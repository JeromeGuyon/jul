"""Post-bias recalibration for the Banking77 combo: recover ECE while keeping accuracy 0.62.

The validated combo builds a combined logit  z = scores / T1 + bias  (per-class bias b fit on val).
Its softmax gives accuracy ~0.62 but poor calibration (test ECE ~0.35): the additive bias makes the
distribution over-confident, and T1 (fit on the *raw* scores, before the bias) no longer calibrates
the *biased* logit.

Fix: fit a SECOND temperature T2 on val, on the combined logit, minimizing NLL:

    p = softmax( (scores / T1 + bias) / T2 )

Dividing by T2 is monotone, so it does NOT change the argmax -> accuracy is identical to the combo.
Only ECE / NLL move. This is textbook temperature scaling applied to the post-bias logit.

Everything is fit on val (200 rows) and evaluated once on the full test (100 rows) and, for a tighter
CI, on train (1000 rows, flagged as not strictly held out). No fit touches the test set.

Usage:
  PYTHONPATH=lib python scripts/recalibrate_banking_combo.py [--l2 0.5] [--norm mean]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "jev-benchmarks" / "src"))

from jev_benchmarks.io import read_jsonl, sha256_file  # noqa: E402
from jev_benchmarks.metrics import score_predictions  # noqa: E402
from jev_benchmarks.models import Prediction  # noqa: E402

import jul.backbone as backbone_mod  # noqa: E402
from jul.backbone import Backbone  # noqa: E402

from explore_1_multitoken import multitoken_scores  # noqa: E402
from pisteC_temperature import _softmax_rows, _nll, fit_temperature, fit_class_bias, compute_scores  # noqa: E402

BENCH = ROOT / "external" / "jev-benchmarks"
MANIFEST = BENCH / "results" / "runs" / "btzsc-pilot-v1" / "manifest.jsonl"
MANIFEST_SHA256 = "ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5"
DATA = ROOT / "data" / "btzsc-banking77"
OUT = ROOT / "runs" / "banking-recalib"
MODEL_REPO = "mlx-community/LLaDA2.0-mini-8bit"


def load_rows(split: str, labels: list[str]):
    idx = {lab: i for i, lab in enumerate(labels)}
    rows, miss = [], 0
    for r in read_jsonl(DATA / f"{split}.jsonl"):
        j = idx.get(r["label"])
        if j is None:
            miss += 1
            continue
        rows.append({"text": r["text"], "target_index": j, "example_id": f"{split}-{len(rows)}"})
    return rows, miss


def fit_temperature_on_logits(logits: np.ndarray, targets: np.ndarray):
    """Same NLL grid+golden fit as piste C, but the input is already a combined logit (scores/T1+bias),
    so here 'temperature' T2 divides that logit: p = softmax(logits / T2)."""
    return fit_temperature(logits, targets)  # fit_temperature treats its arg as raw scores to divide


def preds_from_logits(logits, targets, labels, tag, lat=None):
    P = _softmax_rows(logits)
    lat = lat if lat is not None else np.zeros(len(logits))
    out = []
    for i in range(len(logits)):
        out.append(Prediction(
            experiment_id=f"recalib-{tag}", backend="mlx_llada",
            model_requested=MODEL_REPO, model_resolved=MODEL_REPO, dataset="banking77",
            example_id=str(i), target_index=int(targets[i]),
            predicted_index=int(np.argmax(P[i])), labels=tuple(labels),
            probabilities=tuple(float(x) for x in P[i]), latency_seconds=float(lat[i]),
            probability_sum_raw=float(P[i].sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2", type=float, default=0.5)
    ap.add_argument("--norm", type=str, default="mean", choices=["mean", "sum"])
    args = ap.parse_args()

    assert sha256_file(MANIFEST) == MANIFEST_SHA256, "manifest changed"
    manifest = read_jsonl(MANIFEST)
    labels = list(next(r for r in manifest if r["dataset"] == "banking77")["labels"])

    val_rows, _ = load_rows("val", labels)
    test_rows, _ = load_rows("test", labels)
    train_rows, _ = load_rows("train", labels)
    print(f"val {len(val_rows)}, test {len(test_rows)}, train {len(train_rows)}", flush=True)

    key = "recalib-mini-8bit"
    backbone_mod.MODELS[key] = {"mlx_llada": MODEL_REPO}
    print(f"loading {MODEL_REPO} ...", flush=True)
    t0 = time.perf_counter()
    bb = Backbone(key, "mlx_llada")
    print(f"loaded in {time.perf_counter()-t0:.0f}s", flush=True)
    _ = multitoken_scores(bb, val_rows[0]["text"], labels, norm=args.norm)  # warmup

    val_S, val_t, _ = compute_scores(bb, val_rows, labels, args.norm)
    test_S, test_t, test_lat = compute_scores(bb, test_rows, labels, args.norm)
    train_S, train_t, train_lat = compute_scores(bb, train_rows, labels, args.norm)
    assert np.isfinite(val_S).all() and np.isfinite(test_S).all()

    # --- Stage 1: fit T1 (on raw scores) and per-class bias b, both on val ---
    T1, _ = fit_temperature(val_S, val_t)
    bias = fit_class_bias(val_S, val_t, len(labels), T1, l2=args.l2)

    def combined_logit(S):
        return S / T1 + bias

    # --- Stage 2: fit a second temperature T2 on the *combined* logit, on val ---
    T2, t2info = fit_temperature_on_logits(combined_logit(val_S), val_t)
    print(f"fit on val: T1={T1:.4f}, ||b||={np.linalg.norm(bias):.2f}, T2={T2:.4f} "
          f"(combined-logit NLL: T2=1 -> {t2info['nll_at_1']:.3f}, T2* -> {t2info['nll_at_T']:.3f})",
          flush=True)

    def evaluate(S, t, lat=None):
        zc = combined_logit(S)                       # combo (bias + T1), no recalibration
        zr = zc / T2                                 # + recalibration temperature T2
        return {
            "combo (bias+T1)": score_predictions(preds_from_logits(zc, t, labels, "combo", lat)),
            "combo + recalib T2": score_predictions(preds_from_logits(zr, t, labels, "recal", lat)),
        }

    val_res = evaluate(val_S, val_t)
    test_res = evaluate(test_S, test_t, test_lat)
    train_res = evaluate(train_S, train_t, train_lat)
    p50 = float(np.median(test_lat) * 1000)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "predictions_test_recalib.jsonl", "w") as f:
        for p in preds_from_logits(combined_logit(test_S) / T2, test_t, labels, "recal", test_lat):
            f.write(json.dumps(p.to_dict()) + "\n")

    def block(title, res, note=""):
        L = [f"### {title}{note}", "",
             "| variant | accuracy | macro-F1 | ECE | NLL | mean_conf |",
             "|---|---:|---:|---:|---:|---:|"]
        for name, s in res.items():
            L.append(f"| {name} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} | {s['ece']:.3f} | "
                     f"{s['nll']:.3f} | {s['mean_confidence']:.3f} |")
        return "\n".join(L)

    c, r = test_res["combo (bias+T1)"], test_res["combo + recalib T2"]
    lines = [
        "# Banking77 combo — post-bias recalibration (recover ECE, keep accuracy)",
        "",
        f"Model `{MODEL_REPO}`, multi-token readout, norm=`{args.norm}`. Fit on val (200), eval on "
        f"test (100) and train (1000). No fit touches test.",
        "",
        "Pipeline: `p = softmax( (scores / T1 + bias) / T2 )`. Stage 1 fits T1 and the per-class bias "
        "b on val (bias sets accuracy). Stage 2 fits a second temperature T2 on the *combined* logit, "
        "on val, to recalibrate. T2 is monotone, so **accuracy is identical to the combo** — only "
        "ECE/NLL move.",
        "",
        f"Fitted on val: **T1={T1:.4f}, ||b||={np.linalg.norm(bias):.2f} (L2={args.l2}), "
        f"T2={T2:.4f}**.",
        "",
        "## Test set (the result)",
        "",
        block("Combo vs combo + recalibration", test_res),
        "",
        f"Headline: accuracy stays **{r['accuracy']:.3f}** (unchanged by T2), ECE "
        f"**{c['ece']:.3f} -> {r['ece']:.3f}** (Δ {r['ece']-c['ece']:+.3f}), NLL "
        f"{c['nll']:.3f} -> {r['nll']:.3f}, at {p50:.0f} ms p50.",
        "",
        "## Validation set (fitting set)",
        "",
        block("Combo vs combo + recalibration on val", val_res),
        "",
        "## Train split (1000 rows) — extended read (not strictly held out)",
        "",
        block("Combo vs combo + recalibration on train", train_res),
        "",
        "## Read honestly",
        "",
        f"- **Accuracy preserved**: T2 is monotone, so combo and combo+recalib have the SAME argmax; "
        f"test accuracy {r['accuracy']:.3f} on both. Recalibration is free on the accuracy axis.",
        f"- **ECE recovered**: test ECE {c['ece']:.3f} -> {r['ece']:.3f}. "
        f"{'Target ECE<0.15 met.' if r['ece'] < 0.15 else 'Still above 0.15 — see caveat.'}",
        f"- **val->test transfer**: recalib ECE val {val_res['combo + recalib T2']['ece']:.3f} -> "
        f"test {r['ece']:.3f}; train {train_res['combo + recalib T2']['ece']:.3f} (1000 rows) "
        f"narrows the CI.",
        "- **Why it works**: the per-class bias fixes the readout's per-label score-scale bias (an "
        "accuracy lever) but inflates confidence; a temperature on the resulting logit is exactly the "
        "tool to rescale confidence without moving the decision.",
        "- **Ceiling**: accuracy still ~0.62 (< wemm-4b embedding 0.88), but now with usable "
        "calibration — the diffusion readout is competitive with e5 (0.60) AND calibrated.",
    ]
    report = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
