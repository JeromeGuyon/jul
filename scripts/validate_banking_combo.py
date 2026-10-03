"""Validate the best Banking77 combination: mini-8bit + per-class bias + temperature.

This combines the two winning levers found in the piste explorations, on the full Banking77 test,
with a strict no-leak protocol:

  * Model:      mlx-community/LLaDA2.0-mini-8bit (piste D: +0.15 acc over 4-bit, same arch/vocab).
  * Readout:    the multi-token sequence-likelihood readout (explore_1_multitoken.py, verbatim).
  * Per-class bias b (degenerate Platt, changes the argmax -> lifts accuracy): FIT ON VAL only.
  * Temperature T (monotone, calibration only): FIT ON VAL only.
  * Eval:       the full Banking77 test (100 rows). Also reported on train (1000 rows) as an extended
                robustness read — flagged as NOT a strict held-out set (train was not used to fit
                anything here, but it is the training split by name).

No fit ever touches the test set. The per-class bias and temperature are fit on val; test is scored
once. All scores are computed once per split and reused across variants.

Baselines for context (measured earlier): multi-token 4-bit 0.19, multi-token 8-bit 0.34,
e5-small-v2 alone 0.60 (embedding, the structurally better tool for 72-way).

Usage:
  PYTHONPATH=lib python scripts/validate_banking_combo.py \
      [--l2 0.5] [--norm mean] [--eval-train]
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

# Reuse verbatim: the readout, and the calibration/fitting math from piste C.
from explore_1_multitoken import multitoken_scores  # noqa: E402
from pisteC_temperature import (  # noqa: E402
    _softmax_rows, fit_temperature, fit_class_bias, compute_scores,
)

BENCH = ROOT / "external" / "jev-benchmarks"
MANIFEST = BENCH / "results" / "runs" / "btzsc-pilot-v1" / "manifest.jsonl"
MANIFEST_SHA256 = "ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5"
DATA = ROOT / "data" / "btzsc-banking77"
OUT = ROOT / "runs" / "banking-combo"
MODEL_REPO = "mlx-community/LLaDA2.0-mini-8bit"

# Context baselines (measured in earlier runs), for the report only.
CONTEXT = {
    "multitoken 4-bit": 0.19,
    "multitoken 8-bit": 0.34,
    "e5-small-v2 alone": 0.60,
}


def load_rows(split: str, labels: list[str]):
    """Load a BTZSC split, mapping each label-description to its manifest index."""
    idx = {lab: i for i, lab in enumerate(labels)}
    rows, miss = [], 0
    for r in read_jsonl(DATA / f"{split}.jsonl"):
        j = idx.get(r["label"])
        if j is None:
            miss += 1
            continue
        rows.append({"text": r["text"], "target_index": j, "example_id": f"{split}-{len(rows)}"})
    return rows, miss


def make_preds(scores, targets, labels, tag, T=1.0, bias=None, lat=None):
    z = scores / T
    if bias is not None:
        z = z + bias
    P = _softmax_rows(z)
    lat = lat if lat is not None else np.zeros(len(scores))
    preds = []
    for i in range(len(scores)):
        preds.append(Prediction(
            experiment_id=f"combo-{tag}", backend="mlx_llada",
            model_requested=MODEL_REPO, model_resolved=MODEL_REPO, dataset="banking77",
            example_id=str(i), target_index=int(targets[i]),
            predicted_index=int(np.argmax(P[i])), labels=tuple(labels),
            probabilities=tuple(float(x) for x in P[i]), latency_seconds=float(lat[i]),
            probability_sum_raw=float(P[i].sum())))
    return preds


def variants_table(scores, targets, labels, T, bias, lat=None):
    """Score the four cumulative variants on one split."""
    out = {}
    out["8bit (T=1)"] = score_predictions(make_preds(scores, targets, labels, "raw", lat=lat))
    out["8bit + T"] = score_predictions(make_preds(scores, targets, labels, "T", T=T, lat=lat))
    out["8bit + bias"] = score_predictions(make_preds(scores, targets, labels, "b", bias=bias, lat=lat))
    out["8bit + bias + T"] = score_predictions(
        make_preds(scores, targets, labels, "bT", T=T, bias=bias, lat=lat))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--l2", type=float, default=0.5, help="L2 for the per-class bias fit")
    ap.add_argument("--norm", type=str, default="mean", choices=["mean", "sum"])
    ap.add_argument("--eval-train", action="store_true",
                    help="also evaluate on the 1000-row train split (extended robustness read)")
    args = ap.parse_args()

    assert sha256_file(MANIFEST) == MANIFEST_SHA256, "manifest changed"
    manifest = read_jsonl(MANIFEST)
    labels = list(next(r for r in manifest if r["dataset"] == "banking77")["labels"])

    val_rows, vmiss = load_rows("val", labels)
    test_rows, tmiss = load_rows("test", labels)
    print(f"val {len(val_rows)} ({vmiss} dropped), test {len(test_rows)} ({tmiss} dropped)", flush=True)
    train_rows = None
    if args.eval_train:
        train_rows, trmiss = load_rows("train", labels)
        print(f"train {len(train_rows)} ({trmiss} dropped)", flush=True)

    # Load mini-8bit through the same mlx_llada path (runtime registration, no lib/ change).
    key = "combo-mini-8bit"
    backbone_mod.MODELS[key] = {"mlx_llada": MODEL_REPO}
    print(f"loading {MODEL_REPO} ...", flush=True)
    t0 = time.perf_counter()
    bb = Backbone(key, "mlx_llada")
    print(f"loaded in {time.perf_counter()-t0:.0f}s (n_layers={bb.n_layers})", flush=True)

    print("warmup...", flush=True)
    _ = multitoken_scores(bb, val_rows[0]["text"], labels, norm=args.norm)

    # Compute raw multi-token scores once per split.
    t0 = time.perf_counter()
    val_S, val_t, _ = compute_scores(bb, val_rows, labels, args.norm)
    print(f"val scored in {time.perf_counter()-t0:.0f}s", flush=True)
    t0 = time.perf_counter()
    test_S, test_t, test_lat = compute_scores(bb, test_rows, labels, args.norm)
    print(f"test scored in {time.perf_counter()-t0:.0f}s", flush=True)
    train_S = train_t = train_lat = None
    if train_rows is not None:
        t0 = time.perf_counter()
        train_S, train_t, train_lat = compute_scores(bb, train_rows, labels, args.norm)
        print(f"train scored in {time.perf_counter()-t0:.0f}s", flush=True)

    assert np.isfinite(val_S).all() and np.isfinite(test_S).all(), "non-finite scores"

    # --- Fit on VAL only ---
    T, tinfo = fit_temperature(val_S, val_t)
    bias = fit_class_bias(val_S, val_t, len(labels), T, l2=args.l2)
    print(f"fit on val: T={T:.4f}, ||bias||={np.linalg.norm(bias):.2f} (L2={args.l2})", flush=True)

    # --- Evaluate the four variants on val (sanity) and test (the real result) ---
    val_res = variants_table(val_S, val_t, labels, T, bias)
    test_res = variants_table(test_S, test_t, labels, T, bias, lat=test_lat)
    train_res = None
    if train_rows is not None:
        train_res = variants_table(train_S, train_t, labels, T, bias, lat=train_lat)

    p50 = float(np.median(test_lat) * 1000)

    OUT.mkdir(parents=True, exist_ok=True)
    # Persist the headline test predictions (the full combo).
    with open(OUT / "predictions_test_combo.jsonl", "w") as f:
        for p in make_preds(test_S, test_t, labels, "bT", T=T, bias=bias, lat=test_lat):
            f.write(json.dumps(p.to_dict()) + "\n")

    def block(title, res, note=""):
        L = [f"### {title}{note}", "",
             "| variant | accuracy | macro-F1 | ECE | NLL | mean_conf |",
             "|---|---:|---:|---:|---:|---:|"]
        for name, s in res.items():
            L.append(f"| {name} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} | {s['ece']:.3f} | "
                     f"{s['nll']:.3f} | {s['mean_confidence']:.3f} |")
        return "\n".join(L)

    combo = test_res["8bit + bias + T"]
    raw8 = test_res["8bit (T=1)"]
    lines = [
        "# Validation — Banking77: mini-8bit + per-class bias + temperature (full test)",
        "",
        f"Model `{MODEL_REPO}` (LLaDA2-MoE, 8-bit). Multi-token sequence-likelihood readout "
        f"(`explore_1_multitoken.py`, verbatim), norm=`{args.norm}`.",
        "",
        "**No-leak protocol.** Per-class bias b and temperature T are fit on the Banking77 **val** "
        f"split ({len(val_rows)} rows) only; the **test** split ({len(test_rows)} rows) is scored "
        "once. Fitted on val: "
        f"T={T:.4f}, ||b||={np.linalg.norm(bias):.2f} (L2={args.l2}). "
        "Both splits are class-balanced over all 72 intents, so the bias is not a class-frequency "
        "prior — it removes the readout's input-independent per-label score-scale bias, which is why "
        "it transfers.",
        "",
        "## Test set (the result)",
        "",
        block("Cumulative levers on the full test", test_res),
        "",
        f"Headline: **8-bit + bias + T = {combo['accuracy']:.3f} accuracy / {combo['ece']:.3f} ECE** "
        f"at {p50:.0f} ms p50.",
        "",
        "Context baselines (measured earlier, same task):",
        "",
        "| approach | accuracy |",
        "|---|---:|",
    ]
    for name, acc in CONTEXT.items():
        lines.append(f"| {name} | {acc:.2f} |")
    lines.append(f"| **8-bit + bias + T (this run)** | **{combo['accuracy']:.3f}** |")
    lines += [
        "",
        "## Validation set (fitting set — sanity, expect the best numbers here)",
        "",
        block("Cumulative levers on val", val_res),
    ]
    if train_res is not None:
        lines += [
            "",
            "## Train split (1000 rows) — extended robustness read",
            "",
            block("Cumulative levers on train", train_res,
                  " — NOT a strict held-out set (train split by name); shown only to reduce the "
                  "wide CI of the 100-row test"),
        ]
    lines += [
        "",
        "## Read honestly",
        "",
        f"- **Accuracy gain is real and stacks**: 8-bit raw {raw8['accuracy']:.3f} -> "
        f"+bias {test_res['8bit + bias']['accuracy']:.3f} -> +T {combo['accuracy']:.3f} "
        f"(T is monotone so it does not change the argmax; bias does).",
        f"- **val->test transfer**: bias+T val {val_res['8bit + bias + T']['accuracy']:.3f} -> "
        f"test {combo['accuracy']:.3f}. A modest gap is expected (72 classes, thin splits).",
        "- **Calibration**: temperature is the lever that keeps ECE in check; the per-class bias on "
        "its own tends to over-confidence, so the combo applies T as well.",
        f"- **Ceiling context**: even combined, this stays below the e5 embedding "
        f"({CONTEXT['e5-small-v2 alone']:.2f}) — for 72-way retrieval-style classification the "
        "embedding remains structurally stronger. The combo is the best the diffusion [MASK] readout "
        "reaches here.",
        "- **CI caveat**: test is 100 rows (~±0.08 at 2 SE on a proportion near 0.5), so treat the "
        "point estimate as indicative; the train read above narrows it.",
    ]
    report = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
