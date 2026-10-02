"""Unified Jev-benchmark sweep of ALL LLaDA2 readout/optimization combinations.

Separates configs into ZERO-SHOT STRICT (no task labels used anywhere) and FEW-SHOT CALIBRATED
(a 200-example labelled val slice is used to fit temperature / per-class bias / recalibration). One
table, both models (mini-4bit and mini-8bit), the three Jev datasets.

Efficiency: for each (model, readout) we compute the raw score matrix ONCE on val and test, then
derive every downstream variant (PMI, temperature, bias, recalibration) by post-processing those
scores — no extra forwards. Anchor readout reuses bb.option_logits; multi-token reuses the
sequence-likelihood scorer. Template is FIXED (not val-selected) to keep zero-shot configs honest.

Usage:
  PYTHONPATH=lib python scripts/bench_all_combos.py \
      [--per-dataset 100] [--models 4bit,8bit] [--datasets agnews,banking77,emotiondair]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
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
from explore_1_multitoken import _log_softmax_rows, _label_token_ids, _strip_common_prefix  # noqa: E402
from pisteC_temperature import _softmax_rows, fit_temperature, fit_class_bias  # noqa: E402

BENCH = ROOT / "external" / "jev-benchmarks"
MANIFEST = BENCH / "results" / "runs" / "btzsc-pilot-v1" / "manifest.jsonl"
MANIFEST_SHA256 = "ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5"
DATA = ROOT / "data"
OUT = ROOT / "runs" / "jev-all-combos"
DATASETS = {"agnews": "AG News", "banking77": "Banking77", "emotiondair": "Emotion"}
MODELS = {"4bit": "mlx-community/LLaDA2.0-mini-preview-4bit", "8bit": "mlx-community/LLaDA2.0-mini-8bit"}
FIXED_HEAD = "Which single label best describes the input text?\nThe correct label is:"
NEUTRAL_STATE = "This is a message."  # for the PMI prior (no task label used)


def load_split(dataset, split, labels):
    idx = {lab: i for i, lab in enumerate(labels)}
    rows = []
    for r in read_jsonl(DATA / f"btzsc-{dataset}" / f"{split}.jsonl"):
        j = idx.get(r["label"])
        if j is not None:
            rows.append({"text": r["text"], "target_index": j})
    return rows


# ---- readouts: return a (N, L) raw score matrix -------------------------------------------------

def multitoken_matrix(bb, rows, labels, mask_id, prior_state=None):
    """Sequence-likelihood scores (N, L). If prior_state is set, returns the a-priori score of each
    label under that neutral state (a single row broadcast), used for PMI."""
    label_ids = _label_token_ids(bb, labels)
    suffixes, prefix_len = _strip_common_prefix(label_ids)
    prefix_ids = label_ids[0][:prefix_len]
    tok = bb.tokenizer
    head_ids = tok.encode(FIXED_HEAD, add_special_tokens=False)
    sep = tok.encode("\n", add_special_tokens=False)
    kmax = max(len(s) for s in suffixes)

    def score_state(text):
        base = tok.encode(text, add_special_tokens=False)[:4096] + sep + head_ids + list(prefix_ids)
        positions = list(range(len(base), len(base) + kmax))
        logp = _log_softmax_rows(bb.mask_logits(base + [mask_id] * kmax, positions))
        out = np.empty(len(labels))
        for j, suf in enumerate(suffixes):
            out[j] = sum(logp[i, suf[i]] for i in range(len(suf))) / max(1, len(suf))
        return out

    if prior_state is not None:
        return score_state(prior_state)  # (L,)
    S = np.empty((len(rows), len(labels)))
    lat = np.empty(len(rows))
    for i, r in enumerate(rows):
        t0 = time.perf_counter()
        S[i] = score_state(r["text"])
        lat[i] = time.perf_counter() - t0
    return S, lat


def anchor_matrix(bb, rows, labels, mask_id):
    """Single-anchor readout (N, L): first token of each label description, at one [MASK]."""
    tok = bb.tokenizer
    head_ids = tok.encode(FIXED_HEAD, add_special_tokens=False)
    sep = tok.encode("\n", add_special_tokens=False)
    anchors = [tok.encode(" " + lab, add_special_tokens=False)[0] for lab in labels]
    S = np.empty((len(rows), len(labels)))
    lat = np.empty(len(rows))
    for i, r in enumerate(rows):
        t0 = time.perf_counter()
        base = tok.encode(r["text"], add_special_tokens=False)[:4096] + sep + head_ids
        z = bb.option_logits(base + [mask_id], [len(base)], anchors)[0]  # (L,)
        S[i] = z
        lat[i] = time.perf_counter() - t0
    return S, lat


def scored(S, tgt, labels, ds, lat=None):
    lat = lat if lat is not None else np.zeros(len(S))
    P = _softmax_rows(S)
    preds = [Prediction(experiment_id="all-combos", backend="mlx_llada", model_requested=ds,
                        model_resolved=ds, dataset=ds, example_id=str(i), target_index=int(tgt[i]),
                        predicted_index=int(np.argmax(P[i])), labels=tuple(labels),
                        probabilities=tuple(float(x) for x in P[i]), latency_seconds=float(lat[i]),
                        probability_sum_raw=float(P[i].sum())) for i in range(len(S))]
    return score_predictions(preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=100)
    ap.add_argument("--models", type=str, default="4bit,8bit")
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    args = ap.parse_args()
    models = [m for m in args.models.split(",") if m in MODELS]
    chosen = [d for d in args.datasets.split(",") if d in DATASETS]

    assert sha256_file(MANIFEST) == MANIFEST_SHA256, "manifest changed"
    manifest = read_jsonl(MANIFEST)

    # rows: ds -> (labels, val_rows, test_rows)
    rows_by_ds = {}
    for ds in chosen:
        labels = list(next(r for r in manifest if r["dataset"] == ds)["labels"])
        val = load_split(ds, "val", labels)
        test = [r for r in manifest if r["dataset"] == ds][: args.per_dataset]
        rows_by_ds[ds] = (labels, val, test)

    OUT.mkdir(parents=True, exist_ok=True)
    # results[(model, config)][ds] = {"acc","ece"}, plus "zs" flag per config
    results = defaultdict(dict)
    zeroshot = {}

    for mkey in models:
        repo = MODELS[mkey]
        k = f"combos-{mkey}"
        backbone_mod.MODELS[k] = {"mlx_llada": repo}
        print(f"\n########## loading {repo} ##########", flush=True)
        bb = Backbone(k, "mlx_llada")
        mask_id = bb.mask_id
        # warmup
        _ = anchor_matrix(bb, rows_by_ds[chosen[0]][2][:1], rows_by_ds[chosen[0]][0], mask_id)

        for ds in chosen:
            labels, val, test = rows_by_ds[ds]
            vt = np.array([r["target_index"] for r in val])
            tt = np.array([r["target_index"] for r in test])
            print(f"\n=== {mkey} / {ds} ({len(labels)} labels) ===", flush=True)

            # --- raw readouts, computed once ---
            aS_v, _ = anchor_matrix(bb, val, labels, mask_id)
            aS_t, aLat = anchor_matrix(bb, test, labels, mask_id)
            mS_v, _ = multitoken_matrix(bb, val, labels, mask_id)
            mS_t, mLat = multitoken_matrix(bb, test, labels, mask_id)
            prior = multitoken_matrix(bb, None, labels, mask_id, prior_state=NEUTRAL_STATE)  # (L,)

            def put(config, S_test, tgt, lat, zs):
                results[(mkey, config)][ds] = scored(S_test, tgt, labels, ds, lat)
                zeroshot[config] = zs

            # ===== ZERO-SHOT STRICT =====
            put("anchor", aS_t, tt, aLat, True)
            put("multitoken", mS_t, tt, mLat, True)
            put("multitoken+pmi", mS_t - prior[None, :], tt, mLat, True)

            # ===== FEW-SHOT CALIBRATED (fit on val) =====
            # temperature (calibration only; argmax unchanged -> acc == multitoken)
            T, _ = fit_temperature(mS_v, vt)
            put("multitoken+T", mS_t / T, tt, mLat, False)
            # per-class bias (fit on val, with T) -> accuracy lever
            bias = fit_class_bias(mS_v, vt, len(labels), T, l2=0.5)
            put("multitoken+bias+T", mS_t / T + bias, tt, mLat, False)
            # two-stage recalibration: T2 on the combined logit (keeps argmax, recovers ECE)
            comb_v = mS_v / T + bias
            T2, _ = fit_temperature(comb_v, vt)
            put("multitoken+bias+T+recal", (mS_t / T + bias) / T2, tt, mLat, False)
            print(f"    fit T={T:.3f}, ||b||={np.linalg.norm(bias):.1f}, T2={T2:.3f}", flush=True)

        del bb
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

    # ---- report ----
    configs = ["anchor", "multitoken", "multitoken+pmi",
               "multitoken+T", "multitoken+bias+T", "multitoken+bias+T+recal"]
    lines = [
        "# Jev benchmark — all LLaDA2 readout/optimization combinations",
        "",
        f"Models: {', '.join(models)}. Datasets: {', '.join(DATASETS[d] for d in chosen)}, "
        f"{args.per_dataset} test rows each. Fixed prompt head (not val-selected). "
        "Cells: accuracy / ECE.",
        "",
        "ZS = zero-shot strict (no task labels used). FS = few-shot calibrated (temperature / "
        "per-class bias / recalibration fit on a 200-example val slice; test never used to fit).",
        "",
        "| Model | Config | ZS? | " + " | ".join(DATASETS[d] for d in chosen)
        + " | Mean acc | Mean ECE |",
        "|---|---|:--:|" + "---:|" * (len(chosen) + 2),
    ]
    for mkey in models:
        for cfg in configs:
            if (mkey, cfg) not in results:
                continue
            r = results[(mkey, cfg)]
            cells = " | ".join(f"{r[d]['accuracy']:.2f} / {r[d]['ece']:.3f}" for d in chosen)
            macc = np.mean([r[d]["accuracy"] for d in chosen])
            mece = np.mean([r[d]["ece"] for d in chosen])
            zs = "ZS" if zeroshot.get(cfg) else "FS"
            lines.append(f"| {mkey} | {cfg} | {zs} | {cells} | {macc:.3f} | {mece:.3f} |")

    lines += ["", "## Notes", "",
              "- Fixed head keeps the zero-shot rows honest (a val-selected template would be few-shot).",
              "- anchor and multitoken are the pure readouts; +pmi subtracts each label's a-priori "
              "score under a neutral state (still zero-shot).",
              "- +T is calibration only (accuracy == multitoken by monotonicity); +bias adds accuracy "
              "on many-class tasks; +recal restores ECE after the bias.",
              "- Per-dataset test latency (p50, ms) is saved per config in the JSON dump."]
    report = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(report)
    # machine-readable dump
    dump = {f"{m}|{c}": {d: results[(m, c)][d] for d in chosen}
            for (m, c) in results}
    (OUT / "results.json").write_text(json.dumps(dump, indent=1, default=float))
    print("\n" + report)


if __name__ == "__main__":
    main()
