"""Prompt-template sweep for the multi-token LLaDA2 readout (the lever the literature flags as largest).

The multi-token readout scores each label's text over a block of [MASK]s after a fixed prompt head
("... The correct label is:"). arXiv:2608.14649 shows the question/template wording is a large lever
(Reuters micro 60.6 -> 80.5 just by rewording). Here we sweep several heads, select the best per
dataset **on the validation split**, and report the selected head on **test** vs the default head —
a strict no-leak, val-selected protocol (same budget as the paper's template-tuned rows).

Runs on mlx-community/LLaDA2.0-mini-8bit (the best model from piste D). No lib/ module modified: the
readout is re-implemented locally with a parametric head, reusing the score math from
explore_1_multitoken.

Usage:
  PYTHONPATH=lib python scripts/sweep_template_multitoken.py \
      [--per-dataset 100] [--datasets agnews,banking77,emotiondair]
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

BENCH = ROOT / "external" / "jev-benchmarks"
MANIFEST = BENCH / "results" / "runs" / "btzsc-pilot-v1" / "manifest.jsonl"
MANIFEST_SHA256 = "ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5"
DATA = ROOT / "data"
OUT = ROOT / "runs" / "banking-template"
MODEL_REPO = "mlx-community/LLaDA2.0-mini-8bit"
DATASETS = {"agnews": "AG News", "banking77": "Banking77", "emotiondair": "Emotion"}

# Prompt heads to sweep. {q} is a dataset-agnostic question; the label block of [MASK]s follows.
# These mirror the paper's finding that stating the annotation relation directly helps.
HEADS = {
    "default": "Which single label best describes the input text?\nThe correct label is:",
    "topic": "What is the main topic of this text?\nThe main topic is:",
    "about": "This text is about:",
    "category": "Classify the text into a category.\nThe category is:",
    "isabout": "The text is about the following:",
    "represents": "The single best label for this text is:",
}


def load_split(dataset: str, split: str, labels: list[str]):
    """Load a BTZSC split for a dataset, mapping label-descriptions to manifest index."""
    idx = {lab: i for i, lab in enumerate(labels)}
    rows, miss = [], 0
    path = DATA / f"btzsc-{dataset}" / f"{split}.jsonl"
    for r in read_jsonl(path):
        j = idx.get(r["label"])
        if j is None:
            miss += 1
            continue
        rows.append({"text": r["text"], "target_index": j, "example_id": f"{dataset}-{split}-{len(rows)}"})
    return rows, miss


def multitoken_scores_head(bb, state, labels, head, mask_id, norm="mean"):
    """Multi-token sequence-likelihood scores with a parametric prompt head."""
    label_ids = _label_token_ids(bb, labels)
    suffixes, prefix_len = _strip_common_prefix(label_ids)
    prefix_ids = label_ids[0][:prefix_len]
    tok = bb.tokenizer
    state_ids = tok.encode(state, add_special_tokens=False)[:4096]
    head_ids = tok.encode(head, add_special_tokens=False)
    sep = tok.encode("\n", add_special_tokens=False)
    base = state_ids + sep + head_ids + list(prefix_ids)
    kmax = max(len(s) for s in suffixes)
    tokens = base + [mask_id] * kmax
    positions = list(range(len(base), len(base) + kmax))
    logits = bb.mask_logits(tokens, positions)
    logp = _log_softmax_rows(logits)
    scores = np.empty(len(labels), dtype=np.float64)
    for j, suf in enumerate(suffixes):
        kj = len(suf)
        s = sum(logp[i, suf[i]] for i in range(kj))
        scores[j] = s / kj if norm == "mean" else s
    return scores


def score_split(bb, rows, labels, head, mask_id):
    S = np.empty((len(rows), len(labels)), dtype=np.float64)
    tgt = np.empty(len(rows), dtype=int)
    lat = np.empty(len(rows), dtype=float)
    for i, r in enumerate(rows):
        t0 = time.perf_counter()
        S[i] = multitoken_scores_head(bb, r["text"], labels, head, mask_id)
        lat[i] = time.perf_counter() - t0
        tgt[i] = r["target_index"]
    assert np.isfinite(S).all(), "non-finite scores"
    return S, tgt, lat


def preds_from(S, tgt, labels, ds, lat=None):
    lat = lat if lat is not None else np.zeros(len(S))
    out = []
    for i in range(len(S)):
        z = S[i]
        p = np.exp(z - z.max()); p /= p.sum()
        out.append(Prediction(
            experiment_id="template-sweep", backend="mlx_llada", model_requested=MODEL_REPO,
            model_resolved=MODEL_REPO, dataset=ds, example_id=str(i), target_index=int(tgt[i]),
            predicted_index=int(np.argmax(p)), labels=tuple(labels),
            probabilities=tuple(float(x) for x in p), latency_seconds=float(lat[i]),
            probability_sum_raw=float(p.sum())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=100)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    args = ap.parse_args()
    chosen = [d for d in args.datasets.split(",") if d in DATASETS]

    assert sha256_file(MANIFEST) == MANIFEST_SHA256, "manifest changed"
    manifest = read_jsonl(MANIFEST)

    key = "sweep-mini-8bit"
    backbone_mod.MODELS[key] = {"mlx_llada": MODEL_REPO}
    print(f"loading {MODEL_REPO} ...", flush=True)
    t0 = time.perf_counter()
    bb = Backbone(key, "mlx_llada")
    mask_id = bb.mask_id
    print(f"loaded in {time.perf_counter()-t0:.0f}s", flush=True)

    rows_by_ds = {}
    for ds in chosen:
        labels = list(next(r for r in manifest if r["dataset"] == ds)["labels"])
        val, _ = load_split(ds, "val", labels)
        test = [r for r in manifest if r["dataset"] == ds]
        if args.per_dataset:
            test = test[: args.per_dataset]
        rows_by_ds[ds] = (labels, val, test)

    # warmup
    ds0 = chosen[0]
    _ = multitoken_scores_head(bb, rows_by_ds[ds0][2][0]["text"], rows_by_ds[ds0][0],
                               HEADS["default"], mask_id)

    results = {}  # ds -> {head: (val_acc, test_scored, p50)}
    OUT.mkdir(parents=True, exist_ok=True)
    for ds in chosen:
        labels, val, test = rows_by_ds[ds]
        print(f"\n=== {ds} ({len(labels)} labels, val {len(val)}, test {len(test)}) ===", flush=True)
        per_head = {}
        for hname, head in HEADS.items():
            vS, vt, _ = score_split(bb, val, labels, head, mask_id)
            val_acc = float((vS.argmax(1) == vt).mean())
            tS, tt, tlat = score_split(bb, test, labels, head, mask_id)
            tscored = score_predictions(preds_from(tS, tt, labels, ds, tlat))
            per_head[hname] = {"val_acc": val_acc, "test": tscored,
                               "p50": float(np.median(tlat) * 1000)}
            print(f"  [{hname}] val_acc={val_acc:.3f}  test_acc={tscored['accuracy']:.3f} "
                  f"ECE={tscored['ece']:.3f}", flush=True)
        results[ds] = per_head

    # Report: default vs val-selected head, per dataset.
    lines = [
        "# Prompt-template sweep — multi-token readout on LLaDA2-mini-8bit",
        "",
        f"Model `{MODEL_REPO}`. Multi-token readout with a parametric prompt head. Best head selected "
        "on the **validation** split (micro-accuracy), reported on **test** — no-leak, val-selected.",
        f"Test rows/dataset: {args.per_dataset or 'all'}. Heads swept: " + ", ".join(HEADS) + ".",
        "",
        "## Default head vs val-selected head (test set)",
        "",
        "| Dataset | default test acc | best head (val-selected) | selected test acc | Δacc | ECE def→sel |",
        "|---|---:|---|---:|---:|---|",
    ]
    for ds in chosen:
        ph = results[ds]
        deft = ph["default"]["test"]["accuracy"]
        defece = ph["default"]["test"]["ece"]
        best_head = max(ph, key=lambda h: ph[h]["val_acc"])  # selected on VAL
        selt = ph[best_head]["test"]["accuracy"]
        selece = ph[best_head]["test"]["ece"]
        lines.append(f"| {DATASETS[ds]} | {deft:.3f} | {best_head} | {selt:.3f} | "
                     f"{selt-deft:+.3f} | {defece:.3f}→{selece:.3f} |")

    lines += ["", "## Full sweep (test accuracy / ECE per head; val_acc is the selection key)", ""]
    for ds in chosen:
        lines += [f"### {DATASETS[ds]}", "",
                  "| head | val acc (sel) | test acc | test ECE | p50 ms |",
                  "|---|---:|---:|---:|---:|"]
        ph = results[ds]
        for h in HEADS:
            r = ph[h]
            star = " ★" if h == max(ph, key=lambda k: ph[k]["val_acc"]) else ""
            lines.append(f"| {h}{star} | {r['val_acc']:.3f} | {r['test']['accuracy']:.3f} | "
                        f"{r['test']['ece']:.3f} | {r['p50']:.0f} |")
        lines.append("")

    lines += [
        "## Read honestly",
        "",
        "- ★ marks the head selected on val (the only fair operating point). Δacc is selected−default "
        "on test.",
        "- The paper (arXiv:2608.14649) reports large template effects (Reuters micro 60.6→80.5); this "
        "sweep tests whether the same lever helps the sequence-likelihood readout on these datasets.",
        "- val→test transfer: a head that wins on val but not test is an overfit of the 200-row val "
        "slice; check the star lands on a competitive test cell.",
        "- Multi-token baseline (mini-8bit, default head): Banking77 ~0.34, AG News ~0.87, Emotion "
        "~0.43 (from piste D / earlier runs).",
    ]
    report = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
