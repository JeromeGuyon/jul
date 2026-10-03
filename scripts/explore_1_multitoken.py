"""EXPLORATION 1 — Multi-token (sequence-likelihood) readout for System One on MLX LLaDA2-MoE.

Problem
-------
The baseline System One readout compares only the *first token* of each option marker (anchor
mono-token). On a 72-way task (Banking77) that single-token comparison saturates and collapses to
~0.02 accuracy: the useful signal lives in the *label text*, not in a one-letter marker.

Idea
----
Score the FULL-STRING likelihood of each label. For each option we tokenize its label text into K
tokens, place K consecutive [MASK]s where the answer goes, run ONE bidirectional forward, and read at
each mask position k the log-softmax of the *true* k-th token of that label. The option score is the
(length-normalized) sum of those log-probs. The option with the highest sequence likelihood wins.

Because every label in a dataset shares a long common prefix ("This example ... is about"), we strip
the token-level longest common prefix across the options and only mask/score the *distinctive
suffix*. This focuses the K masks on the discriminating tokens and keeps K small.

This file is self-contained: it imports the existing backbone and calls bb.mask_logits(...) directly.
It does NOT modify mlx_llada.py / mask.py / backbone.py / presets.py. The new readout is the local
functions below.

Usage
-----
  PYTHONPATH=lib python scripts/explore_1_multitoken.py \
      [--per-dataset N] [--datasets agnews,banking77,emotiondair] [--norm mean|sum]
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
# jev_benchmarks lives under external/.../src (the lib/ symlink is stale on this checkout).
sys.path.insert(0, str(ROOT / "external" / "jev-benchmarks" / "src"))

from jev_benchmarks.io import read_jsonl, sha256_file  # noqa: E402
from jev_benchmarks.metrics import score_predictions  # noqa: E402
from jev_benchmarks.models import Prediction  # noqa: E402

from jul.backbone import Backbone  # noqa: E402

BENCH = ROOT / "external" / "jev-benchmarks"
MANIFEST = BENCH / "results" / "runs" / "btzsc-pilot-v1" / "manifest.jsonl"
MANIFEST_SHA256 = "ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5"
DATASETS = {"agnews": "AG News", "banking77": "Banking77", "emotiondair": "Emotion"}
QUESTION = "Which single label best describes the input text?"
OUT = ROOT / "runs" / "explore-1"

# Baseline (anchor mono-token, Jev pilot, per the task brief) for the report comparison.
BASELINE = {
    "agnews": {"accuracy": 0.84, "ece": 0.058, "p50_ms": 50},
    "banking77": {"accuracy": 0.02, "ece": 0.006, "p50_ms": 313},
    "emotiondair": {"accuracy": 0.43, "ece": 0.378, "p50_ms": 46},
    "mean": {"accuracy": 0.430, "ece": 0.147},
}


# --------------------------------------------------------------------------------------------
# Local multi-token readout (the exploration). Calls only the public backbone surface.
# --------------------------------------------------------------------------------------------

def _log_softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax of a (P, V) float32 array."""
    m = logits.max(axis=1, keepdims=True)
    shifted = logits - m
    logsumexp = np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return shifted - logsumexp


def _label_token_ids(bb, labels: list[str]) -> list[list[int]]:
    """Tokenize each label with a leading space (mid-sentence form), as a list of token ids."""
    return [bb.tokenizer.encode(" " + lab, add_special_tokens=False) for lab in labels]


def _strip_common_prefix(seqs: list[list[int]]) -> tuple[list[list[int]], int]:
    """Drop the token-level longest common prefix shared by every sequence.

    Returns (suffixes, prefix_len). All labels in a dataset share a boilerplate head
    ("This example ... is about"); those tokens are common-mode and carry no signal for the argmax,
    so we neither mask nor score them. We always keep at least one token per label.
    """
    if not seqs:
        return seqs, 0
    shortest = min(len(s) for s in seqs)
    prefix_len = 0
    for i in range(shortest - 1):  # keep >=1 token in the shortest label
        col = {s[i] for s in seqs}
        if len(col) == 1:
            prefix_len += 1
        else:
            break
    return [s[prefix_len:] for s in seqs], prefix_len


def _build_tokens(bb, state: str, prefix_ids: list[int], k: int) -> tuple[list[int], list[int]]:
    """Build  <state>\\n<head + shared label prefix> [MASK] x k  ; return (tokens, mask_positions).

    The shared label prefix (e.g. "This example ... is about") is emitted as real (unmasked) context
    right before the mask block, so the model conditions on it while only the distinctive suffix is
    read at the K masks.
    """
    tok = bb.tokenizer
    state_ids = tok.encode(state, add_special_tokens=False)[:4096]
    head = f"{QUESTION}\nThe correct label is:"
    head_ids = tok.encode(head, add_special_tokens=False)
    sep = tok.encode("\n", add_special_tokens=False)
    base = state_ids + sep + head_ids + list(prefix_ids)
    tokens = base + [bb.mask_id] * k
    mask_positions = list(range(len(base), len(base) + k))
    return tokens, mask_positions


def multitoken_scores(bb, state: str, labels: list[str], norm: str = "mean") -> np.ndarray:
    """Sequence-likelihood score per label (higher = more likely). Length-variable K handled.

    Strip the token-level common prefix shared by all labels, place K = max_j len(suffix_j) [MASK]s,
    run ONE bidirectional forward, and read at each mask position i the full-vocab log-softmax. The
    score of option j is the length-normalized sum of log p(suffix_j[i]) over its own K_j <= K
    positions (mean or sum aggregation).

    One forward suffices: every option is scored against the same all-masked block, so the vocab
    distribution at position i is computed once and reused for all options. This is the native
    "parallel" (position-independent) diffusion readout — each position is predicted with the rest of
    the block masked. Variable lengths are handled because option j only reads its first K_j
    positions. (Scoring each option in its own forward gives identical numbers — the token sequence,
    base + K masks, is the same for all options — but costs K x more forwards.)
    """
    suffixes, prefix_len = _strip_common_prefix(_label_token_ids(bb, labels))
    prefix_ids = _label_token_ids(bb, labels)[0][:prefix_len]
    kmax = max(len(s) for s in suffixes)
    tokens, mask_positions = _build_tokens(bb, state, prefix_ids, kmax)

    # ONE forward over the whole K-mask block; full-vocab logits at every mask position.
    logits = bb.mask_logits(tokens, mask_positions)  # (kmax, V), finite-checked by the backbone
    logp = _log_softmax_rows(logits)  # (kmax, V)

    scores = np.empty(len(labels), dtype=np.float32)
    for j, suf in enumerate(suffixes):
        kj = len(suf)
        token_logp = np.array([logp[i, suf[i]] for i in range(kj)], dtype=np.float64)
        s = token_logp.sum()
        scores[j] = s / kj if norm == "mean" else s
    return scores


# --------------------------------------------------------------------------------------------
# Bench harness (same metric code and Prediction contract as the reference bench).
# --------------------------------------------------------------------------------------------

def build(bb, rows, ds, norm):
    labels = list(rows[0]["labels"])
    preds = []
    for r in rows:
        start = time.perf_counter()
        scores = multitoken_scores(bb, r["text"], labels, norm=norm)
        latency = time.perf_counter() - start
        z = scores.astype(np.float64)
        p = np.exp(z - z.max())
        p = p / p.sum()
        preds.append(Prediction(
            experiment_id="explore-1-multitoken", backend="mlx_llada",
            model_requested="llada2-mini-4bit", model_resolved="llada2-mini-4bit",
            dataset=ds, example_id=r["example_id"], target_index=r["target_index"],
            predicted_index=int(np.argmax(p)), labels=tuple(r["labels"]),
            probabilities=tuple(float(x) for x in p), latency_seconds=latency,
            probability_sum_raw=float(p.sum())))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=40)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--norm", type=str, default="mean", choices=["mean", "sum"])
    args = ap.parse_args()
    chosen = [d for d in args.datasets.split(",") if d in DATASETS]

    assert sha256_file(MANIFEST) == MANIFEST_SHA256, "manifest changed"
    manifest = read_jsonl(MANIFEST)
    rows_by_ds = {}
    for ds in chosen:
        rs = [r for r in manifest if r["dataset"] == ds]
        rows_by_ds[ds] = rs[: args.per_dataset] if args.per_dataset else rs

    print("loading backbone (MLX LLaDA2-mini-4bit)...", flush=True)
    bb = Backbone("llada2-mini-4bit", "mlx_llada")

    # Warmup (compile MLX graph, exclude from timing).
    print("warmup...", flush=True)
    _ = multitoken_scores(bb, rows_by_ds[chosen[0]][0]["text"],
                          list(rows_by_ds[chosen[0]][0]["labels"]), norm=args.norm)

    preds = []
    for ds in chosen:
        t0 = time.perf_counter()
        preds += build(bb, rows_by_ds[ds], ds, args.norm)
        print(f"  [{ds}] {len(rows_by_ds[ds])} rows in {time.perf_counter()-t0:.1f}s", flush=True)

    # Score and write the report.
    by = defaultdict(list)
    for p in preds:
        by[p.dataset].append(p)
    scored = {d: score_predictions(by[d]) for d in chosen}
    mean_acc = float(np.mean([scored[d]["accuracy"] for d in chosen]))
    mean_ece = float(np.mean([scored[d]["ece"] for d in chosen]))
    lat = {d: float(np.median([p.latency_seconds for p in by[d]]) * 1000) for d in chosen}

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "predictions.jsonl", "w") as f:
        for p in preds:
            f.write(json.dumps(p.__dict__, default=str) + "\n")

    lines = [
        "# Exploration 1 — Multi-token (sequence-likelihood) readout",
        "",
        f"Model: `mlx-community/LLaDA2.0-mini-preview-4bit` (MLX LLaDA2-MoE), System One (read at [MASK]).",
        f"Rows/dataset: {args.per_dataset or 'all (100)'}. Length normalization: `{args.norm}`.",
        "",
        "Readout: for each label, strip the token-level common prefix shared by all labels, place "
        "K = max distinctive-suffix length [MASK]s, run ONE bidirectional forward, and score each "
        "option = length-normalized sum of log-softmax of its true suffix tokens read at their mask "
        "positions. Argmax over options.",
        "",
        "| Dataset | acc (multi-token) | acc (baseline) | Δacc | ECE (mt) | ECE (base) | p50 ms (mt) | p50 ms (base) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for d in chosen:
        b = BASELINE[d]
        dacc = scored[d]["accuracy"] - b["accuracy"]
        lines.append(
            f"| {DATASETS[d]} | {scored[d]['accuracy']:.3f} | {b['accuracy']:.3f} | "
            f"{dacc:+.3f} | {scored[d]['ece']:.3f} | {b['ece']:.3f} | "
            f"{lat[d]:.0f} | {b['p50_ms']} |"
        )
    lines += [
        "",
        f"**Mean accuracy**: {mean_acc:.3f} (baseline {BASELINE['mean']['accuracy']:.3f}, "
        f"Δ {mean_acc - BASELINE['mean']['accuracy']:+.3f}).  ",
        f"**Mean ECE**: {mean_ece:.3f} (baseline {BASELINE['mean']['ece']:.3f}).",
        "",
        "Per-dataset detail (macro-F1, NLL, mean confidence):",
        "",
        "| Dataset | acc | macro_f1 | ece | nll | mean_conf |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for d in chosen:
        s = scored[d]
        lines.append(
            f"| {DATASETS[d]} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} | {s['ece']:.3f} | "
            f"{s['nll']:.3f} | {s['mean_confidence']:.3f} |"
        )
    report = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
