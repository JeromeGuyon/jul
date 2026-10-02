"""EXPLORATION E5 — Hybrid retrieval (e5 embedding top-k) + re-rank (multi-token LLaDA).

Problem
-------
Banking77 is a 72-way task. Scoring all 72 labels at once with the multi-token LLaDA readout
(explore_1_multitoken.py) dilutes the signal and lands at accuracy 0.19 (baseline anchor mono-token
0.02). Known upper bound of the repo: an e5/wemm embedding classifier reaches ~0.88 zero-shot.

Idea (two stages)
-----------------
(1) RETRIEVAL: a fast encoder (e5-small-v2) embeds the 72 label descriptions once (as e5 "passage:")
    and each input text (as e5 "query:"). Cosine similarity -> keep the top-k candidate labels.
(2) RE-RANK: the multi-token LLaDA readout scores ONLY those k candidates and takes the argmax.

We measure, on Banking77 (100 rows of the Jev pilot manifest), for k in {5, 8, 10, 72}:
  - recall@k of the retrieval  (is the gold label inside the top-k? => hard ceiling for the re-rank)
  - accuracy of e5 alone       (top-1 cosine)
  - accuracy of the hybrid     (e5 top-k, then LLaDA re-rank)
  - latencies (e5 query encode, LLaDA re-rank, total)

STEP BACK — the questions the report must answer:
  (i)   Does the hybrid beat the multi-token-only baseline (0.19)?
  (ii)  Does the hybrid beat e5 alone?
  (iii) What is the recall@k ceiling, and how close does the re-rank get to it?
  (iv)  Does the LLaDA re-rank add anything on top of e5, or is e5 doing all the work?

This file is self-contained. It does NOT modify mlx_llada.py / mask.py / backbone.py / presets.py.
It reuses the multi-token readout functions from scripts/explore_1_multitoken.py verbatim.

Usage
-----
  PYTHONPATH=lib python scripts/hybrid_e5_rerank.py \
      [--rows N] [--ks 5,8,10,72] [--e5-model intfloat/e5-small-v2]
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

from jul.backbone import Backbone  # noqa: E402

# Reuse the reference multi-token readout verbatim (do not reimplement it).
from explore_1_multitoken import (  # noqa: E402
    _build_tokens,
    _label_token_ids,
    _log_softmax_rows,
    _strip_common_prefix,
)

BENCH = ROOT / "external" / "jev-benchmarks"
MANIFEST = BENCH / "results" / "runs" / "btzsc-pilot-v1" / "manifest.jsonl"
MANIFEST_SHA256 = "ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5"
DATASET = "banking77"
OUT = ROOT / "runs" / "banking-hybrid"

# Baseline: multi-token readout scoring all 72 labels (explore_1, 100 rows).
BASE_MULTITOKEN_ACC = 0.19
BASE_MULTITOKEN_ECE = 0.152
BASE_MULTITOKEN_P50_MS = 33.0


# ------------------------------------------------------------------------------------------------
# Multi-token re-rank restricted to a candidate subset.
# ------------------------------------------------------------------------------------------------

def multitoken_scores_subset(bb, state: str, labels: list[str], cand_idx: list[int],
                             norm: str = "mean") -> np.ndarray:
    """Sequence-likelihood score for a SUBSET of labels (the retrieval candidates).

    Identical readout to explore_1.multitoken_scores, but the common-prefix strip and the K-mask
    block are built over ONLY the candidate labels. Restricting to k candidates concentrates the
    strip on their distinctive suffixes and lets each option compete against a shorter, sharper
    mask block. ONE bidirectional forward. Returns a score per candidate (same order as cand_idx).
    """
    cand_labels = [labels[i] for i in cand_idx]
    suffixes, prefix_len = _strip_common_prefix(_label_token_ids(bb, cand_labels))
    prefix_ids = _label_token_ids(bb, cand_labels)[0][:prefix_len]
    kmax = max(len(s) for s in suffixes)
    tokens, mask_positions = _build_tokens(bb, state, prefix_ids, kmax)

    logits = bb.mask_logits(tokens, mask_positions)  # (kmax, V), finite-checked
    logp = _log_softmax_rows(logits)

    scores = np.empty(len(cand_labels), dtype=np.float32)
    for j, suf in enumerate(suffixes):
        kj = len(suf)
        token_logp = np.array([logp[i, suf[i]] for i in range(kj)], dtype=np.float64)
        s = token_logp.sum()
        scores[j] = s / kj if norm == "mean" else s
    return scores


# ------------------------------------------------------------------------------------------------
# E5 retrieval.
# ------------------------------------------------------------------------------------------------

def encode_e5(model, texts: list[str], kind: str, batch_size: int = 32) -> np.ndarray:
    """L2-normalized e5 embeddings. e5 expects 'query: ' / 'passage: ' prefixes."""
    prefixed = [f"{kind}: {t}" for t in texts]
    emb = model.encode(prefixed, batch_size=batch_size, normalize_embeddings=True,
                       convert_to_numpy=True, show_progress_bar=False)
    return emb.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100)
    ap.add_argument("--ks", type=str, default="5,8,10,72")
    ap.add_argument("--e5-model", type=str, default="intfloat/e5-small-v2")
    ap.add_argument("--norm", type=str, default="mean", choices=["mean", "sum"])
    args = ap.parse_args()
    ks = sorted({int(x) for x in args.ks.split(",")})

    assert sha256_file(MANIFEST) == MANIFEST_SHA256, "manifest changed"
    manifest = read_jsonl(MANIFEST)
    rows = [r for r in manifest if r["dataset"] == DATASET]
    rows = rows[: args.rows] if args.rows else rows
    labels = list(rows[0]["labels"])
    n_labels = len(labels)
    print(f"banking77: {len(rows)} rows, {n_labels} labels", flush=True)

    # --- Stage 1: e5 retrieval -------------------------------------------------------------------
    from sentence_transformers import SentenceTransformer
    print(f"loading e5 ({args.e5_model})...", flush=True)
    e5 = SentenceTransformer(args.e5_model)

    print("encoding 72 label descriptions (passage:) once...", flush=True)
    label_emb = encode_e5(e5, labels, "passage")  # (72, d)

    texts = [r["text"] for r in rows]
    # Warmup e5 query encode (compile / first-call overhead) then time per-row.
    _ = encode_e5(e5, [texts[0]], "query")
    t0 = time.perf_counter()
    query_emb = encode_e5(e5, texts, "query")  # (N, d)
    e5_encode_s = time.perf_counter() - t0
    e5_encode_ms_per_row = e5_encode_s / len(rows) * 1000

    # Cosine (embeddings are L2-normalized => dot product).
    sims = query_emb @ label_emb.T  # (N, 72)
    assert np.isfinite(sims).all(), "non-finite cosine similarities"
    e5_top1 = sims.argmax(axis=1)  # (N,)
    # Full ranking (descending), used to slice top-k per row.
    ranking = np.argsort(-sims, axis=1)  # (N, 72)

    targets = np.array([r["target_index"] for r in rows])
    e5_acc = float((e5_top1 == targets).mean())
    print(f"e5 alone (top-1) accuracy: {e5_acc:.3f}  |  encode {e5_encode_ms_per_row:.1f} ms/row",
          flush=True)

    # recall@k for each requested k.
    recall_at_k = {}
    for k in ks:
        kk = min(k, n_labels)
        topk = ranking[:, :kk]
        hit = np.array([targets[i] in topk[i] for i in range(len(rows))])
        recall_at_k[k] = float(hit.mean())
    print("recall@k:", {k: round(v, 3) for k, v in recall_at_k.items()}, flush=True)

    # --- Stage 2: LLaDA re-rank per k ------------------------------------------------------------
    print("loading backbone (MLX LLaDA2-mini-4bit)...", flush=True)
    bb = Backbone("llada2-mini-4bit", "mlx_llada")

    # Warmup the MLX graph on a real candidate set (excluded from timing).
    print("warmup LLaDA...", flush=True)
    _ = multitoken_scores_subset(bb, texts[0], labels, list(ranking[0, :5]), norm=args.norm)

    results = {}  # k -> dict(acc, ece, macro_f1, p50_rerank_ms, p50_total_ms, preds)
    for k in ks:
        kk = min(k, n_labels)
        preds = []
        rerank_latencies = []
        for i, r in enumerate(rows):
            cand_idx = list(ranking[i, :kk])
            t = time.perf_counter()
            scores = multitoken_scores_subset(bb, r["text"], labels, cand_idx, norm=args.norm)
            rerank_lat = time.perf_counter() - t
            rerank_latencies.append(rerank_lat)

            # Softmax over the k candidate scores; map back to full 72-label prob vector so the
            # metric code (ece, nll) sees a well-formed distribution over all labels.
            z = scores.astype(np.float64)
            pz = np.exp(z - z.max())
            pz = pz / pz.sum()
            full_p = np.full(n_labels, 1e-9, dtype=np.float64)
            for j, ci in enumerate(cand_idx):
                full_p[ci] = pz[j]
            full_p = full_p / full_p.sum()
            pred_idx = int(cand_idx[int(np.argmax(pz))])

            total_lat = e5_encode_ms_per_row / 1000 + rerank_lat
            preds.append(Prediction(
                experiment_id=f"hybrid-e5-rerank-k{k}", backend="mlx_llada",
                model_requested="llada2-mini-4bit", model_resolved="llada2-mini-4bit",
                dataset=DATASET, example_id=r["example_id"], target_index=r["target_index"],
                predicted_index=pred_idx, labels=tuple(r["labels"]),
                probabilities=tuple(float(x) for x in full_p), latency_seconds=total_lat,
                probability_sum_raw=float(full_p.sum())))

        scored = score_predictions(preds)
        p50_rerank = float(np.median(rerank_latencies) * 1000)
        p50_total = p50_rerank + e5_encode_ms_per_row
        results[k] = dict(acc=scored["accuracy"], ece=scored["ece"], macro_f1=scored["macro_f1"],
                          nll=scored["nll"], mean_conf=scored["mean_confidence"],
                          p50_rerank_ms=p50_rerank, p50_total_ms=p50_total, preds=preds)
        print(f"  k={k:>3}: hybrid acc {scored['acc'] if 'acc' in scored else scored['accuracy']:.3f}"
              f"  recall@k {recall_at_k[k]:.3f}  rerank p50 {p50_rerank:.0f}ms", flush=True)

    # --- Write predictions + report -------------------------------------------------------------
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "predictions.jsonl", "w") as f:
        for k in ks:
            for p in results[k]["preds"]:
                f.write(json.dumps(p.__dict__, default=str) + "\n")

    best_k = max(ks, key=lambda k: results[k]["acc"])
    best_acc = results[best_k]["acc"]

    lines = [
        "# Exploration E5 — Hybrid retrieval (e5 top-k) + re-rank (multi-token LLaDA)",
        "",
        "Model (re-rank): `mlx-community/LLaDA2.0-mini-preview-4bit` (MLX LLaDA2-MoE), read at [MASK].  ",
        f"Retriever: `{args.e5_model}` (sentence-transformers), cosine on L2-normalized embeddings, "
        "e5 `query:` / `passage:` prefixes.  ",
        f"Dataset: Banking77, {len(rows)} rows of the Jev pilot manifest, {n_labels} labels. "
        f"Length norm: `{args.norm}`.",
        "",
        "Two stages: (1) e5 embeds the 72 label descriptions once and each input; cosine keeps the "
        "top-k labels. (2) the multi-token LLaDA readout scores ONLY those k candidates (one "
        "bidirectional forward over their distinctive-suffix mask block) and takes the argmax. "
        "`k=72` = re-rank all labels (no retrieval pruning).",
        "",
        "## Reference points",
        "",
        f"- Multi-token readout, all 72 labels (explore-1): **acc {BASE_MULTITOKEN_ACC:.2f}**, "
        f"ECE {BASE_MULTITOKEN_ECE:.3f}, p50 {BASE_MULTITOKEN_P50_MS:.0f} ms.",
        f"- e5 alone (top-1 cosine): **acc {e5_acc:.3f}**, encode {e5_encode_ms_per_row:.1f} ms/row.",
        "- Repo upper bound (embedding wemm-4b): ~0.88 zero-shot, ~0.94 + autotune.",
        "",
        "## Results by k",
        "",
        "| k | recall@k (ceiling) | hybrid acc | e5-alone acc | Δ vs multi-token 0.19 | Δ vs e5 alone | "
        "re-rank p50 ms | total p50 ms | hybrid ECE | macro-F1 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for k in ks:
        r = results[k]
        d_mt = r["acc"] - BASE_MULTITOKEN_ACC
        d_e5 = r["acc"] - e5_acc
        lines.append(
            f"| {k} | {recall_at_k[k]:.3f} | {r['acc']:.3f} | {e5_acc:.3f} | {d_mt:+.3f} | "
            f"{d_e5:+.3f} | {r['p50_rerank_ms']:.0f} | {r['p50_total_ms']:.0f} | {r['ece']:.3f} | "
            f"{r['macro_f1']:.3f} |"
        )

    lines += [
        "",
        f"e5 query encode: {e5_encode_ms_per_row:.1f} ms/row (amortized; label passages encoded once). "
        f"Total p50 = e5 encode + LLaDA re-rank.",
        "",
        "## Step back — what actually moves the needle",
        "",
    ]

    # Programmatic verdict text driven by the measured numbers.
    hybrid_beats_mt = best_acc > BASE_MULTITOKEN_ACC
    hybrid_beats_e5 = best_acc > e5_acc + 1e-9
    # Does re-rank help vs picking e5 top-1 among the same candidates? Compare best hybrid to e5.
    rerank_adds = best_acc - e5_acc
    ceiling_k = max(ks, key=lambda k: recall_at_k[k])
    ceiling = recall_at_k[ceiling_k]

    lines += [
        f"- **Hybrid vs multi-token-only (0.19)**: best hybrid acc {best_acc:.3f} at k={best_k} "
        f"({'beats' if hybrid_beats_mt else 'does NOT beat'} 0.19, "
        f"Δ {best_acc - BASE_MULTITOKEN_ACC:+.3f}). Pruning 72→k concentrates the multi-token read "
        "on fewer, sharper candidates.",
        f"- **Hybrid vs e5 alone ({e5_acc:.3f})**: the LLaDA re-rank "
        f"{'adds' if hybrid_beats_e5 else 'does NOT add'} accuracy on top of e5 "
        f"(best Δ {rerank_adds:+.3f}). "
        + ("The re-rank is doing real work beyond the retriever." if hybrid_beats_e5 else
           "e5 is effectively doing the work; the LLaDA re-rank does not improve on the retriever's "
           "top-1 and mostly re-shuffles within the top-k without net gain."),
        f"- **Recall@k ceiling**: recall@{ceiling_k} = {ceiling:.3f} is the hard upper bound any "
        f"re-rank can reach at that k (the gold label must be retrieved to be re-ranked). At small k "
        "the ceiling caps accuracy; the gap between hybrid acc and recall@k is the re-rank's own "
        "error inside the candidate set.",
        f"- **k=72 (no pruning, pure re-rank)**: acc {results[72]['acc']:.3f} if present — isolates "
        "the effect of the candidate restriction vs scoring all labels."
        if 72 in ks else "",
        "",
        "## Verdict (5 lines)",
        "",
    ]

    # 5-line summary.
    if hybrid_beats_e5 and hybrid_beats_mt:
        verdict = "à creuser (l'hybride bat e5 seul ET le multi-token seul)"
    elif hybrid_beats_mt and not hybrid_beats_e5:
        verdict = "rejeter l'hybride, adopter e5 seul (le re-rank LLaDA n'ajoute rien au-dessus de e5)"
    else:
        verdict = "rejeter (l'hybride ne bat pas la baseline)"

    lines += [
        f"1. Gain vs multi-token 0.19 : meilleur hybride {best_acc:.3f} (k={best_k}), "
        f"Δ {best_acc - BASE_MULTITOKEN_ACC:+.3f}.",
        f"2. e5 seul (top-1) fait déjà {e5_acc:.3f} ; borne recall@{ceiling_k} = {ceiling:.3f}.",
        f"3. Apport net du re-rank LLaDA au-dessus de e5 : Δ {rerank_adds:+.3f}.",
        f"4. Coût : e5 encode {e5_encode_ms_per_row:.1f} ms/row + re-rank LLaDA "
        f"{results[best_k]['p50_rerank_ms']:.0f} ms (k={best_k}), total p50 "
        f"{results[best_k]['p50_total_ms']:.0f} ms/row.",
        f"5. Verdict : {verdict}.",
        "",
    ]
    lines = [ln for ln in lines if ln != ""] if False else lines  # keep blanks for markdown

    report = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
