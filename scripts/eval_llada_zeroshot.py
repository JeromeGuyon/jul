"""Zero-shot evaluation of a LLaDA masked-diffusion model, read at [MASK] (Option 1).

This is the cheap gate of the "build a Jev-like" decision tree: does a masked-*diffusion* backbone,
read natively at a single [MASK], generalize in zero-shot better than a causal decoder + trained
pointer head, at equal corpus/holdout/calibration? It changes ONE variable — the backbone family —
and measures accuracy + ECE on the *sealed* holdout, per family, with a separate leave-one-family-out
line (the true zero-shot signal).

It mirrors the working project's `scripts/eval_holdout.py` metrics exactly so results are comparable:
argmax accuracy and Expected Calibration Error over 10 confidence bins, per family and overall.

Holdout format: JSONL, one record per line, with the fields of a MixExample:
    {"family": str, "type": "choice"|"noul"|"score", "instructions": str,
     "state": <str|obj>, "options": {key: description, ...}, "gold": key,
     "split_hint": "eval_only"?  (optional; marks leave-one-family-out families)}

Usage (points at the working project's sealed holdout by default):
    JUL_LLADA_MASK_ID=126336 python scripts/eval_llada_zeroshot.py \
        --model llada-8b-instruct \
        --holdout data/eval-holdout-v2/holdout.jsonl \
        --json runs/llada-8b-instruct-eval-v2.json

For iLLaDA:  --model illada-8b-instruct   (and JUL_LLADA_MASK_ID=5 if needed).
CPU/MPS work but 8B is slow there; a single GPU is recommended. Nothing is trained here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


# --- holdout reading (self-contained: jul-bis has no mix_common) ------------------------------

def read_holdout(path: str):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            yield json.loads(line)


def to_question(ex):
    from jul import Choice, Noul, Score, NoulCriteria
    t = ex["type"]
    if t == "choice":
        return Choice(instructions=ex["instructions"], criteria=dict(ex["options"]))
    if t == "noul":
        o = ex["options"]
        return Noul(instructions=ex["instructions"],
                    criteria=NoulCriteria(true=o.get("true", "Yes."), false=o.get("false", "No.")))
    if t == "score":
        levels = [ex["options"][k] for k in sorted(ex["options"], key=lambda x: int(x))]
        return Score(instructions=ex["instructions"], criteria=levels)
    raise ValueError(t)


def predicted(ex, resp):
    """(pred_key, p_pred, p_gold) — identical to eval_holdout.py."""
    ans = resp.answers["q"]
    t = ex["type"]
    if t == "choice":
        probs = ans.probabilities
        pred = max(probs, key=probs.get)
        return pred, probs[pred], probs.get(ex["gold"], 0.0)
    if t == "noul":
        p_true = ans.noul
        pred = "true" if p_true >= 0.5 else "false"
        p_gold = p_true if ex["gold"] == "true" else 1.0 - p_true
        return pred, max(p_true, 1.0 - p_true), p_gold
    if t == "score":
        probs = ans.probabilities
        pred = max(probs, key=probs.get)
        return str(pred), probs[pred], probs.get(ex["gold"], 0.0)
    raise ValueError(t)


def ece(confidences, corrects, n_bins=10):
    if not confidences:
        return 0.0
    bins = [[] for _ in range(n_bins)]
    for c, ok in zip(confidences, corrects):
        b = min(n_bins - 1, int(c * n_bins))
        bins[b].append((c, ok))
    total, e = len(confidences), 0.0
    for b in bins:
        if not b:
            continue
        conf = sum(c for c, _ in b) / len(b)
        acc = sum(ok for _, ok in b) / len(b)
        e += (len(b) / total) * abs(acc - conf)
    return e


def evaluate(model, holdout, backend, limit=None):
    from jul import TypeSafeClient
    client = (TypeSafeClient(model=model, backend=backend) if backend
              else TypeSafeClient(model=model))
    per_family = defaultdict(lambda: {"n": 0, "correct": 0, "conf": [], "ok": []})
    eval_only = set()
    rows_in = list(read_holdout(holdout))
    if limit:
        rows_in = rows_in[:limit]
    import time as _time
    latencies = []
    for ex in rows_in:
        if ex.get("split_hint") == "eval_only":
            eval_only.add(ex["family"])
        _t = _time.perf_counter()
        resp = client.system_one(state=ex["state"], questions={"q": to_question(ex)})
        latencies.append(_time.perf_counter() - _t)
        pred, p_pred, _ = predicted(ex, resp)
        ok = int(pred == ex["gold"])
        f = per_family[ex["family"]]
        f["n"] += 1; f["correct"] += ok; f["conf"].append(p_pred); f["ok"].append(ok)

    def summarize(fams):
        rows, N, C, allconf, allok = [], 0, 0, [], []
        for fam in fams:
            d = per_family[fam]
            if not d["n"]:
                continue
            rows.append((fam, d["n"], d["correct"] / d["n"], ece(d["conf"], d["ok"])))
            N += d["n"]; C += d["correct"]; allconf += d["conf"]; allok += d["ok"]
        return rows, N, (C / N if N else 0.0), ece(allconf, allok)

    train_fams = [f for f in per_family if f not in eval_only]
    lat = sorted(latencies)
    def pct(p):
        return round(lat[min(len(lat) - 1, int(p * len(lat)))], 4) if lat else 0.0
    latency = {"n": len(lat), "mean_s": round(sum(lat) / len(lat), 4) if lat else 0.0,
               "p50_s": pct(0.50), "p95_s": pct(0.95),
               "n_mask": int(os.environ.get("JUL_LLADA_N_MASK", "1")),
               "n_steps": int(os.environ.get("JUL_LLADA_N_STEPS", "1"))}
    return {"model": model, "in_mix": summarize(train_fams),
            "leave_one_family_out": summarize(list(eval_only)),
            "eval_only_families": sorted(eval_only), "latency": latency}


def print_report(rep):
    def block(title, res):
        rows, n, acc, e = res
        print(f"\n{title}  (n={n})  acc={acc:.3f}  ECE={e:.3f}", file=sys.stderr)
        for fam, fn, facc, fe in sorted(rows, key=lambda r: -r[2]):
            print(f"  {fam:16} n={fn:4}  acc={facc:.3f}  ECE={fe:.3f}", file=sys.stderr)
    print(f"=== {rep['model']} on holdout (mask readout, zero-shot) ===", file=sys.stderr)
    lat = rep.get("latency")
    if lat:
        print(f"latency: mean={lat['mean_s']}s p50={lat['p50_s']}s p95={lat['p95_s']}s "
              f"(n_mask={lat['n_mask']} n_steps={lat['n_steps']}, n={lat['n']})", file=sys.stderr)
    block("IN-MIX FAMILIES", rep["in_mix"])
    if rep["eval_only_families"]:
        block(f"LEAVE-ONE-FAMILY-OUT {rep['eval_only_families']} (true zero-shot)",
              rep["leave_one_family_out"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="llada-8b-instruct", help="llada-8b-instruct | illada-8b-instruct")
    ap.add_argument("--holdout", default="data/eval-holdout-v2/holdout.jsonl")
    ap.add_argument("--backend", default="llada")
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N records (quick signal)")
    ap.add_argument("--json", help="also write the report as JSON here")
    args = ap.parse_args()
    if not Path(args.holdout).exists():
        raise SystemExit(f"holdout not found: {args.holdout}")
    rep = evaluate(args.model, args.holdout, args.backend, args.limit)
    print_report(rep)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nreport -> {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
