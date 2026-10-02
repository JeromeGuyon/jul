"""EXPLORATION 2 — Validate anchor-only against full-vocab on the real project holdout (439 ex).

Goal: prove that reading a typed System One decision with `option_logits` (anchor-only output
projection) yields EXACTLY the same decision quality as the baseline `mask_logits` (full 157k-way
projection then slice), while cutting latency — and measure the latency win per family.

Zero quality risk: for every example we run BOTH readouts on the identical token sequence, mask
positions and anchors (the real MaskReader reading path, n_mask=1/n_steps=1), compare the option
logits bit-for-bit (max abs diff, argmax equality), and time each. Accuracy/ECE are computed from
both paths and must be identical.

We do NOT modify any library file. We reuse jul.mask.MaskReader to build the exact prompt tokens /
anchors (so the comparison is faithful to production), then call the backbone's two readouts
directly.

Usage:
  PYTHONPATH=lib python scripts/explore_2_anchor_holdout.py \
      [--sample 150] [--holdout <path>] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT))

from jul.backbone import Backbone  # noqa: E402
from jul.mask import MaskReader, MaskSpec, _markers  # noqa: E402
from jul.types import Option  # noqa: E402

DEFAULT_HOLDOUT = "data/eval-holdout-v2/holdout.jsonl"
OUT = ROOT / "runs" / "explore-2"


# --- holdout reading (self-contained, mirrors eval_llada_zeroshot.read_holdout) ----------------

def read_holdout(path: str):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _render_state(state):
    """State is a str in this holdout; render objects to labeled text if not."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def question_tuple(ex):
    """Build the MaskReader (kind, instructions, options, opt_keys) from a holdout record.

    Mirrors eval_llada_zeroshot.to_question -> the client's option_of ordering:
      choice: Option(key, text) in dict order.
      noul  : Option('true', ...), Option('false', ...); gold is 'true'/'false'.
      score : levels sorted by int(key); Option(str(i), level_text).
    """
    t = ex["type"]
    o = ex["options"]
    if t == "choice":
        options = [Option(k, v) for k, v in o.items()]
        return "choice", ex["instructions"], options, list(o.keys())
    if t == "noul":
        options = [Option("true", o.get("true", "Yes.")), Option("false", o.get("false", "No."))]
        return "noul", ex["instructions"], options, ["true", "false"]
    if t == "score":
        keys = sorted(o, key=lambda x: int(x))
        options = [Option(str(k), o[k]) for k in keys]
        return "score", ex["instructions"], options, [str(k) for k in keys]
    raise ValueError(t)


def probs_from_logits(z: np.ndarray) -> np.ndarray:
    p = np.exp(z - z.max())
    return p / p.sum()


def pred_key_and_conf(kind, options, opt_keys, z: np.ndarray):
    """Return (pred_key, p_pred, key_to_prob) matching eval_llada_zeroshot.predicted semantics."""
    p = probs_from_logits(z)
    if kind == "noul":
        key_to_prob = {options[i].key: float(p[i]) for i in range(len(options))}
        p_true = key_to_prob["true"]
        pred = "true" if p_true >= 0.5 else "false"
        conf = max(p_true, 1.0 - p_true)
        return pred, conf, key_to_prob
    key_to_prob = {opt_keys[i]: float(p[i]) for i in range(len(opt_keys))}
    pred = max(key_to_prob, key=key_to_prob.get)
    return pred, key_to_prob[pred], key_to_prob


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=DEFAULT_HOLDOUT)
    ap.add_argument("--sample", type=int, default=150, help=">=100; stratified across families")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not Path(args.holdout).exists():
        raise SystemExit(f"holdout not found: {args.holdout}")

    rows = read_holdout(args.holdout)
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r["family"]].append(r)
    rng = random.Random(args.seed)
    for f in by_fam:
        rng.shuffle(by_fam[f])
    fams = sorted(by_fam)
    chosen = []
    i = 0
    while len(chosen) < args.sample and any(by_fam[f][i:] for f in fams):
        for f in fams:
            if i < len(by_fam[f]) and len(chosen) < args.sample:
                chosen.append(by_fam[f][i])
        i += 1
    print(f"sampled {len(chosen)} examples across {len(fams)} families", flush=True)

    print("loading backbone (MLX LLaDA2-mini-4bit)...", flush=True)
    bb = Backbone("llada2-mini-4bit", "mlx_llada")
    reader = MaskReader(bb, MaskSpec.default())

    # Warmup (compile MLX graph, exclude from timings).
    w = chosen[0]
    kind, instr, options, opt_keys = question_tuple(w)
    markers = _markers(kind, options, reader.spec.markers)
    anchors = [reader._anchor_id(m) for m in markers]
    tokens, mask_positions = reader._prompt_tokens(kind, instr, markers, options,
                                                   _render_state(w["state"]))
    _ = bb.mask_logits(tokens, list(mask_positions))[:, anchors]
    _ = bb.option_logits(tokens, list(mask_positions), anchors)

    max_abs_diff = 0.0
    argmax_mismatches = 0
    n = 0
    stats = {"full": defaultdict(lambda: {"conf": [], "ok": []}),
             "anchor": defaultdict(lambda: {"conf": [], "ok": []})}
    lat = {"full": defaultdict(list), "anchor": defaultdict(list)}

    for ex in chosen:
        kind, instr, options, opt_keys = question_tuple(ex)
        markers = _markers(kind, options, reader.spec.markers)
        anchors = [reader._anchor_id(m) for m in markers]
        tokens, mask_positions = reader._prompt_tokens(kind, instr, markers, options,
                                                       _render_state(ex["state"]))
        mp = list(mask_positions)

        # (a) full-vocab: mask_logits then slice anchors
        t0 = time.perf_counter()
        vocab = bb.mask_logits(tokens, mp)          # (P, V)
        z_full_raw = vocab[:, anchors]              # (P, A)
        t_full = time.perf_counter() - t0

        # (b) anchor-only
        t0 = time.perf_counter()
        z_anchor_raw = bb.option_logits(tokens, mp, anchors)  # (P, A)
        t_anchor = time.perf_counter() - t0

        if not (np.isfinite(z_full_raw).all() and np.isfinite(z_anchor_raw).all()):
            raise FloatingPointError(f"non-finite logits for example {ex.get('family')}")

        d = float(np.max(np.abs(z_full_raw - z_anchor_raw)))
        max_abs_diff = max(max_abs_diff, d)

        z_full = z_full_raw.mean(axis=0) / reader.spec.temperature
        z_anchor = z_anchor_raw.mean(axis=0) / reader.spec.temperature
        z_full = reader._to_option_order(kind, options, z_full)
        z_anchor = reader._to_option_order(kind, options, z_anchor)

        gold = ex["gold"]
        fam = ex["family"]
        for path, z in (("full", z_full), ("anchor", z_anchor)):
            pred, conf, _ = pred_key_and_conf(kind, options, opt_keys, z)
            ok = int(pred == gold)
            stats[path][fam]["conf"].append(conf)
            stats[path][fam]["ok"].append(ok)
        pf, _, _ = pred_key_and_conf(kind, options, opt_keys, z_full)
        pa, _, _ = pred_key_and_conf(kind, options, opt_keys, z_anchor)
        if pf != pa:
            argmax_mismatches += 1

        lat["full"][fam].append(t_full)
        lat["anchor"][fam].append(t_anchor)
        n += 1

    def acc_ece(path):
        allconf, allok = [], []
        per = {}
        for fam, d in stats[path].items():
            if not d["ok"]:
                continue
            acc = sum(d["ok"]) / len(d["ok"])
            per[fam] = (len(d["ok"]), acc, ece(d["conf"], d["ok"]))
            allconf += d["conf"]; allok += d["ok"]
        overall_acc = sum(allok) / len(allok) if allok else 0.0
        return per, overall_acc, ece(allconf, allok)

    per_full, acc_full, ece_full = acc_ece("full")
    per_anchor, acc_anchor, ece_anchor = acc_ece("anchor")

    def p50(vals):
        v = sorted(vals)
        return v[len(v) // 2] if v else 0.0

    fams_sorted = sorted(per_full)
    OUT.mkdir(parents=True, exist_ok=True)

    L = []
    L.append("# Exploration 2 — anchor-only vs full-vocab on the real holdout (439-ex holdout-v2)")
    L.append("")
    L.append(f"Sample: **{n}** examples, stratified across **{len(fams_sorted)}** families "
             f"(seed={args.seed}). Reading path: real `MaskReader` prompt/anchors, n_mask=1, n_steps=1.")
    L.append("")
    L.append("Both readouts run on the **identical** token sequence, mask positions and anchors:")
    L.append("- **full-vocab**: `bb.mask_logits(tokens, positions)[:, anchors]` (157k-way projection, then slice).")
    L.append("- **anchor-only**: `bb.option_logits(tokens, positions, anchors)` (project only the anchor rows of `lm_head`).")
    L.append("")
    L.append("## Equivalence proof")
    L.append("")
    L.append(f"- Max abs diff of raw anchor logits over all {n} examples: **{max_abs_diff:.3e}**")
    L.append(f"- Decision (argmax) mismatches: **{argmax_mismatches} / {n}**")
    L.append(f"- Overall accuracy — full: **{acc_full:.4f}**, anchor: **{acc_anchor:.4f}** "
             f"(delta {acc_anchor - acc_full:+.4f})")
    L.append(f"- Overall ECE — full: **{ece_full:.4f}**, anchor: **{ece_anchor:.4f}** "
             f"(delta {ece_anchor - ece_full:+.4f})")
    L.append("- All logits finite: **yes** (checked every example)")
    L.append("")
    L.append("## Per-family: accuracy / ECE (must be identical) + latency (full vs anchor)")
    L.append("")
    L.append("| Family | n | acc full | acc anchor | ECE full | ECE anchor | "
             "p50 full (ms) | p50 anchor (ms) | speedup |")
    L.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    tot_full_ms, tot_anchor_ms = [], []
    for fam in fams_sorted:
        cnt, af, ef = per_full[fam]
        _, aa, ea = per_anchor[fam]
        lf = p50(lat["full"][fam]) * 1000
        la = p50(lat["anchor"][fam]) * 1000
        tot_full_ms += [x * 1000 for x in lat["full"][fam]]
        tot_anchor_ms += [x * 1000 for x in lat["anchor"][fam]]
        sp = (lf / la) if la else 0.0
        L.append(f"| {fam} | {cnt} | {af:.3f} | {aa:.3f} | {ef:.3f} | {ea:.3f} | "
                 f"{lf:.0f} | {la:.0f} | {sp:.2f}x |")
    gf = p50(tot_full_ms)
    ga = p50(tot_anchor_ms)
    L.append(f"| **ALL** | {n} | {acc_full:.3f} | {acc_anchor:.3f} | {ece_full:.3f} | {ece_anchor:.3f} | "
             f"{gf:.0f} | {ga:.0f} | {(gf/ga) if ga else 0:.2f}x |")
    L.append("")
    L.append(f"Global p50 latency: full **{gf:.0f} ms**, anchor **{ga:.0f} ms** "
             f"-> **{(gf/ga - 1) * 100:.0f}%** faster, saving **{gf - ga:.0f} ms/decision**.")
    L.append("")

    report = "\n".join(L) + "\n"
    (OUT / "report.md").write_text(report)
    (OUT / "summary.json").write_text(json.dumps({
        "n": n, "families": fams_sorted, "max_abs_diff": max_abs_diff,
        "argmax_mismatches": argmax_mismatches,
        "acc_full": acc_full, "acc_anchor": acc_anchor,
        "ece_full": ece_full, "ece_anchor": ece_anchor,
        "p50_full_ms": gf, "p50_anchor_ms": ga,
        "per_family": {f: {"n": per_full[f][0], "acc_full": per_full[f][1],
                           "acc_anchor": per_anchor[f][1],
                           "p50_full_ms": p50(lat["full"][f]) * 1000,
                           "p50_anchor_ms": p50(lat["anchor"][f]) * 1000}
                       for f in fams_sorted},
    }, indent=2) + "\n")
    print("\n" + report)
    print(f"report -> {OUT / 'report.md'}")


if __name__ == "__main__":
    main()
