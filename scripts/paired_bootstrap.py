"""Paired bootstrap of (learned head) - (logits readout) accuracy on the SAME bench items.

Both runs predict the same 2159 items. We pair per item, restrict to UNSEEN sources, and bootstrap the
difference in accuracy (head - logits) overall and per type. Paired = we resample items (not runs), so
the correlation between the two systems on the same item is preserved — the right test for "is +5.6 on
score real or 2-3 items of noise".
"""
import json
import sys
import numpy as np
sys.path.insert(0, "/Users/jerome/dev/decision-bench")
from decision_bench.scoring import judge

BENCH = "/Users/jerome/dev/decision-bench/data/bench-v1.jsonl"
LOGITS = "runs/v7/predictions.jsonl"
HEAD = "runs/head2/predictions.jsonl"
SEEN = {"allocine", "civil_comments", "clinc", "mnli", "xnli_fr", "mnli_mt_fr", "go_emotions",
        "pawsx", "dbpedia", "banking77", "agnews", "trec", "imdb", "sst5", "boolq", "amazon", "yelp"}


def base(s):
    return (s or "").split(":")[0].split("+")[0].strip()


def load(path):
    return {json.loads(l)["id"]: json.loads(l) for l in open(path)}


def verdict(rec, item):
    """Official per-item verdict via decision_bench.judge; None if unscored (unknown gold)."""
    if rec is None:
        return None
    return judge(item, None if rec.get("error") else rec.get("pred"))


def main():
    items = load(BENCH)
    lo = load(LOGITS)
    hd = load(HEAD)
    rng = np.random.default_rng(0)

    rows = {"all": [], "choice": [], "noul": [], "score": []}
    for i, it in items.items():
        if base(it.get("source", "")) in SEEN:
            continue
        if i not in lo or i not in hd:
            continue
        cl = verdict(lo[i], it)
        ch = verdict(hd[i], it)
        if cl is None or ch is None:   # unknown-gold / skipped in either run
            continue
        rows["all"].append((int(cl), int(ch)))
        rows[it["type"]].append((int(cl), int(ch)))

    def boot(pairs, n=10000):
        a = np.array(pairs, dtype=float)  # (N, 2): col0 logits, col1 head
        if len(a) == 0:
            return None
        N = len(a)
        diffs = []
        for _ in range(n):
            idx = rng.integers(0, N, N)
            s = a[idx]
            diffs.append(s[:, 1].mean() - s[:, 0].mean())
        diffs = np.array(diffs)
        return (a[:, 0].mean(), a[:, 1].mean(), diffs.mean(),
                np.percentile(diffs, 2.5), np.percentile(diffs, 97.5),
                float((diffs > 0).mean()), N)

    print(f"{'type':8} {'n':>5} {'logits':>7} {'head':>7} {'Δ(head-logits)':>15} {'95% CI':>20} {'P(Δ>0)':>7}")
    for t in ("all", "choice", "noul", "score"):
        r = boot(rows[t])
        if r is None:
            continue
        lo_acc, hd_acc, dm, lo_ci, hi_ci, pgt, N = r
        print(f"{t:8} {N:5d} {lo_acc:7.3f} {hd_acc:7.3f} {dm:+15.3f}  [{lo_ci:+.3f},{hi_ci:+.3f}] {pgt:7.2f}")


if __name__ == "__main__":
    main()
