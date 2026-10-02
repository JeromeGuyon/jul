"""Decontaminate the Mapika corpus against decision-bench bench-v1 (exact + near-duplicate states).

Mapika decontaminated against their OWN eval sets, not against this bench. We drop any training item
whose state exactly matches, or shares a high fraction of 8-word shingles with, any bench state.
"""
import json
import re
import sys
from pathlib import Path

BENCH = Path(os.environ.get("JUL_DBENCH", os.path.expanduser("~/dev/decision-bench"))) / "data/bench-v1.jsonl"
IN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/mix/mapika_teacher.jsonl")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else IN.with_suffix(".clean.jsonl")
NEAR_THRESHOLD = 0.5   # drop if >=50% of an item's shingles are seen in the bench


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def shingles(s: str, k: int = 8):
    w = norm(s).split()
    if len(w) < k:
        return {tuple(w)} if w else set()
    return {tuple(w[i:i + k]) for i in range(len(w) - k + 1)}


def main():
    bench_states = [norm(json.loads(l)["state"]) if isinstance(json.loads(l)["state"], str)
                    else norm(json.dumps(json.loads(l)["state"])) for l in open(BENCH)]
    bench_exact = set(bench_states)
    bench_shingles = set()
    for s in bench_states:
        bench_shingles |= shingles(s)
    print(f"bench: {len(bench_exact)} states, {len(bench_shingles)} shingles")

    kept = exact = near = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for line in open(IN, encoding="utf-8"):
            r = json.loads(line)
            st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"])
            ns = norm(st)
            if ns in bench_exact:
                exact += 1
                continue
            sh = shingles(st)
            if sh:
                overlap = sum(1 for g in sh if g in bench_shingles) / len(sh)
                if overlap >= NEAR_THRESHOLD:
                    near += 1
                    continue
            out.write(line if line.endswith("\n") else line + "\n")
            kept += 1
    total = kept + exact + near
    print(f"in={total} kept={kept} dropped: exact={exact} near={near} "
          f"({100 * (exact + near) / total:.2f}% removed)")
    print("clean ->", OUT)


if __name__ == "__main__":
    main()
