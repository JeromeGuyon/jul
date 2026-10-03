"""Convert Kev's decision-v7 train partition into the decision-bench training format.

Shape: each record has a `state` and a `questions` dict {name: {type, instructions, criteria, label, src}}.
  - score: criteria is a list of level descriptions; label is the integer index.
  - noul:  criteria is {"true": desc, "false": desc}; label is a bool.
  - choice: criteria is {key: desc}; label is a key (or index).
We emit one bench item per question, carrying `source=kev:<src>` so score_split can bucket SEEN/UNSEEN.
"""
import json
import sys
from pathlib import Path

IN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/kev/decision-v7-train.jsonl")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/mix/decision-v7.jsonl")


def _flat_instr(instr):
    """Instructions may be a string or {question, focus, ...}; render to a single string."""
    if isinstance(instr, str):
        return instr
    if isinstance(instr, dict):
        parts = [str(instr[k]) for k in ("question", "focus") if instr.get(k)]
        return " ".join(parts) if parts else " ".join(str(v) for v in instr.values() if v)
    return str(instr) if instr is not None else ""


def _flat_desc(v):
    """An option description may be null, a string, or {what: ...}; render to a string."""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        return str(v.get("what") or " ".join(str(x) for x in v.values() if x))
    return str(v)


def emit(out, state, qtype, instr, options, gold, src):
    instr = _flat_instr(instr)
    if state is not None and not isinstance(state, str):
        from jul.mask import _render  # same state rendering as training/inference
        state = _render(state)
    if not isinstance(state, str) or not instr or gold is None or not options:
        return 0
    rec = {"state": state, "type": qtype, "instructions": instr, "options": options,
           "gold": gold, "family": "other", "lang": "en", "source": f"kev:{src}",
           "license": "public", "split_hint": "train"}
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 1


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    skipped = 0
    with open(OUT, "w", encoding="utf-8") as out:
        for line in open(IN, encoding="utf-8"):
            r = json.loads(line)
            state = r.get("state")
            for q in r.get("questions", {}).values():
                t = q.get("type")
                instr = q.get("instructions", "")
                crit = q.get("criteria")
                lab = q.get("label")
                src = q.get("src", "unknown")
                if t == "score":
                    if not isinstance(crit, list) or len(crit) < 2:
                        skipped += 1
                        continue
                    options = {str(i): _flat_desc(d) for i, d in enumerate(crit)}
                    try:
                        gi = int(lab)
                    except (TypeError, ValueError):
                        skipped += 1
                        continue
                    if 0 <= gi < len(options):
                        n += emit(out, state, "score", instr, options, str(gi), src)
                elif t == "noul":
                    if isinstance(crit, dict):
                        options = {"true": crit.get("true", "Yes."), "false": crit.get("false", "No.")}
                    else:
                        options = {"true": "Yes.", "false": "No."}
                    gold = "true" if (lab is True or lab == "true") else "false"
                    n += emit(out, state, "noul", instr, options, gold, src)
                elif t == "choice":
                    if isinstance(crit, dict):
                        options = {k: _flat_desc(v) for k, v in crit.items()}
                        gold = str(lab)
                    elif isinstance(crit, list):
                        options = {str(i): _flat_desc(d) for i, d in enumerate(crit)}
                        try:
                            gold = str(int(lab))
                        except (TypeError, ValueError):
                            gold = str(lab)
                    else:
                        skipped += 1
                        continue
                    if gold not in options:
                        skipped += 1
                        continue
                    n += emit(out, state, "choice", instr, options, gold, src)
    print(f"converted -> {OUT} | {n} items (skipped {skipped})")
    from collections import Counter
    c = Counter(json.loads(l)["type"] for l in open(OUT))
    print("by type:", dict(c))


if __name__ == "__main__":
    main()
