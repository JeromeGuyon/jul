"""Convert Mapika/decider teacher_data/*.jsonl into the decision-bench training format.

Mapika's recipe (decider v2.1 step 2) mixes these generated families with public sets + replay. We
can reconstruct the GENERATED portion directly from the repo's teacher_data (no HF downloads), convert
it to {state,type,instructions,options,gold,soft}, and (next step) decontaminate against bench-v1.

Two shapes:
  - custom_questions / routing_* / commands: one record = a state + a `questions` list (each with
    type, instructions, criteria, answer, teacher_p). criteria is a dict{key: desc|null} or null.
  - situations: a state (`situation`) + a single choice (`options` list, `answer` is the index).
"""
import json
import sys
from pathlib import Path

SRC = Path("/tmp/decider/teacher_data")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/mix/mapika_teacher.jsonl")


def emit(out, state, qtype, instr, options, gold, soft=None, src="mapika"):
    if not state or not instr or gold is None:
        return 0
    rec = {"state": state, "type": qtype, "instructions": instr, "options": options,
           "gold": gold, "family": "other", "lang": "en", "source": f"mapika:{src}",
           "license": "synthetic", "split_hint": "train"}
    if soft is not None:
        rec["soft"] = soft
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return 1


def conv_questions(path, out):
    n = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        state = r.get("state")
        for q in r.get("questions", []):
            t = q.get("type")
            instr = q.get("instructions", "")
            crit = q.get("criteria")
            ans = q.get("answer")
            tp = q.get("teacher_p")
            if t == "noul":
                options = {"true": "Yes.", "false": "No."}
                gold = "true" if (ans is True or ans == "true") else "false"
                soft = None
                if isinstance(tp, (int, float)):
                    pyes = tp if gold == "true" else 1 - tp
                    soft = {"true": round(pyes, 4), "false": round(1 - pyes, 4)}
                n += emit(out, state, "noul", instr, options, gold, soft, path.stem)
            elif t == "choice":
                if isinstance(crit, dict):
                    options = {k: (v or "") for k, v in crit.items()}
                elif isinstance(crit, list):
                    options = {str(x): "" for x in crit}
                else:
                    continue
                if ans not in options:
                    continue
                n += emit(out, state, "choice", instr, options, str(ans), None, path.stem)
            elif t == "score":
                if isinstance(crit, list) and len(crit) >= 2:
                    options = {str(i): d for i, d in enumerate(crit)}
                    try:
                        gi = int(ans)
                    except (TypeError, ValueError):
                        continue
                    if 0 <= gi < len(options):
                        n += emit(out, state, "score", instr, options, str(gi), None, path.stem)
    return n


def conv_situations(path, out):
    n = 0
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        state = r.get("situation")
        opts = r.get("options") or []
        ans = r.get("answer")
        if not opts or not isinstance(ans, int) or not (0 <= ans < len(opts)):
            continue
        options = {str(i): o for i, o in enumerate(opts)}
        n += emit(out, state, "choice", r.get("question", "What is the best action?"),
                  options, str(ans), None, "situations")
    return n


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    total = {}
    with open(OUT, "w", encoding="utf-8") as out:
        for name in ("custom_questions", "routing_messages", "routing_terse", "commands"):
            p = SRC / f"{name}.jsonl"
            if p.exists():
                total[name] = conv_questions(p, out)
        if (SRC / "situations.jsonl").exists():
            total["situations"] = conv_situations(SRC / "situations.jsonl", out)
    n = sum(1 for _ in open(OUT))
    print("converted ->", OUT, "| total", n, "items")
    print("by source:", total)
    from collections import Counter
    c = Counter(json.loads(l)["type"] for l in open(OUT))
    print("by type:", dict(c))


if __name__ == "__main__":
    main()
