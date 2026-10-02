"""Score a decision-bench run split by whether the bench source was SEEN during LoRA training.

The LoRA CE corpus (jul/data/mix) and decision-bench share 5 dataset sources (allocine,
civil_comments, clinc, mnli, xnli_fr) — 294/2159 items. On those, a trained model is NOT zero-shot,
so trained-vs-Jev is unfair. This splits the bench into SEEN vs UNSEEN sources and scores each, so we
can tell generalization from memorization: a real gain shows up on UNSEEN.

Usage: python score_split.py <bench.jsonl> <predictions.jsonl>
"""
import json
import os
import sys
sys.path.insert(0, os.environ.get("JUL_DBENCH", os.path.expanduser("~/dev/decision-bench")))
from decision_bench.scoring import score  # noqa: E402

# sources of the bench that also appear (same dataset family) in the LoRA training corpus
SEEN_SOURCES = {"allocine", "civil_comments", "clinc", "mnli", "xnli_fr",
                "mnli_mt_fr", "go_emotions", "pawsx"}  # incl. near-family variants, conservative
# extra seen sources per training corpus, e.g. decision-v7 shares dbpedia + mnli with the bench:
#   JUL_SEEN_EXTRA="dbpedia,mnli,banking77,agnews,trec,imdb,sst5,boolq,amazon,yelp"
SEEN_SOURCES |= {s.strip() for s in os.environ.get("JUL_SEEN_EXTRA", "").split(",") if s.strip()}


def base(src: str) -> str:
    return src.split(":")[0].split("+")[0].strip()


def main():
    bench, preds_path = sys.argv[1], sys.argv[2]
    items = {json.loads(l)["id"]: json.loads(l) for l in open(bench)}
    preds = {json.loads(l)["id"]: json.loads(l) for l in open(preds_path)}
    seen = {i: it for i, it in items.items() if base(it.get("source", "")) in SEEN_SOURCES}
    unseen = {i: it for i, it in items.items() if base(it.get("source", "")) not in SEEN_SOURCES}
    for name, sub in (("SEEN (contaminated)", seen), ("UNSEEN (clean zero-shot)", unseen),
                      ("ALL", items)):
        rep = score(sub, {i: preds[i] for i in sub if i in preds})
        g = rep["groups"]; a = g.get("all", {})
        line = f"{name:26} ALL={a.get('accuracy', 0):.3f} n={a.get('n', 0)}"
        for t in ("choice", "noul", "score"):
            d = g.get(f"type|{t}", {})
            if d.get("accuracy") is not None:
                line += f"  {t}={d['accuracy']:.3f}"
        ci = a.get("ci95")
        if ci:
            line += f"  CI95=[{ci[0]:.3f},{ci[1]:.3f}]"
        print(line)


if __name__ == "__main__":
    main()
