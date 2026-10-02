"""SageMaker entry: run decision-bench (quick/full/edge) on LLaDA with the mask readout.

Ships jul-bis's lib/jul and the decision_bench scoring code + a suite JSONL. Loads LLaDA on the GPU,
answers every item through TypeSafeClient.system_one (choice/noul/score), writes predictions.jsonl
and a scored report.json in the decision-bench format, to SM_MODEL_DIR.

decision-bench item: {id, type: choice|noul|score, question, state, options, answer, score, subset,
lang, difficulty, ...}. jul's runner maps: choice -> Choice(criteria={opt:""}), noul -> Noul,
score -> Score(criteria=options). We reproduce that mapping and score with decision_bench.scoring.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import tarfile
import time


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(here, "lib"))          # jul-bis lib
    sys.path.insert(0, os.path.join(here, "dbench_src"))   # decision_bench package
    sys.path.insert(0, here)

    adapter_dir = os.environ.get("SM_CHANNEL_ADAPTER", "")
    if adapter_dir and os.path.isdir(adapter_dir):
        tars = glob.glob(os.path.join(adapter_dir, "*.tar.gz"))
        dest = "/opt/ml/adapter"; os.makedirs(dest, exist_ok=True)
        if tars:
            with tarfile.open(tars[0]) as t:
                t.extractall(dest)
        cand = os.path.join(dest, "lora_adapter")
        os.environ["JUL_LLADA_ADAPTER"] = cand if os.path.isdir(cand) else dest
        print(f"[dbench] adapter -> {os.environ['JUL_LLADA_ADAPTER']}", flush=True)
        head_pt = os.path.join(dest, "read_head.pt")
        if os.path.isfile(head_pt):
            os.environ["JUL_LLADA_HEAD"] = head_pt
            print(f"[dbench] learned read head -> {head_pt}", flush=True)

    from decision_bench.scoring import score
    from jul import Choice, Noul, Score, TypeSafeClient

    model = os.environ.get("JUL_MODEL", "llada-8b-instruct")
    data_dir = os.environ.get("SM_CHANNEL_BENCH", "/opt/ml/input/data/bench")
    suite_files = glob.glob(os.path.join(data_dir, "*.jsonl"))
    items = {}
    for f in suite_files:
        for line in open(f, encoding="utf-8"):
            line = line.strip()
            if line:
                it = json.loads(line)
                items[it["id"]] = it
    print(f"[dbench] model={model} items={len(items)}", flush=True)

    client = TypeSafeClient(model=model, backend="llada")
    client.system_one("warm up", {"q": Noul(instructions="Is this a test?")})

    def question(it):
        if it["type"] == "choice":
            return Choice(instructions=it["question"], criteria={o: "" for o in it["options"]})
        if it["type"] == "noul":
            return Noul(instructions=it["question"])
        return Score(instructions=it["question"], criteria=it["options"])

    preds = {}
    pred_rows = []
    t_start = time.time()
    for i, (iid, it) in enumerate(items.items()):
        t0 = time.perf_counter()
        try:
            r = client.system_one(it["state"], {"q": question(it)})
            a = r.answers["q"]
            if it["type"] == "choice":
                # pipeline guard: a restricted argmax must return one of the given options
                assert a.choice in it["options"], f"OFF-LIST pred {a.choice!r} not in {it['options']}"
                row = {"id": iid, "pred": a.choice, "probs": a.probabilities}
            elif it["type"] == "noul":
                row = {"id": iid, "pred": a.noul}
            else:
                row = {"id": iid, "pred": a.score, "probs": a.probabilities}
            row["latency_ms"] = (time.perf_counter() - t0) * 1000
        except Exception as e:
            row = {"id": iid, "error": str(e)[:300], "latency_ms": (time.perf_counter() - t0) * 1000}
        preds[iid] = row
        pred_rows.append(row)
        if (i + 1) % 50 == 0:
            print(f"[dbench] {i+1}/{len(items)} ({time.time()-t_start:.0f}s)", flush=True)

    report = score(items, preds)
    out_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "predictions.jsonl"), "w", encoding="utf-8") as f:
        for row in pred_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # print the headline groups
    g = report["groups"]
    print("[dbench] === RESULT ===", flush=True)
    for key in ("all", "type|choice", "type|noul", "type|score"):
        if key in g:
            d = g[key]
            print(f"[dbench]   {key:14} acc={d['accuracy']:.3f} (n={d['n']})", flush=True)
    print(f"[dbench]   errors={report['errors']} missing={report['missing']} "
          f"skipped={report['unknown_gold_skipped']}", flush=True)
    print(f"[dbench] report -> {out_dir}/report.json", flush=True)


if __name__ == "__main__":
    main()
