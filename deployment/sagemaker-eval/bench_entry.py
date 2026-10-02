"""SageMaker entry: play the public Jev benchmark (BTZSC pilot v1) on LLaDA + a LoRA adapter.

Reproduces the zero-shot variant of scripts/bench_jul.py — the only variant comparable to Jev — on a
GPU. Ships the jev_benchmarks package, the BTZSC manifest, jul-bis's lib/jul, and (optionally) a LoRA
adapter. Loads LLaDA on the GPU, runs every example through TypeSafeClient.system_one as a Choice,
scores with the benchmark's own metric code, writes a report to SM_MODEL_DIR.

300 examples, 3 datasets (AG News, Banking77, Emotion), 100 each. Published references (from the
report): Jev 0.91 / 0.87 / 0.48, GLiNER 0.70 / 0.61 / 0.44.
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
    sys.path.insert(0, os.path.join(here, "jev_src"))      # jev_benchmarks package
    sys.path.insert(0, here)

    # optional LoRA adapter channel (extract model.tar.gz)
    adapter_dir = os.environ.get("SM_CHANNEL_ADAPTER", "")
    if adapter_dir and os.path.isdir(adapter_dir):
        tars = glob.glob(os.path.join(adapter_dir, "*.tar.gz"))
        dest = "/opt/ml/adapter"; os.makedirs(dest, exist_ok=True)
        if tars:
            with tarfile.open(tars[0]) as t:
                t.extractall(dest)
        cand = os.path.join(dest, "lora_adapter")
        os.environ["JUL_LLADA_ADAPTER"] = cand if os.path.isdir(cand) else dest
        print(f"[bench] adapter -> {os.environ['JUL_LLADA_ADAPTER']}", flush=True)

    import numpy as np
    from pathlib import Path
    from jev_benchmarks.io import read_jsonl
    from jev_benchmarks.metrics import group_scores
    from jev_benchmarks.models import Prediction
    from jul import Choice, TypeSafeClient

    model = os.environ.get("JUL_MODEL", "llada-8b-instruct")
    manifest_path = os.path.join(os.environ.get("SM_CHANNEL_BENCH", "/opt/ml/input/data/bench"),
                                 "manifest.jsonl")
    manifest = read_jsonl(Path(manifest_path))
    datasets = ("agnews", "banking77", "emotiondair")
    rows_by_ds = {ds: [r for r in manifest if r["dataset"] == ds] for ds in datasets}
    question = "Which single label best describes the input text?"
    print(f"[bench] model={model} rows={len(manifest)}", flush=True)

    client = TypeSafeClient(model=model, backend="llada")
    predictions = []
    for ds, rows in rows_by_ds.items():
        labels = list(rows[0]["labels"])
        criteria = {f"label_{i:03d}": label for i, label in enumerate(labels)}
        keys = list(criteria)
        questions = {"label": Choice(instructions=question, criteria=criteria)}
        for r in rows:
            t0 = time.perf_counter()
            ans = client.system_one(state=r["text"], questions=questions).choices["label"]
            latency = time.perf_counter() - t0
            probs = tuple(float(ans.probabilities[k]) for k in keys)
            predictions.append(Prediction(
                experiment_id="llada-bench", backend="llada-zero-shot", model_requested=model,
                model_resolved=model, dataset=ds, example_id=r["example_id"],
                target_index=r["target_index"], predicted_index=int(np.argmax(probs)),
                labels=tuple(r["labels"]), probabilities=probs,
                latency_seconds=latency, probability_sum_raw=float(sum(probs))))
        print(f"[bench] {ds}: done", flush=True)

    scored = group_scores(predictions)
    mean_acc = float(np.mean([scored[d]["accuracy"] for d in datasets]))
    mean_ece = float(np.mean([scored[d]["ece"] for d in datasets]))
    report = {"model": model, "mean_accuracy": mean_acc, "mean_ece": mean_ece,
              "per_dataset": {d: {"accuracy": scored[d]["accuracy"], "ece": scored[d]["ece"],
                                  "latency_p50_seconds": scored[d]["latency_p50_seconds"]}
                              for d in datasets}}
    print("[bench] === RESULT ===", flush=True)
    for d in datasets:
        print(f"[bench]   {d:12} acc={scored[d]['accuracy']:.3f} ece={scored[d]['ece']:.3f}", flush=True)
    print(f"[bench]   MEAN acc={mean_acc:.3f} ece={mean_ece:.3f}", flush=True)
    print(f"[bench]   (ref Jev: agnews 0.91 / banking77 0.87 / emotion 0.48, mean 0.753)", flush=True)

    out_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "jev-bench.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[bench] report -> {out_dir}/jev-bench.json", flush=True)


if __name__ == "__main__":
    main()
