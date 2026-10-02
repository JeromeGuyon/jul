"""Bench the MLX LLaDA2 diffusion backend on the Jev BTZSC pilot: baseline vs the System One optims.

Compares, on the same manifest rows and metric code as scripts/bench_jul.py:

  baseline   n_mask=1, n_steps=1, full-vocab mask_logits then slice to the option anchors.
  optimized  anchor-only output projection (backend.option_logits) + the diffusion inference tricks
             n_mask=3, n_steps=2 (variance reduction + iterative demasking), all read at [MASK].

Both go through MaskReader (the real reading path). The optimized run flips on option_logits
automatically (the backend exposes it) and sets n_mask/n_steps on the MaskSpec.

Usage:
  PYTHONPATH=lib python scripts/bench_llada_mlx_optims.py [--per-dataset N] [--datasets agnews,emotiondair]

Banking77 has 72 labels and is slow on a diffusion model; default samples --per-dataset rows per
dataset (stratified by taking the first N of the manifest order) so the comparison runs in minutes.
Pass --per-dataset 0 for the full 100 rows/dataset.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import dataclasses

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lib"))

from jev_benchmarks.io import read_jsonl, sha256_file  # noqa: E402
from jev_benchmarks.metrics import score_predictions  # noqa: E402
from jev_benchmarks.models import Prediction  # noqa: E402

from jul.backbone import Backbone  # noqa: E402
from jul.mask import MaskReader, MaskSpec  # noqa: E402
from jul.types import Option  # noqa: E402

BENCH = ROOT / "external" / "jev-benchmarks"
MANIFEST = BENCH / "results" / "runs" / "btzsc-pilot-v1" / "manifest.jsonl"
MANIFEST_SHA256 = "ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5"
DATASETS = {"agnews": "AG News", "banking77": "Banking77", "emotiondair": "Emotion"}
QUESTION = "Which single label best describes the input text?"
OUT = ROOT / "runs" / "jev-bench-llada-mlx"


def build(reader, rows, ds):
    labels = list(rows[0]["labels"])
    criteria = {f"label_{i:03d}": label for i, label in enumerate(labels)}
    keys = list(criteria)
    options = [Option(k, v) for k, v in criteria.items()]
    preds = []
    for r in rows:
        start = time.perf_counter()
        zs, _ = reader.logits(r["text"], [("choice", QUESTION, options)])
        z = zs[0]
        latency = time.perf_counter() - start
        p = np.exp(z - z.max())
        p = p / p.sum()
        preds.append(Prediction(
            experiment_id="llada-mlx", backend="mlx_llada", model_requested="llada2-mini-4bit",
            model_resolved="llada2-mini-4bit", dataset=ds, example_id=r["example_id"],
            target_index=r["target_index"], predicted_index=int(np.argmax(p)),
            labels=tuple(r["labels"]), probabilities=tuple(float(x) for x in p),
            latency_seconds=latency, probability_sum_raw=float(p.sum())))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-dataset", type=int, default=30)
    ap.add_argument("--datasets", type=str, default=",".join(DATASETS))
    ap.add_argument("--configs", type=str, default="baseline,anchor-only,early-skip,tricks")
    args = ap.parse_args()
    chosen = [d for d in args.datasets.split(",") if d in DATASETS]
    want_cfgs = [c.strip() for c in args.configs.split(",") if c.strip()]

    assert sha256_file(MANIFEST) == MANIFEST_SHA256, "manifest changed"
    manifest = read_jsonl(MANIFEST)
    rows_by_ds = {}
    for ds in chosen:
        rs = [r for r in manifest if r["dataset"] == ds]
        rows_by_ds[ds] = rs[: args.per_dataset] if args.per_dataset else rs

    print("loading backbone (MLX LLaDA2-mini-4bit)...", flush=True)
    bb = Backbone("llada2-mini-4bit", "mlx_llada")

    # Four configs isolate the levers, which pull in different directions:
    #   baseline     full-vocab projection, no tricks (n_mask=1, n_steps=1)
    #   anchor-only  same reading, only the option rows of lm_head -> pure latency win, identical output
    #   early-skip   anchor-only + stop the layer stack once the mask decision is stable (ES-dLLM)
    #   tricks       anchor-only + n_mask=3, n_steps=2 -> quality/calibration lever, more forwards
    configs = {
        "baseline": (MaskSpec.default(), True, False),                                 # force full vocab
        "anchor-only": (MaskSpec.default(), False, False),                             # use option_logits
        "early-skip": (MaskSpec.default(), False, True),                               # + early skipping
        "tricks": (dataclasses.replace(MaskSpec.default(), n_mask=3, n_steps=2), False, False),
        "multitoken": (dataclasses.replace(MaskSpec.default(), readout="multitoken"), False, False),
    }

    results = {}
    for cfg_name, (spec, force_full, early) in configs.items():
        if cfg_name not in want_cfgs:
            continue
        backbone = _NoFast(bb) if force_full else bb
        bb.early_skip = early  # read by MaskReader._read_option_logits
        reader = MaskReader(backbone, spec)
        preds = []
        for ds in chosen:
            t0 = time.perf_counter()
            preds += build(reader, rows_by_ds[ds], ds)
            print(f"  [{cfg_name}/{ds}] {len(rows_by_ds[ds])} rows in {time.perf_counter()-t0:.1f}s", flush=True)
        results[cfg_name] = preds
    bb.early_skip = False

    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["# MLX LLaDA2-mini-4bit — Jev BTZSC pilot — baseline vs System One optims", "",
             f"Rows/dataset: {args.per_dataset or 'all (100)'}. Cells: accuracy / ECE.", "",
             "- **baseline**: full-vocab projection at the mask, no tricks.",
             "- **anchor-only**: only the option rows of `lm_head` are computed (latency lever, "
             "numerically identical to baseline).",
             "- **early-skip**: anchor-only + stop the layer stack once the mask decision is stable (ES-dLLM).",
             "- **tricks**: anchor-only + n_mask=3, n_steps=2 (quality/calibration lever, more forwards).",
             "",
             "| Config | " + " | ".join(f"{DATASETS[d]} acc / ECE" for d in chosen)
             + " | Mean acc | Mean ECE | p50 latency |",
             "|---|" + "---:|" * (len(chosen) + 3)]
    for cfg_name, preds in results.items():
        by = defaultdict(list)
        for p in preds:
            by[p.dataset].append(p)
        scored = {d: score_predictions(by[d]) for d in chosen}
        mean_acc = np.mean([scored[d]["accuracy"] for d in chosen])
        mean_ece = np.mean([scored[d]["ece"] for d in chosen])
        # per-dataset p50 latency (the global median mixes short- and many-option prompts)
        lat = {d: np.median([p.latency_seconds for p in by[d]]) * 1000 for d in chosen}
        p50 = np.median([p.latency_seconds for p in preds]) * 1000
        cells = " | ".join(f"{scored[d]['accuracy']:.2f} / {scored[d]['ece']:.3f}" for d in chosen)
        latcells = " ".join(f"{DATASETS[d].split()[0]} {lat[d]:.0f}ms" for d in chosen)
        lines.append(f"| **{cfg_name}** | {cells} | {mean_acc:.3f} | {mean_ece:.3f} | {p50:.0f} ms |")
        lines.append(f"| _{cfg_name} per-ds latency_ | {latcells} | | | |")
        with open(OUT / f"{cfg_name}.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p.__dict__, default=str) + "\n")
    report = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(report)
    print("\n" + report)


class _NoFast:
    """Wraps a backbone to hide `option_logits`, forcing MaskReader down the full-vocab path."""

    def __init__(self, bb):
        self._bb = bb

    def __getattr__(self, name):
        if name == "option_logits":
            raise AttributeError(name)
        return getattr(self._bb, name)


if __name__ == "__main__":
    main()
