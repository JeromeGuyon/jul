"""PISTE D — Is the Banking77 0.19 ceiling from quantization / model size, or from the method?

Question
--------
The multi-token (sequence-likelihood) readout of `explore_1_multitoken.py` scores 0.18-0.19 on
Banking77 (72 intents) with `mlx-community/LLaDA2.0-mini-preview-4bit`. Is 0.19 a *method* ceiling
or a *model* ceiling? To tell them apart we re-run the EXACT same readout on strictly more capable
LLaDA2-MoE builds and compare accuracy on the same 100 Banking77 rows.

Two orthogonal levers, both `model_type: llada2_moe` (same vendored MLX arch, same 157184 vocab, so
the readout code and backbone load unchanged):

  * mini-8bit  — SAME model (20 layers, hidden 2048), 8-bit instead of 4-bit -> isolates QUANTIZATION.
  * flash-4bit — BIGGER model (32 layers, hidden 4096), same 4-bit             -> isolates MODEL SIZE.

If neither moves accuracy meaningfully above 0.19, the ceiling is the method, not the model.
If mini-8bit lifts it, 4-bit quantization was hurting. If flash-4bit lifts it, capacity was the limit.

Reuse
-----
The readout is imported verbatim from `scripts/explore_1_multitoken.py`
(`multitoken_scores`, `_strip_common_prefix`, ...). We do NOT modify any lib/ module. The extra
model repos are registered at *runtime* into `jul.backbone.MODELS` (in-memory only; presets.py and
backbone.py on disk are untouched).

Usage
-----
  PYTHONPATH=lib python scripts/pisteD_model.py \
      [--rows 100] [--models mini-8bit,flash-4bit] [--norm mean] [--skip-download-check]

Disk guard: each build is size-checked against free disk before any download; a build that would
not fit (with a safety margin) is SKIPPED and documented rather than downloaded.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "external" / "jev-benchmarks" / "src"))

from jev_benchmarks.io import read_jsonl, sha256_file  # noqa: E402
from jev_benchmarks.metrics import score_predictions  # noqa: E402
from jev_benchmarks.models import Prediction  # noqa: E402

import jul.backbone as backbone_mod  # noqa: E402
from jul.backbone import Backbone  # noqa: E402

# Reuse the reference multi-token readout verbatim (no re-implementation).
from explore_1_multitoken import multitoken_scores  # noqa: E402

BENCH = ROOT / "external" / "jev-benchmarks"
MANIFEST = BENCH / "results" / "runs" / "btzsc-pilot-v1" / "manifest.jsonl"
MANIFEST_SHA256 = "ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5"
OUT = ROOT / "runs" / "banking-D"

# Baseline multi-token Banking77 on mini-4bit (per the task brief and runs/jev-bench-llada-mlx).
BASELINE_ACC = 0.19
BASELINE_ECE = 0.152
BASELINE_P50_MS = 89.0

# Candidate builds. name -> (hf repo, lever, expected GB, note)
CANDIDATES = {
    "mini-4bit": ("mlx-community/LLaDA2.0-mini-preview-4bit", "baseline (4-bit, 20L/2048)", 9.2,
                  "the current baseline model; re-measured here as a sanity check"),
    "mini-8bit": ("mlx-community/LLaDA2.0-mini-8bit", "quantization (8-bit, 20L/2048)", 17.3,
                  "SAME architecture as baseline, 8-bit instead of 4-bit -> isolates quantization"),
    "flash-4bit": ("mlx-community/LLaDA2.0-flash-preview-4bit", "model size (4-bit, 32L/4096)", 57.9,
                   "BIGGER model, same 4-bit -> isolates capability/size"),
}
SAFETY_MARGIN_GB = 30.0  # keep this much free after a download


def free_gb(path: Path) -> float:
    return shutil.disk_usage(path).free / 1e9


def hf_repo_size_gb(repo: str) -> float:
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo, files_metadata=True)
    return sum((f.size or 0) for f in info.siblings) / 1e9


def total_ram_gb() -> float:
    """Physical RAM in GB (unified memory on Apple Silicon = the model residency budget)."""
    try:
        import subprocess
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()
        return int(out) / 1e9
    except Exception:
        import os
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9


#: A build whose weights need more than (RAM - this margin) GB thrashes swap and is skipped.
#: Apple Silicon shares one memory pool between CPU and Metal GPU, so the full weight set must be
#: resident; a model larger than RAM swaps every forward and never finishes in practical time.
RAM_MARGIN_GB = 8.0


def already_cached(repo: str) -> bool:
    """True if the weight shards are already on disk (so no download needed)."""
    from huggingface_hub import scan_cache_dir
    try:
        cache = scan_cache_dir()
    except Exception:
        return False
    for r in cache.repos:
        if r.repo_id == repo:
            # cached with real weights if the repo dir holds >1GB of safetensors
            has_weights = any(
                str(f.file_path).endswith(".safetensors") and f.size_on_disk > 1e8
                for rev in r.revisions for f in rev.files
            )
            if has_weights:
                return True
    return False


def load_build(name: str) -> Backbone:
    """Register the repo at runtime and load through the same mlx_llada path as the baseline."""
    repo, _, _, _ = CANDIDATES[name]
    key = f"pisteD-{name}"
    # In-memory registration only; presets.py / backbone.py on disk are untouched.
    backbone_mod.MODELS[key] = {"mlx_llada": repo}
    return Backbone(key, "mlx_llada")


def build_predictions(bb, rows, model_name, repo, norm):
    labels = list(rows[0]["labels"])
    preds = []
    for r in rows:
        t0 = time.perf_counter()
        scores = multitoken_scores(bb, r["text"], labels, norm=norm)
        latency = time.perf_counter() - t0
        if not np.isfinite(scores).all():
            raise FloatingPointError(f"non-finite scores on {model_name} for {r['example_id']}")
        z = scores.astype(np.float64)
        p = np.exp(z - z.max())
        p = p / p.sum()
        preds.append(Prediction(
            experiment_id=f"pisteD-{model_name}", backend="mlx_llada",
            model_requested=repo, model_resolved=repo,
            dataset="banking77", example_id=r["example_id"], target_index=r["target_index"],
            predicted_index=int(np.argmax(p)), labels=tuple(r["labels"]),
            probabilities=tuple(float(x) for x in p), latency_seconds=latency,
            probability_sum_raw=float(p.sum())))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=100)
    ap.add_argument("--models", type=str, default="mini-8bit,flash-4bit")
    ap.add_argument("--norm", type=str, default="mean", choices=["mean", "sum"])
    ap.add_argument("--skip-download-check", action="store_true",
                    help="do not query the HF API for sizes (use the hardcoded expectations)")
    args = ap.parse_args()
    chosen = [m for m in args.models.split(",") if m in CANDIDATES]

    assert sha256_file(MANIFEST) == MANIFEST_SHA256, "manifest changed"
    manifest = read_jsonl(MANIFEST)
    rows = [r for r in manifest if r["dataset"] == "banking77"]
    rows = rows[: args.rows] if args.rows else rows
    print(f"banking77 rows: {len(rows)}, labels: {len(rows[0]['labels'])}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    results = {}   # name -> scored dict + latency
    skipped = {}   # name -> reason

    for name in chosen:
        repo, lever, exp_gb, note = CANDIDATES[name]
        print(f"\n=== {name}  ({repo})  [{lever}] ===", flush=True)

        # Disk / size guard (unless the build is already fully cached).
        cached = already_cached(repo)
        if not cached:
            if args.skip_download_check:
                size_gb = exp_gb
            else:
                try:
                    size_gb = hf_repo_size_gb(repo)
                except Exception as e:
                    size_gb = exp_gb
                    print(f"  (HF size query failed: {e}; using expected {exp_gb} GB)", flush=True)
            avail = free_gb(ROOT)
            print(f"  not cached. repo ~{size_gb:.1f} GB, free ~{avail:.1f} GB "
                  f"(margin {SAFETY_MARGIN_GB} GB)", flush=True)
            if size_gb + SAFETY_MARGIN_GB > avail:
                reason = (f"would need {size_gb:.1f} GB + {SAFETY_MARGIN_GB} GB margin "
                          f"> {avail:.1f} GB free")
                print(f"  SKIP: {reason}", flush=True)
                skipped[name] = reason
                continue
            print(f"  downloading {size_gb:.1f} GB ...", flush=True)
        else:
            print("  already cached; no download.", flush=True)

        # RAM guard: on Apple Silicon the whole weight set must be resident (unified memory). A build
        # heavier than (RAM - margin) thrashes swap and cannot finish; skip and document it rather
        # than hang. Estimate the resident footprint from the repo size on disk.
        ram = total_ram_gb()
        footprint = exp_gb if not args.skip_download_check else exp_gb
        try:
            footprint = hf_repo_size_gb(repo) if not cached else exp_gb
        except Exception:
            footprint = exp_gb
        if footprint + RAM_MARGIN_GB > ram:
            reason = (f"weights ~{footprint:.1f} GB + {RAM_MARGIN_GB} GB margin > {ram:.0f} GB RAM: "
                      f"memory-bound on this machine (unified memory), would thrash swap and never "
                      f"finish 100 forwards")
            print(f"  SKIP (RAM): {reason}", flush=True)
            skipped[name] = reason
            continue

        # Load + warmup + run.
        t_load = time.perf_counter()
        try:
            bb = load_build(name)
        except Exception as e:
            reason = f"load failed: {type(e).__name__}: {e}"
            print(f"  SKIP: {reason}", flush=True)
            skipped[name] = reason
            continue
        load_s = time.perf_counter() - t_load
        print(f"  loaded in {load_s:.1f}s (n_layers={bb.n_layers}, mask_id={bb.mask_id})", flush=True)

        print("  warmup...", flush=True)
        _ = multitoken_scores(bb, rows[0]["text"], list(rows[0]["labels"]), norm=args.norm)

        t0 = time.perf_counter()
        preds = build_predictions(bb, rows, name, repo, args.norm)
        run_s = time.perf_counter() - t0
        scored = score_predictions(preds)
        p50 = float(np.median([p.latency_seconds for p in preds]) * 1000)
        results[name] = {"scored": scored, "p50_ms": p50, "repo": repo, "lever": lever,
                         "load_s": load_s, "run_s": run_s, "n_layers": bb.n_layers}
        print(f"  [{name}] acc={scored['accuracy']:.3f} ece={scored['ece']:.3f} "
              f"p50={p50:.0f}ms  ({run_s:.1f}s for {len(rows)} rows)", flush=True)

        with open(OUT / f"predictions-{name}.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p.__dict__, default=str) + "\n")

        # Free the model before loading the next (each build is multi-GB in Metal memory).
        del bb
        try:
            import mlx.core as mx
            mx.clear_cache()
        except Exception:
            pass

    write_report(args, results, skipped, len(rows))


def write_report(args, results, skipped, n_rows):
    lines = [
        "# Piste D — Banking77 ceiling: quantization / model size vs method",
        "",
        f"Readout: the multi-token (sequence-likelihood) readout from "
        f"`scripts/explore_1_multitoken.py`, imported verbatim. Rows: {n_rows} Banking77 (72 intents, "
        f"rich label descriptions). Length norm: `{args.norm}`. One bidirectional forward per row.",
        "",
        "Question: is the ~0.19 accuracy a *method* ceiling or a *model* ceiling? We re-run the same "
        "readout on strictly more capable `llada2_moe` builds (same 157184 vocab, same MLX arch).",
        "",
        "| Build | lever | layers | acc | Δ vs 0.19 | ECE | p50 ms | load s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
        f"| mini-4bit (baseline ref) | 4-bit, 20L/2048 | 20 | {BASELINE_ACC:.3f} | — | "
        f"{BASELINE_ECE:.3f} | {BASELINE_P50_MS:.0f} | — |",
    ]
    for name, r in results.items():
        s = r["scored"]
        dacc = s["accuracy"] - BASELINE_ACC
        lines.append(
            f"| {name} | {r['lever']} | {r['n_layers']} | {s['accuracy']:.3f} | {dacc:+.3f} | "
            f"{s['ece']:.3f} | {r['p50_ms']:.0f} | {r['load_s']:.0f} |"
        )
    lines += ["", "Per-build detail (macro-F1, NLL, mean confidence):", "",
              "| Build | acc | macro_f1 | ece | nll | mean_conf |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        s = r["scored"]
        lines.append(
            f"| {name} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} | {s['ece']:.3f} | "
            f"{s['nll']:.3f} | {s['mean_confidence']:.3f} |"
        )

    if skipped:
        lines += ["", "## Skipped builds", ""]
        for name, reason in skipped.items():
            repo, lever, exp_gb, note = CANDIDATES[name]
            lines.append(f"- **{name}** (`{repo}`, ~{exp_gb} GB, {lever}): {reason}. {note}")

    # Verdict.
    lines += ["", "## Verdict", ""]
    if results:
        best_name = max(results, key=lambda k: results[k]["scored"]["accuracy"])
        best_acc = results[best_name]["scored"]["accuracy"]
        best_d = best_acc - BASELINE_ACC
        lines.append(
            f"Best more-capable build: **{best_name}** at acc {best_acc:.3f} "
            f"(Δ {best_d:+.3f} vs baseline 0.19)."
        )
        if best_d < 0.03:
            lines.append(
                "Interpretation: moving to a better-quantized or larger LLaDA2-MoE build does **not** "
                "meaningfully lift Banking77. The 0.19 is a **method ceiling** (the [MASK] "
                "sequence-likelihood readout on 72 near-identical label descriptions), not a "
                "quantization or capacity ceiling. The embedding path (wemm-4b 0.88) remains the "
                "structurally better approach for 72-way retrieval-style classification."
            )
        elif best_d < 0.15:
            lines.append(
                "Interpretation: a more capable build gives a **partial** lift — some of the 0.19 "
                "ceiling is model-driven, but a large gap to the 0.88 embedding bound remains, so the "
                "readout method is still the dominant limiter."
            )
        else:
            lines.append(
                "Interpretation: a more capable build **substantially** lifts Banking77 — the 0.19 was "
                "largely a model (quantization/size) ceiling, not a method ceiling."
            )
    else:
        lines.append(
            "No more-capable build could be evaluated (see Skipped). On disk/size grounds we cannot "
            "empirically separate a method ceiling from a model ceiling in this run."
        )

    report = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(report)
    print("\n" + report, flush=True)


if __name__ == "__main__":
    main()
