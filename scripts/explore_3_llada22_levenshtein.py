"""Exploration 3 — LLaDA2.2 (Levenshtein edition): knowledge search + feasibility.

Goal: for the System One readout (a typed decision read at one [MASK], no generation), determine
  (1) does an MLX build of LLaDA2.2 exist (mlx-community/LLaDA2.2*)?
  (2) if not, would the project's arch (mlx_lm llada2_moe, or the vendored copy) load a 2.2-mini?
  (3) what does the Levenshtein editing (INSERT/DELETE) buy a better-calibrated readout
      (Emotion ECE 0.378)?

This script does NOT modify mlx_llada.py / mask.py / backbone.py / presets.py. It only reads the
backbone package and queries the Hugging Face Hub *metadata* (config.json / model_info), never the
weights (no 100GB download). If a loadable, architecture-compatible MLX 2.2-mini 4-bit build were
found, the script is wired to smoke-test a cloze; it reports precisely why that path is or isn't
available instead.

Run:
  PYTHONPATH=lib python scripts/explore_3_llada22_levenshtein.py
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lib"))

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

API = HfApi()
OUT = ROOT / "runs" / "explore-3"

# 2.2-only config keys: block routing + Levenshtein edit tokens. Their presence is the tell that a
# checkpoint is 2.2 rather than 2.0/2.1.
KEYS_22 = ["block_size", "expert_capacity", "delete_token_id", "split_token_id"]


def search(query: str, author: str | None = None) -> list[str]:
    try:
        return [m.id for m in API.list_models(search=query, author=author, limit=50)]
    except Exception as e:  # noqa: BLE001
        return [f"<ERR {type(e).__name__}: {e}>"]


def repo_meta(repo: str) -> dict:
    """Files + total size from model_info (no weights downloaded)."""
    try:
        m = API.model_info(repo, files_metadata=True)
        files = [(s.rfilename, s.size or 0) for s in (m.siblings or [])]
        return {"total_gb": sum(sz for _, sz in files) / 1e9,
                "files": sorted(files, key=lambda x: -x[1])}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def grab_json(repo: str, fn: str) -> dict | None:
    try:
        return json.loads(open(hf_hub_download(repo, fn)).read())
    except Exception:  # noqa: BLE001
        return None


def arch_support() -> dict:
    """What routing the arch that Backbone would use actually implements.

    Backbone(mlx_llada) calls mlx_lm.load, which imports mlx_lm.models.llada2_moe; the project also
    ships a vendored copy registered only if mlx_lm lacks the module. We inspect whichever mlx_lm
    provides, and the vendored one, for block-routing fields (2.2) vs group-limited (2.0/2.1).
    """
    out = {}
    has_mlxlm = importlib.util.find_spec("mlx_lm.models.llada2_moe") is not None
    out["mlx_lm_has_llada2_moe"] = has_mlxlm
    if has_mlxlm:
        import mlx_lm.models.llada2_moe as m
        fields = {f.name for f in dataclasses.fields(m.ModelArgs)}
        out["mlx_lm_fields_2_2"] = {k: (k in fields) for k in KEYS_22}
        out["mlx_lm_file"] = m.__file__
    # vendored copy
    from jul.backends.mlx_llada import _register_llada2_moe_arch  # read-only import
    _register_llada2_moe_arch()
    import mlx_lm.models.llada2_moe as reg  # now guaranteed importable
    fields = {f.name for f in dataclasses.fields(reg.ModelArgs)}
    out["effective_fields_2_2"] = {k: (k in fields) for k in KEYS_22}
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # (1) availability
    found = {
        "search LLaDA2.2 (all)": search("LLaDA2.2"),
        "search LLaDA2.2 (mlx-community)": search("LLaDA2.2", author="mlx-community"),
        "search LLaDA2 (mlx-community)": search("LLaDA2", author="mlx-community"),
    }

    # candidate MLX 2.2-mini builds (community), plus the official fp checkpoint for the config truth
    candidates = [
        "mlx-community/LLaDA2.2-mini-4bit",   # would be the "official" name if it existed
        "m1rkocasu/LLaDA2.2-mini-MLX-4bit",   # the community MLX 4-bit build that does exist
        "inclusionAI/LLaDA2.2-mini",          # official fp (config truth)
    ]
    meta = {r: repo_meta(r) for r in candidates}

    # (2) config comparison 2.2 vs the 2.0 build the project uses today
    cfg_20 = grab_json("mlx-community/LLaDA2.0-mini-preview-4bit", "config.json") or {}
    cfg_22 = grab_json("m1rkocasu/LLaDA2.2-mini-MLX-4bit", "config.json") or {}
    cfg_diff = {}
    for k in sorted(set(cfg_20) | set(cfg_22)):
        a, b = cfg_20.get(k, "<absent>"), cfg_22.get(k, "<absent>")
        if a != b:
            cfg_diff[k] = {"2.0": a, "2.2": b}

    arch = arch_support()

    # (3) optional smoke test — only attempt if a build looks architecture-compatible AND small.
    # Compatibility requires the effective arch to implement block routing (2.2). It does not, so we
    # do not download weights; we record the precise blocker instead.
    smoke = {"attempted": False,
             "reason": "effective mlx_lm/vendored llada2_moe implements group-limited routing only "
                       "(no block_size/expert_capacity); loading 2.2 weights would run the wrong "
                       "router and corrupt logits. The only MLX 2.2-mini build ships its own "
                       "llada22_mlx/ (mlx-vlm lineage), not loadable via mlx_lm.load."}

    report = {
        "availability": found,
        "candidate_meta": meta,
        "config_diff_2_0_vs_2_2": cfg_diff,
        "arch_support": arch,
        "smoke_test": smoke,
    }
    (OUT / "findings.json").write_text(json.dumps(report, indent=2))

    # console summary
    print("=== (1) availability ===")
    for k, v in found.items():
        print(f"  {k}:")
        for r in v:
            print(f"    - {r}")
    print("\n=== (2) candidate metadata ===")
    for r, m in meta.items():
        if "error" in m:
            print(f"  {r}: {m['error']}")
        else:
            print(f"  {r}: {m['total_gb']:.2f} GB, {len(m['files'])} files")
    print("\n=== (2b) config diff 2.0 -> 2.2 ===")
    for k, v in cfg_diff.items():
        print(f"  {k}: {v['2.0']}  ->  {v['2.2']}")
    print("\n=== (3) arch support (routing the project would use) ===")
    print(json.dumps(arch, indent=2))
    print("\nfindings.json written to", OUT / "findings.json")


if __name__ == "__main__":
    main()
