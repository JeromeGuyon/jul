"""Regressions for the LLaDA learned-head / backend fixes (no model download)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parent.parent


def test_read_head_does_not_save_or_overwrite_the_backbone_embedding(tmp_path):
    from jul.llada_head import ReadHead, load_head

    emb = torch.nn.Embedding(1000, 16)
    head = ReadHead(hidden=16, emb=emb, proj=4)
    state = head.state_dict()
    assert not any(k.startswith("emb.") for k in state), "the vocab table must not be in read_head.pt"
    assert "emb" not in dict(head.named_modules()) and len(list(head.parameters())) == 6
    path = tmp_path / "read_head.pt"
    torch.save(state, path)

    other = torch.nn.Embedding(1000, 16)
    before = other.weight.detach().clone()
    loaded = load_head(str(path), other, {"hidden": 16, "proj": 4, "max_levels": 16})
    assert torch.equal(other.weight, before)
    assert torch.equal(loaded.q.weight, head.q.weight)
    assert loaded.emb is other  # keys still read from the live backbone embedding


def test_load_head_accepts_a_legacy_file_with_the_embedding_and_ignores_it(tmp_path):
    from jul.llada_head import ReadHead, load_head

    head = ReadHead(hidden=8, emb=torch.nn.Embedding(50, 8), proj=4)
    legacy = {**head.state_dict(), "emb.weight": torch.zeros(50, 8)}
    path = tmp_path / "read_head.pt"
    torch.save(legacy, path)
    live = torch.nn.Embedding(50, 8)
    before = live.weight.detach().clone()
    load_head(str(path), live, {"hidden": 8, "proj": 4, "max_levels": 16})
    assert torch.equal(live.weight, before)


def test_load_head_is_strict_on_real_mismatches(tmp_path):
    from jul.llada_head import ReadHead, load_head

    torch.save(ReadHead(hidden=8, emb=torch.nn.Embedding(50, 8), proj=4).state_dict(),
               tmp_path / "read_head.pt")
    with pytest.raises(RuntimeError):
        load_head(str(tmp_path / "read_head.pt"), torch.nn.Embedding(50, 8),
                  {"hidden": 8, "proj": 8, "max_levels": 16})


def test_mask_reader_loads_the_head_from_the_library_not_scripts(tmp_path, monkeypatch):
    from jul.llada_head import ReadHead
    from jul.mask import MaskReader, MaskSpec

    emb = torch.nn.Embedding(5000, 8)
    torch.save(ReadHead(hidden=8, emb=emb, proj=4).state_dict(), tmp_path / "read_head.pt")
    (tmp_path / "read_head.json").write_text(json.dumps({"hidden": 8, "proj": 4, "max_levels": 16}))

    class _Model(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.e = emb
        def get_input_embeddings(s):
            return s.e

    class _BB:
        name = "stub"; mask_id = 7; device = "cpu"
        tokenizer = None
        model = _Model()

    monkeypatch.setattr(sys, "path", [p for p in sys.path if not p.endswith("scripts")])
    sys.modules.pop("llada_head", None)
    monkeypatch.setenv("JUL_LLADA_HEAD", str(tmp_path / "read_head.pt"))
    reader = MaskReader(_BB(), MaskSpec.default())
    assert reader.head is not None and "llada_head" not in sys.modules


def test_transformers_patch_is_scoped_to_the_load():
    rope = pytest.importorskip("transformers.modeling_rope_utils")
    from jul.backends.llada import _patch_remote_code_for_transformers5

    had = "default" in rope.ROPE_INIT_FUNCTIONS
    with _patch_remote_code_for_transformers5("not/a-real-repo"):
        assert "default" in rope.ROPE_INIT_FUNCTIONS
    assert ("default" in rope.ROPE_INIT_FUNCTIONS) == had


def test_decontaminate_mapika_runs(tmp_path):
    bench = tmp_path / "data"; bench.mkdir()
    (bench / "bench-v1.jsonl").write_text(json.dumps({"state": "the bench state that must not leak"}) + "\n")
    src = tmp_path / "in.jsonl"
    src.write_text(json.dumps({"state": "the bench state that must not leak"}) + "\n"
                   + json.dumps({"state": "a perfectly unrelated training state"}) + "\n")
    out = tmp_path / "out.jsonl"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "decontaminate_mapika.py"), str(src), str(out)],
                       env={"JUL_DBENCH": str(tmp_path), "PATH": ""}, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert out.read_text().count("\n") == 1 and "unrelated" in out.read_text()
