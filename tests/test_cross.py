"""The cross reading (jul/cross.py): pairs, heads and routing, on a stub encoder (no model needed)."""

import json

import numpy as np
import pytest

from jul.cross import CrossReader, CrossSpec, cut
from jul.presets import Formulation, Preset
from jul.types import Choice, Noul, NoulCriteria, Score, options_of

D = 4


class StubTokenizer:
    """One token per character; <s> = 0, </s> = 2."""

    def encode(self, text, add_special_tokens=True):
        ids = [10 + (ord(c) % 50) for c in text]
        return [0, *ids, 2] if add_special_tokens else ids


class StubEncoder:
    """Features whose mean half is the pair's length in every dimension, so a head sees the input."""

    architecture = "encoder"
    name = "stub"

    def __init__(self):
        self.tokenizer = StubTokenizer()
        self.seen = []

    def forward_batch(self, queries, layers=(), pools=None, prefix=None):
        self.seen.extend(queries)
        return [{l: np.concatenate([np.full(D, float(len(q))), np.zeros(D)]).astype(np.float32) for l in layers}
                for q in queries]


def write_model(tmp_path, noul_bias=(2.0, 0.0, 1.0), types=("noul", "score")):
    zeros = lambda n: np.zeros((n, D), dtype=np.float32)
    np.savez(tmp_path / "cross_heads.npz", noul_weight=zeros(3), noul_bias=np.array(noul_bias, dtype=np.float32),
             choice_weight=np.ones((1, D), dtype=np.float32), choice_bias=np.zeros(1, dtype=np.float32),
             score_weight=np.ones((1, D), dtype=np.float32), score_bias=np.zeros(1, dtype=np.float32))
    (tmp_path / "cross.json").write_text(json.dumps({"method": "cross", "prefix": "q: ", "max_length": 40,
                                                     "layer": 3, "separator": [2, 2], "types": list(types)}))
    return CrossSpec.load(tmp_path)


def test_cut_is_longest_first():
    a, b = cut(list(range(10)), list(range(4)), 8)
    assert (len(a), len(b)) == (4, 4) and a == [0, 1, 2, 3]
    assert cut([1, 2], [3], 5) == ([1, 2], [3])


def test_first_segments():
    noul = Noul("Is it offensive?")
    assert CrossReader.firsts("noul", noul.instructions, options_of(noul)) == ["Is it offensive?"]
    described = Noul("Is it offensive?", NoulCriteria(true="insults", false="polite"))
    assert CrossReader.firsts("noul", described.instructions, options_of(described)) == \
        ["Is it offensive? (true: insults; false: polite)"]
    score = Score("How urgent?", ["low", "high"])
    assert CrossReader.firsts("score", score.instructions, options_of(score)) == ["How urgent?\nlow", "How urgent?\nhigh"]
    choice = Choice("Which team?", {"billing": "payments", "tech": ""})
    assert CrossReader.firsts("choice", choice.instructions, options_of(choice)) == \
        ["Which team?\nbilling: payments", "Which team?\ntech"]


def test_noul_probability_and_pairs(tmp_path):
    reader = CrossReader(StubEncoder(), write_model(tmp_path))
    q = Noul("Is it late?")
    z, tokens = reader.logits("paid on May 9", "noul", q.instructions, options_of(q))
    p = np.exp(z) / np.exp(z).sum()
    e = np.exp([2.0, 0.0, 1.0])
    yes, unknown = e[0] / e.sum(), e[2] / e.sum()
    by_key = dict(zip([o.key for o in options_of(q)], p))
    assert by_key["true"] == pytest.approx(yes + unknown / 2, abs=1e-6)
    pair = reader.backbone.seen[0]
    first = StubTokenizer().encode("q: Is it late?", add_special_tokens=False)
    assert pair[: len(first)] == first and pair[len(first): len(first) + 2] == [2, 2]
    assert tokens == len(pair) + 2


def test_score_reads_one_pair_per_level_and_cuts(tmp_path):
    reader = CrossReader(StubEncoder(), write_model(tmp_path))
    q = Score("How urgent?", ["low", "a much longer level"])
    z, _ = reader.logits("x" * 100, "score", q.instructions, options_of(q))
    assert z.shape == (2,) and z[1] >= z[0]
    assert all(len(p) + 2 <= 40 for p in reader.backbone.seen)   # + <s> and </s> around the pair


def test_spec_rejects_other_methods(tmp_path):
    (tmp_path / "cross.json").write_text(json.dumps({"method": "pointer", "layer": 1, "separator": [2]}))
    with pytest.raises(ValueError, match="unsupported method"):
        CrossSpec.load(tmp_path)


def test_preset_keeps_its_cross_model(tmp_path):
    p = Preset(name="m", repo="", formulations=(Formulation("one_word", "{state}", 1),), tau=0.02,
               latency_ms="?", quality="", backend="onnx", onnx_repo="dir", cross={"repo": "cross-dir"})
    assert Preset.from_json(p.to_json(), tmp_path).cross == {"repo": "cross-dir"}
    assert "cross" not in Preset.from_json({**p.to_json(), "cross": None}, tmp_path).to_json()


def test_default_method_routes_declared_types_unless_tuned(tmp_path):
    from jul.client import TypeSafeClient
    from jul.context import Context

    class Engine:
        cross = CrossReader(StubEncoder(), write_model(tmp_path))

    client = TypeSafeClient.__new__(TypeSafeClient)
    client._preset = Preset(name="m", repo="", formulations=(), tau=1.0, latency_ms="?", quality="")
    client._backend = "onnx"
    noul, choice = Noul("Is it late?"), Choice("Which?", ["a", "b"])
    assert client._default_method(Engine, None, "noul", noul, options_of(noul)) == "cross"
    assert client._default_method(Engine, None, "choice", choice, options_of(choice)) == "vector"
    ctx = Context()
    ctx.calibration[client._digest("noul", noul, options_of(noul))] = (1.0, [0.0, 0.0])
    assert client._default_method(Engine, ctx, "noul", noul, options_of(noul)) == "vector"
