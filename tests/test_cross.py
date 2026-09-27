"""The cross reading (jul/cross.py): pairs, heads and routing, on a stub encoder (no model needed)."""

import importlib.util
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


# --- on a real (tiny, random) encoder: client, pack and bundle ---------------------------------------

HAS_EXPORT = all(importlib.util.find_spec(m) for m in ("torch", "onnx", "onnxscript", "onnxruntime"))
needs_export = pytest.mark.skipif(not HAS_EXPORT, reason="needs torch, onnx, onnxscript, onnxruntime")


@pytest.fixture
def cross_setup(tiny_encoder, tmp_path, monkeypatch):
    """A vector preset on the tiny onnx encoder, and the same encoder with random heads as its cross model,
    stored in the model repo's cross/ folder the way the published models carry it."""
    import shutil

    import jul.presets
    from jul.cross import find
    from jul.encoder import templates
    from jul.presets import repo_fields, save_preset
    hf_dir, onnx_dir = tiny_encoder
    repo = tmp_path / "model"
    shutil.copytree(onnx_dir, repo)
    shutil.copytree(onnx_dir, repo / "cross")
    rng = np.random.default_rng(0)
    d = 32
    np.savez(repo / "cross" / "cross_heads.npz",
             **{f"{t}_weight": rng.normal(size=(n, d)).astype(np.float32) for t, n in (("noul", 3), ("choice", 1), ("score", 1))},
             **{f"{t}_bias": rng.normal(size=n).astype(np.float32) for t, n in (("noul", 3), ("choice", 1), ("score", 1))})
    (repo / "cross" / "cross.json").write_text(json.dumps(
        {"method": "cross", "prefix": "query: ", "max_length": 64, "layer": 3, "separator": [2, 2],
         "types": ["noul", "score"], "heads": "cross_heads.npz"}))
    monkeypatch.setattr(jul.presets, "PRESET_HOME", tmp_path / "presets")
    t = templates("query: ")
    save_preset(Preset(name="tiny-cross", **repo_fields("onnx", str(repo)),
                       formulations=(Formulation("one_word", t["one_word"], 3),
                                     Formulation("question_options", t["question_options"], 3)),
                       tau=0.05, latency_ms="?", quality="test", center="options", backend="onnx",
                       cross=find(str(repo))), tmp_path / "presets")
    from jul import TypeSafeClient
    client = TypeSafeClient(model="tiny-cross", backend="onnx", context_home=tmp_path / "contexts")
    yield client, repo
    client.close()


MIXED = {"team": Choice("Which team?", {"billing": "payments", "tech": "bugs"}),
         "urgent": Noul("Is it urgent?"),
         "anger": Score("How angry is the customer?", ["calm", "annoyed", "furious"])}
TEXTS = ["I was charged twice", "the app crashes on export", "refund me now or I leave"]


def answers(response):
    return {n: np.array(list(a.probabilities.values())) if hasattr(a, "probabilities") else np.array([a.noul])
            for n, a in response.answers.items()}


@needs_export
def test_the_cross_model_answers_its_types_and_the_bundle_matches(cross_setup, tmp_path):
    from jul.bundle import Bundle, pack
    client, _ = cross_setup
    engine = client._engine_for(None)
    assert engine.cross is not None and engine.cross.spec.types == ("noul", "score")
    vector = client.system_one(state=TEXTS[0], questions=MIXED, method="vector")
    crossed = client.system_one(state=TEXTS[0], questions=MIXED)
    assert np.allclose(answers(vector)["team"], answers(crossed)["team"])          # Choice stays on vectors
    assert not np.allclose(answers(vector)["urgent"], answers(crossed)["urgent"])  # Noul goes to the cross model
    bundle = Bundle.load(pack(client, MIXED, tmp_path / "bundle"))
    assert bundle.models == ["vector", "cross"]
    for text in TEXTS:
        want, got = answers(client.system_one(state=text, questions=MIXED)), answers(bundle.system_one(text))
        assert list(got) == list(MIXED)
        for name in MIXED:
            assert np.allclose(got[name], want[name], atol=1e-4), (text, name)


@needs_export
def test_a_bundle_of_yes_no_and_scores_ships_the_cross_model_alone(cross_setup, tmp_path):
    from jul.bundle import Bundle, pack
    client, _ = cross_setup
    questions = {k: MIXED[k] for k in ("urgent", "anger")}
    bundle = Bundle.load(pack(client, questions, tmp_path / "bundle"))
    assert bundle.models == ["cross"] and bundle.backbone is None
    for text in TEXTS:
        want, got = answers(client.system_one(state=text, questions=questions)), answers(bundle.system_one(text))
        for name in questions:
            assert np.allclose(got[name], want[name], atol=1e-4)


@needs_export
def test_packing_as_vectors_ships_the_vector_model_alone(cross_setup, tmp_path):
    from jul.bundle import Bundle, pack
    client, _ = cross_setup
    bundle = Bundle.load(pack(client, MIXED, tmp_path / "bundle", reading="vector"))
    assert bundle.models == ["vector"] and bundle.cross is None
    want = answers(client.system_one(state=TEXTS[1], questions=MIXED, method="vector"))
    got = answers(bundle.system_one(TEXTS[1]))
    for name in MIXED:
        assert np.allclose(got[name], want[name], atol=1e-4)


@needs_export
def test_each_model_reads_its_own_graph_override(cross_setup, monkeypatch):
    from jul import cross
    client, repo = cross_setup
    entry = {"repo": str(repo), "subfolder": "cross"}
    monkeypatch.setenv("JUL_ONNX_MODEL", str(repo / "missing.onnx"))       # the vector model's, not the cross one's
    assert cross.load(entry, "onnx").spec.types == ("noul", "score")
    monkeypatch.delenv("JUL_ONNX_MODEL")
    monkeypatch.setenv("JUL_ONNX_CROSS_MODEL", str(repo / "missing.onnx"))
    with pytest.raises(Exception):
        cross.load(entry, "onnx")


@needs_export
def test_autotune_keeps_the_cross_model_unless_the_head_beats_it(cross_setup, monkeypatch):
    """The head is judged against the cross model's zero-shot answers; losing, it leaves the question to it."""
    import jul.tuning
    from jul.context import Context
    client, _ = cross_setup
    q = {"urgent": Noul("Is it urgent?")}
    labeled = [(t, {"urgent": i % 2 == 0}) for i, t in enumerate(TEXTS * 4)]
    engine = client._engine_for(None)
    seen = {}
    real_train = jul.tuning.train

    def judged(features, y, zero_shot_scores, *args, **kwargs):
        seen["baseline"] = zero_shot_scores
        return real_train(features, y, zero_shot_scores, *args, **kwargs)

    monkeypatch.setattr(jul.tuning, "train", judged)
    ctx = Context()
    client.autotune(ctx, q, labeled, save=False)
    want = np.stack([engine.cross.logits(t, "noul", "Is it urgent?", options_of(q["urgent"]))[0] for t, _ in labeled])
    assert np.allclose(seen["baseline"], want)                    # judged against the cross model
    digest = client._digest("noul", q["urgent"], options_of(q["urgent"]))
    assert digest not in ctx.calibration                           # no vector calibration to take it over

    def losing(*args, **kwargs):
        return None, jul.tuning.TuningReport("urgent", 12, 12, 2, 0.9, 0.5, False, "does not beat zero-shot")
    monkeypatch.setattr(jul.tuning, "train", losing)
    ctx = Context()
    report = client.autotune(ctx, q, labeled, save=False)["urgent"]
    assert "cross model, which keeps the question" in report.reason
    assert client._default_method(engine, ctx, "noul", q["urgent"], options_of(q["urgent"])) == "cross"

    def winning(features, y, *args, **kwargs):
        head = {"W": np.zeros((features.shape[1], 2)), "b": np.zeros(2), "meta": {"features": "vector"}}
        return head, jul.tuning.TuningReport("urgent", 12, 12, 2, 0.5, 0.9, True, "beats zero-shot")
    monkeypatch.setattr(jul.tuning, "train", winning)
    ctx = Context()
    client.autotune(ctx, q, labeled, save=False)
    assert client._default_method(engine, ctx, "noul", q["urgent"], options_of(q["urgent"])) == "vector"
