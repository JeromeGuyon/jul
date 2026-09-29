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


# --- LoRA cross models: adapters on the preset's own decoder (tiny random Qwen3, torch) --------------

LISTWISE = {"reading": "listwise", "before": 'Text: "', "after": '"\nQuestion: {question}\nOptions:\n',
            "option": "- {option}", "separator": "\n", "answer": "Answer:", "max_state": 16, "max_option": 6,
            "mix": 3.0}


def write_lora(tmp_path, backbone, scale=2.0, zero=False, choice=None):
    """Random adapters on two Linear layers of the tiny decoder, random heads (and a listwise choice)."""
    from jul.cross import LoraSpec
    rng = np.random.default_rng(0)
    dec = backbone._decoder
    weights = {}
    for path, lin in (("layers.0.self_attn.q_proj", dec.layers[0].self_attn.q_proj),
                      ("layers.2.mlp.down_proj", dec.layers[2].mlp.down_proj)):
        lin = getattr(lin, "base", lin)                  # the session's backbone may carry adapters already
        weights[f"{path}.a"] = rng.normal(0, 0.5, (4, lin.in_features)).astype(np.float16)
        weights[f"{path}.b"] = (np.zeros if zero else lambda s: rng.normal(0, 0.5, s))((lin.out_features, 4)).astype(np.float16)
    np.savez(tmp_path / "adapter.npz", **weights)
    d = dec.config.hidden_size
    heads = (("noul", 3), ("choice", 1), ("score", 1)) + ((("choice_q", 8), ("choice_k", 8)) if choice else ())
    np.savez(tmp_path / "cross_heads.npz", **{f"{t}_{k}": rng.normal(0, 1, (n, d) if k == "weight" else (n,)).astype(np.float32)
                                             for t, n in heads for k in ("weight", "bias")})
    (tmp_path / "cross.json").write_text(json.dumps({
        "method": "lora", "base": "tiny", "scale": scale, "adapter": "adapter.npz", "heads": "cross_heads.npz",
        "types": ["noul", "score"] + (["choice"] if choice else []), "max_length": 48,
        "prompt": 'Text: "{state}"\n{first}\nVerdict:',
        "firsts": {"noul": "Question: {question}", "score": "Question: {question}\nCandidate answer: {option}"},
        **({"choice": choice} if choice else {})}))
    return LoraSpec.load(tmp_path)


@needs_export
def test_lora_adapters_leave_the_vector_reading_untouched_and_change_the_cross_one(pair, tmp_path):
    from jul.backbone import PromptTemplate
    from jul.cross import LoraCrossReader
    from jul.presets import ONE_WORD
    torch_bb, _ = pair
    tpl = PromptTemplate(torch_bb, *ONE_WORD.split("{state}"))
    texts = ["I was charged twice", "Le colis n'est jamais arrivé"]
    before = [f[3] for f in tpl.run_batch(texts, layers=[3])]
    reader = LoraCrossReader(torch_bb, write_lora(tmp_path, torch_bb))
    try:
        after = [f[3] for f in tpl.run_batch(texts, layers=[3])]
        assert all(np.array_equal(x, y) for x, y in zip(before, after))          # adapters off: same model
        opts = options_of(Noul(instructions="Was it paid on time?"))
        z, tokens = reader.logits("Paid on May 3, due May 9.", "noul", "Was it paid on time?", opts)
        assert z.shape == (2,) and np.isclose(np.exp(z).sum(), 1.0, atol=1e-5) and tokens > 0
        levels = options_of(Score(instructions="How urgent?", criteria=["low", "medium", "high"]))
        assert reader.logits("x", "score", "How urgent?", levels)[0].shape == (3,)
        after2 = [f[3] for f in tpl.run_batch(texts, layers=[3])]
        assert all(np.array_equal(x, y) for x, y in zip(before, after2))         # switched off again
        # the same prompt through adapters whose B is zero reads as the plain decoder: the adapters did something
        on = reader.logits("Paid on May 3, due May 9.", "noul", "Was it paid on time?", opts)[0]
        for lo in reader._loras:
            lo.b.zero_()
        off = reader.logits("Paid on May 3, due May 9.", "noul", "Was it paid on time?", opts)[0]
        assert not np.allclose(on, off)
    finally:
        for lo in reader._loras:                                                 # the backbone is session-wide
            lo.on = False
            lo.b.zero_()


@needs_export
def test_lora_prompt_cuts_the_text_to_max_length(pair, tmp_path):
    from jul.cross import LoraCrossReader
    torch_bb, _ = pair
    reader = LoraCrossReader(torch_bb, write_lora(tmp_path, torch_bb, zero=True))
    first = reader.firsts("noul", "Is it late?", options_of(Noul(instructions="Is it late?")))[0]
    assert first == "Question: Is it late?"
    ids = reader._prompt(first, "word " * 500)
    assert len(ids) <= 48 and ids[-3:] == torch_bb.encode("\nVerdict:")[-3:]


def test_a_lora_cross_model_needs_the_presets_backbone(tmp_path):
    from jul import cross
    (tmp_path / "cross.json").write_text(json.dumps({"method": "lora"}))
    with pytest.raises(ValueError, match="own backbone"):
        cross.load({"repo": str(tmp_path)}, "torch")


@needs_export
def test_listwise_choice_reads_every_option_in_one_pass(pair, tmp_path):
    from jul.cross import LoraCrossReader
    torch_bb, _ = pair
    reader = LoraCrossReader(torch_bb, write_lora(tmp_path, torch_bb, choice=LISTWISE))
    try:
        q = Choice(instructions="Which team?", criteria={"billing": "charges and invoices", "shipping": "",
                                                         "returns": "refunds"})
        opts = options_of(q)
        ids, ends = reader.listwise("I was charged twice " * 20, "Which team?", opts)
        sep = torch_bb.encode("\n")
        assert len(ends) == 3 and all(ids[e - len(sep) + 1: e + 1] == sep for e in ends)
        assert ids[-len(torch_bb.encode("Answer:")):] == torch_bb.encode("Answer:")
        assert ids[len(torch_bb.encode('Text: "')): ][:16] == torch_bb.encode("I was charged twice " * 20)[:16]
        z, tokens = reader.logits("I was charged twice", "choice", "Which team?", opts)
        assert z.shape == (3,) and np.isfinite(z).all() and tokens > 0
        assert all(not lo.on for lo in reader._loras)                          # switched off after the pass
        # options interact: another option list changes the scores of the ones kept
        z2 = reader.logits("I was charged twice", "choice", "Which team?", options_of(
            Choice(instructions="Which team?", criteria={"billing": "charges and invoices", "shipping": ""})))[0]
        assert not np.allclose(z[:2], z2)
        assert reader.vector_mix("choice") == 3.0 and reader.vector_mix("noul") is None
    finally:
        for lo in reader._loras:
            lo.on = False
            lo.b.zero_()


def test_listwise_choice_needs_its_heads(pair, tmp_path):
    from jul.cross import LoraCrossReader, LoraSpec
    torch_bb, _ = pair
    spec = write_lora(tmp_path, torch_bb, zero=True, choice=LISTWISE)
    w = dict(np.load(tmp_path / "cross_heads.npz"))
    np.savez(tmp_path / "cross_heads.npz", **{k: v for k, v in w.items() if not k.startswith("choice_k")})
    with pytest.raises(ValueError, match="choice_q"):
        LoraCrossReader(torch_bb, LoraSpec.load(tmp_path))
    assert spec.choice["mix"] == 3.0
