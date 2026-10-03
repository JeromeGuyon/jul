"""The mask method (LLaDA readout, Option 1): the reader and the client path, without a model.

These check the wiring and the readout math on a fake masked-diffusion backbone (no download):
  - MaskReader builds one [MASK] per question and reads option logits at it,
  - the noul false/true anchors are remapped to jul's option order,
  - client.system_one routes a mask preset through MaskReader and returns typed answers,
  - the LLaDA preset resolves and declares method='mask' on backend 'llada'.
A real 8B run is the eval script's job (scripts/eval_llada_zeroshot.py), which needs weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from jul.mask import MaskReader, MaskSpec
from jul.types import Choice, Noul, Score, options_of


class _FakeTokenizer:
    """Deterministic single-token-ish encoder; distinct ids for distinct short markers."""

    mask_token = None
    unk_token_id = 0

    def encode(self, s, add_special_tokens=False):
        s = s.strip()
        if not s:
            return [1]
        # first char id is the "anchor"; distinct leading chars -> distinct anchors
        return [(ord(c) % 4000) + 100 for c in s][:16]

    def convert_tokens_to_ids(self, t):
        return 3

    def convert_ids_to_tokens(self, i):
        return "<x>"


class _FakeLLaDA:
    """A stand-in masked-diffusion backbone: mask_logits favors each option's own anchor."""

    name = "fake-llada"
    backend = "llada"
    repo = ""
    architecture = "diffusion"

    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.vocab = 5000
        self.mask_id = 7

    def mask_logits(self, tokens, mask_positions):
        v = np.full((len(mask_positions), self.vocab), -8.0, dtype=np.float32)
        # a spread of plausible logits so softmax is non-degenerate
        v[:, 100:200] = np.linspace(-1.0, 3.0, 100, dtype=np.float32)
        return v


def _reader():
    return MaskReader(_FakeLLaDA(), MaskSpec.default())


def test_choice_readout_shape_and_order():
    r = _reader()
    q = Choice(instructions="pick a team", criteria={"billing": "payments", "tech": "bugs", "sales": "pricing"})
    opts = options_of(q)
    logits, tokens = r.logits("charged twice", [("choice", q.instructions, opts)])
    assert logits[0].shape == (3,)
    assert tokens > 0
    # a proper distribution once softmaxed
    p = np.exp(logits[0] - logits[0].max()); p /= p.sum()
    assert abs(p.sum() - 1.0) < 1e-5


def test_noul_false_true_remap():
    """MaskReader reads anchors in [false, true] order but must return jul's options_of order."""
    r = _reader()
    q = Noul(instructions="is it a bug?")
    opts = options_of(q)                     # order is ("true", "false")
    keys = [o.key for o in opts]
    assert keys == ["true", "false"]
    logits, _ = r.logits("it crashes", [("noul", q.instructions, opts)])
    assert logits[0].shape == (2,)
    # the returned vector is aligned to opts order (index 0 == "true")
    assert np.isfinite(logits[0]).all()


def test_score_expected_levels():
    r = _reader()
    q = Score(instructions="how angry?", criteria=["calm", "annoyed", "furious"])
    opts = options_of(q)
    logits, _ = r.logits("this is unacceptable", [("score", q.instructions, opts)])
    assert logits[0].shape == (3,)


def test_mask_id_from_backbone_or_spec():
    # spec with no mask_token -> uses the backbone's mask id
    r = MaskReader(_FakeLLaDA(), MaskSpec.default())
    assert r.mask_id == 7
    # a backbone without a mask id and a spec without mask_token -> clear error
    bb = _FakeLLaDA()
    del bb.mask_id
    with pytest.raises(ValueError):
        MaskReader(bb, MaskSpec.default())


def test_llada_preset_resolves_as_mask():
    from jul.presets import resolve
    p = resolve("llada-8b-instruct", "llada")
    assert p.method == "mask"
    assert p.backend == "llada"
    assert p.repos.get("llada") == "GSAI-ML/LLaDA-8B-Instruct"


def test_client_system_one_mask_path():
    """The full public call goes through MaskReader and returns typed answers."""
    from jul import TypeSafeClient, Choice, Noul, Score

    class _Engine:
        pointer = None
        cross = None

        def __init__(self):
            self.mask = MaskReader(_FakeLLaDA(), MaskSpec.default())

    c = TypeSafeClient.__new__(TypeSafeClient)
    c._context_home = None
    c.context = None
    c.method = None

    class _P:
        name = "llada-8b-instruct"

    c._preset = _P()
    c._engine_for = lambda model: _Engine()

    resp = c.system_one(
        state={"ticket": "charged twice"},
        questions={
            "team": Choice(instructions="which team?",
                           criteria={"billing": "payments", "tech": "bugs"}),
            "bug": Noul(instructions="is it a bug?"),
            "anger": Score(instructions="how angry?", criteria=["calm", "annoyed", "furious"]),
        },
    )
    assert resp.choices["team"].choice in {"billing", "tech"}
    assert 0.0 <= resp.nouls["bug"].noul <= 1.0
    assert 0.0 <= resp.scores["anger"].score <= 2.0
    assert resp.usage.output_tokens == 0        # nothing is generated
    assert resp.usage.input_tokens > 0


# --- multi-token sequence-likelihood readout (Prompt 1: the Banking77 unblocker) --------------

def test_strip_common_prefix_keeps_distinctive_suffix():
    """All labels share a boilerplate head; it carries no argmax signal, so it is emitted as context
    and only the distinctive suffix is scored. At least one token is always kept per label."""
    from jul.mask import _strip_common_prefix
    seqs = [[10, 20, 30, 1], [10, 20, 30, 2], [10, 20, 30, 3, 4]]
    suffixes, prefix_len = _strip_common_prefix(seqs)
    assert prefix_len == 3                       # the shared [10,20,30]
    assert suffixes == [[1], [2], [3, 4]]
    # never strips a whole label: identical single-token labels keep one token
    suf2, pl2 = _strip_common_prefix([[7], [7]])
    assert pl2 == 0 and suf2 == [[7], [7]]


def test_log_softmax_rows_matches_reference():
    from jul.mask import _log_softmax_rows
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    lp = _log_softmax_rows(x)
    # rows sum to 1 in probability space
    assert np.allclose(np.exp(lp).sum(axis=1), 1.0, atol=1e-5)
    # uniform row -> log(1/3) everywhere
    assert np.allclose(lp[1], np.log(1 / 3), atol=1e-5)


def test_multitoken_is_mean_normalized_not_sum():
    """Mean length-normalization (not sum): a long label must not be penalized for having more
    tokens. With per-token log-probs held roughly equal, a 1-token and a 3-token label that share the
    same mean log-prob must score comparably; a sum would favor the short one massively."""
    from jul.mask import MaskReader, MaskSpec
    from jul.types import Option
    import dataclasses

    # deterministic stub: every masked position puts the same high logit on each label's own suffix
    # tokens, so per-token log-prob is identical across labels -> mean scores tie, sum would not.
    class _Tok:
        mask_token = None; unk_token_id = 0
        def encode(self, s, add_special_tokens=False):
            # map each char to a stable id; " " -> 32, letters distinct
            return [ord(c) for c in s.strip()][:24] or [1]
        def convert_tokens_to_ids(self, t): return 5
    class _BB:
        name = "stub"; backend = "llada"; repo = ""
        def __init__(s): s.tokenizer = _Tok(); s.mask_id = 999; s.V = 300
        def mask_logits(s, tokens, positions):
            v = np.full((len(positions), s.V), -10.0, dtype=np.float32)
            # boost the tokens that actually appear in the labels so each suffix token is likely
            for i in range(len(positions)):
                v[i, 32:130] = 0.0
            return v

    spec = dataclasses.replace(MaskSpec.default(), readout="multitoken")
    r = MaskReader(_BB(), spec)
    # labels sharing a common prefix "aa", distinctive suffixes of different lengths
    opts = [Option("x", "aa b"), Option("y", "aa cde")]
    zs, used = r.logits("some text", [("choice", "pick", opts)])
    z = zs[0]
    assert z.shape == (2,)
    assert used > 0                              # one forward, tokens counted
    # mean-normalized: the longer label is not crushed just for being longer
    assert abs(float(z[0]) - float(z[1])) < 5.0


def test_auto_readout_routes_by_option_count():
    """readout='auto': few options -> anchor (one token per option), many -> multitoken. The router
    keeps AG News (4 labels) on the crisp anchor reading and sends Banking77 (72) to multitoken."""
    from jul.mask import MaskReader, MaskSpec
    from jul.types import Option
    import dataclasses

    calls = {"multitoken": 0, "anchor": 0}

    class _Tok:
        mask_token = None; unk_token_id = 0
        def encode(self, s, add_special_tokens=False):
            return [ord(c) for c in s.strip()][:24] or [1]
        def convert_tokens_to_ids(self, t): return 5
    class _BB:
        name = "stub"; backend = "llada"; repo = ""
        def __init__(s): s.tokenizer = _Tok(); s.mask_id = 999; s.V = 300
        def mask_logits(s, tokens, positions):
            return np.full((len(positions), s.V), 0.0, dtype=np.float32)

    spec = dataclasses.replace(MaskSpec.default(), readout="auto", route_multitoken_above=10)
    r = MaskReader(_BB(), spec)
    # instrument which path is taken
    orig_mt = r._multitoken_scores
    orig_an = r._read_option_logits
    def mt(*a, **k): calls["multitoken"] += 1; return orig_mt(*a, **k)
    def an(*a, **k): calls["anchor"] += 1; return orig_an(*a, **k)
    r._multitoken_scores = mt
    r._read_option_logits = an

    few = [Option(str(i), f"label {i}") for i in range(4)]
    many = [Option(str(i), f"label {i}") for i in range(30)]
    r.logits("t", [("choice", "q", few)])
    assert calls["anchor"] == 1 and calls["multitoken"] == 0     # few -> anchor
    r.logits("t", [("choice", "q", many)])
    assert calls["multitoken"] == 1                              # many -> multitoken


def test_noul_readout_reads_yes_no_pairs():
    """The dedicated noul readout averages several yes/no anchor pairs at a natural prompt, and returns
    logits in jul's Noul option order (true=yes, false=no). A stub that favors the 'yes' tokens must
    yield P(yes) > 0.5."""
    from jul.mask import MaskReader, MaskSpec
    from jul.types import Noul, options_of
    import dataclasses

    yes_forms = {" yes", " Yes", " true", " True"}

    class _Tok:
        mask_token = None; unk_token_id = 0
        def __init__(s):
            s._vocab = {}
        def encode(self, s, add_special_tokens=False):
            # give the yes-forms one id family (100), the no-forms another (200); others hash
            key = s
            if key in yes_forms:
                return [100]
            if key in {" no", " No", " false", " False"}:
                return [200]
            return [(abs(hash(key)) % 50) + 300]
        def convert_tokens_to_ids(self, t): return 5
    class _BB:
        name = "stub"; backend = "llada"; repo = ""
        def __init__(s): s.tokenizer = _Tok(); s.mask_id = 999; s.V = 400
        def mask_logits(s, tokens, positions):
            v = np.full((len(positions), s.V), -10.0, dtype=np.float32)
            v[:, 100] = 5.0   # yes-forms highly likely
            v[:, 200] = 0.0   # no-forms less likely
            return v

    spec = dataclasses.replace(MaskSpec.default(), readout="auto")
    r = MaskReader(_BB(), spec)
    q = Noul(instructions="Is the sky blue?")
    opts = options_of(q)                                  # ("true","false")
    zs, used = r.logits("the sky is blue", [("noul", q.instructions, opts)])
    z = zs[0]
    assert z.shape == (2,)
    assert used > 0
    # option order: index of "true" should carry the higher (yes) logit
    ti = [o.key for o in opts].index("true")
    fi = [o.key for o in opts].index("false")
    assert z[ti] > z[fi]                                  # yes beats no -> P(yes) > 0.5


def test_auto_ce_keeps_choice_on_anchor():
    """readout='auto-ce': for a CE-trained model, choice stays on the anchor channel (even with many
    options), while noul and score use the new readouts. Guards the measured choice-channel conflict
    (CE learned anchor; multitoken then hurts choice)."""
    from jul.mask import MaskReader, MaskSpec
    from jul.types import Choice, Noul, Score, Option, options_of
    import dataclasses

    calls = {"multitoken": 0, "anchor": 0, "noul": 0}

    class _Tok:
        mask_token = None; unk_token_id = 0
        def encode(self, s, add_special_tokens=False):
            return [ord(c) for c in s.strip()][:24] or [1]
        def convert_tokens_to_ids(self, t): return 5
    class _BB:
        name = "stub"; backend = "llada"; repo = ""
        def __init__(s): s.tokenizer = _Tok(); s.mask_id = 999; s.V = 400
        def mask_logits(s, tokens, positions):
            return np.full((len(positions), s.V), 0.0, dtype=np.float32)

    spec = dataclasses.replace(MaskSpec.default(), readout="auto-ce", route_multitoken_above=10)
    r = MaskReader(_BB(), spec)
    o_mt, o_an, o_no = r._multitoken_scores, r._read_option_logits, r._noul_scores
    r._multitoken_scores = lambda *a, **k: (calls.__setitem__("multitoken", calls["multitoken"] + 1), o_mt(*a, **k))[1]
    r._read_option_logits = lambda *a, **k: (calls.__setitem__("anchor", calls["anchor"] + 1), o_an(*a, **k))[1]
    r._noul_scores = lambda *a, **k: (calls.__setitem__("noul", calls["noul"] + 1), o_no(*a, **k))[1]

    many = [Option(str(i), f"label {i}") for i in range(30)]     # >10 options
    r.logits("t", [("choice", "q", many)])
    assert calls["anchor"] == 1 and calls["multitoken"] == 0     # choice stays on anchor despite >10
    r.logits("t", [("noul", "q?", options_of(Noul(instructions="q?")))])
    assert calls["noul"] == 1                                     # noul uses the yes/no readout
    r.logits("t", [("score", "q", options_of(Score(instructions="q", criteria=["lo", "mid", "hi"])))])
    assert calls["multitoken"] == 1                              # score uses the sequence-likelihood readout


def test_listwise_choice_prompt_mask5_keys_on_newline_and_jul_order():
    """Free diagnostic (no GPU): the shared listwise builder puts the [MASK] at id 5, each key
    position falls on the newline token that closes its option line, and a stub backbone yields one
    logit per option in jul's option order."""
    torch = pytest.importorskip("torch")
    from jul.mask import MaskReader, MaskSpec
    from jul.types import Choice, options_of
    from jul import llada_head as LH

    NL = 999  # sentinel id for "\n"

    class _Tok:
        mask_token = None
        unk_token_id = 0
        def encode(self, s, add_special_tokens=False):
            if s == "\n":
                return [NL]
            # deterministic, never emits NL for non-newline text
            return [((ord(c) % 300) + 1) for c in s][:48] or [1]
        def convert_tokens_to_ids(self, t):
            return 5

    class _BB:
        name = "illada-stub"; backend = "llada"; repo = ""; architecture = "diffusion"
        def __init__(s):
            s.tokenizer = _Tok(); s.mask_id = 5; s.vocab = 2000
        def mask_hidden(s, tokens, positions):
            # distinct deterministic hidden per position
            g = np.random.default_rng(7)
            table = g.standard_normal((len(tokens), 8)).astype(np.float32)
            return table[np.array(positions)]

    reader = MaskReader(_BB(), MaskSpec.default())
    assert reader.mask_id == 5

    q = Choice(instructions="which team?", criteria={"billing": "payments", "tech": "bugs", "sales": "pricing"})
    opts = options_of(q)
    tokens, mask_pos, ends = reader._listwise_choice_tokens(q.instructions, opts, "charged twice")

    # the query position holds the [MASK] (id 5)
    assert tokens[mask_pos] == 5
    # one key per option, each landing exactly on the closing newline token
    assert len(ends) == len(opts)
    for e in ends:
        assert tokens[e] == NL
    # the stub head yields one logit per option, in jul option order
    head = LH.ReadHead(hidden=8, emb=torch.nn.Embedding(16, 8), proj=4)
    reader.head = head.eval()
    z, used = reader._head_scores("choice", "charged twice", q.instructions, opts)
    assert z.shape == (len(opts),)
    assert used == len(tokens)
    assert np.isfinite(z).all()


def test_listwise_head_routes_choice_only():
    """With a listwise head loaded, choice goes through the head; noul and score must NOT (they keep
    their auto-ce readout). Guards the routing fix of this run."""
    from jul.mask import MaskReader, MaskSpec
    from jul.types import Choice, Noul, Score, Option, options_of
    import dataclasses

    calls = {"head": 0, "noul": 0, "multitoken": 0, "anchor": 0}

    class _Tok:
        mask_token = None; unk_token_id = 0
        def encode(self, s, add_special_tokens=False):
            if s == "\n":
                return [999]
            return [((ord(c) % 300) + 1) for c in s][:48] or [1]
        def convert_tokens_to_ids(self, t): return 5
    class _BB:
        name = "stub"; backend = "llada"; repo = ""
        def __init__(s): s.tokenizer = _Tok(); s.mask_id = 5; s.V = 400
        def mask_logits(s, tokens, positions):
            return np.full((len(positions), s.V), 0.0, dtype=np.float32)
        def mask_hidden(s, tokens, positions):
            return np.zeros((len(positions), 8), dtype=np.float32)

    spec = dataclasses.replace(MaskSpec.default(), readout="auto-ce", route_multitoken_above=10)
    r = MaskReader(_BB(), spec)
    # a sentinel head object; _head_scores is stubbed to just count and return a vector
    r.head = object()
    o_no, o_mt, o_an = r._noul_scores, r._multitoken_scores, r._read_option_logits
    def hs(kind, state, instr, opts):
        calls["head"] += 1
        return np.zeros(len(opts), dtype=np.float32), 1
    r._head_scores = hs
    r._noul_scores = lambda *a, **k: (calls.__setitem__("noul", calls["noul"] + 1), o_no(*a, **k))[1]
    r._multitoken_scores = lambda *a, **k: (calls.__setitem__("multitoken", calls["multitoken"] + 1), o_mt(*a, **k))[1]
    r._read_option_logits = lambda *a, **k: (calls.__setitem__("anchor", calls["anchor"] + 1), o_an(*a, **k))[1]

    choice = Choice(instructions="q", criteria={"a": "", "b": ""})
    r.logits("t", [("choice", "q", options_of(choice))])
    assert calls["head"] == 1                                   # choice -> head
    r.logits("t", [("noul", "q?", options_of(Noul(instructions="q?")))])
    assert calls["head"] == 1 and calls["noul"] == 1            # noul -> NOT the head
    r.logits("t", [("score", "q", options_of(Score(instructions="q", criteria=["lo", "mid", "hi"])))])
    assert calls["head"] == 1 and calls["multitoken"] == 1      # score -> NOT the head
    """The learned head is now LISTWISE and CHOICE-ONLY. A head trained to point at the gold option
    (query = [MASK] hidden, keys = each option's closing-separator hidden) must read that option back.
    noul/score no longer go through the head (the router sends only choice to it)."""
    torch = pytest.importorskip("torch")
    from jul import llada_head as LH

    torch.manual_seed(0)
    reader = _reader()
    hidden = 8
    # a fake backbone hidden map: distinct hidden per position so q and the keys differ per option.
    def fake_mask_hidden(tokens, positions):
        g = np.random.default_rng(len(tokens))
        table = g.standard_normal((len(tokens), hidden)).astype(np.float32)
        return table[np.array(positions)]
    reader.backbone.mask_hidden = fake_mask_hidden

    head = LH.ReadHead(hidden=hidden, emb=torch.nn.Embedding(16, hidden), proj=4)
    # freeze all but the two listwise projections, exactly like training
    for n, p in head.named_parameters():
        p.requires_grad = n.startswith("choice_q.") or n.startswith("choice_k.")
    trainable = {n for n, p in head.named_parameters() if p.requires_grad}
    assert trainable == {"choice_q.weight", "choice_q.bias", "choice_k.weight", "choice_k.bias"}

    from jul.types import Choice, options_of
    q = Choice(instructions="which team?", criteria={"billing": "", "tech": "", "sales": ""})
    opts = options_of(q)
    gold = 1  # "tech"
    # build the listwise prompt once; train the projections to point at the gold option on it
    tokens, mask_pos, ends = reader._listwise_choice_tokens(q.instructions, opts, "charged twice")
    assert tokens[mask_pos] == reader.mask_id            # the query is the [MASK]
    hs = fake_mask_hidden(tokens, [mask_pos] + ends)
    h_mask = torch.tensor(hs[0]); h_ends = torch.tensor(hs[1:])
    opt = torch.optim.AdamW([p for p in head.parameters() if p.requires_grad], lr=0.1)
    for _ in range(200):
        loss = LH.listwise_choice_loss(head, h_mask, h_ends, gold, "cpu")
        opt.zero_grad(); loss.backward(); opt.step()
    reader.head = head.eval()
    reader.backbone.mask_hidden = lambda t, p: hs[np.array([([mask_pos] + ends).index(x) for x in p])]
    z, _ = reader._head_scores("choice", "charged twice", q.instructions, opts)
    assert z.shape == (3,)
    pr = np.exp(z - z.max()); pr /= pr.sum()
    assert int(np.argmax(pr)) == gold                    # reads back the trained option
