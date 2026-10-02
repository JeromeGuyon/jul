"""Option 2 (LLaDA as a training target): the calibrated losses and the example builder.

Pure-maths tests on CPU (no model, no download). The torch train_step is exercised separately by the
smoke test in docs/llada-training.md; here we lock the properties that make RLCD "calibration-first".
"""

from __future__ import annotations

import numpy as np
import pytest

from jul.mask import MaskReader, MaskSpec
from jul.types import Choice, Noul, options_of

llada_train = pytest.importorskip("llada_train")  # scripts/ on sys.path via conftest/rootdir


def test_cross_entropy_rewards_the_gold():
    right = np.array([5.0, 0.0, 0.0])
    wrong = np.array([0.0, 5.0, 0.0])
    assert llada_train.cross_entropy(right, 0) < llada_train.cross_entropy(wrong, 0)


def test_brier_is_zero_when_certain_and_correct():
    assert llada_train.brier(np.array([50.0, 0.0, 0.0]), 0) == pytest.approx(0.0, abs=1e-6)


def test_kl_distillation_matches_teacher():
    # student == teacher -> KL ~ 0
    z = np.log(np.array([0.7, 0.2, 0.1]))
    assert llada_train.kl_distillation(z, np.array([0.7, 0.2, 0.1])) == pytest.approx(0.0, abs=1e-6)


def test_rlcd_penalizes_confident_error_more_than_uncertain():
    """The load-bearing RLCD property: being confidently wrong costs more than being unsure."""
    confident_wrong = np.array([0.0, 5.0, 0.0])   # gold=0, sure of 1
    unsure_wrong = np.array([0.4, 0.5, 0.1])      # gold=0, barely picks 1
    confident_right = np.array([5.0, 0.0, 0.0])
    assert llada_train.rlcd_loss(confident_wrong, 0) > llada_train.rlcd_loss(unsure_wrong, 0)
    assert llada_train.rlcd_loss(confident_right, 0) < llada_train.rlcd_loss(unsure_wrong, 0)


def test_build_example_reuses_inference_prompt():
    class _Tok:
        mask_token = None
        unk_token_id = 0
        def encode(self, s, add_special_tokens=False):
            s = s.strip()
            return [(ord(c) % 4000) + 100 for c in s][:16] or [1]
        def convert_tokens_to_ids(self, t): return 3

    class _BB:
        name = "toy"; backend = "llada"; repo = ""
        def __init__(self): self.tokenizer = _Tok(); self.mask_id = 7

    reader = MaskReader(_BB(), MaskSpec.default())
    q = Choice(instructions="pick", criteria={"a": "alpha", "b": "beta"})
    opts = options_of(q)
    ex = llada_train.build_example(reader, "choice", q.instructions, opts, "state text", gold_index=1)
    assert ex.tokens[ex.mask_pos] == reader.mask_id       # [MASK] at the read position
    assert len(ex.anchor_ids) == len(opts)
    assert ex.gold_index == 1


def test_build_example_noul_targets_follow_option_order():
    class _Tok:
        mask_token = None
        unk_token_id = 0
        def encode(self, s, add_special_tokens=False):
            return [(ord(c) % 4000) + 100 for c in s.strip()][:16] or [1]
        def convert_tokens_to_ids(self, t): return 3

    class _BB:
        name = "toy"; backend = "llada"; repo = ""
        def __init__(self): self.tokenizer = _Tok(); self.mask_id = 7

    reader = MaskReader(_BB(), MaskSpec.default())
    q = Noul(instructions="is it true?")
    opts = options_of(q)                    # ("true", "false")
    ex = llada_train.build_example(reader, "noul", q.instructions, opts, "state", gold_index=0)
    assert len(ex.anchor_ids) == 2          # anchors reordered to option order (true, false)


# --- masked-diffusion (LLaDA) and GIFT loss properties (pure numpy, no model) ------------------


def test_entropy_is_max_for_uniform_and_zero_for_certain():
    uniform = np.zeros(4)                        # softmax -> uniform, max entropy log(4)
    certain = np.array([50.0, 0.0, 0.0, 0.0])    # softmax -> ~one-hot, ~0 entropy
    assert llada_train.entropy(uniform) == pytest.approx(np.log(4), abs=1e-6)
    assert llada_train.entropy(certain) == pytest.approx(0.0, abs=1e-4)
    assert llada_train.entropy(uniform) > llada_train.entropy(certain)


def test_diffusion_loss_has_the_one_over_t_factor():
    """LLaDA's load-bearing 1/t reweighting: diffusion_loss = (1/t) * cross_entropy."""
    z = np.array([2.0, 0.0, 1.0])
    ce = llada_train.cross_entropy(z, 0)
    for t in (0.1, 0.25, 0.5, 0.9):
        assert llada_train.diffusion_loss(z, 0, t) == pytest.approx(ce / t, rel=1e-9)


def test_diffusion_loss_upweights_small_t():
    """Smaller masking level t (heavier reconstruction burden per token) -> larger weight."""
    z = np.array([1.5, 0.2, 0.3])
    assert llada_train.diffusion_loss(z, 0, 0.1) > llada_train.diffusion_loss(z, 0, 0.9)


def test_diffusion_loss_reduces_to_ce_at_t_one():
    z = np.array([1.0, 0.5, 0.0])
    assert llada_train.diffusion_loss(z, 0, 1.0) == pytest.approx(llada_train.cross_entropy(z, 0))


def test_gift_betas_are_sqrt_of_entropy():
    ents = [0.0, 0.25, 1.0, 4.0]
    betas = llada_train.gift_betas(ents)
    assert np.allclose(betas, np.sqrt(ents))


def test_gift_masking_prob_increases_with_entropy():
    """GIFT Thm 1 monotonicity: a higher-entropy token gets a larger t_i (masked/learned more)."""
    t = 0.3
    ref = 1.0                                    # beta_ref
    low = llada_train.gift_masking_prob(0.5, ref, t)   # beta_i < beta_ref
    mid = llada_train.gift_masking_prob(1.0, ref, t)   # beta_i == beta_ref  -> t_i == t
    high = llada_train.gift_masking_prob(2.0, ref, t)  # beta_i > beta_ref
    assert low < mid < high
    assert mid == pytest.approx(t, abs=1e-9)     # at beta_i == beta_ref, t_i == t exactly


def test_gift_upmasks_uncertain_tokens():
    """The GIFT mechanism (arXiv 2509.20863, Thm 1): at a shared masking level t, a more-uncertain
    decision token gets a *larger masking probability* t_i, so it is masked — and therefore learned —
    more often than a confident one. This is the entropy-importance schedule; the 1/t_i weight then
    makes each masking event an unbiased 1/t_i-scaled reconstruction term (GIFT Alg. 2)."""
    t = 0.3
    high_H = llada_train.entropy(np.zeros(4))               # uniform -> high entropy
    low_H = llada_train.entropy(np.array([50.0, 0, 0, 0]))  # near one-hot -> low entropy
    betas = llada_train.gift_betas([high_H, low_H])
    beta_ref = float(np.mean(betas))
    t_high = llada_train.gift_masking_prob(float(betas[0]), beta_ref, t)
    t_low = llada_train.gift_masking_prob(float(betas[1]), beta_ref, t)
    assert t_high > t_low                                   # uncertain token masked/learned more
    # and the weight is exactly the inverse masking probability (the diffusion change of variable)
    assert llada_train.gift_weight(high_H, beta_ref, t) == pytest.approx(1.0 / t_high, rel=1e-9)
    assert llada_train.gift_weight(low_H, beta_ref, t) == pytest.approx(1.0 / t_low, rel=1e-9)


def test_gift_loss_single_position_matches_weight_times_ce():
    z = np.array([2.0, 0.0, 1.0])
    h = llada_train.entropy(z)
    t = 0.4
    # single position: beta_ref defaults to beta itself -> t_i == t -> weight 1/t
    expected = (1.0 / t) * llada_train.cross_entropy(z, 0)
    assert llada_train.gift_loss(z, 0, h, t) == pytest.approx(expected, rel=1e-9)


def test_gift_loss_batch_aggregates_as_mean_of_weighted_ce():
    """Over a batch of decision positions sharing t, gift_loss is the mean 1/t_i-weighted CE
    (GIFT Alg. 2: S/N), and the uncertain position is masked more often (larger t_i)."""
    certain = np.array([50.0, 0.0, 0.0, 0.0])           # gold=0, near-certain, low entropy
    uncertain = np.array([0.1, 0.0, 0.0, 0.0])          # gold=0, almost uniform, high entropy
    ents = [llada_train.entropy(certain), llada_train.entropy(uncertain)]
    t = 0.3
    betas = llada_train.gift_betas(ents)
    ref = float(np.mean(betas))
    t_certain = llada_train.gift_masking_prob(float(betas[0]), ref, t)
    t_uncertain = llada_train.gift_masking_prob(float(betas[1]), ref, t)
    assert t_uncertain > t_certain                      # uncertain position masked/learned more
    contrib_certain = (1.0 / t_certain) * llada_train.cross_entropy(certain, 0)
    contrib_uncertain = (1.0 / t_uncertain) * llada_train.cross_entropy(uncertain, 0)
    got = llada_train.gift_loss([certain, uncertain], [0, 0], ents, t)
    assert got == pytest.approx((contrib_certain + contrib_uncertain) / 2.0, rel=1e-9)


def test_gift_loss_reduces_toward_ce_when_all_entropies_equal():
    """Equal entropies -> beta_i == beta_ref for all -> t_i == t -> uniform 1/t reweight (LLaDA)."""
    z0 = np.array([1.0, 0.5, 0.0])
    z1 = np.array([0.8, 0.2, 0.1])
    # force identical entropies by using the same logits twice
    ents = [llada_train.entropy(z0), llada_train.entropy(z0)]
    t = 0.5
    got = llada_train.gift_loss([z0, z0], [0, 0], ents, t)
    expected = (1.0 / t) * llada_train.cross_entropy(z0, 0)   # each position == diffusion_loss
    assert got == pytest.approx(expected, rel=1e-9)
