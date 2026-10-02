"""MLX LLaDA2-MoE diffusion backend, read at a [MASK] position.

These load the MLX 4-bit conversion of LLaDA2-mini (~8-9 GB), so they are marked `slow` and only run
with JUL_SLOW=1 on Apple Silicon with mlx installed. They lock in that the custom `llada2_moe`
architecture (mlx_lm/models/llada2_moe.py) loads and that the [MASK] readout is sane end to end.
"""

import importlib.util
import platform

import numpy as np
import pytest

HAS_MLX = (
    platform.system() == "Darwin"
    and platform.machine() == "arm64"
    and importlib.util.find_spec("mlx") is not None
    and importlib.util.find_spec("mlx_lm") is not None
)

REPO = "llada2-mini-4bit"


@pytest.mark.slow
@pytest.mark.skipif(not HAS_MLX, reason="needs MLX on Apple Silicon")
def test_mlx_llada_reads_a_cloze_at_the_mask():
    from jul.backbone import Backbone

    bb = Backbone(REPO, backend="mlx_llada")
    assert bb.backend == "mlx_llada"
    assert bb.mask_id == 156895  # <|mask|> in the LLaDA2 tokenizer

    # "The capital of France is <mask>." — the mask should score " Paris" at the top.
    pre = bb.encode("The capital of France is")
    post = bb.encode(".")
    tokens = pre + [bb.mask_id] + post
    vocab = bb.mask_logits(tokens, [len(pre)])  # (1, V)
    assert vocab.shape[0] == 1
    assert np.isfinite(vocab).all()
    top = int(vocab[0].argmax())
    assert "paris" in bb.tokenizer.decode([top]).strip().lower()


@pytest.mark.slow
@pytest.mark.skipif(not HAS_MLX, reason="needs MLX on Apple Silicon")
def test_mlx_llada_choice_decision_routes_a_ticket():
    from jul.backbone import Backbone
    from jul.mask import MaskReader, MaskSpec
    from jul.types import Option

    bb = Backbone(REPO, backend="mlx_llada")
    reader = MaskReader(bb, MaskSpec.default())
    state = "I was charged twice for my subscription this month."
    options = [
        Option("billing", "payments, invoices, refunds"),
        Option("technical", "bugs, errors, crashes"),
        Option("sales", "pricing, plans, demos"),
    ]
    zs, spent = reader.logits(state, [("choice", "Which team should handle this ticket?", options)])
    z = zs[0]
    assert z.shape == (3,)
    assert spent > 0
    p = np.exp(z - z.max())
    p /= p.sum()
    assert options[int(np.argmax(p))].key == "billing"


@pytest.mark.slow
@pytest.mark.skipif(not HAS_MLX, reason="needs MLX on Apple Silicon")
def test_mlx_llada_multitoken_readout_scores_a_topic():
    """The multi-token readout (readout='multitoken') scores each option's full label text over a
    block of masks. It reads the distinctive suffix of every label in one bidirectional forward and
    picks the highest sequence-likelihood option — the readout that unblocks many-class tasks."""
    import dataclasses

    from jul.backbone import Backbone
    from jul.mask import MaskReader, MaskSpec
    from jul.types import Option

    bb = Backbone(REPO, backend="mlx_llada")
    spec = dataclasses.replace(MaskSpec.default(), readout="multitoken")
    reader = MaskReader(bb, spec)
    options = [
        Option("sport", "sports and athletics"),
        Option("tech", "technology and computers"),
        Option("business", "business and finance"),
        Option("world", "world news and politics"),
    ]
    state = "The team won the championship after a dramatic overtime victory."
    zs, spent = reader.logits(state, [("choice", "Which topic best fits?", options)])
    z = zs[0]
    assert z.shape == (4,)
    assert spent > 0
    assert np.isfinite(z).all()
    assert options[int(np.argmax(z))].key == "sport"
