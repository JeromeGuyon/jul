"""Option 2 — LLaDA as a *training target*: fine-tune the [MASK] readout with a calibrated loss.

Option 1 (backends/llada.py + mask.py) reads a *frozen* LLaDA at [MASK], zero-shot. Option 2 keeps
the very same readout but *trains* it, so the option distribution at the mask is optimized directly.
Two stages, mirroring the project's Harrier pipeline (Stage A distill, Stage B RLCD):

  Stage A — distillation / reconstruction, with a *diffusion-correct* loss.
    For each (state, question, gold) we build the same `state\n<head>[MASK]` sequence as MaskReader,
    take the masked-position logits over the option anchor tokens, and minimize a reconstruction
    loss on the decision token. Three objectives are selectable via `loss=`:
      - 'ce'        : plain cross-entropy (or KL against a teacher when `soft` is present). Baseline.
      - 'diffusion' : the LLaDA masked-diffusion loss (Nie et al. 2025), which reweights the
                      per-token cross-entropy by 1/t where t~U(0,1] is the masking level. That 1/t
                      factor is *the* diffusion correction and is missing from a plain CE.
      - 'gift'      : GIFT (arXiv 2509.20863), which replaces the uniform 1/t by an entropy-aware
                      1/t_i, t_i = 1-(1-t)^(sqrt(H_i)/beta_ref). Uncertain decisions are masked and
                      learned more. Default.
    The "response" of the diffusion objective is the decision itself — the option anchor at the
    [MASK] — not a long generated span, so the 1/t (and 1/t_i) reweighting applies to the decision
    token(s). LoRA on q/k/v/o keeps the diffusion backbone intact (full-FT of an 8B collapses, as
    measured on Harrier: LoRA >> full-FT). The [MASK] readout means no separate pointer head to train.

  Stage B — RLCD (Reinforcement Learning for Calibrated Decisions).
    The load-bearing ingredient of Jev, reproduced: a strictly *proper scoring rule* (Brier / log
    loss) plus an *asymmetric* penalty on confident errors — being wrong at 99% costs far more than
    being wrong at 55%. Minimizing a proper scoring rule is uniquely optimized by reporting one's true
    probabilities, which is exactly calibration. Optionally wrapped in a GRPO-style policy gradient.

This module is kept small and dependency-light so the maths run on CPU with a tiny random model
(`toy_model_and_tokenizer`), exactly like scripts/harrier_model.py. A real run needs a GPU and the
LLaDA weights; see the CLI at the bottom and the design notes in docs/llada-training.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- losses (pure numpy/torch-agnostic maths, unit-testable on CPU) ---------------------------

def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def cross_entropy(option_logits: np.ndarray, gold_index: int) -> float:
    """Stage A hard-label loss: -log p(gold) over the option anchors."""
    p = _softmax(option_logits)
    return float(-np.log(max(p[gold_index], 1e-9)))


def kl_distillation(option_logits: np.ndarray, teacher_probs: np.ndarray) -> float:
    """Stage A soft-label loss: KL(teacher || student) over the option anchors."""
    p = _softmax(option_logits)
    t = np.clip(teacher_probs, 1e-9, 1.0)
    t = t / t.sum()
    return float(np.sum(t * (np.log(t) - np.log(np.clip(p, 1e-9, 1.0)))))


def brier(option_logits: np.ndarray, gold_index: int) -> float:
    """A strictly proper scoring rule: sum (p_k - 1{k==gold})^2. Minimized by the true probabilities."""
    p = _softmax(option_logits)
    y = np.zeros_like(p)
    y[gold_index] = 1.0
    return float(np.sum((p - y) ** 2))


def rlcd_loss(option_logits: np.ndarray, gold_index: int, alpha: float = 3.0) -> float:
    """Stage B: proper scoring rule (Brier) + asymmetric penalty on confident errors.

    penalty = alpha * confidence * wrong, where confidence is the top probability and `wrong` is 1
    when the argmax is not the gold. Being confidently wrong is punished; being uncertain is not.
    This is the calibration-first objective Jev's RLCD is described to optimize.
    """
    p = _softmax(option_logits)
    base = brier(option_logits, gold_index)
    pred = int(np.argmax(p))
    confidence = float(p[pred])
    wrong = 0.0 if pred == gold_index else 1.0
    return base + alpha * confidence * wrong


# --- masked-diffusion losses (LLaDA + GIFT), adapted to the typed decision token --------------
#
# A masked diffusion LM (LLaDA, Nie et al. 2025) is trained by the objective (their Eq. 1)
#
#     L = -sum_i  E_{t~U(0,1)}  [ 1[x_t^i = MASK] * (1/t) * log p_theta(x0^i | x_t) ]
#
# where t is sampled uniformly in (0,1], each token is independently masked with probability t, and
# the *per-token* reconstruction loss is reweighted by 1/t. That 1/t factor is the change of
# variable that turns the diffusion ELBO into this simple Monte-Carlo estimator: at small t few
# tokens are masked but each is worth proportionally more. The old Stage-A loss is a plain
# cross-entropy — it silently drops 1/t and is therefore *not* the LLaDA objective; it over-weights
# easy, near-fully-observed states and under-weights the hard, heavily-masked ones.
#
# Here the "response" is not a long span but the *decision*: the option anchor read at the [MASK]
# position(s). A Choice/Score/Noul readout places n_mask decision positions (usually one). We keep
# the exact diffusion maths and apply it to those decision tokens: the token at a decision position
# is masked (it *is* a [MASK]), so 1[x_t^i = MASK] = 1 there; what we still Monte-Carlo is the level
# t that sets the 1/t weight (and, for GIFT, the per-position schedule t_i and its 1/t_i weight).
#
# GIFT (arXiv 2509.20863, Thm 1 + Alg. 1/2) generalises the uniform masking rate to a per-token rate.
# A first forward with the whole response masked gives the per-position output entropy H_i; set
#   beta_i    = sqrt(H_i)                      (sqrt for stability; raw entropy explodes gradients)
#   beta_ref  = mean_i beta_i                  (over the masked decision positions)
#   t_i       = 1 - (1 - t)^(beta_i / beta_ref)
# then a second forward masks position i with prob t_i and trains it with weight 1/t_i. A higher-
# entropy position has beta_i > beta_ref, hence a *larger* t_i: it is masked (and therefore learned)
# more often. This is the paper's monotonicity — "a token with a larger masking rate beta_i
# corresponds to a larger t_i, making it more likely to be masked and subsequently learned."
# Per Alg. 2 the batch loss is  S/N  with  S = sum_i 1{masked_i} (1/t_i) CE_i  and  N = #masked.


def entropy(option_logits: np.ndarray) -> float:
    """Shannon entropy (nats) of the option distribution at a decision position."""
    p = _softmax(option_logits)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def diffusion_loss(option_logits: np.ndarray, gold_index: int, t: float) -> float:
    """LLaDA masked-diffusion loss for one decision token: (1/t) * (-log p(gold | x_t)).

    `t` is the masking level sampled ~ U(0, 1]; the 1/t factor is the load-bearing LLaDA reweighting
    (Nie et al. 2025, Eq. 1) that a plain cross-entropy omits. Clamped away from 0 so the estimator
    stays finite.
    """
    t = float(min(max(t, 1e-3), 1.0))
    ce = cross_entropy(option_logits, gold_index)
    return (1.0 / t) * ce


def gift_betas(entropies) -> np.ndarray:
    """beta_i = sqrt(H_i) for each decision position (GIFT Eq. 12). sqrt is the stability transform."""
    h = np.asarray(entropies, dtype=np.float64)
    return np.sqrt(np.clip(h, 0.0, None))


def gift_masking_prob(beta_i: float, beta_ref: float, t: float) -> float:
    """t_i = 1 - (1 - t)^(beta_i / beta_ref)  (GIFT Thm 1). Monotone increasing in beta_i.

    A higher-entropy position (beta_i > beta_ref) gets a larger t_i: it is masked, and thus learned,
    more often. Clamped into (0, 1] so 1/t_i stays finite.
    """
    t = float(min(max(t, 1e-6), 1.0))
    beta_ref = float(max(beta_ref, 1e-9))
    ratio = float(max(beta_i, 0.0)) / beta_ref
    t_i = 1.0 - (1.0 - t) ** ratio
    return float(min(max(t_i, 1e-6), 1.0))


def gift_weight(token_entropy: float, beta_ref: float, t: float) -> float:
    """GIFT importance weight 1/t_i for one decision position, given its entropy and beta_ref."""
    beta_i = float(gift_betas([token_entropy])[0])
    return 1.0 / gift_masking_prob(beta_i, beta_ref, t)


def gift_loss(option_logits, gold_indices, entropies, t: float,
              beta_ref: float | None = None) -> float:
    """GIFT importance-weighted SFT loss over one or more decision positions (arXiv 2509.20863).

    Faithful to Alg. 2: per position i, beta_i = sqrt(H_i), beta_ref defaults to mean_i beta_i,
    t_i = 1-(1-t)^(beta_i/beta_ref), and the loss is the 1/t_i-weighted mean cross-entropy

        L = ( sum_i (1/t_i) * CE_i ) / n_positions.

    Accepts either a single position (2-D `option_logits` of shape (n_opt,), scalar `gold_indices`
    and `entropies`) or a batch of decision positions (2-D list/array of per-position option logits,
    a sequence of gold indices, and a sequence of per-position entropies). For the typed decider a
    single decision position is the common case; multiple positions arise with n_mask>1 or when a
    batch of decisions shares one masking level t.
    """
    logits_list, golds, ents = _as_position_batch(option_logits, gold_indices, entropies)
    betas = gift_betas(ents)
    ref = float(np.mean(betas)) if beta_ref is None else float(beta_ref)
    total = 0.0
    for z, g, b in zip(logits_list, golds, betas):
        w = 1.0 / gift_masking_prob(float(b), ref, t)
        total += w * cross_entropy(np.asarray(z, dtype=np.float64), int(g))
    return float(total / max(1, len(logits_list)))


def _as_position_batch(option_logits, gold_indices, entropies):
    """Normalise single-position or multi-position GIFT inputs to parallel lists."""
    arr = np.asarray(option_logits, dtype=np.float64)
    if arr.ndim == 1:                       # single decision position
        return [arr], [int(gold_indices)], [float(entropies)]
    golds = [int(g) for g in gold_indices]
    ents = [float(e) for e in entropies]
    return [np.asarray(z, dtype=np.float64) for z in arr], golds, ents


# --- record + batch build for a masked-diffusion readout --------------------------------------

@dataclass
class MaskExample:
    """One training row for the [MASK] readout."""

    tokens: list[int]          # state\n<head>[MASK]
    mask_pos: int              # index of the [MASK] token in `tokens`
    anchor_ids: list[int]      # vocab id read for each option, in option order
    gold_index: int
    soft: list[float] | None   # teacher distribution over options, if distilling
    kind: str


def build_example(reader, kind: str, instructions: str, options, state: str,
                  gold_index: int, soft=None) -> MaskExample:
    """Build a MaskExample with the *same* prompt/anchors as inference (jul.mask.MaskReader).

    Training and inference must read the state identically, so we reuse the reader's own helpers.
    """
    from jul.mask import _markers, _render  # same marker + state-render logic as inference
    markers = _markers(kind, options, reader.spec.markers)
    anchors = [reader._anchor_id(m) for m in markers]
    state = state if isinstance(state, str) else _render(state)  # match MaskReader.logits exactly
    tokens, mask_positions = reader._prompt_tokens(kind, instructions, markers, options, state)
    mask_pos = mask_positions[0]  # train on the first mask position
    # For noul the reader returns anchors in [false, true]; keep option order for training targets.
    if kind == "noul":
        order = {o.key: i for i, o in enumerate(options)}
        by_opt = [0] * len(options)
        by_opt[order["false"]] = anchors[0]
        by_opt[order["true"]] = anchors[1]
        anchors = by_opt
    return MaskExample(tokens=tokens, mask_pos=mask_pos, anchor_ids=anchors,
                       gold_index=gold_index, soft=soft, kind=kind)


# --- a torch training step (imported lazily; the maths above are enough for CPU tests) --------

def option_logits_from_model(model, ex: MaskExample, device) -> "np.ndarray":
    """One forward pass, read the masked-position logits over the option anchors. Returns a tensor."""
    import torch
    ids = torch.tensor([ex.tokens], dtype=torch.long, device=device)
    out = model(input_ids=ids)
    logits = out.logits if hasattr(out, "logits") else out[0]
    row = logits[0, ex.mask_pos]
    return row[torch.tensor(ex.anchor_ids, device=device)]


def _ordinal_target(n: int, gold: int, sigma: float, device, dtype):
    """A normalized Gaussian bump over n ordered levels, centered on `gold` (std `sigma`).

    Plain CE treats a score scale as unordered classes, so a 3-level error costs the same as a
    1-level error and the model collapses toward the center. This soft target makes neighboring
    levels partly correct, teaching the order and spreading mass off the center.
    """
    import torch
    levels = torch.arange(n, device=device, dtype=torch.float32)
    w = torch.exp(-0.5 * ((levels - float(gold)) / sigma) ** 2)
    w = w / w.sum()
    return w.to(dtype)


def train_step(model, ex: MaskExample, stage: str, device, alpha: float = 3.0,
               loss: str = "ce", t: float | None = None, beta_ref: float | None = None):
    """One optimization step's loss (torch).

    stage: 'a' (Stage A: distillation / masked-diffusion reconstruction) or 'b' (Stage B: RLCD).

    For Stage A, `loss` selects the reconstruction objective on the decision token:
      - 'ce'        : plain cross-entropy / KL — the baseline (unchanged behaviour).
      - 'diffusion' : LLaDA masked-diffusion loss (1/t) * CE  (Nie et al. 2025). `t` is the masking
                      level; if None it is sampled ~ U(0,1]. Distillation KL is likewise 1/t-scaled.
      - 'gift'      : GIFT entropy-importance loss (1/t_i) * CE (arXiv 2509.20863). The decision
                      token's masked-output entropy H sets beta = sqrt(H); t_i = 1-(1-t)^(beta/beta_ref)
                      with `beta_ref` (default: beta itself, i.e. the single-position reference).

    Stage B (RLCD) is calibration-first and independent of `loss`.
    """
    import torch
    z = option_logits_from_model(model, ex, device)          # (n_opt,) with grad
    logp = torch.log_softmax(z, dim=-1)
    if stage == "a":
        # diffusion / GIFT importance weight on the decision token (LLaDA 1/t, GIFT 1/t_i)
        weight = 1.0
        if loss in ("diffusion", "gift"):
            import numpy as _np
            tt = float(_np.random.uniform(1e-3, 1.0)) if t is None else float(min(max(t, 1e-3), 1.0))
            if loss == "diffusion":
                weight = 1.0 / tt
            else:  # gift: entropy of the masked decision distribution sets beta_i
                p_det = logp.detach().exp().float().cpu().numpy()
                p_det = _np.clip(p_det, 1e-12, 1.0)
                h = float(-_np.sum(p_det * _np.log(p_det)))
                beta_i = float(_np.sqrt(max(h, 0.0)))
                ref = beta_i if beta_ref is None else float(beta_ref)
                t_i = gift_masking_prob(beta_i, ref, tt)
                weight = 1.0 / t_i
        if loss == "ordinal" and ex.kind == "score" and len(z) >= 3:
            # ordinal soft target: a Gaussian bump centered on the gold level, so being one level off
            # costs far less than three. Breaks the central-collapse of plain CE on scales. Takes
            # priority over distillation soft labels for score items (the point of this run).
            tgt = _ordinal_target(len(z), ex.gold_index, sigma=1.0, device=device, dtype=logp.dtype)
            ce = torch.sum(tgt * (torch.log(tgt.clamp_min(1e-9)) - logp))  # KL(ordinal||student)
        elif ex.soft is not None:
            tgt = torch.tensor(ex.soft, device=device, dtype=logp.dtype)
            tgt = tgt / tgt.sum()
            ce = torch.sum(tgt * (torch.log(tgt.clamp_min(1e-9)) - logp))  # KL(teacher||student)
        else:
            ce = -logp[ex.gold_index]                                      # cross-entropy
        return weight * ce
    # stage b: Brier (proper scoring rule) + asymmetric confident-error penalty
    p = logp.exp()
    y = torch.zeros_like(p); y[ex.gold_index] = 1.0
    base = torch.sum((p - y) ** 2)
    pred = int(torch.argmax(p))
    penalty = alpha * p[pred] * (0.0 if pred == ex.gold_index else 1.0)
    return base + penalty


def toy_model_and_tokenizer():
    """A tiny random masked-LM-ish model + tokenizer for CPU smoke tests (no download).

    Uses tiny-random-roberta: an encoder with a masked-LM head and bidirectional attention — the
    closest tiny stand-in for LLaDA's [MASK] readout that ships in transformers test fixtures.
    """
    from transformers import AutoTokenizer, AutoConfig, AutoModelForMaskedLM
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-roberta")
    cfg = AutoConfig.from_pretrained("hf-internal-testing/tiny-random-roberta")
    model = AutoModelForMaskedLM.from_config(cfg)
    return model, tok
