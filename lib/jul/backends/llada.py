"""LLaDA backend: a masked *diffusion* language model, read at a [MASK] position in one forward pass.

Why a separate backend. The torch backend (backends/torch.py) is a *causal* decoder: it reads the
last token's hidden state, its attention is left-to-right, and it stops the forward after the deepest
layer needed. LLaDA (arXiv:2502.09992, "Large Language Diffusion Models") is fundamentally different:

  - it is trained by a *masking* objective (predict the tokens hidden behind [MASK]), not next-token,
  - its attention is *bidirectional* — a [MASK] sees the whole state, left and right,
  - so the answer is read at the position of a [MASK] token, not at the last token.

This is exactly the readout Jev's non-autoregressive "parallel sampler" is believed to use, and the
one Laya implements on an encoder — but LLaDA has that readout *natively* (it is its pre-training
objective), while being a generative model with an LLM's world knowledge. We do a single forward
pass (no iterative unmasking): place a [MASK] where the decision goes, read the vocabulary
distribution there, restrict it to the option tokens. That is the whole "System One" idea, on a
diffusion backbone.

Model ids and the mask token:
  - GSAI-ML/LLaDA-8B-Instruct : mask id 126336 (a dedicated <|mdm_mask|> token).
  - iLLaDA-8B-Instruct        : mask id 5 (reuses LLaDA's inference code, per its model card).
The id is read from the tokenizer when possible (added special token), else falls back to MASK_IDS.

This backend implements `mask_logits` (used by decision.MaskReader) and the minimum of the Backbone
interface. It deliberately does NOT implement the causal vector method (`forward`/`last_hidden` for
the pointer/vector readers): LLaDA is meant to be read at [MASK], and mixing readings would muddy the
one experiment this backend exists for. Those raise NotImplementedError with a clear message.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from ..backbone import Backbone
from .torch import _device_dtype

#: Fallback mask-token id per known model family, when the tokenizer has no single mask special token.
MASK_IDS = {"llada-moe": 156895, "llada": 126336, "illada": 5}
#: Env override, e.g. JUL_LLADA_MASK_ID=5 for iLLaDA.
MASK_ID_ENV = "JUL_LLADA_MASK_ID"


def _guess_mask_id(repo: str, tokenizer, config=None) -> int:
    """The [MASK] token id: env override, then config.mask_token_id, then the tokenizer, then family."""
    override = os.environ.get(MASK_ID_ENV)
    if override is not None:
        return int(override)
    # The model config is authoritative: LLaDA declares mask_token_id = 126336.
    if config is not None:
        mid = getattr(config, "mask_token_id", None)
        if isinstance(mid, int) and mid >= 0:
            return mid
    # A tokenizer that declares a mask token (added special) is next.
    mask_tok = getattr(tokenizer, "mask_token", None)
    if mask_tok is not None:
        mid = tokenizer.convert_tokens_to_ids(mask_tok)
        if isinstance(mid, int) and mid >= 0:
            return mid
    for tok in ("<|mdm_mask|>", "<|mask|>", "[MASK]"):
        mid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(mid, int) and mid >= 0 and mid != tokenizer.unk_token_id:
            return mid
    low = repo.lower()
    for family, mid in MASK_IDS.items():
        if family in low:
            return mid
    raise ValueError(
        f"Cannot determine the [MASK] token id for {repo!r}. Set {MASK_ID_ENV}=<id> "
        f"(LLaDA: 126336, iLLaDA: 5)."
    )


@contextlib.contextmanager
def _patch_remote_code_for_transformers5(repo: str):
    """LLaDA's published remote code predates transformers 5.x. Fill in attributes 5.x expects.

    LLaDA-8B-Instruct ships `modeling_llada.LLaDAModelLM`, written for transformers ~4.4x. Loading it
    under transformers 5.x fails in `_finalize_model_loading` because the class lacks
    `all_tied_weights_keys` (a 5.x attribute). We resolve the dynamic class and add the missing
    attribute *before* `from_pretrained` finalizes. This does not downgrade transformers (the rest of
    JuL runs on the installed version) and touches only LLaDA's own class. No-op when the attribute is
    already present (older transformers, or newer remote code).

    Also makes transformers 5.x's ROPE_INIT_FUNCTIONS tolerant of a missing/renamed rope_type (the
    LLaDA-MoE remote code indexes it with a key 5.x may not ship as-is).

    A context manager: wrap the `from_pretrained` call. The `"default"` RoPE entry is a process-global
    registry change, so it is removed again on exit (the remote code reads it while building the
    model); other models loaded later in the same process see the registry unchanged.
    """
    added_default = False
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        # transformers 5.x dropped the "default" entry from ROPE_INIT_FUNCTIONS (the plain-RoPE
        # computation moved into the RotaryEmbedding class). LLaDA's *remote* code still does
        # `ROPE_INIT_FUNCTIONS[self.rope_type]` with rope_type="default", so it raises KeyError:
        # 'default'. Some 5.x paths then re-enter via a config-driven fn that indexes
        # rope_parameters["factor"] -> KeyError: 'factor'. We register a self-contained "default"
        # init that computes plain RoPE inv_freq from the config alone (no 'factor', no scaling),
        # matching the signature ROPE_INIT_FUNCTIONS[*] is called with: (config, device, seq_len=...).
        # This is exactly what a non-scaled RoPE model needs; it never touches 'factor'. It is only
        # registered when missing, so it is a no-op on transformers 4.x (where "default" exists) and
        # does not perturb the dense LLaDA-8B path that already loads.
        if "default" not in ROPE_INIT_FUNCTIONS:
            def _plain_default(config, device=None, seq_len=None, **kw):
                import torch as _t
                # base wavelength: config.rope_theta, else the standardized rope_parameters dict.
                base = getattr(config, "rope_theta", None)
                if base is None or base <= 0:
                    rp = getattr(config, "rope_parameters", None) or {}
                    base = rp.get("rope_theta", 10000.0)
                head_dim = getattr(config, "head_dim", None)
                if not head_dim:
                    head_dim = config.hidden_size // config.num_attention_heads
                # honour a partial rotary factor if the model declares one (LLaDA-MoE uses 1.0).
                prf = getattr(config, "partial_rotary_factor", None)
                if prf is None:
                    rp = getattr(config, "rope_parameters", None) or {}
                    prf = rp.get("partial_rotary_factor", 1.0)
                dim = int(head_dim * (prf or 1.0))
                inv = 1.0 / (base ** (_t.arange(0, dim, 2, dtype=_t.int64).float() / dim))
                # (inv_freq, attention_scaling); scaling is 1.0 for plain RoPE.
                return inv.to(device), 1.0
            ROPE_INIT_FUNCTIONS["default"] = _plain_default
            added_default = True
    except ImportError:
        pass  # transformers without modeling_rope_utils: nothing to register
    # Safe to unregister after loading: the remote RotaryEmbedding classes resolve
    # ROPE_INIT_FUNCTIONS[rope_type] once in __init__ and keep the function on the instance.
    try:
        _patch_llada_classes(repo)
        yield
    finally:
        if added_default:
            ROPE_INIT_FUNCTIONS.pop("default", None)


def _patch_llada_classes(repo: str) -> None:
    """Give LLaDA's own remote classes the attributes transformers 5.x expects (scoped to them)."""
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
    except ImportError:
        return
    for ref in ("modeling_llada.LLaDAModelLM", "modeling_lladamoe.LLaDAMoEModelLM"):
        try:
            cls = get_class_from_dynamic_module(ref, repo)
        except Exception:  # noqa: BLE001 - best effort: this repo does not ship that class
            continue
        if not hasattr(cls, "all_tied_weights_keys"):
            # LLaDA ties nothing beyond its own `_tied_weights_keys`; an empty mapping satisfies 5.x.
            cls.all_tied_weights_keys = {}
        # 5.x calls tie_weights(missing_keys=..., recompute_mapping=...); LLaDA's takes no kwargs.
        # Drop the 5.x kwargs, but let any real error from LLaDA's tie_weights propagate.
        _orig_tie = cls.tie_weights
        if getattr(_orig_tie, "_jul_wrapped", False) is False:
            def _tie_weights(self, *a, _orig=_orig_tie, **k):  # noqa: ANN001
                return _orig(self)
            _tie_weights._jul_wrapped = True
            cls.tie_weights = _tie_weights


class LLaDABackbone(Backbone):
    """A masked diffusion LM (LLaDA / iLLaDA), read at a [MASK] position in a single forward pass."""

    backend = "llada"
    #: bidirectional masked-diffusion; not the causal 'decoder' nor the pooled 'encoder'.
    architecture = "diffusion"

    def __init__(self, name: str, backend: str | None = None, device: str | None = None,
                 dtype: torch.dtype | None = None):
        super().__init__(name)
        self.device, dtype = _device_dtype(device, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.repo, trust_remote_code=True)
        # LLaDA is an AutoModel (masked LM head lives on the base model), loaded with remote code.
        # JUL_LLADA_DEVICE_MAP=auto shards a large model (e.g. LLaDA2.0-mini 32.5 GB) across all GPUs
        # via accelerate; in that mode we must NOT call .to(device) (accelerate owns placement).
        device_map = os.environ.get("JUL_LLADA_DEVICE_MAP") or None
        def _load(**kw):
            if device_map:
                m = AutoModel.from_pretrained(self.repo, trust_remote_code=True,
                                              device_map=device_map, **kw).eval()
                # reads should feed inputs to the first shard's device
                self.device = next(m.parameters()).device
                return m
            return AutoModel.from_pretrained(self.repo, trust_remote_code=True,
                                             **kw).to(self.device).eval()
        with _patch_remote_code_for_transformers5(self.repo):
            try:
                self.model = _load(torch_dtype=dtype)
            except TypeError:
                self.model = _load(dtype=dtype)
        # Optional: load a LoRA adapter trained at the [MASK] readout (Option 2), for the gate.
        adapter = os.environ.get("JUL_LLADA_ADAPTER")
        if adapter and os.path.isdir(adapter):
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter).to(self.device).eval()
            print(f"[llada] loaded LoRA adapter from {adapter}", flush=True)
        self.mask_id = _guess_mask_id(self.repo, self.tokenizer, self.model.config)
        cfg = self.model.config
        # LLaDA's remote forward reads config.use_cache; transformers 5.x no longer defaults it, and a
        # bidirectional diffusion model has no KV cache anyway. Set it defensively (the config's
        # __getattribute__ raises rather than returning None for a missing key).
        try:
            _ = cfg.use_cache
        except AttributeError:
            cfg.use_cache = False
        self.n_layers = int(getattr(cfg, "num_hidden_layers", None)
                            or getattr(cfg, "n_layers", None) or 0)

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    @torch.inference_mode()
    def mask_logits(self, tokens: list[int], mask_positions: list[int]) -> np.ndarray:
        """One forward pass over `tokens`; return the vocabulary logits at each mask position.

        `tokens` already contains `self.mask_id` at each index of `mask_positions`. Returns a
        (len(mask_positions), vocab) float32 array. No iterative unmasking: a single pass, which is all
        a typed decision needs (the option distribution is read straight from the masked position).
        """
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        out = self.model(input_ids=ids)
        logits = out.logits if hasattr(out, "logits") else out[0]  # (1, T, vocab)
        rows = torch.tensor(mask_positions, device=logits.device)
        picked = logits[0, rows].float().cpu().numpy()
        if not np.isfinite(picked).all():
            raise FloatingPointError(
                f"non-finite logits at a [MASK] position in {self.model.dtype}: set JUL_DTYPE=float32"
            )
        return picked

    @torch.inference_mode()
    def mask_hidden(self, tokens: list[int], mask_positions: list[int]) -> np.ndarray:
        """One forward pass; return the last-layer hidden state at each mask position.

        A masked-diffusion LM is bidirectional, so the hidden state at a [MASK] attends to the whole
        state and question — a good representation to read with a learned head (DiffEmbed, 2505.15045),
        instead of (or alongside) the vocabulary logits. Returns a (len(mask_positions), hidden) float32
        array.
        """
        ids = torch.tensor([tokens], dtype=torch.long, device=self.device)
        out = self.model(input_ids=ids, output_hidden_states=True)
        hs = out.hidden_states[-1] if getattr(out, "hidden_states", None) is not None else None
        if hs is None:
            raise RuntimeError("backbone did not return hidden_states; cannot use a learned head")
        rows = torch.tensor(mask_positions, device=hs.device)
        picked = hs[0, rows].float().cpu().numpy()
        if not np.isfinite(picked).all():
            raise FloatingPointError(
                f"non-finite hidden state at a [MASK] in {self.model.dtype}: set JUL_DTYPE=float32"
            )
        return picked

    # --- the causal interface is intentionally not provided on this backbone ------------------

    def forward(self, *a, **k):
        raise NotImplementedError(
            "LLaDABackbone is read at a [MASK] position (method='mask'), not by the causal vector "
            "method. Use a decision.json with method='mask'."
        )

    def forward_batch(self, *a, **k):
        raise NotImplementedError("LLaDABackbone: use mask_logits (method='mask').")

    def cache_prefix(self, tokens):
        # No causal KV prefix cache: attention is bidirectional. The reader re-runs the full sequence.
        return None

    def last_hidden(self, *a, **k):
        raise NotImplementedError("LLaDABackbone: use mask_logits (method='mask').")
