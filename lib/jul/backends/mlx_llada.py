"""MLX backend for LLaDA-family masked *diffusion* models, read at a [MASK] position.

This is the Apple-Silicon counterpart of `backends/llada.py` (torch/transformers): same readout,
same `mask_logits` contract used by `decision.MaskReader`, but the forward runs through `mlx_lm` on
the Metal GPU instead of torch on MPS/CUDA.

It exists because JuL's causal MLX backend (`backends/mlx.py`) taps hidden states for the *vector*
method and assumes a causal decoder; a diffusion LM is read at a [MASK] token with bidirectional
attention, so it needs its own small backend, exactly as the torch side does.

It targets the published MLX conversions of LLaDA2-MoE, e.g.
`mlx-community/LLaDA2.0-mini-preview-4bit`, whose custom architecture (`llada2_moe`) is implemented
in `mlx_lm/models/llada2_moe.py`. The tokenizer is loaded with `trust_remote_code=True` so the load
is non-interactive.

Like `backends/llada.py`, it implements only the [MASK] readout; the causal vector interface raises
NotImplementedError with a clear message.
"""

from __future__ import annotations

import os

import numpy as np

from ..backbone import Backbone
from .llada import MASK_IDS, MASK_ID_ENV


def _register_llada2_moe_arch() -> None:
    """Make `mlx_lm.models.llada2_moe` importable, from our vendored copy if mlx_lm lacks it.

    The MLX LLaDA2-MoE checkpoints declare `model_type: "llada2_moe"`; mlx_lm resolves the
    architecture by importing `mlx_lm.models.<model_type>`. Older mlx_lm releases do not ship that
    module, so we register our vendored implementation under that import path before `load` runs.
    A no-op when mlx_lm already provides it (or once registered)."""
    import importlib
    import importlib.util
    import sys

    name = "mlx_lm.models.llada2_moe"
    if name in sys.modules or importlib.util.find_spec(name) is not None:
        return
    from pathlib import Path

    vendored = Path(__file__).resolve().parent.parent / "vendor" / "llada2_moe.py"
    if not vendored.exists():
        return  # mlx_lm must provide it, else load() will raise a clear "not supported" error
    spec = importlib.util.spec_from_file_location(name, vendored)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


class MLXLLaDABackbone(Backbone):
    """A masked-diffusion LM (LLaDA2-MoE), run through mlx_lm, read at a [MASK] in one forward pass."""

    backend = "mlx_llada"
    #: bidirectional masked-diffusion; not a causal 'decoder' nor a pooled 'encoder'.
    architecture = "diffusion"

    def __init__(self, name: str, backend: str | None = None):
        super().__init__(name)
        import mlx.core as mx  # noqa: F401  (ensures a clear error if mlx is missing)
        from mlx_lm import load

        _register_llada2_moe_arch()
        # trust_remote_code so the tokenizer's custom code loads without an interactive prompt.
        self.model, self.tokenizer = load(
            self.repo, tokenizer_config={"trust_remote_code": True}
        )
        self.mask_id = self._guess_mask_id()
        cfg = getattr(self.model, "args", None)
        self.n_layers = int(getattr(cfg, "num_hidden_layers", 0) or 0)

    def _guess_mask_id(self) -> int:
        """The [MASK] token id: env override, then the tokenizer's mask token, then family default."""
        override = os.environ.get(MASK_ID_ENV)
        if override is not None:
            return int(override)
        mask_tok = getattr(self.tokenizer, "mask_token", None) or "<|mask|>"
        mid = self.tokenizer.convert_tokens_to_ids(mask_tok)
        if isinstance(mid, int) and mid >= 0 and mid != self.tokenizer.unk_token_id:
            return mid
        low = self.repo.lower()
        for family, fid in MASK_IDS.items():
            if family in low:
                return fid
        raise ValueError(
            f"Cannot determine the [MASK] token id for {self.repo!r}. Set {MASK_ID_ENV}=<id>."
        )

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)

    def mask_logits(self, tokens: list[int], mask_positions: list[int]) -> np.ndarray:
        """One bidirectional forward over `tokens`; return the vocab logits at each mask position.

        `tokens` already holds `self.mask_id` at each index in `mask_positions`. Returns a
        (len(mask_positions), vocab) float32 array. No iterative unmasking here: a single pass, which
        is all a typed decision reads (MaskReader may still average several masks or refine over steps).
        """
        import mlx.core as mx

        out = self.model(mx.array([tokens]))  # (1, T, vocab)
        rows = mx.array(mask_positions)
        picked = out[0][rows].astype(mx.float32)
        mx.eval(picked)
        arr = np.array(picked)
        if not np.isfinite(arr).all():
            raise FloatingPointError("non-finite logits at a [MASK] position in the MLX LLaDA backbone")
        return arr

    def option_logits(self, tokens: list[int], mask_positions: list[int], anchors: list[int]) -> np.ndarray:
        """Optimized readout for System One: return only the logits of the `anchors` option tokens
        at each mask position, as a (len(mask_positions), len(anchors)) float32 array.

        This runs the transformer stack (bidirectional, all layers) and the final norm as usual, but
        replaces the full vocabulary projection (hidden -> 157k) at the mask positions with a tiny
        projection onto just the `len(anchors)` rows of `lm_head` we actually read. On a typed
        decision (2-5 options) this is the dominant saving: the 157k-way matmul at each mask position
        never happens. Numerically identical to slicing `mask_logits(...)[:, anchors]`.
        """
        import mlx.core as mx

        # Hidden states after the final norm, at the mask positions only.
        h = self.model.model(mx.array([tokens]))  # (1, T, hidden), post-norm
        rows = mx.array(mask_positions)
        hm = h[0][rows]  # (P, hidden)
        z = self._project_anchors(hm, anchors)
        mx.eval(z)
        arr = np.array(z)
        if not np.isfinite(arr).all():
            raise FloatingPointError("non-finite option logits at a [MASK] position (MLX LLaDA backbone)")
        return arr

    def _project_anchors(self, hm, anchors):
        """Project post-norm hidden states `hm` (P, hidden) onto only the `anchors` rows of lm_head.
        Returns (P, len(anchors)) float32. Dequantizes just the anchor rows for a quantized head."""
        import mlx.core as mx

        lm = self.model.lm_head
        anchor_idx = mx.array(anchors)
        if hasattr(lm, "scales"):  # QuantizedLinear: weight is packed, dequantize the anchor rows
            w = mx.dequantize(
                lm.weight[anchor_idx], lm.scales[anchor_idx], lm.biases[anchor_idx],
                group_size=lm.group_size, bits=lm.bits,
            )  # (A, hidden)
        else:  # plain nn.Linear: weight is (vocab, hidden)
            w = lm.weight[anchor_idx]
        return (hm @ w.T).astype(mx.float32)

    def option_logits_early(
        self,
        tokens: list[int],
        mask_positions: list[int],
        anchors: list[int],
        start_layer: int = 8,
        every: int = 2,
        patience: int = 2,
    ) -> tuple[np.ndarray, int]:
        """Early-skipping readout (ES-dLLM style): run the layers one by one and stop as soon as the
        decision at the mask is stable.

        At each checkpoint (from `start_layer`, every `every` layers) we apply the *final* RMSNorm to
        the current hidden state at the mask positions (a logit-lens view) and project onto the option
        anchors. When the per-position argmax over the anchors is unchanged for `patience` consecutive
        checkpoints, the remaining layers are skipped and the last projected logits are returned. If it
        never stabilizes, the full stack runs and the result equals `option_logits` (up to the same
        logit-lens norm — here the final norm is the real one, so the last checkpoint == exact output).

        Returns (option_logits (P, A) float32, layers_run). `layers_run` lets the bench report the
        average depth actually computed.
        """
        import mlx.core as mx

        inner = self.model.model
        rows = mx.array(mask_positions)
        h = inner.word_embeddings(mx.array([tokens]))  # (1, T, hidden)
        n_layers = len(inner.layers)

        z = None
        prev_argmax = None
        stable = 0
        for i, layer in enumerate(inner.layers):
            h = layer(h, None, None)  # bidirectional, no mask, no cache
            is_checkpoint = (i + 1) >= start_layer and ((i + 1 - start_layer) % every == 0)
            last = i == n_layers - 1
            if not (is_checkpoint or last):
                continue
            hm = inner.norm(h[0][rows])  # logit-lens: final norm applied to the current hidden
            z = self._project_anchors(hm, anchors)
            argmax = mx.argmax(z, axis=-1)
            mx.eval(argmax)
            am = np.array(argmax)
            if prev_argmax is not None and np.array_equal(am, prev_argmax):
                stable += 1
            else:
                stable = 0
            prev_argmax = am
            if stable >= patience or last:
                mx.eval(z)
                arr = np.array(z.astype(mx.float32))
                if not np.isfinite(arr).all():
                    raise FloatingPointError("non-finite early option logits (MLX LLaDA backbone)")
                return arr, i + 1
        # unreachable (the last layer always returns), but keep a safe fallback
        return np.array(z.astype(mx.float32)), n_layers

    # --- the causal vector interface is intentionally not provided ---------------------------

    def forward(self, *a, **k):
        raise NotImplementedError(
            "MLXLLaDABackbone is read at a [MASK] position (method='mask'), not by the causal vector "
            "method. Use a decision.json with method='mask'."
        )

    def forward_batch(self, *a, **k):
        raise NotImplementedError("MLXLLaDABackbone: use mask_logits (method='mask').")

    def cache_prefix(self, tokens):
        # No causal KV prefix cache: attention is bidirectional. The reader re-runs the full sequence.
        return None

    def last_hidden(self, *a, **k):
        raise NotImplementedError("MLXLLaDABackbone: use mask_logits (method='mask').")
