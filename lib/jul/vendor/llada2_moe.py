# LLaDA2-MoE: a masked *diffusion* MoE language model (InclusionAI / Ant Group).
#
# Architecture is the Bailing/Ling MoE family (fused query_key_value, per-head
# query/key RMSNorm, partial RoPE, grouped-sigmoid expert routing with an expert
# bias and a routed-scaling factor, `first_k_dense_replace` dense layers, one
# shared expert) with two differences that matter:
#
#   1. It is a diffusion LM: attention is *bidirectional* (no causal mask). The
#      answer is read at a [MASK] position, not at the last token.
#   2. The published MLX checkpoint always carries query_layernorm/key_layernorm
#      (not gated behind a flag), names the attention submodule `attention`, the
#      output projection `dense`, the router `mlp.gate.weight` + `mlp.gate.expert_bias`,
#      and the token embedding `model.word_embeddings`. Experts are already stacked
#      into `mlp.switch_mlp.*` by the conversion.
#
# This module is written to load `mlx-community/LLaDA2.0-*-preview-4bit` unchanged.

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, scaled_dot_product_attention
from .rope_utils import initialize_rope
from .switch_layers import SwitchGLU


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    intermediate_size: int
    max_position_embeddings: int
    moe_intermediate_size: int
    num_experts: int
    num_shared_experts: int
    norm_topk_prob: bool
    num_attention_heads: int
    num_experts_per_tok: int
    num_hidden_layers: int
    num_key_value_heads: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    first_k_dense_replace: int
    head_dim: Optional[int] = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    use_bias: bool = False
    use_qkv_bias: bool = False
    tie_word_embeddings: bool = False
    partial_rotary_factor: float = 1.0
    rotary_dim: Optional[int] = None
    moe_router_enable_expert_bias: bool = True
    moe_router_enable_routed_scaling: bool = True
    routed_scaling_factor: float = 1.0
    score_function: str = "sigmoid"
    n_group: int = 1
    topk_group: int = 4


@partial(mx.compile, shapeless=True)
def aggregate_expert_outputs(expert_outputs, scores):
    return (expert_outputs * scores[..., None]).sum(axis=-2).astype(expert_outputs.dtype)


class LLaDA2MoeMLP(nn.Module):
    def __init__(self, args: ModelArgs, intermediate_size: Optional[int] = None):
        super().__init__()
        inter = intermediate_size if intermediate_size is not None else args.intermediate_size
        self.gate_proj = nn.Linear(args.hidden_size, inter, bias=args.use_bias)
        self.down_proj = nn.Linear(inter, args.hidden_size, bias=args.use_bias)
        self.up_proj = nn.Linear(args.hidden_size, inter, bias=args.use_bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class LLaDA2MoeAttention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_attention_heads = args.num_attention_heads
        self.num_key_value_heads = args.num_key_value_heads
        self.head_dim = args.head_dim or (args.hidden_size // self.num_attention_heads)
        self.scale = self.head_dim**-0.5

        self.query_key_value = nn.Linear(
            args.hidden_size,
            (self.num_attention_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=args.use_qkv_bias,
        )
        self.dense = nn.Linear(
            self.num_attention_heads * self.head_dim, args.hidden_size, bias=args.use_bias
        )
        # LLaDA2 always applies a per-head RMSNorm to queries and keys.
        self.query_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.key_layernorm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)

        if (rope_dim := args.rotary_dim) is None:
            rope_dim = int(self.head_dim * args.partial_rotary_factor)
        self.rope = initialize_rope(
            rope_dim,
            args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        B, L, _ = x.shape
        qkv = self.query_key_value(x)
        q_size = self.num_attention_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        q, k, v = mx.split(qkv, [q_size, q_size + kv_size], axis=-1)

        queries = q.reshape(B, L, self.num_attention_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = k.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = v.reshape(B, L, self.num_key_value_heads, self.head_dim).transpose(0, 2, 1, 3)

        queries = self.query_layernorm(queries)
        keys = self.key_layernorm(keys)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)


@mx.compile
def group_expert_select(
    gates,
    e_score_correction_bias,
    top_k,
    n_group,
    topk_group,
    routed_scaling_factor,
    norm_topk_prob,
    score_function,
):
    in_type = gates.dtype
    if score_function == "sigmoid":
        scores = mx.sigmoid(gates.astype(mx.float32))
    else:
        scores = mx.softmax(gates.astype(mx.float32), axis=-1)
    orig_scores = scores
    if e_score_correction_bias is not None:
        scores = scores + e_score_correction_bias
    if n_group > 1:
        scores = mx.unflatten(scores, axis=-1, shape=(n_group, -1))
        group_scores = mx.topk(scores, 2, axis=-1).sum(axis=-1, keepdims=True)
        k = n_group - topk_group
        group_idx = mx.argpartition(group_scores, kth=k - 1, axis=-2)[..., :k, :]
        scores = mx.put_along_axis(
            scores, mx.stop_gradient(group_idx), mx.array(0.0, scores.dtype), axis=-2
        )
        scores = mx.flatten(scores, -2, -1)

    k = top_k
    inds = mx.argpartition(scores, kth=-k, axis=-1)[..., -k:]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    if top_k > 1 and norm_topk_prob:
        denominator = scores.sum(axis=-1, keepdims=True) + 1e-20
        scores = scores / denominator
    scores = scores * routed_scaling_factor
    return inds, scores.astype(in_type)


class LLaDA2MoeGate(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.norm_topk_prob = args.norm_topk_prob
        self.top_k = args.num_experts_per_tok
        self.n_group = args.n_group
        self.topk_group = args.topk_group
        self.routed_scaling_factor = args.routed_scaling_factor
        self.score_function = args.score_function
        # The checkpoint stores the router projection as `mlp.gate.weight` (no submodule),
        # so we expose the parameter directly and do a plain linear.
        self.weight = mx.zeros((args.num_experts, args.hidden_size))
        self.expert_bias = (
            mx.zeros((args.num_experts,)) if args.moe_router_enable_expert_bias else None
        )

    def __call__(self, x):
        gates = x @ self.weight.T
        return group_expert_select(
            gates,
            self.expert_bias,
            self.top_k,
            self.n_group,
            self.topk_group,
            self.routed_scaling_factor,
            self.norm_topk_prob,
            self.score_function,
        )


class LLaDA2MoeSparseMoeBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.num_experts_per_tok = args.num_experts_per_tok
        self.switch_mlp = SwitchGLU(
            args.hidden_size, args.moe_intermediate_size, args.num_experts, bias=args.use_bias
        )
        self.gate = LLaDA2MoeGate(args)
        self.shared_experts = (
            LLaDA2MoeMLP(
                args=args,
                intermediate_size=args.moe_intermediate_size * args.num_shared_experts,
            )
            if args.num_shared_experts > 0
            else None
        )

    def __call__(self, x):
        topk_idx, topk_weight = self.gate(x)
        out = self.switch_mlp(x, topk_idx)
        out = aggregate_expert_outputs(out, topk_weight)
        if self.shared_experts is not None:
            out = out + self.shared_experts(x)
        return out


class LLaDA2MoeDecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.attention = LLaDA2MoeAttention(args)
        self.mlp = (
            LLaDA2MoeSparseMoeBlock(args)
            if (args.num_experts is not None and layer_idx >= args.first_k_dense_replace)
            else LLaDA2MoeMLP(args)
        )
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        r = self.attention(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LLaDA2MoeModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            LLaDA2MoeDecoderLayer(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None):
        h = self.word_embeddings(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        # Diffusion model: bidirectional attention, no causal mask (unless one is
        # passed explicitly, e.g. to ignore right padding).
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)
        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = LLaDA2MoeModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, mask: Optional[mx.array] = None, cache=None):
        out = self.model(inputs, mask, cache)
        if self.args.tie_word_embeddings:
            out = self.model.word_embeddings.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        # Experts are already stacked into `switch_mlp` in the published checkpoint;
        # only stack if a future checkpoint ships per-expert tensors.
        for l in range(self.args.num_hidden_layers):
            prefix = f"model.layers.{l}"
            if l >= self.args.first_k_dense_replace:
                for m in ["gate_proj", "down_proj", "up_proj"]:
                    for k in ["weight", "scales", "biases"]:
                        if f"{prefix}.mlp.experts.0.{m}.{k}" in weights:
                            to_join = [
                                weights.pop(f"{prefix}.mlp.experts.{e}.{m}.{k}")
                                for e in range(self.args.num_experts)
                            ]
                            weights[f"{prefix}.mlp.switch_mlp.{m}.{k}"] = mx.stack(to_join)
        return weights

    @property
    def cast_predicate(self):
        # expert_bias is an additive float bias kept in fp32; never cast it.
        def predicate(k):
            return "expert_bias" not in k

        return predicate

    @property
    def layers(self):
        return self.model.layers
