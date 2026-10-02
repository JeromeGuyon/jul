"""Routing-guided MoE LoRA target selection (scripts/moe_lora.py, MoE-Sieve arXiv:2603.24044).

CPU-only, no download: a tiny synthetic model that mirrors LLaDA-MoE's module hierarchy
(model.model.layers.{i}.{self_attn,mlp.gate,mlp.experts.{e}}). We lock the structure discovery, the
router profiler, the hot-expert selection, and the three target-module modes.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
ML = pytest.importorskip("moe_lora")

import torch.nn as nn


class _Cfg:
    num_experts = 8
    num_experts_per_tok = 2
    hidden_size = 16
    num_attention_heads = 4


class _Expert(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class _Attn(nn.Module):
    def __init__(self, h):
        super().__init__()
        for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, p, nn.Linear(h, h, bias=False))

    def forward(self, x):
        return self.o_proj(self.q_proj(x))


class _MoEBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList([_Expert(cfg.hidden_size, 8) for _ in range(cfg.num_experts)])
        self.top_k = cfg.num_experts_per_tok
        self.n = cfg.num_experts

    def forward(self, x):
        b, t, h = x.shape
        xf = x.view(-1, h)
        w = torch.softmax(self.gate(xf), -1)
        w, sel = torch.topk(w, self.top_k, -1)
        out = torch.zeros_like(xf)
        for e in range(self.n):
            mask = (sel == e).any(-1)
            if mask.any():
                out[mask] += self.experts[e](xf[mask])
        return out.view(b, t, h)


class _Layer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_attn = _Attn(cfg.hidden_size)
        self.mlp = _MoEBlock(cfg)

    def forward(self, x):
        return x + self.mlp(x + self.self_attn(x))


class _Inner(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(100, cfg.hidden_size)
        self.layers = nn.ModuleList([_Layer(cfg) for _ in range(3)])

    def forward(self, input_ids=None, **kw):
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return x


class _Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.model = _Inner(cfg)

    def forward(self, input_ids=None, **kw):
        return self.model(input_ids=input_ids)


def _model():
    torch.manual_seed(0)
    return _Model(_Cfg())


def test_discover_structure():
    st = ML.discover_moe_structure(_model())
    assert st.num_layers() == 3
    assert len(st.gates) == 3 and len(st.attns) == 3 and len(st.shared) == 0
    assert st.gates[0].endswith(".mlp.gate")
    # 8 experts per layer
    assert all(len(v) == 8 for v in st.experts.values())


def test_profile_and_hot_selection():
    m = _model()
    batches = [{"input_ids": torch.randint(0, 100, (2, 5))} for _ in range(4)]
    counts = ML.profile_routing(m, batches, "cpu")
    lp = next(iter(counts))
    # 4 batches * 2 seqs * 5 tokens * top_k(2) = 80 selections per layer
    assert int(counts[lp].sum()) == 80
    hot = ML.select_hot_experts(counts, hot_frac=0.25)
    assert len(hot[lp]) == 2                       # 25% of 8 = 2
    # hot experts are indeed the most-selected ones
    top2 = set(int(i) for i in counts[lp].argsort()[::-1][:2])
    assert set(hot[lp]) == top2


def test_target_modules_routing_selects_hot_experts_gate_attention():
    m = _model()
    batches = [{"input_ids": torch.randint(0, 100, (2, 5))} for _ in range(4)]
    tm, info = ML.build_target_modules(m, mode="routing", hot_frac=0.25,
                                       calib_batches=batches, device="cpu")
    assert info["experts_selected"] == 6           # 2 hot * 3 layers
    assert info["experts_total"] == 24
    assert "q_proj" in tm                           # attention
    assert any(t.endswith(".mlp.gate") for t in tm)  # router gate
    assert any(".experts." in t for t in tm)         # hot experts
    # only hot experts, not all 24
    expert_mods = {t.rsplit(".", 1)[0] for t in tm if ".experts." in t}
    assert len(expert_mods) == 6


def test_target_modules_all_and_attention_modes():
    m = _model()
    tm_all, info_all = ML.build_target_modules(m, mode="all")
    assert info_all["experts_selected"] == 24       # every expert
    tm_at, info_at = ML.build_target_modules(m, mode="attention")
    assert info_at["experts_selected"] == 0
    assert tm_at == ["q_proj", "k_proj", "v_proj", "o_proj"]


def test_routing_without_calib_falls_back_to_all():
    m = _model()
    tm, info = ML.build_target_modules(m, mode="routing", calib_batches=None)
    assert info["experts_selected"] == 24           # no calib -> all experts
