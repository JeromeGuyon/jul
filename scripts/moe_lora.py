"""Routing-guided LoRA target selection for LLaDA-MoE (MoE-Sieve, arXiv:2603.24044).

Attention-only LoRA (q/k/v/o) leaves a sparse MoE's experts untouched, so it cannot move the part of
the network that actually does the work — the routed feed-forward experts. But adapting *every*
expert is wasteful and adds noise: with 64 experts per layer and top-8 routing, most experts barely
fire on a given corpus, and adapting a cold expert perturbs a path the data almost never takes.

MoE-Sieve's recipe, reproduced here:
  1. profile the router on a small calibration set — one *free* forward, counting how often each
     (layer, expert) is selected in the top-k,
  2. keep the hottest `hot_frac` (default 25%) of experts *per layer*,
  3. put LoRA on those hot experts + the router/gate + any shared experts + attention. Cold experts
     are left frozen (adapting them only injects noise on rarely-taken paths).

Architecture of inclusionAI/LLaDA-MoE-7B-A1B-Instruct (read from its remote code + config):
    model.model.layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}      # attention
    model.model.layers.{i}.mlp.gate                                     # router (Linear -> 64)
    model.model.layers.{i}.mlp.experts.{e}.{gate_proj,up_proj,down_proj}# 64 routed experts
    model.model.layers.{i}.shared_expert.{gate_proj,up_proj,down_proj}  # only if configured
Config: num_experts=64, num_experts_per_tok=8, 16 layers all MoE, shared_expert_intermediate_size=None
(so this checkpoint has no shared experts — the builder simply finds none and skips them).

PEFT `target_modules` accepts either a bare suffix (matches every module ending in it) or a
*fully-qualified* module name. We use fully-qualified names for the hot experts so exactly those
experts are adapted, and bare suffixes only where we do want all of them (attention on every layer).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


ATTENTION_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")
EXPERT_LINEARS = ("gate_proj", "up_proj", "down_proj")


# --- discovering the MoE structure from a live module tree ------------------------------------

@dataclass
class MoEStructure:
    """The MoE-relevant module names discovered on a loaded model (names relative to the model root)."""

    #: gate/router Linear per MoE layer, e.g. "model.layers.3.mlp.gate".
    gates: list[str] = field(default_factory=list)
    #: {layer_prefix: {expert_index: expert_module_name}}, expert_module_name e.g.
    #: "model.layers.3.mlp.experts.17".
    experts: dict[str, dict[int, str]] = field(default_factory=dict)
    #: shared-expert module names (empty when the checkpoint has none), e.g.
    #: "model.layers.3.shared_expert".
    shared: list[str] = field(default_factory=list)
    #: attention module names, e.g. "model.layers.3.self_attn".
    attns: list[str] = field(default_factory=list)

    def num_layers(self) -> int:
        return len(self.experts)


def discover_moe_structure(model) -> MoEStructure:
    """Walk the module tree of a loaded (LLaDA-)MoE and record gate/expert/shared/attention names.

    Works on the PEFT-unwrapped base model. Robust to the `model.model.` nesting and to the exact
    class names: it keys off attribute names (`gate`, `experts`, `shared_expert`, `self_attn`), which
    are stable in the LLaDA-MoE remote code.
    """
    st = MoEStructure()
    import re
    expert_re = re.compile(r"(.*\.experts)\.(\d+)$")
    for name, _module in model.named_modules():
        leaf = name.rsplit(".", 1)[-1]
        if leaf == "self_attn":
            st.attns.append(name)
        elif leaf == "gate" and name.endswith(".mlp.gate"):
            st.gates.append(name)
        elif leaf == "shared_expert":
            st.shared.append(name)
        else:
            m = expert_re.match(name)
            if m:
                layer_prefix = m.group(1)          # ".../mlp.experts"
                idx = int(m.group(2))
                st.experts.setdefault(layer_prefix, {})[idx] = name
    return st


# --- routing profiler -------------------------------------------------------------------------

def profile_routing(model, calib_batches, device, top_k: int | None = None):
    """One free forward per calibration batch; count how often each (layer, expert) is top-k selected.

    Returns {layer_prefix: numpy array of shape (num_experts,) of selection counts}. Uses forward
    hooks on each `mlp.gate` Linear (the router): the gate output is the router logits, we replay the
    model's own top-k selection (softmax -> topk) and tally the winners. This does not depend on the
    remote code's `output_router_logits` plumbing (which the DecoderLayer does not fully wire up).

    `calib_batches`: iterable of dicts with at least `input_ids` (a LongTensor, or a list of ints).
    `top_k`: experts selected per token; defaults to config.num_experts_per_tok.
    """
    import numpy as np
    import torch

    base = getattr(model, "base_model", None)
    root = getattr(base, "model", model) if base is not None else model
    st = discover_moe_structure(root)
    cfg = root.config
    k = top_k or int(getattr(cfg, "num_experts_per_tok", 8))
    num_experts = int(getattr(cfg, "num_experts", 64))

    counts: dict[str, "np.ndarray"] = {
        lp: np.zeros(num_experts, dtype=np.int64) for lp in st.experts
    }
    # map gate module name -> its layer's expert prefix (".../mlp.gate" -> ".../mlp.experts")
    gate_to_prefix = {g: g[: -len(".gate")] + ".experts" for g in st.gates}

    handles = []
    modules_by_name = dict(root.named_modules())

    def make_hook(layer_prefix):
        def hook(_m, _inp, out):
            logits = out if not isinstance(out, tuple) else out[0]
            # (tokens, num_experts) -> softmax -> top-k winners
            probs = torch.softmax(logits.float(), dim=-1)
            _, sel = torch.topk(probs, k, dim=-1)          # (tokens, k)
            flat = sel.reshape(-1).to("cpu").numpy()
            binc = np.bincount(flat, minlength=num_experts)
            counts[layer_prefix] += binc
        return hook

    for gate_name, prefix in gate_to_prefix.items():
        if prefix in counts and gate_name in modules_by_name:
            handles.append(modules_by_name[gate_name].register_forward_hook(make_hook(prefix)))

    model.eval()
    with torch.no_grad():
        for batch in calib_batches:
            ids = batch["input_ids"]
            if not torch.is_tensor(ids):
                ids = torch.tensor([ids] if isinstance(ids[0], int) else ids, dtype=torch.long)
            ids = ids.to(device)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            model(input_ids=ids)

    for h in handles:
        h.remove()
    return counts


def select_hot_experts(counts, hot_frac: float = 0.25) -> dict[str, list[int]]:
    """Per layer, return the indices of the hottest `hot_frac` experts (at least one)."""
    import numpy as np
    hot: dict[str, list[int]] = {}
    for layer_prefix, c in counts.items():
        n = len(c)
        keep = max(1, int(round(n * hot_frac)))
        order = np.argsort(-c)                     # most-selected first; ties broken by index
        hot[layer_prefix] = sorted(int(i) for i in order[:keep])
    return hot


# --- building PEFT target_modules -------------------------------------------------------------

def build_target_modules(model, mode: str = "routing", hot_frac: float = 0.25,
                          calib_batches=None, device: str = "cpu", top_k: int | None = None):
    """Build the PEFT `target_modules` list for LLaDA-MoE according to `mode`.

    mode:
      - "attention": attention projections only (q/k/v/o) — the previous behaviour, MoE-blind.
      - "all"      : attention + router gate + shared experts + *every* routed expert.
      - "routing"  : attention + router gate + shared experts + only the hot `hot_frac` experts
                     per layer, chosen by profiling `calib_batches`. Falls back to "all" if no
                     calibration data is given (with a printed warning).

    Returns (target_modules, info) where info records the selection (for logging / train_config.json).
    Hot experts are given as fully-qualified module names so PEFT adapts exactly those; attention and
    gate use bare suffixes / names.
    """
    base = getattr(model, "base_model", None)
    root = getattr(base, "model", model) if base is not None else model
    st = discover_moe_structure(root)

    info: dict = {"mode": mode, "hot_frac": hot_frac}

    if mode == "attention" or not st.experts:
        # No MoE (or explicitly attention-only): the classic q/k/v/o suffixes.
        if not st.experts and mode != "attention":
            print("[moe-lora] no MoE experts found; falling back to attention-only target modules",
                  flush=True)
        info["experts_selected"] = 0
        return list(ATTENTION_SUFFIXES), info

    # attention on every layer (bare suffixes) + router gate on every MoE layer (fully-qualified)
    targets: list[str] = list(ATTENTION_SUFFIXES)
    targets += list(st.gates)
    # shared experts, if any (their three linears)
    for sh in st.shared:
        targets += [f"{sh}.{lin}" for lin in EXPERT_LINEARS]

    if mode == "all":
        chosen = {lp: sorted(idxs.keys()) for lp, idxs in st.experts.items()}
    elif mode == "routing":
        if not calib_batches:
            print("[moe-lora] mode='routing' but no calibration data; adapting ALL experts", flush=True)
            chosen = {lp: sorted(idxs.keys()) for lp, idxs in st.experts.items()}
        else:
            counts = profile_routing(model, calib_batches, device, top_k=top_k)
            hot = select_hot_experts(counts, hot_frac=hot_frac)
            chosen = hot
            info["routing_counts"] = {lp: [int(x) for x in c] for lp, c in counts.items()}
    else:
        raise ValueError(f"unknown moe lora mode {mode!r} (use routing|all|attention)")

    n_selected = 0
    for layer_prefix, idxs in chosen.items():
        expert_names = st.experts.get(layer_prefix, {})
        for e in idxs:
            mod = expert_names.get(e)
            if mod is None:
                continue
            targets += [f"{mod}.{lin}" for lin in EXPERT_LINEARS]
            n_selected += 1

    info["experts_selected"] = n_selected
    info["experts_total"] = sum(len(v) for v in st.experts.values())
    info["hot_experts"] = {lp: idxs for lp, idxs in chosen.items()}
    info["num_gates"] = len(st.gates)
    info["num_shared"] = len(st.shared)
    # de-dup while preserving order
    seen = set()
    deduped = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped, info
