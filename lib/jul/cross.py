"""The cross reading: a second, small encoder that reads the question and the text together.

The vector method encodes the text and each option apart, so the model never sees both: a question
about how two things relate (is this a paraphrase of ..., does it follow that ..., was it paid before
the deadline, does it mention a place and not a date) is answered at chance. A cross model is an
encoder fine-tuned on pairs: `<s> question </s></s> text </s>` goes through it once, the mean of its
last layer feeds a small head chosen by the question type.

- noul: one pass, three logits (yes, no, unknown); p(true) = p(yes) + p(unknown) / 2. A Noul's own
  descriptions of true and false, when it has some, are appended to the question as the model was
  trained: "question (true: ...; false: ...)".
- score: one pass per level (`question\\nlevel` against the text), one logit each.
- choice: one pass per option, the same way. Not routed here by default: the vector reading scores as
  well on classification and reads every option once for all calls.

A cross model is a directory (or Hub repo) with the encoder's weights for a backend (transformers for
torch, `jul.backends.onnx_export` for onnx), its tokenizer, and `cross.json`:

    {"method": "cross", "prefix": "query: ", "max_length": 256, "layer": 11, "separator": [2, 2],
     "types": ["noul", "score"], "heads": "cross_heads.npz"}

`separator` is the token ids between the two segments in the tokenizer's pair template (XLM-R and e5:
</s></s>; BERT: [SEP]); `write_spec` reads it from the tokenizer.

`heads` holds `noul_weight` (3, d), `noul_bias` (3,), `choice_weight` (1, d), `choice_bias` (1,),
`score_weight` (1, d), `score_bias` (1,). A preset points to it with `cross: {"repo": ..., "subfolder": ...}`
(a model repo can carry its cross model in `cross/`, which `jul models add` attaches by itself); the
vector model keeps answering everything else. On onnx, `JUL_ONNX_CROSS_MODEL` replaces the cross graph as
`JUL_ONNX_MODEL` replaces the vector one (a local path or `s3://bucket/key`). The pair is cut to `max_length` tokens the way the
model was trained (longest_first: a token off the longer side, until it fits).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .decision import render
from .types import NOUL_DEFAULTS, Option

SPEC_FILE = "cross.json"
TYPES = ("noul", "choice", "score")


@dataclass(frozen=True)
class CrossSpec:
    prefix: str
    max_length: int
    layer: int
    separator: tuple[int, ...]
    types: tuple[str, ...]
    heads_file: str
    directory: Path

    @classmethod
    def load(cls, directory: str | Path) -> "CrossSpec":
        directory = Path(directory)
        d = json.loads((directory / SPEC_FILE).read_text())
        if d.get("method") != "cross":
            raise ValueError(f"{directory / SPEC_FILE}: unsupported method {d.get('method')!r}")
        types = tuple(d.get("types", ("noul", "score")))
        if set(types) - set(TYPES):
            raise ValueError(f"{directory / SPEC_FILE}: unknown types {sorted(set(types) - set(TYPES))}")
        return cls(prefix=d.get("prefix", ""), max_length=int(d.get("max_length", 256)), layer=int(d["layer"]),
                   separator=tuple(int(i) for i in d["separator"]), types=types,
                   heads_file=d.get("heads", "cross_heads.npz"), directory=directory)


SUBFOLDER = "cross"
GRAPH_ENV = "JUL_ONNX_CROSS_MODEL"


def local_dir(repo: str, subfolder: str | None = None) -> Path:
    """The cross model's directory: `repo` itself or its `subfolder`, downloaded (that part only) for a Hub repo."""
    if Path(repo).is_dir():
        return Path(repo) / subfolder if subfolder else Path(repo)
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(repo, allow_patterns=[f"{subfolder}/*"] if subfolder else None))
    return root / subfolder if subfolder else root


def find(repo: str) -> dict | None:
    """A preset's `cross` entry for the cross model a model repo carries in `cross/`, or None."""
    if Path(repo).is_dir():
        return ({"repo": str(Path(repo).resolve()), "subfolder": SUBFOLDER}
                if (Path(repo) / SUBFOLDER / SPEC_FILE).is_file() else None)
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo, f"{SUBFOLDER}/{SPEC_FILE}")
    except Exception:
        return None
    return {"repo": repo, "subfolder": SUBFOLDER}


def location(cross: dict, backend: str) -> Path:
    """The directory a preset's `cross` entry names for `backend` (`repo` may be a {backend: repo} dict)."""
    repo = cross["repo"]
    repo = repo.get(backend) if isinstance(repo, dict) else repo
    if not repo:
        raise ValueError(f"the cross model has no {backend} repo")
    return local_dir(repo, cross.get("subfolder"))


def load(cross: dict, backend: str, base=None):
    """The reader of a preset's `cross` entry: its encoder loaded on `backend`, or, for a LoRA cross model,
    adapters attached to `base` (the preset's own backbone: one model in memory for both readings)."""
    from .backbone import Backbone
    directory = location(cross, backend)
    if json.loads((directory / SPEC_FILE).read_text()).get("method") == "lora":
        if base is None:
            raise ValueError("a LoRA cross model reads with the preset's own backbone: pass `base`")
        return LoraCrossReader(base, LoraSpec.load(directory))
    backbone = Backbone(str(directory), backend, **({"graph_env": GRAPH_ENV} if backend == "onnx" else {}))
    return CrossReader(backbone, CrossSpec.load(directory))


def cut(a: list[int], b: list[int], budget: int) -> tuple[list[int], list[int]]:
    """transformers' "longest_first": drop the last token of the longer side until both fit."""
    a, b = list(a), list(b)
    while len(a) + len(b) > budget:
        if len(a) > len(b):
            a.pop()
        else:
            b.pop()
    return a, b


class CrossReader:
    """Scores questions with a cross model loaded on `backbone` (an encoder backbone)."""

    def __init__(self, backbone, spec: CrossSpec):
        if backbone.architecture != "encoder":
            raise ValueError(f"a cross model must be an encoder, {backbone.name} is a {backbone.architecture}")
        self.backbone, self.spec = backbone, spec
        w = np.load(spec.directory / spec.heads_file)
        self.heads = {t: (w[f"{t}_weight"].astype(np.float32), w[f"{t}_bias"].astype(np.float32)) for t in TYPES}
        from .encoder import special_tokens
        head, tail = special_tokens(backbone.tokenizer.encode)
        self._sep = list(spec.separator)
        self._specials = len(head) + len(tail) + len(self._sep)

    def handles(self, kind: str) -> bool:
        return kind in self.spec.types

    def _pair(self, first: str, text: str) -> list[int]:
        encode = self.backbone.tokenizer.encode
        a = encode(self.spec.prefix + first, add_special_tokens=False)
        b = encode(text, add_special_tokens=False)
        a, b = cut(a, b, self.spec.max_length - self._specials)
        return a + self._sep + b

    @staticmethod
    def firsts(kind: str, instructions: str, options: list[Option]) -> list[str]:
        """The first segment of each pair: the question alone (noul), or the question and one option."""
        if kind == "noul":
            by_key = {o.key: o.description for o in options}
            given = [f"{k}: {by_key[k]}" for k in ("true", "false") if by_key.get(k) and by_key[k] != NOUL_DEFAULTS[k]]
            return [instructions + (f" ({'; '.join(given)})" if given else "")]
        if kind == "score":
            return [f"{instructions}\n{o.description}" for o in options]
        return [f"{instructions}\n{o.key}: {render(o.description)}" if o.description else f"{instructions}\n{o.key}"
                for o in options]

    def logits(self, state: Any, kind: str, instructions: str, options: list[Option]) -> tuple[np.ndarray, int]:
        """Logits in jul's option order (a softmax gives the answer's probabilities), and the tokens run."""
        text = state if isinstance(state, str) else render(state)
        pairs = [self._pair(f, text) for f in self.firsts(kind, instructions, options)]
        feats = self.backbone.forward_batch(pairs, layers=(self.spec.layer,))
        d = feats[0][self.spec.layer].shape[0] // 2
        v = np.stack([f[self.spec.layer][:d] for f in feats])       # mean over the whole pair
        w, b = self.heads[kind]
        z = v @ w.T + b
        tokens = sum(len(p) + self._specials - len(self._sep) for p in pairs)   # + the outer special tokens
        if kind == "noul":
            p = np.exp(z[0] - z[0].max())
            p /= p.sum()
            t = float(np.clip(p[0] + p[2] / 2, 1e-7, 1 - 1e-7))
            by_key = {"true": np.log(t), "false": np.log(1 - t)}
            return np.array([by_key[o.key] for o in options], dtype=np.float32), tokens
        return z[:, 0].astype(np.float32), tokens


# --- LoRA cross models: the preset's own decoder, with adapters switched on for the cross reading ---------

@dataclass(frozen=True)
class LoraSpec:
    """`cross.json` of a LoRA cross model (`"method": "lora"`): adapters for the preset's decoder, read on the
    last token of its last layer after the final norm.

        {"method": "lora", "base": "tencent/WeMM-Embedding-4B", "scale": 2.0, "adapter": "adapter.npz",
         "heads": "cross_heads.npz", "types": ["noul", "score"], "max_length": 320,
         "prompt": "Text: \\"{state}\\"\\n{first}\\nVerdict:",
         "firsts": {"noul": "Question: {question}", "score": "Question: {question}\\nCandidate answer: {option}"}}

    `adapter` holds `<module path>.a` (r, in) and `<module path>.b` (out, r) per wrapped Linear, the path
    relative to the decoder (`layers.3.self_attn.q_proj`), as in transformers and mlx-lm alike.
    """
    base: str
    scale: float
    adapter_file: str
    heads_file: str
    types: tuple[str, ...]
    max_length: int
    prompt: str
    firsts: dict
    directory: Path

    @classmethod
    def load(cls, directory: str | Path) -> "LoraSpec":
        directory = Path(directory)
        d = json.loads((directory / SPEC_FILE).read_text())
        return cls(base=d["base"], scale=float(d["scale"]), adapter_file=d.get("adapter", "adapter.npz"),
                   heads_file=d.get("heads", "cross_heads.npz"), types=tuple(d.get("types", ("noul", "score"))),
                   max_length=int(d.get("max_length", 320)), prompt=d["prompt"], firsts=d["firsts"],
                   directory=directory)


def _attach_mlx(backbone, weights: dict, scale: float) -> list:
    import mlx.core as mx
    import mlx.nn as nn

    class LoRA(nn.Module):
        def __init__(self, base, a, b):
            super().__init__()
            self.base, self.a, self.b = base, a, b
            self.on = False

        def __call__(self, x):
            y = self.base(x)
            if not self.on:
                return y
            if self.a.dtype != x.dtype:
                self.a, self.b = self.a.astype(x.dtype), self.b.astype(x.dtype)
            return y + ((x @ self.a.T) @ self.b.T * scale).astype(y.dtype)

    wrappers = []
    for path in sorted({k.rsplit(".", 1)[0] for k in weights}):
        parts = path.split(".")
        parent = backbone._inner.layers[int(parts[1])]
        parent = getattr(parent, "block", parent)           # jul's tap around each block
        for p in parts[2:-1]:
            parent = getattr(parent, p)
        base = getattr(parent, parts[-1])
        if getattr(base, "_jul_lora", False):              # attached again: replace, never stack
            base = base.base
        lo = LoRA(base, mx.array(weights[f"{path}.a"]), mx.array(weights[f"{path}.b"]))
        lo._jul_lora = True
        setattr(parent, parts[-1], lo)
        wrappers.append(lo)
    return wrappers


def _attach_torch(backbone, weights: dict, scale: float) -> list:
    import torch

    class LoRA(torch.nn.Module):
        def __init__(self, base, a, b):
            super().__init__()
            self.base = base
            self.register_buffer("a", a)
            self.register_buffer("b", b)
            self.on = False

        def forward(self, x):
            y = self.base(x)
            if not self.on:
                return y
            return y + (x.to(self.a.dtype) @ self.a.T @ self.b.T * scale).to(y.dtype)

    wrappers = []
    dtype = next(backbone._decoder.parameters()).dtype
    for path in sorted({k.rsplit(".", 1)[0] for k in weights}):
        parts = path.split(".")
        parent = backbone._decoder.layers[int(parts[1])]
        for p in parts[2:-1]:
            parent = getattr(parent, p)
        base = getattr(parent, parts[-1])
        if getattr(base, "_jul_lora", False):              # attached again: replace, never stack
            base = base.base
        a, b = (torch.from_numpy(weights[f"{path}.{k}"]).to(base.weight.device, dtype) for k in ("a", "b"))
        lo = LoRA(base, a, b)
        lo._jul_lora = True
        setattr(parent, parts[-1], lo)
        wrappers.append(lo)
    return wrappers


class LoraCrossReader:
    """The cross reading of a decoder through LoRA adapters attached to it. The adapters are off except
    inside `logits`, so the vector reading of the same backbone is untouched (one model in memory)."""

    def __init__(self, backbone, spec: LoraSpec):
        if backbone.architecture != "decoder":
            raise ValueError(f"a LoRA cross model needs a decoder backbone, {backbone.name} is a {backbone.architecture}")
        self.backbone, self.spec = backbone, spec
        weights = dict(np.load(spec.directory / spec.adapter_file))
        attach = {"mlx": _attach_mlx, "torch": _attach_torch}.get(backbone.backend)
        if attach is None:
            raise ValueError(f"LoRA cross models run on mlx or torch, not {backbone.backend}")
        self._loras = attach(backbone, weights, spec.scale)
        w = np.load(spec.directory / spec.heads_file)
        self.heads = {t: (w[f"{t}_weight"].astype(np.float32), w[f"{t}_bias"].astype(np.float32)) for t in TYPES}
        self._layer = backbone.n_layers - 1

    def handles(self, kind: str) -> bool:
        return kind in self.spec.types

    def _switch(self, on: bool) -> None:
        for lo in self._loras:
            lo.on = on

    def _final_norm(self, v: np.ndarray) -> np.ndarray:
        bb = self.backbone
        if bb.backend == "mlx":
            import mlx.core as mx
            return np.array(bb._inner.norm(mx.array(v)).astype(mx.float32))
        import torch
        with torch.no_grad():
            return bb._decoder.norm(torch.from_numpy(v).to(bb.device)).float().cpu().numpy()

    def firsts(self, kind: str, instructions: str, options: list[Option]) -> list[str]:
        f = self.spec.firsts[kind]
        if kind == "noul":
            return [f.format(question=CrossReader.firsts("noul", instructions, options)[0])]
        return [f.format(question=instructions, option=o.description if kind == "score" else
                         (f"{o.key}: {render(o.description)}" if o.description else o.key)) for o in options]

    def _prompt(self, first: str, text: str) -> list[int]:
        """As the adapters were trained: the text is cut so that the whole prompt fits in max_length."""
        head, tail = self.spec.prompt.split("{state}")
        tail_ids = self.backbone.encode(tail.replace("{first}", first))
        room = max(8, self.spec.max_length - len(tail_ids))
        return self.backbone.encode(head + text)[:room] + tail_ids

    def logits(self, state: Any, kind: str, instructions: str, options: list[Option]) -> tuple[np.ndarray, int]:
        text = state if isinstance(state, str) else render(state)
        prompts = [self._prompt(f, text) for f in self.firsts(kind, instructions, options)]
        self._switch(True)
        try:
            feats = self.backbone.forward_batch(prompts, layers=(self._layer,))
        finally:
            self._switch(False)
        d = feats[0][self._layer].shape[0] // 2
        v = self._final_norm(np.stack([f[self._layer][:d] for f in feats]).astype(np.float32))
        w, b = self.heads[kind]
        z = v @ w.T + b
        tokens = sum(len(p) for p in prompts)
        if kind == "noul":
            p = np.exp(z[0] - z[0].max())
            p /= p.sum()
            t = float(np.clip(p[0] + p[2] / 2, 1e-7, 1 - 1e-7))
            by_key = {"true": np.log(t), "false": np.log(1 - t)}
            return np.array([by_key[o.key] for o in options], dtype=np.float32), tokens
        return z[:, 0].astype(np.float32), tokens


def write_spec(directory: str | Path, heads_pt: str | Path | None = None, prefix: str = "query: ",
               max_length: int = 256, types=("noul", "score")) -> Path:
    """Make a trained cross model readable by jul: cross.json, and cross_heads.npz from a torch state dict
    of heads (the lab's cross_heads.pt: noul.weight, noul.bias, ...). Needs transformers."""
    from transformers import AutoConfig, AutoTokenizer
    directory = Path(directory)
    tok = AutoTokenizer.from_pretrained(directory)
    from .encoder import special_tokens
    head, tail = special_tokens(tok.encode)
    a, b = tok.encode("a", add_special_tokens=False), tok.encode("b", add_special_tokens=False)
    pair = tok("a", "b")["input_ids"]          # head + a + separator + b + tail
    separator = pair[len(head) + len(a): len(pair) - len(tail) - len(b)]
    if heads_pt:
        import torch
        sd = torch.load(heads_pt, map_location="cpu")
        np.savez(directory / "cross_heads.npz", **{k.replace(".", "_"): v.float().numpy() for k, v in sd.items()})
    spec = {"method": "cross", "prefix": prefix, "max_length": max_length,
            "layer": AutoConfig.from_pretrained(directory).num_hidden_layers - 1,
            "separator": separator, "types": list(types), "heads": "cross_heads.npz"}
    (directory / SPEC_FILE).write_text(json.dumps(spec, indent=1) + "\n")
    return directory / SPEC_FILE
