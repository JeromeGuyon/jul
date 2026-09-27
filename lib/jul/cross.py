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
`score_weight` (1, d), `score_bias` (1,). A preset points to it with `cross: {"repo": ...}`; the
vector model keeps answering everything else. The pair is cut to `max_length` tokens the way the
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


def local_dir(repo: str) -> Path:
    if Path(repo).is_dir():
        return Path(repo)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo))


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
