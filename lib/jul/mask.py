"""The mask method: read a masked-diffusion model (LLaDA) at a single [MASK], in one forward pass.

This is the native readout of a masked diffusion LM. Where the pointer method (decision.py) reads a
*causal* model at delimiter tokens with a trained head, the mask method places one `[MASK]` where the
answer goes, runs a single bidirectional forward, and reads the vocabulary distribution at that mask
— restricted to the tokens that stand for the options. No trained head, no decode loop, no iterative
unmasking: the option distribution is the softmax of the masked-position logits over the option
tokens. This is the reading Jev's "parallel sampler" is believed to use, and LLaDA has it natively
because predicting masked tokens *is* its pre-training objective.

The prompt is one sequence:  <instructions> ... <options listing> ... Answer: [MASK]
For each option we take an anchor token (by default the first token of its key/marker). The masked
logits over those anchors, divided by a temperature, softmaxed, are the answer.

  - choice : anchors are the option markers (A, B, C, ... or the option keys).
  - noul   : two ordered anchors [false, true]; P(yes) = softmax(...)[true].
  - score  : ordered level anchors (0, 1, 2, ...); the expected level is computed by the client.

A `decision.json` with `"method": "mask"` configures it:

    {
      "method": "mask",
      "mask_token": "<|mdm_mask|>",       // optional; else the backbone's mask id is used
      "temperature": 1.0,
      "readout": {"markers": "letters" | "numbers" | "keys"},
      "limits": {"max_state_tokens": 4096, "max_branch_tokens": 2048},
      "prompt": {                           // optional; sensible defaults below
        "template": "{instructions}\nOptions:\n{options}\nAnswer: ",
        "option_line": "{marker}) {text}",
        "score_hint": "Rate on the scale.",
        "noul_hint": "Answer true or false."
      }
    }
"""

from __future__ import annotations

import json
import os
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .types import Option

SPEC_FILE = "decision.json"
LETTERS = string.ascii_uppercase


def _env_int(name: str, default: int) -> int:
    import os
    v = os.environ.get(name)
    return int(v) if v is not None else default

DEFAULT_PROMPT = {
    "template": "{instructions}\n{listing}\nAnswer: ",
    "option_line": "{marker}) {text}",
    "score_hint": "Answer with the number of the level.",
    "noul_hint": "Answer true or false.",
}


@dataclass(frozen=True)
class MaskSpec:
    """The content of a `method='mask'` decision.json."""

    temperature: float
    markers: str                       # "letters" | "numbers" | "keys"
    max_state_tokens: int
    max_branch_tokens: int
    mask_token: str | None
    prompt: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_PROMPT))
    directory: Path | None = None
    #: Diffusion inference tricks (no retraining). n_mask=1/n_steps=1 == the plain single-forward read.
    n_mask: int = 1                    # place K consecutive [MASK]s and average their option logits
    n_steps: int = 1                   # iterative demasking: refine the read over K forward passes
    #: "anchor" (default): read one anchor token per option marker. "multitoken": score the full
    #: sequence-likelihood of each option's label text over a block of [MASK]s (unblocks many-class
    #: tasks; needs a diffusion backbone read at [MASK]). "auto": anchor at/below
    #: `route_multitoken_above` options, multitoken beyond it (best-of-both: anchor is crisp on few
    #: classes, multitoken unblocks many).
    readout: str = "anchor"
    route_multitoken_above: int = 10

    @classmethod
    def load(cls, directory: str | Path, file: str | Path | None = None) -> "MaskSpec":
        directory = Path(directory)
        d = json.loads(Path(file or directory / SPEC_FILE).read_text())
        if d.get("method") != "mask":
            raise ValueError(f"{directory / SPEC_FILE}: not a mask spec (method={d.get('method')!r})")
        limits = d.get("limits", {})
        prompt = {**DEFAULT_PROMPT, **d.get("prompt", {})}
        readout = d.get("readout", {})
        return cls(temperature=float(d.get("temperature", 1.0)),
                   markers=readout.get("markers", "letters"),
                   max_state_tokens=int(limits.get("max_state_tokens", 4096)),
                   max_branch_tokens=int(limits.get("max_branch_tokens", 2048)),
                   mask_token=d.get("mask_token"), prompt=prompt, directory=directory,
                   n_mask=_env_int("JUL_LLADA_N_MASK", int(readout.get("n_mask", 1))),
                   n_steps=_env_int("JUL_LLADA_N_STEPS", int(readout.get("n_steps", 1))),
                   readout=os.environ.get("JUL_LLADA_READOUT", readout.get("mode", "anchor")),
                   route_multitoken_above=_env_int("JUL_LLADA_ROUTE_ABOVE",
                                                   int(readout.get("route_multitoken_above", 10))))

    @classmethod
    def default(cls) -> "MaskSpec":
        """A spec with no file: sensible defaults, for a bare LLaDA repo. Env can turn on the tricks."""
        return cls(temperature=1.0, markers="letters", max_state_tokens=4096,
                   max_branch_tokens=2048, mask_token=None, prompt=dict(DEFAULT_PROMPT),
                   n_mask=_env_int("JUL_LLADA_N_MASK", 1), n_steps=_env_int("JUL_LLADA_N_STEPS", 1),
                   readout=os.environ.get("JUL_LLADA_READOUT", "anchor"),
                   route_multitoken_above=_env_int("JUL_LLADA_ROUTE_ABOVE", 10))


def _markers(kind: str, options: list[Option], mode: str) -> list[str]:
    """The short symbol shown for each option, whose first token is the read anchor."""
    n = len(options)
    if kind == "noul":
        return ["false", "true"]  # anchors read in this order; P(yes)=softmax[...][1]
    if kind == "score":
        return [str(i) for i in range(n)]
    if mode == "keys":
        return [o.key for o in options]
    if mode == "numbers":
        return [str(i) for i in range(n)]
    if n > len(LETTERS):
        # Too many options for single letters: fall back to keys (their first token is the anchor).
        return [o.key for o in options]
    return list(LETTERS[:n])


class MaskReader:
    """Encode a state+question with a trailing [MASK]; read the option distribution at the mask."""

    def __init__(self, backbone, spec: MaskSpec):
        self.backbone, self.spec = backbone, spec
        # The mask id: an explicit token in the spec, else the backbone's own (LLaDABackbone.mask_id).
        if spec.mask_token is not None:
            mid = backbone.tokenizer.convert_tokens_to_ids(spec.mask_token)
            if not isinstance(mid, int) or mid < 0:
                raise ValueError(f"mask_token {spec.mask_token!r} is unknown to {backbone.name}")
            self.mask_id = mid
        else:
            self.mask_id = getattr(backbone, "mask_id", None)
            if self.mask_id is None:
                raise ValueError(
                    f"{backbone.name} has no mask id and the spec sets no mask_token; "
                    "the mask method needs a masked-diffusion backbone (backend='llada')."
                )
        # Optional learned read head on the [MASK] hidden state (DiffEmbed-style). When JUL_LLADA_HEAD
        # points to a read_head.pt, decisions are read with the head instead of vocabulary logits.
        self.head = None
        head_path = os.environ.get("JUL_LLADA_HEAD")
        if head_path:
            self._load_head(head_path)

    def _load_head(self, head_path: str):
        import json as _json
        import torch
        import sys
        sys.path.insert(0, os.path.dirname(head_path) or ".")
        from llada_head import ReadHead
        meta = _json.load(open(os.path.join(os.path.dirname(head_path), "read_head.json")))
        emb = self.backbone.model.get_input_embeddings()
        head = ReadHead(hidden=meta["hidden"], emb=emb, proj=meta["proj"],
                        max_levels=meta.get("max_levels", 16))
        head.load_state_dict(torch.load(head_path, map_location="cpu"), strict=False)
        head.eval()
        dev = getattr(self.backbone, "device", "cpu")
        mdl_dtype = next(self.backbone.model.parameters()).dtype
        self.head = head.to(device=dev, dtype=mdl_dtype)

    # --- anchors ------------------------------------------------------------------------------

    def _anchor_id(self, marker: str) -> int:
        """The token id read for an option: the first token of its marker.

        Prefer a leading-space variant (" A", " true"), the form a token takes mid-sentence, and fall
        back to the bare marker. Returns the first sub-token id.
        """
        for form in (" " + marker, marker):
            ids = self.backbone.tokenizer.encode(form, add_special_tokens=False)
            if ids:
                return ids[0]
        raise ValueError(f"marker {marker!r} does not tokenize")

    # --- prompt building ----------------------------------------------------------------------

    def _listing(self, markers: list[str], options: list[Option]) -> str:
        line = self.spec.prompt["option_line"]
        return "\n".join(line.format(marker=m, text=o.text) for m, o in zip(markers, options))

    def _hint(self, kind: str) -> str:
        if kind == "score":
            return self.spec.prompt.get("score_hint", "")
        if kind == "noul":
            return self.spec.prompt.get("noul_hint", "")
        return ""

    def _prompt_tokens(self, kind: str, instructions: str, markers: list[str],
                       options: list[Option], state: str) -> tuple[list[int], list[int]]:
        """Full token sequence with `n_mask` trailing [MASK]s; returns (tokens, mask_positions)."""
        tok = self.backbone.tokenizer
        state_ids = tok.encode(state, add_special_tokens=False)[: self.spec.max_state_tokens]
        listing = self._listing(markers, options)
        hint = self._hint(kind)
        instr = instructions + (("\n" + hint) if hint else "")
        head = self.spec.prompt["template"].format(instructions=instr, listing=listing)
        head_ids = tok.encode(head, add_special_tokens=False)[: self.spec.max_branch_tokens]
        # State first (so it is fully attended by the bidirectional model), then the question head,
        # then K masks.  <state>\n<head>[MASK]x K
        sep = tok.encode("\n", add_special_tokens=False)
        k = max(1, self.spec.n_mask)
        base = state_ids + sep + head_ids
        tokens = base + [self.mask_id] * k
        mask_positions = list(range(len(base), len(base) + k))
        return tokens, mask_positions

    # --- scoring ------------------------------------------------------------------------------

    def logits(self, state: Any, questions: list[tuple[str, str, list[Option]]]) -> tuple[list[np.ndarray], int]:
        """Same signature as PointerReader.logits.

        `questions`: (kind, instructions, options). Returns option logits in jul's option order,
        already divided by the temperature, and the number of tokens run.

        Diffusion inference tricks (no retraining), off by default (n_mask=n_steps=1):
          - n_mask>1: place K [MASK]s and average their option-logit distributions (variance reduction),
          - n_steps>1: iterative demasking — commit the most confident mask to its argmax option token
            and re-forward, so later reads condition on an increasingly complete answer, the way a
            diffusion sampler unmasks confident tokens first.
        """
        text = state if isinstance(state, str) else _render(state)
        out, spent = [], 0
        for kind, instructions, options in questions:
            # learned head on the [MASK] hidden state (DiffEmbed): read with the trained per-type head
            # instead of vocabulary logits. Highest priority when a head is loaded.
            if self.head is not None:
                z, used = self._head_scores(kind, text, instructions, options)
                spent += used
                out.append(z)
                continue
            # dedicated yes/no readout (auto/auto-ce/multitoken): a natural "Answer (yes/no): [MASK]"
            # prompt read over several yes/no anchor pairs, instead of one arbitrary false/true anchor.
            if kind == "noul" and self.spec.readout in ("auto", "auto-ce", "multitoken"):
                z, used = self._noul_scores(text, instructions, options)
                spent += used
                out.append(z)
                continue
            # per-question readout: "auto" routes choice by option count (anchor few / multitoken many).
            # "auto-ce" keeps choice on anchor always (a CE-trained model learned the anchor channel;
            # multitoken then hurts choice — measured on decision-bench). "multitoken" forces it.
            use_multitoken = (
                kind == "choice" and (
                    self.spec.readout == "multitoken"
                    or (self.spec.readout == "auto"
                        and len(options) > self.spec.route_multitoken_above)
                )
            )
            if use_multitoken:
                z, used = self._multitoken_scores(text, instructions, options)
                spent += used
                out.append(z)  # already in option order, temperature-scaled
                continue
            # score: dedicated ordinal readout (auto/auto-ce/multitoken) reads level *descriptions* by
            # sequence-likelihood, then keeps jul's option order (the client turns it into E[level]).
            if kind == "score" and self.spec.readout in ("auto", "auto-ce", "multitoken"):
                z, used = self._multitoken_scores(text, instructions, options)
                spent += used
                out.append(z)
                continue
            markers = _markers(kind, options, self.spec.markers)
            anchors = [self._anchor_id(m) for m in markers]
            tokens, mask_positions = self._prompt_tokens(kind, instructions, markers, options, text)
            z, used = self._read_option_logits(tokens, list(mask_positions), anchors)
            spent += used
            out.append(self._to_option_order(kind, options, z))
        return out, spent

    # --- yes/no (noul) readout ----------------------------------------------------------------

    #: Anchor pairs read for a yes/no question, each as (yes_form, no_form). Their log-probs at the
    #: mask are averaged in log space, which is far more robust than one arbitrary false/true anchor.
    NOUL_PAIRS = ((" yes", " no"), (" Yes", " No"), (" true", " false"), (" True", " False"))

    def _noul_scores(self, state: str, instructions: str,
                     options: list[Option]) -> tuple[np.ndarray, int]:
        """Read a yes/no question at a single [MASK] after a natural "Answer (yes/no):" prompt.

        The old readout compared the first token of the literal words "false"/"true" listed as options,
        which is an arbitrary anchor and scored near chance. Here the prompt asks the question plainly
        and the mask logits are read over several yes/no *anchor pairs* (yes/no, Yes/No, true/false,
        True/False), averaged in log space. Returns logits in jul's Noul option order (options_of ->
        ("true","false")), temperature-scaled; the client reads P(yes) from the "true" slot.
        """
        tok = self.backbone.tokenizer
        state_ids = tok.encode(state, add_special_tokens=False)[: self.spec.max_state_tokens]
        head = self.spec.prompt.get("noul_head", "{instructions}\nAnswer (yes or no):")
        head_ids = tok.encode(head.format(instructions=instructions),
                              add_special_tokens=False)[: self.spec.max_branch_tokens]
        sep = tok.encode("\n", add_special_tokens=False)
        tokens = state_ids + sep + head_ids + [self.mask_id]
        pos = len(tokens) - 1
        logits = self.backbone.mask_logits(tokens, [pos])[0]          # (V,)
        logp = _log_softmax_rows(logits[None, :])[0]

        def first_id(form: str) -> int:
            ids = tok.encode(form, add_special_tokens=False)
            return ids[0] if ids else tok.unk_token_id

        yes_lp, no_lp = [], []
        for yform, nform in self.NOUL_PAIRS:
            yes_lp.append(float(logp[first_id(yform)]))
            no_lp.append(float(logp[first_id(nform)]))
        yes = float(np.mean(yes_lp))
        no = float(np.mean(no_lp))
        # jul's options_of(Noul) order is ("true", "false") -> ("yes", "no")
        order = {o.key: i for i, o in enumerate(options)}
        z = np.empty(len(options), dtype=np.float32)
        z[order["true"]] = yes
        z[order["false"]] = no
        return z / self.spec.temperature, len(tokens)

    # --- multi-token (sequence-likelihood) readout --------------------------------------------

    def _multitoken_scores(self, state: str, instructions: str,
                           options: list[Option]) -> tuple[np.ndarray, int]:
        """Score each option by the length-normalized log-likelihood of its label text.

        The single-anchor readout compares one token per option and saturates on many-class tasks
        (Banking77: ~0.02). Here each option's *text* is tokenized into a distinctive suffix (the
        token-level common prefix shared by all options is emitted as context, not scored); a block of
        K=max-suffix-length [MASK]s is placed where the answer goes; one bidirectional forward reads
        the full-vocabulary log-softmax at every mask position; option j scores as the mean log-prob of
        its own suffix tokens. Optionally refined over n_steps by committing the block's most confident
        position first (iterative demasking), which lets later reads condition on it.

        Returns (scores in option order, divided by temperature; tokens_used).
        """
        tok = self.backbone.tokenizer
        labels = [o.text for o in options]
        label_ids = [tok.encode(" " + lab, add_special_tokens=False) or [tok.unk_token_id]
                     for lab in labels]
        suffixes, prefix_len = _strip_common_prefix(label_ids)
        prefix_ids = label_ids[0][:prefix_len]

        state_ids = tok.encode(state, add_special_tokens=False)[: self.spec.max_state_tokens]
        head = self.spec.prompt.get("multitoken_head", "{instructions}\nThe correct label is:")
        head_ids = tok.encode(head.format(instructions=instructions),
                              add_special_tokens=False)[: self.spec.max_branch_tokens]
        sep = tok.encode("\n", add_special_tokens=False)
        base = state_ids + sep + head_ids + list(prefix_ids)
        kmax = max(len(s) for s in suffixes)

        tokens = base + [self.mask_id] * kmax
        positions = list(range(len(base), len(base) + kmax))
        steps = max(1, self.spec.n_steps)
        used = 0
        scores = None
        for step in range(steps):
            logits = self.backbone.mask_logits(tokens, positions)  # (len(positions), V)
            used += len(tokens)
            logp = _log_softmax_rows(logits)
            # map absolute position -> row index in the current `logits`
            row_of = {p: i for i, p in enumerate(positions)}
            scores = np.empty(len(options), dtype=np.float64)
            for j, suf in enumerate(suffixes):
                block = base + list(suf)  # suffix tokens sit at base..base+len(suf)-1
                lp = 0.0
                for i in range(len(suf)):
                    ap = len(base) + i
                    lp += logp[row_of[ap], suf[i]] if ap in row_of else 0.0
                scores[j] = lp / max(1, len(suf))
            if steps == 1 or len(positions) <= 1:
                break
            # iterative demasking: commit the block position whose best token is most confident,
            # to the argmax token of the *most likely option* at that position, then re-forward.
            best_option = int(np.argmax(scores))
            suf = suffixes[best_option]
            # find, among still-masked positions, the one with the highest max log-prob
            conf = [(logp[row_of[p]].max(), p) for p in positions]
            _, pos = max(conf)
            off = pos - len(base)
            commit_tok = suf[off] if off < len(suf) else int(np.argmax(logp[row_of[pos]]))
            tokens[pos] = commit_tok
            positions = [p for p in positions if p != pos]
        return (scores / self.spec.temperature).astype(np.float32), used

    # --- anchor readout -----------------------------------------------------------------------

    def _read_option_logits(self, tokens, mask_positions, anchors) -> tuple[np.ndarray, int]:
        """Average option logits over the mask positions, optionally refined over n_steps.

        When the backbone exposes `option_logits(tokens, positions, anchors)` (the MLX LLaDA backend),
        it is used instead of a full-vocabulary `mask_logits` + slicing: only the anchor rows of the
        output projection are computed, which is the dominant saving on a typed decision. The result
        is numerically identical to `mask_logits(...)[:, anchors]`.
        """
        tokens = list(tokens)
        remaining = list(mask_positions)
        steps = max(1, self.spec.n_steps)
        used = 0
        agg = None
        fast = getattr(self.backbone, "option_logits", None)
        early = getattr(self.backbone, "option_logits_early", None)
        use_early = fast is not None and early is not None and getattr(self.backbone, "early_skip", False)
        for step in range(steps):
            if use_early:
                per_pos, _ = early(tokens, remaining, anchors)  # (len(remaining), len(anchors))
            elif fast is not None:
                per_pos = fast(tokens, remaining, anchors)  # (len(remaining), len(anchors))
            else:
                vocab = self.backbone.mask_logits(tokens, remaining)  # (len(remaining), V)
                per_pos = np.stack(
                    [np.array([row[a] for a in anchors], dtype=np.float32) for row in vocab]
                )
            used += len(tokens)
            agg = per_pos.mean(axis=0) / self.spec.temperature
            if steps == 1 or len(remaining) <= 1:
                break
            # commit the most confident position to its argmax option's anchor token, then re-forward
            conf = per_pos.max(axis=1)
            best_local = int(np.argmax(conf))
            pos = remaining[best_local]
            tokens[pos] = anchors[int(np.argmax(per_pos[best_local]))]
            remaining = [p for p in remaining if p != pos]
        return agg, used

    def _head_scores(self, kind: str, state: str, instructions: str,
                     options: list[Option]) -> tuple[np.ndarray, int]:
        """Read a decision with the learned head on the [MASK] hidden state (not vocab logits).

        Builds the same prompt as training, reads the last-layer hidden state at the first [MASK], and
        applies the per-type head. Returns logits in jul's option order, temperature-scaled.
        """
        import torch
        markers = _markers(kind, options, self.spec.markers)
        anchors = [self._anchor_id(m) for m in markers]
        tokens, mask_positions = self._prompt_tokens(kind, instructions, markers, options, state)
        h_np = self.backbone.mask_hidden(tokens, [mask_positions[0]])[0]  # (hidden,)
        dev = next(self.head.parameters()).device
        hd = next(self.head.parameters()).dtype
        h = torch.tensor(h_np, device=dev, dtype=hd)
        with torch.inference_mode():
            if kind == "choice":
                z = self.head.choice_logits(h, torch.tensor(anchors, device=dev)).float().cpu().numpy()
                z = self._to_option_order(kind, options, z)
            elif kind == "noul":
                # The head is trained with gold_index in jul's option order (options_of(Noul) =
                # [true, false], see llada_train.build_example), so its logits are already in option
                # order: no [false, true] remap here (that remap flipped every learned-head answer).
                z = self.head.noul_logits(h).float().cpu().numpy()
            else:  # score
                z = self.head.score_logits(h, len(options)).float().cpu().numpy()
        return z / self.spec.temperature, len(tokens)

    def _to_option_order(self, kind, options, z: np.ndarray) -> np.ndarray:
        if kind == "noul":
            # anchors were built in [false, true]; return jul's options_of order.
            order = {o.key: i for i, o in enumerate(options)}
            ordered = np.empty(len(options), dtype=np.float32)
            ordered[order["false"]] = z[0]
            ordered[order["true"]] = z[1]
            return ordered
        return z  # score/choice: marker order already matches option order


def _render(state: Any) -> str:
    """Serialize a non-string state the same way decision.render does (labeled text)."""
    from .decision import render
    return render(state)


def _log_softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax of a (P, V) float32 array (for the multi-token readout)."""
    m = logits.max(axis=1, keepdims=True)
    shifted = logits - m
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


def _strip_common_prefix(seqs: list[list[int]]) -> tuple[list[list[int]], int]:
    """Drop the token-level longest common prefix shared by every sequence, keeping >=1 token each.

    Option texts in a task often share a boilerplate head; those common-mode tokens carry no signal
    for the argmax, so the multi-token readout emits them as context and scores only the distinctive
    suffix. Returns (suffixes, prefix_len)."""
    if not seqs:
        return seqs, 0
    shortest = min(len(s) for s in seqs)
    prefix_len = 0
    for i in range(shortest - 1):  # always keep at least one token in the shortest label
        if len({s[i] for s in seqs}) == 1:
            prefix_len += 1
        else:
            break
    return [s[prefix_len:] for s in seqs], prefix_len


def spec_source(repo: str) -> str | None:
    """Where a mask decision.json lives (a directory or a Hub repo holding one), else None."""
    if Path(repo).is_dir():
        return str(Path(repo).resolve()) if (Path(repo) / SPEC_FILE).exists() else None
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(repo, SPEC_FILE)
    except Exception:
        return None
    return repo
