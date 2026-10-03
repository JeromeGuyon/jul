"""Learned per-type read head on the [MASK] hidden state (DiffEmbed-style), as an alternative to
reading vocabulary logits at the [MASK].

Motivation: on decision-bench the biggest gap to wemm is on `score` (+22 pts), exactly where reading
raw vocabulary logits is weakest. wemm/Kev/Mapika do not read vocab logits — they read with a learned
head / pointer / projection on a representation. A masked-diffusion LM is bidirectional, so the hidden
state at the [MASK] is a strong representation (DiffEmbed, arXiv 2505.15045). This module puts a small
trained head on that hidden state instead of the vocab logits.

Heads (all consume the last-layer hidden state h ∈ R^D at the [MASK]):
  - choice : bilinear pointer. Query q = Wq·h; each option's key k_i = We·emb(anchor_id_i); the option
             logit is q·k_i / sqrt(d). Works for any option count, like a pointer head.
  - noul   : a 2-logit linear head on h (yes/no), in jul's option order ([true, false]).
  - score  : an ordinal head — K level logits on h, trained with the Gaussian ordinal target so the
             scale is treated as ordered, not as unordered classes.

The head is tiny and trained on top of a (LoRA-adapted or frozen) backbone. It is saved next to the
adapter so inference can load both: `read_head.pt` holds only the head's own weights (q, k, noul,
score), never the backbone embedding. `load_head` also accepts older files that did include it.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReadHead(nn.Module):
    def __init__(self, hidden: int, emb: nn.Embedding, proj: int = 256, max_levels: int = 16):
        super().__init__()
        self.hidden = hidden
        # The backbone's own input embedding, used read-only as the key source for choice. Kept OUT of
        # the module tree (object.__setattr__): otherwise state_dict() saves the whole vocab table
        # (~157k x D for LLaDA2) into read_head.pt, and load_state_dict writes it back over the
        # backbone's embeddings; .to(dtype) on the head would also cast the backbone's table.
        object.__setattr__(self, "emb", emb)
        self.q = nn.Linear(hidden, proj, bias=False)          # choice query (legacy bilinear-on-letter)
        self.k = nn.Linear(emb.embedding_dim, proj, bias=False)  # choice key from option anchor emb (legacy)
        self.noul = nn.Linear(hidden, 2)                      # yes/no
        self.score = nn.Linear(hidden, max_levels)            # ordinal levels (sliced to n)
        self.proj = proj
        # Listwise choice reading, ported from wemm (origin/main:lib/jul/cross.py). The query is the
        # hidden state at the [MASK]; each option's key is the hidden state of the separator token that
        # closes its line (so the key SEES the option text, unlike the legacy letter-embedding key that
        # dropped choice -2.7). Two linear projections, logit_i = (k_i . q) / sqrt(proj). Both project
        # the backbone hidden state, so choice_k's in-dim is `hidden` (not emb_dim).
        self.choice_q = nn.Linear(hidden, proj, bias=True)    # listwise query  (from [MASK] hidden)
        self.choice_k = nn.Linear(hidden, proj, bias=True)    # listwise key    (from each option-end hidden)

    def choice_logits(self, h: torch.Tensor, anchor_ids: torch.Tensor) -> torch.Tensor:
        """LEGACY bilinear-on-letter head (key = embedding of the A/B/C marker). Kept for the old
        read_head.pt files and the regression test; NOT used by the listwise reading. h: (D,) mask
        hidden; anchor_ids: (n,) option anchor token ids. Returns (n,) logits."""
        q = self.q(h)                                   # (proj,)
        with torch.no_grad():
            e = self.emb(anchor_ids)                    # (n, emb_dim) — embedding table is frozen
        k = self.k(e)                                   # (n, proj)
        return (k @ q) / (self.proj ** 0.5)             # (n,)

    def listwise_choice_logits(self, h_mask: torch.Tensor, h_ends: torch.Tensor) -> torch.Tensor:
        """Listwise choice reading (wemm port). h_mask: (D,) hidden at the [MASK] (the query);
        h_ends: (n, D) hidden at each option's closing separator (the keys). Returns (n,) option
        logits = (k_i . q) / sqrt(proj), in the order the options were laid out in the prompt
        (jul's option order)."""
        q = self.choice_q(h_mask)                       # (proj,)
        k = self.choice_k(h_ends)                       # (n, proj)
        return (k @ q) / (self.proj ** 0.5)             # (n,)

    def noul_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.noul(h)                             # (2,) in jul's option order: [true, false]

    def score_logits(self, h: torch.Tensor, n: int) -> torch.Tensor:
        return self.score(h)[:n]                         # (n,)


def _ordinal_target(n: int, gold: int, sigma: float, device, dtype):
    levels = torch.arange(n, device=device, dtype=torch.float32)
    w = torch.exp(-0.5 * ((levels - float(gold)) / sigma) ** 2)
    return (w / w.sum()).to(dtype)


def head_loss(head: ReadHead, h: torch.Tensor, ex, device, sigma: float = 1.0) -> torch.Tensor:
    """Cross-entropy (ordinal soft target for score) on the learned head, for one example.

    `ex` is a MaskExample (kind, anchor_ids, gold_index). `h` is the [MASK] hidden state (D,).
    """
    kind = ex.kind
    gold = ex.gold_index
    if kind == "choice":
        anchor_ids = torch.tensor(ex.anchor_ids, device=device)
        logits = head.choice_logits(h, anchor_ids)
        return F.cross_entropy(logits.unsqueeze(0), torch.tensor([gold], device=device))
    if kind == "noul":
        logits = head.noul_logits(h)                    # option order; gold_index is in option order too
        return F.cross_entropy(logits.unsqueeze(0), torch.tensor([gold], device=device))
    # score: ordinal soft target
    n = len(ex.anchor_ids)
    logits = head.score_logits(h, n)
    target = _ordinal_target(n, gold, sigma, device, logits.dtype)
    return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="sum")


def listwise_choice_loss(head: ReadHead, h_mask: torch.Tensor, h_ends: torch.Tensor,
                         gold_index: int, device) -> torch.Tensor:
    """Cross-entropy on the gold option for the listwise choice reading (wemm port).

    h_mask: (D,) hidden at the [MASK] (query). h_ends: (n, D) hidden at each option's closing
    separator (keys). The option logits are (k_i . q)/sqrt(proj) in prompt (jul option) order, so the
    gold index is used directly. Only `head.choice_q` and `head.choice_k` carry gradient here.
    """
    logits = head.listwise_choice_logits(h_mask, h_ends)
    return F.cross_entropy(logits.unsqueeze(0), torch.tensor([gold_index], device=device))


#: Keys of the backbone embedding that older read_head.pt files carried; never loaded into the head.
_LEGACY_EMB_PREFIX = "emb."


def load_head(path: str, emb: nn.Embedding, meta: dict) -> ReadHead:
    """A ReadHead from `read_head.pt` + its `read_head.json` meta, strictly (no silent key mismatch).

    Older files saved the backbone embedding under `emb.*`; those keys are dropped, never written back.
    """
    head = ReadHead(hidden=meta["hidden"], emb=emb, proj=meta["proj"],
                    max_levels=meta.get("max_levels", 16))
    state = torch.load(path, map_location="cpu")
    state = {k: v for k, v in state.items() if not k.startswith(_LEGACY_EMB_PREFIX)}
    head.load_state_dict(state, strict=True)
    return head.eval()
