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
  - noul   : a 2-logit linear head on h (yes/no).
  - score  : an ordinal head — K level logits on h, trained with the Gaussian ordinal target so the
             scale is treated as ordered, not as unordered classes.

The head is tiny and trained on top of a (LoRA-adapted or frozen) backbone. It is saved next to the
adapter so inference can load both.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReadHead(nn.Module):
    def __init__(self, hidden: int, emb: nn.Embedding, proj: int = 256, max_levels: int = 16):
        super().__init__()
        self.hidden = hidden
        self.emb = emb  # tied to the backbone input embedding (read-only key source for choice)
        self.q = nn.Linear(hidden, proj, bias=False)          # choice query
        self.k = nn.Linear(emb.embedding_dim, proj, bias=False)  # choice key from option anchor emb
        self.noul = nn.Linear(hidden, 2)                      # yes/no
        self.score = nn.Linear(hidden, max_levels)            # ordinal levels (sliced to n)
        self.proj = proj

    def choice_logits(self, h: torch.Tensor, anchor_ids: torch.Tensor) -> torch.Tensor:
        """h: (D,) mask hidden; anchor_ids: (n,) option anchor token ids. Returns (n,) logits."""
        q = self.q(h)                                   # (proj,)
        with torch.no_grad():
            e = self.emb(anchor_ids)                    # (n, emb_dim) — embedding table is frozen
        k = self.k(e)                                   # (n, proj)
        return (k @ q) / (self.proj ** 0.5)             # (n,)

    def noul_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.noul(h)                             # (2,) order [false, true]

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
        logits = head.noul_logits(h)                    # [false, true]
        return F.cross_entropy(logits.unsqueeze(0), torch.tensor([gold], device=device))
    # score: ordinal soft target
    n = len(ex.anchor_ids)
    logits = head.score_logits(h, n)
    target = _ordinal_target(n, gold, sigma, device, logits.dtype)
    return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="sum")
