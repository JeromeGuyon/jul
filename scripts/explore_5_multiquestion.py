"""EXPLORATION 5 — Multi-question in a single bidirectional pass.

A real System One asks SEVERAL typed questions about the SAME state. Today each question is one
forward (sequential reading). In a bidirectional diffusion model we can place ONE [MASK] per
question inside ONE sequence and read them all in a single forward:

    <state>
    Q1: <instr1>
    A) opt B) opt ...
    Answer: [MASK]
    Q2: <instr2>
    ...
    Answer: [MASK]
    ...

This script builds both readouts on the SAME states/questions and compares:
  (a) decision agreement  : do the argmax decisions match between one-pass and sequential?
  (b) probability agreement: how close are the softmax distributions (mean total-variation dist)?
  (c) latency             : total wall time sequential vs one-pass, and the speedup factor.

We only READ the existing backbone (bb.mask_logits over all mask positions, then slice each
question's anchors). We do NOT modify mlx_llada.py / mask.py / backbone.py / presets.py.

Usage:
  PYTHONPATH=lib python scripts/explore_5_multiquestion.py [--n-states 30]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lib"))

from jul.backbone import Backbone  # noqa: E402
from jul.types import Option  # noqa: E402

OUT = ROOT / "runs" / "explore-5"


# --- typed questions used everywhere ------------------------------------------------------------
# Each question: (name, instructions, [Option(key, text), ...]).  Anchors follow the MaskReader
# convention: markers are letters A/B/C..., anchor = first token of " <marker>".

QUESTIONS = [
    ("team", "Route this support message to the right team.",
     [Option("billing", "billing, payments, invoices, refunds"),
      Option("technical", "technical problems, bugs, errors, outages"),
      Option("sales", "sales, pricing, plans, upgrades, new purchases")]),
    ("is_bug", "Is the customer reporting a software bug or defect?",
     [Option("oui", "yes, it is a bug or defect"),
      Option("non", "no, it is not a bug")]),
    ("urgency", "How urgent is this message?",
     [Option("low", "low, can wait"),
      Option("med", "medium, should be handled soon"),
      Option("high", "high, needs immediate attention")]),
]

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def make_states(n: int) -> list[str]:
    """Synthetic support states covering the three questions' answer space."""
    templates = [
        "I was charged twice for my subscription this month and need a refund urgently.",
        "The app crashes every time I try to open the reports page. This is blocking my whole team.",
        "Hi, I'd like to know the pricing for the enterprise plan and whether we can get a demo.",
        "My invoice shows the wrong amount, can someone check it when you have time?",
        "Login button does nothing on Safari, throws a 500 error in the console.",
        "We're considering upgrading to the pro tier next quarter, no rush.",
        "Payment failed three times and now my account is locked. I need this fixed now.",
        "There's a typo on the dashboard, minor thing but wanted to flag it.",
        "Can you send me a quote for 50 seats? Planning our budget for next year.",
        "The export feature returns an empty file since the last update. Urgent for a client meeting.",
        "I don't understand this charge on my card statement labelled 'SVC-99'.",
        "The mobile app freezes on the checkout screen, customers can't pay.",
        "Interested in your annual plan discounts, when is the best time to buy?",
        "Refund request: I cancelled last week but was billed again today.",
        "API returns 403 on all endpoints since this morning, production is down.",
        "Just browsing your plans, might upgrade someday, no urgency at all.",
        "My credit card was declined but you still suspended my account.",
        "Dark mode toggle is broken on the settings page, low priority.",
        "We want to migrate 200 users to your platform, need pricing and onboarding help.",
        "Double billing again this quarter, please refund the extra charge as soon as possible.",
        "The search bar crashes the tab on Chrome, happens intermittently.",
        "Could you clarify what the premium add-on costs per month?",
        "Invoice #4521 was never sent to me, can you resend it sometime?",
        "Server keeps dropping our webhooks, we're losing orders right now.",
        "Thinking about the starter plan, what's included? No rush on the reply.",
        "Unexpected overage fee on my bill, I'd like an explanation and a credit.",
        "The PDF generator throws an error 'null pointer' on every invoice.",
        "Do you offer volume discounts? Evaluating for a possible large purchase.",
        "I was billed after cancelling; urgent because it overdrew my account.",
        "Notifications stopped working across the whole app after the update.",
    ]
    if n <= len(templates):
        return templates[:n]
    # repeat with light variation if more are asked
    out = list(templates)
    i = 0
    while len(out) < n:
        out.append(templates[i % len(templates)] + f" (case #{len(out)})")
        i += 1
    return out


# --- anchor resolution (same rule as jul.mask.MaskReader._anchor_id) ----------------------------

def anchor_id(tok, marker: str) -> int:
    for form in (" " + marker, marker):
        ids = tok.encode(form, add_special_tokens=False)
        if ids:
            return ids[0]
    raise ValueError(f"marker {marker!r} does not tokenize")


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


# --- prompt building ----------------------------------------------------------------------------

def question_block(tok, idx: int, instr: str, options: list[Option]) -> tuple[list[int], list[int]]:
    """Token ids for one question block, and the marker anchors. No [MASK] here; caller appends it."""
    markers = [LETTERS[i] for i in range(len(options))]
    listing = "\n".join(f"{m}) {o.text}" for m, o in zip(markers, options))
    head = f"Q{idx + 1}: {instr}\n{listing}\nAnswer: "
    head_ids = tok.encode(head, add_special_tokens=False)
    anchors = [anchor_id(tok, m) for m in markers]
    return head_ids, anchors


def build_sequential(tok, mask_id, state: str, questions):
    """One (tokens, mask_pos, anchors) per question — each read in its own forward."""
    state_ids = tok.encode(state, add_special_tokens=False)
    sep = tok.encode("\n", add_special_tokens=False)
    seqs = []
    for i, (_name, instr, options) in enumerate(questions):
        head_ids, anchors = question_block(tok, i, instr, options)
        base = state_ids + sep + head_ids
        tokens = base + [mask_id]
        seqs.append((tokens, [len(base)], anchors))
    return seqs


def build_onepass(tok, mask_id, state: str, questions):
    """ONE sequence: state, then all question blocks each ending in a [MASK]. Returns
    (tokens, [mask_pos per question], [anchors per question])."""
    state_ids = tok.encode(state, add_special_tokens=False)
    sep = tok.encode("\n", add_special_tokens=False)
    tokens = list(state_ids)
    mask_positions = []
    anchors_per_q = []
    for i, (_name, instr, options) in enumerate(questions):
        head_ids, anchors = question_block(tok, i, instr, options)
        tokens += sep + head_ids
        mask_positions.append(len(tokens))
        tokens.append(mask_id)
        anchors_per_q.append(anchors)
    return tokens, mask_positions, anchors_per_q


# --- readouts -----------------------------------------------------------------------------------

def read_sequential(bb, state: str, questions):
    """One forward per question. Returns (list of prob arrays, total_forward_tokens, wall_seconds)."""
    seqs = build_sequential(bb.tokenizer, bb.mask_id, state, questions)
    probs = []
    tokens_used = 0
    t0 = time.perf_counter()
    for tokens, mask_pos, anchors in seqs:
        z = bb.option_logits(tokens, mask_pos, anchors)[0]  # (A,)
        probs.append(softmax(z))
        tokens_used += len(tokens)
    dt = time.perf_counter() - t0
    return probs, tokens_used, dt


def read_onepass(bb, state: str, questions):
    """One forward for all questions. Returns (list of prob arrays, total_forward_tokens, wall_seconds)."""
    tokens, mask_positions, anchors_per_q = build_onepass(bb.tokenizer, bb.mask_id, state, questions)
    t0 = time.perf_counter()
    # Distinct questions may have distinct anchor sets, so read the full vocab at the masks once,
    # then slice each question's anchors. This is still a SINGLE forward.
    vocab = bb.mask_logits(tokens, mask_positions)  # (Q, V)
    dt = time.perf_counter() - t0
    probs = []
    for row, anchors in zip(vocab, anchors_per_q):
        z = np.array([row[a] for a in anchors], dtype=np.float32)
        probs.append(softmax(z))
    return probs, len(tokens), dt


def tv_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Total variation distance between two distributions of the same length."""
    return 0.5 * float(np.abs(p - q).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=30)
    args = ap.parse_args()

    print("loading backbone (MLX LLaDA2-mini-4bit)...", flush=True)
    bb = Backbone("llada2-mini-4bit", "mlx_llada")
    print(f"mask_id={bb.mask_id}", flush=True)

    states = make_states(args.n_states)

    # warmup (both paths), not measured
    print("warmup...", flush=True)
    _ = read_sequential(bb, states[0], QUESTIONS)
    _ = read_onepass(bb, states[0], QUESTIONS)

    q_names = [q[0] for q in QUESTIONS]
    n_q = len(QUESTIONS)

    seq_lat, one_lat = [], []
    seq_tokens_total, one_tokens_total = 0, 0
    # per-question agreement and TV distance
    agree = np.zeros(n_q, dtype=np.int64)
    tvs = [[] for _ in range(n_q)]
    finite_ok = True
    n = len(states)

    for si, state in enumerate(states):
        seq_probs, seq_tok, seq_dt = read_sequential(bb, state, QUESTIONS)
        one_probs, one_tok, one_dt = read_onepass(bb, state, QUESTIONS)
        seq_lat.append(seq_dt)
        one_lat.append(one_dt)
        seq_tokens_total += seq_tok
        one_tokens_total += one_tok
        for qi in range(n_q):
            sp, op = seq_probs[qi], one_probs[qi]
            if not (np.isfinite(sp).all() and np.isfinite(op).all()):
                finite_ok = False
            if int(np.argmax(sp)) == int(np.argmax(op)):
                agree[qi] += 1
            tvs[qi].append(tv_distance(sp, op))
        if (si + 1) % 10 == 0:
            print(f"  {si + 1}/{n} states done", flush=True)

    seq_p50 = np.median(seq_lat) * 1000
    one_p50 = np.median(one_lat) * 1000
    seq_mean = np.mean(seq_lat) * 1000
    one_mean = np.mean(one_lat) * 1000
    speedup_p50 = seq_p50 / one_p50 if one_p50 else float("nan")
    speedup_mean = seq_mean / one_mean if one_mean else float("nan")

    overall_agree = agree.sum() / (n * n_q)
    overall_tv = np.mean([np.mean(tvs[qi]) for qi in range(n_q)])

    OUT.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Exploration 5 — Multi-question in a single bidirectional pass",
        "",
        f"States: {n}. Questions per state: {n_q} ({', '.join(q_names)}).",
        "Model: mlx-community/LLaDA2.0-mini-preview-4bit (MLX diffusion, bidirectional).",
        "",
        "Two readouts on the SAME states/questions:",
        "- **sequential**: one forward per question (today's System One), read with `option_logits` "
        "(anchor-only projection).",
        "- **one-pass**: ONE sequence with one `[MASK]` per question, a SINGLE `mask_logits` forward, "
        "each question sliced at its own anchors.",
        "",
        f"Finite logits everywhere: **{finite_ok}**.",
        "",
        "## Latency",
        "",
        "| Mode | p50 (ms) | mean (ms) | total forward tokens |",
        "|---|---:|---:|---:|",
        f"| sequential ({n_q} forwards/state) | {seq_p50:.1f} | {seq_mean:.1f} | {seq_tokens_total} |",
        f"| one-pass (1 forward/state) | {one_p50:.1f} | {one_mean:.1f} | {one_tokens_total} |",
        "",
        f"**Latency speedup (one-pass vs sequential): {speedup_p50:.2f}x on p50, "
        f"{speedup_mean:.2f}x on mean.**",
        f"Forward-token reduction: {seq_tokens_total / one_tokens_total:.2f}x "
        f"(sequential re-encodes the state {n_q} times; one-pass encodes it once).",
        "",
        "## Agreement one-pass vs sequential",
        "",
        "| Question | argmax agreement | mean TV distance (probs) |",
        "|---|---:|---:|",
    ]
    for qi, name in enumerate(q_names):
        lines.append(f"| {name} | {agree[qi] / n:.3f} ({agree[qi]}/{n}) | {np.mean(tvs[qi]):.4f} |")
    lines.append(f"| **overall** | **{overall_agree:.3f}** | **{overall_tv:.4f}** |")
    lines += [
        "",
        "## Reading",
        "",
        "- Agreement close to 1.0 means the shared-context one-pass reaches the SAME decisions as "
        "reading each question in isolation.",
        "- TV distance measures how much the probability distributions drift (0 = identical, "
        "1 = disjoint).",
        "- In one-pass, each `[MASK]` attends to the state AND to the other questions' text (their "
        "masks are still masked). This shared context can help (consistent decisions), harm "
        "(cross-talk between questions), or be neutral.",
        "",
        _verdict(overall_agree, overall_tv, speedup_p50),
        "",
    ]
    report = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(report)
    print("\n" + report)


def _verdict(agree: float, tv: float, speedup: float) -> str:
    if agree >= 0.95 and tv <= 0.05:
        effect = "neutre (le partage de contexte ne change pas les décisions)"
        rec = "ADOPTER" if speedup > 1.2 else "à creuser (gain latence faible)"
    elif agree >= 0.9:
        effect = "quasi-neutre (léger cross-talk entre questions)"
        rec = "à creuser" if speedup > 1.2 else "rejeter (peu de gain, risque de dérive)"
    else:
        effect = "le partage de contexte NUIT (les décisions divergent du mode isolé)"
        rec = "rejeter"
    return (f"## Verdict\n\nEffet du partage de contexte : {effect}. "
            f"Gain latence one-pass ~{speedup:.2f}x. Recommandation : **{rec}**.")


if __name__ == "__main__":
    main()
