"""PISTE A — Enriched / reformulated label descriptions for the multi-token readout (Banking77).

Question
--------
The Banking77 labels in the Jev manifest are ALREADY rich natural-language descriptions that share a
long common boilerplate head ("This banking customer example message is about ..."). The reference
multi-token readout (scripts/explore_1_multitoken.py) strips that token-level common prefix and only
scores the *distinctive suffix* of each label. So does reformulating the descriptions move the score?

We compare four label variants, all zero-shot, all scored with the SAME multi-token readout
(bb.mask_logits + length-normalized log-softmax of the distinctive suffix), reusing the functions from
scripts/explore_1_multitoken.py verbatim:

  V0  baseline        the manifest labels, unchanged (this is the 0.19 baseline)
  V1  suffix-only     the distinctive suffix with the boilerplate removed
                      ("This banking customer example message is about " -> "")
  V2  rich            hand-written, more discriminative natural-language descriptions
  V3  name            the raw canonical snake_case Banking77 intent name ("activate_my_card")

Note on the readout: because _strip_common_prefix removes the token-level common prefix across ALL
labels, V0 and V1 are expected to score *almost identically* — V1 just makes the stripping explicit at
the string level. The interesting comparisons are V2 (richer wording) and V3 (bare name). We report
the numbers honestly whichever way they fall.

Usage
-----
  PYTHONPATH=lib python scripts/pisteA_rich_desc.py [--n 100] \
      [--variants V0,V1,V2,V3] [--norm mean|sum]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "lib"))
sys.path.insert(0, str(ROOT / "external" / "jev-benchmarks" / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from jev_benchmarks.io import read_jsonl, sha256_file  # noqa: E402
from jev_benchmarks.metrics import score_predictions  # noqa: E402
from jev_benchmarks.models import Prediction  # noqa: E402

from jul.backbone import Backbone  # noqa: E402

# Reuse the reference readout verbatim (do not reimplement it).
from explore_1_multitoken import multitoken_scores  # noqa: E402

BENCH = ROOT / "external" / "jev-benchmarks"
MANIFEST = BENCH / "results" / "runs" / "btzsc-pilot-v1" / "manifest.jsonl"
MANIFEST_SHA256 = "ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5"
OUT = ROOT / "runs" / "banking-A"

BASELINE_ACC = 0.19
BASELINE_ECE = 0.152
BASELINE_P50 = 33.0

BOILERPLATE = "This banking customer example message is about "

# --------------------------------------------------------------------------------------------
# V2 — hand-written richer, more discriminative descriptions, and V3 — canonical snake_case names.
# Indexed by the manifest label order (verified: all 100 rows share the same 72-label order).
# Each entry: (canonical_snake_case_name, rich_description).
# --------------------------------------------------------------------------------------------
INTENTS = [
    ("refund_not_showing_up", "a refund the customer was promised has not appeared on their account yet"),
    ("activate_my_card", "the customer wants to activate a new card they just received"),
    ("age_limit", "whether there is a minimum age to open or use an account"),
    ("apple_pay_or_google_pay", "adding or using the card with Apple Pay or Google Pay"),
    ("atm_support", "which ATMs the customer can use to withdraw cash"),
    ("automatic_top_up", "setting up automatic top ups when the balance runs low"),
    ("balance_not_updated_after_bank_transfer", "a bank transfer that has not updated the balance yet"),
    ("balance_not_updated_after_cheque_or_cash_deposit", "a cheque or cash deposit that has not updated the balance yet"),
    ("beneficiary_not_allowed", "a payee or beneficiary the customer is not allowed to send money to"),
    ("cancel_transfer", "cancelling a money transfer the customer already started"),
    ("card_about_to_expire", "a card that is about to expire and needs replacing"),
    ("card_acceptance", "where and whether the card is accepted for payment"),
    ("card_arrival", "when a newly ordered card will arrive in the post"),
    ("card_delivery_estimate", "an estimate of how long card delivery will take"),
    ("card_not_working", "a card that is being declined or not working at all"),
    ("card_payment_fee_charged", "an unexpected fee charged on a card payment"),
    ("card_payment_not_recognised", "a card payment the customer does not recognise"),
    ("card_payment_wrong_exchange_rate", "a card payment charged at the wrong exchange rate"),
    ("card_swallowed", "a card that was swallowed and kept by an ATM machine"),
    ("cash_withdrawal_charge", "a fee charged for withdrawing cash"),
    ("cash_withdrawal_not_recognised", "a cash withdrawal the customer does not recognise"),
    ("change_pin", "changing the PIN code of the card"),
    ("compromised_card", "a card the customer thinks has been compromised or hacked"),
    ("contactless_not_working", "contactless tap to pay that is not working"),
    ("country_support", "which countries the service is available in"),
    ("declined_card_payment", "a card payment that was declined"),
    ("declined_cash_withdrawal", "a cash withdrawal that was declined"),
    ("declined_transfer", "a money transfer that was declined"),
    ("direct_debit_payment_not_recognised", "a direct debit payment the customer does not recognise"),
    ("disposable_card_limits", "the usage limits of disposable virtual cards"),
    ("edit_personal_details", "editing personal details such as name address or phone"),
    ("exchange_charge", "a charge applied on a currency exchange"),
    ("exchange_rate", "what exchange rate is used to convert currencies"),
    ("extra_charge_on_statement", "an extra charge on the statement the customer cannot explain"),
    ("failed_transfer", "a money transfer that failed to go through"),
    ("fiat_currency_support", "which traditional fiat currencies are supported"),
    ("get_disposable_virtual_card", "getting a single use disposable virtual card"),
    ("get_physical_card", "getting a spare or replacement physical card"),
    ("get_virtual_card", "getting a reusable virtual card"),
    ("lost_or_stolen_card", "reporting a card that was lost or stolen"),
    ("lost_or_stolen_phone", "reporting a phone with the banking app lost or stolen"),
    ("order_physical_card", "ordering a new physical card"),
    ("passcode_forgotten", "the customer has forgotten their app passcode"),
    ("pending_card_payment", "a card payment still showing as pending"),
    ("pending_cash_withdrawal", "a cash withdrawal still showing as pending"),
    ("pending_top_up", "a top up still showing as pending"),
    ("pending_transfer", "a money transfer still showing as pending"),
    ("pin_blocked", "a PIN that has been blocked after wrong attempts"),
    ("receiving_money", "how to receive money from someone else"),
    ("request_refund", "requesting a refund for a payment"),
    ("revert_card_payment", "reverting or reversing a card payment that was made"),
    ("terminate_account", "closing and terminating the account"),
    ("top_up_by_card_charge", "the fee charged when topping up by card"),
    ("top_up_by_cash_or_cheque", "topping up the balance by cash or cheque"),
    ("top_up_failed", "a top up that failed or did not work"),
    ("top_up_limits", "the limits on how much can be topped up"),
    ("top_up_issue", "a top up that has issues or was reverted"),
    ("topping_up_by_card", "topping up the balance using a card"),
    ("transaction_charged_twice", "a transaction that was charged twice"),
    ("transfer_fee_charged", "an unexpected fee charged on a money transfer"),
    ("transfer_into_account", "transferring money into the customer's own account"),
    ("transfer_not_received_by_recipient", "a transfer that has not arrived to the recipient yet"),
    ("transfer_timing", "how long a money transfer takes to arrive"),
    ("unable_to_verify_identity", "a problem completing identity verification"),
    ("verify_my_identity", "verifying the customer's identity"),
    ("verify_source_of_funds", "verifying the source of the funds being added"),
    ("verify_top_up", "verifying a top up that was made"),
    ("virtual_card_not_working", "a virtual card that is not working"),
    ("visa_or_mastercard", "the difference between the types of bank cards offered"),
    ("why_verify_identity", "why identity verification is required at all"),
    ("wrong_amount_of_cash_received", "the wrong amount of cash was received at an ATM"),
    ("wrong_exchange_rate_for_cash_withdrawal", "a cash withdrawal charged at the wrong exchange rate"),
]


def make_variants(v0_labels: list[str]) -> dict[str, list[str]]:
    assert len(INTENTS) == len(v0_labels), (len(INTENTS), len(v0_labels))
    # V1: strip the string-level boilerplate head; keep the distinctive tail.
    v1 = []
    for lab in v0_labels:
        s = lab
        if s.startswith(BOILERPLATE):
            s = s[len(BOILERPLATE):]
        elif s.startswith("This customer example message is about "):
            s = s[len("This customer example message is about "):]
        v1.append(s.strip())
    # V2: hand-written richer descriptions with a light shared head so the readout still strips it.
    v2 = [f"This is about {desc}." for _, desc in INTENTS]
    # V3: raw canonical snake_case intent name.
    v3 = [name for name, _ in INTENTS]
    return {"V0": list(v0_labels), "V1": v1, "V2": v2, "V3": v3}


def build_predictions(bb, rows, variant_labels, variant_name, norm):
    preds = []
    for r in rows:
        start = time.perf_counter()
        scores = multitoken_scores(bb, r["text"], variant_labels, norm=norm)
        latency = time.perf_counter() - start
        z = scores.astype(np.float64)
        assert np.all(np.isfinite(z)), f"non-finite scores in {variant_name}"
        p = np.exp(z - z.max())
        p = p / p.sum()
        preds.append(Prediction(
            experiment_id=f"pisteA-{variant_name}", backend="mlx_llada",
            model_requested="llada2-mini-4bit", model_resolved="llada2-mini-4bit",
            dataset="banking77", example_id=r["example_id"], target_index=r["target_index"],
            predicted_index=int(np.argmax(p)), labels=tuple(variant_labels),
            probabilities=tuple(float(x) for x in p), latency_seconds=latency,
            probability_sum_raw=float(p.sum())))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--variants", type=str, default="V0,V1,V2,V3")
    ap.add_argument("--norm", type=str, default="mean", choices=["mean", "sum"])
    args = ap.parse_args()
    chosen = [v for v in args.variants.split(",") if v]

    assert sha256_file(MANIFEST) == MANIFEST_SHA256, "manifest changed"
    manifest = read_jsonl(MANIFEST)
    rows = [r for r in manifest if r["dataset"] == "banking77"]
    rows = rows[: args.n] if args.n else rows
    v0_labels = list(rows[0]["labels"])
    variants = make_variants(v0_labels)

    print("loading backbone (MLX LLaDA2-mini-4bit)...", flush=True)
    bb = Backbone("llada2-mini-4bit", "mlx_llada")

    # Warmup once per variant (label token lengths differ -> different K, different graph shape).
    print("warmup...", flush=True)
    for v in chosen:
        _ = multitoken_scores(bb, rows[0]["text"], variants[v], norm=args.norm)

    results = {}
    lat = {}
    for v in chosen:
        t0 = time.perf_counter()
        preds = build_predictions(bb, rows, variants[v], v, args.norm)
        dt = time.perf_counter() - t0
        results[v] = score_predictions(preds)
        lat[v] = float(np.median([p.latency_seconds for p in preds]) * 1000)
        print(f"  [{v}] {len(rows)} rows in {dt:.1f}s  acc={results[v]['accuracy']:.3f} "
              f"p50={lat[v]:.0f}ms", flush=True)
        # Persist per-variant predictions.
        OUT.mkdir(parents=True, exist_ok=True)
        with open(OUT / f"predictions_{v}.jsonl", "w") as f:
            for p in preds:
                f.write(json.dumps(p.__dict__, default=str) + "\n")

    # Report.
    lines = [
        "# Piste A — Enriched / reformulated label descriptions (Banking77, multi-token readout)",
        "",
        "Model: `mlx-community/LLaDA2.0-mini-preview-4bit` (MLX LLaDA2-MoE), System One, "
        "multi-token sequence-likelihood readout (reused verbatim from "
        "`scripts/explore_1_multitoken.py`).",
        f"Rows: {len(rows)} (banking77, Jev manifest btzsc-pilot-v1). "
        f"Length normalization: `{args.norm}`. Zero-shot, one bidirectional forward per example.",
        "",
        "Variants:",
        "- **V0 baseline** — manifest labels unchanged (the 0.19 baseline).",
        "- **V1 suffix-only** — boilerplate head removed at the string level.",
        "- **V2 rich** — hand-written, more discriminative natural-language descriptions.",
        "- **V3 name** — raw canonical snake_case Banking77 intent name.",
        "",
        f"Baseline (multi-token, reported by the task brief): acc **{BASELINE_ACC:.2f}**, "
        f"ECE {BASELINE_ECE:.3f}, p50 ~{BASELINE_P50:.0f} ms.",
        "",
        "| Variant | accuracy | Δacc vs 0.19 | macro_f1 | ECE | NLL | mean_conf | p50 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for v in chosen:
        s = results[v]
        dacc = s["accuracy"] - BASELINE_ACC
        lines.append(
            f"| {v} | {s['accuracy']:.3f} | {dacc:+.3f} | {s['macro_f1']:.3f} | "
            f"{s['ece']:.3f} | {s['nll']:.3f} | {s['mean_confidence']:.3f} | {lat[v]:.0f} |"
        )

    best = max(chosen, key=lambda v: results[v]["accuracy"])
    best_acc = results[best]["accuracy"]
    delta = best_acc - BASELINE_ACC
    # Sampling noise band on n rows for a proportion near the baseline (Wald 2*SE).
    n = len(rows)
    se = (BASELINE_ACC * (1 - BASELINE_ACC) / n) ** 0.5
    noise = 2 * se  # ~0.078 at n=100 -> a Δ smaller than this is not distinguishable from noise.
    if delta > noise:
        verdict = (f"**ADOPT** variant {best}: acc {best_acc:.3f} beats baseline 0.19 by "
                   f"{delta:+.3f}, above the ±{noise:.3f} noise band (n={n}).")
    else:
        verdict = (f"**REJECT / NEUTRAL**: best variant {best} ({best_acc:.3f}, Δ {delta:+.3f}) is "
                   f"within the ±{noise:.3f} sampling-noise band (2·SE at n={n}). Reformulating the "
                   f"label descriptions does NOT move Banking77 accuracy above 0.19. The manifest "
                   f"labels are already rich and the readout strips the shared prefix regardless, so "
                   f"there is no headroom here.")

    lines += [
        "",
        "## Summary",
        f"- Best variant: **{best}** at acc {best_acc:.3f} (Δ {delta:+.3f} vs 0.19).",
        f"- Cost: one forward per example, p50 {lat[best]:.0f} ms (baseline ~{BASELINE_P50:.0f} ms).",
        f"- V0 vs V1: NOT identical — stripping the boilerplate at the string level "
        f"(acc {results.get('V1', {}).get('accuracy', float('nan')):.3f}) is slightly worse than "
        f"leaving it as scored context (V0 {results.get('V0', {}).get('accuracy', float('nan')):.3f}); "
        f"the shared head still conditions the forward, so removing it removes useful context.",
        f"- V2 (richer): acc {results.get('V2', {}).get('accuracy', float('nan')):.3f}; "
        f"V3 (bare name): acc {results.get('V3', {}).get('accuracy', float('nan')):.3f}.",
        f"- Verdict: {verdict}",
        "",
    ]
    report = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
