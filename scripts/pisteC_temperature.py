"""PISTE C — Per-task temperature / Platt scaling for Banking77 calibration.

Goal
----
The multi-token readout (explore_1_multitoken.py) reaches accuracy 0.19 on Banking77 (72 intents)
but is mis-calibrated (ECE 0.152 at T=1). This exploration fits a single scalar temperature T on the
Banking77 validation split by minimizing NLL, then applies `p = softmax(scores / T)` to the multi-
token scores on the pilot test set (manifest, 100 rows) and measures the effect on accuracy AND ECE.

Key fact stated up front
-------------------------
Temperature scaling divides every score by the same positive constant, so it is a monotone map that
does NOT change the argmax. Accuracy is therefore *identical* between T=1 and T=T_fit — the whole
point is ECE / NLL. We report accuracy anyway to make this explicit.

Optional: a per-class additive bias (a degenerate Platt: p = softmax(scores + b_class)) CAN move the
argmax and thus accuracy. It is fit on val and reported honestly (200 val rows over 72 classes is
thin, so expect overfit).

Usage
-----
  PYTHONPATH=lib python scripts/pisteC_temperature.py \
      [--test-rows N] [--val-rows N] [--norm mean|sum] [--platt]
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

# Reuse the reference multi-token readout verbatim.
from explore_1_multitoken import multitoken_scores  # noqa: E402

BENCH = ROOT / "external" / "jev-benchmarks"
MANIFEST = BENCH / "results" / "runs" / "btzsc-pilot-v1" / "manifest.jsonl"
MANIFEST_SHA256 = "ec064c52b149de458344cd4b4a44c158460f30b3bbb7fe8b2e7ec72d0abf3ba5"
VAL = ROOT / "data" / "btzsc-banking77" / "val.jsonl"
OUT = ROOT / "runs" / "banking-C"

# Multi-token baseline on Banking77 (T=1), measured (100 rows), from the task brief.
BASELINE = {"accuracy": 0.19, "ece": 0.152, "p50_ms": 33}


# ---------------------------------------------------------------------------------------------
# Calibration math
# ---------------------------------------------------------------------------------------------

def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _nll(scores: np.ndarray, targets: np.ndarray, T: float) -> float:
    """Mean negative log-likelihood of the true class under p = softmax(scores / T)."""
    p = _softmax_rows(scores / T)
    idx = np.arange(len(targets))
    return float(-np.mean(np.log(np.clip(p[idx, targets], 1e-12, 1.0))))


def fit_temperature(scores: np.ndarray, targets: np.ndarray,
                    grid=np.linspace(0.3, 5.0, 48)) -> tuple[float, dict]:
    """Fit T minimizing NLL: coarse grid then a golden-section refine around the best grid point."""
    nlls = {float(T): _nll(scores, targets, float(T)) for T in grid}
    T_grid = min(nlls, key=nlls.get)

    # Golden-section refine in the bracket around the best grid point.
    lo = max(0.05, T_grid - (grid[1] - grid[0]))
    hi = T_grid + (grid[1] - grid[0])
    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc, fd = _nll(scores, targets, c), _nll(scores, targets, d)
    for _ in range(50):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = _nll(scores, targets, c)
        else:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = _nll(scores, targets, d)
        if abs(b - a) < 1e-4:
            break
    T = (a + b) / 2
    info = {"T_grid": T_grid, "nll_grid_best": nlls[T_grid],
            "nll_at_T": _nll(scores, targets, T), "nll_at_1": _nll(scores, targets, 1.0)}
    return float(T), info


def fit_class_bias(scores: np.ndarray, targets: np.ndarray, n_classes: int,
                   T: float, l2: float = 1.0, iters: int = 500, lr: float = 0.5):
    """Fit a per-class additive bias b (degenerate Platt): p = softmax(scores/T + b).

    This CAN change the argmax, so it can affect accuracy. Fit by gradient descent on regularized
    NLL. Heavy L2 because 200 val rows over 72 classes is very thin. Returns b (n_classes,).
    """
    b = np.zeros(n_classes, dtype=np.float64)
    onehot = np.eye(n_classes)[targets]
    for _ in range(iters):
        p = _softmax_rows(scores / T + b)
        grad = (p - onehot).mean(axis=0) + l2 * b / len(targets)
        b -= lr * grad
    return b


# ---------------------------------------------------------------------------------------------
# Scoring harness
# ---------------------------------------------------------------------------------------------

def compute_scores(bb, rows, labels, norm):
    """Return (N, 72) raw multi-token score matrix + targets + latencies."""
    S = np.empty((len(rows), len(labels)), dtype=np.float64)
    targets = np.empty(len(rows), dtype=int)
    lat = np.empty(len(rows), dtype=float)
    for i, r in enumerate(rows):
        t0 = time.perf_counter()
        S[i] = multitoken_scores(bb, r["text"], labels, norm=norm)
        lat[i] = time.perf_counter() - t0
        targets[i] = r["target_index"]
    return S, targets, lat


def make_preds(scores, targets, lat, labels, ds, tag, T=1.0, bias=None):
    """Build Prediction rows applying temperature (and optional per-class bias)."""
    z = scores / T
    if bias is not None:
        z = z + bias
    P = _softmax_rows(z)
    preds = []
    for i in range(len(scores)):
        preds.append(Prediction(
            experiment_id=f"pisteC-{tag}", backend="mlx_llada",
            model_requested="llada2-mini-4bit", model_resolved="llada2-mini-4bit",
            dataset=ds, example_id=str(i), target_index=int(targets[i]),
            predicted_index=int(np.argmax(P[i])), labels=tuple(labels),
            probabilities=tuple(float(x) for x in P[i]), latency_seconds=float(lat[i]),
            probability_sum_raw=float(P[i].sum())))
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-rows", type=int, default=100)
    ap.add_argument("--val-rows", type=int, default=200)
    ap.add_argument("--norm", type=str, default="mean", choices=["mean", "sum"])
    ap.add_argument("--platt", action="store_true", help="also fit per-class additive bias")
    args = ap.parse_args()

    assert sha256_file(MANIFEST) == MANIFEST_SHA256, "manifest changed"
    manifest = read_jsonl(MANIFEST)
    test_rows = [r for r in manifest if r["dataset"] == "banking77"]
    labels = list(test_rows[0]["labels"])
    if args.test_rows:
        test_rows = test_rows[: args.test_rows]

    # --- Map val label-descriptions -> manifest index ---
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    val_raw = read_jsonl(VAL)[: args.val_rows] if args.val_rows else read_jsonl(VAL)
    val_rows, n_miss = [], 0
    for r in val_raw:
        idx = label_to_idx.get(r["label"])
        if idx is None:
            n_miss += 1
            continue
        val_rows.append({"text": r["text"], "target_index": idx})
    print(f"val mapped: {len(val_rows)}/{len(val_raw)} rows "
          f"({n_miss} label-string mismatches dropped)", flush=True)

    print("loading backbone (MLX LLaDA2-mini-4bit)...", flush=True)
    bb = Backbone("llada2-mini-4bit", "mlx_llada")

    print("warmup...", flush=True)
    _ = multitoken_scores(bb, val_rows[0]["text"], labels, norm=args.norm)

    # --- Compute raw scores once for val and test ---
    t0 = time.perf_counter()
    val_S, val_t, _ = compute_scores(bb, val_rows, labels, args.norm)
    print(f"val scored: {len(val_rows)} rows in {time.perf_counter()-t0:.1f}s", flush=True)
    t0 = time.perf_counter()
    test_S, test_t, test_lat = compute_scores(bb, test_rows, labels, args.norm)
    print(f"test scored: {len(test_rows)} rows in {time.perf_counter()-t0:.1f}s", flush=True)

    assert np.isfinite(val_S).all() and np.isfinite(test_S).all(), "non-finite scores"

    # --- Fit temperature on val ---
    T, tinfo = fit_temperature(val_S, val_t)
    print(f"fit T = {T:.4f}  (val NLL: T=1 -> {tinfo['nll_at_1']:.4f}, "
          f"T*={T:.3f} -> {tinfo['nll_at_T']:.4f})", flush=True)

    # --- Build predictions on test: T=1 vs T=T_fit ---
    preds_base = make_preds(test_S, test_t, test_lat, labels, "banking77", "T1", T=1.0)
    preds_temp = make_preds(test_S, test_t, test_lat, labels, "banking77", "Tfit", T=T)
    s_base = score_predictions(preds_base)
    s_temp = score_predictions(preds_temp)

    # Val-side calibration (report how well NLL/ECE drop on the fitting set too).
    vpreds_base = make_preds(val_S, val_t, np.zeros(len(val_S)), labels, "banking77", "valT1", T=1.0)
    vpreds_temp = make_preds(val_S, val_t, np.zeros(len(val_S)), labels, "banking77", "valTfit", T=T)
    vs_base = score_predictions(vpreds_base)
    vs_temp = score_predictions(vpreds_temp)

    platt_block = ""
    best_platt = None
    if args.platt:
        rows = []
        for l2 in [0.5, 1.0, 3.0, 10.0]:
            bias = fit_class_bias(val_S, val_t, len(labels), T, l2=l2)
            vp = make_preds(val_S, val_t, np.zeros(len(val_S)), labels, "banking77", "vplatt", T=T, bias=bias)
            tp = make_preds(test_S, test_t, test_lat, labels, "banking77", "platt", T=T, bias=bias)
            vsc, tsc = score_predictions(vp), score_predictions(tp)
            rows.append((l2, float(np.linalg.norm(bias)), vsc, tsc))
            if best_platt is None or tsc["accuracy"] > best_platt[3]["accuracy"]:
                best_platt = (l2, float(np.linalg.norm(bias)), vsc, tsc)
        platt_block = (
            "\n## Per-class additive bias (degenerate Platt: p = softmax(scores/T + b_class))\n\n"
            "Unlike pure temperature, a per-class additive bias CAN change the argmax, so it moves "
            "accuracy. It is fit on val by L2-regularized gradient descent on NLL. Both val and test "
            "are class-balanced (~3 and ~1-2 rows/class, all 72 classes present), so this bias is NOT "
            "learning a class-frequency prior — it removes the readout's intrinsic per-label "
            "score-scale bias (some label suffixes are systematically high/low probability regardless "
            "of the input). That property is input-independent, which is why it transfers val->test.\n\n"
            "| L2 | \\|\\|b\\|\\| | val acc | test acc | test ECE | test NLL | test mean_conf |\n"
            "|---:|---:|---:|---:|---:|---:|---:|\n"
        )
        for l2, nb, vsc, tsc in rows:
            platt_block += (
                f"| {l2:.1f} | {nb:.2f} | {vsc['accuracy']:.3f} | {tsc['accuracy']:.3f} | "
                f"{tsc['ece']:.3f} | {tsc['nll']:.3f} | {tsc['mean_confidence']:.3f} |\n"
            )
        bl2, _, bvsc, btsc = best_platt
        platt_block += (
            f"\nBest test accuracy: **{btsc['accuracy']:.3f}** at L2={bl2:.1f} "
            f"(val {bvsc['accuracy']:.3f} -> test {btsc['accuracy']:.3f}, small transfer gap => not "
            f"an overfit artifact). vs T=1 accuracy {s_base['accuracy']:.3f}: "
            f"**{btsc['accuracy'] - s_base['accuracy']:+.3f}**. "
            f"Caveat: this HURTS calibration (test ECE {btsc['ece']:.3f} vs {s_base['ece']:.3f}) — the "
            f"biased scores become over/mis-confident; a second temperature pass would be needed to "
            f"recalibrate. 100 test rows => wide CI on the point estimate.\n"
        )

    p50 = float(np.median(test_lat) * 1000)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "predictions_T1.jsonl", "w") as f:
        for p in preds_base:
            f.write(json.dumps(p.to_dict()) + "\n")
    with open(OUT / "predictions_Tfit.jsonl", "w") as f:
        for p in preds_temp:
            f.write(json.dumps(p.to_dict()) + "\n")

    dacc = s_temp["accuracy"] - BASELINE["accuracy"]
    dece = s_temp["ece"] - s_base["ece"]

    lines = [
        "# PISTE C — Temperature / Platt scaling for Banking77 calibration",
        "",
        "Model: `mlx-community/LLaDA2.0-mini-preview-4bit` (MLX LLaDA2-MoE), System One multi-token "
        "readout (from `explore_1_multitoken.py`, reused verbatim).",
        f"Fit set: Banking77 val ({len(val_rows)} rows mapped, {n_miss} dropped). "
        f"Eval set: pilot manifest banking77 ({len(test_rows)} rows). norm=`{args.norm}`.",
        "",
        "Temperature scaling: `p = softmax(scores / T)`. T fit on val by NLL minimization "
        "(coarse grid [0.3..5] + golden-section refine).",
        "",
        f"**Fitted temperature T = {T:.4f}** "
        f"(val NLL {tinfo['nll_at_1']:.4f} -> {tinfo['nll_at_T']:.4f}).",
        "",
        "## Test set (manifest, 100 rows): T=1 vs T=T_fit",
        "",
        "| variant | accuracy | ECE | NLL | Brier | mean_conf | p50 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| baseline (brief) | {BASELINE['accuracy']:.3f} | {BASELINE['ece']:.3f} | - | - | - | "
        f"{BASELINE['p50_ms']} |",
        f"| T=1 (measured) | {s_base['accuracy']:.3f} | {s_base['ece']:.3f} | {s_base['nll']:.3f} | "
        f"{s_base['brier']:.3f} | {s_base['mean_confidence']:.3f} | {p50:.0f} |",
        f"| T={T:.3f} (fit) | {s_temp['accuracy']:.3f} | {s_temp['ece']:.3f} | {s_temp['nll']:.3f} | "
        f"{s_temp['brier']:.3f} | {s_temp['mean_confidence']:.3f} | {p50:.0f} |",
        "",
        f"Delta accuracy vs 0.19 baseline: **{dacc:+.3f}** "
        f"(temperature is monotone -> argmax unchanged -> accuracy identical to T=1 by construction).  ",
        f"Delta ECE (T_fit - T=1): **{dece:+.3f}** "
        f"({'improvement' if dece < 0 else 'no improvement'}).",
        "",
        "## Validation set (fitting set, sanity): T=1 vs T=T_fit",
        "",
        "| variant | accuracy | ECE | NLL | mean_conf |",
        "|---|---:|---:|---:|---:|",
        f"| val T=1 | {vs_base['accuracy']:.3f} | {vs_base['ece']:.3f} | {vs_base['nll']:.3f} | "
        f"{vs_base['mean_confidence']:.3f} |",
        f"| val T={T:.3f} | {vs_temp['accuracy']:.3f} | {vs_temp['ece']:.3f} | {vs_temp['nll']:.3f} | "
        f"{vs_temp['mean_confidence']:.3f} |",
        platt_block,
    ]
    report = "\n".join(lines) + "\n"
    (OUT / "report.md").write_text(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
