#!/usr/bin/env python3
"""
Score bin probability forecasts and compare to quantile baseline.

Computes:
1. CRPS from binned probabilities (piecewise-uniform CDF)
2. Binary Brier at each bin threshold (ForecastBench bridge)
3. Calibration diagnostic (observed vs expected bin frequencies)
4. ECI × CRPS correlation by horizon (the key anti-g test)

Usage:
    cd /Users/elsehow/Projects/civbench
    uv run python scripts/score_bin_condition.py
    uv run python scripts/score_bin_condition.py --compare data/results/trajectory_condition/trajectory_results.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# ECI scores for the 5 intervention models
ECI_MAP = {
    "openai/gpt-4.1-2025-04-14": 137,
    "anthropic/claude-sonnet-4-5-20250929": 148,
    "openai/gpt-5-2025-08-07": 150,
    "google/gemini-3-pro-preview": 154,
    "anthropic/claude-opus-4-6": 157,
}


# ── CRPS from binned probabilities ─────────────────────────────────────────

def bins_to_cdf(bin_probs: list[float], bin_edges: list[float]
                ) -> tuple[np.ndarray, np.ndarray]:
    """Convert bin probabilities to a piecewise-uniform CDF.

    Assumes probability mass is uniformly distributed within each bin.

    Args:
        bin_probs: List of 10 probability values (should sum to ~100).
        bin_edges: List of 11 bin edge values.

    Returns:
        (x_points, cdf_values): Arrays defining the piecewise-linear CDF.
        x_points includes all bin edges; cdf_values gives the cumulative
        probability at each edge.
    """
    probs = np.array(bin_probs) / 100.0  # Normalize to [0, 1]
    edges = np.array(bin_edges, dtype=float)

    # CDF at each edge: cumulative sum of bin probabilities
    cdf_at_edges = np.zeros(len(edges))
    for i in range(len(probs)):
        cdf_at_edges[i + 1] = cdf_at_edges[i] + probs[i]

    return edges, cdf_at_edges


def crps_from_bins(bin_probs: list[float], bin_edges: list[float],
                   actual: float) -> float:
    """Compute CRPS from binned probability forecast.

    Uses piecewise-uniform assumption within bins. CRPS is the integral of
    (F(x) - 1{x >= actual})^2 over x.

    Args:
        bin_probs: List of 10 probability values (should sum to ~100).
        bin_edges: List of 11 bin edge values.
        actual: Realized value.

    Returns:
        CRPS value (lower is better, in units of the outcome variable).
    """
    edges, cdf_at_edges = bins_to_cdf(bin_probs, bin_edges)

    crps = 0.0
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if lo == hi:
            # Degenerate bin (e.g., treasury "exactly 0")
            # Treat as a point mass: contributes (cdf_jump)^2 * 0 width = 0
            # to CRPS, but we need to handle the step function correctly.
            # The CDF jumps by probs[i] at this point.
            # CRPS contribution from a point mass p at x0:
            # if actual > x0: p^2 * 0 (no width)
            # The integral is over a zero-width interval, so contribution is 0.
            continue

        width = hi - lo
        cdf_lo = cdf_at_edges[i]
        cdf_hi = cdf_at_edges[i + 1]

        # Within this bin, CDF is linear from cdf_lo to cdf_hi
        # The indicator 1{x >= actual} is a step function
        if actual <= lo:
            # Entire bin is above actual: indicator = 1
            # Integral of (F(x) - 1)^2 from lo to hi, F linear
            # = integral of (a + bx - 1)^2 dx where a,b define the linear CDF
            crps += _integrate_squared_linear_minus_const(
                cdf_lo, cdf_hi, width, 1.0)
        elif actual >= hi:
            # Entire bin is below actual: indicator = 0
            # Integral of F(x)^2 from lo to hi
            crps += _integrate_squared_linear_minus_const(
                cdf_lo, cdf_hi, width, 0.0)
        else:
            # actual is inside this bin — split at actual
            frac = (actual - lo) / width
            cdf_at_actual = cdf_lo + frac * (cdf_hi - cdf_lo)

            # Left part: [lo, actual], indicator = 0
            left_width = actual - lo
            if left_width > 0:
                crps += _integrate_squared_linear_minus_const(
                    cdf_lo, cdf_at_actual, left_width, 0.0)

            # Right part: [actual, hi], indicator = 1
            right_width = hi - actual
            if right_width > 0:
                crps += _integrate_squared_linear_minus_const(
                    cdf_at_actual, cdf_hi, right_width, 1.0)

    return crps


def _integrate_squared_linear_minus_const(
    f_lo: float, f_hi: float, width: float, c: float
) -> float:
    """Integrate (f(x) - c)^2 dx over an interval of given width,
    where f is linear from f_lo to f_hi.

    Using the identity: integral of (a + bt - c)^2 dt from 0 to w
    = (a-c)^2 * w + (a-c)*b*w^2 + b^2*w^3/3
    where b = (f_hi - f_lo) / width.
    """
    a = f_lo - c
    b = (f_hi - f_lo) / width if width > 0 else 0.0
    w = width
    return a * a * w + a * b * w * w + b * b * w * w * w / 3.0


# ── Ranked Probability Score (RPS) ─────────────────────────────────────────

def rps_from_bins(bin_probs: list[float], bin_edges: list[float],
                  actual: float) -> float:
    """Compute Ranked Probability Score from binned probability forecast.

    RPS = (1/(K-1)) × Σ (F_pred(k) - F_obs(k))²

    where F_pred is the cumulative predicted probability and F_obs is the
    cumulative indicator (0 until the bin containing the actual, then 1).

    Args:
        bin_probs: List of 10 probability values (should sum to ~100).
        bin_edges: List of 11 bin edge values.
        actual: Realized value.

    Returns:
        RPS value (lower is better, dimensionless).
    """
    probs = np.array(bin_probs) / 100.0
    n_bins = len(probs)

    # Find which bin the actual falls in
    actual_bin = n_bins - 1
    for i in range(n_bins):
        if i < n_bins - 1 and actual <= bin_edges[i + 1]:
            actual_bin = i
            break

    # Cumulative predicted and observed
    cum_pred = np.cumsum(probs)
    cum_obs = np.zeros(n_bins)
    cum_obs[actual_bin:] = 1.0

    rps = np.sum((cum_pred - cum_obs) ** 2) / (n_bins - 1)
    return float(rps)


# ── Binary Brier at bin thresholds ─────────────────────────────────────────

def bins_to_binary_forecasts(bin_probs: list[float], bin_edges: list[float],
                              actual: float) -> list[dict]:
    """Convert binned forecast to binary forecasts at each bin threshold.

    Each internal bin boundary becomes a binary question: "will the outcome
    exceed this threshold?" The forecast probability is 1 - cumulative
    probability up to that threshold.

    Args:
        bin_probs: List of 10 probability values (should sum to ~100).
        bin_edges: List of 11 bin edge values.
        actual: Realized value.

    Returns:
        List of 9 dicts with keys: threshold, prob_above, actual_above, brier.
    """
    probs = np.array(bin_probs) / 100.0
    cum_prob = np.cumsum(probs)

    results = []
    for i in range(1, len(bin_edges) - 1):  # 9 internal boundaries
        threshold = bin_edges[i]
        prob_above = 1.0 - cum_prob[i - 1]  # P(outcome > threshold)
        actual_above = 1.0 if actual > threshold else 0.0
        brier = (prob_above - actual_above) ** 2

        results.append({
            "threshold": threshold,
            "prob_above": float(prob_above),
            "actual_above": float(actual_above),
            "brier": float(brier),
        })

    return results


# ── Calibration diagnostic ─────────────────────────────────────────────────

def bin_calibration(results: list[dict]) -> dict:
    """Compute calibration diagnostic: observed vs expected bin frequencies.

    For each bin, computes:
    - mean predicted probability (should be ~10% if models are calibrated)
    - fraction of times the actual falls in that bin

    Args:
        results: List of result dicts with 'bin_probabilities', 'bin_edges',
                 and 'ground_truth' keys.

    Returns:
        Dict with per-bin calibration stats.
    """
    n_bins = 10
    predicted_mass = [[] for _ in range(n_bins)]
    actual_in_bin = [0 for _ in range(n_bins)]
    total = 0

    for r in results:
        if r.get("bin_probabilities") is None:
            continue
        probs = r["bin_probabilities"]
        edges = r["bin_edges"]
        truth = r["ground_truth"]
        total += 1

        for i in range(n_bins):
            predicted_mass[i].append(probs[i] / 100.0)

        # Which bin does truth fall in?
        truth_bin = n_bins - 1
        for i in range(n_bins):
            if i < n_bins - 1 and truth <= edges[i + 1]:
                truth_bin = i
                break
        actual_in_bin[truth_bin] += 1

    calibration = {}
    for i in range(n_bins):
        calibration[f"bin_{i}"] = {
            "mean_predicted": float(np.mean(predicted_mass[i])) if predicted_mass[i] else 0,
            "actual_frequency": actual_in_bin[i] / total if total > 0 else 0,
            "n": total,
        }

    return calibration


# ── Analysis ───────────────────────────────────────────────────────────────

def analyze_results(results_file: str, compare_file: str = None):
    """Run full analysis on bin condition results."""
    with open(results_file) as f:
        results = json.load(f)

    valid = [r for r in results if r.get("bin_probabilities") is not None]
    failed = len(results) - len(valid)
    print(f"Loaded {len(results)} results ({len(valid)} valid, {failed} parse failures)")

    # Compute CRPS and RPS for each result
    for r in valid:
        r["crps"] = crps_from_bins(r["bin_probabilities"], r["bin_edges"],
                                    r["ground_truth"])
        r["rps"] = rps_from_bins(r["bin_probabilities"], r["bin_edges"],
                                  r["ground_truth"])
        r["binary_forecasts"] = bins_to_binary_forecasts(
            r["bin_probabilities"], r["bin_edges"], r["ground_truth"])

    # ── CRPS by model and horizon ──
    print(f"\n{'='*70}")
    print("CRPS by model × horizon (bin_probability condition)")
    print(f"{'='*70}\n")

    models = sorted(set(r["model"] for r in valid))
    horizons = sorted(set(r["resolution_turn"] for r in valid))

    # Header
    h_labels = [f"H{(t-60)//30}" for t in horizons]
    print(f"{'Model':<35} {'All':>8} " + " ".join(f"{h:>8}" for h in h_labels))
    print("-" * (35 + 9 + 9 * len(horizons)))

    for model in models:
        model_short = model.split("/")[-1]
        model_results = [r for r in valid if r["model"] == model]
        all_crps = np.mean([r["crps"] for r in model_results])

        crps_by_h = []
        for h in horizons:
            h_results = [r for r in model_results if r["resolution_turn"] == h]
            crps_by_h.append(np.mean([r["crps"] for r in h_results]) if h_results else float("nan"))

        print(f"{model_short:<35} {all_crps:>8.1f} " +
              " ".join(f"{c:>8.1f}" for c in crps_by_h))

    # ── ECI × CRPS correlation by horizon ──
    print(f"\n{'='*70}")
    print("ECI × CRPS correlation by horizon (Spearman)")
    print(f"{'='*70}\n")

    from scipy import stats

    for h in horizons:
        ecis = []
        crps_vals = []
        for model in models:
            eci = ECI_MAP.get(model)
            if eci is None:
                continue
            h_results = [r for r in valid
                         if r["model"] == model and r["resolution_turn"] == h]
            if h_results:
                ecis.append(eci)
                crps_vals.append(np.mean([r["crps"] for r in h_results]))

        if len(ecis) >= 4:
            rho, p = stats.spearmanr(ecis, crps_vals)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            direction = "anti-g" if rho > 0 else "pro-g"
            h_label = f"H{(h-60)//30} (T{h})"
            print(f"  {h_label:<12} ρ={rho:+.3f}  p={p:.3f} {sig:<4} ({direction})")

    # ── Normalized ECI × CRPS correlation by horizon ──
    print(f"\n{'='*70}")
    print("ECI × Normalized CRPS correlation by horizon (Spearman)")
    print(f"{'='*70}\n")

    tmpl_h_model = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in valid:
        tmpl_h_model[r["template_id"]][r["resolution_turn"]][r["model"]].append(r["crps"])

    tmpl_h_medians = {}
    for tmpl in tmpl_h_model:
        for h in tmpl_h_model[tmpl]:
            model_means = [np.mean(scores) for scores in tmpl_h_model[tmpl][h].values()]
            if model_means:
                tmpl_h_medians[(tmpl, h)] = np.median(model_means)

    for h in horizons:
        model_norm = defaultdict(list)
        for tmpl in tmpl_h_model:
            med = tmpl_h_medians.get((tmpl, h))
            if med is None or med == 0:
                continue
            for model, scores in tmpl_h_model[tmpl][h].items():
                model_norm[model].append(np.mean(scores) / med)

        ecis = []
        norm_vals = []
        for model in models:
            eci = ECI_MAP.get(model)
            if eci is None or model not in model_norm:
                continue
            ecis.append(eci)
            norm_vals.append(np.mean(model_norm[model]))

        if len(ecis) >= 4:
            rho, p = stats.spearmanr(ecis, norm_vals)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            direction = "anti-g" if rho > 0 else "pro-g"
            h_label = f"H{(h-60)//30} (T{h})"
            print(f"  {h_label:<12} ρ={rho:+.3f}  p={p:.3f} {sig:<4} ({direction})")

    # ── RPS by model and horizon ──
    print(f"\n{'='*70}")
    print("RPS by model × horizon (bin_probability condition)")
    print(f"{'='*70}\n")

    print(f"{'Model':<35} {'All':>8} " + " ".join(f"{h:>8}" for h in h_labels))
    print("-" * (35 + 9 + 9 * len(horizons)))

    for model in models:
        model_short = model.split("/")[-1]
        model_results = [r for r in valid if r["model"] == model]
        all_rps = np.mean([r["rps"] for r in model_results])

        rps_by_h = []
        for h in horizons:
            h_results = [r for r in model_results if r["resolution_turn"] == h]
            rps_by_h.append(np.mean([r["rps"] for r in h_results]) if h_results else float("nan"))

        print(f"{model_short:<35} {all_rps:>8.4f} " +
              " ".join(f"{c:>8.4f}" for c in rps_by_h))

    # ── ECI × RPS correlation by horizon ──
    print(f"\n{'='*70}")
    print("ECI × RPS correlation by horizon (Spearman)")
    print(f"{'='*70}\n")

    for h in horizons:
        ecis = []
        rps_vals = []
        for model in models:
            eci = ECI_MAP.get(model)
            if eci is None:
                continue
            h_results = [r for r in valid
                         if r["model"] == model and r["resolution_turn"] == h]
            if h_results:
                ecis.append(eci)
                rps_vals.append(np.mean([r["rps"] for r in h_results]))

        if len(ecis) >= 4:
            rho, p = stats.spearmanr(ecis, rps_vals)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            direction = "anti-g" if rho > 0 else "pro-g"
            h_label = f"H{(h-60)//30} (T{h})"
            print(f"  {h_label:<12} ρ={rho:+.3f}  p={p:.3f} {sig:<4} ({direction})")

    # ── Calibration ──
    print(f"\n{'='*70}")
    print("Bin calibration (pooled across all models)")
    print(f"{'='*70}\n")

    cal = bin_calibration(valid)
    print(f"{'Bin':<8} {'Predicted':>10} {'Actual':>10} {'Δ':>8}")
    print("-" * 38)
    for i in range(10):
        c = cal[f"bin_{i}"]
        delta = c["actual_frequency"] - c["mean_predicted"]
        print(f"bin_{i:<4} {c['mean_predicted']:>10.1%} {c['actual_frequency']:>10.1%} {delta:>+8.1%}")

    # ── Binary Brier (FB bridge) ──
    print(f"\n{'='*70}")
    print("Mean binary Brier at bin thresholds (pooled)")
    print(f"{'='*70}\n")

    all_binary = []
    for r in valid:
        all_binary.extend(r["binary_forecasts"])

    by_threshold = defaultdict(list)
    for b in all_binary:
        by_threshold[b["threshold"]].append(b["brier"])

    print(f"{'Threshold':>10} {'Mean Brier':>12} {'N':>8}")
    print("-" * 32)
    for threshold in sorted(by_threshold.keys()):
        briers = by_threshold[threshold]
        print(f"{threshold:>10.0f} {np.mean(briers):>12.4f} {len(briers):>8}")

    # ── Compare to quantile baseline if provided ──
    if compare_file:
        print(f"\n{'='*70}")
        print(f"Comparison: bin_probability vs quantile baseline")
        print(f"{'='*70}\n")

        with open(compare_file) as f:
            baseline = json.load(f)

        # Filter to baseline condition only
        baseline = [r for r in baseline
                    if r.get("condition") == "baseline" and r.get("percentiles")]

        if not baseline:
            print("  No baseline results found in comparison file.")
            return

        # Compute CRPS for baseline (from quantiles)
        from civrealm.evaluation.scoring import crps_from_quantiles
        for r in baseline:
            pcts = r["percentiles"]
            quantiles = [pcts.get(k) for k in ["p10", "p25", "p50", "p75", "p90"]]
            if all(q is not None for q in quantiles):
                r["crps"] = crps_from_quantiles(
                    quantiles, [0.1, 0.25, 0.5, 0.75, 0.9], r["ground_truth"])

        baseline_valid = [r for r in baseline if "crps" in r]

        print(f"  Bin: {len(valid)} results, Baseline: {len(baseline_valid)} results\n")

        print(f"{'Horizon':<12} {'Bin CRPS':>10} {'Base CRPS':>10} {'Δ':>10} {'Better':>8}")
        print("-" * 52)

        for h in horizons:
            bin_crps = [r["crps"] for r in valid if r["resolution_turn"] == h]
            base_crps = [r["crps"] for r in baseline_valid if r["resolution_turn"] == h]

            if bin_crps and base_crps:
                bin_mean = np.mean(bin_crps)
                base_mean = np.mean(base_crps)
                delta = bin_mean - base_mean
                better = "BIN" if delta < 0 else "BASE"
                h_label = f"H{(h-60)//30}"
                print(f"  {h_label:<10} {bin_mean:>10.1f} {base_mean:>10.1f} "
                      f"{delta:>+10.1f} {better:>8}")


def main():
    parser = argparse.ArgumentParser(description="Score bin probability forecasts")
    parser.add_argument("--results", default="data/results/bin_condition/bin_results.json",
                        help="Bin condition results file")
    parser.add_argument("--compare", default=None,
                        help="Quantile baseline results file for comparison")
    args = parser.parse_args()

    analyze_results(args.results, args.compare)


if __name__ == "__main__":
    main()
