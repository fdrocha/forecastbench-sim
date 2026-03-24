#!/usr/bin/env python3
"""
Compare ECI × score correlations under RPS vs CRPS for the bin condition.

Produces a single chart showing the pro-g → anti-g flip under both metrics.

Usage:
    cd /Users/elsehow/Projects/civbench
    uv run python scripts/chart_rps_vs_crps.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from score_bin_condition import crps_from_bins, rps_from_bins

ECI_MAP = {
    "openai/gpt-4.1-2025-04-14": 137,
    "anthropic/claude-sonnet-4-5-20250929": 148,
    "openai/gpt-5-2025-08-07": 150,
    "google/gemini-3-pro-preview": 154,
    "anthropic/claude-opus-4-6": 157,
}

HORIZONS = [90, 120, 150, 180, 210, 240]
H_LABELS = [f"H{(t-60)//30}" for t in HORIZONS]


def compute_eci_correlation(results, score_key):
    """Compute Spearman ρ(ECI, score) by horizon."""
    by_model_h = defaultdict(list)
    for r in results:
        by_model_h[(r["model"], r["resolution_turn"])].append(r[score_key])

    models = sorted(set(r["model"] for r in results))
    correlations = {}
    for h in HORIZONS:
        ecis = []
        vals = []
        for model in models:
            eci = ECI_MAP.get(model)
            if eci is None:
                continue
            key = (model, h)
            if key in by_model_h:
                ecis.append(eci)
                vals.append(np.mean(by_model_h[key]))

        if len(ecis) >= 4:
            rho, p = stats.spearmanr(ecis, vals)
            correlations[h] = {"rho": rho, "p": p}
        else:
            correlations[h] = {"rho": float("nan"), "p": 1.0}
    return correlations


def main():
    results_file = Path("data/results/bin_condition/bin_results.json")
    with open(results_file) as f:
        results = json.load(f)

    valid = [r for r in results if r.get("bin_probabilities") is not None]

    for r in valid:
        r["crps"] = crps_from_bins(r["bin_probabilities"], r["bin_edges"],
                                    r["ground_truth"])
        r["rps"] = rps_from_bins(r["bin_probabilities"], r["bin_edges"],
                                  r["ground_truth"])

    crps_corr = compute_eci_correlation(valid, "crps")
    rps_corr = compute_eci_correlation(valid, "rps")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))

    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.axhspan(0, 1, alpha=0.04, color="red")
    ax.axhspan(-1, 0, alpha=0.04, color="blue")

    for label, corr, color, lw in [
        ("CRPS (continuous)", crps_corr, "#e74c3c", 2.5),
        ("RPS (discrete)", rps_corr, "#3498db", 2.5),
    ]:
        x = [h for h in HORIZONS if h in corr]
        y = [corr[h]["rho"] for h in x]
        p_vals = [corr[h]["p"] for h in x]

        ax.plot(x, y, 'o-', label=label, color=color, linewidth=lw, markersize=6)

        # Filled markers for significant points
        for xi, yi, pi in zip(x, y, p_vals):
            if pi < 0.05:
                ax.plot(xi, yi, 'o', color=color, markersize=9, zorder=10)

    ax.set_xlabel("Horizon (turn)")
    ax.set_ylabel("Spearman ρ(ECI, score)")
    ax.set_title("Capability × accuracy: CRPS vs RPS on bin probability condition\n(filled = p<0.05, N=5 models)")
    ax.set_xticks(HORIZONS)
    ax.set_xticklabels([f"T{h}\n({l})" for h, l in zip(HORIZONS, H_LABELS)])
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.15, 1.15)

    ax.text(HORIZONS[-1] + 5, 0.5, "anti-g\n(worse)", fontsize=8,
            color="red", alpha=0.5, ha="left", va="center")
    ax.text(HORIZONS[-1] + 5, -0.5, "pro-g\n(better)", fontsize=8,
            color="blue", alpha=0.5, ha="left", va="center")

    plt.tight_layout()

    out_path = Path("/Users/elsehow/Projects/fri-vault/_artifacts/static/rps_vs_crps_eci.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
