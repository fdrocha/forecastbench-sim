#!/usr/bin/env python3
"""
Compare bin probability elicitation to all other conditions.

Produces a 2-panel chart:
  Top: CRPS by horizon across all conditions
  Bottom: ECI × CRPS correlation by horizon (anti-g test)

Usage:
    cd /Users/elsehow/Projects/civbench
    uv run python scripts/chart_bin_vs_conditions.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from score_bin_condition import crps_from_bins

# ── Config ──────────────────────────────────────────────────────────────────

ECI_MAP = {
    "openai/gpt-4.1-2025-04-14": 137,
    "anthropic/claude-sonnet-4-5-20250929": 148,
    "openai/gpt-5-2025-08-07": 150,
    "google/gemini-3-pro-preview": 154,
    "anthropic/claude-opus-4-6": 157,
}

# Short model names for agentic results
AGENTIC_MODEL_MAP = {
    "gpt-4.1": "openai/gpt-4.1-2025-04-14",
    "claude-sonnet-4-5": "anthropic/claude-sonnet-4-5-20250929",
    "gpt-5": "openai/gpt-5-2025-08-07",
    "gemini-3-pro-preview": "google/gemini-3-pro-preview",
    "claude-opus-4-6": "anthropic/claude-opus-4-6",
}

HORIZONS = [90, 120, 150, 180, 210, 240]
H_LABELS = [f"H{(t-60)//30}" for t in HORIZONS]

DATA_DIR = Path("data/results")
OUT_DIR = Path(__file__).parent.parent / ".." / "fri-vault" / "_artifacts" / "static"


# ── CRPS from quantile percentiles ─────────────────────────────────────────

def crps_from_quantiles(percentiles: dict, actual: float) -> float | None:
    """Compute CRPS from p10/p25/p50/p75/p90 quantile forecast.

    Uses the weighted interval score (WIS) approximation:
    CRPS ≈ Σ (alpha_k / 2) * IS_k + (1/2) * |median - actual|
    where IS_k is the interval score for the (alpha_k) central interval.
    """
    keys = ["p10", "p25", "p50", "p75", "p90"]
    vals = [percentiles.get(k) for k in keys]
    if any(v is None for v in vals):
        return None

    p10, p25, p50, p75, p90 = vals

    # Interval scores for 80% and 50% central intervals
    def interval_score(lower, upper, alpha, actual):
        width = upper - lower
        penalty_lower = (2 / alpha) * max(0, lower - actual)
        penalty_upper = (2 / alpha) * max(0, actual - upper)
        return width + penalty_lower + penalty_upper

    is_80 = interval_score(p10, p90, 0.2, actual)
    is_50 = interval_score(p25, p75, 0.5, actual)
    abs_error = abs(p50 - actual)

    # WIS with 2 intervals + median
    # Weights: alpha/2 for each interval, 1/2 for median, normalized
    wis = (0.2 / 2) * is_80 + (0.5 / 2) * is_50 + (1 / 2) * abs_error
    # Normalize by number of components (K=2 intervals + 1 median = 3)
    # Actually WIS = (1/(K+0.5)) * [sum + 0.5*|median-y|]
    # With K=2: WIS = (1/2.5) * [0.1*IS_80 + 0.25*IS_50 + 0.5*|med-y|]
    # Simpler: just use the raw sum / (K + 0.5)
    wis = (1 / 2.5) * (0.1 * is_80 + 0.25 * is_50 + 0.5 * abs_error)

    return wis


# ── Load all conditions ─────────────────────────────────────────────────────

def load_quantile_condition(filepath: str, condition_name: str) -> list[dict]:
    """Load a quantile-based condition and compute CRPS."""
    with open(filepath) as f:
        data = json.load(f)

    results = []
    for r in data:
        if r.get("condition") != condition_name:
            continue
        if not r.get("percentiles"):
            continue
        crps = crps_from_quantiles(r["percentiles"], r["ground_truth"])
        if crps is not None:
            results.append({
                "model": r["model"],
                "resolution_turn": r["resolution_turn"],
                "crps": crps,
                "template_id": r["template_id"],
            })
    return results


def load_bin_condition(filepath: str) -> list[dict]:
    """Load bin probability condition and compute CRPS."""
    with open(filepath) as f:
        data = json.load(f)

    results = []
    for r in data:
        if not r.get("bin_probabilities"):
            continue
        crps = crps_from_bins(r["bin_probabilities"], r["bin_edges"],
                              r["ground_truth"])
        results.append({
            "model": r["model"],
            "resolution_turn": r["resolution_turn"],
            "crps": crps,
            "template_id": r["template_id"],
        })
    return results


def load_agentic_condition(filepath: str) -> list[dict]:
    """Load pre-aggregated agentic results."""
    with open(filepath) as f:
        data = json.load(f)

    results = []
    for short_name, model_data in data.items():
        full_name = AGENTIC_MODEL_MAP.get(short_name, short_name)
        for h_label, h_data in model_data.get("by_horizon", {}).items():
            h_num = int(h_label[1])  # H1 -> 1
            turn = 60 + h_num * 30
            results.append({
                "model": full_name,
                "resolution_turn": turn,
                "crps": h_data["crps"],
                "_aggregated": True,  # already averaged
            })
    return results


def compute_crps_by_horizon(results: list[dict]) -> dict:
    """Compute mean CRPS by horizon, pooled across models."""
    by_h = defaultdict(list)
    for r in results:
        by_h[r["resolution_turn"]].append(r["crps"])
    return {h: np.mean(vals) for h, vals in by_h.items()}


def compute_eci_correlation(results: list[dict]) -> dict:
    """Compute Spearman ρ(ECI, mean CRPS) by horizon."""
    # Group by (model, horizon)
    by_model_h = defaultdict(list)
    for r in results:
        by_model_h[(r["model"], r["resolution_turn"])].append(r["crps"])

    correlations = {}
    for h in HORIZONS:
        ecis = []
        crps_vals = []
        for model in sorted(set(r["model"] for r in results)):
            eci = ECI_MAP.get(model)
            if eci is None:
                continue
            key = (model, h)
            if key in by_model_h:
                ecis.append(eci)
                crps_vals.append(np.mean(by_model_h[key]))

        if len(ecis) >= 4:
            rho, p = stats.spearmanr(ecis, crps_vals)
            correlations[h] = {"rho": rho, "p": p, "n": len(ecis)}
        else:
            correlations[h] = {"rho": float("nan"), "p": 1.0, "n": len(ecis)}

    return correlations


def compute_eci_correlation_normalized(results: list[dict]) -> dict:
    """Compute Spearman rho(ECI, normalized_CRPS) by horizon.

    Normalizes per (template, horizon) by cross-model median before aggregating,
    giving each template equal weight regardless of raw scale.
    """
    # Step 1: per-(template, horizon, model) mean CRPS
    tmpl_h_model = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in results:
        tmpl = r.get("template_id")
        if tmpl is None:
            continue
        tmpl_h_model[tmpl][r["resolution_turn"]][r["model"]].append(r["crps"])

    # Step 2: per-(template, horizon) cross-model medians
    tmpl_h_medians = {}
    for tmpl in tmpl_h_model:
        for h in tmpl_h_model[tmpl]:
            model_means = [np.mean(scores) for scores in tmpl_h_model[tmpl][h].values()]
            if model_means:
                tmpl_h_medians[(tmpl, h)] = np.median(model_means)

    # Step 3: normalized aggregate per (model, horizon)
    correlations = {}
    for h in HORIZONS:
        model_norm = defaultdict(list)
        for tmpl in tmpl_h_model:
            med = tmpl_h_medians.get((tmpl, h))
            if med is None or med == 0:
                continue
            for model, scores in tmpl_h_model[tmpl][h].items():
                model_norm[model].append(np.mean(scores) / med)

        ecis = []
        norm_vals = []
        for model in sorted(model_norm.keys()):
            eci = ECI_MAP.get(model)
            if eci is None:
                continue
            ecis.append(eci)
            norm_vals.append(np.mean(model_norm[model]))

        if len(ecis) >= 4:
            rho, p = stats.spearmanr(ecis, norm_vals)
            correlations[h] = {"rho": rho, "p": p, "n": len(ecis)}
        else:
            correlations[h] = {"rho": float("nan"), "p": 1.0, "n": len(ecis)}

    return correlations


# ── Chart ───────────────────────────────────────────────────────────────────

def main():
    conditions = {}

    # Load quantile conditions
    traj_file = DATA_DIR / "trajectory_condition" / "trajectory_results.json"
    if traj_file.exists():
        conditions["Baseline (quantile)"] = load_quantile_condition(str(traj_file), "baseline")
        conditions["Trajectory"] = load_quantile_condition(str(traj_file), "trajectory")

    dk_file = DATA_DIR / "domain_knowledge_condition" / "domain_knowledge_results.json"
    if dk_file.exists():
        conditions["Domain knowledge"] = load_quantile_condition(str(dk_file), "domain_knowledge")

    combined_file = DATA_DIR / "combined_condition" / "combined_results.json"
    if combined_file.exists():
        conditions["Combined (traj+DK)"] = load_quantile_condition(str(combined_file), "combined")

    agentic_file = DATA_DIR / "agentic_baseline" / "scored_results.json"
    if agentic_file.exists():
        conditions["Agentic"] = load_agentic_condition(str(agentic_file))

    # Load bin condition
    bin_file = DATA_DIR / "bin_condition" / "bin_results.json"
    if bin_file.exists():
        conditions["Bin probability"] = load_bin_condition(str(bin_file))
    else:
        print(f"WARNING: {bin_file} not found — run run_bin_condition.py first")

    print(f"Loaded {len(conditions)} conditions:")
    for name, results in conditions.items():
        print(f"  {name}: {len(results)} results")

    # ── Compute metrics ──
    crps_by_condition = {}
    eci_corr_by_condition = {}
    eci_corr_norm_by_condition = {}
    for name, results in conditions.items():
        crps_by_condition[name] = compute_crps_by_horizon(results)
        eci_corr_by_condition[name] = compute_eci_correlation(results)
        # Agentic results are pre-aggregated (no template_id) — skip normalization
        if name == "Agentic":
            eci_corr_norm_by_condition[name] = eci_corr_by_condition[name]
        else:
            eci_corr_norm_by_condition[name] = compute_eci_correlation_normalized(results)

    # ── Plot ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Color scheme: bin probability is highlighted
    colors = {
        "Baseline (quantile)": "#999999",
        "Trajectory": "#66b3ff",
        "Domain knowledge": "#99cc99",
        "Combined (traj+DK)": "#ffcc66",
        "Agentic": "#ff9999",
        "Bin probability": "#e74c3c",
    }
    linewidths = {name: 1.5 for name in conditions}
    linewidths["Bin probability"] = 3.0
    linewidths["Baseline (quantile)"] = 2.5

    # Top panel: CRPS by horizon
    for name in conditions:
        crps_h = crps_by_condition[name]
        x = [h for h in HORIZONS if h in crps_h]
        y = [crps_h[h] for h in x]
        ax1.plot(x, y, 'o-', label=name, color=colors.get(name, "#333"),
                 linewidth=linewidths.get(name, 1.5), markersize=5,
                 zorder=10 if name == "Bin probability" else 5)

    ax1.set_ylabel("Mean CRPS (lower = better)")
    ax1.set_title("Forecast accuracy by condition and horizon")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Bottom panel: ECI × normalized CRPS correlation (anti-g test)
    ax2.axhline(y=0, color="black", linewidth=0.5, linestyle="-")
    for name in conditions:
        corr = eci_corr_norm_by_condition[name]
        x = [h for h in HORIZONS if h in corr]
        y = [corr[h]["rho"] for h in x]
        p_vals = [corr[h]["p"] for h in x]

        ax2.plot(x, y, 'o-', label=name, color=colors.get(name, "#333"),
                 linewidth=linewidths.get(name, 1.5), markersize=5,
                 zorder=10 if name == "Bin probability" else 5)

        # Mark significant points (p < 0.05) with filled markers
        for xi, yi, pi in zip(x, y, p_vals):
            if pi < 0.05:
                ax2.plot(xi, yi, 'o', color=colors.get(name, "#333"),
                         markersize=8, zorder=11 if name == "Bin probability" else 6)

    ax2.set_xlabel("Horizon (turn)")
    ax2.set_ylabel("Spearman ρ(ECI, CRPS)")
    ax2.set_title("Capability × accuracy correlation (>0 = anti-g, filled = p<0.05)")
    ax2.set_xticks(HORIZONS)
    ax2.set_xticklabels([f"T{h}\n({l})" for h, l in zip(HORIZONS, H_LABELS)])
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, 1)

    # Add annotation regions
    ax2.axhspan(0, 1, alpha=0.04, color="red")
    ax2.axhspan(-1, 0, alpha=0.04, color="blue")
    ax2.text(HORIZONS[-1] + 5, 0.5, "anti-g\n(worse)", fontsize=7,
             color="red", alpha=0.5, ha="left", va="center")
    ax2.text(HORIZONS[-1] + 5, -0.5, "pro-g\n(better)", fontsize=7,
             color="blue", alpha=0.5, ha="left", va="center")

    plt.tight_layout()

    out_path = Path("/Users/elsehow/Projects/fri-vault/_artifacts/static/bin_vs_conditions.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved to {out_path}")

    # Also print summary table
    print(f"\n{'='*70}")
    print("CRPS summary (H4-H6 mean)")
    print(f"{'='*70}")
    for name in conditions:
        crps_h = crps_by_condition[name]
        far_h = [crps_h.get(h, float("nan")) for h in [180, 210, 240]]
        print(f"  {name:<25} {np.nanmean(far_h):>8.1f}")

    print(f"\n{'='*70}")
    print("ECI × CRPS at H4-H6 (raw, anti-g test)")
    print(f"{'='*70}")
    for name in conditions:
        corr = eci_corr_by_condition[name]
        for h in [180, 210, 240]:
            if h in corr:
                c = corr[h]
                sig = "***" if c["p"] < 0.001 else "**" if c["p"] < 0.01 else "*" if c["p"] < 0.05 else ""
                h_label = f"H{(h-60)//30}"
                print(f"  {name:<25} {h_label} ρ={c['rho']:+.3f} p={c['p']:.3f} {sig}")

    # ── Normalized ECI × CRPS summary table (Table tab:anti-g-by-condition) ──
    print(f"\n{'='*70}")
    print("ECI × Normalized CRPS — ALL horizons H1-H6 (Spearman)")
    print("(Agentic: raw correlation, no template_id available)")
    print(f"{'='*70}\n")

    cond_names = list(conditions.keys())
    # Header row
    header = f"{'Horizon':<10}" + "".join(f"{n:>27}" for n in cond_names)
    print(header)
    print("-" * len(header))

    for h in HORIZONS:
        h_label = f"H{(h-60)//30}"
        row = f"{h_label:<10}"
        for name in cond_names:
            corr = eci_corr_norm_by_condition[name]
            if h in corr:
                c = corr[h]
                sig = "***" if c["p"] < 0.001 else "**" if c["p"] < 0.01 else "*" if c["p"] < 0.05 else ""
                cell = f"ρ={c['rho']:+.3f} p={c['p']:.3f}{sig}"
            else:
                cell = "n/a"
            row += f"{cell:>27}"
        print(row)

    # Also print a compact ρ-only table for easy copy/paste into paper
    print(f"\n{'='*70}")
    print("Compact normalized ρ table (H1-H6 × condition)")
    print(f"{'='*70}\n")
    compact_header = f"{'Horizon':<8}" + "".join(f"{n[:14]:>16}" for n in cond_names)
    print(compact_header)
    print("-" * len(compact_header))
    for h in HORIZONS:
        h_label = f"H{(h-60)//30}"
        row = f"{h_label:<8}"
        for name in cond_names:
            corr = eci_corr_norm_by_condition[name]
            if h in corr and not np.isnan(corr[h]["rho"]):
                c = corr[h]
                sig = "***" if c["p"] < 0.001 else "**" if c["p"] < 0.01 else "*" if c["p"] < 0.05 else ""
                cell = f"{c['rho']:+.3f}{sig}"
            else:
                cell = "n/a"
            row += f"{cell:>16}"
        print(row)


if __name__ == "__main__":
    main()
