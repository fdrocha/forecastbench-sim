#!/usr/bin/env python3
"""
Compare GenSME vs constant-value baseline on CivBench disruptable templates.

The constant-value baseline predicts the turn-60 value at all future horizons
with zero spread. This script computes CRPS for both and compares.

Key finding: GenSME beats the constant-value baseline pooled (216.9 vs 221.8,
-2.2%), but the result is horizon-dependent. GenSME dominates at H1-H3
(-31% to -52%) but loses at H5-H6 (+47% to +93%). The crossover is at H4,
where the two are approximately equal. This is the opposite of the monolithic
result, where the constant-value baseline dominates at far horizons. GenSME
recovers enough of the integration bottleneck to beat the baseline overall,
but its model-generated disruption conditionals are not as well-calibrated
as the population-derived ones used in the counterfactual mixture.

Per-model: 4 of 5 models beat the baseline under GenSME. The exception is
Gemini 3 Pro (+58.4%), which produces poorly calibrated scenario weights.

Usage:
    cd /Users/elsehow/Projects/civbench
    uv run python scripts/analyze_gensme_vs_baseline.py

Data:
    - GenSME results: data/results/generate_scenario/gensme_results.json
    - Game data (turn-60 values): data/games/{seed}_data.json
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path

GENSME_FILE = Path("data/results/generate_scenario/gensme_results.json")
GAMES_DIR = Path("data/games")

TEMPLATE_TS_KEYS = {
    "cities_count_continuous": "cities_count",
    "population_continuous": "population",
    "territory_continuous": "territory_size",
    "treasury_continuous": "treasury",
}

HORIZON_LABELS = {90: "H1", 120: "H2", 150: "H3", 180: "H4", 210: "H5", 240: "H6"}


def crps_quantile(quantiles: dict, ground_truth: float) -> float:
    """CRPS from 5-quantile forecast using pinball loss."""
    taus = [0.10, 0.25, 0.50, 0.75, 0.90]
    keys = ["p10", "p25", "p50", "p75", "p90"]
    total = 0.0
    for tau, key in zip(taus, keys):
        q = quantiles[key]
        diff = ground_truth - q
        if diff >= 0:
            total += tau * diff
        else:
            total += (tau - 1) * diff
    return total * 2 / len(taus)


def main():
    with open(GENSME_FILE) as f:
        gensme = json.load(f)

    # Load turn-60 values for each seed/template
    t60_values = {}
    seeds = sorted(set(r["seed"] for r in gensme))
    for seed in seeds:
        game_file = GAMES_DIR / f"{seed}_data.json"
        with open(game_file) as f:
            game = json.load(f)
        ts = game["time_series"]
        for tmpl_id, ts_key in TEMPLATE_TS_KEYS.items():
            t60_values[(seed, tmpl_id)] = ts[ts_key]["60"]["0"]

    # Compute CRPS for both methods on each question
    by_horizon = defaultdict(lambda: {"gensme": [], "baseline": []})
    by_template = defaultdict(lambda: {"gensme": [], "baseline": []})
    by_model = defaultdict(lambda: {"gensme": [], "baseline": []})

    for r in gensme:
        seed, tmpl, turn = r["seed"], r["template_id"], r["resolution_turn"]
        truth = r["ground_truth"]
        horizon = HORIZON_LABELS[turn]
        model = r["model"].split("/")[-1]

        gensme_crps = r["mixture_crps"]

        t60 = t60_values.get((seed, tmpl))
        if t60 is None:
            continue
        baseline_pcts = {k: t60 for k in ["p10", "p25", "p50", "p75", "p90"]}
        baseline_crps = crps_quantile(baseline_pcts, truth)

        by_horizon[horizon]["gensme"].append(gensme_crps)
        by_horizon[horizon]["baseline"].append(baseline_crps)
        by_template[tmpl]["gensme"].append(gensme_crps)
        by_template[tmpl]["baseline"].append(baseline_crps)
        by_model[model]["gensme"].append(gensme_crps)
        by_model[model]["baseline"].append(baseline_crps)

    # Print results
    print("=" * 70)
    print("GenSME vs CONSTANT-VALUE BASELINE on CivBench disruptable templates")
    print("=" * 70)

    print(f"\n{'Horizon':<8} {'GenSME':>10} {'Baseline':>10} {'Δ%':>8} {'GenSME wins':>12}")
    print("-" * 52)
    all_g, all_b = [], []
    for h in ["H1", "H2", "H3", "H4", "H5", "H6"]:
        g = by_horizon[h]["gensme"]
        b = by_horizon[h]["baseline"]
        gm, bm = np.mean(g), np.mean(b)
        delta = (gm - bm) / bm * 100
        wins = sum(1 for gi, bi in zip(g, b) if gi < bi)
        print(f"{h:<8} {gm:>9.1f} {bm:>9.1f} {delta:>+7.1f}% {wins}/{len(g)}")
        all_g.extend(g)
        all_b.extend(b)

    gm_all, bm_all = np.mean(all_g), np.mean(all_b)
    delta_all = (gm_all - bm_all) / bm_all * 100
    wins_all = sum(1 for gi, bi in zip(all_g, all_b) if gi < bi)
    print(f"{'POOLED':<8} {gm_all:>9.1f} {bm_all:>9.1f} {delta_all:>+7.1f}% {wins_all}/{len(all_g)}")

    print(f"\n{'Template':<30} {'GenSME':>10} {'Baseline':>10} {'Δ%':>8}")
    print("-" * 62)
    for tmpl in sorted(by_template):
        g = by_template[tmpl]["gensme"]
        b = by_template[tmpl]["baseline"]
        gm, bm = np.mean(g), np.mean(b)
        delta = (gm - bm) / bm * 100
        short = tmpl.replace("_continuous", "")
        print(f"{short:<30} {gm:>9.1f} {bm:>9.1f} {delta:>+7.1f}%")

    print(f"\n{'Model':<30} {'GenSME':>10} {'Baseline':>10} {'Δ%':>8}")
    print("-" * 62)
    for model in sorted(by_model):
        g = by_model[model]["gensme"]
        b = by_model[model]["baseline"]
        gm, bm = np.mean(g), np.mean(b)
        delta = (gm - bm) / bm * 100
        print(f"{model:<30} {gm:>9.1f} {bm:>9.1f} {delta:>+7.1f}%")


if __name__ == "__main__":
    main()
