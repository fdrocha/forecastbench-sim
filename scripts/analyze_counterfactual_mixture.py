#!/usr/bin/env python3
"""
Counterfactual mixture reconstruction analysis.

Tests whether externally mixing the model's own growth forecast with a
population-derived disruption distribution (weighted by the model's own
P(disruption) from sub-task 2) produces better CRPS than the model's
monolithic distributional forecast.

If it does, the integration step is the bottleneck — the model has the
right pieces but fails to combine them in its monolithic output.
"""

import json
import numpy as np
from collections import defaultdict
from pathlib import Path

# ── Paths ───────────────────────────────────────────────────────────────────

BASELINE_RESULTS = Path("data/results/domain_knowledge_condition/domain_knowledge_results.json")
WEIGHTING_RESULTS = Path("data/results/scenario_weighting/scenario_weighting_results.json")
GAMES_DIR = Path("data/games")
SEED_CLASSIFICATIONS = Path("data/seed_classifications.json")

# ── Config ──────────────────────────────────────────────────────────────────

HORIZON_TURNS = [90, 120, 150, 180, 210, 240]
HORIZON_LABELS = {90: "H1", 120: "H2", 150: "H3", 180: "H4", 210: "H5", 240: "H6"}

DISRUPTION_THRESHOLD = 0.20  # >20% decline from peak = disrupted

TS_KEY_MAP = {
    "population_continuous": "population",
    "territory_continuous": "territory_size",
    "treasury_continuous": "treasury",
    "cities_count_continuous": "cities_count",
}

ECI_MAP = {
    "openai/gpt-4.1-2025-04-14": 1471,
    "anthropic/claude-sonnet-4-5-20250929": 1470,
    "openai/gpt-5-2025-08-07": 1537,
    "google/gemini-3-pro-preview": 1474,
    "anthropic/claude-opus-4-6": 1542,
}

MODEL_SHORT = {
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "anthropic/claude-sonnet-4-5-20250929": "Sonnet 4.5",
    "openai/gpt-5-2025-08-07": "GPT-5",
    "google/gemini-3-pro-preview": "Gemini 3 Pro",
    "anthropic/claude-opus-4-6": "Opus 4.6",
}

TEST_SEEDS = ["seed0", "seed1", "seed4", "seed5", "seed9",
              "seed10", "seed13", "seed15", "seed16", "seed20"]


# ── CRPS computation ────────────────────────────────────────────────────────

def crps_quantile(quantiles: dict, ground_truth: float) -> float:
    """Compute CRPS from quantile forecasts using the quantile score decomposition.

    Uses the standard formula: CRPS ≈ (2/K) Σ_k ρ_τk(y - q_k)
    where ρ_τ(u) = u(τ - I(u<0)) is the check/pinball loss.
    """
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


# ── Population disruption distributions ─────────────────────────────────────

def compute_population_disruption_distributions() -> dict:
    """Compute empirical disruption distributions from all available seeds.

    Returns: dict of (template_id, turn) -> {p10, p25, p50, p75, p90}
    for seeds that experienced >20% decline from peak by that turn.
    Also returns the continuation distributions.
    """
    # Load all game files
    all_games = {}
    for game_file in GAMES_DIR.glob("*_data.json"):
        seed_name = game_file.stem.replace("_data", "")
        with open(game_file) as f:
            all_games[seed_name] = json.load(f)

    print(f"Loaded {len(all_games)} game files for population distributions")

    disruption_dists = {}
    continuation_dists = {}

    for tmpl_id, ts_key in TS_KEY_MAP.items():
        for turn in HORIZON_TURNS:
            disrupted_values = []
            continuation_values = []

            for seed_name, game in all_games.items():
                ts = game["time_series"]
                if ts_key not in ts:
                    continue
                if str(turn) not in ts[ts_key]:
                    continue
                if "0" not in ts[ts_key][str(turn)]:
                    continue

                # Get all values for player 0 up to this turn
                values = {}
                for t_str, civs in ts[ts_key].items():
                    t = int(t_str)
                    if "0" in civs and t <= turn:
                        values[t] = civs["0"]

                if not values:
                    continue

                peak = max(values.values())
                val = values.get(turn, 0)

                if peak > 0:
                    decline = (peak - val) / peak
                else:
                    decline = 0

                if decline > DISRUPTION_THRESHOLD:
                    disrupted_values.append(val)
                else:
                    continuation_values.append(val)

            if disrupted_values:
                arr = np.array(disrupted_values)
                disruption_dists[(tmpl_id, turn)] = {
                    "p10": float(np.percentile(arr, 10)),
                    "p25": float(np.percentile(arr, 25)),
                    "p50": float(np.percentile(arr, 50)),
                    "p75": float(np.percentile(arr, 75)),
                    "p90": float(np.percentile(arr, 90)),
                    "n": len(disrupted_values),
                }
            if continuation_values:
                arr = np.array(continuation_values)
                continuation_dists[(tmpl_id, turn)] = {
                    "p10": float(np.percentile(arr, 10)),
                    "p25": float(np.percentile(arr, 25)),
                    "p50": float(np.percentile(arr, 50)),
                    "p75": float(np.percentile(arr, 75)),
                    "p90": float(np.percentile(arr, 90)),
                    "n": len(continuation_values),
                }

    return disruption_dists, continuation_dists


# ── Main analysis ───────────────────────────────────────────────────────────

def main():
    # Load data
    with open(BASELINE_RESULTS) as f:
        baseline_data = json.load(f)
    with open(WEIGHTING_RESULTS) as f:
        weighting_data = json.load(f)

    print("Computing population disruption distributions...")
    disruption_dists, continuation_dists = compute_population_disruption_distributions()

    # Show disruption distribution sizes
    print(f"\nDisruption distribution sizes by template × horizon:")
    for tmpl_id in sorted(TS_KEY_MAP):
        sizes = []
        for turn in HORIZON_TURNS:
            key = (tmpl_id, turn)
            n = disruption_dists.get(key, {}).get("n", 0)
            sizes.append(f"H{(turn-60)//30}: n={n}")
        print(f"  {tmpl_id:<28} {', '.join(sizes)}")

    # Index weighting results: (model, seed, template_id, turn) -> p_disruption
    weight_idx = {}
    for r in weighting_data:
        if r["p_disruption"] is not None:
            key = (r["model"], r["seed"], r["template_id"], r["resolution_turn"])
            weight_idx[key] = r["p_disruption"] / 100.0  # convert to 0-1

    # Process baseline forecasts
    # Use "baseline" condition (the model's monolithic output without DK)
    # and "domain_knowledge" condition
    for condition_name in ["baseline", "domain_knowledge"]:
        print(f"\n{'='*70}")
        print(f"CONDITION: {condition_name}")
        print(f"{'='*70}")

        condition_results = [r for r in baseline_data
                             if r["condition"] == condition_name
                             and r["percentiles"] is not None]

        # Compute CRPS for original and counterfactual mixture
        results_by_model = defaultdict(lambda: {
            "original_crps": [], "mixture_crps": [],
            "original_by_horizon": defaultdict(list),
            "mixture_by_horizon": defaultdict(list),
        })

        skipped = 0
        processed = 0

        for r in condition_results:
            model = r["model"]
            seed = r["seed"]
            tmpl_id = r["template_id"]
            turn = r["resolution_turn"]
            truth = r["ground_truth"]
            horizon = r.get("horizon", HORIZON_LABELS.get(turn, "?"))

            # Get model's scenario weight
            weight_key = (model, seed, tmpl_id, turn)
            if weight_key not in weight_idx:
                skipped += 1
                continue

            p_disruption = weight_idx[weight_key]

            # Get disruption distribution
            dist_key = (tmpl_id, turn)
            if dist_key not in disruption_dists:
                skipped += 1
                continue

            dis_dist = disruption_dists[dist_key]
            orig_pcts = r["percentiles"]

            # Original CRPS
            orig_crps = crps_quantile(orig_pcts, truth)

            # Counterfactual mixture: linear mix in percentile space
            mixture_pcts = {}
            for pk in ["p10", "p25", "p50", "p75", "p90"]:
                mixture_pcts[pk] = (1 - p_disruption) * orig_pcts[pk] + p_disruption * dis_dist[pk]

            mix_crps = crps_quantile(mixture_pcts, truth)

            results_by_model[model]["original_crps"].append(orig_crps)
            results_by_model[model]["mixture_crps"].append(mix_crps)
            results_by_model[model]["original_by_horizon"][horizon].append(orig_crps)
            results_by_model[model]["mixture_by_horizon"][horizon].append(mix_crps)
            processed += 1

        print(f"Processed {processed} forecasts, skipped {skipped}")

        # ── Per-model summary ──
        print(f"\n## Per-model CRPS: original vs counterfactual mixture")
        print(f"{'Model':<15} {'ECI':>5} {'Orig CRPS':>11} {'Mix CRPS':>10} {'Δ':>8} {'% improve':>10}")
        print("-" * 62)

        model_orig = {}
        model_mix = {}
        for model in sorted(ECI_MAP, key=lambda m: ECI_MAP[m]):
            d = results_by_model[model]
            if not d["original_crps"]:
                continue
            orig = np.mean(d["original_crps"])
            mix = np.mean(d["mixture_crps"])
            delta = mix - orig
            pct = delta / orig * 100
            model_orig[model] = orig
            model_mix[model] = mix
            print(f"{MODEL_SHORT.get(model, model):<15} {ECI_MAP[model]:>5} "
                  f"{orig:>10.1f} {mix:>9.1f} {delta:>+7.1f} {pct:>+9.1f}%")

        # Pooled
        all_orig = []
        all_mix = []
        for d in results_by_model.values():
            all_orig.extend(d["original_crps"])
            all_mix.extend(d["mixture_crps"])
        if all_orig:
            orig_mean = np.mean(all_orig)
            mix_mean = np.mean(all_mix)
            delta = mix_mean - orig_mean
            pct = delta / orig_mean * 100
            print(f"{'POOLED':<15} {'':>5} {orig_mean:>10.1f} {mix_mean:>9.1f} "
                  f"{delta:>+7.1f} {pct:>+9.1f}%")

        # ── Per-horizon breakdown ──
        print(f"\n## Per-horizon CRPS: original vs mixture (pooled across models)")
        print(f"{'Horizon':>8} {'Orig CRPS':>11} {'Mix CRPS':>10} {'Δ':>8} {'% improve':>10}")
        print("-" * 50)

        for turn in sorted(HORIZON_LABELS):
            h = HORIZON_LABELS[turn]
            h_orig = []
            h_mix = []
            for d in results_by_model.values():
                h_orig.extend(d["original_by_horizon"].get(h, []))
                h_mix.extend(d["mixture_by_horizon"].get(h, []))
            if h_orig:
                orig_m = np.mean(h_orig)
                mix_m = np.mean(h_mix)
                delta = mix_m - orig_m
                pct = delta / orig_m * 100 if orig_m > 0 else 0
                print(f"  {h:<6} {orig_m:>10.1f} {mix_m:>9.1f} {delta:>+7.1f} {pct:>+9.1f}%")

        # ── ECI correlation ──
        try:
            from scipy.stats import spearmanr
            ecis = []
            improvements = []
            for model in sorted(model_orig):
                ecis.append(ECI_MAP[model])
                improvements.append(model_mix[model] - model_orig[model])
            if len(ecis) >= 3:
                rho, p = spearmanr(ecis, improvements)
                print(f"\nSpearman ρ(ECI, CRPS improvement) = {rho:+.3f} (p={p:.3f})")
                print(f"  {'More capable models benefit more' if rho < 0 else 'Less capable models benefit more'}")
        except ImportError:
            pass

    # ── KEY FINDING ──
    print(f"\n{'='*70}")
    print("KEY FINDING")
    print(f"{'='*70}")
    print("If the mixture CRPS is lower (negative Δ), the model's own components")
    print("(growth forecast + disruption weight) produce a better distribution when")
    print("combined externally than the model's monolithic output. The integration")
    print("step is the bottleneck.")


if __name__ == "__main__":
    main()
