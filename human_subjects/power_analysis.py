#!/usr/bin/env python3
"""Power analysis for TailRiskBench human subjects pilot study.

Design:
- N=30 participants (fixed)
- Within-subject: horizon (H1, H3, H4, H6) × disruptability (disruptable, non-disruptable)
- Binned elicitation (10-bin distributional forecasts → CRPS)
- Question: how many worlds (seeds) do we need?

Key research questions:
1. How hard are these questions for humans? (estimate mean CRPS with CI)
2. Human vs LLM performance (compare CRPS distributions)
3. Disruptable harder than non-disruptable? Longer horizon harder? (2×2 within-subject)

Calibration from LLM binned elicitation data (normalized CRPS = CRPS / mean|GT|):
- Disruptable near (H1,H3):  mean=0.25, sd=0.17
- Disruptable far  (H4,H6):  mean=0.50, sd=0.25
- Non-disruptable near:       mean=0.15, sd=0.13
- Non-disruptable far:        mean=0.49, sd=0.30
(These are LLM baselines; humans may be better or worse.)
"""

import numpy as np
from scipy import stats


def simulate_power(
    n_participants: int,
    n_worlds: int,
    n_questions_per_cell: int,
    effect_sizes: dict,
    n_sims: int = 5000,
    alpha: float = 0.05,
) -> dict:
    """Simulate power for the 2×2 within-subject design.

    Each participant answers questions in a 2 (disruptable vs not) × 2 (near vs far) design.
    We test three contrasts via paired t-tests on participant-level cell means:
      1. Main effect of disruptability (disruptable - non-disruptable)
      2. Main effect of horizon (far - near)
      3. Interaction (does the disruptable gap grow at far horizons?)

    Parameters
    ----------
    n_participants : number of people
    n_worlds : number of game seeds each person sees
    n_questions_per_cell : questions per participant per cell (after subsetting)
        With full templates: disruptable has 4 templates × 5 civs × 2 horizons per near/far = 40
        non-disruptable has 2 templates × 5 civs × 2 horizons = 20
        But we may subset to keep survey manageable.
    effect_sizes : dict with keys 'disruptable_mean', 'disruptable_sd',
        'nondisruptable_mean', 'nondisruptable_sd' for near and far conditions.
    n_sims : number of simulations
    alpha : significance level
    """
    rng = np.random.default_rng(42)

    # Unpack effect sizes (normalized CRPS)
    d_near_mu = effect_sizes["disruptable_near_mean"]
    d_near_sd = effect_sizes["disruptable_near_sd"]
    d_far_mu = effect_sizes["disruptable_far_mean"]
    d_far_sd = effect_sizes["disruptable_far_sd"]
    nd_near_mu = effect_sizes["nondisruptable_near_mean"]
    nd_near_sd = effect_sizes["nondisruptable_near_sd"]
    nd_far_mu = effect_sizes["nondisruptable_far_mean"]
    nd_far_sd = effect_sizes["nondisruptable_far_sd"]

    # Between-participant SD (how much people vary in forecasting ability)
    # Assume moderate individual differences — ~0.5× the within-cell SD
    participant_sd = 0.08

    # Between-world SD (games differ in difficulty)
    # From LLM data: seed is the dominant variance source
    world_sd = 0.10

    sig_disruptability = 0
    sig_horizon = 0
    sig_interaction = 0

    for _ in range(n_sims):
        # Random participant effects
        participant_effects = rng.normal(0, participant_sd, n_participants)

        # Each participant sees n_worlds worlds
        # Random world effects (shared across participants who see that world)
        world_effects = rng.normal(0, world_sd, n_worlds)

        # For each participant, compute cell means
        cell_means = np.zeros((n_participants, 4))  # d_near, d_far, nd_near, nd_far

        for p in range(n_participants):
            p_effect = participant_effects[p]

            for w in range(n_worlds):
                w_effect = world_effects[w]

                # Sample question-level CRPS for each cell
                # n_questions_per_cell questions per world per cell
                base = p_effect + w_effect

                # Disruptable near
                d_near_qs = rng.normal(
                    d_near_mu + base, d_near_sd, n_questions_per_cell
                )
                # Disruptable far
                d_far_qs = rng.normal(
                    d_far_mu + base, d_far_sd, n_questions_per_cell
                )
                # Non-disruptable near
                nd_near_qs = rng.normal(
                    nd_near_mu + base, nd_near_sd, n_questions_per_cell
                )
                # Non-disruptable far
                nd_far_qs = rng.normal(
                    nd_far_mu + base, nd_far_sd, n_questions_per_cell
                )

                cell_means[p, 0] += np.mean(np.clip(d_near_qs, 0, None))
                cell_means[p, 1] += np.mean(np.clip(d_far_qs, 0, None))
                cell_means[p, 2] += np.mean(np.clip(nd_near_qs, 0, None))
                cell_means[p, 3] += np.mean(np.clip(nd_far_qs, 0, None))

            # Average across worlds
            cell_means[p] /= n_worlds

        # Contrasts (paired t-tests on participant-level means)
        # 1. Disruptability: (d_near + d_far)/2 - (nd_near + nd_far)/2
        disruptable_avg = (cell_means[:, 0] + cell_means[:, 1]) / 2
        nondisruptable_avg = (cell_means[:, 2] + cell_means[:, 3]) / 2
        _, p_disrupt = stats.ttest_rel(disruptable_avg, nondisruptable_avg)
        if p_disrupt < alpha:
            sig_disruptability += 1

        # 2. Horizon: (d_far + nd_far)/2 - (d_near + nd_near)/2
        far_avg = (cell_means[:, 1] + cell_means[:, 3]) / 2
        near_avg = (cell_means[:, 0] + cell_means[:, 2]) / 2
        _, p_horizon = stats.ttest_rel(far_avg, near_avg)
        if p_horizon < alpha:
            sig_horizon += 1

        # 3. Interaction: (d_far - nd_far) - (d_near - nd_near)
        interaction = (cell_means[:, 1] - cell_means[:, 3]) - (
            cell_means[:, 0] - cell_means[:, 2]
        )
        _, p_interact = stats.ttest_1samp(interaction, 0)
        if p_interact < alpha:
            sig_interaction += 1

    return {
        "power_disruptability": sig_disruptability / n_sims,
        "power_horizon": sig_horizon / n_sims,
        "power_interaction": sig_interaction / n_sims,
    }


def main():
    # LLM-calibrated effect sizes (normalized CRPS)
    # Using LLM means as baseline — humans might differ but this gives scale
    effect_sizes = {
        "disruptable_near_mean": 0.25,
        "disruptable_near_sd": 0.17,
        "disruptable_far_mean": 0.50,
        "disruptable_far_sd": 0.25,
        "nondisruptable_near_mean": 0.15,
        "nondisruptable_near_sd": 0.13,
        "nondisruptable_far_mean": 0.49,
        "nondisruptable_far_sd": 0.30,
    }

    n_participants = 30

    print("=" * 72)
    print("POWER ANALYSIS: TailRiskBench Human Subjects Pilot")
    print("=" * 72)
    print(f"\nN participants = {n_participants}")
    print(f"Design: 2 (disruptable/non-disruptable) × 2 (near H1,H3 / far H4,H6)")
    print(f"Metric: normalized CRPS (CRPS / mean|GT|)")
    print()

    print("LLM-calibrated cell means (normalized CRPS):")
    print(f"                    Near (H1,H3)   Far (H4,H6)")
    print(
        f"  Disruptable:        {effect_sizes['disruptable_near_mean']:.2f}            {effect_sizes['disruptable_far_mean']:.2f}"
    )
    print(
        f"  Non-disruptable:    {effect_sizes['nondisruptable_near_mean']:.2f}            {effect_sizes['nondisruptable_far_mean']:.2f}"
    )
    print(
        f"  Δ (disrupt effect): {effect_sizes['disruptable_near_mean'] - effect_sizes['nondisruptable_near_mean']:.2f}            {effect_sizes['disruptable_far_mean'] - effect_sizes['nondisruptable_far_mean']:.2f}"
    )
    print()

    # Questions per cell per world:
    # Full design: disruptable has 4 templates × 5 civs = 20 per horizon-pair (near or far)
    # non-disruptable has 2 templates × 5 civs = 10 per horizon-pair
    # Use min (10) as the per-cell count for balanced analysis
    # But we might subset further for survey length

    print("-" * 72)
    print("SCENARIO A: Each person sees all templates, all civs")
    print("  Questions per cell per world: 10 (non-disruptable) to 20 (disruptable)")
    print("  Using 10 per cell (balanced on smaller group)")
    print("-" * 72)
    print(f"{'Worlds':>6} | {'Q/person':>8} | {'Disruptability':>14} | {'Horizon':>8} | {'Interaction':>11}")
    print("-" * 72)

    for n_worlds in [1, 2, 3, 4, 5]:
        # Full templates: 10 qs per cell per world (non-disruptable side)
        # Total per person: 4 cells × 10 qs × n_worlds = 40 × n_worlds
        # (Plus extra 10 per world for disruptable having more templates)
        qs_per_person = n_worlds * (20 + 20 + 10 + 10)  # d_near + d_far + nd_near + nd_far per world
        result = simulate_power(
            n_participants=n_participants,
            n_worlds=n_worlds,
            n_questions_per_cell=10,
            effect_sizes=effect_sizes,
        )
        print(
            f"{n_worlds:>6} | {qs_per_person:>8} | {result['power_disruptability']:>13.0%} | {result['power_horizon']:>7.0%} | {result['power_interaction']:>10.0%}"
        )

    print()
    print("-" * 72)
    print("SCENARIO B: Subset to 1 civ per template (reduce survey length)")
    print("  Questions per cell per world: 2 (non-disruptable) to 4 (disruptable)")
    print("  Using 2 per cell (balanced on smaller group)")
    print("-" * 72)
    print(f"{'Worlds':>6} | {'Q/person':>8} | {'Disruptability':>14} | {'Horizon':>8} | {'Interaction':>11}")
    print("-" * 72)

    for n_worlds in [2, 3, 4, 5, 8, 10]:
        qs_per_person = n_worlds * (4 + 4 + 2 + 2)
        result = simulate_power(
            n_participants=n_participants,
            n_worlds=n_worlds,
            n_questions_per_cell=2,
            effect_sizes=effect_sizes,
        )
        print(
            f"{n_worlds:>6} | {qs_per_person:>8} | {result['power_disruptability']:>13.0%} | {result['power_horizon']:>7.0%} | {result['power_interaction']:>10.0%}"
        )

    print()
    print("-" * 72)
    print("SCENARIO C: 2 civs per template (moderate survey length)")
    print("  Questions per cell per world: 4 (non-disruptable) to 8 (disruptable)")
    print("  Using 4 per cell")
    print("-" * 72)
    print(f"{'Worlds':>6} | {'Q/person':>8} | {'Disruptability':>14} | {'Horizon':>8} | {'Interaction':>11}")
    print("-" * 72)

    for n_worlds in [1, 2, 3, 4, 5]:
        qs_per_person = n_worlds * (8 + 8 + 4 + 4)
        result = simulate_power(
            n_participants=n_participants,
            n_worlds=n_worlds,
            n_questions_per_cell=4,
            effect_sizes=effect_sizes,
        )
        print(
            f"{n_worlds:>6} | {qs_per_person:>8} | {'power_disruptability':>14} | {'power_horizon':>8} | {'power_interaction':>11}"
            if False
            else f"{n_worlds:>6} | {qs_per_person:>8} | {result['power_disruptability']:>13.0%} | {result['power_horizon']:>7.0%} | {result['power_interaction']:>10.0%}"
        )

    print()
    print("=" * 72)
    print("NOTES")
    print("=" * 72)
    print("""
- Power targets: ≥80% for main effects, interaction is exploratory
- "Near" = H1+H3, "Far" = H4+H6
- Calibrated from LLM normalized CRPS (humans may differ in absolute level
  but relative contrasts should be similar or larger)
- Survey time estimate: ~1-2 min per binned elicitation question
- The interaction (disruptable gap growing at far horizons) is WEAK in LLM
  data (d≈0.16) because non-disruptable errors also explode at far horizons.
  We likely cannot detect this with N=30 — that's fine for a pilot.
- Key pilot goals: (1) estimate human CRPS with CIs, (2) detect main effects
  of horizon and disruptability if they exist.
""")


if __name__ == "__main__":
    main()
