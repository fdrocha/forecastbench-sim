"""Export TRUE minimal-pair flip cases for interp probing.

Builds a genuine minimal pair:
  - BOTH arms show the SAME observed situation report (from the CONTROL world).
  - policy_turn is strictly AFTER the snapshot, so before the snapshot the two
    worlds are the same process; showing the control history in both arms is honest.
  - The ONLY textual difference is the appended intervention sentence.
  - Ground truth: unconditional resolved on the control world (YES, R exceeds S);
    conditional resolved on the intervention world (NO, policy suppresses R).

TODO: straight port of pandemic_world.export_minimal_pairs — revisit field names
once the real Micropolis scenario/metric semantics are implemented.

Usage:
    uv run python -m micropolis_world.export_minimal_pairs \
        --out /path/to/micropolis_minimal_pairs.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from fbsim_core.questions.schema import QuestionInstance, classify_horizon
from fbsim_core.questions.resolver import QuestionResolver

from .templates import REGISTRY, POPULATION_COMPARATIVE
from .city_sim import run_region, to_world
from .report import context_blurb

SNAPSHOT_TURN = 20
HORIZONS = [40, 60]
R_ID, S_ID = 0, 1
R_NAME, S_NAME = "Rivertown", "Bay City"
NAMES = {R_ID: R_NAME, S_ID: S_NAME}


def sample_scenarios(n: int, seed: int) -> list[dict]:
    """policy_turn strictly AFTER the snapshot, so the observed window is policy-free."""
    rng = np.random.default_rng(seed)
    scenarios = []
    for i in range(n):
        scenarios.append(dict(
            scenario_id=i,
            growth_rate_R=float(rng.choice([0.040, 0.050, 0.065])),
            growth_rate_S=float(rng.choice([0.030, 0.040, 0.050])),
            policy_magnitude=float(rng.choice([0.7, 0.9])),
            policy_turn=int(rng.choice([25, 35])),      # both > SNAPSHOT_TURN (20)
            policy_coverage=float(rng.choice([0.6, 0.8])),
            sim_seed=int(rng.integers(1, 10_000)),
        ))
    return scenarios


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-scenarios", type=int, default=80)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--max-cases", type=int, default=15)
    args = ap.parse_args()

    assert all(d > SNAPSHOT_TURN for d in (25, 35)), "policy_turn must be after snapshot"

    resolver = QuestionResolver(REGISTRY)
    scenarios = sample_scenarios(args.n_scenarios, args.seed)

    flip_cases = []
    for sc in scenarios:
        kw = dict(policy_magnitude=sc["policy_magnitude"], policy_turn=sc["policy_turn"],
                  policy_coverage=sc["policy_coverage"], seed=sc["sim_seed"])
        s_series = run_region(sc["growth_rate_S"], False, **kw)
        r_control = run_region(sc["growth_rate_R"], False, **kw)
        r_interv = run_region(sc["growth_rate_R"], True, **kw)
        control_world = to_world({R_ID: r_control, S_ID: s_series}, NAMES)
        interv_world = to_world({R_ID: r_interv, S_ID: s_series}, NAMES)

        # The SHARED observed report: control world, no intervention sentence.
        # Honest because policy_turn > snapshot — policy hasn't acted in the window.
        shared_report = context_blurb(control_world, [R_ID, S_ID], NAMES,
                                      SNAPSHOT_TURN, intervention=None)
        intervention_text = (
            f"{R_NAME} will enact a zoning policy change on turn {sc['policy_turn']} "
            f"(magnitude {sc['policy_magnitude']}, {int(sc['policy_coverage']*100)}% coverage). "
            f"{S_NAME} has no planned intervention."
        )
        cond_report = shared_report + f"\n\nPLANNED INTERVENTION: {intervention_text}"

        for T in HORIZONS:
            q_text = POPULATION_COMPARATIVE.question_template.format(
                region_a=R_NAME, region_b=S_NAME, resolution_turn=T)

            def resolve(world):
                q = QuestionInstance(
                    question_id="q", template_id="population_comparative",
                    resolution_turn=T, horizon=classify_horizon(SNAPSHOT_TURN, T),
                    parameters={"region_a": R_NAME, "region_b": S_NAME,
                                "player_id_a": R_ID, "player_id_b": S_ID},
                    question_text=q_text)
                return resolver.resolve(q, world, SNAPSHOT_TURN)

            gt_u = resolve(control_world).answer    # no policy
            gt_c = resolve(interv_world).answer      # policy
            if gt_u is True and gt_c is False:       # flip case
                flip_cases.append({
                    "scenario_id": sc["scenario_id"], "horizon": T,
                    "question_text": q_text,
                    "context_shared": shared_report,            # identical observed history
                    "context_unconditional": shared_report,     # = shared
                    "context_conditional": cond_report,         # = shared + sentence
                    "gt_unconditional": True, "gt_conditional": False,
                    "scenario": {k: sc[k] for k in
                                 ("growth_rate_R", "growth_rate_S", "policy_magnitude",
                                  "policy_turn", "policy_coverage")},
                })

    flip_cases = flip_cases[: args.max_cases]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "description": "TRUE minimal pairs: identical observed report in both arms "
                       "(control world, policy_turn > snapshot), conditional arm adds ONLY "
                       "the intervention sentence. Isolates the sentence's effect.",
        "snapshot_turn": SNAPSHOT_TURN,
        "n_cases": len(flip_cases),
        "cases": flip_cases,
    }, indent=2))
    print(f"[saved] {len(flip_cases)} true-minimal-pair flip cases -> {out}")
    # Sanity: confirm the two contexts differ ONLY by the appended sentence.
    for fc in flip_cases:
        u, c = fc["context_unconditional"], fc["context_conditional"]
        assert c.startswith(u), "conditional must be unconditional + suffix"
        suffix = c[len(u):].strip()
        assert suffix.startswith("PLANNED INTERVENTION"), suffix[:40]
    print(f"[ok] all {len(flip_cases)} pairs differ ONLY by the appended intervention sentence")
    from collections import Counter
    print("policy_turn distribution:", dict(Counter(fc["scenario"]["policy_turn"] for fc in flip_cases)))


if __name__ == "__main__":
    main()
