"""Export TRUE minimal-pair flip cases for interp probing (fixes Exp 20 confound).

Exp 20's "minimal pair" was not minimal: the conditional prompt showed a
different observed trajectory (vaccine already acted by the snapshot, and/or RNG
divergence between the control and intervention sims). The L45 YES->NO flip could
not be attributed to the intervention SENTENCE vs. the model reading an
already-divergent comparison off the data.

This builds a genuine minimal pair:
  - BOTH arms show the SAME observed situation report (from the CONTROL world).
  - vax_day is strictly AFTER the snapshot, so before the snapshot the two worlds
    are the same process; showing the control history in both arms is honest.
  - The ONLY textual difference is the appended intervention sentence.
  - Ground truth: unconditional resolved on the control world (YES, R exceeds S);
    conditional resolved on the intervention world (NO, vaccine suppresses R).

The logit lens on this pair isolates the effect of the intervention sentence
alone — apples-to-apples with the FreeCiv crash prompt where the crash must be
inferred.

Usage:
    uv run python -m pandemic_world.export_minimal_pairs \
        --out /Users/elsehow/Projects/fri-interp-probe/data/starsim_minimal_pairs.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

from fbsim_core.questions.schema import QuestionInstance, classify_horizon
from fbsim_core.questions.resolver import QuestionResolver

from .templates import REGISTRY, CASES_COMPARATIVE
from .runner import run_region, to_world
from .report import context_blurb

SNAPSHOT_DAY = 20
HORIZONS = [40, 60]
R_ID, S_ID = 0, 1
R_NAME, S_NAME = "Riverton", "Southbay"
NAMES = {R_ID: R_NAME, S_ID: S_NAME}


def sample_scenarios(n: int, seed: int) -> list[dict]:
    """vax_day strictly AFTER the snapshot, so the observed window is vaccine-free."""
    rng = np.random.default_rng(seed)
    scenarios = []
    for i in range(n):
        scenarios.append(dict(
            scenario_id=i,
            beta_R=float(rng.choice([0.040, 0.050, 0.065])),
            beta_S=float(rng.choice([0.030, 0.040, 0.050])),
            vax_efficacy=float(rng.choice([0.7, 0.9])),
            vax_day=int(rng.choice([25, 35])),      # both > SNAPSHOT_DAY (20)
            vax_coverage=float(rng.choice([0.6, 0.8])),
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

    assert all(d > SNAPSHOT_DAY for d in (25, 35)), "vax_day must be after snapshot"

    resolver = QuestionResolver(REGISTRY)
    scenarios = sample_scenarios(args.n_scenarios, args.seed)

    flip_cases = []
    for sc in scenarios:
        kw = dict(vax_efficacy=sc["vax_efficacy"], vax_day=sc["vax_day"],
                  vax_coverage=sc["vax_coverage"], seed=sc["sim_seed"])
        s_series = run_region(sc["beta_S"], False, **kw)
        r_control = run_region(sc["beta_R"], False, **kw)
        r_interv = run_region(sc["beta_R"], True, **kw)
        control_world = to_world({R_ID: r_control, S_ID: s_series}, NAMES)
        interv_world = to_world({R_ID: r_interv, S_ID: s_series}, NAMES)

        # The SHARED observed report: control world, no intervention sentence.
        # Honest because vax_day > snapshot — vaccine hasn't acted in the window.
        shared_report = context_blurb(control_world, [R_ID, S_ID], NAMES,
                                      SNAPSHOT_DAY, intervention=None)
        intervention_text = (
            f"{R_NAME} will run a vaccination campaign on day {sc['vax_day']} "
            f"({int(sc['vax_efficacy']*100)}% efficacy, {int(sc['vax_coverage']*100)}% coverage). "
            f"{S_NAME} has no planned intervention."
        )
        cond_report = shared_report + f"\n\nPLANNED INTERVENTION: {intervention_text}"

        for T in HORIZONS:
            q_text = CASES_COMPARATIVE.question_template.format(
                region_a=R_NAME, region_b=S_NAME, resolution_turn=T)

            def resolve(world):
                q = QuestionInstance(
                    question_id="q", template_id="cases_comparative",
                    resolution_turn=T, horizon=classify_horizon(SNAPSHOT_DAY, T),
                    parameters={"region_a": R_NAME, "region_b": S_NAME,
                                "player_id_a": R_ID, "player_id_b": S_ID},
                    question_text=q_text)
                return resolver.resolve(q, world, SNAPSHOT_DAY)

            gt_u = resolve(control_world).answer    # no vaccine
            gt_c = resolve(interv_world).answer      # vaccine
            if gt_u is True and gt_c is False:       # flip case
                flip_cases.append({
                    "scenario_id": sc["scenario_id"], "horizon": T,
                    "question_text": q_text,
                    "context_shared": shared_report,            # identical observed history
                    "context_unconditional": shared_report,     # = shared
                    "context_conditional": cond_report,         # = shared + sentence
                    "gt_unconditional": True, "gt_conditional": False,
                    "scenario": {k: sc[k] for k in
                                 ("beta_R", "beta_S", "vax_efficacy", "vax_day", "vax_coverage")},
                })

    flip_cases = flip_cases[: args.max_cases]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "description": "TRUE minimal pairs: identical observed report in both arms "
                       "(control world, vax_day > snapshot), conditional arm adds ONLY "
                       "the intervention sentence. Isolates the sentence's effect.",
        "snapshot_day": SNAPSHOT_DAY,
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
    print("vax_day distribution:", dict(Counter(fc["scenario"]["vax_day"] for fc in flip_cases)))


if __name__ == "__main__":
    main()
