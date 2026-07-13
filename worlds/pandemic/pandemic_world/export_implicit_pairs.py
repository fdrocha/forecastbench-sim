"""Export implicit-vs-stated minimal pairs (Exp 22).

Tests the PROXIMITY claim from a different angle than Exp 21. There, the vaccine's
consequence was NOT yet in the observed data (vaccine in the future); the stated
sentence cancelled the trend. Here the vaccine has ALREADY acted by the snapshot, so
its consequence IS partly visible in the observed window — but we select cases where
Riverton is STILL ahead of Southbay at the snapshot (the bend is visible but the
current ordering still favors R), and only flips below S by the horizon.

Both arms show the IDENTICAL observed data (the post-vaccine, still-ahead trajectory).
The ONLY difference is whether the cause is labeled:
  - implicit (unconditional arm): bent data, NO mention of a vaccine.
  - stated   (conditional arm):   same data + "A vaccination campaign was administered
                                   on day D (...)."

effect = stated - implicit isolates the value of the causal LABEL holding data fixed.
If proximity matters, the stated label should add NO-commitment (effect negative) even
though the data is identical. If the data already does all the work, effect ~ 0.

Ground truth (both arms): resolved on the same (intervention) world -> the question
"will R exceed S at horizon" is NO (vaccine flips it by the horizon).

Usage:
    uv run python -m pandemic_world.export_implicit_pairs \
        --out /Users/elsehow/Projects/fri-interp-probe/data/starsim_implicit_pairs.json
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
    """vax_day BEFORE the snapshot, so the suppression is partly visible at day 20."""
    rng = np.random.default_rng(seed)
    scenarios = []
    for i in range(n):
        scenarios.append(dict(
            scenario_id=i,
            beta_R=float(rng.choice([0.050, 0.065, 0.080])),   # R grows fast (gets ahead early)
            beta_S=float(rng.choice([0.030, 0.040, 0.050])),
            vax_efficacy=float(rng.choice([0.7, 0.9])),
            vax_day=int(rng.choice([8, 10, 12])),               # all < SNAPSHOT_DAY (20)
            vax_coverage=float(rng.choice([0.6, 0.8])),
            sim_seed=int(rng.integers(1, 10_000)),
        ))
    return scenarios


def cum(series, day):
    return series["cumulative_cases"][day]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-scenarios", type=int, default=120)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--max-cases", type=int, default=15)
    args = ap.parse_args()

    resolver = QuestionResolver(REGISTRY)
    scenarios = sample_scenarios(args.n_scenarios, args.seed)

    cases = []
    for sc in scenarios:
        kw = dict(vax_efficacy=sc["vax_efficacy"], vax_day=sc["vax_day"],
                  vax_coverage=sc["vax_coverage"], seed=sc["sim_seed"])
        s_series = run_region(sc["beta_S"], False, **kw)
        r_interv = run_region(sc["beta_R"], True, **kw)   # vaccinated R (bend visible)
        interv_world = to_world({R_ID: r_interv, S_ID: s_series}, NAMES)

        # Both arms use the SAME (intervention-world) observed data.
        implicit_report = context_blurb(interv_world, [R_ID, S_ID], NAMES,
                                        SNAPSHOT_DAY, intervention=None)
        intervention_text = (
            f"A vaccination campaign was administered in {R_NAME} on day {sc['vax_day']} "
            f"({int(sc['vax_efficacy']*100)}% efficacy, {int(sc['vax_coverage']*100)}% coverage). "
            f"{S_NAME} received no vaccine."
        )
        stated_report = implicit_report + f"\n\nINTERVENTION IN PROGRESS: {intervention_text}"

        for T in HORIZONS:
            # Interesting regime: R still AHEAD at snapshot, but BELOW S by horizon.
            r_snap, s_snap = cum(r_interv, SNAPSHOT_DAY), cum(s_series, SNAPSHOT_DAY)
            r_T, s_T = cum(r_interv, T), cum(s_series, T)
            if not (r_snap > s_snap and r_T < s_T):
                continue

            q_text = CASES_COMPARATIVE.question_template.format(
                region_a=R_NAME, region_b=S_NAME, resolution_turn=T)
            q = QuestionInstance(
                question_id="q", template_id="cases_comparative",
                resolution_turn=T, horizon=classify_horizon(SNAPSHOT_DAY, T),
                parameters={"region_a": R_NAME, "region_b": S_NAME,
                            "player_id_a": R_ID, "player_id_b": S_ID},
                question_text=q_text)
            gt = resolver.resolve(q, interv_world, SNAPSHOT_DAY).answer  # NO (flips by T)
            cases.append({
                "scenario_id": sc["scenario_id"], "horizon": T,
                "question_text": q_text,
                "context_unconditional": implicit_report,   # implicit (no label)
                "context_conditional": stated_report,        # stated (+ label)
                "gt_unconditional": bool(gt), "gt_conditional": bool(gt),  # same world
                "r_snapshot": r_snap, "s_snapshot": s_snap, "r_horizon": r_T, "s_horizon": s_T,
                "scenario": {k: sc[k] for k in
                             ("beta_R", "beta_S", "vax_efficacy", "vax_day", "vax_coverage")},
            })

    cases = cases[: args.max_cases]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "description": "Implicit-vs-stated minimal pairs (Exp 22): identical observed "
                       "(post-vaccine, still-ahead) data in both arms; conditional adds "
                       "ONLY a causal label naming the vaccine. effect=stated-implicit "
                       "isolates the label's value at fixed data. Selected: R ahead of S "
                       "at snapshot, R below S by horizon.",
        "snapshot_day": SNAPSHOT_DAY, "n_cases": len(cases), "cases": cases,
    }, indent=2))
    print(f"[saved] {len(cases)} implicit-vs-stated pairs -> {out}")
    for c in cases:
        u = c["context_unconditional"]; cc = c["context_conditional"]
        assert cc.startswith(u) and "INTERVENTION IN PROGRESS" in cc[len(u):]
        print(f"  s{c['scenario_id']} T{c['horizon']}: R {c['r_snapshot']:.0f}->{c['r_horizon']:.0f} "
              f"S {c['s_snapshot']:.0f}->{c['s_horizon']:.0f}  "
              f"(vax d{c['scenario']['vax_day']} eff{c['scenario']['vax_efficacy']})")
    print(f"[ok] all pairs differ only by the appended causal label")


if __name__ == "__main__":
    main()
