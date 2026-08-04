"""Scenario sampler + corpus builder.

TODO: mirror pandemic_world.scenarios — vary a scenario parameter (e.g. growth
rate) for a policy region R and a comparison region S, plus an intervention
(e.g. zoning change) magnitude/timing/coverage. Matched questions:
  - unconditional: resolved on the control world (R without the policy)
  - conditional:   resolved on the intervention world (R with the policy)
Both resolved by the core QuestionResolver via the micropolis registry.
"""

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
    """TODO: replace placeholder parameter ranges with real Micropolis scenario knobs."""
    rng = np.random.default_rng(seed)
    scenarios = []
    for i in range(n):
        scenarios.append(dict(
            scenario_id=i,
            growth_rate_R=float(rng.choice([0.030, 0.040, 0.050, 0.065])),
            growth_rate_S=float(rng.choice([0.030, 0.040, 0.050, 0.065])),
            policy_magnitude=float(rng.choice([0.5, 0.7, 0.9])),
            policy_turn=int(rng.choice([15, 25, 35])),
            policy_coverage=float(rng.choice([0.4, 0.6, 0.8])),
            sim_seed=int(rng.integers(1, 10_000)),
        ))
    return scenarios


def build_corpus(scenarios: list[dict]) -> list[dict]:
    """TODO: mirror pandemic_world.scenarios.build_corpus once run_region is implemented."""
    resolver = QuestionResolver(REGISTRY)
    corpus = []
    for sc in scenarios:
        kw = dict(policy_magnitude=sc["policy_magnitude"], policy_turn=sc["policy_turn"],
                  policy_coverage=sc["policy_coverage"], seed=sc["sim_seed"])
        s_series = run_region(sc["growth_rate_S"], False, **kw)
        r_control = run_region(sc["growth_rate_R"], False, **kw)
        r_interv = run_region(sc["growth_rate_R"], True, **kw)
        control_world = to_world({R_ID: r_control, S_ID: s_series}, NAMES)
        interv_world = to_world({R_ID: r_interv, S_ID: s_series}, NAMES)

        intervention_text = (
            f"{R_NAME} will enact a zoning policy change on turn {sc['policy_turn']} "
            f"(magnitude {sc['policy_magnitude']}, {int(sc['policy_coverage']*100)}% coverage). "
            f"{S_NAME} has no planned intervention."
        )

        for T in HORIZONS:
            q_text = POPULATION_COMPARATIVE.question_template.format(
                region_a=R_NAME, region_b=S_NAME, resolution_turn=T)

            def mk(world, kind, intervention):
                q = QuestionInstance(
                    question_id=f"s{sc['scenario_id']}_T{T}_{kind}",
                    template_id="population_comparative", resolution_turn=T,
                    horizon=classify_horizon(SNAPSHOT_TURN, T),
                    parameters={"region_a": R_NAME, "region_b": S_NAME,
                                "player_id_a": R_ID, "player_id_b": S_ID},
                    question_text=q_text)
                res = resolver.resolve(q, world, SNAPSHOT_TURN)
                return {
                    "question_id": q.question_id, "kind": kind, "horizon": T,
                    "scenario_id": sc["scenario_id"], "question_text": q_text,
                    "ground_truth": bool(res.answer),
                    "value_R": res.value_a, "value_S": res.value_b,
                    "context": context_blurb(world, [R_ID, S_ID], NAMES, SNAPSHOT_TURN, intervention),
                    "scenario": {k: sc[k] for k in
                                 ("growth_rate_R", "growth_rate_S", "policy_magnitude",
                                  "policy_turn", "policy_coverage")},
                }

            corpus.append(mk(control_world, "unconditional", None))
            corpus.append(mk(interv_world, "conditional", intervention_text))
    return corpus
