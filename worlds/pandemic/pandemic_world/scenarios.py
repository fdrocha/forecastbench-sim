"""Scenario sampler + corpus builder.

Each scenario varies R0 (beta) for a policy region R and a comparison region S,
plus vaccine efficacy/day/coverage. Matched questions:
  - unconditional: resolved on the control world (R unvaccinated)
  - conditional:   resolved on the intervention world (R vaccinated)
Both resolved by the core QuestionResolver via the pandemic registry.
"""

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
    rng = np.random.default_rng(seed)
    scenarios = []
    for i in range(n):
        scenarios.append(dict(
            scenario_id=i,
            beta_R=float(rng.choice([0.030, 0.040, 0.050, 0.065])),
            beta_S=float(rng.choice([0.030, 0.040, 0.050, 0.065])),
            vax_efficacy=float(rng.choice([0.5, 0.7, 0.9])),
            vax_day=int(rng.choice([15, 25, 35])),
            vax_coverage=float(rng.choice([0.4, 0.6, 0.8])),
            sim_seed=int(rng.integers(1, 10_000)),
        ))
    return scenarios


def build_corpus(scenarios: list[dict]) -> list[dict]:
    resolver = QuestionResolver(REGISTRY)
    corpus = []
    for sc in scenarios:
        kw = dict(vax_efficacy=sc["vax_efficacy"], vax_day=sc["vax_day"],
                  vax_coverage=sc["vax_coverage"], seed=sc["sim_seed"])
        s_series = run_region(sc["beta_S"], False, **kw)
        r_control = run_region(sc["beta_R"], False, **kw)
        r_interv = run_region(sc["beta_R"], True, **kw)
        control_world = to_world({R_ID: r_control, S_ID: s_series}, NAMES)
        interv_world = to_world({R_ID: r_interv, S_ID: s_series}, NAMES)

        intervention_text = (
            f"{R_NAME} will run a vaccination campaign on day {sc['vax_day']} "
            f"({int(sc['vax_efficacy']*100)}% efficacy, {int(sc['vax_coverage']*100)}% coverage). "
            f"{S_NAME} has no planned intervention."
        )

        for T in HORIZONS:
            q_text = CASES_COMPARATIVE.question_template.format(
                region_a=R_NAME, region_b=S_NAME, resolution_turn=T)

            def mk(world, kind, intervention):
                q = QuestionInstance(
                    question_id=f"s{sc['scenario_id']}_T{T}_{kind}",
                    template_id="cases_comparative", resolution_turn=T,
                    horizon=classify_horizon(SNAPSHOT_DAY, T),
                    parameters={"region_a": R_NAME, "region_b": S_NAME,
                                "player_id_a": R_ID, "player_id_b": S_ID},
                    question_text=q_text)
                res = resolver.resolve(q, world, SNAPSHOT_DAY)
                return {
                    "question_id": q.question_id, "kind": kind, "horizon": T,
                    "scenario_id": sc["scenario_id"], "question_text": q_text,
                    "ground_truth": bool(res.answer),
                    "value_R": res.value_a, "value_S": res.value_b,
                    "context": context_blurb(world, [R_ID, S_ID], NAMES, SNAPSHOT_DAY, intervention),
                    "scenario": {k: sc[k] for k in
                                 ("beta_R", "beta_S", "vax_efficacy", "vax_day", "vax_coverage")},
                }

            corpus.append(mk(control_world, "unconditional", None))
            corpus.append(mk(interv_world, "conditional", intervention_text))
    return corpus
