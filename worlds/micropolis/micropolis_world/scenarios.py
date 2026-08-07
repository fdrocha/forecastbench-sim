"""Scenario sampler + corpus builder."""

from itertools import product

from fbsim_core.questions.resolver import QuestionResolver
from fbsim_core.questions.schema import QuestionInstance

from .city_sim import CitySimulation, to_world
from .report import gen_world_report
from .templates import ALL_TEMPLATES, REGISTRY

# to_world() keys entities by their position in the dict it is passed, so the
# single city in each scenario is always entity 0.
CITY_ENTITY_ID = 0


def get_single_city_base_scenarios(
    seed: int, cities: list[str], disasters: list[bool]
) -> list[CitySimulation]:
    scenarios = []
    for city, has_disasters in product(cities, disasters):
        sim = CitySimulation(city_name=city, seed=seed, disasters=has_disasters)
        scenarios.append(sim)
    return scenarios


def build_corpus(
    scenarios: list[CitySimulation],
    snapshot_turns: list[int],
    horizons: list[int],
    history_freq: int,
) -> list[dict]:
    resolver = QuestionResolver(REGISTRY)
    corpus = []
    nscenarios = len(scenarios)
    # +1 because the furthest question resolves *at* max(snapshot_turns) +
    # max(horizons), and a run of N turns only covers indices 0..N-1.
    nturns = max(snapshot_turns) + max(horizons) + 1
    for i, sim in enumerate(scenarios):
        sim.run_if_needed_and_load(nturns=nturns, quiet=True)

        world = to_world({"city1": sim})

        scenario_id = sim.get_id_str()
        for SNAPSHOT_TURN in snapshot_turns:
            report_text = gen_world_report(
                sim, turn=SNAPSHOT_TURN, history_freq=history_freq
            )
            for H in horizons:
                T = SNAPSHOT_TURN + H
                for template in ALL_TEMPLATES:
                    q_text = template.question_template.format(resolution_turn=T)
                    question_id = (
                        f"{scenario_id}_T{SNAPSHOT_TURN}_H{H}_{template.template_id}"
                    )
                    q = QuestionInstance(
                        question_id=question_id,
                        template_id=template.template_id,
                        resolution_turn=T,
                        horizon=H,
                        parameters={
                            "player_id": CITY_ENTITY_ID,
                        },
                        question_text=q_text,
                    )

                    res = resolver.resolve(q, world, SNAPSHOT_TURN)
                    # A continuous question resolves to a number in value_at_resolution;
                    # .answer is only a bool saying whether any data was found. A missing
                    # entity id or an out-of-range turn silently yields None here, which
                    # would otherwise be indistinguishable from a real result.
                    if res.value_at_resolution is None:
                        raise ValueError(
                            f"{q.question_id}: no value for {template.signal_name} at turn "
                            f"{T} (entity {CITY_ENTITY_ID}, run has {sim.nturns} turns)"
                        )
                    entry = {
                        "question_id": q.question_id,
                        "metric": template.signal_name,
                        "snapshot_turn": SNAPSHOT_TURN,
                        "horizon": H,  # TODO: this should be one of "H0", "H1", ...
                        "scenario_id": scenario_id,
                        "question_text": q_text,
                        "value": res.value_at_resolution,
                        "context": report_text,
                        "scenario": sim.describe(),
                    }

                    corpus.append(entry)
    ntotal_questions = len(corpus)
    questions_per_scenario = ntotal_questions / nscenarios
    print(
        f"\nbuild_corpus: corpus with {ntotal_questions} questions for {nscenarios} scenarios generated ({questions_per_scenario} questions per scenario)."
    )
    return corpus


def build_prompt_continuous(context: str, question_text: str) -> str:
    return (
        context
        + "\n\n"
        + f"QUESTION: {question_text}\n\n"
        + "Your response should be a single number representing your best estimate of the answer, with no other preamble."
    )
