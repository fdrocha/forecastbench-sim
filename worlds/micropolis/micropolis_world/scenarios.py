"""Scenario sampler + corpus builder."""

from itertools import product

from fbsim_core.questions.resolver import QuestionResolver
from fbsim_core.questions.schema import QuestionInstance, QuestionTemplate

from .city_sim import CitySimulation, to_world
from .report import context_blurb
from .templates import ALL_TEMPLATES, REGISTRY
from .report import gen_report

# a few chosen Micropolis cities. Not a lot of thought put into the selection: dropped scenarios and a few others
CITY_CHOICES = [
    "bluebird",
    "bruce",
    "deadwood",
    "finnigan",
    "freds",
    "haight",
    "happisle",
    "joffburg",
    "kamakura",
    "kobe",
    "kowloon",
    "kyoto",
    "linecity",
    "senri",
    "southpac",
    "splats",
    "wetcity",
    "yokohama",
][:5]  # TODO remove this after testing

TURNS_PER_YEAR = 4 * 12  # 4 ticks per month, 12 months per year
SNAPSHOT_TURN = TURNS_PER_YEAR * 10
HORIZONS = [TURNS_PER_YEAR * y for y in [1, 2, 5]]
FREQ = 4  # report contains data forevery FREQ turns
TURNS = SNAPSHOT_TURN + max(HORIZONS)


def single_city_base_scenarios(seed: int) -> list[CitySimulation]:
    scenarios = []
    for city, disasters in product(CITY_CHOICES, (True, False)):
        sim = CitySimulation(city_name=city, seed=seed, disasters=disasters)
        scenarios.append(sim)
    return scenarios


def build_corpus(scenarios: list[CitySimulation]) -> list[dict]:
    resolver = QuestionResolver(REGISTRY)
    corpus = []
    for sim in scenarios:
        sim.run_if_needed_and_load(nturns=TURNS, quiet=True)

        world = to_world({"city1": sim})

        report_text = gen_report(world, turn=SNAPSHOT_TURN, history_freq=FREQ)
        scenario_id = sim.get_id_str()
        for H in HORIZONS:
            T = SNAPSHOT_TURN + H
            for template in ALL_TEMPLATES:
                q_text = template.format(resolution_turn=T)

                q = QuestionInstance(
                    question_id=f"{scenario_id}_T{T}",
                    template_id=template.template_id,
                    resolution_turn=T,
                    horizon=H,
                    parameters={},
                    question_text=q_text,
                )

                res = resolver.resolve(q, world, SNAPSHOT_TURN)
                entry = {
                    "question_id": q.question_id,
                    "horizon": T,
                    "scenario_id": scenario_id,
                    "question_text": q_text,
                    "ground_truth": res.answer,
                    "context": report_text,
                    "scenario": sim.describe(),
                }

                corpus.append(entry)
    return corpus
