"""Scenario sampler + corpus builder."""

from itertools import product

from fbsim_core.questions.resolver import QuestionResolver
from fbsim_core.questions.schema import QuestionInstance

from .city_sim import CitySimulation, to_world
from .report import gen_world_report
from .templates import ALL_TEMPLATES, REGISTRY

# a few chosen Micropolis cities. Not a lot of thought put into the selection: dropped scenarios and a few others
CITY_CHOICES = [
    #    "bluebird", this is a dead city with no population, nothing happens
    "bruce",
    #    "deadwood", crashes the engine (WASM "memory access out of bounds")
    #    partway through a run when disasters are enabled, at every seed tried
    "finnigan",
    "freds",
    "haight",
    "happisle",
    "joffburg",
    "kamakura",
    #    "kobe",
    "kowloon",
    "kyoto",
    "linecity",
    "senri",
    "southpac",
    "splats",
    "wetcity",
    #    "yokohama",
]

TURNS_PER_YEAR = 4 * 12  # 4 ticks per month, 12 months per year
SNAPSHOT_TURN = TURNS_PER_YEAR * 10
HORIZONS = [TURNS_PER_YEAR * y for y in [1, 2, 5]]
FREQ = 4  # report contains data forevery FREQ turns
# +1 because the furthest question resolves *at* turn SNAPSHOT_TURN + max(HORIZONS),
# and a run of N turns only covers indices 0..N-1.
TURNS = SNAPSHOT_TURN + max(HORIZONS) + 1

# to_world() keys entities by their position in the dict it is passed, so the
# single city in each scenario is always entity 0.
CITY_ENTITY_ID = 0


def get_single_city_base_scenarios(seed: int) -> list[CitySimulation]:
    scenarios = []
    for city, disasters in product(CITY_CHOICES, (True, False)):
        sim = CitySimulation(city_name=city, seed=seed, disasters=disasters)
        scenarios.append(sim)
    return scenarios


def build_corpus(scenarios: list[CitySimulation]) -> list[dict]:
    resolver = QuestionResolver(REGISTRY)
    corpus = []
    nscenarios = len(scenarios)
    for i, sim in enumerate(scenarios):
        print(
            f"build_corpus: Running {(i + 1)}/{nscenarios} scenario: {sim.get_id_str()}",
            end="              \r",
        )
        sim.run_if_needed_and_load(nturns=TURNS, quiet=True)

        world = to_world({"city1": sim})

        report_text = gen_world_report(sim, turn=SNAPSHOT_TURN, history_freq=FREQ)
        scenario_id = sim.get_id_str()
        for H in HORIZONS:
            T = SNAPSHOT_TURN + H
            for template in ALL_TEMPLATES:
                q_text = template.question_template.format(resolution_turn=T)

                q = QuestionInstance(
                    question_id=f"{scenario_id}_H{H}_{template.template_id}",
                    template_id=template.template_id,
                    resolution_turn=T,
                    horizon=H,
                    parameters={"player_id": CITY_ENTITY_ID},
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
                    "horizon": T,  # TODO: this should be one of "H0", "H1", ...
                    "scenario_id": scenario_id,
                    "question_text": q_text,
                    "value": res.value_at_resolution,
                    "context": report_text,
                    "scenario": sim.describe(),
                }

                corpus.append(entry)
    print(
        f"\nbuild_corpus: corpus with {len(corpus)} questions for {nscenarios} scenarios generated."
    )
    return corpus
