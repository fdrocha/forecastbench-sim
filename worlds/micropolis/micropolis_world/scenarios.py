"""Scenario sampler + corpus builder."""

import json
import re
from itertools import product

from fbsim_core.questions.resolver import QuestionResolver
from fbsim_core.questions.schema import QuestionInstance

from .city_sim import CitySimulation, to_world
from .report import gen_world_report
from .templates import ALL_TEMPLATES, REGISTRY

# The quantiles elicited for every continuous question, matching FreeCiv.
PERCENTILE_KEYS = ["p10", "p25", "p50", "p75", "p90"]

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
    """Ask for a p10/p25/p50/p75/p90 quantile forecast.

    The instruction wording and the delimited answer block match FreeCiv's
    build_continuous_batch_prompt (single-question variant), so responses from
    the two worlds are parsed the same way and scored on the same CRPS.
    """
    return (
        context
        + "\n\n"
        + f"QUESTION: {question_text}\n\n"
        + """You MUST provide percentile estimates UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, provide reasonable mid-range estimates, but always return numeric percentile values.

You may analyze the data, but you MUST end your response with your percentile estimates in this exact format:
<<<PERCENTILES>>>
p10=5, p25=10, p50=15, p75=20, p90=25
<<<END>>>

Replace the example values with your actual percentile estimates.
- p10 means you estimate there's a 10% chance the true value is below this number
- p25 means you estimate there's a 25% chance the true value is below this number
- p50 (median) means you estimate there's a 50% chance the true value is below this number
- p75 means you estimate there's a 75% chance the true value is below this number
- p90 means you estimate there's a 90% chance the true value is below this number"""
    )


def _validate_monotonic(
    percentiles: dict[str, float], label: str, quiet: bool
) -> dict[str, float] | None:
    """Return the percentiles if non-decreasing, else warn and return None.

    A quantile function cannot decrease, so p10 > p25 (etc.) means the model
    returned something that isn't a distribution. Scoring it anyway would pass
    CRPS a nonsensical forecast and quietly reward or punish the model for it,
    so the forecast is dropped the same way an unparseable one is.
    """
    values = [percentiles[k] for k in PERCENTILE_KEYS]
    if any(a > b for a, b in zip(values, values[1:])):
        if not quiet:
            pairs = ", ".join(f"{k}={percentiles[k]:g}" for k in PERCENTILE_KEYS)
            print(f"  {label}: percentiles not in increasing order, discarding: {pairs}")
        return None
    return percentiles


def parse_percentiles(
    response: str | None, label: str = "response", quiet: bool = False
) -> dict[str, float] | None:
    """Extract one p10/p25/p50/p75/p90 set from a model response.

    Mirrors FreeCiv's parse_batch_percentiles for the single-question case, and
    accepts the same range of formats: the delimited <<<PERCENTILES>>> block, a
    JSON object/array, "p10=..., p25=..." on a line, or — only as a last resort
    — the first five bare numbers on a line.

    Returns None if no complete set of five percentiles could be read, or if the
    five aren't in non-decreasing order. A partial set is never returned, since
    CRPS needs all five. Either rejection prints a warning naming `label`, unless
    `quiet` is set — used when re-parsing cached responses, whose rejections have
    been reported already.
    """
    if not response:
        if not quiet:
            print(f"  {label}: empty model response")
        return None

    # The delimited block is the requested format and the most reliable, so
    # prefer its contents. Models sometimes emit only the closing tag, having
    # written the percentiles as ordinary prose above it, so fall back to
    # everything before a lone <<<END>>> and finally to the whole response.
    delimiter_match = re.search(
        r"<<<PERCENTILES?>>>(.*?)<<<END>>>", response, re.DOTALL | re.IGNORECASE
    ) or re.search(r"(.*?)<<<END>>>", response, re.DOTALL | re.IGNORECASE)
    content = delimiter_match.group(1).strip() if delimiter_match else response

    # JSON object, or the first object inside a JSON array.
    json_match = re.search(r"\{.*?\}", content, re.DOTALL)
    if json_match:
        try:
            val = json.loads(json_match.group())
            result = {k: float(val[k]) for k in PERCENTILE_KEYS if k in val}
            if len(result) == len(PERCENTILE_KEYS):
                return _validate_monotonic(result, label, quiet)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # Labeled percentiles: "p10=5, p25=10, ..." all on one line, or one per line
    # ("p10=0\np25=0\n..."), or bulleted ("- p10=35"). Scanning the whole content
    # rather than line by line covers all three, since models split the block
    # however they like. The leading (?:^|[^a-zA-Z]) keeps the "p" from matching
    # inside a word such as "pop10".
    result = {}
    for key_digits, value in re.findall(
        r"(?:^|[^a-zA-Z])p(\d+)\s*[=:]\s*(-?[\d,]*\.?\d+)", content, re.IGNORECASE
    ):
        key = f"p{key_digits}"
        if key in PERCENTILE_KEYS:
            try:
                # Last write wins: a model that discusses "p50" in its reasoning
                # before stating it in the final block should be read from the
                # block, which comes last.
                result[key] = float(value.replace(",", ""))
            except ValueError:
                pass
    if len(result) == len(PERCENTILE_KEYS):
        return _validate_monotonic(result, label, quiet)

    # Last resort: five bare numbers on one line, in ascending percentile order.
    # A separate dict from the labeled scan above, whose partial results must not
    # leak into this one.
    for line in content.strip().split("\n"):
        numbers = re.findall(r"-?\d+\.?\d*", line)
        if len(numbers) >= len(PERCENTILE_KEYS):
            try:
                bare = {k: float(n) for k, n in zip(PERCENTILE_KEYS, numbers)}
            except ValueError:
                continue
            return _validate_monotonic(bare, label, quiet)

    if not quiet:
        print(f"  {label}: unable to parse percentiles from model response: {response!r}")
    return None
