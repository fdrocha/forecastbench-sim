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

# What the report gives the model to reason from. The snapshot-only variant has
# no HISTORY table (see report.gen_world_report), so naming "trends" there would
# point at something absent.
_WITH_HISTORY_SOURCES = "the trends, events, and current state"
_SNAPSHOT_ONLY_SOURCES = "the events and current state"

PROMPT_PREAMBLE = """You are an expert superforecaster, familiar with the work of Tetlock and others. Your task is to forecast the evolution of a city in Micropolis, the open-source release of the original SimCity Classic simulation engine.

The city below is running with no mayor: nothing is built, demolished, or rezoned, no tax or funding rates are changed, and no disasters are triggered manually — the simulation engine simply advances on its own from the state described. Disasters can be enabled or disabled and this will be stated in the game report.
Base your forecasts primarily on {sources} given in the game report, using your knowledge of the game's mechanics only to interpret and extrapolate what the report shows. Give calibrated
percentiles that honestly reflect your uncertainty, remembering that an unmanaged city may continue on its current trajectory, plateau, or decline.
Remember that if disasters are enabled, their random occurrence can shift the trajectory abruptly.

"""


def prompt_preamble(snapshot_only_report: bool = False) -> str:
    """The preamble, naming only the report sections the variant actually has."""
    return PROMPT_PREAMBLE.format(
        sources=(
            _SNAPSHOT_ONLY_SOURCES if snapshot_only_report else _WITH_HISTORY_SOURCES
        )
    )


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
    label: str,
    snapshot_only_report: bool = False,
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
                sim,
                turn=SNAPSHOT_TURN,
                history_freq=history_freq,
                label=label,
                snapshot_only=snapshot_only_report,
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


def build_batch_prompt_continuous(
    context: str, questions: list[dict], snapshot_only_report: bool = False
) -> str:
    """Ask for one p10/p25/p50/p75/p90 quantile forecast per question.

    Every question in `questions` shares the game report in `context`, so the
    report — which dominates the token cost — is included once and the
    questions are numbered. The instruction wording and the delimited answer
    block match FreeCiv's build_continuous_batch_prompt, so responses from the
    two worlds are parsed the same way and scored on the same CRPS.
    parse_batch_percentiles reads the answers back.

    `snapshot_only_report` says whether `context` was built without its HISTORY
    table, which only changes which report sections the preamble points the
    model at (see prompt_preamble).
    """
    n = len(questions)
    if n == 1:
        question_section = f"## Question\n{questions[0]['question_text']}"
        must_answer = "You MUST provide percentile estimates UNDER ALL CIRCUMSTANCES."
        format_example = "p10=5, p25=10, p50=15, p75=20, p90=25"
        format_hint = (
            "Replace the example values with your actual percentile estimates."
        )
    else:
        numbered = "\n".join(
            f"{i}. {q['question_text']}" for i, q in enumerate(questions, 1)
        )
        question_section = f"## Questions\n{numbered}"
        must_answer = (
            "You MUST provide percentile estimates for every question "
            "UNDER ALL CIRCUMSTANCES."
        )
        format_example = (
            "Q1: p10=5, p25=10, p50=15, p75=20, p90=25\n"
            "Q2: p10=100, p25=200, p50=300, p75=400, p90=500"
        )
        format_hint = (
            f"Provide one such line for each of the {n} questions, in order, "
            "replacing the example values with your actual percentile estimates."
        )
    return f"""{prompt_preamble(snapshot_only_report)}

## Game report
{context}

{question_section}

{must_answer} If for some reason you can't answer, provide reasonable mid-range estimates, but always return numeric percentile values.

You may analyze the data, but you MUST end your response with your percentile estimates in this exact format:
<<<PERCENTILES>>>
{format_example}
<<<END>>>

{format_hint}
- p10 means you estimate there's a 10% chance the true value is below this number
- p25 means you estimate there's a 25% chance the true value is below this number
- p50 (median) means you estimate there's a 50% chance the true value is below this number
- p75 means you estimate there's a 75% chance the true value is below this number
- p90 means you estimate there's a 90% chance the true value is below this number"""


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
            print(
                f"  {label}: percentiles not in increasing order, discarding: {pairs}"
            )
        return None
    return percentiles


def _scan_labeled_percentiles(text: str) -> dict[str, float] | None:
    """Read one full labeled set ("p10=5, p25=10, ...") out of `text`.

    The labels can be split across lines or bulleted; scanning the whole text
    covers all of it. The leading (?:^|[^a-zA-Z]) keeps the "p" from matching
    inside a word such as "pop10". Later occurrences of a key overwrite earlier
    ones — a model that discusses "p50" in its reasoning before stating it in
    its final answer should be read from the answer, which comes last. Returns
    None unless all five keys were found; a partial set is useless to CRPS.
    """
    result = {}
    for key_digits, value in re.findall(
        r"(?:^|[^a-zA-Z])p(\d+)\s*[=:]\s*(-?[\d,]*\.?\d+)", text, re.IGNORECASE
    ):
        key = f"p{key_digits}"
        if key in PERCENTILE_KEYS:
            try:
                result[key] = float(value.replace(",", ""))
            except ValueError:
                pass
    return result if len(result) == len(PERCENTILE_KEYS) else None


def _scan_bare_percentiles(text: str) -> dict[str, float] | None:
    """Read five bare numbers as p10..p90 in order, or None if fewer."""
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if len(numbers) < len(PERCENTILE_KEYS):
        return None
    try:
        return {k: float(n) for k, n in zip(PERCENTILE_KEYS, numbers)}
    except ValueError:
        return None


def _extract_answer_block(response: str) -> str:
    """The part of a response holding the percentile estimates.

    The delimited <<<PERCENTILES>>> block is the requested format and the most
    reliable, so prefer its contents. Models sometimes emit only the closing
    tag, having written the percentiles as ordinary prose above it, so fall
    back to everything before a lone <<<END>>> and finally to the whole
    response.
    """
    delimiter_match = re.search(
        r"<<<PERCENTILES?>>>(.*?)<<<END>>>", response, re.DOTALL | re.IGNORECASE
    ) or re.search(r"(.*?)<<<END>>>", response, re.DOTALL | re.IGNORECASE)
    return delimiter_match.group(1).strip() if delimiter_match else response


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

    content = _extract_answer_block(response)

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

    result = _scan_labeled_percentiles(content)
    if result is not None:
        return _validate_monotonic(result, label, quiet)

    # Last resort: five bare numbers on one line, in ascending percentile order.
    for line in content.strip().split("\n"):
        bare = _scan_bare_percentiles(line)
        if bare is not None:
            return _validate_monotonic(bare, label, quiet)

    if not quiet:
        print(
            f"  {label}: unable to parse percentiles from model response: {response!r}"
        )
    return None


# A line answering one question of a batch, e.g. "Q3: p10=...", "3. p10=...",
# or "Question 3) ...". Anchored to the line start: a question number mentioned
# mid-sentence is prose, not an answer.
_QUESTION_NUMBER_RE = re.compile(
    r"^\s*(?:question\s*|q)?(\d+)\s*[.:)]\s*", re.IGNORECASE
)


def parse_batch_percentiles(
    response: str | None, labels: list[str], quiet: bool = False
) -> list[dict[str, float] | None]:
    """Extract one p10..p90 set per question from a batched model response.

    `labels` name the questions in warnings — one per question, in prompt
    order — and their count is the number of answers expected. The requested
    format is one "Q<n>: p10=..., ..." line per question; those lines are
    mapped by their stated number, not their position, so a model that skips
    a question can't shift every answer after it. When no line carries a
    question number, full percentile sets are assigned positionally in order
    of appearance instead. Each set is validated by _validate_monotonic like
    a single-question one, and every question left without a usable set gets
    a warning naming its label (unless `quiet`).
    """
    n = len(labels)
    # A single-question prompt asks for the unnumbered single-question format,
    # so read it back with the single-question parser, which also accepts
    # formats (JSON, whole-block scans) that would be ambiguous in a batch.
    if n == 1:
        return [parse_percentiles(response, label=labels[0], quiet=quiet)]

    results: list[dict[str, float] | None] = [None] * n
    if not response:
        if not quiet:
            print(f"  {labels[0]} (+{n - 1} more): empty model response")
        return results

    lines = [
        line for line in _extract_answer_block(response).split("\n") if line.strip()
    ]

    # First pass: numbered answer lines, mapped by their stated number. Later
    # lines overwrite earlier ones for the same number, so an answer restated
    # in a final block wins over one mentioned in the reasoning above it.
    answered = [False] * n
    for line in lines:
        m = _QUESTION_NUMBER_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if not 0 <= idx < n:
            continue
        rest = line[m.end() :]
        parsed = _scan_labeled_percentiles(rest)
        if parsed is None and not re.search(r"[a-zA-Z]", rest):
            # Bare numbers only count when the line is nothing but numbers
            # ("Q1: 5, 10, 15, 20, 25") — a numbered prose sentence in the
            # reasoning can easily contain five figures without being an answer.
            parsed = _scan_bare_percentiles(rest)
        if parsed is not None:
            answered[idx] = True
            results[idx] = _validate_monotonic(parsed, labels[idx], quiet)

    # Positional fallback, only when nothing was numbered: each line holding a
    # full labeled set answers the next question in order. Not tried after a
    # partial numbered parse, where "the next question" would be a guess.
    if not any(answered):
        pos = 0
        for line in lines:
            if pos >= n:
                break
            parsed = _scan_labeled_percentiles(line)
            if parsed is not None:
                answered[pos] = True
                results[pos] = _validate_monotonic(parsed, labels[pos], quiet)
                pos += 1

    if not quiet:
        # _validate_monotonic already explained the answered-but-invalid ones.
        for i in range(n):
            if not answered[i]:
                print(f"  {labels[i]}: no percentiles found in batched response")
    return results
