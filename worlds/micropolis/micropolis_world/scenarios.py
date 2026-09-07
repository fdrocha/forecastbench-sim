"""Scenario sampler + corpus builder."""

import json
import re
from functools import cache
from itertools import pairwise, product
from pathlib import Path

from fbsim_core.questions.resolver import QuestionResolver
from fbsim_core.questions.schema import QuestionInstance

from . import module_globals as g
from .city_sim import CitySimulation, to_world
from .config import (
    DATAFILES_DIR,
    QUESTION_TAGGING_NUMERIC,
    QUESTION_TAGGING_SEMANTIC,
    QUESTION_TAGGINGS,
    QUESTIONS_SORT_TURN,
    QUESTIONS_SORTS,
)
from .report import gen_world_report
from .templates import REGISTRY, asked_templates

# The quantiles elicited for every continuous question, matching FreeCiv.
PERCENTILE_KEYS = ["p10", "p25", "p50", "p75", "p90"]

# to_world() keys entities by their position in the dict it is passed, so the
# single city in each scenario is always entity 0.
CITY_ENTITY_ID = 0

# The preamble a run uses when its config names no "preamble_path". Held as a
# file rather than a string literal so a prompt variant is a new file plus one
# config key, with no code change; the text must contain "{sources}", which
# prompt_preamble fills in.
DEFAULT_PREAMBLE_PATH = DATAFILES_DIR / "preamble1.txt"

# The epilogue a run uses when its config names no "epilogue_path". Held as a
# file for the same reason as the preamble; the text may contain "{n}", which
# read_epilogue fills in with the number of questions in the batch.
DEFAULT_EPILOGUE_PATH = DATAFILES_DIR / "epilogue1.txt"

# The epilogue a semantic-tagging run uses when its config names no
# "epilogue_path": the default one shows "Q1:" answer lines, which are not what
# a semantic prompt asks for.
DEFAULT_SEMANTIC_EPILOGUE_PATH = DATAFILES_DIR / "epilogue2.txt"

# Separates a semantic question's tag from its text. An em dash rather than a
# hyphen so it cannot be confused with a minus sign in the question itself.
SEMANTIC_TAG_SEPARATOR = " — "

# Defaults for the binary yes/no eval's prompt, mirroring the continuous pair.
DEFAULT_BINARY_PREAMBLE_PATH = DATAFILES_DIR / "preamble-binary.txt"
DEFAULT_BINARY_EPILOGUE_PATH = DATAFILES_DIR / "epilogue-binary.txt"

# The answer-block marker a binary response is asked for; accepts the singular
# form too, matching FreeCiv's parser tolerance.
_PROBABILITIES_MARKER_RE = r"PROBABILIT(?:Y|IES)"


def semantic_tag(metric: str, resolution_turn: int) -> str:
    """The "<metric label>@<turn>" tag naming one question in semantic mode.

    Built here rather than at each call site so the tag written into the prompt
    and the tag the parser matches against can never drift apart.
    """
    return f"{g.METRIC_LABELS[metric]}@{resolution_turn}"


@cache
def read_preamble(path: Path | str | None = None) -> str:
    """The raw preamble template at `path`, or the default one when None.

    Cached because the corpus and eval scripts ask for the same preamble once
    per batch, and because a config's key is read afresh at each call site.
    """
    return Path(path or DEFAULT_PREAMBLE_PATH).read_text(encoding="utf-8")


@cache
def _read_epilogue_template(path: Path | str | None = None) -> str:
    """The raw epilogue template at `path`, or the default one when None."""
    return Path(path or DEFAULT_EPILOGUE_PATH).read_text(encoding="utf-8")


def read_epilogue(n: int, path: Path | str | None = None) -> str:
    """The epilogue at `path` with "{n}" replaced by the question count.

    A plain str.replace rather than str.format so an epilogue that never
    mentions the count — or one that contains braces of its own, such as a
    JSON answer template — is returned untouched instead of raising.
    """
    return _read_epilogue_template(path).replace("{n}", str(n))


def get_base_scenarios(
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
    history_length: int = -1,
    report_effectiveness: bool = False,
    censor_city_funds: bool = True,
    questions_sort: str = QUESTIONS_SORT_TURN,
) -> list[dict]:
    """One question per (scenario, snapshot turn, horizon, metric) combination.

    `censor_city_funds` (the default) both hides the city's money from the
    world report and drops the city funds question from the corpus, so the
    metric is neither reported nor asked about. False asks about all of
    templates.Q_METRICS and reports the balance, which is what the prompt
    variants gathered before the flag existed were run with.

    `questions_sort` picks the order the questions come out in, and so the
    order they are numbered in each batch prompt: QUESTIONS_SORT_TURN asks
    every metric for one horizon before moving to the next horizon, while
    QUESTIONS_SORT_METRIC asks every horizon for one metric first. Both ask
    exactly the same questions; only the numbering differs, and since that
    order is part of the prompt text, switching hashes to a new cache entry
    rather than mixing variants.
    """
    if questions_sort not in QUESTIONS_SORTS:
        raise ValueError(
            f"questions_sort must be one of {QUESTIONS_SORTS}, got {questions_sort!r}"
        )
    resolver = QuestionResolver(REGISTRY)
    templates = asked_templates(censor_city_funds)
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
                history_length=history_length,
                report_effectiveness=report_effectiveness,
                censor_city_funds=censor_city_funds,
            )
            # Both orders ask the same questions; only the order they are
            # numbered in the prompt differs, so the pairs are generated once
            # and re-ordered rather than duplicating the loop body.
            pairs = (
                [(H, t) for H in horizons for t in templates]
                if questions_sort == QUESTIONS_SORT_TURN
                else [(H, t) for t in templates for H in horizons]
            )
            for H, template in pairs:
                T = SNAPSHOT_TURN + H
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
                    # Carried on the entry so the prompt builder and the
                    # response parser derive a question's semantic tag from
                    # the same place; unused when tagging is numeric.
                    "semantic_tag": semantic_tag(template.signal_name, T),
                    "resolution_turn": T,
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
    context: str,
    questions: list[dict],
    preamble_path: Path | str | None = None,
    epilogue_path: Path | str | None = None,
    question_tagging: str = QUESTION_TAGGING_NUMERIC,
) -> str:
    """Ask for one p10/p25/p50/p75/p90 quantile forecast per question.

    Every question in `questions` shares the game report in `context`, so the
    report — which dominates the token cost — is included once and the
    questions are numbered. The instruction wording and the delimited answer
    block match FreeCiv's build_continuous_batch_prompt, so responses from the
    two worlds are parsed the same way and scored on the same CRPS.
    parse_batch_percentiles reads the answers back.

    `question_tagging` picks how each question is labeled, and so how its
    answer is matched back to it: QUESTION_TAGGING_NUMERIC numbers them "1.",
    "2.", ...; QUESTION_TAGGING_SEMANTIC prefixes each with its
    "<metric label>@<turn>" tag instead, which lets a model answer out of order
    without its answers sliding onto the wrong questions.

    `preamble_path` and `epilogue_path` name the templates wrapping the report
    and questions, defaulting to DEFAULT_PREAMBLE_PATH and — since the default
    epilogue asks for the numeric answer format — to whichever epilogue matches
    `question_tagging`. All three are part of the prompt, so changing any of
    them misses the response cache rather than mixing variants.
    """
    if question_tagging not in QUESTION_TAGGINGS:
        raise ValueError(
            f"question_tagging must be one of {QUESTION_TAGGINGS}, "
            f"got {question_tagging!r}"
        )
    n = len(questions)
    if question_tagging == QUESTION_TAGGING_SEMANTIC:
        listed_questions = "\n".join(
            f"{q['semantic_tag']}{SEMANTIC_TAG_SEPARATOR}{q['question_text']}"
            for q in questions
        )
        if epilogue_path is None:
            epilogue_path = DEFAULT_SEMANTIC_EPILOGUE_PATH
    else:
        listed_questions = "\n".join(
            f"{i}. {q['question_text']}" for i, q in enumerate(questions, 1)
        )
    return f"""{read_preamble(preamble_path)}

## Game report
{context}

## Questions
{listed_questions}

{read_epilogue(n, epilogue_path)}"""


def build_batch_prompt_binary(
    context: str,
    questions: list[dict],
    preamble_path: Path | str | None = None,
    epilogue_path: Path | str | None = None,
) -> str:
    """Ask for one P(Yes) per binary question, sharing one game report.

    The binary counterpart of build_batch_prompt_continuous: same skeleton,
    numeric tagging only, and the binary preamble/epilogue defaults. Answers
    come back as a <<<PROBABILITIES>>> block of "Q1: 0.65" lines, read by
    parse_batch_probabilities.
    """
    listed_questions = "\n".join(
        f"{i}. {q['question_text']}" for i, q in enumerate(questions, 1)
    )
    return f"""{read_preamble(preamble_path or DEFAULT_BINARY_PREAMBLE_PATH)}

## Game report
{context}

## Questions
{listed_questions}

{read_epilogue(len(questions), epilogue_path or DEFAULT_BINARY_EPILOGUE_PATH)}"""


def _at(source: str | Path | None) -> str:
    """The trailing " <- /path/to/response.txt" a parse warning ends with.

    Empty when the caller did not name a source, so a warning from a test or
    an ad-hoc call still reads correctly.
    """
    return f" <- {source}" if source else ""


def _validate_monotonic(
    percentiles: dict[str, float],
    label: str,
    quiet: bool,
    source: str | Path | None = None,
) -> dict[str, float] | None:
    """Return the percentiles if non-decreasing, else warn and return None.

    A quantile function cannot decrease, so p10 > p25 (etc.) means the model
    returned something that isn't a distribution. Scoring it anyway would pass
    CRPS a nonsensical forecast and quietly reward or punish the model for it,
    so the forecast is dropped the same way an unparseable one is.
    """
    values = [percentiles[k] for k in PERCENTILE_KEYS]
    if any(a > b for a, b in pairwise(values)):
        if not quiet:
            pairs = ", ".join(f"{k}={percentiles[k]:g}" for k in PERCENTILE_KEYS)
            print(
                f"  {label}: percentiles not in increasing order, "
                f"discarding: {pairs}{_at(source)}"
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


def _extract_answer_block(response: str, marker_re: str = r"PERCENTILES?") -> str:
    """The part of a response holding the estimates.

    The delimited <<<{marker}>>> block is the requested format and the most
    reliable, so prefer its contents; `marker_re` names it (percentiles by
    default, probabilities for the binary eval). Models sometimes emit only
    the closing tag, having written the answers as ordinary prose above it, so
    fall back to everything before a lone <<<END>>> and finally to the whole
    response.

    The *last* delimited block wins, not the first. A reasoning model often
    restates the requested format mid-thought ("Format:\n<<<PERCENTILES>>>\nQ1:
    p10=X, ...\n<<<END>>>"), and taking the first match hands the parser that
    placeholder instead of the real answers below it — every question then
    reads as unanswered. This is the same convention the numbered-line passes
    use, where a later restatement overwrites an earlier one.
    """
    matches = list(
        re.finditer(
            rf"<<<{marker_re}>>>(.*?)<<<END>>>", response, re.DOTALL | re.IGNORECASE
        )
    )
    # The lone-<<<END>>> fallback captures everything before the closing tag,
    # so its last match is the one spanning the most text, not the least.
    delimiter_match = (
        matches[-1]
        if matches
        else re.search(r"(.*)<<<END>>>", response, re.DOTALL | re.IGNORECASE)
    )
    return delimiter_match.group(1).strip() if delimiter_match else response


def parse_percentiles(
    response: str | None,
    label: str = "response",
    quiet: bool = False,
    source: str | Path | None = None,
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
            print(f"  {label}: empty model response{_at(source)}")
        return None

    content = _extract_answer_block(response)

    # JSON object, or the first object inside a JSON array.
    json_match = re.search(r"\{.*?\}", content, re.DOTALL)
    if json_match:
        try:
            val = json.loads(json_match.group())
            result = {k: float(val[k]) for k in PERCENTILE_KEYS if k in val}
            if len(result) == len(PERCENTILE_KEYS):
                return _validate_monotonic(result, label, quiet, source)
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    result = _scan_labeled_percentiles(content)
    if result is not None:
        return _validate_monotonic(result, label, quiet, source)

    # Last resort: five bare numbers on one line, in ascending percentile order.
    for line in content.strip().split("\n"):
        bare = _scan_bare_percentiles(line)
        if bare is not None:
            return _validate_monotonic(bare, label, quiet, source)

    if not quiet:
        print(
            f"  {label}: unable to parse percentiles from "
            f"model response: {response!r}{_at(source)}"
        )
    return None


# A line answering one question of a batch, e.g. "Q3: p10=...", "3. p10=...",
# or "Question 3) ...". Anchored to the line start: a question number mentioned
# mid-sentence is prose, not an answer.
_QUESTION_NUMBER_RE = re.compile(
    r"^\s*(?:question\s*|q)?(\d+)\s*[.:)]\s*", re.IGNORECASE
)


def _normalize_tag(tag: str) -> str:
    """A semantic tag reduced to what it must match on.

    Case and internal spacing are the model's to vary — "Average Crime @ 288"
    names the same question as "average crime@288" — so both sides of the
    comparison are folded before matching. Nothing else is stripped: a tag that
    names a different metric or turn must not collide with this one.
    """
    return re.sub(r"\s+", "", tag).lower()


# A semantic answer line: everything up to the first colon is the tag, the rest
# holds the percentiles. Anchored to the line start, like _QUESTION_NUMBER_RE,
# so a tag mentioned mid-sentence in the reasoning is prose, not an answer. The
# tag is matched against the prompt's tags rather than parsed, so a metric label
# containing a space, an "@", or a digit needs no special handling here.
_SEMANTIC_TAG_RE = re.compile(r"^\s*([^:]+?)\s*:\s*")


def parse_batch_percentiles_semantic(
    response: str | None,
    labels: list[str],
    tags: list[str],
    quiet: bool = False,
    source: str | Path | None = None,
) -> list[dict[str, float] | None]:
    """Extract one p10..p90 set per question from a semantically tagged response.

    The semantic counterpart to parse_batch_percentiles: `tags` are the
    questions' "<metric label>@<turn>" tags in prompt order, and an answer line
    is matched to a question by its own tag rather than by position, so a model
    that answers out of order, or skips a question, cannot shift the answers
    after it onto the wrong questions. Later lines overwrite earlier ones for
    the same tag, so an answer restated in a final block wins over one
    mentioned in the reasoning above it.

    A line whose tag matches no question is ignored rather than guessed at —
    there is no positional fallback, since a response that ignored the tag
    format gives no trustworthy way to tell which question it meant. Each set
    is validated by _validate_monotonic, and every question left without a
    usable one gets a warning naming its label (unless `quiet`).
    Every warning ends with `source`, the response file the text came from, so a rejection in a long run can be opened directly.
    """
    n = len(labels)
    results: list[dict[str, float] | None] = [None] * n
    if not response:
        if not quiet:
            print(f"  {labels[0]} (+{n - 1} more): empty model response{_at(source)}")
        return results

    # Built per call rather than cached: the same tag can only appear once in a
    # batch, so the last index wins and duplicates cannot silently shadow.
    by_tag = {_normalize_tag(t): i for i, t in enumerate(tags)}
    answered = [False] * n
    for line in _extract_answer_block(response).split("\n"):
        if not line.strip():
            continue
        m = _SEMANTIC_TAG_RE.match(line)
        if not m:
            continue
        idx = by_tag.get(_normalize_tag(m.group(1)))
        if idx is None:
            continue
        parsed = _scan_labeled_percentiles(line[m.end() :])
        if parsed is not None:
            answered[idx] = True
            results[idx] = _validate_monotonic(parsed, labels[idx], quiet, source)

    if not quiet:
        # _validate_monotonic already explained the answered-but-invalid ones.
        for i in range(n):
            if not answered[i]:
                print(
                    f"  {labels[i]}: no percentiles found in "
                    f"batched response{_at(source)}"
                )
    return results


def parse_batch_percentiles(
    response: str | None,
    labels: list[str],
    quiet: bool = False,
    source: str | Path | None = None,
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
    Every warning ends with `source`, the response file the text came from, so a rejection in a long run can be opened directly.
    """
    n = len(labels)
    # A single-question prompt asks for the unnumbered single-question format,
    # so read it back with the single-question parser, which also accepts
    # formats (JSON, whole-block scans) that would be ambiguous in a batch.
    if n == 1:
        return [
            parse_percentiles(response, label=labels[0], quiet=quiet, source=source)
        ]

    results: list[dict[str, float] | None] = [None] * n
    if not response:
        if not quiet:
            print(f"  {labels[0]} (+{n - 1} more): empty model response{_at(source)}")
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
            results[idx] = _validate_monotonic(parsed, labels[idx], quiet, source)

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
                results[pos] = _validate_monotonic(parsed, labels[pos], quiet, source)
                pos += 1

    if not quiet:
        # _validate_monotonic already explained the answered-but-invalid ones.
        for i in range(n):
            if not answered[i]:
                print(
                    f"  {labels[i]}: no percentiles found in "
                    f"batched response{_at(source)}"
                )
    return results


# A probability at the start of a string: "0.65", ".65", "1", "65%", "65 %".
# Anchored so a number buried in trailing prose ("fewer than 1 in 10") is not
# read as the answer, while "0.65 (earthquake unlikely)" still is.
_PROB_TOKEN_RE = re.compile(r"^\s*(\d+(?:\.\d+)?|\.\d+)\s*(%?)")


def _scan_probability(text: str) -> float | None:
    """The number at the start of `text` read as a probability, or None.

    An explicit "%" divides by 100, so "65%" and "0.65" mean the same thing.
    The range is not checked here — _validate_probability does that, so the
    caller can tell "no answer on this line" from "an answer worth warning
    about".
    """
    m = _PROB_TOKEN_RE.match(text)
    if not m:
        return None
    value = float(m.group(1))
    return value / 100 if m.group(2) else value


def _validate_probability(
    value: float, label: str, quiet: bool, source: str | Path | None = None
) -> float | None:
    """Return `value` if it is in [0, 1], else warn and return None.

    A bare number outside the range is rejected rather than clamped — the
    binary counterpart of _validate_monotonic's convention that an answer
    which isn't a valid forecast is dropped, not repaired.
    """
    if 0.0 <= value <= 1.0:
        return value
    if not quiet:
        print(
            f"  {label}: probability {value:g} outside [0, 1], discarding{_at(source)}"
        )
    return None


def parse_probability(
    response: str | None,
    label: str = "response",
    quiet: bool = False,
    source: str | Path | None = None,
) -> float | None:
    """Extract one P(Yes) from a single-question model response.

    Reads the first line of the <<<PROBABILITIES>>> answer block (or its
    fallbacks, see _extract_answer_block) that starts with a number, with or
    without a "Q1:"-style prefix. Out-of-range values are rejected by
    _validate_probability; either rejection warns naming `label` unless
    `quiet`.
    """
    if not response:
        if not quiet:
            print(f"  {label}: empty model response{_at(source)}")
        return None
    content = _extract_answer_block(response, _PROBABILITIES_MARKER_RE)
    for line in content.split("\n"):
        # A bare decimal like "0.65" also matches the number-prefix pattern
        # (as item "0." followed by "65"), so only strip a prefix that names
        # the one question being asked.
        m = _QUESTION_NUMBER_RE.match(line)
        text = line[m.end() :] if m and m.group(1) == "1" else line
        value = _scan_probability(text)
        if value is not None:
            return _validate_probability(value, label, quiet, source)
    if not quiet:
        print(
            f"  {label}: unable to parse a probability from "
            f"response: {response!r}{_at(source)}"
        )
    return None


def parse_batch_probabilities(
    response: str | None,
    labels: list[str],
    quiet: bool = False,
    source: str | Path | None = None,
) -> list[float | None]:
    """Extract one P(Yes) per question from a batched model response.

    The binary counterpart of parse_batch_percentiles, with the same mapping
    rules: the requested format is one "Q<n>: 0.65" line per question inside a
    <<<PROBABILITIES>>> block, and those lines are mapped by their stated
    number, not their position, so a model that skips a question can't shift
    every answer after it. Later lines overwrite earlier ones for the same
    number. An answer written on one line as "Q1: 0.15, Q2: 0.05, ..." is split
    back apart first. When no line carries a question number, lines that are
    nothing but a single probability are assigned positionally in order of
    appearance.
    Out-of-range values are rejected by _validate_probability, and every
    question left without a usable answer gets a warning naming its label
    (unless `quiet`). Every warning ends with `source`, the response file the text came from, so a rejection in a long run can be opened directly.
    """
    n = len(labels)
    # A single-question prompt asks for the unnumbered single-question format,
    # so read it back with the single-question parser.
    if n == 1:
        return [
            parse_probability(response, label=labels[0], quiet=quiet, source=source)
        ]

    results: list[float | None] = [None] * n
    if not response:
        if not quiet:
            print(f"  {labels[0]} (+{n - 1} more): empty model response{_at(source)}")
        return results

    block = _extract_answer_block(response, _PROBABILITIES_MARKER_RE)
    # Some models put the whole answer on one line, "Q1: 0.15, Q2: 0.05, ...".
    # Break before each question number so those become answer lines too; the
    # split is harmless on properly line-separated answers.
    block = re.sub(
        r"[,;]\s*(?=(?:question\s*|q)?\d+\s*[.:)])", "\n", block, flags=re.IGNORECASE
    )
    lines = [line for line in block.split("\n") if line.strip()]

    # First pass: numbered answer lines, mapped by their stated number. The
    # probability must start right after the number prefix, so a numbered
    # prose sentence in the reasoning is not an answer.
    answered = [False] * n
    for line in lines:
        m = _QUESTION_NUMBER_RE.match(line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if not 0 <= idx < n:
            continue
        value = _scan_probability(line[m.end() :])
        if value is not None:
            answered[idx] = True
            results[idx] = _validate_probability(value, labels[idx], quiet, source)

    # Positional fallback, only when nothing was numbered: each line that is
    # nothing but one probability answers the next question in order. A line
    # with any letters is prose, not an answer.
    if not any(answered):
        pos = 0
        for line in lines:
            if pos >= n:
                break
            if re.search(r"[a-zA-Z]", line):
                continue
            value = _scan_probability(line)
            if value is not None:
                answered[pos] = True
                results[pos] = _validate_probability(value, labels[pos], quiet, source)
                pos += 1

    if not quiet:
        # _validate_probability already explained the answered-but-invalid ones.
        for i in range(n):
            if not answered[i]:
                print(
                    f"  {labels[i]}: no probability found in "
                    f"batched response{_at(source)}"
                )
    return results
