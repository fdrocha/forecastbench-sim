"""Build the Micropolis domain-knowledge prompts from the statement set.

Shuffles statements.all_statements() with a fixed seed, splits the result into
HALVES prompts so that the two members of a true/false pair never share one,
and appends each half, numbered, to the preamble. Every model is asked every
half; the answers are merged back into the order of the global `statements`.
"""

import asyncio
import base64
import hashlib
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .. import messages as msg
from .. import module_globals as g
from ..model_ids import filename_slug
from ..module_globals import warn_if_truncated
from ..prompting import PromptJob, format_eta, format_latency, run_prompts
from ..usage import CallUsage
from ..usage import load_usage as read_usage
from ..usage import save_usage as write_usage
from .statements import Statement, all_statements

__all__ = ["Statement"]

SHUFFLE_SEED = 20260807

HERE = Path(__file__).parent
PREAMBLE_PATH = HERE / "prompt_preamble.txt"

SUBDIR = "knowledge_eval"


def out_dir() -> Path:
    """This eval's directory under the process's data directory.

    A function, not a constant: the data directory is fixed when a config is
    loaded, after this module was imported.
    """
    return g.DATA_DIR / SUBDIR


def cache_dir() -> Path:
    """Prompts, raw model responses and their usage sidecars.

    Below out_dir(), mirroring the continuous eval's cache layout, so out_dir()
    itself holds only analysis outputs (plots, report, scores).
    """
    return out_dir() / "cache"


class Answer(Enum):
    """A model's parsed verdict on a single statement."""

    TRUE = "True"
    FALSE = "False"
    UNKNOWN = "Unknown"
    UNPARSEABLE = "Unparseable"


def build_statement_list() -> list[Statement]:
    """Return the shuffled Statements, in administered order.

    Honeypot statements are all false, so they carry is_true=False alongside
    is_honeypot=True. The shuffle uses random.Random, whose Mersenne Twister
    stream is identical across machines and Python versions for a given seed.

    Statements are sorted by text before shuffling so the administered order
    depends only on the set of statements, not on their order within the three
    source lists.
    """
    statements = sorted(all_statements(), key=lambda s: s.text)
    random.Random(SHUFFLE_SEED).shuffle(statements)
    return statements


statements = build_statement_list()

# How many prompts each model answers. Two is the least that keeps every
# true/false pair apart; the halves are the same size to within one.
HALVES = 2


def split_halves(stmts: list[Statement], halves: int = HALVES) -> list[list[int]]:
    """Positions of `stmts` for each prompt, the two members of a pair apart.

    A pair's true statement goes to the half its pair index selects and the
    false twin to the next one, so with two halves they never meet. Unpaired
    statements are dealt round-robin in shuffled order, which keeps the halves
    balanced. Deterministic, like the shuffle: the prompts, and so the cache
    hashes, depend only on the statement set.
    """
    out: list[list[int]] = [[] for _ in range(halves)]
    unpaired = 0
    for i, s in enumerate(stmts):
        if s.pair is None:
            out[unpaired % halves].append(i)
            unpaired += 1
        else:
            out[(s.pair + (0 if s.is_true else 1)) % halves].append(i)
    return out


@dataclass(frozen=True)
class Prompt:
    """One of a model's prompts: which statements it carries, its text and hash."""

    index: int  # which half, 0-based
    positions: list[int]  # into the global `statements`, in prompt order
    text: str
    hash: str


def prompt_hash(prompt: str) -> str:
    """Short stable digest of a prompt, used to detect a changed statement set.

    SHA-256 rather than hash(), which is randomized per process and so would
    differ between runs. The digest is urlsafe-base64 encoded and truncated to
    8 characters — enough to spot a changed prompt, and filename-safe.
    """
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")[:8]


def build_prompts() -> list[Prompt]:
    """The HALVES prompts, each numbered from 1, with their hashes.

    Writes nothing: the hash is what identifies a cached response, so the
    read-only paths need to derive it without touching out_dir(). Call
    save_prompt() to persist a prompt's text.
    """
    preamble = PREAMBLE_PATH.read_text(encoding="utf-8")
    prompts = []
    for index, positions in enumerate(split_halves(statements)):
        numbered = [
            f" {i}: {statements[pos].text}" for i, pos in enumerate(positions, start=1)
        ]
        text = preamble + "\n".join(numbered) + "\n"
        prompts.append(Prompt(index, positions, text, prompt_hash(text)))
    return prompts


def merge_answers(parts: dict[int, list[Answer]]) -> list[Answer]:
    """Per-half answers back into the order of `statements`.

    `parts` maps a half's index to the answers parsed from its reply, one per
    position of that half. A half that is missing leaves its statements
    UNPARSEABLE, so a caller can tell a partly answered model from a fully
    answered one by the count.
    """
    merged = [Answer.UNPARSEABLE] * len(statements)
    for prompt in build_prompts():
        answers = parts.get(prompt.index)
        if answers is None:
            continue
        for pos, answer in zip(prompt.positions, answers, strict=True):
            merged[pos] = answer
    return merged


def save_prompt(prompt: str, phash: str) -> None:
    """Write the prompt to prompt-<HASH>.txt in cache_dir().

    Keeps every prompt a response was gathered under on disk alongside it.
    """
    cache_dir().mkdir(parents=True, exist_ok=True)
    (cache_dir() / f"prompt-{phash}.txt").write_text(prompt, encoding="utf-8")


def model_slug(model_id: str) -> str:
    """Model slug flattened into a single filename component.

    The shared rule (model_ids.filename_slug): "/" -> "_", ":" -> "+", so a
    ":suffix" stays recoverable in scoring.scores_by_model_name.
    """
    return filename_slug(model_id)


def response_path(model_id: str, phash: str) -> Path:
    """Path of a model's saved response under a given prompt hash.

    These files are the cache: a response present here is reused rather than
    re-fetched. Including the hash keeps responses from different prompt
    revisions side by side rather than overwriting, and means a changed
    statement set simply misses the cache instead of silently reusing answers
    to different questions.
    """
    return cache_dir() / f"response-{model_slug(model_id)}-{phash}.txt"


def usage_path(model_id: str, phash: str) -> Path:
    """Tokens and cost of the call that produced the matching response file.

    A cached response is never re-fetched, so what it cost has to be recorded
    when it is first paid or it is lost on every later run. The slug must match
    response_path's, or the sidecar lands next to nothing.
    """
    return cache_dir() / f"usage-{model_slug(model_id)}-{phash}.json"


def save_usage(model_id: str, phash: str, usage: CallUsage) -> None:
    """Record one call's tokens and cost beside its cached response."""
    write_usage(usage_path(model_id, phash), usage)


def load_usage(model_id: str, phash: str) -> CallUsage | None:
    """The recorded usage for a cached response, or None if unavailable."""
    return read_usage(usage_path(model_id, phash))


def cached_response(model_id: str, phash: str) -> str | None:
    """The saved response for this model and prompt, or None if not cached.

    A file that is empty or all whitespace is treated as absent: a blank reply
    is never cached, but one could be left behind by an interrupted run.
    """
    path = response_path(model_id, phash)
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8")
    return text if text.strip() else None


def cached_models(phash: str) -> list[str]:
    """Model slugs with a stored response for the given prompt hash.

    Recovered from the response filenames, so this reports what is on disk
    rather than what some separate index claims. Filename slugs are returned,
    not the original ids: the "/" separator is not recoverable from a filename
    (a ":suffix" is, escaped as "+").
    """
    prefix, suffix = "response-", f"-{phash}.txt"
    return sorted(
        p.name[len(prefix) : -len(suffix)]
        for p in cache_dir().glob(f"{prefix}*{suffix}")
        if p.read_text(encoding="utf-8").strip()
    )


def parse_response(text: str | None, count: int | None = None) -> list[Answer]:
    """Parse a model's reply into one Answer per statement, in prompt order.

    `count` is how many statements the prompt carried (a half's size); it
    defaults to the whole set, which is what a single-prompt reply holds.

    The prompt asks for lines of the form "12: False". Anything that doesn't
    yield a recognized verdict for a given statement number — a missing line, a
    duplicate, a hedge like "Probably true" — leaves that slot UNPARSEABLE, so
    the returned list always has one entry per statement.

    Matching is deliberately loose about surrounding punctuation and case
    (models like to write "**12:** False."), but strict about the verdict word
    itself: only True/False/Unknown count.
    """
    count = len(statements) if count is None else count
    answers = [Answer.UNPARSEABLE] * count
    if text is None:
        return answers

    seen = set()
    for line in text.splitlines():
        m = re.match(
            r"^\s*[*_>#\-\s]*(\d+)\s*[*_]*\s*[:.)-]\s*[*_\s]*([A-Za-z]+)", line
        )
        if not m:
            continue
        number, word = int(m.group(1)), m.group(2).lower()
        # Out-of-range numbers are hallucinated statements; ignore them. A
        # repeated number keeps the first answer rather than the last.
        if not 1 <= number <= count or number in seen:
            continue
        verdict = {
            "true": Answer.TRUE,
            "false": Answer.FALSE,
            "unknown": Answer.UNKNOWN,
        }.get(word)
        if verdict is None:
            continue
        seen.add(number)
        answers[number - 1] = verdict

    return answers


def get_model_answers(
    models: list[str],
    concurrency: int | None = None,
) -> dict[str, list[Answer]]:
    """Administer the test to each model, one call per half, and parse the replies.

    The response-<MODEL>-<HASH>.txt files in cache_dir() are the cache and the
    source of truth: a half a model has a stored response for is re-read and
    re-parsed rather than prompted again. Because the filename carries the
    prompt hash, editing the statements simply misses the cache instead of
    reusing answers to different questions.

    The uncached (model, half) calls all run concurrently under run_prompts's
    global cap (`concurrency` overrides it), and each response is written to
    its cache file as it lands.

    Returns a dict of model id -> answers, one per entry of the global
    `statements`, in that order, for the models that answered every half. A
    model whose request fails, or which replies with nothing but whitespace,
    on any half is warned about and left out; the halves it did answer stay
    cached, so a re-run only sends what is missing.

    The output cap is the backend's (model_specs.json5, or the provider
    default); it must leave room for one answer line per statement, since a cap
    that truncates the reply shows up as UNPARSEABLE answers for the tail.
    """
    # Imported here so the scoring-only scripts don't pull in an LLM client.
    from fbsim_core.evaluation.models import get_models

    prompts = build_prompts()
    for prompt in prompts:
        save_prompt(prompt.text, prompt.hash)
        print(
            f"prompt {prompt.index + 1}/{len(prompts)}: {len(prompt.positions)}"
            f" statements, hash {prompt.hash}"
            f" ({cache_dir() / f'prompt-{prompt.hash}.txt'})"
        )

    parts: dict[str, dict[int, list[Answer]]] = {m: {} for m in models}

    def record(model_name: str, prompt: Prompt, raw: str) -> None:
        answers = parse_response(raw, len(prompt.positions))
        counts = Counter(answers)
        print(
            f"  half {prompt.index + 1}: answered "
            + ", ".join(f"{counts[a]} {a.value}" for a in Answer)
            + f" (of {len(prompt.positions)} statements)"
        )
        parts[model_name][prompt.index] = answers

    # Cached halves first: their lines print instantly, so handling them before
    # the fan-out keeps them from interleaving with live completions.
    jobs: list[PromptJob] = []
    for model_name, model in zip(models, get_models(models)):
        for prompt in prompts:
            raw = cached_response(model_name, prompt.hash)
            if raw is not None:
                print(
                    f"{model_name}: re-parsing cached response, half {prompt.index + 1}"
                )
                record(model_name, prompt, raw)
            else:
                jobs.append(
                    PromptJob(
                        key=(model_name, prompt.index),
                        model=model,
                        model_name=model_name,
                        messages=[{"role": "user", "content": prompt.text}],
                    )
                )

    total_cost = 0.0
    nunpriced = 0
    failures: list[tuple[str, int, Exception]] = []

    async def consume() -> None:
        nonlocal total_cost, nunpriced
        # Everything below the API call — prints, cache writes, parsing — runs
        # here in the single consumer task, so nothing needs a lock.
        done = 0
        start = time.perf_counter()
        async for result in run_prompts(jobs, limit=concurrency):
            done += 1
            eta = format_eta(start, done, len(jobs))
            model_name, index = result.job.key
            prompt = prompts[index]
            tag = f"[{done}/{len(jobs)}] {model_name}, half {index + 1}"
            if not result.ok:
                err = result.error
                failures.append((model_name, index, err))
                msg.error(f"{tag}: request FAILED: {type(err).__name__}: {err}{eta}")
                continue
            resp = result.response
            raw = resp.text
            # Usage and how long the call took on the header line even for a
            # blank reply: a model that spent its whole budget thinking still
            # billed for it, and this is the only place that spend is reported
            # — there's no response to cache it beside.
            took = format_latency(resp.usage.latency_ms, resp.retries)
            print(f"{tag}: {resp.usage.describe()}{took}{eta}")
            print(f"  finish_reason: {resp.finish_reason}")
            if resp.usage.cost_usd is None:
                nunpriced += 1
            else:
                total_cost += resp.usage.cost_usd
            warn_if_truncated(model_name, resp.finish_reason)
            if raw is None or not raw.strip():
                msg.warn(f"{model_name} returned a blank response; not caching")
                continue
            out_path = response_path(model_name, prompt.hash)
            # Saved as soon as it lands, so an interrupted run keeps what it
            # already paid for. Usage alongside the response, and only when the
            # response is kept, so the two never disagree about whether this
            # call happened.
            out_path.write_text(raw, encoding="utf-8")
            save_usage(model_name, prompt.hash, resp.usage)
            print(f"  saved response to {out_path}")
            record(model_name, prompt, raw)

    if jobs:
        asyncio.run(consume())
        # What this run paid across all fresh calls; cached halves cost nothing.
        summary = f"this run's {len(jobs)} call(s) cost ${total_cost:.2f}"
        if nunpriced:
            summary += f" + {nunpriced} unpriced call(s)"
        print(summary)
        if failures:
            msg.error(
                f"{len(failures)} call(s) failed (not cached; "
                "re-run this script to retry them):"
            )
            for model_name, index, err in failures:
                msg.plain(
                    f"  {model_name}, half {index + 1}: {type(err).__name__}: {err}"
                )

    data: dict[str, list[Answer]] = {}
    for model_name in models:
        got = parts[model_name]
        if len(got) < len(prompts):
            missing = [str(i + 1) for i in range(len(prompts)) if i not in got]
            msg.warn(
                f"{model_name}: no response for half {', '.join(missing)}; "
                "left out until a re-run fills it"
            )
            continue
        data[model_name] = merge_answers(got)
    return data


def get_cached_answers() -> dict[str, list[Answer]]:
    """Answers for every model with a stored response for every current prompt.

    Reads and parses the response files without prompting anything, so it works
    offline and costs nothing. Keys are filename slugs rather than the original
    provider/name ids, which a filename does not preserve. A model with only
    some halves cached is left out: its score would be over a different set.
    """
    prompts = build_prompts()
    slugs = set.intersection(*(set(cached_models(p.hash)) for p in prompts))
    answers = {}
    for slug in sorted(slugs):
        parts = {}
        for prompt in prompts:
            path = cache_dir() / f"response-{slug}-{prompt.hash}.txt"
            parts[prompt.index] = parse_response(
                path.read_text(encoding="utf-8"), len(prompt.positions)
            )
        answers[slug] = merge_answers(parts)
    return answers
