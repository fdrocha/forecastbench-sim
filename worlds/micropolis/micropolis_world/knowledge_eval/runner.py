"""Build the Micropolis domain-knowledge prompt from the statement lists.

Takes the union of true_statements, false_statements and honeypot_statements,
shuffles it with a fixed seed, and appends the numbered result to the preamble.
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
from .statements import false_statements, honeypot_statements, true_statements

SHUFFLE_SEED = 20260807

HERE = Path(__file__).parent
PREAMBLE_PATH = HERE / "prompt_preamble.txt"

OUT_DIR = g.DATA_DIR / "knowledge_eval"

# Prompts, raw model responses and their usage sidecars live below OUT_DIR,
# mirroring the continuous eval's cache layout, so OUT_DIR itself holds only
# analysis outputs (plots, report, scores).
CACHE_DIR = OUT_DIR / "cache"


class Answer(Enum):
    """A model's parsed verdict on a single statement."""

    TRUE = "True"
    FALSE = "False"
    UNKNOWN = "Unknown"
    UNPARSEABLE = "Unparseable"


@dataclass(frozen=True)
class Statement:
    text: str
    is_true: bool
    is_honeypot: bool
    difficulty: int


def build_statement_list() -> list[Statement]:
    """Return the shuffled Statements, in administered order.

    Honeypot statements are all false, so they carry is_true=False alongside
    is_honeypot=True. The shuffle uses random.Random, whose Mersenne Twister
    stream is identical across machines and Python versions for a given seed.

    Statements are sorted by text before shuffling so the administered order
    depends only on the set of statements, not on their order within the three
    source lists.
    """
    statements = (
        [Statement(t, True, False, d) for t, d in true_statements]
        + [Statement(t, False, False, d) for t, d in false_statements]
        + [Statement(t, False, True, d) for t, d in honeypot_statements]
    )
    statements.sort(key=lambda s: s.text)
    random.Random(SHUFFLE_SEED).shuffle(statements)
    return statements


statements = build_statement_list()


def prompt_hash(prompt: str) -> str:
    """Short stable digest of a prompt, used to detect a changed statement set.

    SHA-256 rather than hash(), which is randomized per process and so would
    differ between runs. The digest is urlsafe-base64 encoded and truncated to
    8 characters — enough to spot a changed prompt, and filename-safe.
    """
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii")[:8]


def build_prompt() -> tuple[str, str]:
    """Return the full prompt text and its 8-character hash.

    Writes nothing: the hash is what identifies a cached response, so the
    read-only paths need to derive it without touching OUT_DIR. Call
    save_prompt() to persist the prompt itself.
    """
    preamble = PREAMBLE_PATH.read_text(encoding="utf-8")
    numbered = [f" {i}: {s.text}" for i, s in enumerate(statements, start=1)]
    prompt = preamble + "\n".join(numbered) + "\n"
    return prompt, prompt_hash(prompt)


def save_prompt(prompt: str, phash: str) -> None:
    """Write the prompt to prompt-<HASH>.txt in CACHE_DIR.

    Keeps every prompt a response was gathered under on disk alongside it.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"prompt-{phash}.txt").write_text(prompt, encoding="utf-8")


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
    return CACHE_DIR / f"response-{model_slug(model_id)}-{phash}.txt"


def usage_path(model_id: str, phash: str) -> Path:
    """Tokens and cost of the call that produced the matching response file.

    A cached response is never re-fetched, so what it cost has to be recorded
    when it is first paid or it is lost on every later run. The slug must match
    response_path's, or the sidecar lands next to nothing.
    """
    return CACHE_DIR / f"usage-{model_slug(model_id)}-{phash}.json"


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
        for p in CACHE_DIR.glob(f"{prefix}*{suffix}")
        if p.read_text(encoding="utf-8").strip()
    )


def parse_response(text: str | None) -> list[Answer]:
    """Parse a model's reply into one Answer per statement, in statement order.

    The prompt asks for lines of the form "12: False". Anything that doesn't
    yield a recognized verdict for a given statement number — a missing line, a
    duplicate, a hedge like "Probably true" — leaves that slot UNPARSEABLE, so
    the returned list always has one entry per statement.

    Matching is deliberately loose about surrounding punctuation and case
    (models like to write "**12:** False."), but strict about the verdict word
    itself: only True/False/Unknown count.
    """
    answers = [Answer.UNPARSEABLE] * len(statements)
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
        if not 1 <= number <= len(statements) or number in seen:
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
    """Administer the statement test to each model and parse the replies.

    The response-<MODEL>-<HASH>.txt files in CACHE_DIR are the cache and the
    source of truth: a model with a stored response for the current prompt is
    re-read and re-parsed rather than prompted again. Because the filename
    carries the prompt hash, editing the statements simply misses the cache
    instead of reusing answers to different questions.

    The uncached models are all prompted concurrently under run_prompts's
    global cap (`concurrency` overrides it), and each response is
    written to its cache file as it lands. There is one call per model, so
    every model's block of output still prints whole, in completion order,
    after the cached models' blocks.

    Returns a dict of model id -> answers, one per entry of the global
    `statements`, in the same order. A model whose request fails, or which
    replies with nothing but whitespace, is warned about and left out of both
    the result and the cache.

    The output cap is the backend's (model_specs.json5, or the provider
    default); it must leave room for one answer line per statement, since a cap
    that truncates the reply shows up as UNPARSEABLE answers for the tail.
    """
    # Imported here so the scoring-only scripts don't pull in an LLM client.
    from fbsim_core.evaluation.models import get_models

    prompt, phash = build_prompt()
    save_prompt(prompt, phash)
    print(f"prompt hash {phash} ({CACHE_DIR / f'prompt-{phash}.txt'})")

    data: dict[str, list[Answer]] = {}

    def record_answers(model_name: str, raw: str) -> None:
        answers = parse_response(raw)
        counts = Counter(answers)
        print(
            "  answered "
            + ", ".join(f"{counts[a]} {a.value}" for a in Answer)
            + f" (of {len(statements)} statements)"
        )
        data[model_name] = answers

    # Cached models first: their blocks print instantly, so handling them
    # before the fan-out keeps them from interleaving with live completions.
    jobs: list[PromptJob] = []
    for model_name, model in zip(models, get_models(models)):
        raw = cached_response(model_name, phash)
        if raw is not None:
            print(f"{model_name}: re-parsing cached response")
            record_answers(model_name, raw)
        else:
            jobs.append(
                PromptJob(
                    key=model_name,
                    model=model,
                    model_name=model_name,
                    messages=[{"role": "user", "content": prompt}],
                )
            )

    total_cost = 0.0
    nunpriced = 0
    failures: list[tuple[str, Exception]] = []

    async def consume() -> None:
        nonlocal total_cost, nunpriced
        # Everything below the API call — prints, cache writes, parsing — runs
        # here in the single consumer task, so nothing needs a lock.
        done = 0
        start = time.perf_counter()
        async for result in run_prompts(jobs, limit=concurrency):
            done += 1
            eta = format_eta(start, done, len(jobs))
            model_name = result.job.key
            if not result.ok:
                err = result.error
                failures.append((model_name, err))
                msg.error(
                    f"[{done}/{len(jobs)}] {model_name}: "
                    f"request FAILED: {type(err).__name__}: {err}{eta}"
                )
                continue

            resp = result.response
            raw = resp.text
            # Usage and how long the call took on the header line even for a
            # blank reply: a model that spent its whole budget thinking still
            # billed for it and still made you wait, and this is the only
            # place that spend is ever reported — there's no response to cache
            # it beside, so it isn't recorded on disk.
            took = format_latency(resp.usage.latency_ms, resp.retries)
            print(
                f"[{done}/{len(jobs)}] {model_name}: {resp.usage.describe()}{took}{eta}"
            )
            print(f"  finish_reason: {resp.finish_reason}")
            if resp.usage.cost_usd is None:
                nunpriced += 1
            else:
                total_cost += resp.usage.cost_usd
            warn_if_truncated(model_name, resp.finish_reason)

            if raw is None or not raw.strip():
                msg.warn(f"{model_name} returned a blank response; not caching")
                continue

            out_path = response_path(model_name, phash)
            # Saved as soon as it lands, so an interrupted run keeps what it
            # already paid for. Usage alongside the response, and only when the
            # response is kept, so the two never disagree about whether this
            # call happened.
            out_path.write_text(raw, encoding="utf-8")
            save_usage(model_name, phash, resp.usage)
            print(f"  saved response to {out_path}")
            record_answers(model_name, raw)

    if jobs:
        asyncio.run(consume())
        # What this run paid across all fresh calls; cached models cost nothing.
        summary = f"this run's {len(jobs)} call(s) cost ${total_cost:.2f}"
        if nunpriced:
            summary += f" + {nunpriced} unpriced call(s)"
        print(summary)
        if failures:
            msg.error(
                f"{len(failures)} call(s) failed (not cached; "
                "re-run this script to retry them):"
            )
            for model_name, err in failures:
                msg.plain(f"  {model_name}: {type(err).__name__}: {err}")

    return data


def get_cached_answers() -> dict[str, list[Answer]]:
    """Answers for every model with a stored response for the current prompt.

    Reads and parses the response files without prompting anything, so it works
    offline and costs nothing. Keys are filename slugs rather than the original
    provider/name ids, which a filename does not preserve.
    """
    _, phash = build_prompt()
    answers = {}
    for slug in cached_models(phash):
        path = CACHE_DIR / f"response-{slug}-{phash}.txt"
        answers[slug] = parse_response(path.read_text(encoding="utf-8"))
    return answers
