"""Build the Micropolis domain-knowledge prompt from the statement lists.

Takes the union of true_statements, false_statements and honeypot_statements,
shuffles it with a fixed seed, and appends the numbered result to the preamble.
"""

import base64
import hashlib
import random
import re
from collections import Counter
from enum import Enum
from pathlib import Path
from dataclasses import dataclass

from fbsim_core.evaluation.models import get_models

from .. import module_globals as g
from ..module_globals import prompt_model, warn_if_truncated
from .statements import false_statements, honeypot_statements, true_statements

SHUFFLE_SEED = 20260807

# Statements are one line each; the answers are one short line each. Give enough
# room for the whole answer block plus any preamble a chatty model adds.
MAX_TOKENS = 8000

HERE = Path(__file__).parent
PREAMBLE_PATH = HERE / "prompt_preamble.txt"

OUT_DIR = g.DATA_DIR / "knowledge_eval"


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

    As a side effect the prompt is written to prompt-<HASH>.txt in OUT_DIR, so
    every prompt a response was gathered under stays on disk alongside it.
    """
    preamble = PREAMBLE_PATH.read_text(encoding="utf-8")
    numbered = [f" {i}: {s.text}" for i, s in enumerate(statements, start=1)]
    prompt = preamble + "\n".join(numbered) + "\n"

    phash = prompt_hash(prompt)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / f"prompt-{phash}.txt").write_text(prompt, encoding="utf-8")
    return prompt, phash


def model_slug(model_id: str) -> str:
    """Model id flattened into a single filename component.

    Model ids contain slashes ("anthropic/claude-opus-4"), which can't appear in
    a filename.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")


def response_path(model_id: str, phash: str) -> Path:
    """Path of a model's saved response under a given prompt hash.

    These files are the cache: a response present here is reused rather than
    re-fetched. Including the hash keeps responses from different prompt
    revisions side by side rather than overwriting, and means a changed
    statement set simply misses the cache instead of silently reusing answers
    to different questions.
    """
    return OUT_DIR / f"response-{model_slug(model_id)}-{phash}.txt"


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
    rather than what some separate index claims. Slugs are returned, not the
    original ids: the "/" separator is not recoverable from a filename.
    """
    prefix, suffix = "response-", f"-{phash}.txt"
    return sorted(
        p.name[len(prefix) : -len(suffix)]
        for p in OUT_DIR.glob(f"{prefix}*{suffix}")
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
    models: list[str], max_tokens: int = MAX_TOKENS
) -> dict[str, list[Answer]]:
    """Administer the statement test to each model and parse the replies.

    The response-<MODEL>-<HASH>.txt files in OUT_DIR are the cache and the
    source of truth: a model with a stored response for the current prompt is
    re-read and re-parsed rather than prompted again. Because the filename
    carries the prompt hash, editing the statements simply misses the cache
    instead of reusing answers to different questions.

    Returns a dict of model id -> answers, one per entry of the global
    `statements`, in the same order. A model whose request fails, or which
    replies with nothing but whitespace, is warned about and left out of both
    the result and the cache.

    max_tokens must leave room for one answer line per statement; a cap that
    truncates the reply shows up as UNPARSEABLE answers for the tail.
    """
    prompt, phash = build_prompt()
    print(f"prompt hash {phash} ({OUT_DIR / f'prompt-{phash}.txt'})")

    data = {}
    for model_name, model in zip(models, get_models(models)):
        raw = cached_response(model_name, phash)
        if raw is not None:
            print(f"{model_name}: re-parsing cached response")
        else:
            print(f"{model_name}: prompting...")
            try:
                raw, finish_reason = prompt_model(model, prompt, max_tokens)
            except Exception as e:  # noqa: BLE001
                print(f"  request failed for {model_name}: {e}")
                continue

            print(f"  finish_reason: {finish_reason}")
            warn_if_truncated(model_name, finish_reason, max_tokens)

            if raw is None or not raw.strip():
                print(f"  [warning] {model_name} returned a blank response; not caching")
                continue

            out_path = response_path(model_name, phash)
            out_path.write_text(raw, encoding="utf-8")
            print(f"  saved response to {out_path}")

        answers = parse_response(raw)
        counts = Counter(answers)
        print(
            "  answered "
            + ", ".join(f"{counts[a]} {a.value}" for a in Answer)
            + f" (of {len(statements)} statements)"
        )
        data[model_name] = answers

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
        path = OUT_DIR / f"response-{slug}-{phash}.txt"
        answers[slug] = parse_response(path.read_text(encoding="utf-8"))
    return answers
