"""Build the Micropolis domain-knowledge prompt from the statement lists.

Takes the union of true_statements, false_statements and honeypot_statements,
shuffles it with a fixed seed, and appends the numbered result to the preamble.
"""

import base64
import hashlib
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from dataclasses import dataclass

from fbsim_core.evaluation.models import get_models

from .. import module_globals as g
from .statements import false_statements, honeypot_statements, true_statements

SHUFFLE_SEED = 20260807

# Statements are one line each; the answers are one short line each. Give enough
# room for the whole answer block plus any preamble a chatty model adds.
MAX_TOKENS = 8000

HERE = Path(__file__).parent
PREAMBLE_PATH = HERE / "prompt_preamble.txt"

OUT_DIR = g.DATA_DIR / "knowledge_eval"
CACHE_PATH = OUT_DIR / "response_cache.json"


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


def load_cache() -> dict[str, dict]:
    """Load the whole model_id -> cache entry mapping.

    Each entry carries the parsed answers plus the hash of the prompt they were
    gathered under and when it was run. The raw response text is not cached; it
    lives in the response-<MODEL>-<HASH>.txt file written alongside.
    """
    if not CACHE_PATH.exists():
        return {}
    data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {entry["model_id"]: entry for entry in data}


def save_cache(cache: dict[str, dict]) -> None:
    """Write the whole cache back, preserving entries for other models."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = [entry for _, entry in sorted(cache.items())]
    CACHE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")


def response_path(model_id: str, phash: str) -> Path:
    """Path for a model's saved response text under a given prompt hash.

    Model ids contain slashes ("anthropic/claude-opus-4"), so flatten them into
    a single filename component. Including the hash keeps responses from
    different prompt revisions side by side rather than overwriting.
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", model_id).strip("_")
    return OUT_DIR / f"response-{safe}-{phash}.txt"


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

    A cached response is reused only when it was gathered under the current
    prompt; one recorded against a different prompt hash is stale — the
    statements changed since — so it is reported and re-gathered. The whole
    cache is loaded and saved back, so responses for models outside `models`
    are preserved, but only the requested models are returned.

    On a miss the raw response is written to response-<MODEL>-<HASH>.txt in
    OUT_DIR and its parsed answers are cached; the cache holds the answers
    rather than the text, which the .txt file already keeps. Returns a dict of
    model id -> answers, one per entry of the global `statements`, in the same
    order.

    max_tokens must leave room for one answer line per statement; a cap that
    truncates the reply shows up as UNPARSEABLE answers for the tail.
    """
    prompt, phash = build_prompt()
    print(f"prompt hash {phash} ({OUT_DIR / f'prompt-{phash}.txt'})")

    cache = load_cache()
    n_new = 0
    data = {}
    for model_name, model in zip(models, get_models(models)):
        entry = cache.get(model_name)
        if entry is not None and entry.get("prompt_hash") != phash:
            print(
                f"{model_name}: cached answers were for prompt "
                f"{entry.get('prompt_hash')}, not {phash} — re-prompting"
            )
            entry = None

        if entry is not None:
            answers = [Answer(v) for v in entry["answers"]]
            print(f"{model_name}: using cached answers")
        else:
            print(f"{model_name}: prompting...")
            try:
                raw = model.get_response(prompt, max_tokens=max_tokens)
            except Exception as e:  # noqa: BLE001
                print(f"  request failed for {model_name}: {e}")
                continue

            out_path = response_path(model_name, phash)
            out_path.write_text(raw or "", encoding="utf-8")
            print(f"  saved response to {out_path}")

            answers = parse_response(raw)
            cache[model_name] = {
                "model_id": model_name,
                "prompt_hash": phash,
                "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "answers": [a.value for a in answers],
            }
            n_new += 1
            save_cache(cache)

        counts = Counter(answers)
        print(
            "  answered "
            + ", ".join(f"{counts[a]} {a.value}" for a in Answer)
            + f" (of {len(statements)} statements)"
        )
        data[model_name] = answers

    if n_new:
        save_cache(cache)
    return data
