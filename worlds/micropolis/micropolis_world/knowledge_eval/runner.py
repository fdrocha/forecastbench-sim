"""Build the Micropolis domain-knowledge prompt from the statement lists.

Takes the union of true_statements, false_statements and honeypot_statements,
shuffles it with a fixed seed, and appends the numbered result to the preamble.
"""

import random
from pathlib import Path
from dataclasses import dataclass

from .statements import false_statements, honeypot_statements, true_statements

SHUFFLE_SEED = 20260807

HERE = Path(__file__).parent
PREAMBLE_PATH = HERE / "prompt_preamble.txt"


@dataclass(frozen=True)
class Statement:
    text: str
    is_true: bool
    is_honeypot: bool
    difficulty: int


def build_statement_list(seed: int = SHUFFLE_SEED) -> list[Statement]:
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
    random.Random(seed).shuffle(statements)
    return statements


def build_prompt(seed: int = SHUFFLE_SEED) -> str:
    """Return the full prompt: preamble followed by the numbered statements."""
    preamble = PREAMBLE_PATH.read_text(encoding="utf-8")
    statements = build_statement_list(seed)
    numbered = [f" {i}: {s.text}" for i, s in enumerate(statements, start=1)]
    return preamble + "\n".join(numbered) + "\n"


if __name__ == "__main__":
    print(build_prompt())
