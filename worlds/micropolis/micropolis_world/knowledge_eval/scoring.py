"""Score parsed knowledge-eval answers against the statement key.

Shared by scripts/run_knowledge_eval.py, which gathers the answers, and
scripts/analyze_knowledge.py, which reports on them.
"""

from .runner import Answer, Statement, statements

# Points per answer, from the SCORING block of the prompt preamble. An
# unparseable answer is scored as incorrect, as the preamble warns.
POINTS = {"correct": 1, "wrong": -2, "unknown": 0, "unparseable": -2}


def tally(
    answers: list[Answer], stmts: list[Statement] | None = None
) -> dict[str, int]:
    """Bucket one model's answers against the statement key.

    A statement is correct when the model's verdict matches Statement.is_true,
    wrong when it is the opposite verdict, and otherwise falls into its own
    column.

    `answers` is positional against the global `statements`, one entry each.
    `stmts` restricts scoring to a subset of those statements — a difficulty
    tier, say — and defaults to all of them.
    """
    keep = None if stmts is None else set(stmts)
    counts = {"correct": 0, "wrong": 0, "unknown": 0, "unparseable": 0}
    for statement, answer in zip(statements, answers):
        if keep is not None and statement not in keep:
            continue
        if answer is Answer.UNKNOWN:
            counts["unknown"] += 1
        elif answer is Answer.UNPARSEABLE:
            counts["unparseable"] += 1
        elif (answer is Answer.TRUE) == statement.is_true:
            counts["correct"] += 1
        else:
            counts["wrong"] += 1
    return counts


def score(counts: dict[str, int]) -> float:
    """Normalized score for a tally: 1.0 for all-correct.

    The POINTS total divided by the number of statements scored, which fixes the
    scale independently of how many statements a subset holds: 1.0 all correct,
    0.0 all unknown, -2.0 all wrong. That makes a difficulty tier's score
    directly comparable to the whole set's.
    """
    n = sum(counts.values())
    if n == 0:
        raise ValueError("cannot score an empty tally")
    return sum(POINTS[k] * v for k, v in counts.items()) / n


def scores_by_model_name(stmts: list[Statement] | None = None) -> dict[str, float]:
    """Normalized knowledge score per model, keyed on the bare model name.

    Reads the response cache and scores it; prompts nothing. `stmts` restricts
    the scoring to a subset, as in tally().

    Keyed on the bare name — "gpt-5.6-sol", not "openai/gpt-5.6-sol" — because
    that is what ECI_MAP uses, and what the other evals' model ids reduce to
    once the provider prefix is stripped. The cache's own keys are filename
    slugs, whose first "_" stands in for the "/" of the original id; that holds
    for every provider prefix in use, and a provider name containing "_" would
    be the thing to revisit here.
    """
    from .runner import get_cached_answers

    return {
        slug.split("_", 1)[1]: score(tally(answers, stmts))
        for slug, answers in get_cached_answers().items()
    }
