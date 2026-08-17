"""Unit tests for knowledge-eval scoring.

Covers the tally buckets, subset restriction, and the normalized score's scale.
Nothing here calls a model or touches disk.
"""

from micropolis_world.knowledge_eval.runner import Answer, Statement, statements
from micropolis_world.knowledge_eval.scoring import score, tally


def answers_for(verdict_of) -> list[Answer]:
    """One answer per statement, chosen by a function of the statement."""
    return [verdict_of(s) for s in statements]


def all_correct() -> list[Answer]:
    return answers_for(lambda s: Answer.TRUE if s.is_true else Answer.FALSE)


def all_wrong() -> list[Answer]:
    return answers_for(lambda s: Answer.FALSE if s.is_true else Answer.TRUE)


def test_tally_buckets_every_statement():
    counts = tally(all_correct())
    assert counts["correct"] == len(statements)
    assert sum(counts.values()) == len(statements)


def test_tally_counts_wrong_and_unknown():
    assert tally(all_wrong())["wrong"] == len(statements)
    unknowns = tally([Answer.UNKNOWN] * len(statements))
    assert unknowns["unknown"] == len(statements)
    unparseable = tally([Answer.UNPARSEABLE] * len(statements))
    assert unparseable["unparseable"] == len(statements)


def test_tally_subset_scores_only_that_subset():
    subset = [s for s in statements if s.difficulty == 2]
    counts = tally(all_correct(), subset)
    assert sum(counts.values()) == len(subset)
    assert counts["correct"] == len(subset)


def test_tally_subset_defaults_to_all_statements():
    assert tally(all_correct(), statements) == tally(all_correct())


def test_tally_pairs_answers_positionally():
    # Answer only the first statement, correctly; the rest abstain.
    first = statements[0]
    answers = [Answer.TRUE if first.is_true else Answer.FALSE]
    answers += [Answer.UNKNOWN] * (len(statements) - 1)
    counts = tally(answers, [first])
    assert counts == {"correct": 1, "wrong": 0, "unknown": 0, "unparseable": 0}


def test_score_endpoints():
    n = len(statements)
    assert score({"correct": n, "wrong": 0, "unknown": 0, "unparseable": 0}) == 1.0
    assert score({"correct": 0, "wrong": 0, "unknown": n, "unparseable": 0}) == 0.0
    assert score({"correct": 0, "wrong": n, "unknown": 0, "unparseable": 0}) == -2.0


def test_score_is_independent_of_subset_size():
    # A perfect score is 1.0 whether it covers 11 statements or all of them.
    perfect = {"correct": 11, "wrong": 0, "unknown": 0, "unparseable": 0}
    assert score(perfect) == 1.0


def test_score_treats_unparseable_as_wrong():
    counts = {"correct": 0, "wrong": 0, "unknown": 0, "unparseable": 4}
    assert score(counts) == -2.0


def test_score_rejects_empty_tally():
    try:
        score({"correct": 0, "wrong": 0, "unknown": 0, "unparseable": 0})
    except ValueError:
        return
    raise AssertionError("expected ValueError for an empty tally")


def test_statement_is_hashable_for_subset_membership():
    # tally() puts the subset in a set, which requires Statement be hashable.
    assert len({Statement("a", True, False, 1), Statement("a", True, False, 1)}) == 1
