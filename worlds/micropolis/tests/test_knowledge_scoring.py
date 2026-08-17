"""Unit tests for knowledge-eval scoring.

Covers the tally buckets, subset restriction, the normalized score's scale, and
the statement subsets the analysis breaks its correlations down by. Nothing here
calls a model or touches disk.
"""

import importlib.util
from pathlib import Path

from micropolis_world.knowledge_eval.runner import Answer, Statement, statements
from micropolis_world.knowledge_eval.scoring import score, tally

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def load_analyze_knowledge():
    """Import the analysis script, which is not on the package path."""
    spec = importlib.util.spec_from_file_location(
        "analyze_knowledge", SCRIPTS_DIR / "analyze_knowledge.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_subsets_partition_by_difficulty_and_honeypot():
    subsets = load_analyze_knowledge().subsets()
    by_name = {s.name: s for s in subsets}

    assert by_name["All"].stmts == statements
    # The difficulty tiers partition the whole set.
    tiers = [s for s in subsets if s.name.startswith("Difficulty")]
    assert sum(len(s.stmts) for s in tiers) == len(statements)
    # Honeypot and non-honeypot do too, and are disjoint.
    hp, non_hp = by_name["Honeypot"], by_name["Non-honeypot"]
    assert len(hp.stmts) + len(non_hp.stmts) == len(statements)
    assert not set(hp.stmts) & set(non_hp.stmts)
    assert all(s.is_honeypot for s in hp.stmts)


def test_subset_label_carries_size_and_slug_does_not():
    module = load_analyze_knowledge()
    sub = module.Subset("Difficulty 2", [s for s in statements if s.difficulty == 2])
    assert sub.label == f"Difficulty 2 (n={len(sub.stmts)})"
    # The slug names the plot file, so it must not churn as statements are added.
    assert sub.slug == "difficulty-2"
    assert module.Subset("Non-honeypot", []).slug == "non-honeypot"


def test_subset_slugs_are_unique():
    # Each subset writes its own plot, so a collision would silently overwrite.
    slugs = [s.slug for s in load_analyze_knowledge().subsets()]
    assert len(slugs) == len(set(slugs))
