"""Tests for scripts/analyze_prompts.py: the one-vote-per-model mean, its
question bootstrap, the paired difference and the label trimming. Nothing here
touches disk or a model."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_prompts", SCRIPTS_DIR / "analyze_prompts.py"
    )
    module = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["analyze_prompts"]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


def rows(scores: dict[str, dict[str, float]]) -> list[dict]:
    """{model: {question: score}} as score_forecasts-shaped rows."""
    module = load_module()
    return [
        {"model_id": m, "question_id": q, module.SCORE_KEY: v}
        for m, per_q in scores.items()
        for q, v in per_q.items()
    ]


def test_short_labels_drop_the_shared_prefix_at_a_separator():
    short_labels = load_module().short_labels
    assert short_labels(["prompt-semantic", "prompt-smallbatch"]) == {
        "prompt-semantic": "semantic",
        "prompt-smallbatch": "smallbatch",
    }
    assert short_labels(["default"]) == {"default": "default"}
    assert short_labels(["prompt", "prompt-long"]) == {
        "prompt": "prompt",
        "prompt-long": "prompt-long",
    }


def test_config_mean_gives_every_model_one_vote():
    module = load_module()
    # Model a is scored on three questions, model b on one: the mean is the
    # average of the two per-model means, not the pooled mean over four rows.
    scored = rows({"a": {"q1": 0.0, "q2": 0.0, "q3": 0.0}, "b": {"q1": 1.0}})
    mean, ci, n_models, n_questions = module.config_mean(scored, resamples=200)
    assert mean == pytest.approx(0.5)
    assert (n_models, n_questions) == (2, 3)
    assert ci is not None and ci[0] <= mean <= ci[1]
    assert module.config_mean([]) is None


def test_config_mean_interval_is_tight_when_questions_agree():
    module = load_module()
    scored = rows({"a": {f"q{i}": 0.3 for i in range(20)}})
    mean, ci, _, _ = module.config_mean(scored, resamples=200)
    assert mean == pytest.approx(0.3)
    assert ci == pytest.approx((0.3, 0.3))


def test_paired_difference_uses_shared_models_and_questions_only():
    module = load_module()
    a = module.Scored(
        "a",
        rows({"m1": {"q1": 0.5, "q2": 0.5}, "m2": {"q1": 0.5}}),
        ["m1", "m2"],
        0,
        0,
        0,
    )
    b = module.Scored(
        "b",
        rows({"m1": {"q1": 0.2, "q2": 0.2, "q3": 9.0}, "m3": {"q1": 0.0}}),
        ["m1", "m3"],
        0,
        0,
        0,
    )
    delta, ci, n_models, n_questions = module.paired_difference(a, b, resamples=200)
    # Only m1 is in both, only q1 and q2 are in both: 0.5 - 0.2, q3 ignored.
    assert (n_models, n_questions) == (1, 2)
    assert delta == pytest.approx(0.3)
    assert ci == pytest.approx((0.3, 0.3))


def test_paired_difference_needs_two_shared_questions_and_a_shared_model():
    module = load_module()
    a = module.Scored("a", rows({"m1": {"q1": 0.5}}), ["m1"], 0, 0, 0)
    b = module.Scored("b", rows({"m1": {"q1": 0.2}}), ["m1"], 0, 0, 0)
    assert module.paired_difference(a, b) is None
    c = module.Scored("c", rows({"m2": {"q1": 0.2, "q2": 0.1}}), ["m2"], 0, 0, 0)
    assert module.paired_difference(a, c) is None


def test_question_groups_keep_command_line_order():
    module = load_module()
    same1 = module.Scored("x", rows({"m": {"q1": 0.0, "q2": 0.0}}), ["m"], 0, 0, 0)
    other = module.Scored("y", rows({"m": {"q9": 0.0}}), ["m"], 0, 0, 0)
    same2 = module.Scored("z", rows({"m": {"q2": 1.0, "q1": 1.0}}), ["m"], 0, 0, 0)
    groups = module.question_groups([same1, other, same2])
    assert [[s.label for s in g] for g in groups] == [["x", "z"], ["y"]]


def test_resampled_means_count_a_drawn_question_as_many_times_as_drawn():
    import numpy as np

    module = load_module()
    scores = np.array([[0.0, 1.0], [1.0, np.nan]])  # two questions x two models
    draws = np.array([[1, 1]])  # q2 twice
    means = module.resampled_model_means(scores, draws)
    assert means[0, 0] == pytest.approx(1.0)
    assert np.isnan(means[0, 1])  # model 2 has no score on the drawn question
