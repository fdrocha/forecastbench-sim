"""Tests for the correlation table analyze_binary.py builds on
analyze_continuous.py's machinery: sections by ground truth, the two
bootstrap intervals, and the per-horizon rows carrying both.

Nothing here calls a model or reads the data directory.
"""

import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from micropolis_world.ground_truth import Truth

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [name]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


MODELS = [f"prov/m{i}" for i in range(6)]
PREDICTOR = {m: float(i) for i, m in enumerate(MODELS)}


def rows_for(matrix: np.ndarray, horizons=(240,)) -> list[dict]:
    """Scored rows from a questions x models matrix; nan = unparsed."""
    rows = []
    for qi, question in enumerate(matrix):
        for mj, value in enumerate(question):
            if np.isnan(value):
                continue
            rows.append(
                {
                    "model_id": MODELS[mj],
                    "question_id": f"q{qi}",
                    "horizon": horizons[qi % len(horizons)],
                    "excess_brier": float(value),
                    "brier": float(value),
                }
            )
    return rows


def test_section_of_splits_on_ground_truth_not_qid():
    ab = load("analyze_binary")
    truths = {"x": Truth(0.049, 1000), "y": Truth(0.05, 1000), "z": Truth(0.0, 10)}
    assert ab.section_of({"question_id": "x", "qid": "A1"}, truths) == ab.TAIL
    assert ab.section_of({"question_id": "y", "qid": "B1"}, truths) == ab.MID_RANGE
    assert ab.section_of({"question_id": "z", "qid": "A1"}, truths) == ab.TAIL


def test_model_bootstrap_matches_the_continuous_reports_interval():
    ab, ac = load("analyze_binary"), load("analyze_continuous")
    rng = np.random.default_rng(3)
    # Scores loosely decreasing with the predictor, plus noise.
    matrix = np.array(
        [[0.5 - 0.05 * j + rng.normal(0, 0.1) for j in range(6)] for _ in range(40)]
    )
    score = ab.SCORES[1]
    c = ab.correlate(
        "ECI", PREDICTOR, rows_for(matrix), score.key, MODELS, ab.ALL, resamples=500
    )
    xs = [PREDICTOR[m] for m in MODELS]
    ys = list(matrix.mean(axis=0))
    assert c.rho_models == ac.bootstrap_rho_ci(xs, ys, resamples=500)
    assert c.n_models == 6 and c.n_questions == 40
    assert -1 <= c.rho_questions[0] <= c.rho <= c.rho_questions[1] <= 1
    assert -1 <= c.r_questions[0] <= c.r <= c.r_questions[1] <= 1


def test_question_bootstrap_collapses_when_questions_agree():
    """Every question ranks the models the same way, so redrawing questions
    cannot move Spearman: the interval sits on the point estimate."""
    ab = load("analyze_binary")
    matrix = np.tile(np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1]), (20, 1))
    c = ab.correlate(
        "ECI",
        PREDICTOR,
        rows_for(matrix),
        ab.SCORES[1].key,
        MODELS,
        ab.ALL,
        resamples=300,
    )
    assert c.rho == pytest.approx(-1.0)
    assert c.rho_questions == (pytest.approx(-1.0), pytest.approx(-1.0))


def test_unparsed_forecasts_cost_a_model_a_question_not_everyone():
    ab = load("analyze_binary")
    matrix = np.tile(np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1]), (10, 1))
    matrix[0, 0] = np.nan  # model 0 did not parse question 0
    c = ab.correlate(
        "ECI",
        PREDICTOR,
        rows_for(matrix),
        ab.SCORES[1].key,
        MODELS,
        ab.ALL,
        resamples=100,
    )
    assert c.n_questions == 10 and c.n_models == 6


def test_too_few_models_gives_none():
    ab = load("analyze_binary")
    matrix = np.tile(np.array([0.6, 0.5, 0.4]), (10, 1))
    predictor = {m: PREDICTOR[m] for m in MODELS[:3]}
    assert (
        ab.correlate(
            "ECI", predictor, rows_for(matrix), ab.SCORES[1].key, MODELS, ab.ALL
        )
        is None
    )


def test_table_uses_the_rho_symbol_and_one_row_per_correlation():
    ab = load("analyze_binary")
    matrix = np.tile(np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1]), (12, 1))
    rows = rows_for(matrix, horizons=(240, 480))
    correlations = [
        ab.correlate("ECI", PREDICTOR, rows, ab.SCORES[1].key, MODELS, h, resamples=50)
        for h in (ab.ALL, "5y", "10y")
    ]
    table = ab.format_correlation_table(correlations)
    lines = table.split("\n")
    assert len(lines) == 4
    assert "ρ" in lines[0] and "rho" not in table
    assert lines[1].split()[:2] == ["ECI", ab.ALL]


def test_rows_by_horizon_matches_the_means_based_rows_and_adds_both_bands():
    ac = load("analyze_continuous")
    rng = np.random.default_rng(5)
    matrix = np.array(
        [[0.5 - 0.05 * j + rng.normal(0, 0.1) for j in range(6)] for _ in range(30)]
    )
    rows = rows_for(matrix, horizons=(48, 240, 480))
    by_horizon = {}
    for r in rows:
        by_horizon.setdefault(r["horizon"], {}).setdefault(r["model_id"], []).append(
            r["brier"]
        )
    by_horizon = {
        h: {m: sum(v) / len(v) for m, v in ms.items()} for h, ms in by_horizon.items()
    }
    bare = {m.split("/", 1)[1]: v for m, v in PREDICTOR.items()}
    old = ac.correlate_by_horizon(bare, by_horizon, with_ci=True)
    new = ac.correlate_rows_by_horizon(PREDICTOR, rows, "brier", MODELS)
    assert [row[:4] for row in new] == pytest.approx([row[:4] for row in old])
    assert [row[4] for row in new] == [row[4] for row in old]
    for h, rho, _p, _n, ci_models, ci_questions in new:
        assert ci_models[0] <= rho <= ci_models[1]
        assert ci_questions[0] <= rho <= ci_questions[1]
    text = ac.format_horizon_correlations(new)
    assert text.count("models [") == 3 and text.count("questions [") == 3
    assert "rho" not in text


def test_per_model_intervals_bracket_the_means_the_scatter_draws():
    """The scatter's error bars must come from the same resampled means as the
    coefficient's questions interval, and cover every model it kept."""
    ab = load("analyze_binary")
    rng = np.random.default_rng(11)
    matrix = np.array(
        [[0.5 - 0.05 * j + rng.normal(0, 0.1) for j in range(6)] for _ in range(40)]
    )
    c = ab.correlate(
        "ECI", PREDICTOR, rows_for(matrix), "brier", MODELS, ab.ALL, resamples=400
    )
    assert set(c.scores) == set(MODELS) == set(c.score_questions)
    for model_id, mean in c.scores.items():
        assert mean == pytest.approx(matrix[:, MODELS.index(model_id)].mean())
        lo, hi = c.score_questions[model_id]
        assert lo < mean < hi


def test_error_bars_are_drawn_once_for_the_models_that_have_one():
    """One gray errorbar call for every point with an interval, and no bars at
    all when none of the models carry one."""
    ac = load("analyze_continuous")
    matrix = np.tile(np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1]), (20, 1))
    c = ac.correlate(
        "ECI", PREDICTOR, rows_for(matrix), "brier", MODELS, ac.ALL, resamples=200
    )
    points = [(PREDICTOR[m], c.scores[m], m.split("/")[-1], m) for m in MODELS]

    calls = []

    class FakeAxes:
        def errorbar(self, x, y, **kwargs):
            calls.append((list(x), list(y), kwargs))

    assert ac.draw_score_error_bars(FakeAxes(), c, points) is True
    (xs, ys, kwargs) = calls[0]
    assert len(calls) == 1 and len(xs) == len(ys) == 6
    assert kwargs["label"].startswith("95%")
    # Every bar is non-negative in both directions, or matplotlib would raise.
    assert all(v >= 0 for direction in kwargs["yerr"] for v in direction)

    stripped = dataclasses.replace(c, score_questions={})
    assert ac.draw_score_error_bars(FakeAxes(), stripped, points) is False
