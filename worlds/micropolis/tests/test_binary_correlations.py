"""Tests for analyze_binary.py's correlation table: sections by ground truth,
and the two bootstrap intervals.

Nothing here calls a model or reads the data directory.
"""

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
        "ECI", PREDICTOR, rows_for(matrix), score, MODELS, ab.ALL, resamples=500
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
        "ECI", PREDICTOR, rows_for(matrix), ab.SCORES[1], MODELS, ab.ALL, resamples=300
    )
    assert c.rho == pytest.approx(-1.0)
    assert c.rho_questions == (pytest.approx(-1.0), pytest.approx(-1.0))


def test_unparsed_forecasts_cost_a_model_a_question_not_everyone():
    ab = load("analyze_binary")
    matrix = np.tile(np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1]), (10, 1))
    matrix[0, 0] = np.nan  # model 0 did not parse question 0
    c = ab.correlate(
        "ECI", PREDICTOR, rows_for(matrix), ab.SCORES[1], MODELS, ab.ALL, resamples=100
    )
    assert c.n_questions == 10 and c.n_models == 6


def test_too_few_models_gives_none():
    ab = load("analyze_binary")
    matrix = np.tile(np.array([0.6, 0.5, 0.4]), (10, 1))
    predictor = {m: PREDICTOR[m] for m in MODELS[:3]}
    assert (
        ab.correlate("ECI", predictor, rows_for(matrix), ab.SCORES[1], MODELS, ab.ALL)
        is None
    )


def test_table_uses_the_rho_symbol_and_one_row_per_correlation():
    ab = load("analyze_binary")
    matrix = np.tile(np.array([0.6, 0.5, 0.4, 0.3, 0.2, 0.1]), (12, 1))
    rows = rows_for(matrix, horizons=(240, 480))
    correlations = [
        ab.correlate("ECI", PREDICTOR, rows, ab.SCORES[1], MODELS, h, resamples=50)
        for h in (ab.ALL, "5y", "10y")
    ]
    table = ab.format_correlation_table(correlations)
    lines = table.split("\n")
    assert len(lines) == 4
    assert "ρ" in lines[0] and "rho" not in table
    assert lines[1].split()[:2] == ["ECI", ab.ALL]
