"""Unit tests for the single-city score tables.

Covers the within-column ranking and the ranked cell's padding, which is what
keeps the numbers aligned when some ranks have more digits than others. Nothing
here calls a model or touches disk.
"""

import importlib.util
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def load_module():
    """Import the analysis script, which is not on the package path."""
    spec = importlib.util.spec_from_file_location(
        "analyze_single_city", SCRIPTS_DIR / "analyze_single_city.py"
    )
    module = importlib.util.module_from_spec(spec)
    # The script parses argv at call time, not import time, but keep it clean.
    argv, sys.argv = sys.argv, ["analyze_single_city"]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


def test_ranks_are_best_first():
    ranks = load_module().ranks_within_column({"a": 0.5, "b": 0.1, "c": 0.9})
    assert ranks == {"b": 1, "a": 2, "c": 3}


def test_ranks_share_the_lower_rank_on_ties():
    ranks = load_module().ranks_within_column({"a": 1.0, "b": 1.0, "c": 2.0})
    assert ranks == {"a": 1, "b": 1, "c": 3}


def test_ranks_skip_models_without_a_score():
    ranks = load_module().ranks_within_column({"a": 1.0, "b": None, "c": 2.0})
    assert ranks == {"a": 1, "c": 2}
    assert load_module().ranks_within_column({}) == {}


def test_ranked_cells_are_the_same_width_regardless_of_rank_digits():
    # The point of the padding: a one-digit rank must not shift its score left
    # relative to a two-digit one, or the decimal points stop lining up.
    ranked_cell = load_module().ranked_cell
    narrow = ranked_cell(0.075, 4, ".3f", 2)
    wide = ranked_cell(0.084, 10, ".3f", 2)
    assert narrow == "0.075 ( 4)"
    assert wide == "0.084 (10)"
    assert len(narrow) == len(wide)


def test_ranked_cell_does_not_pad_when_every_rank_is_one_digit():
    assert load_module().ranked_cell(0.075, 3, ".3f", 1) == "0.075 (3)"


def test_ranked_cell_honors_the_number_format():
    assert load_module().ranked_cell(3857.2, 1, ",.1f", 2) == "3,857.2 ( 1)"


def test_ranked_cell_without_a_value_is_not_ranked():
    assert load_module().ranked_cell(None, None, ".3f", 2) == "n/a"


def test_rank_width_counts_the_widest_rank():
    rank_width_for = load_module().rank_width_for
    assert rank_width_for({"a": 1, "b": 10}) == 2
    assert rank_width_for({"a": 1, "b": 2}) == 1
    assert rank_width_for({"a": 100}) == 3
    # No ranks at all still needs a width, or the format spec would be empty.
    assert rank_width_for({}) == 1


def test_correlate_by_horizon_only_uses_models_in_the_predictor():
    # The restriction that keeps every series on one model set: a model absent
    # from the predictor contributes no point, whatever its nCRPS.
    correlate_by_horizon = load_module().correlate_by_horizon
    by_horizon = {
        0: {"p/a": 0.1, "p/b": 0.2, "p/c": 0.3, "p/d": 0.4, "p/e": 0.5},
    }
    predictor = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    (horizon, rho, _p, n) = correlate_by_horizon(predictor, by_horizon)[0]
    assert (horizon, n) == (0, 4)
    # a..d rank the same way in both variables, so the correlation is perfect.
    assert rho == 1.0


def test_correlate_by_horizon_skips_horizons_below_min_n():
    correlate_by_horizon = load_module().correlate_by_horizon
    by_horizon = {0: {"p/a": 0.1, "p/b": 0.2}, 48: {"p/a": 0.1, "p/b": 0.2}}
    predictor = {"a": 1.0, "b": 2.0}
    assert correlate_by_horizon(predictor, by_horizon) == []
    # Same data, but a threshold this small set of models can meet.
    assert len(correlate_by_horizon(predictor, by_horizon, min_n=2)) == 2


def test_correlate_by_horizon_skips_a_constant_column():
    # Every model scoring the same leaves the coefficient undefined; scipy would
    # return nan with a warning rather than raising, so this must be caught here.
    correlate_by_horizon = load_module().correlate_by_horizon
    predictor = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    flat_y = {0: {"p/a": 0.2, "p/b": 0.2, "p/c": 0.2, "p/d": 0.2}}
    assert correlate_by_horizon(predictor, flat_y) == []
    # A constant predictor is just as undefined as a constant score.
    flat_x = {0: {"p/a": 0.1, "p/b": 0.2, "p/c": 0.3, "p/d": 0.4}}
    assert correlate_by_horizon({k: 1.0 for k in predictor}, flat_x) == []


def test_correlate_by_horizon_is_ordered_by_horizon():
    correlate_by_horizon = load_module().correlate_by_horizon
    predictor = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    scores = {"p/a": 0.1, "p/b": 0.2, "p/c": 0.3, "p/d": 0.4}
    by_horizon = {240: dict(scores), 0: dict(scores), 48: dict(scores)}
    horizons = [h for h, _, _, _ in correlate_by_horizon(predictor, by_horizon)]
    assert horizons == [0, 48, 240]
