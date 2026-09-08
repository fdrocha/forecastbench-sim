"""Unit tests for the continuous score tables.

Covers the within-column ranking and the ranked cell's padding, which is what
keeps the numbers aligned when some ranks have more digits than others. Nothing
here calls a model or touches disk.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

from micropolis_world.continuous_eval import GLOBAL_SCALES, make_normalizer

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"

# The baselines are normalized through one of the normalizations, so the
# tests below pass the default one and read its scale rather than restating it.
# The global mode needs no corpus: its scales do not come from the questions.
NORM = make_normalizer("global", [])
POP_SCALE = GLOBAL_SCALES["cityPop"]


def load_module():
    """Import the analysis script, which is not on the package path."""
    spec = importlib.util.spec_from_file_location(
        "analyze_continuous", SCRIPTS_DIR / "analyze_continuous.py"
    )
    module = importlib.util.module_from_spec(spec)
    # The script parses argv at call time, not import time, but keep it clean.
    argv, sys.argv = sys.argv, ["analyze_continuous"]
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


def test_spearman_over_resamples_matches_scipy():
    """The vectorized bootstrap must agree with scipy exactly, ties included.

    Ranks are recomputed per resample rather than taken from the full sample,
    because a resample repeats models and so has ties the full sample lacks.
    Getting that wrong would bias every interval, so it is pinned here.
    """
    import numpy as np
    from scipy import stats

    module = load_module()
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([2.0, 1.0, 4.0, 3.0, 5.0])
    idx = np.array([[0, 1, 2, 3, 4], [0, 0, 1, 2, 3], [4, 3, 2, 1, 0]])
    got = module.spearman_over_resamples(x, y, idx)
    want = [stats.spearmanr(x[i], y[i]).statistic for i in idx]
    assert np.allclose(got, want)


def test_spearman_over_resamples_is_nan_on_a_constant_draw():
    import numpy as np

    module = load_module()
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([5.0, 6.0, 7.0])
    # A resample of one model repeated has no spread, so rho is undefined.
    got = module.spearman_over_resamples(x, y, np.array([[1, 1, 1], [0, 1, 2]]))
    assert np.isnan(got[0])
    assert got[1] == 1.0


def test_bootstrap_ci_brackets_a_strong_correlation():
    module = load_module()
    xs = list(range(12))
    ys = [float(v) for v in range(12)]
    lo, hi = module.bootstrap_rho_ci(xs, ys, resamples=500)
    # Perfectly monotone, so every resample gives rho=1 and the CI collapses.
    assert lo == hi == 1.0


def test_bootstrap_ci_is_deterministic():
    # The band must not move between runs, or a re-run reads as changed data.
    module = load_module()
    xs = [1, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 12]
    ys = [0.5, 0.1, 0.4, 0.9, 0.3, 0.8, 0.2, 0.7, 0.6, 1.0, 0.05, 0.95]
    first = module.bootstrap_rho_ci(xs, ys, resamples=500)
    assert first == module.bootstrap_rho_ci(xs, ys, resamples=500)


def test_correlate_by_horizon_adds_a_ci_only_when_asked():
    module = load_module()
    by_horizon = {0: {"p/a": 0.4, "p/b": 0.3, "p/c": 0.2, "p/d": 0.1}}
    predictor = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    plain = module.correlate_by_horizon(predictor, by_horizon)
    assert len(plain[0]) == 4
    with_ci = module.correlate_by_horizon(predictor, by_horizon, with_ci=True)
    assert len(with_ci[0]) == 5
    lo, hi = with_ci[0][4]
    assert lo <= with_ci[0][1] <= hi


def test_compare_predictors_prefers_the_better_aligned_predictor():
    """The paired comparison must favor the predictor that tracks the scores."""
    module = load_module()
    by_horizon = {0: {"p/a": 0.1, "p/b": 0.2, "p/c": 0.3, "p/d": 0.4, "p/e": 0.5}}
    # `good` is perfectly monotone with nCRPS; `bad` is nearly unrelated.
    good = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    bad = {"a": 3.0, "b": 1.0, "c": 5.0, "d": 2.0, "e": 4.0}
    rows = module.compare_predictors_by_horizon(good, bad, by_horizon)
    assert rows[0]["diff"] > 0  # positive means the first predictor is stronger
    assert rows[0]["lo"] <= rows[0]["diff"] <= rows[0]["hi"]
    # Rarely should the weaker predictor win a resample.
    assert rows[0]["share"] < 0.25
    assert rows[0]["n"] == 5


def test_compare_predictors_is_symmetric_in_sign():
    module = load_module()
    by_horizon = {0: {"p/a": 0.1, "p/b": 0.2, "p/c": 0.3, "p/d": 0.4, "p/e": 0.5}}
    good = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
    bad = {"a": 3.0, "b": 1.0, "c": 5.0, "d": 2.0, "e": 4.0}
    forward = module.compare_predictors_by_horizon(good, bad, by_horizon)[0]
    reverse = module.compare_predictors_by_horizon(bad, good, by_horizon)[0]
    assert forward["diff"] == -reverse["diff"]


def test_read_off_horizon_is_not_a_forecast():
    module = load_module()
    assert not module.is_forecast(module.READ_OFF_HORIZON)
    assert module.is_forecast(module.READ_OFF_HORIZON + 48)


def test_forecast_questions_drops_only_the_read_off_horizon():
    module = load_module()
    corpus = [
        {"question_id": "a", "horizon": module.READ_OFF_HORIZON},
        {"question_id": "b", "horizon": 48},
        {"question_id": "c", "horizon": 240},
    ]
    assert [c["question_id"] for c in module.forecast_questions(corpus)] == ["b", "c"]


def test_horizon_table_all_column_excludes_the_read_off():
    """The "all" column pools horizons, so it must leave the read-off out.

    The read-off scores 0 for every model, so including it would pull "all"
    below the mean of the real horizons — here to 0.10 rather than 0.15.
    """
    module = load_module()
    report = module.MdReport()
    scored = [
        ("m", module.READ_OFF_HORIZON, 0.0),
        ("m", 48, 0.1),
        ("m", 240, 0.2),
    ]
    module.print_horizon_table(
        report, scored, ["m"], [module.READ_OFF_HORIZON, 48, 240], "t", "s", ".3f"
    )
    out = report.render(Path("."))
    row = next(ln for ln in out.splitlines() if ln.startswith("m "))
    assert "0.150" in row, row
    assert "0.100" not in row.split("0.150")[0], row


def test_horizon_table_keeps_the_read_off_as_its_own_column():
    module = load_module()
    report = module.MdReport()
    module.print_horizon_table(
        report,
        [("m", module.READ_OFF_HORIZON, 0.0), ("m", 48, 0.1)],
        ["m"],
        [module.READ_OFF_HORIZON, 48],
        "t",
        "s",
        ".3f",
    )
    out = report.render(Path("."))
    assert f"H{module.READ_OFF_HORIZON}" in out
    assert "all*" in out


def _persistence_corpus():
    """Two horizons of one metric, with a known snapshot value at the read-off.

    cityPop reads 100 at the snapshot, 150 at H48 and 400 at H96, so persistence
    is off by 50 and 300 — two different errors over one shared scale, so a
    helper that divided by the wrong operand would not pass by coincidence.
    """
    return [
        {
            "scenario_id": "s",
            "snapshot_turn": 240,
            "metric": "cityPop",
            "horizon": h,
            "value": v,
        }
        for h, v in [(0, 100), (48, 150), (96, 400)]
    ]


def test_persistence_normalizes_by_the_metric_scale():
    """The snapshot error over the metric's scale, not over either value."""
    module = load_module()
    baseline = module.persistence_by_horizon(_persistence_corpus(), NORM)
    assert baseline[48] == pytest.approx(50 / POP_SCALE)
    assert baseline[96] == pytest.approx(300 / POP_SCALE)


def test_persistence_omits_the_read_off_horizon():
    """It scores 0 there by construction, which would misdraw the reference line.

    Asserted on a corpus whose read-off value disagrees with its resolved value,
    so a kept read-off would contribute a nonzero score rather than being hidden
    by also being dropped for scoring 0.
    """
    module = load_module()
    corpus = [
        {
            "scenario_id": "s",
            "snapshot_turn": 240,
            "metric": "cityPop",
            "horizon": h,
            "value": v,
        }
        # Two read-off questions for one group: the second is a stand-in for a
        # dataset where the read-off row is not the one the snapshot came from,
        # so scoring it would land on a nonzero ratio and show up in the result.
        for h, v in [(0, 100), (0, 200), (48, 150)]
    ]
    assert module.READ_OFF_HORIZON not in module.persistence_by_horizon(corpus, NORM)


def test_persistence_skips_metrics_with_no_scale():
    module = load_module()
    corpus = [
        {
            "scenario_id": "s",
            "snapshot_turn": 240,
            "metric": "totalFunds",
            "horizon": h,
            "value": v,
        }
        for h, v in [(0, 100), (48, 150)]
    ]
    assert module.persistence_by_horizon(corpus, NORM) == {48: None}


def test_persistence_scores_a_zero_actual():
    """A fixed scale is defined where the actual is 0, unlike dividing by it.

    A city that empties out is a real question with a real error, and under a
    global scale nothing has to be dropped to score it.
    """
    module = load_module()
    corpus = [
        {
            "scenario_id": "s",
            "snapshot_turn": 240,
            "metric": "cityPop",
            "horizon": h,
            "value": v,
        }
        for h, v in [(0, 100), (48, 0)]
    ]
    assert module.persistence_by_horizon(corpus, NORM)[48] == pytest.approx(
        100 / POP_SCALE
    )


def test_persistence_reads_the_snapshot_from_its_own_scenario():
    """A shared metric name must not let one run's snapshot score another's."""
    module = load_module()
    corpus = []
    for scenario, snapshot in [("a", 100), ("b", 900)]:
        corpus.append(
            {
                "scenario_id": scenario,
                "snapshot_turn": 240,
                "metric": "cityPop",
                "horizon": 0,
                "value": snapshot,
            }
        )
        corpus.append(
            {
                "scenario_id": scenario,
                "snapshot_turn": 240,
                "metric": "cityPop",
                "horizon": 48,
                "value": snapshot,
            }
        )
    # Each run's H48 equals its own snapshot, so a correct lookup scores 0; a
    # lookup that ignored scenario_id would cross the two and score above 0.
    assert module.persistence_by_horizon(corpus, NORM)[48] == pytest.approx(0.0)


def _flat_history(n: int = 300) -> list[dict]:
    """A history whose metric never moves, so its historical spread is zero."""
    return [{"cityPop": 100} for _ in range(n)]


def test_historical_sigma_uses_only_turns_up_to_the_snapshot():
    """A jump after the snapshot must not widen the interval that precedes it.

    Otherwise the baseline is peeking at the resolution window and is no longer a
    forecast a naive forecaster could have made.
    """
    module = load_module()
    history = _flat_history(300)
    for row in history[250:]:
        row["cityPop"] = 100000
    assert module.historical_sigma(history, "cityPop", 240, 48) == 0.0


def test_historical_sigma_counts_overlapping_changes():
    """n = snapshot + 1 - horizon, since the snapshot turn is itself observable."""
    module = load_module()
    history = [{"cityPop": i} for i in range(300)]
    # A perfectly linear series changes by exactly `horizon` every time, so the
    # spread is 0 while the count is what the window allows.
    assert module.historical_sigma(history, "cityPop", 240, 48) == 0.0
    # Too few changes to have a spread at all: one change is not a sample.
    assert module.historical_sigma(history, "cityPop", 240, 240) is None


def test_historical_sigma_is_none_when_history_is_shorter_than_the_horizon():
    module = load_module()
    assert module.historical_sigma(_flat_history(10), "cityPop", 240, 48) is None


def test_sigma_baseline_matches_persistence_when_nothing_ever_moved(monkeypatch):
    """Zero spread collapses the interval, so the two baselines must agree.

    Pins the sigma baseline to the degenerate one at its boundary case, which is
    the check that it is the same forecast with a different interval rather than
    a different forecast.
    """
    module = load_module()
    corpus = [
        {
            "scenario_id": "s",
            "snapshot_turn": 240,
            "metric": "cityPop",
            "horizon": h,
            "value": v,
        }
        for h, v in [(0, 100), (48, 150)]
    ]
    monkeypatch.setattr(module, "scenario_history", lambda _sid, _seed: _flat_history())
    means, scored = module.persistence_sigma_by_horizon(corpus, 42, NORM)
    assert means[48] == pytest.approx(
        module.persistence_by_horizon(corpus, NORM, scored)[48]
    )


def test_sigma_baseline_beats_persistence_when_the_metric_moves(monkeypatch):
    """A real spread must help, not hurt: that is the whole point of the line."""
    module = load_module()
    corpus = [
        {
            "scenario_id": "s",
            "snapshot_turn": 240,
            "metric": "cityPop",
            "horizon": h,
            "value": v,
        }
        for h, v in [(0, 100), (48, 150)]
    ]
    # A wandering history, so sigma is well above zero.
    history = [{"cityPop": 100 + (i % 7) * 30} for i in range(300)]
    monkeypatch.setattr(module, "scenario_history", lambda _sid, _seed: history)
    means, scored = module.persistence_sigma_by_horizon(corpus, 42, NORM)
    assert means[48] < module.persistence_by_horizon(corpus, NORM, scored)[48]


def test_sigma_baseline_skips_a_scenario_with_no_cached_run(monkeypatch):
    """A missing log is not an error; the line is drawn from what is on disk."""
    module = load_module()
    corpus = [
        {
            "scenario_id": "s",
            "snapshot_turn": 240,
            "metric": "cityPop",
            "horizon": h,
            "value": v,
        }
        for h, v in [(0, 100), (48, 150)]
    ]
    monkeypatch.setattr(module, "scenario_history", lambda _sid, _seed: None)
    means, scored = module.persistence_sigma_by_horizon(corpus, 42, NORM)
    assert means == {48: None}
    assert scored == set()


def test_normal_z_is_symmetric_and_centered_on_the_median():
    """An asymmetric z would shift the median off the snapshot value."""
    module = load_module()
    assert module.NORMAL_Z["p50"] == 0.0
    assert module.NORMAL_Z["p10"] == pytest.approx(-module.NORMAL_Z["p90"])
    assert module.NORMAL_Z["p25"] == pytest.approx(-module.NORMAL_Z["p75"])


def test_the_two_baselines_average_over_the_same_questions(monkeypatch):
    """Where the spread is unavailable, both lines must drop the question.

    Otherwise the H240 comparison comes from two different question sets and the
    gap between the lines reads as a difference between forecasts when it is
    partly a difference in what was scored.
    """
    module = load_module()
    corpus = []
    for scenario in ["keep", "drop"]:
        for h, v in [(0, 100), (48, 150)]:
            corpus.append(
                {
                    "scenario_id": scenario,
                    "snapshot_turn": 240,
                    "metric": "cityPop",
                    "horizon": h,
                    "value": v,
                }
            )
    # Only one scenario has a cached run, so only its questions get a spread.
    monkeypatch.setattr(
        module,
        "scenario_history",
        lambda sid, _seed: _flat_history() if sid == "keep" else None,
    )
    _means, scored = module.persistence_sigma_by_horizon(corpus, 42, NORM)
    assert scored == {("keep", 240, "cityPop", 48)}
    # Unrestricted, plain persistence averages both scenarios; restricted, one.
    assert len(module.persistence_by_horizon(corpus, NORM)) == 1
    restricted = module.persistence_by_horizon(corpus, NORM, scored)
    assert restricted[48] == pytest.approx(50 / POP_SCALE)
