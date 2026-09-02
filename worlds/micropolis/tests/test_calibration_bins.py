"""Unit tests for the calibration line's binning.

Covers the bin assignment, the log-space variant the tail section uses and the
standard error. Nothing here calls a model or touches disk.
"""

import importlib.util
import sys
from itertools import pairwise
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def load_module():
    """Import the analysis script, which is not on the package path."""
    spec = importlib.util.spec_from_file_location(
        "analyze_binary", SCRIPTS_DIR / "analyze_binary.py"
    )
    module = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["analyze_binary"]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


def test_bins_average_the_truth_within_each_forecast_band():
    """Bins split on the forecast; the reported value is the mean truth."""
    module = load_module()
    # Two tight clusters of forecasts, each with a known mean truth.
    points = [(0.1, 0.2), (0.1, 0.4), (0.9, 0.6), (0.9, 0.8)]
    bins = module.calibration_bins(points, 2, log=False, floor=0.0)
    assert len(bins) == 2
    (_, low_truth, _), (_, high_truth, _) = bins
    assert low_truth == pytest.approx(0.3)
    assert high_truth == pytest.approx(0.7)


def test_empty_bins_are_dropped():
    module = load_module()
    points = [(0.01, 0.5), (0.02, 0.5), (0.99, 0.5)]
    bins = module.calibration_bins(points, 10, log=False, floor=0.0)
    # Ten bands over [0.01, 0.99], but only the first and last hold anything.
    assert len(bins) == 2


def test_standard_error_is_sd_over_sqrt_n_and_zero_for_a_lone_point():
    module = load_module()
    bins = module.calibration_bins([(0.5, 0.4), (0.5, 0.6)], 1, log=False, floor=0.0)
    [(_, mean, sem)] = bins
    assert mean == pytest.approx(0.5)
    # sd of {0.4, 0.6} with ddof=1 is 0.1414..., over sqrt(2) is 0.1.
    assert sem == pytest.approx(0.1)

    [(_, _, lone_sem)] = module.calibration_bins(
        [(0.5, 0.4)], 1, log=False, floor=0.0
    )
    assert lone_sem == 0.0


def test_log_bins_split_evenly_in_log_space():
    module = load_module()
    # Three decades of forecasts, one point per decade. Linear bins would put
    # the first two in the same band; log bins separate them.
    points = [(0.001, 0.1), (0.01, 0.2), (1.0, 0.3)]
    bins = module.calibration_bins(points, 3, log=True, floor=1e-4)
    assert len(bins) == 3
    centers = [c for c, _, _ in bins]
    assert centers == sorted(centers)
    # Centers are geometric, so each is the previous times a constant ratio.
    ratios = [b / a for a, b in pairwise(centers)]
    assert abs(ratios[0] - ratios[1]) < 1e-9


def test_log_bins_floor_a_zero_forecast():
    """A zero cannot go on a log axis; it is floored, not dropped."""
    module = load_module()
    bins = module.calibration_bins([(0.0, 0.5), (1.0, 0.5)], 2, log=True, floor=1e-3)
    assert len(bins) == 2
    assert all(c > 0 for c, _, _ in bins)


def test_no_points_gives_no_bins():
    module = load_module()
    assert module.calibration_bins([], 10, log=False, floor=0.0) == []
