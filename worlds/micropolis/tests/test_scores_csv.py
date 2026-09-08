"""Tests for analyze_continuous.py's continuous_scores.csv.

Built on a hand-sized corpus with one model, so every count and mean in the
file can be checked by eye: two metrics, a read-off horizon and two forecast
horizons, and responses that were variously never prompted, unparsed and valid.
"""

import csv
import importlib.util
import math
import sys
from pathlib import Path

import pytest

from micropolis_world.continuous_eval import Normalizer, Response, ResponseId

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_continuous", SCRIPTS_DIR / "analyze_continuous.py"
    )
    module = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["analyze_continuous"]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


MODEL = "prov/model"
# 48 turns to the year, so these read 1y and 2y in the file.
H1, H2 = 48, 96


def _question(metric: str, horizon: int, value: float) -> dict:
    return {
        "question_id": f"{metric}@{horizon}",
        "scenario_id": "s",
        "snapshot_turn": 240,
        "metric": metric,
        "horizon": horizon,
        "value": value,
    }


def _point(value: float) -> dict[str, float]:
    """A degenerate forecast, whose CRPS is |value - actual| exactly."""
    return dict.fromkeys(("p10", "p25", "p50", "p75", "p90"), value)


# Every forecast is off by 10, so raw CRPS is 10 wherever it is defined and the
# normalized columns are 10 over each mode's scale.
CORPUS = [
    _question("cityPop", 0, 1_000.0),  # read-off: never in the file
    _question("cityPop", H1, 1_000.0),
    _question("cityPop", H2, 1_000.0),
    _question("trafficAverage", H1, 50.0),
    _question("trafficAverage", H2, 50.0),
    _question("totalFunds", H1, 5.0),  # unnormalizable: never in the file
]
RESPONSES = {
    ResponseId(MODEL, "cityPop@0"): Response(1_000.0, _point(1_010.0)),
    ResponseId(MODEL, f"cityPop@{H1}"): Response(1_000.0, _point(1_010.0)),
    # Prompted but the answer did not parse.
    ResponseId(MODEL, f"cityPop@{H2}"): Response(1_000.0, None),
    ResponseId(MODEL, f"trafficAverage@{H1}"): Response(50.0, _point(60.0)),
    # trafficAverage@H2 was never prompted: no response at all.
    ResponseId(MODEL, f"totalFunds@{H1}"): Response(5.0, _point(15.0)),
}


def _norm(mode: str, scales: dict[str, float]) -> Normalizer:
    return Normalizer(
        mode=mode, ratio="r", detail="d", scale=lambda c: scales.get(c["metric"])
    )


# Three modes with scales chosen so the columns are told apart at a glance.
NORMS = {
    "global": _norm("global", {"cityPop": 100.0, "trafficAverage": 10.0}),
    "local": _norm("local", {"cityPop": 50.0, "trafficAverage": 5.0}),
    "baseline": _norm("baseline", {"cityPop": 20.0, "trafficAverage": 2.0}),
}


def _rows() -> dict[tuple[str, str], dict]:
    module = load_module()
    rows = module.scores_csv_rows(CORPUS, RESPONSES, [MODEL], NORMS)
    assert all(r["model"] == MODEL for r in rows)
    return {(r["metric"], r["horizon"]): r for r in rows}


def test_rows_cover_every_metric_and_horizon_group_once():
    rows = _rows()
    assert set(rows) == {
        (m, h) for m in ("population", "traffic", "all") for h in ("1y", "2y", "all")
    }


def test_read_off_and_unnormalizable_metric_get_no_row():
    rows = _rows()
    assert not any(h == "0y" for _m, h in rows)
    assert not any(m == "city funds" for m, _h in rows)


def test_counts_separate_prompted_from_parsed():
    rows = _rows()
    # cityPop@H2 was prompted but unparsed; trafficAverage@H2 never prompted.
    assert (
        rows["population", "2y"]["nforecasts"],
        rows["population", "2y"]["nvalid"],
    ) == (1, 0)
    assert (rows["traffic", "2y"]["nforecasts"], rows["traffic", "2y"]["nvalid"]) == (
        0,
        0,
    )
    assert (
        rows["population", "all"]["nforecasts"],
        rows["population", "all"]["nvalid"],
    ) == (2, 1)
    assert (rows["all", "all"]["nforecasts"], rows["all", "all"]["nvalid"]) == (3, 2)


def test_all_horizon_excludes_the_read_off():
    """cityPop@0 has a valid response; it must not lift the pooled counts."""
    rows = _rows()
    assert rows["population", "all"]["nforecasts"] == 2  # H1 and H2, not H0


def test_raw_crps_is_nan_on_the_pooled_metric_row_only():
    rows = _rows()
    assert rows["population", "1y"]["CRPS"] == pytest.approx(10.0)
    assert rows["traffic", "1y"]["CRPS"] == pytest.approx(10.0)
    assert math.isnan(rows["all", "1y"]["CRPS"])
    assert math.isnan(rows["all", "all"]["CRPS"])
    # No valid forecast at all: nan rather than a crash or a zero.
    assert math.isnan(rows["traffic", "2y"]["CRPS"])


def test_normalized_columns_follow_each_modes_scale():
    rows = _rows()
    r = rows["population", "1y"]
    assert r["nCRPS_global"] == pytest.approx(10 / 100)
    assert r["nCRPS_local"] == pytest.approx(10 / 50)
    assert r["nCRPS_baseline"] == pytest.approx(10 / 20)


def test_pooled_row_averages_datapoints_not_metric_means():
    """(all, all) pools cityPop@H1 (10/100) and trafficAverage@H1 (10/10)."""
    rows = _rows()
    assert rows["all", "all"]["nCRPS_global"] == pytest.approx((0.1 + 1.0) / 2)
    assert rows["all", "1y"]["nCRPS_global"] == pytest.approx((0.1 + 1.0) / 2)


def test_horizon_label_is_years_with_a_y():
    module = load_module()
    assert module.horizon_label(48) == "1y"
    assert module.horizon_label(144) == "3y"
    assert module.horizon_label(480) == "10y"
    # A horizon that is not a whole year is not rounded into one.
    assert module.horizon_label(120) == "2.5y"


def test_write_scores_csv_columns_and_nan(tmp_path):
    module = load_module()
    rows = module.scores_csv_rows(CORPUS, RESPONSES, [MODEL], NORMS)
    path = module.write_scores_csv(tmp_path / "x.csv", rows, list(NORMS))
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "model", "metric", "horizon", "nforecasts", "nvalid", "CRPS",
            "nCRPS_global", "nCRPS_local", "nCRPS_baseline",
        ]  # fmt: skip
        back = {(r["metric"], r["horizon"]): r for r in reader}
    assert back["all", "all"]["CRPS"] == "nan"
    assert back["population", "1y"]["model"] == MODEL
    assert float(back["population", "1y"]["nCRPS_local"]) == pytest.approx(0.2)
