"""Tests for analyze_binary.py's binary_scores.csv and results.csv.

A hand-sized corpus with one model: one mid-range and one tail question at two
horizons, with responses that were variously never prompted, unparsed and
valid, so every count and mean can be checked by eye.
"""

import csv
import importlib.util
import math
import sys
from pathlib import Path

import pytest

from micropolis_world.binary_eval import BinaryResponse
from micropolis_world.continuous_eval import ResponseId
from micropolis_world.ground_truth import Truth

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def load_module():
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


MODEL = "prov/model"
H1, H2 = 240, 480  # 5y and 10y


def _question(qid: str, horizon: int, answer: bool, scenario: str = "s") -> dict:
    return {
        "question_id": f"{scenario}:{qid}@{horizon}",
        "scenario_id": scenario,
        "scenario": {"name": "bruce", "seed": 42, "disasters": True},
        "snapshot_turn": 960,
        "qid": qid,
        "horizon": horizon,
        "answer": answer,
    }


CORPUS = [
    _question("A1", H1, True),
    _question("A1", H2, False),
    _question("B1", H1, False),
    _question("B1", H2, False),
]
TRUTHS = {
    f"s:A1@{H1}": Truth(0.5, 100),
    f"s:A1@{H2}": Truth(0.3, 100),
    f"s:B1@{H1}": Truth(0.01, 100),
    f"s:B1@{H2}": Truth(0.0, 100),
}
RESPONSES = {
    ResponseId(MODEL, f"s:A1@{H1}"): BinaryResponse(True, 0.7),
    # Prompted but the answer did not parse.
    ResponseId(MODEL, f"s:A1@{H2}"): BinaryResponse(False, None),
    ResponseId(MODEL, f"s:B1@{H1}"): BinaryResponse(
        False, 0.2, source="s_T960/response-prov_model-abc.txt", line=4
    ),
    # B1@H2 was never prompted: no response at all.
}


def _rows() -> dict[tuple[str, str], dict]:
    module = load_module()
    rows = module.scores_csv_rows(CORPUS, RESPONSES, [MODEL], TRUTHS)
    assert all(r["model"] == MODEL for r in rows)
    return {(r["question_type"], r["horizon"]): r for r in rows}


def test_rows_cover_every_type_and_horizon_group_once_and_never_pool_types():
    rows = _rows()
    assert set(rows) == {
        (t, h) for t in ("mid-range", "tail") for h in ("5y", "10y", "all")
    }


def test_counts_separate_prompted_from_parsed():
    rows = _rows()
    assert (
        rows["mid-range", "10y"]["nforecasts"],
        rows["mid-range", "10y"]["nvalid"],
    ) == (
        1,
        0,
    )
    assert (rows["tail", "10y"]["nforecasts"], rows["tail", "10y"]["nvalid"]) == (0, 0)
    assert (
        rows["mid-range", "all"]["nforecasts"],
        rows["mid-range", "all"]["nvalid"],
    ) == (
        2,
        1,
    )
    assert (rows["tail", "all"]["nforecasts"], rows["tail", "all"]["nvalid"]) == (1, 1)


def test_scores_follow_their_definitions():
    """A1@H1: f=0.7, outcome Yes, p=0.5."""
    r = _rows()["mid-range", "5y"]
    assert r["brier"] == pytest.approx(0.3**2)
    assert r["excess_brier"] == pytest.approx(0.2**2)
    assert r["expected_brier"] == pytest.approx(0.2**2 + 0.5 * 0.5)


def test_rows_with_no_valid_forecast_are_nan():
    r = _rows()["mid-range", "10y"]
    assert all(math.isnan(r[k]) for k in ("brier", "expected_brier", "excess_brier"))


def test_same_qid_in_another_city_does_not_stand_in_for_an_unparsed_answer():
    """Two cities ask A1 at H2; only one parsed, so nvalid is 1, not 2."""
    module = load_module()
    corpus = [_question("A1", H2, False), _question("A1", H2, True, scenario="t")]
    truths = {c["question_id"]: Truth(0.5, 10) for c in corpus}
    responses = {
        ResponseId(MODEL, f"s:A1@{H2}"): BinaryResponse(False, None),
        ResponseId(MODEL, f"t:A1@{H2}"): BinaryResponse(True, 0.5),
    }
    rows = {
        (r["question_type"], r["horizon"]): r
        for r in module.scores_csv_rows(corpus, responses, [MODEL], truths)
    }
    assert (
        rows["mid-range", "10y"]["nforecasts"],
        rows["mid-range", "10y"]["nvalid"],
    ) == (
        2,
        1,
    )


def test_write_scores_csv_columns_and_nan(tmp_path):
    module = load_module()
    rows = module.scores_csv_rows(CORPUS, RESPONSES, [MODEL], TRUTHS)
    path = module.write_csv(tmp_path / "x.csv", module.SCORES_CSV_COLUMNS, rows)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "model", "question_type", "horizon", "nforecasts", "nvalid",
            "brier", "expected_brier", "excess_brier",
        ]  # fmt: skip
        back = {(r["question_type"], r["horizon"]): r for r in reader}
    assert back["mid-range", "10y"]["brier"] == "nan"
    assert float(back["tail", "5y"]["excess_brier"]) == pytest.approx(0.19**2)


def test_results_csv_one_row_per_model_and_question(tmp_path):
    module = load_module()
    rows = module.results_csv_rows(CORPUS, RESPONSES, [MODEL], TRUTHS)
    assert len(rows) == len(CORPUS)
    path = module.write_csv(tmp_path / "r.csv", module.RESULTS_CSV_COLUMNS, rows)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == [
            "model", "seed", "city", "disasters", "snapshot_turn", "horizon",
            "question_id", "forecast", "answer", "real_prob",
            "response_file", "response_line",
        ]  # fmt: skip
        back = {(r["question_id"], int(r["horizon"])): r for r in reader}
    valid = back["A1", H1]
    assert (valid["model"], valid["seed"], valid["city"], valid["disasters"]) == (
        MODEL,
        "42",
        "bruce",
        "1",
    )
    assert (valid["snapshot_turn"], valid["forecast"], valid["answer"]) == (
        "960",
        "0.7",
        "1",
    )
    assert float(valid["real_prob"]) == pytest.approx(0.5)
    # A response that recorded where it was read from.
    traced = back["B1", H1]
    assert traced["response_file"] == "s_T960/response-prov_model-abc.txt"
    assert traced["response_line"] == "4"
    # Without a source on record the trace columns are nan, like the forecast.
    assert valid["response_file"] == valid["response_line"] == "nan"
    # Unparsed and never-prompted forecasts are nan, not dropped.
    assert back["A1", H2]["forecast"] == "nan"
    assert back["B1", H2]["forecast"] == "nan"
    assert back["B1", H2]["response_file"] == "nan"
    assert back["B1", H2]["answer"] == "0"
