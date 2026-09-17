"""Tests for the excess CRPS: the distribution CRPS, its floor and the rows
score_forecasts adds when a question carries its continuation outcomes."""

import numpy as np
import pytest
from fbsim_core.metrics import compute_crps

import micropolis_world.ground_truth as gt
from micropolis_world.continuous_eval import (
    GLOBAL_SCALES,
    Response,
    ResponseId,
    attach_outcomes,
    crps_distribution,
    crps_floor,
    make_normalizer,
    quantile_array,
    score_forecasts,
)

P = {"p10": 80.0, "p25": 90.0, "p50": 100.0, "p75": 110.0, "p90": 120.0}


def test_distribution_crps_on_one_outcome_is_compute_crps():
    for y in (50.0, 95.0, 100.0, 113.0, 200.0):
        assert crps_distribution(quantile_array(P), [y]) == pytest.approx(
            compute_crps(P, y)
        )


def test_distribution_crps_is_the_mean_over_outcomes():
    ys = np.linspace(60.0, 140.0, 17)
    expected = np.mean([compute_crps(P, y) for y in ys])
    assert crps_distribution(quantile_array(P), ys) == pytest.approx(expected)


def test_floor_is_the_least_any_five_quantiles_can_score():
    ys = np.random.default_rng(0).normal(100.0, 10.0, 500)
    floor = crps_floor(ys)
    q = np.quantile(ys, [0.1, 0.25, 0.5, 0.75, 0.9], method="inverted_cdf")
    assert floor == pytest.approx(crps_distribution(q, ys))
    for shift in (-3.0, -0.5, 0.5, 3.0):
        assert crps_distribution(q + shift, ys) >= floor
    assert crps_distribution(quantile_array(P), ys) >= floor


def test_floor_is_zero_when_the_metric_cannot_move():
    assert crps_floor([7.0] * 20) == 0.0


def _question(metric: str, value: float) -> dict:
    return {
        "question_id": f"{metric}@48",
        "scenario_id": "s",
        "snapshot_turn": 240,
        "metric": metric,
        "horizon": 48,
        "value": value,
    }


def test_score_forecasts_without_outcomes_has_no_excess():
    corpus = [_question("cityPop", 5_000.0)]
    responses = {ResponseId("m", corpus[0]["question_id"]): Response(5_000.0, P)}
    (row,) = score_forecasts(corpus, responses, ["m"], make_normalizer("global", []))
    assert row["crps_dist"] is None
    assert row["excess_crps"] is None
    assert row["excess_normalized"] is None


def test_attach_outcomes_gives_score_forecasts_its_excess(monkeypatch):
    corpus = [_question("cityPop", 5_000.0), _question("totalFunds", 100.0)]
    outcomes = {
        "cityPop@48": [4_000.0, 5_000.0, 6_000.0, 7_000.0],
        "totalFunds@48": None,  # the file holds no values for this metric
    }
    monkeypatch.setattr(gt, "load_outcomes", lambda c: outcomes)
    assert attach_outcomes(corpus) == 1
    assert "outcomes" not in corpus[1]

    forecast = {k: v + 5_000.0 for k, v in P.items()}
    responses = {
        ResponseId("m", c["question_id"]): Response(c["value"], forecast)
        for c in corpus
    }
    rows = {
        r["metric"]: r
        for r in score_forecasts(
            corpus, responses, ["m"], make_normalizer("global", [])
        )
    }
    pop = rows["cityPop"]
    ys = np.array(outcomes["cityPop@48"])
    dist = crps_distribution(quantile_array(forecast), ys)
    assert pop["crps_dist"] == pytest.approx(dist)
    assert pop["excess_crps"] == pytest.approx(dist - crps_floor(ys))
    assert pop["excess_normalized"] == pytest.approx(
        pop["excess_crps"] / GLOBAL_SCALES["cityPop"]
    )
    assert pop["excess_crps"] >= 0
    # The realized-outcome CRPS is untouched by the attachment.
    assert pop["crps"] == pytest.approx(compute_crps(forecast, 5_000.0))
    assert rows["totalFunds"]["excess_crps"] is None


def test_a_forecast_equal_to_the_replay_distribution_has_zero_excess():
    ys = np.random.default_rng(1).normal(0.0, 1.0, 1000)
    q = np.quantile(ys, [0.1, 0.25, 0.5, 0.75, 0.9], method="inverted_cdf")
    assert crps_distribution(q, ys) - crps_floor(ys) == pytest.approx(0.0)
