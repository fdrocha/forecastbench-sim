"""Unit tests for the CRPS normalization modes behind --norm."""

import pytest
from fbsim_core.metrics import compute_crps

from micropolis_world import templates
from micropolis_world.continuous_eval import (
    GLOBAL_SCALES,
    UNNORMALIZED_METRICS,
    Response,
    ResponseId,
    make_normalizer,
    score_forecasts,
)


def _question(metric: str, value: float, horizon: int = 48) -> dict:
    return {
        "question_id": f"{metric}-{horizon}",
        "scenario_id": "s",
        "snapshot_turn": 240,
        "metric": metric,
        "horizon": horizon,
        "value": value,
    }


def test_global_scales_are_the_documented_numbers():
    """Pinned, because the whole point of the mode is that they never move."""
    assert GLOBAL_SCALES == {
        "cityPop": 20_000.0,
        "trafficAverage": 20.0,
        "pollutionAverage": 60.0,
        "crimeAverage": 60.0,
        "landValueAverage": 60.0,
    }


def test_every_asked_metric_has_a_scale_or_is_excluded():
    """A metric added to the corpus must not fall out of the normalized tables."""
    for metric in templates.Q_METRICS:
        assert metric in GLOBAL_SCALES or metric in UNNORMALIZED_METRICS, metric


def test_global_scale_ignores_the_question_and_the_actual():
    norm = make_normalizer("global")
    assert norm.scale(_question("cityPop", 1.0)) == 20_000.0
    assert norm.scale(_question("cityPop", 900_000.0, horizon=480)) == 20_000.0
    assert norm.scale(_question("trafficAverage", 0.0)) == 20.0


def test_global_has_no_scale_for_the_excluded_metric():
    assert make_normalizer("global").scale(_question("totalFunds", 500.0)) is None


@pytest.mark.parametrize("mode", ["local", "baseline"])
def test_the_unwritten_modes_say_so(mode):
    with pytest.raises(NotImplementedError, match=mode):
        make_normalizer(mode)


def test_an_unknown_mode_is_an_error():
    with pytest.raises(ValueError, match="unknown normalization mode"):
        make_normalizer("|actual|")


def test_unscaled_metrics_flags_only_what_no_list_covers():
    norm = make_normalizer("global")
    corpus = [
        _question("cityPop", 100.0),
        _question("totalFunds", 100.0),
        _question("cityScore", 100.0),
    ]
    assert norm.unscaled_metrics(corpus) == ["cityScore"]


def _percentiles(center: float) -> dict[str, float]:
    return {
        "p10": center - 10,
        "p25": center - 5,
        "p50": center,
        "p75": center + 5,
        "p90": center + 10,
    }


def test_score_forecasts_divides_by_the_metric_scale():
    norm = make_normalizer("global")
    corpus = [_question("cityPop", 5_000.0), _question("totalFunds", 500.0)]
    responses = {
        ResponseId("m", c["question_id"]): Response(
            actual=c["value"], percentiles=_percentiles(c["value"] + 100)
        )
        for c in corpus
    }
    rows = {r["metric"]: r for r in score_forecasts(corpus, responses, ["m"], norm)}

    pop = rows["cityPop"]
    assert pop["crps"] == pytest.approx(compute_crps(_percentiles(5_100.0), 5_000.0))
    assert pop["normalized"] == pytest.approx(pop["crps"] / 20_000.0)
    # Raw CRPS is still reported for the metric that has no scale.
    assert rows["totalFunds"]["crps"] > 0
    assert rows["totalFunds"]["normalized"] is None
