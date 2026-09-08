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
    norm = make_normalizer("global", [])
    assert norm.scale(_question("cityPop", 1.0)) == 20_000.0
    assert norm.scale(_question("cityPop", 900_000.0, horizon=480)) == 20_000.0
    assert norm.scale(_question("trafficAverage", 0.0)) == 20.0


def test_global_has_no_scale_for_the_excluded_metric():
    assert make_normalizer("global", []).scale(_question("totalFunds", 500.0)) is None


def test_the_unwritten_mode_says_so():
    with pytest.raises(NotImplementedError, match="baseline"):
        make_normalizer("baseline", [])


def test_an_unknown_mode_is_an_error():
    with pytest.raises(ValueError, match="unknown normalization mode"):
        make_normalizer("|actual|", [])


def test_unscaled_metrics_flags_only_what_no_list_covers():
    norm = make_normalizer("global", [])
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
    norm = make_normalizer("global", [])
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


def _write_ground_truth(tmp_path, monkeypatch, averages_by_horizon):
    """A tally file holding just the averages --norm local reads."""
    import micropolis_world.ground_truth as gt

    monkeypatch.setattr(gt, "OUT_DIR", tmp_path)
    lines = [
        {
            "scenario_id": "s",
            "snapshot_turn": 240,
            "horizon": horizon,
            "n_continuations": 1000,
            "averages": averages,
        }
        for horizon, averages in averages_by_horizon.items()
    ]
    gt.write_lines(gt.output_path_for("s", 240), lines)


def test_local_scale_is_the_offset_plus_the_horizons_own_mean(tmp_path, monkeypatch):
    """Each question divides by its own (scenario, snapshot, horizon) mean."""
    _write_ground_truth(
        tmp_path,
        monkeypatch,
        {48: {"cityPop": 5_000.0}, 96: {"cityPop": 9_999.0}},
    )
    corpus = [
        _question("cityPop", 1.0, horizon=48),
        _question("cityPop", 2.0, horizon=96),
    ]
    norm = make_normalizer("local", corpus)
    assert norm.scale(corpus[0]) == pytest.approx(5_001.0)
    assert norm.scale(corpus[1]) == pytest.approx(10_000.0)


def test_local_offsets_a_zero_mean_instead_of_dividing_by_it(tmp_path, monkeypatch):
    """The one case the offset is there for: a metric that averaged 0."""
    _write_ground_truth(tmp_path, monkeypatch, {48: {"pollutionAverage": 0.0}})
    corpus = [_question("pollutionAverage", 3.0)]
    assert make_normalizer("local", corpus).scale(corpus[0]) == pytest.approx(1.0)


def test_local_keeps_the_excluded_metric_out(tmp_path, monkeypatch):
    """City funds has no scale under any mode, whatever the file holds for it."""
    _write_ground_truth(tmp_path, monkeypatch, {48: {"totalFunds": 4_000.0}})
    corpus = [_question("totalFunds", 500.0)]
    assert make_normalizer("local", corpus).scale(corpus[0]) is None


def test_local_has_no_scale_where_the_file_has_no_mean(tmp_path, monkeypatch):
    _write_ground_truth(tmp_path, monkeypatch, {48: {"cityPop": None}})
    corpus = [_question("cityPop", 100.0)]
    norm = make_normalizer("local", corpus)
    assert norm.scale(corpus[0]) is None
    # And that is exactly what the coverage check is there to catch.
    assert norm.unscaled_metrics(corpus) == ["cityPop"]


def test_local_errors_on_a_scenario_with_no_ground_truth(tmp_path, monkeypatch):
    """Rather than normalizing part of the config against something else."""
    _write_ground_truth(tmp_path, monkeypatch, {48: {"cityPop": 5_000.0}})
    absent = dict(_question("cityPop", 100.0), scenario_id="other")
    with pytest.raises(FileNotFoundError, match="extract_ground_truth"):
        make_normalizer("local", [absent])
    with pytest.raises(FileNotFoundError, match="no horizon 480"):
        make_normalizer("local", [_question("cityPop", 100.0, horizon=480)])


def test_local_warns_where_the_offset_sets_the_scale(tmp_path, monkeypatch, capsys):
    """A near-zero mean inflates that cell, so it must not pass in silence."""
    _write_ground_truth(
        tmp_path,
        monkeypatch,
        {48: {"cityPop": 0.2, "crimeAverage": 30.0}},
    )
    corpus = [_question("cityPop", 1.0), _question("crimeAverage", 30.0)]
    make_normalizer("local", corpus)
    err = capsys.readouterr().err
    assert "1 question(s) averaged near 0" in err
    assert "cityPop" in err
    assert "crimeAverage" not in err
