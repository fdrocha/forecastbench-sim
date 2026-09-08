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


def _write_ground_truth(tmp_path, monkeypatch, by_horizon, snapshot=None):
    """A tally file carrying the values the per-question modes read.

    `by_horizon` maps horizon to {metric: [continuation outcomes]}; `snapshot`
    is the log row the persistence baseline forecasts from, patched in as the
    scenario's cached run so no engine is needed.
    """
    import micropolis_world.continuous_eval as ce
    import micropolis_world.ground_truth as gt

    monkeypatch.setattr(gt, "OUT_DIR", tmp_path)
    lines = [
        {
            "scenario_id": "s",
            "snapshot_turn": 240,
            "horizon": horizon,
            "n_continuations": len(next(iter(values.values()))),
            "values": values,
            "averages": {
                m: (sum(v) / len(v) if v else None) for m, v in values.items()
            },
        }
        for horizon, values in by_horizon.items()
    ]
    gt.write_lines(gt.output_path_for("s", 240), lines)
    if snapshot is not None:
        history = [dict.fromkeys(snapshot, 0) for _ in range(241)]
        history[240] = snapshot
        monkeypatch.setattr(ce, "scenario_history", lambda _sid, _seed: history)


def test_local_scale_is_the_horizons_own_mean(tmp_path, monkeypatch):
    """Each question divides by its own (scenario, snapshot, horizon) mean."""
    _write_ground_truth(
        tmp_path,
        monkeypatch,
        {48: {"cityPop": [4_000.0, 6_000.0]}, 96: {"cityPop": [9_000.0, 11_000.0]}},
    )
    corpus = [
        _question("cityPop", 1.0, horizon=48),
        _question("cityPop", 2.0, horizon=96),
    ]
    norm = make_normalizer("local", corpus)
    assert norm.scale(corpus[0]) == pytest.approx(5_000.0)
    assert norm.scale(corpus[1]) == pytest.approx(10_000.0)
    assert norm.floored.n == 0


def test_local_floors_a_zero_mean_at_a_share_of_the_global_scale(tmp_path, monkeypatch):
    """The case the floor exists for: a metric that could not move at all."""
    _write_ground_truth(tmp_path, monkeypatch, {48: {"trafficAverage": [0.0, 0.0]}})
    corpus = [_question("trafficAverage", 3.0)]
    norm = make_normalizer("local", corpus, global_frac=0.01)
    # 1% of trafficAverage's global scale of 20, not a division by zero.
    assert norm.scale(corpus[0]) == pytest.approx(0.2)
    assert norm.floored.n == 1
    assert norm.floored.by_metric == {"trafficAverage": 1}


def test_the_floor_is_configurable(tmp_path, monkeypatch):
    """--norm-global-frac moves the floor, and so what it binds on."""
    _write_ground_truth(tmp_path, monkeypatch, {48: {"trafficAverage": [0.1, 0.1]}})
    corpus = [_question("trafficAverage", 3.0)]
    # A mean of 0.1 clears a 0.005 floor (0.1 of a scale of 20) but not a 0.01 one.
    loose = make_normalizer("local", corpus, global_frac=0.00025)
    assert loose.scale(corpus[0]) == pytest.approx(0.1)
    assert loose.floored.n == 0
    tight = make_normalizer("local", corpus, global_frac=0.05)
    assert tight.scale(corpus[0]) == pytest.approx(1.0)
    assert tight.floored.n == 1


def test_baseline_scale_is_the_expected_persistence_crps(tmp_path, monkeypatch):
    """Mean |snapshot - outcome| over the continuations, not over one draw."""
    _write_ground_truth(
        tmp_path,
        monkeypatch,
        {48: {"cityPop": [900.0, 1_100.0, 1_000.0, 2_000.0]}},
        snapshot={"cityPop": 1_000.0},
    )
    corpus = [_question("cityPop", 12_345.0)]
    norm = make_normalizer("baseline", corpus, seed=42)
    # |1000-900| + |1000-1100| + 0 + 1000, over 4 continuations.
    assert norm.scale(corpus[0]) == pytest.approx(300.0)
    assert norm.floored.n == 0


def test_baseline_floors_a_metric_that_never_moves(tmp_path, monkeypatch):
    """Expected persistence CRPS is 0 exactly where every future is identical."""
    _write_ground_truth(
        tmp_path,
        monkeypatch,
        {48: {"trafficAverage": [0.0, 0.0, 0.0]}},
        snapshot={"trafficAverage": 0.0},
    )
    corpus = [_question("trafficAverage", 0.0)]
    norm = make_normalizer("baseline", corpus, global_frac=0.01, seed=42)
    assert norm.scale(corpus[0]) == pytest.approx(0.2)
    assert norm.floored.n == 1


def test_baseline_needs_a_seed(tmp_path, monkeypatch):
    """The snapshot value comes from a cached run, which the seed names."""
    _write_ground_truth(tmp_path, monkeypatch, {48: {"cityPop": [1.0, 2.0]}})
    with pytest.raises(ValueError, match="seed"):
        make_normalizer("baseline", [_question("cityPop", 1.0)])


def test_the_excluded_metric_stays_unnormalized_under_every_mode(tmp_path, monkeypatch):
    """City funds has no scale to take a share of, so no floor rescues it."""
    _write_ground_truth(
        tmp_path,
        monkeypatch,
        {48: {"totalFunds": [4_000.0, 5_000.0]}},
        snapshot={"totalFunds": 4_000.0},
    )
    corpus = [_question("totalFunds", 500.0)]
    for norm in (
        make_normalizer("local", corpus),
        make_normalizer("baseline", corpus, seed=42),
    ):
        assert norm.scale(corpus[0]) is None, norm.mode
        # And it is not counted as floored: it was never in the running.
        assert norm.floored.n == 0
        assert norm.floored.total == 0


def test_floored_note_reports_the_count_and_share(tmp_path, monkeypatch):
    """The line the report and stdout both carry."""
    _write_ground_truth(
        tmp_path,
        monkeypatch,
        {48: {"trafficAverage": [0.0, 0.0], "cityPop": [5_000.0, 5_000.0]}},
    )
    corpus = [_question("trafficAverage", 1.0), _question("cityPop", 1.0)]
    norm = make_normalizer("local", corpus, global_frac=0.01)
    note = norm.floored.note()
    assert "bound on 1 of 2 question(s) (50.0%)" in note
    assert "average traffic 1" in note
    assert norm.floored.share == pytest.approx(0.5)


def test_floored_note_says_so_when_nothing_was_floored(tmp_path, monkeypatch):
    """Stated either way, so a silent report is not ambiguous."""
    _write_ground_truth(tmp_path, monkeypatch, {48: {"cityPop": [5_000.0, 5_000.0]}})
    corpus = [_question("cityPop", 1.0)]
    norm = make_normalizer("local", corpus)
    assert "bound on no question" in norm.floored.note()


def test_local_errors_on_a_scenario_with_no_ground_truth(tmp_path, monkeypatch):
    """Rather than normalizing part of the config against something else."""
    _write_ground_truth(tmp_path, monkeypatch, {48: {"cityPop": [1.0, 2.0]}})
    absent = dict(_question("cityPop", 100.0), scenario_id="other")
    with pytest.raises(FileNotFoundError, match="extract_ground_truth"):
        make_normalizer("local", [absent])
    with pytest.raises(FileNotFoundError, match="no horizon 480"):
        make_normalizer("local", [_question("cityPop", 100.0, horizon=480)])


def _cfg(data: dict):
    from pathlib import Path

    from micropolis_world.config import Config

    return Config(data, Path("test.json5"))


def test_norm_global_frac_defaults_and_overrides():
    from micropolis_world.config import DEFAULT_NORM_GLOBAL_FRAC

    # Absent from the config: configs written before the key keep working.
    assert _cfg({}).get_norm_global_frac() == DEFAULT_NORM_GLOBAL_FRAC
    assert _cfg({"norm_global_frac": 0.02}).get_norm_global_frac() == 0.02
    # --norm-global-frac wins over the config's own value.
    assert _cfg({"norm_global_frac": 0.02}).get_norm_global_frac(0.005) == 0.005
    # An int is a fine share; a config writing 0 turns the floor off.
    assert _cfg({"norm_global_frac": 0}).get_norm_global_frac() == 0.0


@pytest.mark.parametrize("bad", [1.0, 1.5, -0.1])
def test_norm_global_frac_rejects_shares_outside_the_unit_interval(bad):
    """At 1 every question is floored onto the global scale, which is --norm global."""
    from micropolis_world.config import ConfigError

    with pytest.raises(ConfigError, match="norm_global_frac"):
        _cfg({}).get_norm_global_frac(bad)


def test_norm_global_frac_rejects_a_non_number():
    from micropolis_world.config import ConfigError

    with pytest.raises(ConfigError, match="must be a number"):
        _cfg({"norm_global_frac": "1%"}).get_norm_global_frac()


def _analysis_module():
    """scripts/analyze_continuous.py, which is not importable as a package."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).parent.parent / "scripts" / "analyze_continuous.py"
    spec = importlib.util.spec_from_file_location("analyze_continuous", path)
    module = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["analyze_continuous"]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


def _stub_normalizer(mode: str):
    """A Normalizer of `mode` without the ground truth the real one reads."""
    from micropolis_world.continuous_eval import Normalizer

    return Normalizer(
        mode=mode, ratio="r", detail="d", scale=lambda _c: 1.0, floored=None
    )


def test_norm_suffix_tags_every_mode_but_the_default():
    """The default keeps the unsuffixed names an existing label already has."""
    norm_suffix = _analysis_module().norm_suffix
    assert norm_suffix(make_normalizer("global", [])) == ""
    assert norm_suffix(_stub_normalizer("local")) == "-local"
    assert norm_suffix(_stub_normalizer("baseline")) == "-baseline"
