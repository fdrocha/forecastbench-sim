"""Tests for scripts/analyze_baseline_skill.py.

The skill score is a ratio, which brings failure modes the |actual|-normalized
score does not have: a zero denominator, a zero numerator, and an aggregation
that has to be geometric rather than arithmetic for the scale to stay symmetric.
These pin those down.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"


def load_module():
    """Import the analysis script, which is not on the package path."""
    spec = importlib.util.spec_from_file_location(
        "analyze_baseline_skill", SCRIPTS_DIR / "analyze_baseline_skill.py"
    )
    module = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, ["analyze_baseline_skill"]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


def _rows(*skills: float) -> list[dict]:
    """Rows carrying just what the aggregation helpers read."""
    import math

    return [
        {
            "model_id": "m",
            "metric": "cityPop",
            "horizon": 48,
            "disasters": False,
            "skill": s,
            "log_skill": math.log(s),
        }
        for s in skills
    ]


def test_geometric_mean_is_symmetric_around_parity():
    """Twice as good and twice as bad must average to the baseline, not above it.

    This is the property that makes the ratio averageable at all; an arithmetic
    mean of 0.5 and 2.0 gives 1.25, which would report a model that did equally
    well and badly as worse than the baseline.
    """
    module = load_module()
    assert module.geometric_mean(_rows(0.5, 2.0)) == pytest.approx(1.0)


def test_geometric_mean_resists_a_single_huge_ratio():
    """One question where the baseline nearly nailed it must not swamp a column."""
    module = load_module()
    modest = _rows(*([0.9] * 99))
    with_outlier = _rows(*([0.9] * 99 + [500.0]))
    # Arithmetic would jump from 0.9 to ~5.9; geometric stays near the bulk.
    assert module.geometric_mean(with_outlier) < 1.0
    assert module.geometric_mean(modest) == pytest.approx(0.9)


def test_geometric_mean_of_nothing_is_none():
    module = load_module()
    assert module.geometric_mean([]) is None


def test_baseline_percentiles_plain_is_degenerate():
    """All five quantiles on the snapshot value, so CRPS reduces to |error|."""
    module = load_module()
    got = module.baseline_percentiles(100.0, 25.0, "plain")
    assert set(got.values()) == {100.0}


def test_baseline_percentiles_sigma_widens_symmetrically():
    module = load_module()
    got = module.baseline_percentiles(100.0, 10.0, "sigma")
    assert got["p50"] == pytest.approx(100.0)
    assert got["p10"] == pytest.approx(100.0 - 12.816)
    assert got["p90"] == pytest.approx(100.0 + 12.816)


def test_baseline_percentiles_sigma_needs_a_sigma():
    """A missing spread is a gap, not a zero spread; it must not silently
    collapse to the degenerate baseline, which would score a different forecast
    under the sigma baseline's name."""
    module = load_module()
    assert module.baseline_percentiles(100.0, None, "sigma") is None
    # The plain baseline never consults sigma, so None is fine there.
    assert module.baseline_percentiles(100.0, None, "plain") is not None


def _corpus(snapshot: float, actual: float, metric: str = "cityPop") -> list[dict]:
    return [
        {
            "question_id": f"q-{h}",
            "scenario_id": "s",
            "snapshot_turn": 240,
            "metric": metric,
            "horizon": h,
            "value": v,
            "scenario": {"disasters": False},
        }
        for h, v in [(0, snapshot), (48, actual)]
    ]


def test_baseline_crps_drops_a_zero_scoring_baseline():
    """A baseline that is exactly right gives no ratio, so the question is out.

    The plain baseline hits this whenever the metric did not move, which is 12%
    of the real corpus — the reason the dropped counts are always reported.
    """
    module = load_module()
    # Actual equals the snapshot, so plain persistence scores exactly 0.
    got = module.baseline_crps(_corpus(100, 100), 42, "plain")
    assert got == {}
    # A moved metric is kept.
    assert module.baseline_crps(_corpus(100, 150), 42, "plain") == {"q-48": 50.0}


def test_baseline_crps_excludes_the_read_off_horizon():
    module = load_module()
    got = module.baseline_crps(_corpus(100, 150), 42, "plain")
    assert "q-0" not in got


def test_score_skill_reports_why_pairs_were_dropped(monkeypatch):
    """The tally is what keeps a shrunken table from reading as a complete one."""
    module = load_module()
    from micropolis_world.single_city import Response, ResponseId

    corpus = _corpus(100, 150)
    responses = {
        ResponseId("p/parsed", "q-48"): Response(
            actual=150, percentiles=dict.fromkeys(module.NORMAL_Z, 140.0)
        ),
        ResponseId("p/unparsed", "q-48"): Response(actual=150, percentiles=None),
        ResponseId("p/exact", "q-48"): Response(
            actual=150, percentiles=dict.fromkeys(module.NORMAL_Z, 150.0)
        ),
    }
    rows, dropped = module.score_skill(
        corpus, responses, ["p/parsed", "p/unparsed", "p/exact"], 42, "plain"
    )
    assert [r["model_id"] for r in rows] == ["p/parsed"]
    assert dropped["unparsed"] == 1
    assert dropped["zero_model_crps"] == 1
    # 10 off against a baseline 50 off.
    assert rows[0]["skill"] == pytest.approx(10 / 50)


def test_score_skill_carries_the_disasters_flag():
    """The horizon figures split on it, so it has to travel with the row."""
    module = load_module()
    from micropolis_world.single_city import Response, ResponseId

    corpus = _corpus(100, 150)
    for c in corpus:
        c["scenario"] = {"disasters": True}
    responses = {
        ResponseId("p/m", "q-48"): Response(
            actual=150, percentiles=dict.fromkeys(module.NORMAL_Z, 140.0)
        )
    }
    rows, _ = module.score_skill(corpus, responses, ["p/m"], 42, "plain")
    assert [r["disasters"] for r in rows] == [True]


def test_split_rows_never_mixes_city_funds_with_the_others():
    """The split is the whole point: funds is a near-deterministic series and
    pooling it moved the headline from 2 of 18 models to 6 of 18."""
    module = load_module()
    rows = [
        {"metric": module.FUNDS_METRIC, "skill": 0.01},
        {"metric": "cityPop", "skill": 1.0},
        {"metric": "crimeAverage", "skill": 1.0},
    ]
    assert [r["metric"] for r in module.split_rows(rows, "funds")] == [
        module.FUNDS_METRIC
    ]
    assert module.FUNDS_METRIC not in [
        r["metric"] for r in module.split_rows(rows, "behavioral")
    ]
    # Every row lands on exactly one side.
    assert len(module.split_rows(rows, "funds")) + len(
        module.split_rows(rows, "behavioral")
    ) == len(rows)


def test_skill_by_groups_geometrically():
    module = load_module()
    rows = _rows(0.5, 2.0)
    rows[1]["horizon"] = 96
    assert module.skill_by(rows, "model_id")[("m",)] == pytest.approx(1.0)
    assert module.skill_by(rows, "horizon")[(48,)] == pytest.approx(0.5)
    assert module.skill_by(rows, "horizon")[(96,)] == pytest.approx(2.0)


def test_baseline_and_split_notes_name_every_choice():
    """Each table and figure has to say which baseline and which side it is."""
    module = load_module()
    for kind in module.BASELINES:
        assert module.BASELINES[kind][0] in module.baseline_note(kind)
    for split in module.SPLITS:
        assert module.SPLITS[split][0] in module.split_note(split)


def test_metrics_in_order_all_keeps_funds_in_its_natural_place():
    """metrics_in_order sorts funds last for the |actual|-normalized script; here
    it is a full participant on its own side of the split."""
    module = load_module()
    rows = [{"metric": module.FUNDS_METRIC}, {"metric": "cityPop"}]
    got = module.metrics_in_order_all(rows)
    assert set(got) == {module.FUNDS_METRIC, "cityPop"}


def test_baseline_crps_sigma_path_needs_a_cached_run(monkeypatch):
    """The spread-widened baseline reads the run log; a missing one drops the
    question rather than falling back to a different forecast."""
    module = load_module()
    monkeypatch.setattr(module, "scenario_history", lambda _sid, _seed: None)
    assert module.baseline_crps(_corpus(100, 150), 42, "sigma") == {}


def test_baseline_crps_plain_path_needs_no_cached_run(monkeypatch):
    """Plain persistence never consults the history, so it must not depend on it."""
    module = load_module()

    def fail(*_args):
        raise AssertionError("the plain baseline must not read a run log")

    monkeypatch.setattr(module, "scenario_history", fail)
    assert module.baseline_crps(_corpus(100, 150), 42, "plain") == {"q-48": 50.0}


def test_baseline_crps_sigma_uses_the_history_spread(monkeypatch):
    """A wider history gives a wider interval and so a different baseline CRPS."""
    module = load_module()
    flat = [{"cityPop": 100} for _ in range(300)]
    wandering = [{"cityPop": 100 + (i % 9) * 20} for i in range(300)]

    monkeypatch.setattr(module, "scenario_history", lambda _s, _d: flat)
    tight = module.baseline_crps(_corpus(100, 150), 42, "sigma")
    monkeypatch.setattr(module, "scenario_history", lambda _s, _d: wandering)
    loose = module.baseline_crps(_corpus(100, 150), 42, "sigma")

    # A flat history gives sigma 0, collapsing to the degenerate |error| of 50.
    assert tight["q-48"] == pytest.approx(50.0)
    # A real spread puts probability nearer the actual, so CRPS is lower.
    assert loose["q-48"] < tight["q-48"]


def test_geometric_mean_of_takes_plain_ratios():
    """The figures' mean-over-models line averages cells, not rows."""
    module = load_module()
    assert module.geometric_mean_of([0.5, 2.0]) == pytest.approx(1.0)
    assert module.geometric_mean_of([]) is None


def test_skill_by_keeps_a_cell_whose_mean_is_not_truthy():
    """ "No questions here" and "scored 0 here" must not collapse together.

    A geometric mean of positive ratios cannot be 0 today, so this guards the
    distinction rather than a live bug: a falsiness filter would start dropping
    real cells the moment a 0 became reachable.
    """
    module = load_module()
    import math

    rows = [
        {
            "model_id": "m",
            "metric": "cityPop",
            "horizon": 48,
            "disasters": False,
            "skill": 1.0,
            "log_skill": 0.0,
        }
    ]
    assert module.skill_by(rows, "model_id") == {("m",): 1.0}
    # Patch in a group that averages to 0 and confirm it is reported, not dropped.
    monkey = [dict(rows[0], log_skill=-math.inf, skill=0.0)]
    assert module.skill_by(monkey, "model_id") == {("m",): 0.0}


def test_relabel_correlation_axes_rewrites_the_inherited_footnote():
    """The shared helper's footnote describes the other script's axes.

    It reads "the better-scoring models forecast better", which is wrong here:
    x is a predictor, not a score, and y is a ratio against the baseline.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    module = load_module()
    fig, ax = plt.subplots()
    ax.annotate("ρ<0: the better-scoring models forecast better (pro-g)", xy=(0, 0))
    module.relabel_correlation_axes(ax)
    texts = [t.get_text() for t in ax.texts]
    assert not any("better-scoring" in t for t in texts)
    assert any("predictor" in t for t in texts)
    assert "CRPS_baseline" in ax.get_ylabel()
    plt.close(fig)
