"""Tests for scripts/analyze_baseline_skill.py.

The score is a ratio, which brings failure modes the |actual|-normalized score
does not have: a zero denominator, a zero numerator, and an aggregation that
has to be geometric rather than arithmetic for the scale to stay symmetric.
These pin those down, along with the clustered confidence intervals — which
analyze_skill_by_config.py imports from here, so what is tested once holds for
both scripts.
"""

import importlib.util
import math
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


def _rows(*skills: float, model: str = "p/m", cluster: str | None = None) -> list[dict]:
    """Rows carrying just what the aggregation helpers read.

    Each row gets its own (scenario, snapshot) cluster unless `cluster` pins
    them all to one, so the mean tests are not entangled with the clustering.
    """
    return [
        {
            "model_id": model,
            "metric": "cityPop",
            "horizon": 48,
            "scenario_id": cluster or f"s{i}",
            "snapshot_turn": 1440,
            "skill": s,
            "log_skill": math.log(s),
        }
        for i, s in enumerate(skills)
    ]


def test_score_stats_mean_is_symmetric_around_parity():
    """Twice as good and twice as bad must average to the baseline, not above it.

    This is the property that makes the ratio averageable at all; an arithmetic
    mean of 0.5 and 2.0 gives 1.25, which would report a model that did equally
    well and badly as worse than the baseline.
    """
    module = load_module()
    assert module.score_stats(_rows(0.5, 2.0))[0] == pytest.approx(1.0)


def test_score_stats_mean_resists_a_single_huge_ratio():
    """One question where the baseline nearly nailed it must not swamp the mean."""
    module = load_module()
    modest = module.score_stats(_rows(*([0.9] * 99)))
    with_outlier = module.score_stats(_rows(*([0.9] * 99 + [500.0])))
    # Arithmetic would jump from 0.9 to ~5.9; geometric stays near the bulk.
    assert with_outlier[0] < 1.0
    assert modest[0] == pytest.approx(0.9)


def test_score_stats_needs_a_row():
    module = load_module()
    with pytest.raises(ValueError):
        module.score_stats([])


def test_score_stats_withholds_a_bar_below_min_clusters():
    """Too few trajectories means a point with no bar, not a fabricated one."""
    module = load_module()
    thin = module.score_stats(_rows(0.5, 1.0, 2.0))
    assert thin[1] is None and thin[2] is None
    assert thin[3] == 3
    mean, lo, hi, n = module.score_stats(_rows(0.5, 0.8, 1.25, 2.0))
    assert lo is not None and hi is not None
    assert lo < mean < hi
    assert n == 4


def test_score_stats_interval_clusters_on_trajectories():
    """Repeating a trajectory's questions must not shrink the interval.

    Questions read off one simulated history are not independent draws, so the
    spread is taken over per-cluster means. Doubling every question within its
    own cluster leaves those means untouched; a per-question interval would
    tighten by roughly sqrt(2) and overstate the precision.
    """
    module = load_module()
    base = _rows(0.5, 0.8, 1.25, 2.0)
    doubled = base + [dict(r) for r in base]
    one = module.score_stats(base)
    two = module.score_stats(doubled)
    assert two[0] == pytest.approx(one[0])
    assert two[1] == pytest.approx(one[1])
    assert two[2] == pytest.approx(one[2])
    # Only the question count moves; it reports how much data, not how sure.
    assert (one[3], two[3]) == (4, 8)


def test_score_stats_interval_is_multiplicative():
    """The bounds bracket the geometric mean in ratio space, not additively.

    With cluster means symmetric in log space, lo * hi must equal mean^2 — the
    multiplicative analogue of an interval centered on its estimate.
    """
    module = load_module()
    mean, lo, hi, _n = module.score_stats(_rows(0.5, 1.0, 1.0, 2.0))
    assert lo * hi == pytest.approx(mean * mean)


def test_stats_by_model_groups_by_model():
    module = load_module()
    rows = _rows(0.5, model="p/a") + _rows(2.0, model="p/b")
    got = module.stats_by_model(rows)
    assert got["p/a"][0] == pytest.approx(0.5)
    assert got["p/b"][0] == pytest.approx(2.0)


def test_error_arms_are_asymmetric_and_tolerate_missing_bounds():
    """A multiplicative interval has unequal arms; no bounds means no bar."""
    module = load_module()
    lower, upper = module.error_arms([(1.0, 0.8, 1.5, 4), (0.9, None, None, 2)])
    assert lower == [pytest.approx(0.2), 0.0]
    assert upper == [pytest.approx(0.5), 0.0]


def test_ordered_by_score_puts_best_first_and_unscored_last():
    module = load_module()
    cells = {"p/good": (0.5, None, None, 1), "p/bad": (2.0, None, None, 1)}
    got = module.ordered_by_score(cells, ["p/none", "p/bad", "p/good"])
    assert got == ["p/good", "p/bad", "p/none"]


def test_format_score_cell_prints_the_interval_beside_the_mean():
    module = load_module()
    assert module.format_score_cell(None) == "-"
    assert module.format_score_cell((0.5, None, None, 3)) == "0.500"
    assert module.format_score_cell((0.5, 0.4, 0.6, 30)) == "0.500 [0.400, 0.600]"


def _table_lines(report) -> list[str]:
    """The lines of the table just appended to `report`.

    base_dir is irrelevant here: none of these tests call report.image(), so
    render() never resolves a relative link against it.
    """
    text = report.render(Path("."))
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Model"))
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("```"))
    return lines[start:end]


def test_model_scores_table_is_sorted_best_first_without_ranks():
    module = load_module()
    report = module.MdReport()
    rows = _rows(2.0, model="p/worse") + _rows(0.5, model="p/better")
    module.print_model_scores(report, rows, ["p/worse", "p/better"], "plain")

    lines = _table_lines(report)
    header, body = lines[0], lines[2:]
    assert "#scored" in header and "score [95% CI]" in header
    assert [line.split()[0] for line in body] == ["better", "worse"]
    # Plain means only: the old tables' per-cell "(rank)" annotations are gone.
    assert not any("(" in line for line in body)


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


def test_score_skill_carries_the_cluster_identity():
    """The intervals cluster on (scenario, snapshot), so both travel on the row."""
    module = load_module()
    from micropolis_world.single_city import Response, ResponseId

    corpus = _corpus(100, 150)
    responses = {
        ResponseId("p/m", "q-48"): Response(
            actual=150, percentiles=dict.fromkeys(module.NORMAL_Z, 140.0)
        )
    }
    rows, _ = module.score_skill(corpus, responses, ["p/m"], 42, "plain")
    assert [module.cluster_key(r) for r in rows] == [("s", 240)]


def test_split_rows_never_mixes_city_funds_with_the_others():
    """The split now serves analyze_skill_by_config.py, which imports it from
    here; funds is a near-deterministic series the comparison reports apart."""
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


def test_baseline_and_split_notes_name_every_choice():
    """Each table and figure has to say which baseline and which side it is."""
    module = load_module()
    for kind in module.BASELINES:
        assert module.BASELINES[kind][0] in module.baseline_note(kind)
    for split in module.SPLITS:
        assert module.SPLITS[split][0] in module.split_note(split)


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
