"""Tests for micropolis_world.model_scores' scores.csv export.

The reading side is exercised throughout the analysis tests; what needs pinning
here is the artifact writer: that it re-emits the source file as a strict
superset — every row and column preserved — with the run's own scores joined on
by bare model name.
"""

import csv

import pytest

from micropolis_world import model_scores

# A miniature model_scores.csv: one model with a slug (joinable), one without
# (a leaderboard row this repo cannot run), matching the real file's shape.
SOURCE = """\
ECI,Name,slug,FBName,FBOverall,FBOverallCILo,FBOverallCIHi
150,OpenAI: GPT-5,openai/gpt-5,gpt-5-2025-08-07,60.5,59.5,61.5
130.52,Meta: Llama 4 Scout,,llama-4-scout-17b-16e-instruct,56.6,56.0,57.2
"""

MP_COLUMNS = ["MPScore", "MPScoreLo", "MPScoreHi"]


@pytest.fixture
def source(tmp_path, monkeypatch):
    src = tmp_path / "model_scores.csv"
    src.write_text(SOURCE)
    monkeypatch.setattr(model_scores, "SCORES_PATH", src)
    # load_scores is cached; make sure it reads this file and, afterwards,
    # that no other test inherits the miniature.
    model_scores.load_scores.cache_clear()
    yield src
    model_scores.load_scores.cache_clear()


def test_a_suffixed_slug_joins_its_model_ids_scores(source):
    """Leaderboards score the model, not the effort setting."""
    assert model_scores.eci_of("openai/gpt-5:lowef") == model_scores.eci_of(
        "openai/gpt-5"
    )
    assert model_scores.eci_of("openai/gpt-5:lowef") == 150.0
    # The per-slug key stays distinct: the suffix is part of the label.
    assert model_scores.eci_by_name(["openai/gpt-5", "openai/gpt-5:lowef"]) == {
        "gpt-5": 150.0,
        "gpt-5:lowef": 150.0,
    }


def _written(out) -> tuple[list[str], list[dict]]:
    with out.open(newline="") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames), list(reader)


def test_write_scores_csv_appends_columns_and_joins_on_name(source, tmp_path):
    out = model_scores.write_scores_csv(
        tmp_path / "scores.csv", {"openai/gpt-5": (0.5464, 0.4611, 0.6474)}
    )
    columns, rows = _written(out)
    assert columns == SOURCE.splitlines()[0].split(",") + MP_COLUMNS

    scored, unscored = rows
    assert scored["slug"] == "openai/gpt-5"
    assert [scored[c] for c in MP_COLUMNS] == ["0.5464", "0.4611", "0.6474"]
    # The source columns come through verbatim beside the new ones.
    assert (scored["ECI"], scored["FBOverall"]) == ("150", "60.5")
    # A row this run never scored is kept, with blank score cells.
    assert unscored["Name"] == "Meta: Llama 4 Scout"
    assert [unscored[c] for c in MP_COLUMNS] == ["", "", ""]


def test_write_scores_csv_blanks_an_unbounded_interval(source, tmp_path):
    """Bounds of None mean too few clusters, spelled as the file's blank cell."""
    out = model_scores.write_scores_csv(
        tmp_path / "scores.csv", {"openai/gpt-5": (1.5, None, None)}
    )
    _columns, rows = _written(out)
    assert [rows[0][c] for c in MP_COLUMNS] == ["1.5000", "", ""]


def test_write_scores_csv_appends_a_model_the_source_lacks(source, tmp_path):
    """A scored model missing from the source still lands in the artifact."""
    out = model_scores.write_scores_csv(
        tmp_path / "scores.csv", {"prov/new-model": (0.9, 0.8, 1.0)}
    )
    _columns, rows = _written(out)
    added = rows[-1]
    assert added["slug"] == "prov/new-model"
    assert [added[c] for c in MP_COLUMNS] == ["0.9000", "0.8000", "1.0000"]
    # The external-benchmark columns are honestly blank, not fabricated.
    assert (added["ECI"], added["FBOverall"]) == ("", "")


def test_write_scores_csv_gives_a_suffixed_slug_its_own_row(source, tmp_path):
    """MPScore is per slug, so the variant cannot share its base model's row; it
    is appended as a copy of that row carrying the same external scores."""
    out = model_scores.write_scores_csv(
        tmp_path / "scores.csv",
        {"openai/gpt-5": (0.5, 0.4, 0.6), "openai/gpt-5:lowef": (0.7, 0.6, 0.8)},
    )
    _columns, rows = _written(out)
    base, _unscored, variant = rows
    assert base["slug"] == "openai/gpt-5"
    assert base["MPScore"] == "0.5000"
    assert variant["slug"] == "openai/gpt-5:lowef"
    assert variant["MPScore"] == "0.7000"
    assert (variant["ECI"], variant["FBOverall"]) == ("150", "60.5")
    assert variant["Name"] == base["Name"]
