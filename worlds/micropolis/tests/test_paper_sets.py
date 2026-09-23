"""Tests for the paper's binary question sets and the bands figure built on them.

Nothing here calls a model or reads the data directory.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [name]
    try:
        spec.loader.exec_module(module)
    finally:
        sys.argv = argv
    return module


gp = load("gather_paper_data")
ap = load("analyze_paper")


@pytest.mark.parametrize(
    "q, expected",
    [
        (0.0, gp.ZERO),
        (0.001, gp.TAIL),
        (0.05, gp.TAIL),  # FreeCiv's 0 < q <= 0.05
        (50 / 1000, gp.TAIL),
        (0.051, gp.MID_RANGE),
        (0.949, gp.MID_RANGE),
        (0.95, gp.TOP),  # FreeCiv's mirror set, 0.95 <= q < 1
        (0.999, gp.TOP),
        (1.0, gp.TOP),
    ],
)
def test_question_set_boundaries(q, expected):
    assert gp.question_set(q) == expected


def test_every_band_lies_in_the_set_of_its_members():
    qs = [i / 1000 for i in range(1001)]
    for q in qs:
        assert ap.band_set(ap.band_of(q)) == gp.question_set(q), q


def test_bands_cover_zero_and_one_alone():
    assert ap.band_of(0.0) == 0
    assert ap.band_of(1.0) == len(ap.BAND_EDGES)
    assert ap.band_of(0.001) != 0
    assert ap.band_of(0.999) != len(ap.BAND_EDGES)


def test_assign_sets_rejects_a_stale_section(tmp_path):
    rows = [{"real_prob": 0.05, "section": gp.MID_RANGE}]
    with pytest.raises(SystemExit):
        ap.assign_sets(rows, tmp_path / "x.csv")
    rows = [{"real_prob": 0.05, "section": gp.TAIL}]
    ap.assign_sets(rows, tmp_path / "x.csv")
    assert rows[0]["section"] == gp.TAIL
