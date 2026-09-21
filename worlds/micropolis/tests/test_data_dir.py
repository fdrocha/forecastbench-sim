"""A config's 'data_dir' moves the whole world under another directory of data/.

The point is a fresh response cache: the cache is content-addressed and never
invalidated, so re-asking a prompt means asking it somewhere the earlier run
never wrote. Everything derived from the data directory has to follow, and
has to be read at call time, since the config is loaded after every import.
"""

import json

import pytest

import micropolis_world.module_globals as g
from micropolis_world import binary_eval, continuous_eval, ground_truth
from micropolis_world.config import Config, ConfigError
from micropolis_world.knowledge_eval import runner


@pytest.fixture
def fresh_world(monkeypatch):
    """A process that has not yet fixed its data directory."""
    monkeypatch.setattr(g, "DATA_DIR", g.DATA_ROOT / g.DEFAULT_DATA_SUBDIR)
    monkeypatch.setattr(g, "RUNS_DIR", g.DATA_DIR / "runs")
    monkeypatch.setattr(g, "_data_dir_source", None)


def write_config(tmp_path, **extra):
    data = {"seed": 1, "label": "x", **extra}
    path = tmp_path / "cfg.json5"
    path.write_text(json.dumps(data))
    return path


def test_default_is_the_micropolis_directory(fresh_world, tmp_path):
    Config.load(write_config(tmp_path))
    assert g.DATA_DIR == g.DATA_ROOT / "micropolis"


def test_data_dir_moves_every_derived_path(fresh_world, tmp_path):
    Config.load(write_config(tmp_path, data_dir="micropolis_check"))
    root = g.DATA_ROOT / "micropolis_check"
    assert g.DATA_DIR == root
    assert g.RUNS_DIR == root / "runs"
    assert binary_eval.PATHS.out_dir == root / "binary"
    assert binary_eval.data_path("lbl") == root / "binary" / "lbl" / "data.json"
    assert continuous_eval.PATHS.out_dir == root / "continuous"
    assert continuous_eval.data_path("lbl") == root / "continuous" / "lbl" / "data.json"
    assert (
        ground_truth.output_path_for("s", 240) == root / "ground_truth" / "s_T240.jsonl"
    )
    assert runner.cache_dir() == root / "knowledge_eval" / "cache"
    # The cache path a run would write to is under the new root too.
    assert binary_eval.PATHS.response_path("b", "m", "h").is_relative_to(root)


def test_two_configs_agreeing_is_fine(fresh_world, tmp_path):
    Config.load(write_config(tmp_path, data_dir="micropolis_check"))
    Config.load(write_config(tmp_path, data_dir="micropolis_check"))
    assert g.DATA_DIR.name == "micropolis_check"


def test_two_configs_disagreeing_is_an_error(fresh_world, tmp_path):
    Config.load(write_config(tmp_path, data_dir="micropolis_check"))
    other = tmp_path / "other.json5"
    other.write_text(json.dumps({"seed": 1, "label": "y"}))  # the default world
    with pytest.raises(ConfigError, match="conflicts"):
        Config.load(other)


@pytest.mark.parametrize("bad", ["", ".", "..", "a/b", " x"])
def test_data_dir_must_be_one_directory_name(fresh_world, tmp_path, bad):
    with pytest.raises(ConfigError, match="data_dir"):
        Config.load(write_config(tmp_path, data_dir=bad))
