"""Unit tests for the usage sidecars stored beside cached model responses.

Both eval caches key a response by model and prompt hash, and each stores what
the call cost in a JSON file named the same way. What matters is that the two
filenames agree — the sidecar is useless if it doesn't land next to the response
it describes — and that a response cached before cost tracking existed still
reads back cleanly as "cost unknown".

Nothing here calls a model.
"""

import json

from micropolis_world.continuous_eval import (
    load_usage,
    response_path,
    save_usage,
    usage_path,
)
from micropolis_world.knowledge_eval import runner
from micropolis_world.usage import CallUsage

PHASH = "abcd1234"
MODEL = "anthropic/claude-sonnet-4-5"
USAGE = CallUsage(
    model_id=MODEL,
    response_model="claude-sonnet-4-5",
    input_tokens=1200,
    output_tokens=300,
    total_tokens=1500,
    cost_usd=0.0081,
    latency_ms=942.5,
)


def test_continuous_sidecar_sits_beside_its_response():
    response = response_path("bruce_s42_T240", MODEL, PHASH)
    sidecar = usage_path("bruce_s42_T240", MODEL, PHASH)

    assert sidecar.parent == response.parent
    # Same model slug and prompt hash, so the pair is found or missed together.
    assert sidecar.stem.removeprefix("usage-") == response.stem.removeprefix(
        "response-"
    )
    assert sidecar.suffix == ".json"


def test_knowledge_eval_sidecar_sits_beside_its_response():
    """Its usage_path has to use the same slug function as response_path or
    the sidecar lands next to nothing."""
    response = runner.response_path(MODEL, PHASH)
    sidecar = runner.usage_path(MODEL, PHASH)

    assert sidecar.parent == response.parent
    assert sidecar.stem.removeprefix("usage-") == response.stem.removeprefix(
        "response-"
    )
    assert sidecar.suffix == ".json"


def test_both_caches_name_a_suffixed_slug_the_same_way():
    """One filename rule for both evals, with the ':' escaped rather than kept."""
    slug = "openai/o3:lowef"
    continuous = response_path("bid", slug, PHASH).name
    knowledge = runner.response_path(slug, PHASH).name
    assert continuous == knowledge == f"response-openai_o3+lowef-{PHASH}.txt"


def test_save_then_load_roundtrips(tmp_path):
    path = tmp_path / "usage-test.json"
    save_usage(path, USAGE)
    assert load_usage(path) == USAGE


def test_save_creates_the_cache_directory(tmp_path):
    path = tmp_path / "nested" / "cache" / "usage-test.json"
    save_usage(path, USAGE)
    assert path.exists()


def test_missing_sidecar_is_unknown_not_an_error(tmp_path):
    """The state of every response cached before cost tracking existed."""
    assert load_usage(tmp_path / "nope.json") is None


def test_malformed_sidecar_is_unknown_not_an_error(tmp_path):
    path = tmp_path / "usage-broken.json"
    path.write_text("{not json")
    assert load_usage(path) is None


def test_sidecar_from_a_later_version_still_loads(tmp_path):
    path = tmp_path / "usage-future.json"
    path.write_text(json.dumps({"model_id": MODEL, "field_from_the_future": 1}))

    got = load_usage(path)
    assert got is not None
    assert got.model_id == MODEL


def test_unpriced_call_roundtrips_as_unknown_not_free(tmp_path):
    """cost_usd must survive as None, never collapse to 0.0."""
    path = tmp_path / "usage-unpriced.json"
    save_usage(path, CallUsage(model_id=MODEL, input_tokens=10, cost_usd=None))
    assert load_usage(path).cost_usd is None
