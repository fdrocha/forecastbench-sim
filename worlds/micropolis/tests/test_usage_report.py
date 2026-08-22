"""Unit tests for aggregating past API usage.

Covers the provider split, the sums, the table layout, and how a call that was
never made is distinguished from one whose cost was never recorded. Nothing
here calls a model; the only I/O is sidecars written into tmp_path.
"""

from micropolis_world.usage import CallUsage, save_usage
from micropolis_world.usage_report import (
    Totals,
    by_model,
    by_provider,
    collect_for_batches,
    format_table,
    grand_total,
    provider_of,
)


def usage(model_id="anthropic/claude-haiku-4-5", cost=0.01, **kwargs):
    fields = {
        "input_tokens": 100,
        "output_tokens": 50,
        **kwargs,
    }
    return CallUsage(model_id=model_id, cost_usd=cost, **fields)


# --- provider split ---------------------------------------------------------


def test_provider_is_the_prefix():
    assert provider_of("anthropic/claude-haiku-4-5") == "anthropic"
    assert provider_of("google/gemini-2.5-pro") == "google"


def test_unprefixed_model_is_its_own_bucket():
    """Better a visible odd row than folding it into a provider it may not be."""
    assert provider_of("gpt-4o") == "(unprefixed)"


def test_models_group_under_one_provider():
    got = by_provider([usage("anthropic/a"), usage("anthropic/b"), usage("openai/c")])
    assert set(got) == {"anthropic", "openai"}
    assert got["anthropic"].calls == 2
    assert got["openai"].calls == 1


def test_per_model_keeps_them_apart():
    got = by_model([usage("anthropic/a"), usage("anthropic/b"), usage("anthropic/b")])
    assert got["anthropic/a"].calls == 1
    assert got["anthropic/b"].calls == 2


# --- sums -------------------------------------------------------------------


def test_totals_sum_every_token_column():
    t = Totals()
    t.add(usage(input_tokens=100, output_tokens=50, reasoning_tokens=10))
    t.add(usage(input_tokens=200, output_tokens=25, reasoning_tokens=5))

    assert t.calls == 2
    assert t.input_tokens == 300
    assert t.output_tokens == 75
    assert t.reasoning_tokens == 15
    assert t.cost_usd == 0.02


def test_unreported_counts_are_treated_as_zero():
    """Providers that omit the detail wrappers record None, not 0."""
    t = Totals()
    t.add(CallUsage(model_id="x", input_tokens=1, reasoning_tokens=None, cost_usd=0.5))
    assert t.reasoning_tokens == 0
    assert t.cached_input_tokens == 0


def test_unpriced_calls_are_counted_not_summed_as_free():
    """A None cost must never be added as 0.0 with no trace of it."""
    t = Totals()
    t.add(usage(cost=0.01))
    t.add(usage(cost=None))

    assert t.calls == 2
    assert t.unpriced == 1
    assert t.cost_usd == 0.01


def test_grand_total_covers_every_bucket():
    buckets = by_provider([usage("anthropic/a", 0.01), usage("openai/b", 0.02)])
    total = grand_total(buckets)

    assert total.calls == 2
    assert round(total.cost_usd, 10) == 0.03
    assert total.input_tokens == 200


def test_grand_total_of_nothing_is_zero():
    total = grand_total({})
    assert total.calls == 0
    assert total.cost_usd == 0.0


# --- table ------------------------------------------------------------------


def test_table_has_a_header_a_row_per_bucket_and_a_total():
    buckets = by_provider([usage("anthropic/a", 0.01), usage("openai/b", 0.02)])
    lines = format_table(buckets, "provider").splitlines()

    assert lines[0].split()[0] == "provider"
    assert "cost USD" in lines[0]
    assert lines[-1].startswith("TOTAL")
    # header, rule, two buckets, rule, total
    assert len(lines) == 6


def test_table_lists_the_most_expensive_first():
    buckets = by_provider(
        [usage("cheap/a", 0.01), usage("dear/b", 0.99), usage("mid/c", 0.5)]
    )
    names = [line.split()[0] for line in format_table(buckets, "provider").splitlines()]
    assert names[2:5] == ["dear", "mid", "cheap"]


def test_table_columns_line_up_with_the_header():
    buckets = by_provider([usage("anthropic/a", 0.01), usage("x/averylongname", 0.02)])
    lines = format_table(buckets, "provider").splitlines()
    assert len({len(line) for line in lines}) == 1


# --- matching stored calls to a config --------------------------------------


def make_layout(tmp_path):
    """Response and sidecar path builders over a scratch directory."""
    return (
        lambda bid, model, phash: tmp_path / f"response-{bid}-{model}-{phash}.txt",
        lambda bid, model, phash: tmp_path / f"usage-{bid}-{model}-{phash}.json",
    )


def test_finds_the_sidecar_for_a_stored_call(tmp_path):
    rpath, upath = make_layout(tmp_path)
    rpath("b1", "m1", "h").write_text("answer")
    save_usage(upath("b1", "m1", "h"), usage(cost=0.07))

    got = collect_for_batches({"b1": "h"}, ["m1"], rpath, upath)
    assert len(got.usages) == 1
    assert got.usages[0].cost_usd == 0.07
    assert got.missing == 0
    assert got.unprompted == 0


def test_a_response_with_no_sidecar_is_missing_data(tmp_path):
    """The state of every call made before cost tracking existed."""
    rpath, upath = make_layout(tmp_path)
    rpath("b1", "m1", "h").write_text("answer")

    got = collect_for_batches({"b1": "h"}, ["m1"], rpath, upath)
    assert got.usages == []
    assert got.missing == 1
    assert got.unprompted == 0


def test_no_response_at_all_means_never_prompted(tmp_path):
    """Not a gap in the accounting — the run just hasn't covered it yet."""
    rpath, upath = make_layout(tmp_path)

    got = collect_for_batches({"b1": "h"}, ["m1"], rpath, upath)
    assert got.missing == 0
    assert got.unprompted == 1


def test_counts_every_batch_model_pair(tmp_path):
    rpath, upath = make_layout(tmp_path)
    # b1 answered by both models, one of them with its usage recorded; b2 by none.
    for model in ("m1", "m2"):
        rpath("b1", model, "h1").write_text("answer")
    save_usage(upath("b1", "m1", "h1"), usage())

    got = collect_for_batches({"b1": "h1", "b2": "h2"}, ["m1", "m2"], rpath, upath)
    assert len(got.usages) == 1
    assert got.missing == 1
    assert got.unprompted == 2


def test_a_different_prompt_hash_finds_nothing(tmp_path):
    """The hash is what keeps one config's calls from being read as another's."""
    rpath, upath = make_layout(tmp_path)
    rpath("b1", "m1", "old").write_text("answer")
    save_usage(upath("b1", "m1", "old"), usage())

    got = collect_for_batches({"b1": "new"}, ["m1"], rpath, upath)
    assert got.usages == []
    assert got.unprompted == 1
