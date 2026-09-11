"""Unit tests for aggregating past API usage.

Covers the provider split, the sums, the table layout, the multi-provider
warning, and how a call that was never made is distinguished from one whose
cost was never recorded. Nothing here calls a model; the only I/O is sidecars
written into tmp_path.
"""

from micropolis_world.usage import CallUsage, save_usage
from micropolis_world.usage_report import (
    Totals,
    by_model,
    by_provider,
    collect_for_batches,
    format_provider_warning,
    format_table,
    grand_total,
    merge,
    percentile,
    provider_of,
    providers_by_model,
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


# --- latency ----------------------------------------------------------------


def test_percentile_of_one_call_is_that_call():
    """statistics.quantiles needs two points; a per-model bucket can have one."""
    assert percentile([4.0], 0.5) == 4.0
    assert percentile([4.0], 0.9) == 4.0


def test_percentile_interpolates_between_neighbours():
    assert percentile([0.0, 10.0], 0.5) == 5.0
    assert percentile([0.0, 100.0, 200.0], 0.9) == 180.0


def test_percentile_of_nothing_is_none():
    assert percentile([], 0.5) is None


def test_totals_collect_every_measured_latency():
    t = Totals()
    t.add(usage(latency_ms=1000.0))
    t.add(usage(latency_ms=3000.0))

    assert t.latency_percentile(0.5) == 2000.0


def test_untimed_calls_are_left_out_rather_than_counted_as_zero():
    """Sidecars predating the field record None; a 0 would drag the median down."""
    t = Totals()
    t.add(usage(latency_ms=None))
    t.add(usage(latency_ms=4000.0))

    assert t.latencies == [4000.0]
    assert t.latency_percentile(0.5) == 4000.0


def test_a_bucket_with_no_timed_call_has_no_percentile():
    t = Totals()
    t.add(usage(latency_ms=None))
    assert t.latency_percentile(0.5) is None


def test_grand_total_pools_the_calls_not_the_percentiles():
    """The total's p50 is over every call, not an average of the buckets'."""
    buckets = by_provider(
        [
            usage("a/x", latency_ms=1000.0),
            usage("a/x", latency_ms=1000.0),
            usage("a/x", latency_ms=1000.0),
            usage("b/y", latency_ms=9000.0),
        ]
    )
    assert grand_total(buckets).latency_percentile(0.5) == 1000.0


def test_table_shows_the_latency_percentiles_in_seconds():
    buckets = by_provider([usage("a/x", latency_ms=1500.0)])
    lines = format_table(buckets, "provider").splitlines()

    assert lines[0].endswith("p50 s  p90 s")
    assert lines[2].split()[-2:] == ["1.5", "1.5"]


def test_table_dashes_a_bucket_that_was_never_timed():
    buckets = by_provider([usage("a/x", latency_ms=None)])
    lines = format_table(buckets, "provider").splitlines()
    assert lines[2].split()[-2:] == ["-", "-"]


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


# --- totalling several configs at once --------------------------------------


def test_a_call_two_configs_share_is_counted_once(tmp_path):
    """Overlapping configs paid for the shared batch once, so it counts once."""
    rpath, upath = make_layout(tmp_path)
    rpath("b1", "m1", "h").write_text("answer")
    save_usage(upath("b1", "m1", "h"), usage(cost=0.07))

    seen: set[tuple[str, str, str]] = set()
    first = collect_for_batches({"b1": "h"}, ["m1"], rpath, upath, seen)
    second = collect_for_batches({"b1": "h"}, ["m1"], rpath, upath, seen)

    assert len(first.usages) == 1
    assert second.usages == []
    assert grand_total(by_provider(merge([first, second]).usages)).cost_usd == 0.07


def test_the_second_config_still_counts_what_it_alone_asks_for(tmp_path):
    rpath, upath = make_layout(tmp_path)
    for batch in ("b1", "b2"):
        rpath(batch, "m1", "h").write_text("answer")
        save_usage(upath(batch, "m1", "h"), usage(cost=0.07))

    seen: set[tuple[str, str, str]] = set()
    first = collect_for_batches({"b1": "h"}, ["m1"], rpath, upath, seen)
    second = collect_for_batches({"b1": "h", "b2": "h"}, ["m1"], rpath, upath, seen)

    assert len(first.usages) == 1
    assert len(second.usages) == 1
    assert len(merge([first, second]).usages) == 2


def test_shared_gaps_are_not_double_counted_either(tmp_path):
    """missing and unprompted describe calls too, so they dedupe the same way."""
    rpath, upath = make_layout(tmp_path)
    rpath("b1", "m1", "h").write_text("answer")  # response, no sidecar

    seen: set[tuple[str, str, str]] = set()
    batches = {"b1": "h", "b2": "h"}
    first = collect_for_batches(batches, ["m1"], rpath, upath, seen)
    second = collect_for_batches(batches, ["m1"], rpath, upath, seen)
    total = merge([first, second])

    assert total.missing == 1
    assert total.unprompted == 1


def test_without_a_seen_set_nothing_is_deduplicated(tmp_path):
    """The single-config path is unchanged: no set passed, no cross-call state."""
    rpath, upath = make_layout(tmp_path)
    rpath("b1", "m1", "h").write_text("answer")
    save_usage(upath("b1", "m1", "h"), usage())

    twice = [collect_for_batches({"b1": "h"}, ["m1"], rpath, upath) for _ in range(2)]
    assert len(merge(twice).usages) == 2


def test_merging_nothing_is_empty(tmp_path):
    total = merge([])
    assert total.usages == []
    assert total.missing == 0
    assert total.unprompted == 0


# --- serving provider -------------------------------------------------------


def test_counts_calls_per_serving_provider():
    got = providers_by_model(
        [
            usage(provider="Google AI Studio"),
            usage(provider="Google AI Studio"),
            usage(provider="Google Vertex"),
        ]
    )
    assert got == {
        "anthropic/claude-haiku-4-5": {"Google AI Studio": 2, "Google Vertex": 1}
    }


def test_no_warning_when_every_model_had_one_provider():
    warning = format_provider_warning(
        [
            usage(model_id="a/one", provider="Alpha"),
            usage(model_id="a/one", provider="Alpha"),
            usage(model_id="b/two", provider="Beta"),
        ]
    )
    assert warning == ""


def test_warns_naming_the_split_model_and_its_providers():
    warning = format_provider_warning(
        [
            usage(model_id="a/one", provider="Alpha"),
            usage(model_id="a/one", provider="Beta"),
            usage(model_id="a/one", provider="Beta"),
            usage(model_id="b/two", provider="Gamma"),
        ]
    )
    assert "1 model(s)" in warning
    assert "a/one" in warning
    # Busiest provider first, with its call count.
    assert "Beta (2), Alpha (1)" in warning
    # The model that was served consistently is not named.
    assert "b/two" not in warning


def test_unrecorded_provider_is_not_a_second_provider():
    """Sidecars predating the field have None; that is not a split."""
    assert providers_by_model([usage(), usage()]) == {}
    assert format_provider_warning([usage(), usage(provider="Alpha")]) == ""


def test_blank_provider_is_treated_as_unrecorded():
    assert format_provider_warning([usage(provider=""), usage(provider="Alpha")]) == ""
