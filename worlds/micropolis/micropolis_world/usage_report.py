"""Aggregate the usage sidecars left behind by past API calls.

Every kept model response has a usage-{model}-{hash}.json beside it recording
what that call cost (see continuous_eval.usage_path and knowledge_eval.runner's
equivalent). This module sums those records into per-provider and per-model
tables. It reads only what is already on disk, so it costs nothing and works
offline.

Backs scripts/analyze_usage.py.
"""

import glob
from dataclasses import dataclass
from pathlib import Path

from . import module_globals as g
from .usage import CallUsage, load_usage

# The token columns every table shows, as (attribute, heading). Kept in one
# place so the per-provider and per-model tables can't drift apart.
TOKEN_COLUMNS = [
    ("input_tokens", "input"),
    ("output_tokens", "output"),
    ("reasoning_tokens", "reasoning"),
    ("cached_input_tokens", "cached"),
    ("cache_write_tokens", "cache wr"),
]


def provider_of(model_id: str) -> str:
    """The provider a model id belongs to, e.g. "anthropic/x" -> "anthropic".

    A bare id with no prefix is reported as "(unprefixed)" rather than guessed
    at, so it shows up as its own row instead of being folded into a provider
    it may not belong to.
    """
    return model_id.split("/")[0] if "/" in model_id else "(unprefixed)"


@dataclass
class Totals:
    """Running sums over a set of calls.

    unpriced is tracked apart from cost_usd because a call litellm has no price
    for records None, and adding it as 0.0 would understate the total with no
    way to tell. The table reports both.
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: float = 0.0
    unpriced: int = 0

    def add(self, usage: CallUsage) -> None:
        """Fold one call's usage in. Unreported counts are treated as zero."""
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.reasoning_tokens += usage.reasoning_tokens or 0
        self.cached_input_tokens += usage.cached_input_tokens or 0
        self.cache_write_tokens += usage.cache_write_tokens or 0
        if usage.cost_usd is None:
            self.unpriced += 1
        else:
            self.cost_usd += usage.cost_usd


def aggregate(usages: list[CallUsage], key) -> dict[str, Totals]:
    """Sum `usages` into buckets, `key` naming the bucket for each call."""
    buckets: dict[str, Totals] = {}
    for usage in usages:
        buckets.setdefault(key(usage), Totals()).add(usage)
    return buckets


def by_provider(usages: list[CallUsage]) -> dict[str, Totals]:
    return aggregate(usages, lambda u: provider_of(u.model_id))


def by_model(usages: list[CallUsage]) -> dict[str, Totals]:
    return aggregate(usages, lambda u: u.model_id)


def grand_total(buckets: dict[str, Totals]) -> Totals:
    """One Totals covering every bucket, for the table's last row."""
    total = Totals()
    for bucket in buckets.values():
        total.calls += bucket.calls
        total.input_tokens += bucket.input_tokens
        total.output_tokens += bucket.output_tokens
        total.reasoning_tokens += bucket.reasoning_tokens
        total.cached_input_tokens += bucket.cached_input_tokens
        total.cache_write_tokens += bucket.cache_write_tokens
        total.cost_usd += bucket.cost_usd
        total.unpriced += bucket.unpriced
    return total


def format_table(buckets: dict[str, Totals], key_heading: str) -> str:
    """A fixed-width table of `buckets`, most expensive first, with a TOTAL row.

    Rows are sorted by cost so the spend that matters is at the top. Costs are
    shown to four decimals: a single cheap call is worth a fraction of a cent,
    and two decimals would print most individual providers as 0.00.
    """
    total = grand_total(buckets)
    rows = sorted(buckets.items(), key=lambda kv: -kv[1].cost_usd)
    labelled = [(name, t) for name, t in rows] + [("TOTAL", total)]

    key_width = max(len(key_heading), *(len(name) for name, _ in labelled))
    calls_width = max(len("calls"), *(len(f"{t.calls:,}") for _, t in labelled))
    token_widths = {
        attr: max(
            len(heading),
            *(len(f"{getattr(t, attr):,}") for _, t in labelled),
        )
        for attr, heading in TOKEN_COLUMNS
    }
    cost_width = max(len("cost USD"), *(len(f"{t.cost_usd:,.4f}") for _, t in labelled))

    header = (
        f"{key_heading:<{key_width}}  {'calls':>{calls_width}}  "
        + "  ".join(
            f"{heading:>{token_widths[attr]}}" for attr, heading in TOKEN_COLUMNS
        )
        + f"  {'cost USD':>{cost_width}}"
    )
    lines = [header, "-" * len(header)]
    for i, (name, t) in enumerate(labelled):
        # Rule above the TOTAL row, so it reads as a sum and not another bucket.
        if i == len(labelled) - 1:
            lines.append("-" * len(header))
        row = [f"{name:<{key_width}}", f"{f'{t.calls:,}':>{calls_width}}"]
        row += [
            f"{f'{getattr(t, attr):,}':>{token_widths[attr]}}"
            for attr, _ in TOKEN_COLUMNS
        ]
        row.append(f"{f'{t.cost_usd:,.4f}':>{cost_width}}")
        lines.append("  ".join(row))
    return "\n".join(lines)


@dataclass
class Collected:
    """The sidecars found for a set of calls, and what could not be accounted for.

    Attributes:
        usages: The usage records that were found.
        missing: Calls that were made — there is a cached response for them —
            but whose usage was never recorded, either because they predate
            cost tracking or because the sidecar was lost.
        unprompted: Batch/model pairs the config asks for that have no cached
            response, so no call was ever made for them. Not a gap in the
            accounting; the run simply hasn't covered them yet.
    """

    usages: list[CallUsage]
    missing: int = 0
    unprompted: int = 0


def collect_for_batches(
    batch_hashes: dict[str, str],
    model_names: list[str],
    response_path_for,
    usage_path_for,
    seen: set[tuple[str, str, str]] | None = None,
) -> Collected:
    """Find the sidecar for every call implied by `batch_hashes` x `model_names`.

    A pair with no cached response was never called, and is counted as
    unprompted rather than missing: only a response with no sidecar beside it
    represents a call whose cost is unaccounted for.

    Args:
        batch_hashes: batch id -> prompt hash, as the run script derives them.
        model_names: The models the config asks for.
        response_path_for: (batch_id, model_id, phash) -> the response's path.
        usage_path_for: (batch_id, model_id, phash) -> the sidecar's path.
        seen: Call keys already accounted for, added to as calls are counted.
            Pass one set across several configs to total them without
            double-counting: configs that differ only in, say, their model list
            share every batch the common models answered, and that shared work
            was paid for once. Counted here rather than by deduplicating the
            loaded usages afterwards, since two genuinely distinct calls can
            have identical token counts and cost.
    """
    collected = Collected(usages=[])
    for batch_id, phash in batch_hashes.items():
        for model_id in model_names:
            key = (batch_id, model_id, phash)
            if seen is not None:
                if key in seen:
                    continue
                seen.add(key)
            if not response_path_for(batch_id, model_id, phash).exists():
                collected.unprompted += 1
                continue
            usage = load_usage(usage_path_for(batch_id, model_id, phash))
            if usage is None:
                collected.missing += 1
            else:
                collected.usages.append(usage)
    return collected


def merge(collections: list[Collected]) -> Collected:
    """One Collected covering all of `collections`.

    The counts simply add up: each was gathered against a shared `seen` set, so
    no call is represented in more than one of them.
    """
    return Collected(
        usages=[u for c in collections for u in c.usages],
        missing=sum(c.missing for c in collections),
        unprompted=sum(c.unprompted for c in collections),
    )


def load_from_glob(pattern: str) -> list[CallUsage]:
    """Every sidecar matching `pattern`, which may be absolute or under data/.

    A relative pattern is resolved against the micropolis data directory, so
    "continuous/cache/*/usage-*.json" reaches the continuous cache without
    the caller having to know where that lives.
    """
    if Path(pattern).is_absolute():
        paths = sorted(Path(p) for p in glob.glob(pattern, recursive=True))
    else:
        paths = sorted(g.DATA_DIR.glob(pattern))

    return [u for u in (load_usage(p) for p in paths) if u is not None]
