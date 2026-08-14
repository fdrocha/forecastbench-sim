#!/usr/bin/env -S uv run python3
"""Score the single city eval: CRPS tables by metric and by horizon.

Reads data/micropolis/single_city/data.json, written by
scripts/run_single_city_eval.py. Prompts no models and runs no simulations, so
it is cheap to re-run while changing how the numbers are presented.

The config selects which slice of the dataset to score — its models, cities,
disasters, snapshot_turns and horizons — so one gathered dataset can be viewed
many ways. Naming anything the dataset lacks is an error, not a smaller table.

Usage:
    scripts/analyze_single_city.py
    scripts/analyze_single_city.py subset.json5
"""

import argparse
import sys

import micropolis_world.module_globals as g
from micropolis_world.config import (
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.single_city import (
    DATA_PATH,
    UNNORMALIZED_METRICS,
    DatasetError,
    Responses,
    load_dataset,
    score_forecasts,
    select_for_config,
)


def _mean(values: list[float]) -> float | None:
    """Mean of `values`, or None if there are none to average."""
    return sum(values) / len(values) if values else None


def metrics_in_order(corpus: list[dict]) -> list[str]:
    """The corpus's metrics, the ones excluded from normalization last.

    Keeps the metrics the "norm" column averages together on the left, next to
    that column, and pushes city funds — which no summary column covers — to
    the far right, where it reads as the aside it is.
    """
    metrics = list(dict.fromkeys(c["metric"] for c in corpus))
    return sorted(metrics, key=lambda m: m in UNNORMALIZED_METRICS)


def crps_by_model_and_metric(
    corpus: list[dict], responses: Responses, model_names: list[str]
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], int], list[str]]:
    """Mean CRPS per (model, metric), plus how many questions each cell covers.

    Returns (means, counts, metrics). Cells with no parseable forecast are
    absent from both dicts.
    """
    metrics = metrics_in_order(corpus)
    scores: dict[tuple[str, str], list[float]] = {}
    for row in score_forecasts(corpus, responses, model_names):
        scores.setdefault((row["model_id"], row["metric"]), []).append(row["crps"])

    means = {k: sum(v) / len(v) for k, v in scores.items()}
    counts = {k: len(v) for k, v in scores.items()}
    return means, counts, metrics


def print_crps_table(
    corpus: list[dict], responses: Responses, model_names: list[str]
) -> None:
    """Print models x metrics, each cell the mean CRPS over that model's forecasts.

    Raw CRPS is in each metric's own units, so it compares models down a column
    but never across columns. The "norm" column is the mean of CRPS/|actual|
    over the normalizable metrics, which is unitless and so can be averaged
    across them; rows are sorted by it. A "questions" column gives the number of
    parsed forecasts behind each row, out of the whole corpus.
    """
    means, counts, metrics = crps_by_model_and_metric(corpus, responses, model_names)
    rows = score_forecasts(corpus, responses, model_names)

    # Mean normalized CRPS per model, over whichever metrics are normalizable.
    normalized = {
        model_id: _mean(
            [
                r["normalized"]
                for r in rows
                if r["model_id"] == model_id and r["normalized"] is not None
            ]
        )
        for model_id in model_names
    }

    labels = {m: str(g.METRIC_LABELS.get(m, m)) for m in metrics}
    model_col = max([len("Model")] + [len(m.split("/")[-1]) for m in model_names])
    widths = {m: max(len(labels[m]), 12) for m in metrics}

    # How many of the corpus's questions each model's means actually rest on.
    # Shown as used/total so a model scored on fewer questions than the others
    # can't be compared against them without noticing.
    used = {
        model_id: sum(counts.get((model_id, m), 0) for m in metrics)
        for model_id in model_names
    }
    questions_col = "questions"
    questions_width = max(
        len(questions_col), max(len(f"{u}/{len(corpus)}") for u in used.values())
    )
    norm_col = "norm"
    norm_width = max(len(norm_col), 7)

    # Sorted by normalized CRPS, so the table reads best-first. Models with no
    # normalizable forecast at all sort last rather than crashing the compare.
    ordered = sorted(
        model_names, key=lambda m: (normalized[m] is None, normalized[m] or 0.0)
    )

    normalized_metrics = [m for m in metrics if m not in UNNORMALIZED_METRICS]
    print("\nMean CRPS by model and metric (lower is better)")
    print(
        f"norm = mean CRPS/|actual| over {', '.join(labels[m] for m in normalized_metrics)}"
        f" (excludes {', '.join(labels[m] for m in metrics if m in UNNORMALIZED_METRICS)},"
        " whose actual is sometimes 0)"
    )
    header = (
        f"{'Model':<{model_col}}  {questions_col:>{questions_width}}  "
        f"{norm_col:>{norm_width}}  "
        + "  ".join(f"{labels[m]:>{widths[m]}}" for m in metrics)
    )
    print(header)
    print("-" * len(header))

    for model_id in ordered:
        norm = normalized[model_id]
        row = [
            f"{model_id.split('/')[-1]:<{model_col}}",
            f"{f'{used[model_id]}/{len(corpus)}':>{questions_width}}",
            f"{'n/a' if norm is None else f'{norm:.3f}':>{norm_width}}",
        ]
        for m in metrics:
            mean = means.get((model_id, m))
            cell = "n/a" if mean is None else f"{mean:,.1f}"
            row.append(f"{cell:>{widths[m]}}")
        print("  ".join(row))

    # A cell averaging fewer questions than the corpus holds means some
    # responses failed to parse; say so rather than let the means look complete.
    expected = {m: sum(1 for c in corpus if c["metric"] == m) for m in metrics}
    missing = [
        f"{model_id.split('/')[-1]}/{labels[m]}: {expected[m] - counts.get((model_id, m), 0)}"
        for model_id in ordered
        for m in metrics
        if counts.get((model_id, m), 0) < expected[m]
    ]
    if missing:
        print(f"\nUnparseable forecasts excluded — {', '.join(missing)}")


def print_horizon_table(
    scored: list[tuple[str, int, float]],
    model_names: list[str],
    horizons: list[int],
    title: str,
    subtitle: str,
    fmt: str,
) -> None:
    """Print models x horizons from (model, horizon, score) triples.

    `fmt` is the format spec for a cell, since normalized scores and raw CRPS
    want different precision. Rows are sorted by the "all" column, so the table
    reads best-first, and a model with nothing to average sorts last rather
    than crashing the compare.
    """
    cells = {
        (model_id, h): _mean([v for m, hz, v in scored if m == model_id and hz == h])
        for model_id in model_names
        for h in horizons
    }
    overall = {
        model_id: _mean([v for m, _, v in scored if m == model_id])
        for model_id in model_names
    }

    def cell(value: float | None) -> str:
        return "n/a" if value is None else format(value, fmt)

    model_col = max([len("Model")] + [len(m.split("/")[-1]) for m in model_names])
    h_labels = {h: f"H{h}" for h in horizons}
    # Wide enough for the longest number in the table, so a metric in the
    # hundreds of thousands doesn't push its columns out of alignment.
    width = max(
        [9]
        + [len(cell(v)) for v in cells.values()]
        + [len(cell(v)) for v in overall.values()]
    )
    ordered = sorted(model_names, key=lambda m: (overall[m] is None, overall[m] or 0.0))

    print(f"\n{title}")
    print(subtitle)
    header = f"{'Model':<{model_col}}  {'all':>{width}}  " + "  ".join(
        f"{h_labels[h]:>{width}}" for h in horizons
    )
    print(header)
    print("-" * len(header))

    for model_id in ordered:
        row = [f"{model_id.split('/')[-1]:<{model_col}}"]
        for value in [overall[model_id]] + [cells[(model_id, h)] for h in horizons]:
            row.append(f"{cell(value):>{width}}")
        print("  ".join(row))


def print_normalized_horizon_table(
    corpus: list[dict], responses: Responses, model_names: list[str]
) -> None:
    """Print models x horizons, each cell the mean normalized CRPS.

    Normalizing by |actual| divides every horizon by that horizon's own actual,
    not by a per-horizon cohort statistic, so the horizon trend survives: later
    horizons stay harder rather than being flattened to a common scale.
    """
    scored = [
        (r["model_id"], r["horizon"], r["normalized"])
        for r in score_forecasts(corpus, responses, model_names)
        if r["normalized"] is not None
    ]
    normalized_labels = [
        str(g.METRIC_LABELS.get(m, m))
        for m in metrics_in_order(corpus)
        if m not in UNNORMALIZED_METRICS
    ]
    print_horizon_table(
        scored,
        model_names,
        sorted({c["horizon"] for c in corpus}),
        "Mean normalized CRPS by model and horizon (lower is better)",
        f"CRPS/|actual| over {', '.join(normalized_labels)};"
        " horizons are turns past the snapshot",
        ".3f",
    )


def print_per_metric_horizon_tables(
    corpus: list[dict], responses: Responses, model_names: list[str]
) -> None:
    """One models x horizons table of raw CRPS per metric.

    The normalized table above averages metrics together, which hides how each
    one behaves; these keep them apart. Each table is in its own metric's units,
    so it compares models and horizons within itself but never against another
    table — that is what the normalized one is for.
    """
    rows = score_forecasts(corpus, responses, model_names)
    horizons = sorted({c["horizon"] for c in corpus})

    for metric in metrics_in_order(corpus):
        label = str(g.METRIC_LABELS.get(metric, metric))
        scored = [
            (r["model_id"], r["horizon"], r["crps"])
            for r in rows
            if r["metric"] == metric
        ]
        print_horizon_table(
            scored,
            model_names,
            horizons,
            f"Mean CRPS by model and horizon — {label} (lower is better)",
            f"in {label} units, not normalized; horizons are turns past the snapshot",
            ",.1f",
        )


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    args = ap.parse_args()

    cfg = load_config(args)

    # Both failures are user error with an obvious fix — the gathering step
    # hasn't run, or hasn't run for this config — so say so plainly rather than
    # with a traceback.
    try:
        corpus, responses, models = load_dataset()
        corpus, responses, models = select_for_config(
            corpus, responses, models, cfg, cfg.get_seed(args.seed)
        )
    except (FileNotFoundError, DatasetError) as e:
        sys.exit(f"[error] {e}")

    print("=" * 70)
    print("MICROPOLIS WORLD — single city eval scores")
    print("=" * 70)
    print(f"data:   {DATA_PATH}")
    print(f"config: {cfg.path}")
    print(f"{len(corpus)} questions x {len(models)} models")

    print_crps_table(corpus, responses, models)
    print_normalized_horizon_table(corpus, responses, models)
    print_per_metric_horizon_tables(corpus, responses, models)


if __name__ == "__main__":
    main()
