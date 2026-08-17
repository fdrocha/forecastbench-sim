#!/usr/bin/env -S uv run python3
"""Score the single city eval: CRPS tables by metric and by horizon.

Reads data/micropolis/single_city/data.json, written by
scripts/run_single_city_eval.py. Prompts no models and runs no simulations, so
it is cheap to re-run while changing how the numbers are presented.

The config selects which slice of the dataset to score — its models, cities,
disasters, snapshot_turns and horizons — so one gathered dataset can be viewed
many ways. Naming anything the dataset lacks is an error, not a smaller table.

Also writes scatter plots of normalized CRPS against horizon to
data/micropolis/single_city/plots/ — over all runs, and restricted to the runs
with and without disasters; --no-plot skips them.

Usage:
    scripts/analyze_single_city.py
    scripts/analyze_single_city.py subset.json5
    scripts/analyze_single_city.py --per-metric
    scripts/analyze_single_city.py --no-plot

The per-metric horizon tables are one table per metric and so are the bulk of the
output; --per-metric opts into them.
"""

import argparse
import re
import sys
from pathlib import Path

import micropolis_world.module_globals as g
from micropolis_world.config import (
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.plot_labels import place_labels
from micropolis_world.single_city import (
    DATA_PATH,
    PLOTS_PATH,
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


def normalized_by_model_and_metric(
    corpus: list[dict], responses: Responses, model_names: list[str]
) -> dict[tuple[str, str], float]:
    """Mean normalized CRPS per (model, metric).

    A cell is absent where the metric is excluded from normalization, or where
    no forecast for it had a non-zero actual to divide by.
    """
    scores: dict[tuple[str, str], list[float]] = {}
    for row in score_forecasts(corpus, responses, model_names):
        if row["normalized"] is None:
            continue
        scores.setdefault((row["model_id"], row["metric"]), []).append(row["normalized"])
    return {k: sum(v) / len(v) for k, v in scores.items()}


def normalized_metric_set(normalized: dict[tuple[str, str], float]) -> set[str]:
    """The metrics that have at least one normalized score."""
    return {metric for _model, metric in normalized}


def ranks_within_column(values: dict[str, float | None]) -> dict[str, int]:
    """Rank models by score within one column, 1 being the lowest (best).

    Ties share the lower rank, so two models level on a column are not put in an
    arbitrary order. Models whose score is absent or None are left unranked.
    """
    scored = sorted(
        (m for m, v in values.items() if v is not None), key=lambda m: values[m]
    )
    ranks: dict[str, int] = {}
    for i, model_id in enumerate(scored):
        if i and values[scored[i - 1]] == values[model_id]:
            ranks[model_id] = ranks[scored[i - 1]]
        else:
            ranks[model_id] = i + 1
    return ranks


def ranked_cell(value: float | None, rank: int | None, fmt: str, rank_width: int) -> str:
    """A score with its rank in parens, the rank right-aligned to rank_width.

    Cells are right-aligned as whole strings, so a one-digit rank next to a
    two-digit one would shift the score left by a character and break the
    decimal points down the column. Padding inside the parens keeps the numbers
    aligned regardless of how many models the rank has to count.
    """
    if value is None:
        return "n/a"
    return f"{format(value, fmt)} ({rank:>{rank_width}})"


def rank_width_for(ranks: dict[str, int]) -> int:
    """Digits needed for the widest rank in a column."""
    return max((len(str(r)) for r in ranks.values()), default=1)


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


def print_normalized_crps_table(
    corpus: list[dict], responses: Responses, model_names: list[str]
) -> None:
    """Print models x metrics of normalized CRPS, each cell with its rank.

    Dividing by |actual| makes a cell unitless, so unlike the raw table above
    this one compares a model's performance across metrics as well as down a
    column. The parenthesized rank is the model's standing within that metric,
    1 being best, which is what shows whether a model is uniformly strong or
    carried by one metric.
    """
    normalized = normalized_by_model_and_metric(corpus, responses, model_names)
    # Only the metrics that actually normalize get a column: one that never does
    # would be a column of n/a, which the raw table above already covers.
    metrics = [
        m for m in metrics_in_order(corpus) if m in normalized_metric_set(normalized)
    ]
    ranks = {
        m: ranks_within_column({mid: normalized.get((mid, m)) for mid in model_names})
        for m in metrics
    }

    # The mean of the per-metric means, so every metric counts equally. The raw
    # table's "norm" instead averages the underlying questions, which weights a
    # metric by how many of them parsed; the two differ slightly, hence the
    # distinct column name.
    overall = {
        model_id: _mean(
            [normalized[(model_id, m)] for m in metrics if (model_id, m) in normalized]
        )
        for model_id in model_names
    }

    labels = {m: str(g.METRIC_LABELS.get(m, m)) for m in metrics}
    model_col = max([len("Model")] + [len(m.split("/")[-1]) for m in model_names])
    norm_col, norm_width = "mean", 7

    # One rank width for the whole table: every column ranks the same models, so
    # a shared width keeps the columns reading as one grid.
    rank_width = max(rank_width_for(ranks[m]) for m in metrics) if metrics else 1

    def cell(model_id: str, metric: str) -> str:
        return ranked_cell(
            normalized.get((model_id, metric)),
            ranks[metric].get(model_id),
            ".3f",
            rank_width,
        )

    # Wide enough for the score plus its rank suffix, which the label alone may
    # not cover once a two-digit rank is appended.
    widths = {
        m: max([len(labels[m])] + [len(cell(mid, m)) for mid in model_names])
        for m in metrics
    }

    excluded = [
        str(g.METRIC_LABELS.get(m, m))
        for m in metrics_in_order(corpus)
        if m not in metrics
    ]
    print("\nMean normalized CRPS by model and metric (lower is better)")
    print(
        "CRPS/|actual|, so cells compare across metrics as well as down them;"
        " (n) is the model's rank within that metric"
    )
    print(
        "mean = mean of the per-metric cells, weighting each metric equally"
        + (
            f"; omits {', '.join(excluded)}, whose actual is sometimes 0"
            if excluded
            else ""
        )
    )

    ordered = sorted(model_names, key=lambda m: (overall[m] is None, overall[m] or 0.0))
    header = (
        f"{'Model':<{model_col}}  {norm_col:>{norm_width}}  "
        + "  ".join(f"{labels[m]:>{widths[m]}}" for m in metrics)
    )
    print(header)
    print("-" * len(header))

    for model_id in ordered:
        value = overall[model_id]
        row = [
            f"{model_id.split('/')[-1]:<{model_col}}",
            f"{'n/a' if value is None else f'{value:.3f}':>{norm_width}}",
        ]
        row += [f"{cell(model_id, m):>{widths[m]}}" for m in metrics]
        print("  ".join(row))


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
    want different precision. Each cell carries the model's rank within its own
    column in parens, 1 being best, which is what shows a model gaining or
    losing ground as the horizon lengthens. Rows are sorted by the "all" column,
    so the table reads best-first, and a model with nothing to average sorts
    last rather than crashing the compare.
    """
    overall = {
        model_id: _mean([v for m, _, v in scored if m == model_id])
        for model_id in model_names
    }
    # Keyed by column, "all" included, so ranking is uniform across the table.
    columns = {"all": overall} | {
        h: {
            model_id: _mean([v for m, hz, v in scored if m == model_id and hz == h])
            for model_id in model_names
        }
        for h in horizons
    }
    ranks = {key: ranks_within_column(values) for key, values in columns.items()}
    # One rank width for the whole table, so the columns read as one grid.
    rank_width = max(rank_width_for(r) for r in ranks.values())

    def cell(key, model_id: str) -> str:
        return ranked_cell(
            columns[key][model_id], ranks[key].get(model_id), fmt, rank_width
        )

    model_col = max([len("Model")] + [len(m.split("/")[-1]) for m in model_names])
    labels = {"all": "all"} | {h: f"H{h}" for h in horizons}
    # Wide enough for the longest cell in the table, so a metric in the hundreds
    # of thousands doesn't push its columns out of alignment.
    width = max(
        [9] + [len(cell(key, mid)) for key in columns for mid in model_names]
    )
    ordered = sorted(model_names, key=lambda m: (overall[m] is None, overall[m] or 0.0))

    print(f"\n{title}")
    print(subtitle)
    header = f"{'Model':<{model_col}}  " + "  ".join(
        f"{labels[key]:>{width}}" for key in columns
    )
    print(header)
    print("-" * len(header))

    for model_id in ordered:
        row = [f"{model_id.split('/')[-1]:<{model_col}}"]
        row += [f"{cell(key, model_id):>{width}}" for key in columns]
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


def plot_normalized_by_horizon(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    subset: str = "",
    ymax: float | None = None,
    outdir: Path = PLOTS_PATH,
) -> Path:
    """Scatter normalized CRPS against horizon, one series per model.

    The horizon table says the same thing, but reading a trend across a row of
    numbers is work; here the shape is immediate — how steeply accuracy decays
    with distance, and which models depart from the pack. The mean over models is
    drawn as a thick line so it reads as the summary rather than as one more
    model.

    `subset` names the slice of the corpus being drawn, for the title and the
    filename; empty means the whole of it. `ymax` fixes the top of the y-axis, so
    a set of figures over different slices can be read against each other rather
    than each being scaled to its own worst model.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        r
        for r in score_forecasts(corpus, responses, model_names)
        if r["normalized"] is not None
    ]
    horizons = sorted({c["horizon"] for c in corpus})
    by_model = {
        model_id: {
            h: _mean(
                [
                    r["normalized"]
                    for r in rows
                    if r["model_id"] == model_id and r["horizon"] == h
                ]
            )
            for h in horizons
        }
        for model_id in model_names
    }
    # Averaged over the per-model means, so every model counts equally however
    # many of its forecasts parsed.
    mean_by_horizon = {
        h: _mean([v[h] for v in by_model.values() if v[h] is not None])
        for h in horizons
    }

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # One color per model, matching plot_forecasts.py: tab10 reads more clearly
    # but wraps at 10, so step up once there are more models than that.
    n_colors = 10 if len(model_names) <= 10 else 20
    palette = plt.get_cmap(f"tab{n_colors}")

    # Models in the legend best-first, so its order is itself a ranking.
    overall = {
        model_id: _mean([v for v in by_model[model_id].values() if v is not None])
        for model_id in model_names
    }
    ordered = sorted(model_names, key=lambda m: (overall[m] is None, overall[m] or 0.0))

    # Models bunch tightly at the short horizons, so spread each one's points
    # across a slice of the gap between horizons. Without this the leaders
    # overlap into a single blob and the legend colors can't be matched to
    # anything. The offset is a fixed function of the model's index, not random,
    # so a model sits in the same place in every regenerated figure.
    gap = min((b - a for a, b in zip(horizons, horizons[1:])), default=1)
    # Held well inside the gap so a point stays visibly attached to its own tick;
    # with few horizons a wider spread would put a model nearer the next tick
    # than its own.
    spread = gap * 0.35
    offsets = {
        model_id: (i / max(len(model_names) - 1, 1) - 0.5) * spread
        for i, model_id in enumerate(model_names)
    }

    for i, model_id in enumerate(model_names):
        points = [
            (h, by_model[model_id][h])
            for h in horizons
            if by_model[model_id][h] is not None
        ]
        if not points:
            continue
        ax.scatter(
            [h + offsets[model_id] for h, _ in points],
            [v for _, v in points],
            color=palette(i % n_colors),
            s=38,
            alpha=0.85,
            zorder=3,
            label=model_id.split("/")[-1],
        )

    mean_points = [(h, v) for h, v in mean_by_horizon.items() if v is not None]
    if mean_points:
        ax.plot(
            [h for h, _ in mean_points],
            [v for _, v in mean_points],
            color="black",
            lw=3,
            marker="o",
            ms=8,
            zorder=4,
            label="mean over models",
        )

    ax.set_xlabel(
        "Horizon (turns past the snapshot; model points spread within each tick)"
    )
    ax.set_ylabel("Normalized CRPS (CRPS/|actual|, lower is better)")
    ax.set_title(
        f"Normalized CRPS by horizon{f' — {subset}' if subset else ''}\n"
        f"{len(model_names)} models, {len(corpus)} questions"
    )
    ax.set_xticks(horizons)
    ax.grid(alpha=0.3, zorder=0)
    ax.margins(x=0.04)
    ax.set_ylim(bottom=0, top=ymax)

    # The legend is as tall as the model list, so it goes beside the axes rather
    # than over the points. Entries are ordered best-first, so the legend doubles
    # as a ranking, with the mean on top as the series to find first.
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_order = ["mean over models"] + [m.split("/")[-1] for m in ordered]
    legend_labels = [lbl for lbl in dict.fromkeys(legend_order) if lbl in by_label]
    ax.legend(
        [by_label[lbl] for lbl in legend_labels],
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout()

    suffix = f"-{re.sub(r'[^a-z0-9]+', '-', subset.lower()).strip('-')}" if subset else ""
    out = outdir / f"normalized_crps_by_horizon{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def normalized_by_model(
    corpus: list[dict], responses: Responses, model_names: list[str]
) -> dict[str, float]:
    """Mean normalized CRPS per model, over every forecast that normalizes."""
    scores: dict[str, list[float]] = {}
    for row in score_forecasts(corpus, responses, model_names):
        if row["normalized"] is None:
            continue
        scores.setdefault(row["model_id"], []).append(row["normalized"])
    return {m: sum(v) / len(v) for m, v in scores.items()}


def eci_of(model_id: str) -> int | None:
    """ECI score for a provider/name model id, or None if it has none.

    ECI_MAP is keyed on the bare model name, without the provider prefix.
    """
    return g.ECI_MAP.get(model_id.split("/", 1)[1])


def plot_eci_vs_normalized(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    outdir: Path = PLOTS_PATH,
) -> Path | None:
    """Scatter each model's ECI against its mean normalized CRPS.

    Tests whether forecasting this world tracks general capability. Returns None
    when too few models carry an ECI score for a correlation to mean anything.

    Note the sign: nCRPS is lower-is-better, so a *negative* correlation is the
    pro-g one — the more capable models forecast better. That is the opposite of
    the knowledge eval in scripts/analyze_knowledge.py, whose score is
    higher-is-better.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    scores = normalized_by_model(corpus, responses, model_names)
    points = sorted(
        (eci_of(m), scores[m], m.split("/")[-1])
        for m in model_names
        if m in scores and eci_of(m) is not None
    )
    skipped = sorted(
        m.split("/")[-1] for m in model_names if m in scores and eci_of(m) is None
    )
    if len(points) < 4:
        print(
            "\nECI vs normalized CRPS: only "
            f"{len(points)} model(s) have an ECI score; skipping the plot."
        )
        return None

    ecis = [e for e, _, _ in points]
    values = [v for _, v, _ in points]
    rho, p_rho = stats.spearmanr(ecis, values)
    r, p_r = stats.pearsonr(ecis, values)

    def stars(p: float) -> str:
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    # Lower nCRPS is better, so ECI going up while error goes down is the
    # capability-tracking direction.
    direction = "pro-g" if rho < 0 else "anti-g"
    print("\nECI vs mean normalized CRPS (Spearman)")
    print(
        f"  rho={rho:+.3f}  p={p_rho:.4f} {stars(p_rho):<4} ({direction}, "
        f"n={len(points)})"
    )
    print(f"  Pearson r={r:+.3f}  p={p_r:.4f} {stars(p_r)}")
    print(
        "  nCRPS is lower-is-better, so rho<0 means the more capable models\n"
        "  forecast better (pro-g)."
    )
    if skipped:
        print(f"  no ECI score, excluded: {', '.join(skipped)}")

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(ecis, values, s=70, color="#3266a8", zorder=3)

    fit = stats.linregress(ecis, values)
    xs = [min(ecis), max(ecis)]
    ax.plot(
        xs,
        [fit.intercept + fit.slope * x for x in xs],
        color="#c2432d",
        lw=1.5,
        zorder=2,
        label=(
            f"fit: ρ={rho:+.3f} (p={p_rho:.4f}), r={r:+.3f} (p={p_r:.4f})"
        ),
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_xlabel("ECI (Epoch capability index)")
    ax.set_ylabel("Mean normalized CRPS (CRPS/|actual|, lower is better)")
    ax.set_title(
        f"Forecast skill vs. ECI  ({len(points)} models, {len(corpus)} questions)"
    )
    ax.grid(alpha=0.3, zorder=0)
    ax.margins(x=0.12, y=0.1)
    fig.tight_layout()

    # After the axes are final, so the labels are measured and placed against the
    # limits the figure actually ends up with.
    fig.canvas.draw()
    place_labels(fig, ax, [n for _, _, n in points], ecis, values)

    out = outdir / "eci_vs_normalized_crps.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_horizon_figures(
    corpus: list[dict], responses: Responses, model_names: list[str]
) -> list[Path]:
    """The horizon scatter over all runs, then split by whether disasters ran.

    Disasters are the corpus's one deliberate difficulty axis, so the split says
    whether a model's decay with horizon is about forecasting a city at all or
    about coping with the shocks. Drawn as separate figures rather than one
    overlay: with this many models, two series each would be unreadable.
    """
    subsets = [
        ("", lambda c: True),
        ("disasters", lambda c: c["scenario"]["disasters"]),
        ("no disasters", lambda c: not c["scenario"]["disasters"]),
    ]
    # A config naming only one side of the split leaves the other empty; skip it
    # rather than drawing an axis with nothing on it.
    selections = [
        (subset, [c for c in corpus if keep(c)]) for subset, keep in subsets
    ]
    selections = [(subset, sel) for subset, sel in selections if sel]

    # One y-axis top across the set, so the disasters and no-disasters figures
    # can be read against each other instead of each filling its own axis. Taken
    # from the per-(model, horizon) means, which is what the figures plot.
    ymax = 0.0
    for _subset, selected in selections:
        rows = [
            r
            for r in score_forecasts(selected, responses, model_names)
            if r["normalized"] is not None
        ]
        by_cell: dict[tuple[str, int], list[float]] = {}
        for r in rows:
            by_cell.setdefault((r["model_id"], r["horizon"]), []).append(r["normalized"])
        ymax = max([ymax] + [sum(v) / len(v) for v in by_cell.values()])
    ymax *= 1.08  # headroom so the topmost marker isn't clipped by the frame

    return [
        plot_normalized_by_horizon(selected, responses, model_names, subset, ymax)
        for subset, selected in selections
    ]


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
    ap.add_argument(
        "--per-metric",
        action="store_true",
        help="Also print one CRPS by model and horizon table per metric, in that "
        "metric's own units",
    )
    ap.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip writing the normalized CRPS by horizon scatter plot",
    )
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
    print_normalized_crps_table(corpus, responses, models)
    print_normalized_horizon_table(corpus, responses, models)
    if args.per_metric:
        print_per_metric_horizon_tables(corpus, responses, models)
    else:
        print("\nPer-metric horizon tables omitted; pass --per-metric for them.")

    if args.plot:
        eci_plot = plot_eci_vs_normalized(corpus, responses, models)
        print()
        for out in plot_horizon_figures(corpus, responses, models):
            print(f"Wrote {out}")
        if eci_plot is not None:
            print(f"Wrote {eci_plot}")


if __name__ == "__main__":
    main()
