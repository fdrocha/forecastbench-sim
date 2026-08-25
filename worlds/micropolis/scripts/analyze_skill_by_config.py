#!/usr/bin/env -S uv run python3
"""Compare skill vs the naive baseline across several configs at once.

analyze_baseline_skill.py scores one config's dataset against a naive baseline:

    skill = CRPS_model / CRPS_baseline     per question
            geometric mean over questions

This script runs that same scoring over a *set* of configs — typically the
prompt variants, each of which names its own dataset label — and reports them
against each other:

  - a summary table, one row per config in command-line order, carrying how
    many forecasts back the row, what share of them failed to parse, its
    average skill on the behavioral metrics, on city funds, and over both,
    each with a 95% CI, and how strongly that skill tracks ECI
  - a scatter of ECI against skill pooled over all six metrics, in the same
    format as the per-split ones below it
  - a bar chart of the summary's last skill column, so the headline
    comparison is visible without reading the table
  - a models x configs table of skill scores, with each model's ECI beside
    its name and the rows sorted by it
  - a scatter of ECI against skill with 95% confidence intervals on the
    geometric means as error bars, one color and one fitted line per config,
    once per side of the city-funds split

The summary's averages give every model one vote — the geometric mean of the
per-model geometric means — rather than pooling questions, so a config is not
favored merely for covering more of some model's questions than another config
does. This makes them deliberately different from the models x configs cells,
which stay pooled per model to keep matching the parent script.

Scoring is imported from analyze_baseline_skill.py rather than reimplemented,
so a cell here is by construction the same number as that script's "all"
column for the same config, baseline and split.

The error bars are t intervals on the mean of log skill, exponentiated (so they
are multiplicative and asymmetric around the mean, as a ratio's interval should
be). They are clustered on (scenario, snapshot turn): the questions read off one
simulated trajectory are averaged into a single value first, and the spread is
taken over those cluster means rather than over the questions. Questions sharing
a trajectory are not independent draws — the horizons overlap and the metrics
move together — so treating them as independent understates the interval by
roughly a factor of two on the datasets this was built for.

The clusters are what a rerun would resample: read the bars as "would this
model's skill hold up on a fresh set of scenarios", not as spread over questions.

As in the parent script, city funds is reported separately from the five
behavioral metrics, the read-off horizon is excluded, and --baseline picks the
naive forecast to score against.

Writes a Markdown report and plots to
data/micropolis/single_city/comparisons/{name}/, where {name} defaults to the
config labels joined with '+'. Only the paths written are printed to stdout.

Usage:
    scripts/analyze_skill_by_config.py cfgA.json5 cfgB.json5 [cfgC.json5 ...]
    scripts/analyze_skill_by_config.py configs/*.json5 --baseline plain
    scripts/analyze_skill_by_config.py cfgA.json5 cfgB.json5 --models openai/gpt-4o
    scripts/analyze_skill_by_config.py cfgA.json5 cfgB.json5 --intersect-models
"""

import argparse
import math
import statistics
import sys
import warnings
from pathlib import Path

from micropolis_world.config import Config, ConfigError, main_with_config
from micropolis_world.plot_labels import place_labels
from micropolis_world.single_city import (
    OUT_DIR,
    DatasetError,
    MdReport,
    ResponseId,
    data_path,
    load_dataset,
    select_for_config,
)
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from analyze_baseline_skill import (
    BASELINE_SKILL,
    BASELINES,
    SPLITS,
    baseline_note,
    geometric_mean_of,
    plot_suffix,
    print_dropped,
    score_skill,
    split_note,
    split_rows,
)
from analyze_single_city import eci_of, is_forecast, stars_for

# Two-sided 95%. The t quantile is taken at cluster_count-1 degrees of freedom
# rather than a flat z, since the clustering leaves tens of effective
# observations rather than hundreds and z would be optimistic at that size.
CI_LEVEL = 0.95

# Below this many clusters the spread over cluster means is too noisy to be
# worth drawing, and a bar built from a handful of trajectories would imply a
# precision the data cannot support. Such cells get a point and no bar.
MIN_CLUSTERS = 4


# The pooled view, for the summary scatter that opens the report. Kept out of
# SPLITS — which drives the per-split sections and must stay the two-way funds
# split the parent script defines — while matching its (name, how) shape so the
# plotting code can take either without knowing which it was handed.
POOLED_SPLIT = (
    "all metrics",
    "the five behavioral metrics and city funds pooled",
)


def split_display(split: str) -> tuple[str, str]:
    """(name, description) for a split, including the pooled pseudo-split."""
    return POOLED_SPLIT if split == "all" else SPLITS[split]


def cluster_key(row: dict) -> tuple:
    """What a row's questions are correlated within.

    A (scenario, snapshot turn) pair names one simulated trajectory read at one
    point in time. Every question sharing it — all metrics, all horizons — is
    read off that same history, so they rise and fall together and are not
    independent draws. The scenario id already encodes city and disasters, so
    the pair is the whole grouping.
    """
    return (row["scenario_id"], row["snapshot_turn"])


def comparison_dir(name: str) -> Path:
    """Where one comparison's report and plots go.

    Under its own directory rather than any single label's: the output belongs
    to the set of configs, not to one of them, and writing it into the first
    config's label dir would make the comparison look like that run's property.
    """
    return OUT_DIR / "comparisons" / name


def skill_stats(rows: list[dict]) -> tuple[float, float | None, float | None, int]:
    """(geometric mean, CI low, CI high, n questions) of `rows`' skill ratios.

    The point estimate is the geometric mean over every question, unchanged and
    still matching analyze_baseline_skill.py's "all" column.

    The interval is clustered on cluster_key(): each trajectory's questions are
    averaged into one value, and the spread is taken over those cluster means.
    Questions within a trajectory are not independent — overlapping horizons on
    a shared history — so a per-question interval understates the uncertainty
    substantially. Built in log space and exponentiated, so it is multiplicative
    and brackets the geometric mean it belongs to.

    Note the point estimate weights questions equally while the interval weights
    clusters equally. With the balanced corpora this was built for the two
    agree; where clusters differ in size the mean stays the pooled one so the
    cell keeps matching the parent script, and only the width comes from the
    clusters.

    Returns bounds of None — a point with no bar — when there are too few
    clusters to estimate a spread worth drawing.
    """
    logs = [r["log_skill"] for r in rows]
    if not logs:
        raise ValueError("skill_stats needs at least one row")
    mean = statistics.fmean(logs)

    grouped: dict[tuple, list[float]] = {}
    for r in rows:
        grouped.setdefault(cluster_key(r), []).append(r["log_skill"])
    cluster_means = [statistics.fmean(v) for v in grouped.values()]

    if len(cluster_means) < MIN_CLUSTERS:
        return math.exp(mean), None, None, len(logs)

    sem = statistics.stdev(cluster_means) / math.sqrt(len(cluster_means))
    crit = stats.t.ppf(1 - (1 - CI_LEVEL) / 2, len(cluster_means) - 1)
    return (
        math.exp(mean),
        math.exp(mean - crit * sem),
        math.exp(mean + crit * sem),
        len(logs),
    )


def response_counts(
    corpus: list[dict], responses: dict, model_names: list[str]
) -> tuple[int, int]:
    """(responses cached, responses that parsed) over one config's selection.

    Counts (model, question) pairs rather than distinct questions: each question
    is put to every model, and it is the pair that a response file corresponds
    to. So a config putting 960 questions to 9 models counts 8640, not 960 —
    the corpus size is the same for every config here and would say nothing
    about them. Both numbers are whole-config totals over the forecast horizons
    — they describe how much data stands behind the row, so they are
    deliberately not narrowed to either side of the city-funds split the skill
    columns take.

    The read-off horizon is excluded, matching score_skill: it is a
    comprehension check that is never scored, so counting it would inflate the
    denominator with questions no skill column could ever draw on.

    A pair with a cached response that could not be parsed is present in the
    dataset with null percentiles, so it counts toward the first number and not
    the second. A pair never gathered at all has no row — but select_for_config
    has already refused to return a selection containing one, so within a scored
    config every pair here is cached and the gap between the two numbers is
    exactly the unparseable responses.
    """
    asked = parsed = 0
    for c in corpus:
        if not is_forecast(c["horizon"]):
            continue
        for model_id in model_names:
            r = responses.get(ResponseId(model_id, c["question_id"]))
            if r is None:
                continue
            asked += 1
            if r.percentiles is not None:
                parsed += 1
    return asked, parsed


def mean_of_model_means(rows: list[dict]) -> tuple[float, float | None, float | None] | None:
    """One config's average skill, giving every model one vote.

    The point estimate is the geometric mean of the per-model geometric means,
    not the pooled mean over rows. A config that happens to cover more of one
    model's questions than another's should not thereby weight that model more
    heavily, and configs here differ in exactly that way — the whole point of
    the table is to compare configs, so the average has to be over a comparable
    quantity rather than over whatever each config's row count happens to be.

    Note this is deliberately *not* the number in the models x configs table
    below, whose cells are per-model pooled means; averaging those cells is what
    this does. It will differ from a pooled-over-everything mean whenever models
    have unequal row counts.

    The interval keeps the same (scenario, snapshot turn) clustering used
    throughout: each trajectory's rows — across all models — are averaged into
    one value, and the spread is taken over those cluster means. So it reads as
    "would this config's average hold up on a fresh set of scenarios", holding
    the model set fixed; it does not attempt to also cover model-sampling
    uncertainty.

    Returns None when there are no rows.
    """
    if not rows:
        return None

    by_model: dict[str, list[float]] = {}
    for r in rows:
        by_model.setdefault(r["model_id"], []).append(r["log_skill"])
    mean = statistics.fmean(statistics.fmean(v) for v in by_model.values())

    grouped: dict[tuple, list[float]] = {}
    for r in rows:
        grouped.setdefault(cluster_key(r), []).append(r["log_skill"])
    cluster_means = [statistics.fmean(v) for v in grouped.values()]

    if len(cluster_means) < MIN_CLUSTERS:
        return math.exp(mean), None, None

    sem = statistics.stdev(cluster_means) / math.sqrt(len(cluster_means))
    crit = stats.t.ppf(1 - (1 - CI_LEVEL) / 2, len(cluster_means) - 1)
    return math.exp(mean), math.exp(mean - crit * sem), math.exp(mean + crit * sem)


def stats_by_model(rows: list[dict]) -> dict[str, tuple]:
    """skill_stats per model over `rows`, keyed by model id."""
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["model_id"], []).append(r)
    return {m: skill_stats(v) for m, v in grouped.items()}


def format_cell(cell: tuple | None) -> str:
    """One table cell: the geometric mean alone, or a dash when unscored.

    The interval behind the mean is shown as an error bar on the scatter
    rather than printed here, where it would triple every column's width.
    """
    if cell is None:
        return "-"
    return f"{cell[0]:.3f}"


def ordered_models(per_config: dict[str, list[dict]]) -> list[str]:
    """The union of models across configs, best pooled skill first.

    Pooled over every config's rows rather than ranked within one of them, so
    the column order does not depend on which config was listed first.
    """
    pooled: dict[str, list[float]] = {}
    for rows in per_config.values():
        for r in rows:
            pooled.setdefault(r["model_id"], []).append(r["log_skill"])
    return sorted(pooled, key=lambda m: statistics.fmean(pooled[m]))


def eci_correlation(rows: list[dict]) -> tuple[float, float, int] | None:
    """(Spearman rho, p, n models) between ECI and per-model skill over `rows`.

    Spearman rather than Pearson, and computed the same way the scatter's
    per-config rho is, so the column and the figure cannot disagree about a
    config: one point per model, at that model's geometric-mean skill, with
    models having no ECI score dropped.

    Skill is lower-is-better, so a negative rho is the pro-g direction — the
    more capable models beat the baseline by more. The sign is deliberately
    left as computed rather than flipped to make "higher is better": the
    scatter beside it is drawn on the same convention, and quietly negating one
    of the two would be worse than asking the reader to hold one fact.

    Unlike the scatter, this pools both sides of the city-funds split, matching
    the 'avg skill all' column it sits beside — the row is whole-config, so a
    correlation computed on one split would describe different data than the
    cells around it.

    Returns None when fewer than four models have an ECI score, which is the
    threshold the scatter already refuses to fit a line below: a rho over three
    points is noise with a decimal point on it.
    """
    by_model: dict[str, list[float]] = {}
    for r in rows:
        by_model.setdefault(r["model_id"], []).append(r["log_skill"])

    points = [
        (eci_of(m), math.exp(statistics.fmean(v)))
        for m, v in by_model.items()
        if eci_of(m) is not None
    ]
    if len(points) < 4:
        return None
    # Constant input on either axis leaves rho undefined. scipy warns and
    # returns nan; the nan is handled below, and the warning is silenced
    # because it would print mid-table as though something had gone wrong,
    # when the cell simply has no correlation to show.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", stats.ConstantInputWarning)
        rho, p = stats.spearmanr(
            [e for e, _ in points], [s for _, s in points]
        )
    # nan would format as "nan" and read as a computed result rather than as
    # "not available".
    if math.isnan(rho):
        return None
    return rho, p, len(points)


def format_avg_cell(cell: tuple | None) -> str:
    """One summary-table skill cell: the mean with its 95% interval.

    Unlike the models x configs table — where an interval per cell would triple
    every column's width and the bars are drawn on the scatter instead — this
    table has one row per config and a handful of columns, so the interval fits
    beside the mean and belongs there: the whole reason to compare configs in
    one table is to see whether they differ by more than their own noise.
    """
    if cell is None:
        return "-"
    mean, lo, hi = cell
    if lo is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def print_summary_table(
    report: MdReport,
    per_config: dict[str, list[dict]],
    counts: dict[str, tuple[int, int]],
    kind: str,
) -> dict[str, tuple[float, float | None, float | None]]:
    """One row per config: how much data it has and what it scored on average.

    The report's other tables are per split and per model, which answers "which
    model, under which config" but makes "is this config better than that one"
    something the reader has to assemble by eye across two sections. This is
    that comparison directly, at the top, at the cost of collapsing the model
    axis entirely.

    Rows stay in the order the configs were given on the command line — the
    order the person running it chose, which usually encodes the comparison
    they have in mind — rather than being re-sorted by score.

    Returns the "all" column's cells, which the bar chart draws.
    """
    behavioral = {
        label: mean_of_model_means(split_rows(rows, "behavioral"))
        for label, rows in per_config.items()
    }
    funds = {
        label: mean_of_model_means(split_rows(rows, "funds"))
        for label, rows in per_config.items()
    }
    # "all" pools the two splits rather than averaging the two columns beside
    # it: the splits hold very different numbers of questions, so averaging the
    # columns would silently promote city funds — one metric — to half the
    # weight of the five behavioral ones.
    overall = {
        label: mean_of_model_means(rows) for label, rows in per_config.items()
    }

    report.heading("Summary by config", level=1)
    report.text(
        f"{baseline_note(kind)}\n\n"
        "one row per config, in the order given on the command line. The skill "
        "columns are the geometric mean of the per-model geometric means — one "
        "vote per model, so a config is not favored merely for covering more "
        "of some model's questions — with 95% CIs clustered on (scenario, "
        "snapshot turn). Below 1 beats the baseline.\n\n"
        "'all' pools the behavioral and funds questions rather than averaging "
        "the two columns beside it, so it is not an average of them: city funds "
        "is one metric against five, and averaging the columns would weight it "
        "as half. See analyze_baseline_skill.py for why funds is reported "
        "apart.\n\n"
        "#forecasts is a whole-config total over the forecast horizons, "
        "excluding the never-scored read-off horizon: it counts one per "
        "(model, question) pair that has a cached response, so a config putting "
        "960 questions to 9 models shows 8640, not 960. It includes forecasts "
        "whose numbers could not be parsed.\n\n"
        "pct invalid is the share of those forecasts that failed to parse — the "
        "rest is what the skill columns are computed from. It is a rate rather "
        "than a count so that configs of different sizes can be compared.\n\n"
        "ECI corr is Spearman rho between a model's ECI and its geometric-mean "
        "skill, one point per model, over the same pooled questions as 'avg "
        "skill all'. Skill is lower-is-better, so rho<0 is the pro-g direction: "
        "the more capable models beat the baseline by more. Stars are p<0.05, "
        "p<0.01, p<0.001; a dash means fewer than 4 of the config's models have "
        "an ECI score. The per-split rho beside each scatter below is the same "
        "statistic computed on that split alone."
    )

    def pct_invalid(label: str) -> str:
        """Share of this config's cached responses that did not parse.

        A rate rather than a count, so it is comparable across configs that put
        different numbers of questions to different numbers of models — which is
        the whole reason the column is here, since an absolute count of failures
        says more about a config's size than about its prompt.
        """
        asked, valid = counts[label]
        if not asked:
            return "-"
        return f"{100 * (asked - valid) / asked:.2f}%"

    correlations = {
        label: eci_correlation(rows) for label, rows in per_config.items()
    }

    def eci_corr(label: str) -> str:
        """rho with its significance stars, or a dash where it is not defined."""
        got = correlations[label]
        if got is None:
            return "-"
        rho, p_rho, _n = got
        return f"{rho:+.3f}{stars_for(p_rho)}"

    cols = [
        # Named "forecasts" rather than "#questions": the number is one per
        # (model, question) pair, not per question, and a config putting 960
        # questions to 9 models shows 8640 here. Calling it questions invited
        # reading it as the corpus size, which is the same for every config and
        # would make the column carry nothing. It also matches what the dataset
        # calls these rows.
        ("#forecasts", lambda l: f"{counts[l][0]}"),
        ("pct invalid", pct_invalid),
        ("avg skill behavioral", lambda l: format_avg_cell(behavioral[l])),
        ("avg skill funds", lambda l: format_avg_cell(funds[l])),
        ("avg skill all", lambda l: format_avg_cell(overall[l])),
        ("ECI corr", eci_corr),
    ]
    label_col = max([len("Config")] + [len(l) for l in per_config])
    widths = [
        max(len(head), *(len(fn(l)) for l in per_config)) for head, fn in cols
    ]

    header = f"{'Config':<{label_col}}  " + "  ".join(
        f"{head:>{w}}" for (head, _), w in zip(cols, widths)
    )
    lines = [header, "-" * len(header)]
    for label in per_config:
        row = [f"{label:<{label_col}}"]
        row += [f"{fn(label):>{w}}" for (_, fn), w in zip(cols, widths)]
        lines.append("  ".join(row))
    report.table("\n".join(lines))
    return overall


def plot_avg_skill_bars(
    report: MdReport,
    overall: dict[str, tuple[float, float | None, float | None]],
    kind: str,
    outdir: Path,
) -> Path | None:
    """Bar chart of each config's average skill, with 95% error bars.

    The summary table's last column, drawn: bars are in command-line order so
    the figure and the table read the same way down the page, and the baseline
    is a reference line so whether a bar clears it is visible without reading
    the axis.

    The y axis is logarithmic, as on the scatter — skill is a ratio, and half
    as good should be as far from 1 as twice as good. That means the bars grow
    from the baseline at 1 rather than from 0, which is also the honest
    rendering: 0 is not a reachable score, and a bar based there would make
    every config look similar by burying the differences at the top.

    Returns None if no config has a mean to draw.
    """
    drawable = {l: c for l, c in overall.items() if c is not None}
    if not drawable:
        report.text("no config has scored rows; skipping the summary bar chart.")
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    labels = list(drawable)
    means = [drawable[l][0] for l in labels]
    # Asymmetric, since the interval is multiplicative; a config with too few
    # clusters to bound gets a bar with no whisker rather than a fake zero one.
    lower = [m - (lo if lo is not None else m) for m, lo, _ in drawable.values()]
    upper = [(hi if hi is not None else m) - m for m, _, hi in drawable.values()]

    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(labels) + 2), 6))
    palette = plt.get_cmap("tab10")
    xs = range(len(labels))
    # Spanning baseline -> mean, so a bar's length is how far the config is
    # from breaking even and a config that beats the baseline hangs below the
    # line. Given as (bottom, height) rather than bottom=BASELINE_SKILL: bar()
    # draws bottom to bottom+height, which on a log axis would put a mean of
    # 0.845 at 1.845 — pointing the wrong way, and past the baseline it beat.
    ax.bar(
        xs,
        [m - BASELINE_SKILL for m in means],
        bottom=BASELINE_SKILL,
        color=[palette(i % 10) for i in xs],
        alpha=0.85,
        zorder=3,
    )
    # The mean is marked as well as barred: the bar's end and the interval's
    # center are the same number, but with the whiskers running well past the
    # bar it is otherwise easy to read the bar top as the estimate's edge
    # rather than as the estimate.
    ax.errorbar(
        list(xs),
        means,
        yerr=[lower, upper],
        fmt="_",
        ms=14,
        mec="black",
        mew=1.6,
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        zorder=4,
    )
    ax.axhline(
        BASELINE_SKILL,
        color="crimson",
        lw=2,
        ls="--",
        zorder=2,
        label=f"baseline ({BASELINES[kind][0]})",
    )

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Avg skill vs baseline (CRPS_model / CRPS_baseline, log scale)")
    ax.set_yscale("log")
    lo_lim, hi_lim = ax.get_ylim()
    # Finer than the scatter's ladder: configs of the same family land within a
    # factor of two of each other, so that axis's steps would leave a plot
    # spanning 0.7-1.1 labeled at one or two ticks.
    candidates = [
        0.1, 0.125, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
        1.0, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 6, 8, 12, 16,
    ]
    ticks = [t for t in candidates if lo_lim <= t <= hi_lim]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}x" if t != 1 else "1x (baseline)" for t in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(
        "Average skill vs baseline by config — all metrics\n"
        "one vote per model; 95% CIs clustered on (scenario, snapshot)\n"
        f"baseline: {BASELINES[kind][0]}; below the dashed line beats it"
    )
    ax.grid(alpha=0.3, axis="y", which="both", zorder=0)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    out = outdir / f"avg_skill_by_config{plot_suffix(kind)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


def print_skill_table(
    report: MdReport, per_config: dict[str, list[dict]], kind: str, split: str
) -> None:
    """Print models x configs of geometric-mean skill, with each model's ECI.

    Models are rows, as in the parent script's tables, and configs are columns:
    how a model's score moves as the config changes reads across one row, and
    the model list — the longer of the two axes — grows the table down rather
    than sideways. Rows are sorted by ECI, most capable first, so the table
    reads as a coarse text rendering of the scatter below it; models without
    an ECI score sink to the bottom, best pooled skill first.
    """
    models = sorted(
        ordered_models(per_config),
        key=lambda m: (eci_of(m) is None, -(eci_of(m) or 0)),
    )
    cells = {
        label: stats_by_model(rows) for label, rows in per_config.items()
    }

    report.heading(
        f"Skill vs baseline by model and config — {SPLITS[split][0]} "
        "(below 1 beats the baseline)"
    )
    report.text(
        f"{split_note(split)}\n\n{baseline_note(kind)}\n\n"
        "each cell is the geometric mean of CRPS_model/CRPS_baseline over the "
        "config's scored questions; rows are sorted by ECI, models without one "
        "last. The scatter's error bars are 95% CIs on these means, clustered "
        "on (scenario, snapshot turn)"
    )

    def eci_cell(model_id: str) -> str:
        eci = eci_of(model_id)
        return "nan" if eci is None else str(eci)

    model_col = max([len("Model")] + [len(m.split("/")[-1]) for m in models])
    eci_col = max([len("ECI")] + [len(eci_cell(m)) for m in models])
    widths = {
        label: max(
            len(label),
            *(len(format_cell(cells[label].get(m))) for m in models),
        )
        for label in per_config
    }

    header = f"{'Model':<{model_col}}  {'ECI':>{eci_col}}  " + "  ".join(
        f"{label:>{widths[label]}}" for label in per_config
    )
    lines = [header, "-" * len(header)]
    for m in models:
        row = [f"{m.split('/')[-1]:<{model_col}}", f"{eci_cell(m):>{eci_col}}"]
        row += [
            f"{format_cell(cells[label].get(m)):>{widths[label]}}"
            for label in per_config
        ]
        lines.append("  ".join(row))
    report.table("\n".join(lines))

    counts = []
    for label, rows in per_config.items():
        ns = [cells[label][m][3] for m in models if m in cells[label]]
        if ns:
            per_model = (
                f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}-{max(ns)}"
            )
            # Both counts, since they answer different questions: the questions
            # say how much was scored, the clusters say how wide the bars are.
            n_clusters = len({cluster_key(r) for r in rows})
            counts.append(
                f"  {label}: {per_model} scored questions per model, "
                f"over {n_clusters} (scenario, snapshot) clusters"
            )
    report.text("\n".join(["questions behind each cell:"] + counts))


def plot_eci_vs_skill_by_config(
    report: MdReport,
    per_config: dict[str, list[dict]],
    kind: str,
    split: str,
    outdir: Path,
) -> Path | None:
    """Scatter ECI against skill with 95% error bars, one series per config.

    Each config gets its points, their intervals, and a fit line in its own
    color, so the figure answers two questions at once: does capability predict
    skill within a config, and does one config sit systematically above another
    at the same capability. The fit is in log space, matching the axis and the
    geometric mean the points are computed with.

    Returns None when no config has enough models with an ECI score for a fit
    to mean anything.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # (eci, mean, lo, hi, model) per config, models without an ECI dropped.
    series: dict[str, list[tuple]] = {}
    skipped: set[str] = set()
    for label, rows in per_config.items():
        points = []
        for model_id, cell in sorted(stats_by_model(rows).items()):
            eci = eci_of(model_id)
            if eci is None:
                skipped.add(model_id.split("/")[-1])
                continue
            mean, lo, hi, _n = cell
            points.append((eci, mean, lo, hi, model_id))
        if len(points) >= 4:
            series[label] = sorted(points)
    if not series:
        report.text(
            f"ECI vs skill ({split_display(split)[0]}): no config has 4+ "
            "models with an ECI score; skipping the plot."
        )
        return None

    report.heading(
        f"ECI vs skill vs baseline by config — {split_display(split)[0]}"
    )
    lines = [
        baseline_note(kind),
        "skill is lower-is-better, so rho<0 means the more capable models beat "
        "the baseline by more (pro-g).",
    ]

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    palette = plt.get_cmap("tab10")

    for i, (label, points) in enumerate(series.items()):
        color = palette(i % 10)
        ecis = [p[0] for p in points]
        means = [p[1] for p in points]
        # Asymmetric multiplicative intervals, so the two arms are computed
        # separately; a cell with too few clusters to bound gets no bar.
        lower = [m - (lo if lo is not None else m) for _, m, lo, _, _ in points]
        upper = [(hi if hi is not None else m) - m for _, m, _, hi, _ in points]
        ax.errorbar(
            ecis,
            means,
            yerr=[lower, upper],
            fmt="o",
            ms=6,
            color=color,
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            alpha=0.85,
            zorder=3,
            label=label,
        )

        rho, p_rho = stats.spearmanr(ecis, means)
        lines.append(
            f"  {label}: rho={rho:+.3f}  p={p_rho:.4f} {stars_for(p_rho):<4} "
            f"(n={len(points)})"
        )
        # Fitted in log space: a straight fit in ratio space would let one
        # model far above the baseline pull the line more than one equally far
        # below it pulls back.
        fit = stats.linregress(ecis, [math.log(m) for m in means])
        xs = [min(ecis), max(ecis)]
        ax.plot(
            xs,
            [math.exp(fit.intercept + fit.slope * x) for x in xs],
            color=color,
            lw=1.5,
            ls="-",
            alpha=0.7,
            zorder=2,
        )

    if skipped:
        lines.append(f"no ECI score, excluded: {', '.join(sorted(skipped))}")
    report.text("\n".join(lines))

    ax.axhline(
        BASELINE_SKILL,
        color="crimson",
        lw=2,
        ls="--",
        zorder=2,
        label=f"baseline ({BASELINES[kind][0]})",
    )

    ax.set_xlabel("ECI (Epoch capability index)")
    ax.set_ylabel("Skill vs baseline (CRPS_model / CRPS_baseline, log scale)")
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    candidates = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4, 6, 8, 12, 16]
    ticks = [t for t in candidates if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}x" if t != 1 else "1x (baseline)" for t in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_title(
        f"Skill vs baseline against ECI, by config — {split_display(split)[0]}\n"
        f"{len(series)} configs; 95% CIs clustered on (scenario, snapshot)\n"
        f"baseline: {BASELINES[kind][0]}; below the dashed line beats it"
    )
    ax.grid(alpha=0.3, which="both", zorder=0)
    ax.margins(x=0.12, y=0.1)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    # One name per model rather than one per point: a model sits at the same
    # ECI in every config, so it is labeled once, at the geometric mean of its
    # per-config scores, and the colors tie the points above and below the
    # label back to their configs.
    by_model: dict[str, list[tuple[int, float]]] = {}
    for points in series.values():
        for eci, mean, _lo, _hi, model_id in points:
            by_model.setdefault(model_id, []).append((eci, mean))
    names = [m.split("/")[-1] for m in by_model]
    xs = [pts[0][0] for pts in by_model.values()]
    ys = [geometric_mean_of([v for _, v in pts]) for pts in by_model.values()]
    fig.canvas.draw()
    place_labels(fig, ax, names, xs, ys)

    out = outdir / f"eci_vs_skill_by_config-{split}{plot_suffix(kind)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


def intersect_models(
    per_config: dict[str, list[dict]],
) -> tuple[dict[str, list[dict]], list[str]]:
    """Restrict every config to the models scored in all of them.

    Without this a column can be an average over a different model set than the
    column beside it, which makes the two look comparable when they are not: a
    config that happens to cover only the strongest models would show better
    skill for that reason alone. Returns the narrowed rows and the models that
    were dropped, so the report can say what it left out rather than quietly
    shrinking.

    Intersects the models that actually produced scored rows, not the models the
    configs name — a model whose forecasts all failed to parse contributes no
    cell either way, and keeping it would leave a dash in the row it is meant to
    make comparable.
    """
    by_config = [
        {r["model_id"] for r in rows} for rows in per_config.values()
    ]
    common = set.intersection(*by_config) if by_config else set()
    dropped = sorted(set.union(*by_config) - common) if by_config else []
    narrowed = {
        label: [r for r in rows if r["model_id"] in common]
        for label, rows in per_config.items()
    }
    return narrowed, dropped


def load_and_score(
    config_path: str, seed_override: int | None, models: list[str] | None, kind: str
) -> tuple[str, list[dict], dict[str, int], dict]:
    """One config's (label, scored rows, dropped tally, selection).

    Loads the dataset the config's label names and narrows it to what the
    config asks for, exactly as the single-config script does, so a config that
    scores there scores identically here.

    The narrowed selection is returned alongside the scores because the summary
    table counts responses that produced no scored row at all — a response that
    failed to parse has no row to be counted from, so the counts have to come
    from the selection rather than from `rows`.
    """
    cfg = Config.load(config_path)
    seed = cfg.get_seed(seed_override)
    label = cfg.get_label(None)
    corpus, responses, model_names = load_dataset(data_path(label))
    corpus, responses, model_names = select_for_config(
        corpus, responses, model_names, cfg, seed, models=models
    )
    rows, dropped = score_skill(corpus, responses, model_names, seed, kind)
    selection = {
        "corpus": corpus,
        "responses": responses,
        "model_names": model_names,
    }
    return label, rows, dropped, selection


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "configs",
        nargs="+",
        help="Two or more JSON5 config files, each naming a gathered dataset",
    )
    ap.add_argument(
        "--baseline",
        choices=sorted(BASELINES),
        default="plain",
        help=(
            "Which naive forecast to score against. 'plain' (default) puts "
            "all five quantiles on the snapshot value; 'sigma' widens the "
            "interval by the metric's historical volatility"
        ),
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed to score at, overriding every config's 'seed'",
    )
    ap.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        default=None,
        help="Model ids to score, overriding every config's 'models' list",
    )
    ap.add_argument(
        "--intersect-models",
        action="store_true",
        help=(
            "Keep only the models that have scored rows in every config, so "
            "the columns are compared over one common set of models rather "
            "than each over whatever it happens to cover"
        ),
    )
    ap.add_argument(
        "--name",
        default=None,
        help="Name of the output directory under "
        f"{OUT_DIR / 'comparisons'} (default: the config labels joined with '+')",
    )
    ap.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip writing the figures",
    )
    args = ap.parse_args()

    per_config: dict[str, list[dict]] = {}
    tallies: dict[str, dict[str, int]] = {}
    selections: dict[str, dict] = {}
    for path in args.configs:
        try:
            label, rows, dropped, selection = load_and_score(
                path, args.seed, args.models, args.baseline
            )
        except (FileNotFoundError, DatasetError, ConfigError) as e:
            sys.exit(f"[error] {path}: {e}")
        if label in per_config:
            sys.exit(
                f"[error] two configs share the label {label!r}; they would "
                "score the same dataset twice"
            )
        per_config[label] = rows
        tallies[label] = dropped
        selections[label] = selection

    dropped_models: list[str] = []
    if args.intersect_models:
        per_config, dropped_models = intersect_models(per_config)
        # The counts describe the same data the skill columns do, so they are
        # narrowed to the common model set as well; leaving them at the config's
        # full coverage would put a #questions in the row that no other cell in
        # it was computed from.
        common = {r["model_id"] for rows in per_config.values() for r in rows}
        for sel in selections.values():
            sel["model_names"] = [
                m for m in sel["model_names"] if m in common
            ]
        if not any(per_config.values()):
            sys.exit(
                "[error] --intersect-models: no model has scored rows in every "
                f"config ({', '.join(per_config)}); nothing left to compare"
            )

    name = args.name or "+".join(per_config)
    outdir = comparison_dir(name)

    print("=" * 70)
    print("MICROPOLIS WORLD — skill vs a naive baseline, across configs")
    print("=" * 70)
    print(f"configs: {', '.join(per_config)}")
    print(f"output:  {outdir}")
    print(baseline_note(args.baseline))
    if dropped_models:
        print(
            "--intersect-models dropped "
            f"{len(dropped_models)} model(s) missing from some config: "
            f"{', '.join(dropped_models)}"
        )

    report = MdReport()
    report.text(
        "score = CRPS_model / CRPS_baseline per question, geometric mean over "
        "questions; below 1 beats the baseline. Scoring is imported from "
        "analyze_baseline_skill.py, so each cell matches that script's 'all' "
        "column for the same config. City funds is reported separately from "
        "the other five metrics — see analyze_baseline_skill.py for why.\n\n"
        f"{baseline_note(args.baseline)}\n\n"
        f"configs compared: {', '.join(per_config)}"
    )
    if args.intersect_models:
        common = sorted(
            {r["model_id"] for rows in per_config.values() for r in rows}
        )
        note = (
            "--intersect-models: every table and figure below is restricted to "
            f"the {len(common)} model(s) scored in all "
            f"{len(per_config)} configs"
        )
        if dropped_models:
            note += f"; excluded for missing at least one config: {', '.join(dropped_models)}"
        report.text(note)
    for label, dropped in tallies.items():
        report.text(f"**{label}**")
        print_dropped(report, dropped, len(per_config[label]))

    counts = {
        label: response_counts(
            sel["corpus"], sel["responses"], sel["model_names"]
        )
        for label, sel in selections.items()
        if label in per_config
    }

    written = []
    # Before the per-split sections: the summary is the comparison the script
    # exists to make, and a reader who wants only "which config did better"
    # should not have to scroll past two full models x configs tables to find
    # it. Configs with no scored rows are omitted rather than shown as dashes.
    scored = {label: rows for label, rows in per_config.items() if rows}
    if scored:
        overall = print_summary_table(report, scored, counts, args.baseline)
        if args.plot:
            # The pooled scatter first: it is the same figure the per-split
            # sections end with, over all six metrics at once, and it answers
            # the report's headline question — does capability predict skill,
            # and does one config sit above another — before the reader has to
            # decide which split to look at.
            fig = plot_eci_vs_skill_by_config(
                report, scored, args.baseline, "all", outdir
            )
            if fig is not None:
                written.append(fig)
            fig = plot_avg_skill_bars(report, overall, args.baseline, outdir)
            if fig is not None:
                written.append(fig)
        empty = [label for label in per_config if not per_config[label]]
        if empty:
            report.text(
                "omitted from the summary, no scored rows: " + ", ".join(empty)
            )

    for split in SPLITS:
        selected = {
            label: split_rows(rows, split) for label, rows in per_config.items()
        }
        selected = {label: rows for label, rows in selected.items() if rows}
        name_, how = SPLITS[split]
        report.heading(f"{name_.upper()} — {how}", level=1)
        if not selected:
            report.text(
                "no scored forecasts on this side of the split; nothing to report"
            )
            continue
        print_skill_table(report, selected, args.baseline, split)
        if args.plot:
            fig = plot_eci_vs_skill_by_config(
                report, selected, args.baseline, split, outdir
            )
            if fig is not None:
                written.append(fig)

    if written:
        print()
        for out in written:
            print(f"Wrote {out}")

    md_name = f"skill-by-config{plot_suffix(args.baseline)}.md"
    out_path = report.write(
        outdir / md_name,
        f"Single city eval — skill vs baseline across configs ({args.baseline})",
    )
    print(out_path)


if __name__ == "__main__":
    main()
