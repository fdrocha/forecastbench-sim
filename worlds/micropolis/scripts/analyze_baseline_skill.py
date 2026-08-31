#!/usr/bin/env -S uv run python3
"""Score the single city eval against a naive baseline instead of against |actual|.

scripts/analyze_single_city.py normalizes CRPS by |actual|, which answers "how
large is the error relative to the thing being forecast". That number has no
zero point: 0.09 is only good or bad relative to how hard the question was. This
script divides by the CRPS of a naive baseline forecast instead, so the scale
carries its own meaning — below 1 beats the baseline, above 1 loses to it.

    score = CRPS_model / CRPS_baseline     per question
            geometric mean over questions

Two baselines are available through --baseline, both of which predict that
nothing changes from the snapshot value:

  plain (default)  all five quantiles at the snapshot value. CRPS of a
                   degenerate forecast reduces to absolute error, so this
                   scores 0 whenever the metric did not move at all — 12% of
                   the corpus — and those questions have no defined score and
                   are dropped.
  sigma            p50 at the snapshot value and the other quantiles at
                   p50 + z*sigma, sigma being the metric's own historical
                   volatility over a window the length of the horizon. The
                   fairer comparison, since it submits a real interval as the
                   models do, and it is almost never exactly right.

Both baselines and their reasoning live in analyze_single_city.py, which this
imports rather than reimplements, so the two scripts cannot disagree about what
the baseline is.

The geometric mean is what makes a ratio averageable: it is symmetric between
twice-as-good and twice-as-bad, and one question where the baseline nearly
nailed the answer cannot swamp a mean the way it would under an arithmetic one.

This is not a rescaling of the other script's numbers. The denominator varies
per question, so the reweighting reorders the models, and city funds is included
here, having been excluded there only because |actual| is sometimes 0. Read the
two as different analyses of the same forecasts.

The read-off horizon is excluded throughout: it is a comprehension check, the
baseline resolves it exactly by construction, and a ratio against a zero
denominator says nothing.

The report is five sections past the preamble: a per-model table of overall
scores and the bar chart that draws it, score against horizon as one small
panel per model, score against ECI and against ForecastBench, and a
model-by-city heatmap of the same scores, which is the only place the report
breaks the cities apart rather than pooling them. All six
metrics are pooled everywhere, city funds included; the funds split lives on
only in analyze_skill_by_config.py, which imports its machinery from here.
Every interval is a 95% t interval on mean log score, clustered on (scenario,
snapshot turn) — questions read off one simulated trajectory are not
independent draws — and that computation is likewise shared with
analyze_skill_by_config.py by import, so a bar there and a bar here cannot
disagree.

Writes the report to data/micropolis/single_city/{label}/analysis-skill.md
(--baseline plain, the default) or analysis-skill-sigma.md (--baseline sigma),
with plots in data/micropolis/single_city/{label}/plots/with_baseline/. Beside
the report it writes scores.csv: model_scores.csv with MPScore/MPScoreLo/
MPScoreHi columns appended, carrying the Model scores table's numbers in a
form the next analysis can load rather than parse out of fixed-width text.
The two baselines' plots and exports are also distinguished by a -sigma
suffix, so running both never overwrites the other's files. Only the paths
written and the report's own path are printed to stdout.

Usage:
    scripts/analyze_baseline_skill.py
    scripts/analyze_baseline_skill.py --baseline sigma
    scripts/analyze_baseline_skill.py my_config.json5 --no-plot
"""

import argparse
import math
import statistics
import sys
from pathlib import Path

from fbsim_core.metrics import compute_crps
from scipy import stats

from micropolis_world import model_scores
from micropolis_world.config import (
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.plot_labels import place_labels
from micropolis_world.single_city import (
    DatasetError,
    MdReport,
    ResponseId,
    Responses,
    data_path,
    label_dir,
    load_dataset,
    plots_path,
    scenario_history,
    select_for_config,
)

# Imported rather than reimplemented so the baseline this scores against is
# provably the same one analyze_single_city.py draws on its figures.
sys.path.insert(0, str(Path(__file__).parent))
from analyze_single_city import (
    NORMAL_Z,
    READ_OFF_HORIZON,
    eci_of,
    historical_sigma,
    is_forecast,
)


def out_dir(label: str) -> Path:
    # Plots go in their own directory so this analysis never overwrites a figure
    # from the |actual|-normalized one, which stays as it was.
    return plots_path(label) / "with_baseline"


def plot_suffix(kind: str) -> str:
    """Filename suffix distinguishing the sigma baseline's plots from plain's.

    "plain" keeps the original, unsuffixed names — it was the only variant
    before --baseline existed — so only "sigma" is tagged; otherwise the two
    variants' figures would silently overwrite each other whenever both are
    plotted against the same label.
    """
    return "-sigma" if kind == "sigma" else ""


# A score of exactly 1 is the baseline's own score, so it is the reference
# every axis and table is read against.
BASELINE_SKILL = 1.0

# City funds is a different kind of series from the other five metrics: an
# unmanaged city's funds follow a near-deterministic arithmetic progression — a
# fixed tax income against fixed expenses — so a model that spots the pattern
# predicts it to a fraction of a percent, reaching scores of 0.002 against a
# baseline interval far too wide for a series that barely wanders. This
# script's own report pools it with the rest regardless; the split machinery
# below stays because analyze_skill_by_config.py still reports the two sides
# separately and imports it from here.
FUNDS_METRIC = "totalFunds"

# The two sides of the city-funds split, as analyze_skill_by_config.py reports
# them. Not used by this script's own report, which pools all six metrics.
SPLITS = {
    "behavioral": (
        "behavioral metrics",
        f"the five metrics other than {FUNDS_METRIC}",
    ),
    "funds": (
        "city funds",
        "reported alone, being a near-deterministic series unlike the other five",
    ),
}

# The two baselines --baseline chooses between, and how each is described
# wherever a table or figure has to say which one it scored against.
BASELINES = {
    "sigma": (
        "persistence + historical spread",
        "p50 at the snapshot value, other quantiles at p50 + z*sigma",
    ),
    "plain": (
        "plain persistence",
        "all five quantiles at the snapshot value",
    ),
}


def baseline_percentiles(
    snapshot: float, sigma: float | None, kind: str
) -> dict[str, float] | None:
    """The baseline's five quantiles for one question, or None if unavailable.

    `sigma` is only consulted for the spread-widened baseline, and None there
    means the history was too short to estimate it — which is a real gap, not a
    zero spread, so the question has no baseline rather than a degenerate one.
    """
    if kind == "plain":
        return dict.fromkeys(NORMAL_Z, snapshot)
    if sigma is None:
        return None
    return {k: snapshot + z * sigma for k, z in NORMAL_Z.items()}


def snapshot_values(corpus: list[dict]) -> dict[tuple[str, int, str], float]:
    """Each (scenario, snapshot, metric) group's value at its snapshot turn.

    Read off the read-off horizon's own question, which resolves at the snapshot
    turn by definition, so no re-simulation is needed.
    """
    return {
        (c["scenario_id"], c["snapshot_turn"], c["metric"]): c["value"]
        for c in corpus
        if c["horizon"] == READ_OFF_HORIZON
    }


def baseline_crps(corpus: list[dict], seed: int, kind: str) -> dict[str, float]:
    """CRPS of the chosen baseline per question id, over the forecast horizons.

    A question missing from the result has no usable baseline, and so no score:
    either its history was too short to estimate a spread, or its run log is
    not cached, or the baseline scored exactly 0 and the ratio would be
    undefined. Callers report the count rather than dropping them silently —
    which of the two baselines is in use changes that count by a lot.
    """
    snapshot = snapshot_values(corpus)
    # One log read per scenario; every question in a scenario shares it. Only the
    # spread-widened baseline needs the history at all, so the plain baseline is
    # not made to depend on a cached run it never reads.
    histories = (
        {
            scenario_id: scenario_history(scenario_id, seed)
            for scenario_id in {c["scenario_id"] for c in corpus}
        }
        if kind == "sigma"
        else {}
    )

    out = {}
    for c in corpus:
        if not is_forecast(c["horizon"]):
            continue
        key = (c["scenario_id"], c["snapshot_turn"], c["metric"])
        value = snapshot.get(key)
        if value is None:
            continue
        sigma = None
        if kind == "sigma":
            history = histories.get(c["scenario_id"])
            if history is None:
                continue
            sigma = historical_sigma(
                history, c["metric"], c["snapshot_turn"], c["horizon"]
            )
        percentiles = baseline_percentiles(value, sigma, kind)
        if percentiles is None:
            continue
        crps = compute_crps(percentiles, c["value"])
        # A baseline that is exactly right leaves no room for a ratio. Dropping
        # beats an epsilon, which would invent a score of ~1e9 and dominate any
        # mean it entered.
        if crps <= 0:
            continue
        out[c["question_id"]] = crps
    return out


def score_skill(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    seed: int,
    kind: str,
) -> tuple[list[dict], dict[str, int]]:
    """One row per scored (model, question), plus a tally of what was dropped.

    A row carries the score ratio and its log, since every aggregate here is a
    geometric mean and taking the log once per row keeps that cheap and keeps
    the two definitions from drifting apart. It also carries the metric and
    horizon, which are what the tables and figures group by, and the scenario
    and snapshot turn it came from, which identify the cluster of questions
    sharing one simulated trajectory.

    The tally counts question-model pairs by why they have no score, and is
    printed alongside the tables: with the plain baseline a large fraction of
    the corpus has no defined ratio, and a table that quietly averaged what was
    left would hide it.
    """
    baselines = baseline_crps(corpus, seed, kind)
    forecasts = [c for c in corpus if is_forecast(c["horizon"])]

    rows = []
    dropped = {"no_baseline": 0, "unparsed": 0, "zero_model_crps": 0}
    for c in forecasts:
        base = baselines.get(c["question_id"])
        for model_id in model_names:
            r = responses.get(ResponseId(model_id, c["question_id"]))
            if r is None or r.percentiles is None:
                dropped["unparsed"] += 1
                continue
            if base is None:
                dropped["no_baseline"] += 1
                continue
            crps = compute_crps(r.percentiles, c["value"])
            # A model that is exactly right has score 0, whose log is -inf and
            # would take any geometric mean it entered to 0. Rare enough to drop
            # and report rather than to floor at an arbitrary epsilon.
            if crps <= 0:
                dropped["zero_model_crps"] += 1
                continue
            rows.append(
                {
                    "model_id": model_id,
                    "metric": c["metric"],
                    "horizon": c["horizon"],
                    # Which trajectory this question was read off. Questions
                    # sharing a (scenario, snapshot) share a simulated history,
                    # so they are not independent draws; carried here so an
                    # interval can cluster on it. See cluster_key().
                    "scenario_id": c["scenario_id"],
                    "snapshot_turn": c["snapshot_turn"],
                    "skill": crps / base,
                    "log_skill": math.log(crps / base),
                }
            )
    return rows, dropped


def geometric_mean_of(values: list[float]) -> float | None:
    """Geometric mean of already-computed ratios, or None if there are none.

    For averaging cells that are themselves means — the figures' mean-over-models
    line — where score_stats' row dicts would have to be fabricated.
    """
    if not values:
        return None
    return math.exp(statistics.fmean(math.log(v) for v in values))


# ---------------------------------------------------------------------------
# Confidence intervals.
#
# analyze_skill_by_config.py imports everything in this block rather than
# redefining it, so a bar drawn there is by construction the same computation
# as a bar drawn here.

# Two-sided 95%. The t quantile is taken at cluster_count-1 degrees of freedom
# rather than a flat z, since the clustering leaves tens of effective
# observations rather than hundreds and z would be optimistic at that size.
CI_LEVEL = 0.95

# Below this many clusters the spread over cluster means is too noisy to be
# worth drawing, and a bar built from a handful of trajectories would imply a
# precision the data cannot support. Such cells get a point and no bar.
MIN_CLUSTERS = 4


def cluster_key(row: dict) -> tuple:
    """What a row's questions are correlated within.

    A (scenario, snapshot turn) pair names one simulated trajectory read at one
    point in time. Every question sharing it — all metrics, all horizons — is
    read off that same history, so they rise and fall together and are not
    independent draws. The scenario id already encodes city and disasters, so
    the pair is the whole grouping.
    """
    return (row["scenario_id"], row["snapshot_turn"])


def city_of(row: dict) -> str:
    """The city a row's question was read off, from its scenario id.

    The id is built as {city}_{disasters flag}_seed{n} by CitySimulation, so the
    city is the leading segment. Both disaster settings of one city fold into
    the same row of the heatmap: the cell is about the map, and splitting it
    would halve every cell's question count for a distinction the figure is not
    asking about.
    """
    return row["scenario_id"].partition("_")[0]


def score_stats(rows: list[dict]) -> tuple[float, float | None, float | None, int]:
    """(geometric mean, CI low, CI high, n questions) of `rows`' scores.

    The point estimate is the geometric mean over every question.

    The interval is clustered on cluster_key(): each trajectory's questions are
    averaged into one value, and the spread is taken over those cluster means.
    Questions within a trajectory are not independent — overlapping horizons on
    a shared history — so a per-question interval understates the uncertainty
    substantially. Built in log space and exponentiated, so it is multiplicative
    and brackets the geometric mean it belongs to.

    Note the point estimate weights questions equally while the interval weights
    clusters equally. With the balanced corpora this was built for the two
    agree; where clusters differ in size the mean stays the pooled one, and only
    the width comes from the clusters.

    Returns bounds of None — a point with no bar — when there are too few
    clusters to estimate a spread worth drawing.

    Read the bars as "would this score hold up on a fresh set of scenarios",
    not as spread over questions: the clusters are what a rerun would resample.
    """
    logs = [r["log_skill"] for r in rows]
    if not logs:
        raise ValueError("score_stats needs at least one row")
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


def stats_by_model(rows: list[dict]) -> dict[str, tuple]:
    """score_stats per model over `rows`, keyed by model id."""
    grouped: dict[str, list[dict]] = {}
    for r in rows:
        grouped.setdefault(r["model_id"], []).append(r)
    return {m: score_stats(v) for m, v in grouped.items()}


def error_arms(cells: list[tuple]) -> tuple[list[float], list[float]]:
    """(below, above) errorbar arm lengths for a list of score_stats cells.

    Computed separately per arm, since the interval is multiplicative and so
    asymmetric around the mean. A cell with too few clusters to bound gets
    zero-length arms — a point with no bar — rather than a fabricated one.
    """
    lower = [m - (lo if lo is not None else m) for m, lo, _hi, *_ in cells]
    upper = [(hi if hi is not None else m) - m for m, _lo, hi, *_ in cells]
    return lower, upper


def split_rows(rows: list[dict], split: str) -> list[dict]:
    """`rows` narrowed to one side of the city-funds split.

    For analyze_skill_by_config.py; this script's own report pools the sides.
    """
    if split == "funds":
        return [r for r in rows if r["metric"] == FUNDS_METRIC]
    return [r for r in rows if r["metric"] != FUNDS_METRIC]


def split_note(split: str) -> str:
    """One line naming which side of the split a table or figure covers."""
    name, how = SPLITS[split]
    return f"{name}: {how}"


def baseline_note(kind: str) -> str:
    """One line naming the baseline in use, for a table subtitle or plot title."""
    name, how = BASELINES[kind]
    return f"baseline: {name} ({how})"


def ci_note() -> str:
    """One line saying what every interval in the report is."""
    return (
        f"intervals are {int(CI_LEVEL * 100)}% t intervals on mean log score, "
        "clustered on (scenario, snapshot turn): questions read off one "
        "simulated trajectory are not independent draws, so the spread is "
        "taken over per-trajectory means"
    )


def print_dropped(report: MdReport, dropped: dict[str, int], total: int) -> None:
    """Report what had no score and why, as a share of the pairs attempted.

    Always reported, including when nothing was dropped, so the absence of a
    warning is informative rather than ambiguous. The plain baseline drops far
    more than the spread-widened one — it is exactly right whenever the metric
    did not move — and that difference is the main thing to know when comparing
    a run under one baseline against a run under the other.
    """
    labels = {
        "no_baseline": "no usable baseline (baseline scored 0, or no spread estimate)",
        "unparsed": "model forecast did not parse",
        "zero_model_crps": "model was exactly right (score 0, no log)",
    }
    n = sum(dropped.values())
    attempted = total + n
    if not attempted:
        return
    lines = [f"{n} of {attempted} (model, question) pairs have no score:"]
    for key, count in dropped.items():
        if count:
            lines.append(f"  {count:>6}  {labels[key]}")
    if not n:
        lines.append("  none")
    report.text("\n".join(lines))


# ---------------------------------------------------------------------------
# The report's sections.


def format_score_cell(cell: tuple | None) -> str:
    """One table cell: the mean with its 95% interval, or a dash when unscored.

    The interval is printed rather than left to the figures: the table is the
    number a reader will quote, and a mean quoted without its width invites
    reading a 0.02 gap between models as a ranking when it is inside the noise.
    """
    if cell is None:
        return "-"
    mean, lo, hi, _n = cell
    if lo is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def ordered_by_score(cells: dict[str, tuple], model_names: list[str]) -> list[str]:
    """`model_names` sorted best (lowest) score first, unscored models last."""
    return sorted(
        model_names,
        key=lambda m: (m not in cells, cells[m][0] if m in cells else 0.0),
    )


def print_model_scores(
    report: MdReport, rows: list[dict], model_names: list[str], kind: str
) -> dict[str, tuple]:
    """The per-model table: model, questions scored, pooled score with its CI.

    One row per model, best score first. Each cell pools every scored question
    — all six metrics, all forecast horizons — so this is the single number the
    rest of the report breaks down by horizon and correlates with the external
    benchmarks.

    Returns the per-model stats, so the bar chart draws exactly these cells.
    """
    cells = stats_by_model(rows)
    ordered = ordered_by_score(cells, model_names)
    counts = {m: cells[m][3] if m in cells else 0 for m in model_names}

    report.heading("Model scores")
    report.text(
        f"{baseline_note(kind)}\n\n"
        "each score is the geometric mean of CRPS_model / CRPS_baseline over "
        "every scored question, all six metrics pooled; below 1 beats the "
        f"baseline. {ci_note()}. Sorted best score first."
    )

    score_header = "score [95% CI]"
    model_col = max([len("Model")] + [len(m.split("/")[-1]) for m in ordered])
    scored_col = max(len("#scored"), max(len(str(c)) for c in counts.values()))
    score_col = max(
        [len(score_header)] + [len(format_score_cell(cells.get(m))) for m in ordered]
    )

    header = (
        f"{'Model':<{model_col}}  {'#scored':>{scored_col}}  "
        f"{score_header:>{score_col}}"
    )
    lines = [header, "-" * len(header)]
    for m in ordered:
        lines.append(
            f"{m.split('/')[-1]:<{model_col}}  {counts[m]:>{scored_col}}  "
            f"{format_score_cell(cells.get(m)):>{score_col}}"
        )
    report.table("\n".join(lines))
    return cells


# Tick ladders for the log score axes, labeled as ratios — 0.5 is "half the
# baseline's error" — rather than left to the log locator, which on ranges this
# narrow labels only the decade boundary and would leave a single tick at 1.
# The fine ladder is for the bar chart and the horizon panels, whose ranges
# span well under a decade (the panels thin it further via max_ticks); the
# coarse one keeps the scatters uncluttered.
SCATTER_TICKS = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4, 6, 8, 12, 16]
FINE_TICKS = [
    0.1, 0.125, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,
    1.0, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 6, 8, 12, 16,
]  # fmt: skip


def set_ratio_ticks(
    ax, candidates: list[float], max_ticks: int | None = None
) -> None:
    """Put ratio-labeled ticks on a log score axis, spanning its current limits.

    `max_ticks` thins a dense ladder down for a small axis by dropping every
    other tick until it fits. A range narrow enough to catch fewer than two
    candidates keeps matplotlib's own ticks instead of being pinned to a
    single label.
    """
    from matplotlib import ticker

    lo, hi = ax.get_ylim()
    ticks = [t for t in candidates if lo <= t <= hi]
    if len(ticks) < 2:
        return
    while max_ticks is not None and len(ticks) > max_ticks:
        ticks = ticks[::2]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}x" if t != 1 else "1x (baseline)" for t in ticks])
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


SCORE_AXIS_LABEL = "Score (CRPS_model / CRPS_baseline, log scale)"

# The heatmap's color span, as a factor either side of the baseline: a cell at
# 2x is the reddest red, one at 0.5x the bluest blue. Fixed rather than fitted
# to the data so one runaway cell cannot flatten the rest of the grid to
# neutral, and so two runs' grids are comparable at a glance.
HEATMAP_SPAN = 2.0

# The blank strip, in cell widths, between the heatmap body and its pooled
# margins. Half a cell is enough to break the grid — the margins are aggregates
# over a whole row or column, not one more model or city — without pushing them
# so far out that a reader loses the alignment.
MARGIN_GAP = 0.5


def plot_model_scores(
    report: MdReport,
    cells: dict[str, tuple],
    ordered: list[str],
    kind: str,
    outdir: Path,
) -> Path | None:
    """Bar chart of the model-scores table, in the same order, with 95% bars.

    The y axis is logarithmic — score is a ratio, and half as good should sit
    as far from 1 as twice as good — so the bars grow from the baseline at 1
    rather than from 0, which is also the honest rendering: 0 is not a
    reachable score, and a bar based there would bury the differences at the
    top. A model that beats the baseline hangs below the line.

    Returns None when no model has a score to draw.
    """
    drawable = [m for m in ordered if m in cells]
    if not drawable:
        report.text("no model has scored rows; skipping the bar chart.")
        return None

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    means = [cells[m][0] for m in drawable]
    lower, upper = error_arms([cells[m] for m in drawable])

    fig, ax = plt.subplots(figsize=(max(8.0, 0.75 * len(drawable) + 2), 6))
    palette = plt.get_cmap("tab20" if len(drawable) > 10 else "tab10")
    xs = range(len(drawable))
    # Spanning baseline -> mean, so a bar's length is how far the model is from
    # breaking even. Given as (bottom, height) rather than bottom=BASELINE_SKILL:
    # bar() draws bottom to bottom+height, which on a log axis would put a mean
    # of 0.845 at 1.845 — pointing the wrong way, and past the baseline it beat.
    ax.bar(
        xs,
        [m - BASELINE_SKILL for m in means],
        bottom=BASELINE_SKILL,
        color=[palette(i % palette.N) for i in xs],
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
    ax.set_xticklabels([m.split("/")[-1] for m in drawable], rotation=30, ha="right")
    ax.set_ylabel(SCORE_AXIS_LABEL)
    ax.set_yscale("log")
    set_ratio_ticks(ax, FINE_TICKS)
    ax.set_title(
        "Model scores — all metrics pooled\n"
        "95% CIs clustered on (scenario, snapshot); "
        f"baseline: {BASELINES[kind][0]}\n"
        "below the dashed line beats the baseline"
    )
    ax.grid(alpha=0.3, axis="y", which="both", zorder=0)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    out = outdir / f"model_scores{plot_suffix(kind)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


def plot_score_by_horizon(
    report: MdReport, rows: list[dict], model_names: list[str], kind: str, outdir: Path
) -> Path | None:
    """One small panel per model: score against horizon, with 95% bars.

    Panels rather than one shared axes: with a dozen models the error bars at a
    shared horizon collapse into a single unreadable stack, and dodging them
    sideways misstates the x value the bar belongs to. Each panel repeats two
    references so it can be read alone — the baseline at 1, and the geometric
    mean over models — so a panel says at a glance whether its model beats the
    field as well as the baseline, and whether either advantage survives the
    longer horizons.

    Panels are sorted like the table, best overall score first, and share the
    x axis. Each panel's y range is fitted to its own model: a shared range
    flattened every line to the spread between the best and worst model, hiding
    the within-model dependency on horizon the panel exists to show. Read
    levels off a panel's own ticks, not across panels.

    Returns None when no model has a scored (model, horizon) cell.
    """
    report.heading("Score vs horizon")

    horizons = sorted({r["horizon"] for r in rows})
    grouped: dict[tuple, list[dict]] = {}
    for r in rows:
        grouped.setdefault((r["model_id"], r["horizon"]), []).append(r)
    cells = {k: score_stats(v) for k, v in grouped.items()}
    overall = stats_by_model(rows)
    ordered = [m for m in ordered_by_score(overall, model_names) if m in overall]
    if not ordered:
        report.text("no model has scored rows; skipping the plot.")
        return None

    # Averaged over the per-model cells, not over the raw questions, so every
    # model counts equally however many of its forecasts had a usable baseline.
    mean_points = [
        (h, v)
        for h in horizons
        for v in [
            geometric_mean_of(
                [cells[(m, h)][0] for m in ordered if (m, h) in cells]
            )
        ]
        if v is not None
    ]

    report.text(
        f"{baseline_note(kind)}\n\n"
        "one panel per model, best overall score first; x is the horizon in "
        "turns past the snapshot, and the thin gray line is the geometric mean "
        "over all models, repeated in every panel as a reference. Each panel's "
        "y range is fitted to its own model so the horizon trend is visible — "
        "compare levels via the ticks, not across panels, and note the mean "
        f"line and the baseline can leave a frame. {ci_note()}."
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ncols = 2
    nrows = math.ceil(len(ordered) / ncols)
    # Each panel is deliberately short — about a fifth of the report's other
    # figures — so the grid reads as a ranked column of sparklines rather than
    # as a stack of full charts.
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(10, 1.3 * nrows + 1.2),
        sharex=True,
        layout="constrained",
    )
    panels = list(axes.flat)

    for ax, model_id in zip(panels, ordered):
        present = [h for h in horizons if (model_id, h) in cells]
        model_cells = [cells[(model_id, h)] for h in present]
        lower, upper = error_arms(model_cells)

        ax.set_yscale("log")
        # The y range is the panel's own, fitted to this model's points and
        # interval arms and padded multiplicatively since the axis is
        # logarithmic. Deliberately not extended to cover the reference lines:
        # forcing the baseline into a panel that sits far from it re-flattens
        # exactly the trend the per-panel range exists to show, so those lines
        # run off some frames instead — the section text says so.
        spans = [c[0] for c in model_cells] + [
            b for _m, lo, hi, _n in model_cells for b in (lo, hi) if b is not None
        ]
        ax.set_ylim(min(spans) / 1.15, max(spans) * 1.15)

        ax.axhline(BASELINE_SKILL, color="crimson", lw=1.2, ls="--", zorder=2)
        if mean_points:
            ax.plot(
                [h for h, _ in mean_points],
                [v for _, v in mean_points],
                color="0.45",
                lw=1.0,
                alpha=0.9,
                zorder=3,
            )
        ax.errorbar(
            present,
            [c[0] for c in model_cells],
            yerr=[lower, upper],
            fmt="o-",
            color="#3266a8",
            ms=3.5,
            lw=1.4,
            elinewidth=1.0,
            capsize=2.5,
            zorder=4,
        )
        ax.set_title(model_id.split("/")[-1], fontsize=8, pad=2)
        ax.grid(alpha=0.25, which="both", zorder=0)
        set_ratio_ticks(ax, FINE_TICKS, max_ticks=4)
        ax.tick_params(labelsize=7)
    for ax in panels[len(ordered) :]:
        ax.set_visible(False)
    # sharex puts x tick labels only on each column's bottom panel; with an odd
    # model count that panel is the hidden spare, which would take the right
    # column's labels with it.
    for col in range(ncols):
        column = [p for p in panels[col::ncols] if p.get_visible()]
        if column:
            column[-1].tick_params(labelbottom=True)

    panels[0].set_xticks(horizons)

    fig.suptitle(
        "Score vs horizon — one panel per model, best overall first\n"
        f"baseline: {BASELINES[kind][0]}; below the dashed line beats it",
        fontsize=11,
    )
    fig.supylabel(SCORE_AXIS_LABEL, fontsize=9)
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#3266a8",
                marker="o",
                ms=3.5,
                lw=1.4,
                label="model score, 95% CI",
            ),
            Line2D([], [], color="0.45", lw=1.0, label="geometric mean over models"),
            Line2D(
                [],
                [],
                color="crimson",
                lw=1.2,
                ls="--",
                label=f"baseline ({BASELINES[kind][0]})",
            ),
        ],
        # The bottom edge is the legend's alone: a supxlabel would share its
        # strip and be printed through, so the x axis is named in the report
        # text instead.
        loc="outside lower center",
        ncol=3,
        fontsize=8,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"score_by_horizon{plot_suffix(kind)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


def plot_score_vs_predictor(
    report: MdReport,
    rows: list[dict],
    model_names: list[str],
    kind: str,
    outdir: Path,
    predictor: str,
) -> Path | None:
    """Scatter an external benchmark against the models' scores, with 95% bars.

    One function for both x variables, differing only in what the x axis is:

      eci            Epoch's capability index. Does forecasting this world
                     track general capability?
      forecastbench  the model's overall ForecastBench score, with its
                     published 95% interval as horizontal bars. The narrower
                     and more pointed question — does it track real-event
                     forecasting ability specifically, which is what would
                     license using a simulated world as a stand-in for the
                     real benchmark.

    Both x variables are higher-is-better and the score is lower-is-better, so
    the agreeing direction is a negative correlation. The fit is in log space,
    matching the axis and the geometric mean the points are computed with — a
    straight fit in ratio space would render as a curve here, and would let one
    model far above the baseline pull the line more than one equally far below
    it pulls back. Pearson is computed on log score for the same reason;
    Spearman only uses ranks and needs no transform. Both go in the legend, on
    the fit line's label.

    Returns None when fewer than four models carry the predictor, where a fit
    would be noise with a decimal point on it.
    """
    heading, x_label, x_of, x_err_of = {
        "eci": (
            "Score vs ECI",
            "ECI (Epoch capability index)",
            eci_of,
            lambda m: None,
        ),
        "forecastbench": (
            "Score vs Forecastbench",
            (
                "ForecastBench overall score "
                "(higher is better; horizontal bars are its published "
                "95% CI)"
            ),
            model_scores.fb_overall_of,
            lambda m: (s := model_scores.scores_of(m)) and s.fb_error,
        ),
    }[predictor]

    report.heading(heading)

    cells = stats_by_model(rows)
    entries = sorted(
        (
            (x_of(m), cells[m], x_err_of(m), m.split("/")[-1])
            for m in model_names
            if m in cells and x_of(m) is not None
        ),
        key=lambda e: (e[0], e[1][0]),
    )
    skipped = sorted(
        m.split("/")[-1] for m in model_names if m in cells and x_of(m) is None
    )
    if len(entries) < 4:
        report.text(
            f"only {len(entries)} model(s) have a score on this benchmark; "
            "skipping the plot."
        )
        return None

    xs_data = [x for x, _, _, _ in entries]
    means = [c[0] for _, c, _, _ in entries]
    rho, p_rho = stats.spearmanr(xs_data, means)
    r, p_r = stats.pearsonr(xs_data, [math.log(v) for v in means])

    lines = [
        baseline_note(kind),
        (
            "score is lower-is-better and the x axis is higher-is-better, "
            "so a negative correlation means the models the benchmark rates "
            "higher beat the baseline by more."
        ),
    ]
    if skipped:
        lines.append(f"no {heading.split(' vs ')[1]} score, excluded: {', '.join(skipped)}")
    report.text("\n".join(lines))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6.5))

    lower, upper = error_arms([c for _, c, _, _ in entries])
    # Drawn point by point rather than vectorized, since a model can carry an x
    # value but no published x interval; those get a plain marker instead of a
    # zero-width bar, which would read as a precise estimate.
    for i, (x, cell, x_err, _name) in enumerate(entries):
        ax.errorbar(
            x,
            cell[0],
            yerr=[[lower[i]], [upper[i]]],
            xerr=None if x_err is None else [[x_err[0]], [x_err[1]]],
            fmt="o",
            ms=7,
            color="#3266a8",
            ecolor="#3266a8",
            elinewidth=1.2,
            capsize=3,
            alpha=0.9,
            zorder=3,
        )

    fit = stats.linregress(xs_data, [math.log(v) for v in means])
    fit_xs = [min(xs_data), max(xs_data)]
    ax.plot(
        fit_xs,
        [math.exp(fit.intercept + fit.slope * x) for x in fit_xs],
        color="#c2432d",
        lw=1.5,
        zorder=2,
        label=(
            f"log-space fit: Spearman ρ={rho:+.3f} (p={p_rho:.4f}), "
            f"Pearson r={r:+.3f} (p={p_r:.4f})"
        ),
    )
    # The parity line is the whole point of this scale: it splits the models
    # that add something over the naive forecast from those that do not.
    ax.axhline(
        BASELINE_SKILL,
        color="crimson",
        lw=2,
        ls="--",
        zorder=2,
        label=f"baseline ({BASELINES[kind][0]})",
    )
    # Below the axes rather than in a corner: with error bars on every point an
    # in-axes legend covers a real interval at whichever corner it is put.
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=1,
        fontsize=9,
        framealpha=0.9,
    )

    ax.set_xlabel(x_label)
    ax.set_ylabel(SCORE_AXIS_LABEL)
    ax.set_yscale("log")
    set_ratio_ticks(ax, SCATTER_TICKS)
    ax.set_title(
        f"{heading} — all metrics pooled\n"
        f"{len(entries)} models; vertical bars are 95% CIs clustered on "
        "(scenario, snapshot)\n"
        f"baseline: {BASELINES[kind][0]}; below the dashed line beats it"
    )
    ax.grid(alpha=0.3, which="both", zorder=0)
    # Wider y-margin than a bare scatter would need: labels are pushed away
    # from their markers to clear the interval bars, so the outermost ones
    # need more room than a marker-sized offset would leave.
    ax.margins(x=0.12, y=0.18)
    fig.tight_layout()

    fig.canvas.draw()
    # The vertical bars are deliberately not passed as obstacles: with a bar on
    # every point the labels get pushed so far from their markers that the
    # association is lost, and a name crossing a thin whisker reads better
    # than one floating four rows away from its point.
    place_labels(
        fig,
        ax,
        [name for _, _, _, name in entries],
        xs_data,
        means,
        xerr=[e for _, _, e, _ in entries],
    )

    out = outdir / f"score_vs_{predictor}{plot_suffix(kind)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


def plot_score_heatmap(
    report: MdReport, rows: list[dict], model_names: list[str], kind: str, outdir: Path
) -> Path | None:
    """Score per (model, city) as a heatmap: models across, cities down.

    The rest of the report pools cities together, so a model that is strong
    everywhere and one that is strong on half the maps and lost on the other
    half arrive at the same overall number. This is the figure that separates
    them, and it also reads the other way down a row: a city every model loses
    on is a property of the map, not of the models.

    Color is diverging around the baseline, in log space, so a cell twice the
    baseline's error is as far from neutral as one half of it — the same
    symmetry the score's geometric mean is built on. The scale is clipped to a
    fixed span rather than fitted to the data: a single extreme cell would
    otherwise wash the whole grid to neutral, and a fixed span also means two
    runs' heatmaps can be laid side by side. Cells past the span keep the end
    color and are still labeled with their own number, so nothing is hidden by
    the clip.

    Columns are ordered by overall score like every other figure here; rows are
    ordered by the city's own score over all models, worst-forecast city first.

    Returns None when no (model, city) cell has a score.
    """
    report.heading("Score by model and city")

    grouped: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        grouped.setdefault((r["model_id"], city_of(r)), []).append(r)
    if not grouped:
        report.text("no scored rows; skipping the heatmap.")
        return None
    cells = {k: score_stats(v) for k, v in grouped.items()}

    overall = stats_by_model(rows)
    models = [m for m in ordered_by_score(overall, model_names) if m in overall]

    by_city: dict[str, list[dict]] = {}
    for r in rows:
        by_city.setdefault(city_of(r), []).append(r)
    city_stats = {c: score_stats(v) for c, v in by_city.items()}
    # Worst first, so the cities that defeat the field are at the top where a
    # reader starts rather than buried at the bottom of a fifteen-row grid.
    cities = sorted(city_stats, key=lambda c: -city_stats[c][0])

    report.text(
        f"{baseline_note(kind)}\n\n"
        "one cell per (model, city): the geometric mean of "
        "CRPS_model / CRPS_baseline over that pair's questions, all six metrics "
        "and every forecast horizon pooled. Both disaster settings of a city "
        "share a row. Blue beats the baseline, red loses to it, and the color "
        f"is symmetric in log space around 1 — clipped at "
        f"{1 / HEATMAP_SPAN:g}x and {HEATMAP_SPAN:g}x, past which the cell keeps "
        "the end color but still prints its own number. Columns run best "
        "overall score first (the Model scores order), rows worst-forecast city "
        "first. The rightmost column and the bottom row are the pooled margins, "
        "not cells — the same numbers the per-model table carries. Intervals "
        "are omitted here for room; a single cell rests on far fewer questions "
        "than the table's rows do, so read the grid for pattern and the table "
        "for level."
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    outdir.mkdir(parents=True, exist_ok=True)

    # The margins get their own trailing column and row, separated from the grid
    # by a gap: they are aggregates over a whole row or column, and butted
    # straight against the cells they summarize they would read as one more
    # model and one more city.
    n_col, n_row = len(models), len(cities)
    grid = [[cells.get((m, c)) for m in models] for c in cities]

    fig, ax = plt.subplots(
        figsize=(max(8.0, 0.62 * n_col + 3.4), max(5.0, 0.42 * n_row + 3.0))
    )
    norm = LogNorm(vmin=1 / HEATMAP_SPAN, vmax=HEATMAP_SPAN)
    # Reversed so the low, baseline-beating end is the cool one: the colormap
    # runs red -> blue as the value rises, and here rising is worse.
    cmap = plt.get_cmap("RdBu_r")

    def draw(col: int, row: int, cell: tuple | None) -> None:
        """One patch plus its number, at grid position (col, row)."""
        if cell is None:
            ax.add_patch(
                plt.Rectangle((col, row), 1, 1, facecolor="0.92", edgecolor="white")
            )
            return
        value = cell[0]
        rgba = cmap(norm(value))
        ax.add_patch(plt.Rectangle((col, row), 1, 1, facecolor=rgba, edgecolor="white"))
        # Ink color follows the patch's own lightness rather than the value:
        # both ends of a diverging map are dark, so a fixed black would vanish
        # at both extremes at once.
        light = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        ax.text(
            col + 0.5,
            row + 0.5,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=7,
            color="white" if light < 0.55 else "0.15",
        )

    for row, city in enumerate(cities):
        for col in range(n_col):
            draw(col, row, grid[row][col])
        draw(n_col + MARGIN_GAP, row, city_stats[city])
    for col, m in enumerate(models):
        draw(col, n_row + MARGIN_GAP, overall[m])
    # The corner is the whole corpus pooled: the number the report opens with.
    draw(n_col + MARGIN_GAP, n_row + MARGIN_GAP, score_stats(rows))

    ax.set_xlim(0, n_col + MARGIN_GAP + 1)
    # Inverted so row 0 — the worst city — is at the top, reading downward.
    ax.set_ylim(n_row + MARGIN_GAP + 1, 0)
    ax.set_xticks([c + 0.5 for c in range(n_col)] + [n_col + MARGIN_GAP + 0.5])
    ax.set_xticklabels(
        [m.split("/")[-1] for m in models] + ["all models"],
        rotation=40,
        ha="right",
        fontsize=8,
    )
    ax.set_yticks([r + 0.5 for r in range(n_row)] + [n_row + MARGIN_GAP + 0.5])
    ax.set_yticklabels(cities + ["all cities"], fontsize=8)
    # The margin ticks are italicized so a reader scanning the axis can tell
    # the pooled row and column from the models and cities beside them.
    for label in (ax.get_xticklabels()[-1], ax.get_yticklabels()[-1]):
        label.set_fontstyle("italic")
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # extendfrac is pinned small: the default sizes each arrow as a fraction of
    # the bar, and on an axes this tall that leaves two arrowheads taking a
    # third of the ladder between them. The arrows only need to say the scale
    # is clipped.
    bar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        extend="both",
        extendfrac=0.02,
        pad=0.02,
    )
    bar.set_ticks([t for t in FINE_TICKS if 1 / HEATMAP_SPAN <= t <= HEATMAP_SPAN])
    bar.set_ticklabels(
        [
            f"{t:g}x" if t != 1 else "1x (baseline)"
            for t in FINE_TICKS
            if 1 / HEATMAP_SPAN <= t <= HEATMAP_SPAN
        ]
    )
    bar.ax.tick_params(labelsize=8)
    bar.set_label(SCORE_AXIS_LABEL, fontsize=9)

    ax.set_title(
        "Score by model and city — all metrics pooled\n"
        f"baseline: {BASELINES[kind][0]}; blue beats it, red loses to it\n"
        "margins are the pooled row and column, not cells",
        fontsize=11,
    )
    fig.tight_layout()

    out = outdir / f"score_heatmap{plot_suffix(kind)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument(
        "--baseline",
        choices=sorted(BASELINES),
        default="plain",
        help=(
            "Which naive forecast to score against. 'plain' (default) puts all "
            "five quantiles on the snapshot value, and so is exactly right — and "
            "gives no usable ratio — whenever the metric did not move; 'sigma' "
            "widens the interval by the metric's historical volatility"
        ),
    )
    ap.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip writing the figures",
    )
    args = ap.parse_args()

    cfg = load_config(args)
    seed = cfg.get_seed(args.seed)
    label = cfg.get_label(args.label)
    data_file = data_path(label)
    outdir = out_dir(label)

    try:
        corpus, responses, models = load_dataset(data_file)
        corpus, responses, models = select_for_config(
            corpus,
            responses,
            models,
            cfg,
            seed,
            cities=args.cities,
            disasters=args.disasters,
            models=args.models,
        )
    except (FileNotFoundError, DatasetError) as e:
        sys.exit(f"[error] {e}")

    if not models:
        sys.exit(
            "[error] no models to score: the config's 'models' list selected nothing"
        )
    forecasts = [c for c in corpus if is_forecast(c["horizon"])]
    if not forecasts:
        sys.exit(
            "[error] no forecast questions: the config's cities, disasters, "
            f"snapshot_turns and horizons selected only H{READ_OFF_HORIZON}, "
            "which this analysis excludes"
        )

    print("=" * 70)
    print("MICROPOLIS WORLD — single city eval, scores vs a naive baseline")
    print("=" * 70)
    print(f"data:   {data_file}")
    print(f"config: {cfg.path}")
    print(f"label:  {label}")
    print(f"plots:  {outdir}")
    print(f"{len(forecasts)} forecast questions x {len(models)} models")
    print(baseline_note(args.baseline))

    report = MdReport()
    report.heading(f"Scores normalized by baseline ({args.baseline})")
    report.text(
        "score = CRPS_model / CRPS_baseline per question, geometric mean over "
        "questions; below 1 beats the baseline. All six metrics are pooled, "
        "city funds included.\n\n"
        f"{baseline_note(args.baseline)}\n\n"
        f"H{READ_OFF_HORIZON} (the read-off) is excluded throughout: the "
        "baseline resolves it exactly by construction.\n\n"
        f"{len(forecasts)} forecast questions x {len(models)} models"
    )

    rows, dropped = score_skill(corpus, responses, models, seed, args.baseline)
    print_dropped(report, dropped, len(rows))
    if not rows:
        sys.exit(
            "[error] no forecast has a usable score; nothing to report. "
            "The counts above say why"
        )

    cells = print_model_scores(report, rows, models, args.baseline)

    # The CSV export of that table: model_scores.csv's columns plus this run's
    # own scores with their 95% CIs, exactly the cells printed above. Written
    # even under --no-plot, being data rather than a figure, and suffixed like
    # the plots so a sigma run never overwrites a plain run's export.
    csv_path = model_scores.write_scores_csv(
        label_dir(label) / f"scores{plot_suffix(args.baseline)}.csv",
        {m: cell[:3] for m, cell in cells.items()},
    )

    written = [csv_path]
    if args.plot:
        figures = [
            plot_model_scores(
                report, cells, ordered_by_score(cells, models), args.baseline, outdir
            ),
            plot_score_by_horizon(report, rows, models, args.baseline, outdir),
            plot_score_vs_predictor(
                report, rows, models, args.baseline, outdir, "eci"
            ),
            plot_score_vs_predictor(
                report, rows, models, args.baseline, outdir, "forecastbench"
            ),
            plot_score_heatmap(report, rows, models, args.baseline, outdir),
        ]
        written += [f for f in figures if f is not None]

    if written:
        print()
        for out in written:
            print(f"Wrote {out}")

    md_name = f"analysis-skill{plot_suffix(args.baseline)}.md"
    out_path = report.write(
        label_dir(label) / md_name,
        f"Single city eval — scores normalized by baseline ({args.baseline})",
    )
    print(out_path)


if __name__ == "__main__":
    main()
