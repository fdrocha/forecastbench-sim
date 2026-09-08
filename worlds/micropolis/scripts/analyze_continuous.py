#!/usr/bin/env -S uv run python3
"""Score the continuous eval: CRPS tables by metric and by horizon.

Reads data/micropolis/continuous/{label}/data.json, written by
scripts/run_eval_continuous.py. Prompts no models and runs no simulations, so
it is cheap to re-run while changing how the numbers are presented.

The config selects which slice of the dataset to score — its models, cities,
disasters, snapshot_turns and horizons — so one gathered dataset can be viewed
many ways. Naming anything the dataset lacks is an error, not a smaller table.

Writes every table and figure to one Markdown report per normalization,
data/micropolis/continuous/{label}/analysis-crps-{norm}.md, rather than to
stdout: normalized CRPS against horizon; forecast skill against ECI; and the
correlation of ECI and the knowledge-eval score against horizon, compared
against each other. --no-plot skips the figures. Only the paths written and
the reports' own paths are printed to stdout.

CRPS is reported normalized — divided by something that makes it unitless —
and there are three things worth dividing by. One run computes all three, each
to its own report and figures, tagged -{norm} in the filename so the sets sit
side by side rather than overwrite each other:

  global    a fixed per-metric scale, the same for every question, so a cell
            is comparable across scenarios, snapshots and horizons.
  local     the mean the question's own metric took over the reseeded
            continuations of its scenario, snapshot and horizon.
  baseline  the expected CRPS of the persistence forecast: the mean of
            |snapshot - outcome| over those same continuations. 1.0 is then
            "as good as assuming nothing changes", which is the one mode whose
            scale carries its own zero point.

The last two read data/micropolis/ground_truth/, so they need
scripts/extract_ground_truth.py to have covered the config — and since every
mode is computed on every run, so does the script as a whole. Both divide by a
number the question itself supplies, which goes to 0 where a metric provably
could not move — a city whose traffic is pinned at 0 across every continuation
— and near 0 where it barely could; both therefore floor the denominator at
--norm-global-frac of the metric's global scale, and each report says how many
questions that floor bound. The modes are not comparable with each other, so
the mode is stated wherever a normalized number is.

Beside the reports it writes continuous_scores.csv: one row per model x metric
x horizon, plus an "all" metric and an "all" horizon per model, with the
forecast counts, the raw CRPS and the normalized CRPS under every mode side by
side — the tables' numbers in a form the next analysis can load rather than
parse out of fixed-width text. One file for all three modes, so unsuffixed.

Usage:
    scripts/analyze_continuous.py                   # configs/continuous.json5
    scripts/analyze_continuous.py subset.json5
    scripts/analyze_continuous.py --norm-global-frac 0.005
    scripts/analyze_continuous.py --no-plot
    scripts/analyze_continuous.py --cities kyoto --disasters false
    scripts/analyze_continuous.py --models openai/gpt-5.6-sol --label myrun
"""

import argparse
import csv
import math
import statistics
import sys
from itertools import pairwise
from pathlib import Path

from fbsim_core.metrics import compute_crps

import micropolis_world.module_globals as g
from micropolis_world import model_scores
from micropolis_world.config import (
    CONFIG_DIR,
    DEFAULT_NORM_GLOBAL_FRAC,
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.continuous_eval import (
    NORM_MODES,
    UNNORMALIZED_METRICS,
    DatasetError,
    MdReport,
    Normalizer,
    ResponseId,
    Responses,
    data_path,
    label_dir,
    load_dataset,
    make_normalizer,
    plots_path,
    scenario_history,
    score_forecasts,
    select_for_config,
)

DEFAULT_CONTINUOUS_CONFIG_PATH = CONFIG_DIR / "continuous.json5"

# The nearest horizon asks for a value the snapshot report already prints, so it
# is a comprehension check — did the model read the report and follow the answer
# format — rather than a forecast. It is kept as its own point in every
# per-horizon table and figure, and excluded from everything that pools horizons
# together, where averaging a read-off in with real forecasts would flatter every
# model by the same trick and let a model that merely copies numbers well outrank
# one that forecasts better.
READ_OFF_HORIZON = 0

# Said wherever an aggregate has had the read-off removed, so no pooled number
# goes out without the exclusion attached to it.
READ_OFF_NOTE = f"excludes H{READ_OFF_HORIZON} (a read-off, not a forecast)"

# 48 city-time units make one game year (knowledge_eval's statements.py states
# the same fact); horizons are reported in years rather than turns.
TURNS_PER_YEAR = 48


def horizon_label(horizon: int) -> str:
    """A horizon in turns past the snapshot, as years (e.g. 144 -> "3y")."""
    return f"{horizon / TURNS_PER_YEAR:g}y"


# A model's identity in a figure: a color and a marker shape. Two dozen models
# is more than any palette separates by hue alone — tab20 pairs a light and a
# dark of each hue, and at scatter size those read as one color — so shape
# carries the difference the eye can't get from color. The two lists are
# coprime in length, so a (color, marker) pair does not repeat until every
# model has one: within a hue the shapes differ, and within a shape the hues do.
MODEL_COLORS = [
    "#4269d0", "#efb118", "#ff725c", "#6cc5b0", "#3ca951", "#ff8ab7",
    "#a463f2", "#97bbf5", "#9c6b4e", "#9498a0", "#e45756", "#72b7b2",
]
MODEL_MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


def model_style(index: int) -> tuple[str, str]:
    """The (color, marker) a model's position in the config's order earns it."""
    return MODEL_COLORS[index % len(MODEL_COLORS)], MODEL_MARKERS[
        index % len(MODEL_MARKERS)
    ]


# Named in one place because the string is both the legend entry and the key the
# legend is reordered by, and the two silently disagreeing would drop the
# baseline out of the legend while leaving it on the axes.
PERSISTENCE_LABEL = "persistence baseline (no change from snapshot)"


def norm_suffix(norm: Normalizer) -> str:
    """Filename suffix keeping one normalization's outputs off another's.

    Every figure and report is per-mode: the three normalizations turn the
    same forecasts into three different sets of numbers, and one run writes
    all three, so the mode has to be in the name for them to coexist. Files
    from before the suffix existed — an unsuffixed analysis-crps.md and its
    plots — are left behind rather than overwritten, and are stale from here
    on.
    """
    return f"-{norm.mode}"


def is_forecast(horizon: int) -> bool:
    """Whether `horizon` asks the model to predict rather than to read off."""
    return horizon != READ_OFF_HORIZON


def forecast_questions(corpus: list[dict]) -> list[dict]:
    """`corpus` without the read-off horizon.

    Used by the aggregates that pool horizons. Narrowing the corpus rather than
    filtering the scored rows keeps the counts, the metric list and the question
    totals reported alongside an aggregate describing the same set of questions
    the aggregate was computed over.
    """
    return [c for c in corpus if is_forecast(c["horizon"])]


def _mean(values: list[float]) -> float | None:
    """Mean of `values`, or None if there are none to average."""
    return sum(values) / len(values) if values else None


def question_key(c: dict) -> tuple[str, int, str, int]:
    """A question's identity for comparing which ones a baseline could score."""
    return (c["scenario_id"], c["snapshot_turn"], c["metric"], c["horizon"])


def persistence_by_horizon(
    corpus: list[dict],
    norm: Normalizer,
    scored: set[tuple[str, int, str, int]] | None = None,
) -> dict[int, float | None]:
    """Mean normalized CRPS of a persistence forecast, per horizon.

    Persistence predicts that nothing changes: whatever the metric reads at the
    snapshot turn is what it will read at the resolution turn. It is the
    reference every forecast should be measured against — a model that cannot
    beat "assume the city stands still" has not demonstrated any understanding
    of the dynamics, however low its absolute score looks.

    Scored on exactly the same footing as the models. CRPS of a point forecast
    (all five quantiles equal) reduces to absolute error, so the pinball loss is
    never actually needed here; the baseline's normalized score for one question
    is |snapshot - actual| over `norm`'s scale for that question, averaged the
    same way and over the same questions as a model's.

    The snapshot value is read from the read-off horizon's own question, which
    resolves at the snapshot turn by definition, so this costs nothing and needs
    no re-simulation. That also means the read-off horizon itself is left out of
    the result: persistence scores 0 there by construction, which is a property
    of the question, not evidence about the baseline. Questions whose group has
    no read-off value are skipped; a horizon with nothing usable comes back None.

    `scored`, if given, is the set of (scenario, snapshot, metric, horizon) keys
    to average over, and anything outside it is dropped. Callers drawing this
    beside persistence_sigma_by_horizon pass that function's keys, so the two
    lines are means over the same questions — the spread estimate is unavailable
    for some of them, and comparing a line over all questions with a line over a
    subset would attribute the difference between two question sets to the
    difference between two forecasts.
    """
    # Keyed on everything but the horizon, so a question can find the snapshot
    # reading of its own metric in its own run.
    snapshot_value = {
        (c["scenario_id"], c["snapshot_turn"], c["metric"]): c["value"]
        for c in corpus
        if c["horizon"] == READ_OFF_HORIZON
    }

    # Forecast horizons only, so the read-off is never a key here and its
    # questions are never scored. Persistence resolves the read-off exactly right
    # by construction, and a reference line dropping to zero there would imply
    # every model is infinitely worse at a horizon where the baseline is not
    # making a forecast either.
    horizons = sorted({c["horizon"] for c in corpus if is_forecast(c["horizon"])})

    scores: dict[int, list[float]] = {h: [] for h in horizons}
    for c in corpus:
        if c["horizon"] not in scores:
            continue
        if scored is not None and question_key(c) not in scored:
            continue
        # Normalized through the same scale as score_forecasts, so the
        # baseline line and the model points are means over an identical
        # question set as well as in the same units.
        scale = norm.scale(c)
        if not scale:
            continue
        snapshot = snapshot_value.get(
            (c["scenario_id"], c["snapshot_turn"], c["metric"])
        )
        if snapshot is None:
            continue
        scores[c["horizon"]].append(abs(snapshot - c["value"]) / scale)

    return {h: _mean(v) for h, v in scores.items()}


# Normal-distribution quantiles, for widening the persistence baseline's
# interval from a single spread estimate. Using z rather than the empirical
# quantiles of the historical changes keeps the estimate stable where only a
# handful of changes are available — the annual metrics have as few as one at the
# longest horizon — at the cost of assuming the changes are roughly symmetric.
NORMAL_Z = {"p10": -1.2816, "p25": -0.6745, "p50": 0.0, "p75": 0.6745, "p90": 1.2816}

# Names the second baseline wherever it is drawn or ranked, for the same reason
# PERSISTENCE_LABEL does.
PERSISTENCE_SIGMA_LABEL = "persistence + historical spread"


def historical_sigma(
    history: list[dict], metric: str, snapshot_turn: int, horizon: int
) -> float | None:
    """Std dev of `metric`'s change over `horizon` turns, before the snapshot.

    The spread a naive forecaster could have known at forecast time: every pair
    of turns `horizon` apart within the history the snapshot report was built
    from, and nothing after it. Returns None where fewer than two changes are
    available, which is where the estimate would be meaningless rather than
    merely noisy.

    The changes overlap — turn 0->48 and turn 1->49 share 47 turns — so the
    effective sample size is nearer len(changes)/horizon than len(changes), and
    this is a rougher estimate at the long horizons than the count suggests. It
    is deliberately still a point estimate: the baseline is meant to be naive.
    """
    # Rows are turn-indexed, and the snapshot turn is itself observable at
    # forecast time, so the history runs 0..snapshot_turn inclusive.
    values = [row[metric] for row in history[: snapshot_turn + 1]]
    changes = [values[t + horizon] - values[t] for t in range(len(values) - horizon)]
    if len(changes) < 2:
        return None
    return statistics.stdev(changes)


def persistence_sigma_by_horizon(
    corpus: list[dict], seed: int, norm: Normalizer
) -> tuple[dict[int, float | None], set[tuple[str, int, str, int]]]:
    """Mean normalized CRPS of persistence widened by historical spread.

    Same median as persistence_by_horizon — the snapshot value, so this is not a
    better central estimate — but the other four quantiles are placed at
    median + z * sigma, with sigma the metric's own historical volatility over a
    window the length of the horizon (see historical_sigma) and z the normal
    quantiles. That makes it a calibrated-interval version of the same forecast,
    which under CRPS is the fairer reference: a degenerate forecast is scored as
    pure absolute error and so is never penalized for its false confidence, while
    a model that hedges honestly is.

    It should therefore score better than plain persistence wherever the metric
    moves at all, raising the bar the models have to clear rather than lowering
    it.

    Needs the run logs for the history, which the dataset does not carry. A
    scenario whose log is not cached is skipped, so this can come back None where
    persistence_by_horizon does not; the caller draws what it has.

    Returns the per-horizon means and the set of question keys they were computed
    over. The keys matter because the spread estimate is not available for every
    question — the annual metrics have too few historical changes at the longest
    horizon from an early snapshot — so a caller drawing this beside plain
    persistence must hold that line to the same set rather than let the two
    differ in both forecast and question set at once.
    """
    snapshot_value = {
        (c["scenario_id"], c["snapshot_turn"], c["metric"]): c["value"]
        for c in corpus
        if c["horizon"] == READ_OFF_HORIZON
    }

    # One log read per scenario rather than per question; each is ~1000 rows and
    # every question in a scenario reads the same one.
    histories = {
        scenario_id: scenario_history(scenario_id, seed)
        for scenario_id in {c["scenario_id"] for c in corpus}
    }

    horizons = sorted({c["horizon"] for c in corpus if is_forecast(c["horizon"])})
    scores: dict[int, list[float]] = {h: [] for h in horizons}
    scored: set[tuple[str, int, str, int]] = set()
    for c in corpus:
        if c["horizon"] not in scores:
            continue
        # Same scale as score_forecasts and persistence_by_horizon, so all
        # three are means over an identical question set.
        scale = norm.scale(c)
        if not scale:
            continue
        key = (c["scenario_id"], c["snapshot_turn"], c["metric"])
        snapshot = snapshot_value.get(key)
        history = histories.get(c["scenario_id"])
        if snapshot is None or history is None:
            continue
        sigma = historical_sigma(history, c["metric"], c["snapshot_turn"], c["horizon"])
        if sigma is None:
            continue
        percentiles = {k: snapshot + z * sigma for k, z in NORMAL_Z.items()}
        scores[c["horizon"]].append(compute_crps(percentiles, c["value"]) / scale)
        scored.add(question_key(c))

    return {h: _mean(v) for h, v in scores.items()}, scored


def metrics_in_order(corpus: list[dict]) -> list[str]:
    """The corpus's metrics, the ones excluded from normalization last.

    Keeps the metrics the "norm" column averages together on the left, next to
    that column, and pushes city funds — which no summary column covers — to
    the far right, where it reads as the aside it is.
    """
    metrics = list(dict.fromkeys(c["metric"] for c in corpus))
    return sorted(metrics, key=lambda m: m in UNNORMALIZED_METRICS)


def crps_by_model_and_metric(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    norm: Normalizer,
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], int], list[str]]:
    """Mean CRPS per (model, metric), plus how many questions each cell covers.

    Returns (means, counts, metrics). Cells with no parseable forecast are
    absent from both dicts.
    """
    metrics = metrics_in_order(corpus)
    scores: dict[tuple[str, str], list[float]] = {}
    for row in score_forecasts(corpus, responses, model_names, norm):
        scores.setdefault((row["model_id"], row["metric"]), []).append(row["crps"])

    means = {k: sum(v) / len(v) for k, v in scores.items()}
    counts = {k: len(v) for k, v in scores.items()}
    return means, counts, metrics


def normalized_by_model_and_metric(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    norm: Normalizer,
) -> dict[tuple[str, str], float]:
    """Mean normalized CRPS per (model, metric).

    A cell is absent where no forecast for that metric had a scale to divide
    by under `norm`.
    """
    scores: dict[tuple[str, str], list[float]] = {}
    for row in score_forecasts(corpus, responses, model_names, norm):
        if row["normalized"] is None:
            continue
        scores.setdefault((row["model_id"], row["metric"]), []).append(
            row["normalized"]
        )
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


def ranked_cell(
    value: float | None, rank: int | None, fmt: str, rank_width: int
) -> str:
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
    report: MdReport,
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    norm: Normalizer,
) -> None:
    """Append models x metrics, each cell the mean CRPS over that model's forecasts.

    Raw CRPS is in each metric's own units, so it compares models down a column
    but never across columns. Rows keep the config's model order rather than
    being ranked: ranking needs one number per model, and the only one that can
    be averaged across these columns is normalized, which the table below
    reports properly. A "questions" column gives the number of parsed forecasts
    behind each row, out of the whole corpus.

    Metric labels drop the "average " that g.METRIC_LABELS carries, which is
    just noise repeated across four of the column heads in a table this wide.

    Every cell pools the horizons, so the read-off horizon is dropped first and
    the counts are out of the remaining questions.
    """
    corpus = forecast_questions(corpus)
    # Both would otherwise surface as a bare max() on an empty sequence while
    # measuring the column widths, several frames from the cause.
    if not model_names:
        raise ValueError(
            "no models to tabulate: the config's 'models' list selected nothing"
        )
    if not corpus:
        raise ValueError(
            "no questions to tabulate: the config's cities, disasters, "
            "snapshot_turns and horizons selected nothing from the dataset"
        )
    means, counts, metrics = crps_by_model_and_metric(
        corpus, responses, model_names, norm
    )

    labels = {
        m: str(g.METRIC_LABELS.get(m, m)).removeprefix("average ") for m in metrics
    }
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

    report.heading("Mean CRPS by model and metric (lower is better)")
    report.text(f"pooled over every forecast horizon; {READ_OFF_NOTE}")
    header = (
        f"{'Model':<{model_col}}  {questions_col:>{questions_width}}  "
        + "  ".join(f"{labels[m]:>{widths[m]}}" for m in metrics)
    )
    lines = [header, "-" * len(header)]

    for model_id in model_names:
        row = [
            f"{model_id.split('/')[-1]:<{model_col}}",
            f"{f'{used[model_id]}/{len(corpus)}':>{questions_width}}",
        ]
        for m in metrics:
            mean = means.get((model_id, m))
            cell = "n/a" if mean is None else f"{mean:,.1f}"
            row.append(f"{cell:>{widths[m]}}")
        lines.append("  ".join(row))
    report.table("\n".join(lines))

    # A cell averaging fewer questions than the corpus holds means some
    # responses failed to parse; say so rather than let the means look complete.
    expected = {m: sum(1 for c in corpus if c["metric"] == m) for m in metrics}
    missing = [
        f"{model_id.split('/')[-1]}/{labels[m]}: {expected[m] - counts.get((model_id, m), 0)}"
        for model_id in model_names
        for m in metrics
        if counts.get((model_id, m), 0) < expected[m]
    ]
    if missing:
        report.text(f"Unparseable forecasts excluded — {', '.join(missing)}")


def print_normalized_crps_table(
    report: MdReport,
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    norm: Normalizer,
) -> None:
    """Print models x metrics of normalized CRPS.

    Dividing by `norm`'s scale makes a cell unitless, so unlike the raw table
    above this one compares a model's performance across metrics as well as down
    a column.

    Pools the horizons into each cell, so the read-off horizon is dropped first.
    """
    corpus = forecast_questions(corpus)
    normalized = normalized_by_model_and_metric(corpus, responses, model_names, norm)
    # Only the metrics that actually normalize get a column: one that never does
    # would be a column of n/a, which the raw table above already covers.
    metrics = [
        m for m in metrics_in_order(corpus) if m in normalized_metric_set(normalized)
    ]

    # Pooled over every normalized forecast rather than averaged over the
    # per-metric cells, so a metric with more parsed forecasts weighs more.
    # The same pooling as the ECI figures and continuous_scores.csv's (all,
    # all) row, so one number per model is quoted the same way everywhere.
    pooled = normalized_by_model(corpus, responses, model_names, norm)
    overall = {model_id: pooled.get(model_id) for model_id in model_names}

    labels = {
        m: str(g.METRIC_LABELS.get(m, m)).removeprefix("average ") for m in metrics
    }
    model_col = max([len("Model")] + [len(m.split("/")[-1]) for m in model_names])
    norm_col, norm_width = "overall", 7

    def cell(model_id: str, metric: str) -> str:
        value = normalized.get((model_id, metric))
        return "n/a" if value is None else format(value, ".3f")

    widths = {
        m: max([len(labels[m])] + [len(cell(mid, m)) for mid in model_names])
        for m in metrics
    }

    excluded = [
        str(g.METRIC_LABELS.get(m, m)).removeprefix("average ")
        for m in metrics_in_order(corpus)
        if m not in metrics
    ]
    report.heading("Mean normalized CRPS by model and metric (lower is better)")
    report.text(
        f"pooled over every forecast horizon; {READ_OFF_NOTE}\n\n"
        f"{norm.ratio} ({norm.mode} normalization), so cells compare across metrics as"
        " well as down them\n\n"
        "overall = mean over every normalized forecast pooled, so a metric with "
        "more parsed forecasts weighs more"
        + (f"; omits {', '.join(excluded)}, which no scale covers" if excluded else "")
    )

    ordered = sorted(model_names, key=lambda m: (overall[m] is None, overall[m] or 0.0))
    header = f"{'Model':<{model_col}}  {norm_col:>{norm_width}}  " + "  ".join(
        f"{labels[m]:>{widths[m]}}" for m in metrics
    )
    lines = [header, "-" * len(header)]

    for model_id in ordered:
        value = overall[model_id]
        row = [
            f"{model_id.split('/')[-1]:<{model_col}}",
            f"{'n/a' if value is None else f'{value:.3f}':>{norm_width}}",
        ]
        row += [f"{cell(model_id, m):>{widths[m]}}" for m in metrics]
        lines.append("  ".join(row))
    report.table("\n".join(lines))


def print_horizon_table(
    report: MdReport,
    scored: list[tuple[str, int, float]],
    model_names: list[str],
    horizons: list[int],
    title: str,
    subtitle: str,
    fmt: str,
) -> None:
    """Print models x horizons from (model, horizon, score) triples.

    `fmt` is the format spec for a cell, since normalized scores and raw CRPS
    want different precision. Rows are sorted by the "overall" column, so the
    table reads best-first, and a model with nothing to average sorts last
    rather than crashing the compare.

    The read-off horizon keeps its own column — it is the comprehension check —
    but is left out of "overall", which is the column the row order comes from.
    """
    overall = {
        model_id: _mean([v for m, h, v in scored if m == model_id and is_forecast(h)])
        for model_id in model_names
    }
    # Keyed by column, "overall" included.
    columns = {"overall": overall} | {
        h: {
            model_id: _mean([v for m, hz, v in scored if m == model_id and hz == h])
            for model_id in model_names
        }
        for h in horizons
    }

    def cell(key, model_id: str) -> str:
        value = columns[key][model_id]
        return "n/a" if value is None else format(value, fmt)

    model_col = max([len("Model")] + [len(m.split("/")[-1]) for m in model_names])
    labels = {"overall": "overall"} | {h: horizon_label(h) for h in horizons}
    # Wide enough for the longest cell in the table, so a metric in the hundreds
    # of thousands doesn't push its columns out of alignment.
    width = max([9] + [len(cell(key, mid)) for key in columns for mid in model_names])
    ordered = sorted(model_names, key=lambda m: (overall[m] is None, overall[m] or 0.0))

    report.heading(title)
    report.text(subtitle)
    header = f"{'Model':<{model_col}}  " + "  ".join(
        f"{labels[key]:>{width}}" for key in columns
    )
    lines = [header, "-" * len(header)]

    for model_id in ordered:
        row = [f"{model_id.split('/')[-1]:<{model_col}}"]
        row += [f"{cell(key, model_id):>{width}}" for key in columns]
        lines.append("  ".join(row))
    report.table("\n".join(lines))


def print_normalized_horizon_table(
    report: MdReport,
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    norm: Normalizer,
) -> None:
    """Print models x horizons, each cell the mean normalized CRPS.

    The scale does not depend on the horizon, so the horizon trend survives:
    later horizons stay harder rather than each being flattened against a
    statistic of its own cohort.
    """
    rows = [
        r
        for r in score_forecasts(corpus, responses, model_names, norm)
        if r["normalized"] is not None
    ]
    scored = [(r["model_id"], r["horizon"], r["normalized"]) for r in rows]
    # The metrics that actually made it into the cells, so the note names the
    # question set the means are over rather than the corpus's whole metric
    # list.
    covered = {r["metric"] for r in rows}
    normalized_labels = [
        str(g.METRIC_LABELS.get(m, m)) for m in metrics_in_order(corpus) if m in covered
    ]
    print_horizon_table(
        report,
        scored,
        model_names,
        sorted({c["horizon"] for c in corpus}),
        "Mean normalized CRPS by model and horizon (lower is better)",
        f"{norm.ratio} ({norm.mode} normalization) over {', '.join(normalized_labels)};"
        " horizons are years past the snapshot"
        f"\noverall {READ_OFF_NOTE}; the {horizon_label(READ_OFF_HORIZON)} column is"
        " kept as the comprehension check it is",
        ".3f",
    )


# ---------------------------------------------------------------------------
# continuous_scores.csv

SCORES_CSV_NAME = "continuous_scores.csv"

# The pooled row's label in both the metric and the horizon column.
ALL = "all"


def horizon_label(horizon: int) -> str:
    """A horizon in years, "3y", the unit a reader of the CSV thinks in."""
    return f"{horizon / g.TURNS_PER_YEAR:g}y"


def metric_label(metric: str) -> str:
    """The metric's report label without the "average " four of them carry."""
    return str(g.METRIC_LABELS.get(metric, metric)).removeprefix("average ")


def scores_csv_rows(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    norms: dict[str, Normalizer],
) -> list[dict]:
    """The rows of continuous_scores.csv, in the order they are written.

    One row per model x metric x horizon, then an "all" metric row per
    (model, horizon) and an "all" horizon row per (model, metric), and the
    (all, all) corner. A pooled row averages the underlying (model, question)
    scores rather than the per-metric means — so a metric with more parsed
    forecasts weighs more, as in the normalized table's "mean" column — and
    its raw CRPS is nan, since raw CRPS is in each metric's own units.

    Per row, nforecasts counts the questions the model was actually prompted
    with — a response on record, parsed or not — and nvalid those whose answer
    parsed; the means are over the latter. The two differ per model under
    --incomplete and agree otherwise. The normalized columns are one per
    normalization mode, computed from the same forecasts, so a row compares the
    modes on an identical question set.

    Only the normalizable metrics get rows: city funds has no scale under any
    mode, and a row of nans would say nothing the raw table does not. The
    read-off horizon gets none either, "all" included — it is a comprehension
    check, not a forecast.
    """
    corpus = forecast_questions(corpus)
    metrics = [m for m in metrics_in_order(corpus) if m not in UNNORMALIZED_METRICS]
    horizons = sorted({c["horizon"] for c in corpus})

    # Every mode's rows, keyed so a question's scores under each can be read
    # side by side. The raw CRPS is identical across modes; it is read off the
    # first.
    scored = {
        mode: {
            (r["model_id"], r["question_id"]): r
            for r in score_forecasts(corpus, responses, model_names, norm)
        }
        for mode, norm in norms.items()
    }
    first = next(iter(scored.values()))

    def nan_mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else math.nan

    metric_groups = [(metric_label(m), [m]) for m in metrics] + [(ALL, metrics)]
    horizon_groups = [(horizon_label(h), [h]) for h in horizons] + [(ALL, horizons)]

    rows = []
    for model_id in model_names:
        for metric_name, metric_group in metric_groups:
            for horizon_name, horizon_group in horizon_groups:
                keys = [
                    (model_id, c["question_id"])
                    for c in corpus
                    if c["metric"] in metric_group and c["horizon"] in horizon_group
                ]
                valid = [k for k in keys if k in first]
                row = {
                    "model": model_id,
                    "metric": metric_name,
                    "horizon": horizon_name,
                    "nforecasts": sum(
                        1 for m, q in keys if ResponseId(m, q) in responses
                    ),
                    "nvalid": len(valid),
                    "CRPS": (
                        math.nan
                        if metric_name == ALL
                        else nan_mean([first[k]["crps"] for k in valid])
                    ),
                }
                for mode, by_key in scored.items():
                    row[f"nCRPS_{mode}"] = nan_mean(
                        [
                            by_key[k]["normalized"]
                            for k in valid
                            if by_key[k]["normalized"] is not None
                        ]
                    )
                rows.append(row)
    return rows


def write_scores_csv(path: Path, rows: list[dict], modes: list[str]) -> Path:
    """Write scores_csv_rows' output, columns in the documented order.

    nan lands as the literal "nan", which pandas and R both read back as
    missing without being told to.
    """
    columns = ["model", "metric", "horizon", "nforecasts", "nvalid", "CRPS"] + [
        f"nCRPS_{mode}" for mode in modes
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_normalized_by_horizon(
    report: MdReport,
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    seed: int,
    outdir: Path,
    norm: Normalizer,
) -> Path:
    """Scatter normalized CRPS against horizon, one series per model.

    The horizon table says the same thing, but reading a trend across a row of
    numbers is work; here the shape is immediate — how steeply accuracy decays
    with distance, and which models depart from the pack. The mean over models is
    drawn as a thick line so it reads as the summary rather than as one more
    model, and the two persistence baselines as dashed and dotted lines, so the
    figure answers "is this good?" and not only "who is best?" — absolute nCRPS
    values carry no scale of their own, and a whole field can sit below the
    baseline.

    The two baselines make the same central guess and differ only in their
    interval, which separates two ways of losing: distance above the dashed line
    is a bad central estimate, and the gap between the lines is what honest
    uncertainty is worth on this corpus.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        r
        for r in score_forecasts(corpus, responses, model_names, norm)
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

    # Models in the legend best-first, so its order is itself a ranking. Ranked
    # on the forecast horizons only, matching the tables' "all" column; ranking
    # on a mean that included the read-off would disagree with them.
    overall = {
        model_id: _mean(
            [
                v
                for h, v in by_model[model_id].items()
                if v is not None and is_forecast(h)
            ]
        )
        for model_id in model_names
    }
    ordered = sorted(model_names, key=lambda m: (overall[m] is None, overall[m] or 0.0))

    # Models bunch tightly at the short horizons, so spread each one's points
    # across a slice of the gap between horizons. Without this the leaders
    # overlap into a single blob and the legend colors can't be matched to
    # anything. The offset is a fixed function of the model's index, not random,
    # so a model sits in the same place in every regenerated figure.
    gap = min((b - a for a, b in pairwise(horizons)), default=1)
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
        color, marker = model_style(i)
        ax.scatter(
            [h + offsets[model_id] for h, _ in points],
            [v for _, v in points],
            color=color,
            marker=marker,
            s=44,
            alpha=0.9,
            linewidths=0.5,
            edgecolors="white",
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

    # The baseline, drawn last so it sits above the model points. Dashed and in
    # a color no model can be assigned, so it reads as a reference level rather
    # than as another series: points below the line beat "nothing changes",
    # points above it are worse than assuming the city stands still.
    # Sigma first, because its question set is the narrower of the two and the
    # plain line is then held to it.
    sigma_by_horizon, scored = persistence_sigma_by_horizon(corpus, seed, norm)
    baseline_points = [
        (h, v)
        for h, v in persistence_by_horizon(corpus, norm, scored).items()
        if v is not None
    ]
    if baseline_points:
        ax.plot(
            [h for h, _ in baseline_points],
            [v for _, v in baseline_points],
            color="crimson",
            lw=2.5,
            ls="--",
            marker="D",
            ms=7,
            zorder=5,
            label=PERSISTENCE_LABEL,
        )

    # The same forecast with an honestly-wide interval. Drawn in the same hue so
    # the pair reads as two versions of one reference rather than two unrelated
    # lines, and lighter, since it is the harder of the two bars to clear.
    sigma_points = [(h, v) for h, v in sigma_by_horizon.items() if v is not None]
    if sigma_points:
        ax.plot(
            [h for h, _ in sigma_points],
            [v for _, v in sigma_points],
            color="darkorange",
            lw=2.5,
            ls=":",
            marker="s",
            ms=7,
            zorder=5,
            label=PERSISTENCE_SIGMA_LABEL,
        )

    ax.set_xlabel(
        "Horizon (years past the snapshot; model points spread within each tick)"
    )
    ax.set_ylabel(f"Normalized CRPS ({norm.ratio}, lower is better)")
    ax.set_title(
        "Normalized CRPS by horizon\n"
        f"{len(model_names)} models, {len(corpus)} questions, "
        f"{norm.mode} normalization\n"
        f"legend ranks on the forecast horizons only ({READ_OFF_NOTE})\n"
        "below dashed beats persistence; below dotted also beats it "
        "with an honest interval"
    )
    ax.set_xticks(horizons)
    ax.set_xticklabels([horizon_label(h) for h in horizons])
    ax.grid(alpha=0.3, zorder=0)
    ax.margins(x=0.04)

    # The baseline counts toward the top too. On the harder scenarios it sits
    # above every model, and a top set from the models alone would push the
    # reference line off the figure — losing exactly the comparison it is
    # drawn for, and silently, since a clipped line still plots. Includes the
    # read-off horizon: its cells sit near zero and so never set the top, but
    # the figure still draws them and an axis that excluded them could clip a
    # point that is on the plot.
    ymax = max(
        [v for v in mean_by_horizon.values() if v is not None]
        + [v for _, v in baseline_points]
        + [v for _, v in sigma_points]
        or [0.0]
    )
    if ymax <= 0:
        raise ValueError(
            "no normalized scores to plot: every forecast either failed to "
            "parse or resolved on a metric with no scale under the "
            f"{norm.mode} normalization ({norm.detail})"
        )
    ymax *= 1.08  # headroom so the topmost marker isn't clipped by the frame
    ax.set_ylim(bottom=0, top=ymax)

    # Drawn here as well as on the correlation figures: the read-off is on this
    # axis as a real point, and near-zero error at the nearest tick reads as
    # models being superb at short range unless it is named.
    annotate_read_off(ax, [(h,) for h in horizons])

    # The legend is as tall as the model list, so it goes beside the axes rather
    # than over the points. Entries are ordered best-first, so the legend doubles
    # as a ranking, with the mean on top as the series to find first.
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_order = [
        "mean over models",
        PERSISTENCE_LABEL,
        PERSISTENCE_SIGMA_LABEL,
    ] + [m.split("/")[-1] for m in ordered]
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

    out = outdir / f"normalized_crps_by_horizon{norm_suffix(norm)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


def normalized_by_model(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    norm: Normalizer,
) -> dict[str, float]:
    """Mean normalized CRPS per model, over every forecast that normalizes.

    Pools the horizons, so the read-off horizon is left out.
    """
    scores: dict[str, list[float]] = {}
    for row in score_forecasts(corpus, responses, model_names, norm):
        if row["normalized"] is None or not is_forecast(row["horizon"]):
            continue
        scores.setdefault(row["model_id"], []).append(row["normalized"])
    return {m: sum(v) / len(v) for m, v in scores.items()}


def plot_overall_bars(
    report: MdReport,
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    outdir: Path,
    norm: Normalizer,
) -> Path | None:
    """Bar the pooled mean normalized CRPS per model, best-first.

    The report's headline: one number per model, the same one the normalized
    table's "overall" column carries, in the order that column sorts. Returns
    None when nothing normalized.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scores = normalized_by_model(corpus, responses, model_names, norm)
    ranked = sorted(
        ((m, scores[m]) for m in model_names if m in scores), key=lambda p: p[1]
    )
    if not ranked:
        return None

    outdir.mkdir(parents=True, exist_ok=True)
    # 4:1 is the plotting area, not the file: the rotated model names below the
    # bars take as much height again, and sizing the figure instead would leave
    # the bars themselves nearer 8:1. Set after tight_layout, which measures the
    # labels, by giving the axes the box it worked out at 1/4 of its width.
    fig, ax = plt.subplots(figsize=(18, 8))

    # Each bar in the model's own color, so a bar can be matched to that model's
    # points on the figures below it.
    order = {m: i for i, m in enumerate(model_names)}
    xs = range(len(ranked))
    ax.bar(
        xs,
        [v for _, v in ranked],
        color=[model_style(order[m])[0] for m, _ in ranked],
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
    )

    for x, (_m, v) in zip(xs, ranked):
        ax.annotate(
            f"{v:.3f}",
            xy=(x, v),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color="#333333",
        )

    ax.set_xticks(list(xs))
    ax.set_xticklabels(
        [m.split("/")[-1] for m, _ in ranked], rotation=40, ha="right", fontsize=8
    )
    ax.set_ylabel(f"Mean nCRPS ({norm.ratio})")
    ax.set_title(
        f"Overall forecast skill — mean normalized CRPS, lower is better  "
        f"({len(ranked)} models, {norm.mode} normalization)\n{READ_OFF_NOTE}"
    )
    ax.margins(x=0.01)
    ax.grid(alpha=0.3, axis="y", zorder=0)
    fig.tight_layout()

    # 4:1 for the bars, with the label and title space tight_layout measured
    # kept as it is; the figure is then trimmed to whatever that leaves.
    box = ax.get_position()
    width_in = box.width * fig.get_figwidth()
    ax.set_position([box.x0, box.y0, box.width, (width_in / 4) / fig.get_figheight()])

    out = outdir / f"overall_normalized_crps{norm_suffix(norm)}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    report.image(out)
    return out


def eci_of(model_id: str) -> float | None:
    """ECI score for a provider/name model id, or None if it has none.

    Re-exported from micropolis_world.model_scores, which reads them from
    model_scores.csv, rather than imported directly by the scripts downstream of
    this one: they already import a dozen helpers from here, and keeping the
    name in one place means the CSV's join rule is stated once.
    """
    return model_scores.eci_of(model_id)


def plot_eci_vs_normalized(
    report: MdReport,
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    outdir: Path,
    norm: Normalizer,
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

    scores = normalized_by_model(corpus, responses, model_names, norm)
    forecasts = forecast_questions(corpus)
    # Carries each model's index in the config's order, so its color and marker
    # here are the ones the horizon figure gave it and the two can be read
    # together.
    points = sorted(
        (eci_of(m), scores[m], m.split("/")[-1], i)
        for i, m in enumerate(model_names)
        if m in scores and eci_of(m) is not None
    )
    skipped = sorted(
        m.split("/")[-1] for m in model_names if m in scores and eci_of(m) is None
    )
    if len(points) < 4:
        report.text(
            f"ECI vs normalized CRPS: only {len(points)} model(s) have an ECI "
            "score; skipping the plot."
        )
        return None

    ecis = [e for e, _, _, _ in points]
    values = [v for _, v, _, _ in points]
    rho, p_rho = stats.spearmanr(ecis, values)
    r, p_r = stats.pearsonr(ecis, values)

    def stars(p: float) -> str:
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

    # Lower nCRPS is better, so ECI going up while error goes down is the
    # capability-tracking direction.
    direction = "pro-g" if rho < 0 else "anti-g"
    report.heading(f"ECI vs mean normalized CRPS (Spearman) — {READ_OFF_NOTE}")
    lines = [
        f"rho={rho:+.3f}  p={p_rho:.4f} {stars(p_rho):<4} ({direction}, n={len(points)})",
        f"Pearson r={r:+.3f}  p={p_r:.4f} {stars(p_r)}",
        (
            "nCRPS is lower-is-better, so rho<0 means the more capable "
            "models forecast better (pro-g)."
        ),
    ]
    if skipped:
        lines.append(f"no ECI score, excluded: {', '.join(skipped)}")
    report.text("\n".join(lines))

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # One scatter call per model rather than one for all of them: the legend
    # replaces the labels that used to sit on the points, and it needs a handle
    # per model to do that. Plotted best-first so the legend doubles as a
    # ranking, the same order the horizon figure's legend uses.
    for eci, value, name, i in sorted(points, key=lambda p: p[1]):
        color, marker = model_style(i)
        ax.scatter(
            [eci],
            [value],
            color=color,
            marker=marker,
            s=90,
            alpha=0.9,
            linewidths=0.5,
            edgecolors="white",
            zorder=3,
            label=name,
        )

    fit = stats.linregress(ecis, values)
    xs = [min(ecis), max(ecis)]
    ax.plot(
        xs,
        [fit.intercept + fit.slope * x for x in xs],
        color="#c2432d",
        lw=1.5,
        zorder=2,
        label=(f"fit: ρ={rho:+.3f} (p={p_rho:.4f}), r={r:+.3f} (p={p_r:.4f})"),
    )

    ax.set_xlabel("ECI (Epoch capability index)")
    ax.set_ylabel(f"Mean normalized CRPS ({norm.ratio}, lower is better)")
    ax.set_title(
        f"Forecast skill vs. ECI  ({len(points)} models,"
        f" {len(forecasts)} questions)\n{READ_OFF_NOTE}"
    )
    ax.grid(alpha=0.3, zorder=0)
    ax.margins(x=0.12, y=0.1)

    # Beside the axes, as on the horizon figure: the legend is as tall as the
    # model list, and over the points it would cover the scatter it explains.
    # The fit line goes on top, since it is the figure's summary and not one
    # more model.
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda j: not labels[j].startswith("fit:"))
    ax.legend(
        [handles[j] for j in order],
        [labels[j] for j in order],
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        framealpha=0.9,
    )
    fig.tight_layout()

    out = outdir / f"eci_vs_normalized_crps{norm_suffix(norm)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


def eci_by_name(model_names: list[str]) -> dict[str, float]:
    """ECI per model, keyed on the bare name, skipping the models without one."""
    return model_scores.eci_by_name(model_names)


def normalized_by_model_and_horizon(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    norm: Normalizer,
) -> dict[int, dict[str, float]]:
    """Mean normalized CRPS per model, per horizon."""
    per_horizon: dict[int, dict[str, list[float]]] = {}
    for row in score_forecasts(corpus, responses, model_names, norm):
        if row["normalized"] is None:
            continue
        by_model = per_horizon.setdefault(row["horizon"], {})
        by_model.setdefault(row["model_id"], []).append(row["normalized"])
    return {
        h: {m: sum(v) / len(v) for m, v in by_model.items()}
        for h, by_model in per_horizon.items()
    }


# Resamples behind each confidence interval. At this many the interval is stable
# to about +/-0.03 across seeds, far finer than the intervals themselves are wide.
BOOTSTRAP_RESAMPLES = 5000

# Fixed so re-running the analysis doesn't move the error bars. The intervals are
# a property of the data, and a band that shifted every run would read as though
# the underlying numbers had changed.
BOOTSTRAP_SEED = 0


def resample_indices(n: int, resamples: int, seed: int):
    """Bootstrap draws over `n` models, as a (resamples, n) index array."""
    import numpy as np

    return np.random.default_rng(seed).integers(0, n, (resamples, n))


def spearman_over_resamples(x, y, idx):
    """Spearman rho for every row of `idx`, as an array with nan where undefined.

    Vectorized over resamples rather than looping: ranks are recomputed per row,
    since a resample repeats models and so introduces ties the full sample does
    not have. Ranking the full sample once and indexing into it would be wrong.

    nan marks a resample that drew a constant column, whose coefficient is
    undefined — this is a real case at the nearest horizon, where many models
    score identically.
    """
    import numpy as np
    from scipy import stats

    rx = stats.rankdata(x[idx], axis=1)
    ry = stats.rankdata(y[idx], axis=1)
    rxc = rx - rx.mean(axis=1, keepdims=True)
    ryc = ry - ry.mean(axis=1, keepdims=True)
    denominator = np.sqrt((rxc**2).sum(axis=1) * (ryc**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0, (rxc * ryc).sum(axis=1) / denominator, np.nan)


def bootstrap_rho_ci(
    xs: list[float],
    ys: list[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float] | None:
    """Percentile bootstrap 95% CI for Spearman rho, resampling models.

    The model is the unit of independence here: each model contributes one
    (predictor, mean nCRPS) pair per horizon, and the many forecasts behind that
    mean are not independent draws of the thing being estimated — a model's skill
    is a property of the model. So the resampling is over models, which is what
    makes the interval an honest statement about generalizing to other models.

    Returns None if too few resamples yield a defined coefficient, which happens
    when a variable is nearly constant and most resamples come out degenerate.
    """
    import numpy as np

    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    rhos = spearman_over_resamples(x, y, resample_indices(len(x), resamples, seed))
    rhos = rhos[~np.isnan(rhos)]
    if len(rhos) < resamples // 2:
        return None
    lo, hi = np.percentile(rhos, [2.5, 97.5])
    return float(lo), float(hi)


def correlate_by_horizon(
    predictor: dict[str, float],
    by_horizon: dict[int, dict[str, float]],
    min_n: int = 4,
    with_ci: bool = False,
) -> list[tuple]:
    """Spearman correlation of `predictor` against nCRPS, one row per horizon.

    `predictor` is keyed on the bare model name and supplies the x values;
    `by_horizon` gives the y values. Only the models present in both count, so
    passing a predictor over a restricted model set restricts the correlation.

    Returns (horizon, rho, p, n) ascending by horizon, skipping any horizon with
    too few models or with no spread in either variable — a constant column
    makes the coefficient undefined, which is a live case at the nearest horizon
    where many models are exactly right.

    With `with_ci`, each row gains a fifth element: the bootstrap 95% interval,
    or None where one could not be computed.
    """
    from scipy import stats

    results = []
    for h in sorted(by_horizon):
        points = [
            (predictor[m.split("/", 1)[1]], v)
            for m, v in by_horizon[h].items()
            if m.split("/", 1)[1] in predictor
        ]
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        if len(points) < min_n or len(set(xs)) < 2 or len(set(ys)) < 2:
            continue
        rho, p = stats.spearmanr(xs, ys)
        row = (h, rho, p, len(points))
        if with_ci:
            row += (bootstrap_rho_ci(xs, ys),)
        results.append(row)
    return results


def compare_predictors_by_horizon(
    a: dict[str, float],
    b: dict[str, float],
    by_horizon: dict[int, dict[str, float]],
    min_n: int = 4,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[int, dict[str, float]]:
    """Per horizon, how much better `a` predicts nCRPS than `b` does.

    Returns {horizon: {diff, lo, hi, share, n}}, where `diff` is
    |rho_a| - |rho_b| on the full sample (positive means `a` is the stronger
    predictor), `lo`/`hi` bound it at 95%, and `share` is the fraction of
    resamples in which `b` came out stronger.

    Both predictors are correlated against the *same* resampled models, which is
    the point: the two correlations share the model sample and the nCRPS
    variable, and are strongly dependent, so the difference is estimated far more
    tightly than either coefficient. Comparing the two marginal intervals instead
    would be the wrong test — they can overlap almost entirely while the
    difference is consistently signed.

    Read `lo`/`hi` as the result, not `share`: the share is a one-sided tail
    fraction and is not a significance test, so a share of 0.05 alongside an
    interval spanning zero means "leans this way, not established".
    """
    import numpy as np

    out: dict[int, dict[str, float]] = {}
    for h in sorted(by_horizon):
        scores = {m.split("/", 1)[1]: v for m, v in by_horizon[h].items()}
        names = [n for n in scores if n in a and n in b]
        if len(names) < min_n:
            continue
        xa = np.array([a[n] for n in names], dtype=float)
        xb = np.array([b[n] for n in names], dtype=float)
        y = np.array([scores[n] for n in names], dtype=float)
        # One index array for both predictors: the same resampled models are
        # scored by each, which is what makes the comparison paired.
        idx = resample_indices(len(names), resamples, seed)
        ra = spearman_over_resamples(xa, y, idx)
        rb = spearman_over_resamples(xb, y, idx)
        usable = ~np.isnan(ra) & ~np.isnan(rb)
        if usable.sum() < resamples // 2:
            continue
        diffs = np.abs(ra[usable]) - np.abs(rb[usable])
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        full_a = spearman_over_resamples(xa, y, np.arange(len(names))[None, :])[0]
        full_b = spearman_over_resamples(xb, y, np.arange(len(names))[None, :])[0]
        out[h] = {
            "diff": float(abs(full_a) - abs(full_b)),
            "lo": float(lo),
            "hi": float(hi),
            "share": float((diffs < 0).sum() / len(diffs)),
            "n": len(names),
        }
    return out


def stars_for(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def format_horizon_correlations(results: list[tuple], indent: str = "  ") -> str:
    """One rho/p line per horizon, with the interval where there is one."""
    lines = []
    for row in results:
        h, rho, p, n = row[:4]
        ci = row[4] if len(row) > 4 else None
        band = f"  95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else ""
        lines.append(
            f"{indent}H{h:<4} rho={rho:+.3f}  p={p:.4f} {stars_for(p):<4} (n={n}){band}"
        )
    return "\n".join(lines)


def read_off_caveat(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    results: list[tuple[int, float, float, int]],
) -> str | None:
    """Warn that the read-off horizon is not really a forecast, if it is in play.

    It asks about a value the snapshot already prints, so almost every model puts
    its median on the actual. What separates them there is whether they also
    collapsed the interval onto it: a model that hedges a value it could have
    copied scores worse without having read anything wrong. So the coefficient at
    this horizon is largely about how confidently a known value is restated, not
    about forecasting, and is not comparable to the rest.

    Both shares are reported because the gap between them is the point — quoting
    only the CRPS-zero one reads as models failing to read the report. Returns
    None when the read-off horizon isn't in play, so callers can skip it outright.
    """
    if not results or results[0][0] != READ_OFF_HORIZON:
        return None
    scored = []
    for c in corpus:
        if c["horizon"] != READ_OFF_HORIZON:
            continue
        for model_id in model_names:
            r = responses.get(ResponseId(model_id, c["question_id"]))
            if r is not None and r.percentiles is not None:
                scored.append((r.percentiles, c["value"]))
    if not scored:
        return None
    median = sum(1 for p, actual in scored if p["p50"] == actual) / len(scored)
    exact = sum(
        1 for p, actual in scored if all(v == actual for v in p.values())
    ) / len(scored)
    return (
        f"{horizon_label(READ_OFF_HORIZON)} is a read-off, not a forecast:"
        f" {median:.0%} of its forecasts put the median on the "
        f"actual and {exact:.0%} collapse the whole interval onto it, so its rho"
        " is mostly about "
        "how confidently a known value is restated; treat it apart."
    )


def draw_horizon_correlation_axes(
    ax,
    series: list[tuple[str, str, list[tuple[int, float, float, int]]]],
    title: str,
) -> None:
    """Draw one or more rho-versus-horizon lines onto `ax`.

    `series` is (label, color, results) per line, with results as returned by
    correlate_by_horizon(). Significant points are filled and the rest hollow,
    so a coefficient that could be noise does not read as a finding.

    Rows carrying a bootstrap interval (from correlate_by_horizon(with_ci=True))
    get a shaded band. Bands rather than capped error bars: with several series
    on one axes, caps at shared horizons collide and read as a grid, while the
    filled region keeps each series' uncertainty attached to its own line.
    """
    all_rhos = []
    all_hs: list[int] = []
    for label, color, results in series:
        hs = [h for h, *_ in results]
        rhos = [row[1] for row in results]
        sig = [row[2] < 0.05 for row in results]
        all_rhos += rhos
        all_hs += [h for h in hs if h not in all_hs]

        # Drawn under the lines, and only where every row has an interval — a band
        # that silently skipped a horizon would misstate where it narrows.
        cis = [row[4] if len(row) > 4 else None for row in results]
        if cis and all(ci is not None for ci in cis):
            ax.fill_between(
                hs,
                [ci[0] for ci in cis],
                [ci[1] for ci in cis],
                color=color,
                alpha=0.12,
                lw=0,
                zorder=1,
            )
            all_rhos += [b for ci in cis for b in ci]
        ax.plot(hs, rhos, color=color, lw=2, zorder=2, label=label)
        ax.scatter(
            [h for h, s in zip(hs, sig) if s],
            [r for r, s in zip(rhos, sig) if s],
            s=80,
            color=color,
            zorder=3,
        )
        ax.scatter(
            [h for h, s in zip(hs, sig) if not s],
            [r for r, s in zip(rhos, sig) if not s],
            s=80,
            facecolors="none",
            edgecolors=color,
            zorder=3,
        )
    ax.axhline(0, color="#666666", lw=1, ls="--", zorder=1)

    ax.set_xlabel("Horizon (years past the snapshot)")
    ax.set_ylabel("Spearman ρ vs. normalized CRPS")
    ax.set_title(title)
    ax.set_xticks(sorted(all_hs))
    ax.set_xticklabels([horizon_label(h) for h in sorted(all_hs)])
    # Zero included so distance from "no relationship" is visible, and the pro-g
    # half of the axis labelled, since the sign is the easy thing to misread.
    low, high = min(all_rhos + [0.0]), max(all_rhos + [0.0])
    pad = (high - low) * 0.18 or 0.1
    ax.set_ylim(low - pad, high + pad)
    ax.annotate(
        "ρ<0: the better-scoring models forecast better (pro-g)",
        xy=(0.5, 0.03),
        xycoords="axes fraction",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    ax.grid(alpha=0.3, zorder=0)
    ax.margins(x=0.06)


def band_handles(color: str, plt) -> list:
    """Legend proxy explaining the shaded band."""
    from matplotlib.patches import Patch

    return [Patch(facecolor=color, alpha=0.12, label="95% CI (bootstrap over models)")]


def significance_handles(color: str, plt) -> list:
    """Legend proxies explaining the filled/hollow marker convention."""
    return [
        plt.Line2D(
            [], [], marker="o", ls="", color=color, markersize=8, label="p < 0.05"
        ),
        plt.Line2D(
            [],
            [],
            marker="o",
            ls="",
            markerfacecolor="none",
            markeredgecolor=color,
            markersize=8,
            label="not significant",
        ),
    ]


def annotate_read_off(ax, rows: list[tuple]) -> None:
    """Mark the read-off horizon as not comparable to the others, if present.

    `rows` is any sequence of tuples whose first element is a horizon — the
    correlation figures pass correlate_by_horizon() results, the scatter passes
    its horizons — since all this needs is whether the read-off is on the axis.

    Anchored to the read-off tick rather than to any one series' point: the
    caveat is about the horizon, and with several series stacked there a leader
    line to one of them reads as singling that series out.
    """
    if READ_OFF_HORIZON not in [h for h, *_ in rows]:
        return
    ax.annotate(
        "read-off,\nnot a forecast",
        xy=(READ_OFF_HORIZON, 0),
        xycoords=("data", "axes fraction"),
        xytext=(0, 26),
        textcoords="offset points",
        fontsize=8,
        color="#777777",
        ha="center",
        arrowprops={"arrowstyle": "-", "color": "#bbbbbb", "lw": 0.8, "shrinkB": 2},
    )


def format_predictor_comparison(
    restricted: list[tuple[str, str, dict[str, float]]],
    by_horizon: dict[int, dict[str, float]],
) -> str:
    """Report how much more closely each predictor tracks nCRPS than the first.

    The per-coefficient intervals in the plot are marginal, and at this many
    models they overlap heavily — which understates what the data can say,
    because the predictors are strongly correlated with each other and share the
    nCRPS variable. This paired resampling asks the question those intervals
    cannot: holding the resampled model set fixed, which predictor tracks skill
    more closely? Reported as text rather than plotted, so the figure stays one
    axes. Returns "" when there is only one predictor to compare.
    """
    if len(restricted) < 2:
        return ""
    base_label, _color, base = restricted[0]
    lines = [
        f"Correlation strength vs {base_label}: |rho_{base_label}| - |rho_other|,",
        f"paired bootstrap over models (positive favors {base_label})",
    ]
    for label, _color, predictor in restricted[1:]:
        rows = compare_predictors_by_horizon(base, predictor, by_horizon)
        if not rows:
            continue
        lines.append(f"  vs {label}")
        for h, r in sorted(rows.items()):
            # Flagged only where the interval clears zero, which is the actual
            # test; the sign of diff alone is not evidence of a difference.
            mark = "*" if r["lo"] > 0 or r["hi"] < 0 else ""
            lines.append(
                f"    H{h:<4} diff={r['diff']:+.3f}"
                f"  95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}] {mark}"
            )
        # Whether any interval clears zero is read off the rows rather than
        # assumed: with a larger model set some of them will, and a hardcoded
        # "none clears zero" would then contradict the stars printed above it.
        signs = [r["diff"] > 0 for r in rows.values() if r["diff"] != 0]
        cleared = sum(r["lo"] > 0 or r["hi"] < 0 for r in rows.values())
        n = next(iter(rows.values()))["n"]
        if signs and all(signs):
            if cleared:
                lines.append(
                    f"    {base_label} leads at every horizon, and {cleared} of"
                    f" {len(rows)} intervals clear zero (n={n})."
                )
            else:
                lines.append(
                    f"    {base_label} leads at every horizon, but no interval"
                    f" clears zero at n={n}, so the consistency"
                    " across horizons is the evidence rather than any one"
                    " horizon."
                )
    return "\n".join(lines)


def format_tie_warnings(
    restricted: list[tuple[str, str, dict[str, float]]], min_distinct: int = 6
) -> str:
    """Flag a predictor too coarse for a rank correlation to resolve.

    Spearman works on ranks, so a predictor with many models tied has less
    resolution than its model count suggests, and its correlation is attenuated
    for a reason that is not about the world. Worth saying outright, because a
    weak coefficient otherwise reads as a substantive finding. Returns "" when
    no predictor is too coarse.
    """
    lines = []
    for label, _color, predictor in restricted:
        distinct = len(set(predictor.values()))
        if distinct < min_distinct and predictor:
            lines.append(
                f"{label}: only {distinct} distinct values across"
                f" {len(predictor)} models, so ties limit how much rank"
                " correlation it can show; read its weakness as partly"
                " granularity, not only signal."
            )
    return "\n".join(lines)


def knowledge_predictor(model_names: list[str]) -> dict[str, float] | None:
    """Knowledge-eval score per model, as a predictor of forecast skill.

    Maps bare model name to normalized score over the whole statement set.
    Returns None when the knowledge eval has no cached responses for the models
    in this dataset, which is the case before its gathering step has run.

    The scores come from the response cache, so this reads no models and needs no
    credentials.
    """
    from micropolis_world.knowledge_eval.scoring import scores_by_model_name

    scores = scores_by_model_name()
    if not {m.split("/", 1)[1] for m in model_names} & set(scores):
        return None
    return scores


def plot_predictors_correlation_by_horizon(
    report: MdReport,
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    outdir: Path,
    norm: Normalizer,
) -> Path | None:
    """Compare ECI and knowledge-eval score as predictors of forecast skill.

    Two rho-versus-horizon lines: ECI and the knowledge-eval score. The question
    is whether knowing this world's facts predicts forecasting it any better than
    a general capability index does.

    Both lines are restricted to the models carrying an ECI score, so they run
    over one model set and their coefficients are directly comparable.

    Returns None when the model set is too small to correlate, or when the
    knowledge eval has no cached answers for these models.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    knowledge = knowledge_predictor(model_names)
    if knowledge is None:
        report.text(
            "Predictor comparison by horizon: no cached knowledge-eval answers"
            " for these models; skipping the plot."
        )
        return None

    eci = eci_by_name(model_names)
    # Restricted to the ECI-scored models, and further to those that also sat the
    # knowledge eval, so both lines cover the same models. Restricting the
    # predictors rather than the y values keeps correlate_by_horizon's
    # present-in-both rule doing the work.
    shared = set(eci) & set(knowledge)
    series_defs = [
        ("ECI", "#3266a8", eci),
        ("Knowledge score", "#c2432d", knowledge),
    ]

    by_horizon = normalized_by_model_and_horizon(corpus, responses, model_names, norm)
    restricted = [
        (label, color, {k: v for k, v in predictor.items() if k in shared})
        for label, color, predictor in series_defs
    ]
    series = []
    for label, color, predictor in restricted:
        results = correlate_by_horizon(predictor, by_horizon, with_ci=True)
        if results:
            series.append((label, color, results))

    if not series:
        report.text(
            f"Predictor comparison by horizon: only {len(shared)} model(s) have"
            " both an ECI score and knowledge-eval answers; skipping the plot."
        )
        return None

    report.heading(
        f"Predictors of nCRPS by horizon (Spearman, {len(shared)} shared models)"
    )
    lines = []
    for label, _color, results in series:
        lines.append(f"{label}")
        lines.append(format_horizon_correlations(results, indent="  "))
    caveat = read_off_caveat(corpus, responses, model_names, series[0][2])
    if caveat:
        lines.append(caveat)
    comparison = format_predictor_comparison(restricted, by_horizon)
    if comparison:
        lines.append(comparison)
    tie_warnings = format_tie_warnings(restricted)
    if tie_warnings:
        lines.append(tie_warnings)
    report.text("\n\n".join(lines))

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    draw_horizon_correlation_axes(
        ax,
        series,
        "What predicts forecast skill: general capability or world knowledge?\n"
        f"{len(shared)} models with both scores, {len(corpus)} questions",
    )
    annotate_read_off(ax, series[0][2])
    handles, labels = ax.get_legend_handles_labels()
    extra = significance_handles("#666666", plt) + band_handles("#666666", plt)
    ax.legend(
        handles=handles + extra,
        labels=labels + [h.get_label() for h in extra],
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )
    fig.tight_layout()

    out = outdir / f"predictors_correlation_by_horizon{norm_suffix(norm)}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default=DEFAULT_CONTINUOUS_CONFIG_PATH)
    ap.add_argument(
        "--norm-global-frac",
        type=float,
        default=None,
        help="Floor the per-question denominators of the local and baseline "
        "normalizations at this share of the metric's global scale "
        f"(config 'norm_global_frac', default {DEFAULT_NORM_GLOBAL_FRAC:g}). "
        "Only the questions it binds are affected, and each report counts them",
    )
    ap.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Skip writing the figures",
    )
    ap.add_argument(
        "--incomplete",
        action="store_true",
        help="Score only the questions gathered for every selected model, "
        "instead of failing when the dataset is missing forecasts",
    )
    args = ap.parse_args()

    cfg = load_config(args)
    label = cfg.get_label(args.label)
    seed = cfg.get_seed(args.seed)
    data_file = data_path(label)
    outdir = plots_path(label)

    # Both failures are user error with an obvious fix — the gathering step
    # hasn't run, or hasn't run for this config — so say so plainly rather than
    # with a traceback.
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
            incomplete=args.incomplete,
        )
    except (FileNotFoundError, DatasetError) as e:
        sys.exit(f"[error] {e}")

    print("=" * 70)
    print("MICROPOLIS WORLD — continuous eval scores")
    print("=" * 70)
    print(f"data:   {data_file}")
    print(f"config: {cfg.path}")
    print(f"label:  {label}")
    print(f"{len(corpus)} questions x {len(models)} models")

    # Every normalization, built after the selection since two of them need
    # per-question numbers for the slice being scored. The failure is the
    # user's to fix — a gathering step to run — so it gets no stack trace.
    try:
        norms = {
            mode: make_normalizer(
                mode,
                corpus,
                global_frac=cfg.get_norm_global_frac(args.norm_global_frac),
                seed=seed,
            )
            for mode in NORM_MODES
        }
    except (NotImplementedError, FileNotFoundError) as e:
        sys.exit(
            f"[error] {e}\n"
            "  every normalization is computed on every run, so the ground truth "
            "is needed even for the global report"
        )

    # A metric with no scale and no place on the exclusion list would drop out
    # of every normalized table without saying so, leaving them quietly
    # narrower than the raw ones. Checked for every mode before anything is
    # written, so a failing mode cannot leave the others' files half-updated.
    for norm in norms.values():
        unscaled = norm.unscaled_metrics(corpus)
        if unscaled:
            sys.exit(
                f"[error] the {norm.mode} normalization has no scale for: "
                f"{', '.join(unscaled)}\n"
                "  add one to continuous_eval.GLOBAL_SCALES, or to "
                "UNNORMALIZED_METRICS to leave the metric out of normalized CRPS"
            )

    # One report and one set of figures per normalization, each self-contained:
    # the raw CRPS table is the same under every mode but is repeated in each,
    # so a report can be read on its own.
    for norm in norms.values():
        print()
        print(f"norm:   {norm.mode} ({norm.detail})")
        if norm.floored is not None:
            print(f"floor:  {norm.floored.note()}")

        report = MdReport()
        report.text(f"Normalized CRPS is {norm.ratio}: {norm.detail}.")
        # Beside the numbers it affected rather than only on stdout: a floored
        # cell is scored against the floor, not against its own scenario, and
        # a reader of the report alone has to be told how much of the table
        # that covers.
        if norm.floored is not None:
            report.text(norm.floored.note().capitalize() + ".")

        # The headline figure, so it is appended before the tables it summarizes
        # rather than with the rest of the plots.
        overall_bars = (
            plot_overall_bars(report, corpus, responses, models, outdir, norm)
            if args.plot
            else None
        )

        print_crps_table(report, corpus, responses, models, norm)
        print_normalized_crps_table(report, corpus, responses, models, norm)
        print_normalized_horizon_table(report, corpus, responses, models, norm)

        if args.plot:
            plots = [
                overall_bars,
                plot_normalized_by_horizon(
                    report, corpus, responses, models, seed, outdir, norm
                ),
                plot_eci_vs_normalized(report, corpus, responses, models, outdir, norm),
                plot_predictors_correlation_by_horizon(
                    report, corpus, responses, models, outdir, norm
                ),
            ]
            for out in plots:
                if out is not None:
                    print(f"Wrote {out}")

        out_path = report.write(
            label_dir(label) / f"analysis-crps{norm_suffix(norm)}.md",
            f"Continuous eval — CRPS ({norm.mode} normalization)",
        )
        print(out_path)

    # Data rather than a figure, so written under --no-plot too; and one file
    # for every mode, hence no suffix. Last, so the tables' own emptiness
    # checks have already fired before a CSV of nothing is written.
    csv_path = write_scores_csv(
        label_dir(label) / SCORES_CSV_NAME,
        scores_csv_rows(corpus, responses, models, norms),
        list(NORM_MODES),
    )
    print()
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
