#!/usr/bin/env -S uv run python3
"""Score the single city eval against a naive baseline instead of against |actual|.

scripts/analyze_single_city.py normalizes CRPS by |actual|, which answers "how
large is the error relative to the thing being forecast". That number has no
zero point: 0.09 is only good or bad relative to how hard the question was. This
script divides by the CRPS of a naive baseline forecast instead, so the scale
carries its own meaning — below 1 beats the baseline, above 1 loses to it.

    skill = CRPS_model / CRPS_baseline     per question
            geometric mean over questions

Two baselines are available through --baseline, both of which predict that
nothing changes from the snapshot value:

  sigma (default)  p50 at the snapshot value and the other quantiles at
                   p50 + z*sigma, sigma being the metric's own historical
                   volatility over a window the length of the horizon. The
                   fairer comparison, since it submits a real interval as the
                   models do, and it is almost never exactly right.
  plain            all five quantiles at the snapshot value. CRPS of a
                   degenerate forecast reduces to absolute error, so this
                   scores 0 whenever the metric did not move at all — 12% of
                   the corpus — and those questions have no defined skill
                   score and are dropped.

Both baselines and their reasoning live in analyze_single_city.py, which this
imports rather than reimplements, so the two scripts cannot disagree about what
the baseline is.

The geometric mean is what makes a ratio averageable: it is symmetric between
twice-as-good and twice-as-bad, and one question where the baseline nearly
nailed the answer cannot swamp a column the way it would under an arithmetic
mean.

This is not a rescaling of the other script's numbers. The denominator varies per
question, so the reweighting reorders the models — Spearman rho between the two
rankings is about +0.81, not +1 — and city funds is included here, having been
excluded there only because |actual| is sometimes 0. Read the two as different
analyses of the same forecasts.

The read-off horizon is excluded throughout: it is a comprehension check, the
baseline resolves it exactly by construction, and a ratio against a zero
denominator says nothing.

Writes tables to stdout and plots to
data/micropolis/single_city/plots/with_baseline/.

Usage:
    scripts/analyze_baseline_skill.py
    scripts/analyze_baseline_skill.py --baseline plain
    scripts/analyze_baseline_skill.py my_config.json5 --no-plot
"""

import argparse
import math
import re
import statistics
import sys
from pathlib import Path

from fbsim_core.metrics import compute_crps

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
    DatasetError,
    ResponseId,
    Responses,
    load_dataset,
    scenario_history,
    select_for_config,
)

# Imported rather than reimplemented so the baseline this scores against is
# provably the same one analyze_single_city.py draws on its figures.
sys.path.insert(0, str(Path(__file__).parent))
from analyze_single_city import (
    NORMAL_Z,
    READ_OFF_HORIZON,
    band_handles,
    correlate_by_horizon,
    draw_horizon_correlation_axes,
    eci_by_name,
    eci_of,
    historical_sigma,
    is_forecast,
    knowledge_predictor,
    print_horizon_correlations,
    print_predictor_comparison,
    print_tie_warnings,
    rank_width_for,
    ranked_cell,
    ranks_within_column,
    significance_handles,
    stars_for,
)

# Plots go in their own directory so this analysis never overwrites a figure
# from the |actual|-normalized one, which stays as it was.
OUT_DIR = PLOTS_PATH / "with_baseline"

# A skill score of exactly 1 is the baseline's own score, so it is the reference
# every axis and table is read against.
BASELINE_SKILL = 1.0

# City funds is reported entirely separately from the other five metrics, never
# pooled with them. It is not a forecast of the same kind: an unmanaged city's
# funds follow a near-deterministic arithmetic progression — a fixed tax income
# against fixed expenses — so a model that spots the pattern predicts it to a
# fraction of a percent, reaching skill ratios of 0.002, while the baseline's
# interval is far too wide for a series that barely wanders. Pooling it moved the
# headline from 2 of 18 models beating the baseline to 6 of 18 and shifted
# individual models by five places, all on the strength of one metric measuring
# something the other five do not.
FUNDS_METRIC = "totalFunds"

# The two analyses, each getting its own tables and figures: the behavioral
# metrics that describe how the city evolves, and city funds on its own.
SPLITS = {
    "behavioral": (
        "behavioral metrics",
        f"the five metrics other than {FUNDS_METRIC}",
    ),
    "funds": (
        "city funds",
        "reported alone: a near-deterministic series, unlike the other five",
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

    A question missing from the result has no usable baseline, and so no skill
    score: either its history was too short to estimate a spread, or its run log
    is not cached, or the baseline scored exactly 0 and the ratio would be
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
        # beats an epsilon, which would invent a skill score of ~1e9 and
        # dominate any mean it entered.
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

    A row carries the skill ratio and its log, since every aggregate here is a
    geometric mean and taking the log once per row keeps that cheap and keeps the
    two definitions from drifting apart. It also carries the metric, horizon and
    disasters flag, which are what the tables and figures group by.

    The tally counts question-model pairs by why they have no skill score, and is
    printed alongside the tables: with the plain baseline a large fraction of the
    corpus has no defined ratio, and a table that quietly averaged what was left
    would hide it.
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
            # A model that is exactly right has skill 0, whose log is -inf and
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
                    # Carried on the row so the disasters split is a property of
                    # the score rather than something the plotting code has to
                    # rediscover from the corpus by question id.
                    "disasters": c["scenario"]["disasters"],
                    "skill": crps / base,
                    "log_skill": math.log(crps / base),
                }
            )
    return rows, dropped


def geometric_mean_of(values: list[float]) -> float | None:
    """Geometric mean of already-computed ratios, or None if there are none.

    For averaging cells that are themselves means — the figures' mean-over-models
    line — where geometric_mean's row dicts would have to be fabricated.
    """
    if not values:
        return None
    return math.exp(statistics.fmean(math.log(v) for v in values))


def geometric_mean(rows: list[dict]) -> float | None:
    """Geometric mean of `rows`' skill ratios, or None if there are none.

    The mean of the logs, exponentiated. Averaging ratios arithmetically would
    let one question where the baseline nearly nailed the answer — a ratio in the
    hundreds — outweigh a hundred questions where the model was slightly better,
    and would break the symmetry that makes 0.5 and 2.0 equally far from parity.
    """
    if not rows:
        return None
    return math.exp(statistics.fmean(r["log_skill"] for r in rows))


def skill_by(rows: list[dict], *keys: str) -> dict[tuple, float]:
    """Geometric-mean skill grouped by the named row fields.

    One helper for every table and plot here, so "the skill of this cell" is
    computed one way throughout.
    """
    grouped: dict[tuple, list[dict]] = {}
    for r in rows:
        grouped.setdefault(tuple(r[k] for k in keys), []).append(r)
    # Tested against None rather than for truthiness: a geometric mean of
    # positive ratios cannot be 0 today, but "no questions in this cell" and "this
    # cell scored 0" are different facts and a falsiness test would merge them.
    means = {k: geometric_mean(v) for k, v in grouped.items()}
    return {k: v for k, v in means.items() if v is not None}


def split_rows(rows: list[dict], split: str) -> list[dict]:
    """`rows` narrowed to one side of the city-funds split.

    Every table and figure here takes one side or the other; nothing averages
    across the split. See FUNDS_METRIC for why.
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


def print_dropped(dropped: dict[str, int], total: int) -> None:
    """Say what had no skill score and why, as a share of the pairs attempted.

    Always printed, including when nothing was dropped, so the absence of a
    warning is informative rather than ambiguous. The plain baseline drops far
    more than the spread-widened one — it is exactly right whenever the metric
    did not move — and that difference is the main thing to know when comparing
    a run under one baseline against a run under the other.
    """
    labels = {
        "no_baseline": "no usable baseline (baseline scored 0, or no spread estimate)",
        "unparsed": "model forecast did not parse",
        "zero_model_crps": "model was exactly right (skill 0, no log)",
    }
    n = sum(dropped.values())
    attempted = total + n
    if not attempted:
        return
    print(f"\n{n} of {attempted} (model, question) pairs have no skill score:")
    for key, count in dropped.items():
        if count:
            print(f"  {count:>6}  {labels[key]}")
    if not n:
        print("  none")


def print_skill_by_metric(
    rows: list[dict], model_names: list[str], kind: str, split: str
) -> None:
    """Print models x metrics of geometric-mean skill, each cell with its rank.

    Unlike raw CRPS, these cells are comparable across columns as well as down
    them: the units cancel in the ratio, so a 0.8 on population and a 0.8 on
    pollution both mean "20% better than the baseline". That is what this table
    has over the raw-CRPS one in the |actual|-normalized script, which can only
    be read down a column.
    """
    metrics = metrics_in_order_all(rows)
    labels = {m: str(g.METRIC_LABELS.get(m, m)) for m in metrics}
    cells = skill_by(rows, "model_id", "metric")
    overall = skill_by(rows, "model_id")

    columns = {"all": {m: overall.get((m,)) for m in model_names}} | {
        metric: {m: cells.get((m, metric)) for m in model_names} for metric in metrics
    }
    ranks = {key: ranks_within_column(values) for key, values in columns.items()}
    rank_width = max(rank_width_for(r) for r in ranks.values())

    counts = {
        model_id: sum(1 for r in rows if r["model_id"] == model_id)
        for model_id in model_names
    }

    def cell(key, model_id: str) -> str:
        return ranked_cell(
            columns[key][model_id], ranks[key].get(model_id), ".3f", rank_width
        )

    model_col = max([len("Model")] + [len(m.split("/")[-1]) for m in model_names])
    header_labels = {"all": "all"} | labels
    width = max(
        [9]
        + [len(cell(key, mid)) for key in columns for mid in model_names]
        + [len(header_labels[key]) for key in columns]
    )
    questions_width = max(len("scored"), max(len(str(c)) for c in counts.values()))
    ordered = sorted(
        model_names, key=lambda m: (overall.get((m,)) is None, overall.get((m,)) or 0.0)
    )

    print("\nSkill vs baseline by model and metric (below 1 beats the baseline)")
    print(split_note(split))
    print(baseline_note(kind))
    print(
        "each cell is the geometric mean of CRPS_model/CRPS_baseline; "
        f"H{READ_OFF_HORIZON} excluded"
    )
    header = f"{'Model':<{model_col}}  {'scored':>{questions_width}}  " + "  ".join(
        f"{header_labels[key]:>{width}}" for key in columns
    )
    print(header)
    print("-" * len(header))
    for model_id in ordered:
        row = [
            f"{model_id.split('/')[-1]:<{model_col}}",
            f"{counts[model_id]:>{questions_width}}",
        ]
        row += [f"{cell(key, model_id):>{width}}" for key in columns]
        print("  ".join(row))


def metrics_in_order_all(rows: list[dict]) -> list[str]:
    """The metrics present in `rows`, in the corpus's canonical order.

    metrics_in_order sorts the metrics excluded from |actual| normalization to
    the far right, which is the right call there and the wrong one here: city
    funds is a full participant under a ratio, so it takes its natural place.
    """
    present = {r["metric"] for r in rows}
    return [m for m in g.METRICS if m in present]


def print_skill_by_horizon(
    rows: list[dict], model_names: list[str], kind: str, split: str
) -> None:
    """Print models x horizons of geometric-mean skill, each cell with its rank.

    The interesting question this answers that the metric table cannot: whether a
    model's advantage over the baseline holds up with distance. A model can beat
    persistence at short range simply by nudging the snapshot value in the right
    direction, and still have nothing to say at 240 turns.
    """
    horizons = sorted({r["horizon"] for r in rows})
    cells = skill_by(rows, "model_id", "horizon")
    overall = skill_by(rows, "model_id")

    columns = {"all": {m: overall.get((m,)) for m in model_names}} | {
        h: {m: cells.get((m, h)) for m in model_names} for h in horizons
    }
    ranks = {key: ranks_within_column(values) for key, values in columns.items()}
    rank_width = max(rank_width_for(r) for r in ranks.values())

    def cell(key, model_id: str) -> str:
        return ranked_cell(
            columns[key][model_id], ranks[key].get(model_id), ".3f", rank_width
        )

    model_col = max([len("Model")] + [len(m.split("/")[-1]) for m in model_names])
    labels = {"all": "all"} | {h: f"H{h}" for h in horizons}
    width = max(
        [9]
        + [len(cell(key, mid)) for key in columns for mid in model_names]
        + [len(labels[key]) for key in columns]
    )
    ordered = sorted(
        model_names, key=lambda m: (overall.get((m,)) is None, overall.get((m,)) or 0.0)
    )

    print("\nSkill vs baseline by model and horizon (below 1 beats the baseline)")
    print(split_note(split))
    print(baseline_note(kind))
    print(
        "horizons are turns past the snapshot; "
        f"H{READ_OFF_HORIZON} excluded (a read-off, and the baseline resolves it "
        "exactly)"
    )
    header = f"{'Model':<{model_col}}  " + "  ".join(
        f"{labels[key]:>{width}}" for key in columns
    )
    print(header)
    print("-" * len(header))
    for model_id in ordered:
        row = [f"{model_id.split('/')[-1]:<{model_col}}"]
        row += [f"{cell(key, model_id):>{width}}" for key in columns]
        print("  ".join(row))


def plot_skill_by_horizon(
    rows: list[dict],
    model_names: list[str],
    kind: str,
    split: str,
    subset: str = "",
    ylim: tuple[float, float] | None = None,
    outdir: Path = OUT_DIR,
) -> Path:
    """Scatter skill against horizon, one series per model, on a log y-axis.

    The baseline is a horizontal line at 1 rather than a curve: it is its own
    reference, so what the figure shows directly is who is under it and whether
    they stay under it as the horizon grows. That is the question the
    |actual|-normalized version of this figure could only answer by eye, by
    comparing two sloping lines.

    The y-axis is logarithmic so that twice-as-good and twice-as-bad sit equally
    far from the line, matching the geometric mean the points are computed with.
    On a linear axis the region below 1 is compressed into a tenth of the height
    while the region above it runs to 6, which would make a model that halves the
    baseline's error look like a rounding difference.

    `ylim` fixes the axis across a set of figures so they can be read against
    each other; `subset` names the slice for the title and filename.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = sorted({r["horizon"] for r in rows})
    cells = skill_by(rows, "model_id", "horizon")
    by_model = {
        model_id: [
            (h, cells[(model_id, h)]) for h in horizons if (model_id, h) in cells
        ]
        for model_id in model_names
    }
    # Averaged over the per-model cells, not over the raw questions, so every
    # model counts equally however many of its forecasts had a usable baseline.
    # This is deliberately not the same number as a table cell: the tables average
    # a model's own questions, and the two differ by up to about 1% at H240, where
    # the models' question counts diverge most.
    mean_by_horizon = {
        h: geometric_mean_of([cells[(m, h)] for m in model_names if (m, h) in cells])
        for h in horizons
    }

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    n_colors = 10 if len(model_names) <= 10 else 20
    palette = plt.get_cmap(f"tab{n_colors}")

    overall = skill_by(rows, "model_id")
    ordered = sorted(
        model_names,
        key=lambda m: (overall.get((m,)) is None, overall.get((m,)) or 0.0),
    )

    # Same fixed per-model dodge as the |actual|-normalized figure, so a model
    # sits in the same place in every regenerated figure and the leaders don't
    # overlap into a single blob at the short horizons.
    gap = min((b - a for a, b in zip(horizons, horizons[1:])), default=1)
    spread = gap * 0.35
    offsets = {
        model_id: (i / max(len(model_names) - 1, 1) - 0.5) * spread
        for i, model_id in enumerate(model_names)
    }

    for i, model_id in enumerate(model_names):
        points = by_model[model_id]
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
            label="geometric mean over models",
        )

    # The baseline itself. Drawn across the full width rather than as a series,
    # since it is 1 at every horizon by construction.
    ax.axhline(
        BASELINE_SKILL,
        color="crimson",
        lw=2.5,
        ls="--",
        zorder=5,
        label=f"baseline ({BASELINES[kind][0]})",
    )

    ax.set_yscale("log")
    ax.set_xlabel(
        "Horizon (turns past the snapshot; model points spread within each tick)"
    )
    ax.set_ylabel("Skill vs baseline (CRPS_model / CRPS_baseline, log scale)")
    ax.set_title(
        f"Forecast skill vs baseline by horizon{f' — {subset}' if subset else ''}\n"
        f"{len(model_names)} models, {SPLITS[split][0]}\n"
        f"baseline: {BASELINES[kind][0]}; below the dashed line beats it"
    )
    ax.set_xticks(horizons)
    ax.grid(alpha=0.3, which="both", zorder=0)
    ax.margins(x=0.04)
    if ylim:
        ax.set_ylim(*ylim)

    # Ticks read as ratios rather than as powers of ten, which is what the axis
    # means: 0.5 is "half the baseline's error". Placed at fixed ratios spanning
    # the data rather than left to the log locator, which on a range this narrow
    # labels only the decade boundary and so would leave the axis with a single
    # tick at 1.
    lo, hi = ax.get_ylim()
    candidates = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2, 3, 4, 6, 8, 12, 16]
    ticks = [t for t in candidates if lo <= t <= hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}x" if t != 1 else "1x (baseline)" for t in ticks])
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_order = [
        "geometric mean over models",
        f"baseline ({BASELINES[kind][0]})",
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

    suffix = f"-{slugify(subset)}" if subset else ""
    out = outdir / f"skill_by_horizon-{split}{suffix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def slugify(text: str) -> str:
    """`text` as a filename component."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def skill_by_model_and_horizon(
    rows: list[dict], model_names: list[str]
) -> dict[int, dict[str, float]]:
    """{horizon: {model_id: skill}}, the shape the correlation helpers want."""
    cells = skill_by(rows, "model_id", "horizon")
    horizons = sorted({r["horizon"] for r in rows})
    return {
        h: {m: cells[(m, h)] for m in model_names if (m, h) in cells} for h in horizons
    }


def plot_eci_vs_skill(
    rows: list[dict],
    model_names: list[str],
    kind: str,
    split: str,
    outdir: Path = OUT_DIR,
) -> Path | None:
    """Scatter each model's ECI against its overall skill vs the baseline.

    Same question as the |actual|-normalized script's ECI figure — does
    forecasting this world track general capability — but on a scale with a
    meaningful zero, so the figure also shows *where* the capability frontier
    crosses from losing to the baseline to beating it. Returns None when too few
    models carry an ECI score for a correlation to mean anything.

    Sign: skill is lower-is-better, so a negative correlation is the pro-g one.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    scores = {k[0]: v for k, v in skill_by(rows, "model_id").items()}
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
            f"\nECI vs skill ({SPLITS[split][0]}): only {len(points)} model(s) "
            "have an ECI score; skipping the plot."
        )
        return None

    ecis = [e for e, _, _ in points]
    values = [v for _, v, _ in points]
    rho, p_rho = stats.spearmanr(ecis, values)
    # Pearson on the logs, since the score is a ratio: r on the raw ratios would
    # be driven by the handful of models several times worse than the baseline,
    # which have far more room above 1 than any model has below it.
    r, p_r = stats.pearsonr(ecis, [math.log(v) for v in values])
    direction = "pro-g" if rho < 0 else "anti-g"

    print(f"\nECI vs skill vs baseline — {SPLITS[split][0]}")
    print(f"  {baseline_note(kind)}")
    print(
        f"  rho={rho:+.3f}  p={p_rho:.4f} {stars_for(p_rho):<4} ({direction}, "
        f"n={len(points)})"
    )
    print(f"  Pearson r={r:+.3f}  p={p_r:.4f} {stars_for(p_r)} (on log skill)")
    print(
        "  skill is lower-is-better, so rho<0 means the more capable models\n"
        "  beat the baseline by more (pro-g)."
    )
    beaten = sum(1 for v in values if v < BASELINE_SKILL)
    print(f"  {beaten} of {len(values)} models with an ECI score beat the baseline")
    if skipped:
        print(f"  no ECI score, excluded: {', '.join(skipped)}")

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(ecis, values, s=70, color="#3266a8", zorder=3)

    # Fitted in log space, matching both the axis and the geometric mean the
    # points are computed with. A straight fit in ratio space would render as a
    # curve here, and would let one model 3x worse than the baseline pull the line
    # further than one 3x better pulls it back.
    fit = stats.linregress(ecis, [math.log(v) for v in values])
    xs = [min(ecis), max(ecis)]
    ax.plot(
        xs,
        [math.exp(fit.intercept + fit.slope * x) for x in xs],
        color="#c2432d",
        lw=1.5,
        zorder=2,
        label=f"log-space fit: ρ={rho:+.3f} (p={p_rho:.4f}), r={r:+.3f} (p={p_r:.4f})",
    )
    # The parity line is the whole point of this scale: it splits the models that
    # add something over the naive forecast from those that do not.
    ax.axhline(
        BASELINE_SKILL,
        color="crimson",
        lw=2,
        ls="--",
        zorder=2,
        label=f"baseline ({BASELINES[kind][0]})",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

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
        f"Skill vs baseline against ECI — {SPLITS[split][0]}\n"
        f"{len(points)} models; baseline: {BASELINES[kind][0]}\n"
        "below the dashed line beats the baseline"
    )
    ax.grid(alpha=0.3, which="both", zorder=0)
    ax.margins(x=0.12, y=0.1)
    fig.tight_layout()

    fig.canvas.draw()
    place_labels(fig, ax, [n for _, _, n in points], ecis, values)

    out = outdir / f"eci_vs_skill-{split}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def relabel_correlation_axes(ax) -> None:
    """Fix up the shared correlation axes for a skill-vs-baseline y variable.

    draw_horizon_correlation_axes is written for the |actual|-normalized script
    and hardcodes both its y-label and a footnote reading "the better-scoring
    models forecast better", which describes neither axis here — x is a predictor
    (ECI or knowledge score), not a score, and y is a ratio against the baseline.
    Rewriting them in place keeps that helper shared, and unchanged, rather than
    forking it or editing the script it belongs to.
    """
    ax.set_ylabel("Spearman ρ vs. skill (CRPS_model / CRPS_baseline)")
    for child in ax.texts:
        if "better-scoring" in child.get_text():
            child.set_text(
                "ρ<0: the models scoring higher on the predictor beat the "
                "baseline by more"
            )


def plot_eci_correlation_by_horizon(
    rows: list[dict],
    model_names: list[str],
    kind: str,
    split: str,
    outdir: Path = OUT_DIR,
) -> Path | None:
    """Spearman rho of ECI against skill, per horizon, with bootstrap intervals.

    Says whether capability predicts skill more strongly the further out the
    forecast goes. Returns None when no horizon has enough models to correlate.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predictor = eci_by_name(model_names)
    by_horizon = skill_by_model_and_horizon(rows, model_names)
    results = correlate_by_horizon(predictor, by_horizon, with_ci=True)
    if not results:
        print(
            f"\nECI correlation by horizon ({SPLITS[split][0]}): no horizon has "
            "enough models with an ECI score; skipping the plot."
        )
        return None

    print(f"\nECI vs skill by horizon (Spearman) — {SPLITS[split][0]}")
    print(f"  {baseline_note(kind)}")
    print_horizon_correlations(results)

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_horizon_correlation_axes(
        ax,
        [("ECI", "#3266a8", results)],
        f"Does capability predict beating the baseline? — {SPLITS[split][0]}\n"
        f"Spearman ρ of ECI vs skill, by horizon; baseline: {BASELINES[kind][0]}",
    )
    relabel_correlation_axes(ax)
    # The proxies carry their own labels, so matplotlib reads them off directly.
    ax.legend(
        handles=band_handles("#3266a8", plt) + significance_handles("#3266a8", plt),
        loc="lower right",
        fontsize=9,
        framealpha=0.9,
    )
    fig.tight_layout()
    out = outdir / f"eci_correlation_by_horizon-{split}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_predictors_correlation_by_horizon(
    rows: list[dict],
    model_names: list[str],
    kind: str,
    split: str,
    outdir: Path = OUT_DIR,
) -> Path | None:
    """Compare ECI and knowledge-eval score as predictors of skill vs baseline.

    Whether knowing this world's facts predicts beating the naive forecast any
    better than a general capability index does. Both lines are restricted to the
    models carrying both scores, so their coefficients are comparable.

    Returns None when the model set is too small, or when the knowledge eval has
    no cached answers for these models.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    knowledge = knowledge_predictor(model_names)
    if knowledge is None:
        print(
            f"\nPredictor comparison ({SPLITS[split][0]}): no cached knowledge-eval"
            " answers for these models; skipping the plot."
        )
        return None

    eci = eci_by_name(model_names)
    shared = set(eci) & set(knowledge)
    restricted = [
        (label, color, {k: v for k, v in predictor.items() if k in shared})
        for label, color, predictor in [
            ("ECI", "#3266a8", eci),
            ("Knowledge score", "#c2432d", knowledge),
        ]
    ]

    by_horizon = skill_by_model_and_horizon(rows, model_names)
    series = []
    for label, color, predictor in restricted:
        results = correlate_by_horizon(predictor, by_horizon, with_ci=True)
        if results:
            series.append((label, color, results))
    if not series:
        print(
            f"\nPredictor comparison ({SPLITS[split][0]}): only {len(shared)} "
            "model(s) have both scores; skipping the plot."
        )
        return None

    print(
        f"\nPredictors of skill by horizon — {SPLITS[split][0]} "
        f"({len(shared)} shared models)"
    )
    print(f"  {baseline_note(kind)}")
    for label, _color, results in series:
        print(f"  {label}")
        print_horizon_correlations(results, indent="    ")
    print_predictor_comparison(restricted, by_horizon)
    print_tie_warnings(restricted)

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    draw_horizon_correlation_axes(
        ax,
        series,
        "What predicts beating the baseline: capability or world knowledge?\n"
        f"{SPLITS[split][0]}; {len(shared)} models with both scores\n"
        f"baseline: {BASELINES[kind][0]}",
    )
    relabel_correlation_axes(ax)
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

    out = outdir / f"predictors_correlation_by_horizon-{split}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_horizon_figures(
    rows: list[dict], model_names: list[str], kind: str, split: str
) -> list[Path]:
    """The skill-by-horizon scatter over all runs, then split by disasters.

    Same rationale as the |actual|-normalized script: disasters are the corpus's
    one deliberate difficulty axis, and the split says whether a model's decay
    with horizon is about forecasting a city at all or about coping with shocks.

    Under a ratio the split gains a sharper reading than it had there. Both the
    models and the baseline face the same disasters, so a model whose skill holds
    up with disasters on is adding something the naive forecast cannot — rather
    than merely facing an easier question.
    """
    subsets = [("", None), ("disasters", True), ("no disasters", False)]

    # One y-axis across the set so the three figures can be read against each
    # other, taken from the per-(model, horizon) cells the figures plot.
    limits = []
    grouped = {}
    for subset, want in subsets:
        selected = [r for r in rows if want is None or r["disasters"] == want]
        if not selected:
            continue
        grouped[subset] = selected
        limits += list(skill_by(selected, "model_id", "horizon").values())
    if not limits:
        raise ValueError(
            "no skill scores to plot: every selected forecast either failed to "
            "parse or had no usable baseline"
        )
    # Padded multiplicatively, since the axis is logarithmic; parity is always
    # inside the range so the reference line cannot fall off the figure.
    ylim = (min(limits + [BASELINE_SKILL]) / 1.3, max(limits + [BASELINE_SKILL]) * 1.3)

    return [
        plot_skill_by_horizon(selected, model_names, kind, split, subset, ylim)
        for subset, selected in grouped.items()
    ]


def run_split(
    rows: list[dict],
    model_names: list[str],
    kind: str,
    split: str,
    plot: bool,
) -> list[Path]:
    """Every table and figure for one side of the city-funds split.

    Returns the paths written, so main can list them together after the tables
    rather than interleaving "Wrote" lines with the correlation output.
    """
    selected = split_rows(rows, split)
    name, how = SPLITS[split]
    print()
    print("=" * 70)
    print(f"{name.upper()} — {how}")
    print("=" * 70)
    if not selected:
        print("no scored forecasts on this side of the split; nothing to report")
        return []

    print_skill_by_metric(selected, model_names, kind, split)
    print_skill_by_horizon(selected, model_names, kind, split)
    if not plot:
        return []
    figures = [
        plot_eci_vs_skill(selected, model_names, kind, split),
        plot_eci_correlation_by_horizon(selected, model_names, kind, split),
        plot_predictors_correlation_by_horizon(selected, model_names, kind, split),
    ]
    return plot_horizon_figures(selected, model_names, kind, split) + [
        f for f in figures if f is not None
    ]


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument(
        "--baseline",
        choices=sorted(BASELINES),
        default="sigma",
        help=(
            "Which naive forecast to score against. 'sigma' (default) widens the "
            "interval by the metric's historical volatility; 'plain' puts all "
            "five quantiles on the snapshot value, and so is exactly right — and "
            "gives no usable ratio — whenever the metric did not move"
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

    try:
        corpus, responses, models = load_dataset()
        corpus, responses, models = select_for_config(
            corpus, responses, models, cfg, seed
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
    print("MICROPOLIS WORLD — single city eval, skill vs a naive baseline")
    print("=" * 70)
    print(f"data:   {DATA_PATH}")
    print(f"config: {cfg.path}")
    print(f"plots:  {OUT_DIR}")
    print(f"{len(forecasts)} forecast questions x {len(models)} models")
    print(baseline_note(args.baseline))
    print(
        "score = CRPS_model / CRPS_baseline per question, geometric mean over "
        "questions;\nbelow 1 beats the baseline. City funds is reported "
        "separately from the other\nfive metrics — see the module docstring for "
        "why."
    )

    rows, dropped = score_skill(corpus, responses, models, seed, args.baseline)
    print_dropped(dropped, len(rows))
    if not rows:
        sys.exit(
            "[error] no forecast has a usable skill score; nothing to report. "
            "The counts above say why"
        )

    written = []
    for split in SPLITS:
        written += run_split(rows, models, args.baseline, split, args.plot)

    if written:
        print()
        for out in written:
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()
