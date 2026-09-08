#!/usr/bin/env -S uv run python3
"""Score the binary eval: Brier and calibration tables by question and horizon.

Reads data/micropolis/binary/{label}/data.json, written by
scripts/run_eval_binary.py, and the continuation tallies under
data/micropolis/ground_truth/, written by scripts/extract_ground_truth.py.
Prompts no models and runs no simulations.

Every forecast f gets two scores: the Brier score (f - outcome)^2 against the
realized answer, and the calibration error (f - p)^2 against p, the share of
reseeded continuations that resolved Yes. The report is split into two
sections — "Binary forecasts", the mid-range A questions, and "Tail
probabilities", the B ones — each with the same structure: a base-rate
heatmap, then the two scores paired within each view (a models x questions
heatmap, a per-model bar panel, a by-horizon figure and the ECI scatter), and
finally a grid of per-model calibration scatters. The tail section draws
its color and its scatter axes on a log scale, since its probabilities span
two decades. Everything goes to one Markdown report,
data/micropolis/binary/{label}/analysis-brier.md; --no-plot skips the figures.
Beside it goes binary_scores.csv: one row per model x question type (regular =
A, tail = B) x horizon in years plus an "all" horizon row, with the prompted
and parsed counts and the mean Brier, expected Brier and calibration error.

Usage:
    scripts/analyze_binary.py                       # configs/binary.json5
    scripts/analyze_binary.py subset.json5
    scripts/analyze_binary.py --no-plot
    scripts/analyze_binary.py --cities kyoto --disasters false
    scripts/analyze_binary.py --models openai/gpt-5 --label myrun
"""

import argparse
import csv
import math
import sys
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

from fbsim_core.metrics import compute_brier_score

from micropolis_world.binary_eval import (
    BinaryResponses,
    data_path,
    label_dir,
    load_dataset_binary,
)
from micropolis_world.binary_questions import QUESTION_IDS
from micropolis_world.config import (
    CONFIG_DIR,
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.continuous_eval import (
    DatasetError,
    MdReport,
    ResponseId,
    select_for_config,
)
from micropolis_world.ground_truth import Truth, load_truths
from micropolis_world.plot_labels import place_labels

# Imported rather than reimplemented so the table formatting, the correlation
# machinery and the CI conventions are provably the same ones the continuous
# report uses.
sys.path.insert(0, str(Path(__file__).parent))
from analyze_continuous import (
    _mean,
    band_handles,
    correlate_by_horizon,
    draw_horizon_correlation_axes,
    eci_by_name,
    eci_of,
    format_horizon_correlations,
    format_predictor_comparison,
    format_tie_warnings,
    knowledge_predictor,
    significance_handles,
    stars_for,
)

DEFAULT_BINARY_CONFIG_PATH = CONFIG_DIR / "binary.json5"

# The report's two halves. The A questions target mid-range base rates and the
# B questions tail ones (binary_forecasts.md §3), so their scores live on
# different scales — an always-No forecast is already near-perfect on B — and
# averaging them together would let the tail questions dilute the mid-range
# signal. Everything below is computed per section.
SECTIONS = [
    ("A", "Binary forecasts", "mid-range questions (target P(Yes) ~ 10-90%)"),
    ("B", "Tail probabilities", "tail questions (target P(Yes) ~ 0.5-5%)"),
]

# The sections' names in binary_scores.csv's question_type column.
QUESTION_TYPES = {"A": "regular", "B": "tail"}


@dataclass(frozen=True)
class Score:
    """One way of scoring a forecast; every table and figure is made per score."""

    key: str  # the row field holding the score, and the figure file prefix
    name: str  # as it appears in titles: "Mean {name} by ..."
    definition: str


SCORES = [
    Score("brier", "Brier", "(f - outcome)^2 against the realized answer"),
    Score(
        "calibration",
        "calibration error",
        "(f - p)^2 against p, the share of reseeded continuations resolving Yes",
    ),
]

PLOT_BLUE = "#3266a8"

# Bins for the calibration line. Ten over ~1700 tail forecasts per model
# leaves each bin with enough to average; more would make the standard errors
# swamp the line.
NCAL_BINS = 10

# The engine runs 48 turns to the simulated year, so turn counts are reported
# as years: a snapshot turn as the city's age, a horizon as its span.
TURNS_PER_YEAR = 48


def years(turns: int) -> str:
    """A turn count as years, without a trailing ".0" on the whole ones."""
    y = turns / TURNS_PER_YEAR
    return f"{y:.0f}y" if y == int(y) else f"{y:g}y"


def window_label(snapshot_turn: int, horizon: int) -> str:
    """A (snapshot, horizon) window as "Y20: 5y" — the city's age, then the span."""
    return f"Y{snapshot_turn // TURNS_PER_YEAR}: {years(horizon)}"


def plots_path(label: str) -> Path:
    return label_dir(label) / "plots"


def score_forecasts_binary(
    corpus: list[dict],
    responses: BinaryResponses,
    model_names: list[str],
    truths: dict[str, Truth],
) -> list[dict]:
    """One row per parsed forecast: its Brier score, its calibration error,
    its expected Brier score, the forecast and ground truth behind them, and
    its qid and horizon.

    The binary counterpart of continuous_eval.score_forecasts. The scores are
    unitless and bounded, so there is no normalized twin. The expected Brier
    is what the Brier score averages to over the continuations' outcomes,
    (f - p)^2 + p(1 - p): the calibration error plus the irreducible variance
    of the event itself.
    """
    rows = []
    for c in corpus:
        truth = truths[c["question_id"]]
        for model_id in model_names:
            r = responses.get(ResponseId(model_id, c["question_id"]))
            if r is None or r.probability is None:
                continue
            rows.append(
                {
                    "model_id": model_id,
                    "question_id": c["question_id"],
                    "qid": c["qid"],
                    "horizon": c["horizon"],
                    "forecast": r.probability,
                    "truth": truth,
                    "brier": compute_brier_score([r.probability], [c["answer"]]),
                    "calibration": (r.probability - truth.p) ** 2,
                    "expected_brier": (r.probability - truth.p) ** 2
                    + truth.p * (1 - truth.p),
                }
            )
    return rows


def score_by_model(
    rows: list[dict], model_names: list[str], score: Score
) -> dict[str, float]:
    """Mean score per model over every scored forecast in `rows`."""
    return {
        m: v
        for m in model_names
        if (v := _mean([r[score.key] for r in rows if r["model_id"] == m])) is not None
    }


def score_by_model_and_horizon(
    rows: list[dict], score: Score
) -> dict[int, dict[str, float]]:
    """Mean score per model, per horizon — the correlation machinery's shape."""
    per_horizon: dict[int, dict[str, list[float]]] = {}
    for r in rows:
        by_model = per_horizon.setdefault(r["horizon"], {})
        by_model.setdefault(r["model_id"], []).append(r[score.key])
    return {
        h: {m: sum(v) / len(v) for m, v in by_model.items()}
        for h, by_model in per_horizon.items()
    }


# ---------------------------------------------------------------------------
# binary_scores.csv

SCORES_CSV_NAME = "binary_scores.csv"
SCORES_CSV_COLUMNS = [
    "model",
    "question_type",
    "horizon",
    "nforecasts",
    "nvalid",
    "brier",
    "expected_brier",
    "calibration",
]

# The pooled row's label in the horizon column.
ALL = "all"


def scores_csv_rows(
    corpus: list[dict],
    responses: BinaryResponses,
    model_names: list[str],
    truths: dict[str, Truth],
) -> list[dict]:
    """The rows of binary_scores.csv, in the order they are written.

    One row per model x question type x horizon, then an "all" horizon row per
    (model, question type). The two question types are never pooled: their
    scores live on different scales (see SECTIONS). A pooled row averages the
    underlying forecasts, not the per-horizon means.

    Per row, nforecasts counts the questions the model was actually prompted
    with — a response on record, parsed or not — and nvalid those whose answer
    parsed; the means are over the latter and nan when there are none.
    """
    scored = {
        (r["model_id"], r["question_id"]): r
        for r in score_forecasts_binary(corpus, responses, model_names, truths)
    }
    horizons = sorted({c["horizon"] for c in corpus})
    horizon_groups = [(years(h), [h]) for h in horizons] + [(ALL, horizons)]

    def nan_mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else math.nan

    rows = []
    for model_id in model_names:
        for prefix, question_type in QUESTION_TYPES.items():
            for horizon_name, horizon_group in horizon_groups:
                asked = [
                    c
                    for c in corpus
                    if c["qid"].startswith(prefix) and c["horizon"] in horizon_group
                ]
                valid = [
                    scored[k]
                    for c in asked
                    if (k := (model_id, c["question_id"])) in scored
                ]
                row = {
                    "model": model_id,
                    "question_type": question_type,
                    "horizon": horizon_name,
                    "nforecasts": sum(
                        1
                        for c in asked
                        if ResponseId(model_id, c["question_id"]) in responses
                    ),
                    "nvalid": len(valid),
                }
                for key in ("brier", "expected_brier", "calibration"):
                    row[key] = nan_mean([r[key] for r in valid])
                rows.append(row)
    return rows


def write_scores_csv(path: Path, rows: list[dict]) -> Path:
    """Write scores_csv_rows' output; nan lands as the literal "nan"."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SCORES_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def plot_base_rates(
    report: MdReport,
    corpus: list[dict],
    truths: dict[str, Truth],
    outdir: Path,
    section: str,  # display name, for titles
    prefix: str,  # qid prefix, for figure filenames
    qids: list[str],
    log: bool,
) -> Path:
    """Questions x windows: ground-truth P(Yes) as color, realized count as text.

    What the scores below are read against — how often each event fired across
    reseeded continuations of the same state, and how many of the section's
    scenarios actually realized it. Both live in one grid because they are the
    same quantity measured two ways; the color carries the probability, which
    rests on a thousand continuations, and the number carries the count, which
    rests on the handful of scenarios the eval actually asked about. The scale
    tops out at the section's own maximum rather than 1 — on the tail questions
    every probability is under 5% and a 0-1 scale would render the grid blank.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    windows = sorted({(c["snapshot_turn"], c["horizon"]) for c in corpus})
    nscenarios = len({c["scenario_id"] for c in corpus})
    yes = Counter(
        (c["qid"], c["snapshot_turn"], c["horizon"]) for c in corpus if c["answer"]
    )
    ps: dict[tuple[str, int, int], list[float]] = {}
    for c in corpus:
        ps.setdefault((c["qid"], c["snapshot_turn"], c["horizon"]), []).append(
            truths[c["question_id"]].p
        )
    ncont = sorted({t.n for t in truths.values()})

    counts = np.array(
        [[yes[(qid, t, h)] for t, h in windows] for qid in qids], dtype=float
    )
    truth = np.array(
        [[_mean(ps.get((qid, t, h), [])) or np.nan for t, h in windows] for qid in qids]
    )

    labels = [window_label(t, h) for t, h in windows]
    outdir.mkdir(parents=True, exist_ok=True)
    # Square cells at a fixed size, so the axes is sized by the grid rather
    # than stretched to the page: a 16-row section is tall and an 11-row one
    # shorter, and both keep the same cell. The figure is only as wide as the
    # grid plus its margins, and the axes is centered in what is left over.
    # Every length here is in inches and scaled together, so `scale` shrinks
    # the whole figure without changing its proportions.
    scale = 0.6
    cell = 0.515 * scale
    label_w, bar_w, ticks_h, title_h = (
        0.75 * scale,
        1.15 * scale,
        1.15 * scale,
        0.85 * scale,
    )
    grid_w, grid_h = cell * len(windows), cell * len(qids)
    width = max(grid_w + label_w + bar_w, 5.6 * scale)
    height = grid_h + ticks_h + title_h
    fig, ax = plt.subplots(figsize=(width, height))
    left = (width - grid_w - bar_w) / 2 / width
    fig.subplots_adjust(
        left=left,
        right=left + grid_w / width,
        bottom=ticks_h / height,
        top=1 - title_h / height,
    )

    vmax = float(np.nanmax(truth))
    if log:
        # A zero has no place on a log ramp; floor it at half the smallest
        # probability a continuation count can express, as the tail
        # calibration scatter does.
        floor = 0.5 / min(t.n for t in truths.values())
        shade = np.where(np.isnan(truth), np.nan, np.maximum(truth, floor))
        norm = matplotlib.colors.LogNorm(vmin=floor, vmax=vmax)
    else:
        floor = 0.0
        shade = truth
        norm = matplotlib.colors.Normalize(vmin=0.0, vmax=vmax)
    im = ax.imshow(shade, cmap="Reds", norm=norm, aspect="auto")
    for i in range(len(qids)):
        for j in range(len(windows)):
            if np.isnan(truth[i, j]):
                continue
            color = "white" if norm(shade[i, j]) > 0.55 else "black"
            ax.text(
                j,
                i,
                f"{counts[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=8 * scale,
                color=color,
            )
    ax.set_xticks(
        range(len(windows)), labels, fontsize=8 * scale, rotation=45, ha="right"
    )
    ax.set_yticks(range(len(qids)), qids, fontsize=8 * scale)
    ax.tick_params(length=0)
    bar = fig.colorbar(
        im,
        ax=ax,
        fraction=0.03,
        pad=0.04,
        label="probability (log scale)" if log else "probability",
    )
    bar.ax.tick_params(labelsize=7 * scale)
    # The label wears the default axis-label size, which does not scale with
    # the rest and would tower over the shrunk grid.
    bar.ax.yaxis.label.set_size(8 * scale)

    # Wrapped to the figure's width, which is set by the grid, not the prose.
    fig.suptitle(
        f"How often each question resolved Yes — {section}\n"
        f"color: ground-truth P(Yes) over {'/'.join(map(str, ncont))}"
        " reseeded continuations\n"
        f"number: how many of {nscenarios} scenarios realized Yes",
        fontsize=8 * scale,
    )

    out = outdir / f"base_rates-{prefix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


def plot_score_heatmap(
    report: MdReport,
    corpus: list[dict],
    rows: list[dict],
    model_names: list[str],
    outdir: Path,
    section: str,  # display name, for titles
    prefix: str,  # qid prefix, for figure filenames
    score: Score,
    qids: list[str],
) -> Path:
    """Models x questions mean score as an annotated heatmap.

    Replaces a models x questions table: at 13+ models by up to 16 questions
    the grid reads better as color than as a wall of numbers. Rows sort
    best-first by the section mean, drawn as its own separated leftmost column
    since it pools what the other columns split. Both scores are unitless, so
    one color scale serves the whole grid; it runs from 0 so a cell's darkness
    reads as absolute error, not error relative to the section's worst.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    scores: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        scores.setdefault((r["model_id"], r["qid"]), []).append(r[score.key])
    means = {k: _mean(v) for k, v in scores.items()}
    overall = score_by_model(rows, model_names, score)
    ordered = sorted(model_names, key=lambda m: (m not in overall, overall.get(m, 0.0)))

    columns = ["mean"] + qids
    data = np.full((len(ordered), len(columns)), np.nan)
    for i, model_id in enumerate(ordered):
        if model_id in overall:
            data[i, 0] = overall[model_id]
        for j, qid in enumerate(qids, start=1):
            value = means.get((model_id, qid))
            if value is not None:
                data[i, j] = value

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(
        figsize=(0.62 * len(columns) + 3.4, 0.42 * len(ordered) + 1.8)
    )
    vmax = float(np.nanmax(data))
    im = ax.imshow(data, cmap="Reds", vmin=0.0, vmax=vmax, aspect="auto")

    for i in range(len(ordered)):
        for j in range(len(columns)):
            value = data[i, j]
            if np.isnan(value):
                continue
            # ".118" rather than "0.118": every value is below 1, so the
            # leading zero is a column-width tax with no information in it.
            text = f"{value:.3f}".removeprefix("0")
            color = "white" if value > 0.55 * vmax else "black"
            ax.text(j, i, text, ha="center", va="center", fontsize=7, color=color)

    ax.set_xticks(range(len(columns)), columns)
    ax.set_yticks(range(len(ordered)), [m.split("/")[-1] for m in ordered])
    ax.tick_params(length=0)
    # The mean column pools what the rest split, so wall it off visually.
    ax.axvline(x=0.5, color="black", lw=1.2)
    ax.set_title(
        f"Mean {score.name} by model and question — {section}"
        " (lower is better)\n"
        "rows sorted best-first; mean pools every scored forecast in the section"
    )
    fig.colorbar(im, ax=ax, label=f"mean {score.name}", fraction=0.03, pad=0.02)
    fig.tight_layout()

    out = outdir / f"{score.key}_heatmap-{prefix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    report.heading(
        f"Mean {score.name} by model and question — {section} (lower is better)"
    )
    report.image(out)

    # A row resting on fewer questions than the section holds means some
    # responses failed to parse; say so rather than let the means look complete.
    counts = {k: len(v) for k, v in scores.items()}
    used = {m: sum(counts.get((m, q), 0) for q in qids) for m in model_names}
    missing = [
        f"{m.split('/')[-1]}: {len(corpus) - used[m]}"
        for m in ordered
        if used[m] < len(corpus)
    ]
    if missing:
        report.text(f"Unparseable forecasts excluded — {', '.join(missing)}")
    return out


def plot_score_bars(
    report: MdReport,
    corpus: list[dict],
    rows: list[dict],
    model_names: list[str],
    outdir: Path,
    section: str,  # display name, for titles
    prefix: str,  # qid prefix, for figure filenames
) -> Path:
    """Mean Brier and mean calibration error per model, as stacked bar panels.

    Replaces the two models x horizons tables. Those split each model's score
    across horizons; this pools it and puts the models side by side, which is
    the comparison the section is actually for — the horizon breakdown lives
    in the by-horizon figures below. Both panels share the x axis, ordered by
    calibration error, so a model's two bars sit in one column and the panels
    can be read against each other: where the Brier order departs from the
    calibration order is a model whose accuracy and whose calibration
    disagree.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    brier, calibration = SCORES
    by_score = {score.key: score_by_model(rows, model_names, score) for score in SCORES}
    # Sorted by calibration error, best first; a model with nothing to average
    # sorts last rather than crashing the compare.
    ordered = sorted(
        model_names,
        key=lambda m: (
            m not in by_score[calibration.key],
            by_score[calibration.key].get(m, 0.0),
        ),
    )

    outdir.mkdir(parents=True, exist_ok=True)
    # 1:4 vertical to horizontal per panel, and wide enough that the model
    # labels do not collide.
    width = max(0.62 * len(ordered) + 2.2, 9.0)
    panel_h = width / 4
    fig, axes = plt.subplots(2, 1, figsize=(width, 2 * panel_h + 1.5), sharex=True)

    for ax, score in zip(axes, SCORES):
        values = [by_score[score.key].get(m) for m in ordered]
        ax.bar(
            range(len(ordered)),
            [v if v is not None else 0.0 for v in values],
            width=1.0,
            color=PLOT_BLUE,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        for i, v in enumerate(values):
            if v is None:
                continue
            ax.text(
                i,
                v,
                f"{v:.3f}".removeprefix("0"),
                ha="center",
                va="bottom",
                fontsize=7,
                zorder=4,
            )
        ax.set_ylabel(f"mean {score.name}")
        ax.set_title(f"Mean {score.name} (lower is better)", fontsize=9)
        ax.grid(axis="y", alpha=0.3, zorder=0)
        ax.margins(x=0.01, y=0.14)
        ax.set_ylim(bottom=0)

    axes[-1].set_xticks(
        range(len(ordered)),
        [m.split("/")[-1] for m in ordered],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    fig.suptitle(
        f"Forecast skill by model — {section}\n"
        f"models ordered by {calibration.name}, best first;"
        f" {len(corpus)} questions",
        fontsize=10,
    )
    fig.tight_layout()

    out = outdir / f"score_bars-{prefix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    report.heading(f"Mean {brier.name} and {calibration.name} by model — {section}")
    report.image(out)
    return out


def plot_scores_by_horizon(
    report: MdReport,
    corpus: list[dict],
    rows: list[dict],
    model_names: list[str],
    outdir: Path,
    section: str,  # display name, for titles
    prefix: str,  # qid prefix, for figure filenames
) -> Path:
    """Scatter both scores against horizon, one series per model, two panels.

    Shows how sharply accuracy decays with distance and which models depart
    from the pack. The mean over models is a thick line in each panel, so it
    reads as the summary rather than as one more model. Both panels share the
    x axis and one legend, ordered by calibration error like the bar figure,
    so a model keeps one color and one legend position across the whole
    section.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = sorted({c["horizon"] for c in corpus})
    by_score = {
        score.key: {
            model_id: {
                h: _mean(
                    [
                        r[score.key]
                        for r in rows
                        if r["model_id"] == model_id and r["horizon"] == h
                    ]
                )
                for h in horizons
            }
            for model_id in model_names
        }
        for score in SCORES
    }

    n_colors = 10 if len(model_names) <= 10 else 20
    palette = plt.get_cmap(f"tab{n_colors}")
    # One color and one legend slot per model across both panels, ordered by
    # calibration error to match the bar figure above.
    calibration = SCORES[1]
    overall = score_by_model(rows, model_names, calibration)
    ordered = sorted(model_names, key=lambda m: (m not in overall, overall.get(m, 0.0)))
    colors = {m: palette(i % n_colors) for i, m in enumerate(model_names)}

    # Models bunch tightly, so spread each one's points across a slice of the
    # gap between horizons — fixed per model, not random, so a model sits in the
    # same place in every regenerated figure.
    gap = min((b - a for a, b in pairwise(horizons)), default=1)
    spread = gap * 0.35
    offsets = {
        model_id: (i / max(len(model_names) - 1, 1) - 0.5) * spread
        for i, model_id in enumerate(model_names)
    }

    outdir.mkdir(parents=True, exist_ok=True)
    # 1:2 height to width per panel. Bands are in inches so the panel aspect
    # is exact, rather than whatever is left after the legend takes its share.
    panel_w, legend_w = 8.0, 2.9
    panel_h = panel_w / 2
    title_h, xlabel_h, gap_h = 0.75, 0.75, 0.5
    width = panel_w + legend_w + 0.85
    height = 2 * panel_h + title_h + xlabel_h + gap_h
    fig, axes = plt.subplots(2, 1, figsize=(width, height), sharex=True)
    fig.subplots_adjust(
        left=0.85 / width,
        right=(0.85 + panel_w) / width,
        top=1 - title_h / height,
        bottom=xlabel_h / height,
        hspace=gap_h / panel_h,
    )

    for ax, score in zip(axes, SCORES):
        by_model = by_score[score.key]
        for model_id in model_names:
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
                color=colors[model_id],
                s=38,
                alpha=0.85,
                zorder=3,
                label=model_id.split("/")[-1],
            )
        # Averaged over the per-model means, so every model counts equally
        # however many of its forecasts parsed.
        mean_points = [
            (h, v)
            for h in horizons
            if (v := _mean([m[h] for m in by_model.values() if m[h] is not None]))
            is not None
        ]
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
        ax.set_ylabel(f"mean {score.name}")
        ax.set_title(f"Mean {score.name} (lower is better)", fontsize=9)
        ax.grid(alpha=0.3, zorder=0)
        ax.margins(x=0.04)
        ax.set_ylim(bottom=0)

    axes[-1].set_xlabel(
        "Horizon in simulated years (model points spread within each tick)"
    )
    axes[-1].set_xticks(horizons, [years(h) for h in horizons])

    # One legend for both panels: the series are the same models, so a legend
    # per panel would be the same box printed twice.
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    legend_order = ["mean over models"] + [m.split("/")[-1] for m in ordered]
    legend_labels = [lbl for lbl in dict.fromkeys(legend_order) if lbl in by_label]
    fig.legend(
        [by_label[lbl] for lbl in legend_labels],
        legend_labels,
        loc="center left",
        bbox_to_anchor=((0.85 + panel_w + 0.15) / width, 0.5),
        fontsize=8,
        framealpha=0.9,
    )
    fig.suptitle(
        f"Forecast skill by horizon — {section}\n"
        f"{len(model_names)} models, {len(corpus)} questions;"
        f" legend ordered by {calibration.name}",
        fontsize=10,
    )

    out = outdir / f"scores_by_horizon-{prefix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.heading(f"Mean score by horizon — {section}")
    report.image(out)
    return out


def plot_eci_vs_score(
    report: MdReport,
    corpus: list[dict],
    rows: list[dict],
    model_names: list[str],
    outdir: Path,
    section: str,  # display name, for titles
    prefix: str,  # qid prefix, for figure filenames
    score: Score,
) -> Path | None:
    """Scatter each model's ECI against its mean score for the section.

    Tests whether forecasting this world tracks general capability. Both scores
    are lower-is-better like nCRPS, so a *negative* correlation is the pro-g
    one. Returns None when too few models carry an ECI score.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    scores = score_by_model(rows, model_names, score)
    points = sorted(
        (eci_of(m), scores[m], m.split("/")[-1])
        for m in model_names
        if m in scores and eci_of(m) is not None
    )
    skipped = sorted(
        m.split("/")[-1] for m in model_names if m in scores and eci_of(m) is None
    )
    if len(points) < 4:
        report.text(
            f"ECI vs mean {score.name} ({section}): only {len(points)} model(s)"
            " have an ECI score; skipping the plot."
        )
        return None

    ecis = [e for e, _, _ in points]
    values = [v for _, v, _ in points]
    rho, p_rho = stats.spearmanr(ecis, values)
    r, p_r = stats.pearsonr(ecis, values)

    direction = "pro-g" if rho < 0 else "anti-g"
    report.heading(f"ECI vs mean {score.name} — {section} (Spearman)")
    lines = [
        f"rho={rho:+.3f}  p={p_rho:.4f} {stars_for(p_rho):<4} ({direction}, n={len(points)})",
        f"Pearson r={r:+.3f}  p={p_r:.4f} {stars_for(p_r)}",
        (
            f"{score.name[0].upper()}{score.name[1:]} is lower-is-better, so rho<0"
            " means the more capable models forecast better (pro-g)."
        ),
    ]
    if skipped:
        lines.append(f"no ECI score, excluded: {', '.join(skipped)}")
    report.text("\n".join(lines))

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.scatter(ecis, values, s=70, color=PLOT_BLUE, zorder=3)

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
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    ax.set_xlabel("ECI (Epoch capability index)")
    ax.set_ylabel(f"Mean {score.name} (lower is better)")
    ax.set_title(
        f"Forecast skill vs. ECI — {section}, {score.name}"
        f"  ({len(points)} models, {len(corpus)} questions)"
    )
    ax.grid(alpha=0.3, zorder=0)
    ax.margins(x=0.12, y=0.1)
    fig.tight_layout()

    fig.canvas.draw()
    place_labels(fig, ax, [n for _, _, n in points], ecis, values)

    out = outdir / f"eci_vs_{score.key}-{prefix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


def plot_predictors_correlation_by_horizon_binary(
    report: MdReport,
    corpus: list[dict],
    rows: list[dict],
    model_names: list[str],
    outdir: Path,
    section: str,  # display name, for titles
    prefix: str,  # qid prefix, for figure filenames
    score: Score,
) -> Path | None:
    """Compare ECI and knowledge-eval score as predictors of forecast skill.

    Both lines are restricted to the models carrying both scores, so their
    coefficients are directly comparable. Returns None when the model set is
    too small or the knowledge eval has no cached answers for these models.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    knowledge = knowledge_predictor(model_names)
    if knowledge is None:
        report.text(
            f"Predictor comparison by horizon ({section}, {score.name}): no cached"
            " knowledge-eval answers for these models; skipping the plot."
        )
        return None

    eci = eci_by_name(model_names)
    shared = set(eci) & set(knowledge)
    series_defs = [
        ("ECI", PLOT_BLUE, eci),
        ("Knowledge score", "#c2432d", knowledge),
    ]

    by_horizon = score_by_model_and_horizon(rows, score)
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
            f"Predictor comparison by horizon ({section}, {score.name}): only"
            f" {len(shared)} model(s) have both an ECI score and knowledge-eval"
            " answers; skipping the plot."
        )
        return None

    report.heading(
        f"Predictors of {score.name} by horizon — {section}"
        f" (Spearman, {len(shared)} shared models)"
    )
    lines = []
    for label, _color, results in series:
        lines.append(f"{label}")
        lines.append(format_horizon_correlations(results, indent="  "))
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
        f"{score.name}, {section}: {len(shared)} models with both"
        f" scores, {len(corpus)} questions",
    )
    ax.set_ylabel(f"Spearman ρ vs. mean {score.name}")
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

    out = outdir / f"predictors_correlation_by_horizon-{score.key}-{prefix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


def calibration_bins(
    points: list[tuple[float, float]], nbins: int, log: bool, floor: float
) -> list[tuple[float, float, float]]:
    """Bin forecasts by their own probability and average the truth in each.

    `points` are (forecast, truth) pairs. Bins are equal-width in the
    forecast, in log space when `log`, since the tail section's forecasts span
    two decades and equal-width linear bins would put nearly all of them in
    the first. Returns one (bin center, mean truth, standard error) per
    non-empty bin, the standard error being sd/sqrt(n) over the truths in the
    bin — so a bin holding one forecast reports an error of zero, which is
    honest about the spread and silent about the uncertainty.
    """
    import numpy as np

    if not points:
        return []
    fs = np.array([max(f, floor) if log else f for f, _ in points])
    ps = np.array([max(p, floor) if log else p for _, p in points])
    xs = np.log10(fs) if log else fs
    edges = np.linspace(xs.min(), xs.max(), nbins + 1)
    # Values equal to the top edge belong to the last bin, not past it.
    idx = np.clip(np.digitize(xs, edges[1:-1]), 0, nbins - 1)

    out = []
    for b in range(nbins):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        center = (edges[b] + edges[b + 1]) / 2
        sem = float(ps[sel].std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        out.append((float(10**center if log else center), float(ps[sel].mean()), sem))
    return out


def plot_calibration(
    report: MdReport,
    corpus: list[dict],
    rows: list[dict],
    model_names: list[str],
    outdir: Path,
    section: str,  # display name, for titles
    prefix: str,  # qid prefix, for figure filenames
    log: bool,
) -> Path:
    """One scatter per model of ground-truth p against forecast f, in three columns.

    The diagonal is perfect calibration; points below it are overconfident
    Yes — the model said more than happened — and points above it
    overconfident No. The tail section is drawn log-log since
    its probabilities span two decades, and there a zero — a question no
    continuation resolved Yes, or a model that answered 0 anyway — is clipped
    to half a continuation's worth so it stays on the page rather than
    vanishing at -inf. Panels sort best-first by mean calibration error.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    calibration = SCORES[1]
    overall = score_by_model(rows, model_names, calibration)
    ordered = sorted(model_names, key=lambda m: overall.get(m, math.inf))
    ncols = 3
    nrows = math.ceil(len(ordered) / ncols)

    # Half a continuation: the smallest nonzero p is 1/n, so 0 lands one
    # "step" below it, distinguishable from a real 1/n and still in frame.
    n_min = min(r["truth"].n for r in rows) if rows else 1
    floor = 0.5 / n_min
    clip = (lambda x: max(x, floor)) if log else (lambda x: x)
    lo, hi = (floor * 0.7, 1.0) if log else (-0.02, 1.02)

    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(9, 3.0 * nrows + 1.1),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for ax in axes.flat[len(ordered) :]:
        ax.set_visible(False)

    for ax, model_id in zip(axes.flat, ordered):
        mine = [r for r in rows if r["model_id"] == model_id]
        clipped = sum(1 for r in mine if r["forecast"] <= 0 or r["truth"].p <= 0)
        ax.plot([lo, hi], [lo, hi], color="#888888", lw=1, ls="--", zorder=1)
        ax.scatter(
            [clip(r["forecast"]) for r in mine],
            [clip(r["truth"].p) for r in mine],
            s=14,
            alpha=0.45,
            color=PLOT_BLUE,
            edgecolors="none",
            zorder=3,
        )
        # The binned average: within each band of forecast probability, where
        # did the truth actually land? A line tracking the diagonal is a
        # calibrated model; one flatter than it is a model whose probabilities
        # move more than reality does.
        bins = calibration_bins(
            [(r["forecast"], r["truth"].p) for r in mine], NCAL_BINS, log, floor
        )
        if bins:
            ax.errorbar(
                [f for f, _, _ in bins],
                [p for _, p, _ in bins],
                yerr=[e for _, _, e in bins],
                color="#c2432d",
                lw=1.6,
                marker="o",
                ms=4,
                capsize=2.5,
                elinewidth=1.0,
                zorder=5,
                label="binned mean",
            )
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3, zorder=0)
        mean = overall.get(model_id)
        note = f"cal. err. {mean:.4f}" if mean is not None else "no forecasts"
        # Three columns leaves little width, so the count and the score share
        # a line and the clipped-zero count gets its own.
        lines = [model_id.split("/")[-1], f"{len(mine)} forecasts, {note}"]
        if log and clipped:
            lines.append(f"{clipped} zero(s) clipped")
        ax.set_title("\n".join(lines), fontsize=8)

    # The bottom visible panel in each column carries the x label: the last
    # row is partly empty whenever the model count is not a multiple of ncols.
    for col in range(ncols):
        column = [axes[r][col] for r in range(nrows) if axes[r][col].get_visible()]
        if column:
            column[-1].set_xlabel("forecast P(Yes)", fontsize=8)
    for row in axes:
        row[0].set_ylabel("ground-truth P(Yes)", fontsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower right",
            fontsize=8,
            framealpha=0.9,
            bbox_to_anchor=(0.99, 0.01),
        )
    scale = "log-log; zeros drawn at half a continuation" if log else "linear"
    fig.suptitle(
        f"Calibration: ground truth vs. forecast — {section} ({scale})\n"
        "dashed diagonal is perfect calibration; panels sorted best-first",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out = outdir / f"calibration_scatter-{prefix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    report.heading(f"Calibration plots — {section}")
    report.text(
        "Each panel scatters the share of reseeded continuations that resolved"
        " Yes against the model's forecast; the dashed diagonal is perfect"
        f" calibration. The red line bins the forecasts into {NCAL_BINS} equal"
        + (" log-width" if log else " width")
        + " bands by forecast probability and plots the mean ground truth in"
        " each, with bars at one standard error."
        + (
            f" Axes are log-log; a zero on either axis is drawn at {floor:g}"
            " (half a continuation) so it stays in frame."
            if log
            else ""
        )
    )
    report.image(out)
    return out


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default=DEFAULT_BINARY_CONFIG_PATH)
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
    data_file = data_path(label)
    outdir = plots_path(label)

    try:
        corpus, responses, models = load_dataset_binary(data_file)
        corpus, responses, models = select_for_config(
            corpus,
            responses,
            models,
            cfg,
            cfg.get_seed(args.seed),
            cities=args.cities,
            disasters=args.disasters,
            models=args.models,
            rerun_hint="scripts/run_eval_binary.py",
            incomplete=args.incomplete,
        )
        truths = load_truths(corpus)
    except (FileNotFoundError, DatasetError) as e:
        sys.exit(f"[error] {e}")

    print("=" * 70)
    print("MICROPOLIS WORLD — binary eval scores")
    print("=" * 70)
    print(f"data:   {data_file}")
    print(f"config: {cfg.path}")
    print(f"label:  {label}")
    print(f"{len(corpus)} questions x {len(models)} models")

    report = MdReport()
    report.text(
        "Scores per forecast f, both lower-is-better:\n"
        + "\n".join(f"- {s.name}: {s.definition}" for s in SCORES)
        + f"\n\n{SCORES_CSV_NAME} carries their means per model, question type"
        " and horizon, plus the expected Brier score (f - p)^2 + p(1 - p):"
        " the calibration error plus the event's own variance."
    )
    written: list[Path] = []
    for prefix, section, description in SECTIONS:
        qids = [q for q in QUESTION_IDS if q.startswith(prefix)]
        section_corpus = [c for c in corpus if c["qid"].startswith(prefix)]
        if not section_corpus:
            continue
        rows = score_forecasts_binary(section_corpus, responses, models, truths)
        # The prefix of every per-score call below, spelled once.
        common = (report, section_corpus, rows, models, outdir, section, prefix)

        report.heading(f"{section} — {description}", level=1)
        if args.plot:
            written.append(
                plot_base_rates(
                    report,
                    section_corpus,
                    truths,
                    outdir,
                    section,
                    prefix,
                    qids,
                    log=(prefix == "B"),
                )
            )

        # The two scores are shown side by side per view rather than in two
        # separate runs of every view: the pair invites comparison — where a
        # model's Brier and its calibration error disagree is the interesting
        # cell — and that reads far better adjacent than a page apart.
        if args.plot:
            for score in SCORES:
                written.append(plot_score_heatmap(*common, score, qids))
        if args.plot:
            written.append(
                plot_score_bars(
                    report, section_corpus, rows, models, outdir, section, prefix
                )
            )
            written.append(plot_scores_by_horizon(*common))
            for score in SCORES:
                written += [
                    p
                    for p in [
                        plot_eci_vs_score(*common, score),
                        plot_predictors_correlation_by_horizon_binary(*common, score),
                    ]
                    if p is not None
                ]

            report.heading(f"{section} — calibration plots", level=1)
            written.append(
                plot_calibration(
                    report,
                    section_corpus,
                    rows,
                    models,
                    outdir,
                    section,
                    prefix,
                    log=(prefix == "B"),
                )
            )

    print()
    for out in written:
        print(f"Wrote {out}")
    csv_path = write_scores_csv(
        label_dir(label) / SCORES_CSV_NAME,
        scores_csv_rows(corpus, responses, models, truths),
    )
    print(f"Wrote {csv_path}")
    out_path = report.write(
        label_dir(label) / "analysis-brier.md", "Binary eval — Brier and calibration"
    )
    print(out_path)


if __name__ == "__main__":
    main()
