#!/usr/bin/env -S uv run python3
"""Score the binary eval: Brier and calibration tables by question and horizon.

Reads data/micropolis/binary/{label}/data.json, written by
scripts/run_eval_binary.py, and the continuation tallies under
data/micropolis/ground_truth/, written by scripts/extract_ground_truth.py.
Prompts no models and runs no simulations.

Every forecast f gets two scores: the Brier score (f - outcome)^2 against the
realized answer, and the calibration error (f - p)^2 against p, the share of
reseeded continuations that resolved Yes. The report is split into two
sections — the mid-range A questions and the tail B questions — each with the
same structure: Yes counts, then for each score a models x questions heatmap,
a models x horizons table, a by-horizon figure and the ECI correlations, and
finally a grid of per-model calibration scatters (f against p; log-log for
the tail section). Everything goes to one Markdown report,
data/micropolis/binary/{label}/analysis-brier.md; --no-plot skips the figures.

Usage:
    scripts/analyze_binary.py                       # configs/binary.json5
    scripts/analyze_binary.py subset.json5
    scripts/analyze_binary.py --no-plot
    scripts/analyze_binary.py --cities kyoto --disasters false
    scripts/analyze_binary.py --models openai/gpt-5 --label myrun
"""

import argparse
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
    print_horizon_table,
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
    ("A", "mid-range questions (target P(Yes) ~ 10-90%)"),
    ("B", "tail questions (target P(Yes) ~ 0.5-5%)"),
]


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


def plots_path(label: str) -> Path:
    return label_dir(label) / "plots"


def score_forecasts_binary(
    corpus: list[dict],
    responses: BinaryResponses,
    model_names: list[str],
    truths: dict[str, Truth],
) -> list[dict]:
    """One row per parsed forecast: its Brier score, its calibration error,
    the forecast and ground truth behind them, and its qid and horizon.

    The binary counterpart of continuous_eval.score_forecasts. Both scores are
    unitless and bounded, so there is no normalized twin.
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
                    "qid": c["qid"],
                    "horizon": c["horizon"],
                    "forecast": r.probability,
                    "truth": truth,
                    "brier": compute_brier_score([r.probability], [c["answer"]]),
                    "calibration": (r.probability - truth.p) ** 2,
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


def print_yes_counts_table(report: MdReport, corpus: list[dict], qids: list[str]) -> None:
    """Per-question Yes counts per window, so the scores below can be read
    against how often each event actually fired."""
    windows = sorted({(c["snapshot_turn"], c["horizon"]) for c in corpus})
    yes = Counter(
        (c["qid"], c["snapshot_turn"], c["horizon"]) for c in corpus if c["answer"]
    )
    nscenarios = len({c["scenario_id"] for c in corpus})
    header = f"  {'':>4} " + "  ".join(f"T{t}+{h}" for t, h in windows)
    lines = [header, "-" * len(header)]
    for qid in qids:
        counts = "  ".join(
            f"{yes[(qid, t, h)]:>{len(f'T{t}+{h}')}}" for t, h in windows
        )
        lines.append(f"  {qid:>4} {counts}")
    report.text(f"Yes counts out of {nscenarios} scenario(s) per window:")
    report.table("\n".join(lines))


def print_ground_truth_table(
    report: MdReport, corpus: list[dict], truths: dict[str, Truth], qids: list[str]
) -> None:
    """Per-question mean ground-truth P(Yes) per window, the calibration
    error's counterpart to the Yes counts."""
    windows = sorted({(c["snapshot_turn"], c["horizon"]) for c in corpus})
    ps: dict[tuple[str, int, int], list[float]] = {}
    for c in corpus:
        ps.setdefault((c["qid"], c["snapshot_turn"], c["horizon"]), []).append(
            truths[c["question_id"]].p
        )
    n = sorted({t.n for t in truths.values()})
    header = f"  {'':>4} " + "  ".join(f"T{t}+{h}" for t, h in windows)
    lines = [header, "-" * len(header)]
    for qid in qids:
        cells = "  ".join(
            f"{_mean(ps.get((qid, t, h), [])) or 0.0:>{len(f'T{t}+{h}')}.3f}"
            for t, h in windows
        )
        lines.append(f"  {qid:>4} {cells}")
    report.text(
        "Ground-truth P(Yes) averaged over scenarios per window, from "
        f"{'/'.join(map(str, n))} reseeded continuations each:"
    )
    report.table("\n".join(lines))


def plot_score_heatmap(
    report: MdReport,
    corpus: list[dict],
    rows: list[dict],
    model_names: list[str],
    outdir: Path,
    section: str,
    qids: list[str],
    score: Score,
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
    ordered = sorted(
        model_names, key=lambda m: (m not in overall, overall.get(m, 0.0))
    )

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
        f"Mean {score.name} by model and question — section {section}"
        " (lower is better)\n"
        "rows sorted best-first; mean pools every scored forecast in the section"
    )
    fig.colorbar(im, ax=ax, label=f"mean {score.name}", fraction=0.03, pad=0.02)
    fig.tight_layout()

    out = outdir / f"{score.key}_heatmap-{section}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    report.heading(
        f"Mean {score.name} by model and question — section {section}"
        " (lower is better)"
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


def print_score_horizon_table(
    report: MdReport,
    corpus: list[dict],
    rows: list[dict],
    model_names: list[str],
    section: str,
    score: Score,
) -> None:
    """Models x horizons of mean score, via the continuous report's table shape."""
    print_horizon_table(
        report,
        [(r["model_id"], r["horizon"], r[score.key]) for r in rows],
        model_names,
        sorted({c["horizon"] for c in corpus}),
        f"Mean {score.name} by model and horizon — section {section}"
        " (lower is better)",
        "horizons are turns past the snapshot\nall* pools every horizon",
        ".3f",
    )


def plot_score_by_horizon(
    report: MdReport,
    corpus: list[dict],
    rows: list[dict],
    model_names: list[str],
    outdir: Path,
    section: str,
    score: Score,
) -> Path:
    """Scatter mean score against horizon, one series per model.

    The horizon table says the same thing, but the shape is immediate here —
    how sharply accuracy decays with distance, and which models depart from the
    pack. The mean over models is drawn as a thick line so it reads as the
    summary rather than as one more model.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizons = sorted({c["horizon"] for c in corpus})
    by_model = {
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
    # Averaged over the per-model means, so every model counts equally however
    # many of its forecasts parsed.
    mean_by_horizon = {
        h: _mean([v[h] for v in by_model.values() if v[h] is not None])
        for h in horizons
    }

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    n_colors = 10 if len(model_names) <= 10 else 20
    palette = plt.get_cmap(f"tab{n_colors}")

    # Legend best-first, so its order is itself a ranking.
    overall = {
        model_id: _mean([v for v in by_model[model_id].values() if v is not None])
        for model_id in model_names
    }
    ordered = sorted(model_names, key=lambda m: (overall[m] is None, overall[m] or 0.0))

    # Models bunch tightly, so spread each one's points across a slice of the
    # gap between horizons — fixed per model, not random, so a model sits in the
    # same place in every regenerated figure.
    gap = min((b - a for a, b in pairwise(horizons)), default=1)
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
    ax.set_ylabel(f"Mean {score.name} (lower is better)")
    ax.set_title(
        f"Mean {score.name} by horizon — section {section}\n"
        f"{len(model_names)} models, {len(corpus)} questions"
    )
    ax.set_xticks(horizons)
    ax.grid(alpha=0.3, zorder=0)
    ax.margins(x=0.04)
    ax.set_ylim(bottom=0)

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

    out = outdir / f"{score.key}_by_horizon-{section}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


def plot_eci_vs_score(
    report: MdReport,
    corpus: list[dict],
    rows: list[dict],
    model_names: list[str],
    outdir: Path,
    section: str,
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
    report.heading(f"ECI vs mean {score.name} — section {section} (Spearman)")
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
        f"Forecast skill vs. ECI — section {section}, {score.name}"
        f"  ({len(points)} models, {len(corpus)} questions)"
    )
    ax.grid(alpha=0.3, zorder=0)
    ax.margins(x=0.12, y=0.1)
    fig.tight_layout()

    fig.canvas.draw()
    place_labels(fig, ax, [n for _, _, n in points], ecis, values)

    out = outdir / f"eci_vs_{score.key}-{section}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


def plot_eci_correlation_by_horizon_binary(
    report: MdReport,
    corpus: list[dict],
    rows: list[dict],
    model_names: list[str],
    outdir: Path,
    section: str,
    score: Score,
) -> Path | None:
    """Plot the ECI x score Spearman correlation against horizon.

    The single scatter pools every horizon into one coefficient; this asks
    whether capability predicts forecast skill more or less strongly as the
    question gets harder. Returns None when too few models carry an ECI score.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    by_horizon = score_by_model_and_horizon(rows, score)
    results = correlate_by_horizon(eci_by_name(model_names), by_horizon, with_ci=True)
    if not results:
        report.text(
            f"ECI x {score.name} by horizon ({section}): too few models with an"
            " ECI score; skipping the plot."
        )
        return None

    report.heading(
        f"ECI x {score.name} correlation by horizon — section {section} (Spearman)"
    )
    report.text(format_horizon_correlations(results))

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_horizon_correlation_axes(
        ax,
        [("ECI", PLOT_BLUE, results)],
        "Does capability predict forecast skill at every horizon?\n"
        f"section {section}, {score.name}: {results[0][3]} models with an ECI"
        f" score, {len(corpus)} questions",
    )
    ax.set_ylabel(f"Spearman ρ of ECI vs. mean {score.name}")
    ax.legend(
        handles=significance_handles(PLOT_BLUE, plt) + band_handles(PLOT_BLUE, plt),
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
    )
    fig.tight_layout()

    out = outdir / f"eci_correlation_by_horizon-{score.key}-{section}.png"
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
    section: str,
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
        f"Predictors of {score.name} by horizon — section {section}"
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
        f"section {section}, {score.name}: {len(shared)} models with both"
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

    out = outdir / f"predictors_correlation_by_horizon-{score.key}-{section}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


def plot_calibration(
    report: MdReport,
    corpus: list[dict],
    rows: list[dict],
    model_names: list[str],
    outdir: Path,
    section: str,
    log: bool,
) -> Path:
    """One scatter per model of forecast f against ground-truth p, in two columns.

    The diagonal is perfect calibration; points above it are overconfident
    Yes, below it overconfident No. The tail section is drawn log-log since
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
    ncols = 2
    nrows = math.ceil(len(ordered) / ncols)

    # Half a continuation: the smallest nonzero p is 1/n, so 0 lands one
    # "step" below it, distinguishable from a real 1/n and still in frame.
    n_min = min(r["truth"].n for r in rows) if rows else 1
    floor = 0.5 / n_min
    clip = (lambda x: max(x, floor)) if log else (lambda x: x)
    lo, hi = (floor * 0.7, 1.0) if log else (-0.02, 1.02)

    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(9, 4.1 * nrows), squeeze=False, sharex=True, sharey=True
    )
    for ax in axes.flat[len(ordered):]:
        ax.set_visible(False)

    for ax, model_id in zip(axes.flat, ordered):
        mine = [r for r in rows if r["model_id"] == model_id]
        clipped = sum(1 for r in mine if r["forecast"] <= 0 or r["truth"].p <= 0)
        ax.plot([lo, hi], [lo, hi], color="#888888", lw=1, ls="--", zorder=1)
        ax.scatter(
            [clip(r["truth"].p) for r in mine],
            [clip(r["forecast"]) for r in mine],
            s=14,
            alpha=0.45,
            color=PLOT_BLUE,
            edgecolors="none",
            zorder=3,
        )
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3, zorder=0)
        mean = overall.get(model_id)
        note = f"mean cal. err. {mean:.4f}" if mean is not None else "no forecasts"
        if log and clipped:
            note += f", {clipped} zero(s) clipped"
        ax.set_title(
            f"{model_id.split('/')[-1]}  ({len(mine)} forecasts)\n{note}", fontsize=9
        )

    for ax in axes[-1]:
        ax.set_xlabel("ground-truth P(Yes) from continuations")
    for row in axes:
        row[0].set_ylabel("forecast P(Yes)")
    scale = "log-log; zeros drawn at half a continuation" if log else "linear"
    fig.suptitle(
        f"Calibration: forecast vs. ground truth — section {section} ({scale})\n"
        "dashed diagonal is perfect calibration; panels sorted best-first",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out = outdir / f"calibration_scatter-{section}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    report.heading(f"Calibration plots — section {section}")
    report.text(
        "Each panel scatters a model's forecasts against the share of reseeded"
        " continuations that resolved Yes; the dashed diagonal is perfect"
        " calibration."
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
    )
    written: list[Path] = []
    for prefix, description in SECTIONS:
        qids = [q for q in QUESTION_IDS if q.startswith(prefix)]
        section_corpus = [c for c in corpus if c["qid"].startswith(prefix)]
        if not section_corpus:
            continue
        rows = score_forecasts_binary(section_corpus, responses, models, truths)

        report.heading(f"Section {prefix} — {description}", level=1)
        print_yes_counts_table(report, section_corpus, qids)
        print_ground_truth_table(report, section_corpus, truths, qids)

        for score in SCORES:
            report.heading(f"Section {prefix} — {score.name}", level=1)
            if args.plot:
                written.append(
                    plot_score_heatmap(
                        report, section_corpus, rows, models, outdir, prefix, qids, score
                    )
                )
            print_score_horizon_table(
                report, section_corpus, rows, models, prefix, score
            )
            if args.plot:
                written.append(
                    plot_score_by_horizon(
                        report, section_corpus, rows, models, outdir, prefix, score
                    )
                )
                written += [
                    p
                    for p in [
                        plot_eci_vs_score(
                            report, section_corpus, rows, models, outdir, prefix, score
                        ),
                        plot_eci_correlation_by_horizon_binary(
                            report, section_corpus, rows, models, outdir, prefix, score
                        ),
                        plot_predictors_correlation_by_horizon_binary(
                            report, section_corpus, rows, models, outdir, prefix, score
                        ),
                    ]
                    if p is not None
                ]

        if args.plot:
            report.heading(f"Section {prefix} — calibration plots", level=1)
            written.append(
                plot_calibration(
                    report,
                    section_corpus,
                    rows,
                    models,
                    outdir,
                    prefix,
                    log=(prefix == "B"),
                )
            )

    print()
    for out in written:
        print(f"Wrote {out}")
    out_path = report.write(
        label_dir(label) / "analysis-brier.md", "Binary eval — Brier and calibration"
    )
    print(out_path)


if __name__ == "__main__":
    main()
