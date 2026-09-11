#!/usr/bin/env -S uv run python3
"""Score the binary eval: Brier and excess Brier tables by question and horizon.

Reads data/micropolis/binary/{label}/data.json, written by
scripts/run_eval_binary.py, and the continuation tallies under
data/micropolis/ground_truth/, written by scripts/extract_ground_truth.py.
Prompts no models and runs no simulations.

Every forecast f gets two scores: the Brier score (f - outcome)^2 against the
realized answer, and the excess Brier (f - p)^2 against p, the share of
reseeded continuations that resolved Yes. The report is split into two
sections by that p: "Mid-range probabilities" (p >= 5%) and "Tail
probabilities" (p < 5%). The split is per question instance, so one qid can be
mid-range in one city and tail in another. Each section has the same
structure: a histogram of its questions per ground-truth p, the two scores
paired within each view (a per-model bar panel, a by-horizon figure), a table
of correlations between the per-model scores and the capability predictors
(ECI, knowledge eval) with bootstrap intervals over models and over
questions, the ECI scatter and predictor comparison, and finally a grid of
per-model calibration scatters. The tail section draws its histogram and its
scatter axes on a log scale, since its probabilities span two decades.
Everything goes to one Markdown report,
data/micropolis/binary/{label}/analysis-brier.md; --no-plot skips the figures.
Beside it goes binary_scores.csv: one row per model x question type (mid-range
or tail, by the same rule) x horizon in years plus an "all" horizon row, with
the prompted and parsed counts and the mean Brier, expected Brier and excess
Brier. And
results.csv: one row per model x question, with the model's forecast (nan
when its response did not parse), the realized answer, the ground-truth p and
the cached response file and line the forecast was read from.

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

# Imported rather than reimplemented so the table formatting, the correlation
# machinery and the CI conventions are provably the same ones the continuous
# report uses.
sys.path.insert(0, str(Path(__file__).parent))
from analyze_continuous import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
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
    model_style,
    resample_indices,
    significance_handles,
    stars_for,
)

DEFAULT_BINARY_CONFIG_PATH = CONFIG_DIR / "binary.json5"

# The report's two halves, split on the ground-truth probability of each
# question instance rather than on its qid: a question is tail when fewer than
# TAIL_THRESHOLD of the reseeded continuations resolved Yes. Tail scores live
# on a different scale — an always-No forecast is already near-perfect there —
# and averaging them with the mid-range ones would let the tail dilute the
# signal. Everything below is computed per section, and the section key is
# also the question_type in binary_scores.csv and the figure filename prefix.
TAIL_THRESHOLD = 0.05
MID_RANGE, TAIL = "mid-range", "tail"
SECTIONS = [
    (MID_RANGE, "Mid-range probabilities", "ground-truth P(Yes) ≥ 5%"),
    (TAIL, "Tail probabilities", "ground-truth P(Yes) < 5%"),
]


def section_of(question: dict, truths: dict[str, Truth]) -> str:
    """The section a corpus question falls in, by its ground-truth P(Yes)."""
    return TAIL if truths[question["question_id"]].p < TAIL_THRESHOLD else MID_RANGE


@dataclass(frozen=True)
class Score:
    """One way of scoring a forecast; every table and figure is made per score."""

    key: str  # the row field holding the score, and the figure file prefix
    name: str  # as it appears in titles: "Mean {name} by ..."
    definition: str


SCORES = [
    Score("brier", "Brier", "(f - outcome)^2 against the realized answer"),
    Score(
        "excess_brier",
        "excess Brier",
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


def plots_path(label: str) -> Path:
    return label_dir(label) / "plots"


def score_forecasts_binary(
    corpus: list[dict],
    responses: BinaryResponses,
    model_names: list[str],
    truths: dict[str, Truth],
) -> list[dict]:
    """One row per parsed forecast: its Brier score, its excess Brier,
    its expected Brier score, the forecast and ground truth behind them, and
    its qid and horizon.

    The binary counterpart of continuous_eval.score_forecasts. The scores are
    unitless and bounded, so there is no normalized twin. The expected Brier
    is what the Brier score averages to over the continuations' outcomes,
    (f - p)^2 + p(1 - p): the excess Brier plus the irreducible variance
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
                    "excess_brier": (r.probability - truth.p) ** 2,
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
    "excess_brier",
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

    One row per model x question type (a SECTIONS key, by section_of) x
    horizon, then an "all" horizon row per (model, question type). The two
    question types are never pooled: their scores live on different scales
    (see SECTIONS). A pooled row averages the underlying forecasts, not the
    per-horizon means.

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

    sections = {c["question_id"]: section_of(c, truths) for c in corpus}
    rows = []
    for model_id in model_names:
        for question_type, _name, _description in SECTIONS:
            for horizon_name, horizon_group in horizon_groups:
                asked = [
                    c
                    for c in corpus
                    if sections[c["question_id"]] == question_type
                    and c["horizon"] in horizon_group
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
                for key in ("brier", "expected_brier", "excess_brier"):
                    row[key] = nan_mean([r[key] for r in valid])
                rows.append(row)
    return rows


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> Path:
    """Write `rows` under `columns`; nan lands as the literal "nan"."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# results.csv

RESULTS_CSV_NAME = "results.csv"
RESULTS_CSV_COLUMNS = [
    "model",
    "seed",
    "city",
    "disasters",
    "snapshot_turn",
    "horizon",
    "question_id",
    "forecast",
    "answer",
    "real_prob",
    "response_file",
    "response_line",
]


def results_csv_rows(
    corpus: list[dict],
    responses: BinaryResponses,
    model_names: list[str],
    truths: dict[str, Truth],
) -> list[dict]:
    """The rows of results.csv: one per model x question, in corpus order.

    question_id is the short qid (A1, B3, ...); horizon is in turns, like
    snapshot_turn; disasters is 1/0. forecast is nan when the model's response
    is missing or did not parse. answer is the main run's outcome as 1/0 and
    real_prob the share of reseeded continuations that resolved Yes.
    response_file is the cached response, relative to the binary cache root,
    and response_line the 1-based line of it the forecast was read from; nan
    when unknown (never gathered, unparsed, or a dataset written before the
    gather recorded them).
    """

    def or_nan(value):
        return math.nan if value is None else value

    rows = []
    for model_id in model_names:
        for c in corpus:
            r = responses.get(ResponseId(model_id, c["question_id"]))
            rows.append(
                {
                    "model": model_id,
                    "seed": c["scenario"]["seed"],
                    "city": c["scenario"]["name"],
                    "disasters": int(c["scenario"]["disasters"]),
                    "snapshot_turn": c["snapshot_turn"],
                    "horizon": c["horizon"],
                    "question_id": c["qid"],
                    "forecast": math.nan if r is None else or_nan(r.probability),
                    "answer": int(c["answer"]),
                    "real_prob": truths[c["question_id"]].p,
                    "response_file": math.nan if r is None else or_nan(r.source),
                    "response_line": math.nan if r is None else or_nan(r.line),
                }
            )
    return rows


def plot_truth_histogram(
    report: MdReport,
    corpus: list[dict],
    truths: dict[str, Truth],
    outdir: Path,
    section: str,  # display name, for titles
    prefix: str,  # section key, for figure filenames
    log: bool,
) -> Path:
    """How many of the section's questions sit at each ground-truth P(Yes).

    What the scores below are read against. Every model is asked every
    question, so the count per bin is the count per model. The tail section
    bins in log p, since its probabilities span two decades and linear bins
    would put nearly all of them in the first; a question no continuation
    resolved Yes has no log, so those sit in their own bar, labeled 0, one bin
    below the smallest probability a continuation count can express.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    ps = np.array([truths[c["question_id"]].p for c in corpus])
    ncont = min(t.n for t in truths.values())

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.7))
    # The axes is 4:1 whatever the figure's margins come out as.
    ax.set_box_aspect(0.25)
    if log:
        lo = 1 / ncont  # the smallest nonzero p a continuation count can express
        edges = np.logspace(np.log10(lo), np.log10(TAIL_THRESHOLD), 13)
        ratio = edges[1] / edges[0]
        ax.hist(ps[ps > 0], bins=edges, color=PLOT_BLUE, edgecolor="white", zorder=3)
        ax.set_xscale("log")
        ticks = [t for t in (0.001, 0.002, 0.005, 0.01, 0.02, 0.05) if lo <= t]
        labels = [f"{t:g}" for t in ticks]
        zeros = int((ps == 0).sum())
        if zeros:
            # One bin wide, one bin's gap below the first real bin.
            left, right = lo / ratio**2, lo / ratio
            ax.bar(
                left,
                zeros,
                width=right - left,
                align="edge",
                color=PLOT_BLUE,
                edgecolor="white",
                hatch="///",
                zorder=3,
            )
            ticks.insert(0, math.sqrt(left * right))
            labels.insert(0, "0")
        ax.set_xticks(ticks, labels)
        ax.set_xlim(lo / ratio**2.3, TAIL_THRESHOLD * 1.05)
        ax.set_xlabel("ground-truth P(Yes), log scale (hatched bar: p = 0)")
    else:
        edges = np.linspace(TAIL_THRESHOLD, 1.0, 20)
        ax.hist(ps, bins=edges, color=PLOT_BLUE, edgecolor="white", zorder=3)
        ax.set_xlim(TAIL_THRESHOLD, 1.0)
        ax.set_xlabel("ground-truth P(Yes)")
    ax.set_ylabel("questions (per model)")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_title(
        f"Questions per ground-truth P(Yes) — {section}\n"
        f"{len(corpus)} questions, each asked to every model;"
        f" p over {ncont}+ reseeded continuations",
        fontsize=10,
    )
    fig.tight_layout()

    out = outdir / f"truth_histogram-{prefix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


# ---------------------------------------------------------------------------
# Correlations with capability, with intervals from two resampling units

# A correlation over fewer models than this is not reported, matching
# correlate_by_horizon's default.
MIN_MODELS = 4


@dataclass(frozen=True)
class Correlation:
    """One predictor against the per-model mean score of one question slice.

    Each coefficient carries two 95% percentile-bootstrap intervals. `models`
    resamples the models — the unit that would have to generalize, and the
    convention the continuous report uses — and so says how far the
    coefficient could move with a different model set. `questions` resamples
    the questions with the models fixed, moving every model's forecast on a
    question together, and so says how stable the coefficient is to which
    questions were asked. Either is None when too few resamples had a defined
    coefficient.
    """

    predictor: str
    horizon: str  # a years label, or ALL for the pooled slice
    n_models: int
    n_questions: int
    rho: float
    rho_p: float
    rho_models: tuple[float, float] | None
    rho_questions: tuple[float, float] | None
    r: float
    r_p: float
    r_models: tuple[float, float] | None
    r_questions: tuple[float, float] | None


def _correlations_by_row(x, y):
    """Spearman and Pearson of each row of `x` against the same row of `y`.

    Both (resamples, models) arrays. nan where a row has no spread — a
    resample that drew one model several times over, or a constant score.
    """
    import numpy as np
    from scipy import stats

    def pearson(a, b):
        ac = a - a.mean(axis=1, keepdims=True)
        bc = b - b.mean(axis=1, keepdims=True)
        denominator = np.sqrt((ac**2).sum(axis=1) * (bc**2).sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(
                denominator > 0, (ac * bc).sum(axis=1) / denominator, np.nan
            )

    return (
        pearson(stats.rankdata(x, axis=1), stats.rankdata(y, axis=1)),
        pearson(x, y),
    )


def _percentile_interval(values, resamples: int) -> tuple[float, float] | None:
    """The 2.5–97.5 percentile band, or None when most resamples were undefined."""
    import numpy as np

    values = values[~np.isnan(values)]
    if len(values) < resamples // 2:
        return None
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def correlate(
    predictor_name: str,
    predictor: dict[str, float],
    rows: list[dict],
    score: Score,
    model_names: list[str],
    horizon_name: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> Correlation | None:
    """Correlate `predictor` (keyed by model id) with the mean score per model.

    The point estimates and p-values are scipy's on the per-model means over
    every parsed forecast in `rows`. The model bootstrap resamples those means
    with analyze_continuous's draws, so its Spearman interval is the one
    bootstrap_rho_ci would give. The question bootstrap redraws the question
    set and recomputes each model's mean over its parsed forecasts among the
    drawn questions, so an unparsed forecast costs a model a question rather
    than costing every model that question. Returns None when fewer than
    MIN_MODELS models carry both a predictor value and a forecast, or when a
    variable has no spread.
    """
    import numpy as np
    from scipy import stats

    questions = sorted({r["question_id"] for r in rows})
    models = [m for m in model_names if m in predictor]
    qpos = {q: i for i, q in enumerate(questions)}
    mpos = {m: j for j, m in enumerate(models)}
    scores = np.full((len(questions), len(models)), np.nan)
    for r in rows:
        if r["model_id"] in mpos:
            scores[qpos[r["question_id"]], mpos[r["model_id"]]] = r[score.key]
    present = ~np.isnan(scores)
    keep = present.any(axis=0)
    models = [m for m, k in zip(models, keep) if k]
    scores, present = scores[:, keep], present[:, keep]
    if len(models) < MIN_MODELS:
        return None
    x = np.array([predictor[m] for m in models])
    y = np.nanmean(scores, axis=0)
    if len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rho, rho_p = stats.spearmanr(x, y)
    r, r_p = stats.pearsonr(x, y)

    idx = resample_indices(len(models), resamples, seed)
    rho_m, r_m = _correlations_by_row(x[idx], y[idx])

    # Each resample as a count per question, so the resampled means are one
    # matrix product rather than a (resamples x questions x models) array.
    nq = len(questions)
    draws = np.random.default_rng(seed).integers(0, nq, (resamples, nq))
    weights = np.zeros((resamples, nq))
    for i, draw in enumerate(draws):
        weights[i] = np.bincount(draw, minlength=nq)
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (weights @ np.where(present, scores, 0.0)) / (weights @ present)
    rho_q, r_q = _correlations_by_row(np.broadcast_to(x, means.shape), means)

    return Correlation(
        predictor=predictor_name,
        horizon=horizon_name,
        n_models=len(models),
        n_questions=nq,
        rho=float(rho),
        rho_p=float(rho_p),
        rho_models=_percentile_interval(rho_m, resamples),
        rho_questions=_percentile_interval(rho_q, resamples),
        r=float(r),
        r_p=float(r_p),
        r_models=_percentile_interval(r_m, resamples),
        r_questions=_percentile_interval(r_q, resamples),
    )


def capability_predictors(model_names: list[str]) -> list[tuple[str, dict[str, float]]]:
    """(name, predictor keyed by model id) for ECI and, when cached, the knowledge eval.

    Both analyze_continuous helpers key on the bare model name; this rekeys
    them on the ids the rows carry.
    """
    bare = {m: m.split("/", 1)[1] for m in model_names}
    predictors = [("ECI", eci_by_name(model_names))]
    knowledge = knowledge_predictor(model_names)
    if knowledge is not None:
        predictors.append(("Knowledge", knowledge))
    return [
        (name, {m: p[bare[m]] for m in model_names if bare[m] in p})
        for name, p in predictors
    ]


def section_correlations(
    rows: list[dict], model_names: list[str], score: Score
) -> list[Correlation]:
    """Every predictor against the pooled slice and against each horizon."""
    horizons = sorted({r["horizon"] for r in rows})
    slices = [(ALL, rows)] + [
        (years(h), [r for r in rows if r["horizon"] == h]) for h in horizons
    ]
    out = []
    for name, predictor in capability_predictors(model_names):
        for horizon_name, slice_rows in slices:
            c = correlate(name, predictor, slice_rows, score, model_names, horizon_name)
            if c is not None:
                out.append(c)
    return out


def format_correlation_table(correlations: list[Correlation]) -> str:
    """Fixed-width rows of the coefficients and both intervals per coefficient."""

    def band(ci):
        return f"[{ci[0]:+.2f}, {ci[1]:+.2f}]" if ci else "—".center(14)

    header = (
        f"{'predictor':<10} {'horizon':<7} {'models':>6} {'questions':>9}  "
        f"{'ρ':>5} {'p':>6} {'CI models':>14} {'CI questions':>14}  "
        f"{'r':>5} {'p':>6} {'CI models':>14} {'CI questions':>14}"
    )
    lines = [header]
    for c in correlations:
        lines.append(
            f"{c.predictor:<10} {c.horizon:<7} {c.n_models:>6} {c.n_questions:>9}  "
            f"{c.rho:+.2f} {c.rho_p:6.3f} {band(c.rho_models):>14} {band(c.rho_questions):>14}  "
            f"{c.r:+.2f} {c.r_p:6.3f} {band(c.r_models):>14} {band(c.r_questions):>14}"
        )
    return "\n".join(lines)


def report_correlations(
    report: MdReport, rows: list[dict], model_names: list[str], section: str
) -> None:
    """The correlation table per score, ahead of the figures that draw them."""
    report.heading(f"Correlations with capability — {section}")
    report.text(
        "Spearman ρ and Pearson r between each predictor and the mean score per"
        " model, pooled and per horizon; both scores are lower-is-better, so a"
        " negative coefficient means the more capable models forecast better."
        f" Each coefficient has two 95% percentile-bootstrap intervals"
        f" ({BOOTSTRAP_RESAMPLES:,} draws): 'models' resamples the models and asks"
        " how far the coefficient could move with a different model set;"
        " 'questions' resamples the questions with the models fixed and asks"
        " how stable it is to which questions were asked. Questions from one"
        " report share a prompt, so the latter is somewhat optimistic."
    )
    for score in SCORES:
        correlations = section_correlations(rows, model_names, score)
        if not correlations:
            report.text(f"{score.name}: too few models with a predictor to correlate.")
            continue
        report.text(f"{score.name[0].upper()}{score.name[1:]}")
        report.table(format_correlation_table(correlations))


def plot_score_bars(
    report: MdReport,
    corpus: list[dict],
    rows: list[dict],
    model_names: list[str],
    outdir: Path,
    section: str,  # display name, for titles
    prefix: str,  # qid prefix, for figure filenames
) -> Path:
    """Mean Brier and mean excess Brier per model, as stacked bar panels.

    Replaces the two models x horizons tables. Those split each model's score
    across horizons; this pools it and puts the models side by side, which is
    the comparison the section is actually for — the horizon breakdown lives
    in the by-horizon figures below. Both panels share the x axis, ordered by
    excess Brier, so a model's two bars sit in one column and the panels can be
    read against each other: where the Brier order departs from the excess
    Brier order is a model whose accuracy and whose calibration disagree.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    brier, excess = SCORES
    by_score = {score.key: score_by_model(rows, model_names, score) for score in SCORES}
    # Sorted by excess Brier, best first; a model with nothing to average
    # sorts last rather than crashing the compare.
    ordered = sorted(
        model_names,
        key=lambda m: (
            m not in by_score[excess.key],
            by_score[excess.key].get(m, 0.0),
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
        f"models ordered by {excess.name}, best first;"
        f" {len(corpus)} questions",
        fontsize=10,
    )
    fig.tight_layout()

    out = outdir / f"score_bars-{prefix}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)

    report.heading(f"Mean {brier.name} and {excess.name} by model — {section}")
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
    x axis and one legend, ordered by excess Brier like the bar figure,
    so a model keeps one legend position across the whole section. Its color
    and marker come from model_style, keyed on its index in the config's
    order, so they are the ones the ECI figures and the continuous report use.
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

    # One legend slot per model across both panels, ordered by excess Brier
    # to match the bar figure above.
    excess = SCORES[1]
    overall = score_by_model(rows, model_names, excess)
    ordered = sorted(model_names, key=lambda m: (m not in overall, overall.get(m, 0.0)))

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
        f" legend ordered by {excess.name}",
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
    one. Returns None when too few models carry an ECI score. Each model keeps
    the color and marker the by-horizon figure gave it, and the legend beside
    the axes names them, ordered best-first.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    scores = score_by_model(rows, model_names, score)
    # Carries each model's index in the config's order, so its color and marker
    # here are the ones the by-horizon figure gave it.
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
            f"ECI vs mean {score.name} ({section}): only {len(points)} model(s)"
            " have an ECI score; skipping the plot."
        )
        return None

    ecis = [e for e, _, _, _ in points]
    values = [v for _, v, _, _ in points]
    rho, p_rho = stats.spearmanr(ecis, values)
    r, p_r = stats.pearsonr(ecis, values)

    direction = "pro-g" if rho < 0 else "anti-g"
    report.heading(f"ECI vs mean {score.name} — {section} (Spearman)")
    lines = [
        f"ρ={rho:+.3f}  p={p_rho:.4f} {stars_for(p_rho):<4} ({direction}, n={len(points)})",
        f"Pearson r={r:+.3f}  p={p_r:.4f} {stars_for(p_r)}",
        (
            f"{score.name[0].upper()}{score.name[1:]} is lower-is-better, so ρ<0"
            " means the more capable models forecast better (pro-g)."
        ),
    ]
    if skipped:
        lines.append(f"no ECI score, excluded: {', '.join(skipped)}")
    report.text("\n".join(lines))

    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))

    # One scatter call per model: the legend replaces the labels that used to
    # sit on the points, and it needs a handle per model to do that. Plotted
    # best-first so the legend doubles as a ranking.
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
    ax.set_ylabel(f"Mean {score.name} (lower is better)")
    ax.set_title(
        f"Forecast skill vs. ECI — {section}, {score.name}"
        f"  ({len(points)} models, {len(corpus)} questions)"
    )
    ax.grid(alpha=0.3, zorder=0)
    ax.margins(x=0.12, y=0.1)

    # Beside the axes, as on the by-horizon figure: the legend is as tall as
    # the model list, and over the points it would cover the scatter it
    # explains. The fit line goes on top, since it is the figure's summary and
    # not one more model.
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
    vanishing at -inf. Panels sort best-first by mean excess Brier.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    excess = SCORES[1]
    overall = score_by_model(rows, model_names, excess)
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
        note = f"excess Brier {mean:.4f}" if mean is not None else "no forecasts"
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
        " the excess Brier plus the event's own variance."
    )
    written: list[Path] = []
    for prefix, section, description in SECTIONS:
        section_corpus = [c for c in corpus if section_of(c, truths) == prefix]
        if not section_corpus:
            continue
        rows = score_forecasts_binary(section_corpus, responses, models, truths)
        # The prefix of every per-score call below, spelled once.
        common = (report, section_corpus, rows, models, outdir, section, prefix)
        log = prefix == TAIL

        report.heading(f"{section} — {description}", level=1)
        if args.plot:
            written.append(
                plot_truth_histogram(
                    report, section_corpus, truths, outdir, section, prefix, log
                )
            )

        # The two scores are shown side by side per view rather than in two
        # separate runs of every view: the pair invites comparison — where a
        # model's Brier and its excess Brier disagree is the interesting
        # cell — and that reads far better adjacent than a page apart.
        if args.plot:
            written.append(
                plot_score_bars(
                    report, section_corpus, rows, models, outdir, section, prefix
                )
            )
            written.append(plot_scores_by_horizon(*common))
        # The table stands whether or not the figures are drawn: it is the
        # report's statement of the correlations, the figures illustrate it.
        report_correlations(report, rows, models, section)
        if args.plot:
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
                    log,
                )
            )

    print()
    for out in written:
        print(f"Wrote {out}")
    for name, columns, csv_rows in [
        (SCORES_CSV_NAME, SCORES_CSV_COLUMNS, scores_csv_rows),
        (RESULTS_CSV_NAME, RESULTS_CSV_COLUMNS, results_csv_rows),
    ]:
        csv_path = write_csv(
            label_dir(label) / name,
            columns,
            csv_rows(corpus, responses, models, truths),
        )
        print(f"Wrote {csv_path}")
    out_path = report.write(label_dir(label) / "analysis-brier.md", "Binary eval")
    print(out_path)


if __name__ == "__main__":
    main()
