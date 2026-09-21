#!/usr/bin/env -S uv run python3
"""Compare configs — typically the prompt variants — on excess normalized CRPS.

Each config names a gathered continuous dataset. Every parsed forecast is
scored by excess nCRPS as the article defines it: its mean CRPS over the
question's reseeded continuations, minus the floor the continuations' own five
quantiles score, divided by the city's scale for the metric — the metric's mean
over the turns up to the main continuous config's first snapshot, floored, the
same numbers gather_paper_data.py writes to city_metric_scales.csv and
analyze_paper.py divides by. The scale is per city, not per snapshot, so a
cell is comparable across the snapshots the configs take. Zero is a forecast
equal to the replay distribution and lower is better. extract_ground_truth.py
must have covered every config; a question without continuation outcomes is
not scored.

The report, under data/micropolis/continuous/comparisons/{name}/:
  - a summary table, one row per config in command-line order: forecasts
    cached, the share that did not parse, the mean excess nCRPS with one vote
    per model and a 95% bootstrap interval over questions, and Spearman rho
    with ECI with a 95% bootstrap interval over models
  - paired differences: configs asking the same questions are compared with
    the first of them given, over the models they share, with the same
    questions redrawn for both sides of every difference
  - a bar chart of the summary's means
  - a models x configs table of per-model means, rows sorted by ECI
  - a scatter of ECI against per-model mean excess nCRPS, one series per config
  - scores.csv: model_scores.csv with MPScore columns pooled over the configs

Usage:
    scripts/analyze_prompts.py "configs/prompt variants"/prompt-*.json5 --name variants
    scripts/analyze_prompts.py cfgA.json5 cfgB.json5 --intersect-models
"""

import argparse
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from micropolis_world import model_scores
from micropolis_world.config import Config, ConfigError, main_with_config
from micropolis_world.continuous_eval import (
    DatasetError,
    MdReport,
    ResponseId,
    attach_outcomes,
    data_path,
    load_dataset,
    make_normalizer,
    out_dir,
    score_forecasts,
    select_for_config,
)
from micropolis_world.plot_labels import place_labels

sys.path.insert(0, str(Path(__file__).parent))
from analyze_continuous import (
    ALL,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    DEFAULT_CONTINUOUS_CONFIG_PATH,
    EXCESS,
    Correlation,
    _percentile_interval,
    by_model_id,
    correlate,
    eci_by_name,
    eci_of,
    format_band,
    is_forecast,
    resample_indices,
    stars_for,
)
from get_city_scales import SCALES_START_TURN, paper_scales

# The score every table and figure here is built from.
SCORE_KEY = "excess_ncrps"

# The config whose cities and first snapshot define the scales: the main run's.
SCALES_CONFIG_PATH = DEFAULT_CONTINUOUS_CONFIG_PATH


def load_scales() -> tuple[dict[str, dict[str, float]], str]:
    """The article's per-city scales and a line describing them."""
    cfg = Config.load(SCALES_CONFIG_PATH)
    snapshot = cfg.get_int_list("snapshot_turns")[0]
    scales = paper_scales(cfg, cfg.get_seed(None))
    if not scales:
        sys.exit(
            f"[error] no cached sim log for any city of {SCALES_CONFIG_PATH}\n"
            "  run scripts/run_sim.py on it; the scales are read from those logs"
        )
    return scales, (
        f"the metric's mean over turns {SCALES_START_TURN}-{snapshot} of the "
        f"city, floored ({SCALES_CONFIG_PATH.name}), as in analyze_paper.py"
    )


def normalize_rows(
    rows: list[dict], cities: dict[str, str], scales: dict[str, dict[str, float]]
) -> list[dict]:
    """Add SCORE_KEY = excess CRPS over the city's scale; drop rows without excess.

    `cities` maps a question id to its city. A scored row whose city or
    metric has no scale is an error: the tables average what is present, so a
    silent gap would move every number.
    """
    kept = []
    for r in rows:
        if r["excess_crps"] is None:
            continue
        city = cities[r["question_id"]]
        scale = scales.get(city, {}).get(r["metric"])
        if not scale:
            sys.exit(
                f"[error] no scale for {city}/{r['metric']}: the city is not in "
                f"{SCALES_CONFIG_PATH.name}, or has no cached sim log"
            )
        r[SCORE_KEY] = r["excess_crps"] / scale
        kept.append(r)
    return kept


@dataclass
class Scored:
    """One config's scored forecasts and the counts behind them."""

    label: str
    rows: list[dict]  # score_forecasts rows at forecast horizons with an excess score
    model_names: list[str]
    asked: int  # (model, question) pairs with a cached response
    parsed: int  # of those, the ones whose percentiles parsed
    without_outcomes: int  # questions the ground truth does not cover

    @property
    def question_ids(self) -> frozenset[str]:
        return frozenset(r["question_id"] for r in self.rows)


# Display names for the configs, keyed by label, once a prefix every label
# shares has been dropped. Set once in main(); unknown keys fall back to the
# label itself.
SHORT: dict[str, str] = {}


def short(label: str) -> str:
    """What to call `label` in a header, a legend or a tick."""
    return SHORT.get(label, label)


def short_labels(labels: list[str]) -> dict[str, str]:
    """Each label minus a prefix all of them share, cut at a separator.

    'prompt-semantic' and 'prompt-smallbatch' share 'prompt-s' but lose only
    'prompt-'. Left alone when there is one label, nothing is shared, or the
    trim would empty a label.
    """
    if len(labels) < 2:
        return {l: l for l in labels}
    common = ""
    for chars in zip(*labels):
        if len(set(chars)) > 1:
            break
        common += chars[0]
    cut = max(common.rfind(sep) for sep in "-_.")
    if cut < 0:
        return {l: l for l in labels}
    trimmed = {l: l[cut + 1 :] for l in labels}
    if not all(trimmed.values()):
        return {l: l for l in labels}
    return trimmed


def comparison_dir(name: str) -> Path:
    """Where one comparison's report and plots go."""
    return out_dir() / "comparisons" / name


def load_and_score(
    config_path: str,
    seed_override: int | None,
    models: list[str] | None,
    scales: dict[str, dict[str, float]],
    incomplete: bool,
) -> Scored:
    """Load the dataset a config names, narrow it to the config, score it."""
    cfg = Config.load(config_path)
    seed = cfg.get_seed(seed_override)
    label = cfg.get_label(None)
    corpus, responses, model_names = load_dataset(data_path(label))
    corpus, responses, model_names = select_for_config(
        corpus, responses, model_names, cfg, seed, models=models, incomplete=incomplete
    )
    corpus = [c for c in corpus if is_forecast(c["horizon"])]
    # score_forecasts wants a normalizer for its own normalized column, which
    # is not read here; the score is the raw excess over the article's scale.
    norm = make_normalizer("global", corpus)
    without_outcomes = attach_outcomes(corpus)
    cities = {c["question_id"]: c["scenario"]["name"] for c in corpus}
    asked = parsed = 0
    for c in corpus:
        for model_id in model_names:
            r = responses.get(ResponseId(model_id, c["question_id"]))
            if r is None:
                continue
            asked += 1
            parsed += r.percentiles is not None
    rows = normalize_rows(
        score_forecasts(corpus, responses, model_names, norm), cities, scales
    )
    return Scored(label, rows, model_names, asked, parsed, without_outcomes)


# ---------------------------------------------------------------------------
# Means with one vote per model, and their question bootstrap


def score_matrix(rows: list[dict], models: list[str], questions: list[str]):
    """(questions x models) array of scores, nan where a model has none."""
    import numpy as np

    qpos = {q: i for i, q in enumerate(questions)}
    mpos = {m: j for j, m in enumerate(models)}
    scores = np.full((len(questions), len(models)), np.nan)
    for r in rows:
        if r["model_id"] in mpos and r["question_id"] in qpos:
            scores[qpos[r["question_id"]], mpos[r["model_id"]]] = r[SCORE_KEY]
    return scores


def resampled_model_means(scores, draws):
    """Each model's mean over the questions each draw selects.

    `draws` is a (resamples, questions) index array; a question drawn twice
    counts twice. A model's unscored questions cost it those questions only.
    Returns a (resamples, models) array.
    """
    import numpy as np

    present = ~np.isnan(scores)
    nq = scores.shape[0]
    weights = np.zeros(draws.shape, dtype=float)
    for i, draw in enumerate(draws):
        weights[i] = np.bincount(draw, minlength=nq)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (weights @ np.where(present, scores, 0.0)) / (weights @ present)


def per_model_means(rows: list[dict]) -> dict[str, float]:
    """Mean score per model over its scored rows."""
    by_model: dict[str, list[float]] = {}
    for r in rows:
        by_model.setdefault(r["model_id"], []).append(r[SCORE_KEY])
    return {m: statistics.fmean(v) for m, v in by_model.items()}


def config_mean(
    rows: list[dict],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, tuple[float, float] | None, int, int] | None:
    """(mean of the per-model means, 95% CI over questions, n models, n questions).

    One vote per model, so a config is not favored for covering more of one
    model's questions than another's. The interval redraws the questions with
    the models fixed. None without rows.
    """
    import numpy as np

    if not rows:
        return None
    models = sorted({r["model_id"] for r in rows})
    questions = sorted({r["question_id"] for r in rows})
    scores = score_matrix(rows, models, questions)
    mean = float(np.nanmean(scores, axis=0).mean())
    draws = resample_indices(len(questions), resamples, seed)
    means = resampled_model_means(scores, draws)
    with np.errstate(invalid="ignore"):
        pooled = np.nanmean(means, axis=1)
    return mean, _percentile_interval(pooled, resamples), len(models), len(questions)


def paired_difference(
    a: Scored,
    b: Scored,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, tuple[float, float] | None, int, int] | None:
    """Mean of `a` minus mean of `b`, over shared models and questions.

    Both means are one vote per model. The questions are redrawn once per
    resample and applied to both sides, so the interval is on the paired
    difference rather than on two independent means. None when the two
    configs share fewer than two questions or no model.
    """
    import numpy as np

    models = sorted({r["model_id"] for r in a.rows} & {r["model_id"] for r in b.rows})
    questions = sorted(a.question_ids & b.question_ids)
    if not models or len(questions) < 2:
        return None
    sa = score_matrix(a.rows, models, questions)
    sb = score_matrix(b.rows, models, questions)
    delta = float(np.nanmean(sa, axis=0).mean() - np.nanmean(sb, axis=0).mean())
    draws = resample_indices(len(questions), resamples, seed)
    with np.errstate(invalid="ignore"):
        deltas = np.nanmean(resampled_model_means(sa, draws), axis=1) - np.nanmean(
            resampled_model_means(sb, draws), axis=1
        )
    return delta, _percentile_interval(deltas, resamples), len(models), len(questions)


def question_groups(scored: list[Scored]) -> list[list[Scored]]:
    """Configs bucketed by the questions they scored, in command-line order."""
    groups: dict[frozenset[str], list[Scored]] = {}
    for s in scored:
        groups.setdefault(s.question_ids, []).append(s)
    return list(groups.values())


def eci_correlation(s: Scored) -> Correlation | None:
    """ECI against the per-model mean score of one config, both intervals."""
    return correlate(
        "ECI",
        by_model_id(eci_by_name(s.model_names), s.model_names),
        s.rows,
        SCORE_KEY,
        s.model_names,
        ALL,
    )


def intersect_models(scored: list[Scored]) -> list[str]:
    """Narrow every config to the models scored in all of them; returns the dropped."""
    sets = [{r["model_id"] for r in s.rows} for s in scored]
    common = set.intersection(*sets) if sets else set()
    dropped = sorted(set.union(*sets) - common) if sets else []
    for s in scored:
        s.rows = [r for r in s.rows if r["model_id"] in common]
        s.model_names = [m for m in s.model_names if m in common]
    return dropped


# ---------------------------------------------------------------------------
# Report


def format_mean(cell: tuple | None) -> str:
    """ "mean [lo, hi]" to three decimals, or a dash."""
    if cell is None:
        return "-"
    mean, ci = cell[0], cell[1]
    if ci is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def format_rho(c: Correlation | None) -> str:
    """rho with stars and its models interval, or a dash."""
    if c is None:
        return "-"
    return f"{c.rho:+.2f}{stars_for(c.rho_p):<3} {format_band(c.rho_models, 0)}"


def fixed_table(headers: list[str], rows: list[list[str]], left: int = 1) -> str:
    """Fixed-width text table; the first `left` columns are left-aligned."""
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt(cells):
        return "  ".join(
            f"{c:<{w}}" if i < left else f"{c:>{w}}"
            for i, (c, w) in enumerate(zip(cells, widths))
        )

    header = fmt(headers)
    return "\n".join([header, "-" * len(header)] + [fmt(r) for r in rows])


def print_summary_table(
    report: MdReport, scored: list[Scored], scale_note: str
) -> dict[str, tuple]:
    """One row per config; returns the mean cells the bar chart draws."""
    means = {s.label: config_mean(s.rows) for s in scored}
    correlations = {s.label: eci_correlation(s) for s in scored}

    report.heading("Summary by config", level=1)
    report.text(
        f"One row per config, in the order given on the command line. Excess "
        f"nCRPS is {EXCESS.definition}; the scale is {scale_note}. 0 is a "
        "forecast equal to the replay distribution; lower is better.\n\n"
        "mean is the mean of the per-model means — one vote per model — with a "
        f"95% percentile-bootstrap interval ({BOOTSTRAP_RESAMPLES:,} draws) over "
        "questions, the models fixed. #forecasts counts (model, question) pairs "
        "with a cached response; pct invalid is the share of those that did not "
        "parse. ECI rho is Spearman between a model's ECI and its mean, one point "
        "per model, with stars for p<0.05, p<0.01, p<0.001 and a 95% bootstrap "
        "interval over models; rho<0 means the more capable models forecast better."
    )
    rows = []
    for s in scored:
        invalid = f"{100 * (s.asked - s.parsed) / s.asked:.2f}%" if s.asked else "-"
        cell = means[s.label]
        rows.append(
            [
                short(s.label),
                f"{s.asked}",
                invalid,
                f"{cell[2]}" if cell else "-",
                f"{cell[3]}" if cell else "-",
                format_mean(cell),
                format_rho(correlations[s.label]),
            ]
        )
    report.table(
        fixed_table(
            [
                "Config",
                "#forecasts",
                "pct invalid",
                "models",
                "questions",
                "mean excess nCRPS",
                "ECI rho",
            ],
            rows,
        )
    )
    return {label: cell for label, cell in means.items() if cell is not None}


def print_paired_table(report: MdReport, scored: list[Scored]) -> None:
    """Paired differences within each group of configs asking the same questions."""
    groups = [g for g in question_groups(scored) if len(g) > 1]
    report.heading("Paired differences", level=1)
    if not groups:
        report.text("no two configs ask the same questions; nothing to pair.")
        return
    report.text(
        "Configs that ask the same questions, each against the first of them "
        "given on the command line: the difference of the two means (one vote "
        "per model, over the models both scored), negative when the config beats "
        "its reference, with a 95% bootstrap interval that redraws the same "
        "questions for both sides."
    )
    rows = []
    for group in groups:
        ref = group[0]
        for s in group[1:]:
            d = paired_difference(s, ref)
            if d is None:
                rows.append([short(s.label), short(ref.label), "-", "-", "-"])
                continue
            delta, ci, n_models, n_questions = d
            band = f" [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""
            rows.append(
                [
                    short(s.label),
                    short(ref.label),
                    f"{n_models}",
                    f"{n_questions}",
                    f"{delta:+.3f}{band}",
                ]
            )
    report.table(
        fixed_table(["Config", "vs", "models", "questions", "difference"], rows, left=2)
    )


def print_models_table(report: MdReport, scored: list[Scored]) -> None:
    """Models x configs of per-model mean excess nCRPS, rows by ECI."""
    cells = {s.label: per_model_means(s.rows) for s in scored}
    models = sorted(
        {m for c in cells.values() for m in c},
        key=lambda m: (eci_of(m) is None, -(eci_of(m) or 0), m),
    )
    report.heading("Mean excess nCRPS by model and config", level=1)
    report.text(
        "Each cell is the model's mean over the config's scored questions; rows "
        "are sorted by ECI, models without one last. The scatter below puts a "
        "95% bootstrap interval over questions on these means."
    )
    rows = [
        [m.split("/")[-1], model_scores.format_eci(eci_of(m))]
        + [f"{cells[s.label][m]:.3f}" if m in cells[s.label] else "-" for s in scored]
        for m in models
    ]
    report.table(fixed_table(["Model", "ECI"] + [short(s.label) for s in scored], rows))


def plot_mean_bars(
    report: MdReport, means: dict[str, tuple], outdir: Path
) -> Path | None:
    """Bar chart of each config's mean excess nCRPS with its interval."""
    if not means:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    labels = list(means)
    values = [means[l][0] for l in labels]
    lower = [v - (means[l][1][0] if means[l][1] else v) for l, v in zip(labels, values)]
    upper = [(means[l][1][1] if means[l][1] else v) - v for l, v in zip(labels, values)]
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * len(labels) + 2), 5))
    palette = plt.get_cmap("tab10")
    xs = range(len(labels))
    ax.bar(xs, values, color=[palette(i % 10) for i in xs], alpha=0.85, zorder=3)
    ax.errorbar(
        list(xs),
        values,
        yerr=[lower, upper],
        fmt="none",
        ecolor="black",
        elinewidth=1.2,
        capsize=4,
        zorder=4,
    )
    ax.axhline(0, color="black", lw=0.8, zorder=2)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([short(l) for l in labels], rotation=20, ha="right")
    ax.set_ylabel("Mean excess nCRPS")
    ax.set_title(
        "Mean excess nCRPS by config — one vote per model\n"
        "95% bootstrap intervals over questions; 0 is the replay distribution"
    )
    ax.grid(alpha=0.3, axis="y", zorder=0)
    fig.tight_layout()
    out = outdir / "excess_by_config.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


def plot_eci_scatter(
    report: MdReport, scored: list[Scored], outdir: Path
) -> Path | None:
    """ECI against per-model mean excess nCRPS, one series and fit per config."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    series: dict[str, Correlation] = {}
    for s in scored:
        c = eci_correlation(s)
        if c is not None:
            series[s.label] = c
    report.heading("ECI vs mean excess nCRPS by config", level=1)
    if not series:
        report.text("no config has 4+ models with an ECI score; skipping the plot.")
        return None
    lines = ["rho<0 means the more capable models forecast better (pro-g)."]
    outdir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    palette = plt.get_cmap("tab10")
    predictor = None
    for i, (label, c) in enumerate(series.items()):
        color = palette(i % 10)
        predictor = by_model_id(eci_by_name(list(c.scores)), list(c.scores))
        models = sorted(c.scores, key=lambda m: predictor[m])
        ecis = [predictor[m] for m in models]
        values = [c.scores[m] for m in models]
        lower = [
            v - c.score_questions.get(m, (v, v))[0] for m, v in zip(models, values)
        ]
        upper = [
            c.score_questions.get(m, (v, v))[1] - v for m, v in zip(models, values)
        ]
        ax.errorbar(
            ecis,
            values,
            yerr=[lower, upper],
            fmt="o",
            ms=6,
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=3,
            alpha=0.85,
            zorder=3,
            label=short(label),
        )
        fit = stats.linregress(ecis, values)
        span = [min(ecis), max(ecis)]
        ax.plot(
            span,
            [fit.intercept + fit.slope * x for x in span],
            color=color,
            lw=1.5,
            alpha=0.7,
            zorder=2,
        )
        lines.append(
            f"  {short(label)}: rho={c.rho:+.3f} p={c.rho_p:.4f} {stars_for(c.rho_p):<4}"
            f" CI models {format_band(c.rho_models, 0)} questions {format_band(c.rho_questions, 0)}"
            f" (n={c.n_models} models, {c.n_questions} questions)"
        )
    report.text("\n".join(lines))

    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.set_xlabel("ECI (Epoch capability index)")
    ax.set_ylabel("Mean excess nCRPS (lower is better)")
    ax.set_title(
        f"Excess nCRPS against ECI, by config — {len(series)} configs\n"
        "bars are 95% bootstrap intervals over questions"
    )
    ax.grid(alpha=0.3, zorder=0)
    ax.margins(x=0.12, y=0.1)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    # One name per model, at the mean of its per-config means.
    by_model: dict[str, list[float]] = {}
    for c in series.values():
        for m, v in c.scores.items():
            by_model.setdefault(m, []).append(v)
    names = [m.split("/")[-1] for m in by_model]
    xs = [predictor[m] if m in predictor else eci_of(m) for m in by_model]
    ys = [statistics.fmean(v) for v in by_model.values()]
    fig.canvas.draw()
    place_labels(fig, ax, names, xs, ys)

    out = outdir / "eci_vs_excess_by_config.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    report.image(out)
    return out


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "configs", nargs="+", help="JSON5 config files, each naming a gathered dataset"
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
        help="Keep only the models scored in every config",
    )
    ap.add_argument(
        "--name",
        default=None,
        help=f"Name of the output directory under {out_dir() / 'comparisons'} "
        "(default: the config labels joined with '+')",
    )
    ap.add_argument(
        "--no-plot", dest="plot", action="store_false", help="Skip writing the figures"
    )
    ap.add_argument(
        "--incomplete",
        action="store_true",
        help="Score only the questions gathered for every selected model, "
        "instead of failing when the dataset is missing forecasts",
    )
    args = ap.parse_args()

    scales, scale_note = load_scales()
    scored: list[Scored] = []
    for path in args.configs:
        try:
            s = load_and_score(path, args.seed, args.models, scales, args.incomplete)
        except (FileNotFoundError, DatasetError, ConfigError, NotImplementedError) as e:
            sys.exit(
                f"[error] {path}: {e}\n  the excess measure needs the ground truth; "
                "run extract_ground_truth.py on the config first"
            )
        if any(t.label == s.label for t in scored):
            sys.exit(f"[error] two configs share the label {s.label!r}")
        scored.append(s)

    dropped_models = intersect_models(scored) if args.intersect_models else []
    if args.intersect_models and not any(s.rows for s in scored):
        sys.exit("[error] --intersect-models: no model has scored rows in every config")

    SHORT.update(short_labels([s.label for s in scored]))
    name = args.name or "+".join(s.label for s in scored)
    outdir = comparison_dir(name)

    print("=" * 70)
    print("MICROPOLIS WORLD — excess nCRPS across configs")
    print("=" * 70)
    print(f"configs: {', '.join(s.label for s in scored)}")
    print(f"scale:   {scale_note}")
    print(f"output:  {outdir}")
    if dropped_models:
        print(
            f"--intersect-models dropped {len(dropped_models)} model(s): {', '.join(dropped_models)}"
        )

    report = MdReport()
    report.text(
        f"score = excess nCRPS per forecast: {EXCESS.definition}, where the "
        f"scale is {scale_note}. The excess CRPS is analyze_continuous.py's; the "
        "scale is the article's, so a cell here matches analyze_paper.py rather "
        "than that script's normalized columns.\n\n"
        f"configs compared: {', '.join(s.label for s in scored)}"
    )
    trimmed = [s.label for s in scored if SHORT[s.label] != s.label]
    if trimmed:
        shared = trimmed[0][: len(trimmed[0]) - len(SHORT[trimmed[0]])]
        report.text(
            f"every config's label starts with {shared!r}; the tables and figures "
            f"drop it, so the column '{SHORT[trimmed[0]]}' is the config '{trimmed[0]}'."
        )
    if args.intersect_models:
        common = sorted({r["model_id"] for s in scored for r in s.rows})
        note = f"--intersect-models: restricted to the {len(common)} model(s) scored in all configs"
        if dropped_models:
            note += f"; excluded: {', '.join(dropped_models)}"
        report.text(note)
    for s in scored:
        notes = []
        if s.asked - s.parsed:
            notes.append(f"{s.asked - s.parsed} of {s.asked} responses did not parse")
        if s.without_outcomes:
            notes.append(
                f"{s.without_outcomes} question(s) have no continuation outcomes"
            )
        if notes:
            report.text(f"**{s.label}**: " + "; ".join(notes))

    with_rows = [s for s in scored if s.rows]
    empty = [s.label for s in scored if not s.rows]
    written = []
    if with_rows:
        means = print_summary_table(report, with_rows, scale_note)
        if empty:
            report.text("omitted, no scored rows: " + ", ".join(empty))
        if args.plot:
            fig = plot_mean_bars(report, means, outdir)
            if fig is not None:
                written.append(fig)
        print_paired_table(report, with_rows)
        print_models_table(report, with_rows)
        if args.plot:
            fig = plot_eci_scatter(report, with_rows, outdir)
            if fig is not None:
                written.append(fig)

    # model_scores.csv plus each model's mean over every compared config, with
    # its question interval; written under --no-plot too.
    all_rows = [r for s in scored for r in s.rows]
    if all_rows:
        models = sorted({r["model_id"] for r in all_rows})
        stats_by_model = {}
        for m in models:
            cell = config_mean([r for r in all_rows if r["model_id"] == m])
            if cell is not None:
                mean, ci, _, _ = cell
                stats_by_model[m] = (mean, ci[0] if ci else None, ci[1] if ci else None)
        outdir.mkdir(parents=True, exist_ok=True)
        csv_path = outdir / "scores.csv"
        model_scores.write_scores_csv(csv_path, stats_by_model)
        written.append(csv_path)

    if written:
        print()
        for out in written:
            print(f"Wrote {out}")
    out_path = report.write(
        outdir / "prompts.md", "Continuous eval — excess nCRPS across configs"
    )
    print(out_path)


if __name__ == "__main__":
    main()
