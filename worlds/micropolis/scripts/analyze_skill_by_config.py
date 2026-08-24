#!/usr/bin/env -S uv run python3
"""Compare skill vs the naive baseline across several configs at once.

analyze_baseline_skill.py scores one config's dataset against a naive baseline:

    skill = CRPS_model / CRPS_baseline     per question
            geometric mean over questions

This script runs that same scoring over a *set* of configs — typically the
prompt variants, each of which names its own dataset label — and reports them
against each other:

  - a models x configs table of skill scores, with each model's ECI beside
    its name and the rows sorted by it
  - a scatter of ECI against skill with 95% confidence intervals on the
    geometric means as error bars, one color and one fitted line per config

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
from pathlib import Path

from micropolis_world.config import Config, ConfigError, main_with_config
from micropolis_world.plot_labels import place_labels
from micropolis_world.single_city import (
    OUT_DIR,
    DatasetError,
    MdReport,
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
from analyze_single_city import eci_of, stars_for

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
            f"ECI vs skill ({SPLITS[split][0]}): no config has 4+ models with "
            "an ECI score; skipping the plot."
        )
        return None

    report.heading(f"ECI vs skill vs baseline by config — {SPLITS[split][0]}")
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
        f"Skill vs baseline against ECI, by config — {SPLITS[split][0]}\n"
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
) -> tuple[str, list[dict], dict[str, int]]:
    """One config's (label, scored rows, dropped tally).

    Loads the dataset the config's label names and narrows it to what the
    config asks for, exactly as the single-config script does, so a config that
    scores there scores identically here.
    """
    cfg = Config.load(config_path)
    seed = cfg.get_seed(seed_override)
    label = cfg.get_label(None)
    corpus, responses, model_names = load_dataset(data_path(label))
    corpus, responses, model_names = select_for_config(
        corpus, responses, model_names, cfg, seed, models=models
    )
    rows, dropped = score_skill(corpus, responses, model_names, seed, kind)
    return label, rows, dropped


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
    for path in args.configs:
        try:
            label, rows, dropped = load_and_score(
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

    dropped_models: list[str] = []
    if args.intersect_models:
        per_config, dropped_models = intersect_models(per_config)
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

    written = []
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
