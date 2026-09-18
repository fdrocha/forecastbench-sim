#!/usr/bin/env -S uv run python3
"""Analyze the gathered paper data: the article's figures, as PDF for LaTeX.

The reporting half of the paper pipeline. Reads only what
scripts/gather_paper_data.py wrote — binary_forecasts.csv,
continuous_forecasts.csv, city_metric_scales.csv and its copy of
model_scores.csv, which the package's own parser is pointed at so the ECI
plotted is the one the gather run captured. No dataset, no ground truth, no
config, and nothing outside that directory: rerunning this to nudge a legend
costs a second and cannot change a number.

The continuous CRPS arrives unnormalized and is divided here, per row, by the
scale its own city and metric carry in city_metric_scales.csv — each metric's
mean over that city's turns up to the first snapshot, floored. One city's
population moving by a thousand is not the same event as another's, so the
division has to happen before the rows are averaged together, not after.

Writes to data/micropolis/paper/figures/extra/:

- eci_vs_excess_brier-mid-range.pdf
- eci_vs_excess_brier-tail.pdf
- eci_vs_excess_bits-tail.pdf        (FreeCiv's tail score)
- eci_vs_excess_ncrps-city.pdf       (the continuous eval)

"extra" is the figures the article does not currently place; --no-extra draws
only the rest, which today is none of them, so the flag is a way to refresh the
numbers without spending the drawing time. The numbers are written either way.

And to data/micropolis/paper/:

- micropolis-macros.tex              \\MPD* macros for the article's prose

The three headline correlations — ECI against the mid-range excess Brier, the
tail excess bits and the continuous excess nCRPS — are printed to stdout and
defined as LaTeX macros, each with its p-value, its model count, its question
count and both bootstrap intervals. They are not recomputed here: the figures'
own correlate() call is intercepted, so a macro and its figure cannot disagree.
Under --no-extra no figure is drawn, and the same correlations are computed
directly instead.

Each is the report's own ECI scatter — the same points, fit line, Spearman ρ
and Pearson r, and 95% bootstrap intervals, from analyze_binary.py's and
analyze_continuous.py's plot functions — restyled for an article: the title
dropped, since the caption does that job; the per-model legend moved from
beside the axes to below them in four columns, since two dozen models in one
column is wider than the scatter it explains; and the whole sized for a
\\textwidth of about 6.5in with Type 42 fonts.

Model order fixes each model's color and marker, and is taken from the CSV's
own order of first appearance, so a model keeps one identity across the
figures without a config being read. Models are named by model id, the gather
step having dropped this world's ":suffix", so nothing here or in the figures
carries a ":loeff".

Usage:
    scripts/analyze_paper.py
    scripts/analyze_paper.py --no-extra
    scripts/analyze_paper.py --datadir /tmp/paper-data --outdir /tmp/figures
"""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from micropolis_world import model_scores
from micropolis_world.continuous_eval import MdReport

sys.path.insert(0, str(Path(__file__).parent))
import analyze_binary
from analyze_binary import (
    EXCESS_BITS,
    MID_RANGE,
    SCORES,
    TAIL,
    Score,
    capability_predictors,
    plot_eci_vs_score,
)
from analyze_continuous import ALL, EXCESS, correlate, format_band
from gather_paper_data import (
    BINARY_CSV_NAME,
    CONTINUOUS_CSV_NAME,
    MODEL_SCORES_CSV_NAME,
    OUT_DIR,
    SCALES_CSV_NAME,
)

FIGURES_DIR = OUT_DIR / "figures"

# The figures the article does not currently place. They are still the paper's
# own figures, drawn from the paper's own data — kept apart only so --no-extra
# can skip them while the numbers below are refreshed.
EXTRA_SUBDIR = "extra"

MACROS_NAME = "micropolis-macros.tex"

# The prefix every macro carries, so a \\MPD in the article's source is
# unambiguously a number this script wrote and not one typed by hand.
MACRO_PREFIX = "MPD"

# Names the normalization in the continuous figure's filename and axis note,
# the way the reports' modes name theirs: here the denominator is per city and
# metric rather than per metric alone.
NORM_MODE = "city"

# A \textwidth figure in a single-column article. The report's 10x6.5 would be
# scaled down by \includegraphics and take the fonts with it. This is the axes'
# share; the legend below adds its own rows, and bbox_inches="tight" grows the
# box to fit them.
FIG_WIDTH, FIG_HEIGHT = 7.0, 4.3

# Two dozen models in one column beside the axes takes more width than the
# scatter it explains; below the axes the scatter gets the full figure width and
# the legend costs height instead, which a \textwidth figure has to spare.
LEGEND_NCOLS = 4

# How the reports name each section, for the axis labels the scatters keep.
SECTION_NAMES = {MID_RANGE: "Mid-range probabilities", TAIL: "Tail probabilities"}

# The binary scatters, as (section, Score). The tail carries the excess bits as
# well: at p below 5% an always-No forecast has a near-zero excess Brier and
# separates nothing, so the ratio f/p is what tells the models apart there.
EXCESS_BRIER = next(s for s in SCORES if s.key == "excess_brier")
BINARY_FIGURES = [
    (MID_RANGE, EXCESS_BRIER),
    (TAIL, EXCESS_BRIER),
    (TAIL, EXCESS_BITS),
]

# The continuous figure is drawn by analyze_binary's scatter rather than
# analyze_continuous's: the two draw the same axes, but the binary one takes
# rows already scored, which is what a CSV holds, while
# plot_eci_vs_normalized() rescores a corpus against a Normalizer and so needs
# the dataset and the ground truth this script deliberately does not read.
# The Score names the CSV column and the axis; the wording is EXCESS's, so the
# figure says what the continuous report's says.
EXCESS_NCRPS = Score("excess_ncrps", EXCESS.name, EXCESS.definition)
# What the continuous scatter's section slot carries: it is one eval-wide
# figure, not a section of one, so this names the eval instead.
CONTINUOUS_SECTION = "continuous eval"


@dataclass
class Headline:
    r"""One correlation the article quotes, by the name its macros carry.

    `macro` is the stem: \MPDRho{stem}, \MPDP{stem}, \MPDNModels{stem} and so
    on. `what` is the one-line gloss the .tex comments with and stdout prints,
    since "Binary" alone does not say which of the two binary sections it is or
    which score it scored them with.
    """

    macro: str
    what: str
    figure: str  # the figure whose correlate() call produces it


# The three the article quotes. The mid-range slice is named "Binary" and the
# tail one "Tail" because that is how the article's prose refers to them; the
# gloss carries the precision the names drop. The fourth figure — the tail
# excess Brier — is drawn but not quoted: at p below 5% an always-No forecast
# has a near-zero excess Brier, so the excess bits is the tail's headline.
HEADLINES = [
    Headline(
        "Binary",
        "ECI vs excess Brier, mid-range probabilities",
        "eci_vs_excess_brier-mid-range",
    ),
    Headline(
        "Tail",
        "ECI vs excess bits, tail probabilities",
        "eci_vs_excess_bits-tail",
    ),
    Headline(
        "Continuous",
        "ECI vs per-city normalized excess CRPS, continuous eval",
        "eci_vs_excess_ncrps-city",
    ),
]


def use_copied_model_scores(datadir: Path) -> Path:
    """Point the package's score reader at the gather run's copy.

    Every ECI on these figures is reached through model_scores.eci_of, down
    inside the reports' plot functions, so redirecting the module's path is
    what makes the whole run read the paper's directory instead of the repo's
    datafiles/ — no plot function has to learn where the numbers came from.
    The parsed view is cached, so the cache is dropped in case something has
    already read it.
    """
    path = datadir / MODEL_SCORES_CSV_NAME
    if not path.exists():
        sys.exit(
            f"[error] {path} not found\n"
            "  run scripts/gather_paper_data.py first; it copies model_scores.csv"
            " into the paper's directory"
        )
    model_scores.SCORES_PATH = path
    model_scores.load_scores.cache_clear()
    return path


def read_rows(path: Path) -> list[dict]:
    """A gathered CSV as the scored rows the plot functions expect.

    The score columns are floats and the rest stay strings; `model` becomes
    `model_id`, the key every scoring helper in the reports reads. Nothing
    else is reconstructed: the plot functions want model_id, question_id and
    the score column, and the horizon only to slice on.
    """
    if not path.exists():
        sys.exit(
            f"[error] {path} not found\n"
            "  run scripts/gather_paper_data.py first; it writes the CSVs this"
            " script draws from"
        )
    numeric = {
        "forecast",
        "real_prob",
        "brier",
        "excess_brier",
        "excess_bits",
        "crps",
        "excess_crps",
    }
    with path.open(newline="") as f:
        rows = []
        for raw in csv.DictReader(f):
            row = {
                k: (float(v) if k in numeric and v != "" else v) for k, v in raw.items()
            }
            row["model_id"] = row.pop("model")
            rows.append(row)
    if not rows:
        sys.exit(f"[error] {path} holds no rows")
    return rows


def read_scales(path: Path) -> dict[str, dict[str, float]]:
    """city_metric_scales.csv as {city: {metric: scale}}.

    The metric columns are the engine's own field names, the same ones the
    forecast rows carry, so the join is by name and nothing here has to know
    which metrics the paper covers.
    """
    if not path.exists():
        sys.exit(
            f"[error] {path} not found\n"
            "  run scripts/gather_paper_data.py first; it writes the per-city"
            " scales this script normalizes with"
        )
    with path.open(newline="") as f:
        scales = {
            row["city"]: {k: float(v) for k, v in row.items() if k != "city"}
            for row in csv.DictReader(f)
        }
    if not scales:
        sys.exit(f"[error] {path} holds no rows")
    return scales


def normalize(rows: list[dict], scales: dict[str, dict[str, float]]) -> list[dict]:
    """Divide each row's CRPS by its city's scale for that metric.

    Adds "ncrps" and "excess_ncrps" and returns the rows the paper can plot.
    A missing city or metric is an error: the figures average over whatever is
    present, so a silent gap would move every number without saying so. A row
    whose excess is empty — a question the gather run found no continuations
    for — is dropped instead, since the figure it would join averages the
    excess and cannot carry a blank.
    """
    kept = []
    for r in rows:
        city, metric = r["city"], r["metric"]
        scale = scales.get(city, {}).get(metric)
        if not scale:
            sys.exit(
                f"[error] no scale for {city}/{metric} in {SCALES_CSV_NAME}\n"
                "  rerun scripts/gather_paper_data.py so the scales cover the"
                " same run as the forecasts"
            )
        r["ncrps"] = r["crps"] / scale
        if r["excess_crps"] == "":
            continue
        r["excess_ncrps"] = r["excess_crps"] / scale
        kept.append(r)
    if len(kept) != len(rows):
        print(f"[warn] dropped {len(rows) - len(kept)} rows with no excess CRPS")
    return kept


def models_in_order(rows: list[dict]) -> list[str]:
    """The models, in the CSV's order of first appearance.

    Their position is what model_style keys a color and marker on, so taking it
    from the file rather than from a config keeps this script off the configs
    while still giving a model one identity across the figures.
    """
    return list(dict.fromkeys(r["model_id"] for r in rows))


def shorten_ylabels(fig) -> None:
    """Drop the parenthetical from any y label too long for the paper's axes.

    The continuous scatter's — "Mean excess normalized CRPS (CRPS/scale, lower
    is better)" — is longer than the shortened axes are tall, so it runs off
    the figure and into the legend. The ratio belongs in the caption, and
    every score in the paper is lower-is-better.
    """
    for ax in fig.axes:
        label = ax.get_ylabel()
        if len(label) > 40 and "(" in label:
            ax.set_ylabel(label[: label.index("(")].strip())


def relegend(fig) -> None:
    """Move a figure's legend under its axes, in LEGEND_NCOLS columns.

    The handles and labels come off the existing legend, so the order the plot
    function chose — fit line and CI entry first, then the models best-first —
    is kept.
    """
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        handles = legend.legend_handles
        labels = [t.get_text() for t in legend.get_texts()]
        legend.remove()
        ax.legend(
            handles,
            labels,
            loc="upper center",
            # Clear of the x-axis label, which sits just under the axes.
            bbox_to_anchor=(0.5, -0.22),
            ncol=LEGEND_NCOLS,
            fontsize=6,
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.0,
            borderaxespad=0.0,
        )


def macro_band(ci: tuple[float, float] | None) -> str:
    r"""A bootstrap interval as self-contained math: $[-0.88,\,-0.49]$.

    Math-mode so the article can drop the macro into prose without wrapping
    it, and a thin space after the comma because a bare one sets too tight
    beside a minus sign. Two decimals, as the reports' own bands use; an
    absent interval becomes a dash rather than a number that is not there.
    """
    if ci is None:
        return "---"
    return f"$[{ci[0]:+.2f},\\,{ci[1]:+.2f}]$"


def macro_p(p: float) -> str:
    r"""A p-value for prose: "<0.001" below that, else three decimals.

    The threshold is the smallest the reports' own stars distinguish, and
    printing 0.000 would claim a precision the bootstrap does not have.

    The "<" goes through \ensuremath, not a bare "$<$": the article writes
    these inside math as often as beside it, and a "$<$" dropped into
    $p\,\MPDPBinary$ closes the math and leaves a bare "<", which OT1 sets as
    an inverted exclamation mark. \ensuremath is right either way.
    """
    return r"\ensuremath{<}0.001" if p < 0.001 else f"{p:.3f}"


def macro_lines(h: Headline, c) -> list[str]:
    r"""The \newcommand lines for one headline correlation.

    Six macros: the coefficient, its two intervals, its p-value and the two
    counts the interval widths depend on. The counts are macros rather than
    prose so a rerun that adds a model or a city cannot leave a stale n
    behind in the article.
    """
    pre = MACRO_PREFIX
    return [
        f"% {h.what}",
        f"\\newcommand{{\\{pre}Rho{h.macro}}}{{{c.rho:+.3f}}}",
        f"\\newcommand{{\\{pre}Rho{h.macro}CIModels}}{{{macro_band(c.rho_models)}}}",
        (
            f"\\newcommand{{\\{pre}Rho{h.macro}CIQuestions}}"
            f"{{{macro_band(c.rho_questions)}}}"
        ),
        f"\\newcommand{{\\{pre}P{h.macro}}}{{{macro_p(c.rho_p)}}}",
        f"\\newcommand{{\\{pre}NModels{h.macro}}}{{{c.n_models}}}",
        f"\\newcommand{{\\{pre}NQuestions{h.macro}}}{{{c.n_questions:,}}}",
        "",
    ]


def write_macros(path: Path, found: dict[str, object]) -> Path:
    r"""Write micropolis-macros.tex: every quoted number as a \newcommand.

    A headline whose correlation was not computed is skipped rather than
    written as a placeholder: \MPDRhoTail expanding to a dash in the article
    would read as a result, while an undefined macro fails the LaTeX run and
    says which number is missing.
    """
    lines = [
        "% Generated by worlds/micropolis/scripts/analyze_paper.py -- do not edit.",
        "% Every macro is a number from the Micropolis world's two forecasting",
        "% evals. Each Rho is a Spearman correlation between a model's ECI and",
        "% its mean score, so a negative value is the pro-g direction. The two",
        "% CIs are 95% percentile-bootstrap intervals: CIModels resamples the",
        "% models, CIQuestions resamples the questions with the models fixed.",
        "",
    ]
    missing = []
    for h in HEADLINES:
        c = found.get(h.figure)
        if c is None:
            missing.append(h.macro)
            continue
        lines += macro_lines(h, c)
    if missing:
        print(f"[warn] no correlation for {', '.join(missing)}; macros not defined")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def print_headlines(found: dict[str, object]) -> None:
    """The quoted correlations, as the reports print them."""
    print("Spearman ECI vs mean score, all horizons pooled")
    print("-" * 70)
    for h in HEADLINES:
        c = found.get(h.figure)
        print(f"{h.what}  [\\{MACRO_PREFIX}Rho{h.macro}]")
        if c is None:
            print("  not computed: too few models with an ECI, or no spread\n")
            continue
        direction = "pro-g" if c.rho < 0 else "anti-g"
        print(
            f"  ρ={c.rho:+.3f}  p={c.rho_p:.4f}  ({direction},"
            f" n={c.n_models} models, {c.n_questions:,} questions)"
        )
        print(
            f"  95% CI  models    {format_band(c.rho_models, 0)}\n"
            f"          questions {format_band(c.rho_questions, 0)}\n"
        )


class PaperFigures:
    """Draws the reports' ECI scatters as paper PDFs.

    The plot functions are the reports' own, so they take an MdReport and save
    a PNG named for the report's conventions. This wraps each call: it sets the
    paper's rc params, then intercepts the figure on its way to disk to drop
    the title, move the legend and resize, and writes a PDF under the paper's
    name. The restyling is applied after the fact rather than by threading a
    flag per difference through the plot functions: the point of importing them
    is that the paper draws the same axes the reports do, and a parameter per
    stylistic difference would let the two drift.
    """

    def __init__(self, outdir: Path):
        self.outdir = outdir
        self.written: list[Path] = []
        # figure name -> the Correlation its own plot call computed.
        self.correlations: dict[str, object] = {}

    def draw(self, name: str, plot) -> Path | None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        report = MdReport()
        # plot_eci_vs_score correlates the same per-model means it scatters,
        # and analyze_binary binds correlate into its own namespace, so
        # swapping it there hands this the very object behind the figure's fit
        # line. Recomputing it here would give the same number today and
        # could drift from the figure tomorrow.
        captured = []
        original_correlate = analyze_binary.correlate

        def capturing(*a, **kw):
            c = original_correlate(*a, **kw)
            captured.append(c)
            return c

        with plt.rc_context(
            {
                "figure.figsize": (FIG_WIDTH, FIG_HEIGHT),
                # Type 42 (TrueType) rather than Type 3, which many publishers
                # reject.
                "pdf.fonttype": 42,
                "font.size": 9,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
            }
        ):
            # The plot functions save and close their own figure, so the
            # restyling has to happen before that: patching savefig is the one
            # hook that runs with the figure still open.
            original = plt.Figure.savefig

            def savefig(fig, path, *a, **kw):
                fig.suptitle("")
                for ax in fig.axes:
                    ax.set_title("")
                relegend(fig)
                shorten_ylabels(fig)
                # The plot functions pass figsize=(10, 6.5) to subplots
                # explicitly, so figure.figsize never reaches them; the resize
                # has to happen here, before tight_layout, so the labels are
                # laid out at the final size rather than scaled into it.
                fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
                fig.tight_layout()
                out = self.outdir / f"{name}.pdf"
                out.parent.mkdir(parents=True, exist_ok=True)
                # The caller's kwargs (dpi) are dropped: a PDF is vector.
                return original(fig, out, format="pdf", bbox_inches="tight")

            plt.Figure.savefig = savefig
            analyze_binary.correlate = capturing
            try:
                # The plot functions mkdir their outdir and name their own PNG;
                # the patched savefig ignores that name, so this only has to be
                # a real directory.
                drawn = plot(report, self.outdir)
            finally:
                plt.Figure.savefig = original
                analyze_binary.correlate = original_correlate

        # The scatter correlates once. A figure that declined still correlated
        # first, so the number survives a skipped plot.
        if captured and captured[0] is not None:
            self.correlations[name] = captured[0]

        if drawn is None:
            # Declined — too few models with an ECI score, or no spread to
            # correlate. The reason is in the report text, which is the whole
            # of this throwaway report.
            why = report.render(self.outdir).strip().splitlines()
            print(f"[skipped] {name}: {why[-1] if why else 'no figure drawn'}")
            return None
        out = self.outdir / f"{name}.pdf"
        self.written.append(out)
        return out


def figure_specs(binary: list[dict], continuous: list[dict]) -> list[dict]:
    """Every figure the paper draws, as (name, rows, models, score, plot).

    One description per figure, so the drawing loop and the --no-extra path
    that only wants the numbers work from the same definition of each slice
    rather than each carving its own.
    """
    specs = []
    binary_models = models_in_order(binary)
    for section_key, score in BINARY_FIGURES:
        rows = [r for r in binary if r["section"] == section_key]
        if not rows:
            print(f"[skipped] eci_vs_{score.key}-{section_key}: no such rows")
            continue
        # The corpus argument is only read for the question count in the title,
        # which the restyling drops; one entry per question keeps that count
        # honest for anything that does look at it.
        corpus = [{"question_id": q} for q in {r["question_id"] for r in rows}]
        specs.append(
            {
                "name": f"eci_vs_{score.key}-{section_key}",
                "rows": rows,
                "models": binary_models,
                "score": score,
                "plot": (
                    lambda report, outdir, c=corpus, r=rows, s=score, k=section_key: (
                        plot_eci_vs_score(
                            report, c, r, binary_models, outdir, SECTION_NAMES[k], k, s
                        )
                    )
                ),
            }
        )

    continuous_models = models_in_order(continuous)
    continuous_corpus = [
        {"question_id": q} for q in {r["question_id"] for r in continuous}
    ]
    specs.append(
        {
            "name": f"eci_vs_excess_ncrps-{NORM_MODE}",
            "rows": continuous,
            "models": continuous_models,
            "score": EXCESS_NCRPS,
            "plot": lambda report, outdir: plot_eci_vs_score(
                report,
                continuous_corpus,
                continuous,
                continuous_models,
                outdir,
                CONTINUOUS_SECTION,
                NORM_MODE,
                EXCESS_NCRPS,
            ),
        }
    )
    return specs


def correlations_without_drawing(specs: list[dict]) -> dict[str, object]:
    """The headline correlations for a run that draws no figure.

    Under --no-extra there is no plot call to intercept, so the same
    correlate() the scatter would have made is called here, on the same rows
    with the same predictor and the same pooled horizon. Only the quoted
    slices are computed: the rest cost a bootstrap each and nothing reads them.
    """
    wanted = {h.figure for h in HEADLINES}
    found = {}
    for spec in specs:
        if spec["name"] not in wanted:
            continue
        c = correlate(
            "ECI",
            capability_predictors(spec["models"])[0][1],
            spec["rows"],
            spec["score"].key,
            spec["models"],
            ALL,
        )
        if c is not None:
            found[spec["name"]] = c
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=FIGURES_DIR,
        help=f"Where to write the PDFs (default: {FIGURES_DIR})",
    )
    ap.add_argument(
        "--datadir",
        type=Path,
        default=OUT_DIR,
        help=f"Where to read the gathered CSVs from (default: {OUT_DIR})",
    )
    ap.add_argument(
        "--no-extra",
        action="store_true",
        help=(
            f"Skip the figures under {EXTRA_SUBDIR}/ (today, all of them), and"
            f" write only the correlations and {MACROS_NAME}"
        ),
    )
    args = ap.parse_args()

    scores_path = use_copied_model_scores(args.datadir)
    binary = read_rows(args.datadir / BINARY_CSV_NAME)
    continuous = read_rows(args.datadir / CONTINUOUS_CSV_NAME)
    scales = read_scales(args.datadir / SCALES_CSV_NAME)
    continuous = normalize(continuous, scales)

    extra_dir = args.outdir / EXTRA_SUBDIR
    print("=" * 70)
    print("MICROPOLIS WORLD — paper figures and numbers")
    print("=" * 70)
    print(f"data:   {args.datadir}")
    print(f"scores: {scores_path}")
    print(f"out:    {args.outdir}")
    print(f"binary:     {len(binary)} scored forecasts")
    print(
        f"continuous: {len(continuous)} scored forecasts, normalized by"
        f" {len(scales)} cities' scales"
    )
    print()

    specs = figure_specs(binary, continuous)
    figures = PaperFigures(extra_dir)
    if args.no_extra:
        print(f"[--no-extra] drawing no figures; {EXTRA_SUBDIR}/ left as it is\n")
        found = correlations_without_drawing(specs)
    else:
        for spec in specs:
            figures.draw(spec["name"], spec["plot"])
        found = figures.correlations
        print()

    print_headlines(found)
    macros = write_macros(args.datadir / MACROS_NAME, found)

    for out in figures.written:
        print(f"Wrote {out}")
    print(f"Wrote {macros}")


if __name__ == "__main__":
    main()
