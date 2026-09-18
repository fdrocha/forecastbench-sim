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

And to data/micropolis/paper/figures/:

- fig_micropolis_capability.pdf      the article's own capability figure

And to data/micropolis/paper/figures/:

- fig_micropolis_horizon.pdf         each binary score by horizon

And to data/micropolis/paper/:

- micropolis-macros.tex              \\MPD* macros for the article's prose
- micropolis_models.tex              the appendix's per-model table
- micropolis_horizon.tex             the same by horizon
- micropolis_continuous.tex          excess nCRPS by horizon and metric
- micropolis_model_scores.csv        the three cells the combined score fits

fig_micropolis_capability.pdf is the one figure here the article places, so
--no-extra keeps drawing it. Three ECI scatters side by side — continuous,
tail, binary, in the order fbs-paper's fig_freeciv_capability.pdf lays its
four out — styled on that figure so the two worlds' capability figures read as
a pair: 5.5 x 2.04in at the paper's own \\textwidth, text set by LaTeX in
Computer Modern, dark-green points with the best and worst model named in
orange, and no legend, which at 26 models would be wider than the panel it
explains.

Correlations here are **sign-adjusted** — ECI against minus the score, so a
positive rho means more capable models forecast better, which is the article's
convention and FreeCiv's. The reports' own figures under figures/extra/ keep
the raw negative rho that analysis-brier.md shows. Every number this script
prints, defines as a macro or draws is adjusted; nothing under extra/ is.

The three headline correlations — ECI against the mid-range excess Brier, the
tail excess bits and the continuous excess nCRPS — are printed to stdout and
defined as LaTeX macros, each with its p-value, its model count, its question
count and both bootstrap intervals, plus \\MPDCapCapability, the capability
figure's caption written in terms of those macros. They are not recomputed here: the figures'
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

When PAPER_REPO_PATH is set — it is, in worlds/micropolis/.env, which
module_globals loads — the run ends by delivering into that checkout of the
article: micropolis-macros.tex to its root, beside math_commands.tex, and
every figure the article places to its figures/, the three appendix tables to
its data/appendix_tables/, and every CSV in the paper's directory to its
data/micropolis/ — the rows every number in the article was computed from.
Nothing from extra/ is copied, that being what extra/ means. An unset variable is a note, since a
machine that only gathers data has no article to deliver to; a variable
pointing at a directory that does not exist is an error, since the alternative
is rebuilding the paper from stale figures and not being told.

Usage:
    scripts/analyze_paper.py
    scripts/analyze_paper.py --no-extra
    scripts/analyze_paper.py --datadir /tmp/paper-data --outdir /tmp/figures
"""

import argparse
import csv
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from micropolis_world import model_scores
from micropolis_world.continuous_eval import MdReport
from micropolis_world.model_scores import eci_of, fb_by_name

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
from analyze_continuous import (
    ALL,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EXCESS,
    by_model_id,
    correlate,
    format_band,
)
from gather_paper_data import (
    BINARY_CSV_NAME,
    CONTINUOUS_CSV_NAME,
    COVERAGE_CSV_NAME,
    MODEL_SCORES_CSV_NAME,
    OUT_DIR,
    SCALES_CSV_NAME,
)

FIGURES_DIR = OUT_DIR / "figures"

# The figures the article does not currently place. They are still the paper's
# own figures, drawn from the paper's own data — kept apart only so --no-extra
# can skip them while the numbers below are refreshed.
EXTRA_SUBDIR = "extra"

CAPABILITY_FIG_NAME = "fig_micropolis_capability.pdf"

# fbs-paper's \textwidth is 5.5in exactly (iclr2027_conference.sty), and
# figures/fig_freeciv_capability.pdf is 5.5 x 2.04in. Matching it means the two
# worlds' capability figures set at the same size with no \includegraphics
# scaling, so their fonts come out the same size on the page.
CAPABILITY_SIZE = (5.5, 2.04)

# The FreeCiv figure's own palette, read off its PDF: a near-black green for
# the models and an orange for the best and worst.
POINT_COLOR = "#102b23"
EXTREME_COLOR = "#e8632c"

MACROS_NAME = "micropolis-macros.tex"

# The by-horizon figure the appendix places, and the three per-model tables it
# \input. Generated here so a rerun moves them with the rest of the paper's
# numbers; the hand-maintained versions they replace came from a generator that
# is not in either repository.
HORIZON_FIG_NAME = "fig_micropolis_horizon.pdf"

# The by-horizon figure's size, matching the one it replaces so the article's
# layout does not move. Taller than the capability figure because its panels
# carry a legend each.
HORIZON_SIZE = (5.5, 2.6)

# Its two panels, as (section, Score, axis label, panel title). The tail is
# scored in excess bits here, as everywhere else in the paper now: the figure
# it replaces used excess Brier, which at q below 5% separates nothing.
HORIZON_PANELS = [
    (
        MID_RANGE,
        "excess_brier",
        "Excess Brier score",
        r"Mid-range questions ($p \geq 5\%$)",
    ),
    (TAIL, "excess_bits", "Excess bits", r"Tail questions ($p < 5\%$)"),
]
TABLE_NAMES = {
    "models": "micropolis_models.tex",
    "horizon": "micropolis_horizon.tex",
    "continuous": "micropolis_continuous.tex",
}

# Where the tables land in the article: it \inputs them from data/appendix_tables/.
PAPER_REPO_TABLES = "data/appendix_tables"

# Where this world's CSVs land in the article: everything the paper's own
# numbers were computed from, beside the other worlds' data directories, so a
# reader or a coauthor's script can reach the rows behind any figure.
PAPER_REPO_DATA = "data/micropolis"

# The three per-model cell scores the cross-world combined score of the
# article's validation section fits on, as a CSV rather than a table. Its
# generator used to scrape micropolis_models.tex by column position, which
# breaks whenever that table's layout changes; a file with named columns
# cannot break that way. It goes with the other CSVs, not with the tables:
# it is data the article computes from, not something it typesets.
CELLS_CSV_NAME = "micropolis_model_scores.csv"
CELLS_COLUMNS = [
    "model",
    "mid_range_excess_brier",
    "tail_excess_bits",
    "excess_ncrps",
]

# The horizons every binary figure and table slices on, in the order the
# article reads them.
HORIZONS = ["3y", "5y", "7y", "10y"]

# A control sequence cannot contain a digit — \MPDRhoBinaryH3 parses as
# \MPDRhoBinaryH followed by a "3" — so a horizon's macro is named in words.
HORIZON_WORDS = {"3y": "Hthree", "5y": "Hfive", "7y": "Hseven", "10y": "Hten"}

# Where the article lives, from worlds/micropolis/.env (loaded by
# module_globals, which this script reaches through gather_paper_data). Unset
# means "nobody has the paper checked out here", which is the normal state on
# a machine that only gathers data, so its absence is a note and not an error.
PAPER_REPO_ENV = "PAPER_REPO_PATH"

# Inside the paper repo: the figures go where \includegraphics resolves them,
# and the macros to the root, beside math_commands.tex, which is where the
# article's other \input of definitions sits — \input{micropolis-macros.tex}.
PAPER_REPO_FIGURES = "figures"

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


# The capability figure's three panels, in the order FreeCiv's four are laid
# out (continuous, tails, binary) so the two figures' panels line up when the
# article places them near each other. `label` is the paper's word for the
# slice — "Binary" is the mid-range questions and "Tail" the tail ones, since
# "binary" covers both literally and only the pair reads unambiguously.
PANELS = [
    ("Continuous", "eci_vs_excess_ncrps-city", "Excess nCRPS"),
    ("Tail", "eci_vs_excess_bits-tail", "Excess bits"),
    ("Binary", "eci_vs_excess_brier-mid-range", "Excess Brier"),
]

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


# The two capability scales the article correlates against. ECI covers every
# model in the panel; ForecastBench covers the 17 with a published overall, so
# its n differs and its macros carry their own count. Both come from the
# paper's own copy of model_scores.csv.
PREDICTORS = [
    ("", "ECI", lambda models: by_model_id(eci_by_name_of(models), models)),
    (
        "FB",
        "ForecastBench overall",
        lambda models: by_model_id(fb_by_name(models), models),
    ),
]


def eci_by_name_of(models: list[str]) -> dict[str, float]:
    """ECI keyed on the bare name, as capability_predictors builds it.

    Wrapped so PREDICTORS can name it beside fb_by_name with one signature;
    both go through by_model_id to be rekeyed onto the ids the rows carry.
    """
    return {m.split("/", 1)[-1]: eci_of(m) for m in models if eci_of(m) is not None}


def read_coverage(path: Path) -> list[dict]:
    """model_coverage.csv as rows with integer counts.

    Prompted against parsed, per model and eval — the one thing the forecast
    CSVs cannot carry, since an unparsed forecast leaves no scored row.
    """
    if not path.exists():
        sys.exit(
            f"[error] {path} not found\n"
            "  rerun scripts/gather_paper_data.py; it writes the per-model"
            " coverage this script reports parse rates from"
        )
    with path.open(newline="") as f:
        rows = [
            {**r, "nforecasts": int(r["nforecasts"]), "nvalid": int(r["nvalid"])}
            for r in csv.DictReader(f)
        ]
    if not rows:
        sys.exit(f"[error] {path} holds no rows")
    return rows


def parse_rates(coverage: list[dict]) -> dict[str, float]:
    """The worst model's parse rate per eval, as a percentage.

    The article quotes "every model parsed on at least X% of its questions",
    so the number it needs is the minimum over the panel — which is exactly
    the number that moves when the panel changes, and the one it had wrong.
    """
    out = {}
    for name in sorted({r["eval"] for r in coverage}):
        rates = [
            100.0 * r["nvalid"] / r["nforecasts"]
            for r in coverage
            if r["eval"] == name and r["nforecasts"]
        ]
        if rates:
            out[name] = min(rates)
    return out


def paper_repo_dir() -> Path | None:
    """The article's checkout from $PAPER_REPO_PATH, or None when unset.

    The variable comes from worlds/micropolis/.env, which module_globals loads
    at import time — this script reaches that through gather_paper_data, so the
    value is in the environment before main() runs and nothing here has to
    load it again.

    Unset is the normal state on a machine with no checkout of the article, so
    it is reported and skipped rather than treated as a failure. A value that
    does not exist, though, is a typo worth stopping for: silently not
    delivering the figures is how a paper ends up rebuilt from stale ones.
    """
    raw = os.environ.get(PAPER_REPO_ENV, "").strip()
    if not raw:
        return None
    repo = Path(raw).expanduser()
    if not repo.is_dir():
        sys.exit(
            f"[error] {PAPER_REPO_ENV}={raw} is not a directory\n"
            f"  fix it in {Path(__file__).resolve().parents[1] / '.env'},"
            " or unset it to skip copying into the article"
        )
    return repo


def paper_figures(outdir: Path) -> list[Path]:
    """The PDFs the article places: outdir's own, never extra/'s.

    Taken from the directory rather than from a list of names, so a figure
    added to the paper later is delivered without this having to be kept in
    step. The split is exactly the one EXTRA_SUBDIR already draws: a figure
    sits beside extra/ when the article places it, and inside extra/ when it
    does not, and nothing recurses into it.
    """
    return sorted(p for p in outdir.glob("*.pdf") if p.is_file())


def paper_csvs(datadir: Path) -> list[Path]:
    """Every CSV in the paper's directory, for delivery into the article.

    Taken from the directory rather than a list of names, so a file added to
    the gathered data reaches the article without this being kept in step.
    These are what every number in the paper was computed from: the two
    forecast files, the per-city scales, the per-model coverage, the
    leaderboard copy the ECI came from, and the three cells the cross-world
    combined score fits on.
    """
    return sorted(p for p in datadir.glob("*.csv") if p.is_file())


def deliver(paths: list[Path], dest: Path) -> list[Path]:
    """Copy each path into `dest`, saying which ones replaced something.

    The article's own figures and macros, so the paper builds from this run
    without a manual copy. Only the files the paper places are passed in —
    nothing from figures/extra/, which exists precisely because the article
    does not use it.

    An overwrite is called out per file: these land in a git repo, and knowing
    that a figure was replaced rather than added is what tells the difference
    between "new figure" and "the numbers moved" when the diff is a binary
    PDF. shutil.copyfile, not cp, since cp is aliased interactively on this
    machine and copies nothing over an existing file.
    """
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for src in paths:
        target = dest / src.name
        existed = target.exists()
        shutil.copyfile(src, target)
        print(f"{'Replaced' if existed else 'Copied  '} {target}")
        written.append(target)
    return written


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


def display_names(path: Path) -> dict[str, str]:
    """Model id -> the leaderboard's display name, from the paper's own copy.

    model_scores.csv carries a "Name" column ("OpenAI: GPT 4.1 Nano") that the
    package's parser drops, and it is what the article's figures label points
    with — "GPT-5 Nano" reads where "gpt-5-nano-2025-08-07" does not. Read
    here rather than added to ModelScores so the shared package keeps its
    shape; the provider prefix is dropped since the panel has no room for it.
    """
    out = {}
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            slug = (row.get("slug") or "").strip()
            name = (row.get("Name") or "").strip()
            if slug and name:
                out[slug] = name.split(":", 1)[-1].strip()
    return out


def adjusted(value: float | None) -> float | None:
    """Flip a coefficient's sign, to the article's convention.

    Every score in this world is lower-is-better, so ECI correlates negatively
    with skill and the reports print a negative rho. The article states its
    correlations against *minus* the score, so that a positive rho reads as
    "more capable models forecast better" — the same convention FreeCiv's
    figures and the paper's own prose use. Only the sign changes: Spearman on
    a negated variable is the same coefficient reflected, and its p-value and
    the width of its intervals are untouched.
    """
    return None if value is None else -value


def adjusted_band(ci: tuple[float, float] | None) -> tuple[float, float] | None:
    """Sign-adjust an interval, which also reverses its ends.

    Negating [-0.87, -0.45] gives [0.45, 0.87], not [0.87, 0.45]: the lower
    bound of the negated coefficient is minus the upper bound of the original.
    """
    return None if ci is None else (-ci[1], -ci[0])


def draw_capability_figure(
    path: Path, found: dict[str, object], names: dict[str, str]
) -> Path | None:
    """The article's capability figure: three ECI scatters side by side.

    Styled on fbs-paper's figures/fig_freeciv_capability.pdf, so the two
    worlds' capability figures can sit near each other and read as one pair:
    the same 5.5 x 2.04in at the paper's \textwidth, the same Computer Modern
    through LaTeX, the same dark-green points with the best and worst model in
    orange, the same "(a) ..." panel captions under the axes, and rho sign-
    adjusted so positive means more capable models forecast better.

    Unlike the figures/extra/ scatters this does not go through the reports'
    plot functions: those draw one panel with a per-model legend, which is the
    right figure for a report and the wrong one for a 2in-tall panel. The
    numbers are still the reports' own — the rho annotated on each panel is
    the Correlation the matching extra/ figure computed, only sign-adjusted.
    """
    import matplotlib

    # pgf rather than Agg: the text is set by LaTeX itself, which is what puts
    # the figure in the paper's own Computer Modern instead of a sans-serif
    # approximation of it.
    matplotlib.use("pgf")
    import matplotlib.pyplot as plt

    missing = [name for name, fig, _ in PANELS if found.get(fig) is None]
    if missing:
        print(
            f"[skipped] {CAPABILITY_FIG_NAME}: no correlation for {', '.join(missing)}"
        )
        return None

    with plt.rc_context(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "font.family": "serif",
            # Let LaTeX pick the fonts rather than matplotlib naming them, so
            # the result is the document's Computer Modern.
            "pgf.rcfonts": False,
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=CAPABILITY_SIZE, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.04, h_pad=0.02, wspace=0.02)
        for i, (ax, (name, figname, ylabel)) in enumerate(zip(axes, PANELS)):
            letter = chr(ord("a") + i)
            _draw_panel(ax, found[figname], name, ylabel, letter, names)
        path.parent.mkdir(parents=True, exist_ok=True)
        # No bbox_inches="tight": the figure is sized to the paper's
        # \textwidth exactly, and a tight box would grow it past that, so
        # \includegraphics[width=\linewidth] would scale it back down and take
        # the fonts with it. constrained layout fits the labels inside instead.
        fig.savefig(path)
        plt.close(fig)
    return path


def draw_horizon_figure(
    path: Path, binary: list[dict], names: dict[str, str]
) -> Path | None:
    """Each binary score by forecast horizon, mid-range and tail.

    Replaces the figure the appendix carried, which came from the aggregated
    per-model file and disagreed with the per-question scores in direction:
    it had the mean mid-range excess Brier rising with horizon where these
    rows have it falling. Drawn from the same CSV as everything else here, so
    the two cannot part company again.

    Mean, median and interquartile range across models, plus the model with
    the best pooled score — the shape of the figure it replaces, on the
    paper's own scale and panel.
    """
    import matplotlib

    matplotlib.use("pgf")
    import matplotlib.pyplot as plt
    import numpy as np

    with plt.rc_context(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "font.family": "serif",
            "pgf.rcfonts": False,
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=HORIZON_SIZE, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.04, h_pad=0.02, wspace=0.03)
        drawn = False
        for ax, (section, key, ylabel, title) in zip(axes, HORIZON_PANELS):
            rows = [r for r in binary if r["section"] == section]
            if not rows:
                continue
            drawn = True
            models = models_in_order(rows)
            # (model, horizon) -> mean score, so every series below reads off
            # one table rather than re-filtering the rows per line.
            per = {m: [_mean_of(rows, m, h, key) for h in HORIZONS] for m in models}
            xs = np.arange(len(HORIZONS))
            stack = np.array(
                [v for v in per.values() if all(x is not None for x in v)], dtype=float
            )
            ax.fill_between(
                xs,
                np.percentile(stack, 25, axis=0),
                np.percentile(stack, 75, axis=0),
                color="0.85",
                lw=0,
                label=f"Interquartile range across {len(stack)} models",
            )
            ax.plot(
                xs,
                stack.mean(axis=0),
                color=POINT_COLOR,
                lw=1.0,
                marker="o",
                ms=2.5,
                label=f"Mean of {len(stack)} models",
            )
            ax.plot(
                xs,
                np.median(stack, axis=0),
                color="0.45",
                lw=0.8,
                ls="--",
                label=f"Median of {len(stack)} models",
            )
            # The best model pooled over horizons, named as the old figure
            # named it, so a reader comparing drafts sees the same series.
            pooled = {m: _mean_of(rows, m, None, key) for m in models}
            best = min(
                (m for m in pooled if pooled[m] is not None), key=lambda m: pooled[m]
            )
            ax.plot(
                xs,
                per[best],
                color=EXTREME_COLOR,
                lw=1.0,
                marker="s",
                ms=2.5,
                label=f"{tex_escape(names.get(best, best))} (best overall)",
            )
            ax.set_xticks(xs)
            ax.set_xticklabels([h.rstrip("y") for h in HORIZONS])
            ax.set_xlabel("Forecast horizon (game years)")
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=7)
            # Headroom for the legend, which sits top-left over the band.
            ax.set_ylim(
                0,
                max(np.percentile(stack, 75, axis=0).max(), stack.mean(axis=0).max())
                * 1.38,
            )
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(
                fontsize=5,
                frameon=False,
                loc="upper left",
                borderaxespad=0.2,
                handlelength=1.4,
                handletextpad=0.5,
                labelspacing=0.25,
            )
        if not drawn:
            plt.close(fig)
            print(f"[skipped] {HORIZON_FIG_NAME}: no binary rows")
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return path


def _mean_of(rows: list[dict], model: str, horizon: str | None, key: str):
    """Mean `key` for one model, at one horizon or pooled over all of them."""
    vals = [
        r[key]
        for r in rows
        if r["model_id"] == model
        and (horizon is None or r["horizon"] == horizon)
        and r[key] != ""
    ]
    return sum(vals) / len(vals) if vals else None


def _draw_panel(
    ax, c, name: str, ylabel: str, letter: str, names: dict[str, str]
) -> None:
    """One panel: the models' (ECI, mean score), a fit line and the rho note.

    The per-model means come from `c.scores`, the ones the correlation itself
    used, so a point cannot sit somewhere the coefficient does not describe.
    """
    import numpy as np

    points = sorted(
        (eci_of(m), v, names.get(m, m.split("/")[-1]))
        for m, v in c.scores.items()
        if eci_of(m) is not None
    )
    x = np.array([e for e, _, _ in points])
    y = np.array([v for _, v, _ in points])

    # Best and worst by score, in the reference's orange. Lower is better in
    # every panel, so best is the minimum.
    best = int(np.argmin(y))
    worst = int(np.argmax(y))
    colors = [POINT_COLOR] * len(points)
    colors[best] = colors[worst] = EXTREME_COLOR
    ax.scatter(x, y, s=7, c=colors, linewidths=0, zorder=3, clip_on=False)

    # A least-squares line, as the reference draws: it shows the direction the
    # rank correlation reports without claiming the fit is the estimate.
    if len(points) > 1:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, slope * xs + intercept, color="0.72", lw=0.6, zorder=1)

    # Headroom for the rho note, which sits top-left: without it the note
    # lands on whichever model is worst at the low-ECI end.
    lo, hi = min(y), max(y)
    ax.set_ylim(lo - 0.13 * (hi - lo), hi + 0.26 * (hi - lo))
    label_models(ax, points, best, worst)

    rho, band = adjusted(c.rho), adjusted_band(c.rho_models)
    note = f"$\\rho = {rho:.2f}$"
    if band:
        note += f" $[{band[0]:.2f}, {band[1]:.2f}]$"
    ax.text(
        0.03,
        0.955,
        note,
        transform=ax.transAxes,
        fontsize=6,
        color=POINT_COLOR,
        va="top",
        ha="left",
    )

    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    # "ECI" then the panel caption under it, as the reference sets them: two
    # lines of one xlabel rather than an xlabel plus a title, so tight_layout
    # reserves room for both and the caption cannot land on the panel below.
    ax.set_xlabel(f"ECI\n\\textrm{{({letter}) {name}}}")


def label_models(ax, points: list[tuple], best: int, worst: int) -> None:
    """Name the best and worst model beside their points.

    A 26-entry legend is wider than a 1.8in panel, so the figure names only
    the two models a reader looks for, the way the reference figure's orange
    pair does. Labeling the middle of the ranking was tried and dropped: at
    5pt the names of models whose scores differ by a percent land on each
    other and on the tick labels, and the figure's claim is the trend, not the
    identity of every point.

    Each label is placed on the side away from the data, and vertically away
    from the fit line, so it cannot sit on the line or the axis.
    """
    xs = [p[0] for p in points]
    xlo, xhi = min(xs), max(xs)
    xr = (xhi - xlo) or 1.0
    for i, below in ((best, True), (worst, False)):
        eci, value, label = points[i]
        right = (eci - xlo) / xr > 0.5
        ax.annotate(
            label,
            (eci, value),
            textcoords="offset points",
            # The best model is at the bottom of the panel and the worst at
            # the top, so pushing each further that way clears the cloud.
            xytext=(-4.5 if right else 4.5, -5.0 if below else 3.0),
            ha="right" if right else "left",
            va="top" if below else "bottom",
            fontsize=5.4,
            color=EXTREME_COLOR,
            annotation_clip=False,
        )


def macro_band(ci: tuple[float, float] | None) -> str:
    r"""A bootstrap interval as self-contained math: $[0.45,\,0.87]$.

    Math-mode so the article can drop the macro into prose without wrapping
    it, and a thin space after the comma because a bare one sets too tight
    beside a minus sign. Two decimals, as the reports' own bands use; an
    absent interval becomes a dash rather than a number that is not there.
    The interval is expected sign-adjusted already — see adjusted_band, which
    reverses its ends as well as its signs.
    """
    if ci is None:
        return "---"
    return f"$[{ci[0]:.2f},\\,{ci[1]:.2f}]$"


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
    rho = adjusted(c.rho)
    return [
        f"% {h.what}",
        f"\\newcommand{{\\{pre}Rho{h.macro}}}{{{rho:.3f}}}",
        (
            f"\\newcommand{{\\{pre}Rho{h.macro}CIModels}}"
            f"{{{macro_band(adjusted_band(c.rho_models))}}}"
        ),
        (
            f"\\newcommand{{\\{pre}Rho{h.macro}CIQuestions}}"
            f"{{{macro_band(adjusted_band(c.rho_questions))}}}"
        ),
        f"\\newcommand{{\\{pre}P{h.macro}}}{{{macro_p(c.rho_p)}}}",
        f"\\newcommand{{\\{pre}NModels{h.macro}}}{{{c.n_models}}}",
        f"\\newcommand{{\\{pre}NQuestions{h.macro}}}{{{c.n_questions:,}}}",
        "",
    ]


def extremes_lines(h: Headline, c, names: dict[str, str]) -> list[str]:
    r"""\MPDBest/\MPDWorst macros: the range each score runs over.

    The article quotes "runs from X (model) to Y (model)" for every score, and
    those were the numbers most at risk of going stale — they move whenever a
    model is added. Four macros per slice: the two values and the two display
    names. The values come from c.scores, the per-model means the coefficient
    itself used, so the range and the correlation describe one set of numbers.
    """
    pre = MACRO_PREFIX
    if not c.scores:
        return []
    best = min(c.scores, key=c.scores.get)
    worst = max(c.scores, key=c.scores.get)
    return [
        f"\\newcommand{{\\{pre}Best{h.macro}}}{{{c.scores[best]:.3g}}}",
        (
            f"\\newcommand{{\\{pre}Best{h.macro}Model}}"
            f"{{{tex_escape(names.get(best, best))}}}"
        ),
        f"\\newcommand{{\\{pre}Worst{h.macro}}}{{{c.scores[worst]:.3g}}}",
        (
            f"\\newcommand{{\\{pre}Worst{h.macro}Model}}"
            f"{{{tex_escape(names.get(worst, worst))}}}"
        ),
    ]


def tex_escape(text: str) -> str:
    """Escape what a model's display name can carry into LaTeX.

    These come from a leaderboard's Name column, so they are plain words and
    digits today; the escape is here because the column is edited by hand and
    an underscore or ampersand in it would otherwise break the article's build
    with an error pointing at the macro file rather than at the CSV.
    """
    for char, repl in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        text = text.replace(char, repl)
    return text


def horizon_lines(h: Headline, by_horizon: dict[str, object]) -> list[str]:
    r"""\MPDRho{stem}H{years} per horizon.

    The article says the correlation "holds at every horizon" and quotes the
    two ends, which were the last bare numbers in the results paragraph. One
    macro per horizon rather than just the ends, so the sentence can be
    rewritten without another trip to the data.
    """
    pre = MACRO_PREFIX
    out = []
    for horizon in HORIZONS:
        c = by_horizon.get(horizon)
        if c is None:
            continue
        out.append(
            f"\\newcommand{{\\{pre}Rho{h.macro}{HORIZON_WORDS[horizon]}}}"
            f"{{{adjusted(c.rho):.2f}}}"
        )
    return out


def predictor_lines(
    prefix: str, what: str, per_headline: dict[str, object]
) -> list[str]:
    r"""One predictor's \MPD{prefix}* macros for every headline slice.

    ECI carries no prefix, since it is the article's main predictor and its
    macros are the ones already in use; ForecastBench is "FB". Its n differs —
    only 17 models have a published overall — so the count is a macro per
    slice rather than assumed shared.
    """
    pre = MACRO_PREFIX
    lines = [f"% {what}"]
    for h in HEADLINES:
        c = per_headline.get(h.figure)
        if c is None:
            continue
        stem = f"{pre}{prefix}"
        lines += [
            f"\\newcommand{{\\{stem}Rho{h.macro}}}{{{adjusted(c.rho):.3f}}}",
            (
                f"\\newcommand{{\\{stem}Rho{h.macro}CIModels}}"
                f"{{{macro_band(adjusted_band(c.rho_models))}}}"
            ),
            f"\\newcommand{{\\{stem}P{h.macro}}}{{{macro_p(c.rho_p)}}}",
            f"\\newcommand{{\\{stem}NModels{h.macro}}}{{{c.n_models}}}",
        ]
    return lines + [""]


def bootstrap_lines() -> list[str]:
    r"""\MPDResamples and \MPDSeed: how the intervals were drawn.

    The article's table captions state both, and stated them wrong — 10,000
    resamples at seed 2026, where this world draws BOOTSTRAP_RESAMPLES at
    BOOTSTRAP_SEED. Macros so a caption cannot describe a bootstrap that did
    not happen.
    """
    pre = MACRO_PREFIX
    return [
        "% How every interval above was drawn.",
        f"\\newcommand{{\\{pre}Resamples}}{{{BOOTSTRAP_RESAMPLES:,}}}",
        f"\\newcommand{{\\{pre}Seed}}{{{BOOTSTRAP_SEED}}}",
        "",
    ]


def parse_lines(rates: dict[str, float]) -> list[str]:
    r"""\MPDMinParse* : the worst model's parse rate per eval.

    Floored to one decimal the way the article quotes it. "At least" is the
    claim, so rounding down keeps the sentence true: 98.44 becomes 98.4, never
    98.5.
    """
    import math

    pre = MACRO_PREFIX
    lines = ["% Lowest parse rate over the panel, per eval (percent)."]
    for name, rate in sorted(rates.items()):
        stem = name.capitalize()
        lines.append(
            f"\\newcommand{{\\{pre}MinParse{stem}}}{{{math.floor(rate * 10) / 10:.1f}}}"
        )
    return lines + [""]


def caption_lines(found: dict[str, object]) -> list[str]:
    r"""\MPDCapCapability: the capability figure's caption.

    Written in terms of the other macros rather than with the numbers
    substituted, so the .tex shows what the caption depends on and a rerun
    that moves a coefficient moves the caption with it. Only the question
    counts vary per panel, and those are macros too.
    """
    pre = MACRO_PREFIX
    if any(found.get(f) is None for _, f, _ in PANELS):
        return []
    body = (
        "Micropolis forecasting scores against the Epoch Capabilities Index"
        f" for \\{pre}NModelsContinuous\\ models; lower is better in every"
        " panel."
        f" (a) Continuous: excess nCRPS, \\{pre}NQuestionsContinuous\\"
        " questions."
        f" (b) Tail questions ($p<5\\%$): excess bits,"
        f" \\{pre}NQuestionsTail\\ questions."
        f" (c) Binary questions ($p\\geq5\\%$): excess Brier,"
        f" \\{pre}NQuestionsBinary\\ questions."
        " $\\rho$ is Spearman rank correlation of ECI with $-$score,"
        " sign-adjusted so that positive means more capable models forecast"
        " better; brackets are 95\\% percentile intervals from"
        f" {BOOTSTRAP_RESAMPLES:,} bootstrap resamples over models."
        " Orange marks the best and worst model."
    )
    horizon = (
        "Micropolis forecasting scores by forecast horizon (3, 5, 7 and 10 game"
        " years); lower is better in both panels. Left: mid-range questions"
        " ($p \\geq 5\\%$), excess Brier score. Right: tail questions ($p<5\\%$),"
        " excess bits. Solid line: mean over the"
        f" \\{pre}NModelsBinary\\ models; dashed: median; band: interquartile"
        " range across models; orange: the best model pooled over horizons"
        f" (\\{pre}BestBinaryModel\\ for mid-range, \\{pre}BestTailModel\\ for"
        " the tail)."
    )
    return [
        "% The capability figure's caption. Depends on the macros above, so a",
        "% rerun that moves a coefficient moves the caption with it.",
        f"\\newcommand{{\\{pre}CapCapability}}{{{body}}}",
        "",
        "% The by-horizon figure's caption.",
        f"\\newcommand{{\\{pre}CapHorizon}}{{{horizon}}}",
        "",
    ]


def correlations_by_horizon(specs: list[dict]) -> dict[str, dict[str, object]]:
    """Each quoted slice correlated within each horizon.

    The pooled figures answer "does skill track capability"; these answer
    "at which horizons", which is the claim the article makes in one sentence.
    Only the quoted slices are computed — a bootstrap per horizon is not free.
    """
    wanted = {h.figure for h in HEADLINES}
    out: dict[str, dict[str, object]] = {}
    for spec in specs:
        if spec["name"] not in wanted:
            continue
        per = {}
        for horizon in HORIZONS:
            rows = [r for r in spec["rows"] if r["horizon"] == horizon]
            if not rows:
                continue
            c = correlate(
                "ECI",
                by_model_id(eci_by_name_of(spec["models"]), spec["models"]),
                rows,
                spec["score"].key,
                spec["models"],
                horizon,
            )
            if c is not None:
                per[horizon] = c
        out[spec["name"]] = per
    return out


def correlations_for(specs: list[dict], predictor) -> dict[str, object]:
    """Each quoted slice against one predictor, pooled over horizons.

    Used for ForecastBench: the ECI numbers come from the figures' own
    correlate() call, but a second predictive scale has no figure to intercept,
    so it is computed here on the same rows.
    """
    wanted = {h.figure for h in HEADLINES}
    out = {}
    for spec in specs:
        if spec["name"] not in wanted:
            continue
        c = correlate(
            "predictor",
            predictor(spec["models"]),
            spec["rows"],
            spec["score"].key,
            spec["models"],
            ALL,
        )
        if c is not None:
            out[spec["name"]] = c
    return out


# The metric columns of the continuous table, in the order the article reads
# them, with the abbreviation each column head uses.
METRIC_COLUMNS = [
    ("cityPop", "Popul."),
    ("trafficAverage", "Traffic"),
    ("pollutionAverage", "Pollut."),
    ("crimeAverage", "Crime"),
    ("landValueAverage", "Land"),
]


def table_file(lines: list[str], source: str) -> str:
    """A generated table as its file's text, header comment included."""
    return "\n".join(
        [
            (
                "% Generated by worlds/micropolis/scripts/analyze_paper.py"
                " -- do not edit by hand."
            ),
            f"% {source}",
            *lines,
        ]
    )


def cell(value, fmt: str = "{:.4f}") -> str:
    """One table cell: the number, or a dash where a model has no score."""
    return "--" if value is None else fmt.format(value)


def models_table(
    binary: list[dict],
    continuous: list[dict],
    coverage: list[dict],
    names: dict[str, str],
) -> str:
    """Per-model scores pooled over horizons, with parse rates.

    One row per model of the panel, ordered by ECI as the article's tables
    are: the two binary scores, the continuous one, and the share of each
    eval's questions the model returned a readable forecast for.
    """
    mid = [r for r in binary if r["section"] == MID_RANGE]
    tail = [r for r in binary if r["section"] == TAIL]
    parsed = {
        (r["model"], r["eval"]): (
            100.0 * r["nvalid"] / r["nforecasts"] if r["nforecasts"] else None
        )
        for r in coverage
    }
    rows = []
    for m in sorted(models_in_order(binary), key=lambda m: -(eci_of(m) or 0)):
        rows.append(
            " & ".join(
                [
                    tex_escape(names.get(m, m)),
                    cell(eci_of(m), "{:.1f}"),
                    cell(_mean_of(mid, m, None, "excess_brier")),
                    cell(_mean_of(tail, m, None, "excess_bits")),
                    cell(_mean_of(continuous, m, None, "excess_ncrps")),
                    cell(parsed.get((m, "binary")), "{:.1f}"),
                    cell(parsed.get((m, "continuous")), "{:.1f}"),
                ]
            )
            + r" \\"
        )
    return table_file(
        [
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{lrrrrrr}",
            r"\toprule",
            (
                r"Model & ECI & Excess Brier & Excess bits & Excess nCRPS"
                r" & \multicolumn{2}{c}{Parsed (\%)} \\"
            ),
            r"\cmidrule(lr){6-7}",
            r" & & Mid-range & Tail & & Binary & Continuous \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ],
        "Per model, horizons pooled. Sources: binary_forecasts.csv,"
        " continuous_forecasts.csv, model_coverage.csv.",
    )


def horizon_table(binary: list[dict], names: dict[str, str]) -> str:
    """Each binary score by horizon, mid-range then tail."""
    mid = [r for r in binary if r["section"] == MID_RANGE]
    tail = [r for r in binary if r["section"] == TAIL]
    rows = []
    for m in sorted(models_in_order(binary), key=lambda m: -(eci_of(m) or 0)):
        cells = [tex_escape(names.get(m, m)), cell(eci_of(m), "{:.1f}")]
        for src, key in ((mid, "excess_brier"), (tail, "excess_bits")):
            cells += [cell(_mean_of(src, m, h, key)) for h in HORIZONS]
        rows.append(" & ".join(cells) + r" \\")
    return table_file(
        [
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{lrrrrrrrrr}",
            r"\toprule",
            (
                r"Model & ECI & \multicolumn{4}{c}{Mid-range excess Brier, by"
                r" horizon (years)} & \multicolumn{4}{c}{Tail excess bits, by"
                r" horizon (years)} \\"
            ),
            r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
            r" & & 3 & 5 & 7 & 10 & 3 & 5 & 7 & 10 \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ],
        "Per model and horizon. Source: binary_forecasts.csv.",
    )


def continuous_table(continuous: list[dict], names: dict[str, str]) -> str:
    """Excess nCRPS by horizon and by metric."""
    rows = []
    for m in sorted(models_in_order(continuous), key=lambda m: -(eci_of(m) or 0)):
        cells = [tex_escape(names.get(m, m)), cell(eci_of(m), "{:.1f}")]
        cells += [
            cell(_mean_of(continuous, m, h, "excess_ncrps"), "{:.3f}") for h in HORIZONS
        ]
        for metric, _ in METRIC_COLUMNS:
            vals = [
                r["excess_ncrps"]
                for r in continuous
                if r["model_id"] == m and r["metric"] == metric
            ]
            cells.append(cell(sum(vals) / len(vals) if vals else None, "{:.3f}"))
        rows.append(" & ".join(cells) + r" \\")
    heads = " & ".join(h for _, h in METRIC_COLUMNS)
    return table_file(
        [
            r"\setlength{\tabcolsep}{3pt}",
            r"\begin{tabular}{lrrrrrrrrrr}",
            r"\toprule",
            (
                r"Model & ECI & \multicolumn{4}{c}{By horizon (years)}"
                r" & \multicolumn{5}{c}{By metric} \\"
            ),
            r"\cmidrule(lr){3-6}\cmidrule(lr){7-11}",
            rf" & & 3 & 5 & 7 & 10 & {heads} \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
        ],
        "Per model, excess nCRPS by horizon and metric."
        " Source: continuous_forecasts.csv.",
    )


def write_cells(
    datadir: Path,
    binary: list[dict],
    continuous: list[dict],
) -> Path:
    """One row per model with the three scores the combined score fits on.

    The article's cross-world latent-skill fit needs Micropolis as three cells
    beside StarSim's and FreeCiv's. Its generator read them out of
    micropolis_models.tex by column number, so changing that table's columns
    or its model names silently redefined a cell or stopped the parse. Named
    columns instead, so the dependency is explicit and survives the table
    being re-laid-out.

    Models are named by model id, as every other CSV here names them. A
    consumer wanting the display names joins the `slug` column of
    model_scores.csv, which sits in the same directory.
    """
    mid = [r for r in binary if r["section"] == MID_RANGE]
    tail = [r for r in binary if r["section"] == TAIL]
    rows = []
    for m in sorted(models_in_order(binary), key=lambda m: -(eci_of(m) or 0)):
        rows.append(
            {
                "model": m,
                "mid_range_excess_brier": _mean_of(mid, m, None, "excess_brier"),
                "tail_excess_bits": _mean_of(tail, m, None, "excess_bits"),
                "excess_ncrps": _mean_of(continuous, m, None, "excess_ncrps"),
            }
        )
    path = datadir / CELLS_CSV_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CELLS_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return path


def write_tables(
    datadir: Path,
    binary: list[dict],
    continuous: list[dict],
    coverage: list[dict],
    names: dict[str, str],
) -> list[Path]:
    """The three appendix tables, as files the article \\inputs."""
    built = {
        "models": models_table(binary, continuous, coverage, names),
        "horizon": horizon_table(binary, names),
        "continuous": continuous_table(continuous, names),
    }
    out = []
    for key, text in built.items():
        path = datadir / TABLE_NAMES[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
        out.append(path)
    return out


def write_macros(
    path: Path,
    found: dict[str, object],
    names: dict[str, str],
    by_horizon: dict[str, dict[str, object]],
    fb: dict[str, object],
    rates: dict[str, float],
) -> Path:
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
        "% its mean score, SIGN-ADJUSTED: the correlation is against -score,",
        "% so a positive value means more capable models forecast better. The",
        "% reports and figures/extra/ show the raw negative rho instead.",
        "% The two CIs are 95% percentile-bootstrap intervals: CIModels",
        "% resamples the models, CIQuestions the questions with models fixed.",
        "",
    ]
    missing = []
    for h in HEADLINES:
        c = found.get(h.figure)
        if c is None:
            missing.append(h.macro)
            continue
        lines += (
            macro_lines(h, c)[:-1]
            + extremes_lines(h, c, names)
            + horizon_lines(h, by_horizon.get(h.figure, {}))
            + [""]
        )
    lines += parse_lines(rates)
    lines += bootstrap_lines()
    if fb:
        lines += predictor_lines("FB", "ForecastBench overall", fb)
    lines += caption_lines(found)
    if missing:
        print(f"[warn] no correlation for {', '.join(missing)}; macros not defined")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


def print_headlines(found: dict[str, object]) -> None:
    """The quoted correlations, sign-adjusted as the article states them."""
    print("Spearman ECI vs mean score, all horizons pooled")
    print("(sign-adjusted: positive means more capable models forecast better)")
    print("-" * 70)
    for h in HEADLINES:
        c = found.get(h.figure)
        print(f"{h.what}  [\\{MACRO_PREFIX}Rho{h.macro}]")
        if c is None:
            print("  not computed: too few models with an ECI, or no spread\n")
            continue
        direction = "pro-g" if c.rho < 0 else "anti-g"
        print(
            f"  ρ={adjusted(c.rho):+.3f}  p={c.rho_p:.4f}  ({direction},"
            f" n={c.n_models} models, {c.n_questions:,} questions)"
        )
        print(
            f"  95% CI  models    {format_band(adjusted_band(c.rho_models), 0)}\n"
            f"          questions {format_band(adjusted_band(c.rho_questions), 0)}\n"
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
            f"Skip the figures under {EXTRA_SUBDIR}/, keeping the article's own"
            f" {CAPABILITY_FIG_NAME}, the correlations and {MACROS_NAME}"
        ),
    )
    args = ap.parse_args()

    scores_path = use_copied_model_scores(args.datadir)
    binary = read_rows(args.datadir / BINARY_CSV_NAME)
    continuous = read_rows(args.datadir / CONTINUOUS_CSV_NAME)
    scales = read_scales(args.datadir / SCALES_CSV_NAME)
    continuous = normalize(continuous, scales)
    coverage = read_coverage(args.datadir / COVERAGE_CSV_NAME)

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
        print(f"[--no-extra] {EXTRA_SUBDIR}/ left as it is\n")
        found = correlations_without_drawing(specs)
    else:
        for spec in specs:
            figures.draw(spec["name"], spec["plot"])
        found = figures.correlations
        print()

    print_headlines(found)
    names = display_names(args.datadir / MODEL_SCORES_CSV_NAME)
    by_horizon = correlations_by_horizon(specs)
    fb = correlations_for(specs, PREDICTORS[1][2])
    rates = parse_rates(coverage)
    for name, rate in sorted(rates.items()):
        print(f"lowest parse rate, {name}: {rate:.1f}%")
    print()
    macros = write_macros(
        args.datadir / MACROS_NAME, found, names, by_horizon, fb, rates
    )
    tables = write_tables(args.datadir, binary, continuous, coverage, names)
    cells = write_cells(args.datadir, binary, continuous)
    # The article's own figure, which is not one of the extra ones: --no-extra
    # skips the figures the paper does not place, and this is the one it does.
    capability = draw_capability_figure(
        args.outdir / CAPABILITY_FIG_NAME,
        found,
        names,
    )

    horizon_fig = draw_horizon_figure(args.outdir / HORIZON_FIG_NAME, binary, names)

    for out in figures.written:
        print(f"Wrote {out}")
    for out in (capability, horizon_fig):
        if out:
            print(f"Wrote {out}")
    print(f"Wrote {macros}")
    for out in [*tables, cells]:
        print(f"Wrote {out}")

    # Deliver into the article, when there is one checked out here: the macros
    # to its root, where its other \\input of definitions lives, and the
    # figures it places to figures/.
    repo = paper_repo_dir()
    if repo is None:
        print(f"\n[note] {PAPER_REPO_ENV} unset; not copying into the article")
        return
    print()
    deliver([macros], repo)
    deliver(paper_figures(args.outdir), repo / PAPER_REPO_FIGURES)
    deliver(tables, repo / PAPER_REPO_TABLES)
    deliver(paper_csvs(args.datadir), repo / PAPER_REPO_DATA)


if __name__ == "__main__":
    main()
