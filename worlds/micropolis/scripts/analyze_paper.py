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
count and both bootstrap intervals. They are not recomputed here: the figures'
own correlate() call is intercepted, so a macro and its figure cannot disagree.
Under --no-extra no figure is drawn, and the same correlations are computed
directly instead.

Figure captions are not generated. They live in the article's own .tex, where
they are edited, and quote the macros defined here for every number in them;
this script defines those numbers and nothing about the prose around them.

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
    TAIL_THRESHOLD,
    Score,
    capability_predictors,
    plot_eci_vs_score,
)
from analyze_continuous import (
    ALL,
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EXCESS,
    MIN_MODELS,
    _correlations_by_row,
    _percentile_interval,
    by_model_id,
    correlate,
    format_band,
)
from gather_paper_data import (
    BATCHING_CSV_NAME,
    BATCHING_RUNS_CSV_NAME,
    BINARY_CSV_NAME,
    CONTINUOUS_CSV_NAME,
    COVERAGE_CSV_NAME,
    MODEL_SCORES_CSV_NAME,
    OUT_DIR,
    RECHECK_CSV_NAME,
    SCALES_CSV_NAME,
    USAGE_CSV_NAME,
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

# The question set's own composition: what ground-truth probabilities the
# 3,000 binary questions actually take, which is the base rate every score
# here is measured against. The article asked for this as "the project plan's
# Figure 1" and the box for it asked for the per-question ground-truth files;
# binary_forecasts.csv already carries q per question as `real_prob`, so it
# needs nothing that is not already gathered.
BANDS_FIG_NAME = "fig_micropolis_bands.pdf"
BANDS_SIZE = (5.5, 2.15)

# The bands q is counted in. Deliberately not ten equal deciles: the set is
# built to put most of its mass below 5% (that is what makes the tail section
# a tail), so equal deciles would show one bar and nine slivers. The edges
# below split that first decile and keep TAIL_THRESHOLD as a boundary, so the
# figure's leftmost bars are exactly the tail section.
BAND_EDGES = [0.0, 0.005, 0.02, TAIL_THRESHOLD, 0.1, 0.25, 0.5, 0.75, 0.95, 1.0]

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
    "batching": "micropolis_batching.tex",
    "batching_models": "micropolis_batching_models.tex",
}

# The questions-per-prompt ablation (appendix D). Three panels: the two binary
# scores against the number of questions a prompt carried, and what each
# setting cost. The pooled line is a mean of per-model means with a PAIRED
# question bootstrap — one shared draw of questions across settings — so the
# difference between two settings is what the interval speaks to.
BATCHING_FIG_NAME = "fig_micropolis_batching.pdf"
BATCHING_SIZE = (5.5, 2.15)
BATCHING_PANELS = [
    (MID_RANGE, "excess_brier", "Excess Brier score", "(a) Mid-range questions"),
    (TAIL, "excess_bits", "Excess bits", "(b) Tail questions"),
]

# Macro names cannot hold digits, so each cap gets a word. Keyed by the
# config's cap, which is stable, where the size a prompt actually carried
# (7--8 under a cap of 8) is what the article prints.
CAP_WORDS = {
    1: "One",
    2: "Two",
    4: "Four",
    8: "Eight",
    16: "Sixteen",
    32: "ThirtyTwo",
    64: "SixtyFour",
    100: "Hundred",
}

# The two probabilities the binary epilogue shows as its example answer block
# ("Q1: 0.65 / Q2: 0.03"). Some models return them verbatim; the appendix
# measures how often. Prompt constants rather than data, so they are named
# here: analyze_paper.py reads only the paper's directory.
EXAMPLE_VALUES = (0.65, 0.03)

# Where the macros land in the article: it \inputs them from data/, beside
# the other generated definitions, not from its root.
PAPER_REPO_MACROS = "data"

# Where the tables land in the article: it \inputs them from data/appendix_tables/.
PAPER_REPO_TABLES = "data/appendix_tables"

# The two cost tables the article shares between worlds. Neither is ours to
# write: StarSim's data/starsim/scripts/build_metadata.py generates them too,
# by reading the file and rewriting only the cells it owns. So this script
# edits them the same way — in place, Micropolis cells only — and the two
# generators compose in either order.
#
# hosting_cost_table.tex gets two new columns. They go BEFORE StarSim's pair,
# because build_metadata.py addresses its own cells as cells[-2:]; appended at
# the end, ours would be the ones it overwrote.
SHARED_TABLES = {
    "hosting": "data/hosting_cost_table.tex",
    "cost_per_item": "data/appendix_tables/cost_per_item.tex",
}

# What the article's cost_per_item rows say Micropolis asked, per eval. The
# item counts are checked against the gathered rows rather than written from
# here, so a config change shows up as an error instead of a stale table.
COST_ROWS = [
    ("binary", "Binary"),
    ("continuous", "Continuous"),
]

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


def read_usage(path: Path) -> list[dict]:
    """model_usage.csv as rows with numbers parsed.

    What the run cost, per model and eval. Like the coverage file this cannot
    be recovered from the forecast rows: a row there is a score, and the call
    that produced it left its cost in a sidecar beside the cached response.
    """
    if not path.exists():
        sys.exit(
            f"[error] {path} not found\n"
            "  rerun scripts/gather_paper_data.py; it writes the per-model"
            " usage the article's cost tables report"
        )
    with path.open(newline="") as f:
        rows = [
            {
                **r,
                "ncalls": int(r["ncalls"]),
                "nprompts": int(r["nprompts"]),
                "questions_per_prompt": int(r["questions_per_prompt"]),
                "cost_usd": float(r["cost_usd"]),
            }
            for r in csv.DictReader(f)
        ]
    if not rows:
        sys.exit(f"[error] {path} holds no rows")
    return rows


def usage_by_eval(usage: list[dict]) -> dict[str, dict[str, float]]:
    """Per-eval totals: calls, cost, and the prompts each model was sent.

    nprompts is the same for every model of an eval — it is a property of the
    corpus and the batch size, not of the model — so it is asserted rather
    than averaged: a spread there would mean the rows came from two different
    runs, and the cost table's "calls per model" column would be a fiction.
    """
    out = {}
    for name in sorted({r["eval"] for r in usage}):
        rows = [r for r in usage if r["eval"] == name]
        prompts = {r["nprompts"] for r in rows}
        if len(prompts) != 1:
            sys.exit(
                f"[error] {USAGE_CSV_NAME}: models of the {name} eval were sent"
                f" different numbers of prompts ({sorted(prompts)})\n"
                "  the rows are from more than one run; rerun"
                " scripts/gather_paper_data.py"
            )
        out[name] = {
            "calls": sum(r["ncalls"] for r in rows),
            "cost": sum(r["cost_usd"] for r in rows),
            "nprompts": prompts.pop(),
            "per_prompt": max(r["questions_per_prompt"] for r in rows),
            "nmodels": len(rows),
        }
    return out


def items_per_model(coverage: list[dict]) -> dict[str, int]:
    """How many questions one model was asked, per eval.

    The denominator of the per-item cost. Taken from what was prompted, not
    from what parsed: a model that returned nothing for a question was still
    charged for it.

    Every model of an eval is asked the same corpus, so a spread here means
    the rows came from more than one run and no single item count describes
    the table's row.
    """
    out = {}
    for name in sorted({r["eval"] for r in coverage}):
        counts = {r["nforecasts"] for r in coverage if r["eval"] == name}
        if len(counts) != 1:
            sys.exit(
                f"[error] {COVERAGE_CSV_NAME}: models of the {name} eval were"
                f" asked different numbers of questions ({sorted(counts)})\n"
                "  rerun scripts/gather_paper_data.py"
            )
        out[name] = counts.pop()
    return out


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
    """The paper's data files, for delivery into the article.

    Taken from the directory by extension rather than from a list of names, so
    a file added to the gathered data reaches the article without this being
    kept in step. These are what every number in the paper was computed from:
    the two forecast files, the per-city scales, the per-model coverage and
    usage, the leaderboard copy the ECI came from, and the cells the
    cross-world combined score fits on.

    versions.txt goes with them: the article names this world's question-set
    version and cites that file for the commit behind it, so the two have to
    travel together.
    """
    return sorted(
        p for p in datadir.iterdir() if p.is_file() and p.suffix in (".csv", ".txt")
    )


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


def city_of_row(row: dict) -> str | None:
    """The city a scored row belongs to.

    The continuous rows carry it as a column; the binary ones encode it in the
    question id, as "{city}_{disasters}_seed{n}_T{turn}_H{horizon}_{qid}", and
    a city name can itself hold an underscore ("med_isle"), so the suffixes
    are stripped rather than the first segment taken — the same rule
    gather_paper_data.city_of applies to scenario ids.
    """
    if row.get("city"):
        return row["city"]
    qid = row.get("question_id")
    if not qid:
        return None
    head = qid.split("_seed", 1)[0]
    return head.removesuffix("_disasters").removesuffix("_nodisasters") or None


def cluster_ci(
    predictor: dict[str, float],
    rows: list[dict],
    score_key: str,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[tuple[float, float] | None, int]:
    """A 95% interval for the correlation, resampling cities rather than questions.

    The article's validation table has a "CI (worlds)" column: for FreeCiv a
    cluster bootstrap over its eight anchor games, since questions drawn from
    one world are not independent of each other. This world's analogue is its
    cities — 15 of them, each replayed 1,000 times — so a city is resampled
    with all of its questions at once.

    Wider than the question bootstrap, since 15 clusters carry less
    information than 3,000 independent draws, and narrower than the model
    bootstrap, which resamples the 24 points the correlation is computed over.
    It is the one to quote for "would this hold on other cities".

    Built on the same (question x model) matrix and the same weight-matrix
    trick as analyze_continuous.correlate, so the only difference from its
    question interval is what gets resampled. Returns (interval, n_cities).
    """
    import numpy as np

    rows = [r for r in rows if r[score_key] is not None]
    questions = sorted({r["question_id"] for r in rows})
    models = sorted({r["model_id"] for r in rows} & set(predictor))
    if len(questions) < 2 or len(models) < MIN_MODELS:
        return None, 0
    qpos = {q: i for i, q in enumerate(questions)}
    mpos = {m: j for j, m in enumerate(models)}
    scores = np.full((len(questions), len(models)), np.nan)
    city_of_q: dict[str, str] = {}
    for r in rows:
        if r["model_id"] in mpos:
            scores[qpos[r["question_id"]], mpos[r["model_id"]]] = r[score_key]
        city = city_of_row(r)
        if city:
            city_of_q[r["question_id"]] = city
    if len(city_of_q) != len(questions):
        return None, 0
    cities = sorted(set(city_of_q.values()))
    if len(cities) < 2:
        return None, len(cities)
    present = ~np.isnan(scores)
    x = np.array([predictor[m] for m in models])

    # Which questions belong to each city, as a (cities x questions) indicator,
    # so a resample of cities becomes a weight per question in one product.
    cpos = {c: i for i, c in enumerate(cities)}
    member = np.zeros((len(cities), len(questions)))
    for q, city in city_of_q.items():
        member[cpos[city], qpos[q]] = 1.0

    nc = len(cities)
    draws = np.random.default_rng(seed).integers(0, nc, (resamples, nc))
    city_w = np.stack([np.bincount(d, minlength=nc) for d in draws]).astype(float)
    weights = city_w @ member
    with np.errstate(invalid="ignore", divide="ignore"):
        means = (weights @ np.where(present, scores, 0.0)) / (weights @ present)
    rho_c, _ = _correlations_by_row(np.broadcast_to(x, means.shape), means)
    return _percentile_interval(rho_c, resamples), nc


def band_of(q: float) -> int:
    """Which BAND_EDGES bucket a ground-truth probability falls in."""
    for i in range(len(BAND_EDGES) - 1):
        if q < BAND_EDGES[i + 1]:
            return i
    return len(BAND_EDGES) - 2


def band_label(i: int) -> str:
    """A band's axis label, as a percentage range."""

    def pct(x: float) -> str:
        return f"{x * 100:g}"

    return f"{pct(BAND_EDGES[i])}--{pct(BAND_EDGES[i + 1])}"


def band_counts(binary: list[dict]) -> tuple[list[list[int]], int]:
    """Questions per (band, horizon), and the questions per model.

    Counted over one model's rows, not all of them: every model is asked the
    same 3,000 questions, so counting all 24 would report the same set 24
    times. The model is the one with a complete set of rows, since a model
    that failed to parse an answer has no row for that question and would
    undercount the set it was nonetheless asked.
    """
    per_model: dict[str, list[dict]] = {}
    for r in binary:
        per_model.setdefault(r["model_id"], []).append(r)
    if not per_model:
        return [], 0
    rows = max(per_model.values(), key=len)
    counts = [[0] * len(HORIZONS) for _ in range(len(BAND_EDGES) - 1)]
    for r in rows:
        if r["horizon"] in HORIZONS:
            counts[band_of(r["real_prob"])][HORIZONS.index(r["horizon"])] += 1
    return counts, len(rows)


def draw_bands_figure(path: Path, binary: list[dict]) -> Path | None:
    """The question set's composition by ground-truth probability.

    Left: how many of the questions fall in each probability band, stacked by
    horizon. Right: the same counts as each horizon's share of its own
    questions, which is what shows whether the composition shifts as the
    horizon lengthens — a longer window makes rare events less rare, so the
    tail share should fall.

    The tail/mid-range boundary is drawn on both panels, because the section
    split every other figure here conditions on is exactly that line.
    """
    import matplotlib

    matplotlib.use("pgf")
    import matplotlib.pyplot as plt
    import numpy as np

    counts, n_questions = band_counts(binary)
    if not n_questions:
        print(f"[skipped] {BANDS_FIG_NAME}: no binary rows")
        return None
    grid = np.array(counts, dtype=float)
    nbands = grid.shape[0]
    # Where the tail ends: the first edge at or past the threshold.
    split = BAND_EDGES.index(TAIL_THRESHOLD)

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
        fig, axes = plt.subplots(1, 2, figsize=BANDS_SIZE, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.04, h_pad=0.02, wspace=0.03)
        xs = np.arange(nbands)
        # One grey per horizon, light to dark, so the stack reads as an
        # ordered variable rather than as four unrelated categories.
        shades = [str(v) for v in np.linspace(0.78, 0.30, len(HORIZONS))]

        ax = axes[0]
        bottom = np.zeros(nbands)
        for j, h in enumerate(HORIZONS):
            ax.bar(
                xs,
                grid[:, j],
                bottom=bottom,
                color=shades[j],
                edgecolor="white",
                lw=0.3,
                width=0.82,
                label=f"{h.rstrip('y')} years",
            )
            bottom += grid[:, j]
        ax.set_ylabel("Questions")
        ax.set_title(
            rf"Composition of the {n_questions:,} binary questions", fontsize=7
        )
        ax.legend(fontsize=5, frameon=False, loc="upper center", ncols=2)
        ax.set_ylim(0, bottom.max() * 1.30)

        ax = axes[1]
        # Column-normalized: each horizon's bars sum to 100%, so the panel
        # compares composition rather than counts.
        share = 100.0 * grid / grid.sum(axis=0, keepdims=True)
        width = 0.82 / len(HORIZONS)
        for j, h in enumerate(HORIZONS):
            ax.bar(
                xs + (j - (len(HORIZONS) - 1) / 2) * width,
                share[:, j],
                color=shades[j],
                edgecolor="white",
                lw=0.2,
                width=width,
                label=f"{h.rstrip('y')} years",
            )
        ax.set_ylabel(r"Share of the horizon's questions (\%)")
        ax.set_title("Composition by forecast horizon", fontsize=7)
        ax.set_ylim(0, share.max() * 1.30)

        for ax in axes:
            # Between the last tail band and the first mid-range one.
            ax.axvline(split - 0.5, color=EXTREME_COLOR, lw=0.7, ls=":")
            ax.set_xticks(xs)
            ax.set_xticklabels([band_label(i) for i in range(nbands)], rotation=90)
            ax.set_xlabel(r"Ground-truth probability $p$ (\%)")
            ax.spines[["top", "right"]].set_visible(False)
            ax.margins(x=0.02)

        # Named once, on the right panel: the left one's legend already
        # occupies the space above the split.
        axes[1].text(
            split - 0.62,
            share.max() * 1.22,
            r"tail $\mid$ mid-range",
            fontsize=5,
            color=EXTREME_COLOR,
            ha="right",
            va="top",
        )
        path.parent.mkdir(parents=True, exist_ok=True)
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


def cost_lines(totals: dict[str, dict[str, float]], items: dict[str, int]) -> list[str]:
    r"""\MPDCost* : what the run cost, per eval and in total.

    Dollars to the cent and the grand total too, because the article quotes
    all three in prose. The per-item figure is in cents, as the shared cost
    table reports it, and is computed from the same totals so the prose and
    the table cannot disagree.
    """
    pre = MACRO_PREFIX
    lines = [
        "% What the run cost, from the usage sidecars beside the cached",
        "% responses (model_usage.csv). Dollars per model summed over the",
        "% panel; CentsPerItem divides by the items one model was asked.",
    ]
    for name, t in sorted(totals.items()):
        stem = name.capitalize()
        n = items[name]
        cents = 100.0 * t["cost"] / (t["nmodels"] * n) if t["nmodels"] and n else 0.0
        per_prompt = n // t["nprompts"] if t["nprompts"] else 0
        lines += [
            f"\\newcommand{{\\{pre}Cost{stem}}}{{{t['cost']:.2f}}}",
            f"\\newcommand{{\\{pre}Calls{stem}}}{{{t['calls']:,}}}",
            f"\\newcommand{{\\{pre}Prompts{stem}}}{{{t['nprompts']:,}}}",
            f"\\newcommand{{\\{pre}Items{stem}}}{{{n:,}}}",
            f"\\newcommand{{\\{pre}Cents{stem}}}{{{cents:.2f}}}",
            f"\\newcommand{{\\{pre}PerPrompt{stem}}}{{{per_prompt}}}",
        ]
    total = sum(t["cost"] for t in totals.values())
    calls = sum(t["calls"] for t in totals.values())
    lines += [
        f"\\newcommand{{\\{pre}CostTotal}}{{{total:.2f}}}",
        f"\\newcommand{{\\{pre}CallsTotal}}{{{calls:,}}}",
        "",
    ]
    return lines


def city_ci_lines(
    binary: list[dict], continuous: list[dict], models: list[str]
) -> list[str]:
    r"""\MPDRho*CICities : each headline correlation's city cluster interval.

    The article's validation table carries a "CI (worlds)" column, which for
    FreeCiv is a cluster bootstrap over its eight anchor games. This world's
    rows had the question interval repeated there, which is not the same
    claim: it asks how much the correlation depends on which questions were
    drawn, where the column asks how much it depends on which worlds were.
    These are that column, resampling the 15 cities.
    """
    pre = MACRO_PREFIX
    predictor = by_model_id(eci_by_name_of(models), models)
    mid = [r for r in binary if r["section"] == MID_RANGE]
    tail = [r for r in binary if r["section"] == TAIL]
    lines = [
        "% Each headline correlation's 95% interval from a cluster bootstrap",
        "% over this world's cities, resampling a city with all its questions:",
        "% the article's 'CI (worlds)' column, FreeCiv's anchor-game bootstrap",
        "% applied to the Micropolis cities. Wider than the question",
        "% interval and narrower than the model one; it is the interval to",
        "% quote for whether a correlation would hold on other cities.",
    ]
    ncities = 0
    for macro, rows, key in [
        ("Binary", mid, "excess_brier"),
        ("Tail", tail, "excess_bits"),
        ("Continuous", continuous, "excess_ncrps"),
    ]:
        ci, n = cluster_ci(predictor, rows, key)
        ncities = max(ncities, n)
        if ci is None:
            print(f"[warn] no city interval for \\{pre}Rho{macro}CICities")
            continue
        band = macro_band(adjusted_band(ci))
        lines.append(f"\\newcommand{{\\{pre}Rho{macro}CICities}}{{{band}}}")
        print(f"  95% CI cities, {macro:<10} {format_band(adjusted_band(ci), 0)}")
    if ncities:
        lines.append(f"\\newcommand{{\\{pre}NCities}}{{{ncities}}}")
    return lines + [""]


def band_lines(binary: list[dict]) -> list[str]:
    r"""\MPDBand* : how the question set's ground truth is distributed.

    The share below the tail threshold and in the top band are what the
    composition paragraph quotes; the per-horizon tail shares are what let it
    state the direction the composition moves in as the window lengthens.
    """
    pre = MACRO_PREFIX
    counts, total = band_counts(binary)
    if not total:
        return []
    import numpy as np

    grid = np.array(counts, dtype=float)
    split = BAND_EDGES.index(TAIL_THRESHOLD)
    tail = grid[:split].sum()
    lines = [
        "% The binary set's composition by ground-truth probability p, over",
        "% the questions one model is asked (every model is asked the same).",
        f"\\newcommand{{\\{pre}BandNQuestions}}{{{int(total):,}}}",
        f"\\newcommand{{\\{pre}BandTailShare}}{{{100.0 * tail / total:.1f}}}",
        f"\\newcommand{{\\{pre}BandTopShare}}{{{100.0 * grid[-1].sum() / total:.1f}}}",
        f"\\newcommand{{\\{pre}BandTailPct}}{{{100 * TAIL_THRESHOLD:g}}}",
    ]
    # The tail's share within each horizon, shortest and longest: the two the
    # article contrasts.
    per_h = 100.0 * grid[:split].sum(axis=0) / grid.sum(axis=0)
    for h, share in zip(HORIZONS, per_h):
        lines.append(
            f"\\newcommand{{\\{pre}BandTailShare{HORIZON_WORDS[h]}}}{{{share:.1f}}}"
        )
    return lines + [""]


def horizon_years_lines() -> list[str]:
    r"""\MPDHorizonYears: the forecast horizons in game years, as prose.

    The by-horizon figure's caption lists them, so the list comes from
    HORIZONS rather than being retyped in the article.
    """
    pre = MACRO_PREFIX
    years = [h.removesuffix("y") for h in HORIZONS]
    if len(years) > 1:
        joined = ", ".join(years[:-1]) + f" and {years[-1]}"
    else:
        joined = years[0]
    return [
        "% The forecast horizons in game years, for the caption that lists them.",
        f"\\newcommand{{\\{pre}HorizonYears}}{{{joined}}}",
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


def write_shared_tables(
    repo: Path, usage: list[dict], totals: dict, items: dict, names: dict[str, str]
) -> list[Path]:
    """Refill this world's cells of the article's two shared cost tables.

    Read-modify-write, in the article's own checkout: these files are shared
    with StarSim's generator, so each side edits only its own cells and leaves
    the rest of the bytes alone. Nothing is written when a table's cells are
    already right, so a rerun that changes no number leaves no diff.
    """
    written = []
    for key, edit in [
        ("hosting", lambda t: edit_hosting(t, usage, names)),
        ("cost_per_item", lambda t: edit_cost_per_item(t, totals, items)),
    ]:
        path = repo / SHARED_TABLES[key]
        if not path.exists():
            print(f"[warn] {path} not in the article; its cells not filled")
            continue
        before = path.read_text()
        after = edit(before)
        if after == before:
            print(f"Unchanged {path}")
            continue
        path.write_text(after)
        print(f"Filled   {path}")
        written.append(path)
    return written


def read_batching(path: Path) -> list[dict]:
    """batching_forecasts.csv: the paper's binary rows with a setting prepended."""
    rows = read_rows(path)
    for r in rows:
        r["cap"] = int(r["cap"])
        r["questions_per_prompt"] = float(r["questions_per_prompt"])
        r["prompts_per_model"] = int(r["prompts_per_model"])
    return rows


def read_batching_runs(path: Path) -> list[dict]:
    """batching_runs.csv: per (setting, model) what was asked, parsed and paid."""
    if not path.exists():
        sys.exit(
            f"[error] {path} not found\n"
            "  rerun scripts/gather_paper_data.py; it writes the ablation's"
            " per-run rows"
        )
    numeric = {
        "cap": int,
        "questions_per_prompt": float,
        "prompts_per_model": int,
        "nforecasts": int,
        "nvalid": int,
        "ncalls": int,
        "cost_usd": float,
        "latency_ms_sum": float,
        "latency_ms_p50": float,
    }
    with path.open(newline="") as f:
        rows = [
            {
                k: (numeric[k](v) if k in numeric and v != "" else v)
                for k, v in r.items()
            }
            for r in csv.DictReader(f)
        ]
    if not rows:
        sys.exit(f"[error] {path} holds no rows")
    return rows


def read_recheck(path: Path) -> list[dict] | None:
    """gpt5_check_forecasts.csv, or None when the rerun has not been gathered.

    Unparsed forecasts are rows with an empty forecast here, unlike every
    other forecast file: the count of them is the point.
    """
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            r["forecast"] = float(r["forecast"]) if r["forecast"] != "" else None
            r["real_prob"] = float(r["real_prob"])
            r["has_block"] = int(r["has_block"])
            r["model_id"] = r.pop("model")
            rows.append(r)
    return rows


def batching_settings(rows: list[dict]) -> list[dict]:
    """The ablation's settings, smallest prompt first.

    Each is {cap, qpp, prompts, setting}; qpp is the mean number of questions
    a prompt actually carried, which is what every axis and table prints.
    """
    seen = {}
    for r in rows:
        seen[r["cap"]] = {
            "cap": r["cap"],
            "qpp": r["questions_per_prompt"],
            "prompts": r["prompts_per_model"],
            "setting": r["setting"],
        }
    return sorted(seen.values(), key=lambda s: s["qpp"])


def size_label(qpp: float, tex: bool = False) -> str:
    """A prompt size for an axis or a table: "4", or "7--8" for an uneven split."""
    import math

    if abs(qpp - round(qpp)) < 1e-6:
        return f"{round(qpp)}"
    dash = "--" if tex else "–"
    return f"{math.floor(qpp)}{dash}{math.ceil(qpp)}"


def cap_word(cap: int) -> str:
    """The macro-name word for a cap, failing loudly on one not in the table."""
    if cap not in CAP_WORDS:
        sys.exit(
            f"[error] no macro word for a cap of {cap}; add it to CAP_WORDS in"
            " analyze_paper.py"
        )
    return CAP_WORDS[cap]


def production_setting(settings: list[dict], qpp: float) -> dict | None:
    """The ablation setting whose prompts are the paper's own run, if any.

    The main run and the ablation split a snapshot's questions the same way,
    so a setting that carried the same number per prompt asked the very same
    prompts and, the cache being shared, got the very same responses.
    """
    for s in settings:
        if abs(s["qpp"] - qpp) < 0.5:
            return s
    return None


def _score_matrix(rows: list[dict], key: str, questions: list[str], models: list[str]):
    """rows as a (questions x models) array, NaN where a model has no score."""
    import numpy as np

    qpos = {q: i for i, q in enumerate(questions)}
    mpos = {m: j for j, m in enumerate(models)}
    a = np.full((len(questions), len(models)), np.nan)
    for r in rows:
        if r["question_id"] in qpos and r["model_id"] in mpos and r[key] != "":
            a[qpos[r["question_id"]], mpos[r["model_id"]]] = r[key]
    return a


def batching_bootstrap(
    rows: list[dict],
    settings: list[dict],
    section: str,
    key: str,
    models: list[str],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[int, dict[str, float]]:
    """Pooled score per setting with paired question-bootstrap intervals.

    The pooled score is the mean over models of each model's mean over
    questions. One draw of question indices is shared by every setting, so
    the interval on the difference between two settings reflects only which
    questions were asked — the eight settings answered the same 1,000.

    Returns, per cap: point, lo, hi, and delta/dlo/dhi against the smallest
    setting (zero for that setting itself).
    """
    import numpy as np

    sec = [r for r in rows if r["section"] == section]
    questions = sorted({r["question_id"] for r in sec})
    nq = len(questions)
    draws = np.random.default_rng(seed).integers(0, nq, (resamples, nq))
    weights = np.zeros((resamples, nq))
    for i, d in enumerate(draws):
        weights[i] = np.bincount(d, minlength=nq)

    pooled = {}
    points = {}
    for s in settings:
        a = _score_matrix(
            [r for r in sec if r["cap"] == s["cap"]], key, questions, models
        )
        present = ~np.isnan(a)
        with np.errstate(invalid="ignore", divide="ignore"):
            means = (weights @ np.where(present, a, 0.0)) / (weights @ present)
        pooled[s["cap"]] = np.nanmean(means, axis=1)
        points[s["cap"]] = float(np.nanmean(np.nanmean(a, axis=0)))

    base = settings[0]["cap"]
    out = {}
    for s in settings:
        cap = s["cap"]
        lo, hi = np.percentile(pooled[cap], [2.5, 97.5])
        d = pooled[cap] - pooled[base]
        dlo, dhi = np.percentile(d, [2.5, 97.5])
        out[cap] = {
            "point": points[cap],
            "lo": float(lo),
            "hi": float(hi),
            "delta": points[cap] - points[base],
            "dlo": float(dlo),
            "dhi": float(dhi),
        }
    return out


def batching_rho(
    rows: list[dict], settings: list[dict], section: str, key: str, models: list[str]
) -> dict[int, object]:
    """ECI correlation per setting, on the section's per-model means."""
    predictor = by_model_id(eci_by_name_of(models), models)
    out = {}
    for s in settings:
        slice_rows = [
            r for r in rows if r["section"] == section and r["cap"] == s["cap"]
        ]
        out[s["cap"]] = correlate("", predictor, slice_rows, key, models, ALL)
    return out


def per_model_mean(rows: list[dict], cap: int, section: str, key: str, model: str):
    return _mean_of(
        [r for r in rows if r["cap"] == cap and r["section"] == section],
        model,
        None,
        key,
    )


def run_stat(runs: list[dict], cap: int, model: str | None, field: str):
    """One run field for a (cap, model), or its mean over models when model is None."""
    vals = [
        r[field]
        for r in runs
        if r["cap"] == cap and (model is None or r["model"] == model) and r[field] != ""
    ]
    return sum(vals) / len(vals) if vals else None


def parse_rate(runs: list[dict], cap: int, model: str) -> float | None:
    for r in runs:
        if r["cap"] == cap and r["model"] == model and r["nforecasts"]:
            return 100.0 * r["nvalid"] / r["nforecasts"]
    return None


def example_share(rows: list[dict], cap: int, model: str, value: float) -> float:
    """Share (%) of a model's parsed answers at a cap that equal `value` exactly."""
    mine = [r for r in rows if r["cap"] == cap and r["model_id"] == model]
    if not mine:
        return 0.0
    return 100.0 * sum(1 for r in mine if abs(r["forecast"] - value) < 1e-9) / len(mine)


def without_value(rows: list[dict], value: float) -> list[dict]:
    return [r for r in rows if abs(r["forecast"] - value) >= 1e-9]


def pooled_mean(rows: list[dict], cap: int, section: str, key: str, models: list[str]):
    """Mean over models of each model's mean: the ablation's pooled score."""
    vals = [per_model_mean(rows, cap, section, key, m) for m in models]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def draw_batching_figure(
    path: Path,
    rows: list[dict],
    runs: list[dict],
    settings: list[dict],
    models: list[str],
    boot: dict[str, dict[int, dict[str, float]]],
    production: dict | None,
) -> Path | None:
    """Scores and cost against questions per prompt; see BATCHING_FIG_NAME."""
    import matplotlib

    matplotlib.use("pgf")
    import matplotlib.pyplot as plt
    import numpy as np

    if not settings:
        print(f"[skipped] {BATCHING_FIG_NAME}: no ablation rows")
        return None
    xs = np.array([s["qpp"] for s in settings])
    caps = [s["cap"] for s in settings]
    labels = [size_label(s["qpp"]) for s in settings]

    with plt.rc_context(
        {
            "pgf.texsystem": "pdflatex",
            "text.usetex": True,
            "font.family": "serif",
            "pgf.rcfonts": False,
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.0,
            "ytick.major.size": 2.0,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=BATCHING_SIZE, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.04, h_pad=0.02, wspace=0.04)

        for ax, (section, key, ylabel, title) in zip(axes[:2], BATCHING_PANELS):
            b = boot[section]
            for m in models:
                ys = [per_model_mean(rows, c, section, key, m) for c in caps]
                if all(y is not None for y in ys):
                    ax.plot(xs, ys, color="0.78", lw=0.6, zorder=1)
            ax.fill_between(
                xs,
                [b[c]["lo"] for c in caps],
                [b[c]["hi"] for c in caps],
                color=POINT_COLOR,
                alpha=0.15,
                lw=0,
                zorder=2,
            )
            ax.plot(
                xs,
                [b[c]["point"] for c in caps],
                color=POINT_COLOR,
                lw=1.2,
                marker="o",
                ms=2.8,
                zorder=3,
                label=f"Mean of {len(models)} models",
            )
            if production is not None:
                ax.plot(
                    [production["qpp"]],
                    [b[production["cap"]]["point"]],
                    marker="s",
                    ms=5,
                    mfc="none",
                    mec=EXTREME_COLOR,
                    mew=1.0,
                    ls="none",
                    zorder=4,
                    label="The paper's run",
                )
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=7)
            ax.set_ylim(bottom=0)

        # Cost per model, log-log, against the 1/n line a fixed per-prompt
        # cost would give: the gap past ~8 is the answer text, which does not
        # shrink with the shared report.
        ax = axes[2]
        for m in models:
            ys = [run_stat(runs, c, m, "cost_usd") for c in caps]
            if all(y for y in ys):
                ax.plot(xs, ys, color="0.78", lw=0.6, zorder=1)
        cost = [run_stat(runs, c, None, "cost_usd") for c in caps]
        ax.plot(
            xs,
            cost,
            color=POINT_COLOR,
            lw=1.2,
            marker="o",
            ms=2.8,
            zorder=3,
            label=f"Mean of {len(models)} models",
        )
        ax.plot(
            xs,
            [cost[0] * xs[0] / x for x in xs],
            color="0.45",
            lw=0.8,
            ls="--",
            zorder=2,
            label=r"$\propto 1/n$",
        )
        if production is not None:
            ax.plot(
                [production["qpp"]],
                [run_stat(runs, production["cap"], None, "cost_usd")],
                marker="s",
                ms=5,
                mfc="none",
                mec=EXTREME_COLOR,
                mew=1.0,
                ls="none",
                zorder=4,
            )
        ax.set_yscale("log")
        ax.set_ylabel(r"Cost per model (\$)")
        ax.set_title("(c) Cost", fontsize=7)

        for ax in axes:
            ax.set_xscale("log")
            ax.set_xticks(xs)
            # Rotated: "7–8" and "14–15" sit a log-step apart and collide flat.
            ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
            ax.minorticks_off()
            ax.set_xlabel("Questions per prompt")
            ax.spines[["top", "right"]].set_visible(False)
        # Where the data is not: the mid-range panel is empty below its lowest
        # model, the cost panel below its cheapest at small n.
        axes[0].legend(fontsize=5, frameon=False, loc="lower left")
        axes[2].legend(fontsize=5, frameon=False, loc="lower left")
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path)
        plt.close(fig)
    return path


def batching_table(
    rows: list[dict],
    runs: list[dict],
    settings: list[dict],
    models: list[str],
    boot: dict[str, dict[int, dict[str, float]]],
    rho: dict[str, dict[int, object]],
    production: dict | None,
) -> str:
    """Per setting: size, prompts, the two pooled scores, parse rate, cost, rho."""
    lines = [
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{rrrrrrrrr}",
        r"\toprule",
        (
            r"\multicolumn{2}{c}{Per model} & \multicolumn{2}{c}{Pooled score}"
            r" & Parsed & \multicolumn{2}{c}{Cost per model} & \multicolumn{2}{c}{$\rho$ with ECI} \\"
        ),
        r"\cmidrule(lr){1-2}\cmidrule(lr){3-4}\cmidrule(lr){6-7}\cmidrule(lr){8-9}",
        r"Q/prompt & Prompts & Mid-range & Tail & (\%) & \$ & \textcent/question & Mid-range & Tail \\",
        r"\midrule",
    ]
    for s in settings:
        cap = s["cap"]
        size = size_label(s["qpp"], tex=True)
        if production is not None and cap == production["cap"]:
            size += r"$^\dagger$"
        parsed = min(parse_rate(runs, cap, m) or 0.0 for m in models)
        cost = run_stat(runs, cap, None, "cost_usd")
        nq = run_stat(runs, cap, None, "nforecasts")
        r_mid, r_tail = rho[MID_RANGE][cap], rho[TAIL][cap]
        lines.append(
            " & ".join(
                [
                    size,
                    f"{s['prompts']:,}",
                    cell(boot[MID_RANGE][cap]["point"], "{:.4f}"),
                    cell(boot[TAIL][cap]["point"], "{:.3f}"),
                    f"{parsed:.1f}",
                    cell(cost, "{:.2f}"),
                    cell(100.0 * cost / nq if cost and nq else None, "{:.2f}"),
                    cell(adjusted(r_mid.rho) if r_mid else None, "{:.2f}"),
                    cell(adjusted(r_tail.rho) if r_tail else None, "{:.2f}"),
                ]
            )
            + r" \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return table_file(
        lines,
        "Questions-per-prompt ablation, per setting. Sources: batching_forecasts.csv,"
        " batching_runs.csv. Pooled score = mean of per-model means; Parsed = the"
        " lowest parse rate over the models; rho sign-adjusted.",
    )


def batching_models_table(
    rows: list[dict], settings: list[dict], models: list[str], names: dict[str, str]
) -> str:
    """Per model x setting: mid-range excess Brier, then tail excess bits."""
    ncol = len(settings)
    head = " & ".join(size_label(s["qpp"], tex=True) for s in settings)
    lines = [
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{l" + "r" * ncol + "}",
        r"\toprule",
        rf"Model & \multicolumn{{{ncol}}}{{c}}{{Questions per prompt}} \\",
        rf"\cmidrule(lr){{2-{ncol + 1}}}",
        f" & {head} " + r"\\",
    ]
    for section, key, label, fmt in [
        (
            MID_RANGE,
            "excess_brier",
            "Mid-range questions, excess Brier score",
            "{:.3f}",
        ),
        (TAIL, "excess_bits", "Tail questions, excess bits", "{:.3f}"),
    ]:
        lines += [
            r"\midrule",
            rf"\multicolumn{{{ncol + 1}}}{{l}}{{\emph{{{label}}}}} \\",
        ]
        for m in sorted(models, key=lambda m: -(eci_of(m) or 0)):
            vals = [per_model_mean(rows, s["cap"], section, key, m) for s in settings]
            lines.append(
                " & ".join([tex_escape(names.get(m, m)), *(cell(v, fmt) for v in vals)])
                + r" \\"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    return table_file(
        lines,
        "Questions-per-prompt ablation, per model. Source: batching_forecasts.csv."
        " Models ordered by ECI.",
    )


def batching_lines(
    rows: list[dict],
    runs: list[dict],
    settings: list[dict],
    models: list[str],
    names: dict[str, str],
    boot: dict[str, dict[int, dict[str, float]]],
    rho: dict[str, dict[int, object]],
    production: dict | None,
    recheck: list[dict] | None,
) -> list[str]:
    r"""\MPDBatch* : every number the questions-per-prompt appendix quotes."""
    pre = MACRO_PREFIX
    nc = lambda name, value: f"\\newcommand{{\\{pre}Batch{name}}}{{{value}}}"
    first, last = settings[0], settings[-1]
    lines = [
        "% The questions-per-prompt ablation (appendix D). Pooled scores are",
        "% means of per-model means; Delta* are paired against the smallest",
        "% prompt, with the shared question draw's 95% interval; Rho* are",
        "% sign-adjusted as every other correlation here.",
    ]
    sec_rows = {
        MID_RANGE: [r for r in rows if r["section"] == MID_RANGE],
        TAIL: [r for r in rows if r["section"] == TAIL],
    }
    one = [r for r in rows if r["cap"] == first["cap"]]
    nq = len({r["question_id"] for r in rows})
    # A snapshot's questions, which is what the batcher splits. The largest
    # cap holds a whole snapshot in one prompt, so its prompt count is the
    # number of snapshots asked and nq over that is the group's size.
    nsnapshots = last["prompts"] if last["cap"] >= last["qpp"] else 0
    group = nq / nsnapshots if nsnapshots else 0
    # The first setting whose cap does not divide a snapshot evenly, so the
    # caption's example of two prompt sizes is always one the run produced.
    uneven = next(
        (s_ for s_ in settings if abs(s_["qpp"] - round(s_["qpp"])) > 1e-6), None
    )
    lines += [
        nc("NModels", len(models)),
        nc("NQuestions", f"{nq:,}"),
        nc("NMid", f"{len({r['question_id'] for r in sec_rows[MID_RANGE]}):,}"),
        nc("NTail", f"{len({r['question_id'] for r in sec_rows[TAIL]}):,}"),
        nc("NSettings", len(settings)),
        nc("SmallestSize", size_label(first["qpp"], tex=True)),
        nc("LargestSize", size_label(last["qpp"], tex=True)),
        nc("SnapshotQuestions", f"{group:.0f}" if group else "---"),
        nc(
            "UnevenExample",
            size_label(uneven["qpp"], tex=True) if uneven else "---",
        ),
        nc("CostTotal", f"{sum(r['cost_usd'] for r in runs):.2f}"),
    ]
    if production is not None:
        lines += [
            nc("ProductionSize", size_label(production["qpp"], tex=True)),
            nc("ProductionCap", production["cap"]),
            nc(
                "ProductionPrompts",
                f"{production['prompts'] / nsnapshots:.0f}" if nsnapshots else "---",
            ),
        ]
    # Per setting.
    for s in settings:
        cap, w = s["cap"], cap_word(s["cap"])
        cost = run_stat(runs, cap, None, "cost_usd")
        nf = run_stat(runs, cap, None, "nforecasts")
        parsed = min(parse_rate(runs, cap, m) or 0.0 for m in models)
        lat = run_stat(runs, cap, None, "latency_ms_sum")
        lines += [
            nc(f"Size{w}", size_label(s["qpp"], tex=True)),
            nc(f"Prompts{w}", f"{s['prompts']:,}"),
            nc(f"Mid{w}", f"{boot[MID_RANGE][cap]['point']:.4f}"),
            nc(f"Tail{w}", f"{boot[TAIL][cap]['point']:.3f}"),
            nc(f"DeltaMid{w}", f"${boot[MID_RANGE][cap]['delta']:+.4f}$"),
            nc(
                f"DeltaMid{w}CI",
                macro_band((boot[MID_RANGE][cap]["dlo"], boot[MID_RANGE][cap]["dhi"])),
            ),
            nc(f"DeltaTail{w}", f"${boot[TAIL][cap]['delta']:+.3f}$"),
            nc(
                f"DeltaTail{w}CI",
                macro_band((boot[TAIL][cap]["dlo"], boot[TAIL][cap]["dhi"])),
            ),
            nc(f"Parse{w}", f"{parsed:.1f}"),
            nc(f"Cost{w}", f"{cost:.2f}" if cost is not None else "---"),
            nc(f"Cents{w}", f"{100 * cost / nf:.2f}" if cost and nf else "---"),
            nc(f"LatencyMin{w}", f"{lat / 60000:.0f}" if lat else "---"),
        ]
        for section, tag in ((MID_RANGE, "Mid"), (TAIL, "Tail")):
            c = rho[section][cap]
            lines += [
                nc(f"Rho{tag}{w}", f"{adjusted(c.rho):.2f}" if c else "---"),
                nc(
                    f"Rho{tag}{w}CIModels",
                    macro_band(adjusted_band(c.rho_models)) if c else "---",
                ),
            ]
    # The mechanism: mean (forecast - p) on tail questions, and the Pearson
    # correlation between a model's forecasts and the ground truth over all
    # its questions, each a mean over models. Overshoot that shrinks with the
    # prompt while discrimination does not is a calibration effect.
    import numpy as np

    for s_ in settings:
        w = cap_word(s_["cap"])
        biases, corrs = [], []
        for m in models:
            mine = [r for r in rows if r["cap"] == s_["cap"] and r["model_id"] == m]
            tail_m = [r for r in mine if r["section"] == TAIL]
            if tail_m:
                biases.append(
                    sum(r["forecast"] - r["real_prob"] for r in tail_m) / len(tail_m)
                )
            if len(mine) > 2:
                f = np.array([r["forecast"] for r in mine])
                q = np.array([r["real_prob"] for r in mine])
                if f.std() > 0 and q.std() > 0:
                    corrs.append(float(np.corrcoef(f, q)[0, 1]))
        lines += [
            nc(
                f"TailBias{w}",
                f"${sum(biases) / len(biases):+.3f}$" if biases else "---",
            ),
            nc(f"Discrim{w}", f"{sum(corrs) / len(corrs):.2f}" if corrs else "---"),
        ]
    discrims = [
        float(v.split("}{")[1].rstrip("}"))
        for v in lines
        if "BatchDiscrim" in v and not v.endswith("{---}")
    ]
    if discrims:
        lines += [
            nc("DiscrimMin", f"{min(discrims):.2f}"),
            nc("DiscrimMax", f"{max(discrims):.2f}"),
        ]
    # Across settings.
    for section, tag in ((MID_RANGE, "Mid"), (TAIL, "Tail")):
        rhos = [
            adjusted(rho[section][s["cap"]].rho)
            for s in settings
            if rho[section][s["cap"]]
        ]
        lines += [
            nc(f"Rho{tag}Min", f"{min(rhos):.2f}"),
            nc(f"Rho{tag}Max", f"{max(rhos):.2f}"),
        ]
    c1, cN = (
        run_stat(runs, first["cap"], None, "cost_usd"),
        run_stat(runs, last["cap"], None, "cost_usd"),
    )
    l1, lN = (
        run_stat(runs, first["cap"], None, "latency_ms_sum"),
        run_stat(runs, last["cap"], None, "latency_ms_sum"),
    )
    p1 = run_stat(runs, first["cap"], None, "latency_ms_p50")
    pN = run_stat(runs, last["cap"], None, "latency_ms_p50")
    lines += [
        nc("CostRatio", f"{c1 / cN:.0f}" if c1 and cN else "---"),
        nc("LatencyRatio", f"{l1 / lN:.0f}" if l1 and lN else "---"),
        nc("LatencyMedianSmallest", f"{p1 / 1000:.0f}" if p1 else "---"),
        nc("LatencyMedianLargest", f"{pN / 1000:.0f}" if pN else "---"),
    ]

    # The example-value effect. The model that copies the example most at the
    # smallest prompt, and the runner-up, are found rather than named.
    ex = EXAMPLE_VALUES[0]
    shares = sorted(
        ((example_share(rows, first["cap"], m, ex), m) for m in models), reverse=True
    )
    anchor, copier = shares[0][1], shares[1][1]
    second = settings[1]
    beyond = [s["cap"] for s in settings[2:]]
    lines += [
        "% The epilogue's example answers, returned verbatim.",
        nc("ExampleValue", f"{ex:g}"),
        nc("ExampleValueTwo", f"{EXAMPLE_VALUES[1]:g}"),
        nc("AnchorModel", tex_escape(names.get(anchor, anchor))),
        nc("AnchorShareSmallest", f"{shares[0][0]:.1f}"),
        nc(
            "AnchorShareSecond", f"{example_share(rows, second['cap'], anchor, ex):.1f}"
        ),
        nc(
            "AnchorShareBeyondMax",
            f"{max(example_share(rows, c, anchor, ex) for c in beyond):.1f}",
        ),
        nc(
            "AnchorTailCountSmallest",
            sum(
                1
                for r in one
                if r["model_id"] == anchor
                and r["section"] == TAIL
                and abs(r["forecast"] - ex) < 1e-9
            ),
        ),
        nc("CopierModel", tex_escape(names.get(copier, copier))),
        nc("CopierShareSmallest", f"{shares[1][0]:.1f}"),
        nc(
            "CopierShareSecond", f"{example_share(rows, second['cap'], copier, ex):.1f}"
        ),
        nc(
            "CopierSecondValueShareSecond",
            f"{example_share(rows, second['cap'], copier, EXAMPLE_VALUES[1]):.1f}",
        ),
        nc(
            "CopierShareBeyondMax",
            f"{max(example_share(rows, c, copier, ex) for c in beyond):.1f}",
        ),
        nc(
            "OthersShareMax",
            f"{max(example_share(rows, s['cap'], m, ex) for s in settings for m in models if m not in (anchor, copier)):.1f}",
        ),
    ]
    # What the copies cost: pooled tail bits at the smallest prompt with and
    # without them, and the anchor model's own scores likewise.
    wo = without_value(rows, ex)
    lines += [
        nc(
            "TailSmallestNoExample",
            f"{pooled_mean(wo, first['cap'], TAIL, 'excess_bits', models):.3f}",
        ),
        nc(
            "MidSmallestNoExample",
            f"{pooled_mean(wo, first['cap'], MID_RANGE, 'excess_brier', models):.4f}",
        ),
        nc(
            "NoExampleGapBeyondMax",
            f"{max(abs(pooled_mean(rows, s['cap'], TAIL, 'excess_bits', models) - pooled_mean(wo, s['cap'], TAIL, 'excess_bits', models)) for s in settings[1:]):.3f}",
        ),
        nc(
            "AnchorTailSmallest",
            f"{per_model_mean(rows, first['cap'], TAIL, 'excess_bits', anchor):.3f}",
        ),
        nc(
            "AnchorTailSmallestNoExample",
            f"{per_model_mean(wo, first['cap'], TAIL, 'excess_bits', anchor):.3f}",
        ),
        nc(
            "AnchorMidSmallest",
            f"{per_model_mean(rows, first['cap'], MID_RANGE, 'excess_brier', anchor):.3f}",
        ),
        nc(
            "AnchorMidSmallestNoExample",
            f"{per_model_mean(wo, first['cap'], MID_RANGE, 'excess_brier', anchor):.3f}",
        ),
    ]
    # The anchor model's unanswered questions, by setting.
    unparsed = [
        (100.0 - (parse_rate(runs, s["cap"], anchor) or 100.0), s) for s in settings
    ]
    peak = max(unparsed, key=lambda u: u[0])
    others_unparsed = max(
        100.0 - (parse_rate(runs, s["cap"], m) or 100.0)
        for s in settings
        for m in models
        if m != anchor
    )
    lines += [
        nc("AnchorUnparsedSmallest", f"{unparsed[0][0]:.1f}"),
        nc("AnchorUnparsedPeak", f"{peak[0]:.1f}"),
        nc("AnchorUnparsedPeakSize", size_label(peak[1]["qpp"], tex=True)),
        nc(
            "AnchorUnparsedZeroFrom",
            size_label(
                next(
                    s["qpp"]
                    for u, s in unparsed
                    if u == 0.0 and s["qpp"] > peak[1]["qpp"]
                ),
                tex=True,
            ),
        ),
        nc("OthersUnparsedMax", f"{others_unparsed:.1f}"),
    ]
    # Robustness: the pooled picture without the anchor model.
    rest = [m for m in models if m != anchor]
    b_rest = batching_bootstrap(rows, settings, TAIL, "excess_bits", rest)
    rho_rest = batching_rho(rows, settings, TAIL, "excess_bits", rest)
    lines += [
        nc("NModelsNoAnchor", len(rest)),
        nc("TailSmallestNoAnchor", f"{b_rest[first['cap']]['point']:.3f}"),
        nc(
            "DeltaTailFourNoAnchor",
            f"${b_rest[4]['delta']:+.3f}$" if 4 in b_rest else "---",
        ),
        nc(
            "DeltaTailFourNoAnchorCI",
            macro_band((b_rest[4]["dlo"], b_rest[4]["dhi"])) if 4 in b_rest else "---",
        ),
        nc(
            "RhoTailSmallestNoAnchor",
            f"{adjusted(rho_rest[first['cap']].rho):.2f}"
            if rho_rest[first["cap"]]
            else "---",
        ),
        nc(
            "RhoTailNoAnchorMin",
            f"{min(adjusted(c.rho) for c in rho_rest.values() if c):.2f}",
        ),
    ]
    # The rerun in its own data directory, when gathered.
    if recheck:
        mine = [r for r in recheck if r["model_id"] == anchor]
        parsed = [r for r in mine if r["forecast"] is not None]
        stop = sum(1 for r in mine if r["finish_reason"] == "stop")
        noblock = [r for r in mine if not r["has_block"]]
        old = {r["question_id"]: r["forecast"] for r in one if r["model_id"] == anchor}
        pairs = [
            (old[r["question_id"]], r["forecast"])
            for r in parsed
            if r["question_id"] in old
        ]
        old65 = [(a, b) for a, b in pairs if abs(a - ex) < 1e-9]
        tail = [r for r in parsed if r["section"] == TAIL]
        mid = [r for r in parsed if r["section"] == MID_RANGE]
        import math

        def bits(f, p):
            f = min(max(f, 0.001), 0.999)
            v = 0.0
            if p > 0:
                v += p * math.log2(p / f)
            if p < 1:
                v += (1 - p) * math.log2((1 - p) / (1 - f))
            return v

        lines += [
            "% The GPT-5 mini rerun at the smallest prompt, in its own data directory.",
            nc("RecheckN", f"{len(mine):,}"),
            nc("RecheckUnparsed", len(mine) - len(parsed)),
            nc(
                "RecheckStopShare", f"{100.0 * stop / len(mine):.1f}" if mine else "---"
            ),
            nc("RecheckNoBlock", len(noblock)),
            nc(
                "RecheckNoBlockStop",
                sum(1 for r in noblock if r["finish_reason"] == "stop"),
            ),
            nc(
                "RecheckExampleShare",
                f"{100.0 * sum(1 for r in parsed if abs(r['forecast'] - ex) < 1e-9) / len(parsed):.1f}"
                if parsed
                else "---",
            ),
            nc(
                "RecheckTail",
                f"{sum(bits(r['forecast'], r['real_prob']) for r in tail) / len(tail):.3f}"
                if tail
                else "---",
            ),
            nc(
                "RecheckMid",
                f"{sum((r['forecast'] - r['real_prob']) ** 2 for r in mid) / len(mid):.3f}"
                if mid
                else "---",
            ),
            nc(
                "RecheckMeanAbsDelta",
                f"{sum(abs(a - b) for a, b in pairs) / len(pairs):.3f}"
                if pairs
                else "---",
            ),
            nc(
                "RecheckIdenticalShare",
                f"{100.0 * sum(1 for a, b in pairs if abs(a - b) < 1e-9) / len(pairs):.0f}"
                if pairs
                else "---",
            ),
            nc(
                "RecheckExampleAgainShare",
                f"{100.0 * sum(1 for a, b in old65 if abs(b - ex) < 1e-9) / len(old65):.0f}"
                if old65
                else "---",
            ),
        ]
    return lines


def write_tables(
    datadir: Path,
    binary: list[dict],
    continuous: list[dict],
    coverage: list[dict],
    names: dict[str, str],
    extra: dict[str, str] | None = None,
) -> list[Path]:
    """The appendix tables, as files the article \\inputs.

    `extra` carries tables built elsewhere (the ablation's), keyed like
    TABLE_NAMES.
    """
    built = {
        "models": models_table(binary, continuous, coverage, names),
        "horizon": horizon_table(binary, names),
        "continuous": continuous_table(continuous, names),
        **(extra or {}),
    }
    out = []
    for key, text in built.items():
        path = datadir / TABLE_NAMES[key]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
        out.append(path)
    return out


def cells_of(line: str) -> list[str]:
    r"""A table row's cells. The same split build_metadata.py uses."""
    return [c.strip() for c in line.removesuffix(r"\\").split("&")]


def table_row(cells: list[str]) -> str:
    """Cells back into a row, as build_metadata.py writes them."""
    return " & ".join(cells) + r" \\"


def is_row(line: str) -> bool:
    """Whether a line of a generated table is a data row."""
    return "&" in line and line.rstrip().endswith(r"\\")


def edit_hosting(text: str, usage: list[dict], names: dict[str, str]) -> str:
    """Fill this world's two columns of the shared hosting table.

    Adds "Micropolis calls" and "Cost" before StarSim's pair, or refills them
    when a previous run put them there. Before, not after, because StarSim's
    generator addresses its own cells from the right-hand end.

    The join is on the article's display name, which is the table's first
    column: every other key the two sides share (the ECI, the OpenRouter id)
    is absent from this table. A name the panel has no row for is left alone
    rather than guessed at — the other worlds ran models this one did not.
    """
    per_model: dict[str, dict[str, float]] = {}
    for r in usage:
        acc = per_model.setdefault(r["model"], {"calls": 0, "cost": 0.0})
        acc["calls"] += r["ncalls"]
        acc["cost"] += r["cost_usd"]
    by_name = {tex_escape(names.get(m, m)): v for m, v in per_model.items()}

    lines = text.splitlines()
    header = next(i for i, ln in enumerate(lines) if ln.startswith("Model &"))
    ncols = len(cells_of(lines[header]))
    # Where our pair sits. A rerun finds them already present; a first run
    # inserts them, which is the only time the column count changes.
    fresh = "Micropolis calls" not in lines[header]
    at = ncols - 2 if fresh else cells_of(lines[header]).index("Micropolis calls")

    def put(cells: list[str], pair: list[str]) -> list[str]:
        out = list(cells)
        if fresh:
            out[at:at] = pair
        else:
            out[at : at + 2] = pair
        return out

    seen = set()
    for i, line in enumerate(lines):
        if not is_row(line):
            continue
        cells = cells_of(line)
        if i == header:
            lines[i] = table_row(put(cells, ["Micropolis calls", r"Cost (\$)"]))
            continue
        name = cells[0]
        if name in by_name:
            v = by_name[name]
            seen.add(name)
            lines[i] = table_row(
                put(cells, [f"{int(v['calls']):,}", f"{v['cost']:.2f}"])
            )
        elif name == "Total":
            calls = sum(v["calls"] for v in per_model.values())
            cost = sum(v["cost"] for v in per_model.values())
            lines[i] = table_row(put(cells, [f"{int(calls):,}", f"{cost:.2f}"]))
        else:
            # A model this world did not run: an empty pair, not a zero, which
            # would claim the calls were made and cost nothing.
            lines[i] = table_row(put(cells, ["--", "--"]))
    missing = sorted(set(by_name) - seen)
    if missing:
        sys.exit(
            "[error] these models have Micropolis usage but no row in"
            f" {SHARED_TABLES['hosting']}: {', '.join(missing)}\n"
            "  the join is on the display name of model_scores.csv; the"
            " article's table must name them the same way"
        )
    if fresh:
        lines = widen_tabular(lines, at)
    return "\n".join(lines) + "\n"


def widen_tabular(lines: list[str], at: int) -> list[str]:
    r"""Add two right-aligned columns to the \begin{tabular} preamble.

    Only on the run that first inserts them. The column spec is read and
    rewritten rather than replaced wholesale, so whatever alignment and
    @{}-padding the article chose survives.
    """
    for i, line in enumerate(lines):
        if not line.startswith(r"\begin{tabular}"):
            continue
        spec = line[line.index("{", len(r"\begin{tabular}") - 1) :]
        inner = spec.strip()[1:-1]
        # Column letters only; @{} and >{} groups are positional padding that
        # must not be counted as columns.
        out, seen, done = [], 0, False
        for ch in inner:
            if not done and ch in "lcr" and seen == at:
                out.append("r r ")
                done = True
            out.append(ch)
            if ch in "lcr":
                seen += 1
        if not done:
            out.append(" r r")
        lines[i] = r"\begin{tabular}{" + "".join(out) + "}"
        return lines
    sys.exit(f"[error] no \\begin{{tabular}} in {SHARED_TABLES['hosting']}")


def edit_cost_per_item(
    text: str, totals: dict[str, dict[str, float]], items: dict[str, int]
) -> str:
    """Fill this world's rows of the shared per-item cost table.

    Rewrites the Micropolis rows' label, calls, cost and cents per item from
    the run, leaving the other worlds' rows and the whole StarSim block
    untouched. The row's label carries the batch size, which is a property of
    the run and was stale in the committed file, so it is written here too.
    """
    lines = text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.startswith("Micropolis &")), None
    )
    if start is None:
        sys.exit(
            f"[error] no Micropolis block in {SHARED_TABLES['cost_per_item']}\n"
            '  the row must begin "Micropolis &" for this script to find it'
        )
    for j, (name, label) in enumerate(COST_ROWS):
        i = start + j
        if i >= len(lines) or not is_row(lines[i]):
            sys.exit(
                f"[error] {SHARED_TABLES['cost_per_item']}: expected"
                f" {len(COST_ROWS)} Micropolis rows, found {j}"
            )
        cells = cells_of(lines[i])
        t = totals[name]
        n = items[name]
        cost = t["cost"] / t["nmodels"] if t["nmodels"] else 0.0
        # The items a prompt actually carried, not the config's cap: a
        # continuous snapshot has fewer scored lines than the cap allows, so
        # the cap would misdescribe the prompt the model saw.
        cells[1] = f"{label} ({n // t['nprompts']} questions per prompt)"
        cells[2] = f"{n:,}"
        cells[3] = f"{t['nprompts']:,}"
        cells[4] = f"{cost:.2f}"
        cells[5] = f"{100.0 * cost / n:.2f}" if n else "--"
        lines[i] = table_row(cells)
    return "\n".join(lines) + "\n"


def write_macros(
    path: Path,
    found: dict[str, object],
    names: dict[str, str],
    by_horizon: dict[str, dict[str, object]],
    fb: dict[str, object],
    rates: dict[str, float],
    totals: dict[str, dict[str, float]],
    items: dict[str, int],
    binary: list[dict],
    continuous: list[dict],
    models_for_ci: list[str],
    extra: list[str] | None = None,
) -> Path:
    r"""Write micropolis-macros.tex: every quoted number as a \newcommand.

    `extra` is appended verbatim: macro lines built elsewhere in this script
    (the ablation's), so one file still holds every number the paper quotes.

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
    lines += cost_lines(totals, items)
    lines += band_lines(binary)
    lines += horizon_years_lines()
    lines += city_ci_lines(binary, continuous, models_for_ci)
    lines += extra or []
    lines += bootstrap_lines()
    if fb:
        lines += predictor_lines("FB", "ForecastBench overall", fb)
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
    usage = read_usage(args.datadir / USAGE_CSV_NAME)
    batching = read_batching(args.datadir / BATCHING_CSV_NAME)
    batching_runs = read_batching_runs(args.datadir / BATCHING_RUNS_CSV_NAME)
    recheck = read_recheck(args.datadir / RECHECK_CSV_NAME)

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
    totals = usage_by_eval(usage)
    # The items one model was asked, from the coverage rows rather than from
    # the config: it is the denominator of the per-item cost, so it has to be
    # the count the scores were actually computed over.
    items = items_per_model(coverage)
    for name, t in sorted(totals.items()):
        print(
            f"cost, {name}: ${t['cost']:.2f} over {t['calls']:,} calls"
            f" ({items[name]:,} items per model)"
        )
    print(f"cost, total: ${sum(t['cost'] for t in totals.values()):.2f}")
    print()

    # The questions-per-prompt ablation. Its production setting is the one
    # whose prompts carried as many questions as the main run's did.
    settings = batching_settings(batching)
    ablation_models = models_in_order(batching)
    production = production_setting(
        settings, items["binary"] / totals["binary"]["nprompts"]
    )
    boot = {
        section: batching_bootstrap(batching, settings, section, key, ablation_models)
        for section, key, _, _ in BATCHING_PANELS
    }
    rho = {
        section: batching_rho(batching, settings, section, key, ablation_models)
        for section, key, _, _ in BATCHING_PANELS
    }
    print(
        f"questions-per-prompt ablation: {len(settings)} settings x"
        f" {len(ablation_models)} models"
        + (
            f"; the paper's run is the {size_label(production['qpp'])}-per-prompt setting"
            if production
            else ""
        )
    )
    for s_ in settings:
        b_m, b_t = boot[MID_RANGE][s_["cap"]], boot[TAIL][s_["cap"]]
        print(
            f"  {size_label(s_['qpp']):>6} per prompt: mid {b_m['point']:.4f}"
            f" ({b_m['delta']:+.4f} vs smallest)  tail bits {b_t['point']:.3f}"
            f" ({b_t['delta']:+.3f}, CI {format_band((b_t['dlo'], b_t['dhi']), 0)})"
        )
    if recheck is None:
        print(f"  [note] {RECHECK_CSV_NAME} absent; rerun macros not defined")
    print()
    batch_macros = batching_lines(
        batching,
        batching_runs,
        settings,
        ablation_models,
        names,
        boot,
        rho,
        production,
        recheck,
    )
    macros = write_macros(
        args.datadir / MACROS_NAME,
        found,
        names,
        by_horizon,
        fb,
        rates,
        totals,
        items,
        binary,
        continuous,
        models_in_order(binary),
        batch_macros,
    )
    tables = write_tables(
        args.datadir,
        binary,
        continuous,
        coverage,
        names,
        {
            "batching": batching_table(
                batching,
                batching_runs,
                settings,
                ablation_models,
                boot,
                rho,
                production,
            ),
            "batching_models": batching_models_table(
                batching, settings, ablation_models, names
            ),
        },
    )
    cells = write_cells(args.datadir, binary, continuous)
    # The article's own figure, which is not one of the extra ones: --no-extra
    # skips the figures the paper does not place, and this is the one it does.
    capability = draw_capability_figure(
        args.outdir / CAPABILITY_FIG_NAME,
        found,
        names,
    )

    horizon_fig = draw_horizon_figure(args.outdir / HORIZON_FIG_NAME, binary, names)
    bands_fig = draw_bands_figure(args.outdir / BANDS_FIG_NAME, binary)
    batching_fig = draw_batching_figure(
        args.outdir / BATCHING_FIG_NAME,
        batching,
        batching_runs,
        settings,
        ablation_models,
        boot,
        production,
    )

    for out in figures.written:
        print(f"Wrote {out}")
    for out in (capability, horizon_fig, bands_fig, batching_fig):
        if out:
            print(f"Wrote {out}")
    print(f"Wrote {macros}")
    for out in [*tables, cells]:
        print(f"Wrote {out}")

    # Deliver into the article, when there is one checked out here: the macros
    # to data/, where its other \\input of generated definitions lives, and
    # the figures it places to figures/.
    repo = paper_repo_dir()
    if repo is None:
        print(f"\n[note] {PAPER_REPO_ENV} unset; not copying into the article")
        return
    print()
    deliver([macros], repo / PAPER_REPO_MACROS)
    deliver(paper_figures(args.outdir), repo / PAPER_REPO_FIGURES)
    deliver(tables, repo / PAPER_REPO_TABLES)
    deliver(paper_csvs(args.datadir), repo / PAPER_REPO_DATA)
    # Edited in place rather than delivered: the article and StarSim's
    # generator own the rest of these two files.
    write_shared_tables(repo, usage, totals, items, names)


if __name__ == "__main__":
    main()
