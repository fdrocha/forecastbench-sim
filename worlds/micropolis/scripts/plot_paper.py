#!/usr/bin/env -S uv run python3
"""Draw the article's figures from data/micropolis/paper/, as PDF for LaTeX.

The drawing half of the paper pipeline. Reads only what
scripts/gather_paper_data.py wrote — binary_forecasts.csv and
continuous_forecasts.csv — plus each model's ECI from the package's
model_scores.csv, which is where every other script gets it. No dataset, no
ground truth, no config: rerunning this to nudge a legend costs a second and
cannot change a number.

Writes to data/micropolis/paper/figures/:

- eci_vs_excess_brier-mid-range.pdf
- eci_vs_excess_brier-tail.pdf
- eci_vs_excess_bits-tail.pdf        (FreeCiv's tail score)
- eci_vs_excess_ncrps-global.pdf     (the continuous eval)

Each is the report's own ECI scatter — the same points, fit line, Spearman ρ
and Pearson r, and 95% bootstrap intervals, from analyze_binary.py's and
analyze_continuous.py's plot functions — restyled for an article: the title
dropped, since the caption does that job; the per-model legend moved from
beside the axes to below them in four columns, since two dozen models in one
column is wider than the scatter it explains; and the whole sized for a
\\textwidth of about 6.5in with Type 42 fonts.

Model order fixes each model's color and marker, and is taken from the CSV's
own order of first appearance, so a model keeps one identity across the
figures without a config being read.

Usage:
    scripts/plot_paper.py
    scripts/plot_paper.py --outdir /tmp/figures
    scripts/plot_paper.py --datadir /tmp/paper-data --outdir /tmp/figures
"""

import argparse
import csv
import sys
from pathlib import Path

from micropolis_world.continuous_eval import MdReport

sys.path.insert(0, str(Path(__file__).parent))
from analyze_binary import (
    EXCESS_BITS,
    MID_RANGE,
    SCORES,
    TAIL,
    Score,
    plot_eci_vs_score,
)
from analyze_continuous import EXCESS
from gather_paper_data import (
    BINARY_CSV_NAME,
    CONTINUOUS_CSV_NAME,
    NORM_MODE,
    OUT_DIR,
)

FIGURES_DIR = OUT_DIR / "figures"

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
        "ncrps",
        "excess_crps",
        "excess_ncrps",
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

    def draw(self, name: str, plot) -> Path | None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        report = MdReport()
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
            try:
                # The plot functions mkdir their outdir and name their own PNG;
                # the patched savefig ignores that name, so this only has to be
                # a real directory.
                drawn = plot(report, self.outdir)
            finally:
                plt.Figure.savefig = original

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
    args = ap.parse_args()

    binary = read_rows(args.datadir / BINARY_CSV_NAME)
    continuous = read_rows(args.datadir / CONTINUOUS_CSV_NAME)

    print("=" * 70)
    print("MICROPOLIS WORLD — paper figures")
    print("=" * 70)
    print(f"data:   {args.datadir}")
    print(f"out:    {args.outdir}")
    print(f"binary:     {len(binary)} scored forecasts")
    print(f"continuous: {len(continuous)} scored forecasts")
    print()

    figures = PaperFigures(args.outdir)

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

        def plot(report, outdir, c=corpus, r=rows, s=score, k=section_key):
            return plot_eci_vs_score(
                report, c, r, binary_models, outdir, SECTION_NAMES[k], k, s
            )

        figures.draw(f"eci_vs_{score.key}-{section_key}", plot)

    continuous_models = models_in_order(continuous)
    continuous_corpus = [
        {"question_id": q} for q in {r["question_id"] for r in continuous}
    ]
    figures.draw(
        f"eci_vs_excess_ncrps-{NORM_MODE}",
        lambda report, outdir: plot_eci_vs_score(
            report,
            continuous_corpus,
            continuous,
            continuous_models,
            outdir,
            CONTINUOUS_SECTION,
            NORM_MODE,
            EXCESS_NCRPS,
        ),
    )

    print()
    for out in figures.written:
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
