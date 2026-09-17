#!/usr/bin/env -S uv run python3
"""Paper figures and score tables: the subset of the two analyses the article uses.

Reads the datasets the binary and continuous evals already gathered and writes,
under data/micropolis/paper/:

- binary_scores.csv and continuous_scores.csv, byte-for-byte what
  analyze_binary.py and analyze_continuous.py write beside their own reports —
  the row builders are imported, not reimplemented, so the paper's tables and
  the reports' can never disagree.
- figures/*.pdf, the ECI scatters, in PDF for \\includegraphics: excess Brier
  for the mid-range and the tail binary sections, excess bits for the tail, and
  excess nCRPS (global normalization) for the continuous eval.

The figures are the report scatters with the title dropped — in an article the
caption does that job — and sized for a \\textwidth of about 6.5in. The
per-model legend, the fit line with its ρ and r, and the 95% CI bars over
questions are kept.

Both datasets are required: the paper's outputs are all-or-nothing, so a
missing data.json or ground truth is an error, not a partial run. No
--incomplete here — ragged selections make per-model figures cover different
question sets, which is a testing aid, not something to publish.

Usage:
    scripts/analyze_paper.py
    scripts/analyze_paper.py --binary-config configs/binary-testing.json5
    scripts/analyze_paper.py --continuous-config configs/prompt-a.json5
"""

import argparse
import sys
from pathlib import Path

import micropolis_world.module_globals as g
from micropolis_world.binary_eval import data_path as binary_data_path
from micropolis_world.binary_eval import load_dataset_binary
from micropolis_world.config import CONFIG_DIR, Config, ConfigError, main_with_config
from micropolis_world.continuous_eval import (
    NORM_MODES,
    DatasetError,
    MdReport,
    attach_outcomes,
    data_path,
    load_dataset,
    make_normalizer,
    select_for_config,
)
from micropolis_world.ground_truth import load_truths
from micropolis_world.messages import error

# Imported rather than reimplemented so the paper's CSVs are the reports' CSVs
# and its scatters the reports' scatters.
sys.path.insert(0, str(Path(__file__).parent))
from analyze_binary import (
    MID_RANGE,
    TAIL,
    plot_eci_vs_score,
    score_forecasts_binary,
    scores_for,
    section_of,
    write_csv,
)
from analyze_binary import (
    SCORES_CSV_COLUMNS as BINARY_CSV_COLUMNS,
)
from analyze_binary import (
    SCORES_CSV_NAME as BINARY_CSV_NAME,
)
from analyze_binary import (
    scores_csv_rows as binary_scores_csv_rows,
)
from analyze_continuous import (
    EXCESS,
    plot_eci_vs_normalized,
    write_scores_csv,
)
from analyze_continuous import (
    SCORES_CSV_NAME as CONTINUOUS_CSV_NAME,
)
from analyze_continuous import (
    scores_csv_rows as continuous_scores_csv_rows,
)

DEFAULT_CONTINUOUS_CONFIG_PATH = CONFIG_DIR / "continuous.json5"
DEFAULT_BINARY_CONFIG_PATH = CONFIG_DIR / "binary.json5"

# Everything the paper needs, in one place and flat: the label picks which
# data.json is read, not where the outputs land, since the article cites one
# set of figures by a fixed path.
OUT_DIR = g.DATA_DIR / "paper"
FIGURES_DIR = OUT_DIR / "figures"

# The one continuous normalization the paper reports. The three modes are not
# comparable with each other, and `global` is the one whose scale is fixed per
# metric — so a cell means the same thing across scenarios, snapshots and
# horizons, and derive_scales.py can check it against FreeCiv's.
NORM_MODE = "global"

# A \textwidth figure in a single-column article. The report's 10x6.5 would be
# scaled down by \includegraphics and take the fonts with it. The height is the
# axes' share; the legend below adds its own rows on top of it, and
# bbox_inches="tight" grows the box to fit them.
FIG_WIDTH, FIG_HEIGHT = 7.0, 4.3

# The legend goes under the axes rather than beside them, in this many columns.
# Two dozen models in one column beside the axes takes more width than the
# scatter it explains; below, the scatter gets the full figure width and the
# legend costs height instead, which a \textwidth figure has to spare.
LEGEND_NCOLS = 4

# The binary scatters the paper carries: (section key, score key). The tail's
# excess bits is FreeCiv's tail score — at p below 5% an always-No forecast has
# a near-zero excess Brier and separates nothing, so the ratio f/p is what
# tells the models apart there.
BINARY_FIGURES = [
    (MID_RANGE, "excess_brier"),
    (TAIL, "excess_brier"),
    (TAIL, "excess_bits"),
]

# How the report names each section in the axis labels the scatters keep.
SECTION_NAMES = {MID_RANGE: "Mid-range probabilities", TAIL: "Tail probabilities"}


def load_config_at(path: Path | str) -> Config:
    """Load one config by path, exiting with a message rather than a traceback."""
    try:
        return Config.load(path)
    except ConfigError as e:
        error(str(e))
        sys.exit(1)


# The continuous scatter's y label — "Mean excess normalized CRPS (CRPS/scale,
# lower is better)" — is longer than the shortened axes are tall, so it runs off
# the figure and into the legend. The parenthetical is what goes: the ratio is
# stated in the caption, and "lower is better" is on every score in the paper.
def shorten_ylabels(fig) -> None:
    """Drop the parenthetical from any y label too long for the paper's axes."""
    for ax in fig.axes:
        label = ax.get_ylabel()
        if len(label) > 40 and "(" in label:
            ax.set_ylabel(label[: label.index("(")].strip())


def relegend(fig) -> None:
    """Move a figure's legend under its axes, in LEGEND_NCOLS columns.

    The report scatters put the legend beside the axes, where a two-dozen-model
    list is wider than the scatter. The handles and labels are taken off the
    existing legend — so the order the plot function chose, fit line and CI
    entry first, is kept — and a new one is drawn below, outside the axes.
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
            # Clear of the x-axis label, which sits just under the axes; -0.13
            # put the legend's first row on top of it.
            bbox_to_anchor=(0.5, -0.22),
            ncol=LEGEND_NCOLS,
            fontsize=6,
            frameon=False,
            handletextpad=0.4,
            columnspacing=1.0,
            borderaxespad=0.0,
        )


class PaperFigures:
    """Collects the report scatters as paper PDFs.

    The plot functions are the reports' own, so they take an MdReport and write
    a PNG named for the report's conventions. This wraps each call: it sets the
    paper's rc params, then intercepts the figure on its way to disk to drop
    the title, move the legend below the axes and resize, and writes a PDF
    under the paper's own name. The restyling happens after the fact rather
    than by threading flags through the plot functions: the point of importing
    them is that the paper draws the same axes the report does, and a parameter
    per stylistic difference would let the two drift.
    """

    def __init__(self, outdir: Path):
        self.outdir = outdir
        self.written: list[Path] = []

    def draw(self, name: str, plot) -> Path | None:
        """Run `plot(report, tmpdir)`, strip its titles, save it as name.pdf."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        report = MdReport()
        with plt.rc_context(
            {
                "figure.figsize": (FIG_WIDTH, FIG_HEIGHT),
                "savefig.format": "pdf",
                # Type 42 (TrueType) rather than Type 3, which many
                # publishers reject.
                "pdf.fonttype": 42,
                "font.size": 9,
                "axes.titlesize": 9,
                "axes.labelsize": 9,
                "legend.fontsize": 6,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
            }
        ):
            # The plot functions save and close their own figure, so the
            # titles have to go before that: patching savefig is the one hook
            # that runs with the figure still open.
            original = plt.Figure.savefig

            def savefig(fig, path, *a, **kw):
                fig.suptitle("")
                for ax in fig.axes:
                    ax.set_title("")
                relegend(fig)
                shorten_ylabels(fig)
                # The plot functions pass figsize=(10, 6.5) to subplots
                # explicitly, so figure.figsize never reaches them; the resize
                # has to happen here. Done before tight_layout so the labels
                # are laid out at the final size rather than scaled into it.
                fig.set_size_inches(FIG_WIDTH, FIG_HEIGHT)
                fig.tight_layout()
                out = self.outdir / f"{name}.pdf"
                out.parent.mkdir(parents=True, exist_ok=True)
                # The mode's own kwargs (dpi) are dropped: a PDF is vector.
                return original(fig, out, format="pdf", bbox_inches="tight")

            plt.Figure.savefig = savefig
            try:
                # The plot functions mkdir their outdir and name their own
                # PNG; the patched savefig ignores that name, so this only has
                # to be a real directory.
                drawn = plot(report, self.outdir)
            finally:
                plt.Figure.savefig = original

        if drawn is None:
            # The plot function declined — too few models with an ECI score,
            # or no spread to correlate. It said why in the report text, which
            # is the whole of this throwaway report.
            why = report.render(self.outdir).strip().splitlines()
            print(f"[skipped] {name}: {why[-1] if why else 'no figure drawn'}")
            return None
        out = self.outdir / f"{name}.pdf"
        self.written.append(out)
        return out


def binary_half(cfg: Config, figures: PaperFigures) -> list[dict]:
    """The binary eval's CSV rows, its scatters drawn on the way.

    Same selection, sectioning and scoring as analyze_binary.py: the section a
    question instance falls in is decided by its ground-truth P(Yes), so a qid
    can be mid-range in one city and tail in another.
    """
    label = cfg.get_label(None)
    data_file = binary_data_path(label)
    try:
        corpus, responses, models = load_dataset_binary(data_file)
        corpus, responses, models = select_for_config(
            corpus,
            responses,
            models,
            cfg,
            cfg.get_seed(None),
            rerun_hint="scripts/run_eval_binary.py",
        )
        truths = load_truths(corpus)
    except (FileNotFoundError, DatasetError) as e:
        sys.exit(f"[error] {e}")

    print(f"binary:     {data_file}")
    print(f"            {len(corpus)} questions x {len(models)} models")

    for section_key, score_key in BINARY_FIGURES:
        section_corpus = [c for c in corpus if section_of(c, truths) == section_key]
        if not section_corpus:
            print(f"[skipped] eci_vs_{score_key}-{section_key}: no such questions")
            continue
        rows = score_forecasts_binary(section_corpus, responses, models, truths)
        # The Score object the report uses, so the axis label and the row field
        # are the ones analyze_binary.py defines.
        score = next(s for s in scores_for(section_key) if s.key == score_key)

        def plot(report, outdir, sc=section_corpus, r=rows, s=score, k=section_key):
            return plot_eci_vs_score(
                report, sc, r, models, outdir, SECTION_NAMES[k], k, s
            )

        figures.draw(f"eci_vs_{score_key}-{section_key}", plot)
    return binary_scores_csv_rows(corpus, responses, models, truths)


def continuous_half(cfg: Config, figures: PaperFigures) -> tuple[list[dict], list[str]]:
    """The continuous eval's CSV rows and its one scatter.

    The CSV is the full one — every normalization mode, both measures — since
    it is the reports' own row builder; only the figure is restricted to
    NORM_MODE.
    """
    label = cfg.get_label(None)
    seed = cfg.get_seed(None)
    data_file = data_path(label)
    try:
        corpus, responses, models = load_dataset(data_file)
        corpus, responses, models = select_for_config(
            corpus, responses, models, cfg, seed
        )
    except (FileNotFoundError, DatasetError) as e:
        sys.exit(f"[error] {e}")

    print(f"continuous: {data_file}")
    print(f"            {len(corpus)} questions x {len(models)} models")

    # Every mode, because continuous_scores.csv carries every mode's columns
    # and the reports' row builder expects them all.
    try:
        norms = {
            mode: make_normalizer(
                mode,
                corpus,
                global_frac=cfg.get_norm_global_frac(None),
                seed=seed,
            )
            for mode in NORM_MODES
        }
        without_outcomes = attach_outcomes(corpus)
    except (NotImplementedError, FileNotFoundError) as e:
        sys.exit(
            f"[error] {e}\n"
            "  the excess measure and two of the normalizations need the ground"
            " truth; run scripts/extract_ground_truth.py for this config"
        )
    if without_outcomes:
        print(
            f"            {without_outcomes} question(s) have no continuation"
            " outcomes and get no excess CRPS"
        )

    figures.draw(
        f"eci_vs_excess_ncrps-{NORM_MODE}",
        lambda report, outdir: plot_eci_vs_normalized(
            report, corpus, responses, models, outdir, norms[NORM_MODE], EXCESS
        ),
    )
    return (
        continuous_scores_csv_rows(corpus, responses, models, norms),
        list(NORM_MODES),
    )


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--continuous-config",
        default=DEFAULT_CONTINUOUS_CONFIG_PATH,
        help="JSON5 config for the continuous half "
        f"(default: {DEFAULT_CONTINUOUS_CONFIG_PATH})",
    )
    ap.add_argument(
        "--binary-config",
        default=DEFAULT_BINARY_CONFIG_PATH,
        help="JSON5 config for the binary half "
        f"(default: {DEFAULT_BINARY_CONFIG_PATH})",
    )
    args = ap.parse_args()

    continuous_cfg = load_config_at(args.continuous_config)
    binary_cfg = load_config_at(args.binary_config)

    print("=" * 70)
    print("MICROPOLIS WORLD — paper figures and tables")
    print("=" * 70)
    print(f"configs:    {continuous_cfg.path}")
    print(f"            {binary_cfg.path}")
    print(f"out:        {OUT_DIR}")
    print()

    figures = PaperFigures(FIGURES_DIR)
    binary_rows = binary_half(binary_cfg, figures)
    continuous_rows, modes = continuous_half(continuous_cfg, figures)

    print()
    for out in figures.written:
        print(f"Wrote {out}")
    binary_csv = write_csv(OUT_DIR / BINARY_CSV_NAME, BINARY_CSV_COLUMNS, binary_rows)
    continuous_csv = write_scores_csv(
        OUT_DIR / CONTINUOUS_CSV_NAME, continuous_rows, modes
    )
    print(f"Wrote {binary_csv}")
    print(f"Wrote {continuous_csv}")


if __name__ == "__main__":
    main()
