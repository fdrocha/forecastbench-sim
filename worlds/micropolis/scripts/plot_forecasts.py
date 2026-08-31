#!/usr/bin/env -S uv run python3
"""Plot single city metric trajectories with model forecasts overlaid.

One figure per scenario, carrying every snapshot taken from it and every horizon
asked from those snapshots. Reads
data/micropolis/single_city/{label}/data.json, written by
scripts/run_single_city_eval.py, and the simulation logs the questions came
from; prompts no models.

The config selects which slice of the dataset to draw — its models, cities,
disasters, snapshot_turns and horizons — so one gathered dataset can be plotted
many ways. Naming anything the dataset lacks is an error, not a smaller figure.

Writes one figure per scenario to
data/micropolis/single_city/{label}/plots/forecasts/, panelled by
metric.

Defaults to the forecasts_plots.json5 config, which subsets the models to a
number these figures can legibly carry; name another config to override it.

Usage:
    scripts/plot_forecasts.py
    scripts/plot_forecasts.py subset.json5
    scripts/plot_forecasts.py micropolis_world/configs/default.json5
    scripts/plot_forecasts.py --cities kyoto --models openai/gpt-5.6-sol
"""

import argparse
import sys
from itertools import pairwise
from pathlib import Path

import micropolis_world.module_globals as g
from micropolis_world.city_sim import CitySimulation, turn_of
from micropolis_world.config import (
    CONFIG_DIR,
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.plot_sim import PANEL_GRID, PANEL_METRICS
from micropolis_world.single_city import (
    DatasetError,
    ResponseId,
    Responses,
    data_path,
    load_dataset,
    plots_path,
    select_for_config,
)


def forecasts_path(label: str) -> Path:
    # Written to their own directory: one figure per scenario is many files, and
    # they would otherwise be mixed in with the handful of summary plots the
    # scoring script writes alongside them.
    return plots_path(label) / "forecasts"

# These figures carry one marker per (model, snapshot, horizon) per panel, so the
# full model set makes them unreadable; this config subsets to a legible few.
# Named here as the default rather than in default.json5 so the scoring scripts,
# which want every model, are unaffected. Pass a config to override.
DEFAULT_CONFIG_PATH = CONFIG_DIR / "forecasts_plots.json5"

# The panels come from plot_sim.py's list, so a metric is drawn the same color
# and in the same position whether the figure carries forecasts or not, and
# titles come from g.METRIC_LABELS so no panel can disagree with how the same
# metric is named in a report.
PLOT_METRICS = [
    (metric, str(g.METRIC_LABELS[metric]).capitalize(), color)
    for metric, color in PANEL_METRICS
]


def plot_forecasts(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    outdir: Path,
) -> list[Path]:
    """One figure per scenario: trajectories plus every snapshot's forecasts.

    One panel per metric that has forecast questions, gridded and drawn against
    turn rather than date, so a forecast sits at the turn it resolves on
    (snapshot_turn + horizon), give or take the small per-model offset that keeps
    them apart. Every horizon asked from a snapshot shares the figure: a model's
    forecasts then read left to right as a fan widening with distance from the
    snapshot, which is what comparing horizons is about.

    Every snapshot of a scenario shares the figure too, distinguished by marker
    shape, since they share one actual trajectory and comparing what a model said
    from turn 240 against what it said from 480 is the other half of the picture.
    Disaster lines are omitted.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    # One color per model, stable across every figure. tab10 is the clearer
    # palette but wraps at 10, which would give two models the same color, so
    # step up to tab20 once there are more models than that.
    n_colors = 10 if len(model_names) <= 10 else 20
    palette = plt.get_cmap(f"tab{n_colors}")
    model_colors = {m: palette(i % n_colors) for i, m in enumerate(model_names)}

    # One figure per scenario, carrying every snapshot taken from it. The actual
    # trajectory is a property of the scenario, not of the snapshot, so splitting
    # by snapshot would redraw the same line once per figure.
    by_figure: dict[str, list[dict]] = {}
    for c in corpus:
        by_figure.setdefault(c["scenario_id"], []).append(c)

    # A scenario_id maps to exactly one simulation; load each one once.
    sims: dict[str, CitySimulation] = {}
    written = []

    for scenario_id, entries in sorted(by_figure.items()):
        if scenario_id not in sims:
            sc = entries[0]["scenario"]
            sim = CitySimulation(
                city_name=sc["name"], seed=sc["seed"], disasters=sc["disasters"]
            )
            sim.load_from_disk()
            sims[scenario_id] = sim
        sim = sims[scenario_id]
        rows = sim.log_data or []
        turns = [turn_of(r) for r in rows]

        snapshot_turns = sorted({c["snapshot_turn"] for c in entries})
        # Marker shape per snapshot, since color is already spent on the model.
        # Position alone would not separate them: a late snapshot's nearest
        # forecasts land among an early snapshot's most distant ones.
        markers = ["o", "s", "^", "D", "v", "P"]
        snapshot_markers = {
            turn: markers[i % len(markers)] for i, turn in enumerate(snapshot_turns)
        }

        # Every model answers the same question, so without this they all land
        # on one x and hide each other. Spread them across a slot narrower than
        # the gap between adjacent resolve turns, centered on the turn the
        # forecast is actually for, so the groups stay distinct. Computed over
        # every snapshot's resolve turns at once, so clusters from different
        # snapshots that land close together still do not merge.
        resolve_turns = sorted({c["snapshot_turn"] + c["horizon"] for c in entries})
        gaps = [b - a for a, b in pairwise(resolve_turns)]
        slot = 0.6 * min(gaps) if gaps else 0.04 * (max(turns) - min(turns))
        dodge = {
            model_id: slot * ((i + 0.5) / len(model_names) - 0.5)
            for i, model_id in enumerate(model_names)
        }

        nrows, ncols = PANEL_GRID
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(8 * ncols, 3.6 * nrows),
            sharex=True,
            squeeze=False,
        )
        flat_axes = [ax for row in axes for ax in row]
        for ax, (metric, title, color) in zip(flat_axes, PLOT_METRICS):
            actual = [r[metric] for r in rows]
            # Unlabelled: each panel draws its metric in its own color, so one
            # panel's handle would key the legend with a color five other panels
            # do not use. A neutral proxy stands in below.
            ax.plot(turns, actual, color=color)
            # Everything left of one of these lines was in that snapshot's
            # prompt; everything right of it is what the models were asked to
            # forecast from it.
            for turn in snapshot_turns:
                ax.axvline(turn, color="0.4", linestyle="--", linewidth=1.2)
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            # Overlay each model's forecasts for this metric at the turn they
            # resolve on: the median as a marker, p25-p75 as an error bar.
            # Unparseable answers have no percentiles; skip them.
            all_medians = []
            for model_id in model_names:
                for snapshot_turn in snapshot_turns:
                    xs, medians, lo, hi = [], [], [], []
                    for c in entries:
                        if c["metric"] != metric or c["snapshot_turn"] != snapshot_turn:
                            continue
                        r = responses.get(ResponseId(model_id, c["question_id"]))
                        if r is None or r.percentiles is None:
                            continue
                        p = r.percentiles
                        xs.append(c["snapshot_turn"] + c["horizon"] + dodge[model_id])
                        medians.append(p["p50"])
                        # errorbar wants distances from the median, not absolute
                        # positions. Both are non-negative because
                        # parse_percentiles rejects any set not in increasing
                        # order.
                        lo.append(p["p50"] - p["p25"])
                        hi.append(p["p75"] - p["p50"])
                    all_medians.extend(medians)
                    if not xs:
                        continue
                    # Unlabelled: the model legend is built from color proxies
                    # below. Labelling a real series here would tie each model's
                    # legend marker to whichever snapshot it happened to have data
                    # for, so a model missing the first snapshot would show the
                    # wrong shape.
                    ax.errorbar(
                        xs,
                        medians,
                        yerr=[lo, hi],
                        fmt=snapshot_markers[snapshot_turn],
                        markersize=6,
                        zorder=5,
                        alpha=0.85,
                        color=model_colors[model_id],
                        markeredgecolor="black",
                        markeredgewidth=0.5,
                        elinewidth=1.2,
                        capsize=3,
                    )

            # Scale to the trajectory and the medians only. A single model with
            # a wildly wide interval would otherwise set the range for the whole
            # panel and flatten everything else into a line; the bars are still
            # drawn, they just run off the top or bottom.
            visible = actual + all_medians
            span = max(visible) - min(visible)
            pad = 0.05 * span if span else 1.0
            ax.set_ylim(min(visible) - pad, max(visible) + pad)

        # A grid wider than the metric list would leave panels showing only their
        # axes, which read as a missing plot rather than an empty slot.
        for ax in flat_axes[len(PLOT_METRICS) :]:
            ax.set_visible(False)

        # sharex leaves tick labels only on the bottom row, so that is where the
        # x label belongs; putting it under a panel whose ticks are hidden would
        # leave it floating mid-figure with no axis to read it against.
        for ax in axes[-1]:
            if ax.get_visible():
                ax.set_xlabel("Turn")

        # One shared legend. The trajectory and snapshot lines come off the axes;
        # models are color proxies and snapshots shape proxies, so the two
        # encodings are each stated once instead of being crossed into one entry
        # per (model, snapshot).
        handles = [
            plt.Line2D([], [], color="0.35", lw=1.5),
            plt.Line2D([], [], color="0.4", lw=1.2, linestyle="--"),
        ]
        labels = ["actual (per-panel color)", "snapshot"]
        handles += [
            plt.Line2D(
                [],
                [],
                marker="o",
                color=model_colors[model_id],
                linestyle="",
                markersize=6,
                markeredgecolor="black",
                markeredgewidth=0.5,
            )
            for model_id in model_names
        ]
        labels += [m.split("/")[-1] for m in model_names]
        if len(snapshot_turns) > 1:
            handles += [
                plt.Line2D(
                    [],
                    [],
                    marker=snapshot_markers[turn],
                    color="0.35",
                    linestyle="",
                    markersize=6,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                )
                for turn in snapshot_turns
            ]
            labels += [f"from turn {turn}" for turn in snapshot_turns]
        legend = fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(6, len(labels)),
            fontsize="small",
        )
        horizons = sorted({c["horizon"] for c in entries})
        fig.suptitle(
            f"{scenario_id} — forecasts from turns "
            f"{', '.join(str(t) for t in snapshot_turns)}; "
            f"horizons {', '.join(str(h) for h in horizons)}"
        )
        fig.tight_layout()
        # Reserve the strip the legend occupies, measured from the drawn figure
        # rather than estimated from the label count: it wraps to a number of rows
        # that depends on the label widths, so any fixed fraction either overlaps
        # the axes or leaves a gap.
        #
        # tight_layout has already left room below the axes for the tick labels
        # and x label, but it ran before the legend existed. subplots_adjust
        # measures from the figure edge, so that room has to be added back on top
        # of the legend's height or the x labels end up behind the legend.
        fig.canvas.draw()
        bottom_axes = [ax for ax in axes[-1] if ax.get_visible()]
        below_axes = max(
            (
                ax.get_position().y0 - ax.get_tightbbox().y0 / fig.bbox.height
                for ax in bottom_axes
            ),
            default=0.0,
        )
        legend_height = legend.get_window_extent().height / fig.bbox.height
        fig.subplots_adjust(bottom=legend_height + below_axes + 0.01)

        out = outdir / f"forecasts_{scenario_id}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

    return written


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default=DEFAULT_CONFIG_PATH)
    args = ap.parse_args()

    cfg = load_config(args)
    label = cfg.get_label(args.label)
    data_file = data_path(label)
    outdir = forecasts_path(label)

    # Both failures are user error with an obvious fix — the gathering step
    # hasn't run, or hasn't run for this config — so say so plainly rather than
    # with a traceback.
    try:
        corpus, responses, models = load_dataset(data_file)
        corpus, responses, models = select_for_config(
            corpus,
            responses,
            models,
            cfg,
            cfg.get_seed(args.seed),
            cities=args.cities,
            disasters=args.disasters,
            models=args.models,
        )
    except (FileNotFoundError, DatasetError) as e:
        sys.exit(f"[error] {e}")

    print("=" * 70)
    print("MICROPOLIS WORLD — single city forecast plots")
    print("=" * 70)
    print(f"data:   {data_file}")
    print(f"config: {cfg.path}")
    print(f"label:  {label}")

    written = plot_forecasts(corpus, responses, models, outdir)
    print(f"Wrote {len(written)} plots -> {outdir}")


if __name__ == "__main__":
    main()
