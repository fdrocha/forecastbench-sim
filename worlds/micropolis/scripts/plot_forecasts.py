#!/usr/bin/env -S uv run python3
"""Plot single city metric trajectories with model forecasts overlaid.

One figure per (scenario, snapshot turn), carrying every horizon asked from
that snapshot. Reads data/micropolis/single_city/data.json, written by
scripts/run_single_city_eval.py, and the simulation logs the questions came
from; prompts no models.

The config selects which slice of the dataset to draw — its models, cities,
disasters, snapshot_turns and horizons — so one gathered dataset can be plotted
many ways. Naming anything the dataset lacks is an error, not a smaller figure.

Usage:
    scripts/plot_forecasts.py
    scripts/plot_forecasts.py subset.json5
"""

import argparse
import sys
from pathlib import Path

from micropolis_world.city_sim import CitySimulation, turn_of
from micropolis_world.config import (
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.single_city import (
    DATA_PATH,
    PLOTS_PATH,
    DatasetError,
    Responses,
    ResponseId,
    load_dataset,
    select_for_config,
)

# The metrics charted per figure, in subplot order — the three that have
# question templates, so every panel carries forecasts as well as the actual.
PLOT_METRICS = [
    ("totalFunds", "Funds ($)", "tab:green"),
    ("cityPop", "Population", "tab:blue"),
    ("crimeAverage", "Crime Average", "tab:red"),
]


def plot_forecasts(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    outdir: Path = PLOTS_PATH,
) -> list[Path]:
    """One figure per (scenario, snapshot turn): trajectories + model forecasts.

    Metrics are stacked vertically and drawn against turn rather than date, so
    a forecast sits at the turn it resolves on (snapshot_turn + horizon), give
    or take the small per-model offset that keeps them apart. Every
    horizon asked from a snapshot shares the figure: a model's forecasts then
    read left to right as a fan widening with distance from the snapshot, which
    is what comparing horizons is about. Disaster lines are omitted.
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

    # Group questions by the figure they belong to, then by which subplot.
    by_figure: dict[tuple[str, int], list[dict]] = {}
    for c in corpus:
        by_figure.setdefault((c["scenario_id"], c["snapshot_turn"]), []).append(c)

    # A scenario_id maps to exactly one simulation; load each one once.
    sims: dict[str, CitySimulation] = {}
    written = []

    for (scenario_id, snapshot_turn), entries in sorted(by_figure.items()):
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

        # Every model answers the same question, so without this they all land
        # on one x and hide each other. Spread them across a slot narrower than
        # the gap between adjacent resolve turns, centered on the turn the
        # forecast is actually for, so the groups stay distinct.
        resolve_turns = sorted({c["snapshot_turn"] + c["horizon"] for c in entries})
        gaps = [b - a for a, b in zip(resolve_turns, resolve_turns[1:])]
        slot = 0.6 * min(gaps) if gaps else 0.04 * (max(turns) - min(turns))
        dodge = {
            model_id: slot * ((i + 0.5) / len(model_names) - 0.5)
            for i, model_id in enumerate(model_names)
        }

        fig, axes = plt.subplots(
            len(PLOT_METRICS), 1, figsize=(11, 4 * len(PLOT_METRICS)), sharex=True
        )
        for ax, (metric, title, color) in zip(axes, PLOT_METRICS):
            actual = [r[metric] for r in rows]
            ax.plot(turns, actual, color=color, label="actual")
            # Everything left of this line was in the prompt; everything right
            # of it is what the models were asked to forecast.
            ax.axvline(
                snapshot_turn,
                color="0.4",
                linestyle="--",
                linewidth=1.2,
                label=f"snapshot (turn {snapshot_turn})",
            )
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            # Overlay each model's forecasts for this metric at the turn they
            # resolve on: the median as a marker, p25-p75 as an error bar.
            # Unparseable answers have no percentiles; skip them.
            all_medians = []
            for model_id in model_names:
                xs, medians, lo, hi = [], [], [], []
                for c in entries:
                    if c["metric"] != metric:
                        continue
                    r = responses.get(ResponseId(model_id, c["question_id"]))
                    if r is None or r.percentiles is None:
                        continue
                    p = r.percentiles
                    xs.append(c["snapshot_turn"] + c["horizon"] + dodge[model_id])
                    medians.append(p["p50"])
                    # errorbar wants distances from the median, not absolute
                    # positions. Both are non-negative because parse_percentiles
                    # rejects any set that isn't in increasing order.
                    lo.append(p["p50"] - p["p25"])
                    hi.append(p["p75"] - p["p50"])
                all_medians.extend(medians)
                if xs:
                    ax.errorbar(
                        xs,
                        medians,
                        yerr=[lo, hi],
                        fmt="o",
                        markersize=6,
                        zorder=5,
                        alpha=0.85,
                        color=model_colors[model_id],
                        markeredgecolor="black",
                        markeredgewidth=0.5,
                        elinewidth=1.2,
                        capsize=3,
                        label=model_id.split("/")[-1],
                    )

            # Scale to the trajectory and the medians only. A single model with
            # a wildly wide interval would otherwise set the range for the whole
            # panel and flatten everything else into a line; the bars are still
            # drawn, they just run off the top or bottom.
            visible = actual + all_medians
            span = max(visible) - min(visible)
            pad = 0.05 * span if span else 1.0
            ax.set_ylim(min(visible) - pad, max(visible) + pad)

        axes[-1].set_xlabel("Turn")

        # One shared legend; every subplot has the same series. It grows with
        # the model count, so wrap it and give the rows it needs back to the
        # figure rather than letting it eat the bottom subplot.
        handles, labels = axes[0].get_legend_handles_labels()
        ncol = min(5, len(labels))
        legend_rows = -(-len(labels) // ncol)
        fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize="small")
        horizons = sorted({c["horizon"] for c in entries})
        fig.suptitle(
            f"{scenario_id} — forecasts from turn {snapshot_turn}, "
            f"horizons {', '.join(str(h) for h in horizons)}"
        )
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.02 + 0.022 * legend_rows)

        out = outdir / f"forecasts_{scenario_id}_turn{snapshot_turn}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

    return written


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    args = ap.parse_args()

    cfg = load_config(args)

    # Both failures are user error with an obvious fix — the gathering step
    # hasn't run, or hasn't run for this config — so say so plainly rather than
    # with a traceback.
    try:
        corpus, responses, models = load_dataset()
        corpus, responses, models = select_for_config(
            corpus, responses, models, cfg, cfg.get_seed(args.seed)
        )
    except (FileNotFoundError, DatasetError) as e:
        sys.exit(f"[error] {e}")

    print("=" * 70)
    print("MICROPOLIS WORLD — single city forecast plots")
    print("=" * 70)
    print(f"data:   {DATA_PATH}")
    print(f"config: {cfg.path}")

    written = plot_forecasts(corpus, responses, models)
    print(f"Wrote {len(written)} plots -> {PLOTS_PATH}")


if __name__ == "__main__":
    main()
