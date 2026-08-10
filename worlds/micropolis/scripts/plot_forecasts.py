#!/usr/bin/env -S uv run python3
"""Plot single city metric trajectories with model forecasts overlaid.

One figure per (scenario, horizon). Reads
data/micropolis/single_city/data.json, written by
scripts/run_single_city_eval.py, and the simulation logs the questions came
from; prompts no models.

The config selects which slice of the dataset to draw — its models, cities,
disasters, snapshot_turns and horizons — so one gathered dataset can be plotted
many ways. Naming anything the dataset lacks is an error, not a smaller figure.

Usage:
    uv run python scripts/plot_forecasts.py
    uv run python scripts/plot_forecasts.py subset.json5
    uv run python scripts/plot_forecasts.py --outdir /tmp/plots
    uv run python scripts/plot_forecasts.py --data other/data.json
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

# The four metrics charted per figure, in subplot order. Only the metrics that
# have question templates get forecast points overlaid; the rest are context.
PLOT_METRICS = [
    ("cityPop", "Population", "tab:blue"),
    ("totalFunds", "Funds ($)", "tab:green"),
    ("crimeAverage", "Crime Average", "tab:red"),
    ("pollutionAverage", "Pollution Average", "tab:orange"),
]


def plot_forecasts(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    outdir: Path = PLOTS_PATH,
) -> list[Path]:
    """One figure per (scenario, horizon): metric trajectories + model forecasts.

    Each figure has the same four metric subplots as micropolis_world.plot_sim, drawn
    against turn rather than date so forecasts can be placed at the turn they
    resolve on (snapshot_turn + horizon). Disaster lines are omitted.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    # One color per model, stable across every figure.
    palette = plt.get_cmap("tab10")
    model_colors = {m: palette(i % 10) for i, m in enumerate(model_names)}

    # Group questions by the figure they belong to, then by which subplot.
    by_figure: dict[tuple[str, int], list[dict]] = {}
    for c in corpus:
        by_figure.setdefault((c["scenario_id"], c["horizon"]), []).append(c)

    # A scenario_id maps to exactly one simulation; load each one once.
    sims: dict[str, CitySimulation] = {}
    written = []

    for (scenario_id, horizon), entries in sorted(by_figure.items()):
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

        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
        for ax, (metric, title, color) in zip(axes.flat, PLOT_METRICS):
            ax.plot(turns, [r[metric] for r in rows], color=color, label="actual")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            # Overlay each model's forecasts for this metric at the turn they
            # resolve on: the median as a marker, p10-p90 as an error bar.
            # Unparseable answers have no percentiles; skip them.
            for model_id in model_names:
                xs, medians, lo, hi = [], [], [], []
                for c in entries:
                    if c["metric"] != metric:
                        continue
                    r = responses.get(ResponseId(model_id, c["question_id"]))
                    if r is None or r.percentiles is None:
                        continue
                    p = r.percentiles
                    xs.append(c["snapshot_turn"] + horizon)
                    medians.append(p["p50"])
                    # errorbar wants distances from the median, not absolute
                    # positions. Both are non-negative because parse_percentiles
                    # rejects any set that isn't in increasing order.
                    lo.append(p["p50"] - p["p10"])
                    hi.append(p["p90"] - p["p50"])
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

        for ax in axes[1]:
            ax.set_xlabel("Turn")

        # One shared legend; every subplot has the same series.
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=max(2, len(labels)),
            fontsize="small",
        )
        fig.suptitle(f"{scenario_id} — forecasts at horizon {horizon}")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.13)

        out = outdir / f"forecasts_{scenario_id}_{horizon}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

    return written


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument(
        "--data",
        type=Path,
        default=DATA_PATH,
        help=f"dataset written by run_single_city_eval.py (default: {DATA_PATH})",
    )
    ap.add_argument(
        "--outdir",
        type=Path,
        default=PLOTS_PATH,
        help=f"where to write the figures (default: {PLOTS_PATH})",
    )
    args = ap.parse_args()

    cfg = load_config(args)

    # Both failures are user error with an obvious fix — the gathering step
    # hasn't run, or hasn't run for this config — so say so plainly rather than
    # with a traceback.
    try:
        corpus, responses, models = load_dataset(args.data)
        corpus, responses, models = select_for_config(
            corpus, responses, models, cfg, cfg.get_seed(args.seed)
        )
    except (FileNotFoundError, DatasetError) as e:
        sys.exit(f"[error] {e}")

    print("=" * 70)
    print("MICROPOLIS WORLD — single city forecast plots")
    print("=" * 70)
    print(f"data:   {args.data}")
    print(f"config: {cfg.path}")

    written = plot_forecasts(corpus, responses, models, outdir=args.outdir)
    print(f"Wrote {len(written)} plots -> {args.outdir}")


if __name__ == "__main__":
    main()
