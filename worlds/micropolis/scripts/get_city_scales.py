#!/usr/bin/env -S uv run python3
"""Per-city metric tables (rows = cities, columns = metrics), four views:
the starting turn, the mean over the turns up to the first snapshot, the
first snapshot itself, and the maximum over that same window.

Reads the cached sim logs only; never runs the engine.

Usage:
    scripts/get_city_scales.py
    scripts/get_city_scales.py configs/binary.json5
    scripts/get_city_scales.py --cities kyoto,kobe --csv scales.csv
"""

import argparse
import csv
import statistics
import sys

from micropolis_world.city_sim import CitySimulation
from micropolis_world.config import (
    add_config_args,
    load_config,
    main_with_config,
    scenarios_from,
)
from micropolis_world.messages import error

# The scale-bearing metrics: no city score (bounded) and no city funds (no scale).
METRICS = [
    "cityPop",
    "trafficAverage",
    "pollutionAverage",
    "crimeAverage",
    "landValueAverage",
]

# Column headers, the METRIC_LABELS without their "average" qualifier.
LABELS = {
    "cityPop": "population",
    "trafficAverage": "traffic",
    "pollutionAverage": "pollution",
    "crimeAverage": "crime",
    "landValueAverage": "land value",
}


def print_table(title: str, rows: dict[str, dict[str, float]]) -> None:
    labels = [LABELS[m] for m in METRICS]
    city_w = max(len(c) for c in rows)
    widths = [max(len(lab), 12) for lab in labels]
    header = f"{'city':<{city_w}}  " + "  ".join(
        f"{lab:>{w}}" for lab, w in zip(labels, widths)
    )
    print(title)
    print(header)
    print("-" * len(header))
    for city, values in rows.items():
        cells = "  ".join(f"{values[m]:>{w},.0f}" for m, w in zip(METRICS, widths))
        print(f"{city:<{city_w}}  {cells}")
    print()


def collect_views(
    cfg,
    seed: int,
    start: int,
    snapshot: int,
    cities: list[str] | None = None,
    disasters: list[bool] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """The four views, each city -> metric -> value, read from the cached logs.

    Values do not depend on the disaster setting before anything strikes, but
    the window views do, so a city is read from the first variant the config
    lists for it. A city with no cached log is reported and skipped.
    """
    views: dict[str, dict[str, dict[str, float]]] = {
        "start": {},
        "mean": {},
        "snapshot": {},
        "max": {},
    }
    for city, dis in scenarios_from(cfg, cities, disasters):
        if city in views["start"]:
            continue
        sim = CitySimulation(city_name=city, seed=seed, disasters=dis)
        try:
            sim.load_from_disk()
        except FileNotFoundError as e:
            error(f"{city}: {e}")
            continue
        assert sim.log_data is not None
        if snapshot >= len(sim.log_data):
            error(f"{city}: only {len(sim.log_data)} turns logged, need turn {snapshot}")
            continue
        window = sim.log_data[start : snapshot + 1]
        views["start"][city] = {m: sim.log_data[start][m] for m in METRICS}
        views["mean"][city] = {m: statistics.fmean(r[m] for r in window) for m in METRICS}
        views["snapshot"][city] = {m: sim.log_data[snapshot][m] for m in METRICS}
        views["max"][city] = {m: max(r[m] for r in window) for m in METRICS}
    return views


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument(
        "--turn",
        type=int,
        default=0,
        help="the starting turn the tables read from (default 0)",
    )
    ap.add_argument("--csv", help="also write the four tables to this path")
    args = ap.parse_args()

    cfg = load_config(args)
    seed = cfg.get_seed(args.seed)
    start = args.turn
    snapshot = cfg.get_int_list("snapshot_turns")[0]
    if snapshot < start:
        sys.exit(f"[error] first snapshot turn {snapshot} is before start turn {start}")

    views = collect_views(cfg, seed, start, snapshot, args.cities, args.disasters)
    if not views["start"]:
        sys.exit("[error] no city had a cached log; run scripts/run_sim.py first")

    titles = {
        "start": f"Starting values (turn {start})",
        "mean": f"Mean over turns {start}-{snapshot}",
        "snapshot": f"Values at the first snapshot (turn {snapshot})",
        "max": f"Maximum over turns {start}-{snapshot}",
    }
    for key, rows in views.items():
        print_table(titles[key], rows)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["view", "city", *METRICS])
            for key, rows in views.items():
                for city, values in rows.items():
                    w.writerow([key, city, *(values[m] for m in METRICS)])
        print(args.csv, file=sys.stderr)


if __name__ == "__main__":
    main()
