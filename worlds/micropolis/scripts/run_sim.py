#!/usr/bin/env -S uv run python3
"""Run Micropolis city simulations via CitySimulation.run.

Runs every (city, disasters) combination in the config file.

Usage:
    uv run python scripts/run_sim.py
    uv run python scripts/run_sim.py my_config.json --plot
    uv run python scripts/run_sim.py my_config.json --seed 7 --quiet
"""

import argparse

from micropolis_world.city_sim import CitySimulation
from micropolis_world.config import (
    add_config_args,
    load_config,
    main_with_config,
    scenarios_from,
)
from plot_run import plot_run


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--plot", action="store_true", help="Plot the simulation results after running"
    )
    args = ap.parse_args()

    cfg = load_config(args)
    seed = cfg.get_seed(args.seed)
    turns = cfg.get_int("turns")
    scenarios = scenarios_from(cfg)

    for i, (city, disasters) in enumerate(scenarios):
        sim = CitySimulation(city_name=city, seed=seed, disasters=disasters)
        print(f"[{i + 1}/{len(scenarios)}] running {sim.get_id_str()} for {turns} turns")
        sim.run(nturns=turns, quiet=args.quiet)

        if args.plot:
            out = sim.get_plot_path()
            out.parent.mkdir(parents=True, exist_ok=True)
            plot_run(sim, output=str(out))


if __name__ == "__main__":
    main()
