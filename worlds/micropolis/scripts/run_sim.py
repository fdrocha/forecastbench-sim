#!/usr/bin/env -S uv run python3
"""Run a single Micropolis city simulation via CitySimulation.run.

Usage:
    uv run python scripts/run_sim.py --city haight --seed 1 --turns 1000 --disasters
    uv run python scripts/run_sim.py --city haight --seed 1 --turns 1000 --plot
"""

import argparse

import micropolis_world.module_globals as g
from micropolis_world.city_sim import CitySimulation
from plot_run import plot_run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=g.CITY_CHOICES, default="haight")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--turns", type=int, default=100)
    ap.add_argument("--disasters", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--plot", action="store_true", help="Plot the simulation results after running"
    )
    ap.add_argument(
        "-o",
        "--output",
        help="Save the plot to this PNG file instead of showing it interactively (implies --plot)",
    )
    args = ap.parse_args()

    sim = CitySimulation(city_name=args.city, seed=args.seed, disasters=args.disasters)
    sim.run(nturns=args.turns, quiet=args.quiet)

    if args.plot or args.output:
        plot_run(sim, output=args.output)


if __name__ == "__main__":
    main()
