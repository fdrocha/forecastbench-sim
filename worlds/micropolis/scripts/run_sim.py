#!/usr/bin/env -S uv run python3
"""Run a single Micropolis city simulation via CitySimulation.run.

Usage:
    uv run python scripts/run_sim.py --city haight --seed 1 --turns 1000 --disasters
"""

import argparse

from micropolis_world.city_sim import CITY_CHOICES, CitySimulation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=CITY_CHOICES, default="haight")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--turns", type=int, default=100)
    ap.add_argument("--disasters", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    sim = CitySimulation(city_name=args.city, seed=args.seed, disasters=args.disasters)
    sim.run(nturns=args.turns, quiet=args.quiet)


if __name__ == "__main__":
    main()
