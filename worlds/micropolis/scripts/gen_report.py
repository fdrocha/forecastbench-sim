#!/usr/bin/env -S uv run python3
"""Print the model-facing world report for an already-run Micropolis simulation.

Loads the sim's log/events files from disk (does not run the sim).

Usage:
    uv run python scripts/gen_report.py
    uv run python scripts/gen_report.py --city haight --seed 1 --disasters --turn 500
"""

import argparse
import sys

import micropolis_world.module_globals as g
from micropolis_world.city_sim import CitySimulation
from micropolis_world.report import gen_world_report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--city", choices=g.CITY_CHOICES, default="haight")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--disasters", action="store_true", default=True)
    ap.add_argument("--no-disasters", dest="disasters", action="store_false")
    ap.add_argument(
        "--turn",
        type=int,
        default=-1,
        help="Snapshot turn; negative counts back from the last logged turn",
    )
    ap.add_argument("--history-freq", type=int, default=100)
    args = ap.parse_args()

    sim = CitySimulation(city_name=args.city, seed=args.seed, disasters=args.disasters)
    try:
        sim.load_from_disk()
    except FileNotFoundError as e:
        print(
            f"[error] {e}\nDid you run the simulation first? "
            f"e.g. uv run python scripts/run_sim.py --city {args.city} --seed {args.seed}"
            f"{' --disasters' if args.disasters else ''}",
            file=sys.stderr,
        )
        sys.exit(1)

    assert sim.log_data is not None
    turn = args.turn if args.turn >= 0 else len(sim.log_data) + args.turn
    print(gen_world_report(sim, turn=turn, history_freq=args.history_freq))


if __name__ == "__main__":
    main()
