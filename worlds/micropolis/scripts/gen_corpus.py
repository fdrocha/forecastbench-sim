#!/usr/bin/env -S uv run python3
"""Build the single-city Micropolis question corpus and write it to JSON.

Runs (or reuses cached) simulations for every base scenario, then resolves
each question template at every horizon.

Usage:
    scripts/gen_corpus.py
    scripts/gen_corpus.py my_config.json --seed 7
"""

import argparse
import json

import micropolis_world.module_globals as g
from micropolis_world.config import (
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.scenarios import build_corpus, get_single_city_base_scenarios


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    args = ap.parse_args()

    cfg = load_config(args)
    seed = cfg.get_seed(args.seed)

    scenarios = get_single_city_base_scenarios(
        seed=seed, cities=cfg.get_cities(), disasters=cfg.get_bool_list("disasters")
    )
    corpus = build_corpus(
        scenarios,
        cfg.get_int_list("snapshot_turns"),
        cfg.get_int_list("horizons"),
        cfg.get_int("history_freq"),
    )

    out = g.DATA_DIR / f"corpus-seed{seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(corpus, indent=2))
    print(f"[saved] {len(corpus)} questions -> {out}")


if __name__ == "__main__":
    main()
