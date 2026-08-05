#!/usr/bin/env -S uv run python3
"""Build the single-city Micropolis question corpus and write it to JSON.

Runs (or reuses cached) simulations for every base scenario, then resolves
each question template at every horizon.

Usage:
    uv run python scripts/gen_corpus.py
    uv run python scripts/gen_corpus.py --seed 7 --out /tmp/corpus.json
"""

import argparse
import json
from pathlib import Path

import micropolis_world.module_globals as g
from micropolis_world.scenarios import build_corpus, get_single_city_base_scenarios


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <data>/micropolis/corpus-seed<seed>.json)",
    )
    args = ap.parse_args()

    scenarios = get_single_city_base_scenarios(seed=args.seed)
    corpus = build_corpus(scenarios)

    out = args.out or g.DATA_DIR / f"corpus-seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(corpus, indent=2))
    print(f"[saved] {len(corpus)} questions -> {out}")


if __name__ == "__main__":
    main()
