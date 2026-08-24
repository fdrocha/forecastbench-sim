#!/usr/bin/env -S uv run python3
"""Report what past API calls have cost, from the usage sidecars on disk.

Every kept model response has a usage-{model}-{hash}.json beside it recording
the tokens and dollars of the call that produced it. This script sums those
records; it never prompts anything, so it costs nothing and works offline.

Two ways to choose which calls to report on:

  Config mode (the default) takes the same config file and overrides as the
  run* scripts, rebuilds the same corpus, and reports on exactly the calls that
  config implies — so the number it prints is the cost of the run those same
  arguments would reproduce. Calls the config asks for that were never made are
  counted as not yet prompted, not as missing data.

  --glob reports on every sidecar matching a pattern instead, ignoring the
  config. Relative patterns resolve under data/micropolis/. Use this to look
  across labels and configs, or to reach the knowledge eval's cache, which
  config mode does not cover.

Usage:
    scripts/analyze_usage.py
    scripts/analyze_usage.py my_config.json --per-model
    scripts/analyze_usage.py --cities kyoto --disasters false
    scripts/analyze_usage.py --glob 'single_city/cache/*/usage-*.json'
    scripts/analyze_usage.py --glob 'knowledge_eval/usage-*.json' --per-model
    scripts/analyze_usage.py --glob '**/usage-*.json' --per-model
"""

import argparse

from micropolis_world import usage_report as ur
from micropolis_world.config import (
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.scenarios import (
    build_batch_prompt_continuous,
    build_corpus,
    get_single_city_base_scenarios,
)
from micropolis_world.single_city import (
    group_into_batches,
    prompt_hash,
    response_path,
    usage_path,
)


def collect_from_config(args: argparse.Namespace) -> ur.Collected:
    """The sidecars for the calls this config's single-city eval implies.

    Rebuilds the corpus and batch prompts the same way run_single_city_eval.py
    does, because the sidecar's filename carries the prompt's hash: without
    re-deriving the prompt there is no way to tell which stored call belongs to
    this config rather than to some other variant cached beside it.
    """
    cfg = load_config(args)
    seed = cfg.get_seed(args.seed)
    label = cfg.get_label(args.label)
    model_names = cfg.get_models(args.models)
    snapshot_only = cfg.get_bool_or("snapshot_only_report", False)
    preamble_path = cfg.get_preamble_path()

    print(f"config: {cfg.path}")
    print(f"label:  {label}")

    scenarios = get_single_city_base_scenarios(
        seed=seed,
        cities=cfg.get_cities(args.cities),
        disasters=cfg.get_disasters(args.disasters),
    )
    corpus = build_corpus(
        scenarios,
        cfg.get_int_list("snapshot_turns"),
        cfg.get_int_list("horizons"),
        cfg.get_int("history_freq"),
        label,
        snapshot_only,
        cfg.get_int_or("history_length", -1),
    )

    batches = group_into_batches(corpus)
    batch_hashes = {
        bid: prompt_hash(
            build_batch_prompt_continuous(
                questions[0]["context"], questions, snapshot_only, preamble_path
            )
        )
        for bid, questions in batches.items()
    }
    return ur.collect_for_batches(
        batch_hashes, model_names, response_path, usage_path
    )


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument(
        "--glob",
        metavar="PATTERN",
        help="report on every sidecar matching PATTERN instead of on a config's "
        "calls; relative patterns resolve under data/micropolis/",
    )
    ap.add_argument(
        "--per-model",
        action="store_true",
        help="also break the totals down by individual model",
    )
    args = ap.parse_args()

    print("=" * 70)
    print("MICROPOLIS WORLD — API usage")
    print("=" * 70)

    if args.glob:
        print(f"glob:   {args.glob}")
        collected = ur.Collected(usages=ur.load_from_glob(args.glob))
    else:
        collected = collect_from_config(args)

    print()
    if not collected.usages:
        print("No usage data found.")
    else:
        print(ur.format_table(ur.by_provider(collected.usages), "provider"))
        if args.per_model:
            print()
            print(ur.format_table(ur.by_model(collected.usages), "model"))

    # One line rather than a warning per file: an old cache legitimately has no
    # sidecars at all, and a screen of warnings would bury the totals.
    if collected.missing:
        print(f"\nUsage data not stored for {collected.missing} API calls")
    if collected.unprompted:
        print(f"{collected.unprompted} batch/model pairs not yet prompted")

    total = ur.grand_total(ur.by_provider(collected.usages))
    if total.unpriced:
        # Excluded from the cost column above, so the total is a floor.
        print(
            f"{total.unpriced} call(s) had no litellm price; "
            "the cost above excludes them"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
