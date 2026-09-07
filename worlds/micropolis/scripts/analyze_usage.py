#!/usr/bin/env -S uv run python3
"""Report what past API calls have cost, from the usage sidecars on disk.

Every kept model response has a usage-{model}-{hash}.json beside it recording
the tokens and dollars of the call that produced it. This script sums those
records; it never prompts anything, so it costs nothing and works offline.

Two ways to choose which calls to report on:

  Config mode (the default) takes the same config files and overrides as the
  run* scripts, rebuilds the corpora of both the continuous and the binary
  eval, and reports on exactly the calls those configs imply — so the number
  it prints is the cost of the runs those same arguments would reproduce.
  Calls a config asks for that were never made are counted as not yet
  prompted, not as missing data; a config used for only one of the two evals
  therefore reports the other eval's calls entirely under that count.

  Several configs may be named at once, and their totals are then reported
  together. A call shared by two of them — configs commonly overlap, differing
  only in their model list or in a prompt variant that leaves most batches
  untouched — is counted once, since the cached response was paid for once.

  --glob reports on every sidecar matching a pattern instead, ignoring the
  config. Relative patterns resolve under data/micropolis/. Use this to look
  across labels and configs, or to reach the knowledge eval's cache, which
  config mode does not cover.

Usage:
    scripts/analyze_usage.py
    scripts/analyze_usage.py my_config.json --per-model
    scripts/analyze_usage.py cfgA.json5 cfgB.json5 cfgC.json5
    scripts/analyze_usage.py configs/*.json5 --per-config
    scripts/analyze_usage.py --cities kyoto --disasters false
    scripts/analyze_usage.py --glob 'continuous/cache/*/usage-*.json'
    scripts/analyze_usage.py --glob 'knowledge_eval/cache/usage-*.json' --per-model
    scripts/analyze_usage.py --glob '**/usage-*.json' --per-model
"""

import argparse

from micropolis_world import usage_report as ur
from micropolis_world.binary_eval import PATHS as BINARY_PATHS
from micropolis_world.binary_questions import build_corpus_binary
from micropolis_world.config import (
    Config,
    add_config_args,
    load_configs,
    main_with_config,
)
from micropolis_world.continuous_eval import (
    group_into_batches,
    prompt_hash,
    response_path,
    usage_path,
)
from micropolis_world.scenarios import (
    build_batch_prompt_binary,
    build_batch_prompt_continuous,
    build_corpus,
    get_base_scenarios,
)


def collect_from_config(
    cfg: Config,
    args: argparse.Namespace,
    seen: set[tuple[str, str, str]],
) -> dict[str, ur.Collected]:
    """The sidecars for the calls this config implies, keyed by eval.

    Rebuilds the corpora and batch prompts the same way run_eval_continuous.py
    and run_eval_binary.py do, because the sidecar's filename carries the
    prompt's hash: without re-deriving the prompt there is no way to tell
    which stored call belongs to this config rather than to some other variant
    cached beside it. Both evals are covered since one config can drive either
    run script; an eval the config was never run through contributes nothing
    but its unprompted count.

    `seen` carries the calls already counted for earlier configs, so a call two
    configs share is reported once — see usage_report.collect_for_batches. The
    two evals never collide in it: their prompts, and so their hashes, differ.
    """
    seed = cfg.get_seed(args.seed)
    label = cfg.get_label(args.label)
    model_names = cfg.get_models(args.models)
    snapshot_only = cfg.get_bool_or("snapshot_only_report", False)
    preamble_path = cfg.get_preamble_path()
    epilogue_path = cfg.get_epilogue_path()
    question_tagging = cfg.get_question_tagging()
    questions_per_prompt = cfg.get_questions_per_prompt()

    print(f"config: {cfg.path}")
    print(f"label:  {label}")

    scenarios = get_base_scenarios(
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
        cfg.get_bool_or("report_effectiveness", False),
        cfg.get_bool_or("censorCityFunds", True),
        cfg.get_questions_sort(),
    )

    # A config written for the continuous eval can name windows the binary
    # questions reject — horizon 0 is a read-off there but an empty window
    # here. run_eval_binary.py could never run such a config, so it implies no
    # binary calls at all.
    try:
        corpus_binary = build_corpus_binary(
            scenarios,
            cfg.get_int_list("snapshot_turns"),
            cfg.get_int_list("horizons"),
            cfg.get_int("history_freq"),
            label,
            snapshot_only,
            cfg.get_int_or("history_length", -1),
            cfg.get_bool_or("report_effectiveness", False),
            cfg.get_bool_or("censorCityFunds", True),
            cfg.get_bool_or("report_census", True),
        )
    except ValueError as e:
        print(f"binary eval not applicable to this config: {e}")
        corpus_binary = []

    def batch_hashes(corpus: list[dict], build_prompt) -> dict[str, str]:
        batches = group_into_batches(corpus, questions_per_prompt)
        return {
            bid: prompt_hash(build_prompt(questions[0]["context"], questions))
            for bid, questions in batches.items()
        }

    continuous_hashes = batch_hashes(
        corpus,
        lambda context, questions: build_batch_prompt_continuous(
            context, questions, preamble_path, epilogue_path, question_tagging
        ),
    )
    binary_hashes = batch_hashes(
        corpus_binary,
        lambda context, questions: build_batch_prompt_binary(
            context, questions, preamble_path, epilogue_path
        ),
    )
    return {
        "continuous": ur.collect_for_batches(
            continuous_hashes, model_names, response_path, usage_path, seen
        ),
        "binary": ur.collect_for_batches(
            binary_hashes,
            model_names,
            BINARY_PATHS.response_path,
            BINARY_PATHS.usage_path,
            seen,
        ),
    }


def report_tables(collected: ur.Collected, per_model: bool) -> str:
    """The provider table for `collected`, and the per-model one when asked for."""
    if not collected.usages:
        return "No usage data found."
    parts = [ur.format_table(ur.by_provider(collected.usages), "provider")]
    if per_model:
        parts.append(ur.format_table(ur.by_model(collected.usages), "model"))
    return "\n\n".join(parts)


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_config_args(ap, many=True)
    ap.add_argument(
        "--per-config",
        action="store_true",
        help="also print each config's own table, in the order given; a call "
        "two configs share is counted under the first of them",
    )
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
        # One `seen` across every config, so a call more than one of them asks
        # for is counted once: the response was cached and paid for once.
        seen: set[tuple[str, str, str]] = set()
        per_config = []
        for i, cfg in enumerate(load_configs(args)):
            if i:
                print()
            per_config.append((cfg, collect_from_config(cfg, args, seen)))
        if args.per_config and len(per_config) > 1:
            # Each config's own share of the total: the calls it is the first to
            # ask for. A config listed later than one it overlaps therefore shows
            # only what the earlier one did not already account for, which is
            # what makes these tables add up to the total below.
            for cfg, one in per_config:
                print()
                print(f"--- {cfg.path} ---")
                print(report_tables(ur.merge(list(one.values())), args.per_model))
        collected = ur.merge([c for _, one in per_config for c in one.values()])
        if len(per_config) > 1:
            print()
            print(f"TOTAL over {len(per_config)} configs, shared calls counted once")

    print()
    print(report_tables(collected, args.per_model))

    # One line rather than a warning per file: an old cache legitimately has no
    # sidecars at all, and a screen of warnings would bury the totals.
    if collected.missing:
        print(f"\nUsage data not stored for {collected.missing} API calls")
    if collected.unprompted:
        by_eval = ""
        if not args.glob:
            shares = {
                eval_name: sum(one[eval_name].unprompted for _, one in per_config)
                for eval_name in ("continuous", "binary")
            }
            by_eval = " (" + ", ".join(
                f"{n} {eval_name}" for eval_name, n in shares.items() if n
            ) + ")"
        print(f"{collected.unprompted} batch/model pairs not yet prompted{by_eval}")

    total = ur.grand_total(ur.by_provider(collected.usages))
    if total.unpriced:
        # Excluded from the cost column above, so the total is a floor.
        print(
            f"{total.unpriced} call(s) had no price; "
            "the cost above excludes them"
        )
    print("=" * 70)


if __name__ == "__main__":
    main()
