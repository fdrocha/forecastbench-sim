#!/usr/bin/env -S uv run python3
"""Gather each model's answers to the Micropolis domain-knowledge test.

Asks every configured model to classify the statement set as True/False/Unknown
and caches the replies, so re-running only prompts models that have not answered
yet. This is the only script in the knowledge eval that calls a model; to score
what it gathered, run scripts/analyze_knowledge.py.

Usage:
    scripts/run_eval_knowledge.py
    scripts/run_eval_knowledge.py my_config.json5
    scripts/run_eval_knowledge.py --models openai/gpt-4o xai/grok-4-0709

Without a config argument this uses configs/knowledge_eval.json5. --models
overrides the config's list; see data/micropolis/available_models.md for what
each provider offers.
"""

import argparse

import micropolis_world.module_globals as g
from micropolis_world import messages as msg
from micropolis_world.config import (
    CONFIG_DIR,
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.knowledge_eval.runner import (
    CACHE_DIR,
    get_model_answers,
    statements,
)

DEFAULT_CONFIG = CONFIG_DIR / "knowledge_eval.json5"


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    args = ap.parse_args()

    print("=" * 70)
    print("MICROPOLIS WORLD — domain knowledge eval")
    print("=" * 70)

    # The config argument is optional; fall back to this script's own.
    if args.config is None:
        args.config = DEFAULT_CONFIG
    cfg = load_config(args)
    models = cfg.get_models(args.models)

    print(f"config: {cfg.path}")
    print(f"{len(statements)} statements, {len(models)} model(s)\n")

    g.ensure_api_keys()
    answers = get_model_answers(models, concurrency=cfg.get_concurrency())

    print(f"\n{'=' * 70}")
    print(f"Gathered answers from {len(answers)} of {len(models)} model(s)")
    print("=" * 70)

    missing = [m for m in models if m not in answers]
    if missing:
        msg.warn(f"no response from: {', '.join(missing)}")
    print(f"Responses saved under {CACHE_DIR}")
    print("\nTo score them, run scripts/analyze_knowledge.py")


if __name__ == "__main__":
    main()
