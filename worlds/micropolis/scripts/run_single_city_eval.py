#!/usr/bin/env -S uv run python3
"""Single city scenario evaluation: gather model forecasts.

Builds the question corpus for every (city, disasters) combination in the
config file, prompts each configured model on it, and writes the questions and
their forecasts to data/micropolis/single_city/data.json.

Scoring and plotting read that file:
    scripts/analyze_single_city.py   tables of CRPS
    scripts/plot_forecasts.py        trajectories with forecasts overlaid

Usage:
    scripts/run_single_city_eval.py
    scripts/run_single_city_eval.py my_config.json --seed 7
    scripts/run_single_city_eval.py my_config.json --dry-run
    scripts/run_single_city_eval.py --models openai/gpt-4o xai/grok-4-0709
    scripts/run_single_city_eval.py --verbose-reparse

--models overrides the config's list; see data/micropolis/available_models.md
for what each provider offers.
"""

import argparse

import micropolis_world.module_globals as g
from fbsim_core.evaluation.models import get_models
from micropolis_world.config import (
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.scenarios import (
    build_corpus,
    build_prompt_continuous,
    get_single_city_base_scenarios,
    parse_percentiles,
)
from micropolis_world.single_city import (
    DATA_PATH,
    Response,
    ResponseId,
    Responses,
    load_cache,
    save_cache,
    save_dataset,
)
from tqdm import tqdm


def gather_responses(
    corpus: list[dict],
    model_names: list[str],
    max_tokens: int,
    verbose_reparse: bool = False,
) -> Responses:
    """Prompt each model on each corpus question, reusing cached responses.

    The whole cache is loaded and saved back, so responses for other corpora and
    models are preserved, but only the responses for this call's corpus and
    model_names are returned.

    verbose_reparse is passed to load_cache, printing why each cached response
    that no longer parses was rejected instead of just how many.
    """
    n_new = 0
    cache = load_cache(verbose_reparse=verbose_reparse)
    # Re-parsing is quiet unless asked otherwise, so say how many stored
    # responses have no usable forecast rather than letting them vanish.
    n_bad_cached = sum(1 for r in cache.values() if r.percentiles is None)
    if n_bad_cached:
        hint = "" if verbose_reparse else "; --verbose-reparse to see why"
        print(
            f"{n_bad_cached} of {len(cache)} cached responses have no usable "
            f"percentiles (re-parsed from the stored text{hint})"
        )
    responses: Responses = {}
    models = get_models(model_names)
    nmodels = len(models)
    for model_idx, (model_name, model) in enumerate(zip(model_names, models)):
        print(f"Prompting {model_name} ({model_idx + 1}/{nmodels})")
        for c in tqdm(corpus):
            response_id = ResponseId(model_id=model_name, question_id=c["question_id"])
            if response_id in cache:
                response = cache[response_id]
            else:
                # prompt_model rather than model.get_response, because the
                # finish reason is what distinguishes a model that answered
                # badly from one that never got to answer at all.
                raw, finish_reason = g.prompt_model(
                    model,
                    build_prompt_continuous(c["context"], c["question_text"]),
                    max_tokens,
                )
                g.warn_if_truncated(model_name, finish_reason, max_tokens)
                # parse_percentiles reports its own reason for rejecting a
                # response, so only the question id needs adding here.
                percentiles = parse_percentiles(raw, label=response_id.question_id)
                response = Response(
                    response_text=raw,
                    actual=c["value"],
                    percentiles=percentiles,
                )
                cache[response_id] = response

                n_new += 1
                if n_new % 10 == 0:
                    save_cache(cache)
            responses[response_id] = response

    if n_new % 10:
        save_cache(cache)
    return responses


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        default=None,
        help="Model ids to prompt, overriding the config's 'models' list. "
        "Ids are in provider/name form; see data/micropolis/available_models.md",
    )
    ap.add_argument(
        "--verbose-reparse",
        action="store_true",
        help="print the full warning for every cached response whose percentiles "
        "no longer parse, rather than only the count",
    )
    args = ap.parse_args()

    cfg = load_config(args)
    seed = cfg.get_seed(args.seed)
    models = args.models if args.models else cfg.get_str_list("models")

    print("=" * 70)
    print("MICROPOLIS WORLD — single city eval")
    print("=" * 70)
    print(f"config: {cfg.path}")

    scenarios = get_single_city_base_scenarios(
        seed=seed, cities=cfg.get_cities(), disasters=cfg.get_bool_list("disasters")
    )
    print(f"\nRunning {len(scenarios)} with seed={seed}; building corpus...")
    corpus = build_corpus(
        scenarios,
        cfg.get_int_list("snapshot_turns"),
        cfg.get_int_list("horizons"),
        cfg.get_int("history_freq"),
    )

    if args.dry_run:
        print("\nDry run. Exiting")
        return

    print("\nGathering model responses...")
    g.ensure_api_keys()
    responses = gather_responses(
        corpus,
        models,
        cfg.get_int("max_tokens"),
        verbose_reparse=args.verbose_reparse,
    )
    print("Done gathering")

    save_dataset(corpus, responses, models)
    usable = sum(1 for r in responses.values() if r.percentiles is not None)
    print(f"\nWrote {len(corpus)} questions x {len(models)} models -> {DATA_PATH}")
    print(f"  {usable} of {len(responses)} forecasts usable")
    print("=" * 70)
    print("Next: scripts/analyze_single_city.py, scripts/plot_forecasts.py")


if __name__ == "__main__":
    main()
