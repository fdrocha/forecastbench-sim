#!/usr/bin/env -S uv run python3
"""Single city scenario evaluation: gather model forecasts.

Builds the question corpus for every (city, disasters) combination in the
config file, prompts each configured model on it, and writes the questions and
their forecasts to data/micropolis/single_city/data.json.

Questions that share a game report — same scenario and snapshot turn — are
asked together in one numbered prompt, so the report is paid for once per
batch instead of once per question. Each batch's prompt and raw responses are
cached in data/micropolis/single_city/cache/{batch_id}/ as prompt.txt and one
response-{model}.txt per model; cached responses are never re-requested, and
deleting a batch directory re-gathers it. data.json is regenerated from the
cached responses on every run, so parser improvements take effect without
re-prompting.

Scoring and plotting read that file:
    scripts/analyze_single_city.py   tables of CRPS
    scripts/plot_forecasts.py        trajectories with forecasts overlaid

Usage:
    scripts/run_single_city_eval.py
    scripts/run_single_city_eval.py my_config.json --seed 7
    scripts/run_single_city_eval.py my_config.json --dry-run
    scripts/run_single_city_eval.py --models openai/gpt-4o xai/grok-4-0709

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
    build_batch_prompt_continuous,
    build_corpus,
    get_single_city_base_scenarios,
    parse_batch_percentiles,
)
from micropolis_world.single_city import (
    DATA_PATH,
    Response,
    ResponseId,
    Responses,
    batch_dir,
    batch_id_for,
    prompt_path,
    response_path,
    save_dataset,
)
from tqdm import tqdm


def group_into_batches(corpus: list[dict]) -> dict[str, list[dict]]:
    """Group corpus questions by batch_id, preserving corpus order."""
    batches: dict[str, list[dict]] = {}
    for c in corpus:
        batches.setdefault(batch_id_for(c), []).append(c)
    return batches


def check_prompts_against_cache(prompts: dict[str, str]) -> None:
    """Refuse to run when a cached batch was prompted with different questions.

    A batch directory answers the exact prompt stored in it, but batch_id names
    only the scenario and snapshot turn: a config change to horizons, templates
    or history_freq changes the prompt under the same batch_id, and reusing the
    cached responses would silently attach them to different questions. Deleting
    the stale directories is left to the user, so nothing gathered is thrown
    away without them seeing what and why.
    """
    stale = [
        bid
        for bid, prompt in prompts.items()
        if prompt_path(bid).exists() and prompt_path(bid).read_text() != prompt
    ]
    if stale:
        dirs = "\n".join(f"  {batch_dir(bid)}" for bid in stale)
        raise SystemExit(
            f"{len(stale)} cached batch(es) were prompted with different "
            "questions than this config builds (changed horizons, templates, "
            "or history_freq?). Delete the stale batch directories to "
            f"re-gather them:\n{dirs}"
        )


def gather_responses(
    corpus: list[dict],
    model_names: list[str],
    max_tokens: int,
) -> Responses:
    """Prompt each model on each batch of questions, reusing cached responses.

    A model is only queried for batches with no cached response file, so
    re-running is free once a run completes. An empty reply — a reasoning
    model can burn the whole token budget thinking — is not cached, so the
    next run retries it; a non-empty reply is cached even when unparseable,
    since retrying greedy decoding would return the same text.
    """
    batches = group_into_batches(corpus)
    prompts = {
        bid: build_batch_prompt_continuous(questions[0]["context"], questions)
        for bid, questions in batches.items()
    }
    check_prompts_against_cache(prompts)
    for bid, prompt in prompts.items():
        if not prompt_path(bid).exists():
            batch_dir(bid).mkdir(parents=True, exist_ok=True)
            prompt_path(bid).write_text(prompt)

    responses: Responses = {}
    models = get_models(model_names)
    nmodels = len(models)
    for model_idx, (model_name, model) in enumerate(zip(model_names, models)):
        print(f"Prompting {model_name} ({model_idx + 1}/{nmodels})")
        for bid, questions in tqdm(batches.items()):
            rpath = response_path(bid, model_name)
            if rpath.exists():
                raw = rpath.read_text()
            else:
                # prompt_model rather than model.get_response, because the
                # finish reason is what distinguishes a model that answered
                # badly from one that never got to answer at all.
                raw, finish_reason = g.prompt_model(model, prompts[bid], max_tokens)
                g.warn_if_truncated(model_name, finish_reason, max_tokens)
                if raw:
                    rpath.write_text(raw)
            # The same question_id appears once per model, so name both.
            labels = [f"{model_name} {q['question_id']}" for q in questions]
            percentile_sets = parse_batch_percentiles(raw, labels)
            for q, percentiles in zip(questions, percentile_sets):
                responses[ResponseId(model_name, q["question_id"])] = Response(
                    actual=q["value"],
                    percentiles=percentiles,
                    response_text=raw,
                )
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
    responses = gather_responses(corpus, models, cfg.get_int("max_tokens"))
    print("Done gathering")

    save_dataset(corpus, responses, models)
    usable = sum(1 for r in responses.values() if r.percentiles is not None)
    print(f"\nWrote {len(corpus)} questions x {len(models)} models -> {DATA_PATH}")
    print(f"  {usable} of {len(responses)} forecasts usable")
    print("=" * 70)
    print("Next: scripts/analyze_single_city.py, scripts/plot_forecasts.py")


if __name__ == "__main__":
    main()
