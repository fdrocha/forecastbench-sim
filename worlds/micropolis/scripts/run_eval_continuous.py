#!/usr/bin/env -S uv run python3
"""Continuous forecast evaluation: gather model forecasts.

Builds the question corpus for every (city, disasters) combination in the
config file, prompts each configured model on it, and writes the questions and
their forecasts to data/micropolis/continuous/data.json.

Questions that share a game report — same scenario and snapshot turn — are
asked together in one numbered prompt, so the report is paid for once per
batch instead of once per question. The uncached batch calls all run
concurrently, capped per provider (see micropolis_world/prompting.py; a
config's "provider_concurrency" map adjusts the caps), with one progress line
printed as each response lands and the response written to disk right then. Each batch's prompt and raw responses are
cached in data/micropolis/continuous/cache/{batch_id}/ as prompt-{hash}.txt
and one response-{model}-{hash}.txt per model, where {hash} is the first 8
characters of the prompt's SHA-256 digest (see knowledge_eval/runner.py's
prompt_hash) — the same convention the knowledge eval uses. A response is only
reused when its hash matches the freshly built prompt, so trying a different
prompt variant (template, history_freq, ...) never mixes its answers with an
older variant's; it just adds new files alongside them. data.json is
regenerated from the cached responses on every run, so parser improvements
take effect without re-prompting.

Scoring and plotting read that file:
    scripts/analyze_continuous.py   tables of CRPS
    scripts/plot_forecasts.py        trajectories with forecasts overlaid

Usage:
    scripts/run_eval_continuous.py
    scripts/run_eval_continuous.py my_config.json --seed 7
    scripts/run_eval_continuous.py my_config.json --dry-run
    scripts/run_eval_continuous.py --models openai/gpt-4o xai/grok-4-0709
    scripts/run_eval_continuous.py --cities kyoto bruce --disasters false
    scripts/run_eval_continuous.py --label kyoto_only --cities kyoto

--models overrides the config's list; see data/micropolis/available_models.md
for what each provider offers.
"""

import argparse
from pathlib import Path

import micropolis_world.module_globals as g
from micropolis_world.config import (
    QUESTION_TAGGING_NUMERIC,
    QUESTION_TAGGING_SEMANTIC,
    QUESTIONS_SORT_TURN,
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.continuous_eval import (
    PATHS,
    Response,
    ResponseId,
    Responses,
    data_path,
    save_dataset,
)
from micropolis_world.gather import gather_raw_responses
from micropolis_world.scenarios import (
    build_batch_prompt_continuous,
    build_corpus,
    get_base_scenarios,
    parse_batch_percentiles,
    parse_batch_percentiles_semantic,
)


def gather_responses(
    corpus: list[dict],
    model_names: list[str],
    max_tokens: int,
    provider_limits: dict[str, int] | None = None,
    preamble_path: Path | None = None,
    epilogue_path: Path | None = None,
    question_tagging: str = QUESTION_TAGGING_NUMERIC,
    questions_per_prompt: int = -1,
) -> Responses:
    """Prompt each model on each batch, then read percentiles out of the replies.

    The gathering itself — batching, the prompt hash cache, the concurrent
    calls, the cost accounting — is gather.gather_raw_responses, shared with
    the binary eval. What is continuous-specific is the prompt this asks for
    and the percentile sets read back out of it.
    """
    batches, raws = gather_raw_responses(
        corpus,
        model_names,
        max_tokens,
        paths=PATHS,
        build_prompt=lambda context, questions: build_batch_prompt_continuous(
            context, questions, preamble_path, epilogue_path, question_tagging
        ),
        provider_limits=provider_limits,
        questions_per_prompt=questions_per_prompt,
    )

    # Parse after the gather, in the stable model x batch order. Cached text,
    # fresh text and empty replies (text None) all take the same path; a failed
    # call has no raws entry and so gets no Response rows at all, which
    # downstream (select_for_config) reads as "never gathered; re-run" —
    # exactly right, since re-running retries it.
    responses: Responses = {}
    for model_name in model_names:
        for bid, questions in batches.items():
            if (bid, model_name) not in raws:
                continue
            raw = raws[(bid, model_name)]
            # The same question_id appears once per model, so name both.
            labels = [f"{model_name} {q['question_id']}" for q in questions]
            if question_tagging == QUESTION_TAGGING_SEMANTIC:
                percentile_sets = parse_batch_percentiles_semantic(
                    raw, labels, [q["semantic_tag"] for q in questions]
                )
            else:
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
    args = ap.parse_args()

    cfg = load_config(args)
    seed = cfg.get_seed(args.seed)
    label = cfg.get_label(args.label)
    models = cfg.get_models(args.models)
    out_path = data_path(label)
    # Read once and passed to both build_corpus and gather_responses, so the
    # report the corpus carries and the templates the prompt names it with can
    # never disagree about which variant this run is.
    snapshot_only = cfg.get_bool_or("snapshot_only_report", False)
    censor_city_funds = cfg.get_bool_or("censorCityFunds", True)
    preamble_path = cfg.get_preamble_path()
    epilogue_path = cfg.get_epilogue_path()
    questions_sort = cfg.get_questions_sort()
    question_tagging = cfg.get_question_tagging()
    questions_per_prompt = cfg.get_questions_per_prompt()

    print("=" * 70)
    print("MICROPOLIS WORLD — continuous eval")
    print("=" * 70)
    print(f"config: {cfg.path}")
    print(f"label:  {label}")
    if preamble_path is not None:
        print(f"preamble: {preamble_path}")
    if epilogue_path is not None:
        print(f"epilogue: {epilogue_path}")
    if snapshot_only:
        print("report: snapshot only (no HISTORY table)")
    if censor_city_funds:
        print("report: city funds censored")
    if questions_sort != QUESTIONS_SORT_TURN:
        print(f"questions sorted by: {questions_sort}")
    if question_tagging != QUESTION_TAGGING_NUMERIC:
        print(f"question tagging: {question_tagging}")
    if questions_per_prompt > 0:
        print(f"questions per prompt: at most {questions_per_prompt}")

    scenarios = get_base_scenarios(
        seed=seed,
        cities=cfg.get_cities(args.cities),
        disasters=cfg.get_disasters(args.disasters),
    )
    print(f"\nRunning {len(scenarios)} with seed={seed}; building corpus...")
    corpus = build_corpus(
        scenarios,
        cfg.get_int_list("snapshot_turns"),
        cfg.get_int_list("horizons"),
        cfg.get_int("history_freq"),
        label,
        snapshot_only,
        cfg.get_int_or("history_length", -1),
        cfg.get_bool_or("report_effectiveness", False),
        censor_city_funds,
        questions_sort,
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
        cfg.get_provider_concurrency(),
        preamble_path,
        epilogue_path,
        question_tagging,
        questions_per_prompt,
    )
    print("Done gathering")

    save_dataset(corpus, responses, models, out_path)
    usable = sum(1 for r in responses.values() if r.percentiles is not None)
    print(f"\nWrote {len(corpus)} questions x {len(models)} models -> {out_path}")
    print(f"  {usable} of {len(responses)} forecasts usable")
    print("=" * 70)
    print("Next: scripts/analyze_continuous.py, scripts/plot_forecasts.py")


if __name__ == "__main__":
    main()
