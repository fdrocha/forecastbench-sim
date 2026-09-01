#!/usr/bin/env -S uv run python3
"""Binary yes/no evaluation: gather model probability forecasts.

The binary counterpart of run_eval_continuous.py: builds the 27-question
corpus from binary_questions.py for every (city, disasters) combination in the
config, prompts each configured model for one P(Yes) per question, and writes
the questions and forecasts to data/micropolis/binary/{label}/data.json.

Questions sharing a game report — same scenario and snapshot turn — are asked
together in one numbered prompt. Prompts and raw responses are cached under
data/micropolis/binary/cache/{batch_id}/ with the same content-addressed
naming as the continuous eval, so a config change misses the cache rather
than mixing variants, and data.json is regenerated from cached responses on
every run. The gather loop below mirrors run_eval_continuous.gather_responses
(minus its semantic-tagging branches); a third eval variant should extract the
shared machinery rather than copy it again.

Usage:
    scripts/run_eval_binary.py                      # configs/binary.json5
    scripts/run_eval_binary.py my_config.json5 --seed 7
    scripts/run_eval_binary.py --dry-run            # corpus + Yes counts only
    scripts/run_eval_binary.py --cities kyoto bruce
"""

import argparse
import asyncio
import time
from collections import Counter, defaultdict
from pathlib import Path

from fbsim_core.evaluation.models import get_models

import micropolis_world.module_globals as g
from micropolis_world.binary_eval import (
    BinaryResponse,
    BinaryResponses,
    batch_dir,
    data_path,
    prompt_path,
    response_path,
    save_dataset_binary,
    usage_path,
)
from micropolis_world.binary_questions import QUESTION_IDS, build_corpus_binary
from micropolis_world.config import (
    CONFIG_DIR,
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.continuous_eval import (
    ResponseId,
    group_into_batches,
    prompt_hash,
    save_usage,
)
from micropolis_world.prompting import (
    PromptJob,
    PromptResult,
    format_eta,
    format_latency,
    run_prompts,
)
from micropolis_world.scenarios import (
    build_batch_prompt_binary,
    get_base_scenarios,
    parse_batch_probabilities,
)

DEFAULT_BINARY_CONFIG_PATH = CONFIG_DIR / "binary.json5"


def gather_responses_binary(
    corpus: list[dict],
    model_names: list[str],
    max_tokens: int,
    provider_limits: dict[str, int] | None = None,
    preamble_path: Path | None = None,
    epilogue_path: Path | None = None,
    questions_per_prompt: int = -1,
) -> BinaryResponses:
    """Prompt each model on each batch of questions, reusing cached responses.

    Same contract as run_eval_continuous.gather_responses: cache files are
    named with the prompt's hash, so a response is only reused under the exact
    prompt being asked now; an empty reply is not cached (retried next run), a
    non-empty one is cached even when unparseable; uncached calls run
    concurrently capped per provider, each response written the moment it
    lands; a failed call is reported and skipped, so re-running retries
    exactly the failures.
    """
    batches = group_into_batches(corpus, questions_per_prompt)
    prompts = {
        bid: build_batch_prompt_binary(
            questions[0]["context"],
            questions,
            preamble_path,
            epilogue_path,
        )
        for bid, questions in batches.items()
    }
    phashes = {bid: prompt_hash(prompt) for bid, prompt in prompts.items()}
    ppaths = {bid: prompt_path(bid, phashes[bid]) for bid in prompts}
    for bid, prompt in prompts.items():
        if not ppaths[bid].exists():
            batch_dir(bid).mkdir(parents=True, exist_ok=True)
            ppaths[bid].write_text(prompt)

    models = get_models(model_names)
    rpaths = {
        (bid, model_name): response_path(bid, model_name, phashes[bid])
        for bid in batches
        for model_name in model_names
    }
    ncached = sum(1 for p in rpaths.values() if p.exists())
    print(
        f"{ncached} of {len(rpaths)} batch responses cached; "
        f"generating {len(rpaths) - ncached}"
    )

    raws: dict[tuple[str, str], str | None] = {}
    jobs: list[PromptJob] = []
    for model_name, model in zip(model_names, models):
        for bid in batches:
            key = (bid, model_name)
            if rpaths[key].exists():
                raws[key] = rpaths[key].read_text()
            else:
                jobs.append(
                    PromptJob(
                        key=key,
                        model=model,
                        model_name=model_name,
                        messages=[{"role": "user", "content": prompts[bid]}],
                        max_tokens=max_tokens,
                    )
                )

    model_cost: dict[str, float] = defaultdict(float)
    nunpriced: Counter = Counter()
    failures: list[PromptResult] = []

    async def consume() -> None:
        # Warm litellm's slow first import in the loop that will use it.
        import litellm  # noqa: F401

        done = 0
        start = time.perf_counter()
        async for result in run_prompts(jobs, limits=provider_limits):
            done += 1
            eta = format_eta(start, done, len(jobs))
            bid, model_name = result.job.key
            prefix = f"[{done}/{len(jobs)}] {model_name} <- {ppaths[bid]}"
            if not result.ok:
                failures.append(result)
                err = result.error
                print(f"{prefix}  FAILED: {type(err).__name__}: {err}{eta}", flush=True)
                continue
            resp = result.response
            took = format_latency(resp.usage.latency_ms, resp.retries)
            cost = resp.usage.cost_usd
            if cost is None:
                nunpriced[model_name] += 1
                print(
                    f"{prefix}  cost unknown, {resp.usage.tokens()}{took}{eta}",
                    flush=True,
                )
            else:
                model_cost[model_name] += cost
                print(
                    f"{prefix}  {cost * 100:.3f}c, {resp.usage.tokens()}{took}{eta}",
                    flush=True,
                )
            g.warn_if_truncated(model_name, resp.finish_reason, max_tokens)
            raws[result.job.key] = resp.text
            if resp.text:
                rpaths[result.job.key].write_text(resp.text)
                save_usage(usage_path(bid, model_name, phashes[bid]), resp.usage)

    if jobs:
        asyncio.run(consume())

    # Parse after the gather, in the stable model x batch order. A failed call
    # has no raws entry and so gets no response rows at all — "never gathered;
    # re-run", exactly right since re-running retries it.
    responses: BinaryResponses = {}
    for model_name in model_names:
        for bid, questions in batches.items():
            if (bid, model_name) not in raws:
                continue
            raw = raws[(bid, model_name)]
            labels = [f"{model_name} {q['question_id']}" for q in questions]
            probabilities = parse_batch_probabilities(raw, labels)
            for q, probability in zip(questions, probabilities):
                responses[ResponseId(model_name, q["question_id"])] = BinaryResponse(
                    actual=q["answer"],
                    probability=probability,
                    response_text=raw,
                )

    print("Per-model totals for this run:")
    for model_name in model_names:
        total = f"${model_cost[model_name]:.2f}"
        if nunpriced[model_name]:
            total += f" + {nunpriced[model_name]} call(s) litellm could not price"
        print(f"  {model_name}: {total}")
    if failures:
        print(
            f"{len(failures)} call(s) failed (not cached; "
            "re-run this script to retry them):"
        )
        for result in failures:
            bid, model_name = result.job.key
            print(f"  {model_name} <- {ppaths[bid]}")
            print(f"    {type(result.error).__name__}: {result.error}")
    return responses


def print_yes_counts(corpus: list[dict]) -> None:
    """Per-question Yes counts, for eyeballing against the doc's P(Yes) ranges."""
    windows = sorted({(c["snapshot_turn"], c["horizon"]) for c in corpus})
    yes = Counter(
        (c["qid"], c["snapshot_turn"], c["horizon"]) for c in corpus if c["answer"]
    )
    nscenarios = len({c["scenario_id"] for c in corpus})
    header = "  ".join(f"T{t}+{h}" for t, h in windows)
    print(f"\nYes counts out of {nscenarios} scenario(s) per window:")
    print(f"  {'':>4} {header}")
    for qid in QUESTION_IDS:
        counts = "  ".join(
            f"{yes[(qid, t, h)]:>{len(f'T{t}+{h}')}}" for t, h in windows
        )
        print(f"  {qid:>4} {counts}")


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default=DEFAULT_BINARY_CONFIG_PATH)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args)
    seed = cfg.get_seed(args.seed)
    label = cfg.get_label(args.label)
    models = cfg.get_models(args.models)
    out_path = data_path(label)
    censor_city_funds = cfg.get_bool_or("censorCityFunds", True)
    report_census = cfg.get_bool_or("report_census", True)
    preamble_path = cfg.get_preamble_path()
    epilogue_path = cfg.get_epilogue_path()
    questions_per_prompt = cfg.get_questions_per_prompt()

    print("=" * 70)
    print("MICROPOLIS WORLD — binary eval")
    print("=" * 70)
    print(f"config: {cfg.path}")
    print(f"label:  {label}")
    if preamble_path is not None:
        print(f"preamble: {preamble_path}")
    if epilogue_path is not None:
        print(f"epilogue: {epilogue_path}")
    if not report_census:
        print("report: census section omitted")
    if questions_per_prompt > 0:
        print(f"questions per prompt: at most {questions_per_prompt}")

    scenarios = get_base_scenarios(
        seed=seed,
        cities=cfg.get_cities(args.cities),
        disasters=cfg.get_disasters(args.disasters),
    )
    print(f"\nRunning {len(scenarios)} with seed={seed}; building corpus...")
    corpus = build_corpus_binary(
        scenarios,
        cfg.get_int_list("snapshot_turns"),
        cfg.get_int_list("horizons"),
        cfg.get_int("history_freq"),
        label,
        cfg.get_bool_or("snapshot_only_report", False),
        cfg.get_int_or("history_length", -1),
        cfg.get_bool_or("report_effectiveness", False),
        censor_city_funds,
        report_census,
    )

    if args.dry_run:
        print_yes_counts(corpus)
        print("\nDry run. Exiting")
        return

    print("\nGathering model responses...")
    g.ensure_api_keys()
    responses = gather_responses_binary(
        corpus,
        models,
        cfg.get_int("max_tokens"),
        cfg.get_provider_concurrency(),
        preamble_path,
        epilogue_path,
        questions_per_prompt,
    )
    print("Done gathering")

    save_dataset_binary(corpus, responses, models, out_path)
    usable = sum(1 for r in responses.values() if r.probability is not None)
    print(f"\nWrote {len(corpus)} questions x {len(models)} models -> {out_path}")
    print(f"  {usable} of {len(responses)} forecasts usable")
    print("=" * 70)


if __name__ == "__main__":
    main()
