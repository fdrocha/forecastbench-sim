#!/usr/bin/env -S uv run python3
"""Single city scenario evaluation: gather model forecasts.

Builds the question corpus for every (city, disasters) combination in the
config file, prompts each configured model on it, and writes the questions and
their forecasts to data/micropolis/single_city/data.json.

Questions that share a game report — same scenario and snapshot turn — are
asked together in one numbered prompt, so the report is paid for once per
batch instead of once per question. The uncached batch calls all run
concurrently, capped per provider (see micropolis_world/prompting.py; a
config's "provider_concurrency" map adjusts the caps), with one progress line
printed as each response lands and the response written to disk right then. Each batch's prompt and raw responses are
cached in data/micropolis/single_city/cache/{batch_id}/ as prompt-{hash}.txt
and one response-{model}-{hash}.txt per model, where {hash} is the first 8
characters of the prompt's SHA-256 digest (see knowledge_eval/runner.py's
prompt_hash) — the same convention the knowledge eval uses. A response is only
reused when its hash matches the freshly built prompt, so trying a different
prompt variant (template, history_freq, ...) never mixes its answers with an
older variant's; it just adds new files alongside them. data.json is
regenerated from the cached responses on every run, so parser improvements
take effect without re-prompting.

Scoring and plotting read that file:
    scripts/analyze_single_city.py   tables of CRPS
    scripts/plot_forecasts.py        trajectories with forecasts overlaid

Usage:
    scripts/run_single_city_eval.py
    scripts/run_single_city_eval.py my_config.json --seed 7
    scripts/run_single_city_eval.py my_config.json --dry-run
    scripts/run_single_city_eval.py --models openai/gpt-4o xai/grok-4-0709
    scripts/run_single_city_eval.py --cities kyoto bruce --disasters false
    scripts/run_single_city_eval.py --label kyoto_only --cities kyoto

--models overrides the config's list; see data/micropolis/available_models.md
for what each provider offers.
"""

import argparse
import asyncio
import time
from collections import Counter, defaultdict
from pathlib import Path

import micropolis_world.module_globals as g
from fbsim_core.evaluation.models import get_models
from micropolis_world.config import (
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.prompting import (
    PromptJob,
    PromptResult,
    format_eta,
    run_prompts,
)
from micropolis_world.scenarios import (
    build_batch_prompt_continuous,
    build_corpus,
    get_single_city_base_scenarios,
    parse_batch_percentiles,
)
from micropolis_world.single_city import (
    Response,
    ResponseId,
    Responses,
    batch_dir,
    data_path,
    group_into_batches,
    prompt_hash,
    prompt_path,
    response_path,
    save_dataset,
    save_usage,
    usage_path,
)


def gather_responses(
    corpus: list[dict],
    model_names: list[str],
    max_tokens: int,
    snapshot_only_report: bool = False,
    provider_limits: dict[str, int] | None = None,
    preamble_path: Path | None = None,
) -> Responses:
    """Prompt each model on each batch of questions, reusing cached responses.

    Cache files are named with the prompt's hash (see single_city.prompt_hash),
    so a response is only ever reused when it was gathered under the exact
    prompt being asked now — a config change to horizons, templates,
    history_freq, history_length, snapshot_only_report or preamble_path changes
    the hash, which simply misses the cache rather than risking a stale match.
    An empty reply — a reasoning model can burn the whole token budget
    thinking — is not cached, so the next run retries it; a non-empty reply is
    cached even when unparseable, since retrying greedy decoding would return
    the same text.

    All uncached (batch, model) calls run concurrently, capped per provider by
    run_prompts, and each response is written to its cache file the moment it
    lands, so an interrupted run keeps what it already paid for. A failed call
    is reported and skipped rather than aborting the run — nothing is cached
    for it, so re-running the script retries exactly the failures.
    """
    batches = group_into_batches(corpus)
    prompts = {
        bid: build_batch_prompt_continuous(
            questions[0]["context"], questions, snapshot_only_report, preamble_path
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

    # Split cache hits from the calls still to make. raws holds the text to
    # parse per (batch_id, model_name); a key that is still absent at parse
    # time means the call failed this run.
    raws: dict[tuple[str, str], str | None] = {}
    jobs: list[PromptJob] = []
    for model_name, model in zip(model_names, models):
        for bid in batches:
            key = (bid, model_name)
            if rpaths[key].exists():
                raws[key] = rpaths[key].read_text()
            else:
                # prompt_model_async rather than model.get_response, because
                # the finish reason is what distinguishes a model that answered
                # badly from one that never got to answer at all, and the usage
                # is what it cost either way.
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
        # Warm litellm's slow first import in the loop that will use it, rather
        # than paying for it under the first worker's provider semaphore.
        import litellm  # noqa: F401

        # The workers only make API calls; every print, dict update and disk
        # write happens here in the single consumer task, so nothing needs a
        # lock. One complete line per completed call — completions from
        # different providers interleave, so a line can't be left dangling for
        # its cost the way the serial version's was.
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
                print(
                    f"{prefix}  FAILED: {type(err).__name__}: {err}{eta}", flush=True
                )
                continue
            resp = result.response
            cost = resp.usage.cost_usd
            if cost is None:
                # Reported, not counted: a model litellm has no price for
                # would otherwise be summed into the total as free.
                nunpriced[model_name] += 1
                print(f"{prefix}  cost unknown, {resp.usage.tokens()}{eta}", flush=True)
            else:
                # Cents: a single batch is a fraction of a cent to a few
                # cents, which dollars would print as 0.00.
                model_cost[model_name] += cost
                print(
                    f"{prefix}  {cost * 100:.3f}c, {resp.usage.tokens()}{eta}",
                    flush=True,
                )
            g.warn_if_truncated(model_name, resp.finish_reason, max_tokens)
            raws[result.job.key] = resp.text
            if resp.text:
                # Saved as soon as it lands, so an interrupted run keeps what it
                # already paid for. Usage is written only alongside a kept
                # response, so the pair never disagrees about whether the call
                # needs paying for again.
                rpaths[result.job.key].write_text(resp.text)
                save_usage(usage_path(bid, model_name, phashes[bid]), resp.usage)

    if jobs:
        asyncio.run(consume())

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
            percentile_sets = parse_batch_percentiles(raw, labels)
            for q, percentiles in zip(questions, percentile_sets):
                responses[ResponseId(model_name, q["question_id"])] = Response(
                    actual=q["value"],
                    percentiles=percentiles,
                    response_text=raw,
                )

    # What this run paid, per model. Cached batches cost nothing, so a fully
    # cached model reports $0.00 rather than what it originally cost.
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
    # report the corpus carries and the preamble the prompt names it with can
    # never disagree about which variant this run is.
    snapshot_only = cfg.get_bool_or("snapshot_only_report", False)
    preamble_path = cfg.get_preamble_path()

    print("=" * 70)
    print("MICROPOLIS WORLD — single city eval")
    print("=" * 70)
    print(f"config: {cfg.path}")
    print(f"label:  {label}")
    if preamble_path is not None:
        print(f"preamble: {preamble_path}")
    if snapshot_only:
        print("report: snapshot only (no HISTORY table)")

    scenarios = get_single_city_base_scenarios(
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
        snapshot_only,
        cfg.get_provider_concurrency(),
        preamble_path,
    )
    print("Done gathering")

    save_dataset(corpus, responses, models, out_path)
    usable = sum(1 for r in responses.values() if r.percentiles is not None)
    print(f"\nWrote {len(corpus)} questions x {len(models)} models -> {out_path}")
    print(f"  {usable} of {len(responses)} forecasts usable")
    print("=" * 70)
    print("Next: scripts/analyze_single_city.py, scripts/plot_forecasts.py")


if __name__ == "__main__":
    main()
