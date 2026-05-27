#!/usr/bin/env python3
"""
Bin probability elicitation experiment.

Tests whether the anti-g overconfidence finding persists when models forecast
using binned probabilities (assign % to outcome bins summing to 100%) instead
of direct quantile elicitation.

Matches the baseline condition (no domain knowledge, no tutorial, disruptable
templates only) — changes only the elicitation format.

Usage:
    cd /Users/elsehow/Projects/civbench
    uv run python scripts/run_bin_condition.py --dry-run
    uv run python scripts/run_bin_condition.py
    uv run python scripts/run_bin_condition.py --models anthropic/claude-opus-4-6 --seeds seed0 seed1
"""

import argparse
import asyncio
import json
import re
import sys
import time
from functools import partial
from pathlib import Path

# Force unbuffered output
print = partial(print, flush=True)

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fbsim_core.evaluation.models import load_api_keys_from_gcp, LiteLLMModel
from freeciv_world.evaluation.bin_probability_prompt import (
    build_bin_probability_prompt,
    bin_labels_for_template,
    BIN_EDGES,
)

_keys_loaded = False
def ensure_api_keys():
    global _keys_loaded
    if not _keys_loaded:
        load_api_keys_from_gcp()
        _keys_loaded = True

# ── Config ──────────────────────────────────────────────────────────────────

TEST_SEEDS = ["seed0", "seed1", "seed4", "seed5", "seed9",
              "seed10", "seed13", "seed15", "seed16", "seed20"]

MODELS = [
    "openai/gpt-4.1-2025-04-14",
    "anthropic/claude-sonnet-4-5-20250929",
    "openai/gpt-5-2025-08-07",
    "google/gemini-3-pro-preview",
    "anthropic/claude-opus-4-6",
]

DISRUPTABLE_TEMPLATES = {
    "population_continuous", "territory_continuous",
    "treasury_continuous", "cities_count_continuous",
}

HORIZON_TURNS = [90, 120, 150, 180, 210, 240]
HORIZON_LABELS = {90: "H1", 120: "H2", 150: "H3", 180: "H4", 210: "H5", 240: "H6"}

TEMPLATE_DEFS = {
    "population_continuous": ("population", "What will {civ}'s population be at turn {turn}?"),
    "territory_continuous": ("territory_size", "How many tiles will {civ} control at turn {turn}?"),
    "treasury_continuous": ("treasury", "How much gold will {civ} have at turn {turn}?"),
    "cities_count_continuous": ("cities_count", "How many cities will {civ} have at turn {turn}?"),
}

DATA_DIR = Path("data/conditional/republic/baseline")
GAMES_DIR = Path("data/games")
RESULTS_DIR = Path("data/results/bin_condition")


# ── Data loading ────────────────────────────────────────────────────────────

def load_disruptable_questions(seed: str) -> list[dict]:
    """Load all-horizon disruptable continuous questions for a seed."""
    existing = {}
    qfile = DATA_DIR / seed / "questions.json"
    if qfile.exists():
        with open(qfile) as f:
            data = json.load(f)
        for q in data["questions"]:
            if (q["template_id"] in DISRUPTABLE_TEMPLATES
                    and q["question_type"] == "continuous"):
                key = (q["template_id"], q["resolution_turn"])
                existing[key] = q

    game_file = GAMES_DIR / f"{seed}_data.json"
    with open(game_file) as f:
        game = json.load(f)

    ts = game["time_series"]
    civ_name = game["civilizations"]["0"]["name"]

    questions = []
    for tmpl_id in sorted(DISRUPTABLE_TEMPLATES):
        ts_key, text_template = TEMPLATE_DEFS[tmpl_id]
        for turn in HORIZON_TURNS:
            key = (tmpl_id, turn)
            if key in existing:
                questions.append(existing[key])
            else:
                value = ts[ts_key][str(turn)]["0"]
                questions.append({
                    "question_id": f"{HORIZON_LABELS[turn]}_{tmpl_id}",
                    "template_id": tmpl_id,
                    "resolution_turn": turn,
                    "horizon": HORIZON_LABELS[turn],
                    "question_type": "continuous",
                    "question_text": text_template.format(civ=civ_name, turn=turn),
                    "resolution": {"value_at_resolution": value},
                })

    return questions


def load_world_report(seed: str) -> str:
    """Load the turn 060 world report for a seed."""
    report_path = DATA_DIR / seed / "world_report" / "turn_060_report.txt"
    with open(report_path) as f:
        return f.read()


# ── Parsing ─────────────────────────────────────────────────────────────────

def parse_bin_probabilities(response: str, num_questions: int,
                            questions: list[dict]) -> list[dict | None]:
    """Parse bin probability estimates from model response.

    Expects format inside <<<BINS>>>...<<<END>>> block:
        Q1: [0-2]=15%, [3-10]=20%, ...
        Q2: [0-2]=10%, [3-10]=25%, ...

    Returns:
        List of dicts with keys 'bins' (list of 10 floats) and 'labels'
        (list of 10 strings), or None for parse failures.
    """
    results = []
    match = re.search(r"<<<BINS>>>(.*?)<<<END>>>", response, re.DOTALL)
    if not match:
        return [None] * num_questions

    block = match.group(1).strip()
    lines = [l.strip() for l in block.split("\n") if l.strip()]

    for line in lines:
        # Skip lines that don't look like data (no = or % in them)
        if "=" not in line and "%" not in line:
            continue

        # Strip Q prefix
        line = re.sub(r"^Q\d+:\s*", "", line)

        # Extract all numeric values from [label]=value% patterns
        values = re.findall(r"\[.*?\]\s*=\s*([\d.]+)\s*%?", line)
        if not values:
            # Try alternative: just comma-separated numbers
            values = re.findall(r"(\d+(?:\.\d+)?)\s*%", line)

        if len(values) >= 8:  # Allow some tolerance
            bins = [float(v) for v in values[:10]]
            # Pad with 0 if fewer than 10
            while len(bins) < 10:
                bins.append(0.0)
            results.append({"bins": bins})
        else:
            results.append(None)

    while len(results) < num_questions:
        results.append(None)
    return results[:num_questions]


# ── Main ────────────────────────────────────────────────────────────────────

async def run_experiment(models: list[str], seeds: list[str] = None,
                          dry_run: bool = False):
    """Run the bin probability condition experiment."""
    ensure_api_keys()

    seeds = seeds or TEST_SEEDS

    # Pre-load all test world data
    test_data = {}
    for seed in seeds:
        report = load_world_report(seed)
        questions = load_disruptable_questions(seed)
        test_data[seed] = (report, questions)
        print(f"Loaded {seed}: {len(questions)} questions "
              f"({len(set(q['template_id'] for q in questions))} templates × "
              f"{len(set(q['resolution_turn'] for q in questions))} horizons)")

    all_results = []
    total_calls = len(models) * len(seeds)
    call_num = 0

    for model_id in models:
        model_short = model_id.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {model_id}")
        print(f"{'='*70}")

        model = LiteLLMModel(model_id)

        for seed in seeds:
            call_num += 1
            report, questions = test_data[seed]

            prompt = build_bin_probability_prompt(questions, report)

            print(f"\n  [{call_num}/{total_calls}] {model_short} × bin_probability × {seed}: "
                  f"{len(questions)} Qs, ~{len(prompt)//4} tokens")

            if dry_run:
                print(f"    [DRY RUN] Skipping API call")
                print(f"    Prompt length: {len(prompt)} chars")
                # Show first question's bin labels
                first_tmpl = questions[0]["template_id"]
                labels = bin_labels_for_template(first_tmpl)
                print(f"    Example bins ({first_tmpl}): {labels}")
                for q in questions[:4]:
                    truth = q["resolution"]["value_at_resolution"]
                    print(f"      {q['template_id']:<28} T{q['resolution_turn']:<4} truth={truth}")
                print(f"      ... ({len(questions)} total)")
                continue

            start = time.monotonic()
            try:
                response = await model.get_response_async(prompt, temperature=0.0)
                latency = (time.monotonic() - start) * 1000
                print(f"    Response in {latency:.0f}ms")

                parsed = parse_bin_probabilities(response, len(questions), questions)

                for q, bins_data in zip(questions, parsed):
                    truth = q["resolution"]["value_at_resolution"]
                    tmpl = q["template_id"]
                    edges = BIN_EDGES[tmpl]
                    labels = bin_labels_for_template(tmpl)

                    result = {
                        "condition": "bin_probability",
                        "model": model_id,
                        "seed": seed,
                        "template_id": tmpl,
                        "horizon": q.get("horizon", HORIZON_LABELS.get(q["resolution_turn"], "?")),
                        "resolution_turn": q["resolution_turn"],
                        "question_text": q["question_text"],
                        "ground_truth": truth,
                        "bin_edges": edges,
                        "bin_labels": labels,
                        "bin_probabilities": bins_data["bins"] if bins_data else None,
                    }
                    all_results.append(result)

                    if bins_data:
                        bins = bins_data["bins"]
                        total = sum(bins)
                        # Find which bin the truth falls in
                        truth_bin = len(edges) - 2  # default to last bin
                        for i in range(len(edges) - 1):
                            if truth <= edges[i + 1]:
                                truth_bin = i
                                break
                        truth_prob = bins[truth_bin] if truth_bin < len(bins) else 0
                        print(f"      {tmpl:<28} {q.get('horizon', '?'):<4} "
                              f"truth={truth:<8} bin=[{labels[truth_bin]}] "
                              f"p={truth_prob:.0f}% sum={total:.0f}%")
                    else:
                        print(f"      {tmpl:<28} {q.get('horizon', '?'):<4} PARSE FAILED")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Bin probability elicitation experiment")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show prompts without API calls")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Models to test")
    parser.add_argument("--seeds", nargs="+", default=TEST_SEEDS,
                        help="Test seeds to evaluate")
    args = parser.parse_args()

    results = asyncio.run(run_experiment(args.models, seeds=args.seeds,
                                         dry_run=args.dry_run))

    if results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = RESULTS_DIR / "bin_results.json"

        # Load existing results if any, and merge
        existing = []
        if out_file.exists():
            with open(out_file) as f:
                existing = json.load(f)
            print(f"\nLoaded {len(existing)} existing results")

        # Remove duplicates (same model, seed, template, horizon)
        new_keys = set()
        for r in results:
            new_keys.add((r["model"], r["seed"],
                          r["template_id"], r["resolution_turn"]))

        merged = [r for r in existing
                  if (r["model"], r["seed"],
                      r["template_id"], r["resolution_turn"]) not in new_keys]
        merged.extend(results)

        with open(out_file, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        print(f"\nResults saved to {out_file} ({len(merged)} total results)")

        # Print summary
        from collections import Counter
        counts = Counter(r["model"].split("/")[-1] for r in merged)
        parsed_ok = sum(1 for r in merged if r["bin_probabilities"] is not None)
        print(f"\nParse rate: {parsed_ok}/{len(merged)} ({parsed_ok/len(merged)*100:.0f}%)")
        print("\nResult counts:")
        for model, count in sorted(counts.items()):
            print(f"  {model:<35} {count} results")


if __name__ == "__main__":
    main()
