#!/usr/bin/env python3
"""
Domain knowledge prompting experiment.

Tests whether explaining FreeCiv mechanics (without crash rates or forecasting
guidance) improves continuous forecasting calibration.

Conditions:
  1. baseline: Current prompt (no domain explanation)
  2. domain_knowledge: Adds FreeCiv domain knowledge prefix

Uses the same 5 models, 10 test worlds, 4 disruptable templates as other interventions.
Question format is interleaved (baseline style) — trajectory format is tested separately.

Usage:
    cd /Users/elsehow/Projects/civbench
    uv run python scripts/run_domain_knowledge_condition.py --dry-run
    uv run python scripts/run_domain_knowledge_condition.py
    uv run python scripts/run_domain_knowledge_condition.py --conditions domain_knowledge
    uv run python scripts/run_domain_knowledge_condition.py --models openai/gpt-4.1-2025-04-14 --seeds seed0
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

from civrealm.evaluation.models import load_api_keys_from_gcp, LiteLLMModel
from civrealm.evaluation.domain_knowledge_prompt import build_domain_knowledge_continuous_prompt

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

ALL_CONDITIONS = ["baseline", "domain_knowledge"]

DATA_DIR = Path("data/conditional/republic/baseline")
GAMES_DIR = Path("data/games")
RESULTS_DIR = Path("data/results/domain_knowledge_condition")

DISRUPTABLE_TEMPLATES = {
    "population_continuous", "territory_continuous",
    "treasury_continuous", "cities_count_continuous",
}

HORIZON_TURNS = [90, 120, 150, 180, 210, 240]

TEMPLATE_DEFS = {
    "population_continuous": ("population", "What will {civ}'s population be at turn {turn}?"),
    "territory_continuous": ("territory_size", "How many tiles will {civ} control at turn {turn}?"),
    "treasury_continuous": ("treasury", "How much gold will {civ} have at turn {turn}?"),
    "cities_count_continuous": ("cities_count", "How many cities will {civ} have at turn {turn}?"),
}

HORIZON_LABELS = {90: "H1", 120: "H2", 150: "H3", 180: "H4", 210: "H5", 240: "H6"}


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


# ── Prompt builders ────────────────────────────────────────────────────────

def build_baseline_prompt(questions: list[dict], world_report: str) -> str:
    """Build the standard interleaved prompt (control condition)."""
    num_questions = len(questions)
    questions_text = "\n".join(
        f"{i+1}. {q['question_text']}" for i, q in enumerate(questions)
    )

    instruction = f"""You may analyze the data, but you MUST end your response with percentile estimates in this exact format:
<<<PERCENTILES>>>
Q1: p10=5, p25=10, p50=15, p75=20, p90=25
Q2: p10=100, p25=200, p50=300, p75=400, p90=500
<<<END>>>

For each question, provide one line with percentile estimates for all {num_questions} questions, in order.
- p10 means you estimate there's a 10% chance the true value is below this number
- p25 means you estimate there's a 25% chance the true value is below this number
- p50 (median) means you estimate there's a 50% chance the true value is below this number
- p75 means you estimate there's a 75% chance the true value is below this number
- p90 means you estimate there's a 90% chance the true value is below this number"""

    return f"""You are an expert superforecaster, familiar with the work of Tetlock and others. You are analyzing a FreeCiv game simulation. Make predictions based on the world report below.

You MUST provide percentile estimates for each question UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, provide reasonable mid-range estimates, but always return numeric percentile values.

## World Report
{world_report}

## Questions
{questions_text}

{instruction}"""


# ── Parsing ─────────────────────────────────────────────────────────────────

def parse_percentiles(response: str, num_questions: int) -> list[dict | None]:
    """Parse percentile estimates from model response."""
    results = []
    match = re.search(r"<<<PERCENTILES>>>(.*?)<<<END>>>", response, re.DOTALL)
    if not match:
        return [None] * num_questions

    block = match.group(1).strip()
    lines = [l.strip() for l in block.split("\n") if l.strip()]

    for line in lines:
        if "p10" not in line.lower():
            continue

        line = re.sub(r"^Q\d+:\s*", "", line)
        percentile_dict = {}
        for key in ["p10", "p25", "p50", "p75", "p90"]:
            m = re.search(rf"{key}\s*=\s*([-\d.]+)", line)
            if m:
                percentile_dict[key] = float(m.group(1))
        if len(percentile_dict) >= 3:
            results.append(percentile_dict)
        else:
            results.append(None)

    while len(results) < num_questions:
        results.append(None)
    return results[:num_questions]


# ── Main ────────────────────────────────────────────────────────────────────

async def run_experiment(models: list[str], conditions: list[str],
                          seeds: list[str] = None,
                          dry_run: bool = False):
    """Run the domain knowledge condition experiment."""
    ensure_api_keys()

    seeds = seeds or TEST_SEEDS

    prompt_builders = {
        "baseline": build_baseline_prompt,
        "domain_knowledge": build_domain_knowledge_continuous_prompt,
    }

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
    total_calls = len(models) * len(conditions) * len(seeds)
    call_num = 0

    for model_id in models:
        model_short = model_id.split("/")[-1]
        print(f"\n{'='*70}")
        print(f"MODEL: {model_id}")
        print(f"{'='*70}")

        model = LiteLLMModel(model_id)

        for condition in conditions:
            build_prompt = prompt_builders[condition]
            print(f"\n--- Condition: {condition} ---")

            for seed in seeds:
                call_num += 1
                report, questions = test_data[seed]

                prompt = build_prompt(questions, report)

                print(f"\n  [{call_num}/{total_calls}] {model_short} × {condition} × {seed}: "
                      f"{len(questions)} Qs, ~{len(prompt)//4} tokens")

                if dry_run:
                    print(f"    [DRY RUN] Skipping API call")
                    for q in questions:
                        truth = q["resolution"]["value_at_resolution"]
                        print(f"      {q['template_id']:<28} T{q['resolution_turn']:<4} truth={truth}")
                    continue

                start = time.monotonic()
                try:
                    response = await model.get_response_async(prompt, temperature=0.0)
                    latency = (time.monotonic() - start) * 1000
                    print(f"    Response in {latency:.0f}ms")

                    percentiles = parse_percentiles(response, len(questions))

                    for q, pcts in zip(questions, percentiles):
                        truth = q["resolution"]["value_at_resolution"]

                        result = {
                            "condition": condition,
                            "model": model_id,
                            "seed": seed,
                            "template_id": q["template_id"],
                            "horizon": q.get("horizon", HORIZON_LABELS.get(q["resolution_turn"], "?")),
                            "resolution_turn": q["resolution_turn"],
                            "question_text": q["question_text"],
                            "ground_truth": truth,
                            "percentiles": pcts,
                        }
                        all_results.append(result)

                        if pcts:
                            p50 = pcts.get("p50", "?")
                            p10 = pcts.get("p10", "?")
                            p90 = pcts.get("p90", "?")
                            overshoot = f"{p50/truth:.1f}x" if truth and isinstance(p50, (int, float)) else "?"
                            print(f"      {q['template_id']:<28} {q.get('horizon', '?'):<4} "
                                  f"truth={truth:<8} p10={p10:<8} p50={p50:<8}({overshoot}) p90={p90:<8}")
                        else:
                            print(f"      {q['template_id']:<28} {q.get('horizon', '?'):<4} PARSE FAILED")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Domain knowledge prompting experiment")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show prompts without API calls")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Models to test")
    parser.add_argument("--conditions", nargs="+", default=ALL_CONDITIONS,
                        help="Conditions to run (baseline, domain_knowledge)")
    parser.add_argument("--seeds", nargs="+", default=TEST_SEEDS,
                        help="Test seeds to evaluate")
    args = parser.parse_args()

    for c in args.conditions:
        if c not in ALL_CONDITIONS:
            print(f"ERROR: Unknown condition '{c}'. Choose from: {ALL_CONDITIONS}")
            sys.exit(1)

    results = asyncio.run(run_experiment(args.models, args.conditions,
                                         seeds=args.seeds, dry_run=args.dry_run))

    if results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = RESULTS_DIR / "domain_knowledge_results.json"

        existing = []
        if out_file.exists():
            with open(out_file) as f:
                existing = json.load(f)
            print(f"\nLoaded {len(existing)} existing results")

        new_keys = set()
        for r in results:
            new_keys.add((r["model"], r["condition"], r["seed"],
                          r["template_id"], r["resolution_turn"]))

        merged = [r for r in existing
                  if (r["model"], r["condition"], r["seed"],
                      r["template_id"], r["resolution_turn"]) not in new_keys]
        merged.extend(results)

        with open(out_file, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        print(f"\nResults saved to {out_file} ({len(merged)} total results)")

        from collections import Counter
        counts = Counter((r["model"].split("/")[-1], r["condition"]) for r in merged)
        print("\nResult counts:")
        for (model, cond), count in sorted(counts.items()):
            print(f"  {model:<35} {cond:<20} {count} results")


if __name__ == "__main__":
    main()
