#!/usr/bin/env python3
"""
Three-condition tutorial experiment: crash-only, growth-only, both tutorials.

Tests whether the TYPE of tutorial world matters for LLM tail risk performance.

Conditions:
  1. crash_tutorial: Resolved seed7 (crash) as tutorial → each test seed independently
  2. growth_tutorial: Resolved seed3 (growth) as tutorial → each test seed independently
  3. both_tutorials: Resolved seed7 + seed3 as tutorials → each test seed independently

Test seeds (10): seed0 seed1 seed4 seed8 seed9 seed10 seed11 seed16 seed18 seed20
Tutorial seeds: seed7 (crash), seed3 (growth)

Also runs baseline (no tutorial) for any model missing baseline data.

Usage:
    cd /Users/elsehow/Projects/civbench
    uv run python scripts/run_tutorial_conditions.py --dry-run
    uv run python scripts/run_tutorial_conditions.py
    uv run python scripts/run_tutorial_conditions.py --conditions crash_tutorial growth_tutorial
    uv run python scripts/run_tutorial_conditions.py --models anthropic/claude-opus-4-6 --conditions baseline
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

_keys_loaded = False
def ensure_api_keys():
    global _keys_loaded
    if not _keys_loaded:
        load_api_keys_from_gcp()
        _keys_loaded = True

# ── Config ──────────────────────────────────────────────────────────────────

TUTORIAL_CRASH_SEED = "seed7"   # Monegasque: peak 83 → T240 15, -81.9%
TUTORIAL_GROWTH_SEED = "seed3"  # Jolof: peak 84 → T240 149, +77.4%

TEST_SEEDS = ["seed0", "seed1", "seed4", "seed8", "seed9",
              "seed10", "seed11", "seed16", "seed18", "seed20"]

# Seeds with existing H6 questions in questions.json
SEEDS_WITH_H6 = {"seed0", "seed1", "seed4", "seed8", "seed9", "seed10",
                  "seed3", "seed7"}

MODELS = [
    "openai/gpt-4.1-2025-04-14",
    "anthropic/claude-sonnet-4-5-20250929",
    "openai/gpt-5-2025-08-07",
    "google/gemini-3-pro-preview",
    "anthropic/claude-opus-4-6",
]

ALL_CONDITIONS = ["crash_tutorial", "growth_tutorial", "both_tutorials", "baseline"]

DATA_DIR = Path("data/conditional/republic/baseline")
GAMES_DIR = Path("data/games")
RESULTS_DIR = Path("data/results/tutorial_conditions")

CONTINUOUS_TEMPLATES = {
    "population_continuous", "territory_continuous", "treasury_continuous",
    "techs_continuous", "scores_continuous", "cities_count_continuous",
}


# ── Data loading ────────────────────────────────────────────────────────────

def load_h6_continuous_questions(seed: str) -> list[dict]:
    """Load H6 continuous questions for a seed.

    For seeds with existing H6 questions, load from questions.json.
    For seeds without (11, 16, 18, 20), generate from game data.
    """
    if seed in SEEDS_WITH_H6:
        qfile = DATA_DIR / seed / "questions.json"
        with open(qfile) as f:
            data = json.load(f)
        questions = []
        for q in data["questions"]:
            if (q["template_id"] in CONTINUOUS_TEMPLATES
                    and q["question_type"] == "continuous"
                    and "turn 240" in q["question_text"]):
                questions.append(q)
        return questions
    else:
        return _generate_h6_questions(seed)


def _generate_h6_questions(seed: str) -> list[dict]:
    """Generate H6 continuous questions from game data for seeds missing them."""
    game_file = GAMES_DIR / f"{seed}_data.json"
    with open(game_file) as f:
        d = json.load(f)

    ts = d["time_series"]
    civ_name = d["civilizations"]["0"]["name"]

    templates = {
        "techs_continuous": {
            "text": f"How many technologies will {civ_name} have discovered by turn 240?",
            "value": ts["techs_known"]["240"]["0"],
        },
        "treasury_continuous": {
            "text": f"How much gold will {civ_name} have at turn 240?",
            "value": ts["treasury"]["240"]["0"],
        },
        "population_continuous": {
            "text": f"What will {civ_name}'s population be at turn 240?",
            "value": ts["population"]["240"]["0"],
        },
        "cities_count_continuous": {
            "text": f"How many cities will {civ_name} have at turn 240?",
            "value": ts["cities_count"]["240"]["0"],
        },
        "territory_continuous": {
            "text": f"How many tiles will {civ_name} control at turn 240?",
            "value": ts["territory_size"]["240"]["0"],
        },
        "scores_continuous": {
            "text": f"What will {civ_name}'s score be at turn 240?",
            "value": ts["scores"]["240"]["0"],
        },
    }

    questions = []
    for tmpl_id, info in templates.items():
        questions.append({
            "question_id": f"h6_{tmpl_id}",
            "template_id": tmpl_id,
            "question_text": info["text"],
            "question_type": "continuous",
            "resolution": {"value_at_resolution": info["value"]},
        })
    return questions


def load_world_report(seed: str) -> str:
    """Load the turn 060 world report for a seed."""
    report_path = DATA_DIR / seed / "world_report" / "turn_060_report.txt"
    with open(report_path) as f:
        return f.read()


# ── Tutorial context builders ──────────────────────────────────────────────

def build_tutorial_context(seed: str, label: str = "Example World") -> str:
    """Build tutorial section showing a resolved world with outcomes."""
    questions = load_h6_continuous_questions(seed)
    report = load_world_report(seed)

    outcomes = "\n".join(
        f"- {q['question_text']} → Answer: {q['resolution']['value_at_resolution']}"
        for q in questions
    )

    return f"""## {label} (for orientation)

Below is an example FreeCiv world report with its actual outcomes at turn 240.
Study this to understand the report format and the kinds of dynamics that can occur.

### World Report
{report}

### Actual Outcomes (Turn 240)
{outcomes}

---

"""


def build_crash_tutorial_prefix() -> str:
    """Build tutorial prefix with crash world only."""
    ctx = build_tutorial_context(TUTORIAL_CRASH_SEED, "Example World (Crash Scenario)")
    return ctx + """Now, forecast the following world. Note that each world is independent — outcomes
from the example world above do not predict outcomes in the world below.

"""


def build_growth_tutorial_prefix() -> str:
    """Build tutorial prefix with growth world only."""
    ctx = build_tutorial_context(TUTORIAL_GROWTH_SEED, "Example World (Growth Scenario)")
    return ctx + """Now, forecast the following world. Note that each world is independent — outcomes
from the example world above do not predict outcomes in the world below.

"""


def build_both_tutorials_prefix() -> str:
    """Build tutorial prefix with both crash and growth worlds."""
    crash_ctx = build_tutorial_context(TUTORIAL_CRASH_SEED, "Example World 1 (Crash Scenario)")
    growth_ctx = build_tutorial_context(TUTORIAL_GROWTH_SEED, "Example World 2 (Growth Scenario)")
    return crash_ctx + growth_ctx + """Now, forecast the following world. Note that each world is independent — outcomes
from the example worlds above do not predict outcomes in the world below. Both crash
and growth outcomes are possible.

"""


# ── Prompt builder ─────────────────────────────────────────────────────────

def build_continuous_prompt(questions: list[dict], world_report: str,
                            prefix: str = "") -> str:
    """Build continuous forecast prompt with optional prefix context."""
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

{prefix}## World Report
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
    """Run the tutorial conditions experiment."""
    ensure_api_keys()

    seeds = seeds or TEST_SEEDS

    # Build tutorial prefixes (only for conditions we're running)
    prefix_builders = {
        "crash_tutorial": build_crash_tutorial_prefix,
        "growth_tutorial": build_growth_tutorial_prefix,
        "both_tutorials": build_both_tutorials_prefix,
        "baseline": lambda: "",  # No prefix for baseline
    }

    # Pre-load all test world data
    test_data = {}
    for seed in seeds:
        report = load_world_report(seed)
        questions = load_h6_continuous_questions(seed)
        test_data[seed] = (report, questions)
        print(f"Loaded {seed}: {len(questions)} H6 questions")

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
            prefix = prefix_builders[condition]()
            print(f"\n--- Condition: {condition} ---")
            print(f"    Prefix length: {len(prefix)} chars (~{len(prefix)//4} tokens)")

            for seed in seeds:
                call_num += 1
                report, questions = test_data[seed]

                prompt = build_continuous_prompt(
                    questions, report, prefix=prefix
                )

                print(f"\n  [{call_num}/{total_calls}] {model_short} × {condition} × {seed}: "
                      f"{len(questions)} Qs, ~{len(prompt)//4} tokens")

                if dry_run:
                    print(f"    [DRY RUN] Skipping API call")
                    for q in questions:
                        truth = q["resolution"]["value_at_resolution"]
                        print(f"      {q['template_id']:<28} truth={truth}")
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
                            print(f"      {q['template_id']:<28} truth={truth:<8} "
                                  f"p10={p10:<8} p50={p50:<8}({overshoot}) p90={p90:<8}")
                        else:
                            print(f"      {q['template_id']:<28} PARSE FAILED")

                except Exception as e:
                    print(f"    ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Three-condition tutorial experiment")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show prompts without API calls")
    parser.add_argument("--models", nargs="+", default=MODELS,
                        help="Models to test")
    parser.add_argument("--conditions", nargs="+", default=["crash_tutorial", "growth_tutorial", "both_tutorials"],
                        help="Conditions to run (crash_tutorial, growth_tutorial, both_tutorials, baseline)")
    parser.add_argument("--seeds", nargs="+", default=TEST_SEEDS,
                        help="Test seeds to evaluate")
    args = parser.parse_args()

    # Validate conditions
    for c in args.conditions:
        if c not in ALL_CONDITIONS:
            print(f"ERROR: Unknown condition '{c}'. Choose from: {ALL_CONDITIONS}")
            sys.exit(1)

    results = asyncio.run(run_experiment(args.models, args.conditions,
                                         seeds=args.seeds, dry_run=args.dry_run))

    if results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = RESULTS_DIR / "tutorial_conditions_results.json"

        # Load existing results if any, and merge
        existing = []
        if out_file.exists():
            with open(out_file) as f:
                existing = json.load(f)
            print(f"\nLoaded {len(existing)} existing results")

        # Remove duplicates (same model, condition, seed, template)
        existing_keys = set()
        for r in results:
            existing_keys.add((r["model"], r["condition"], r["seed"], r["template_id"]))

        merged = [r for r in existing
                  if (r["model"], r["condition"], r["seed"], r["template_id"]) not in existing_keys]
        merged.extend(results)

        with open(out_file, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        print(f"\nResults saved to {out_file} ({len(merged)} total results)")

        # Print summary counts
        from collections import Counter
        counts = Counter((r["model"].split("/")[-1], r["condition"]) for r in merged)
        print("\nResult counts:")
        for (model, cond), count in sorted(counts.items()):
            print(f"  {model:<35} {cond:<20} {count} results")


if __name__ == "__main__":
    main()
