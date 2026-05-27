#!/usr/bin/env python3
"""
Test: Does domain knowledge about FreeCiv mechanics fix tail risk blindness?

Hypothesis: Models overshoot population at long horizons because they don't know
that wars/conquests cause population crashes in FreeCiv. If we tell them about
these mechanics, their p10 should widen to contain the actual crash values.

Design:
- Take the worst population crash cases (seed7/T240 GT=15, seed1/T240 GT=33)
- Add FreeCiv domain knowledge to the continuous prompt
- Compare p10 calibration: augmented vs original
- Run on 3 models: GPT-4.1 (low ECI), GPT-5 (high ECI), Gemini 3 Pro (highest ECI)
- Also run control questions (non-crash populations) to check we don't over-correct

Usage:
    cd /Users/elsehow/Projects/civbench
    uv run python scripts/test_domain_knowledge_tail_risk.py
    uv run python scripts/test_domain_knowledge_tail_risk.py --dry-run  # print prompts only
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from functools import partial
from pathlib import Path

# Force unbuffered output for background execution
print = partial(print, flush=True)

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from fbsim_core.evaluation.models import LiteLLMModel, load_api_keys_from_gcp

# Only load GCP keys when actually querying models (not for dry run)
_keys_loaded = False
def ensure_api_keys():
    global _keys_loaded
    if not _keys_loaded:
        load_api_keys_from_gcp()
        _keys_loaded = True

# ── Config ──────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent.parent / "data" / "questions"
RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"

# Models to test (low ECI → high ECI)
MODELS = [
    "anthropic/claude-sonnet-4-5-20250929",
    "anthropic/claude-opus-4-6",
]

# Test questions: population crashes at H5-H6
CRASH_QUESTIONS = [
    # seed7 Monegasque T240 — worst case (GT=15, 43x overshoot)
    {"game_id": "seed7", "question_id": "cond_q0242",
     "question_text": "What will Monegasque's population be at turn 240?",
     "ground_truth": 15.0},
    # seed7 Monegasque T210
    {"game_id": "seed7", "question_id": "cond_q0201",
     "question_text": "What will Monegasque's population be at turn 210?",
     "ground_truth": 18.0},
    # seed1 Dacian T240 — the example from the writeup (GT=33)
    {"game_id": "seed1", "question_id": "cond_q0236",
     "question_text": "What will Dacian's population be at turn 240?",
     "ground_truth": 33.0},
    # seed1 Dacian T210
    {"game_id": "seed1", "question_id": "cond_q0196",
     "question_text": "What will Dacian's population be at turn 210?",
     "ground_truth": 48.0},
    # seed6 Californian T240
    {"game_id": "seed6", "question_id": "cond_q0248",
     "question_text": "What will Californian's population be at turn 240?",
     "ground_truth": 115.0},
]

# Control questions: population at near horizons (where models are calibrated)
CONTROL_QUESTIONS = [
    # seed1 Dacian T90 — same civ, near horizon (should be fine)
    {"game_id": "seed1", "question_id": "cond_q0036",
     "question_text": "What will Dacian's population be at turn 90?",
     "ground_truth": 42.0},
    # seed0 Pontic T90
    {"game_id": "seed0", "question_id": "cond_q0038",
     "question_text": "What will Pontic's population be at turn 90?",
     "ground_truth": 79.0},
    # seed1 Dacian T120
    {"game_id": "seed1", "question_id": "cond_q0076",
     "question_text": "What will Dacian's population be at turn 120?",
     "ground_truth": 66.0},
]

# ── Domain knowledge supplement ─────────────────────────────────────────────

DOMAIN_KNOWLEDGE = """## FreeCiv Game Mechanics: Key Dynamics for Forecasting

**Population dynamics:** Population in FreeCiv reflects the total citizen count across all cities. It can grow through city expansion, food surplus, and founding new cities — but it can also DECLINE sharply. Common causes of population decline:
- **Warfare and conquest:** When a city is conquered, it typically loses 1-2 citizens. If a civilization loses multiple cities in a war, population can drop by 50-80% in a few turns.
- **City destruction:** Cities can be destroyed entirely (razed), eliminating all their population.
- **Famine:** Cities with negative food surplus lose population each turn.
- **Civil disorder:** Prolonged unhappiness can cause cities to stagnate or shrink.

**Historical patterns in this simulation format:** In games run to turn 240, roughly 30-50% of civilizations that are growing at turn 60 experience a major population decline (>40% drop from peak) by turn 240 due to warfare. Civilizations with smaller militaries relative to neighbors are especially vulnerable. Population at turn 60 is NOT a reliable floor for future population — it can go much lower.

**Territorial dynamics:** Territory follows similar patterns — civilizations can lose significant territory through conquest. A civ controlling 100+ tiles at turn 60 might control <30 by turn 240 if it loses a war.

**Key forecasting implication:** When predicting population/territory at distant future turns (180-240), your pessimistic scenarios (p10) should account for the possibility of major military defeat. A p10 estimate that assumes continued growth from current levels will systematically miss crash scenarios."""


def build_augmented_continuous_prompt(questions, world_report, with_domain_knowledge=True):
    """Build continuous prompt with optional domain knowledge."""
    num_questions = len(questions)
    questions_text = "\n".join(
        f"{i+1}. {q['question_text']}"
        for i, q in enumerate(questions)
    )

    if num_questions == 1:
        question_section = "## Question"
        instruction = """You may analyze the data, but you MUST end your response with your percentile estimates in this exact format:
<<<PERCENTILES>>>
p10=5, p25=10, p50=15, p75=20, p90=25
<<<END>>>

Replace the example values with your actual percentile estimates.
- p10 means you estimate there's a 10% chance the true value is below this number
- p25 means you estimate there's a 25% chance the true value is below this number
- p50 (median) means you estimate there's a 50% chance the true value is below this number
- p75 means you estimate there's a 75% chance the true value is below this number
- p90 means you estimate there's a 90% chance the true value is below this number"""
    else:
        question_section = "## Questions"
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

    domain_section = f"\n{DOMAIN_KNOWLEDGE}\n" if with_domain_knowledge else ""

    return f"""You are an expert superforecaster, familiar with the work of Tetlock and others. You are analyzing a FreeCiv game simulation. Make predictions based on the world report below.

You MUST provide percentile estimates for each question UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, provide reasonable mid-range estimates, but always return numeric percentile values.
{domain_section}
## World Report
{world_report}

{question_section}
{questions_text}

{instruction}"""


def load_world_report(game_id):
    """Load world report for a game."""
    txt_path = DATA_DIR / game_id / "world_report" / "turn_060_report.txt"
    if txt_path.exists():
        return txt_path.read_text()
    return ""


def parse_percentiles(response_text):
    """Parse percentile estimates from model response."""
    import re
    results = []

    # Find the PERCENTILES block
    match = re.search(r'<<<PERCENTILES>>>(.*?)<<<END>>>', response_text, re.DOTALL)
    if not match:
        return []

    block = match.group(1).strip()
    for line in block.strip().split('\n'):
        line = line.strip()
        if not line:
            continue
        # Remove Q1:, Q2: prefix if present
        line = re.sub(r'^Q\d+:\s*', '', line)
        percentiles = {}
        for m in re.finditer(r'p(\d+)\s*=\s*([\d.]+)', line):
            percentiles[f'p{m.group(1)}'] = float(m.group(2))
        if percentiles:
            results.append(percentiles)

    return results


async def query_model(model, prompt):
    """Query a single model."""
    ensure_api_keys()
    m = LiteLLMModel(id=model)
    response = await m.get_response_async(prompt, temperature=0.0)
    return response


async def run_experiment(dry_run=False):
    """Run the domain knowledge experiment."""
    all_questions = CRASH_QUESTIONS + CONTROL_QUESTIONS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Group questions by game_id
    by_game = {}
    for q in all_questions:
        by_game.setdefault(q['game_id'], []).append(q)

    results = []

    for game_id, questions in by_game.items():
        world_report = load_world_report(game_id)
        if not world_report:
            print(f"WARNING: No world report for {game_id}")
            continue

        # Build both prompts
        prompt_augmented = build_augmented_continuous_prompt(questions, world_report, with_domain_knowledge=True)
        prompt_baseline = build_augmented_continuous_prompt(questions, world_report, with_domain_knowledge=False)

        if dry_run:
            print(f"\n{'='*80}")
            print(f"GAME: {game_id} ({len(questions)} questions)")
            print(f"{'='*80}")
            print(f"\n--- AUGMENTED PROMPT (first 500 chars of domain knowledge section) ---")
            # Show just the domain knowledge part
            dk_start = prompt_augmented.find("## FreeCiv Game Mechanics")
            dk_end = prompt_augmented.find("## World Report")
            print(prompt_augmented[dk_start:dk_end][:500])
            print(f"\n--- QUESTIONS ---")
            for q in questions:
                print(f"  {q['question_id']}: {q['question_text']} (GT={q['ground_truth']})")
            continue

        # Query each model with both conditions
        for model_id in MODELS:
            model_short = model_id.split("/")[-1]
            print(f"\n{'='*60}")
            print(f"{model_short} × {game_id} ({len(questions)} Qs)")
            print(f"{'='*60}")

            for condition, prompt in [("baseline", prompt_baseline), ("augmented", prompt_augmented)]:
                print(f"\n  [{condition}] Querying {model_short}...")
                try:
                    response = await query_model(model_id, prompt)
                    percentiles_list = parse_percentiles(response)

                    if len(percentiles_list) != len(questions):
                        print(f"  WARNING: Expected {len(questions)} answers, got {len(percentiles_list)}")
                        print(f"  Response excerpt: {response[-300:]}")

                    for i, q in enumerate(questions):
                        p = percentiles_list[i] if i < len(percentiles_list) else {}
                        gt = q['ground_truth']
                        is_crash = q in CRASH_QUESTIONS

                        result = {
                            "model": model_short,
                            "game_id": game_id,
                            "question_id": q['question_id'],
                            "question_text": q['question_text'],
                            "ground_truth": gt,
                            "condition": condition,
                            "is_crash": is_crash,
                            "percentiles": p,
                            "p10_contains_truth": p.get('p10', float('inf')) >= gt if not is_crash else p.get('p10', float('inf')) <= gt,
                            "overshoot": p.get('p50', 0) / gt if gt > 0 else None,
                        }
                        results.append(result)

                        tag = "CRASH" if is_crash else "CTRL"
                        p10 = p.get('p10', '?')
                        p50 = p.get('p50', '?')
                        p90 = p.get('p90', '?')
                        miss = "MISS" if is_crash and isinstance(p10, (int, float)) and p10 > gt else ("HIT" if is_crash else "")
                        print(f"  [{tag}] {q['question_id']}: GT={gt:>6.0f} | p10={str(p10):>6} p50={str(p50):>6} p90={str(p90):>6} {miss}")

                except Exception as e:
                    print(f"  ERROR: {e}")

    if dry_run:
        return

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Compare p10 miss rates: baseline vs augmented
    for model_id in MODELS:
        model_short = model_id.split("/")[-1]
        print(f"\n--- {model_short} ---")

        for condition in ["baseline", "augmented"]:
            crash_results = [r for r in results if r['model'] == model_short
                           and r['condition'] == condition and r['is_crash']]
            if not crash_results:
                continue

            miss_count = sum(1 for r in crash_results
                           if r['percentiles'].get('p10', float('inf')) > r['ground_truth'])
            total = len(crash_results)
            avg_overshoot = sum(r['overshoot'] for r in crash_results if r['overshoot']) / total

            print(f"  {condition:>10}: p10 miss rate = {miss_count}/{total} ({miss_count/total*100:.0f}%), "
                  f"avg p50 overshoot = {avg_overshoot:.1f}x")

    # ── Save results ─────────────────────────────────────────────────────
    out_path = RESULTS_DIR / f"domain_knowledge_test_{timestamp}.json"
    with open(out_path, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "models": MODELS,
            "domain_knowledge": DOMAIN_KNOWLEDGE,
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without querying models")
    args = parser.parse_args()

    asyncio.run(run_experiment(dry_run=args.dry_run))
