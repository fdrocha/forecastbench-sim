#!/usr/bin/env python3
"""
Standardized domain knowledge intervention experiment.

Runs the domain knowledge prompt (FreeCiv warfare mechanics + crash base rates)
on the same standardized 10-seed set used by other intervention experiments.

Templates: population_continuous, territory_continuous (baseline condition)
           conditional_population_continuous, conditional_territory_continuous (conditional condition)

Usage:
    uv run python scripts/run_domain_knowledge_standardized.py --model "openai/gpt-4.1-2025-04-14" --seeds seed0 seed1 seed4
    uv run python scripts/run_domain_knowledge_standardized.py --dry-run --model "openai/gpt-4.1-2025-04-14" --seeds seed0
"""

import argparse
import asyncio
import json
import re
import sys
import time
from functools import partial
from pathlib import Path

# Force unbuffered output for background execution
print = partial(print, flush=True)

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from civrealm.evaluation.models import load_api_keys_from_gcp, LiteLLMModel

load_api_keys_from_gcp()

# ── Config ──────────────────────────────────────────────────────────────────

SEEDS = ["seed0", "seed1", "seed4", "seed8", "seed9", "seed10", "seed11", "seed16", "seed18", "seed20"]
MODEL_ID = "anthropic/claude-sonnet-4-5-20250929"

TEMPLATES_BASELINE = {"population_continuous", "territory_continuous"}
TEMPLATES_CONDITIONAL = {"conditional_population_continuous", "conditional_territory_continuous"}
CONDITIONS = ["baseline", "conditional"]

OUTPUT_DIR = Path("data/results/domain_knowledge_standardized")

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


def load_ground_truth_map() -> dict:
    """Load ground truth from existing results files (questions.json has stale values)."""
    gt_map = {}  # (question_text, condition) -> ground_truth
    for path, condition in [
        (Path("data/results/republic_baseline_continuous_all.json"), "baseline"),
        (Path("data/results/republic_conditional_continuous_all.json"), "conditional"),
    ]:
        if not path.exists():
            continue
        with open(path) as f:
            content = f.read().replace('NaN', 'null').replace('Infinity', 'null')
            data = json.loads(content)
        for q in data['questions']:
            gt_map[(q['question_text'], condition)] = q['ground_truth']
    return gt_map


# ── Prompt builder ──────────────────────────────────────────────────────────

def build_continuous_prompt(questions: list[dict], world_report: str, treatment: bool = False) -> str:
    """Build continuous prompt with optional domain knowledge supplement."""
    num_questions = len(questions)
    questions_text = "\n".join(
        f"{i+1}. {q['question_text']}" for i, q in enumerate(questions)
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

    domain_section = f"\n{DOMAIN_KNOWLEDGE}\n" if treatment else ""

    return f"""You are an expert superforecaster, familiar with the work of Tetlock and others. You are analyzing a FreeCiv game simulation. Make predictions based on the world report below.

You MUST provide percentile estimates for each question UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, provide reasonable mid-range estimates, but always return numeric percentile values.
{domain_section}
## World Report
{world_report}

{question_section}
{questions_text}

{instruction}"""


def parse_percentiles(response: str, num_questions: int) -> list[dict | None]:
    """Parse percentile estimates from model response."""
    results = []

    # Find the percentiles block
    match = re.search(r'<<<PERCENTILES>>>(.*?)<<<END>>>', response, re.DOTALL)
    if not match:
        return [None] * num_questions

    block = match.group(1).strip()
    lines = [l.strip() for l in block.split('\n') if l.strip()]

    for line in lines:
        # Remove Q1:, Q2: prefix if present
        line = re.sub(r'^Q\d+:\s*', '', line)

        percentile_dict = {}
        for key in ['p10', 'p25', 'p50', 'p75', 'p90']:
            m = re.search(rf'{key}\s*=\s*([-\d.]+)', line)
            if m:
                percentile_dict[key] = float(m.group(1))

        if len(percentile_dict) == 5:
            results.append(percentile_dict)
        else:
            results.append(None)

    # Pad or truncate
    while len(results) < num_questions:
        results.append(None)
    return results[:num_questions]


# ── Main ────────────────────────────────────────────────────────────────────

async def run_experiment(model_id: str, seeds: list[str], dry_run: bool = False):
    model = LiteLLMModel(model_id)
    all_results = []
    gt_map = load_ground_truth_map()

    for condition in CONDITIONS:
        if condition == "baseline":
            data_dir = Path("data/conditional/republic/baseline")
            q_file = "questions.json"
        else:
            data_dir = Path("data/conditional/republic/conditional")
            q_file = "conditional_questions.json"

        for seed in seeds:
            seed_dir = data_dir / seed

            # Load questions
            with open(seed_dir / q_file) as f:
                qdata = json.load(f)

            templates = TEMPLATES_CONDITIONAL if condition == "conditional" else TEMPLATES_BASELINE
            questions = [
                q for q in qdata['questions']
                if q['template_id'] in templates and q['question_type'] == 'continuous'
            ]

            if not questions:
                print(f"  No matching questions for {condition}/{seed}")
                continue

            # Load world report
            report_path = seed_dir / "world_report" / "turn_060_report.txt"
            with open(report_path) as f:
                world_report = f.read()

            # Build prompt (treatment only — control data already exists)
            prompt = build_continuous_prompt(questions, world_report, treatment=True)

            print(f"\n{'='*60}")
            print(f"  {condition}/{seed}: {len(questions)} questions")
            print(f"  Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")

            if dry_run:
                print(f"  [DRY RUN] Prompt preview:")
                print(f"  {prompt[:500]}...")
                for q in questions:
                    truth = gt_map.get((q['question_text'], condition), '?')
                    print(f"    {q['template_id']}: {q['question_text']} -> truth={truth}")
                continue

            # Query model
            print(f"  Querying {model_id}...")
            start = time.monotonic()
            try:
                response = await model.get_response_async(prompt, temperature=0.0)
                latency = (time.monotonic() - start) * 1000
                print(f"  Response in {latency:.0f}ms ({len(response)} chars)")

                # Parse
                percentiles = parse_percentiles(response, len(questions))

                for i, (q, pcts) in enumerate(zip(questions, percentiles)):
                    truth = gt_map.get((q['question_text'], condition), q['resolution']['value_at_resolution'])
                    result = {
                        'condition': condition,
                        'seed': seed,
                        'question_id': q['question_id'],
                        'template_id': q['template_id'],
                        'horizon': q['horizon'],
                        'resolution_turn': q['resolution_turn'],
                        'question_text': q['question_text'],
                        'ground_truth': truth,
                        'treatment_percentiles': pcts,
                        'treatment_p50': pcts['p50'] if pcts else None,
                        'signed_error': (pcts['p50'] - truth) if pcts else None,
                        'latency_ms': latency,
                    }
                    all_results.append(result)

                    status = "ok" if pcts else "FAIL"
                    p50 = pcts['p50'] if pcts else '???'
                    overshoot = f"{pcts['p50']/truth:.1f}x" if pcts and truth else '?'
                    print(f"    {status} {q['template_id']:<35} H{q['horizon'][-1]}: "
                          f"p50={p50:<8} truth={truth:<8} ({overshoot})")

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    return all_results


async def main():
    parser = argparse.ArgumentParser(description="Standardized domain knowledge intervention experiment")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seeds", nargs="+", default=SEEDS)
    parser.add_argument("--model", default=MODEL_ID)
    args = parser.parse_args()

    results = await run_experiment(args.model, args.seeds, args.dry_run)

    if results:
        # Save
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model_short = args.model.split('/')[-1]
        out_file = OUTPUT_DIR / f"dk_{model_short}.json"
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_file}")
        print(f"Total results: {len(results)}")


if __name__ == "__main__":
    asyncio.run(main())
