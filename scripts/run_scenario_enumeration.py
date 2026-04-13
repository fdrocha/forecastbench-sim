#!/usr/bin/env python3
"""
Scenario enumeration experiment (mixture decomposition sub-task 1).

Tests whether models spontaneously identify disruption scenarios when asked
to enumerate possible futures for each metric. No probabilities or values
are requested — only scenario descriptions.

Usage:
    cd /Users/elsehow/Projects/civbench
    uv run python scripts/run_scenario_enumeration.py --dry-run
    uv run python scripts/run_scenario_enumeration.py
    uv run python scripts/run_scenario_enumeration.py --models openai/gpt-4.1-2025-04-14 --seeds seed0
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
from civrealm.evaluation.scenario_enumeration_prompt import build_scenario_enumeration_prompt

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

DATA_DIR = Path("data/conditional/republic/baseline")
GAMES_DIR = Path("data/games")
RESULTS_DIR = Path("data/results/scenario_enumeration")

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

# Canonical template names for parsing (match the prompt output format)
TEMPLATE_PARSE_NAMES = {
    "cities count": "cities_count_continuous",
    "cities_count": "cities_count_continuous",
    "population": "population_continuous",
    "territory": "territory_continuous",
    "territory size": "territory_continuous",
    "treasury": "treasury_continuous",
}


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

# Keywords indicating a disruption scenario
DISRUPTION_KEYWORDS = [
    "war", "conflict", "attack", "invade", "invasion", "conquer", "conquest",
    "capture", "destroy", "destruction", "collapse", "decline", "loss",
    "lose", "lost", "defeat", "raid", "siege", "military",
    "pillage", "plunder", "raze", "annex", "hostile",
]

def parse_scenarios(response: str) -> dict[str, list[dict]]:
    """Parse scenario enumeration from model response.

    Returns:
        Dict mapping template_id -> list of scenario dicts with keys:
        label, trajectory, description, is_disruption
    """
    match = re.search(r"<<<SCENARIOS>>>(.*?)<<<END>>>", response, re.DOTALL)
    if not match:
        return {}

    block = match.group(1).strip()
    results = {}
    current_template = None

    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Check for template header
        tmpl_match = re.match(r"TEMPLATE:\s*(.+)", line, re.IGNORECASE)
        if tmpl_match:
            tmpl_name = tmpl_match.group(1).strip().lower()
            current_template = TEMPLATE_PARSE_NAMES.get(tmpl_name)
            if current_template:
                results[current_template] = []
            continue

        # Check for scenario line
        s_match = re.match(r"S\d+:\s*(.+)", line)
        if s_match and current_template:
            parts = s_match.group(1).split("|")
            label = parts[0].strip() if len(parts) > 0 else ""
            trajectory = parts[1].strip().lower() if len(parts) > 1 else ""
            description = parts[2].strip() if len(parts) > 2 else ""

            full_text = (label + " " + description).lower()
            is_disruption = any(kw in full_text for kw in DISRUPTION_KEYWORDS)
            is_declining = "declining" in trajectory or "non-monotonic" in trajectory

            results[current_template].append({
                "label": label,
                "trajectory": trajectory,
                "description": description,
                "is_disruption": is_disruption,
                "is_declining": is_declining,
            })

    return results


# ── Main ────────────────────────────────────────────────────────────────────

async def run_experiment(models: list[str],
                          seeds: list[str] = None,
                          dry_run: bool = False):
    """Run the scenario enumeration experiment."""
    ensure_api_keys()

    seeds = seeds or TEST_SEEDS

    # Pre-load all test world data
    test_data = {}
    for seed in seeds:
        report = load_world_report(seed)
        questions = load_disruptable_questions(seed)
        test_data[seed] = (report, questions)
        print(f"Loaded {seed}: {len(questions)} questions")

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

            prompt = build_scenario_enumeration_prompt(questions, report)

            print(f"\n  [{call_num}/{total_calls}] {model_short} × {seed}: ~{len(prompt)//4} tokens")

            if dry_run:
                print(f"    [DRY RUN] Skipping API call")
                print(f"    Prompt length: {len(prompt)} chars")
                continue

            start = time.monotonic()
            try:
                response = await model.get_response_async(prompt, temperature=0.0)
                latency = (time.monotonic() - start) * 1000
                print(f"    Response in {latency:.0f}ms ({len(response)} chars)")

                scenarios_by_template = parse_scenarios(response)

                for tmpl_id, scenarios in scenarios_by_template.items():
                    n_disruption = sum(1 for s in scenarios if s["is_disruption"])
                    n_declining = sum(1 for s in scenarios if s["is_declining"])
                    print(f"    {tmpl_id:<28} {len(scenarios)} scenarios, "
                          f"{n_disruption} disruption, {n_declining} declining")

                result = {
                    "model": model_id,
                    "seed": seed,
                    "scenarios_by_template": scenarios_by_template,
                    "raw_response": response,
                    "parse_success": len(scenarios_by_template) > 0,
                    "templates_parsed": list(scenarios_by_template.keys()),
                }
                all_results.append(result)

                if not scenarios_by_template:
                    print(f"    WARNING: No scenarios parsed!")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Scenario enumeration experiment")
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
        out_file = RESULTS_DIR / "scenario_enumeration_results.json"

        # Save without raw_response for the main results file
        clean_results = []
        for r in results:
            clean = {k: v for k, v in r.items() if k != "raw_response"}
            clean_results.append(clean)

        with open(out_file, "w") as f:
            json.dump(clean_results, f, indent=2)
        print(f"\nResults saved to {out_file} ({len(clean_results)} results)")

        # Save raw responses separately
        raw_file = RESULTS_DIR / "raw_responses.json"
        raw = [{"model": r["model"], "seed": r["seed"], "response": r["raw_response"]}
               for r in results]
        with open(raw_file, "w") as f:
            json.dump(raw, f, indent=2)
        print(f"Raw responses saved to {raw_file}")

        # Quick summary
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        from collections import defaultdict
        by_model = defaultdict(list)
        for r in results:
            by_model[r["model"]].append(r)

        for model_id, model_results in sorted(by_model.items()):
            model_short = model_id.split("/")[-1]
            n_seeds = len(model_results)
            n_parsed = sum(1 for r in model_results if r["parse_success"])

            # Count disruption identification across templates and seeds
            total_template_seeds = 0
            total_with_disruption = 0
            for r in model_results:
                for tmpl_id, scenarios in r["scenarios_by_template"].items():
                    total_template_seeds += 1
                    if any(s["is_disruption"] for s in scenarios):
                        total_with_disruption += 1

            pct = (total_with_disruption / total_template_seeds * 100) if total_template_seeds > 0 else 0
            print(f"  {model_short:<35} {n_parsed}/{n_seeds} parsed, "
                  f"disruption identified: {total_with_disruption}/{total_template_seeds} ({pct:.0f}%)")


if __name__ == "__main__":
    main()
