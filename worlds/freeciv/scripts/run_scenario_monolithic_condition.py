#!/usr/bin/env python3
"""
CoT ablation — condition (c): scenarios enumerated + monolithic percentiles.

Tests whether the format affordance (per-scenario quantile slots in
GenSME) is what recovers the CRPS deficit, vs. scenario enumeration +
reasoning alone. 5 models × 10 seeds × 1 call per (model, seed) = 50 calls.

Usage:
    cd /Users/elsehow/Projects/civbench
    uv run python scripts/run_scenario_monolithic_condition.py --dry-run
    uv run python scripts/run_scenario_monolithic_condition.py --models openai/gpt-4.1-2025-04-14 --seeds seed0
    uv run python scripts/run_scenario_monolithic_condition.py
"""

import argparse
import asyncio
import json
import re
import sys
import time
from functools import partial
from pathlib import Path

print = partial(print, flush=True)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fbsim_core.evaluation.models import load_api_keys_from_gcp, LiteLLMModel
from freeciv_world.evaluation.scenario_monolithic_prompt import build_scenario_monolithic_prompt

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
    # Original 5-model intervention set
    "openai/gpt-4.1-2025-04-14",
    "anthropic/claude-sonnet-4-5-20250929",
    "openai/gpt-5-2025-08-07",
    "google/gemini-3-pro-preview",
    "anthropic/claude-opus-4-6",
    # Additional 22 models matching the gensme_all_models generalizability set
    "anthropic/claude-3-haiku-20240307",
    "anthropic/claude-haiku-4-5-20251001",
    "anthropic/claude-sonnet-4-20250514",
    "google/gemini-2.0-flash-lite-001",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "openai/gpt-3.5-turbo-0125",
    "openai/gpt-4o",
    "openai/gpt-5-mini-2025-08-07",
    "openai/gpt-5.1-2025-11-13",
    "openai/o3-mini-2025-01-31",
    "openai/o3-2025-04-16",
    "openai/o4-mini-2025-04-16",
    "mistral/mistral-large-2407",
    "mistral/mistral-large-2411",
    "mistral/mistral-large-latest",
    "together_ai/mistralai/Mixtral-8x7B-Instruct-v0.1",
    "xai/grok-4-0709",
    "xai/grok-4-fast-non-reasoning",
    "xai/grok-4-fast-reasoning",
    "xai/grok-4-1-fast-non-reasoning",
    "xai/grok-4-1-fast-reasoning",
]

DATA_DIR = Path("data/conditional/republic/baseline")
GAMES_DIR = Path("data/games")
RESULTS_DIR = Path("data/results/scenario_monolithic_condition")

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


# ── Data loading (same as run_gensme_all_models.py) ────────────────────────

def load_disruptable_questions(seed: str) -> list[dict]:
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
    report_path = DATA_DIR / seed / "world_report" / "turn_060_report.txt"
    with open(report_path) as f:
        return f.read()


# ── Parsing ─────────────────────────────────────────────────────────────────

def parse_monolithic(response: str, num_questions: int) -> list[dict | None]:
    """Extract p10/p25/p50/p75/p90 per question from <<<PERCENTILES>>> block.

    Also records the scenario labels declared per template (for auditing that
    the model actually enumerated scenarios; not used in scoring).
    """
    match = re.search(r"<<<PERCENTILES>>>(.*?)<<<END>>>", response, re.DOTALL)
    if not match:
        return [None] * num_questions

    block = match.group(1).strip()
    results = [None] * num_questions
    current_scenarios = []
    current_template = None

    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.upper().startswith("TEMPLATE:"):
            current_template = line.split(":", 1)[1].strip()
            continue

        if line.upper().startswith("SCENARIOS:"):
            scenario_str = line.split(":", 1)[1].strip()
            current_scenarios = []
            for part in scenario_str.split(","):
                part = part.strip()
                if "=" in part:
                    label = part.split("=")[0].strip()
                    current_scenarios.append(label)
            continue

        q_match = re.match(r"Q(\d+):\s*(.*)", line)
        if q_match:
            q_idx = int(q_match.group(1)) - 1
            content = q_match.group(2)

            pcts = {}
            for pk in ["p10", "p25", "p50", "p75", "p90"]:
                m = re.search(rf"\b{pk}\s*=\s*([-\d.]+)", content)
                if m:
                    pcts[pk] = float(m.group(1))

            if len(pcts) == 5 and 0 <= q_idx < num_questions:
                # Sanity check: non-decreasing percentiles
                vals = [pcts[k] for k in ["p10", "p25", "p50", "p75", "p90"]]
                if vals == sorted(vals):
                    results[q_idx] = {
                        "percentiles": pcts,
                        "template": current_template,
                        "scenarios_declared": list(current_scenarios),
                        "n_scenarios_declared": len(current_scenarios),
                    }

    return results


def crps_quantile(quantiles: dict, ground_truth: float) -> float:
    """Pinball-based CRPS estimate from 5 quantiles (matches gensme scoring)."""
    taus = [0.10, 0.25, 0.50, 0.75, 0.90]
    keys = ["p10", "p25", "p50", "p75", "p90"]
    total = 0.0
    for tau, key in zip(taus, keys):
        q = quantiles[key]
        diff = ground_truth - q
        if diff >= 0:
            total += tau * diff
        else:
            total += (tau - 1) * diff
    return total * 2 / len(taus)


# ── Main ────────────────────────────────────────────────────────────────────

async def run_experiment(models: list[str], seeds: list[str] = None,
                          dry_run: bool = False):
    ensure_api_keys()
    seeds = seeds or TEST_SEEDS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "scenario_monolithic_results.json"
    existing_results = []
    existing_keys = set()
    if out_file.exists():
        with open(out_file) as f:
            existing_results = json.load(f)
        for r in existing_results:
            existing_keys.add((r["model"], r["seed"]))
        print(f"Loaded {len(existing_results)} existing results ({len(existing_keys)} model-seed pairs)")

    test_data = {}
    for seed in seeds:
        report = load_world_report(seed)
        questions = load_disruptable_questions(seed)
        test_data[seed] = (report, questions)

    all_results = list(existing_results)
    calls_needed = [(m, s) for m in models for s in seeds if (m, s) not in existing_keys]
    total_calls = len(calls_needed)
    print(f"{total_calls} API calls needed ({len(models)} models × {len(seeds)} seeds - {len(existing_keys)} already done)")

    call_num = 0
    for model_id, seed in calls_needed:
        call_num += 1
        model_short = model_id.split("/")[-1]
        report, questions = test_data[seed]

        prompt = build_scenario_monolithic_prompt(questions, report)

        print(f"\n  [{call_num}/{total_calls}] {model_short} × {seed}: ~{len(prompt)//4} tokens")

        if dry_run:
            print(f"    [DRY RUN] Skipping")
            continue

        model = LiteLLMModel(model_id)
        start = time.monotonic()
        try:
            response = await model.get_response_async(prompt, temperature=0.0)
            latency = (time.monotonic() - start) * 1000
            print(f"    Response in {latency:.0f}ms ({len(response)} chars)")

            parsed = parse_monolithic(response, len(questions))
            n_parsed = sum(1 for p in parsed if p is not None)

            # Align parsed results to the canonical question ordering used by gensme
            ordered_questions = []
            for tmpl_id in sorted(DISRUPTABLE_TEMPLATES):
                for q in questions:
                    if q["template_id"] == tmpl_id:
                        ordered_questions.append(q)

            for q, p in zip(ordered_questions, parsed):
                truth = q["resolution"]["value_at_resolution"]
                if p is not None:
                    crps = crps_quantile(p["percentiles"], truth)
                    all_results.append({
                        "model": model_id,
                        "seed": seed,
                        "template_id": q["template_id"],
                        "horizon": q.get("horizon", HORIZON_LABELS.get(q["resolution_turn"], "?")),
                        "resolution_turn": q["resolution_turn"],
                        "ground_truth": truth,
                        "percentiles": p["percentiles"],
                        "n_scenarios_declared": p["n_scenarios_declared"],
                        "scenarios_declared": p["scenarios_declared"],
                        "crps": crps,
                    })

            print(f"    Parsed: {n_parsed}/{len(questions)}")

            with open(out_file, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

        except Exception as e:
            print(f"    ERROR: {e}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Condition (c): scenarios + monolithic percentiles")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--seeds", nargs="+", default=TEST_SEEDS)
    args = parser.parse_args()

    results = asyncio.run(run_experiment(args.models, seeds=args.seeds,
                                          dry_run=args.dry_run))

    if results:
        print(f"\n{'='*70}")
        print(f"DONE: {len(results)} total results")
        models_done = set(r["model"] for r in results)
        for m in sorted(models_done):
            n = sum(1 for r in results if r["model"] == m)
            print(f"  {m}: {n}")


if __name__ == "__main__":
    main()
