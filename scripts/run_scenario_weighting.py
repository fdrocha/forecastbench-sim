#!/usr/bin/env python3
"""
Scenario weighting experiment (mixture decomposition sub-task 2).

Gives models fixed scenarios (continuation vs. disruption) and asks only
for P(disruption) at each horizon. Tests whether models assign appropriate
probability to disruption when the scenario is already identified for them.

Usage:
    cd /Users/elsehow/Projects/civbench
    uv run python scripts/run_scenario_weighting.py --dry-run
    uv run python scripts/run_scenario_weighting.py
    uv run python scripts/run_scenario_weighting.py --models openai/gpt-4.1-2025-04-14 --seeds seed0
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

from civrealm.evaluation.models import load_api_keys_from_gcp, LiteLLMModel
from civrealm.evaluation.scenario_weighting_prompt import build_scenario_weighting_prompt

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
RESULTS_DIR = Path("data/results/scenario_weighting")

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

DISRUPTION_THRESHOLD = 0.20  # >20% decline from peak = disruption


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
    report_path = DATA_DIR / seed / "world_report" / "turn_060_report.txt"
    with open(report_path) as f:
        return f.read()


def compute_ground_truth(seed: str) -> dict:
    """Compute per-template per-horizon disruption ground truth.

    Returns dict: (template_id, resolution_turn) -> {disrupted: bool, decline_pct: float, peak: float, value: float}
    """
    game_file = GAMES_DIR / f"{seed}_data.json"
    with open(game_file) as f:
        game = json.load(f)

    ts = game["time_series"]
    ground_truth = {}

    ts_key_map = {
        "population_continuous": "population",
        "territory_continuous": "territory_size",
        "treasury_continuous": "treasury",
        "cities_count_continuous": "cities_count",
    }

    for tmpl_id, ts_key in ts_key_map.items():
        # Get all values for player 0
        values = {}
        for t_str, civs in ts[ts_key].items():
            t = int(t_str)
            if "0" in civs:
                values[t] = civs["0"]

        for turn in HORIZON_TURNS:
            # Peak up to this horizon
            peak = max(v for t, v in values.items() if t <= turn)
            val_at_turn = values.get(turn, 0)
            decline_pct = (peak - val_at_turn) / peak if peak > 0 else 0
            disrupted = decline_pct > DISRUPTION_THRESHOLD

            ground_truth[(tmpl_id, turn)] = {
                "disrupted": disrupted,
                "decline_pct": decline_pct * 100,
                "peak": peak,
                "value": val_at_turn,
            }

    return ground_truth


# ── Parsing ─────────────────────────────────────────────────────────────────

def parse_weights(response: str, num_questions: int) -> list[float | None]:
    """Parse P(disruption) weights from model response."""
    match = re.search(r"<<<WEIGHTS>>>(.*?)<<<END>>>", response, re.DOTALL)
    if not match:
        return [None] * num_questions

    block = match.group(1).strip()
    results = []

    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Match Q<n>: <number>
        m = re.match(r"Q\d+:\s*([\d.]+)", line)
        if m:
            val = float(m.group(1))
            results.append(min(max(val, 0), 100))  # clamp 0-100

    while len(results) < num_questions:
        results.append(None)
    return results[:num_questions]


# ── Main ────────────────────────────────────────────────────────────────────

async def run_experiment(models: list[str], seeds: list[str] = None,
                          dry_run: bool = False):
    ensure_api_keys()
    seeds = seeds or TEST_SEEDS

    # Pre-load data and ground truth
    test_data = {}
    for seed in seeds:
        report = load_world_report(seed)
        questions = load_disruptable_questions(seed)
        ground_truth = compute_ground_truth(seed)
        test_data[seed] = (report, questions, ground_truth)
        n_disrupted = sum(1 for v in ground_truth.values() if v["disrupted"])
        print(f"Loaded {seed}: {len(questions)} Qs, {n_disrupted}/24 disrupted")

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
            report, questions, ground_truth = test_data[seed]

            prompt = build_scenario_weighting_prompt(questions, report)

            print(f"\n  [{call_num}/{total_calls}] {model_short} × {seed}: ~{len(prompt)//4} tokens")

            if dry_run:
                print(f"    [DRY RUN] Skipping API call")
                continue

            start = time.monotonic()
            try:
                response = await model.get_response_async(prompt, temperature=0.0)
                latency = (time.monotonic() - start) * 1000
                print(f"    Response in {latency:.0f}ms ({len(response)} chars)")

                weights = parse_weights(response, len(questions))
                parsed = sum(1 for w in weights if w is not None)

                # Build question order matching prompt (sorted by template, then horizon)
                ordered_questions = []
                for tmpl_id in sorted(DISRUPTABLE_TEMPLATES):
                    for q in questions:
                        if q["template_id"] == tmpl_id:
                            ordered_questions.append(q)

                for q, w in zip(ordered_questions, weights):
                    gt = ground_truth.get((q["template_id"], q["resolution_turn"]), {})
                    disrupted = gt.get("disrupted", None)
                    decline = gt.get("decline_pct", 0)

                    result = {
                        "model": model_id,
                        "seed": seed,
                        "template_id": q["template_id"],
                        "horizon": q.get("horizon", HORIZON_LABELS.get(q["resolution_turn"], "?")),
                        "resolution_turn": q["resolution_turn"],
                        "p_disruption": w,
                        "ground_truth_disrupted": disrupted,
                        "ground_truth_decline_pct": round(decline, 1),
                    }
                    all_results.append(result)

                    marker = "✓" if disrupted else "·"
                    w_str = f"{w:.0f}%" if w is not None else "FAIL"
                    print(f"    {q['template_id']:<28} {q.get('horizon','?'):<4} "
                          f"P(dis)={w_str:<6} actual={marker} (decline={decline:.0f}%)")

                if parsed < len(questions):
                    print(f"    WARNING: only {parsed}/{len(questions)} parsed")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Scenario weighting experiment")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--seeds", nargs="+", default=TEST_SEEDS)
    args = parser.parse_args()

    results = asyncio.run(run_experiment(args.models, seeds=args.seeds,
                                          dry_run=args.dry_run))

    if results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = RESULTS_DIR / "scenario_weighting_results.json"

        with open(out_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {out_file} ({len(results)} results)")

        # Quick summary: avg P(disruption) for disrupted vs non-disrupted
        from collections import defaultdict
        by_model = defaultdict(lambda: {"disrupted_weights": [], "stable_weights": []})
        for r in results:
            if r["p_disruption"] is None:
                continue
            model = r["model"]
            if r["ground_truth_disrupted"]:
                by_model[model]["disrupted_weights"].append(r["p_disruption"])
            else:
                by_model[model]["stable_weights"].append(r["p_disruption"])

        print(f"\n{'='*70}")
        print("SUMMARY: Avg P(disruption) by actual outcome")
        print(f"{'='*70}")
        print(f"{'Model':<35} {'When disrupted':>15} {'When stable':>15} {'Gap':>8}")
        print("-" * 75)
        for model in sorted(by_model):
            d = by_model[model]
            avg_d = sum(d["disrupted_weights"]) / len(d["disrupted_weights"]) if d["disrupted_weights"] else 0
            avg_s = sum(d["stable_weights"]) / len(d["stable_weights"]) if d["stable_weights"] else 0
            gap = avg_d - avg_s
            model_short = model.split("/")[-1]
            print(f"  {model_short:<33} {avg_d:>14.1f}% {avg_s:>14.1f}% {gap:>+7.1f}pp")


if __name__ == "__main__":
    main()
