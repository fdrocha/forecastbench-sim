#!/usr/bin/env python3
"""
Generate-scenario SME experiment.

Models generate their own scenarios, weight them, and provide conditional
distributions — all in one prompt. Mixtures are reconstructed externally.

Usage:
    cd /Users/elsehow/Projects/civbench
    uv run python scripts/run_generate_scenario.py --dry-run
    uv run python scripts/run_generate_scenario.py
    uv run python scripts/run_generate_scenario.py --models openai/gpt-4.1-2025-04-14 --seeds seed0
"""

import argparse
import asyncio
import json
import re
import sys
import time
import numpy as np
from functools import partial
from pathlib import Path
from collections import defaultdict

print = partial(print, flush=True)

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from civrealm.evaluation.models import load_api_keys_from_gcp, LiteLLMModel
from civrealm.evaluation.generate_scenario_prompt import build_generate_scenario_prompt

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
RESULTS_DIR = Path("data/results/generate_scenario")

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

ECI_MAP = {
    "openai/gpt-4.1-2025-04-14": 1471,
    "anthropic/claude-sonnet-4-5-20250929": 1470,
    "openai/gpt-5-2025-08-07": 1537,
    "google/gemini-3-pro-preview": 1474,
    "anthropic/claude-opus-4-6": 1542,
}

MODEL_SHORT = {
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "anthropic/claude-sonnet-4-5-20250929": "Sonnet 4.5",
    "openai/gpt-5-2025-08-07": "GPT-5",
    "google/gemini-3-pro-preview": "Gemini 3 Pro",
    "anthropic/claude-opus-4-6": "Opus 4.6",
}

# Template parse names
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

def parse_gensme(response: str, num_questions: int) -> list[dict | None]:
    """Parse generate-scenario SME output.

    Returns list of dicts with keys:
    - scenarios: list of (label, weight) per scenario
    - percentiles: dict of scenario_label -> {p10, p25, p50, p75, p90}
    - mixture: {p10, p25, p50, p75, p90} (externally mixed)
    """
    match = re.search(r"<<<GENSME>>>(.*?)<<<END>>>", response, re.DOTALL)
    if not match:
        return [None] * num_questions

    block = match.group(1).strip()
    results = []
    current_scenarios = []  # list of scenario labels like ['A', 'B', 'C']

    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue

        # Template header
        if line.upper().startswith("TEMPLATE:"):
            continue

        # Scenario definition line
        if line.upper().startswith("SCENARIOS:"):
            # Parse "A=Steady growth, B=War disruption, C=Stagnation"
            scenario_str = line.split(":", 1)[1].strip()
            current_scenarios = []
            for part in scenario_str.split(","):
                part = part.strip()
                if "=" in part:
                    label = part.split("=")[0].strip()
                    current_scenarios.append(label)
            continue

        # Question line
        q_match = re.match(r"Q\d+:\s*(.*)", line)
        if q_match and current_scenarios:
            content = q_match.group(1)

            # Parse weights
            weights = {}
            for label in current_scenarios:
                w_match = re.search(rf"w{label}\s*=\s*([\d.]+)", content)
                if w_match:
                    weights[label] = float(w_match.group(1))

            # Parse conditional percentiles
            conditionals = {}
            for label in current_scenarios:
                pcts = {}
                for pk in ["p10", "p25", "p50", "p75", "p90"]:
                    m = re.search(rf"{label}_{pk}\s*=\s*([-\d.]+)", content)
                    if m:
                        pcts[pk] = float(m.group(1))
                if len(pcts) == 5:
                    conditionals[label] = pcts

            # Validate
            if len(weights) >= 2 and len(conditionals) >= 2:
                # Normalize weights to sum to 1
                total_w = sum(weights.values())
                if total_w > 0:
                    norm_weights = {k: v / total_w for k, v in weights.items()}
                else:
                    norm_weights = {k: 1.0 / len(weights) for k in weights}

                # Compute mixture
                mixture = {}
                for pk in ["p10", "p25", "p50", "p75", "p90"]:
                    val = 0.0
                    for label in conditionals:
                        if label in norm_weights:
                            val += norm_weights[label] * conditionals[label][pk]
                    mixture[pk] = val

                results.append({
                    "scenario_labels": current_scenarios,
                    "weights": weights,
                    "weights_normalized": norm_weights,
                    "conditionals": conditionals,
                    "mixture": mixture,
                    "n_scenarios": len(current_scenarios),
                })
            else:
                results.append(None)

    while len(results) < num_questions:
        results.append(None)
    return results[:num_questions]


# ── CRPS ────────────────────────────────────────────────────────────────────

def crps_quantile(quantiles: dict, ground_truth: float) -> float:
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

    test_data = {}
    for seed in seeds:
        report = load_world_report(seed)
        questions = load_disruptable_questions(seed)
        test_data[seed] = (report, questions)
        print(f"Loaded {seed}: {len(questions)} Qs")

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

            prompt = build_generate_scenario_prompt(questions, report)

            print(f"\n  [{call_num}/{total_calls}] {model_short} × {seed}: ~{len(prompt)//4} tokens")

            if dry_run:
                print(f"    [DRY RUN] Skipping")
                continue

            start = time.monotonic()
            try:
                response = await model.get_response_async(prompt, temperature=0.0)
                latency = (time.monotonic() - start) * 1000
                print(f"    Response in {latency:.0f}ms ({len(response)} chars)")

                gensme_results = parse_gensme(response, len(questions))
                parsed = sum(1 for s in gensme_results if s is not None)

                # Build ordered questions
                ordered_questions = []
                for tmpl_id in sorted(DISRUPTABLE_TEMPLATES):
                    for q in questions:
                        if q["template_id"] == tmpl_id:
                            ordered_questions.append(q)

                for q, gs in zip(ordered_questions, gensme_results):
                    truth = q["resolution"]["value_at_resolution"]

                    if gs is not None:
                        mix_crps = crps_quantile(gs["mixture"], truth)

                        result = {
                            "model": model_id,
                            "seed": seed,
                            "template_id": q["template_id"],
                            "horizon": q.get("horizon", HORIZON_LABELS.get(q["resolution_turn"], "?")),
                            "resolution_turn": q["resolution_turn"],
                            "ground_truth": truth,
                            "n_scenarios": gs["n_scenarios"],
                            "scenario_labels": gs["scenario_labels"],
                            "weights": gs["weights"],
                            "conditionals": gs["conditionals"],
                            "mixture": gs["mixture"],
                            "mixture_crps": mix_crps,
                        }
                        all_results.append(result)

                        w_str = ", ".join(f"{k}={v:.0f}" for k, v in gs["weights"].items())
                        print(f"    {q['template_id']:<28} {q.get('horizon','?'):<4} "
                              f"CRPS={mix_crps:<8.1f} [{w_str}] truth={truth}")
                    else:
                        print(f"    {q['template_id']:<28} {q.get('horizon','?'):<4} PARSE FAILED")

                print(f"    Parsed: {parsed}/{len(questions)}")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    return all_results


def print_summary(results):
    if not results:
        return

    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Per-model
    print(f"\n{'Model':<15} {'ECI':>5} {'N':>5} {'GenSME CRPS':>12}")
    print("-" * 40)

    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r["mixture_crps"])

    for model in sorted(ECI_MAP, key=lambda m: ECI_MAP[m]):
        vals = by_model[model]
        if vals:
            print(f"{MODEL_SHORT.get(model, model):<15} {ECI_MAP[model]:>5} {len(vals):>5} {np.mean(vals):>11.1f}")
    all_vals = [r["mixture_crps"] for r in results]
    print(f"{'POOLED':<15} {'':>5} {len(all_vals):>5} {np.mean(all_vals):>11.1f}")

    # Per-horizon
    print(f"\n{'Horizon':>8} {'GenSME CRPS':>12}")
    print("-" * 22)
    for turn in sorted(HORIZON_LABELS):
        h = HORIZON_LABELS[turn]
        h_vals = [r["mixture_crps"] for r in results if r["resolution_turn"] == turn]
        if h_vals:
            print(f"  {h:<6} {np.mean(h_vals):>11.1f}")

    # Avg number of scenarios
    avg_n = np.mean([r["n_scenarios"] for r in results])
    print(f"\nAvg scenarios per question: {avg_n:.1f}")

    # Parse rate
    print(f"Total parsed: {len(results)}")


def main():
    parser = argparse.ArgumentParser(description="Generate-scenario SME")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--seeds", nargs="+", default=TEST_SEEDS)
    args = parser.parse_args()

    results = asyncio.run(run_experiment(args.models, seeds=args.seeds,
                                          dry_run=args.dry_run))

    if results:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_file = RESULTS_DIR / "gensme_results.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {out_file} ({len(results)} results)")

        print_summary(results)


if __name__ == "__main__":
    main()
