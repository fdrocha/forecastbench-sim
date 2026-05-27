#!/usr/bin/env python3
"""Generate survey files and answer keys for the agentic baseline experiment.

Assembles survey_seed{N}.txt and answer_key_seed{N}.json from:
  - World report: data/conditional/republic/baseline/{seed}/world_report/turn_060_report.txt
  - Game data:    data/games/{seed}_data.json  (civ names + ground truth values)

Usage:
    cd /Users/elsehow/Projects/civbench
    uv run python scripts/generate_agentic_surveys.py          # all 10 seeds
    uv run python scripts/generate_agentic_surveys.py seed4    # single seed
"""

import json
import sys
from pathlib import Path

CIVBENCH_DIR = Path(__file__).parent.parent
RESULTS_DIR = CIVBENCH_DIR / "data" / "results" / "agentic_baseline"
GAMES_DIR = CIVBENCH_DIR / "data" / "games"
BASELINE_DIR = CIVBENCH_DIR / "data" / "conditional" / "republic" / "baseline"

ALL_SEEDS = ["seed0", "seed1", "seed4", "seed5", "seed9",
             "seed10", "seed13", "seed15", "seed16", "seed20"]

HORIZONS = [90, 120, 150, 180, 210, 240]
HORIZON_LABELS = {90: "H1", 120: "H2", 150: "H3", 180: "H4", 210: "H5", 240: "H6"}

# Template definitions: (ts_key, group_label, question_template)
TEMPLATES = [
    ("cities_count", "cities_count_continuous",
     "number of cities across future turns",
     "How many cities will {civ} have at turn {turn}?"),
    ("population", "population_continuous",
     "population across future turns",
     "What will {civ}'s population be at turn {turn}?"),
    ("territory_size", "territory_continuous",
     "territory (tiles controlled) across future turns",
     "How many tiles will {civ} control at turn {turn}?"),
    ("treasury", "treasury_continuous",
     "treasury (gold) across future turns",
     "How much gold will {civ} have at turn {turn}?"),
]


def generate_survey(seed: str):
    """Generate survey file and answer key for a single seed."""
    # Load world report
    report_path = BASELINE_DIR / seed / "world_report" / "turn_060_report.txt"
    world_report = report_path.read_text()

    # Load game data
    game_path = GAMES_DIR / f"{seed}_data.json"
    with open(game_path) as f:
        game = json.load(f)

    civ_name = game["civilizations"]["0"]["name"]
    ts = game["time_series"]

    # Build questions section
    questions_lines = []
    answer_key = []
    q_num = 1

    for ts_key, template_id, group_label, q_template in TEMPLATES:
        start = q_num
        end = q_num + len(HORIZONS) - 1
        questions_lines.append(f"Questions {start}-{end}: {group_label}.")

        for turn in HORIZONS:
            q_text = q_template.format(civ=civ_name, turn=turn)
            questions_lines.append(f"{q_num}. {q_text}")

            gt = ts[ts_key][str(turn)]["0"]
            answer_key.append({
                "question_number": q_num,
                "question_id": f"{HORIZON_LABELS[turn]}_{template_id}",
                "template_id": template_id,
                "horizon": HORIZON_LABELS[turn],
                "resolution_turn": turn,
                "question_text": q_text,
                "ground_truth": gt,
            })
            q_num += 1

        questions_lines.append("")  # blank line between groups

    questions_text = "\n".join(questions_lines).rstrip()

    # Assemble survey
    survey = f"## World Report\n{world_report}\n## Questions\n{questions_text}\n"

    # Write files
    survey_path = RESULTS_DIR / f"survey_{seed}.txt"
    survey_path.write_text(survey)

    key_path = RESULTS_DIR / f"answer_key_{seed}.json"
    with open(key_path, "w") as f:
        json.dump(answer_key, f, indent=2)

    print(f"{seed}: {civ_name}, {len(answer_key)} questions -> {survey_path.name}, {key_path.name}")


def main():
    seeds = sys.argv[1:] if len(sys.argv) > 1 else ALL_SEEDS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        generate_survey(seed)


if __name__ == "__main__":
    main()
