#!/usr/bin/env python3
"""Generate survey materials for Zach to paste into Quorum.

Outputs:
  survey_materials/
    example_worlds/
      seed10_world_report.md
      seed1_world_report.md
    worlds/
      seed{N}/
        world_report.md
        questions.md        # trajectory-format questions with bins
"""

import json
import shutil
from pathlib import Path

REPO = Path(__file__).parent.parent
HUMAN_SUBJECTS = Path(__file__).parent
OUTPUT = HUMAN_SUBJECTS / "survey_materials"

EXAMPLE_SEEDS = ["seed10", "seed1"]
SURVEY_SEEDS = ["seed1976", "seed1515", "seed1327", "seed1525"]

TEMPLATES = ["treasury_continuous", "cities_count_continuous", "techs_continuous"]
HORIZONS = ["H1", "H3", "H4", "H6"]
HORIZON_TURNS = {"H1": 90, "H3": 150, "H4": 180, "H6": 240}

BIN_EDGES = {
    "treasury_continuous": [0, 0, 98, 167, 236, 331, 469, 648, 891, 1280, 6578],
    "cities_count_continuous": [0, 1, 5, 8, 10, 13, 16, 20, 24, 29, 102],
    "techs_continuous": [0, 24, 31, 36, 40, 44, 47, 49, 52, 56, 113],
}

TEMPLATE_LABELS = {
    "treasury_continuous": "treasury (gold)",
    "cities_count_continuous": "number of cities",
    "techs_continuous": "technologies discovered",
}

# Adapted from src/civrealm/evaluation/domain_knowledge_prompt.py
FREECIV_BACKGROUND = """\
## About This Game

FreeCiv is a turn-based strategy game inspired by the Civilization series. \
Each game simulates multiple civilizations developing on a shared map over \
hundreds of turns.

### How the game works

- Civilizations start with a single settler and must found cities, research \
technologies, and manage resources.
- Turns advance simultaneously for all civilizations. The game runs \
autonomously (AI-controlled players).
- Civilizations interact through diplomacy, trade, and warfare. Wars can lead \
to city capture, territory loss, and population collapse.

### Reading the world report

- **Population:** Total population across all cities. Grows with city \
development; drops sharply if cities are captured or destroyed.
- **Territory:** Tiles controlled. Expands with new cities and border growth; \
shrinks if cities are lost.
- **Treasury:** Gold reserves. Accumulates from trade; can deplete rapidly \
during wars or government transitions.
- **Cities:** Number of cities. Increases from founding; decreases if captured \
by rivals.
- **Techs:** Technologies discovered. Follows a research tree; never \
decreases (knowledge is not lost).
- **Score:** Composite metric reflecting overall civilization development.
- **Government:** Political system (Despotism, Republic, etc.). Affects \
economic output and military capability. Transitions cause temporary \
"Anarchy" (zero output).
- **Civilizations with 0 population and "Anarchy" government at turn 60 have \
not yet been founded** — they may emerge later in the game.

### What you are forecasting

The world report shows the first 60 of ~300 turns. You will be forecasting \
what happens over the remaining ~240 turns.
"""


def bin_labels(template_id: str) -> list[str]:
    edges = BIN_EDGES[template_id]
    labels = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == 0 and lo == hi:
            labels.append(f"{lo}")
        elif i == 0:
            labels.append(f"{lo}–{hi}")
        elif i == len(edges) - 2:
            labels.append(f"{lo + 1}+")
        else:
            labels.append(f"{lo + 1}–{hi}")
    return labels


def txt_to_markdown(txt_path: Path, maps_dir: Path, maps_output_dir: Path) -> str:
    """Convert a TXT world report to markdown, embedding territory maps."""
    lines = txt_path.read_text().splitlines()
    md = []

    for line in lines:
        # Skip original territory map file paths and header
        if line.startswith("Turn ") and "territory" in line.lower() and "/" in line:
            continue
        if line == "TERRITORY MAPS (PNG)":
            continue
        # Convert section headers
        if line.isupper() and line.strip() and not line.startswith("---") and "|" not in line:
            md.append(f"\n## {line.strip()}\n")
        elif "|" in line and "---" not in line:
            md.append(line)
        elif line.startswith("---+"):
            # Convert table separators
            parts = line.split("+")
            md_sep = "|".join("-" * len(p) for p in parts)
            md.append(f"|{md_sep}|")
        else:
            md.append(line)

    # Append territory maps section
    if maps_dir.exists():
        map_files = sorted(maps_dir.glob("territory_turn_*.png"))
        if map_files:
            maps_output_dir.mkdir(parents=True, exist_ok=True)
            md.append("\n## TERRITORY MAPS\n")
            for mf in map_files:
                # Extract turn number from filename
                turn_str = mf.stem.replace("territory_turn_", "")
                turn_num = int(turn_str)
                # Copy map to output
                dest = maps_output_dir / mf.name
                shutil.copy2(mf, dest)
                # Relative path from the world_report.md to the maps dir
                rel_path = f"maps/{mf.name}"
                md.append(f"### Turn {turn_num}\n")
                md.append(f"![Territory map at turn {turn_num}]({rel_path})\n")

    return "\n".join(md)


def generate_questions_md(seed: str) -> str:
    """Generate trajectory-format questions with bins for a survey world."""
    questions_path = HUMAN_SUBJECTS / "worlds" / seed / "questions.json"
    data = json.load(open(questions_path))
    qs = data["questions"]

    game_path = REPO / "data" / "games" / f"{seed}_data.json"
    game = json.load(open(game_path))
    civ_name = game["civilizations"]["0"]["name"]

    lines = [f"# Survey Questions — {seed}\n"]
    lines.append(f"**Civilization:** {civ_name} (player 0)\n")
    lines.append("For each question, assign a percentage to each bin.")
    lines.append("Your percentages must sum to 100%.\n")

    q_num = 1
    for template in TEMPLATES:
        label = TEMPLATE_LABELS[template]
        labels = bin_labels(template)

        # Get questions for player 0, this template, target horizons
        template_qs = [
            q for q in qs
            if q["template_id"] == template
            and q["horizon"] in HORIZONS
            and q["parameters"].get("player_id") == 0
        ]
        template_qs.sort(key=lambda q: HORIZON_TURNS[q["horizon"]])

        if not template_qs:
            continue

        lines.append(f"\n---\n")
        lines.append(f"### Questions {q_num}–{q_num + len(template_qs) - 1}: {civ_name}'s {label}\n")
        lines.append(f"**Outcome bins:** {' | '.join(f'[{l}]' for l in labels)}\n")

        for q in template_qs:
            turn = q["resolution_turn"]
            horizon = q["horizon"]
            lines.append(f"**Q{q_num}.** {q['question_text']}")
            lines.append("")
            for l in labels:
                lines.append(f"- [{l}]: ___%")
            lines.append("")
            q_num += 1

    return "\n".join(lines)


def main():
    # Example worlds
    example_dir = OUTPUT / "example_worlds"
    example_dir.mkdir(parents=True, exist_ok=True)

    # FreeCiv background (shown once, before everything else)
    bg_path = OUTPUT / "freeciv_background.md"
    bg_path.write_text(FREECIV_BACKGROUND)
    print(f"  {bg_path}")

    for seed in EXAMPLE_SEEDS:
        txt_path = HUMAN_SUBJECTS / "examples" / seed / "turn_301_report.txt"
        maps_dir = HUMAN_SUBJECTS / "examples" / seed / "turn_301_territory_maps"
        maps_output = example_dir / f"{seed}_maps"
        md = txt_to_markdown(txt_path, maps_dir, maps_output)
        out = example_dir / f"{seed}_world_report.md"
        out.write_text(md)
        print(f"  {out}")

    # Survey worlds
    for seed in SURVEY_SEEDS:
        world_dir = OUTPUT / "worlds" / seed
        world_dir.mkdir(parents=True, exist_ok=True)

        # World report
        txt_path = HUMAN_SUBJECTS / "worlds" / seed / "world_report" / "turn_060_report.txt"
        maps_dir = HUMAN_SUBJECTS / "worlds" / seed / "world_report" / "turn_060_territory_maps"
        maps_output = world_dir / "maps"
        md = txt_to_markdown(txt_path, maps_dir, maps_output)
        out_report = world_dir / "world_report.md"
        out_report.write_text(md)
        print(f"  {out_report}")

        # Questions
        questions_md = generate_questions_md(seed)
        out_qs = world_dir / "questions.md"
        out_qs.write_text(questions_md)
        print(f"  {out_qs}")


if __name__ == "__main__":
    main()
