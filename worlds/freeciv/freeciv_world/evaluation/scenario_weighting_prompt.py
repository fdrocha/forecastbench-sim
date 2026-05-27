"""
Scenario weighting prompt for the mixture decomposition experiment (sub-task 2).

Gives models a fixed two-scenario decomposition (continuation vs. disruption)
and asks only for probability weights. Removes identification from the task
to isolate whether models assign appropriate P(disruption).

Uses the same domain knowledge as other intervention conditions.
"""

FREECIV_DOMAIN_KNOWLEDGE = """FreeCiv is a turn-based 4X strategy game (explore, expand, exploit, exterminate) inspired by the Civilization series. Each game simulates multiple civilizations developing on a shared map over hundreds of turns.

**How the game works:**
- Civilizations start with a single settler and must found cities, research technologies, and manage resources.
- Turns advance simultaneously for all civilizations. The game runs autonomously (AI-controlled players).
- Civilizations interact through diplomacy, trade, and warfare. Wars can lead to city capture, territory loss, and population collapse.

**Reading the world report:**
- **Population:** Total population across all cities. Grows with city development; drops sharply if cities are captured or destroyed.
- **Territory:** Tiles controlled. Expands with new cities and border growth; shrinks if cities are lost.
- **Treasury:** Gold reserves. Accumulates from trade; can deplete rapidly during wars or government transitions.
- **Cities:** Number of cities. Increases from founding; decreases if captured by rivals.
- **Techs:** Technologies discovered. Follows a research tree; never decreases (knowledge is not lost).
- **Score:** Composite metric reflecting overall civilization development.
- **Government:** Political system (Despotism, Republic, etc.). Affects economic output and military capability. Transitions cause temporary "Anarchy" (zero output).
- **Civilizations with 0 population and "Anarchy" government at turn 60 have not yet been founded** — they may emerge later in the game.

**The report shows the first 60 of ~300 turns.** You are forecasting what happens over the remaining ~240 turns.

"""

# Scenario descriptions per template
SCENARIO_DESCRIPTIONS = {
    "cities_count_continuous": {
        "continuation": "Continued growth — no major conflict disrupts city founding. The civilization continues expanding, founding new cities and growing its urban network.",
        "disruption": "War disruption — a neighboring civilization attacks, leading to city capture or destruction. The city count drops significantly (>20% from its peak) due to military conflict.",
    },
    "population_continuous": {
        "continuation": "Continued growth — population grows steadily through city development and expansion. No major wars or catastrophes cause significant population loss.",
        "disruption": "War-driven collapse — military conflict leads to city captures or destruction, causing population to drop significantly (>20% from its peak).",
    },
    "territory_continuous": {
        "continuation": "Continued expansion — territory grows through new cities and border growth. No major losses to rival civilizations.",
        "disruption": "Territorial loss — war or rival expansion causes significant territory loss (>20% from peak). Cities captured or borders pushed back by military force.",
    },
    "treasury_continuous": {
        "continuation": "Continued accumulation — treasury grows through trade and economic output. No major economic shocks or war-related expenses drain the reserves.",
        "disruption": "Economic disruption — war costs, government transitions, or loss of trade routes cause the treasury to drop significantly (>20% from its peak).",
    },
}


def build_scenario_weighting_prompt(
    questions: list[dict], world_report: str
) -> str:
    """Build a prompt that asks models to weight fixed scenarios.

    For each template × horizon, presents two fixed scenarios and asks
    for P(disruption). Output is a single number per question.

    Args:
        questions: List of question dicts with template_id, resolution_turn, question_text.
        world_report: World report text.

    Returns:
        Formatted prompt string.
    """
    from collections import defaultdict
    by_template = defaultdict(list)
    for q in questions:
        by_template[q["template_id"]].append(q)

    for tmpl in by_template:
        by_template[tmpl].sort(key=lambda q: q["resolution_turn"])

    # Build per-template scenario weighting blocks
    template_blocks = []
    q_num = 0
    q_map = []  # track question numbering for parsing

    for tmpl_id in sorted(by_template.keys()):
        qs = by_template[tmpl_id]
        metric_name = tmpl_id.replace("_continuous", "").replace("_", " ")
        scenarios = SCENARIO_DESCRIPTIONS[tmpl_id]

        block = f"### {metric_name}\n\n"
        block += f"**Scenario A (Continuation):** {scenarios['continuation']}\n\n"
        block += f"**Scenario B (Disruption):** {scenarios['disruption']}\n\n"
        block += "For each horizon below, estimate the probability (0-100%) that Scenario B (Disruption) occurs by that turn:\n\n"

        for q in qs:
            q_num += 1
            block += f"Q{q_num}. By turn {q['resolution_turn']}: P(Disruption) = ?\n"
            q_map.append({"q_num": q_num, "template_id": tmpl_id,
                          "resolution_turn": q["resolution_turn"]})

        template_blocks.append(block)

    templates_text = "\n\n".join(template_blocks)

    return f"""You are an expert superforecaster analyzing a FreeCiv game simulation. For each metric below, two scenarios are described. Your task is to estimate the probability that the Disruption scenario occurs by each horizon.

**Important:** Consider the game state carefully. Look at which civilizations are nearby, their military strength relative to the player, and whether early signs of conflict are visible.

## About This Game

{FREECIV_DOMAIN_KNOWLEDGE}## World Report
{world_report}

## Task

{templates_text}

Format your response as follows (you may reason before this block, but you MUST include it):

<<<WEIGHTS>>>
Q1: 45
Q2: 55
Q3: 65
Q4: 70
Q5: 75
Q6: 80
Q7: 30
...
<<<END>>>

Each line should contain only the question number and the probability (0-100, no % sign). Provide one line for each of the {q_num} questions."""
