"""
Scenario enumeration prompt for the mixture decomposition experiment (sub-task 1).

Asks models to enumerate possible scenarios for each metric without assigning
probabilities. Tests whether models spontaneously identify disruption pathways
(war, conquest, city loss) when given game state data.

This isolates scenario *identification* from weighting and conditional forecasting.
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

**The report shows the first 60 of ~300 turns.** You are thinking about what could happen over the remaining ~240 turns.

"""


def build_scenario_enumeration_prompt(
    questions: list[dict], world_report: str
) -> str:
    """Build a prompt that asks models to enumerate scenarios without probabilities.

    Groups questions by template (metric) and asks for scenarios at each horizon.
    Output format is structured text for parsing.

    Args:
        questions: List of question dicts with template_id, resolution_turn, question_text.
        world_report: World report text.

    Returns:
        Formatted prompt string.
    """
    # Group questions by template
    from collections import defaultdict
    by_template = defaultdict(list)
    for q in questions:
        by_template[q["template_id"]].append(q)

    # Sort each group by resolution_turn
    for tmpl in by_template:
        by_template[tmpl].sort(key=lambda q: q["resolution_turn"])

    # Build per-template scenario request
    template_blocks = []
    for tmpl_id in sorted(by_template.keys()):
        qs = by_template[tmpl_id]
        # Use the metric name from the first question
        metric_name = tmpl_id.replace("_continuous", "").replace("_", " ")
        horizons = ", ".join(f"turn {q['resolution_turn']}" for q in qs)

        template_blocks.append(
            f"### {metric_name}\n"
            f"Horizons: {horizons}\n"
            f"For this metric, list 2-5 distinct scenarios that could plausibly unfold. "
            f"Each scenario should describe a qualitatively different trajectory for {metric_name} "
            f"over these horizons. Include at least one scenario where the metric declines "
            f"significantly from its current trajectory."
        )

    templates_text = "\n\n".join(template_blocks)

    return f"""You are analyzing a FreeCiv game simulation. Your task is to enumerate the distinct scenarios that could plausibly unfold for each metric below.

**Important:** Do NOT assign probabilities or forecast specific values. Just describe what could happen and why.

## About This Game

{FREECIV_DOMAIN_KNOWLEDGE}## World Report
{world_report}

## Task

For each metric below, list the distinct scenarios that could unfold over the specified horizons. For each scenario, give:
- A short label (e.g., "Steady growth", "War-driven collapse", "Stagnation")
- A 1-2 sentence description of what happens and what causes it
- Whether the trajectory is: rising, flat, declining, or non-monotonic

{templates_text}

Format your response as follows:

<<<SCENARIOS>>>
TEMPLATE: cities count
S1: [label] | [rising/flat/declining/non-monotonic] | [description]
S2: [label] | [rising/flat/declining/non-monotonic] | [description]
S3: [label] | [rising/flat/declining/non-monotonic] | [description]

TEMPLATE: population
S1: [label] | [rising/flat/declining/non-monotonic] | [description]
S2: [label] | [rising/flat/declining/non-monotonic] | [description]

TEMPLATE: territory
S1: [label] | [rising/flat/declining/non-monotonic] | [description]
S2: [label] | [rising/flat/declining/non-monotonic] | [description]

TEMPLATE: treasury
S1: [label] | [rising/flat/declining/non-monotonic] | [description]
S2: [label] | [rising/flat/declining/non-monotonic] | [description]
<<<END>>>

You may reason about the game state before the formatted block, but you MUST include the <<<SCENARIOS>>>...<<<END>>> block."""
