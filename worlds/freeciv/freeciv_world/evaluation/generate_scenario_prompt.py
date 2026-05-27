"""
Generate-scenario SME prompt.

Like SME but the model generates its own scenarios instead of receiving
fixed ones. Combines all three sub-tasks: enumerate scenarios, weight them,
and provide conditional distributions per scenario.

External mixing reconstructs the final distribution.
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


def build_generate_scenario_prompt(
    questions: list[dict], world_report: str
) -> str:
    """Build a prompt where the model generates scenarios, weights, and conditionals.

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

    # Build per-template blocks
    template_blocks = []
    q_num = 0

    for tmpl_id in sorted(by_template.keys()):
        qs = by_template[tmpl_id]
        metric_name = tmpl_id.replace("_continuous", "").replace("_", " ")

        block = f"### {metric_name}\n\n"
        block += "Questions:\n"
        q_start = q_num + 1
        for q in qs:
            q_num += 1
            block += f"Q{q_num}. {q['question_text']}\n"
        q_end = q_num

        block += f"\nFor {metric_name} (Q{q_start}-Q{q_end}):\n"
        block += "1. **Generate 2-3 scenarios** that could plausibly unfold. At least one should involve significant decline (>20% from peak). Give each a short label.\n"
        block += "2. **Weight** each scenario (probabilities summing to 100%) at each horizon.\n"
        block += "3. **Forecast** conditional percentiles (p10/p25/p50/p75/p90) under each scenario at each horizon.\n"

        template_blocks.append(block)

    templates_text = "\n\n".join(template_blocks)

    return f"""You are an expert superforecaster analyzing a FreeCiv game simulation. For each metric below, you will:
1. Generate plausible scenarios
2. Assign probability weights to each scenario
3. Provide conditional forecasts (percentiles) under each scenario

The scenarios and weights should reflect your analysis of the game state. Think carefully about what could happen — growth, conflict, stagnation, collapse — and how likely each path is.

## About This Game

{FREECIV_DOMAIN_KNOWLEDGE}## World Report
{world_report}

## Task

{templates_text}

You may reason before the output block. You MUST end your response with a block in this exact format:

<<<GENSME>>>
TEMPLATE: cities count
SCENARIOS: A=Steady growth, B=War disruption, C=Stagnation
Q1: wA=70, wB=15, wC=15, A_p10=28, A_p25=30, A_p50=33, A_p75=36, A_p90=40, B_p10=10, B_p25=14, B_p50=18, B_p75=22, B_p90=26, C_p10=20, C_p25=23, C_p50=26, C_p75=28, C_p90=30
Q2: wA=60, wB=25, wC=15, A_p10=32, A_p25=36, A_p50=42, A_p75=48, A_p90=55, B_p10=5, B_p25=10, B_p50=15, B_p75=20, B_p90=25, C_p10=22, C_p25=26, C_p50=30, C_p75=34, C_p90=38
...

TEMPLATE: population
SCENARIOS: A=Continued expansion, B=Collapse
Q7: wA=65, wB=35, A_p10=50, A_p25=60, A_p50=75, A_p75=90, A_p90=110, B_p10=5, B_p25=12, B_p50=22, B_p75=35, B_p90=48
...
<<<END>>>

Rules:
- Weights for each question must sum to 100 (e.g., wA + wB + wC = 100)
- Use 2-3 scenarios per template (labeled A, B, C)
- All percentile values are the actual predicted quantity (not percentages)
- Provide one line per question, {q_num} questions total
- Scenario labels in the SCENARIOS line must match the prefixes in each Q line"""
