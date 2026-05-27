"""
Structured Mixture Elicitation (SME) prompt.

Asks models for scenario weights AND conditional distributions per scenario
in a single call. The distributions are reconstructed externally as a mixture.

This tests whether the integration step is the bottleneck: if the externally-
mixed distribution outperforms the monolithic baseline, the model can produce
the right components but fails to combine them internally.
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


def build_structured_mixture_prompt(
    questions: list[dict], world_report: str
) -> str:
    """Build a prompt asking for scenario weights + conditional distributions.

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
        scenarios = SCENARIO_DESCRIPTIONS[tmpl_id]

        block = f"### {metric_name}\n\n"
        block += f"**Scenario A (Continuation):** {scenarios['continuation']}\n\n"
        block += f"**Scenario B (Disruption):** {scenarios['disruption']}\n\n"
        block += "For each horizon:\n"
        block += "1. Estimate P(Disruption) — the probability (0-100%) that Scenario B occurs by that turn.\n"
        block += "2. Provide conditional percentiles ASSUMING Scenario A occurs (p10/p25/p50/p75/p90).\n"
        block += "3. Provide conditional percentiles ASSUMING Scenario B occurs (p10/p25/p50/p75/p90).\n\n"

        for q in qs:
            q_num += 1
            block += f"Q{q_num}. {q['question_text']}\n"

        template_blocks.append(block)

    templates_text = "\n\n".join(template_blocks)

    return f"""You are an expert superforecaster analyzing a FreeCiv game simulation. For each metric below, two scenarios are described. Your task is to:
1. Estimate the probability of disruption at each horizon
2. Provide percentile forecasts conditional on each scenario

**Important:** The conditional forecasts should reflect what you'd expect IF that scenario occurs. The continuation forecast should assume no major disruption. The disruption forecast should assume significant decline occurs.

## About This Game

{FREECIV_DOMAIN_KNOWLEDGE}## World Report
{world_report}

## Task

{templates_text}

You MUST end your response with the following block. For each question, provide one line with the weight and both conditional distributions:

<<<SME>>>
Q1: w=45, cont_p10=30, cont_p25=35, cont_p50=42, cont_p75=50, cont_p90=60, dis_p10=5, dis_p25=10, dis_p50=18, dis_p75=25, dis_p90=32
Q2: w=55, cont_p10=35, cont_p25=42, cont_p50=50, cont_p75=60, cont_p90=75, dis_p10=3, dis_p25=8, dis_p50=15, dis_p75=22, dis_p90=30
<<<END>>>

Where:
- w = P(Disruption) as a percentage (0-100)
- cont_p10..cont_p90 = percentiles assuming Continuation
- dis_p10..dis_p90 = percentiles assuming Disruption
- All percentile values should be numeric (the actual predicted quantity, not percentages)

Provide one line for each of the {q_num} questions, in order."""
