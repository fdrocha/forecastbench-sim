"""
Condition (c) ablation prompt: scenarios enumerated + monolithic percentiles.

Identical to GenSME in domain knowledge and scenario-enumeration instructions,
but the model produces ONE p10-p90 set per question instead of per-scenario
weights + conditional distributions. Tests whether the format affordance
(per-scenario quantile slots) is what drives decomposed-elicitation recovery,
vs. scenario enumeration + reasoning alone.
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


def build_scenario_monolithic_prompt(
    questions: list[dict], world_report: str
) -> str:
    """Build a prompt where the model generates scenarios then outputs ONE
    set of percentiles per question (mixture integration is done internally).

    Differs from GenSME only in the output format: no per-scenario weights
    or conditionals, just a single p10/p25/p50/p75/p90 per question that
    the model must integrate across scenarios itself.
    """
    from collections import defaultdict
    by_template = defaultdict(list)
    for q in questions:
        by_template[q["template_id"]].append(q)

    for tmpl in by_template:
        by_template[tmpl].sort(key=lambda q: q["resolution_turn"])

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
        block += "2. **Consider** how likely each scenario is at each horizon.\n"
        block += "3. **Forecast percentiles** (p10/p25/p50/p75/p90) that integrate across your scenarios — a single percentile set per question that captures the full mixture of possibilities, weighted by how likely each scenario is.\n"

        template_blocks.append(block)

    templates_text = "\n\n".join(template_blocks)

    return f"""You are an expert superforecaster analyzing a FreeCiv game simulation. For each metric below, you will:
1. Generate plausible scenarios
2. Consider how likely each scenario is
3. Provide a SINGLE set of percentile forecasts (p10/p25/p50/p75/p90) per question that integrates across your scenarios

The scenarios should reflect your analysis of the game state. Think carefully about what could happen — growth, conflict, stagnation, collapse — and combine them into your final percentile forecast.

## About This Game

{FREECIV_DOMAIN_KNOWLEDGE}## World Report
{world_report}

## Task

{templates_text}

You may reason before the output block. You MUST end your response with a block in this exact format:

<<<PERCENTILES>>>
TEMPLATE: cities count
SCENARIOS: A=Steady growth, B=War disruption, C=Stagnation
Q1: p10=18, p25=24, p50=30, p75=34, p90=38
Q2: p10=15, p25=26, p50=35, p75=45, p90=52
...

TEMPLATE: population
SCENARIOS: A=Continued expansion, B=Collapse
Q7: p10=12, p25=28, p50=55, p75=80, p90=100
...
<<<END>>>

Rules:
- Exactly ONE set of percentiles per question (no per-scenario forecasts, no weights)
- Use 2-3 scenarios per template (labeled A, B, C) — these inform your reasoning
- Your p10/p25/p50/p75/p90 must reflect the FULL MIXTURE of scenarios weighted by their likelihood. If you think there is a 30% chance of collapse to values near 10, your p10 should reflect that lower-tail mass.
- All percentile values are the actual predicted quantity (not percentages)
- Provide one line per question, {q_num} questions total
- Percentiles must be non-decreasing within each question: p10 <= p25 <= p50 <= p75 <= p90"""
