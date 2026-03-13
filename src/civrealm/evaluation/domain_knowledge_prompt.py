"""
Domain knowledge prefix for FreeCiv forecasting prompts.

Explains what FreeCiv is and what the report columns mean, without including
crash rates or forecasting guidance (which cause narrative anchoring).

Used by both LLM interventions and human superforecaster briefings.
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


def build_domain_knowledge_continuous_prompt(
    questions: list[dict], world_report: str
) -> str:
    """Build a continuous forecast prompt with domain knowledge prefix.

    Uses the standard interleaved question format (baseline) but prepends
    the FreeCiv domain knowledge explanation.

    Args:
        questions: List of question dicts with 'question_text' key.
        world_report: World report text.

    Returns:
        Formatted prompt string.
    """
    num_questions = len(questions)
    questions_text = "\n".join(
        f"{i+1}. {q['question_text']}" for i, q in enumerate(questions)
    )

    instruction = f"""You may analyze the data, but you MUST end your response with percentile estimates in this exact format:
<<<PERCENTILES>>>
Q1: p10=5, p25=10, p50=15, p75=20, p90=25
Q2: p10=100, p25=200, p50=300, p75=400, p90=500
<<<END>>>

Provide one line of percentile estimates per question for all {num_questions} questions, in order.
- p10 means you estimate there's a 10% chance the true value is below this number
- p25 means you estimate there's a 25% chance the true value is below this number
- p50 (median) means you estimate there's a 50% chance the true value is below this number
- p75 means you estimate there's a 75% chance the true value is below this number
- p90 means you estimate there's a 90% chance the true value is below this number"""

    return f"""You are an expert superforecaster, familiar with the work of Tetlock and others. You are analyzing a FreeCiv game simulation. Make predictions based on the world report below.

You MUST provide percentile estimates for each question UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, provide reasonable mid-range estimates, but always return numeric percentile values.

## About This Game

{FREECIV_DOMAIN_KNOWLEDGE}## World Report
{world_report}

## Questions
{questions_text}

{instruction}"""
