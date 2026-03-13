"""
Combined trajectory + domain knowledge prompt for continuous forecasting.

Merges the per-time-series question format (trajectory_prompt.py) with the
FreeCiv domain knowledge prefix (domain_knowledge_prompt.py). This is the
non-agentic equivalent of the agentic baseline's AGENTS.md prompt.
"""

from civrealm.evaluation.domain_knowledge_prompt import FREECIV_DOMAIN_KNOWLEDGE
from civrealm.evaluation.trajectory_prompt import group_questions_by_template, TEMPLATE_LABELS


def build_trajectory_domain_knowledge_continuous_prompt(
    questions: list[dict], world_report: str
) -> str:
    """Build a prompt combining trajectory grouping with domain knowledge.

    Uses trajectory grouping from trajectory_prompt.py and the FreeCiv
    domain knowledge prefix from domain_knowledge_prompt.py.

    Args:
        questions: List of question dicts with 'question_text', 'template_id',
                   and 'resolution_turn' keys.
        world_report: World report text.

    Returns:
        Formatted prompt string.
    """
    grouped = group_questions_by_template(questions)
    num_questions = len(questions)

    # Build questions section with trajectory grouping
    sections = []
    q_num = 1
    for tmpl, group in grouped:
        label = TEMPLATE_LABELS.get(tmpl, tmpl)
        start = q_num
        end = q_num + len(group) - 1

        lines = [f"Questions {start}-{end}: {label} across future turns."]
        for q in group:
            lines.append(f"{q_num}. {q['question_text']}")
            q_num += 1

        sections.append("\n".join(lines))

    questions_text = "\n\n".join(sections)

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
