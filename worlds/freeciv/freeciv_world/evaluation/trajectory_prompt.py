"""
Trajectory (per-time-series) question format for continuous forecasting.

Groups questions by template and presents all horizons together, so the model
sees each metric as a trajectory rather than independent estimation problems.
Response format stays flat (Q1, Q2, ... QN) for compatibility with
parse_batch_percentiles.
"""

from collections import defaultdict

# Human-readable names for template IDs
TEMPLATE_LABELS = {
    "population_continuous": "population",
    "territory_continuous": "territory (tiles controlled)",
    "treasury_continuous": "treasury (gold)",
    "cities_count_continuous": "number of cities",
    "techs_continuous": "technologies discovered",
    "scores_continuous": "score",
}


def group_questions_by_template(
    questions: list[dict],
) -> list[tuple[str, list[dict]]]:
    """Group questions by template_id, sorted by resolution_turn within each group.

    Args:
        questions: List of question dicts with 'template_id' and 'resolution_turn' keys.

    Returns:
        List of (template_id, sorted_questions) tuples.
        Template order matches first appearance in input.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    order: list[str] = []
    for q in questions:
        tmpl = q["template_id"]
        if tmpl not in groups:
            order.append(tmpl)
        groups[tmpl].append(q)

    return [
        (tmpl, sorted(groups[tmpl], key=lambda q: q["resolution_turn"]))
        for tmpl in order
    ]


def build_trajectory_continuous_prompt(
    questions: list[dict], world_report: str
) -> str:
    """Build a continuous forecast prompt with questions grouped by template.

    Questions are grouped by template_id and sorted by resolution_turn within
    each group. A trajectory header introduces each group. Questions are
    numbered sequentially across all groups (Q1-QN) so the flat response format
    used by parse_batch_percentiles works unchanged.

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

## World Report
{world_report}

## Questions
{questions_text}

{instruction}"""
