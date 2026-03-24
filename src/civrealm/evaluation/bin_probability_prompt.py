"""
Bin probability elicitation format for continuous forecasting.

Instead of asking for quantiles (p10, p25, p50, p75, p90), asks forecasters to
assign probability mass (%) to discrete outcome bins summing to 100%. This is
the standard format used by the Survey of Professional Forecasters and Good
Judgment Inc.

Bins are corpus deciles: for each template, decile boundaries computed from
ground-truth outcomes pooled across all horizons (H1-H6) and all 1,019
generated worlds. Each bin captures ~10% of actual outcomes by construction.
"""

from collections import defaultdict

TEMPLATE_LABELS = {
    "population_continuous": "population",
    "territory_continuous": "territory (tiles controlled)",
    "treasury_continuous": "treasury (gold)",
    "cities_count_continuous": "number of cities",
    "techs_continuous": "technologies discovered",
    "scores_continuous": "score",
}

# Corpus decile bin edges (N=30,517 per template, all 1019 seeds × H1-H6).
# Each list has 11 values defining 10 bins.
BIN_EDGES = {
    "population_continuous": [0, 2, 10, 18, 26, 35, 44, 54, 67, 82, 375],
    "territory_continuous": [0, 15, 52, 82, 109, 137, 168, 206, 249, 312, 1142],
    "treasury_continuous": [0, 0, 98, 167, 236, 331, 469, 648, 891, 1280, 6578],
    "cities_count_continuous": [0, 1, 5, 8, 10, 13, 16, 20, 24, 29, 102],
}


def bin_labels_for_template(template_id: str) -> list[str]:
    """Return human-readable bin labels for a template.

    Returns:
        List of 10 strings like '0-2', '3-10', ..., '83+'.
    """
    edges = BIN_EDGES[template_id]
    labels = []
    for i in range(len(edges) - 1):
        lo = edges[i]
        hi = edges[i + 1]
        if i == 0 and lo == hi:
            # Degenerate bin (e.g., treasury "exactly 0")
            labels.append(f"{lo}")
        elif i == 0:
            labels.append(f"{lo}-{hi}")
        elif i == len(edges) - 2:
            # Last bin is open-ended
            labels.append(f"{lo + 1}+")
        else:
            labels.append(f"{lo + 1}-{hi}")
    return labels


def format_bin_line(labels: list[str]) -> str:
    """Format bin labels into a single response line template."""
    parts = [f"[{lbl}]=__%" for lbl in labels]
    return ", ".join(parts)


def group_questions_by_template(
    questions: list[dict],
) -> list[tuple[str, list[dict]]]:
    """Group questions by template_id, sorted by resolution_turn within each group."""
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


def build_bin_probability_prompt(
    questions: list[dict], world_report: str
) -> str:
    """Build a bin probability forecast prompt with questions grouped by template.

    Questions are grouped by template_id and sorted by resolution_turn within
    each group. For each question, the model assigns probability mass (%) to
    10 outcome bins. Questions are numbered sequentially (Q1-QN).

    Args:
        questions: List of question dicts with 'question_text', 'template_id',
                   and 'resolution_turn' keys.
        world_report: World report text.

    Returns:
        Formatted prompt string.
    """
    grouped = group_questions_by_template(questions)
    num_questions = len(questions)

    # Build questions section with trajectory grouping and bin format
    sections = []
    q_num = 1
    for tmpl, group in grouped:
        label = TEMPLATE_LABELS.get(tmpl, tmpl)
        labels = bin_labels_for_template(tmpl)
        bin_line = format_bin_line(labels)
        start = q_num
        end = q_num + len(group) - 1

        lines = [
            f"Questions {start}-{end}: {label} across future turns.",
            f"Bins for {label}: {', '.join(f'[{lbl}]' for lbl in labels)}",
        ]
        for q in group:
            lines.append(
                f"{q_num}. {q['question_text']}\n"
                f"   {bin_line}"
            )
            q_num += 1

        sections.append("\n".join(lines))

    questions_text = "\n\n".join(sections)

    # Build example using first template's bins
    first_tmpl = grouped[0][0]
    example_labels = bin_labels_for_template(first_tmpl)
    example_line = ", ".join(
        f"[{lbl}]={v}%" for lbl, v in zip(example_labels, [5, 10, 15, 20, 15, 10, 10, 5, 5, 5])
    )

    instruction = f"""You may analyze the data, but you MUST end your response with probability estimates in this exact format:
<<<BINS>>>
Q1: {example_line}
Q2: {example_line}
<<<END>>>

Provide one line of bin probability estimates per question for all {num_questions} questions, in order.
- Each value is the probability (%) that the true outcome falls in that range
- Values for each question MUST sum to 100%
- Use integer percentages (e.g., 15%, not 14.7%)"""

    return f"""You are an expert superforecaster, familiar with the work of Tetlock and others. You are analyzing a FreeCiv game simulation. Make predictions based on the world report below.

You MUST provide probability estimates for each question UNDER ALL CIRCUMSTANCES. If for some reason you can't answer, provide reasonable estimates that sum to 100%, but always return numeric probability values.

## World Report
{world_report}

## Questions
{questions_text}

{instruction}"""
