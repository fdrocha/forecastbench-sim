#!/usr/bin/env -S uv run python3
"""Score each model's domain knowledge of Micropolis.

Asks every configured model to classify the statement set as True/False/Unknown,
then prints a table of how many each model got correct, wrong, unknown and
unparseable. Responses are cached, so re-running only prompts models that have
not answered yet.

Usage:
    uv run python scripts/run_knowledge_eval.py
    uv run python scripts/run_knowledge_eval.py my_config.json5
    uv run python scripts/run_knowledge_eval.py --models openai/gpt-4o xai/grok-4-0709

--models overrides the config's list; see data/micropolis/available_models.md
for what each provider offers.
"""

import argparse

import micropolis_world.module_globals as g
from micropolis_world.config import (
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.knowledge_eval.runner import (
    Answer,
    OUT_DIR,
    get_model_answers,
    statements,
)

# Points per answer, from the SCORING block of the prompt preamble. An
# unparseable answer is scored as incorrect, as the preamble warns.
POINTS = {"correct": 1, "wrong": -2, "unknown": 0, "unparseable": -2}


def tally(answers: list[Answer]) -> dict[str, int]:
    """Bucket one model's answers against the statement key.

    A statement is correct when the model's verdict matches Statement.is_true,
    wrong when it is the opposite verdict, and otherwise falls into its own
    column.
    """
    counts = {"correct": 0, "wrong": 0, "unknown": 0, "unparseable": 0}
    for statement, answer in zip(statements, answers):
        if answer is Answer.UNKNOWN:
            counts["unknown"] += 1
        elif answer is Answer.UNPARSEABLE:
            counts["unparseable"] += 1
        elif (answer is Answer.TRUE) == statement.is_true:
            counts["correct"] += 1
        else:
            counts["wrong"] += 1
    return counts


def print_table(results: dict[str, dict[str, int]]) -> None:
    """Print the per-model score table, best score first."""
    if not results:
        print("No model answers to report.")
        return

    headers = ["Model", "Correct", "Wrong", "Unknown", "Unparseable", "Score"]
    rows = []
    for model, counts in sorted(
        results.items(), key=lambda kv: -sum(POINTS[k] * v for k, v in kv[1].items())
    ):
        score = sum(POINTS[k] * v for k, v in counts.items())
        rows.append(
            [
                model,
                str(counts["correct"]),
                str(counts["wrong"]),
                str(counts["unknown"]),
                str(counts["unparseable"]),
                str(score),
            ]
        )

    widths = [
        max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)
    ]
    # Model names left-aligned, counts right-aligned under their headers.
    def fmt(cells: list[str]) -> str:
        return "  ".join(
            c.ljust(widths[i]) if i == 0 else c.rjust(widths[i])
            for i, c in enumerate(cells)
        )

    print(fmt(headers))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        default=None,
        help="Model ids to prompt, overriding the config's 'models' list. "
        "Ids are in provider/name form; see data/micropolis/available_models.md",
    )
    args = ap.parse_args()

    cfg = load_config(args)
    models = args.models if args.models else cfg.get_str_list("models")
    max_tokens = cfg.get_int("max_tokens")

    print("=" * 70)
    print("MICROPOLIS WORLD — domain knowledge eval")
    print("=" * 70)
    print(f"config: {cfg.path}")
    print(f"{len(statements)} statements, {len(models)} model(s)\n")

    g.ensure_api_keys()
    answers = get_model_answers(models, max_tokens)

    print(f"\n{'=' * 70}")
    print("RESULTS (scoring: correct +1, wrong -2, unknown 0)")
    print("=" * 70)
    print_table({model: tally(a) for model, a in answers.items()})

    missing = [m for m in models if m not in answers]
    if missing:
        print(f"\nNo response from: {', '.join(missing)}")
    print(f"\nResponses saved under {OUT_DIR}")


if __name__ == "__main__":
    main()
