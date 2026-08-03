"""Model-facing world report (situation report) + prompt builder."""

# TODO: replace with a real constant from runner.py once the sim is implemented
# (e.g. starting population, city size). Kept here only as a report footnote.
N_REGIONS = 2


def context_blurb(world: dict, region_ids: list[int], names: dict[int, str],
                  snapshot_turn: int, intervention: str | None) -> str:
    """Per-region population/funds trajectory up to the snapshot turn.

    TODO: adjust metric names/formatting once runner.METRICS is finalized.
    """
    ts = world["time_series"]
    lines = [f"MICROPOLIS SITUATION REPORT — Turn {snapshot_turn}", ""]
    for rid in region_ids:
        pop = [ts["population"][str(t)][str(rid)] for t in range(0, snapshot_turn + 1, 5)]
        funds = ts["funds"][str(snapshot_turn)][str(rid)]
        pollution = ts["pollution"][str(snapshot_turn)][str(rid)]
        traj = " -> ".join(f"t{t}:{int(p)}" for t, p in zip(range(0, snapshot_turn + 1, 5), pop))
        lines.append(f"{names[rid]}: population {traj}")
        lines.append(f"  (turn {snapshot_turn}: funds {int(funds)}, pollution {int(pollution)})")
    lines.append(f"\nRegions are isolated (no inter-region migration or trade).")
    if intervention:
        lines.append(f"\nPLANNED INTERVENTION: {intervention}")
    return "\n".join(lines)


def build_prompt(context: str, question_text: str) -> str:
    return (
        context
        + "\n\n"
        + f"QUESTION: {question_text}\n\n"
        + "Give your probability that the answer is YES, as a single number "
        + "between 0 and 1. Respond with ONLY the number (e.g. 0.73)."
    )
