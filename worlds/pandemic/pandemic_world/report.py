"""Model-facing world report (situation report) + prompt builder."""

from .runner import N_AGENTS


def context_blurb(world: dict, region_ids: list[int], names: dict[int, str],
                  snapshot_day: int, intervention: str | None) -> str:
    """Per-region cases/deaths trajectory up to the snapshot day."""
    ts = world["time_series"]
    lines = [f"PANDEMIC SITUATION REPORT — Day {snapshot_day}", ""]
    for rid in region_ids:
        cases = [ts["cumulative_cases"][str(d)][str(rid)] for d in range(0, snapshot_day + 1, 5)]
        active = ts["active_infections"][str(snapshot_day)][str(rid)]
        deaths = ts["cumulative_deaths"][str(snapshot_day)][str(rid)]
        traj = " -> ".join(f"d{d}:{int(c)}" for d, c in zip(range(0, snapshot_day + 1, 5), cases))
        lines.append(f"{names[rid]}: cumulative cases {traj}")
        lines.append(f"  (day {snapshot_day}: {int(active)} active, {int(deaths)} deaths)")
    lines.append(f"\nPopulation per region: {N_AGENTS}. Regions are isolated (no inter-region travel).")
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
