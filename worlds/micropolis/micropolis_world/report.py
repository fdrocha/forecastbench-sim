"""Model-facing world report (situation report) + prompt builder."""

from collections import Counter

from .city_sim import METRICS, CitySimulation, turn_of

# cityClass as reported by the engine's evaluation pass, indexed 0..5.
CITY_CLASSES = ["Village", "Town", "City", "Capital", "Metropolis", "Megalopolis"]

# sendMessage messageNum values that mean a disaster actually struck, as opposed
# to a standing advisory like "Pollution very high". See the engine's text.h for
# the enum and message.cpp's sound-effect switch for the disaster subset.
DISASTER_MESSAGES = {
    20: "Fire",
    21: "Monster",
    22: "Tornado",
    23: "Earthquake",
    24: "Plane crash",
    25: "Shipwreck",
    26: "Train crash",
    27: "Helicopter crash",
    30: "Firebombing",
    32: "Explosion",
    42: "Flooding",
    43: "Nuclear meltdown",
    44: "Riots",
}

# Report labels for the city_sim.METRICS keys.
METRIC_LABELS = {
    "cityScore": "Score",
    "cityPop": "Population",
    "totalFunds": "Funds ($)",
    "trafficAverage": "Traffic",
    "pollutionAverage": "Pollution",
    "crimeAverage": "Crime",
    "landValueAverage": "Land value",
}

# Extra log fields worth showing in the snapshot, beyond METRICS.
SNAPSHOT_COMPOSITION = [
    ("resPop", "Residential"),
    ("comPop", "Commercial"),
    ("indPop", "Industrial"),
]
SNAPSHOT_INFRASTRUCTURE = [
    ("roadTotal", "Roads"),
    ("railTotal", "Rail"),
    ("policeStationPop", "Police stations"),
    ("fireStationPop", "Fire stations"),
    ("seaportPop", "Seaports"),
    ("airportPop", "Airports"),
    ("coalPowerPop", "Coal plants"),
    ("nuclearPowerPop", "Nuclear plants"),
    ("poweredZoneCount", "Powered zones"),
    ("unpoweredZoneCount", "Unpowered zones"),
]


def _snapshot_section(row: dict) -> list[str]:
    """Current-state block from a single log_data row."""
    lines = [f"CURRENT STATE (turn {turn_of(row)})", ""]
    city_class = row.get("cityClass")
    if city_class is not None and 0 <= city_class < len(CITY_CLASSES):
        lines.append(f"  City class: {CITY_CLASSES[city_class]}")
    for metric in METRICS:
        label = METRIC_LABELS.get(metric, metric)
        lines.append(f"  {label}: {row[metric]}")
    lines.append(f"  Cash flow: {row['cashFlow']}")
    lines.append(f"  Tax rate: {row['cityTax']}%")
    lines.append("")
    # Skipping this for now, it has weird units and is hard to interpret
    #    lines.append("  Zone population — " + ", ".join(
    #        f"{label} {row[key]}" for key, label in SNAPSHOT_COMPOSITION))
    lines.append(
        "  Infrastructure — "
        + ", ".join(f"{label} {row[key]}" for key, label in SNAPSHOT_INFRASTRUCTURE)
    )
    return lines


def _history_section(log_data: list[dict], turn: int, history_freq: int) -> list[str]:
    """Metric history sampled every history_freq turns, ending on the snapshot turn."""
    turns = sorted(set(range(turn, -1, -history_freq)))
    header = ["Turn"] + [METRIC_LABELS.get(m, m) for m in METRICS]
    rows = [[str(t)] + [str(log_data[t][m]) for m in METRICS] for t in turns]
    return [f"HISTORY (every {history_freq} turns)", "", ",".join(header)] + [
        ",".join(row) for row in rows
    ]


def _events_section(events_data: list[dict], cutoff_tick: int) -> list[str]:
    """Disasters (individually, with dates) and recurring advisories (aggregated)."""
    messages = [
        e
        for e in events_data
        if e.get("event") == "sendMessage" and e["tick"] <= cutoff_tick
    ]
    disasters = [e for e in messages if e["messageNum"] in DISASTER_MESSAGES]
    advisories = [e for e in messages if e["messageNum"] not in DISASTER_MESSAGES]

    lines = ["EVENTS", ""]

    if disasters:
        lines.append(f"  Disasters ({len(disasters)} total):")
        # One incident often fires several messages on the same turn (a plane
        # crash triggers a helicopter response plus explosions), so collapse
        # repeats of the same message within a turn into a count.
        counts = Counter((turn_of(e), e["messageText"]) for e in disasters)
        seen = set()
        for e in disasters:
            key = (turn_of(e), e["messageText"])
            if key in seen:
                continue
            seen.add(key)
            suffix = f" (x{counts[key]})" if counts[key] > 1 else ""
            lines.append(f"    turn {key[0]}: {key[1]}{suffix}")
    else:
        lines.append("  Disasters: none reported.")

    if advisories:
        lines.append("")
        lines.append("  Standing advisories (first and last reported):")
        counts = Counter(e["messageText"] for e in advisories)
        first = {}
        last = {}
        for e in advisories:
            first.setdefault(e["messageText"], turn_of(e))
            last[e["messageText"]] = turn_of(e)
        for text, n in counts.most_common():
            span = (
                f"turn {first[text]}"
                if first[text] == last[text]
                else f"turns {first[text]}–{last[text]}"
            )
            lines.append(f"    {text} (x{n}, {span})")

    return lines


def gen_world_report(sim: CitySimulation, turn: int, history_freq: int) -> str:
    """Build the model-facing situation report for `sim` as of `turn`.

    Args:
        sim: A CitySimulation with load_from_disk() already called.
        turn: Index into sim.log_data for the snapshot. Everything after it
            (later log rows, later events) is excluded to avoid leakage.
        history_freq: Sample the metric history every this many turns.
    """
    if sim.log_data is None or sim.events_data is None:
        raise ValueError("Simulation data not loaded; call sim.load_from_disk() first")
    if not 0 <= turn < len(sim.log_data):
        raise IndexError(
            f"turn {turn} out of range for {len(sim.log_data)} logged turns"
        )
    if history_freq < 1:
        raise ValueError(f"history_freq must be >= 1, got {history_freq}")

    row = sim.log_data[turn]
    sections = [
        [
            "MICROPOLIS WORLD REPORT",
            f"City: {sim.city_name} (seed {sim.seed})",
            f"Disasters: {'enabled' if sim.disasters else 'disabled'}",
            f"Snapshot: turn {turn} of {len(sim.log_data) - 1}",
        ],
        _snapshot_section(row),
        _history_section(sim.log_data, turn, history_freq),
        _events_section(sim.events_data, row["tick"]),
    ]
    return "\n\n".join("\n".join(section) for section in sections)


def build_prompt(context: str, question_text: str) -> str:
    return (
        context
        + "\n\n"
        + f"QUESTION: {question_text}\n\n"
        + "Give your probability that the answer is YES, as a single number "
        + "between 0 and 1. Respond with ONLY the number (e.g. 0.73)."
    )
