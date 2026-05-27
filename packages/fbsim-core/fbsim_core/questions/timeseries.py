"""Explicit time_series layout.

The canonical layout is TURN-MAJOR:

    time_series[metric][turn][entity_id] = value

Every world serializes into this layout. Core does NOT infer layout from whether
keys happen to be integer-castable (the old `_get_signal_value` heuristic silently
misread entity-major data when both ids and turns were numeric). Keys are strings.
"""

from typing import Any


def signal_at(signal_series: dict[str, Any], entity, turn) -> Any | None:
    """Look up one value in a single metric's turn-major series."""
    turn_data = signal_series.get(str(turn))
    if not isinstance(turn_data, dict):
        return None
    return turn_data.get(str(entity))


def value_at(game_data: dict[str, Any], metric: str, entity, turn) -> Any | None:
    """Look up time_series[metric][turn][entity] from a full game_data dict."""
    return signal_at(game_data.get("time_series", {}).get(metric, {}), entity, turn)
