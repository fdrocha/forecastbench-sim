"""Micropolis city-sim runner + serializer into the fbsim-core turn-major schema.

TODO: plug in the actual city-simulation engine (zoning, population growth,
traffic, pollution, budget, etc). Each region/city is simulated independently
(no inter-region interaction) and produces one time series per metric, one
value per simulated turn.
"""

import numpy as np

N_REGIONS_DEFAULT = 2
N_TURNS = 90
METRICS = ["population", "funds", "pollution"]


def run_region(growth_rate: float, policy_enabled: bool, seed: int, **policy_kwargs) -> dict:
    """Run one city/region's simulation; return per-turn metric arrays.

    Args:
        growth_rate: TODO — placeholder scenario parameter analogous to pandemic's beta.
        policy_enabled: TODO — whether the intervention (e.g. zoning/tax policy) is active.
        seed: RNG seed for reproducibility.
        **policy_kwargs: TODO — intervention-specific parameters (e.g. policy_turn, magnitude).

    Returns:
        dict mapping metric name -> list of per-turn values, length N_TURNS.
    """
    raise NotImplementedError("TODO: implement Micropolis city simulation")


def to_world(region_series: dict[int, dict], names: dict[int, str]) -> dict:
    """Assemble game_data in the core TURN-MAJOR schema:
    time_series[metric][turn][region_id] = value.
    """
    ts = {m: {} for m in METRICS}
    for rid, series in region_series.items():
        for m in METRICS:
            for turn, val in enumerate(series[m]):
                ts[m].setdefault(str(turn), {})[str(rid)] = val
    return {
        "time_series": ts,
        "civilizations": {str(rid): {"name": names[rid], "nation_id": str(rid)}
                          for rid in region_series},
    }
