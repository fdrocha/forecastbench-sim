"""Defines some global variables used throughout the Micropolis world."""

import os
from pathlib import Path

from dotenv import load_dotenv

PKG_DIR = Path(__file__).resolve().parent.parent  # forecastbench-sim/worlds/micropolis
FBS_DIR = PKG_DIR.parent.parent  # forecastbench-sim
DATA_DIR = FBS_DIR / "data" / "micropolis"

load_dotenv(PKG_DIR / ".env")
MICROPOLIS_APP_PATH = Path(os.environ["MICROPOLIS_CORE_PATH"]) / "apps" / "micropolis"

_keys_loaded = False


def ensure_api_keys() -> None:
    """Fill in any unset provider API keys from GCP Secret Manager.

    Call this before prompting models. Keys already set — by the shell or by
    .env — take precedence and are left alone, so only the providers missing
    locally are fetched. Requires GOOGLE_CLOUD_PROJECT and gcloud application
    default credentials; see docs/notes/evaluation_setup.md.

    Loading is lazy and done once per process: it makes a network call per
    missing key, and the scripts that never prompt a model shouldn't pay for it.
    """
    global _keys_loaded
    if _keys_loaded:
        return
    # Imported here so the simulation-only scripts don't pull in litellm.
    from fbsim_core.evaluation.models import load_api_keys_from_gcp

    load_api_keys_from_gcp()
    _keys_loaded = True

CITY_CHOICES = [
    "about",
    "badnews",
    "bluebird",
    "bruce",
    "deadwood",
    "finnigan",
    "freds",
    "haight",
    "happisle",
    "joffburg",
    "kamakura",
    "kobe",
    "kowloon",
    "kyoto",
    "linecity",
    "med_isle",
    "ndulls",
    "neatmap",
    "radial",
    "scenario_bern",
    "scenario_boston",
    "scenario_detroit",
    "scenario_dullsville",
    "scenario_hamburg",
    "scenario_rio_de_janeiro",
    "scenario_san_francisco",
    "scenario_tokyo",
    "senri",
    "southpac",
    "splats",
    "wetcity",
    "yokohama",
]

METRICS = [
    "cityScore",
    "cityPop",
    "totalFunds",
    "trafficAverage",
    "pollutionAverage",
    "crimeAverage",
    "landValueAverage",
]

# The engine logs one row every 16 ticks, so a row's turn is its tick // 16,
# which equals the row's index in log_data. Events carry raw ticks only, so
# this is also how an event is placed on the same turn axis.
TICKS_PER_TURN = 16

TURNS_PER_YEAR = 4 * 12  # 4 ticks per month, 12 months per year

# Cities, snapshot turns, horizons and the rest of the per-run parameters now
# live in the JSON config files under micropolis_world/configs/ — see the README
# there. Notes on the city selection, for when you edit a config's "cities":
#   - "bluebird" is a dead city with no population; nothing happens.
#   - "deadwood" crashes the engine (WASM "memory access out of bounds")
#     partway through a run when disasters are enabled, at every seed tried.

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
    "cityScore": "city score",
    "cityPop": "population",
    "totalFunds": "city funds",
    "trafficAverage": "average traffic",
    "pollutionAverage": "average pollution",
    "crimeAverage": "average crime",
    "landValueAverage": "average land value",
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
