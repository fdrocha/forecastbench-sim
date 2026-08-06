"""Defines some global variables used throughout the Micropolis world."""

import os
from pathlib import Path

from dotenv import load_dotenv

PKG_DIR = Path(__file__).resolve().parent.parent  # forecastbench-sim/worlds/micropolis
FBS_DIR = PKG_DIR.parent.parent  # forecastbench-sim
DATA_DIR = FBS_DIR / "data" / "micropolis"

load_dotenv(PKG_DIR / ".env")
MICROPOLIS_APP_PATH = Path(os.environ["MICROPOLIS_CORE_PATH"]) / "apps" / "micropolis"

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

# Defaults for corpus generation and evaluation; override via the --cities,
# --snapshots and --horizons CLI flags in scripts/.
#
# A few chosen Micropolis cities. Not a lot of thought put into the selection:
# dropped scenarios and a few others.
DEFAULT_CITIES = [
    #    "bluebird", this is a dead city with no population, nothing happens
    "bruce",
    #    "deadwood", crashes the engine (WASM "memory access out of bounds")
    #    partway through a run when disasters are enabled, at every seed tried
    "finnigan",
    #    "freds",
    "haight",
    #    "happisle",
    #    "joffburg",
    #    "kamakura",
    #    #    "kobe",
    # "kowloon",
    # "kyoto",
    # "linecity",
    # "senri",
    # "southpac",
    # "splats",
    # "wetcity",
    # #    "yokohama",
]

DEFAULT_SNAPSHOT_TURNS = [TURNS_PER_YEAR * y for y in [1, 2, 3, 4, 5]]
DEFAULT_HORIZONS = [TURNS_PER_YEAR * y for y in [1, 5, 10]]

# The world report contains data every FREQ turns.
FREQ = 4

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
