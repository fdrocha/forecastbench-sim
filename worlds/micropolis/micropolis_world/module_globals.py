"""Defines some global variables used throughout the Micropolis world."""

import os
from pathlib import Path

from dotenv import load_dotenv

PKG_DIR = Path(__file__).resolve().parent.parent  # forecastbench-sim/worlds/micropolis
FBS_DIR = PKG_DIR.parent.parent  # forecastbench-sim
DATA_DIR = FBS_DIR / "data" / "micropolis"
# Per-run simulation output (log/events/report/plot files), one directory per
# city. The engine's run_sim.js appends the city name to the base dir it's
# given, so this is passed to it as --output-base-dir verbatim.
RUNS_DIR = DATA_DIR / "runs"

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


def prompt_model(model, prompt: str, max_tokens: int) -> tuple[str | None, str | None]:
    """Send `prompt` to `model`, returning its text and the finish reason.

    LiteLLMModel.get_response() returns only the text, but the finish reason is
    what explains an empty reply: a reasoning model can spend the whole token
    budget thinking and stop at "length" with nothing written, which is a
    successful call the caller would otherwise see as a silent blank.

    Mirrors get_response()'s handling of the parameters some models reject.
    """
    # Imported here so the simulation-only scripts don't pull in litellm.
    import litellm
    from litellm import completion

    # We need this so we can use models that don't support temperature
    # It still sets T=0 for models that do support it, but drops it silently for ones that don't
    litellm.drop_params = True

    kwargs = {
        "model": model._litellm_model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }
    if model.supports_temperature:
        kwargs["temperature"] = 0.0
    else:
        # Left at the provider default, so this model's answers are sampled
        # rather than greedy and will vary between runs.
        print(f"  [warning] {model.id} does not support temperature; omitting it")

    choice = completion(**kwargs).choices[0]
    return choice.message.content, choice.finish_reason


def warn_if_truncated(
    model_id: str, finish_reason: str | None, max_tokens: int
) -> None:
    """Warn when a reply stopped because it ran out of tokens.

    Worth saying explicitly: for a reasoning model the cap covers thinking as
    well as the answer, so the reply can come back empty rather than merely cut
    short, which looks like an unparseable answer instead of a budget problem.
    """
    if finish_reason == "length":
        print(
            f"  [warning] {model_id} hit the {max_tokens}-token cap before "
            "finishing. Raise max_tokens; for a reasoning model the cap "
            "covers thinking as well as the answer, so it can be spent "
            "before any answer is written."
        )


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
    ("hospitalPop", "Hospitals"),
    ("stadiumPop", "Stadiums"),
    ("coalPowerPop", "Coal plants"),
    ("nuclearPowerPop", "Nuclear plants"),
    ("poweredZoneCount", "Powered zones"),
    ("unpoweredZoneCount", "Unpowered zones"),
]

# Scores taken from https://epoch.ai/eci?subset-view=graph&view=graph&tab=leaderboard&subset-tab=Software%20engineering
# on 2026-08-17. Note that the "Software engineering" in the URL is for a table further down on the page
# and does not affect these numbers
ECI_MAP = {
    "gpt-4.1-2025-04-14": 137,
    "gpt-5-2025-08-07": 150,
    "gpt-5.6-luna": 156,
    "gpt-5.6-terra": 159,
    "gpt-5.6-sol": 162,
    "claude-haiku-4-5-20251001": 143,
    "claude-sonnet-4-5-20250929": 147,
    "claude-opus-4-5-20251101": 150,
    "claude-opus-4-6": 155,
    "gemini-2.5-pro": 146,  # Jun 2025
    "gemini-2.5-flash": 143,  # Sep 2025
    "gemini-3-pro-preview": 153,
    "gemini-3.1-pro-preview": 155,
    # "gemini-3.7-flash": 1,  # no score
    "grok-4.20-0309-reasoning": 152,  # there is only one grok-4.20 score, I believe it is for the reasoning model
    # "grok-4.20-0309-non-reasoning": 1, # no score
    # "grok-4.6": 1, # no score
    # "mistral-small-2603": 1, # I believe this corresponds to Mistral Small 4 which is not on the leaderboard
    "mistral-medium-2604": 143,  # 2604 corresponds to Medium 3.5 on the table
    # "mistral-large-2512": 121, # This is Mistral Large 3, not on the table
}
