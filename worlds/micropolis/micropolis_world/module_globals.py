"""Defines some global variables used throughout the Micropolis world."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv

from . import messages as msg
from .usage import LLMResponse, usage_from_response

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

    OPENROUTER_API_KEY is not among the secrets fetched, so when the OpenRouter
    backend is live it has to come from the environment or .env. Checked here
    rather than left to the first call, which would fail per-request as an
    opaque 401 after the run had already started.
    """
    global _keys_loaded
    if _keys_loaded:
        return
    # Imported here so the simulation-only scripts don't pull in an LLM client.
    from fbsim_core.evaluation.models import load_api_keys_from_gcp

    from .llm_backend import REQUIRED_ENV_KEYS

    load_api_keys_from_gcp()
    missing = [k for k in REQUIRED_ENV_KEYS if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"{', '.join(missing)} not set — the active LLM backend needs it. "
            f"Put it in {PKG_DIR / '.env'} or export it."
        )
    _keys_loaded = True


def prompt_model(model, prompt: str) -> LLMResponse:
    """Send `prompt` to `model`, returning its text, finish reason and cost.

    LiteLLMModel.get_response() returns only the text, discarding the two
    things needed to explain and price an empty reply: the finish reason, and
    the tokens the provider charged for producing nothing. A reasoning model
    can spend the whole budget thinking and stop at "length" with nothing
    written — a successful, billed call the caller would otherwise see as a
    silent blank. So the call is made here instead, and the whole response is
    read before it is dropped.

    Sampling parameters and the output cap are left unset so every model runs
    on its provider defaults; the cap is a per-endpoint limit and belongs to
    the backend's model registry.
    """
    # Imported here so the simulation-only scripts don't pull in an LLM client.
    from .llm_backend import completion

    kwargs = {
        "model": model.id,  # the slug; the backend maps it to its own id
        "messages": [{"role": "user", "content": prompt}],
    }

    start = time.perf_counter()
    response = completion(**kwargs)
    latency_ms = (time.perf_counter() - start) * 1000

    choice = response.choices[0]  # type: ignore
    return LLMResponse(
        text=choice.message.content,
        finish_reason=choice.finish_reason,
        usage=usage_from_response(response, model.id, latency_ms),
    )


def warn_if_truncated(model_id: str, finish_reason: str | None) -> None:
    """Warn when a reply stopped because it ran out of tokens.

    Worth saying explicitly: for a reasoning model the cap covers thinking as
    well as the answer, so the reply can come back empty rather than merely cut
    short, which looks like an unparseable answer instead of a budget problem.
    The cap itself is the backend's — see the model's entry in
    model_specs.json5, or the provider default when it has none.
    """
    if finish_reason == "length":
        msg.warn(
            f"{model_id} hit its output-token cap before finishing. Raise "
            "max_tokens in the model's model_specs.json5 entry; for a "
            "reasoning model the cap covers thinking as well as the answer, "
            "so it can be spent before any answer is written."
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

# The one METRICS entry that is money rather than a behavioral reading, named
# so the report's censorCityFunds variant can drop it by name.
FUNDS_METRIC = "totalFunds"

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

# The row["census"] tile counts shown when a report asks for the census section
# (report_census=True) — the fields the binary tile-count questions resolve on.
SNAPSHOT_CENSUS = [
    ("rubble", "rubble tiles"),
    ("fire", "tiles on fire"),
    ("road", "road tiles"),
]
