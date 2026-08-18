"""Shared pieces of the single city eval: response cache, dataset, scoring.

The eval is split across three scripts — run_single_city_eval.py gathers model
responses, analyze_single_city.py scores them, plot_forecasts.py draws them —
and this module holds what more than one of them needs. The dataset written by
the first is the only thing the other two read, so they never re-simulate or
re-prompt.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from fbsim_core.metrics import compute_crps

from . import module_globals as g
from .city_sim import CitySimulation
from .config import Config, scenarios_from

# Everything the eval writes. Raw prompts and model responses are cached in one
# directory per batch (see batch_dir), so they survive across runs and configs;
# data.json is regenerated from them on every run.
OUT_DIR = g.DATA_DIR / "single_city"
DATA_PATH = OUT_DIR / "data.json"
PLOTS_PATH = OUT_DIR / "plots"

# Metrics left out of normalized CRPS. Normalizing by |actual| is undefined
# where the actual is 0, and city funds legitimately sits at 0 for long
# stretches — a bankrupt city stays broke — so the whole metric is excluded
# rather than dropping the individual questions and averaging over a
# silently different question set per scenario.
UNNORMALIZED_METRICS = {"totalFunds"}


@dataclass(frozen=True)
class ResponseId:
    model_id: str
    question_id: str


@dataclass(frozen=True)
class Response:
    actual: float
    percentiles: dict[str, float] | None
    response_text: str | None = None


Responses = dict[ResponseId, Response]


def batch_id_for(question: dict) -> str:
    """The batch a corpus question is prompted in.

    Questions sharing a scenario and snapshot turn share a game report — the
    bulk of the prompt — so they are asked together in one numbered prompt.
    """
    return f"{question['scenario_id']}_T{question['snapshot_turn']}"


def batch_dir(batch_id: str) -> Path:
    """Where a batch's prompt and raw model responses are cached.

    Holds prompt.txt plus one response-{model}.txt per model that has answered
    it (see prompt_path and response_path). Deleting the directory re-gathers
    the batch from scratch on the next run.
    """
    return OUT_DIR / "cache" / batch_id


def prompt_path(batch_id: str) -> Path:
    return batch_dir(batch_id) / "prompt.txt"


def response_path(batch_id: str, model_id: str) -> Path:
    # Model ids are provider/name; the slash would nest a directory.
    return batch_dir(batch_id) / f"response-{model_id.replace('/', '_')}.txt"


def save_dataset(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    path: Path = DATA_PATH,
) -> Path:
    """Write the corpus and this run's forecasts as one self-contained file.

    The analysis and plotting scripts read only this, so it carries everything
    they need: the questions with their resolved values, and each model's
    percentiles per question. The full response text is deliberately left out —
    it stays in the response cache, and including it here would multiply the
    file size for something no downstream script reads.
    """
    # Prompts are megabytes of world report repeated per question; the
    # downstream scripts want the question, not the prompt that produced it.
    dropped = {"context"}
    questions = [{k: v for k, v in c.items() if k not in dropped} for c in corpus]

    forecasts = [
        {
            "model_id": model_id,
            "question_id": c["question_id"],
            "percentiles": r.percentiles,
        }
        for c in corpus
        for model_id in model_names
        for r in [responses.get(ResponseId(model_id, c["question_id"]))]
        if r is not None
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"models": model_names, "questions": questions, "forecasts": forecasts},
            indent=2,
        )
    )
    return path


def load_dataset(path: Path = DATA_PATH) -> tuple[list[dict], Responses, list[str]]:
    """Read back what save_dataset wrote, as (corpus, responses, model_names).

    Returns the same shapes the gathering script works with, so the analysis and
    plotting code is identical whether it was handed live results or a file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/run_single_city_eval.py first"
        )
    data = json.loads(path.read_text())
    corpus = data["questions"]
    actual = {c["question_id"]: c["value"] for c in corpus}
    responses: Responses = {
        ResponseId(f["model_id"], f["question_id"]): Response(
            actual=actual[f["question_id"]],
            percentiles=f["percentiles"],
        )
        for f in data["forecasts"]
    }
    return corpus, responses, data["models"]


def scenario_history(scenario_id: str, seed: int) -> list[dict] | None:
    """The cached log rows of a scenario run, or None if it isn't on disk.

    Indexed by turn: row i is turn i, matching how build_corpus resolves a
    question at snapshot_turn + horizon. Reads the cache only — never simulates —
    so the analysis scripts stay offline and cost nothing.

    The scenario id encodes city, disasters and seed, which is what names the log
    file, so a run can be found from a corpus entry alone. `seed` is taken as an
    argument rather than parsed back out of the id: the id is built by
    CitySimulation and this should not depend on how it is spelled.
    """
    city, _, rest = scenario_id.partition("_")
    if not rest:
        return None
    sim = CitySimulation(
        city_name=city, seed=seed, disasters=rest.startswith("disasters")
    )
    try:
        sim.load_from_disk()
    except FileNotFoundError:
        return None
    return sim.log_data


class DatasetError(Exception):
    """The dataset on disk doesn't cover what the config asked for."""


def scenario_ids_from(cfg: Config, seed: int) -> list[str]:
    """The scenario ids a config's cities x disasters cross product names.

    Built through CitySimulation so the ids match the ones build_corpus wrote;
    constructing one runs nothing, it only holds the parameters.
    """
    return [
        CitySimulation(city_name=city, seed=seed, disasters=disasters).get_id_str()
        for city, disasters in scenarios_from(cfg)
    ]


def select_for_config(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    cfg: Config,
    seed: int,
) -> tuple[list[dict], Responses, list[str]]:
    """Narrow a dataset to what `cfg` asks for, or fail saying what is missing.

    Lets one gathered dataset serve many views — a subset of models, cities or
    horizons — without re-prompting. Anything the config names that the dataset
    lacks is an error rather than a silently smaller table, since a missing
    model or city would otherwise look like a legitimately empty result.
    """
    wanted_models = cfg.get_str_list("models")
    wanted_scenarios = scenario_ids_from(cfg, seed)
    wanted_snapshots = cfg.get_int_list("snapshot_turns")
    wanted_horizons = cfg.get_int_list("horizons")

    missing = []
    for name, wanted, present in [
        ("models", wanted_models, set(model_names)),
        ("cities/disasters", wanted_scenarios, {c["scenario_id"] for c in corpus}),
        ("snapshot_turns", wanted_snapshots, {c["snapshot_turn"] for c in corpus}),
        ("horizons", wanted_horizons, {c["horizon"] for c in corpus}),
    ]:
        absent = [w for w in wanted if w not in present]
        if absent:
            missing.append(
                f"  {name}: {', '.join(str(a) for a in absent)}\n"
                f"    dataset has: {', '.join(str(p) for p in sorted(present, key=str))}"
            )
    if missing:
        raise DatasetError(
            "the dataset does not cover this config:\n"
            + "\n".join(missing)
            + "\n  re-run scripts/run_single_city_eval.py with this config to gather it"
        )

    selected_scenarios = set(wanted_scenarios)
    selected_snapshots = set(wanted_snapshots)
    selected_horizons = set(wanted_horizons)
    selected_corpus = [
        c
        for c in corpus
        if c["scenario_id"] in selected_scenarios
        and c["snapshot_turn"] in selected_snapshots
        and c["horizon"] in selected_horizons
    ]

    # Every selected question needs a row for every selected model. A model that
    # answered unusably still has a row, with null percentiles, so a genuinely
    # absent row means that pair was never gathered.
    ungathered = [
        (model_id, c["question_id"])
        for c in selected_corpus
        for model_id in wanted_models
        if ResponseId(model_id, c["question_id"]) not in responses
    ]
    if ungathered:
        shown = ", ".join(f"{m} / {q}" for m, q in ungathered[:3])
        more = f" (+{len(ungathered) - 3} more)" if len(ungathered) > 3 else ""
        raise DatasetError(
            f"the dataset is missing {len(ungathered)} forecast(s) the config asks "
            f"for: {shown}{more}\n"
            "  re-run scripts/run_single_city_eval.py with this config to gather them"
        )

    selected_responses = {
        ResponseId(model_id, c["question_id"]): responses[
            ResponseId(model_id, c["question_id"])
        ]
        for c in selected_corpus
        for model_id in wanted_models
    }
    return selected_corpus, selected_responses, wanted_models


def score_forecasts(
    corpus: list[dict], responses: Responses, model_names: list[str]
) -> list[dict]:
    """Score every parsed forecast, raw and normalized.

    One row per (model, question) that produced a usable forecast, carrying the
    metric and horizon so callers can group as they like. "normalized" is CRPS
    over |actual|, and is None where that is undefined or the metric is
    excluded, so a caller averaging it must skip the Nones.
    """
    rows = []
    for c in corpus:
        for model_id in model_names:
            r = responses.get(ResponseId(model_id, c["question_id"]))
            if r is None or r.percentiles is None:
                continue
            crps = compute_crps(r.percentiles, c["value"])
            actual = abs(c["value"])
            # Guard on the value rather than trusting the metric to be
            # non-zero: which metrics can hit 0 depends on the cities in the
            # config, and a division by zero here would be silent.
            normalizable = c["metric"] not in UNNORMALIZED_METRICS and actual != 0
            rows.append(
                {
                    "model_id": model_id,
                    "metric": c["metric"],
                    "horizon": c["horizon"],
                    "crps": crps,
                    "normalized": crps / actual if normalizable else None,
                }
            )
    return rows
