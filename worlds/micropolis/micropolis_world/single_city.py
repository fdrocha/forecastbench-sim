"""Shared pieces of the single city eval: response cache, dataset, scoring.

The eval is split across three scripts — run_single_city_eval.py gathers model
responses, analyze_single_city.py scores them, plot_forecasts.py draws them —
and this module holds what more than one of them needs. The dataset written by
the first is the only thing the other two read, so they never re-simulate or
re-prompt.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from fbsim_core.metrics import compute_crps

from . import module_globals as g
from .scenarios import PERCENTILE_KEYS, parse_percentiles

# Everything the eval writes. The response cache is keyed by (model, question)
# and survives across corpora, so it lives beside the dataset rather than in it.
OUT_DIR = g.DATA_DIR / "single_city"
DATA_PATH = OUT_DIR / "data.json"
CACHE_PATH = g.DATA_DIR / "response_cache.json"
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


def load_cache(verbose_reparse: bool = False) -> Responses:
    """Load cached responses, re-parsing the percentiles from the raw text.

    response_text is the source of truth, as it is in the knowledge eval: the
    stored percentiles are a convenience, so an improved parse_percentiles takes
    effect on the next run instead of needing the whole cache re-queried. An
    entry that has no response_text — nothing left to re-parse — keeps whatever
    percentiles it was stored with.

    Re-parsing is quiet by default: a response that was rejected when first
    fetched would otherwise reprint its warning on every subsequent run, and
    callers report the total instead. Set verbose_reparse to get the full
    per-question warning back, which is what you want when investigating why a
    particular cached response yields no forecast.
    """
    r: Responses = {}
    data = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else []
    for entry in data:
        stored = entry.get("percentiles")
        stored = (
            {k: float(stored[k]) for k in PERCENTILE_KEYS}
            if stored is not None
            else None
        )
        raw = entry.get("response_text", None)
        response_id = ResponseId(
            model_id=entry["model_id"], question_id=entry["question_id"]
        )
        # The same question_id appears once per model, so name both.
        label = f"{response_id.model_id} {response_id.question_id}"
        percentiles = (
            parse_percentiles(raw, label=label, quiet=not verbose_reparse)
            if raw is not None
            else stored
        )

        r[response_id] = Response(
            actual=entry["actual"],
            percentiles=percentiles,
            response_text=raw,
        )
    return r


def save_cache(cache: Responses) -> None:
    data = [asdict(k) | asdict(v) for k, v in cache.items()]
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=2))


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
