#!/usr/bin/env -S uv run python3
"""Single city scenario evaluation script.

Usage
  TODO
  scripts/eval_single_city.py
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from math import isnan
from tqdm import tqdm

import micropolis_world.module_globals as g
from fbsim_core.evaluation.models import get_models
from fbsim_core.metrics import compute_brier_score
from micropolis_world.scenarios import (
    build_corpus,
    build_prompt_continuous,
    get_single_city_base_scenarios,
)

CACHE_PATH = g.DATA_DIR / "response_cache.json"
RESULTS_PATH = g.DATA_DIR / "single_city_results.json"

DEFAULT_MODELS = [
    # ForecastBench models
    # "anthropic/claude-3-7-sonnet-20250219",
    # "anthropic/claude-opus-4-1-20250805",
    # "anthropic/claude-sonnet-4-20250514",
    # "openai/o3-2025-04-16",
    # "openai/gpt-4.1-2025-04-14",
    # "openai/gpt-5-2025-08-07",
    # "openai/gpt-5-mini-2025-08-07",
    # "google/gemini-2.5-pro",
    # "google/gemini-2.5-flash",
    # # "together/DeepSeek-V3.1",
    # # "together/Qwen3-235B-A22B-fp8-tput",
    # # "together/Kimi-K2-Instruct",
    # # "together/GLM-4.5-Air-FP8",
    # # "mistral/mistral-large-2411",
    # # Frontier Models
    # "anthropic/claude-opus-4-5-20251101",
    # "anthropic/claude-sonnet-4-5-20250929",
    # "google/gemini-3-pro-preview",
    # "openai/gpt-5.1-2025-11-13",
    # # Pandemic Models
    # "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct",
    # "deepinfra/meta-llama/Meta-Llama-3.1-70B-Instruct",
    # "deepinfra/Qwen/Qwen2.5-72B-Instruct",
    # For testing, currently cheapest recent OpenAI model
    # "openai/gpt-5.6-luna", # Note this one doesn't support temperature=0
    "openai/gpt-3.5-turbo-1106",
]


@dataclass(frozen=True)
class ResponseId:
    model_id: str
    question_id: str


@dataclass(frozen=True)
class Response:
    actual: float
    predicted: float
    response_text: str | None = None


Responses = dict[ResponseId, Response]
fnan = float("nan")


def parse_answer(text: str) -> float:
    """Extract the model's numeric estimate: the last number in the response.

    Last match wins because verbose models put the answer at the end, after a
    preamble full of incidental numbers (turn numbers, figures from the report).

    Thousands separators are accepted only in strict 3-digit groups ("161,000"),
    so a comma-separated list like "1,2,3" still reads as three numbers.

    Returns nan if unable to parse
    """
    if text is None:
        return fnan
    matches = re.findall(
        r"[-+]?(?:\d{1,3}(?:,\d{3})+(?!\d)|\d+)(?:\.\d*)?(?:[eE][-+]?\d+)?|[-+]?\.\d+(?:[eE][-+]?\d+)?",
        text,
    )
    if not matches:
        return fnan
    return float(matches[-1].replace(",", ""))


def load_cache() -> Responses:
    r: Responses = {}
    data = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    for entry in data:
        response_id = ResponseId(
            model_id=entry["model_id"], question_id=entry["question_id"]
        )
        response = Response(
            actual=entry["actual"],
            predicted=entry["predicted"],
            response_text=entry.get("response_text", None),
        )
        r[response_id] = response
    return r


def save_cache(cache: Responses) -> None:
    data = [asdict(k) | asdict(v) for k, v in cache.items()]
    CACHE_PATH.write_text(json.dumps(data, indent=2))


# TODO
# def make_chart(agg: dict, base_rates: dict) -> None:
#     import matplotlib

#     matplotlib.use("Agg")
#     import matplotlib.pyplot as plt

#     models = list(agg.keys())
#     labels = [
#         m.replace("Meta-Llama-3.1-", "Llama-3.1-").replace("-Instruct", "")
#         for m in models
#     ]
#     x = np.arange(len(models))
#     w = 0.38

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(11, 2.6 * len(models)), 5))

#     uncond = [agg[m]["unconditional"]["brier"] for m in models]
#     cond = [agg[m]["conditional"]["brier"] for m in models]
#     b1 = ax1.bar(x - w / 2, uncond, w, label="Unconditional", color="#4C72B0")
#     b2 = ax1.bar(
#         x + w / 2, cond, w, label="Conditional (given policy)", color="#C44E52"
#     )
#     ax1.axhline(0.25, ls="--", c="gray", lw=1, label="Uninformed (0.25)")
#     ax1.set_ylabel("Brier score (lower = better)")
#     ax1.set_title("Forecast accuracy")
#     ax1.set_xticks(x)
#     ax1.set_xticklabels(labels, rotation=15, ha="right")
#     ax1.legend(fontsize=8)
#     for bars in (b1, b2):
#         for b in bars:
#             ax1.annotate(
#                 f"{b.get_height():.3f}",
#                 (b.get_x() + b.get_width() / 2, b.get_height()),
#                 ha="center",
#                 va="bottom",
#                 fontsize=8,
#             )

#     mp_u = [agg[m]["unconditional"]["mean_pred"] for m in models]
#     mp_c = [agg[m]["conditional"]["mean_pred"] for m in models]
#     ax2.bar(
#         x - w / 2,
#         mp_u,
#         w,
#         label="Mean P(yes) unconditional",
#         color="#4C72B0",
#         alpha=0.85,
#     )
#     ax2.bar(
#         x + w / 2, mp_c, w, label="Mean P(yes) conditional", color="#C44E52", alpha=0.85
#     )
#     ax2.axhline(
#         base_rates["unconditional"],
#         ls="--",
#         c="#1f3a5f",
#         lw=1.5,
#         label=f"True rate uncond ({base_rates['unconditional']:.2f})",
#     )
#     ax2.axhline(
#         base_rates["conditional"],
#         ls="--",
#         c="#7a1f25",
#         lw=1.5,
#         label=f"True rate cond ({base_rates['conditional']:.2f})",
#     )
#     ax2.set_ylabel("Mean P(yes)")
#     ax2.set_title("Does the model lower P(yes) when told about the policy?")
#     ax2.set_xticks(x)
#     ax2.set_xticklabels(labels, rotation=15, ha="right")
#     ax2.legend(fontsize=7)

#     fig.suptitle(
#         "Conditional (intervention) reasoning: does the model shift toward "
#         "the true rate when told about the policy change?",
#         fontsize=11,
#     )
#     fig.tight_layout()
#     fig.savefig(CHART_PATH, dpi=150)


def gather_responses(corpus: list[dict], model_names: list[str]) -> Responses:
    """Prompt each model on each corpus question, reusing cached responses.

    The whole cache is loaded and saved back, so responses for other corpora and
    models are preserved, but only the responses for this call's corpus and
    model_names are returned.
    """
    n_new = 0
    cache = load_cache()
    responses: Responses = {}
    models = get_models(model_names)
    nmodels = len(models)
    for model_idx, (model_name, model) in enumerate(zip(model_names, models)):
        print(f"Prompting {model_name} ({model_idx + 1}/{nmodels})")
        for c in tqdm(corpus):
            response_id = ResponseId(model_id=model_name, question_id=c["question_id"])
            if response_id in cache:
                response = cache[response_id]
            else:
                raw = model.get_response(
                    build_prompt_continuous(c["context"], c["question_text"]),
                    max_tokens=4000,  # TOD increase this when I go to other modles
                )
                ans_value = parse_answer(raw)
                if isnan(ans_value):
                    print(
                        f"  {response_id.question_id}: unable to parse numeric answer from model response: {raw}"
                    )
                response = Response(
                    response_text=raw,
                    actual=c["value"],
                    predicted=ans_value,
                )
                cache[response_id] = response

                n_new += 1
                if n_new % 10 == 0:
                    save_cache(cache)
            responses[response_id] = response

    if n_new % 10:
        save_cache(cache)
    return responses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = ap.parse_args()

    print("=" * 70)
    print("MICROPOLIS WORLD — single city eval")
    print("=" * 70)

    scenarios = get_single_city_base_scenarios(args.seed)
    print(f"\nRunning {len(scenarios)} with seed={args.seed}; building corpus...")
    corpus = build_corpus(scenarios)

    if args.dry_run:
        print("\nDry run. Exiting")
        return

    print("\nGathering model responses...")
    cache = gather_responses(corpus, args.models)
    print("Done gathering")
    print("=" * 70)

    # TODO
    # - compute crps, maybe with different aggregations
    # - chart of crps per horizon?
    # - chart of


if __name__ == "__main__":
    main()
