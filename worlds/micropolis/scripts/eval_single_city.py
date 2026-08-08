#!/usr/bin/env -S uv run python3
"""Single city scenario evaluation script.

Builds the question corpus for every (city, disasters) combination in the
config file, prompts each configured model on it, and plots the forecasts.

Usage:
    uv run python scripts/eval_single_city.py
    uv run python scripts/eval_single_city.py my_config.json --seed 7
    uv run python scripts/eval_single_city.py my_config.json --dry-run
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass
from math import isnan
from pathlib import Path
from tqdm import tqdm

import micropolis_world.module_globals as g
from fbsim_core.evaluation.models import get_models
from fbsim_core.metrics import compute_brier_score
from micropolis_world.city_sim import CitySimulation, turn_of
from micropolis_world.config import (
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.scenarios import (
    build_corpus,
    build_prompt_continuous,
    get_single_city_base_scenarios,
)

CACHE_PATH = g.DATA_DIR / "response_cache.json"
RESULTS_PATH = g.DATA_DIR / "single_city_results.json"
PLOTS_PATH = g.DATA_DIR / "single_city_plots"

# The four metrics charted per figure, in subplot order. Only the metrics that
# have question templates get forecast points overlaid; the rest are context.
PLOT_METRICS = [
    ("cityPop", "Population", "tab:blue"),
    ("totalFunds", "Funds ($)", "tab:green"),
    ("crimeAverage", "Crime Average", "tab:red"),
    ("pollutionAverage", "Pollution Average", "tab:orange"),
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


def plot_forecasts(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    outdir: Path = PLOTS_PATH,
) -> list[Path]:
    """One figure per (scenario, horizon): metric trajectories + model forecasts.

    Each figure has the same four metric subplots as micropolis_world.plot_sim, drawn
    against turn rather than date so forecasts can be placed at the turn they
    resolve on (snapshot_turn + horizon). Disaster lines are omitted.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    # One color per model, stable across every figure.
    palette = plt.get_cmap("tab10")
    model_colors = {m: palette(i % 10) for i, m in enumerate(model_names)}

    # Group questions by the figure they belong to, then by which subplot.
    by_figure: dict[tuple[str, int], list[dict]] = {}
    for c in corpus:
        by_figure.setdefault((c["scenario_id"], c["horizon"]), []).append(c)

    # A scenario_id maps to exactly one simulation; load each one once.
    sims: dict[str, CitySimulation] = {}
    written = []

    for (scenario_id, horizon), entries in sorted(by_figure.items()):
        if scenario_id not in sims:
            sc = entries[0]["scenario"]
            sim = CitySimulation(
                city_name=sc["name"], seed=sc["seed"], disasters=sc["disasters"]
            )
            sim.load_from_disk()
            sims[scenario_id] = sim
        sim = sims[scenario_id]
        rows = sim.log_data or []
        turns = [turn_of(r) for r in rows]

        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True)
        for ax, (metric, title, color) in zip(axes.flat, PLOT_METRICS):
            ax.plot(turns, [r[metric] for r in rows], color=color, label="actual")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

            # Overlay each model's forecasts for this metric at the turn they
            # resolve on. Unparseable answers came back as nan; skip them.
            for model_id in model_names:
                xs, ys = [], []
                for c in entries:
                    if c["metric"] != metric:
                        continue
                    r = responses.get(ResponseId(model_id, c["question_id"]))
                    if r is None or isnan(r.predicted):
                        continue
                    xs.append(c["snapshot_turn"] + horizon)
                    ys.append(r.predicted)
                if xs:
                    ax.scatter(
                        xs, ys, s=45, zorder=5, alpha=0.85,
                        color=model_colors[model_id],
                        edgecolors="black", linewidths=0.5,
                        label=model_id.split("/")[-1],
                    )

        for ax in axes[1]:
            ax.set_xlabel("Turn")

        # One shared legend; every subplot has the same series.
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=max(2, len(labels)),
                   fontsize="small")
        fig.suptitle(f"{scenario_id} — forecasts at horizon {horizon}")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.13)

        out = outdir / f"forecasts_{scenario_id}_{horizon}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

    return written


def gather_responses(
    corpus: list[dict], model_names: list[str], max_tokens: int
) -> Responses:
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
                    max_tokens=max_tokens,
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


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args)
    seed = cfg.get_seed(args.seed)
    models = cfg.get_str_list("models")

    print("=" * 70)
    print("MICROPOLIS WORLD — single city eval")
    print("=" * 70)
    print(f"config: {cfg.path}")

    scenarios = get_single_city_base_scenarios(
        seed=seed, cities=cfg.get_cities(), disasters=cfg.get_bool_list("disasters")
    )
    print(f"\nRunning {len(scenarios)} with seed={seed}; building corpus...")
    corpus = build_corpus(
        scenarios,
        cfg.get_int_list("snapshot_turns"),
        cfg.get_int_list("horizons"),
        cfg.get_int("history_freq"),
    )

    if args.dry_run:
        print("\nDry run. Exiting")
        return

    print("\nGathering model responses...")
    g.ensure_api_keys()
    responses = gather_responses(corpus, models, cfg.get_int("max_tokens"))
    print("Done gathering")

    print("\nPlotting forecasts...")
    written = plot_forecasts(corpus, responses, models)
    print(f"Wrote {len(written)} plots -> {PLOTS_PATH}")
    print("=" * 70)

    # TODO
    # - compute crps, maybe with different aggregations
    # - chart of crps per horizon?
    # - chart of


if __name__ == "__main__":
    main()
