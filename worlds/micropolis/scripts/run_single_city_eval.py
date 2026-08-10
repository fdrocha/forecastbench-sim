#!/usr/bin/env -S uv run python3
"""Single city scenario evaluation script.

Builds the question corpus for every (city, disasters) combination in the
config file and prompts each configured model on it.

Usage:
    uv run python scripts/run_single_city_eval.py
    uv run python scripts/run_single_city_eval.py my_config.json --seed 7
    uv run python scripts/run_single_city_eval.py my_config.json --dry-run
    uv run python scripts/run_single_city_eval.py --scenario-plots
    uv run python scripts/run_single_city_eval.py --verbose-reparse
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import micropolis_world.module_globals as g
from fbsim_core.evaluation.models import get_models
from fbsim_core.metrics import compute_crps
from micropolis_world.city_sim import CitySimulation, turn_of
from micropolis_world.config import (
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.scenarios import (
    PERCENTILE_KEYS,
    build_corpus,
    build_prompt_continuous,
    get_single_city_base_scenarios,
    parse_percentiles,
)
from tqdm import tqdm

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
    data = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
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
    CACHE_PATH.write_text(json.dumps(data, indent=2))


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
            # resolve on: the median as a marker, p10-p90 as an error bar.
            # Unparseable answers have no percentiles; skip them.
            for model_id in model_names:
                xs, medians, lo, hi = [], [], [], []
                for c in entries:
                    if c["metric"] != metric:
                        continue
                    r = responses.get(ResponseId(model_id, c["question_id"]))
                    if r is None or r.percentiles is None:
                        continue
                    p = r.percentiles
                    xs.append(c["snapshot_turn"] + horizon)
                    medians.append(p["p50"])
                    # errorbar wants distances from the median, not absolute
                    # positions. Both are non-negative because parse_percentiles
                    # rejects any set that isn't in increasing order.
                    lo.append(p["p50"] - p["p10"])
                    hi.append(p["p90"] - p["p50"])
                if xs:
                    ax.errorbar(
                        xs,
                        medians,
                        yerr=[lo, hi],
                        fmt="o",
                        markersize=6,
                        zorder=5,
                        alpha=0.85,
                        color=model_colors[model_id],
                        markeredgecolor="black",
                        markeredgewidth=0.5,
                        elinewidth=1.2,
                        capsize=3,
                        label=model_id.split("/")[-1],
                    )

        for ax in axes[1]:
            ax.set_xlabel("Turn")

        # One shared legend; every subplot has the same series.
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=max(2, len(labels)),
            fontsize="small",
        )
        fig.suptitle(f"{scenario_id} — forecasts at horizon {horizon}")
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.13)

        out = outdir / f"forecasts_{scenario_id}_{horizon}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        written.append(out)

    return written


def crps_by_model_and_metric(
    corpus: list[dict], responses: Responses, model_names: list[str]
) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], int], list[str]]:
    """Mean CRPS per (model, metric), plus how many questions each cell covers.

    Returns (means, counts, metrics), where metrics is in corpus order. Cells
    with no parseable forecast are absent from both dicts.
    """
    metrics = list(dict.fromkeys(c["metric"] for c in corpus))
    scores: dict[tuple[str, str], list[float]] = {}
    for c in corpus:
        for model_id in model_names:
            r = responses.get(ResponseId(model_id, c["question_id"]))
            if r is None or r.percentiles is None:
                continue
            crps = compute_crps(r.percentiles, c["value"])
            scores.setdefault((model_id, c["metric"]), []).append(crps)

    means = {k: sum(v) / len(v) for k, v in scores.items()}
    counts = {k: len(v) for k, v in scores.items()}
    return means, counts, metrics


def print_crps_table(
    corpus: list[dict], responses: Responses, model_names: list[str]
) -> None:
    """Print models x metrics, each cell the mean CRPS over that model's forecasts.

    CRPS is in each metric's own units, so it compares models down a column but
    never across columns.
    """
    means, counts, metrics = crps_by_model_and_metric(corpus, responses, model_names)

    labels = {m: g.METRIC_LABELS.get(m, m) for m in metrics}
    model_col = max([len("Model")] + [len(m.split("/")[-1]) for m in model_names])
    widths = {m: max(len(labels[m]), 12) for m in metrics}

    print("\nMean CRPS by model and metric (lower is better)")
    header = f"{'Model':<{model_col}}  " + "  ".join(
        f"{labels[m]:>{widths[m]}}" for m in metrics
    )
    print(header)
    print("-" * len(header))

    for model_id in model_names:
        row = [f"{model_id.split('/')[-1]:<{model_col}}"]
        for m in metrics:
            mean = means.get((model_id, m))
            cell = "n/a" if mean is None else f"{mean:,.1f}"
            row.append(f"{cell:>{widths[m]}}")
        print("  ".join(row))

    # A cell averaging fewer questions than the corpus holds means some
    # responses failed to parse; say so rather than let the means look complete.
    expected = {m: sum(1 for c in corpus if c["metric"] == m) for m in metrics}
    missing = [
        f"{model_id.split('/')[-1]}/{labels[m]}: {expected[m] - counts.get((model_id, m), 0)}"
        for model_id in model_names
        for m in metrics
        if counts.get((model_id, m), 0) < expected[m]
    ]
    if missing:
        print(f"\nUnparseable forecasts excluded — {', '.join(missing)}")


def gather_responses(
    corpus: list[dict],
    model_names: list[str],
    max_tokens: int,
    verbose_reparse: bool = False,
) -> Responses:
    """Prompt each model on each corpus question, reusing cached responses.

    The whole cache is loaded and saved back, so responses for other corpora and
    models are preserved, but only the responses for this call's corpus and
    model_names are returned.

    verbose_reparse is passed to load_cache, printing why each cached response
    that no longer parses was rejected instead of just how many.
    """
    n_new = 0
    cache = load_cache(verbose_reparse=verbose_reparse)
    # Re-parsing is quiet unless asked otherwise, so say how many stored
    # responses have no usable forecast rather than letting them vanish.
    n_bad_cached = sum(1 for r in cache.values() if r.percentiles is None)
    if n_bad_cached:
        hint = "" if verbose_reparse else "; --verbose-reparse to see why"
        print(
            f"{n_bad_cached} of {len(cache)} cached responses have no usable "
            f"percentiles (re-parsed from the stored text{hint})"
        )
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
                # prompt_model rather than model.get_response, because the
                # finish reason is what distinguishes a model that answered
                # badly from one that never got to answer at all.
                raw, finish_reason = g.prompt_model(
                    model,
                    build_prompt_continuous(c["context"], c["question_text"]),
                    max_tokens,
                )
                g.warn_if_truncated(model_name, finish_reason, max_tokens)
                # parse_percentiles reports its own reason for rejecting a
                # response, so only the question id needs adding here.
                percentiles = parse_percentiles(raw, label=response_id.question_id)
                response = Response(
                    response_text=raw,
                    actual=c["value"],
                    percentiles=percentiles,
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
    ap.add_argument(
        "--scenario-plots",
        action="store_true",
        help="also plot metric trajectories with forecasts overlaid, one figure "
        "per (scenario, horizon)",
    )
    ap.add_argument(
        "--verbose-reparse",
        action="store_true",
        help="print the full warning for every cached response whose percentiles "
        "no longer parse, rather than only the count",
    )
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
    responses = gather_responses(
        corpus,
        models,
        cfg.get_int("max_tokens"),
        verbose_reparse=args.verbose_reparse,
    )
    print("Done gathering")

    if args.scenario_plots:
        print("\nPlotting forecasts...")
        written = plot_forecasts(corpus, responses, models)
        print(f"Wrote {len(written)} plots -> {PLOTS_PATH}")
    print("=" * 70)

    print_crps_table(corpus, responses, models)


if __name__ == "__main__":
    main()
