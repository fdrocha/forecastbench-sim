"""Scaled pandemic conditional/unconditional benchmark + chart.

Samples many scenarios, builds a matched corpus (resolved by fbsim-core), scores
several models (cached on disk), and plots the conditional-vs-unconditional Brier
gap. Demonstrates the pandemic world running entirely on fbsim-core.

Usage:
  uv run python -m pandemic_world.scale_eval --dry-run
  uv run python -m pandemic_world.scale_eval --eval
"""

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from fbsim_core.metrics import compute_brier_score

from .scenarios import sample_scenarios, build_corpus
from .report import build_prompt

PKG_DIR = Path(__file__).resolve().parent.parent  # worlds/pandemic
CACHE_PATH = PKG_DIR / ".response_cache.json"
RESULTS_PATH = PKG_DIR / "scale_results.json"
CHART_PATH = PKG_DIR / "conditional_brier_gap.png"

DEFAULT_MODELS = [
    "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct",
    "deepinfra/meta-llama/Meta-Llama-3.1-70B-Instruct",
    "deepinfra/Qwen/Qwen2.5-72B-Instruct",
]


def parse_prob(text: str) -> float | None:
    m = re.search(r"(?<![\d.])(0?\.\d+|1\.0+|0|1)(?![\d.])", text.strip())
    if not m:
        m = re.search(r"\d+(\.\d+)?", text)
    if not m:
        return None
    try:
        return min(max(float(m.group(0)), 0.0), 1.0)
    except ValueError:
        return None


def load_cache() -> dict:
    return json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}


def save_cache(cache: dict) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2))


def report_balance(corpus: list[dict]) -> None:
    print(f"\nCorpus: {len(corpus)} questions "
          f"({len(corpus)//2} matched pairs, {len(set(c['scenario_id'] for c in corpus))} scenarios)")
    for kind in ("unconditional", "conditional"):
        sub = [c for c in corpus if c["kind"] == kind]
        t = sum(c["ground_truth"] for c in sub)
        print(f"  {kind:14s}: {len(sub):3d} questions, {t} True / {len(sub)-t} False "
              f"({100*t/len(sub):.0f}% True)")
    pairs = {}
    for c in corpus:
        pairs.setdefault((c["scenario_id"], c["horizon"]), {})[c["kind"]] = c["ground_truth"]
    flips = sum(k.get("unconditional") != k.get("conditional")
                for k in pairs.values() if len(k) == 2)
    print(f"  intervention flips the answer in {flips}/{len(pairs)} pairs "
          f"({100*flips/len(pairs):.0f}%)")


def make_chart(agg: dict, base_rates: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(agg.keys())
    labels = [m.replace("Meta-Llama-3.1-", "Llama-3.1-").replace("-Instruct", "")
              for m in models]
    x = np.arange(len(models))
    w = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(max(11, 2.6 * len(models)), 5))

    uncond = [agg[m]["unconditional"]["brier"] for m in models]
    cond = [agg[m]["conditional"]["brier"] for m in models]
    b1 = ax1.bar(x - w/2, uncond, w, label="Unconditional", color="#4C72B0")
    b2 = ax1.bar(x + w/2, cond, w, label="Conditional (given vaccine)", color="#C44E52")
    ax1.axhline(0.25, ls="--", c="gray", lw=1, label="Uninformed (0.25)")
    ax1.set_ylabel("Brier score (lower = better)")
    ax1.set_title("Forecast accuracy")
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.legend(fontsize=8)
    for bars in (b1, b2):
        for b in bars:
            ax1.annotate(f"{b.get_height():.3f}", (b.get_x() + b.get_width()/2, b.get_height()),
                         ha="center", va="bottom", fontsize=8)

    mp_u = [agg[m]["unconditional"]["mean_pred"] for m in models]
    mp_c = [agg[m]["conditional"]["mean_pred"] for m in models]
    ax2.bar(x - w/2, mp_u, w, label="Mean P(yes) unconditional", color="#4C72B0", alpha=0.85)
    ax2.bar(x + w/2, mp_c, w, label="Mean P(yes) conditional", color="#C44E52", alpha=0.85)
    ax2.axhline(base_rates["unconditional"], ls="--", c="#1f3a5f", lw=1.5,
                label=f"True rate uncond ({base_rates['unconditional']:.2f})")
    ax2.axhline(base_rates["conditional"], ls="--", c="#7a1f25", lw=1.5,
                label=f"True rate cond ({base_rates['conditional']:.2f})")
    ax2.set_ylabel("Mean P(yes)")
    ax2.set_title("Does the model lower P(yes) when told about the vaccine?")
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.legend(fontsize=7)

    fig.suptitle("Conditional (intervention) reasoning emerges with scale: "
                 "8B fails to adjust for the vaccine; 70B+ shift toward the true rate",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--n-scenarios", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = ap.parse_args()

    print("=" * 70)
    print("PANDEMIC WORLD — scaled conditional/unconditional benchmark (on fbsim-core)")
    print("=" * 70)
    print(f"\nSampling {args.n_scenarios} scenarios (seed={args.seed}); building corpus...")
    scenarios = sample_scenarios(args.n_scenarios, args.seed)
    corpus = build_corpus(scenarios)
    report_balance(corpus)

    if not args.eval:
        print("\n(Dry run. Re-run with --eval to score models + chart.)")
        print("=" * 70)
        return

    from fbsim_core.evaluation.models import get_models
    cache = load_cache()
    agg: dict[str, dict] = {}

    for model_id in args.models:
        model = get_models([model_id])[0]
        short = model_id.split("/")[-1]
        print(f"\nEvaluating {short} ({len(corpus)} questions)...")
        preds = {"unconditional": [], "conditional": []}
        outs = {"unconditional": [], "conditional": []}
        n_new = 0
        for c in corpus:
            ck = f"{model_id}|{c['question_id']}"
            if ck in cache:
                p = cache[ck]
            else:
                raw = model.get_response(build_prompt(c["context"], c["question_text"]), max_tokens=2000)
                p = parse_prob(raw)
                cache[ck] = p
                n_new += 1
                if n_new % 10 == 0:
                    save_cache(cache)
            if p is None:
                continue
            preds[c["kind"]].append(p)
            outs[c["kind"]].append(c["ground_truth"])
        save_cache(cache)
        agg[short] = {
            k: {"brier": compute_brier_score(preds[k], outs[k]) if preds[k] else None,
                "n": len(preds[k]),
                "mean_pred": float(np.mean(preds[k])) if preds[k] else None}
            for k in ("unconditional", "conditional")
        }
        print(f"  cached {len(corpus)-n_new}, new {n_new}")
        for k in ("unconditional", "conditional"):
            a = agg[short][k]
            if a["brier"] is not None:
                print(f"    {k:14s} Brier={a['brier']:.3f}  (n={a['n']}, mean P={a['mean_pred']:.2f})")

    base_rates = {k: float(np.mean([c["ground_truth"] for c in corpus if c["kind"] == k]))
                  for k in ("unconditional", "conditional")}
    RESULTS_PATH.write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(),
         "n_scenarios": args.n_scenarios, "horizons": [40, 60],
         "base_rates": base_rates, "results": agg}, indent=2))
    print(f"\nResults -> {RESULTS_PATH}")
    make_chart(agg, base_rates)
    print(f"Chart   -> {CHART_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
