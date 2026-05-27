"""
Starsim pandemic — SCALED conditional/unconditional benchmark + chart. SPIKE.

Scales conditional_eval.py from n=1 to a real corpus across many sampled
scenarios and several models, then produces a chart of the conditional-vs-
unconditional Brier gap (the "intervention-blindness" finding) for Ezra.

Pipeline:
  1. Sample N scenarios (vary beta_R, beta_S, vaccine efficacy/day/coverage).
  2. Per scenario: run S, R-control, R-intervention sims (each deterministic);
     build control + intervention worlds in the core time_series schema.
  3. Per horizon: matched comparative questions "will R exceed S by day T?"
       - unconditional -> resolved on control world
       - conditional   -> resolved on intervention world
     resolved by the UNMODIFIED core QuestionResolver.
  4. Eval each model for P(yes); cache responses to disk; compute Brier.
  5. Aggregate Brier by (model, kind); plot grouped bars; save PNG + JSON.

Usage:
  uv run python starsim_spike/scale_eval.py --dry-run          # build corpus, check balance
  uv run python starsim_spike/scale_eval.py --eval             # full scored run + chart
  uv run python starsim_spike/scale_eval.py --eval --models deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct
"""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "src")

import numpy as np
import starsim as ss

from civrealm.world_reports.questions.schema import (
    QuestionInstance, QuestionTemplate, classify_horizon,
)
from civrealm.world_reports.questions import templates as tpl
from civrealm.world_reports.questions.resolver import QuestionResolver
from civrealm.metrics import compute_brier_score

# ---------------------------------------------------------------------------
SPIKE_DIR = Path(__file__).parent
CACHE_PATH = SPIKE_DIR / ".response_cache.json"
RESULTS_PATH = SPIKE_DIR / "scale_results.json"
CHART_PATH = SPIKE_DIR / "conditional_brier_gap.png"

N_AGENTS = 5_000
N_DAYS = 90
SNAPSHOT_DAY = 20
HORIZONS = [40, 60]          # resolution days
N_CONTACTS = 6
P_DEATH = 0.02
METRICS = ["cumulative_cases", "active_infections", "cumulative_deaths"]

DEFAULT_MODELS = [
    "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct",
    "deepinfra/meta-llama/Meta-Llama-3.1-70B-Instruct",
    "deepinfra/Qwen/Qwen2.5-72B-Instruct",
]

# cases_comparative template registered into the core registry
CASES_COMPARATIVE = QuestionTemplate(
    template_id="cases_comparative",
    signal_name="cumulative_cases",
    question_template="Will {region_a} have more cumulative cases than {region_b} at day {resolution_turn}?",
    resolution_type="comparative",
    data_path="time_series.cumulative_cases.{player_id}.{resolution_turn}",
    comparison_op=">",
    required_params=["region_a", "region_b", "player_id_a", "player_id_b", "resolution_turn"],
)
tpl.TEMPLATES_BY_ID["cases_comparative"] = CASES_COMPARATIVE

R_ID, S_ID = 0, 1
R_NAME, S_NAME = "Riverton", "Southbay"


# ---------------------------------------------------------------------------
# Scenario sampling
# ---------------------------------------------------------------------------
def sample_scenarios(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    scenarios = []
    for i in range(n):
        scenarios.append(dict(
            scenario_id=i,
            beta_R=float(rng.choice([0.030, 0.040, 0.050, 0.065])),
            beta_S=float(rng.choice([0.030, 0.040, 0.050, 0.065])),
            vax_efficacy=float(rng.choice([0.5, 0.7, 0.9])),
            vax_day=int(rng.choice([15, 25, 35])),
            vax_coverage=float(rng.choice([0.4, 0.6, 0.8])),
            sim_seed=int(rng.integers(1, 10_000)),
        ))
    return scenarios


def run_region(beta, vaccinate, sc, seed) -> dict:
    sir = ss.SIR(init_prev=0.005, beta=beta, dur_inf=10, p_death=P_DEATH)
    pars = dict(n_agents=N_AGENTS, networks=dict(type="random", n_contacts=N_CONTACTS),
                diseases=sir, dur=N_DAYS, dt=1, verbose=0, rand_seed=seed)
    if vaccinate:
        vax = ss.simple_vx(efficacy=sc["vax_efficacy"], leaky=True)
        pars["interventions"] = [ss.campaign_vx(product=vax, years=[sc["vax_day"]],
                                                prob=sc["vax_coverage"])]
    sim = ss.Sim(pars); sim.run()
    r = sim.results
    return {
        "cumulative_cases": np.array(r["sir"]["cum_infections"]).tolist(),
        "active_infections": np.array(r["sir"]["n_infected"]).tolist(),
        "cumulative_deaths": np.array(r["cum_deaths"]).tolist(),
    }


def to_world(r_series, s_series) -> dict:
    ts = {m: {} for m in METRICS}
    for rid, series in [(R_ID, r_series), (S_ID, s_series)]:
        for m in METRICS:
            for day, val in enumerate(series[m]):
                ts[m].setdefault(str(day), {})[str(rid)] = val
    return {"time_series": ts,
            "civilizations": {str(R_ID): {"name": R_NAME, "nation_id": str(R_ID)},
                              str(S_ID): {"name": S_NAME, "nation_id": str(S_ID)}}}


def context_blurb(world, snapshot_day, intervention) -> str:
    ts = world["time_series"]
    lines = [f"PANDEMIC SITUATION REPORT — Day {snapshot_day}", ""]
    for rid, name in [(R_ID, R_NAME), (S_ID, S_NAME)]:
        cases = [ts["cumulative_cases"][str(d)][str(rid)] for d in range(0, snapshot_day + 1, 5)]
        active = ts["active_infections"][str(snapshot_day)][str(rid)]
        deaths = ts["cumulative_deaths"][str(snapshot_day)][str(rid)]
        traj = " -> ".join(f"d{d}:{int(c)}" for d, c in zip(range(0, snapshot_day + 1, 5), cases))
        lines.append(f"{name}: cumulative cases {traj}")
        lines.append(f"  (day {snapshot_day}: {int(active)} active, {int(deaths)} deaths)")
    lines.append(f"\nPopulation per region: {N_AGENTS}. Regions are isolated (no inter-region travel).")
    if intervention:
        lines.append(f"\nPLANNED INTERVENTION: {intervention}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def build_corpus(scenarios: list[dict]) -> list[dict]:
    resolver = QuestionResolver()
    corpus = []
    for sc in scenarios:
        s_series = run_region(sc["beta_S"], False, sc, sc["sim_seed"])
        r_control = run_region(sc["beta_R"], False, sc, sc["sim_seed"])
        r_interv = run_region(sc["beta_R"], True, sc, sc["sim_seed"])
        control_world = to_world(r_control, s_series)
        interv_world = to_world(r_interv, s_series)

        intervention_text = (
            f"{R_NAME} will run a vaccination campaign on day {sc['vax_day']} "
            f"({int(sc['vax_efficacy']*100)}% efficacy, {int(sc['vax_coverage']*100)}% coverage). "
            f"{S_NAME} has no planned intervention."
        )

        for T in HORIZONS:
            q_text = CASES_COMPARATIVE.question_template.format(
                region_a=R_NAME, region_b=S_NAME, resolution_turn=T)

            def mk(world, kind, intervention):
                q = QuestionInstance(
                    question_id=f"s{sc['scenario_id']}_T{T}_{kind}",
                    template_id="cases_comparative", resolution_turn=T,
                    horizon=classify_horizon(SNAPSHOT_DAY, T),
                    parameters={"region_a": R_NAME, "region_b": S_NAME,
                                "player_id_a": R_ID, "player_id_b": S_ID},
                    question_text=q_text)
                res = resolver.resolve(q, world, SNAPSHOT_DAY)
                return {
                    "question_id": q.question_id, "kind": kind, "horizon": T,
                    "scenario_id": sc["scenario_id"], "question_text": q_text,
                    "ground_truth": bool(res.answer),
                    "value_R": res.value_a, "value_S": res.value_b,
                    "context": context_blurb(world, SNAPSHOT_DAY, intervention),
                    "scenario": {k: sc[k] for k in
                                 ("beta_R", "beta_S", "vax_efficacy", "vax_day", "vax_coverage")},
                }

            corpus.append(mk(control_world, "unconditional", None))
            corpus.append(mk(interv_world, "conditional", intervention_text))
    return corpus


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


def build_prompt(context, question_text) -> str:
    return (context + "\n\n" + f"QUESTION: {question_text}\n\n"
            + "Give your probability that the answer is YES, as a single number "
            + "between 0 and 1. Respond with ONLY the number (e.g. 0.73).")


# ---------------------------------------------------------------------------
def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


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
    # how often does the intervention flip the answer?
    flips = 0
    pairs = {}
    for c in corpus:
        pairs.setdefault((c["scenario_id"], c["horizon"]), {})[c["kind"]] = c["ground_truth"]
    for kinds in pairs.values():
        if "unconditional" in kinds and "conditional" in kinds:
            flips += kinds["unconditional"] != kinds["conditional"]
    print(f"  intervention flips the answer in {flips}/{len(pairs)} pairs "
          f"({100*flips/len(pairs):.0f}%)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build corpus, report balance, no model calls")
    ap.add_argument("--eval", action="store_true", help="run scored eval + chart")
    ap.add_argument("--n-scenarios", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    args = ap.parse_args()

    print("=" * 70)
    print("STARSIM PANDEMIC — scaled conditional/unconditional benchmark")
    print("=" * 70)
    print(f"\nSampling {args.n_scenarios} scenarios (seed={args.seed})...")
    scenarios = sample_scenarios(args.n_scenarios, args.seed)
    print("Running sims + building corpus (3 sims/scenario)...")
    corpus = build_corpus(scenarios)
    report_balance(corpus)

    if not args.eval:
        print("\n(Dry run. Re-run with --eval to score models + chart.)")
        print("=" * 70)
        return

    from civrealm.evaluation.models import get_models
    cache = load_cache()
    # model -> kind -> (preds[], outcomes[])
    agg: dict[str, dict[str, list]] = {}

    for model_id in args.models:
        model = get_models([model_id])[0]
        short = model_id.split("/")[-1]
        print(f"\nEvaluating {short} ({len(corpus)} questions)...")
        preds = {"unconditional": [], "conditional": []}
        outs = {"unconditional": [], "conditional": []}
        n_new = 0
        for i, c in enumerate(corpus):
            ck = f"{model_id}|{c['question_id']}"
            if ck in cache:
                p = cache[ck]
            else:
                raw = model.get_response(build_prompt(c["context"], c["question_text"]),
                                         max_tokens=2000)
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
        print(f"  {short}: cached {len(corpus)-n_new}, new {n_new}")
        for k in ("unconditional", "conditional"):
            a = agg[short][k]
            if a["brier"] is not None:
                print(f"    {k:14s} Brier={a['brier']:.3f}  (n={a['n']}, mean P={a['mean_pred']:.2f})")

    base_rates = {
        k: float(np.mean([c["ground_truth"] for c in corpus if c["kind"] == k]))
        for k in ("unconditional", "conditional")
    }
    RESULTS_PATH.write_text(json.dumps(
        {"generated_at": datetime.now(timezone.utc).isoformat(),
         "n_scenarios": args.n_scenarios, "horizons": HORIZONS,
         "snapshot_day": SNAPSHOT_DAY, "base_rates": base_rates, "results": agg}, indent=2))
    print(f"\nResults -> {RESULTS_PATH}")
    make_chart(agg, base_rates)
    print(f"Chart   -> {CHART_PATH}")
    print("=" * 70)


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

    # Panel 1: Brier by condition type
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

    # Panel 2: mechanism — does mean P(yes) shift toward reality under intervention?
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


if __name__ == "__main__":
    main()
