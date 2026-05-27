"""
Starsim pandemic — CONDITIONAL forecasting + end-to-end scored eval. SPIKE.

Builds on spike.py. Demonstrates the full pipeline for a non-FreeCiv world:

  1. Per-region independent SIR sims (separate populations, no cross-region
     transmission — counties with no travel). Region R additionally gets a
     vaccinated variant. A never-vaccinated region S is the comparison and is
     unaffected by R's policy, so the counterfactual is clean.
  2. Two worlds, both serialized to the core time_series schema (turn-major):
       - control world:      R unvaccinated, S unvaccinated
       - intervention world: R vaccinated day D, S unvaccinated
  3. Matched questions resolved by the UNMODIFIED core resolver:
       - unconditional: "Will R have more cases than S by day T?"  -> control world
       - conditional:   "GIVEN R vaccinates day D, will R have more
                         cases than S by day T?"                    -> intervention world
  4. Minimal pandemic context blurb (replaces FreeCiv world-report renderer).
  5. Scored eval: query a cheap model for P(yes), compute Brier vs ground truth,
     report conditional vs unconditional.

Run (resolve only, no model):  uv run python starsim_spike/conditional_eval.py
Run (with scored eval):        uv run python starsim_spike/conditional_eval.py --eval
"""

import argparse
import re
import sys
from datetime import datetime, timezone

sys.path.insert(0, "src")

import numpy as np
import starsim as ss

from civrealm.world_reports.questions.schema import (
    QuestionInstance,
    QuestionTemplate,
    classify_horizon,
)
from civrealm.world_reports.questions import templates as tpl
from civrealm.world_reports.questions.resolver import QuestionResolver
from civrealm.metrics import compute_brier_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_AGENTS = 5_000
N_DAYS = 90
SNAPSHOT_DAY = 20
RESOLUTION_DAY = 60
VACCINE_DAY = 25
EVAL_MODEL = "deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct"

# Region R (policy region) and comparison region S. Independent populations.
R_ID, S_ID = 0, 1
R_NAME, S_NAME = "Riverton", "Southbay"
REGION_PARS = {
    R_ID: dict(name=R_NAME, beta=0.045, p_death=0.02),
    S_ID: dict(name=S_NAME, beta=0.040, p_death=0.02),
}
METRICS = ["cumulative_cases", "active_infections", "cumulative_deaths"]


# ---------------------------------------------------------------------------
# 1. Run regions
# ---------------------------------------------------------------------------
def run_region(beta, p_death, vaccinate, seed) -> dict:
    sir = ss.SIR(init_prev=0.005, beta=beta, dur_inf=10, p_death=p_death)
    pars = dict(
        n_agents=N_AGENTS,
        networks=dict(type="random", n_contacts=6),
        diseases=sir,
        dur=N_DAYS,
        dt=1,
        verbose=0,
        rand_seed=seed,
    )
    if vaccinate:
        vax = ss.simple_vx(efficacy=0.9, leaky=True)
        pars["interventions"] = [ss.campaign_vx(product=vax, years=[VACCINE_DAY], prob=0.7)]
    sim = ss.Sim(pars)
    sim.run()
    r = sim.results
    return {
        "cumulative_cases": np.array(r["sir"]["cum_infections"]).tolist(),
        "active_infections": np.array(r["sir"]["n_infected"]).tolist(),
        "cumulative_deaths": np.array(r["cum_deaths"]).tolist(),
    }


def to_game_data(region_series: dict[int, dict], world: str) -> dict:
    """region_series: {region_id: {metric: [per-day values]}} -> core schema (turn-major)."""
    time_series = {m: {} for m in METRICS}
    for rid, series in region_series.items():
        for m in METRICS:
            for day, val in enumerate(series[m]):
                time_series[m].setdefault(str(day), {})[str(rid)] = val
    return {
        "metadata": {"world": f"starsim_pandemic_{world}", "generated_at":
                     datetime.now(timezone.utc).isoformat()},
        "time_series": time_series,
        "civilizations": {str(rid): {"name": REGION_PARS[rid]["name"], "nation_id": str(rid)}
                          for rid in REGION_PARS},
    }


# ---------------------------------------------------------------------------
# 3. Templates (registered into core registry)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# 4. Minimal pandemic context blurb (model-facing world report)
# ---------------------------------------------------------------------------
def context_blurb(game_data: dict, regions: list[int], snapshot_day: int,
                  intervention: str | None) -> str:
    """Per-region cases/deaths trajectory up to the snapshot day."""
    lines = [f"PANDEMIC SITUATION REPORT — Day {snapshot_day}", ""]
    ts = game_data["time_series"]
    for rid in regions:
        name = REGION_PARS[rid]["name"]
        cases = [ts["cumulative_cases"][str(d)][str(rid)] for d in range(0, snapshot_day + 1, 5)]
        deaths = ts["cumulative_deaths"][str(snapshot_day)][str(rid)]
        active = ts["active_infections"][str(snapshot_day)][str(rid)]
        traj = " -> ".join(f"d{d}:{int(c)}" for d, c in zip(range(0, snapshot_day + 1, 5), cases))
        lines.append(f"{name}: cumulative cases {traj}")
        lines.append(f"  (day {snapshot_day}: {int(active)} active, {int(deaths)} cumulative deaths)")
    lines.append("")
    lines.append(f"Population per region: {N_AGENTS}. Regions are isolated (no inter-region travel).")
    if intervention:
        lines.append("")
        lines.append(f"PLANNED INTERVENTION: {intervention}")
    return "\n".join(lines)


def build_prompt(context: str, question_text: str) -> str:
    return (
        context
        + "\n\n"
        + f"QUESTION: {question_text}\n\n"
        + "Give your probability that the answer is YES, as a single number "
        + "between 0 and 1. Respond with ONLY the number (e.g. 0.73)."
    )


# ---------------------------------------------------------------------------
# Build matched conditional / unconditional question pair set
# ---------------------------------------------------------------------------
def make_pair(qid: str, world_kind: str) -> QuestionInstance:
    """A cases_comparative question: does R have more cases than S at resolution day?"""
    return QuestionInstance(
        question_id=qid,
        template_id="cases_comparative",
        resolution_turn=RESOLUTION_DAY,
        horizon=classify_horizon(SNAPSHOT_DAY, RESOLUTION_DAY),
        parameters={"region_a": R_NAME, "region_b": S_NAME,
                    "player_id_a": R_ID, "player_id_b": S_ID, "world_kind": world_kind},
        question_text=CASES_COMPARATIVE.question_template.format(
            region_a=R_NAME, region_b=S_NAME, resolution_turn=RESOLUTION_DAY),
    )


def parse_prob(text: str) -> float | None:
    m = re.search(r"(?<![\d.])(0?\.\d+|1\.0+|0|1)(?![\d.])", text.strip())
    if not m:
        m = re.search(r"\d+(\.\d+)?", text)
    if not m:
        return None
    try:
        v = float(m.group(0))
    except ValueError:
        return None
    return min(max(v, 0.0), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true", help="run scored model eval")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print("=" * 68)
    print("STARSIM PANDEMIC — conditional vs unconditional, end-to-end")
    print("=" * 68)

    # --- Run the three region-variants ---
    print(f"\n[1] Running region sims (seed={args.seed})...")
    s_series = run_region(**{k: v for k, v in REGION_PARS[S_ID].items() if k != "name"},
                          vaccinate=False, seed=args.seed)
    r_control = run_region(**{k: v for k, v in REGION_PARS[R_ID].items() if k != "name"},
                          vaccinate=False, seed=args.seed)
    r_interv = run_region(**{k: v for k, v in REGION_PARS[R_ID].items() if k != "name"},
                          vaccinate=True, seed=args.seed)

    control_world = to_game_data({R_ID: r_control, S_ID: s_series}, "control")
    interv_world = to_game_data({R_ID: r_interv, S_ID: s_series}, "intervention")

    r_c = r_control["cumulative_cases"][RESOLUTION_DAY]
    r_i = r_interv["cumulative_cases"][RESOLUTION_DAY]
    s_c = s_series["cumulative_cases"][RESOLUTION_DAY]
    print(f"    day {RESOLUTION_DAY} cumulative cases:")
    print(f"      {S_NAME} (comparison):        {s_c:.0f}")
    print(f"      {R_NAME} control (no vax):    {r_c:.0f}")
    print(f"      {R_NAME} vaccinated (day {VACCINE_DAY}): {r_i:.0f}  "
          f"({100*(r_c-r_i)/r_c:.0f}% fewer cases)")

    # --- Build + resolve matched questions via core resolver ---
    print("\n[2] Resolving matched questions via UNMODIFIED core resolver...")
    resolver = QuestionResolver()

    q_uncond = make_pair("uncond_0", "control")
    q_uncond.resolution = resolver.resolve(q_uncond, control_world, SNAPSHOT_DAY)

    q_cond = make_pair("cond_0", "intervention")
    q_cond.resolution = resolver.resolve(q_cond, interv_world, SNAPSHOT_DAY)

    gt_uncond = q_uncond.resolution.answer
    gt_cond = q_cond.resolution.answer
    print(f"    UNCONDITIONAL: '{q_uncond.question_text}'")
    print(f"      -> {gt_uncond}  ({R_NAME}={q_uncond.resolution.value_a:.0f} "
          f"vs {S_NAME}={q_uncond.resolution.value_b:.0f})")
    print(f"    CONDITIONAL (given {R_NAME} vaccinates day {VACCINE_DAY}): same question")
    print(f"      -> {gt_cond}  ({R_NAME}={q_cond.resolution.value_a:.0f} "
          f"vs {S_NAME}={q_cond.resolution.value_b:.0f})")
    flip = gt_uncond != gt_cond
    print(f"    conditional_effect: answer {'FLIPS' if flip else 'unchanged'} "
          f"under the intervention.")

    if not args.eval:
        print("\n(Resolve-only. Re-run with --eval to score a model.)")
        print("=" * 68)
        return

    # --- Scored eval ---
    print(f"\n[3] Scored eval with {EVAL_MODEL.split('/')[-1]}...")
    from civrealm.evaluation.models import get_models
    model = get_models([EVAL_MODEL])[0]

    snapshot_day = SNAPSHOT_DAY
    jobs = [
        ("unconditional", q_uncond, gt_uncond,
         context_blurb(control_world, [R_ID, S_ID], snapshot_day, intervention=None)),
        ("conditional", q_cond, gt_cond,
         context_blurb(control_world, [R_ID, S_ID], snapshot_day,
                       intervention=f"{R_NAME} will run a vaccination campaign on day "
                                    f"{VACCINE_DAY} (90% efficacy, 70% coverage). "
                                    f"{S_NAME} has no planned intervention.")),
    ]

    results = []
    for kind, q, gt, ctx in jobs:
        prompt = build_prompt(ctx, q.question_text)
        raw = model.get_response(prompt, max_tokens=2000)
        p = parse_prob(raw)
        if p is None:
            print(f"    [{kind}] could not parse: {raw!r}")
            continue
        brier = compute_brier_score([p], [gt])
        results.append((kind, p, gt, brier))
        print(f"    [{kind}] model P(yes)={p:.2f}  truth={gt}  Brier={brier:.3f}")

    print("\n" + "=" * 68)
    if results:
        print("SCORED RESULTS (lower Brier = better; 0.25 = uninformed 50/50):")
        for kind, p, gt, brier in results:
            print(f"  {kind:14s} P={p:.2f}  truth={gt}  Brier={brier:.3f}")
        print("\nFull pipeline runs end-to-end on a pandemic world: "
              "sim -> schema -> core resolver -> context -> model -> Brier.")
    print("=" * 68)


if __name__ == "__main__":
    main()
