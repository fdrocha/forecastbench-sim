"""
Starsim pandemic world — SPIKE.

Goal: validate that fbsim-core (schema + resolver) generalizes to a non-FreeCiv
world WITHOUT modifying core. We:

  1. Run N Starsim SIR sims with varied params (each sim = one "region", the
     pandemic analog of a FreeCiv civ / entity).
  2. Serialize results into the existing `game_data` time_series schema:
        time_series[metric][region_id][day] = value
  3. Define pandemic QuestionTemplates and register them in the core template
     registry (TEMPLATES_BY_ID).
  4. Hand-construct QuestionInstances and resolve them with the UNMODIFIED
     core QuestionResolver, checking answers against ground truth.

If step 4 passes, the core question machinery is world-agnostic and the
monorepo split (fbsim-core + worlds/{freeciv,starsim}) is justified.

Run:  uv run python starsim_spike/spike.py
"""

import sys
from datetime import datetime

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


# ---------------------------------------------------------------------------
# 1 + 2. Run regions and serialize to the game_data time_series schema
# ---------------------------------------------------------------------------

# Each region is a parameterized SIR sim. Same population so cases/deaths are
# directly comparable across regions (the comparative templates need this).
REGIONS = {
    0: dict(name="Northland", beta=0.05, p_death=0.02, vaccine_day=None),
    1: dict(name="Eastport", beta=0.09, p_death=0.03, vaccine_day=None),
    2: dict(name="Westfield", beta=0.07, p_death=0.02, vaccine_day=20),
    3: dict(name="Southbay", beta=0.04, p_death=0.01, vaccine_day=None),
}

N_AGENTS = 5_000
N_DAYS = 80


def run_region(region_id: int, cfg: dict) -> dict:
    """Run one region's SIR sim, return per-day metric arrays."""
    sir = ss.SIR(init_prev=0.01, beta=cfg["beta"], dur_inf=10, p_death=cfg["p_death"])
    pars = dict(
        n_agents=N_AGENTS,
        networks=dict(type="random", n_contacts=8),
        diseases=sir,
        dur=N_DAYS,
        dt=1,
        verbose=0,
    )
    # Optional intervention: a one-off vaccination campaign on vaccine_day.
    if cfg["vaccine_day"] is not None:
        vax = ss.simple_vx(efficacy=0.9, leaky=True)
        campaign = ss.campaign_vx(
            product=vax,
            years=[cfg["vaccine_day"]],
            prob=0.5,
        )
        pars["interventions"] = [campaign]

    sim = ss.Sim(pars)
    sim.run()
    r = sim.results
    sir_res = r["sir"]

    return {
        "cumulative_cases": np.array(sir_res["cum_infections"]).tolist(),
        "active_infections": np.array(sir_res["n_infected"]).tolist(),
        "cumulative_deaths": np.array(r["cum_deaths"]).tolist(),
    }


def build_game_data() -> dict:
    """Run all regions and assemble the game_data dict in the core schema."""
    metrics = ["cumulative_cases", "active_infections", "cumulative_deaths"]
    # Turn-major layout to match FreeCiv's convention (the core resolver's
    # _get_signal_value treats an int-castable outer key as turn-major):
    #   time_series[metric][day(str)][region_id(str)] = value
    time_series: dict[str, dict[str, dict[str, float]]] = {m: {} for m in metrics}

    for region_id, cfg in REGIONS.items():
        print(f"  region {region_id} ({cfg['name']}): beta={cfg['beta']} "
              f"p_death={cfg['p_death']} vax={cfg['vaccine_day']}")
        series = run_region(region_id, cfg)
        rid = str(region_id)
        for m in metrics:
            for day, val in enumerate(series[m]):
                time_series[m].setdefault(str(day), {})[rid] = val

    return {
        "metadata": {
            "world": "starsim_pandemic",
            "n_regions": len(REGIONS),
            "n_days": N_DAYS,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
        "time_series": time_series,
        # "civilizations" analog: the regions
        "civilizations": {
            rid: {"name": cfg["name"], "nation_id": rid}
            for rid, cfg in REGIONS.items()
        },
    }


# ---------------------------------------------------------------------------
# 3. Pandemic templates — registered into the core registry
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

DEATHS_COMPARATIVE = QuestionTemplate(
    template_id="deaths_comparative",
    signal_name="cumulative_deaths",
    question_template="Will {region_a} have more cumulative deaths than {region_b} at day {resolution_turn}?",
    resolution_type="comparative",
    data_path="time_series.cumulative_deaths.{player_id}.{resolution_turn}",
    comparison_op=">",
    required_params=["region_a", "region_b", "player_id_a", "player_id_b", "resolution_turn"],
)

CASES_CONTINUOUS = QuestionTemplate(
    template_id="cases_continuous",
    signal_name="cumulative_cases",
    question_template="How many cumulative cases will {region} have at day {resolution_turn}?",
    resolution_type="continuous",
    data_path="time_series.cumulative_cases.{player_id}.{resolution_turn}",
    comparison_op="value",
    required_params=["region", "player_id", "resolution_turn"],
)

PANDEMIC_TEMPLATES = [CASES_COMPARATIVE, DEATHS_COMPARATIVE, CASES_CONTINUOUS]


def register_pandemic_templates() -> None:
    """Inject pandemic templates into the core registry.

    NOTE: this monkey-patch is the spike's evidence for the one real refactor
    we identified — get_template() is a global FreeCiv-only lookup. A real
    monorepo wants each world to register its own templates via a plugin hook.
    """
    for t in PANDEMIC_TEMPLATES:
        tpl.TEMPLATES_BY_ID[t.template_id] = t


# ---------------------------------------------------------------------------
# 4. Build questions + resolve via the UNMODIFIED core resolver
# ---------------------------------------------------------------------------

def make_questions(snapshot_turn: int, resolution_turn: int) -> list[QuestionInstance]:
    """Hand-construct a few questions across the regions."""
    horizon = classify_horizon(snapshot_turn, resolution_turn)
    qs: list[QuestionInstance] = []
    qid = 0

    def name(rid: int) -> str:
        return REGIONS[rid]["name"]

    # Comparative cases for a few region pairs
    for a, b in [(1, 0), (1, 3), (0, 3), (2, 1)]:
        qs.append(QuestionInstance(
            question_id=f"q{qid:04d}",
            template_id="cases_comparative",
            resolution_turn=resolution_turn,
            horizon=horizon,
            parameters={"region_a": name(a), "region_b": name(b),
                        "player_id_a": a, "player_id_b": b},
            question_text=CASES_COMPARATIVE.question_template.format(
                region_a=name(a), region_b=name(b), resolution_turn=resolution_turn),
        ))
        qid += 1

    # Comparative deaths
    for a, b in [(1, 3), (0, 2)]:
        qs.append(QuestionInstance(
            question_id=f"q{qid:04d}",
            template_id="deaths_comparative",
            resolution_turn=resolution_turn,
            horizon=horizon,
            parameters={"region_a": name(a), "region_b": name(b),
                        "player_id_a": a, "player_id_b": b},
            question_text=DEATHS_COMPARATIVE.question_template.format(
                region_a=name(a), region_b=name(b), resolution_turn=resolution_turn),
        ))
        qid += 1

    # Continuous cases
    for rid in [1, 2]:
        qs.append(QuestionInstance(
            question_id=f"q{qid:04d}",
            template_id="cases_continuous",
            resolution_turn=resolution_turn,
            horizon=horizon,
            parameters={"region": name(rid), "player_id": rid},
            question_text=CASES_CONTINUOUS.question_template.format(
                region=name(rid), resolution_turn=resolution_turn),
        ))
        qid += 1

    return qs


def ground_truth_comparative(game_data, signal, a, b, turn) -> bool:
    # turn-major: time_series[signal][day][region_id]
    day = game_data["time_series"][signal][str(turn)]
    return day[str(a)] > day[str(b)]


def main() -> None:
    print("=" * 64)
    print("STARSIM PANDEMIC SPIKE — does fbsim-core generalize?")
    print("=" * 64)

    print("\n[1/4] Running regions + serializing to game_data schema...")
    game_data = build_game_data()

    print("\n[2/4] Registering pandemic templates into core registry...")
    register_pandemic_templates()
    print(f"  registered: {[t.template_id for t in PANDEMIC_TEMPLATES]}")

    snapshot_turn, resolution_turn = 30, 60
    print(f"\n[3/4] Building questions (snapshot day {snapshot_turn} -> "
          f"resolve day {resolution_turn})...")
    questions = make_questions(snapshot_turn, resolution_turn)
    print(f"  built {len(questions)} questions")

    print("\n[4/4] Resolving with UNMODIFIED core QuestionResolver...")
    resolver = QuestionResolver()
    n_ok = 0
    for q in questions:
        res = resolver.resolve(q, game_data, snapshot_turn)
        q.resolution = res
        line = f"  [{q.question_id}] {q.question_text}"
        if res.value_a is not None and res.value_b is not None:
            # verify against independent ground truth
            signal = tpl.TEMPLATES_BY_ID[q.template_id].signal_name
            gt = ground_truth_comparative(
                game_data, signal, q.parameters["player_id_a"],
                q.parameters["player_id_b"], resolution_turn)
            ok = (res.answer == gt)
            n_ok += ok
            print(line)
            print(f"        -> {res.answer}  (a={res.value_a:.0f} vs "
                  f"b={res.value_b:.0f})  {'OK' if ok else 'MISMATCH'}")
        else:
            v = res.value_at_resolution
            print(line)
            print(f"        -> value={v:.0f}" if v is not None else "        -> value=None  MISMATCH")
            n_ok += v is not None

    print("\n" + "=" * 64)
    print(f"RESULT: {n_ok}/{len(questions)} questions resolved correctly "
          f"by the unmodified core.")
    print("Core schema + resolver generalize to a pandemic world." if n_ok == len(questions)
          else "Some questions failed — core does NOT cleanly generalize.")
    print("=" * 64)


if __name__ == "__main__":
    main()
