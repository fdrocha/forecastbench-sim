"""Export micropolis conditional flip-case prompts for external interp probing.

A "flip case" is a (scenario, horizon) where the intervention flips the ground-truth
answer to "Will R have more population than S?": trend says YES (R has higher
growth rate), but the intervention makes it NO. These are the cases where
trend-extrapolation and intervention-knowledge disagree.

Emits, per flip case, the SAME question with two contexts (a minimal pair):
  - context_unconditional: situation report, no intervention   (GT: YES)
  - context_conditional:   same report + policy sentence        (GT: NO)

TODO: straight port of pandemic_world.export_probe_corpus — revisit field names
once the real Micropolis scenario/metric semantics are implemented.

Usage:
    uv run python -m micropolis_world.export_probe_corpus \
        --out /path/to/micropolis_flip_cases.json
"""

import argparse
import json
from pathlib import Path

from .scenarios import sample_scenarios, build_corpus


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-scenarios", type=int, default=30)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-cases", type=int, default=15)
    args = ap.parse_args()

    scenarios = sample_scenarios(args.n_scenarios, args.seed)
    corpus = build_corpus(scenarios)

    # Pair unconditional + conditional by (scenario_id, horizon)
    pairs: dict[tuple, dict] = {}
    for c in corpus:
        pairs.setdefault((c["scenario_id"], c["horizon"]), {})[c["kind"]] = c

    flip_cases = []
    for (sid, T), kinds in sorted(pairs.items()):
        u, cnd = kinds.get("unconditional"), kinds.get("conditional")
        if not (u and cnd):
            continue
        # Trend-vs-intervention disagreement, intervention-correct direction:
        # unconditional YES (R exceeds S) -> conditional NO (policy suppresses R).
        if u["ground_truth"] is True and cnd["ground_truth"] is False:
            flip_cases.append({
                "scenario_id": sid,
                "horizon": T,
                "question_text": u["question_text"],
                "context_unconditional": u["context"],
                "context_conditional": cnd["context"],
                "gt_unconditional": u["ground_truth"],   # True  (YES)
                "gt_conditional": cnd["ground_truth"],    # False (NO)
                "value_R_unconditional": u["value_R"],
                "value_R_conditional": cnd["value_R"],
                "value_S": u["value_S"],
                "scenario": u["scenario"],
            })

    flip_cases = flip_cases[: args.max_cases]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "description": "Micropolis conditional flip-cases: trend says YES (R exceeds S), "
                       "policy intervention flips ground truth to NO. Minimal pair = "
                       "same question, context with/without the intervention sentence.",
        "n_cases": len(flip_cases),
        "cases": flip_cases,
    }, indent=2))
    print(f"[saved] {len(flip_cases)} flip cases -> {out}")
    for fc in flip_cases:
        print(f"  s{fc['scenario_id']} T{fc['horizon']}: "
              f"R {fc['value_R_unconditional']:.0f}->{fc['value_R_conditional']:.0f} "
              f"vs S {fc['value_S']:.0f}  "
              f"(policy mag={fc['scenario']['policy_magnitude']} turn={fc['scenario']['policy_turn']})")


if __name__ == "__main__":
    main()
