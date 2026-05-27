"""Score agentic baseline responses against answer keys.

Usage:
    cd /Users/elsehow/Projects/civbench
    uv run python scripts/score_agentic_baseline.py
"""
import json
import os
import re
import sys

sys.path.insert(0, ".")
from fbsim_core.metrics import compute_crps, compute_mae


RESULTS_DIR = "data/results/agentic_baseline"
SEEDS = ["seed0", "seed1", "seed4", "seed5", "seed9",
         "seed10", "seed13", "seed15", "seed16", "seed20"]
MODELS = ["gpt-4.1", "claude-sonnet-4-5", "gpt-5",
          "gemini-3-pro-preview", "claude-opus-4-6"]
TEMPLATES = ["cities_count", "population", "territory", "treasury"]


def parse_percentiles(text):
    """Extract first <<<PERCENTILES>>>...<<<END>>> block from response text."""
    match = re.search(r"<<<PERCENTILES>>>(.*?)<<<END>>>", text, re.DOTALL)
    if not match:
        return None
    questions = {}
    for line in match.group(1).strip().split("\n"):
        m = re.match(
            r"Q(\d+):\s*p10=(-?[\d.]+),\s*p25=(-?[\d.]+),\s*p50=(-?[\d.]+),\s*p75=(-?[\d.]+),\s*p90=(-?[\d.]+)",
            line.strip(),
        )
        if m:
            qnum = int(m.group(1))
            questions[qnum] = {
                "p10": float(m.group(2)),
                "p25": float(m.group(3)),
                "p50": float(m.group(4)),
                "p75": float(m.group(5)),
                "p90": float(m.group(6)),
            }
    return questions


def load_response(model, seed):
    """Load response text, preferring sandbox file (Opus writes answers there)."""
    for suffix in ["_sandbox.txt", ".txt"]:
        fpath = os.path.join(RESULTS_DIR, f"response_{model}_{seed}{suffix}")
        if os.path.exists(fpath):
            with open(fpath) as f:
                text = f.read()
            if "<<<PERCENTILES>>>" in text:
                return text, fpath
    # Fallback to regular file even without percentiles
    fpath = os.path.join(RESULTS_DIR, f"response_{model}_{seed}.txt")
    with open(fpath) as f:
        return f.read(), fpath


def main():
    # Load answer keys
    answer_keys = {}
    for seed in SEEDS:
        with open(os.path.join(RESULTS_DIR, f"answer_key_{seed}.json")) as f:
            answer_keys[seed] = json.load(f)

    # Score each model
    all_results = {}
    for model in MODELS:
        by_template = {}
        by_horizon = {}
        calibration = {q: 0 for q in ["p10", "p25", "p50", "p75", "p90"]}
        total_qs = 0

        for seed in SEEDS:
            text, fpath = load_response(model, seed)
            pcts = parse_percentiles(text)
            if not pcts:
                print(f"WARN: no percentiles in {fpath}")
                continue

            for q in answer_keys[seed]:
                qnum = q["question_number"]
                if qnum not in pcts:
                    continue
                gt = q["ground_truth"]
                t = q["template_id"].replace("_continuous", "")
                h = q["horizon"]
                p = pcts[qnum]

                crps = compute_crps(p, gt)
                mae = compute_mae(p["p50"], gt)

                for key in [t, f"{t}_{h}"]:
                    if key not in by_template:
                        by_template[key] = {"crps": [], "mae": []}
                    by_template[key]["crps"].append(crps)
                    by_template[key]["mae"].append(mae)

                if h not in by_horizon:
                    by_horizon[h] = {"crps": [], "mae": []}
                by_horizon[h]["crps"].append(crps)
                by_horizon[h]["mae"].append(mae)

                total_qs += 1
                for pkey in calibration:
                    if gt <= p[pkey]:
                        calibration[pkey] += 1

        # Compute averages
        all_crps = []
        all_mae = []
        for t in TEMPLATES:
            if t in by_template:
                all_crps.extend(by_template[t]["crps"])
                all_mae.extend(by_template[t]["mae"])

        avg_crps = sum(all_crps) / len(all_crps) if all_crps else float("nan")
        avg_mae = sum(all_mae) / len(all_mae) if all_mae else float("nan")

        all_results[model] = {
            "crps": avg_crps,
            "mae": avg_mae,
            "n": total_qs,
            "by_template": {
                t: {
                    "crps": sum(by_template[t]["crps"]) / len(by_template[t]["crps"]),
                    "mae": sum(by_template[t]["mae"]) / len(by_template[t]["mae"]),
                }
                for t in TEMPLATES
                if t in by_template
            },
            "by_horizon": {
                h: {
                    "crps": sum(v["crps"]) / len(v["crps"]),
                    "mae": sum(v["mae"]) / len(v["mae"]),
                }
                for h, v in sorted(by_horizon.items())
            },
            "calibration": {
                k: v / total_qs if total_qs > 0 else float("nan")
                for k, v in calibration.items()
            },
        }

    # Print summary
    print(f"{'Model':<30} {'CRPS':>8} {'MAE':>8}  |  cities   pop     terr    treas")
    print("-" * 85)
    for model, r in all_results.items():
        tmpl = r["by_template"]
        print(
            f"{model:<30} {r['crps']:>8.1f} {r['mae']:>8.1f}  | "
            + "  ".join(f"{tmpl.get(t, {}).get('crps', float('nan')):>6.1f}" for t in TEMPLATES)
        )

    print()
    print(f"{'Model':<30}  p10    p25    p50    p75    p90   (ideal: 10%  25%  50%  75%  90%)")
    print("-" * 75)
    for model, r in all_results.items():
        cal = r["calibration"]
        print(
            f"{model:<30} "
            + "  ".join(f"{cal[k]*100:>4.0f}%" for k in ["p10", "p25", "p50", "p75", "p90"])
        )

    # Save JSON
    outpath = os.path.join(RESULTS_DIR, "scored_results.json")
    with open(outpath, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {outpath}")


if __name__ == "__main__":
    main()
