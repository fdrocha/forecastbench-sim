"""Chart: Direction accuracy vs chance (left) and magnitude of update (right).

Shows models get direction partially right but magnitude is near zero.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "results"

DATASETS = {
    "republic": {
        "baseline": DATA_DIR / "republic_baseline_binary_all.json",
        "conditional": DATA_DIR / "republic_conditional_binary_all.json",
    },
    "gold500": {
        "baseline": DATA_DIR / "gold500_baseline_binary_all.json",
        "conditional": DATA_DIR / "gold500_conditional_binary_all.json",
    },
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def short_name(model_id: str) -> str:
    name = model_id.split('/')[-1]
    for suffix in ['-20251101', '-20250929', '-2025-04-14', '-2025-08-07',
                   '-2025-11-13', '-2025-04-16', '-preview']:
        name = name.replace(suffix, '')
    renames = {
        'claude-opus-4-5': 'Opus 4.5',
        'claude-sonnet-4-5': 'Sonnet 4.5',
        'gemini-2.5-flash': 'Gemini 2.5 Flash',
        'gemini-2.5-pro': 'Gemini 2.5 Pro',
        'gemini-3-pro': 'Gemini 3 Pro',
        'gpt-4.1': 'GPT-4.1',
        'gpt-5': 'GPT-5',
        'gpt-5-mini': 'GPT-5 mini',
        'gpt-5.1': 'GPT-5.1',
        'o3': 'o3',
    }
    return renames.get(name, name)


def collect_direction_magnitude():
    """Collect per-model direction accuracy and magnitude fractions, pooled across interventions."""
    model_stats = defaultdict(lambda: {"correct": 0, "wrong": 0, "no_pred": 0,
                                        "mag_fracs": []})

    for interv_name, paths in DATASETS.items():
        base_data = load_json(paths["baseline"])
        cond_data = load_json(paths["conditional"])
        base_by_id = {q["question_id"]: q for q in base_data["questions"]}

        for cond_q in cond_data["questions"]:
            if cond_q["question_type"] != "binary":
                continue

            base_id = cond_q["question_id"].replace("_intervention", "")
            base_q = base_by_id.get(base_id)
            if base_q is None:
                continue

            baseline_truth = base_q["ground_truth"]
            fork_truth = cond_q["ground_truth"]

            if baseline_truth == fork_truth:
                continue  # truth didn't change — skip

            truth_direction = 1 if fork_truth else -1

            for model_id, cond_pred in cond_q["predictions"].items():
                base_pred = base_q["predictions"].get(model_id)
                if base_pred is None:
                    continue

                p_base = base_pred["probability"]
                p_cond = cond_pred["probability"]
                if p_base is None or p_cond is None:
                    continue

                pred_delta = p_cond - p_base
                needed_shift = float(fork_truth) - float(baseline_truth)

                if abs(pred_delta) < 1e-6:
                    model_stats[model_id]["no_pred"] += 1
                    continue

                mag_frac = pred_delta / needed_shift if needed_shift != 0 else 0

                if (pred_delta > 0 and truth_direction > 0) or (pred_delta < 0 and truth_direction < 0):
                    model_stats[model_id]["correct"] += 1
                else:
                    model_stats[model_id]["wrong"] += 1

                model_stats[model_id]["mag_fracs"].append(mag_frac)

    return model_stats


# --- Compute ---
model_stats = collect_direction_magnitude()

results = []
for model_id, stats in model_stats.items():
    total = stats["correct"] + stats["wrong"]
    if total == 0:
        continue
    dir_acc = stats["correct"] / total
    med_mag = np.median(stats["mag_fracs"]) if stats["mag_fracs"] else 0
    # Conditional Brier for sorting (recompute quickly)
    cond_brier_scores = []
    for interv_name, paths in DATASETS.items():
        cond_data = load_json(paths["conditional"])
        for q in cond_data["questions"]:
            if q["question_type"] != "binary":
                continue
            pred = q["predictions"].get(model_id)
            if pred and pred.get("error") is None:
                cond_brier_scores.append((pred["probability"] - float(q["ground_truth"])) ** 2)
    cond_brier = np.mean(cond_brier_scores) if cond_brier_scores else 999

    results.append({
        "model_id": model_id,
        "label": short_name(model_id),
        "dir_acc": dir_acc,
        "med_mag": med_mag,
        "cond_brier": cond_brier,
    })

# Sort by conditional Brier ascending (consistent with other charts)
results.sort(key=lambda r: r["cond_brier"])

labels = [r["label"] for r in results]
dir_accs = [r["dir_acc"] for r in results]
med_mags = [r["med_mag"] for r in results]

# --- Plot ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
y = np.arange(len(labels))

# Left: Direction accuracy
bars = ax1.barh(y, dir_accs, height=0.6, color='#332288', edgecolor='none')
ax1.axvline(x=0.5, color='#CC6677', linewidth=2, linestyle='--', label='Chance (50%)')
ax1.set_yticks(y)
ax1.set_yticklabels(labels, fontsize=10)
ax1.set_xlabel('Direction accuracy', fontsize=11)
ax1.set_title('Direction: above chance', fontsize=11, fontweight='bold')
ax1.set_xlim(0, 0.75)
ax1.invert_yaxis()
ax1.legend(fontsize=9, loc='lower right')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.grid(axis='x', alpha=0.3, zorder=0)
ax1.tick_params(axis='both', which='both', length=4)

# Per-model percentage labels
for i, acc in enumerate(dir_accs):
    ax1.text(acc + 0.01, y[i], f'{acc:.0%}', ha='left', va='center', fontsize=8,
             color='#555555')

# Right: Magnitude (median fraction of needed shift)
bars2 = ax2.barh(y, med_mags, height=0.6, color='#DDCC77', edgecolor='none')
ax2.axvline(x=1.0, color='#CC6677', linewidth=2, linestyle='--', label='Full update')
ax2.set_yticks(y)
ax2.set_yticklabels(labels, fontsize=10)
ax2.set_xlabel('Median fraction of needed shift', fontsize=11)
ax2.set_title('Magnitude: near zero', fontsize=11, fontweight='bold')
ax2.set_xlim(0, 1.1)
ax2.invert_yaxis()
ax2.legend(fontsize=9, loc='lower right')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.grid(axis='x', alpha=0.3, zorder=0)
ax2.tick_params(axis='both', which='both', length=4)

# Per-model percentage labels
for i, mag in enumerate(med_mags):
    ax2.text(mag + 0.015, y[i], f'{mag:.0%}', ha='left', va='center', fontsize=8,
             color='#555555')

plt.tight_layout()

out_base = '/Users/elsehow/Projects/fri-vault/_artifacts/static/study2_direction_magnitude'
for ext in ['pdf', 'png']:
    plt.savefig(f'{out_base}.{ext}', bbox_inches='tight', dpi=300)
    print(f'Saved: {out_base}.{ext}')

# Summary
avg_dir = np.mean(dir_accs)
avg_mag = np.mean(med_mags)
print(f"\nAverage direction accuracy: {avg_dir:.1%}")
print(f"Average median magnitude: {avg_mag:.1%}")
