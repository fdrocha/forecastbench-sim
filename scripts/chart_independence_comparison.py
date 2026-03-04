"""Chart: Conditional performance vs independence baseline.

Shows that models are ~80% anchored to unconditional forecasts.
For each model: baseline Brier (floor), independence Brier (ceiling),
and conditional Brier (barely below independence).
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "results"


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


def collect_matched_brier():
    """Collect per-model Brier scores on matched question pairs, pooled across interventions.

    Returns dict: model_id -> {baseline: [...], conditional: [...], independence: [...]}
    where independence = baseline_prediction scored against fork_truth (Cross Brier).
    """
    model_scores = defaultdict(lambda: {"baseline": [], "conditional": [], "independence": []})

    for intervention in ['republic', 'gold500']:
        base_data = load_json(DATA_DIR / f"{intervention}_baseline_binary_all.json")
        cond_data = load_json(DATA_DIR / f"{intervention}_conditional_binary_all.json")

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

            for model_id, cond_pred in cond_q["predictions"].items():
                base_pred = base_q["predictions"].get(model_id)
                if base_pred is None:
                    continue

                p_base = base_pred["probability"]
                p_cond = cond_pred["probability"]
                if p_base is None or p_cond is None:
                    continue

                model_scores[model_id]["baseline"].append((p_base - float(baseline_truth)) ** 2)
                model_scores[model_id]["conditional"].append((p_cond - float(fork_truth)) ** 2)
                model_scores[model_id]["independence"].append((p_base - float(fork_truth)) ** 2)

    return model_scores


# --- Compute ---
model_scores = collect_matched_brier()
models = sorted(model_scores.keys())

results = []
for m in models:
    s = model_scores[m]
    results.append((
        m,
        short_name(m),
        np.mean(s["baseline"]),
        np.mean(s["conditional"]),
        np.mean(s["independence"]),
    ))

# Sort by conditional Brier (ascending)
results.sort(key=lambda r: r[3])

labels = [r[1] for r in results]
baseline = np.array([r[2] for r in results])
conditional = np.array([r[3] for r in results])
independence = np.array([r[4] for r in results])

# Compute pooled signal capture
pooled_improvement = np.mean(independence) - np.mean(conditional)
pooled_room = np.mean(independence) - np.mean(baseline)
pooled_pct = pooled_improvement / pooled_room * 100

# --- Plot ---
fig, ax = plt.subplots(figsize=(9, 5.5))

y = np.arange(len(labels))

# For each model, draw:
# 1. A gray bar from baseline to independence (total available room)
# 2. A colored bar from conditional to independence (captured signal)
# 3. Dots at each level

bar_h = 0.55

for i in range(len(labels)):
    # Full room: baseline to independence (light gray)
    ax.barh(y[i], independence[i] - baseline[i], height=bar_h,
            left=baseline[i], color='#e0e0e0', edgecolor='none', zorder=1)
    # Captured: conditional to independence (blue)
    ax.barh(y[i], independence[i] - conditional[i], height=bar_h,
            left=conditional[i], color='#332288', alpha=0.7, edgecolor='none', zorder=2)

# Dots
ax.scatter(baseline, y, color='black', s=40, zorder=4, label='Baseline')
ax.scatter(conditional, y, color='#332288', s=40, zorder=4, marker='D', label='Conditional')
ax.scatter(independence, y, color='#CC6677', s=40, zorder=4, marker='s', label='Independence P(Y)')

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Brier score', fontsize=11)
ax.invert_yaxis()

# Legend
legend_elements = [
    mpatches.Patch(facecolor='#e0e0e0', label='Unconditional performance gap'),
    mpatches.Patch(facecolor='#332288', alpha=0.7, label=f'Signal captured ({pooled_pct:.0f}% pooled)'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
               markersize=7, label='Unconditional Brier'),
    plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#332288',
               markersize=7, label='Conditional Brier'),
    plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#CC6677',
               markersize=7, label='Independence P(Y)'),
]
ax.legend(handles=legend_elements, frameon=False, fontsize=8.5, loc='upper right')

# Clean academic style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3, zorder=0)
ax.tick_params(axis='both', which='both', length=4)

# Per-model % captured annotation (right side)
for i in range(len(labels)):
    room = independence[i] - baseline[i]
    captured = independence[i] - conditional[i]
    pct = captured / room * 100 if room > 0 else 0
    ax.text(independence[i] + 0.005, y[i], f'{pct:.0f}%',
            ha='left', va='center', fontsize=8, color='#555555')

plt.tight_layout()

out_base = '/Users/elsehow/Projects/fri-vault/_artifacts/static/study2_independence_comparison'
for ext in ['pdf', 'png']:
    plt.savefig(f'{out_base}.{ext}', bbox_inches='tight', dpi=300)
    print(f'Saved: {out_base}.{ext}')

# Print summary
print(f"\nPooled: {pooled_pct:.1f}% of available signal captured")
print(f"  Independence Brier: {np.mean(independence):.4f}")
print(f"  Conditional Brier:  {np.mean(conditional):.4f}")
print(f"  Baseline Brier:     {np.mean(baseline):.4f}")
