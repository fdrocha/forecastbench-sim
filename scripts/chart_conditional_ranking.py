"""Chart: Conditional Brier ranking with bootstrap 95% CIs and tier annotations.

Shows that models form a clear 4-tier ranking on conditional questions —
confidence intervals don't all overlap.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "results"

# --- Load and match questions across both interventions ---

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


def collect_conditional_brier_scores():
    """Collect per-question conditional Brier scores for each model, pooled across interventions."""
    model_scores = defaultdict(list)

    for intervention in ['republic', 'gold500']:
        cond_data = load_json(DATA_DIR / f"{intervention}_conditional_binary_all.json")

        for q in cond_data["questions"]:
            if q["question_type"] != "binary":
                continue
            truth = q["ground_truth"]
            for model_id, pred in q["predictions"].items():
                p = pred["probability"]
                if p is None:
                    continue
                brier = (p - float(truth)) ** 2
                model_scores[model_id].append(brier)

    return model_scores


def bootstrap_ci(scores, n_boot=10000, alpha=0.05):
    """Bootstrap 95% CI for the mean."""
    scores = np.array(scores)
    n = len(scores)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        scores[rng.integers(0, n, size=n)].mean()
        for _ in range(n_boot)
    ])
    lo = np.percentile(boot_means, 100 * alpha / 2)
    hi = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lo, hi


# --- Compute ---
model_scores = collect_conditional_brier_scores()
models = sorted(model_scores.keys())

results = []
for m in models:
    scores = model_scores[m]
    mean_brier = np.mean(scores)
    lo, hi = bootstrap_ci(scores)
    results.append((m, short_name(m), mean_brier, lo, hi))

# Sort by conditional Brier (ascending = best first)
results.sort(key=lambda r: r[2])

labels = [r[1] for r in results]
means = np.array([r[2] for r in results])
los = np.array([r[3] for r in results])
his = np.array([r[4] for r in results])

# --- Tier definitions (from experiment writeup) ---
tiers = [
    (0, 4, 'Tier 1'),   # o3, gpt-5-mini, opus 4.5, sonnet 4.5
    (4, 6, 'Tier 2'),   # gpt-5.1, gpt-5
    (6, 9, 'Tier 3'),   # gemini-3-pro, gemini-2.5-pro, gemini-2.5-flash
    (9, 10, 'Tier 4'),  # gpt-4.1
]
tier_colors = ['#e8f0fe', '#fff3e0', '#fce4ec', '#f3e5f5']

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))

y = np.arange(len(labels))

# Tier background shading
for (start, end, tier_label), color in zip(tiers, tier_colors):
    ax.axhspan(start - 0.5, end - 0.5, color=color, alpha=0.5, zorder=0)
    # Tier label on right
    mid = (start + end - 1) / 2
    ax.text(max(his) + 0.006, mid, tier_label,
            ha='left', va='center', fontsize=9, fontstyle='italic', color='#555555')

# Error bars
ax.errorbar(means, y, xerr=[means - los, his - means],
            fmt='o', color='#332288', ecolor='#332288', elinewidth=1.5,
            capsize=4, capthick=1.5, markersize=7, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Conditional Brier score (pooled Republic + Gold, 95% CI)', fontsize=11)
ax.invert_yaxis()

# Clean academic style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3, zorder=0)
ax.tick_params(axis='both', which='both', length=4)

# Pairwise separation count
n_pairs = len(results) * (len(results) - 1) // 2
separated = 0
for i in range(len(results)):
    for j in range(i + 1, len(results)):
        if his[i] < los[j] or his[j] < los[i]:
            separated += 1

ax.text(0.02, 0.02, f'{separated}/{n_pairs} pairwise CIs separated',
        transform=ax.transAxes, fontsize=8, color='#555555',
        ha='left', va='bottom')

plt.tight_layout()

out_base = '/Users/elsehow/Projects/fri-vault/_artifacts/static/conditional_ranking_with_cis'
for ext in ['pdf', 'png']:
    plt.savefig(f'{out_base}.{ext}', bbox_inches='tight', dpi=300)
    print(f'Saved: {out_base}.{ext}')
