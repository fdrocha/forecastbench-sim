"""Chart: Murphy decomposition of conditional gap, pooled Republic + Gold.

Horizontal stacked bars sorted by conditional Brier (ascending),
matching the orientation of chart_independence_comparison.py.
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "results"

MODEL_NAMES = {
    "openai/o3-2025-04-16": "o3",
    "openai/gpt-5.1-2025-11-13": "GPT-5.1",
    "openai/gpt-5-2025-08-07": "GPT-5",
    "openai/gpt-5-mini-2025-08-07": "GPT-5 mini",
    "openai/gpt-4.1-2025-04-14": "GPT-4.1",
    "google/gemini-3-pro-preview": "Gemini 3 Pro",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "anthropic/claude-opus-4-5-20251101": "Opus 4.5",
    "anthropic/claude-sonnet-4-5-20250929": "Sonnet 4.5",
}

N_BINS = 10


def load_predictions(eval_path, model_id):
    data = json.load(open(eval_path))
    preds, truths = [], []
    for q in data["questions"]:
        pred = q["predictions"].get(model_id)
        if pred and pred.get("error") is None:
            preds.append(pred["probability"])
            truths.append(1.0 if q["ground_truth"] else 0.0)
    return np.array(preds), np.array(truths)


def murphy_decomposition(preds, truths, n_bins=N_BINS):
    n = len(preds)
    base_rate = truths.mean()
    unc = base_rate * (1 - base_rate)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    rel = res = 0.0
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        mask = (preds >= lo) & (preds <= hi) if i == n_bins - 1 else (preds >= lo) & (preds < hi)
        n_b = mask.sum()
        if n_b == 0:
            continue
        f_b = truths[mask].mean()
        p_b = preds[mask].mean()
        rel += n_b * (f_b - p_b) ** 2
        res += n_b * (f_b - base_rate) ** 2
    rel /= n
    res /= n
    brier = ((preds - truths) ** 2).mean()
    return {"brier": brier, "rel": rel, "res": res, "unc": unc, "n": n}


# --- Compute pooled decomposition ---
interventions = {
    "republic": {
        "baseline": DATA_DIR / "republic_baseline_binary_all.json",
        "conditional": DATA_DIR / "republic_conditional_binary_all.json",
    },
    "gold500": {
        "baseline": DATA_DIR / "gold500_baseline_binary_all.json",
        "conditional": DATA_DIR / "gold500_conditional_binary_all.json",
    },
}

results = {}
for model_id, short in MODEL_NAMES.items():
    # Pool predictions across both interventions
    bl_preds_all, bl_truths_all = [], []
    cd_preds_all, cd_truths_all = [], []

    for interv, paths in interventions.items():
        bl_p, bl_t = load_predictions(paths["baseline"], model_id)
        cd_p, cd_t = load_predictions(paths["conditional"], model_id)
        bl_preds_all.append(bl_p)
        bl_truths_all.append(bl_t)
        cd_preds_all.append(cd_p)
        cd_truths_all.append(cd_t)

    bl_preds = np.concatenate(bl_preds_all)
    bl_truths = np.concatenate(bl_truths_all)
    cd_preds = np.concatenate(cd_preds_all)
    cd_truths = np.concatenate(cd_truths_all)

    bl_decomp = murphy_decomposition(bl_preds, bl_truths)
    cd_decomp = murphy_decomposition(cd_preds, cd_truths)

    d_brier = cd_decomp["brier"] - bl_decomp["brier"]
    d_rel = cd_decomp["rel"] - bl_decomp["rel"]
    d_res = -(cd_decomp["res"] - bl_decomp["res"])
    d_unc = cd_decomp["unc"] - bl_decomp["unc"]

    results[short] = {
        "cond_brier": cd_decomp["brier"],
        "d_brier": d_brier,
        "d_rel": d_rel,
        "d_res": d_res,
        "d_unc": d_unc,
        "pct_rel": d_rel / d_brier * 100 if abs(d_brier) > 0.001 else 0,
        "pct_unc": d_unc / d_brier * 100 if abs(d_brier) > 0.001 else 0,
    }

# Sort by conditional Brier ascending (matching independence comparison chart)
sorted_models = sorted(results.keys(), key=lambda m: results[m]["cond_brier"])

labels = sorted_models
d_rel = np.array([results[m]["d_rel"] for m in sorted_models])
d_res = np.array([results[m]["d_res"] for m in sorted_models])
d_unc = np.array([results[m]["d_unc"] for m in sorted_models])
pct_rel = np.array([results[m]["pct_rel"] for m in sorted_models])

# --- Plot ---
c_rel = '#332288'   # indigo
c_res = '#88CCEE'   # cyan
c_unc = '#DDCC77'   # sand

fig, ax = plt.subplots(figsize=(9, 5.5))
y = np.arange(len(labels))
bar_h = 0.6

# Stacked horizontal bars
ax.barh(y, d_rel, height=bar_h, color=c_rel, edgecolor='none',
        label=r'$\Delta$REL (miscalibration)')
ax.barh(y, d_res, height=bar_h, left=d_rel, color=c_res, edgecolor='none',
        label=r'$-\Delta$RES (resolution)')
ax.barh(y, d_unc, height=bar_h, left=d_rel + d_res, color=c_unc, edgecolor='none',
        label=r'$\Delta$UNC (difficulty)')

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=10)
ax.set_xlabel('Brier score gap (conditional \u2212 unconditional)', fontsize=11)
ax.invert_yaxis()

# Per-model %REL annotation
for i in range(len(labels)):
    total = d_rel[i] + d_res[i] + d_unc[i]
    ax.text(total + 0.003, y[i], f'{pct_rel[i]:.0f}% miscal.',
            ha='left', va='center', fontsize=8, color='#555555')

ax.legend(frameon=False, fontsize=9, loc='upper right')

# Clean academic style
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.grid(axis='x', alpha=0.3, zorder=0)
ax.tick_params(axis='both', which='both', length=4)

plt.tight_layout()

out_base = '/Users/elsehow/Projects/fri-vault/_artifacts/static/study2_brier_decomposition_pooled'
for ext in ['pdf', 'png']:
    plt.savefig(f'{out_base}.{ext}', bbox_inches='tight', dpi=300)
    print(f'Saved: {out_base}.{ext}')

# Summary
avg_pct_rel = np.mean(pct_rel)
avg_pct_unc = np.mean([results[m]["pct_unc"] for m in sorted_models])
print(f"\nPooled average: {avg_pct_rel:.0f}% ΔREL, {avg_pct_unc:.0f}% ΔUNC")
