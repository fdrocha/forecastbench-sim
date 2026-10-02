#!/usr/bin/env python
"""Every mid-range (bank) number the FreeCiv prose quotes, as LaTeX macros.

    python worlds/freeciv/fbsim_v3/paper/make_freeciv_midrange_macros.py --paper-root /path/to/paper

Run after update_shared_tables.py and make_freeciv_figs.py (it reads their freeciv_validation_stats.json and
freeciv_summary.json and checks its own numbers against them).  Writes data/freeciv_midrange_macros.tex and
data/freeciv/freeciv_midrange_numbers.json.  In a weighted run every mean, correlation and moment over bank items uses
the inverse-selection weights w; in an unweighted run it reproduces the numbers typed in the paper before October 2026.
"""
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import (BANK_WEIGHTS, DATA, PAPER_DATA, RUN, SCORE_ITEMS, WEIGHTED, load_capability, load_items, load_wide,
                     models_by_eci, rel, spearman_signed)

HZ = [90, 120, 150, 180, 210]
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
NAMES = {"meta-llama/llama-4-scout": "Llama 4 Scout", "google/gemini-3.7-flash": "Gemini 3.7 Flash", "google/gemini-3-flash-preview": "Gemini 3 Flash",
         "openai/gpt-5": "GPT-5", "openai/o3": "o3", "openai/gpt-5-nano": "GPT-5 Nano", "openai/gpt-5.5": "GPT-5.5", "google/gemini-2.5-flash": "Gemini 2.5 Flash"}
WORST_N = 3
UNDER_FAMILIES = ["NB6_event", "NB1_threshold"]   # "event and threshold questions"
WORST_NAMED = ["S7_any_destroyed", "S4_gov_change_count", "W4_lose_k"]   # "any city destroyed, the number of government changes, and cities lost"


def wavg(x, w):
    return float(np.average(np.asarray(x, float), weights=np.asarray(w, float)))


def wcorr(x, y, w):
    """Weighted Pearson correlation."""
    x, y, w = (np.asarray(v, float) for v in (x, y, w))
    mx, my = wavg(x, w), wavg(y, w)
    return float(wavg((x - mx) * (y - my), w) / np.sqrt(wavg((x - mx) ** 2, w) * wavg((y - my) ** 2, w)))


def decompose(f, q, w):
    """excess = squared bias + var f + var q - 2 cov(f, q), weighted moments."""
    mf, mq = wavg(f, w), wavg(q, w)
    return dict(bias2=(mf - mq) ** 2, var_f=wavg((f - mf) ** 2, w), var_q=wavg((q - mq) ** 2, w),
                cov2=2 * wavg((f - mf) * (q - mq), w), total=wavg((f - q) ** 2, w))


items = load_items()
b = items[items.set == "bank"].copy()
assert b.item.nunique() == 750 and b.model.nunique() == 24
b["e"] = b.p - b.q
cap = load_capability()
models = models_by_eci(load_wide())          # descending ECI, the order the bootstraps resample in
eci = cap.eci.reindex(models)
g = b.groupby("model")
ex = g.apply(lambda d: wavg(d.excess_brier, d.w), include_groups=False).reindex(models)
bias = g.apply(lambda d: wavg(d.e, d.w), include_groups=False).reindex(models)
disc = g.apply(lambda d: wcorr(d.p, d.q, d.w), include_groups=False).reindex(models)
uq = b.drop_duplicates("item")
flat = wavg((0.5 - uq.q) ** 2, uq.w)
horizon = {T: float(np.mean([wavg(d.excess_brier, d.w) for _, d in b[b["T"] == T].groupby("model")])) for T in HZ}

VAL = {r["column"]: r for r in json.load(open(DATA / "freeciv_validation_stats.json"))["rows"]}
SUM = json.load(open(DATA / "freeciv_summary.json"))
mid = VAL["bank_all_excess_brier"]
assert max(abs(mid["per_model_score"][m] - ex[m]) for m in models) < 1e-9, "validation stats are not from this run"
assert np.allclose([horizon[T] for T in HZ], SUM["fig_freeciv_horizon"]["bank"]["mean_over_models"], atol=1e-12)
rho_disc, p_disc, (disc_lo, disc_hi), _ = spearman_signed(eci.values, disc.values, lower_is_better=False)

# reliability figure: pooled bands and per-model bias, as the figure draws them
rel_ = SUM["fig_freeciv_reliability"]
pq, pp = rel_["pooled"]["q"], rel_["pooled"]["p"]
above = [i for i in range(len(pq)) if pp[i] > pq[i]]
k = len(above)
if above == list(range(k)):
    where = f"they lie above the truth in the {WORDS[k]} lowest intervals and below it in the {WORDS[10 - k]} highest"
elif above == list(range(10 - k, 10)):
    where = f"they lie below the truth in the {WORDS[10 - k]} lowest intervals and above it in the {WORDS[k]} highest"
else:
    where = f"they lie above the truth in {WORDS[k]} of the ten intervals"
assert np.allclose([rel_["per_model_bias_bank"][m] for m in models], bias.values, atol=1e-12)

# forecast-binned rows and the decomposition (per model, then averaged over models)
low, high = b[b.p < 0.1], b[b.p >= 0.9]
dec = pd.DataFrame([decompose(d.p.values, d.q.values, d.w.values) for _, d in g]).mean()

# families: pooled over models, weighted
fam = b.groupby("family").apply(lambda d: pd.Series(dict(items=d.item.nunique(), bias=wavg(d.e, d.w), disc=wcorr(d.p, d.q, d.w) if d.q.nunique() > 1 else np.nan,
                                                        loss=float((d.excess_brier * d.w).sum()))), include_groups=False)
fh = pd.read_csv(DATA / "family_horizon_scores.csv")
fh = fh[fh.set == "bank"].copy()
fh["wt"] = fh["wsum"] if "wsum" in fh.columns else fh["n"]
order = fh.groupby("family").apply(lambda d: (d.wt * d.excess).sum() / d.wt.sum(), include_groups=False).sort_values(ascending=False)   # the difficulty table's order
worst = list(order.index[:WORST_N])
assert worst == WORST_NAMED, f"the prose names {WORST_NAMED} as the hardest families, now {worst}"
comp = fam.loc["EX_comparative"]
assert comp.disc == fam.disc.max(), "comparative questions are no longer the most discriminated family"
under = fam.loc[UNDER_FAMILIES, "bias"]

fb = VAL["bank_all_excess_brier"]["fb"]
cl = VAL["bank_all_excess_brier"]["eci_cluster"]
rhos = [VAL[c]["eci"]["rho"] for c in VAL]
N = dict(
    run=RUN, weighted=WEIGHTED, weights=rel(BANK_WEIGHTS) if WEIGHTED else None, generated=str(date.today()), source=rel(SCORE_ITEMS),
    mean_over_models=float(ex.mean()), per_model=ex.to_dict(), best=ex.idxmin(), worst=ex.idxmax(),
    eci=dict(rho=mid["eci"]["rho"], ci=[mid["eci"]["ci_lo"], mid["eci"]["ci_hi"]], p=mid["eci"]["p"], games_ci=[cl["ci_lo"], cl["ci_hi"]]),
    fb=dict(rho=fb["rho"], ci=[fb["ci_lo"], fb["ci_hi"]], p=fb["p"], n=fb["n"]),
    freeciv_eci_rho_range=[min(rhos), max(rhos)],
    flat_0p5=flat, models_below_flat=int((ex < flat).sum()), models_negative_bias=int((bias < 0).sum()), bias_range=[float(bias.min()), float(bias.max())],
    discrimination=dict(per_model=disc.to_dict(), min=float(disc.min()), min_model=disc.idxmin(), max=float(disc.max()), max_model=disc.idxmax(),
                        rho_eci=rho_disc, ci=[disc_lo, disc_hi], p=p_disc),
    rho_bias_eci=float(spearmanr(eci, bias)[0]), rho_excess_absbias=float(spearmanr(ex, bias.abs())[0]), rho_excess_disc=float(spearmanr(ex, disc)[0]),
    horizon_mean=horizon,
    reliability=dict(p_low=pp[0], q_low=pq[0], p_high=pp[-1], q_high=pq[-1], where=where),
    forecast_bins=dict(low=dict(f=wavg(low.p, low.w), q=wavg(low.q, low.w), rows=len(low)), high=dict(f=wavg(high.p, high.w), q=wavg(high.q, high.w), rows=len(high))),
    decomposition=dec.to_dict(), decomposition_bias_share=float(dec.bias2 / dec.total),
    families=fam.to_dict(orient="index"), worst_families=worst, worst_items=int(fam.loc[worst, "items"].sum()), worst_loss_share=float(fam.loc[worst, "loss"].sum() / fam.loss.sum()),
    worst_weight_share=float(uq[uq.family.isin(worst)].w.sum() / uq.w.sum()),
    comparative=dict(bias=float(comp.bias), disc=float(comp.disc)), under_families=under.to_dict(),
    mean_q_selected=float(uq.q.mean()), mean_q_weighted=wavg(uq.q, uq.w),
    effective_items=float(uq.w.sum() ** 2 / (uq.w ** 2).sum()),
    pool_total=int(pd.read_csv(BANK_WEIGHTS).drop_duplicates(["band", "T", "family"]).n_pool.sum()) if WEIGHTED else None,
    combinations=int(len(pd.read_csv(BANK_WEIGHTS).drop_duplicates(["band", "T", "family"]))) if WEIGHTED else None, max_weight_ratio=float(uq.w.max() / uq.w.median()),
    weight_share_by_band={str(k_): float(v) for k_, v in (uq.groupby(pd.cut(uq.q, [0.05, 0.23, 0.41, 0.59, 0.77, 0.95]), observed=False).w.sum() / uq.w.sum()).items()},
)
(DATA / "freeciv_midrange_numbers.json").write_text(json.dumps(N, indent=1, default=float))


MINUS, PLUS = r"\ensuremath{-}", r"\ensuremath{+}"   # valid in text and in math


def f2(x):
    return f"{x:.2f}".replace("-", MINUS)


def f3(x):
    return f"{x:.3f}".replace("-", MINUS)


def signed2(x):
    return (PLUS if x >= 0 else MINUS) + f"{abs(x):.2f}"


M = {
    "FCMidMean": f3(N["mean_over_models"]),
    "FCMidBest": f3(ex.min()), "FCMidBestModel": NAMES.get(N["best"], N["best"]), "FCMidWorst": f3(ex.max()), "FCMidWorstModel": NAMES.get(N["worst"], N["worst"]),
    "FCMidRho": f2(N["eci"]["rho"]), "FCMidRhoLo": f2(N["eci"]["ci"][0]), "FCMidRhoHi": f2(N["eci"]["ci"][1]), "FCMidRhoP": f"{N['eci']['p']:.3f}",
    "FCMidRhoGamesLo": f2(N["eci"]["games_ci"][0]), "FCMidRhoGamesHi": f2(N["eci"]["games_ci"][1]),
    "FCMidFBRho": f2(N["fb"]["rho"]), "FCMidFBLo": f2(N["fb"]["ci"][0]), "FCMidFBHi": f2(N["fb"]["ci"][1]), "FCMidFBP": f"{N['fb']['p']:.3f}",
    "FCRhoMin": f2(N["freeciv_eci_rho_range"][0]), "FCRhoMax": f2(N["freeciv_eci_rho_range"][1]),
    "FCMidFlat": f3(flat), "FCMidBelowFlat": str(N["models_below_flat"]), "FCMidNegBias": str(N["models_negative_bias"]),
    "FCMidBiasMin": f2(N["bias_range"][0]), "FCMidBiasMax": f2(N["bias_range"][1]),
    "FCMidDiscMin": f2(disc.min()), "FCMidDiscMinModel": NAMES.get(disc.idxmin(), disc.idxmin()), "FCMidDiscMax": f2(disc.max()), "FCMidDiscMaxModel": NAMES.get(disc.idxmax(), disc.idxmax()),
    "FCMidDiscRho": f2(rho_disc), "FCMidDiscLo": f2(disc_lo), "FCMidDiscHi": f2(disc_hi),
    "FCMidBiasEciRho": f2(N["rho_bias_eci"]), "FCMidExcessAbsBiasRho": f2(N["rho_excess_absbias"]), "FCMidExcessDiscRho": f2(N["rho_excess_disc"]),
    "FCMidCompBias": signed2(comp.bias), "FCMidCompDisc": f2(comp.disc), "FCMidUnderLo": f2(-under.max()), "FCMidUnderHi": f2(-under.min()),
    "FCMidHorizonLo": f3(min(horizon.values())), "FCMidHorizonHi": f3(max(horizon.values())), "FCMidHorizonFirst": f3(horizon[90]), "FCMidHorizonLast": f3(horizon[210]),
    "FCMidRelPLow": f2(pp[0]), "FCMidRelQLow": f2(pq[0]), "FCMidRelPHigh": f2(pp[-1]), "FCMidRelQHigh": f2(pq[-1]), "FCMidRelWhere": where,
    "FCMidLowF": f3(N["forecast_bins"]["low"]["f"]), "FCMidLowQ": f3(N["forecast_bins"]["low"]["q"]), "FCMidLowRows": f"{len(low):,}",
    "FCMidHighF": f3(N["forecast_bins"]["high"]["f"]), "FCMidHighQ": f3(N["forecast_bins"]["high"]["q"]), "FCMidHighRows": f"{len(high):,}",
    "FCMidDecTotal": f3(dec.total), "FCMidDecBias": f3(dec.bias2), "FCMidDecVarF": f3(dec.var_f), "FCMidDecVarQ": f3(dec.var_q), "FCMidDecCov": f3(dec.cov2),
    "FCMidDecBiasShare": f"{100 * N['decomposition_bias_share']:.0f}\\%",
    "FCMidWorstItems": str(N["worst_items"]), "FCMidWorstShare": f"{100 * N['worst_loss_share']:.0f}\\%", "FCMidWorstWeightShare": f"{100 * N['worst_weight_share']:.0f}\\%",
    "FCMidMeanQ": f3(N["mean_q_selected"]), "FCMidMeanQWeighted": f3(N["mean_q_weighted"]),
    "FCMidPoolTotal": f"{N['pool_total']:,}" if WEIGHTED else "--",
    "FCMidCombinations": f"{N['combinations']:,}" if WEIGHTED else "--",
    "FCMidEffItems": f"{N['effective_items']:.0f}", "FCMidMaxWeightRatio": f"{N['max_weight_ratio']:.0f}",
}
lines = [f"% Generated by worlds/freeciv/fbsim_v3/paper/make_freeciv_midrange_macros.py (forecastbench-sim) from results/{RUN}; do not edit by hand.",
         "% FreeCiv mid-range (bank) numbers quoted in the text" + (", weighted by inverse selection probability (data/freeciv/bank_weights.csv)." if WEIGHTED else ", unweighted.")]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
(PAPER_DATA / "freeciv_midrange_macros.tex").write_text("\n".join(lines) + "\n")
print(f"wrote {rel(PAPER_DATA / 'freeciv_midrange_macros.tex')} ({len(M)} macros) and {rel(DATA / 'freeciv_midrange_numbers.json')}")
for k_, v in M.items():
    print(f"  {k_:24s} {v}")
