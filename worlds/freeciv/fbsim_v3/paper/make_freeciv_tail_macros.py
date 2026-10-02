#!/usr/bin/env python
"""Every tail-set number the FreeCiv prose quotes, as LaTeX macros.

    python worlds/freeciv/fbsim_v3/paper/make_freeciv_tail_macros.py --paper-root /path/to/paper

Run after update_shared_tables.py and make_freeciv_figs.py (it reads freeciv_validation_stats.json and
freeciv_summary.json and checks its numbers against them).  Writes data/freeciv_tail_macros.tex and
data/freeciv/freeciv_tail_numbers.json.  In a run with tail weights every mean over tail items uses the
inverse-selection weights w; unweighted, it reproduces the numbers typed in the paper before October 2026.
"""
import json
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from _common import (DATA, PAPER_DATA, RUN, SCORE_ITEMS, TAILS_WEIGHTED, TAILS_WEIGHTS, load_capability, load_items, load_wide,
                     models_by_eci, rel)

HZ = [90, 120, 150, 180, 210]
WORST_N = 3
WORST_NAMED = ["EX_government_at", "W6_peace_at", "NW5_wonder"]   # "a government in place at the horizon, a peace in effect, a wonder completed"
MINUS = r"\ensuremath{-}"


def wavg(x, w):
    return float(np.average(np.asarray(x, float), weights=np.asarray(w, float)))


def kl_bits(q, p):
    q = np.clip(np.asarray(q, float), 1e-9, 1 - 1e-9)
    return q * np.log2(q / p) + (1 - q) * np.log2((1 - q) / (1 - p))


def f2(x):
    return f"{x:.2f}".replace("-", MINUS)


items = load_items()
t = items[items.set == "tails"].copy()
assert t.item.nunique() == 300 and t.model.nunique() == 24
models = models_by_eci(load_wide())
eci = load_capability().eci.reindex(models)
g = t.groupby("model")
bits = g.apply(lambda d: wavg(d.excess_bits, d.w), include_groups=False).reindex(models)
meanf = g.apply(lambda d: wavg(d.p, d.w), include_groups=False).reindex(models)
uq = t.drop_duplicates("item")
flat05 = wavg(kl_bits(uq.q, 0.05), uq.w)
horizon = {T: float(np.mean([wavg(d.excess_bits, d.w) for _, d in t[t["T"] == T].groupby("model")])) for T in HZ}

VAL = {r["column"]: r for r in json.load(open(DATA / "freeciv_validation_stats.json"))["rows"]}
SUM = json.load(open(DATA / "freeciv_summary.json"))
row = VAL["tails_all_excess_bits"]
assert max(abs(row["per_model_score"][m] - bits[m]) for m in models) < 1e-9, "validation stats are not from this run"
assert np.allclose([horizon[T] for T in HZ], SUM["fig_freeciv_horizon"]["tails"]["mean_over_models"], atol=1e-12)
assert abs(SUM["reference_forecasters"]["tails_flat_0p05_bits"] - flat05) < 1e-12

fh = pd.read_csv(DATA / "family_horizon_scores.csv")
fh = fh[fh.set == "tails"].copy()
fh["wt"] = fh["wsum"] if "wsum" in fh.columns else fh["n"]
order = fh.groupby("family").apply(lambda d: (d.wt * d.bits).sum() / d.wt.sum(), include_groups=False).sort_values(ascending=False)   # the difficulty table's order
worst = list(order.index[:WORST_N])
assert worst == WORST_NAMED, f"the prose names {WORST_NAMED} as the hardest tail families, now {worst}"
loss = (t.excess_bits * t.w)
worst_share = float(loss[t.family.isin(worst)].sum() / loss.sum())
worst_items = int(uq.family.isin(worst).sum())
worst_wshare = float(uq[uq.family.isin(worst)].w.sum() / uq.w.sum())
cl, fb = row["eci_cluster"], row["fb"]
N = dict(
    run=RUN, weighted=TAILS_WEIGHTED, weights=rel(TAILS_WEIGHTS) if TAILS_WEIGHTED else None, generated=str(date.today()), source=rel(SCORE_ITEMS),
    mean_over_models=float(bits.mean()), per_model=bits.to_dict(), best=bits.idxmin(), worst=bits.idxmax(),
    eci=dict(rho=row["eci"]["rho"], ci=[row["eci"]["ci_lo"], row["eci"]["ci_hi"]], p=row["eci"]["p"], games_ci=[cl["ci_lo"], cl["ci_hi"]]),
    fb=dict(rho=fb["rho"], ci=[fb["ci_lo"], fb["ci_hi"]], p=fb["p"], n=fb["n"]),
    mean_forecast=dict(min=float(meanf.min()), max=float(meanf.max()), per_model=meanf.to_dict()),
    mean_q=wavg(uq.q, uq.w), mean_q_selected=float(uq.q.mean()),
    flat_0p05_bits=flat05, models_below_flat=int((bits < flat05).sum()),
    rho_bits_meanforecast=float(spearmanr(bits, meanf)[0]),
    horizon_mean=horizon,
    worst_families=worst, worst_items=worst_items, worst_loss_share=worst_share, worst_weight_share=worst_wshare,
    effective_items=float(uq.w.sum() ** 2 / (uq.w ** 2).sum()), max_weight_ratio=float(uq.w.max() / uq.w.median()),
    pool_total=int(pd.read_csv(TAILS_WEIGHTS).drop_duplicates(["T", "family"]).n_pool.sum()) if TAILS_WEIGHTED else None,
    groups=int(len(pd.read_csv(TAILS_WEIGHTS).drop_duplicates(["T", "family"]))) if TAILS_WEIGHTED else None,
)
# claims the prose makes in words; a change needs a rewrite, not just new numbers
assert N["models_below_flat"] == 0, "the prose says no model scores below a constant 0.05"
assert (N["eci"]["ci"][0] > 0) == TAILS_WEIGHTED, "sections 2 and 7 count the tail interval as excluding zero only in the weighted run"
(DATA / "freeciv_tail_numbers.json").write_text(json.dumps(N, indent=1, default=float))
M = {
    "FCTailMean": f2(N["mean_over_models"]),
    "FCTailRho": f2(N["eci"]["rho"]), "FCTailRhoLo": f2(N["eci"]["ci"][0]), "FCTailRhoHi": f2(N["eci"]["ci"][1]), "FCTailRhoP": f"{N['eci']['p']:.3f}",
    "FCTailRhoGamesLo": f2(N["eci"]["games_ci"][0]), "FCTailRhoGamesHi": f2(N["eci"]["games_ci"][1]),
    "FCTailFBRho": f2(N["fb"]["rho"]), "FCTailFBLo": f2(N["fb"]["ci"][0]), "FCTailFBHi": f2(N["fb"]["ci"][1]), "FCTailFBP": f"{N['fb']['p']:.3f}",
    "FCTailMeanFMin": f2(meanf.min()), "FCTailMeanFMax": f2(meanf.max()), "FCTailMeanQ": f2(N["mean_q"]), "FCTailMeanQThree": f"{N['mean_q']:.3f}",
    "FCTailFlat": f2(flat05), "FCTailBelowFlat": str(N["models_below_flat"]),
    "FCTailRhoMeanF": f2(N["rho_bits_meanforecast"]),
    "FCTailHorizonFirst": f2(horizon[90]), "FCTailHorizonLast": f2(horizon[210]),
    "FCTailWorstItems": str(worst_items), "FCTailWorstShare": f"{100 * worst_share:.0f}\\%", "FCTailWorstWeightShare": f"{100 * worst_wshare:.0f}\\%",
    "FCTailEffItems": f"{N['effective_items']:.0f}", "FCTailMaxWeightRatio": f"{N['max_weight_ratio']:.0f}",
    "FCTailPoolTotal": f"{N['pool_total']:,}" if TAILS_WEIGHTED else "--", "FCTailGroups": str(N["groups"]) if TAILS_WEIGHTED else "--",
}
lines = [f"% Generated by worlds/freeciv/fbsim_v3/paper/make_freeciv_tail_macros.py (forecastbench-sim) from results/{RUN}; do not edit by hand.",
         "% FreeCiv tail-set numbers quoted in the text" + (", weighted by inverse selection probability (data/freeciv/tails_weights.csv)." if TAILS_WEIGHTED else ", unweighted.")]
lines += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()]
(PAPER_DATA / "freeciv_tail_macros.tex").write_text("\n".join(lines) + "\n")
print(f"wrote {rel(PAPER_DATA / 'freeciv_tail_macros.tex')} ({len(M)} macros) and {rel(DATA / 'freeciv_tail_numbers.json')}")
for k_, v in M.items():
    print(f"  {k_:24s} {v}")
