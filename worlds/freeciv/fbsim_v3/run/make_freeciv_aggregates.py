#!/usr/bin/env python3
"""make_freeciv_aggregates.py SCORE_ITEMS OUT_DIR — the two aggregate files the paper's figures and difficulty table read.

  OUT_DIR/family_horizon_scores.csv  model x set (bank, tails, mirrors, continuous) x family x horizon: n, bias (mean p-q),
                                     bits (mean excess bits), excess (mean excess Brier, or mean excess nCRPS for continuous),
                                     ncrps (mean nCRPS), cov90, mederr, p, q; four decimals
  OUT_DIR/reliability_bands.csv      per model, the bank forecasts in ten equal-width bands of q on [0.05, 0.95] (right-closed):
                                     band, n, mean q, mean p

If score_items carries a weight column w (score_v2.py --bank-weights), every mean is weighted by it and both files gain a
column wsum, the sum of the weights behind each row; without it the files are as before.
Reproduces the run-1 files of 2026-09-09 exactly (imputed binary rows included, as the scorer left them) from scores_v1/score_items.csv(.gz); run 2 uses the same code.
"""
import sys, os
import numpy as np, pandas as pd

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
si = pd.read_csv(src, low_memory=False)
# imputed binary rows (p = 0.5 for an unparsed answer) stay in, as in run 1; unparsed continuous rows carry NaN metrics and drop out of the means
WEIGHTED = "w" in si.columns


def wm(x, w):
    """Mean of x weighted by w over the non-missing x; the plain mean when unweighted."""
    if not WEIGHTED:
        return x.mean()
    ok = x.notna()
    return float((x[ok] * w[ok]).sum() / w[ok].sum())


rows = []
for (m, s, fam, T), g in si[si["set"].isin(["bank", "tails", "mirrors", "continuous"])].groupby(["model", "set", "family", "T"]):
    r = dict(model=m, set=s, family=fam, horizon=int(T), n=len(g))
    w = g["w"] if WEIGHTED else None
    if WEIGHTED:
        r["wsum"] = float(w.sum())
    if s == "continuous":
        r.update(excess=wm(g["excess_ncrps_global"], w), ncrps=wm(g["ncrps_global"], w), cov90=wm(g["cov90"], w), mederr=wm(g["median_err"], w))
    else:
        r.update(bias=wm(g["p"] - g["q"], w), bits=wm(g["excess_bits"], w), excess=wm(g["excess_brier"], w), p=wm(g["p"], w), q=wm(g["q"], w))
    rows.append(r)
fh = pd.DataFrame(rows)[["model", "set", "family", "horizon", "bias", "bits", "cov90", "excess", "mederr", "n", "ncrps", "p", "q"] + (["wsum"] if WEIGHTED else [])]
fh = fh.sort_values(["model", "set", "horizon", "family"]).round(4)
fh.to_csv(os.path.join(out, "family_horizon_scores.csv"), index=False)
bank = si[si["set"] == "bank"].copy()
edges = np.round(np.linspace(0.05, 0.95, 11), 6)                                     # rounded, so q values on an edge bin as the figure script bins them
bank["band"] = pd.cut(bank["q"], edges, labels=False, right=True, include_lowest=True).astype(int)   # right-closed bins of width 0.09
if WEIGHTED:
    rb = bank.groupby(["model", "band"]).apply(lambda g: pd.Series(dict(n=len(g), q=wm(g["q"], g["w"]), p=wm(g["p"], g["w"]), wsum=g["w"].sum())), include_groups=False).reset_index()
    rb["n"] = rb["n"].astype(int)
    rb = rb.round(4)
else:
    rb = bank.groupby(["model", "band"]).agg(n=("q", "size"), q=("q", "mean"), p=("p", "mean")).reset_index().round(4)
rb.to_csv(os.path.join(out, "reliability_bands.csv"), index=False)
print(f"wrote {out}/family_horizon_scores.csv ({len(fh)} rows) and reliability_bands.csv ({len(rb)} rows)")
