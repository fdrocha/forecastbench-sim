#!/usr/bin/env python3
"""bank_weights_v1.py DRAW_DIR — inverse-selection weights for the 750 bank items.

Reads DRAW_DIR/bank_750.json and DRAW_DIR/selection_fractions_bank_tails_mirrors.json (Jaeho, October 2, 2026: candidate
count n_pool and selected count per (band, horizon, family) stratum of draw_v1.py). Writes DRAW_DIR/bank_weights.csv
with w = n_pool / n_selected per item, rescaled so the 750 weights sum to 750, and prints a summary.
"""
import ast, csv, json, sys, collections

D = sys.argv[1]
BANDS = [(0.05, 0.23), (0.23, 0.41), (0.41, 0.59), (0.59, 0.77), (0.77, 0.95)]   # as draw_v1.py: lo < q <= hi


def band_of(q):
    return next(i for i, (lo, hi) in enumerate(BANDS) if lo < q <= hi)


bank = json.load(open(f"{D}/bank_750.json"))
frac = {ast.literal_eval(k): v for k, v in json.load(open(f"{D}/selection_fractions_bank_tails_mirrors.json")).items()}
key = {c["id"]: (band_of(c["qAll"]), c["T"], c["family"]) for c in bank}
n_sel = collections.Counter(key.values())
missing = sorted(set(n_sel) - set(frac), key=str)
assert not missing, f"bank strata without a pool count: {missing}"
# The selected counts in the JSON match the bank except where it also counted the q = 0.95 mirror item in band 4;
# the bank's own count is used.
fixed = {k: (frac[k]["n_selected"], n_sel[k]) for k in n_sel if frac[k]["n_selected"] != n_sel[k]}
pool_total = sum(frac[k]["n_pool"] for k in n_sel)
scale = len(bank) / pool_total
rows = []
for c in bank:
    k = key[c["id"]]
    n_pool = frac[k]["n_pool"]
    assert n_pool >= n_sel[k], (k, n_pool, n_sel[k])
    rows.append(dict(item=c["id"], world=c["world"], family=c["family"], T=c["T"], q=c["qAll"], band=k[0], n_pool=n_pool,
                     n_selected=n_sel[k], r=n_sel[k] / n_pool, w=n_pool / n_sel[k] * scale))
with open(f"{D}/bank_weights.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)

w = [r["w"] for r in rows]
ess = sum(w) ** 2 / sum(x * x for x in w)
print(f"wrote {D}/bank_weights.csv: {len(rows)} items, {len(n_sel)} strata, pool {pool_total:,}")
print(f"  selected counts corrected from the JSON: {fixed}")
print(f"  weights: min {min(w):.3f} median {sorted(w)[len(w) // 2]:.3f} max {max(w):.3f}; effective sample size {ess:.0f} of {len(w)}")
for name, idx in (("band", "band"), ("horizon", "T")):
    share = collections.defaultdict(float)
    for r in rows:
        share[r[idx]] += r["w"] / len(rows)
    print(f"  weighted share by {name}: " + ", ".join(f"{k}: {v:.3f}" for k, v in sorted(share.items())))
