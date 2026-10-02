#!/usr/bin/env python3
"""tails_weights_v1.py DRAW_DIR — inverse-selection weights for the 300 tail items.

Tails were drawn per horizon with a family round-robin and no probability bands (draw_v1.py), so the groups are
(horizon, family). Reads DRAW_DIR/tails_300.json and DRAW_DIR/selection_fractions_bank_tails_mirrors.json (keys
('tail', T, family)). Writes DRAW_DIR/tails_weights.csv with w = n_pool / n_selected per item, rescaled so the 300
weights sum to 300, in the columns of bank_weights.csv (band = 'tail'), and prints a summary.
"""
import ast, csv, json, sys, collections

D = sys.argv[1]
tails = json.load(open(f"{D}/tails_300.json"))
assert all(0 < c["qAll"] <= 0.05 for c in tails)
frac = {ast.literal_eval(k): v for k, v in json.load(open(f"{D}/selection_fractions_bank_tails_mirrors.json")).items()}
key = {c["id"]: ("tail", c["T"], c["family"]) for c in tails}
n_sel = collections.Counter(key.values())
missing = sorted(set(n_sel) - set(frac), key=str)
assert not missing, f"tail groups without a pool count: {missing}"
mismatch = {k: (frac[k]["n_selected"], n_sel[k]) for k in n_sel if frac[k]["n_selected"] != n_sel[k]}
assert not mismatch, f"selected counts differ from tails_300.json: {mismatch}"
extra = sorted(k for k in frac if k[0] == "tail" and k not in n_sel)
assert not extra, f"tail groups in the JSON with no selected item: {extra}"
pool_total = sum(frac[k]["n_pool"] for k in n_sel)
scale = len(tails) / pool_total
rows = []
for c in tails:
    k = key[c["id"]]
    n_pool = frac[k]["n_pool"]
    assert n_pool >= n_sel[k], (k, n_pool, n_sel[k])
    rows.append(dict(item=c["id"], world=c["world"], family=c["family"], T=c["T"], q=c["qAll"], band="tail", n_pool=n_pool,
                     n_selected=n_sel[k], r=n_sel[k] / n_pool, w=n_pool / n_sel[k] * scale))
with open(f"{D}/tails_weights.csv", "w", newline="") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)

w = [r["w"] for r in rows]
ess = sum(w) ** 2 / sum(x * x for x in w)
print(f"wrote {D}/tails_weights.csv: {len(rows)} items, {len(n_sel)} groups, pool {pool_total:,}")
print(f"  weights: min {min(w):.3f} median {sorted(w)[len(w) // 2]:.3f} max {max(w):.3f}; effective sample size {ess:.0f} of {len(w)}")
share = collections.defaultdict(float)
for r in rows:
    share[r["T"]] += r["w"] / len(rows)
print("  weighted share by horizon: " + ", ".join(f"{k}: {v:.3f}" for k, v in sorted(share.items())))
