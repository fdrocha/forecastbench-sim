#!/usr/bin/env python3
"""selection_fractions.py — per-stratum selection fractions r = n_selected / n_pool for the FreeCiv binary sets, for 1/r
importance weights (Fabio Rocha's correction, 2026-09-30).

  uv run python selection_fractions.py [--corpus ../fbsim_v3_corpus] [--out ../fbsim_v3_corpus/draw_v1/selection_fractions_bank_tails_mirrors.json]
                                       [--results ../fbsim_v3_corpus/elicit_v1 --reweighted-csv ../fbsim_v3_corpus/results_v1/freeciv_binary_reweighted.csv]

Pool: every binary instance the generator enumerated (bank_v1/<world>.pkl, key 'bin'), keeping 0 < q < 1 (23,241 items over 8
worlds). Item ids are '<world>:b<index into that list>', the same ids used in the draw files, so selection is exact set membership.
Strata: (band, T, family) where band = index of q in the five bank bands (.05,.23], (.23,.41], (.41,.59], (.59,.77], (.77,.95],
or 'tail' (q <= .05) or 'mirror' (q >= .95); T = horizon; family = question template. These are the cells the draw balanced
(draw_v1.py: 30 bank items per band x horizon filled round-robin over families; 60 tails and 10 mirrors per horizon likewise),
so the inverse fraction 1/r is the importance weight that maps a sample mean back to the enumerated pool.
Output JSON: {"(band, T, 'family')": {"n_pool": int, "n_selected": int, "r": float}} for every stratum with n_selected > 0.
With --results, also writes a per-model CSV of unweighted bank excess Brier and the 1/r-weighted whole-pool excess Brier
(bank + tails + mirrors) from a results directory of elicit_v1.py rows.
"""
import argparse, collections, csv, glob, json, os, pickle, sys
BANDS = [(.05, .23), (.23, .41), (.41, .59), (.59, .77), (.77, .95)]
def band(q): return next((i for i, (lo, hi) in enumerate(BANDS) if lo < q <= hi), None)
def stratum(c):
    q = c['qAll']; b = band(q)
    return (b if b is not None else ('tail' if q <= 0.05 else 'mirror'), c['T'], c['family'])
def load_pool(corpus):
    idx = {}
    for f in sorted(glob.glob(f'{corpus}/bank_v1/*.pkl')):
        d = pickle.load(open(f, 'rb'))
        for k, c in enumerate(d['bin']):
            if 0 < c['qAll'] < 1: idx[f"{d['world']}:b{k}"] = dict(c, world=d['world'])
    return idx
def load_selected(corpus):
    sel = {}
    for s, f in [('bank', 'bank_750'), ('tails', 'tails_300'), ('mirrors', 'mirrors_50')]:
        for i in json.load(open(f'{corpus}/draw_v1/{f}.json')): sel[i['id']] = s
    return sel
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument('--corpus', default=os.environ.get('FBSIM_V3_CORPUS', f'{here}/../fbsim_v3_corpus'))
    ap.add_argument('--out', default=None, help='default: <corpus>/draw_v1/selection_fractions_bank_tails_mirrors.json')
    ap.add_argument('--results', default=None, help='elicit_v1.py results dir; enables the reweighted CSV')
    ap.add_argument('--reweighted-csv', default=None, help='default: <corpus>/results_v1/freeciv_binary_reweighted.csv')
    a = ap.parse_args(); out = a.out or f'{a.corpus}/draw_v1/selection_fractions_bank_tails_mirrors.json'
    idx = load_pool(a.corpus); sel = load_selected(a.corpus)
    missing = [i for i in sel if i not in idx]
    if missing: sys.exit(f'{len(missing)} selected ids not in the pool, e.g. {missing[:3]}')
    npool = collections.Counter(stratum(c) for c in idx.values()); nsel = collections.Counter(stratum(idx[i]) for i in sel)
    r = {k: nsel[k] / npool[k] for k in nsel}
    json.dump({str(k): dict(n_pool=npool[k], n_selected=nsel[k], r=v) for k, v in r.items()}, open(out, 'w'), indent=1)
    print(f"pool {len(idx)} items ({collections.Counter('tail' if c['qAll'] <= .05 else 'mirror' if c['qAll'] >= .95 else 'bank' for c in idx.values())}); "
          f"selected {len(sel)}; strata with selections {len(r)}; r range {min(r.values()):.4f}-{max(r.values()):.1f}; wrote {out}")
    if a.results:
        sys.path.insert(0, here); import numpy as np, score_v1 as sv
        S, C, N = sv.load_sets(f'{a.corpus}/draw_v1'); R = sv.load_results([a.results])
        rows = [x for x in sv.score_items(R, S, C, N, impute=0.5) if x['set'] in ('bank', 'tails', 'mirrors')]
        meta = {m['openrouter_id']: m for m in csv.DictReader(open(os.path.join(os.environ.get('FBSIM_FREECIV_V3_DATA', f'{here}/..'), 'run', 'models_v1.csv')))}
        csv_out = a.reweighted_csv or f'{a.corpus}/results_v1/freeciv_binary_reweighted.csv'
        with open(csv_out, 'w', newline='') as f:
            w = csv.writer(f); w.writerow(['model', 'eci', 'bank_excess_brier_unweighted', 'whole_pool_excess_brier_weighted_1_over_r'])
            for m in sorted({x['model'] for x in rows}, key=lambda m: -float(meta[m]['eci'])):
                g = [x for x in rows if x['model'] == m]; e = np.array([x['excess_brier'] for x in g]); wts = np.array([1 / r[stratum(idx[x['item']])] for x in g])
                w.writerow([m, meta[m]['eci'], round(float(np.mean([x['excess_brier'] for x in g if x['set'] == 'bank'])), 4), round(float(np.average(e, weights=wts)), 4)])
        print('wrote', csv_out)
if __name__ == '__main__': main()
