#!/usr/bin/env python3
"""analyze_logprob.py DIR — per model, the three probabilities on the same items: verbal (stated number), ev_verbal (expectation of
the model's own distribution over the stated number), token (normalised Yes/No first-token probability).
Reports bias, slope, Spearman(p,q), excess Brier, reliability by truth band on the bank; excess bits on the tails; token coverage;
and the within-model correlation between token and verbal probabilities."""
import sys, os, glob, json, csv, collections, math, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import score_v1 as sv
S, C, N = sv.load_sets(); d = sys.argv[1]
def bits(p, q): return sv.bits(p, q)[1]
print(f"{'model':34s} {'arm':10s} {'n':>4s} {'bias':>6s} {'slope':>6s} {'rho':>5s} {'excess':>7s} {'tail bits':>9s} {'cover':>6s} {'p<=.25':>6s} {'p>=.75':>6s}")
summary = {}
for f in sorted(glob.glob(f'{d}/*/results.jsonl')):
    rows = {}
    for l in open(f): r = json.loads(l); rows[(r['item'], r['arm'])] = r
    mid = next(iter(rows.values()))['model']; summary[mid] = {}
    arms = {'verbal': lambda r: r.get('value'), 'ev_verbal': lambda r: r.get('ev_verbal'), 'token': lambda r: r.get('value')}
    for arm, get in arms.items():
        src = 'verbal' if arm != 'token' else 'token'
        pts = [(S[i]['qAll'], get(r), S[i]['set'], r) for (i, a), r in rows.items() if a == src and i in S and get(r) is not None]
        bank = [(q, p) for q, p, s, _ in pts if s == 'bank']; tails = [(q, p) for q, p, s, _ in pts if s == 'tails']
        if len(bank) < 20: continue
        q = np.array([x for x, _ in bank]); p = np.array([y for _, y in bank]); p = np.clip(p, 0, 1)
        cov = np.mean([r.get('coverage') or 0 for _, _, _, r in pts]) if arm == 'token' else float('nan')
        tb = np.mean([bits(min(max(y, 1e-3), 1 - 1e-3), x) for x, y in tails]) if tails else float('nan')
        rec = dict(n=len(bank), bias=float(np.mean(p - q)), slope=float(np.polyfit(q, p, 1)[0]), rho=spearmanr(p, q).correlation, excess=float(np.mean((p - q) ** 2)), tail_bits=tb, coverage=cov, low=float(np.mean(p <= .25)), high=float(np.mean(p >= .75)))
        summary[mid][arm] = rec | dict(q=q, p=p)
        print(f"{mid:34s} {arm:10s} {rec['n']:4d} {rec['bias']:+6.2f} {rec['slope']:6.2f} {rec['rho']:5.2f} {rec['excess']:7.3f} {rec['tail_bits']:9.3f} {rec['coverage']:6.2f} {rec['low']:6.2f} {rec['high']:6.2f}")
    v = {i: r['value'] for (i, a), r in rows.items() if a == 'verbal' and r.get('value') is not None}; t = {i: r['value'] for (i, a), r in rows.items() if a == 'token' and r.get('value') is not None}
    both = [(v[i], t[i]) for i in v if i in t]
    if len(both) > 20: print(f"{'':34s} token vs verbal on {len(both)} items: Spearman {spearmanr([a for a,_ in both],[b for _,b in both]).correlation:.2f}; mean |diff| {np.mean([abs(a-b) for a,b in both]):.2f}")
print("\nReliability on the bank (mean p by truth band), verbal | token:")
bands = [(0.05, 0.23), (0.23, 0.41), (0.41, 0.59), (0.59, 0.77), (0.77, 0.95)]
print(f"{'model':34s} " + ' '.join(f"{lo:.2f}-{hi:.2f}".rjust(11) for lo, hi in bands))
for mid, arms in summary.items():
    for arm in ('verbal', 'token'):
        if arm not in arms: continue
        q, p = arms[arm]['q'], arms[arm]['p']
        print(f"{(mid.split('/')[-1] + ' ' + arm):34s} " + ' '.join(f"{np.mean(p[(q > lo) & (q <= hi)]):11.2f}" if ((q > lo) & (q <= hi)).any() else f"{'-':>11s}" for lo, hi in bands))
flat = float(np.mean([(0.5 - S[i]['qAll']) ** 2 for i in S if S[i]['set'] == 'bank'])); print(f"\nflat-0.5 excess Brier on the full bank: {flat:.3f}")
