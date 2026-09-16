#!/usr/bin/env -S uv run python3
"""How questions-per-prompt affects the binary eval.

Reads data/micropolis/binary/{label}/results.csv for each batching label plus
the usage sidecars beside the cached responses, and writes
data/micropolis/binary/batching/summary.json (everything the report page
draws) and batching.csv (mean scores per model x n x section).

Bootstrap intervals resample question instances with one shared draw across
models and batch sizes, so differences between n are paired.

Usage:
    scripts/analyze_batching.py
    scripts/analyze_batching.py --labels 1q 2q 4q 8q 16q 32q 64q 100q
    scripts/analyze_batching.py --exclude openai/gpt-5-mini:loeff --suffix no-gpt5mini
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from micropolis_world.model_scores import eci_of
from micropolis_world.module_globals import DATA_DIR

sys.path.insert(0, str(Path(__file__).parent))
from analyze_continuous import (  # noqa: E402
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    resample_indices,
    spearman_over_resamples,
)

BINARY_DIR = DATA_DIR / "binary"
OUT_DIR = BINARY_DIR / "batching"
DEFAULT_LABELS = ["1q", "2q", "4q", "8q", "16q", "32q", "64q", "100q"]
TAIL_THRESHOLD = 0.05
SECTIONS = {"all": None, "mid": lambda p: p >= TAIL_THRESHOLD, "tail": lambda p: p < TAIL_THRESHOLD}
QKEY = ("city", "snapshot_turn", "horizon", "question_id")


def load_results(label: str) -> list[dict]:
    with (BINARY_DIR / label / "results.csv").open() as f:
        return list(csv.DictReader(f))


def usage_path(response_file: str) -> Path:
    d, base = response_file.split("/", 1)
    return BINARY_DIR / "cache" / d / ("usage-" + base.removeprefix("response-").removesuffix(".txt") + ".json")


def ci(a: np.ndarray, axis=0) -> tuple[list, list]:
    lo, hi = np.nanpercentile(a, [2.5, 97.5], axis=axis)
    return lo, hi


def flt(x) -> float | None:
    x = float(x)
    return None if math.isnan(x) else x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=DEFAULT_LABELS)
    ap.add_argument("--exclude", nargs="*", default=[], help="model slugs to drop")
    ap.add_argument("--suffix", default="", help="appended to the output file names")
    args = ap.parse_args()

    labels = args.labels
    ns = [int(l.removesuffix("q")) for l in labels]
    results = {l: load_results(l) for l in labels}

    # Model order follows first appearance in the 1q file (the config order).
    models = [m for m in dict.fromkeys(r["model"] for r in results[labels[0]]) if m not in args.exclude]
    results = {l: [r for r in rows if r["model"] in models] for l, rows in results.items()}
    keys = list(dict.fromkeys(tuple(r[k] for k in QKEY) for r in results[labels[0]]))
    qi = {k: i for i, k in enumerate(keys)}
    mi = {m: i for i, m in enumerate(models)}
    M, Q = len(models), len(keys)

    p = np.full(Q, np.nan)
    F = {}  # n -> (M, Q) forecasts, nan where unparsed
    answer = np.full(Q, np.nan)
    for l, n in zip(labels, ns):
        f = np.full((M, Q), np.nan)
        for r in results[l]:
            q = qi[tuple(r[k] for k in QKEY)]
            f[mi[r["model"]], q] = float(r["forecast"])
            p[q] = float(r["real_prob"])
            answer[q] = float(r["answer"])
        F[n] = f
    assert not np.isnan(p).any()

    ecis = np.array([eci_of(m) if eci_of(m) is not None else np.nan for m in models])

    out = {
        "ns": ns,
        "models": [{"id": m, "name": m.split("/")[-1].split(":")[0], "eci": flt(e)} for m, e in zip(models, ecis)],
        "n_questions": Q,
        "sections": {},
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "excluded": args.exclude,
    }
    csv_rows = []

    for sec, pred in SECTIONS.items():
        mask = np.ones(Q, bool) if pred is None else pred(p)
        qs = np.where(mask)[0]
        Qs = len(qs)
        idx = resample_indices(Qs, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)  # (B, Qs), shared across n and models
        ps = p[qs]
        ans = answer[qs]

        per_model = {m: {} for m in models}
        point = {}  # n -> (M,) mean excess Brier
        boot = {}  # n -> (B, M)
        bias = {}
        disc = {}
        for n in ns:
            f = F[n][:, qs]
            ex = (f - ps) ** 2
            br = (f - ans) ** 2
            pm = np.nanmean(ex, axis=1)
            point[n] = pm
            b = np.stack([np.nanmean(ex[m][idx], axis=1) for m in range(M)], axis=1)
            boot[n] = b
            lo, hi = ci(b)
            for m, mid in enumerate(models):
                valid = ~np.isnan(f[m])
                r = stats.pearsonr(f[m][valid], ps[valid])[0] if valid.sum() > 2 and np.std(f[m][valid]) > 0 else np.nan
                per_model[mid][str(n)] = {
                    "excess": flt(pm[m]),
                    "lo": flt(lo[m]),
                    "hi": flt(hi[m]),
                    "brier": flt(np.nanmean(br[m])),
                    "nvalid": int(valid.sum()),
                    "bias": flt(np.nanmean(f[m] - ps)),
                    "corr_p": flt(r),
                }
                csv_rows.append(dict(section=sec, model=mid, n=n, excess_brier=pm[m], lo=lo[m], hi=hi[m],
                                     brier=np.nanmean(br[m]), nvalid=int(valid.sum())))

        # Pooled over models: mean of per-model means, paired across n.
        pooled_pt = {n: float(np.nanmean(point[n])) for n in ns}
        pooled_b = {n: np.nanmean(boot[n], axis=1) for n in ns}  # (B,)
        pooled = {}
        delta = {}
        for n in ns:
            lo, hi = ci(pooled_b[n])
            pooled[str(n)] = {"excess": pooled_pt[n], "lo": float(lo), "hi": float(hi)}
            d = pooled_b[n] - pooled_b[ns[0]]
            dlo, dhi = ci(d)
            delta[str(n)] = {
                "delta": pooled_pt[n] - pooled_pt[ns[0]],
                "lo": float(dlo),
                "hi": float(dhi),
                "p_worse": float(np.mean(d > 0)),
            }
        delta_model = {m: {} for m in models}
        for m, mid in enumerate(models):
            for n in ns:
                d = boot[n][:, m] - boot[ns[0]][:, m]
                dlo, dhi = ci(d)
                delta_model[mid][str(n)] = {
                    "delta": float(point[n][m] - point[ns[0]][m]),
                    "lo": float(dlo),
                    "hi": float(dhi),
                    "p_worse": float(np.mean(d > 0)),
                }

        # P(best n): share of paired draws where each n has the lowest mean.
        stack = np.stack([boot[n] for n in ns], axis=0)  # (N, B, M)
        best = np.argmin(stack, axis=0)  # (B, M)
        pbest = {
            mid: {str(n): float(np.mean(best[:, m] == i)) for i, n in enumerate(ns)}
            for m, mid in enumerate(models)
        }
        pstack = np.stack([pooled_b[n] for n in ns], axis=0)
        pbest["pooled"] = {str(n): float(np.mean(np.argmin(pstack, axis=0) == i)) for i, n in enumerate(ns)}

        # Pairwise: P(n_a beats n_b) matrix for the pooled score.
        pairwise = [[float(np.mean(pstack[a] < pstack[b])) for b in range(len(ns))] for a in range(len(ns))]

        # ECI correlation, two intervals: over models and over questions.
        has = ~np.isnan(ecis)
        midx = resample_indices(int(has.sum()), BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
        eci_corr = {}
        for n in ns:
            x = ecis[has]
            y = point[n][has]
            rho, rho_p = stats.spearmanr(x, y)
            r, r_p = stats.pearsonr(x, y)
            rm = spearman_over_resamples(x, y, midx)
            rq = np.array([stats.spearmanr(x, boot[n][b][has])[0] for b in range(0, BOOTSTRAP_RESAMPLES, 5)])
            rm = rm[~np.isnan(rm)]
            rq = rq[~np.isnan(rq)]
            eci_corr[str(n)] = {
                "rho": float(rho), "rho_p": float(rho_p), "r": float(r), "r_p": float(r_p),
                "models_ci": [float(v) for v in np.percentile(rm, [2.5, 97.5])],
                "questions_ci": [float(v) for v in np.percentile(rq, [2.5, 97.5])],
                "n_models": int(has.sum()),
            }

        # Rank stability of the model ordering against n=1.
        rank_stab = {
            str(n): float(stats.spearmanr(point[ns[0]], point[n])[0]) for n in ns
        }

        out["sections"][sec] = {
            "n_questions": Qs,
            "per_model": per_model,
            "pooled": pooled,
            "delta": delta,
            "delta_model": delta_model,
            "pbest": pbest,
            "pairwise": pairwise,
            "eci": eci_corr,
            "rank_stability": rank_stab,
        }

    # Parse failures and forecast agreement with n=1 (all questions).
    out["failures"] = {
        m: {str(n): float(np.isnan(F[n][i]).mean()) for n in ns} for i, m in enumerate(models)
    }
    out["agreement"] = {}
    for i, m in enumerate(models):
        out["agreement"][m] = {}
        for n in ns:
            a, b = F[ns[0]][i], F[n][i]
            ok = ~np.isnan(a) & ~np.isnan(b)
            out["agreement"][m][str(n)] = {
                "mean_abs_diff": float(np.mean(np.abs(a[ok] - b[ok]))),
                "frac_same": float(np.mean(np.isclose(a[ok], b[ok]))),
                "corr": float(stats.pearsonr(a[ok], b[ok])[0]) if ok.sum() > 2 else None,
            }
    # Position within the prompt: paired excess-Brier change vs n=1 by relative
    # position quintile, pooled over models. Rows in results.csv follow the
    # prompt order within a response file.
    NBINS = 5
    out["position"] = {}
    ex1 = (F[ns[0]] - p) ** 2
    for l, n in zip(labels, ns):
        if n < 8:
            continue
        pos = np.full((M, Q), np.nan)
        counts: dict[str, int] = {}
        for r in results[l]:
            counts[r["response_file"]] = counts.get(r["response_file"], 0) + 1
        seen: dict[str, int] = {}
        for r in results[l]:
            rf = r["response_file"]
            k = seen.get(rf, 0)
            seen[rf] = k + 1
            pos[mi[r["model"]], qi[tuple(r[x] for x in QKEY)]] = k / max(counts[rf] - 1, 1)
        d = (F[n] - p) ** 2 - ex1  # (M, Q)
        bins = np.minimum((pos * NBINS).astype(int), NBINS - 1)
        rows = []
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        for b in range(NBINS):
            sel = (bins == b) & ~np.isnan(d)
            vals = d[sel]
            if len(vals) == 0:
                continue
            boots = rng.choice(vals, (2000, len(vals))).mean(axis=1)
            lo, hi = np.percentile(boots, [2.5, 97.5])
            rows.append({"bin": b, "delta": float(vals.mean()), "lo": float(lo), "hi": float(hi), "count": int(len(vals))})
        out["position"][str(n)] = rows

    # Anchoring on the format example: the epilogue's example block reads
    # "Q1: 0.65 / Q2: 0.03", and some models return those values verbatim.
    EXAMPLE = 0.65
    out["anchor"] = {"example": EXAMPLE, "share": {}, "n1": {}}
    for i, m in enumerate(models):
        out["anchor"]["share"][m] = {}
        for n in ns:
            f = F[n][i]
            ok = ~np.isnan(f)
            out["anchor"]["share"][m][str(n)] = float(np.isclose(f[ok], EXAMPLE).mean())
        f = F[ns[0]][i]
        ok = ~np.isnan(f)
        hit = ok & np.isclose(f, EXAMPLE)
        ex = (f - p) ** 2
        out["anchor"]["n1"][m] = {
            "count": int(hit.sum()),
            "tail_count": int((hit & (p < TAIL_THRESHOLD)).sum()),
            "excess_all": float(np.nanmean(ex[ok])),
            "excess_without": float(np.nanmean(ex[ok & ~hit])) if (ok & ~hit).any() else None,
            "excess_only": float(np.nanmean(ex[hit])) if hit.any() else None,
            "tail_excess_all": float(np.nanmean(ex[ok & (p < TAIL_THRESHOLD)])),
            "tail_excess_without": float(np.nanmean(ex[ok & ~hit & (p < TAIL_THRESHOLD)])),
            "tail_share_ge_02": float((f[ok & ~hit & (p < TAIL_THRESHOLD)] >= 0.2).mean()),
        }

    # Cost from the usage sidecars of every response the scored forecasts came from.
    out["cost"] = {}
    for l, n in zip(labels, ns):
        files = sorted({r["response_file"] for r in results[l]})
        tot = {"total_usd": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0, "unpriced": 0,
               "per_model": {m: {"usd": 0.0, "input_tokens": 0, "output_tokens": 0, "calls": 0,
                                  "latency_ms": []} for m in models}}
        for rf in files:
            u = json.loads(usage_path(rf).read_text())
            pm = tot["per_model"][u["model_id"]]
            tot["calls"] += 1
            pm["calls"] += 1
            if u.get("cost_usd") is None:
                tot["unpriced"] += 1
            else:
                tot["total_usd"] += u["cost_usd"]
                pm["usd"] += u["cost_usd"]
            for k in ("input_tokens", "output_tokens"):
                tot[k] += u[k]
                pm[k] += u[k]
            if u.get("latency_ms") is not None:
                pm["latency_ms"].append(u["latency_ms"])
        for m in models:
            lat = tot["per_model"][m].pop("latency_ms")
            tot["per_model"][m]["latency_p50_s"] = float(np.median(lat)) / 1000 if lat else None
            tot["per_model"][m]["latency_total_s"] = float(np.sum(lat)) / 1000 if lat else None
        out["cost"][str(n)] = tot

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"-{args.suffix}" if args.suffix else ""
    (OUT_DIR / f"summary{tag}.json").write_text(json.dumps(out, indent=1))
    with (OUT_DIR / f"batching{tag}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"wrote {OUT_DIR / f'summary{tag}.json'} and batching{tag}.csv")


if __name__ == "__main__":
    main()
