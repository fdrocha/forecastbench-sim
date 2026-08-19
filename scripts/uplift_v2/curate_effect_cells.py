#!/usr/bin/env python3
"""Curate the headline effect-cell benchmark from natcond_cells_v2 output.

Selection (effect cells only, certified on half A):
  rank by certification strength z = |delta_A| / SE_A, then apply balance caps
  greedily in rank order:
    - per world:            <= MAX_PER_WORLD
    - per event (per world): <= MAX_PER_EVENT  (stops one conquest event
                              dominating, cf. seed2 pilot skew)
    - per template family:  <= MAX_PER_TEMPLATE
  aiming for AT LEAST MIN_PER_HORIZON cells at each of H1/H2/H3 (a horizon
  quota pass runs first so long horizons are not crowded out by H1's larger
  certified pool).

Also emits a matched placebo set (same size, sampled deterministically,
stratified by world x horizon) for the selectivity arm, and a summary table.

Usage:
  uv run python scripts/uplift_v2/curate_effect_cells.py \
      --cells tmp/natcond_v2/cells.json --target 300 \
      --out tmp/natcond_v2/curated.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict

MAX_PER_WORLD = 45
MAX_PER_EVENT = 8       # per (world, event_id)
MAX_PER_TEMPLATE = 70
MIN_PER_HORIZON = 50


def is_quasi(c: dict) -> bool:
    """Identity pairs: the revealed event names the same (entity, target) the
    question asks about, and the conditional truth is exactly 0 or 1. These are
    either logically implied or mechanically determined — excluded from the
    headline set and reported separately (cf. v1 quasi quadrant)."""
    q, e = c["question"].lower(), c["event_desc"].lower()
    m = re.search(r"(discovered|changed government.*to|adopt\w*)\s+([a-z' ]+)$", e)
    if not m:
        return False
    target = m.group(2).strip()
    entity = e.split()[0]
    return (target in q and entity in q
            and c.get("half_b", {}).get("p_yx") in (0.0, 1.0))


def tfam(q: str) -> str:
    ql = q.lower()
    for k in ("technolog", "wonder", "government", "rank", "score", "cit",
              "population", "territor", "tile", "treasur"):
        if k in ql:
            return k
    return "other"


def zscore(c: dict) -> float:
    ha = c.get("half_a", {})
    d, se = abs(ha.get("delta", c["delta"])), max(ha.get("se_delta", c["se_delta"]), 1e-6)
    return d / se


def pick(cells: list[dict], target: int) -> list[dict]:
    ranked = sorted(cells, key=zscore, reverse=True)
    chosen: list[dict] = []
    w_ct: Counter = Counter()
    e_ct: Counter = Counter()
    t_ct: Counter = Counter()
    h_ct: Counter = Counter()

    def ok(c: dict) -> bool:
        w, e, t = c["game_id"], (c["game_id"], c["event_id"]), tfam(c["question"])
        return (w_ct[w] < MAX_PER_WORLD and e_ct[e] < MAX_PER_EVENT
                and t_ct[t] < MAX_PER_TEMPLATE)

    def take(c: dict) -> None:
        chosen.append(c)
        w_ct[c["game_id"]] += 1
        e_ct[(c["game_id"], c["event_id"])] += 1
        t_ct[tfam(c["question"])] += 1
        h_ct[c.get("horizon", "H1")] += 1

    # pass 1: horizon quotas, rank order within horizon
    for hz in ("H3", "H2", "H1"):  # scarcest first
        for c in ranked:
            if h_ct[hz] >= MIN_PER_HORIZON or len(chosen) >= target:
                break
            if c.get("horizon", "H1") == hz and c not in chosen and ok(c):
                take(c)
    # pass 2: fill to target in global rank order
    for c in ranked:
        if len(chosen) >= target:
            break
        if c not in chosen and ok(c):
            take(c)
    return chosen


def matched_placebos(cells: list[dict], chosen: list[dict]) -> list[dict]:
    strata = Counter((c["game_id"], c.get("horizon", "H1")) for c in chosen)
    pool = defaultdict(list)
    for c in cells:
        if c["cls"] == "placebo":
            pool[(c["game_id"], c.get("horizon", "H1"))].append(c)
    out = []
    for key, n in strata.items():
        cand = sorted(pool.get(key, []),
                      key=lambda c: hashlib.sha256(
                          f"{c['game_id']}:{c['qid']}:{c['event_id']}".encode()).hexdigest())
        out.extend(cand[:n])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--target", type=int, default=300)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = json.load(open(args.cells))
    cells = payload["cells"] if isinstance(payload, dict) and "cells" in payload else payload
    eff_all = [c for c in cells if c.get("cls") == "effect" and c.get("half_b")]
    quasi = [c for c in eff_all if is_quasi(c)]
    eff = [c for c in eff_all if not is_quasi(c)]
    chosen = pick(eff, args.target)
    placebos = matched_placebos(cells, chosen)

    def table(rows: list[dict]) -> dict:
        return {
            "by_world": dict(Counter(c["game_id"] for c in rows)),
            "by_horizon": dict(Counter(c.get("horizon", "H1") for c in rows)),
            "by_template": dict(Counter(tfam(c["question"]) for c in rows)),
            "z_median": sorted(zscore(c) for c in rows)[len(rows) // 2] if rows else None,
        }

    out = {
        "meta": {
            "source": args.cells,
            "certified_effect_pool": len(eff),
            "target": args.target,
            "caps": {"per_world": MAX_PER_WORLD, "per_event": MAX_PER_EVENT,
                     "per_template": MAX_PER_TEMPLATE, "min_per_horizon": MIN_PER_HORIZON},
        },
        "summary": {"effect": table(chosen), "placebo": table(placebos),
                    "quasi_excluded": len(quasi)},
        "effect_cells": chosen,
        "placebo_cells": placebos,
        "quasi_cells": quasi,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"pool {len(eff_all)} certified ({len(quasi)} quasi excluded) -> curated {len(chosen)} "
          f"+ {len(placebos)} matched placebos -> {args.out}")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
