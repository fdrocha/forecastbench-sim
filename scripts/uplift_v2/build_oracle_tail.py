#!/usr/bin/env python3
"""Build the oracle-tail question set from fork-ensemble truth.

The natural-conditional cells carry, per (question, horizon), the held-out
half-B unconditional probability p_y measured over ~500 replays of that exact
world. This module selects the questions whose p_y is in the low tail — events
that genuinely almost never happen *in this world*, as opposed to events that
are merely rare across a population of worlds.

Selection:
  tail   : 0 < p_y <= --tail-hi      (default 0.05)
  mirror : --mirror-lo <= p_y < 1    (default 0.95)

The mirror set exists so a constant "always answer 2%" strategy cannot score
well: mirrors are the same instrument with the answer flipped. Degenerate
p_y in {0, 1} are excluded (no measurable calibration target) but counted.

Balance: per-anchor and per-horizon caps keep one anchor or one horizon from
dominating; within a stratum, items are ranked by split-half agreement
(|p_y_A - p_y_B|, smaller first) so the most reliably-measured tails are kept.

Usage:
  uv run python scripts/uplift_v2/build_oracle_tail.py \
      --cells tmp/natcond_v3/cells_all.json --out tmp/natcond_v3/oracle_tail.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

TAIL_HI = 0.05
MIRROR_LO = 0.95
MAX_PER_ANCHOR = 60
MAX_PER_HORIZON = 120


def load_cells(paths: list[str]) -> list[dict]:
    out = []
    for p in paths:
        payload = json.load(open(p))
        cells = payload["cells"] if isinstance(payload, dict) and "cells" in payload else payload
        out.extend(cells)
    return out


def unique_questions(cells: list[dict]) -> list[dict]:
    """One record per (anchor, qid, horizon) — cells repeat it per event."""
    best: dict[tuple, dict] = {}
    for c in cells:
        hb = c.get("half_b") or {}
        if "p_y" not in hb:
            continue
        key = (c.get("game_id"), c.get("qid"), c.get("horizon", "H1"))
        if key in best:
            continue
        ha = c.get("half_a") or {}
        best[key] = {
            "anchor": c.get("game_id"),
            "qid": c.get("qid"),
            "horizon": c.get("horizon", "H1"),
            "resolution_turn": c.get("resolution_turn"),
            "question": c.get("question"),
            "template_id": c.get("template_id"),
            "parameters": c.get("parameters"),
            "p_y": hb.get("p_y"),
            "p_y_half_a": ha.get("p_y"),
            "resolution_outcome": c.get("resolution_outcome"),
        }
    return list(best.values())


def reliability(q: dict) -> float:
    a, b = q.get("p_y_half_a"), q.get("p_y")
    if a is None or b is None:
        return 1.0
    return abs(a - b)


def pick(pool: list[dict], cap_anchor: int, cap_horizon: int) -> list[dict]:
    chosen: list[dict] = []
    a_ct: Counter = Counter()
    h_ct: Counter = Counter()
    for q in sorted(pool, key=reliability):
        if a_ct[q["anchor"]] >= cap_anchor or h_ct[q["horizon"]] >= cap_horizon:
            continue
        chosen.append(q)
        a_ct[q["anchor"]] += 1
        h_ct[q["horizon"]] += 1
    return chosen


def histogram(values: list[float], edges: list[float]) -> dict:
    out = {}
    for lo, hi in zip(edges, edges[1:]):
        out[f"{lo:g}-{hi:g}"] = sum(1 for v in values if lo <= v < hi)
    return out


def table(rows: list[dict]) -> dict:
    ps = sorted(r["p_y"] for r in rows)
    return {
        "n": len(rows),
        "by_anchor": dict(Counter(r["anchor"] for r in rows)),
        "by_horizon": dict(Counter(r["horizon"] for r in rows)),
        "by_template": dict(Counter(r["template_id"] for r in rows)),
        "p_y_quantiles": {
            q: (ps[int(f * (len(ps) - 1))] if ps else None)
            for q, f in (("p10", 0.1), ("p50", 0.5), ("p90", 0.9))
        },
        "mean_split_half_gap": (
            round(sum(reliability(r) for r in rows) / len(rows), 4) if rows else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--tail-hi", type=float, default=TAIL_HI)
    ap.add_argument("--mirror-lo", type=float, default=MIRROR_LO)
    ap.add_argument("--max-per-anchor", type=int, default=MAX_PER_ANCHOR)
    ap.add_argument("--max-per-horizon", type=int, default=MAX_PER_HORIZON)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    qs = unique_questions(load_cells(args.cells))
    tail_pool = [q for q in qs if 0 < q["p_y"] <= args.tail_hi]
    mirror_pool = [q for q in qs if args.mirror_lo <= q["p_y"] < 1]
    degenerate_zero = [q for q in qs if q["p_y"] == 0]
    degenerate_one = [q for q in qs if q["p_y"] == 1]

    tail = pick(tail_pool, args.max_per_anchor, args.max_per_horizon)
    mirror = pick(mirror_pool, args.max_per_anchor, args.max_per_horizon)

    out = {
        "meta": {
            "source_cells": args.cells,
            "bands": {"tail": (0, args.tail_hi), "mirror": (args.mirror_lo, 1)},
            "caps": {"per_anchor": args.max_per_anchor, "per_horizon": args.max_per_horizon},
            "truth": "half-B held-out unconditional probability over ~500 replays",
            "pool_sizes": {
                "questions_total": len(qs),
                "tail_pool": len(tail_pool),
                "mirror_pool": len(mirror_pool),
                "excluded_p_y_zero": len(degenerate_zero),
                "excluded_p_y_one": len(degenerate_one),
            },
        },
        "summary": {
            "tail": table(tail),
            "mirror": table(mirror),
            "tail_p_y_histogram": histogram(
                [q["p_y"] for q in tail], [0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
            ),
        },
        "tail_questions": tail,
        "mirror_questions": mirror,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"questions {len(qs)} -> tail {len(tail)}/{len(tail_pool)} + mirror "
          f"{len(mirror)}/{len(mirror_pool)}; excluded p=0:{len(degenerate_zero)} "
          f"p=1:{len(degenerate_one)} -> {args.out}")
    print(json.dumps(out["summary"], indent=1))


if __name__ == "__main__":
    main()
