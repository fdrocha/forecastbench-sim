#!/usr/bin/env python3
"""Score v2 natural-conditional elicitations against half-B ground truth.

Primary metric: excess log loss KL(q || p_hat) in bits, with p_hat clipped to
[0.001, 0.999]. q is the held-out half-B truth (p_yx for conditional stages,
p_y for baseline calibration) and is never clipped. Secondary: squared error
(p_hat - q)^2.

Skill keeps the v1 ladder CUS form, per band:
  CUS = 1 - loss(updated) / loss(held baseline)
where both losses are against the conditional truth q = half_b.p_yx — i.e.
how much updating on the revealed fact beat holding the baseline forecast.
Direction accuracy (certified effect cells only) is against the half-B delta:
correct iff (p_cond - p_base) * delta_B > 0.

Confidence intervals: deterministic paired qid-cluster percentile bootstrap.

Usage:
  uv run python scripts/uplift_v2/natcond_score_v2.py \
      --cells tmp/natcond_v2/w01_cells.json \
      --results tmp/natcond_v2/elicit_w01_oss120.json \
      --output tmp/natcond_v2/score_w01_oss120.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

CLIP_LO, CLIP_HI = 0.001, 0.999


def clip(p: float) -> float:
    return min(CLIP_HI, max(CLIP_LO, float(p)))


def kl_bits(q: float, p: float) -> float:
    """Bernoulli KL(q || p) in bits, with the 0*log(0) := 0 convention.

    q is a true probability (may be exactly 0 or 1); p must already be
    clipped away from {0, 1}.
    """
    out = 0.0
    if q > 0:
        out += q * math.log2(q / p)
    if q < 1:
        out += (1 - q) * math.log2((1 - q) / (1 - p))
    return out


def sq_err(q: float, p: float) -> float:
    return (p - q) ** 2


LOSSES = {"kl_bits": kl_bits, "sq": sq_err}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_predictions(payload: dict) -> tuple[dict, dict]:
    """qid -> mean base p; (qid, event_id) -> mean cond p (over samples)."""
    base: dict[str, list[float]] = defaultdict(list)
    conditional: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in payload["results"]:
        p = row.get("p")
        if p is None:
            continue
        if row["stage"] == "base":
            base[row["qid"]].append(float(p))
        elif row["stage"] == "cond":
            conditional[(row["qid"], row["event_id"])].append(float(p))
        else:
            raise RuntimeError(f"unknown stage {row['stage']!r}")
    return ({k: mean(v) for k, v in base.items()},
            {k: mean(v) for k, v in conditional.items()})


def join_rows(cells: list[dict], payload: dict) -> tuple[list[dict], dict]:
    """Attach pb/pc predictions and half-B truths to each scorable cell."""
    base, conditional = load_predictions(payload)
    rows, missing = [], {"no_base": 0, "no_cond": 0}
    for cell in cells:
        key = (cell["qid"], cell["event_id"])
        if cell["qid"] not in base:
            missing["no_base"] += 1
            continue
        if key not in conditional:
            missing["no_cond"] += 1
            continue
        row = dict(cell)
        row["pb"] = base[cell["qid"]]
        row["pc"] = conditional[key]
        rows.append(row)
    return rows, missing


def band_metrics(rows: list[dict]) -> dict[str, Any] | None:
    """Losses / CUS / direction for one band of joined rows."""
    if not rows:
        return None
    out: dict[str, Any] = {"n": len(rows)}
    for name, loss in LOSSES.items():
        updated = mean([loss(r["half_b"]["p_yx"], clip(r["pc"])) for r in rows])
        held = mean([loss(r["half_b"]["p_yx"], clip(r["pb"])) for r in rows])
        out[name] = {
            "loss_updated": updated,
            "loss_baseline": held,
            "cus": 1 - updated / held if held > 0 else None,
        }
    effects = [r for r in rows if r["cls"] == "effect"]
    if effects:
        correct = sum(1 for r in effects
                      if (r["pc"] - r["pb"]) * r["half_b"]["delta"] > 0)
        out["direction"] = {"correct": correct, "total": len(effects),
                            "accuracy": correct / len(effects)}
    placebos = [r for r in rows if r["cls"] == "placebo"]
    if effects and placebos:
        eff_upd = mean([abs(r["pc"] - r["pb"]) for r in effects])
        pla_upd = mean([abs(r["pc"] - r["pb"]) for r in placebos])
        out["mean_abs_update_effect"] = eff_upd
        out["mean_abs_update_placebo"] = pla_upd
        out["selectivity"] = eff_upd / pla_upd if pla_upd > 0 else None
    out["baseline_calibration_kl_bits"] = mean(
        [kl_bits(r["half_b"]["p_y"], clip(r["pb"])) for r in rows])
    return out


def cus_for_qids(rows_by_qid: dict, qids: list[str], loss_name: str) -> float:
    num = den = 0.0
    loss = LOSSES[loss_name]
    for qid in qids:
        for r in rows_by_qid.get(qid, ()):
            num += loss(r["half_b"]["p_yx"], clip(r["pc"]))
            den += loss(r["half_b"]["p_yx"], clip(r["pb"]))
    return 1 - num / den if den > 0 else float("nan")


def bootstrap_ci(rows: list[dict], loss_name: str,
                 replicates: int, seed: int) -> list[float] | None:
    """Deterministic paired qid-cluster percentile bootstrap of the CUS."""
    rows_by_qid: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        rows_by_qid[r["qid"]].append(r)
    qids = sorted(rows_by_qid)
    if not qids:
        return None
    rng = random.Random(seed)
    values = []
    for _ in range(replicates):
        sample = [rng.choice(qids) for _ in qids]
        values.append(cus_for_qids(rows_by_qid, sample, loss_name))
    finite = [v for v in values if math.isfinite(v)]
    if not finite:
        return None
    return [quantile(finite, 0.025), quantile(finite, 0.975)]


def score(cells: list[dict], payload: dict,
          bootstrap: int = 2000, seed: int = 20260818) -> dict[str, Any]:
    rows, missing = join_rows(cells, payload)
    bands: dict[str, list[dict]] = {"overall": rows}
    for cls in ("effect", "placebo"):
        bands[cls] = [r for r in rows if r["cls"] == cls]
    for hz in sorted({r["horizon"] for r in rows if r.get("horizon")}):
        bands[hz] = [r for r in rows if r.get("horizon") == hz]

    report: dict[str, Any] = {
        "schema": "natcond_score_v2",
        "mode": payload.get("mode"),
        "tag": payload.get("tag"),
        "model": payload.get("model"),
        "clip": [CLIP_LO, CLIP_HI],
        "n_cells": len(cells),
        "n_scored": len(rows),
        "missing": missing,
        "bootstrap": {"method": "paired qid-cluster percentile",
                      "replicates": bootstrap, "seed": seed},
        "bands": {},
    }
    for name, band_rows in bands.items():
        metrics = band_metrics(band_rows)
        if metrics is None:
            continue
        if name in ("overall", "effect") and bootstrap > 0:
            for loss_name in LOSSES:
                metrics[loss_name]["cus_95pct_ci"] = bootstrap_ci(
                    band_rows, loss_name, bootstrap, seed)
        report["bands"][name] = metrics
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    cells = json.loads(args.cells.read_text())
    payload = json.loads(args.results.read_text())
    for cell in cells:
        if "half_b" not in cell:
            raise RuntimeError(
                f"cell {cell.get('qid')}/{cell.get('event_id')} has no half_b "
                f"truth — score_v2 needs cells from natcond_cells_v2.py")

    report = score(cells, payload, args.bootstrap, args.seed)
    report["inputs"] = {"cells_sha256": sha256(args.cells),
                        "results_sha256": sha256(args.results)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
