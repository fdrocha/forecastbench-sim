#!/usr/bin/env -S uv run python3
"""Derive FreeCiv-style normalization constants per metric and compare them
with continuous_eval.GLOBAL_SCALES.

FreeCiv pools excess CRPS across question families by dividing each family's
items by one constant fixed at the draw: the median over the family's items
of the ground truth's p05-p95 range, rounded to one significant figure. Every
Micropolis continuous question is a value-at-T question, so the families are
the metrics, and the analog is computed here from the reseeded continuations
of the config's corpus (data/micropolis/ground_truth/): per metric, the median
over its forecast-horizon questions of the continuations' p10-p90 range (the
levels Micropolis asks; p05-p95 shown beside it), rounded to one significant
figure. Prints a table to stdout; changes nothing. Reads the gathered dataset
so the corpus is exactly the slice analyze_continuous.py scores.

Usage:
    scripts/derive_scales.py                   # configs/continuous.json5
    scripts/derive_scales.py --cities kyoto --disasters false
"""

import argparse
import math
import statistics
import sys

import numpy as np

import micropolis_world.module_globals as g
from micropolis_world.config import (
    CONFIG_DIR,
    add_config_args,
    load_config,
    main_with_config,
)
from micropolis_world.continuous_eval import (
    GLOBAL_SCALES,
    UNNORMALIZED_METRICS,
    DatasetError,
    data_path,
    load_dataset,
    select_for_config,
)
from micropolis_world.ground_truth import load_outcomes

DEFAULT_CONFIG_PATH = CONFIG_DIR / "continuous.json5"

# Horizon 0 is a read-off with no spread, so it is left out as everywhere.
READ_OFF_HORIZON = 0


def round_to_one_significant_figure(x: float) -> float:
    if x == 0:
        return 0.0
    return round(x, -math.floor(math.log10(abs(x))))


def spread(values: list[float], lo: float, hi: float) -> float:
    """The lo-hi quantile range of the continuations, inverted_cdf like the floor."""
    q = np.quantile(np.asarray(values, dtype=float), [lo, hi], method="inverted_cdf")
    return float(q[1] - q[0])


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_config_args(ap, default=DEFAULT_CONFIG_PATH)
    args = ap.parse_args()
    cfg = load_config(args)

    try:
        corpus, responses, models = load_dataset(data_path(cfg.get_label(args.label)))
        corpus, _responses, _models = select_for_config(
            corpus,
            responses,
            models,
            cfg,
            cfg.get_seed(args.seed),
            cities=args.cities,
            disasters=args.disasters,
            models=args.models,
        )
        outcomes = load_outcomes(corpus)
    except (FileNotFoundError, DatasetError) as e:
        sys.exit(f"[error] {e}")

    # Per metric, one range per forecast question.
    ranges: dict[str, dict[str, list[float]]] = {}
    for c in corpus:
        values = outcomes.get(c["question_id"])
        if c["horizon"] == READ_OFF_HORIZON or not values:
            continue
        per = ranges.setdefault(c["metric"], {"p10-p90": [], "p05-p95": []})
        per["p10-p90"].append(spread(values, 0.10, 0.90))
        per["p05-p95"].append(spread(values, 0.05, 0.95))

    metrics = list(dict.fromkeys(c["metric"] for c in corpus))
    label = {m: str(g.METRIC_LABELS.get(m, m)) for m in metrics}
    width = max(len(v) for v in label.values())
    header = (
        f"{'metric':<{width}}  {'questions':>9}  {'median p10-p90':>14}  "
        f"{'rounded':>8}  {'median p05-p95':>14}  {'rounded':>8}  {'GLOBAL_SCALES':>13}  ratio"
    )
    print(header)
    print("-" * len(header))
    for m in metrics:
        current = GLOBAL_SCALES.get(m)
        cur = (
            "excluded"
            if m in UNNORMALIZED_METRICS
            else (f"{current:,.0f}" if current is not None else "none")
        )
        if m not in ranges:
            print(
                f"{label[m]:<{width}}  {0:>9}  {'n/a':>14}  {'n/a':>8}  {'n/a':>14}  {'n/a':>8}  {cur:>13}"
            )
            continue
        med10 = statistics.median(ranges[m]["p10-p90"])
        med05 = statistics.median(ranges[m]["p05-p95"])
        c10 = round_to_one_significant_figure(med10)
        c05 = round_to_one_significant_figure(med05)
        ratio = f"{c10 / current:.2f}" if current else ""
        print(
            f"{label[m]:<{width}}  {len(ranges[m]['p10-p90']):>9}  {med10:>14,.2f}  "
            f"{c10:>8,g}  {med05:>14,.2f}  {c05:>8,g}  {cur:>13}  {ratio}"
        )
    print()
    print(
        "rounded = the median to one significant figure, FreeCiv's rule for C_family;"
        " ratio = the rounded p10-p90 constant over the current GLOBAL_SCALES entry."
    )


if __name__ == "__main__":
    main()
