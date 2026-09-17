#!/usr/bin/env -S uv run python3
"""Gather the article's data: one long CSV per eval, under data/micropolis/paper/.

The scoring half of the paper pipeline. Reads the two evals' gathered datasets
and the ground truth, scores every forecast with the reports' own scoring
functions, and writes one row per (model, question) — the grain the scores are
actually computed at, not a per-model mean:

- binary_forecasts.csv: model, question, section (mid-range or tail, by the
  question instance's ground-truth P(Yes)), horizon, the forecast, that p, and
  the Brier, excess Brier and excess bits.
- continuous_forecasts.csv: model, question, metric, horizon, the raw CRPS and
  its global normalization, and the excess CRPS and its normalization.
- city_metric_scales.csv: one row per city of the continuous config, the mean
  each metric took over the turns up to the first snapshot, floored at a
  per-metric minimum so a quiet city cannot give a near-zero scale.
- model_scores.csv, copied verbatim from the package's datafiles/, so the
  paper's directory carries the ECI and ForecastBench numbers its figures plot
  against rather than depending on the repo's copy at drawing time.

Model names are written as model *ids*: this world's ":suffix" (":loeff", the
reasoning effort a run was gathered under) is dropped, since the paper reports
one run per model and the leaderboards score the model rather than the effort
setting. The suffix stays the canonical id everywhere upstream — configs, the
response cache, data.json — so it is stripped here, at the boundary, and not
before.

Aggregates are deliberately not written here: every mean, correlation and
bootstrap interval the paper reports is recoverable from these rows, and a
pre-aggregated file beside them would be a second source of truth to keep in
step. The reports' own binary_scores.csv and continuous_scores.csv still live
beside the reports, under the eval labels.

Unparsed forecasts are dropped rather than written as nan: a row here means a
score, and the counts of what was prompted against what parsed belong to the
reports. Both datasets are required — the paper's data is all-or-nothing — and
there is no --incomplete, since a ragged selection makes per-model figures
cover different question sets.

scripts/analyze_paper.py draws the figures from this directory alone.

Usage:
    scripts/gather_paper_data.py
    scripts/gather_paper_data.py --binary-config configs/binary-testing.json5
    scripts/gather_paper_data.py --continuous-config configs/prompt-a.json5
"""

import argparse
import shutil
import sys
from pathlib import Path

import micropolis_world.module_globals as g
from micropolis_world.binary_eval import (
    data_path as binary_data_path,
)
from micropolis_world.binary_eval import (
    load_dataset_binary,
)
from micropolis_world.config import CONFIG_DIR, Config, ConfigError, main_with_config
from micropolis_world.continuous_eval import (
    DatasetError,
    attach_outcomes,
    data_path,
    load_dataset,
    make_normalizer,
    score_forecasts,
    select_for_config,
)
from micropolis_world.ground_truth import load_truths
from micropolis_world.messages import error, warn
from micropolis_world.model_ids import to_model_id
from micropolis_world.model_scores import SCORES_PATH

# Imported rather than reimplemented so the paper's scores are the reports'
# scores: the same section rule, the same Brier/excess/bits, the same CRPS.
sys.path.insert(0, str(Path(__file__).parent))
from analyze_binary import (
    TURNS_PER_YEAR,
    score_forecasts_binary,
    section_of,
    write_csv,
)
from analyze_continuous import UNNORMALIZED_METRICS, forecast_questions, is_forecast
from get_city_scales import METRICS as SCALE_METRICS
from get_city_scales import collect_views

DEFAULT_CONTINUOUS_CONFIG_PATH = CONFIG_DIR / "continuous.json5"
DEFAULT_BINARY_CONFIG_PATH = CONFIG_DIR / "binary.json5"

# The paper's own directory, flat: the figures are cited from the article by a
# fixed path, so the config's label picks which data.json is read, not where
# these land.
OUT_DIR = g.DATA_DIR / "paper"

# The one continuous normalization the paper reports. The three modes are not
# comparable with each other, and `global` is the one whose scale is fixed per
# metric — so a cell means the same thing across scenarios, snapshots and
# horizons, and derive_scales.py can check it against FreeCiv's. The other two
# modes are not built at all, which is also what keeps this script off
# load_averages and load_expected_persistence.
NORM_MODE = "global"

BINARY_CSV_NAME = "binary_forecasts.csv"
BINARY_COLUMNS = [
    "model",
    "question_id",
    "qid",
    "section",
    "horizon",
    "forecast",
    "real_prob",
    "brier",
    "excess_brier",
    "excess_bits",
]

# Copied beside the scores rather than read from the package when the figures
# are drawn: the paper's directory should hold everything its numbers rest on,
# and the repo's copy will keep gaining rows as leaderboards publish.
MODEL_SCORES_CSV_NAME = "model_scores.csv"

SCALES_CSV_NAME = "city_metric_scales.csv"
SCALES_COLUMNS = ["city", *SCALE_METRICS]

# The turn the scale window starts at: the city as shipped, before any of it
# has run.
SCALES_START_TURN = 0

# Floors on those means, per metric. A city that never gets going would
# otherwise contribute a scale near zero, which as a denominator turns its
# small absolute errors into large relative ones.
SCALE_FLOORS = {
    "cityPop": 10_000,
    "trafficAverage": 10,
    "pollutionAverage": 40,
    "crimeAverage": 40,
    "landValueAverage": 40,
}

CONTINUOUS_CSV_NAME = "continuous_forecasts.csv"
CONTINUOUS_COLUMNS = [
    "model",
    "question_id",
    "metric",
    "horizon",
    "crps",
    "ncrps",
    "excess_crps",
    "excess_ncrps",
]


def load_config_at(path: Path | str) -> Config:
    """Load one config by path, exiting with a message rather than a traceback."""
    try:
        return Config.load(path)
    except ConfigError as e:
        error(str(e))
        sys.exit(1)


def years(turns: int) -> str:
    """A horizon in turns as the years the CSV reports it in."""
    return f"{turns / TURNS_PER_YEAR:g}y"


def check_no_suffix_collisions(models: list[str]) -> None:
    """Fail if dropping the ":suffix" would merge two of the run's models.

    The paper names models by model id, so two slugs of one base model — "o3"
    and "o3:loeff" gathered together — would land in the CSV under one name and
    have their forecasts averaged as if they were one model. That is a config
    the paper cannot report as it stands, so it is an error here rather than a
    silent merge downstream.
    """
    merged: dict[str, list[str]] = {}
    for slug in models:
        merged.setdefault(to_model_id(slug), []).append(slug)
    clashes = {mid: slugs for mid, slugs in merged.items() if len(slugs) > 1}
    if clashes:
        sys.exit(
            "[error] dropping the ':suffix' would merge these models:\n"
            + "\n".join(f"  {mid}: {', '.join(s)}" for mid, s in clashes.items())
            + "\n  the paper names models by model id, so select one variant per"
            " model (--models, or the config's list)"
        )


def binary_rows(cfg: Config) -> list[dict]:
    """One row per scored binary forecast.

    The section is decided per question *instance* by its ground-truth P(Yes),
    the reports' rule, so a qid can be mid-range in one city and tail in
    another — which is why it is a column here rather than something the
    plotting script could derive from the qid.
    """
    label = cfg.get_label(None)
    data_file = binary_data_path(label)
    try:
        corpus, responses, models = load_dataset_binary(data_file)
        corpus, responses, models = select_for_config(
            corpus,
            responses,
            models,
            cfg,
            cfg.get_seed(None),
            rerun_hint="scripts/run_eval_binary.py",
        )
        truths = load_truths(corpus)
    except (FileNotFoundError, DatasetError) as e:
        sys.exit(f"[error] {e}")

    print(f"binary:     {data_file}")
    print(f"            {len(corpus)} questions x {len(models)} models")
    check_no_suffix_collisions(models)

    sections = {c["question_id"]: section_of(c, truths) for c in corpus}
    scored = score_forecasts_binary(corpus, responses, models, truths)
    rows = [
        {
            "model": to_model_id(r["model_id"]),
            "question_id": r["question_id"],
            "qid": r["qid"],
            "section": sections[r["question_id"]],
            "horizon": years(r["horizon"]),
            "forecast": r["forecast"],
            "real_prob": r["truth"].p,
            "brier": r["brier"],
            "excess_brier": r["excess_brier"],
            "excess_bits": r["excess_bits"],
        }
        for r in scored
    ]
    counts = {
        s: sum(1 for r in rows if r["section"] == s) for s in set(sections.values())
    }
    print(
        f"            {len(rows)} scored forecasts: "
        + ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
    )
    return rows


def continuous_rows(cfg: Config) -> list[dict]:
    """One row per scored continuous forecast, under the global normalization.

    The read-off horizon is excluded, as in every pooled figure the reports
    draw: it asks for a number the snapshot report already prints, so averaging
    it in with real forecasts flatters every model by the same trick.
    Unnormalized metrics (city funds) are dropped too — they have no scale
    under any mode, so their normalized columns would be empty.
    """
    label = cfg.get_label(None)
    seed = cfg.get_seed(None)
    data_file = data_path(label)
    try:
        corpus, responses, models = load_dataset(data_file)
        corpus, responses, models = select_for_config(
            corpus, responses, models, cfg, seed
        )
    except (FileNotFoundError, DatasetError) as e:
        sys.exit(f"[error] {e}")

    print(f"continuous: {data_file}")
    print(f"            {len(corpus)} questions x {len(models)} models")
    check_no_suffix_collisions(models)

    scorable = [
        c for c in forecast_questions(corpus) if c["metric"] not in UNNORMALIZED_METRICS
    ]
    try:
        norm = make_normalizer(
            NORM_MODE,
            scorable,
            global_frac=cfg.get_norm_global_frac(None),
            seed=seed,
        )
        without_outcomes = attach_outcomes(scorable)
    except (NotImplementedError, FileNotFoundError) as e:
        sys.exit(
            f"[error] {e}\n"
            "  the excess measure needs the ground truth; run "
            "scripts/extract_ground_truth.py for this config"
        )
    # A metric with no scale would silently drop out of the normalized columns,
    # leaving the CSV quieter than it looks.
    unscaled = norm.unscaled_metrics(scorable)
    if unscaled:
        sys.exit(
            f"[error] the {NORM_MODE} normalization has no scale for: "
            f"{', '.join(unscaled)}\n"
            "  add one to continuous_eval.GLOBAL_SCALES, or to "
            "UNNORMALIZED_METRICS to leave the metric out"
        )
    if without_outcomes:
        warn(
            f"{without_outcomes} continuous question(s) have no continuation"
            " outcomes; their excess columns are empty"
        )

    rows = [
        {
            "model": to_model_id(r["model_id"]),
            "question_id": r["question_id"],
            "metric": r["metric"],
            "horizon": years(r["horizon"]),
            "crps": r["crps"],
            "ncrps": r["normalized"],
            "excess_crps": r["excess_crps"],
            "excess_ncrps": r["excess_normalized"],
        }
        for r in score_forecasts(scorable, responses, models, norm)
        # Belt and braces: forecast_questions already removed the read-off.
        if is_forecast(r["horizon"])
    ]
    print(f"            {len(rows)} scored forecasts, {NORM_MODE} normalization")
    return rows


def scale_rows(cfg: Config) -> list[dict]:
    """One row per city: its metric means up to the first snapshot, floored."""
    seed = cfg.get_seed(None)
    snapshot = cfg.get_int_list("snapshot_turns")[0]
    views = collect_views(cfg, seed, SCALES_START_TURN, snapshot)
    means = views["mean"]
    if len(means) != len(cfg.get_cities(None)):
        sys.exit(
            "[error] no cached sim log for: "
            + ", ".join(c for c in cfg.get_cities(None) if c not in means)
            + "\n  run scripts/run_sim.py for this config"
        )
    print(f"scales:     {len(means)} cities, mean over turns "
          f"{SCALES_START_TURN}-{snapshot}, floored")
    return [
        {"city": city, **{m: max(values[m], SCALE_FLOORS[m]) for m in SCALE_METRICS}}
        for city, values in means.items()
    ]


@main_with_config
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--continuous-config",
        default=DEFAULT_CONTINUOUS_CONFIG_PATH,
        help="JSON5 config for the continuous half "
        f"(default: {DEFAULT_CONTINUOUS_CONFIG_PATH})",
    )
    ap.add_argument(
        "--binary-config",
        default=DEFAULT_BINARY_CONFIG_PATH,
        help="JSON5 config for the binary half "
        f"(default: {DEFAULT_BINARY_CONFIG_PATH})",
    )
    args = ap.parse_args()

    continuous_cfg = load_config_at(args.continuous_config)
    binary_cfg = load_config_at(args.binary_config)

    print("=" * 70)
    print("MICROPOLIS WORLD — paper data")
    print("=" * 70)
    print(f"configs:    {continuous_cfg.path}")
    print(f"            {binary_cfg.path}")
    print(f"out:        {OUT_DIR}")
    print()

    binary = binary_rows(binary_cfg)
    continuous = continuous_rows(continuous_cfg)
    scales = scale_rows(continuous_cfg)

    print()
    for name, columns, rows in [
        (BINARY_CSV_NAME, BINARY_COLUMNS, binary),
        (CONTINUOUS_CSV_NAME, CONTINUOUS_COLUMNS, continuous),
        (SCALES_CSV_NAME, SCALES_COLUMNS, scales),
    ]:
        print(f"Wrote {write_csv(OUT_DIR / name, columns, rows)}")

    # Verbatim, header and blank cells included: analyze_paper.py reads it
    # through the package's own parser, so it has to stay in that format.
    scores_copy = OUT_DIR / MODEL_SCORES_CSV_NAME
    scores_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SCORES_PATH, scores_copy)
    print(f"Wrote {scores_copy}")


if __name__ == "__main__":
    main()
