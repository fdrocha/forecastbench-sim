#!/usr/bin/env -S uv run python3
"""Gather the article's data: one long CSV per eval, under data/micropolis/paper/.

The scoring half of the paper pipeline. Reads the two evals' gathered datasets
and the ground truth, scores every forecast with the reports' own scoring
functions, and writes one row per (model, question) — the grain the scores are
actually computed at, not a per-model mean:

- binary_forecasts.csv: model, question, section (mid-range or tail, by the
  question instance's ground-truth P(Yes)), horizon, the forecast, that p, and
  the Brier, excess Brier and excess bits.
- continuous_forecasts.csv: model, question, city, metric, horizon, the raw
  CRPS and the excess CRPS. Both are unnormalized: the paper divides by the
  city's own scale, which is a join analyze_paper.py does against the file
  below, so the scale can be changed without re-scoring.
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
    Normalizer,
    attach_outcomes,
    data_path,
    load_dataset,
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

# The paper normalizes downstream, per city and metric, so nothing is divided
# here: score_forecasts still takes a Normalizer, and this one has no scale for
# any question, which leaves its normalized columns empty. They are not
# written. Scoring stays the reports' own; only the denominator moves.
RAW = Normalizer(
    mode="none",
    ratio="CRPS",
    detail="nothing — the paper's rows are unnormalized",
    scale=lambda c: None,
)

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
    "city",
    "metric",
    "horizon",
    "crps",
    "excess_crps",
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


def city_of(scenario_id: str) -> str:
    """The city a scenario id names.

    CitySimulation builds it as {city}_{disasters flag}_seed{n}, and a city
    name can itself hold underscores ("med_isle"), so the suffixes are stripped
    rather than the first segment taken.
    """
    head = scenario_id.rsplit("_seed", 1)[0]
    return head.removesuffix("_disasters").removesuffix("_nodisasters")


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
    """One row per scored continuous forecast, unnormalized.

    The read-off horizon is excluded, as in every pooled figure the reports
    draw: it asks for a number the snapshot report already prints, so averaging
    it in with real forecasts flatters every model by the same trick.
    Unnormalized metrics (city funds) are dropped too — they have no scale, so
    the paper could not pool them with the rest.

    The city is carried as a column: it is what analyze_paper.py joins the
    per-city scales on, and the question id encodes it only by convention.
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
        without_outcomes = attach_outcomes(scorable)
    except (NotImplementedError, FileNotFoundError) as e:
        sys.exit(
            f"[error] {e}\n"
            "  the excess measure needs the ground truth; run "
            "scripts/extract_ground_truth.py for this config"
        )
    # A metric the scales table has no column for could not be normalized
    # downstream, and would drop out of the paper's figures there rather than
    # here, where the corpus is in hand to say so.
    unscaled = sorted({c["metric"] for c in scorable} - set(SCALE_METRICS))
    if unscaled:
        sys.exit(
            f"[error] {SCALES_CSV_NAME} has no scale for: {', '.join(unscaled)}\n"
            "  add the metric to get_city_scales.METRICS, or to "
            "UNNORMALIZED_METRICS to leave it out of the paper"
        )
    if without_outcomes:
        warn(
            f"{without_outcomes} continuous question(s) have no continuation"
            " outcomes; their excess columns are empty"
        )

    cities = {c["question_id"]: city_of(c["scenario_id"]) for c in scorable}
    rows = [
        {
            "model": to_model_id(r["model_id"]),
            "question_id": r["question_id"],
            "city": cities[r["question_id"]],
            "metric": r["metric"],
            "horizon": years(r["horizon"]),
            "crps": r["crps"],
            "excess_crps": r["excess_crps"],
        }
        for r in score_forecasts(scorable, responses, models, RAW)
        # Belt and braces: forecast_questions already removed the read-off.
        if is_forecast(r["horizon"])
    ]
    print(f"            {len(rows)} scored forecasts, unnormalized")
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
