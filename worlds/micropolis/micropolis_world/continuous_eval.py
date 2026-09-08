"""Shared pieces of the continuous eval: response cache, dataset, scoring.

The eval is split across three scripts — run_eval_continuous.py gathers model
responses, analyze_continuous.py scores them, plot_forecasts.py draws them —
and this module holds what more than one of them needs. The dataset written by
the first is the only thing the other two read, so they never re-simulate or
re-prompt.
"""

import json
import os
import re
import subprocess
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from fbsim_core.metrics import compute_crps

from . import messages as msg
from . import module_globals as g
from .city_sim import CitySimulation
from .config import Config, scenarios_from
from .gather import (  # noqa: F401 - re-exported for the scripts and tests
    EvalPaths,
    batch_id_for,
    group_into_batches,
    write_dataset,
)
from .knowledge_eval.runner import prompt_hash  # noqa: F401 - re-exported
from .usage import load_usage, save_usage  # noqa: F401 - re-exported for the scripts

# Batch prompts and raw model responses are cached here, shared across every
# label: the cache filename already carries the prompt's hash (see
# batch_dir), so two labels asking an identical prompt reuse the same cached
# response rather than paying for it twice.
OUT_DIR = g.DATA_DIR / "continuous"


def label_dir(label: str) -> Path:
    """Where one label's dataset, plots and reports are written.

    Kept apart per label — unlike OUT_DIR's shared cache — so runs made under
    different prompt variants never overwrite each other's output.
    """
    return OUT_DIR / label


def data_path(label: str) -> Path:
    return label_dir(label) / "data.json"


def plots_path(label: str) -> Path:
    return label_dir(label) / "plots"


# Metrics no normalization mode covers, so they appear in the raw CRPS tables
# and nowhere else. City funds swings through zero and changes sign — a
# bankrupt city stays broke — which leaves it with no scale worth dividing by
# under any of the modes below; the whole metric is excluded rather than
# dropping the individual questions and averaging over a silently different
# question set per scenario. Named as well as absent from GLOBAL_SCALES so a
# metric with no scale can be told from one whose scale was forgotten.
UNNORMALIZED_METRICS = {"totalFunds"}

# Denominators for --norm global: one fixed scale per metric, the same for
# every question. Round numbers on the order of a metric's plausible range over
# a run rather than anything fitted to the data, so they hold still as cities,
# snapshots and models come and go, and a normalized cell can be compared
# across all of them. The alternative modes divide by something the question
# itself supplies, which makes a cell easier to interpret in isolation and
# harder to compare.
GLOBAL_SCALES: dict[str, float] = {
    "cityPop": 20_000.0,
    "trafficAverage": 20.0,
    "pollutionAverage": 60.0,
    "crimeAverage": 60.0,
    "landValueAverage": 60.0,
}

# --norm's values, "global" first because it is the default.
NORM_MODES = ("global", "local", "baseline")
DEFAULT_NORM = "global"

# The per-question modes below need the ground-truth continuations, which
# scripts/extract_ground_truth.py writes; "global" reads nothing.
GROUND_TRUTH_NORMS = ("local", "baseline")


@dataclass(frozen=True)
class Floored:
    """How often a mode's denominator floor was the binding one.

    A per-question denominator goes to 0 exactly where the metric provably
    could not move — a city whose traffic is pinned at 0 across every
    continuation — and near 0 on the scenarios where it barely could. Flooring
    at a share of the metric's global scale keeps every question in the corpus
    at a bounded denominator, but a floored cell is measuring the floor rather
    than the question, so the count goes in the report beside the numbers it
    affected.
    """

    n: int
    total: int
    frac: float
    by_metric: dict[str, int]

    @property
    def share(self) -> float:
        return self.n / self.total if self.total else 0.0

    def note(self) -> str:
        """One line for the report and for stdout."""
        floor = f"{self.frac:.3g}% of the metric's global scale"
        if not self.n:
            return f"denominator floor ({floor}) bound on no question"
        where = ", ".join(
            f"{g.METRIC_LABELS.get(m, m)} {n}"
            for m, n in sorted(self.by_metric.items(), key=lambda kv: -kv[1])
        )
        return (
            f"denominator floor ({floor}) bound on {self.n} of {self.total} "
            f"question(s) ({self.share:.1%}) — {where}; those cells are scored "
            "against the floor rather than against their own scenario"
        )


@dataclass(frozen=True)
class Normalizer:
    """How CRPS is divided into a unitless score, plus the prose that says so.

    `scale` returns one question's denominator, or None where the question
    cannot be normalized at all: such a forecast keeps its raw CRPS and is left
    out of every normalized aggregate. `ratio` names the quantity on an axis or
    in a column note and `detail` is the one line saying what the denominator
    is, so a report never shows a normalized number without stating what it was
    divided by — the modes are not comparable with each other.

    `floored` is None for a mode with no floor to report — "global" divides by
    the scale itself — and otherwise says how many questions the floor bound.
    """

    mode: str
    ratio: str
    detail: str
    scale: Callable[[dict], float | None]
    floored: Floored | None = None

    def unscaled_metrics(self, corpus: list[dict]) -> list[str]:
        """Corpus metrics this mode has no scale for and does not exclude.

        A metric added to the corpus without a scale would otherwise drop out
        of the normalized tables silently, leaving them narrower than the raw
        ones with nothing to say why.
        """
        scaled = {c["metric"] for c in corpus if self.scale(c) is not None}
        return sorted({c["metric"] for c in corpus} - UNNORMALIZED_METRICS - scaled)


def describe_global_scales() -> str:
    """GLOBAL_SCALES as report prose, in the table's own order."""
    return ", ".join(
        f"{g.METRIC_LABELS.get(m, m)} {scale:,.0f}"
        for m, scale in GLOBAL_SCALES.items()
    )


def floor_scales(
    corpus: list[dict], raw: dict[str, float | None], global_frac: float
) -> tuple[dict[str, float | None], Floored]:
    """`raw` denominators floored at `global_frac` of each metric's scale.

    The floor is what keeps a per-question mode bounded: the raw denominator is
    0 where a metric could not move and arbitrarily small where it barely
    could, and either would let one cell dominate every mean it entered.
    Returns the floored denominators and the tally of where the floor won, so
    the caller can report it rather than let it pass silently.

    A metric on UNNORMALIZED_METRICS, or one with no global scale to take a
    share of, stays None — unnormalized under every mode — however large its
    raw denominator is.
    """
    scales: dict[str, float | None] = {}
    by_metric: Counter[str] = Counter()
    normalizable = 0
    for c in corpus:
        metric = c["metric"]
        scale = GLOBAL_SCALES.get(metric)
        if metric in UNNORMALIZED_METRICS or scale is None:
            scales[c["question_id"]] = None
            continue
        normalizable += 1
        value = raw.get(c["question_id"])
        floor = global_frac * scale
        if value is None or abs(value) < floor:
            by_metric[metric] += 1
            scales[c["question_id"]] = floor
        else:
            scales[c["question_id"]] = abs(value)
    tally = Floored(
        n=sum(by_metric.values()),
        total=normalizable,
        frac=global_frac * 100,
        by_metric=dict(by_metric),
    )
    return scales, tally


def snapshot_rows(corpus: list[dict], seed: int) -> dict[tuple[str, int], dict | None]:
    """Each (scenario, snapshot)'s cached log row at its snapshot turn.

    What the persistence baseline forecasts, read from the run logs the
    analysis already relies on. None for a scenario with no cached run, which
    leaves its questions on the floor rather than unscored.
    """
    rows: dict[tuple[str, int], dict | None] = {}
    histories: dict[str, list[dict] | None] = {}
    for c in corpus:
        key = (c["scenario_id"], c["snapshot_turn"])
        if key in rows:
            continue
        if c["scenario_id"] not in histories:
            histories[c["scenario_id"]] = scenario_history(c["scenario_id"], seed)
        history = histories[c["scenario_id"]]
        rows[key] = (
            history[c["snapshot_turn"]]
            if history and c["snapshot_turn"] < len(history)
            else None
        )
    return rows


def make_normalizer(
    mode: str,
    corpus: list[dict],
    global_frac: float = 0.01,
    seed: int | None = None,
) -> Normalizer:
    """The Normalizer for a --norm mode, over the corpus about to be scored.

    `global_frac` floors the two per-question modes' denominators at that share
    of the metric's global scale (see floor_scales); `seed` names the run logs
    the baseline mode reads its snapshot values from, and is required there.
    """
    if mode == "global":
        return Normalizer(
            mode=mode,
            ratio="CRPS/scale",
            detail=f"a fixed per-metric scale — {describe_global_scales()}",
            scale=lambda c: GLOBAL_SCALES.get(c["metric"]),
        )
    if mode == "local":
        from .ground_truth import load_averages

        scales, floored = floor_scales(corpus, load_averages(corpus), global_frac)
        return Normalizer(
            mode=mode,
            ratio="CRPS/mean",
            detail=(
                "the metric's mean over the reseeded continuations of that "
                f"question's own scenario, snapshot and horizon, floored at "
                f"{global_frac:.3g} of its global scale"
            ),
            scale=lambda c: scales.get(c["question_id"]),
            floored=floored,
        )
    if mode == "baseline":
        from .ground_truth import load_expected_persistence

        if seed is None:
            raise ValueError("--norm baseline needs the config's seed")
        raw = load_expected_persistence(corpus, snapshot_rows(corpus, seed))
        scales, floored = floor_scales(corpus, raw, global_frac)
        return Normalizer(
            mode=mode,
            ratio="CRPS/CRPS_persistence",
            detail=(
                "the expected CRPS of the persistence forecast — the mean of "
                "|snapshot - outcome| over the reseeded continuations of that "
                f"question — floored at {global_frac:.3g} of the metric's "
                "global scale; 1.0 is as good as assuming nothing changes"
            ),
            scale=lambda c: scales.get(c["question_id"]),
            floored=floored,
        )
    raise ValueError(f"unknown normalization mode: {mode!r}")


@dataclass(frozen=True)
class ResponseId:
    model_id: str
    question_id: str


@dataclass(frozen=True)
class Response:
    actual: float
    percentiles: dict[str, float] | None
    response_text: str | None = None


Responses = dict[ResponseId, Response]


# The continuous eval's cache layout. The helpers below are kept as
# module-level functions because the scripts and tests import them by name;
# PATHS is what the shared gather machinery is handed.
PATHS = EvalPaths(OUT_DIR)


def batch_dir(batch_id: str) -> Path:
    """Where a batch's prompt and raw model responses are cached.

    Holds prompt-{hash}.txt plus one response-{model}-{hash}.txt per model that
    has answered it (see prompt_path and response_path), named the same way as
    the knowledge eval's cache. The hash is the prompt's content, so a batch
    directory can hold more than one prompt variant — a template or history_freq
    change simply adds new files alongside the old ones instead of colliding
    with or invalidating them. Deleting the directory re-gathers the batch from
    scratch on the next run.
    """
    return PATHS.batch_dir(batch_id)


def prompt_path(batch_id: str, phash: str) -> Path:
    return PATHS.prompt_path(batch_id, phash)


def response_path(batch_id: str, model_id: str, phash: str) -> Path:
    return PATHS.response_path(batch_id, model_id, phash)


def usage_path(batch_id: str, model_id: str, phash: str) -> Path:
    """Tokens and cost of the call that produced the matching response file."""
    return PATHS.usage_path(batch_id, model_id, phash)


def save_dataset(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    path: Path,
) -> Path:
    """Write the corpus and this run's percentile forecasts to `path`.

    The continuous eval's shape of gather.write_dataset: one "percentiles"
    entry per (question, model) that was gathered.
    """
    forecasts = [
        {
            "model_id": model_id,
            "question_id": c["question_id"],
            "percentiles": r.percentiles,
        }
        for c in corpus
        for model_id in model_names
        for r in [responses.get(ResponseId(model_id, c["question_id"]))]
        if r is not None
    ]
    return write_dataset(corpus, forecasts, model_names, path)


def load_dataset(path: Path) -> tuple[list[dict], Responses, list[str]]:
    """Read back what save_dataset wrote, as (corpus, responses, model_names).

    Returns the same shapes the gathering script works with, so the analysis and
    plotting code is identical whether it was handed live results or a file.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/run_eval_continuous.py first"
        )
    data = json.loads(path.read_text())
    corpus = data["questions"]
    actual = {c["question_id"]: c["value"] for c in corpus}
    responses: Responses = {
        ResponseId(f["model_id"], f["question_id"]): Response(
            actual=actual[f["question_id"]],
            percentiles=f["percentiles"],
        )
        for f in data["forecasts"]
    }
    return corpus, responses, data["models"]


def scenario_history(scenario_id: str, seed: int) -> list[dict] | None:
    """The cached log rows of a scenario run, or None if it isn't on disk.

    Indexed by turn: row i is turn i, matching how build_corpus resolves a
    question at snapshot_turn + horizon. Reads the cache only — never simulates —
    so the analysis scripts stay offline and cost nothing.

    The scenario id encodes city, disasters and seed, which is what names the log
    file, so a run can be found from a corpus entry alone. `seed` is taken as an
    argument rather than parsed back out of the id: the id is built by
    CitySimulation and this should not depend on how it is spelled.
    """
    city, _, rest = scenario_id.partition("_")
    if not rest:
        return None
    sim = CitySimulation(
        city_name=city, seed=seed, disasters=rest.startswith("disasters")
    )
    try:
        sim.load_from_disk()
    except FileNotFoundError:
        return None
    return sim.log_data


def git_commit_note(repo: Path | None = None, count_untracked: bool = True) -> str:
    """The short commit hash HEAD is at, flagged '(dirty)' if the tree differs.

    Recorded at the top of every analysis report: the tables and figures are
    computed from code, and a report generated mid-edit should say so rather
    than imply it came from a clean, identifiable commit.

    `repo` is the working directory to ask about, defaulting to this process's.
    `count_untracked` treats untracked files as making the tree dirty, which is
    right for this repo — a new, not-yet-added source file can change the
    numbers — and wrong for the engine checkout, where build and run artifacts
    sit untracked permanently and would pin the flag on forever.
    """
    status = ["git", "status", "--porcelain"]
    if not count_untracked:
        status.append("--untracked-files=no")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repo,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                status, capture_output=True, text=True, check=False, cwd=repo
            ).stdout.strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return "unknown (not a git checkout, or git is unavailable)"
    return f"{commit} (dirty)" if dirty else commit


def engine_commit_note() -> str:
    """git_commit_note for the MicropolisCore checkout the sims are run with.

    The engine decides the trajectories every question resolves against, so it
    is as much a part of a result's provenance as this repo is. Untracked files
    are ignored: the checkout carries build output and sim-runs directories that
    are not part of the engine's source.
    """
    return git_commit_note(g.MICROPOLIS_APP_PATH, count_untracked=False)


class MdReport:
    """Accumulates one script run's tables, figures and commentary as Markdown.

    Every print_*/plot_* function that used to write straight to stdout instead
    appends to a report instance passed in as an argument, so a run's whole
    output ends up in one .md file rather than scattered across the console.
    Tables are kept as the fixed-width text they were already formatted into —
    reformatting them as native Markdown tables would mean redoing the column
    alignment logic for no reader benefit — wrapped in a code fence so a
    Markdown viewer renders the alignment as written.
    """

    def __init__(self) -> None:
        self._parts: list[str] = []

    def heading(self, text: str, level: int = 2) -> None:
        self._parts.append(f"{'#' * level} {text}")

    def text(self, text: str = "") -> None:
        self._parts.append(text)

    def table(self, text: str) -> None:
        """A preformatted, fixed-width table or block, wrapped in a code fence."""
        self._parts.append(f"```\n{text}\n```")

    def image(self, path: Path, caption: str = "") -> None:
        """Embed a figure, linked relative to wherever write() ends up putting it.

        `path` is stored absolute and resolved to a relative link in write(),
        rather than assumed to be one fixed number of directories below the
        report — analyze_continuous.py's figures sit directly under
        plots/, but analyze_baseline_skill.py's sit one level deeper, under
        plots/with_baseline/, so a hardcoded "plots/{name}" link would be wrong
        for the second caller.
        """
        alt = caption or path.stem
        self._parts.append(f"![{alt}]({path.resolve()})")
        if caption:
            self._parts.append(f"*{caption}*")

    def render(self, base_dir: Path) -> str:
        """Join the accumulated parts, rewriting image links relative to base_dir."""
        text = "\n\n".join(self._parts) + "\n"
        return re.sub(
            r"!\[([^\]]*)\]\(([^)]+)\)",
            lambda m: f"![{m.group(1)}]({os.path.relpath(m.group(2), base_dir)})",
            text,
        )

    def write(self, path: Path, title: str) -> Path:
        """Write the accumulated report to `path`, with a title and commit notes.

        Both repos are named: this one produced the tables, and the engine
        checkout produced the trajectories they score against, so neither alone
        identifies what a report came from.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# {title}\n\n"
            f"generated at forecastbench-sim commit {git_commit_note()}\n"
            f"Micropolis engine commit {engine_commit_note()}"
        )
        path.write_text(header + "\n\n" + self.render(path.parent))
        return path


class DatasetError(Exception):
    """The dataset on disk doesn't cover what the config asked for."""


def scenario_ids_from(
    cfg: Config,
    seed: int,
    cities: list[str] | None = None,
    disasters: list[bool] | None = None,
) -> list[str]:
    """The scenario ids a config's cities x disasters cross product names.

    Built through CitySimulation so the ids match the ones build_corpus wrote;
    constructing one runs nothing, it only holds the parameters. `cities` and
    `disasters` override the config's lists, for the matching command-line flags.
    """
    return [
        CitySimulation(city_name=city, seed=seed, disasters=dis).get_id_str()
        for city, dis in scenarios_from(cfg, cities, disasters)
    ]


def select_for_config(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    cfg: Config,
    seed: int,
    *,
    cities: list[str] | None = None,
    disasters: list[bool] | None = None,
    models: list[str] | None = None,
    rerun_hint: str = "scripts/run_eval_continuous.py",
    incomplete: bool = False,
) -> tuple[list[dict], Responses, list[str]]:
    """Narrow a dataset to what `cfg` asks for, or fail saying what is missing.

    Lets one gathered dataset serve many views — a subset of models, cities or
    horizons — without re-prompting. Anything the config names that the dataset
    lacks is an error rather than a silently smaller table, since a missing
    model or city would otherwise look like a legitimately empty result.
    `cities`, `disasters` and `models` override the config's lists, for the
    matching command-line flags; keyword-only, since three same-shaped list
    arguments in a row are easy to pass in the wrong order. `rerun_hint` names
    the gathering script in the error messages — the binary eval reuses this
    with its own script.

    `incomplete` downgrades the never-gathered rows from an error to a warning
    and keeps every (question, model) pair that *was* gathered, for testing the
    analysis scripts against a run still missing batches. The selection is then
    ragged: each model is scored on the questions it answered, so per-model
    aggregates cover different question sets and are not strictly comparable
    with one another. The warning says so and lists each model's coverage. This
    is a testing aid, not a reporting mode.

    It does not relax the coverage check above: a model, city or horizon the
    dataset knows nothing about is a config/dataset mismatch rather than sparse
    data. A selected model with no gathered forecast at all is likewise an
    error, since every figure it appears in would be empty.
    """
    wanted_models = cfg.get_models(models)
    wanted_scenarios = scenario_ids_from(cfg, seed, cities, disasters)
    wanted_snapshots = cfg.get_int_list("snapshot_turns")
    wanted_horizons = cfg.get_int_list("horizons")

    missing = []
    for name, wanted, present in [
        ("models", wanted_models, set(model_names)),
        ("cities/disasters", wanted_scenarios, {c["scenario_id"] for c in corpus}),
        ("snapshot_turns", wanted_snapshots, {c["snapshot_turn"] for c in corpus}),
        ("horizons", wanted_horizons, {c["horizon"] for c in corpus}),
    ]:
        absent = [w for w in wanted if w not in present]
        if absent:
            missing.append(
                f"  {name}: {', '.join(str(a) for a in absent)}\n"
                f"    dataset has: {', '.join(str(p) for p in sorted(present, key=str))}"
            )
    if missing:
        raise DatasetError(
            "the dataset does not cover this config:\n"
            + "\n".join(missing)
            + f"\n  re-run {rerun_hint} with this config to gather it"
        )

    selected_scenarios = set(wanted_scenarios)
    selected_snapshots = set(wanted_snapshots)
    selected_horizons = set(wanted_horizons)
    selected_corpus = [
        c
        for c in corpus
        if c["scenario_id"] in selected_scenarios
        and c["snapshot_turn"] in selected_snapshots
        and c["horizon"] in selected_horizons
    ]

    # Every selected question needs a row for every selected model. A model that
    # answered unusably still has a row, with null percentiles, so a genuinely
    # absent row means that pair was never gathered.
    ungathered = [
        (model_id, c["question_id"])
        for c in selected_corpus
        for model_id in wanted_models
        if ResponseId(model_id, c["question_id"]) not in responses
    ]
    if ungathered and not incomplete:
        shown = ", ".join(f"{m} / {q}" for m, q in ungathered[:3])
        more = f" (+{len(ungathered) - 3} more)" if len(ungathered) > 3 else ""
        raise DatasetError(
            f"the dataset is missing {len(ungathered)} forecast(s) the config asks "
            f"for: {shown}{more}\n"
            f"  re-run {rerun_hint} with this config to gather them\n"
            f"  or pass --incomplete to score only the fully gathered questions"
        )
    if ungathered:
        # Every pair that was gathered is kept, so the selection is ragged:
        # each model is scored on the questions it actually answered. That is
        # what makes this a testing aid and not a reporting mode — per-model
        # aggregates are then means over different question sets, so the
        # columns are not strictly comparable with each other. The per-model
        # counts below are printed for exactly that reason.
        by_model = Counter(m for m, _ in ungathered)
        nq = len(selected_corpus)
        msg.warn(
            f"--incomplete: {len(ungathered)} of {nq * len(wanted_models)} "
            f"(question, model) pair(s) were never gathered; scoring the rest. "
            "Per-model figures cover different question sets and are not "
            "directly comparable."
        )
        for model_id in wanted_models:
            n = by_model.get(model_id, 0)
            if n:
                msg.plain(
                    f"  {model_id}: {nq - n} of {nq} question(s)", color=msg.YELLOW
                )
        msg.plain(
            f"  re-run {rerun_hint} with this config to gather them",
            color=msg.YELLOW,
        )
        starved = [m for m in wanted_models if by_model.get(m, 0) >= nq]
        if starved:
            raise DatasetError(
                "no forecast at all was gathered for: " + ", ".join(starved) + "\n"
                f"  re-run {rerun_hint} with this config, or drop them from --models"
            )

    # Absent pairs are skipped rather than indexed, so an --incomplete
    # selection can be ragged; without the flag the loop above has already
    # proved every pair is present.
    selected_responses = {
        rid: responses[rid]
        for c in selected_corpus
        for model_id in wanted_models
        if (rid := ResponseId(model_id, c["question_id"])) in responses
    }
    return selected_corpus, selected_responses, wanted_models


def score_forecasts(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    norm: Normalizer,
) -> list[dict]:
    """Score every parsed forecast, raw and normalized.

    One row per (model, question) that produced a usable forecast, carrying the
    metric and horizon so callers can group as they like. "normalized" is CRPS
    over `norm`'s scale for that question, and is None where the question has
    no scale, so a caller averaging it must skip the Nones.
    """
    rows = []
    for c in corpus:
        for model_id in model_names:
            r = responses.get(ResponseId(model_id, c["question_id"]))
            if r is None or r.percentiles is None:
                continue
            crps = compute_crps(r.percentiles, c["value"])
            # A zero scale is treated as no scale: a mode reading the
            # denominator off the question can land on 0 for a metric that
            # happens to sit there, and dividing by it would raise several
            # frames from the cause.
            scale = norm.scale(c)
            rows.append(
                {
                    "model_id": model_id,
                    "metric": c["metric"],
                    "horizon": c["horizon"],
                    "crps": crps,
                    "normalized": crps / scale if scale else None,
                }
            )
    return rows
