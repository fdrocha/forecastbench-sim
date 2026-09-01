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
from dataclasses import dataclass
from pathlib import Path

from fbsim_core.metrics import compute_crps

from . import module_globals as g
from .city_sim import CitySimulation
from .config import Config, scenarios_from
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


# Metrics left out of normalized CRPS. Normalizing by |actual| is undefined
# where the actual is 0, and city funds legitimately sits at 0 for long
# stretches — a bankrupt city stays broke — so the whole metric is excluded
# rather than dropping the individual questions and averaging over a
# silently different question set per scenario.
UNNORMALIZED_METRICS = {"totalFunds"}


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


def batch_id_for(question: dict) -> str:
    """The batch a corpus question is prompted in.

    Questions sharing a scenario and snapshot turn share a game report — the
    bulk of the prompt — so they are asked together in one numbered prompt.
    """
    return f"{question['scenario_id']}_T{question['snapshot_turn']}"


def _split_evenly(questions: list[dict], limit: int) -> list[list[dict]]:
    """`questions` cut into consecutive chunks of at most `limit` each.

    The chunk count is what `limit` really fixes: enough chunks that none
    exceeds it. Their sizes are then evened out rather than filling each chunk
    before starting the next, so 20 questions at a limit of 12 come out 10 and
    10 instead of 12 and 8 — no prompt is left with a lopsided tail asking
    about only a question or two.
    """
    nchunks = -(-len(questions) // limit)  # ceiling division
    base, extra = divmod(len(questions), nchunks)
    chunks = []
    start = 0
    for i in range(nchunks):
        # The first `extra` chunks take one more, so sizes differ by at most 1.
        stop = start + base + (1 if i < extra else 0)
        chunks.append(questions[start:stop])
        start = stop
    return chunks


def group_into_batches(
    corpus: list[dict], questions_per_prompt: int = -1
) -> dict[str, list[dict]]:
    """Group corpus questions by batch_id, preserving corpus order.

    `questions_per_prompt` caps how many questions one prompt may ask. The
    default -1 means no cap: a scenario's whole snapshot goes in one prompt,
    which is the cheapest way to ask them since they share a game report. A
    positive value splits a batch that would exceed it into consecutive chunks
    (see _split_evenly), each becoming its own batch — its own prompt repeating
    the report, its own cache directory, and its own response — so a split
    batch stays one hash per batch id for analyze_usage, and a chunk that fails
    is retried on its own. Chunk ids are suffixed "_c{i}of{n}"; an unsplit
    batch keeps the plain id, so existing caches stay addressable.
    """
    batches: dict[str, list[dict]] = {}
    for c in corpus:
        batches.setdefault(batch_id_for(c), []).append(c)
    if questions_per_prompt < 0:
        return batches

    if questions_per_prompt == 0:
        raise ValueError("questions_per_prompt must be -1 or a positive integer")

    split: dict[str, list[dict]] = {}
    for bid, questions in batches.items():
        if len(questions) <= questions_per_prompt:
            split[bid] = questions
            continue
        chunks = _split_evenly(questions, questions_per_prompt)
        for i, chunk in enumerate(chunks, 1):
            split[f"{bid}_c{i}of{len(chunks)}"] = chunk
    return split


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
    return OUT_DIR / "cache" / batch_id


def prompt_path(batch_id: str, phash: str) -> Path:
    return batch_dir(batch_id) / f"prompt-{phash}.txt"


def response_path(batch_id: str, model_id: str, phash: str) -> Path:
    # Model ids are provider/name; the slash would nest a directory.
    return batch_dir(batch_id) / f"response-{model_id.replace('/', '_')}-{phash}.txt"


def usage_path(batch_id: str, model_id: str, phash: str) -> Path:
    """Tokens and cost of the call that produced the matching response file.

    A cached response is never re-fetched, so what it cost has to be recorded
    when it is first paid or it is lost on every later run. Written beside the
    response and keyed the same way, so the pair stays together — note the slug
    must match response_path's exactly, or the sidecar lands next to nothing.
    """
    return batch_dir(batch_id) / f"usage-{model_id.replace('/', '_')}-{phash}.json"


def save_dataset(
    corpus: list[dict],
    responses: Responses,
    model_names: list[str],
    path: Path,
) -> Path:
    """Write the corpus and this run's forecasts as one self-contained file.

    The analysis and plotting scripts read only this, so it carries everything
    they need: the questions with their resolved values, and each model's
    percentiles per question. The full response text is deliberately left out —
    it stays in the response cache, and including it here would multiply the
    file size for something no downstream script reads.
    """
    # Prompts are megabytes of world report repeated per question; the
    # downstream scripts want the question, not the prompt that produced it.
    dropped = {"context"}
    questions = [{k: v for k, v in c.items() if k not in dropped} for c in corpus]

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

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"models": model_names, "questions": questions, "forecasts": forecasts},
            indent=2,
        )
    )
    return path


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
) -> tuple[list[dict], Responses, list[str]]:
    """Narrow a dataset to what `cfg` asks for, or fail saying what is missing.

    Lets one gathered dataset serve many views — a subset of models, cities or
    horizons — without re-prompting. Anything the config names that the dataset
    lacks is an error rather than a silently smaller table, since a missing
    model or city would otherwise look like a legitimately empty result.
    `cities`, `disasters` and `models` override the config's lists, for the
    matching command-line flags; keyword-only, since three same-shaped list
    arguments in a row are easy to pass in the wrong order.
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
            + "\n  re-run scripts/run_eval_continuous.py with this config to gather it"
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
    if ungathered:
        shown = ", ".join(f"{m} / {q}" for m, q in ungathered[:3])
        more = f" (+{len(ungathered) - 3} more)" if len(ungathered) > 3 else ""
        raise DatasetError(
            f"the dataset is missing {len(ungathered)} forecast(s) the config asks "
            f"for: {shown}{more}\n"
            "  re-run scripts/run_eval_continuous.py with this config to gather them"
        )

    selected_responses = {
        ResponseId(model_id, c["question_id"]): responses[
            ResponseId(model_id, c["question_id"])
        ]
        for c in selected_corpus
        for model_id in wanted_models
    }
    return selected_corpus, selected_responses, wanted_models


def score_forecasts(
    corpus: list[dict], responses: Responses, model_names: list[str]
) -> list[dict]:
    """Score every parsed forecast, raw and normalized.

    One row per (model, question) that produced a usable forecast, carrying the
    metric and horizon so callers can group as they like. "normalized" is CRPS
    over |actual|, and is None where that is undefined or the metric is
    excluded, so a caller averaging it must skip the Nones.
    """
    rows = []
    for c in corpus:
        for model_id in model_names:
            r = responses.get(ResponseId(model_id, c["question_id"]))
            if r is None or r.percentiles is None:
                continue
            crps = compute_crps(r.percentiles, c["value"])
            actual = abs(c["value"])
            # Guard on the value rather than trusting the metric to be
            # non-zero: which metrics can hit 0 depends on the cities in the
            # config, and a division by zero here would be silent.
            normalizable = c["metric"] not in UNNORMALIZED_METRICS and actual != 0
            rows.append(
                {
                    "model_id": model_id,
                    "metric": c["metric"],
                    "horizon": c["horizon"],
                    "crps": crps,
                    "normalized": crps / actual if normalizable else None,
                }
            )
    return rows
