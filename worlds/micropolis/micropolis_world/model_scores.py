"""External benchmark scores for the models this world evaluates.

Reads worlds/micropolis/model_scores.csv, which is the single source for every
score this repo did not measure itself:

  ECI         Epoch's capability index, a general-capability number the
              analyses correlate their own skill scores against.
  FBOverall   The model's overall ForecastBench score, with a 95% confidence
              interval. Real-world forecasting ability, against which this
              world's simulated forecasting is the thing being validated.

The file also carries the model's name as OpenRouter and ForecastBench spell it,
which nothing here reads — they are there so a row can be traced back to the
leaderboard it was copied from.

Models are joined on LiteLLMSlug, the id the eval configs and the response cache
use, so a row without one takes part in no analysis. That is deliberate rather
than an oversight to route around: a near-miss slug ("gemini-3.1-flash-lite" for
a row recording "gemini-3.1-flash-lite-preview") is a different model checkpoint
with a different score, and guessing that the two are the same would silently
attribute one's benchmark number to the other's forecasts. Filling in the blank
LiteLLMSlug in the CSV is the way to bring such a model in.

A missing cell is a missing score, not a zero: ECI, FBOverall and the interval
are each None when blank, and callers drop the model from that particular
figure rather than plotting a hole at the origin.
"""

import csv
from dataclasses import dataclass
from functools import cache
from pathlib import Path

# worlds/micropolis/model_scores.csv, beside the package rather than inside it:
# it is data to be edited by hand as new leaderboard numbers land, not code.
SCORES_PATH = Path(__file__).resolve().parent.parent / "model_scores.csv"


@dataclass(frozen=True)
class ModelScores:
    """One model's externally-measured scores, keyed by its LiteLLM slug."""

    # The bare model name, without the provider prefix — "gpt-5.5", not
    # "openai/gpt-5.5". What the analyses key on once they have stripped the
    # prefix, and what the knowledge eval's cache reduces to.
    name: str
    eci: float | None
    # ForecastBench overall, and the bounds of its 95% interval. The interval is
    # what makes the comparison honest: several of these models are within a
    # point of each other, which is inside their own error bars.
    fb_overall: float | None
    fb_ci_lo: float | None
    fb_ci_hi: float | None

    @property
    def fb_error(self) -> tuple[float, float] | None:
        """The interval as (below, above) distances, the shape errorbar wants.

        None unless the score and both bounds are present, since half an
        interval is not one that can be drawn.
        """
        if self.fb_overall is None or self.fb_ci_lo is None or self.fb_ci_hi is None:
            return None
        return (self.fb_overall - self.fb_ci_lo, self.fb_ci_hi - self.fb_overall)


def _number(value: str | None) -> float | None:
    """A CSV cell as a float, or None when it is blank.

    Blank means "not published for this model", which every caller has to handle
    anyway; raising here would make one missing leaderboard entry break every
    analysis rather than just its own point.
    """
    text = (value or "").strip()
    return float(text) if text else None


@cache
def load_scores() -> dict[str, ModelScores]:
    """Every scored model, keyed on its bare name.

    Keyed on the name with the provider prefix stripped — "gpt-5.5" rather than
    "openai/gpt-5.5" — because that is the form the analyses and the knowledge
    eval's response cache both reduce their model ids to.

    Cached: the file is small, but it is read from inside per-model helpers
    called once per point on a figure.
    """
    out: dict[str, ModelScores] = {}
    with SCORES_PATH.open(newline="") as f:
        for row in csv.DictReader(f):
            slug = (row.get("LiteLLMSlug") or "").strip()
            if not slug:
                continue
            name = slug.split("/", 1)[-1]
            out[name] = ModelScores(
                name=name,
                eci=_number(row.get("ECI")),
                fb_overall=_number(row.get("FBOverall")),
                fb_ci_lo=_number(row.get("FBOverallCILo")),
                fb_ci_hi=_number(row.get("FBOverallCIHi")),
            )
    return out


def scores_of(model_id: str) -> ModelScores | None:
    """Every score for a provider/name model id, or None if the CSV has no row."""
    return load_scores().get(model_id.split("/", 1)[-1])


def eci_of(model_id: str) -> float | None:
    """ECI for a provider/name model id, or None if it has no published score."""
    scores = scores_of(model_id)
    return scores.eci if scores else None


def fb_overall_of(model_id: str) -> float | None:
    """ForecastBench overall for a provider/name id, or None if it has none."""
    scores = scores_of(model_id)
    return scores.fb_overall if scores else None


def eci_by_name(model_ids: list[str]) -> dict[str, float]:
    """ECI per model, keyed on the bare name, skipping those without one."""
    return {m.split("/", 1)[-1]: eci_of(m) for m in model_ids if eci_of(m) is not None}


def fb_by_name(model_ids: list[str]) -> dict[str, float]:
    """ForecastBench overall per model, keyed on bare name, skipping the unscored."""
    return {
        m.split("/", 1)[-1]: fb_overall_of(m)
        for m in model_ids
        if fb_overall_of(m) is not None
    }


def format_eci(eci: float | None, missing: str = "nan") -> str:
    """An ECI for a table cell, at a fixed two decimals.

    Fixed width rather than %g: the published scores mix whole numbers with two
    decimals ("150" beside "137.52"), and a column that alternated between the
    two reads as if the round ones were measured less precisely than they were.
    """
    return missing if eci is None else f"{eci:.2f}"


def write_scores_csv(
    path: Path,
    mp_stats: dict[str, tuple[float, float | None, float | None]],
) -> Path:
    """Write model_scores.csv with MPScore/MPScoreLo/MPScoreHi columns appended.

    `mp_stats` maps a provider/name model id to (score, CI low, CI high) from
    one of this world's own analyses. Bounds of None — an interval too few
    clusters could support — land as blank cells, matching how the source file
    spells "not available" everywhere else.

    Every row and column of the source file is kept verbatim and in order: the
    models the run never scored keep blank MPScore cells rather than being
    dropped, so the artifact stays a strict superset of its source and a later
    reader can join it back on any of the original columns. A scored model the
    source file does not list is appended at the end with only its slug filled
    in. The join is on the bare model name, the same rule scores_of applies in
    the other direction.
    """
    stats_by_name = {m.split("/", 1)[-1]: v for m, v in mp_stats.items()}

    # Read raw rather than through load_scores(): the artifact must preserve
    # the rows without a slug and the name columns that the parsed view drops.
    with SCORES_PATH.open(newline="") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        rows = list(reader)

    def score_cells(stats: tuple) -> dict[str, str]:
        score, lo, hi = stats
        return {
            "MPScore": f"{score:.4f}",
            "MPScoreLo": "" if lo is None else f"{lo:.4f}",
            "MPScoreHi": "" if hi is None else f"{hi:.4f}",
        }

    joined = set()
    for row in rows:
        slug = (row.get("LiteLLMSlug") or "").strip()
        name = slug.split("/", 1)[-1] if slug else ""
        stats = stats_by_name.get(name)
        if stats is not None:
            row.update(score_cells(stats))
            joined.add(name)
    for model_id in sorted(mp_stats):
        name = model_id.split("/", 1)[-1]
        if name not in joined:
            rows.append({"LiteLLMSlug": model_id} | score_cells(stats_by_name[name]))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=columns + ["MPScore", "MPScoreLo", "MPScoreHi"],
            restval="",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path
