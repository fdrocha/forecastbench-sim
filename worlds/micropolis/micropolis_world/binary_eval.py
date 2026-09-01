"""Output paths + dataset IO for the binary yes/no eval.

The binary counterpart of single_city.py's cache/path/dataset half, rooted at
data/micropolis/binary/ instead of single_city/. Batching, hashing and usage
sidecars are reused from single_city/usage unchanged.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from . import module_globals as g
from .single_city import ResponseId

OUT_DIR = g.DATA_DIR / "binary"


def label_dir(label: str) -> Path:
    return OUT_DIR / label


def data_path(label: str) -> Path:
    return label_dir(label) / "data.json"


def batch_dir(batch_id: str) -> Path:
    """Where a batch's prompt and raw model responses are cached.

    Same layout and content-addressing as single_city.batch_dir (shared across
    labels, one file per prompt hash), under the binary root.
    """
    return OUT_DIR / "cache" / batch_id


def prompt_path(batch_id: str, phash: str) -> Path:
    return batch_dir(batch_id) / f"prompt-{phash}.txt"


def response_path(batch_id: str, model_id: str, phash: str) -> Path:
    # Model ids are provider/name; the slash would nest a directory.
    return batch_dir(batch_id) / f"response-{model_id.replace('/', '_')}-{phash}.txt"


def usage_path(batch_id: str, model_id: str, phash: str) -> Path:
    return batch_dir(batch_id) / f"usage-{model_id.replace('/', '_')}-{phash}.json"


@dataclass(frozen=True)
class BinaryResponse:
    actual: bool
    probability: float | None
    response_text: str | None = None


BinaryResponses = dict[ResponseId, BinaryResponse]


def save_dataset_binary(
    corpus: list[dict],
    responses: BinaryResponses,
    model_names: list[str],
    path: Path,
) -> Path:
    """Write the corpus and this run's forecasts as one self-contained file.

    Mirrors single_city.save_dataset: questions keep their resolved bool
    "answer", forecasts carry each model's P(Yes) (null = answered unusably;
    an absent row = never gathered), and the report ("context") stays in the
    prompt cache rather than being repeated per question here.
    """
    dropped = {"context"}
    questions = [{k: v for k, v in c.items() if k not in dropped} for c in corpus]

    forecasts = [
        {
            "model_id": model_id,
            "question_id": c["question_id"],
            "probability": r.probability,
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


def load_dataset_binary(path: Path) -> tuple[list[dict], BinaryResponses, list[str]]:
    """Read back what save_dataset_binary wrote, as (corpus, responses, models)."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run scripts/run_eval_binary.py first"
        )
    data = json.loads(path.read_text())
    corpus = data["questions"]
    actual = {c["question_id"]: c["answer"] for c in corpus}
    responses: BinaryResponses = {
        ResponseId(f["model_id"], f["question_id"]): BinaryResponse(
            actual=actual[f["question_id"]],
            probability=f["probability"],
        )
        for f in data["forecasts"]
    }
    return corpus, responses, data["models"]
