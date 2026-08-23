#!/usr/bin/env python3
"""Derive H2-H5 questions deterministically from a world's H1 bank.

We deliberately do NOT generate fresh H2-H5 questions with the unseeded
QuestionGenerator (its target sampling is nondeterministic run-to-run).
Instead every H1 question is re-emitted at resolution turns 120 (H2), 150
(H3), 180 (H4), and 210 (H5): same template, target, and subject civs (the
production fleet rolls forks out to t210, so all five horizons resolve). The
qid scheme keeps the horizon family linked with the H1 qid as the family key:

    q0007 (H1, t90)  ->  q0007_h2 (H2, t120),  ...,  q0007_h5 (H5, t210)

"turn {rt}" occurrences in the question text and the parameters'
resolution_turn are rewritten; the base game's H1 "resolution" field is
dropped from derived questions (their truth comes from rollouts). Output is
deterministic and idempotent: derived questions are always regenerated from
the H1 originals (never re-derived from _h2.._h5), so running the script on
its own output changes nothing — including output produced by the earlier
H2/H3-only revision of this script. Non-H1 questions that are not our
derivations pass through untouched.

Usage:
  uv run python scripts/uplift_v2/derive_horizon_bank.py \
      --bank data/questions_mc/seed0/questions.json \
      --out tmp/natcond_v2/fleet_truth/seed0/questions.json
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

DERIVED_HORIZONS = {"H2": 120, "H3": 150, "H4": 180, "H5": 210}
_SUFFIX_RE = re.compile(
    r"^(?P<base>.+)_h[%s]$"
    % "".join(sorted(h[1:].lower() for h in DERIVED_HORIZONS)))


def family_qid(qid: str) -> str:
    """The horizon family key: the H1 qid (q0007_h2 -> q0007)."""
    m = _SUFFIX_RE.match(qid)
    return m.group("base") if m else qid


def derive_question(q: dict, horizon: str, rt: int) -> dict:
    """One H1 question re-emitted at a later resolution turn."""
    base_rt = q["resolution_turn"]
    d = copy.deepcopy(q)
    d["question_id"] = f"{q['question_id']}_h{horizon[1:].lower()}"
    d["horizon"] = horizon
    d["resolution_turn"] = rt
    d["question_text"] = q["question_text"].replace(f"turn {base_rt}", f"turn {rt}")
    if d.get("parameters", {}).get("resolution_turn") == base_rt:
        d["parameters"]["resolution_turn"] = rt
    d.pop("resolution", None)  # base game's H1 answer; rollouts provide truth
    return d


def derive_bank(bank: dict, horizons: dict[str, int] = DERIVED_HORIZONS) -> dict:
    """H1 bank dict -> H1+derived bank dict (deterministic, idempotent)."""
    h1 = [q for q in bank["questions"] if q.get("horizon") == "H1"]
    h1_ids = {q["question_id"] for q in h1}
    kept = [q for q in bank["questions"]
            if family_qid(q["question_id"]) == q["question_id"]
            or family_qid(q["question_id"]) not in h1_ids]  # not our derivation
    derived = [derive_question(q, hz, rt)
               for hz, rt in sorted(horizons.items())
               for q in h1]
    out = copy.deepcopy(bank)
    out["questions"] = copy.deepcopy(kept) + derived
    out["derived_horizons"] = dict(sorted(horizons.items()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    bank = json.loads(Path(args.bank).read_text())
    out = derive_bank(bank)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    from collections import Counter
    hz = Counter(q.get("horizon") for q in out["questions"])
    print(f"{args.bank} -> {args.out}: "
          + ", ".join(f"{h}={n}" for h, n in sorted(hz.items())))


if __name__ == "__main__":
    main()
