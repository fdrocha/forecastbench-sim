#!/usr/bin/env python3
"""Post-hoc H2-H5 ground truth for the mined rollout fleets (one pre-step).

The pod manifests only resolve the H1 banks at t90 (resolution_turns
[90, 150] against H1-only questions), so t120/t150/t180/t210 truth is
produced here, post hoc, per world under --fleet-dir:

  1. derive the H2-H5 bank deterministically from the world's H1 bank
     (derive_horizon_bank.py — the unseeded QuestionGenerator is never run);
  2. re-resolve the derived questions against every serialized rollout
     (QuestionResolver over the rollout's time_series/events — the
     re-resolution path that reproduced the archived seed2 fleet's p_y
     54/54);
  3. write a self-contained arm at <workdir>/<world>/: questions.json
     (H1+derived), manifest.json (pod H1 answers merged across lane shards +
     the newly resolved derived answers), and laneNN symlinks so the rollout
     events stay reachable for event mining.

--horizons restricts the derived set (default: all of H2,H3,H4,H5). Legacy
fleets mined only to t150 need --horizons H2,H3; on the production fleet
(rollouts to t210) all five horizons resolve. A derived question at a turn
past a fleet's last serialized turn resolves to null everywhere and is
dropped by the survival guard below, so the default is safe either way.

After this pre-step the standard cells miner consumes the workdir
transparently (natcond_cells_v2 auto-uses <world>/questions.json):

  uv run python scripts/uplift_v2/mine_natcond_truth.py \
      --fleet-dir tmp/mined/rollouts --workdir tmp/natcond_v2/fleet_truth
  uv run python scripts/uplift_v2/natcond_cells_v2.py \
      --fleet-dir tmp/natcond_v2/fleet_truth --out tmp/natcond_v2/cells.json

Guard: a derived question whose target/metric is unreadable from the rollout
serializations at the later turn resolves to null everywhere; such questions
are warned about, dropped from the world's manifest, and reported (survival
counts per horizon). Resume: rollout tags already resolved in the workdir
manifest are skipped on re-run, so this can be re-invoked as the fleet syncs.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from derive_horizon_bank import DERIVED_HORIZONS, derive_bank  # noqa: E402
import natcond_cells_v2 as cells_v2  # noqa: E402


def parse_horizons(spec: str) -> dict[str, int]:
    """--horizons spec ("H2,H3,H4,H5") -> {horizon: resolution turn}.

    H1 entries are ignored (H1 truth comes from the pod manifests, never
    derived); unknown horizons are an error."""
    out = {}
    for hz in spec.split(","):
        hz = hz.strip()
        if not hz or hz == "H1":
            continue
        if hz not in DERIVED_HORIZONS:
            raise ValueError(f"unknown derived horizon {hz!r}; "
                             f"derivable: {sorted(DERIVED_HORIZONS)}")
        out[hz] = DERIVED_HORIZONS[hz]
    return out


def build_resolver():
    """QuestionResolver + bank parser; lazy so unit tests import this module
    without the world package. Supports both package names (main branch:
    freeciv_world; whatif-exploration: civrealm)."""
    try:
        from freeciv_world.world_reports.questions import (
            QuestionResolver, REGISTRY, dict_to_question_bank)
        return QuestionResolver(REGISTRY), dict_to_question_bank
    except ImportError:
        from civrealm.world_reports.questions import (  # type: ignore
            QuestionResolver, dict_to_question_bank)
        return QuestionResolver(), dict_to_question_bank


def rollout_files(world_dir: str) -> list[str]:
    """Serialized rollouts for one fleet world (lane-sharded or plain arm)."""
    return (sorted(glob.glob(f"{world_dir}/rollouts/*.json.gz"))
            or sorted(glob.glob(f"{world_dir}/lane*/rollouts/*.json.gz")))


def dead_derived_qids(answers: dict, derived_qids: set) -> list[str]:
    """Derived questions with no non-null answer anywhere: their target/metric
    is unreadable from the rollout serializations at the later turn. Reported
    and warned about here; downstream the cells miner skips them naturally
    (all-null answer lists yield no scorable rollouts). Records are kept in
    the manifest so resume stays consistent."""
    return [qid for qid in sorted(derived_qids)
            if qid in answers
            and all(r["answer"] is None for r in answers[qid])]


def process_world(world_dir: str, game_id: str, bank_path: str,
                  workdir: Path, smoke: bool,
                  horizons: dict[str, int] = DERIVED_HORIZONS) -> dict:
    out_dir = workdir / game_id
    out_dir.mkdir(parents=True, exist_ok=True)

    bank = json.loads(Path(bank_path).read_text())
    derived_bank = derive_bank(bank, horizons)
    (out_dir / "questions.json").write_text(json.dumps(derived_bank, indent=1))

    resolver, to_bank = build_resolver()
    parsed = to_bank(derived_bank)
    derived_qs = [q for q in parsed.questions
                  if q.resolution_turn in horizons.values()]
    derived_qids = {q.question_id for q in derived_qs}

    # resume: keep previously resolved derived answers
    manifest_path = out_dir / "manifest.json"
    answers: dict[str, list] = {}
    done_tags: set[str] = set()
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text())
        answers = {qid: recs for qid, recs in prior.get("answers", {}).items()
                   if qid in derived_qids}
        done_tags = set(prior.get("config", {}).get("resolved_tags", []))

    fps = rollout_files(world_dir)
    todo = [(Path(fp).name.split(".")[0], fp) for fp in fps
            if Path(fp).name.split(".")[0] not in done_tags]
    todo.sort(key=lambda tf: cells_v2._tag_sort_key(tf[0]))
    t0 = time.time()
    n_fail = 0
    for tag, fp in todo:
        try:
            gd = json.load(gzip.open(fp, "rt"))
        except Exception:
            print(f"  !! {game_id}: unreadable rollout {tag}, skipped")
            n_fail += 1
            continue
        for q in derived_qs:
            try:
                ans = bool(resolver.resolve(q, gd, parsed.snapshot_turn).answer)
            except Exception:
                ans = None
            answers.setdefault(q.question_id, []).append(
                {"tag": tag, "answer": ans})
        done_tags.add(tag)

    # pod H1 answers, merged across lane shards (refreshed on every run)
    pod = cells_v2.load_answers(world_dir)
    for qid, amap in pod.items():
        answers[qid] = [{"tag": t, "answer": v} for t, v in
                        sorted(amap.items(), key=lambda kv: cells_v2._tag_sort_key(kv[0]))]

    dead = dead_derived_qids(answers, derived_qids)
    for qid in dead:
        print(f"  !! {game_id}: derived {qid} never resolves in "
              f"{len(done_tags)} rollouts — metric gap, will be skipped")

    manifest_path.write_text(json.dumps({
        "config": {
            "game_id": game_id, "snapshot_turn": parsed.snapshot_turn,
            "resolution_turns": sorted({90, *horizons.values()}),
            "questions": str(out_dir / "questions.json"),
            "source_fleet_world": str(world_dir),
            "source_bank": str(bank_path),
            "resolved_tags": sorted(done_tags, key=cells_v2._tag_sort_key),
            "derived_truth": True, "smoke": smoke,
        },
        "answers": answers}, indent=1))

    # lane symlinks so natcond_cells_v2 finds the rollout events
    for lane in sorted(Path(world_dir).glob("lane*")):
        link = out_dir / lane.name
        if not link.exists():
            link.symlink_to(lane.resolve())
    if (Path(world_dir) / "rollouts").exists() and not (out_dir / "rollouts").exists():
        (out_dir / "rollouts").symlink_to((Path(world_dir) / "rollouts").resolve())

    # survival accounting per horizon
    by_hz = {}
    dead_set = set(dead)
    n_h1 = sum(1 for q in bank["questions"] if q.get("horizon") == "H1")
    for hz in sorted(horizons):
        qids = {q.question_id for q in derived_qs
                if q.resolution_turn == horizons[hz]}
        alive = [q for q in sorted(qids)
                 if q in answers and q not in dead_set]
        partial = sum(1 for q in alive
                      if any(r["answer"] is None for r in answers[q]))
        by_hz[hz] = {"derived": len(qids), "survived": len(alive),
                     "dropped": len(qids) - len(alive),
                     "partial_null": partial}
    stats = {"game_id": game_id, "h1": n_h1, "rollouts_resolved": len(done_tags),
             "unreadable": n_fail, "horizons": by_hz,
             "dropped_qids": dead, "sec": round(time.time() - t0, 1)}
    print(f"  {game_id}: {len(done_tags)} rollouts ({len(todo)} new), "
          + ", ".join(f"{hz} {v['survived']}/{v['derived']}"
                      for hz, v in by_hz.items())
          + f" survived ({stats['sec']}s)")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet-dir", required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--questions-pattern",
                    default="data/questions_mc/{game_id}/questions.json",
                    help="source H1 bank per world")
    ap.add_argument("--horizons", default=",".join(sorted(DERIVED_HORIZONS)),
                    help="derived horizons to emit/resolve (subset of "
                         f"{','.join(sorted(DERIVED_HORIZONS))}; legacy t150 "
                         "fleets want H2,H3)")
    ap.add_argument("--smoke", action="store_true",
                    help="label outputs as a smoke run over partial fleet data")
    args = ap.parse_args()
    try:
        horizons = parse_horizons(args.horizons)
    except ValueError as e:
        ap.error(str(e))
    if not horizons:
        ap.error("--horizons selects no derivable horizon")
    if args.smoke:
        print("== SMOKE RUN: partial fleet data; truth is provisional ==")

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    report = []
    for d in sorted(Path(args.fleet_dir).iterdir()):
        if not (d.is_dir() and cells_v2.manifest_paths(str(d))):
            continue
        cfg = json.loads(cells_v2.manifest_paths(str(d))[0].read_text()).get("config", {})
        game_id = cfg.get("game_id") or d.name
        bank_path = args.questions_pattern.format(game_id=game_id)
        if not Path(bank_path).exists():
            print(f"  !! {game_id}: no source bank at {bank_path}, skipped")
            continue
        report.append(process_world(str(d), game_id, bank_path, workdir,
                                    args.smoke, horizons))

    (workdir / "truth_report.json").write_text(json.dumps(
        {"smoke": args.smoke, "derived_horizons": horizons,
         "worlds": report}, indent=1))
    total = Counter()
    for w in report:
        for hz, v in w["horizons"].items():
            total[f"{hz}_derived"] += v["derived"]
            total[f"{hz}_survived"] += v["survived"]
    print(f"\n{len(report)} worlds -> {workdir}/truth_report.json; "
          + ", ".join(f"{hz} {total[f'{hz}_survived']}/{total[f'{hz}_derived']}"
                      for hz in sorted(horizons)))


if __name__ == "__main__":
    main()
