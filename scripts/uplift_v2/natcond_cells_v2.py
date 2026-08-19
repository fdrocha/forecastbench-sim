#!/usr/bin/env python3
"""Build split-half natural-conditional benchmark cells from fleet rollouts (v2).

v1 (natcond_cells.py) mined H1-only cells from one archived arm with pooled
ground truth. v2 targets the mined fleets under tmp/mined/rollouts/<world>/
(8 worlds x 1,000 rollouts, t60 -> t150) and adds:

  * horizons H1/H2/H3 (t90/t120/t150), taken from the world's question bank;
  * split-half certification: rollouts are split into two fixed halves by
    sorted rollout index (even index -> half A, odd -> half B; numeric-aware
    sort of the s<seed> tags, deterministic across runs). Conditioning events
    are selected and effect cells certified on half A only
    (|delta_A| > 2*SE_A); half-B p_y / p_yx are emitted as scoring truth;
  * a shrinkage diagnostic per cell, sqrt(max(delta_A^2 - SE_A^2, 0)) — the
    selection-debiased effect-size estimate (0 when the observed delta is
    within one SE of noise);
  * a summary table: cells per world x template-family x horizon x
    effect/placebo.

Cells keep the v1 schema (game_id, qid, question, event_id, event_kind,
event_desc, window, freq, n_x, p_y, p_yx, delta, se_delta, cls — all carrying
the half-A / certification values, so v1 consumers keep working) plus new
fields: horizon, resolution_turn, template_id, half_a{...}, half_b{...},
shrunk_abs_delta.

Arm dir layout (as written by scripts/causal_pilot.py): manifest.json with
answers[qid] = [{"tag": ..., "answer": bool|null}], and rollouts/<tag>.json.gz
serialized game_data with an "events" list. Flat <arm>/<tag>.json.gz files and
bare event-list payloads are also accepted.

Usage (single world):
  uv run python scripts/uplift_v2/natcond_cells_v2.py --game-id w01 \
      --arm-dir tmp/mined/rollouts/w01 --out tmp/natcond_v2/w01_cells.json
Fleet mode (every subdir with a manifest.json; merged cells, one summary):
  uv run python scripts/uplift_v2/natcond_cells_v2.py \
      --fleet-dir tmp/mined/rollouts --out tmp/natcond_v2/fleet_cells.json \
      --questions-pattern "data/questions_mc/{game_id}/questions.json"
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path

DEFAULT_HORIZONS = "H1,H2,H3"
SPECIFIC_EVENT_TYPES = ("tech_discovered", "government_change", "wonder_completed")


def vague_predicates(civs: list[str]):
    """Event-class predicates: (name, human description template, fn(events, a, b))."""
    preds = []
    for civ in civs:
        preds.append((
            f"{civ}_loses_city",
            f"{civ} loses at least one city (conquered or destroyed)",
            lambda evs, a, b, civ=civ: any(
                (e["type"] == "city_conquered" and f"from {civ}" in e["description"])
                or (e["type"] == "city_destroyed" and f"({civ})" in e["description"])
                for e in evs if a < e["turn"] <= b)))
        preds.append((
            f"{civ}_anarchy",
            f"{civ} falls into Anarchy (any government collapse)",
            lambda evs, a, b, civ=civ: any(
                e["type"] == "government_change" and e["description"].startswith(civ)
                and "to Anarchy" in e["description"]
                for e in evs if a < e["turn"] <= b)))
    preds.append((
        "any_wonder",
        "any civilization completes a Wonder of the World",
        lambda evs, a, b: any(e["type"] == "wonder_completed"
                              for e in evs if a < e["turn"] <= b)))
    preds.append((
        "pirate_conquest",
        "Pirates conquer at least one city from any civilization",
        lambda evs, a, b: any(e["type"] == "city_conquered"
                              and "by Pirate" in e["description"]
                              for e in evs if a < e["turn"] <= b)))
    return preds


def _tag_sort_key(tag: str):
    """Numeric-aware sort so s999 < s1000 (lexicographic would flip them)."""
    m = re.fullmatch(r"([A-Za-z_]*)(\d+)", tag)
    return (m.group(1), int(m.group(2))) if m else (tag, -1)


def split_tags(tags) -> tuple[set, set]:
    """Two fixed halves, deterministic by rollout index.

    Sort tags (numeric-aware), then even index -> half A, odd -> half B.
    Interleaving is robust to any monotonic drift across the seed range.
    """
    ordered = sorted(tags, key=_tag_sort_key)
    half_a = {t for i, t in enumerate(ordered) if i % 2 == 0}
    return half_a, set(ordered) - half_a


def half_stats(amap: dict, half_tags: set, members: set) -> dict | None:
    """p_y / p_yx / delta / se_delta over one half's answered rollouts.

    amap: tag -> bool answer for one question (nulls already dropped).
    SE formula matches v1 (binomial, subset-dominated, 0.25/n floor on the
    subset variance).
    """
    tags = [t for t in half_tags if t in amap]
    if not tags:
        return None
    p_y = sum(amap[t] for t in tags) / len(tags)
    sub = [amap[t] for t in tags if t in members]
    if not sub:
        return None
    p_yx = sum(sub) / len(sub)
    se = math.sqrt(max(p_yx * (1 - p_yx), 0.25 / len(sub)) / len(sub)
                   + p_y * (1 - p_y) / len(tags))
    return {
        "n": len(tags), "n_x": len(sub),
        "freq": round(len(members & half_tags) / len(half_tags), 3),
        "p_y": round(p_y, 4), "p_yx": round(p_yx, 4),
        "delta": round(p_yx - p_y, 4), "se_delta": round(se, 4),
    }


def load_rollout_events(arm_dir: str) -> dict[str, list]:
    """tag -> events list. Accepts <arm>/rollouts/*.json.gz or flat *.json.gz,
    and either full game_data dicts (with an "events" key) or bare event lists."""
    fps = (sorted(glob.glob(f"{arm_dir}/rollouts/*.json.gz"))
           or sorted(glob.glob(f"{arm_dir}/*.json.gz")))
    rollouts = {}
    for fp in fps:
        tag = Path(fp).name.split(".")[0]
        try:
            gd = json.load(gzip.open(fp, "rt"))
        except Exception:
            print(f"  !! skipping unreadable rollout {tag}")
            continue
        rollouts[tag] = gd["events"] if isinstance(gd, dict) else gd
    return rollouts


def load_questions(questions_path: str, horizons: set[str]) -> dict[str, dict]:
    """qid -> {question, horizon, resolution_turn, template_id} for kept horizons."""
    qbank = json.loads(Path(questions_path).read_text())
    out = {}
    for q in qbank["questions"]:
        if q.get("horizon") in horizons:
            out[q["question_id"]] = {
                "question": q["question_text"], "horizon": q["horizon"],
                "resolution_turn": q.get("resolution_turn"),
                "template_id": q.get("template_id"),
            }
    return out


def mine_world(game_id: str, arm_dir: str, questions_path: str,
               horizons: set[str], windows: list[tuple[int, int]],
               lo: float, hi: float, max_specific: int, max_vague: int) -> list[dict]:
    """Mine one world's cells. Event selection + certification on half A only;
    half B rides along as held-out scoring truth."""
    manifest = json.loads((Path(arm_dir) / "manifest.json").read_text())
    answers = {qid: {r["tag"]: r["answer"] for r in recs if r["answer"] is not None}
               for qid, recs in manifest["answers"].items()}
    qinfo = load_questions(questions_path, horizons)
    missing_hz = horizons - {qi["horizon"] for qi in qinfo.values()}
    if missing_hz:
        print(f"  !! {game_id}: question bank has no {sorted(missing_hz)} questions")

    rollouts = load_rollout_events(arm_dir)
    civs = set()
    for evs in rollouts.values():
        for e in evs:
            if e["type"] == "tech_discovered" and " discovered " in e["description"]:
                civs.add(e["description"].split(" discovered ")[0])
    civs = sorted(c for c in civs if c not in ("Pirate",))

    tags_a, tags_b = split_tags(rollouts)
    n_a, n_b = len(tags_a), len(tags_b)
    min_nx_a = max(12, int(0.15 * n_a))
    min_nx_b = max(12, int(0.15 * n_b))
    lo = max(lo, min_nx_a / n_a) if n_a else lo  # freq band consistent with n_x floor
    print(f"{game_id}: {len(rollouts)} rollouts (A={n_a}, B={n_b}), "
          f"civs={civs}, min_nx A/B={min_nx_a}/{min_nx_b}, band=[{lo:.2f},{hi}]")

    events = []  # (kind, name, description, window, member_tags)
    # specific events: exact description within window; frequency measured on half A
    for a, b in windows:
        counts = Counter()
        members = {}
        for tag, evs in rollouts.items():
            seen = set()
            for e in evs:
                if a < e["turn"] <= b and e["type"] in SPECIFIC_EVENT_TYPES:
                    d = e["description"]
                    if d not in seen:
                        seen.add(d)
                        members.setdefault(d, []).append(tag)
                        if tag in tags_a:
                            counts[d] += 1
        cands = [(d, c) for d, c in counts.items() if lo <= c / n_a <= hi]
        # spread across event types, prefer mid-band frequency
        cands.sort(key=lambda dc: abs(dc[1] / n_a - 0.4))
        used_types = Counter()
        for d, c in cands:
            ty = ("gov" if "government" in d else
                  "wonder" if "completed" in d else "tech")
            if used_types[ty] >= max(1, max_specific // 3):
                continue
            used_types[ty] += 1
            events.append(("specific", f"sp_{len(events)}", d, (a, b), members[d]))
            if sum(used_types.values()) >= max_specific // len(windows) + 1:
                break
    # vague events (frequency measured on half A)
    for a, b in windows:
        scored = []
        for name, desc, fn in vague_predicates(civs):
            mem = [t for t, evs in rollouts.items() if fn(evs, a, b)]
            f = len([t for t in mem if t in tags_a]) / n_a if n_a else 0.0
            if lo <= f <= hi:
                scored.append((abs(f - 0.4), name, desc, mem))
        scored.sort()
        for _, name, desc, mem in scored[:max_vague // len(windows) + 1]:
            events.append(("vague", f"vg_{len(events)}", desc, (a, b), mem))

    # build cells: half A certifies, half B is held-out truth
    cells = []
    for kind, eid, desc, (a, b), mem in events:
        mem = set(mem)
        for qid, amap in answers.items():
            if qid not in qinfo:
                continue
            sa = half_stats(amap, tags_a, mem)
            sb = half_stats(amap, tags_b, mem)
            if not sa or not sb or sa["n_x"] < min_nx_a or sb["n_x"] < min_nx_b:
                continue
            shrunk = math.sqrt(max(sa["delta"] ** 2 - sa["se_delta"] ** 2, 0.0))
            cells.append({
                # v1 fields (half-A / certification values)
                "game_id": game_id, "qid": qid,
                "question": qinfo[qid]["question"],
                "event_id": eid, "event_kind": kind, "event_desc": desc,
                "window": [a, b], "freq": sa["freq"], "n_x": sa["n_x"],
                "p_y": sa["p_y"], "p_yx": sa["p_yx"],
                "delta": sa["delta"], "se_delta": sa["se_delta"],
                "cls": "effect" if abs(sa["delta"]) > 2 * sa["se_delta"] else "placebo",
                # v2 fields
                "horizon": qinfo[qid]["horizon"],
                "resolution_turn": qinfo[qid]["resolution_turn"],
                "template_id": qinfo[qid]["template_id"],
                "half_a": sa, "half_b": sb,
                "shrunk_abs_delta": round(shrunk, 4),
            })
    for kind, eid, desc, w, mem in events:
        fa = len(set(mem) & tags_a) / n_a if n_a else 0.0
        print(f"  [{kind}] {w} fA={fa:.2f}  {desc[:60]}")
    return cells


def summarize(cells: list[dict]) -> list[dict]:
    """Cells per world x template-family x horizon x effect/placebo."""
    counts = Counter((c["game_id"], c["template_id"], c["horizon"], c["cls"])
                     for c in cells)
    rows = [{"game_id": g, "template_id": t, "horizon": h, "cls": cls, "cells": n}
            for (g, t, h, cls), n in sorted(counts.items())]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", default=None,
                    help="world id (default: manifest config game_id or dir name)")
    ap.add_argument("--arm-dir", default=None, help="one world's arm dir")
    ap.add_argument("--fleet-dir", default=None,
                    help="parent dir; every subdir with a manifest.json is a world")
    ap.add_argument("--questions-pattern",
                    default="data/questions_mc/{game_id}/questions.json")
    ap.add_argument("--horizons", default=DEFAULT_HORIZONS)
    ap.add_argument("--windows", default="60-75,75-90")
    ap.add_argument("--freq-band", default="0.15,0.65")
    ap.add_argument("--max-specific", type=int, default=6)
    ap.add_argument("--max-vague", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary-out", default=None,
                    help="default: <out stem>_summary.json")
    args = ap.parse_args()
    if bool(args.arm_dir) == bool(args.fleet_dir):
        ap.error("exactly one of --arm-dir / --fleet-dir is required")
    lo, hi = map(float, args.freq_band.split(","))
    windows = [tuple(map(int, w.split("-"))) for w in args.windows.split(",")]
    horizons = set(args.horizons.split(","))

    worlds = []
    if args.arm_dir:
        worlds.append((args.game_id, args.arm_dir))
    else:
        for d in sorted(Path(args.fleet_dir).iterdir()):
            if (d / "manifest.json").exists():
                worlds.append((None, str(d)))
        if not worlds:
            ap.error(f"no <world>/manifest.json under {args.fleet_dir}")

    cells = []
    for game_id, arm_dir in worlds:
        if game_id is None:
            cfg = json.loads((Path(arm_dir) / "manifest.json").read_text()).get("config", {})
            game_id = cfg.get("game_id") or Path(arm_dir).name
        qpath = args.questions_pattern.format(game_id=game_id)
        cells.extend(mine_world(game_id, arm_dir, qpath, horizons, windows,
                                lo, hi, args.max_specific, args.max_vague))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(cells, open(args.out, "w"), indent=1)
    rows = summarize(cells)
    summary_out = args.summary_out or str(Path(args.out).with_name(
        Path(args.out).stem + "_summary.json"))
    json.dump({"meta": {"worlds": sorted({c["game_id"] for c in cells}),
                        "horizons": sorted(horizons), "windows": args.windows,
                        "freq_band": [lo, hi], "split": "even-odd by rollout index",
                        "n_cells": len(cells)},
               "counts": rows},
              open(summary_out, "w"), indent=1)

    eff = sum(1 for c in cells if c["cls"] == "effect")
    print(f"\n{len(cells)} cells ({eff} effect, {len(cells) - eff} placebo) "
          f"-> {args.out}\nsummary (world x template x horizon x cls) -> {summary_out}")
    print(f"{'world':12s} {'template':22s} {'hz':3s} {'cls':8s} cells")
    for r in rows:
        print(f"{r['game_id']:12s} {str(r['template_id']):22s} "
              f"{r['horizon']:3s} {r['cls']:8s} {r['cells']:5d}")


if __name__ == "__main__":
    main()
