#!/usr/bin/env python3
"""Assemble the uplift-v2 Verification Report (JSON + markdown).

One document that lets a reader audit the production benchmark bundle
without re-running anything, framed pre-registered-gates-first: Section 1
states every acceptance gate and its threshold as registered (plus the
verbatim gate-revision log), and only then do the evidence sections report
what was measured. Sections:

  (a) pilot gate verdicts   — the frozen pilot_gates.py verdict JSONs
      (P1-only, full P1+P2, and the preserved pre-revision-2 snapshot);
  (b) world health          — recomputed per-world frozen-civ count,
      tech-event coverage, and savegame_coverage stamp over --worlds-dir
      (*_data.json[.gz]), via world_publish_gate.py's stats/gate logic;
  (c) fork exchangeability  — pilot_gates.py's P2 computations re-run over
      --forks-dir + anchor artifacts (per-anchor section);
  (d) truth quality         — split-half reliability (Pearson r between the
      half-A and held-out half-B values) of p_y, p_yx, and delta per horizon
      from a natcond_cells_v2.py --cells file;
  (e) provenance            — sha256 of every file under --bundle-dir, plus
      the gate-revision log (verbatim, also quoted in the markdown).

Every input is optional: absent inputs yield a section marked missing, so the
report can be assembled incrementally as the fleet lands.

Usage:
  uv run python scripts/uplift_v2/verification_report.py \
      --p1-verdict tmp/pilot_v2/pilot_gates_p1_verdict.json \
      --p2-verdict tmp/pilot_v2/pilot_gates_verdict.json \
      --prerev-verdict tmp/pilot_v2/pilot_gates_verdict_prerev2.json \
      --worlds-dir tmp/mined/worlds \
      --forks-dir tmp/pilot_v2/pod_out/forks \
      --anchor tmp/pilot_v2/pod_out/worlds/seed1000_data.json.gz \
      --anchor-savegames tmp/pilot_v2/pod_out/worlds/savegames/seed1000 \
      --pairs tmp/pilot_v2/pairs.json \
      --anchor-bank tmp/pilot_v2/anchor_bank.json \
      --cells tmp/natcond_v2/fleet_cells.json \
      --bundle-dir tmp/bundle_v2 \
      --out-json tmp/verification/verification_report.json

Exit codes: 0 = assembled (regardless of gate outcomes; the verdicts speak
for themselves), 2 = usage error.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import pilot_gates  # noqa: E402
import world_publish_gate as wpg  # noqa: E402

# ---------------------------------------------------------------------------
# Gate-revision log. VERBATIM entries — never rephrase: these are the exact
# coordinator-verified records of every change to a pre-registered gate.
# ---------------------------------------------------------------------------
GATE_REVISION_LOG = [
    "Revision 1: p1_tech_events band [400,600]->[300,805]; evidence: "
    "52-recording healthy reference min=314/med=517/max=732, legacy "
    "seed1=644, zero duplicate (turn,pid,tech) events in out-of-band worlds "
    "1004=645/1010=615/1021=397; commit 7bbd6764",
    "Revision 2: p2_p0_next_tech_modal replaced by "
    "p2_anchor_path_not_excluded; evidence: anchor path Literacy t61 -> "
    "The Republic t69 is the fork modal path 100/100 and 99/100, p0 held 99 "
    "banked Literacy bulbs at t60 making tech #1 legitimately deterministic; "
    "commit 779c06e0",
]

# The world publish gate's registered conditions (world_publish_gate.py).
WORLD_GATE_CONDITIONS = [
    "frozen_civs == 0 (every reference-turn civ has >=1 tech event)",
    "wonder_completed events > 0",
    "diplomacy relations non-empty",
    "savegame_coverage == 1.0 stamped in metadata",
]


# ---------------------------------------------------------------------------
# (a) pilot gate verdicts
# ---------------------------------------------------------------------------

def load_pilot_verdicts(p1_path: str | None, p2_path: str | None,
                        prerev_path: str | None) -> dict:
    """The frozen pilot_gates.py verdict JSONs, echoed into the report.

    'pre_revision_snapshot' is the P1+P2 verdict preserved BEFORE gate
    revision #2 landed (it shows the single p2_p0_next_tech_modal FAIL that
    motivated the revision)."""
    out = {}
    for key, path in (("p1", p1_path), ("p1_p2", p2_path),
                      ("pre_revision_snapshot", prerev_path)):
        if path and os.path.exists(path):
            v = wpg.load_world(path)  # plain/gz JSON loader
            out[key] = {"path": path, "missing": False,
                        "overall": v.get("overall"),
                        "n_pass": v.get("n_pass"), "n_fail": v.get("n_fail"),
                        "n_missing": v.get("n_missing"),
                        "inputs": v.get("inputs", {}),
                        "gates": v.get("gates", [])}
        else:
            out[key] = {"path": path, "missing": True, "overall": None,
                        "gates": []}
    return out


# ---------------------------------------------------------------------------
# (b) production world health (world_publish_gate.py logic/format)
# ---------------------------------------------------------------------------

def world_health_table(worlds_dir: str) -> list[dict]:
    """Per-world health rows recomputed from the serialized worlds."""
    rows = []
    for path in wpg.collect_world_files(worlds_dir):
        wid = wpg.world_id_from_path(path)
        try:
            data = wpg.load_world(path)
        except Exception as e:  # noqa: BLE001
            rows.append({"world": wid, "error": repr(e), "publish": False,
                         "reasons": [f"unreadable: {e!r}"]})
            continue
        verdict = wpg.gate_world(data)
        s = verdict["stats"]
        rows.append({
            "world": wid,
            "frozen_civs": s["frozen_civs"],
            "n_civs": s["n_civs"],
            "tech_events": s["tech_events"],
            "tech_covers_all_civs": (None if s["frozen_civs"] is None
                                     else s["frozen_civs"] == 0),
            "wonder_completed_events": s["wonder_completed_events"],
            "diplomacy_pairs": s["diplomacy_pairs"],
            "savegame_coverage": s["savegame_coverage"],
            "publish": verdict["publish"],
            "reasons": verdict["reasons"],
        })
    return rows


# ---------------------------------------------------------------------------
# (c) per-anchor fork exchangeability (pilot_gates.py P2 computations)
# ---------------------------------------------------------------------------

def fork_exchangeability(forks_dir: str, anchor: str | None,
                         anchor_savegames: str | None, pairs: str | None,
                         anchor_bank: str | None, fork_turn: int = 60,
                         window_end: int = 90,
                         expected_forks: int = 100) -> dict:
    """Re-run every P2 gate over one anchor's fork ensemble."""
    gates = pilot_gates.run_p2_gates(
        forks_dir, anchor, anchor_savegames, pairs, anchor_bank,
        fork_turn, window_end, expected_forks)
    counts = Counter(g["status"] for g in gates)
    anchor_id = (wpg.world_id_from_path(anchor) if anchor
                 else Path(forks_dir).name)
    return {"anchor": anchor_id, "forks_dir": forks_dir,
            "fork_turn": fork_turn, "window_end": window_end,
            "expected_forks": expected_forks,
            "n_pass": counts.get(pilot_gates.PASS, 0),
            "n_fail": counts.get(pilot_gates.FAIL, 0),
            "n_missing": counts.get(pilot_gates.MISSING, 0),
            "gates": gates}


# ---------------------------------------------------------------------------
# (d) truth quality: split-half reliability
# ---------------------------------------------------------------------------

def pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r; None when undefined (n < 2 or a degenerate margin)."""
    n = len(xs)
    if n < 2 or n != len(ys):
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0 or syy == 0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)

def truth_quality(cells: list[dict]) -> dict:
    """Per-horizon split-half reliability of the mined truth.

    For each horizon, the Pearson r between the half-A (certification) and
    half-B (held-out scoring truth) values of p_y, p_yx, and delta across
    that horizon's cells. High r = the mined probabilities replicate across
    independent rollout halves; r on delta is the reliability of the
    conditional effect itself."""
    by_hz: dict[str, list[dict]] = {}
    for c in cells:
        if c.get("half_a") and c.get("half_b") and c.get("horizon"):
            by_hz.setdefault(c["horizon"], []).append(c)
    out = {}
    for hz in sorted(by_hz):
        cs = by_hz[hz]
        row: dict = {"n_cells": len(cs),
                     "resolution_turns": sorted(
                         {c.get("resolution_turn") for c in cs})}
        for field in ("p_y", "p_yx", "delta"):
            r = pearson([c["half_a"][field] for c in cs],
                        [c["half_b"][field] for c in cs])
            row[f"r_{field}"] = None if r is None else round(r, 4)
        out[hz] = row
    return out


# ---------------------------------------------------------------------------
# (e) provenance
# ---------------------------------------------------------------------------

def bundle_hashes(bundle_dir: str) -> dict[str, str]:
    """sha256 of every file under bundle_dir, keyed by relative path."""
    root = Path(bundle_dir)
    out = {}
    for fp in sorted(p for p in root.rglob("*") if p.is_file()):
        h = hashlib.sha256()
        with open(fp, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[str(fp.relative_to(root))] = h.hexdigest()
    return out


# ---------------------------------------------------------------------------
# Assembly + rendering
# ---------------------------------------------------------------------------

def assemble_report(pilot: dict | None = None,
                    world_health: list[dict] | None = None,
                    fork_exch: dict | None = None,
                    truth: dict | None = None,
                    provenance: dict | None = None) -> dict:
    provenance = dict(provenance or {})
    provenance["gate_revision_log"] = GATE_REVISION_LOG
    return {
        "report": "uplift_v2 verification report",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "framing": "pre-registered gates stated first; evidence follows",
        "sections": {
            "pilot_gate_verdicts": pilot,
            "world_health": world_health,
            "fork_exchangeability": fork_exch,
            "truth_quality": truth,
            "provenance": provenance,
        },
    }


def _gate_threshold_rows(pilot: dict | None) -> list[tuple[str, str]]:
    """(gate, threshold) as registered, deduped, from the frozen verdicts
    (the fullest verdict wins per gate name)."""
    rows: dict[str, str] = {}
    for key in ("p1", "p1_p2"):
        for g in ((pilot or {}).get(key) or {}).get("gates", []):
            rows.setdefault(g["gate"], g.get("threshold", "-"))
    return sorted(rows.items())


def _md_table(headers: list[str], rows: list[list]) -> list[str]:
    def cell(v):
        return "-" if v is None else str(v).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(" --- " for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    return lines

def _gates_md(gates: list[dict]) -> list[str]:
    return _md_table(
        ["gate", "status", "value", "threshold"],
        [[g["gate"], g["status"], g.get("value"), g.get("threshold")]
         for g in gates])


def render_markdown(report: dict) -> str:
    s = report["sections"]
    pilot, health = s["pilot_gate_verdicts"], s["world_health"]
    fork, truth, prov = (s["fork_exchangeability"], s["truth_quality"],
                         s["provenance"])
    L: list[str] = []
    L += ["# Uplift v2 Verification Report",
          "",
          f"Generated: {report['generated']}",
          "",
          "Framing: the acceptance gates below were pre-registered before "
          "the production runs; two were revised during the pilot with "
          "coordinator-verified evidence (Section 2, quoted verbatim). "
          "Everything after Section 2 is evidence measured against these "
          "gates.",
          ""]

    # -- Section 1: pre-registered gates ---------------------------------
    L += ["## 1. Pre-registered gates", ""]
    thr = _gate_threshold_rows(pilot)
    if thr:
        L += ["### P1 world-health and P2 fork-exchangeability gates "
              "(as registered)", ""]
        L += _md_table(["gate", "threshold"], [list(t) for t in thr])
        L += [""]
    else:
        L += ["(pilot verdict files not provided; registered thresholds "
              "not echoed)", ""]
    L += ["### World publish gate (per serialized world)", ""]
    L += [f"- {c}" for c in WORLD_GATE_CONDITIONS]
    L += ["",
          "### Truth-quality check",
          "",
          "- split-half reliability: Pearson r between half-A "
          "(certification) and half-B (held-out truth) values of p_y, "
          "p_yx, delta, per horizon (reported as evidence; the halves are "
          "fixed by rollout index before any selection)",
          ""]

    # -- Section 2: gate revision log (verbatim) --------------------------
    L += ["## 2. Gate revision log (verbatim)", ""]
    L += [f"- {entry}" for entry in prov["gate_revision_log"]]
    L += [""]

    # -- Section 3: pilot gate verdicts -----------------------------------
    L += ["## 3. Evidence: pilot gate verdicts", ""]
    if pilot:
        for key, title in (("p1", "P1 verdict (world health, re-mined pilot "
                                  "worlds)"),
                           ("p1_p2", "P1+P2 verdict (current gates)"),
                           ("pre_revision_snapshot",
                            "Pre-revision-2 snapshot (preserved; shows the "
                            "verdict under the superseded gate)")):
            v = pilot.get(key) or {}
            L += [f"### {title}", ""]
            if v.get("missing"):
                L += [f"MISSING: {v.get('path')}", ""]
                continue
            L += [f"Overall: **{v['overall']}** ({v['n_pass']} pass, "
                  f"{v['n_fail']} fail, {v['n_missing']} missing) — "
                  f"`{v['path']}`", ""]
            L += _gates_md(v["gates"]) + [""]
    else:
        L += ["Not assembled (verdict inputs not provided).", ""]

    # -- Section 4: production world health --------------------------------
    L += ["## 4. Evidence: production world health", ""]
    if health is not None:
        n_pub = sum(1 for r in health if r.get("publish"))
        L += [f"{n_pub}/{len(health)} worlds publishable under the world "
              "publish gate.", ""]
        L += _md_table(
            ["world", "frozen civs", "tech events", "tech covers all civs",
             "wonders", "diplomacy pairs", "savegame coverage", "verdict"],
            [[r.get("world"), r.get("frozen_civs"), r.get("tech_events"),
              r.get("tech_covers_all_civs"), r.get("wonder_completed_events"),
              r.get("diplomacy_pairs"), r.get("savegame_coverage"),
              ("PUBLISH" if r.get("publish")
               else "FLAGGED: " + "; ".join(r.get("reasons", [])))]
             for r in health])
        L += [""]
    else:
        L += ["Not assembled (--worlds-dir not provided).", ""]

    # -- Section 5: fork exchangeability -----------------------------------
    L += ["## 5. Evidence: fork exchangeability", ""]
    if fork:
        L += [f"Anchor `{fork['anchor']}` — fork turn t{fork['fork_turn']}, "
              f"window end t{fork['window_end']}; {fork['n_pass']} pass, "
              f"{fork['n_fail']} fail, {fork['n_missing']} missing "
              f"({fork['forks_dir']}).", ""]
        L += _gates_md(fork["gates"]) + [""]
    else:
        L += ["Not assembled (--forks-dir not provided).", ""]

    # -- Section 6: truth quality ------------------------------------------
    L += ["## 6. Evidence: truth quality (split-half reliability)", ""]
    if truth is not None:
        L += _md_table(
            ["horizon", "resolution turns", "cells", "r(p_y)", "r(p_yx)",
             "r(delta)"],
            [[hz, ",".join(str(t) for t in row["resolution_turns"]),
              row["n_cells"], row["r_p_y"], row["r_p_yx"], row["r_delta"]]
             for hz, row in truth.items()])
        L += [""]
    else:
        L += ["Not assembled (--cells not provided).", ""]

    # -- Section 7: provenance ---------------------------------------------
    L += ["## 7. Provenance", ""]
    if prov.get("bundle_dir"):
        L += [f"Bundle: `{prov['bundle_dir']}` "
              f"({len(prov.get('sha256', {}))} files)", ""]
        L += _md_table(["file", "sha256"],
                       [[fp, h] for fp, h in prov.get("sha256", {}).items()])
        L += [""]
    else:
        L += ["Bundle hashes not assembled (--bundle-dir not provided).", ""]
    L += ["The gate-revision log in Section 2 is part of this report's "
          "provenance record.", ""]
    return "\n".join(L)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Assemble the uplift-v2 Verification Report")
    ap.add_argument("--p1-verdict",
                    default="tmp/pilot_v2/pilot_gates_p1_verdict.json")
    ap.add_argument("--p2-verdict",
                    default="tmp/pilot_v2/pilot_gates_verdict.json")
    ap.add_argument("--prerev-verdict",
                    default="tmp/pilot_v2/pilot_gates_verdict_prerev2.json")
    ap.add_argument("--worlds-dir", default=None,
                    help="production *_data.json[.gz] worlds to health-check")
    ap.add_argument("--forks-dir", default=None,
                    help="fork outputs for the exchangeability section")
    ap.add_argument("--anchor", default=None)
    ap.add_argument("--anchor-savegames", default=None)
    ap.add_argument("--pairs", default=None)
    ap.add_argument("--anchor-bank", default=None)
    ap.add_argument("--fork-turn", type=int, default=60)
    ap.add_argument("--window-end", type=int, default=90)
    ap.add_argument("--expected-forks", type=int, default=100)
    ap.add_argument("--cells", default=None,
                    help="natcond_cells_v2.py output for the truth-quality "
                         "section")
    ap.add_argument("--bundle-dir", default=None,
                    help="bundle files to sha256 for provenance")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", default=None,
                    help="default: <out-json stem>.md")
    args = ap.parse_args(argv)

    pilot = load_pilot_verdicts(args.p1_verdict, args.p2_verdict,
                                args.prerev_verdict)
    health = (world_health_table(args.worlds_dir)
              if args.worlds_dir and os.path.isdir(args.worlds_dir) else None)
    fork = (fork_exchangeability(args.forks_dir, args.anchor,
                                 args.anchor_savegames, args.pairs,
                                 args.anchor_bank, args.fork_turn,
                                 args.window_end, args.expected_forks)
            if args.forks_dir else None)
    truth = (truth_quality(json.loads(Path(args.cells).read_text()))
             if args.cells and os.path.exists(args.cells) else None)
    prov = {}
    if args.bundle_dir and os.path.isdir(args.bundle_dir):
        prov = {"bundle_dir": args.bundle_dir,
                "sha256": bundle_hashes(args.bundle_dir)}

    report = assemble_report(pilot=pilot, world_health=health, fork_exch=fork,
                             truth=truth, provenance=prov)
    md = render_markdown(report)

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=1))
    out_md = Path(args.out_md) if args.out_md else out_json.with_suffix(".md")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md)

    n_worlds = len(health) if health is not None else 0
    print(f"verification report -> {out_json} + {out_md}\n"
          f"  pilot verdicts: "
          + ", ".join(f"{k}={v['overall'] or 'MISSING'}"
                      for k, v in pilot.items())
          + f"\n  world health: {n_worlds} worlds"
          + (f"; fork gates: {fork['n_pass']}P/{fork['n_fail']}F/"
             f"{fork['n_missing']}M" if fork else "")
          + (f"; truth horizons: {sorted(truth)}" if truth else "")
          + (f"; bundle files hashed: {len(prov.get('sha256', {}))}"
             if prov else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
