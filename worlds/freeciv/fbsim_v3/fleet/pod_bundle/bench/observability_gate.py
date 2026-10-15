#!/usr/bin/env python3
"""Observability gate for serialized FreeCiv rollouts.

The recorder is a client attached to player 0, so any per-player metric taken
from client state files is fog-of-war limited: other players' cities/units/
population/territory are undercounted, scores are -1 without an embassy and
treasury is 0 when unknown. The serializer must override those from the
omniscient server savegame (metadata.truth_source == "savegame"). This gate
refuses a serialized file that still carries the client view.

Gate conditions (ALL must hold):
    1. metadata.truth_source present with source == "savegame" and every turn
       overridden (turns_overridden == turns_total, no turns_without_savegame)
    2. metadata.savegame_coverage == 1.0
    3. no score == -1 at any turn (time_series.scores) or in any snapshot
    4. time_series.is_alive and time_series.government present for every turn
    5. (with --recording-dir) every turn's savegame re-parsed independently:
       every player in the savegame must have a cities_count cell >= the
       city table length (a player shown with fewer cities than the server
       has is the fog-of-war failure mode), scores must equal [scoreN].total
       and is_alive must match. This is the checkable form of "cities_count
       < savegame-derived city count"; the data-only rules 1-4 are what the
       gate can check without savegames.

Informational only (reported in stats, never fails the gate): the city-table
vs [scoreN].cities cross-check mismatches recorded in truth_source. The
server computes [scoreN] before the turn-change city founding/capture that
the save reflects, so ~0.5% of cells legitimately differ by one; the city
table is the value the client also sees for its own player.

Usage:
    uv run python bench/observability_gate.py path/to/X_data.json[.gz] \
        [--recording-dir logs/recordings/<username>] [--username <username>]

Exit codes: 0 = pass, 1 = flagged, 2 = usage/input error.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from typing import Dict, List, Optional

TRUTH_SOURCE_SAVEGAME = "savegame"
REQUIRED_SERIES = ("scores", "cities_count", "units_count", "military_units_count",
                   "population", "territory_size", "treasury", "is_alive", "government")


def load_world(path: str) -> dict:
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def _turns(series: dict) -> List[int]:
    return sorted(int(t) for t in series.keys())


def _row(series: dict, turn: int) -> dict:
    if turn in series:
        return series[turn]
    return series.get(str(turn), {})


def _import_parser():
    """Import the bundle's savegame parser (bench/ sits next to worlds/)."""
    here = os.path.dirname(os.path.abspath(__file__))
    world_pkg = os.path.join(here, '..', 'worlds', 'freeciv')
    if world_pkg not in sys.path:
        sys.path.insert(0, world_pkg)
    from freeciv_world.world_reports.utils import savegame_parser  # noqa: WPS433
    return savegame_parser


def gate_data(data: dict, recording_dir: Optional[str] = None,
              username: Optional[str] = None, max_examples: int = 5) -> dict:
    """Apply the observability gate. Returns {'ok', 'reasons', 'stats'}."""
    reasons: List[str] = []
    stats: Dict[str, object] = {}
    metadata = data.get('metadata') or {}
    time_series = data.get('time_series') or {}

    # 1. truth_source marker
    truth = metadata.get('truth_source')
    if not isinstance(truth, dict) or truth.get('source') != TRUTH_SOURCE_SAVEGAME:
        reasons.append("metadata.truth_source missing or != 'savegame' "
                       "(per-player metrics are the fog-of-war client view)")
    else:
        if truth.get('turns_overridden') != truth.get('turns_total'):
            reasons.append(f"truth_source: only {truth.get('turns_overridden')}/"
                           f"{truth.get('turns_total')} turns overridden from savegames")
        if truth.get('turns_without_savegame'):
            reasons.append(f"truth_source: turns without savegame: "
                           f"{truth['turns_without_savegame'][:max_examples]}")
        # Informational: [scoreN] is computed before turn-change city
        # founding/capture, so a few one-off mismatches are expected.
        stats['cities_crosscheck_mismatches'] = len(truth.get('cities_crosscheck_mismatches') or [])

    # 2. savegame coverage
    cov = metadata.get('savegame_coverage')
    if cov is None:
        reasons.append('metadata.savegame_coverage stamp missing')
    elif cov != 1.0:
        reasons.append(f'savegame_coverage={cov:.4f} != 1.0')

    # 3. -1 scores
    scores = time_series.get('scores') or {}
    turns = _turns(scores)
    stats['turns'] = len(turns)
    neg = [(t, pid) for t in turns for pid, v in _row(scores, t).items() if v == -1]
    stats['negative_scores'] = len(neg)
    if neg:
        reasons.append(f"{len(neg)} score==-1 cells in time_series.scores, "
                       f"e.g. {neg[:max_examples]}")
    snap_neg = []
    for turn, snap in (data.get('snapshots') or {}).items():
        for pid, v in (snap.get('scores') or {}).items():
            if v == -1:
                snap_neg.append((turn, pid))
    if snap_neg:
        reasons.append(f"{len(snap_neg)} score==-1 cells in snapshots, e.g. {snap_neg[:max_examples]}")

    # 4. required series present on every turn
    for name in REQUIRED_SERIES:
        series = time_series.get(name)
        if series is None:
            reasons.append(f"time_series.{name} missing")
            continue
        empty = [t for t in turns if not _row(series, t)]
        if empty and name in ("is_alive", "government"):
            reasons.append(f"time_series.{name} empty at {len(empty)} turns, e.g. {empty[:max_examples]}")

    # 5. independent re-check against the savegames
    if recording_dir:
        sp = _import_parser()
        if not username:
            username = os.path.basename(os.path.normpath(recording_dir))
        cities_series = time_series.get('cities_count') or {}
        alive_series = time_series.get('is_alive') or {}
        missing, under, score_mismatch, alive_mismatch = [], [], [], []
        for t in turns:
            sg = sp.extract_complete_data_from_savegame(username, t, recording_dir=recording_dir)
            if not sg:
                missing.append(t)
                continue
            c_row, s_row, a_row = _row(cities_series, t), _row(scores, t), _row(alive_series, t)
            for pid, city_list in (sg.get('cities') or {}).items():
                got = c_row.get(str(pid), c_row.get(pid))
                truth_n = len(city_list)
                if got is None or got < truth_n:
                    under.append((t, pid, got, truth_n))
            for pid, row in (sg.get('scores') or {}).items():
                got = s_row.get(str(pid), s_row.get(pid))
                if got != row.get('total', 0):
                    score_mismatch.append((t, pid, got, row.get('total')))
            for pid, is_alive in (sg.get('alive') or {}).items():
                got = a_row.get(str(pid), a_row.get(pid))
                if got is None or bool(got) != bool(is_alive):
                    alive_mismatch.append((t, pid, got, is_alive))
        stats.update({'savegames_missing': len(missing), 'cities_undercount': len(under),
                      'score_mismatch': len(score_mismatch), 'alive_mismatch': len(alive_mismatch)})
        if missing:
            reasons.append(f"{len(missing)} turns without a parseable savegame, e.g. {missing[:max_examples]}")
        if under:
            reasons.append(f"{len(under)} (turn,pid) cells with cities_count < savegame cities "
                           f"(turn,pid,got,truth), e.g. {under[:max_examples]}")
        if score_mismatch:
            reasons.append(f"{len(score_mismatch)} score cells != [scoreN].total, e.g. {score_mismatch[:max_examples]}")
        if alive_mismatch:
            reasons.append(f"{len(alive_mismatch)} is_alive cells != savegame, e.g. {alive_mismatch[:max_examples]}")

    return {'ok': not reasons, 'reasons': reasons, 'stats': stats}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Observability gate for serialized FreeCiv rollouts')
    parser.add_argument('data', help='serialized *_data.json or *_data.json.gz')
    parser.add_argument('--recording-dir', default=None,
                        help='recording dir with savegames/ to re-check every turn against')
    parser.add_argument('--username', default=None,
                        help='savegame username prefix (default: basename of --recording-dir)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args(argv)

    if not os.path.isfile(args.data):
        print(f'ERROR: no such file {args.data}')
        return 2
    try:
        data = load_world(args.data)
    except Exception as e:  # noqa: BLE001
        print(f'ERROR: unreadable {args.data}: {e!r}')
        return 2

    result = gate_data(data, recording_dir=args.recording_dir, username=args.username)
    verdict = 'PASS' if result['ok'] else 'FLAGGED'
    print(f"observability gate: {verdict} ({args.data})")
    if not args.quiet:
        for reason in result['reasons']:
            print(f"  - {reason}")
        print(f"  stats: {result['stats']}")
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
