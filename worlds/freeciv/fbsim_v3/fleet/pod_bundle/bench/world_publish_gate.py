#!/usr/bin/env python3
"""Publish gate for serialized FreeCiv worlds.

A serialized world (seed{N}_data.json[.gz]) may be published only if it shows
the full-visibility signature. Worlds serialized without per-turn savegame
parsing silently degrade to the attached player's embassy-gated view and show
the "dark" signature instead: frozen civs, zero wonder events, empty
diplomacy. This gate makes publishing such a world impossible.

Gate conditions (ALL must hold):
    1. frozen_civs == 0        - every reference-turn civ has >=1 tech event
    2. wonder_completed  > 0   - at least one wonder_completed event
    3. diplomacy relations != {} - pairwise diplomacy was extracted
    4. savegame_coverage == 1.0  - every serialized turn's savegame parsed
       (metadata stamp; absent on legacy corpora - see --coverage-optional)

Definitions (matched against tmp/tail_v2/world_coverage_audit.json):
    reference turn = 60 if turn 60 is in the series, else the first turn
    civs           = player ids present in time_series.techs_known at the
                     reference turn
    frozen civ     = such a player id with zero tech_discovered events over
                     the whole serialized window

Usage:
    # Gate a single world before publishing (exit 0 = publish allowed):
    uv run python scripts/uplift_v2/world_publish_gate.py path/to/seed1000_data.json.gz

    # Dry-run over a corpus and cross-check against the coverage audit:
    uv run python scripts/uplift_v2/world_publish_gate.py tmp/mined/worlds \
        --coverage-optional --audit tmp/tail_v2/world_coverage_audit.json

Exit codes: 0 = all gated worlds pass (and audit matches, if given), 1 = any
world flagged (or audit mismatch), 2 = usage/input error.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from typing import Dict, List, Optional


REFERENCE_TURN = 60


def load_world(path: str) -> dict:
    """Load a serialized world JSON (.json or .json.gz)."""
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def _turn_key(series: dict, turn: int):
    """time_series keys may be int or str depending on (de)serialization."""
    if turn in series:
        return turn
    if str(turn) in series:
        return str(turn)
    return None


def compute_world_stats(data: dict) -> dict:
    """Compute the gate-relevant statistics for one serialized world."""
    events = data.get('events', [])
    techs_known = data.get('time_series', {}).get('techs_known', {})

    turns = sorted(int(t) for t in techs_known.keys())
    if not turns:
        return {
            'reference_turn': None, 'n_civs': 0, 'frozen_civs': None,
            'frozen_civ_ids': [], 'wonder_completed_events': 0,
            'diplomacy_pairs': 0, 'savegame_coverage': None,
            'tech_events': 0,
        }

    ref_turn = REFERENCE_TURN if _turn_key(techs_known, REFERENCE_TURN) is not None else turns[0]
    ref_row = techs_known[_turn_key(techs_known, ref_turn)]
    civ_ids = sorted(int(p) for p in ref_row.keys())

    tech_events = [e for e in events if e.get('type') == 'tech_discovered']
    tech_pids = {e.get('player_id') for e in tech_events}
    frozen_ids = [p for p in civ_ids if p not in tech_pids]

    wonders = sum(1 for e in events if e.get('type') == 'wonder_completed')
    relations = (data.get('diplomacy') or {}).get('relations') or {}

    coverage = data.get('metadata', {}).get('savegame_coverage')

    return {
        'reference_turn': ref_turn,
        'n_civs': len(civ_ids),
        'frozen_civs': len(frozen_ids),
        'frozen_civ_ids': frozen_ids,
        'wonder_completed_events': wonders,
        'diplomacy_pairs': len(relations),
        'savegame_coverage': coverage,
        'tech_events': len(tech_events),
    }


def gate_world(data: dict, coverage_optional: bool = False) -> dict:
    """Apply the publish gate to one serialized world.

    Returns a dict with 'publish' (bool), 'reasons' (list of failed checks),
    and 'stats'.
    """
    stats = compute_world_stats(data)
    reasons: List[str] = []

    if stats['frozen_civs'] is None:
        reasons.append('no techs_known time series')
    elif stats['frozen_civs'] != 0:
        reasons.append(
            f"frozen_civs={stats['frozen_civs']} (ids {stats['frozen_civ_ids']})")

    if stats['wonder_completed_events'] <= 0:
        reasons.append('wonder_completed_events=0')

    if stats['diplomacy_pairs'] <= 0:
        reasons.append('diplomacy relations empty')

    cov = stats['savegame_coverage']
    if cov is None:
        if not coverage_optional:
            reasons.append('savegame_coverage stamp missing '
                           '(legacy serialization? see --coverage-optional)')
    elif cov != 1.0:
        reasons.append(f'savegame_coverage={cov:.4f} != 1.0')

    return {'publish': not reasons, 'reasons': reasons, 'stats': stats}


def collect_world_files(target: str) -> List[str]:
    if os.path.isdir(target):
        files = sorted(
            glob.glob(os.path.join(target, '*_data.json.gz'))
            + glob.glob(os.path.join(target, '*_data.json')))
        return files
    if os.path.isfile(target):
        return [target]
    return []


def world_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    for suffix in ('_data.json.gz', '_data.json'):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def run_audit_comparison(results: Dict[str, dict], audit_path: str) -> dict:
    """Compare gate verdicts against a coverage audit (frozen>=2 == dark).

    Returns a confusion-matrix dict; a perfect gate flags exactly the audit's
    dark worlds and passes exactly its healthy ones.
    """
    with open(audit_path) as f:
        audit = json.load(f)

    matrix = {
        'dark_flagged': 0,      # audit dark, gate flagged (true positive)
        'dark_passed': 0,       # audit dark, gate passed (MISS)
        'healthy_flagged': 0,   # audit healthy, gate flagged (FALSE ALARM)
        'healthy_passed': 0,    # audit healthy, gate passed (true negative)
        'not_in_audit': 0,
        'misses': [],
        'false_alarms': [],
    }

    for world_id, result in results.items():
        entry = audit.get(world_id)
        if entry is None:
            matrix['not_in_audit'] += 1
            continue
        dark = entry.get('frozen', 0) >= 2
        flagged = not result['publish']
        if dark and flagged:
            matrix['dark_flagged'] += 1
        elif dark and not flagged:
            matrix['dark_passed'] += 1
            matrix['misses'].append(world_id)
        elif not dark and flagged:
            matrix['healthy_flagged'] += 1
            matrix['false_alarms'].append(world_id)
        else:
            matrix['healthy_passed'] += 1

    matrix['exact'] = (matrix['dark_passed'] == 0
                       and matrix['healthy_flagged'] == 0)
    return matrix


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='Publish gate for serialized FreeCiv worlds')
    parser.add_argument('target',
                        help='A world data JSON(.gz) file or a directory of them')
    parser.add_argument('--coverage-optional', action='store_true',
                        help='Do not fail worlds lacking the savegame_coverage '
                             'metadata stamp (legacy corpora only; new '
                             'serializations always carry the stamp)')
    parser.add_argument('--audit', default=None,
                        help='world_coverage_audit.json to cross-check the '
                             'verdicts against (dry-run validation)')
    parser.add_argument('--json-out', default=None,
                        help='Write the full machine-readable verdict here')
    parser.add_argument('--quiet', action='store_true',
                        help='Only print the summary lines')
    args = parser.parse_args(argv)

    files = collect_world_files(args.target)
    if not files:
        print(f'ERROR: no world data files found at {args.target}')
        return 2

    results: Dict[str, dict] = {}
    for path in files:
        world_id = world_id_from_path(path)
        try:
            data = load_world(path)
        except Exception as e:
            results[world_id] = {
                'publish': False,
                'reasons': [f'unreadable: {e!r}'],
                'stats': {},
            }
            continue
        results[world_id] = gate_world(
            data, coverage_optional=args.coverage_optional)

    n_pass = sum(1 for r in results.values() if r['publish'])
    n_flag = len(results) - n_pass

    if not args.quiet:
        for world_id in sorted(results):
            r = results[world_id]
            verdict = 'PUBLISH' if r['publish'] else 'FLAGGED'
            reason = '' if r['publish'] else '  [' + '; '.join(r['reasons']) + ']'
            print(f'{world_id}: {verdict}{reason}')
        print()

    print(f'Publish gate: {n_pass} publishable, {n_flag} flagged '
          f'(of {len(results)})')

    exit_code = 0 if n_flag == 0 else 1

    audit_matrix = None
    if args.audit:
        audit_matrix = run_audit_comparison(results, args.audit)
        print('Audit cross-check (dark = audit frozen>=2):')
        print(f"  dark & flagged   : {audit_matrix['dark_flagged']}")
        print(f"  dark & passed    : {audit_matrix['dark_passed']}"
              + (f"  MISSES: {audit_matrix['misses'][:10]}"
                 if audit_matrix['misses'] else ''))
        print(f"  healthy & flagged: {audit_matrix['healthy_flagged']}"
              + (f"  FALSE ALARMS: {audit_matrix['false_alarms'][:10]}"
                 if audit_matrix['false_alarms'] else ''))
        print(f"  healthy & passed : {audit_matrix['healthy_passed']}")
        print(f"  not in audit     : {audit_matrix['not_in_audit']}")
        print(f"  exact separation : {audit_matrix['exact']}")
        # In dry-run mode the gate is judged against the audit, not against
        # whether dark worlds exist in the corpus (they are supposed to be
        # flagged there).
        exit_code = 0 if audit_matrix['exact'] else 1

    if args.json_out:
        payload = {
            'target': args.target,
            'coverage_optional': args.coverage_optional,
            'n_worlds': len(results),
            'n_publishable': n_pass,
            'n_flagged': n_flag,
            'results': results,
        }
        if audit_matrix is not None:
            payload['audit_matrix'] = audit_matrix
        with open(args.json_out, 'w') as f:
            json.dump(payload, f, indent=1)
        print(f'Wrote verdict JSON: {args.json_out}')

    return exit_code


if __name__ == '__main__':
    sys.exit(main())
