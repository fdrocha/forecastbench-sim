#!/usr/bin/env python3
"""End-to-end check that per-turn metrics come from the omniscient savegame.

The recorder is a FreeCiv client attached to player 0, so in a serialized
``*_data.json.gz`` the client-derived series (cities/units/population/
territory/treasury/scores) are fog-of-war limited for players 1..N while the
values for player 0 are exact. This test therefore uses the client's player-0
series as the oracle for the savegame-derived override, and the savegame as
the oracle for everybody else.

Run from the pod bundle directory:

    uv run python bench/test_observability.py

or from the main repo with the bundle on PYTHONPATH:

    PYTHONPATH=<bundle>/worlds/freeciv:<bundle>/packages/fbsim-core \
        uv run python <bundle>/bench/test_observability.py

Parts:
  (a) per-player client-vs-savegame table at T60 and T150 (rng5001)
  (b) MetricsCollector.collect_time_series driven end-to-end on rng5001 with
      minimal client states; player-0 equality vs the client data for
      cities/units/military/population/treasury/scores over all turns,
      score equality for every (turn, player) where the client had a score,
      savegame cities >= client cities everywhere, no -1 scores left
  (c) bench/observability_gate.py FAILS on the original data.json.gz and
      PASSES on the regenerated series
  (d) optional spot check on the rng5418 checkpoint fork (report only)

Test data locations can be overridden with FBSIM_TEST_RNG5001_DIR and
FBSIM_TEST_RNG5418_DIR.
"""

from __future__ import annotations

import glob
import gzip
import json
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
BUNDLE = os.path.dirname(HERE)
for p in (os.path.join(BUNDLE, 'worlds', 'freeciv'),
          os.path.join(BUNDLE, 'packages', 'fbsim-core'), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from freeciv_world.world_reports.extractors.metrics_collector import (  # noqa: E402
    MetricsCollector, TRUTH_SOURCE_SAVEGAME)
from freeciv_world.world_reports.utils import savegame_parser as sp  # noqa: E402
import observability_gate  # noqa: E402

CORPUS = '/Users/jaeholee0404/civbench/tmp/fbsim_v2_corpus/rollouts'
RNG5001_DIR = os.environ.get(
    'FBSIM_TEST_RNG5001_DIR', os.path.join(CORPUS, 'forks-s1000-a', 'forks', 'rng5001'))
RNG5001_USER = 'seed1000forkrng5001'
RNG5418_DIR = os.environ.get(
    'FBSIM_TEST_RNG5418_DIR', os.path.join(CORPUS, 'fix-s3c', 'forks', 'rng5418'))
RNG5418_USER = 'seed3forkrng5418'

# client series name -> (savegame accessor)
ORACLE_SERIES = ['cities_count', 'units_count', 'military_units_count',
                 'population', 'treasury', 'scores']

_failures: List[str] = []


def check(cond: bool, msg: str) -> None:
    print(('  PASS  ' if cond else '  FAIL  ') + msg)
    if not cond:
        _failures.append(msg)


def load_client_data(recording_dir: str, username: str) -> dict:
    path = os.path.join(recording_dir, f'{username}_data.json.gz')
    with gzip.open(path, 'rt') as f:
        return json.load(f)


def client_value(ts: dict, series: str, turn: int, pid: int):
    return (ts.get(series) or {}).get(str(turn), {}).get(str(pid))


def savegame_row(sg: dict, pid: int) -> Dict[str, Any]:
    """Flatten one player's savegame state into the client series names."""
    cities = (sg.get('cities') or {}).get(pid)
    units = (sg.get('units') or {}).get(pid)
    score = (sg.get('scores') or {}).get(pid) or {}
    barb = (sg.get('barbarian') or {}).get(pid, 'None') != 'None'
    mil = None if units is None else sp.military_unit_count(
        units, score, barb, sg.get('rulesetdir') or '')
    return {
        'cities_count': None if cities is None else len(cities),
        'units_count': None if units is None else len(units),
        'military_units_count': mil,
        'population': None if cities is None else sum(int(c['size']) for c in cities),
        'territory_size': (sg.get('tile_ownership') or {}).get(pid, 0),
        'treasury': (sg.get('gold') or {}).get(pid),
        'scores': score.get('total'),
        'is_alive': (sg.get('alive') or {}).get(pid),
        'government': (sg.get('governments') or {}).get(pid),
        'score_cities': score.get('cities'),
    }


def fmt(v) -> str:
    if v is None:
        return '-'
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    if isinstance(v, bool):
        return 'T' if v else 'F'
    return str(v)


def print_table(recording_dir: str, username: str, client: dict, turn: int) -> None:
    sg = sp.extract_complete_data_from_savegame(username, turn, recording_dir=recording_dir)
    if not sg:
        print(f'  (no savegame for T{turn})')
        return
    ts = client['time_series']
    cols = ['cities', 'units', 'mil', 'pop', 'terr', 'gold', 'score']
    print(f'\n  T{turn}  (client/savegame; player 0 is the recorder so its client values are exact)')
    print('  pid  ' + ''.join(f'{c:>14}' for c in cols) + f'{"alive":>7}{"gov":>12}')
    pids = sorted(set(sg.get('cities') or {}) | set(sg.get('scores') or {}))
    for pid in pids:
        s = savegame_row(sg, pid)
        row = []
        for series in ['cities_count', 'units_count', 'military_units_count',
                       'population', 'territory_size', 'treasury', 'scores']:
            row.append(f'{fmt(client_value(ts, series, turn, pid))}/{fmt(s[series])}')
        print(f'  {pid:<4} ' + ''.join(f'{r:>14}' for r in row)
              + f'{fmt(s["is_alive"]):>7}{(s["government"] or "-")[:11]:>12}')


def part_a(client: dict) -> None:
    print('\n== (a) client vs savegame per-player table, rng5001 ==')
    for turn in (60, 150):
        print_table(RNG5001_DIR, RNG5001_USER, client, turn)


def part_b(client: dict):
    print('\n== (b) collect_time_series end-to-end on rng5001 ==')
    ts_client = client['time_series']
    turns = sorted(int(t) for t in ts_client['scores'])
    check(turns[0] == 60 and turns[-1] == 211 and len(turns) == 152,
          f'client data covers T60..T211 (152 turns): {turns[0]}..{turns[-1]} n={len(turns)}')

    # Minimal client states: only the client-view score per listed player.
    states = {
        t: {'player': {pid: {'score': score}
                       for pid, score in ts_client['scores'][str(t)].items()}}
        for t in turns
    }
    civilizations = {int(pid): dict(info) for pid, info in client['civilizations'].items()}
    config = SimpleNamespace(recording_dir=RNG5001_DIR, savegame_username=RNG5001_USER)
    check(sp.extract_username_from_config(config) == RNG5001_USER,
          'extract_username_from_config honours config.savegame_username')
    check(sp.extract_username_from_config(SimpleNamespace(recording_dir=RNG5001_DIR)) == RNG5001_USER,
          'extract_username_from_config infers the username from savegame filenames')

    sp.reset_savegame_coverage(RNG5001_DIR)
    collector = MetricsCollector()
    ts = collector.collect_time_series(states, config, civilizations, data_loader=None, events=[])
    cov = sp.compute_savegame_coverage(RNG5001_DIR, turns)
    check(cov['coverage'] == 1.0, f'savegame coverage == 1.0 (parsed {cov["parsed"]}/{cov["expected"]})')
    check(collector.truth_source.get('source') == TRUTH_SOURCE_SAVEGAME,
          'truth_source marker == "savegame"')
    check(collector.truth_source.get('turns_overridden') == len(turns),
          f'every turn overridden from a savegame ({collector.truth_source.get("turns_overridden")}/{len(turns)})')
    mism = collector.truth_source.get('cities_crosscheck_mismatches') or []
    cells = sum(len(ts['cities_count'][t]) for t in turns)
    print(f'  INFO  city table vs [scoreN].cities: {len(mism)}/{cells} cells differ '
          f'(server scores before turn-change founding/capture), e.g. {mism[:3]}')
    check(all(abs(m['city_table'] - m['score_cities']) <= 1 for m in mism),
          'every city-table vs score.cities mismatch is off by exactly one (timing, not parsing)')

    # Player 0 oracle: client values are exact for the recorder's own player.
    # population tolerates a documented client/server snapshot skew: the client
    # state and the autosave are taken at slightly different points of the turn
    # change, so a single growth/shrink can be visible in one and not the
    # other (rng5001: 1 of 152 turns, off by one).
    for series in ORACLE_SERIES:
        bad = []
        for t in turns:
            c = client_value(ts_client, series, t, 0)
            s = ts[series][t].get(0)
            if c is None or s is None or float(c) != float(s):
                bad.append((t, c, s))
        if series == 'population':
            ok = (len(bad) <= max(1, len(turns) // 100)
                  and all(c is not None and s is not None and abs(float(c) - float(s)) <= 1
                          for _, c, s in bad))
            check(ok, f'player 0 population: savegame == client on {len(turns) - len(bad)}/{len(turns)} '
                  f'turns, remaining cells off by <= 1 (turn, client, savegame) {bad}')
        else:
            check(not bad, f'player 0 {series}: savegame == client on all {len(turns)} turns'
                  + (f' (mismatches e.g. {bad[:4]})' if bad else ''))

    # Territory is reported, not asserted (client counts from the visible tile
    # map, the savegame from the [map] owner rows; document the gap)
    terr_bad = []
    for t in turns:
        c = client_value(ts_client, 'territory_size', t, 0)
        s = ts['territory_size'][t].get(0)
        if c is None or s is None or int(c) != int(s):
            terr_bad.append((t, c, s))
    print(f'  INFO  player 0 territory_size client != savegame on {len(terr_bad)}/{len(turns)} turns'
          + (f', e.g. (turn, client, savegame) {terr_bad[:5]}' if terr_bad else ''))
    if terr_bad:
        ratios = [s / c for _, c, s in terr_bad if c]
        print(f'        savegame/client ratio min {min(ratios):.3f} max {max(ratios):.3f}')

    # Every other player: scores agree wherever the client had intelligence
    score_bad, score_checked = [], 0
    for t in turns:
        for pid_s, c in ts_client['scores'][str(t)].items():
            pid = int(pid_s)
            if c == -1:
                continue
            score_checked += 1
            s = ts['scores'][t].get(pid)
            if s is None or int(s) != int(c):
                score_bad.append((t, pid, c, s))
    check(not score_bad, f'scores equal wherever client != -1 ({score_checked} cells checked)'
          + (f', mismatches e.g. {score_bad[:4]}' if score_bad else ''))

    # Savegame cities vs client cities. The savegame can only be BELOW the
    # client where the client holds a stale fog-of-war ownership record (a city
    # that changed hands or was destroyed out of the observer's sight keeps its
    # last-seen owner; rng5001: pirate cities Tavoy/Muse/Thaton vanish from the
    # savegame at T94 while the client keeps listing 2 pirate cities). Player 0
    # can never be stale, so the >= rule is asserted for player 0 and the
    # aggregate, and reported per player for the rest.
    under, cells, gained = [], 0, 0
    under_by_pid: Dict[int, int] = {}
    for t in turns:
        for pid_s, c in ts_client['cities_count'][str(t)].items():
            pid = int(pid_s)
            s = ts['cities_count'][t].get(pid)
            cells += 1
            if s is None or s < c:
                under.append((t, pid, c, s))
                under_by_pid[pid] = under_by_pid.get(pid, 0) + 1
            elif s > c:
                gained += 1
        # a player listed by the savegame but never by the client this turn
        for pid in ts['cities_count'][t]:
            if str(pid) not in ts_client['cities_count'][str(t)]:
                gained += 1
    check(not [u for u in under if u[1] == 0],
          f'player 0 never has fewer savegame cities than client cities ({cells} cells)')
    check(gained > len(under),
          f'savegame cities > client on {gained} cells vs < on {len(under)} cells (stale client records)')
    print(f'  INFO  savegame < client cities per player {under_by_pid}, e.g. (turn,pid,client,savegame) {under[:3]}')

    neg = [(t, pid) for t in turns for pid, v in ts['scores'][t].items() if v == -1]
    check(not neg, f'no -1 scores remain in the regenerated series ({len(neg)} found)')
    neg_client = sum(1 for t in turns for v in ts_client['scores'][str(t)].values() if v == -1)
    print(f'  INFO  client data had {neg_client} score==-1 cells')

    for name in ('is_alive', 'government'):
        empty = [t for t in turns if not ts[name][t]]
        check(not empty, f'time_series.{name} populated on every turn')
    print(f'  INFO  T211 is_alive={ts["is_alive"][211]} government={ts["government"][211]}')

    return ts, collector, states, config, turns, civilizations


def part_c(client: dict, ts, collector, states, config, turns) -> None:
    print('\n== (c) observability gate ==')
    orig_path = os.path.join(RNG5001_DIR, f'{RNG5001_USER}_data.json.gz')
    orig = observability_gate.gate_data(client)
    check(not orig['ok'], f'gate FAILS on the original {os.path.basename(orig_path)}')
    for r in orig['reasons']:
        print(f'          - {r}')
    rc = observability_gate.main([orig_path, '--quiet'])
    check(rc == 1, f'gate CLI exit code on original file == 1 (got {rc})')

    snapshots = collector.collect_snapshots(states, [109, 160, 211], config)
    cov = sp.compute_savegame_coverage(RNG5001_DIR, turns)
    regenerated = {
        'metadata': {
            'turn': turns[-1], 'turns_analyzed': turns, 'username': RNG5001_USER,
            'truth_source': dict(collector.truth_source),
            'savegame_coverage': cov['coverage'],
        },
        'civilizations': client['civilizations'],
        'time_series': ts,
        'events': [],
        'snapshots': snapshots,
    }
    for turn, snap in snapshots.items():
        check(snap.get('truth_source') == TRUTH_SOURCE_SAVEGAME and -1 not in snap['scores'].values(),
              f'snapshot T{turn} overridden from savegame, scores={snap["scores"]}')
    regen = observability_gate.gate_data(regenerated)
    check(regen['ok'], 'gate PASSES on the regenerated series (data-only checks)')
    for r in regen['reasons']:
        print(f'          - {r}')
    regen_full = observability_gate.gate_data(regenerated, recording_dir=RNG5001_DIR, username=RNG5001_USER)
    check(regen_full['ok'], f'gate PASSES on the regenerated series with savegame re-check: {regen_full["stats"]}')
    for r in regen_full['reasons']:
        print(f'          - {r}')


def part_d() -> None:
    print('\n== (d) rng5418 checkpoint-fork spot check (report only) ==')
    data_path = os.path.join(RNG5418_DIR, f'{RNG5418_USER}_data.json.gz')
    if not os.path.isfile(data_path):
        print('  skipped (no rng5418 data)')
        return
    client = load_client_data(RNG5418_DIR, RNG5418_USER)
    saves = glob.glob(os.path.join(RNG5418_DIR, 'savegames', f'{RNG5418_USER}_T*'))
    turns_with_save = sorted({int(os.path.basename(p).split('_T')[1].split('_')[0]) for p in saves})
    print(f'  savegame turns: {turns_with_save}')
    ts = client['time_series']
    for t in turns_with_save:
        if str(t) not in ts['scores']:
            continue
        sg = sp.extract_complete_data_from_savegame(RNG5418_USER, t, recording_dir=RNG5418_DIR)
        if not sg:
            print(f'  T{t}: savegame unparseable')
            continue
        s = savegame_row(sg, 0)
        diffs = {k: (client_value(ts, k, t, 0), s[k]) for k in ORACLE_SERIES
                 if client_value(ts, k, t, 0) is None or float(client_value(ts, k, t, 0)) != float(s[k])}
        print(f'  T{t} player 0: ' + ('all oracle series agree' if not diffs else f'client/savegame diffs {diffs}'))
    gate = observability_gate.gate_data(client)
    print(f'  gate on rng5418 original: {"PASS" if gate["ok"] else "FLAGGED"} ({len(gate["reasons"])} reasons)')


def main() -> int:
    if not os.path.isdir(os.path.join(RNG5001_DIR, 'savegames')):
        print(f'ERROR: test data missing: {RNG5001_DIR}/savegames (set FBSIM_TEST_RNG5001_DIR)')
        return 2
    client = load_client_data(RNG5001_DIR, RNG5001_USER)
    part_a(client)
    ts, collector, states, config, turns, _ = part_b(client)
    part_c(client, ts, collector, states, config, turns)
    part_d()
    print()
    if _failures:
        print(f'{len(_failures)} CHECK(S) FAILED:')
        for f in _failures:
            print(f'  - {f}')
        return 1
    print('ALL CHECKS PASSED')
    return 0


if __name__ == '__main__':
    sys.exit(main())
