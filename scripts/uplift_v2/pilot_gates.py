#!/usr/bin/env python3
"""Pilot gate-checkers for the fixed FreeCiv mining/fork bundle.

Given a directory of pilot outputs, computes PASS/FAIL for every P1 (re-mined
worlds) and P2 (fork exchangeability) gate from the root-cause fix plan, and
emits a machine-readable JSON verdict plus a human-readable table.

Expected layout (all pieces optional; gates whose inputs are absent report
MISSING rather than guessing):

  P1 (--p1-dir):
      <p1-dir>/seed{N}_data.json[.gz]     re-mined world serializations
      <p1-dir>/**/*.log                   lane/driver logs (zstd-traceback scan)

  P2 (--p2-dir, --anchor, --anchor-savegames):
      <p2-dir>/<fork_id>/*_data.json[.gz]         fork serialization
      <p2-dir>/<fork_id>/savegames/*.sav.[xz|zst] fork autosaves
      (a flat <p2-dir>/<fork_id>_data.json.gz layout also works, with
       savegames under <p2-dir>/savegames/<fork_id>/)
      --anchor            anchor world data JSON (healthy pod-mined world)
      --anchor-savegames  dir with the anchor's T{fork-turn}/T{window-end} saves
      --pairs             JSON: {"same_seed": [[a,b],...], "diff_seed": [[a,b],...]}
      --anchor-bank       JSON: {qid: {"obs": 0|1, "p_mc": float}, ...}

P1 gates: 10/10 frozen=0; wonder>=1/world with median in [7,25]; tech events
400-600/world covering all reference-turn pids; diplomacy pairs>=10;
savegame_coverage==1.0 stamped; no zstd tracebacks in logs.

P2 gates (fork turn t60, window end t90 by default): frozen 0/5 in >=98% of
forks; t60 research state identical to anchor in 100%; t60 diplomacy identical
in 100%; >=30% of forks witness a wonder in-window; p0 next-tech modal
frequency <=0.95; same-seed pairs dynamics-identical at t70 while diff-seed
pairs diverge; goal_name != A_UNSET in 100%; anchor t90 techs inside the fork
ensemble min-max in >=92% of player-cells (23/25); pooled |obs-p_mc| < 0.05
over the anchor bank; [settings] block identical to the anchor T60 save
(gameseed/metamessage rows excluded — the per-fork reseed rewrites gameseed by
design).

Usage:
    uv run python scripts/uplift_v2/pilot_gates.py \
        --p1-dir tmp/pilot/p1_worlds \
        --p2-dir tmp/pilot/p2_forks --anchor tmp/pilot/anchor_data.json.gz \
        --anchor-savegames tmp/pilot/anchor_savegames \
        --pairs tmp/pilot/pairs.json --anchor-bank tmp/pilot/anchor_bank.json \
        --out tmp/pilot/pilot_gates_verdict.json

Exit codes: 0 = every evaluated gate PASS and none MISSING; 1 = any FAIL;
3 = no FAIL but some gates MISSING inputs.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import lzma
import os
import re
import statistics
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

PASS, FAIL, MISSING = 'PASS', 'FAIL', 'MISSING'


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def read_savegame_text(path: str) -> str:
    """Decompress a .sav[.xz|.zst] file to text (stdlib/zstandard only)."""
    with open(path, 'rb') as f:
        raw = f.read()
    if path.endswith('.zst'):
        import io
        import zstandard
        dctx = zstandard.ZstdDecompressor()
        with dctx.stream_reader(io.BytesIO(raw)) as reader:
            return reader.read().decode('utf-8')
    if path.endswith('.xz'):
        return lzma.decompress(raw).decode('utf-8')
    return raw.decode('utf-8')


def savegame_turn_from_name(path: str) -> Optional[int]:
    m = re.search(r'_T(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def find_savegame_for_turn(directory: str, turn: int,
                           slack: int = 1) -> Optional[str]:
    """Find a savegame labeled T{turn} (or up to T{turn+slack}) in a dir."""
    candidates = []
    for path in glob.glob(os.path.join(directory, '**', '*.sav*'),
                          recursive=True):
        t = savegame_turn_from_name(path)
        if t is not None and turn <= t <= turn + slack:
            candidates.append((t, path))
    if not candidates:
        return None
    return min(candidates)[1]


# ---------------------------------------------------------------------------
# Savegame section parsers (self-contained; formats per Freeciv 3.x saves)
# ---------------------------------------------------------------------------

def extract_section(content: str, name: str) -> Optional[str]:
    start = content.find(f'[{name}]')
    if start == -1:
        return None
    end = content.find('\n[', start + 1)
    return content[start:] if end == -1 else content[start:end]


RESEARCH_ROW = re.compile(r'^(\d+),"([^"]*)",(\d+),.*,"([01]+)"\s*$',
                          re.MULTILINE)


def parse_research_rows(content: str) -> Dict[int, dict]:
    """Parse [research] rows -> {pid: {goal, techs, done}}."""
    section = extract_section(content, 'research')
    if section is None:
        return {}
    rows = {}
    for m in RESEARCH_ROW.finditer(section):
        pid, goal, techs, done = m.groups()
        rows[int(pid)] = {'goal': goal, 'techs': int(techs), 'done': done}
    return rows


def parse_diplstates(content: str) -> Dict[int, List[str]]:
    """Parse each [playerN] diplstate table -> {pid: [current states]}."""
    out: Dict[int, List[str]] = {}
    for m in re.finditer(r'\[player(\d+)\]', content):
        pid = int(m.group(1))
        section_end = content.find('\n[', m.end())
        section = content[m.end(): section_end if section_end != -1 else None]
        dm = re.search(r'diplstate=\{(.*?)\n\}', section, re.DOTALL)
        if not dm:
            continue
        lines = [ln for ln in dm.group(1).split('\n') if ln.strip()]
        states = []
        # First line is the header row ("current","closest",...)
        for line in lines[1:]:
            first = line.split(',')[0].strip()
            states.append(first.strip('"'))
        out[pid] = states
    return out


def parse_settings_rows(content: str,
                        exclude: Tuple[str, ...] = ()) -> List[str]:
    """Parse [settings] set-table rows, optionally excluding named settings."""
    section = extract_section(content, 'settings')
    if section is None:
        return []
    rows = []
    for line in section.split('\n'):
        line = line.strip()
        if not line.startswith('"'):
            continue
        name = line.split(',')[0].strip('"')
        if name in exclude:
            continue
        rows.append(line)
    return rows


def dynamics_fingerprint(content: str) -> str:
    """A fork-identity-independent fingerprint of game dynamics state.

    Combines the [random] RNG block, all [research] rows, and per-player
    gold= lines. Two rollouts with identical dynamics through this turn have
    identical fingerprints even though their savegames differ in fork
    username/timestamp fields; divergent dynamics differ here with
    overwhelming probability.
    """
    random_block = extract_section(content, 'random') or ''
    research = json.dumps(parse_research_rows(content), sort_keys=True)
    gold = ','.join(re.findall(r'^gold=\d+', content, re.MULTILINE))
    import hashlib
    return hashlib.sha256(
        (random_block + research + gold).encode('utf-8')).hexdigest()


# ---------------------------------------------------------------------------
# World-data statistics (shared with world_publish_gate definitions)
# ---------------------------------------------------------------------------

def world_stats(data: dict, reference_turn: int = 60) -> dict:
    events = data.get('events', [])
    techs_known = data.get('time_series', {}).get('techs_known', {})
    turns = sorted(int(t) for t in techs_known.keys())
    if not turns:
        return {'ok': False}

    def row(turn):
        return techs_known.get(turn, techs_known.get(str(turn)))

    ref = reference_turn if row(reference_turn) is not None else turns[0]
    civ_ids = sorted(int(p) for p in row(ref).keys())
    tech_events = [e for e in events if e.get('type') == 'tech_discovered']
    tech_pids = {e.get('player_id') for e in tech_events}
    frozen = [p for p in civ_ids if p not in tech_pids]
    wonders = [e for e in events if e.get('type') == 'wonder_completed']
    relations = (data.get('diplomacy') or {}).get('relations') or {}
    return {
        'ok': True,
        'reference_turn': ref,
        'civ_ids': civ_ids,
        'frozen': len(frozen),
        'frozen_ids': frozen,
        'n_tech_events': len(tech_events),
        'tech_pids': sorted(p for p in tech_pids if p is not None),
        'n_wonders': len(wonders),
        'wonder_turns': sorted(e.get('turn', -1) for e in wonders),
        'diplomacy_pairs': len(relations),
        'coverage': data.get('metadata', {}).get('savegame_coverage'),
        'max_turn': turns[-1],
    }


def gate(name: str, status: str, value, threshold: str, detail: str = '') -> dict:
    return {'gate': name, 'status': status, 'value': value,
            'threshold': threshold, 'detail': detail}


# ---------------------------------------------------------------------------
# P1 gates
# ---------------------------------------------------------------------------

def run_p1_gates(p1_dir: Optional[str], expected_worlds: int,
                 reference_turn: int) -> List[dict]:
    gates: List[dict] = []
    if not p1_dir or not os.path.isdir(p1_dir):
        for g in ('p1_frozen', 'p1_wonders', 'p1_tech_events',
                  'p1_diplomacy', 'p1_coverage', 'p1_no_zstd_tracebacks'):
            gates.append(gate(g, MISSING, None, '-', f'--p1-dir not found: {p1_dir}'))
        return gates

    files = sorted(glob.glob(os.path.join(p1_dir, '*_data.json.gz'))
                   + glob.glob(os.path.join(p1_dir, '*_data.json')))
    stats = {}
    for path in files:
        wid = os.path.basename(path).split('_data.json')[0]
        try:
            stats[wid] = world_stats(load_json(path), reference_turn)
        except Exception as e:
            stats[wid] = {'ok': False, 'error': repr(e)}

    n = len(stats)
    ok_stats = {w: s for w, s in stats.items() if s.get('ok')}

    if n == 0:
        for g in ('p1_frozen', 'p1_wonders', 'p1_tech_events',
                  'p1_diplomacy', 'p1_coverage'):
            gates.append(gate(g, MISSING, None, '-',
                              f'no world data files in {p1_dir}'))
    else:
        # Gate: 10/10 frozen=0
        frozen_ok = [w for w, s in ok_stats.items() if s['frozen'] == 0]
        status = PASS if (len(frozen_ok) == n and n >= expected_worlds) else FAIL
        bad = {w: s.get('frozen_ids', '?') for w, s in stats.items()
               if not s.get('ok') or s.get('frozen', 1) != 0}
        gates.append(gate(
            'p1_frozen', status, f'{len(frozen_ok)}/{n}',
            f'{expected_worlds}/{expected_worlds} worlds with frozen=0',
            f'violations: {bad}' if bad else ''))

        # Gate: wonder>=1 per world, median across worlds in [7,25]
        wcounts = [s['n_wonders'] for s in ok_stats.values()]
        med = statistics.median(wcounts) if wcounts else 0
        wonder_all = all(c >= 1 for c in wcounts) and len(wcounts) == n
        status = PASS if (wonder_all and 7 <= med <= 25) else FAIL
        gates.append(gate(
            'p1_wonders', status,
            f'min={min(wcounts) if wcounts else 0}, median={med}',
            '>=1/world and median in [7,25]',
            f'per-world: { {w: s["n_wonders"] for w, s in ok_stats.items()} }'))

        # Gate: tech events 400-600/world covering all reference-turn pids
        tech_ok, tech_bad = [], {}
        for w, s in ok_stats.items():
            covers = set(s['civ_ids']).issubset(set(s['tech_pids']))
            in_range = 400 <= s['n_tech_events'] <= 600
            if covers and in_range:
                tech_ok.append(w)
            else:
                tech_bad[w] = {'n': s['n_tech_events'],
                               'missing_pids': sorted(
                                   set(s['civ_ids']) - set(s['tech_pids']))}
        status = PASS if len(tech_ok) == n else FAIL
        gates.append(gate(
            'p1_tech_events', status, f'{len(tech_ok)}/{n}',
            '400-600 tech events/world covering all reference-turn pids',
            f'violations: {tech_bad}' if tech_bad else ''))

        # Gate: diplomacy pairs >= 10 per world
        dip_bad = {w: s['diplomacy_pairs'] for w, s in ok_stats.items()
                   if s['diplomacy_pairs'] < 10}
        status = PASS if (not dip_bad and len(ok_stats) == n) else FAIL
        gates.append(gate(
            'p1_diplomacy', status,
            f'{n - len(dip_bad)}/{n}', '>=10 diplomacy pairs per world',
            f'violations: {dip_bad}' if dip_bad else ''))

        # Gate: savegame_coverage == 1.0 stamped
        cov_bad = {w: s['coverage'] for w, s in ok_stats.items()
                   if s['coverage'] != 1.0}
        status = PASS if (not cov_bad and len(ok_stats) == n) else FAIL
        gates.append(gate(
            'p1_coverage', status, f'{n - len(cov_bad)}/{n}',
            'savegame_coverage==1.0 stamped in every world',
            f'violations: {cov_bad}' if cov_bad else ''))

    # Gate: no zstd tracebacks anywhere in the pilot logs
    log_files = glob.glob(os.path.join(p1_dir, '**', '*.log'), recursive=True)
    hits = []
    for lf in log_files:
        try:
            with open(lf, errors='replace') as f:
                text = f.read()
        except OSError:
            continue
        if re.search(r"(FileNotFoundError|Traceback)[^\n]*\n(.*\n){0,20}?[^\n]*'?zstd'?",
                     text) or "No such file or directory: 'zstd'" in text:
            hits.append(lf)
    gates.append(gate(
        'p1_no_zstd_tracebacks', PASS if not hits else FAIL,
        f'{len(hits)} log(s) with zstd tracebacks '
        f'({len(log_files)} scanned)',
        '0 zstd tracebacks', f'hits: {hits[:5]}' if hits else ''))

    return gates


# ---------------------------------------------------------------------------
# P2 helpers
# ---------------------------------------------------------------------------

def discover_forks(p2_dir: str) -> Dict[str, dict]:
    """Map fork_id -> {'data': path|None, 'savegames': dir|None}."""
    forks: Dict[str, dict] = {}
    if not os.path.isdir(p2_dir):
        return forks
    for entry in sorted(os.listdir(p2_dir)):
        full = os.path.join(p2_dir, entry)
        if os.path.isdir(full) and entry != 'savegames':
            data_files = (glob.glob(os.path.join(full, '*_data.json.gz'))
                          + glob.glob(os.path.join(full, '*_data.json')))
            sav_dir = None
            for cand in (os.path.join(full, 'savegames'), full):
                if glob.glob(os.path.join(cand, '*.sav*')):
                    sav_dir = cand
                    break
            forks[entry] = {'data': data_files[0] if data_files else None,
                            'savegames': sav_dir}
        elif entry.endswith(('_data.json.gz', '_data.json')):
            fid = entry.split('_data.json')[0]
            forks.setdefault(fid, {'data': None, 'savegames': None})
            forks[fid]['data'] = full
    # flat savegames/<fork_id>/ layout
    flat_sav = os.path.join(p2_dir, 'savegames')
    if os.path.isdir(flat_sav):
        for fid in list(forks):
            cand = os.path.join(flat_sav, fid)
            if forks[fid]['savegames'] is None and os.path.isdir(cand):
                forks[fid]['savegames'] = cand
    return forks


def run_p2_gates(p2_dir: Optional[str], anchor_path: Optional[str],
                 anchor_sav_dir: Optional[str], pairs_path: Optional[str],
                 bank_path: Optional[str], fork_turn: int, window_end: int,
                 expected_forks: int) -> List[dict]:
    gates: List[dict] = []
    p2_names = ['p2_frozen', 'p2_t60_research_identical',
                'p2_t60_diplomacy_identical', 'p2_wonder_window',
                'p2_p0_next_tech_modal', 'p2_seed_pairs',
                'p2_goal_not_unset', 'p2_anchor_in_ensemble',
                'p2_calibration', 'p2_settings_identical']

    if not p2_dir or not os.path.isdir(p2_dir):
        for g in p2_names:
            gates.append(gate(g, MISSING, None, '-',
                              f'--p2-dir not found: {p2_dir}'))
        return gates

    forks = discover_forks(p2_dir)
    n_forks = len(forks)
    if n_forks == 0:
        for g in p2_names:
            gates.append(gate(g, MISSING, None, '-',
                              f'no fork outputs found in {p2_dir}'))
        return gates

    # Load fork serializations + world stats
    fork_stats: Dict[str, dict] = {}
    fork_data: Dict[str, dict] = {}
    for fid, paths in forks.items():
        if paths['data']:
            try:
                fork_data[fid] = load_json(paths['data'])
                fork_stats[fid] = world_stats(fork_data[fid], fork_turn)
            except Exception as e:
                fork_stats[fid] = {'ok': False, 'error': repr(e)}

    # Anchor artifacts
    anchor = load_json(anchor_path) if (
        anchor_path and os.path.exists(anchor_path)) else None
    anchor_t60_save = (find_savegame_for_turn(anchor_sav_dir, fork_turn)
                       if anchor_sav_dir and os.path.isdir(anchor_sav_dir)
                       else None)
    anchor_t90_save = (find_savegame_for_turn(anchor_sav_dir, window_end)
                       if anchor_sav_dir and os.path.isdir(anchor_sav_dir)
                       else None)
    anchor_t60_text = read_savegame_text(anchor_t60_save) if anchor_t60_save else None
    anchor_t90_text = read_savegame_text(anchor_t90_save) if anchor_t90_save else None

    # --- Gate: frozen 0/5 in >=98% of forks -------------------------------
    with_data = [fid for fid in forks if fork_stats.get(fid, {}).get('ok')]
    if with_data:
        frozen_ok = [fid for fid in with_data if fork_stats[fid]['frozen'] == 0]
        frac = len(frozen_ok) / len(with_data)
        status = PASS if (frac >= 0.98 and len(with_data) >= expected_forks) else FAIL
        gates.append(gate(
            'p2_frozen', status,
            f'{len(frozen_ok)}/{len(with_data)} forks frozen=0 ({frac:.1%})',
            f'>=98% of >={expected_forks} forks',
            f"violators: {[f for f in with_data if fork_stats[f]['frozen'] != 0][:10]}"))
    else:
        gates.append(gate('p2_frozen', MISSING, None, '>=98% forks frozen=0',
                          'no fork serializations found'))

    # --- Savegame-based gates ---------------------------------------------
    fork_first_saves: Dict[str, str] = {}
    fork_t70_saves: Dict[str, str] = {}
    fork_t90_saves: Dict[str, str] = {}
    for fid, paths in forks.items():
        if paths['savegames']:
            s60 = find_savegame_for_turn(paths['savegames'], fork_turn)
            if s60:
                fork_first_saves[fid] = s60
            s70 = find_savegame_for_turn(paths['savegames'], 70)
            if s70:
                fork_t70_saves[fid] = s70
            s90 = find_savegame_for_turn(paths['savegames'], window_end)
            if s90:
                fork_t90_saves[fid] = s90

    # Gate: t60 research state identical to anchor (100%)
    if anchor_t60_text and fork_first_saves:
        anchor_rows = parse_research_rows(anchor_t60_text)
        anchor_done = {p: r['done'] for p, r in anchor_rows.items()}
        mismatches = []
        for fid, sav in sorted(fork_first_saves.items()):
            rows = parse_research_rows(read_savegame_text(sav))
            done = {p: r['done'] for p, r in rows.items()}
            if done != anchor_done:
                mismatches.append(fid)
        n_cmp = len(fork_first_saves)
        status = PASS if (not mismatches and n_cmp >= expected_forks) else FAIL
        gates.append(gate(
            'p2_t60_research_identical', status,
            f'{n_cmp - len(mismatches)}/{n_cmp} identical',
            f'100% of >={expected_forks} forks',
            f'mismatches: {mismatches[:10]}'))
    else:
        gates.append(gate(
            'p2_t60_research_identical', MISSING, None, '100% identical',
            'needs anchor T60 savegame (--anchor-savegames) and fork savegames'))

    # Gate: t60 diplomacy identical to anchor (100%)
    if anchor_t60_text and fork_first_saves:
        anchor_dipl = parse_diplstates(anchor_t60_text)
        mismatches = []
        for fid, sav in sorted(fork_first_saves.items()):
            if parse_diplstates(read_savegame_text(sav)) != anchor_dipl:
                mismatches.append(fid)
        n_cmp = len(fork_first_saves)
        status = PASS if (not mismatches and n_cmp >= expected_forks) else FAIL
        gates.append(gate(
            'p2_t60_diplomacy_identical', status,
            f'{n_cmp - len(mismatches)}/{n_cmp} identical',
            f'100% of >={expected_forks} forks',
            f'mismatches: {mismatches[:10]}'))
    else:
        gates.append(gate(
            'p2_t60_diplomacy_identical', MISSING, None, '100% identical',
            'needs anchor T60 savegame and fork savegames'))

    # --- Gate: >=30% of forks witness >=1 wonder in-window ----------------
    if with_data:
        witnessed = []
        for fid in with_data:
            wonder_turns = fork_stats[fid].get('wonder_turns', [])
            if any(fork_turn < t <= window_end for t in wonder_turns):
                witnessed.append(fid)
        frac = len(witnessed) / len(with_data)
        gates.append(gate(
            'p2_wonder_window', PASS if frac >= 0.30 else FAIL,
            f'{len(witnessed)}/{len(with_data)} forks ({frac:.1%})',
            f'>=30% witness a wonder in ({fork_turn},{window_end}]', ''))
    else:
        gates.append(gate('p2_wonder_window', MISSING, None,
                          '>=30% witness a wonder in-window', 'no fork data'))

    # --- Gate: p0 next-tech modal frequency <= 0.95 -----------------------
    if with_data:
        firsts = []
        for fid in with_data:
            evs = [e for e in fork_data[fid].get('events', [])
                   if e.get('type') == 'tech_discovered'
                   and e.get('player_id') == 0
                   and e.get('turn', 0) > fork_turn]
            if evs:
                first = min(evs, key=lambda e: e.get('turn', 10**9))
                name = (first.get('metadata') or {}).get(
                    'tech_name', first.get('description', '?'))
                firsts.append(name)
        if firsts:
            counts = Counter(firsts)
            modal, modal_n = counts.most_common(1)[0]
            frac = modal_n / len(firsts)
            gates.append(gate(
                'p2_p0_next_tech_modal', PASS if frac <= 0.95 else FAIL,
                f'modal={modal!r} at {modal_n}/{len(firsts)} ({frac:.1%})',
                'modal frequency <= 0.95',
                f'distribution: {dict(counts.most_common(5))}'))
        else:
            gates.append(gate(
                'p2_p0_next_tech_modal', FAIL, 'no p0 post-fork tech events',
                'modal frequency <= 0.95',
                'no fork produced a player-0 tech discovery in-window'))
    else:
        gates.append(gate('p2_p0_next_tech_modal', MISSING, None,
                          'modal frequency <= 0.95', 'no fork data'))

    # --- Gate: same-seed pairs identical at t70, diff-seed diverge --------
    pairs = None
    if pairs_path and os.path.exists(pairs_path):
        pairs = load_json(pairs_path)
    if pairs and fork_t70_saves:
        def fp(fid):
            sav = fork_t70_saves.get(fid)
            return dynamics_fingerprint(read_savegame_text(sav)) if sav else None

        same_bad, same_skipped = [], []
        for a, b in pairs.get('same_seed', []):
            fa, fb = fp(a), fp(b)
            if fa is None or fb is None:
                same_skipped.append((a, b))
            elif fa != fb:
                same_bad.append((a, b))
        diff_bad, diff_skipped = [], []
        for a, b in pairs.get('diff_seed', []):
            fa, fb = fp(a), fp(b)
            if fa is None or fb is None:
                diff_skipped.append((a, b))
            elif fa == fb:
                diff_bad.append((a, b))
        n_pairs = len(pairs.get('same_seed', [])) + len(pairs.get('diff_seed', []))
        evaluated = n_pairs - len(same_skipped) - len(diff_skipped)
        if evaluated == 0:
            gates.append(gate('p2_seed_pairs', MISSING, None,
                              'same-seed identical @t70, diff-seed diverge',
                              'no T70 savegames found for listed pairs'))
        else:
            status = PASS if not (same_bad or diff_bad or same_skipped
                                  or diff_skipped) else FAIL
            gates.append(gate(
                'p2_seed_pairs', status,
                f'{evaluated}/{n_pairs} pairs evaluated; '
                f'{len(same_bad)} same-seed diverged, '
                f'{len(diff_bad)} diff-seed identical',
                'same-seed dynamics-identical @t70; diff-seed diverged',
                f'same_bad={same_bad} diff_bad={diff_bad} '
                f'skipped={same_skipped + diff_skipped}'))
    else:
        gates.append(gate(
            'p2_seed_pairs', MISSING, None,
            'same-seed identical @t70, diff-seed diverge',
            'needs --pairs JSON and fork T70 savegames'))

    # --- Gate: goal_name != A_UNSET in 100% of forks ----------------------
    if fork_first_saves:
        unset = []
        for fid, sav in sorted(fork_first_saves.items()):
            rows = parse_research_rows(read_savegame_text(sav))
            goal = rows.get(0, {}).get('goal')
            if goal is None or goal == 'A_UNSET':
                unset.append((fid, goal))
        n_cmp = len(fork_first_saves)
        status = PASS if (not unset and n_cmp >= expected_forks) else FAIL
        gates.append(gate(
            'p2_goal_not_unset', status,
            f'{n_cmp - len(unset)}/{n_cmp} with a live research goal',
            f"goal_name != 'A_UNSET' in 100% of >={expected_forks} forks",
            f'violations: {unset[:10]}'))
    else:
        gates.append(gate(
            'p2_goal_not_unset', MISSING, None, "goal_name != 'A_UNSET' 100%",
            'no fork savegames at the fork turn found'))

    # --- Gate: anchor t90 techs inside fork min-max (>=92% of cells) ------
    if anchor_t90_text and fork_t90_saves:
        anchor_techs = {p: r['techs']
                        for p, r in parse_research_rows(anchor_t90_text).items()}
        ensemble: Dict[int, List[int]] = {}
        for fid, sav in fork_t90_saves.items():
            for p, r in parse_research_rows(read_savegame_text(sav)).items():
                ensemble.setdefault(p, []).append(r['techs'])
        cells, inside = 0, 0
        outside = {}
        for p, a in sorted(anchor_techs.items()):
            vals = ensemble.get(p)
            if not vals:
                continue
            cells += 1
            if min(vals) <= a <= max(vals):
                inside += 1
            else:
                outside[p] = {'anchor': a, 'min': min(vals), 'max': max(vals)}
        frac = inside / cells if cells else 0.0
        status = PASS if (cells > 0 and frac >= 23 / 25) else FAIL
        gates.append(gate(
            'p2_anchor_in_ensemble', status,
            f'{inside}/{cells} player-cells inside ({frac:.1%})',
            '>=92% of cells (23/25) contain the anchor t90 tech count',
            f'outside: {outside}' if outside else ''))
    else:
        gates.append(gate(
            'p2_anchor_in_ensemble', MISSING, None,
            '>=92% of player-cells inside fork min-max',
            'needs anchor T90 savegame and fork T90 savegames'))

    # --- Gate: pooled |obs - p_mc| < 0.05 over the anchor bank ------------
    if bank_path and os.path.exists(bank_path):
        bank = load_json(bank_path)
        items = bank.items() if isinstance(bank, dict) else enumerate(bank)
        obs, pmc = [], []
        for _, rec in items:
            if rec.get('obs') is None or rec.get('p_mc') is None:
                continue
            obs.append(float(rec['obs']))
            pmc.append(float(rec['p_mc']))
        if obs:
            pooled = abs(sum(obs) / len(obs) - sum(pmc) / len(pmc))
            mae = sum(abs(o - p) for o, p in zip(obs, pmc)) / len(obs)
            gates.append(gate(
                'p2_calibration', PASS if pooled < 0.05 else FAIL,
                f'pooled |mean(obs)-mean(p_mc)|={pooled:.4f} over {len(obs)} qs',
                'pooled |obs - p_mc| < 0.05',
                f'per-question MAE={mae:.4f}'))
        else:
            gates.append(gate('p2_calibration', MISSING, None,
                              'pooled |obs - p_mc| < 0.05',
                              f'no resolvable (obs, p_mc) pairs in {bank_path}'))
    else:
        gates.append(gate('p2_calibration', MISSING, None,
                          'pooled |obs - p_mc| < 0.05',
                          'needs --anchor-bank JSON with obs and p_mc'))

    # --- Gate: [settings] identical to anchor T60 -------------------------
    # gameseed is excluded: the default per-fork reseed zeroes it by design.
    # metamessage is excluded: it embeds the host username/port.
    exclude = ('gameseed', 'metamessage')
    if anchor_t60_text and fork_first_saves:
        anchor_settings = parse_settings_rows(anchor_t60_text, exclude)
        mismatches = {}
        for fid, sav in sorted(fork_first_saves.items()):
            rows = parse_settings_rows(read_savegame_text(sav), exclude)
            if rows != anchor_settings:
                diff = (set(rows) ^ set(anchor_settings))
                mismatches[fid] = sorted(diff)[:6]
        n_cmp = len(fork_first_saves)
        status = PASS if (not mismatches and n_cmp >= expected_forks) else FAIL
        gates.append(gate(
            'p2_settings_identical', status,
            f'{n_cmp - len(mismatches)}/{n_cmp} identical',
            f'[settings] identical to anchor T60 (excl. {exclude})',
            f'mismatches: {dict(list(mismatches.items())[:5])}'))
    else:
        gates.append(gate(
            'p2_settings_identical', MISSING, None,
            '[settings] identical to anchor T60',
            'needs anchor T60 savegame and fork first autosaves'))

    return gates


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_table(gates: List[dict]) -> None:
    name_w = max(len(g['gate']) for g in gates) + 2
    print(f"{'GATE':<{name_w}}{'STATUS':<9}VALUE / THRESHOLD")
    print('-' * (name_w + 60))
    for g in gates:
        val = g['value'] if g['value'] is not None else '-'
        print(f"{g['gate']:<{name_w}}{g['status']:<9}{val}  [{g['threshold']}]")
        if g['detail']:
            print(f"{'':<{name_w}}{'':<9}{g['detail'][:300]}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description='PASS/FAIL checker for every P1/P2 pilot gate')
    parser.add_argument('--p1-dir', default=None,
                        help='Directory of re-mined P1 world data JSONs (+logs)')
    parser.add_argument('--p1-expected-worlds', type=int, default=10)
    parser.add_argument('--p2-dir', default=None,
                        help='Directory of P2 fork outputs')
    parser.add_argument('--p2-expected-forks', type=int, default=100)
    parser.add_argument('--anchor', default=None,
                        help='Anchor world data JSON(.gz)')
    parser.add_argument('--anchor-savegames', default=None,
                        help='Directory holding anchor T60/T90 savegames')
    parser.add_argument('--pairs', default=None,
                        help='JSON with same_seed/diff_seed fork-id pairs')
    parser.add_argument('--anchor-bank', default=None,
                        help='JSON question bank with obs + p_mc per question')
    parser.add_argument('--fork-turn', type=int, default=60)
    parser.add_argument('--window-end', type=int, default=90)
    parser.add_argument('--reference-turn', type=int, default=60)
    parser.add_argument('--out', default=None,
                        help='Write machine-readable JSON verdict here')
    args = parser.parse_args(argv)

    gates = []
    gates += run_p1_gates(args.p1_dir, args.p1_expected_worlds,
                          args.reference_turn)
    gates += run_p2_gates(args.p2_dir, args.anchor, args.anchor_savegames,
                          args.pairs, args.anchor_bank, args.fork_turn,
                          args.window_end, args.p2_expected_forks)

    n_fail = sum(1 for g in gates if g['status'] == FAIL)
    n_missing = sum(1 for g in gates if g['status'] == MISSING)
    n_pass = sum(1 for g in gates if g['status'] == PASS)
    overall = ('FAIL' if n_fail else
               ('INCOMPLETE' if n_missing else 'PASS'))

    print_table(gates)
    print()
    print(f'OVERALL: {overall}  ({n_pass} pass, {n_fail} fail, '
          f'{n_missing} missing inputs)')

    if args.out:
        payload = {
            'overall': overall,
            'n_pass': n_pass, 'n_fail': n_fail, 'n_missing': n_missing,
            'inputs': {
                'p1_dir': args.p1_dir, 'p2_dir': args.p2_dir,
                'anchor': args.anchor,
                'anchor_savegames': args.anchor_savegames,
                'pairs': args.pairs, 'anchor_bank': args.anchor_bank,
                'fork_turn': args.fork_turn, 'window_end': args.window_end,
            },
            'gates': gates,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(payload, f, indent=1)
        print(f'Wrote verdict JSON: {args.out}')

    if n_fail:
        return 1
    if n_missing:
        return 3
    return 0


if __name__ == '__main__':
    sys.exit(main())
