"""P0 smoke tests for the savegame-blind / fork-dynamics fix bundle.

Covers:
  - grant_player_tech modifies the [research] row of a REAL T60 save (or raises)
  - a bogus fork modification type raises (strict mod loop)
  - .sav.zst decompresses via the python-zstandard fallback (no zstd binary)
  - set_rng_from_seed rewrites the [random] block; clear_rng_state defaults
  - set_player_name rewrites name= AND username=/ranked_username=
  - savegame coverage tracking + FBSIM_REQUIRE_SAVEGAME_COVERAGE semantics
  - per-username port-claim release (replacement for destructive Ports.clear)
  - publish-gate dry-run flags EXACTLY the 176 dark / passes the 124 healthy
"""
from __future__ import annotations

import glob
import json
import lzma
import os
import re
import subprocess
from pathlib import Path

import pytest

from conftest import ROOT, load_script

# ---------------------------------------------------------------------------
# Fixtures / data discovery
# ---------------------------------------------------------------------------

_MINED_CANDIDATES = [
    os.environ.get('FBSIM_MINED_WORLDS'),
    ROOT / 'tmp' / 'mined' / 'worlds',
    Path('/Users/jaeholee0404/civbench/tmp/mined/worlds'),
]
_AUDIT_CANDIDATES = [
    os.environ.get('FBSIM_COVERAGE_AUDIT'),
    ROOT / 'tmp' / 'tail_v2' / 'world_coverage_audit.json',
    Path('/Users/jaeholee0404/civbench/tmp/tail_v2/world_coverage_audit.json'),
]


def _first_existing(candidates):
    for p in candidates:
        if p and Path(p).exists():
            return Path(p)
    return None


MINED_WORLDS = _first_existing(_MINED_CANDIDATES)
AUDIT_JSON = _first_existing(_AUDIT_CANDIDATES)


def _find_t60_save():
    if MINED_WORLDS is None:
        return None
    matches = sorted(glob.glob(
        str(MINED_WORLDS / 'savegames' / 'seed1000' / '*T60*.sav.xz')))
    return matches[0] if matches else None


T60_SAVE = _find_t60_save()

needs_t60 = pytest.mark.skipif(
    T60_SAVE is None, reason='seed1000 T60 savegame not available')
needs_corpus = pytest.mark.skipif(
    MINED_WORLDS is None or AUDIT_JSON is None,
    reason='mined worlds corpus / coverage audit not available')


@pytest.fixture(scope='module')
def t60_modifier():
    from freeciv_world.forking.savegame_modifier import SavegameModifier
    return SavegameModifier(T60_SAVE)


def _fresh_modifier():
    from freeciv_world.forking.savegame_modifier import SavegameModifier
    return SavegameModifier(T60_SAVE)


# ---------------------------------------------------------------------------
# grant_player_tech on a real T60 save
# ---------------------------------------------------------------------------

@needs_t60
def test_grant_player_tech_modifies_research_row():
    m = _fresh_modifier()
    section = m.content[m.content.find('[research]'):]
    row = re.search(r'^0,"[^"]*",(\d+),.*,"([01]+)"\s*$', section, re.MULTILINE)
    assert row, 'expected a [research] row for player 0 in the real save'
    before_count, before_bits = int(row.group(1)), row.group(2)

    # Pick a tech the player does NOT know yet
    tech_id = before_bits.index('0')
    m.grant_player_tech(0, tech_id)

    section = m.content[m.content.find('[research]'):]
    row = re.search(r'^0,"[^"]*",(\d+),.*,"([01]+)"\s*$', section, re.MULTILINE)
    after_count, after_bits = int(row.group(1)), row.group(2)

    assert after_bits[tech_id] == '1'
    assert after_count == before_count + 1
    # Only that one bit changed
    diffs = [i for i, (a, b) in enumerate(zip(before_bits, after_bits)) if a != b]
    assert diffs == [tech_id]


@needs_t60
def test_grant_player_tech_already_known_is_noop():
    m = _fresh_modifier()
    before = m.content
    known_id = re.search(
        r'^0,"[^"]*",\d+,.*,"([01]+)"\s*$',
        m.content[m.content.find('[research]'):], re.MULTILINE).group(1).index('1')
    m.grant_player_tech(0, known_id)
    assert m.content == before


@needs_t60
def test_grant_player_tech_raises_on_missing_player():
    m = _fresh_modifier()
    with pytest.raises(ValueError, match=r'no \[research\] row matched'):
        m.grant_player_tech(99, 1)


def test_grant_player_tech_raises_when_no_tech_section_at_all():
    from freeciv_world.forking.savegame_modifier import SavegameModifier
    m = SavegameModifier.__new__(SavegameModifier)
    m.content = '[player0]\nname="X"\ngold=5\n'
    with pytest.raises(ValueError):
        m.grant_player_tech(0, 3)


# ---------------------------------------------------------------------------
# Strict fork mod loop
# ---------------------------------------------------------------------------

def test_bogus_mod_type_raises():
    from freeciv_world.forking.fork_manager import ForkManager

    class DummyModifier:
        def set_player_gold(self, *a):
            pass

        def clear_rng_state(self):
            pass

    with pytest.raises(ValueError, match='Unknown modification type'):
        ForkManager._apply_mods(DummyModifier(), [{'type': 'bogus_mod', 'x': 1}])


def test_rng_seed_mod_dispatches_and_suppresses_default_reseed():
    from freeciv_world.forking.fork_manager import ForkManager

    calls = []

    class DummyModifier:
        def set_rng_from_seed(self, seed):
            calls.append(('set', seed))

        def clear_rng_state(self):
            calls.append(('clear',))

    ForkManager._apply_mods(DummyModifier(), [{'type': 'rng_seed', 'value': 5001}])
    assert calls == [('set', 5001)]

    calls.clear()
    ForkManager._apply_mods(DummyModifier(), [])
    assert calls == [('clear',)], 'no rng_seed mod => default reseed'


# ---------------------------------------------------------------------------
# zstd fallback decompression
# ---------------------------------------------------------------------------

def test_sav_zst_decompresses_via_zstandard_fallback(monkeypatch):
    import zstandard
    from freeciv_world.world_reports.utils import savegame_parser as sp

    payload = '[savegame]\nversion=3\nturn=60\n'
    compressed = zstandard.ZstdCompressor().compress(payload.encode())

    def no_binary(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory: 'zstd'")

    monkeypatch.setattr(sp.subprocess, 'run', no_binary)
    out = sp.decompress_savegame_content(compressed, 'x_T60_x.sav.zst')
    assert out == payload


def test_sav_zst_stream_frame_without_content_size(monkeypatch):
    """zstd CLI writes frames without a content-size header; the fallback
    must handle those (one-shot decompress() rejects them)."""
    import io
    import zstandard
    from freeciv_world.world_reports.utils import savegame_parser as sp

    payload = b'[savegame]\nturn=61\n' * 100
    buf = io.BytesIO()
    cctx = zstandard.ZstdCompressor()
    with cctx.stream_writer(buf, closefd=False) as w:  # no content size header
        w.write(payload)

    monkeypatch.setattr(sp.subprocess, 'run',
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    out = sp.decompress_savegame_content(buf.getvalue(), 'y.sav.zst')
    assert out == payload.decode()


def test_sav_xz_falls_back_to_stdlib_lzma(monkeypatch):
    from freeciv_world.world_reports.utils import savegame_parser as sp

    payload = '[savegame]\nturn=1\n'
    compressed = lzma.compress(payload.encode())
    monkeypatch.setattr(sp.subprocess, 'run',
                        lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    assert sp.decompress_savegame_content(compressed, 'z.sav.xz') == payload


# ---------------------------------------------------------------------------
# RNG state rewrite
# ---------------------------------------------------------------------------

@needs_t60
def test_set_rng_from_seed_rewrites_random_block():
    m = _fresh_modifier()
    before_block = re.search(r'\[random\].*?(?=\n\[)', m.content, re.DOTALL).group(0)

    m.set_rng_from_seed(4242)
    block = re.search(r'\[random\].*?(?=\n\[)', m.content, re.DOTALL).group(0)

    assert block != before_block
    assert 'saved=TRUE' in block
    assert re.search(r'index_J=\d+', block)
    assert re.search(r'index_K=\d+', block)
    assert re.search(r'index_X=\d+', block)
    tables = re.findall(r'table(\d)="([0-9a-f ]+)"', block)
    assert [int(t[0]) for t in tables] == list(range(8))
    for _, words in tables:
        assert len(words.split()) == 7

    # Deterministic per seed; different seeds differ
    m2 = _fresh_modifier()
    m2.set_rng_from_seed(4242)
    block2 = re.search(r'\[random\].*?(?=\n\[)', m2.content, re.DOTALL).group(0)
    assert block2 == block

    m3 = _fresh_modifier()
    m3.set_rng_from_seed(4243)
    block3 = re.search(r'\[random\].*?(?=\n\[)', m3.content, re.DOTALL).group(0)
    assert block3 != block


def test_set_rng_from_seed_raises_without_random_block():
    from freeciv_world.forking.savegame_modifier import SavegameModifier
    m = SavegameModifier.__new__(SavegameModifier)
    m.content = '[game]\nturn=5\n[player0]\nname="X"\n'
    with pytest.raises(RuntimeError, match=r'\[random\]'):
        m.set_rng_from_seed(1)


@needs_t60
def test_clear_rng_state_unsaves_table_and_zeroes_gameseed():
    m = _fresh_modifier()
    assert 'saved=TRUE' in m.content
    m.clear_rng_state()
    block = re.search(r'\[random\].*?(?=\n\[)', m.content, re.DOTALL).group(0)
    assert 'saved=FALSE' in block
    assert 'saved=TRUE' not in block
    gs = re.search(r'"gameseed",(\d+),(\d+)', m.content)
    assert gs.group(1) == '0', 'live gameseed value must be zeroed'


# ---------------------------------------------------------------------------
# Username-rewrite attach (F4)
# ---------------------------------------------------------------------------

@needs_t60
def test_set_player_name_rewrites_identity_fields():
    m = _fresh_modifier()
    m.set_player_name(0, 'seed1000forkx1')
    section_start = m.content.find('[player0]')
    section_end = m.content.find('\n[', section_start + 1)
    section = m.content[section_start:section_end]
    assert 'name="seed1000forkx1"' in section
    assert 'username="seed1000forkx1"' in section
    assert 'ranked_username="seed1000forkx1"' in section
    # Other players untouched
    p1 = m.content.find('[player1]')
    assert 'seed1000forkx1' not in m.content[p1:p1 + 2000]


def test_set_player_name_raises_on_missing_section():
    from freeciv_world.forking.savegame_modifier import SavegameModifier
    m = SavegameModifier.__new__(SavegameModifier)
    m.content = '[game]\nturn=5\n'
    with pytest.raises(ValueError, match=r'\[player3\]'):
        m.set_player_name(3, 'x')


# ---------------------------------------------------------------------------
# Savegame coverage tracking (F2)
# ---------------------------------------------------------------------------

def test_savegame_coverage_accounting():
    from freeciv_world.world_reports.utils import savegame_parser as sp

    rd = '/tmp/fake-recdir-for-coverage-test'
    sp.reset_savegame_coverage(rd)
    sp.note_savegame_status(rd, 1, 'parsed')
    sp.note_savegame_status(rd, 2, 'failed')
    sp.note_savegame_status(rd, 2, 'parsed')   # later success upgrades
    sp.note_savegame_status(rd, 3, 'missing')
    sp.note_savegame_status(rd, 3, 'failed')

    cov = sp.compute_savegame_coverage(rd, [1, 2, 3, 4])
    assert cov['expected'] == 4
    assert cov['parsed'] == 2
    assert cov['coverage'] == pytest.approx(0.5)
    assert cov['failed_turns'] == [3]
    assert cov['missing_turns'] == [4]

    sp.reset_savegame_coverage(rd)
    cov = sp.compute_savegame_coverage(rd, [1, 2])
    assert cov['coverage'] == 0.0


def test_parse_failure_is_recorded_and_returns_none(tmp_path, monkeypatch):
    from freeciv_world.world_reports.utils import savegame_parser as sp

    rd = tmp_path / 'rec'
    (rd / 'savegames').mkdir(parents=True)
    # A syntactically valid xz file whose parsing will explode
    sav = rd / 'savegames' / 'u1_T7_2026-01-01-00_00.sav.xz'
    with lzma.open(sav, 'wt') as f:
        f.write('[research]\nnot really parseable in a way that matters\n')

    monkeypatch.setattr(sp, 'parse_city_production',
                        lambda content: (_ for _ in ()).throw(RuntimeError('boom')))
    sp.reset_savegame_coverage(str(rd))
    out = sp.extract_complete_data_from_savegame('u1', 7, recording_dir=str(rd))
    assert out is None
    cov = sp.compute_savegame_coverage(str(rd), [7])
    assert cov['failed_turns'] == [7]
    assert cov['coverage'] == 0.0


def test_require_coverage_env_raises(tmp_path, monkeypatch):
    """collect_all must hard-fail on incomplete coverage when the env is set."""
    from freeciv_world.world_reports.extractors.metrics_collector import MetricsCollector

    rec = tmp_path / 'recordings'
    rec.mkdir()  # exists, but holds no savegames -> coverage 0.0

    class Cfg:
        recording_dir = str(rec)

    class Loader:
        ruleset = None

    states = {1: {'player': {'0': {'name': 'X'}}},
              2: {'player': {'0': {'name': 'X'}}}}

    monkeypatch.setenv('FBSIM_REQUIRE_SAVEGAME_COVERAGE', '1')
    with pytest.raises(RuntimeError, match='savegame_coverage'):
        MetricsCollector().collect_all(states=states, config=Cfg(),
                                       data_loader=Loader())

    # Without the env it degrades loudly but returns data with the stamp
    monkeypatch.delenv('FBSIM_REQUIRE_SAVEGAME_COVERAGE')
    data = MetricsCollector().collect_all(states=states, config=Cfg(),
                                          data_loader=Loader())
    assert data['metadata']['savegame_coverage'] == 0.0
    assert data['metadata']['savegame_turns_expected'] == 2


# ---------------------------------------------------------------------------
# Per-username port-claim release (F6)
# ---------------------------------------------------------------------------

def test_release_user_claims_only_touches_own_rows(tmp_path):
    from freeciv_world.freeciv.utils.port_utils import PortStatus

    ps = PortStatus.__new__(PortStatus)  # no network in __init__
    ps.lock_file = str(tmp_path / 'ports.lock')
    ps.occupied_ports_file = str(tmp_path / 'occupied.txt')

    rows = [
        '6301 100 0 1 seed1000forka\n',
        '6302 100 0 1 seed1000forkb\n',
        '6303 100 0 1\n',                 # legacy 4-column row
        '6304 100 0 1 seed1000forka\n',
    ]
    Path(ps.occupied_ports_file).write_text(''.join(rows))

    released = ps.release_user_claims('seed1000forka')
    assert released == 2

    remaining = Path(ps.occupied_ports_file).read_text().splitlines()
    assert [r.split()[0] for r in remaining] == ['6302', '6303']
    assert remaining[1].split()[4] == '-'  # legacy row normalized, kept

    # Releasing a user with no claims is a no-op
    assert ps.release_user_claims('nobody') == 0


def test_ports_import_is_lazy():
    """Importing the module (e.g. via fork_manager) must not hit the network."""
    from freeciv_world.freeciv.utils.port_utils import Ports, _LazyPortStatus
    assert isinstance(Ports, _LazyPortStatus)


# ---------------------------------------------------------------------------
# Settings ack tracking (W4)
# ---------------------------------------------------------------------------

def test_settings_ack_roundtrip():
    from freeciv_world.freeciv.connectivity.client_state import ClientState

    cs = ClientState.__new__(ClientState)
    cs.pending_settings = {}

    class WS:
        sent = []

        def send_message(self, msg):
            WS.sent.append(msg)

    cs.ws_client = WS()
    cs._send_set('aifill', 5)
    cs._send_set('fogofwar', 'disabled')
    assert WS.sent == ['/set aifill 5', '/set fogofwar disabled']
    assert set(cs.pending_settings) == {'aifill', 'fogofwar'}

    cs.ack_setting_from_message("Option: aifill has been set to 5.")
    assert set(cs.pending_settings) == {'fogofwar'}
    assert cs.warn_unacked_settings() == ['fogofwar']


def test_dead_server_commands_removed():
    src = (ROOT / 'worlds/freeciv/freeciv_world/freeciv/connectivity/'
           'client_state.py').read_text()
    body = src[src.find('def set_multiplayer_game'):
               src.find('def update_state')]
    code_lines = [ln for ln in body.split('\n')
                  if not ln.strip().startswith('#')]
    code = '\n'.join(code_lines)
    assert 'cmdlevel' not in code
    assert 'fogofwar' not in code
    assert 'revealmap' not in code


# ---------------------------------------------------------------------------
# Publish-gate dry run: exactly 176 dark flagged, 124 healthy passed
# ---------------------------------------------------------------------------

@needs_corpus
def test_publish_gate_dry_run_flags_exactly_176_of_300():
    wpg = load_script('scripts/uplift_v2/world_publish_gate.py')

    files = wpg.collect_world_files(str(MINED_WORLDS))
    assert len(files) == 300

    results = {}
    for path in files:
        data = wpg.load_world(path)
        results[wpg.world_id_from_path(path)] = wpg.gate_world(
            data, coverage_optional=True)

    flagged = [w for w, r in results.items() if not r['publish']]
    passed = [w for w, r in results.items() if r['publish']]
    assert len(flagged) == 176
    assert len(passed) == 124

    matrix = wpg.run_audit_comparison(results, str(AUDIT_JSON))
    assert matrix['exact'] is True
    assert matrix['dark_flagged'] == 176
    assert matrix['healthy_passed'] == 124
    assert matrix['dark_passed'] == 0
    assert matrix['healthy_flagged'] == 0


@needs_corpus
def test_publish_gate_requires_coverage_stamp_by_default():
    """New serializations must carry savegame_coverage==1.0; a legacy world
    without the stamp is NOT publishable unless --coverage-optional."""
    wpg = load_script('scripts/uplift_v2/world_publish_gate.py')
    healthy = MINED_WORLDS / 'seed1001_data.json.gz'
    data = wpg.load_world(str(healthy))

    strict = wpg.gate_world(data, coverage_optional=False)
    assert not strict['publish']
    assert any('savegame_coverage' in r for r in strict['reasons'])

    data['metadata']['savegame_coverage'] = 1.0
    assert wpg.gate_world(data, coverage_optional=False)['publish']

    data['metadata']['savegame_coverage'] = 0.9967
    assert not wpg.gate_world(data, coverage_optional=False)['publish']


# ---------------------------------------------------------------------------
# Pilot gate checker plumbing
# ---------------------------------------------------------------------------

def test_pilot_gates_reports_missing_inputs(tmp_path):
    pg = load_script('scripts/uplift_v2/pilot_gates.py')
    out = tmp_path / 'verdict.json'
    rc = pg.main(['--out', str(out)])
    assert rc == 3  # nothing failed, everything missing
    verdict = json.loads(out.read_text())
    assert verdict['overall'] == 'INCOMPLETE'
    assert verdict['n_fail'] == 0
    assert verdict['n_missing'] == 16  # 6 P1 + 10 P2 gates


def test_pilot_gates_p1_pass_on_synthetic_healthy_worlds(tmp_path):
    pg = load_script('scripts/uplift_v2/pilot_gates.py')

    p1 = tmp_path / 'p1'
    p1.mkdir()
    for i in range(10):
        events = []
        for pid in range(5):
            for k in range(90):  # 450 tech events total
                events.append({'type': 'tech_discovered', 'player_id': pid,
                               'turn': 10 + k, 'metadata': {'tech_name': f't{k}'}})
        for k in range(9):
            events.append({'type': 'wonder_completed', 'player_id': k % 5,
                           'turn': 100 + k})
        world = {
            'metadata': {'turn': 302, 'savegame_coverage': 1.0},
            'time_series': {'techs_known': {
                '60': {str(p): 1 for p in range(5)},
                '302': {str(p): 90 for p in range(5)}}},
            'events': events,
            'diplomacy': {'relations': {
                f'{a}_{b}': {} for a in range(5) for b in range(5) if a != b}},
        }
        with open(p1 / f'seed{i}_data.json', 'w') as f:
            json.dump(world, f)

    gates = pg.run_p1_gates(str(p1), expected_worlds=10, reference_turn=60)
    by_name = {g['gate']: g for g in gates}
    for name in ('p1_frozen', 'p1_wonders', 'p1_tech_events',
                 'p1_diplomacy', 'p1_coverage', 'p1_no_zstd_tracebacks'):
        assert by_name[name]['status'] == 'PASS', by_name[name]


@needs_t60
def test_pilot_gate_savegame_parsers_on_real_save():
    pg = load_script('scripts/uplift_v2/pilot_gates.py')
    content = pg.read_savegame_text(T60_SAVE)

    rows = pg.parse_research_rows(content)
    assert set(rows) == {0, 1, 2, 3, 4}
    assert rows[2]['goal'] == 'A_UNSET'   # known property of this save
    assert all(set(r['done']) <= {'0', '1'} for r in rows.values())

    dipl = pg.parse_diplstates(content)
    assert 0 in dipl and len(dipl[0]) >= 5

    settings = pg.parse_settings_rows(content, exclude=('gameseed', 'metamessage'))
    assert any(s.startswith('"aifill"') for s in settings)
    assert not any(s.startswith('"gameseed"') for s in settings)
