"""Unit tests for verification_report.py assembly (small synthetic inputs)."""
import gzip
import hashlib
import json

from conftest import load_script

vr = load_script("scripts/uplift_v2/verification_report.py")

REV1 = ("Revision 1: p1_tech_events band [400,600]->[300,805]; evidence: "
        "52-recording healthy reference min=314/med=517/max=732, legacy "
        "seed1=644, zero duplicate (turn,pid,tech) events in out-of-band "
        "worlds 1004=645/1010=615/1021=397; commit 7bbd6764")
REV2 = ("Revision 2: p2_p0_next_tech_modal replaced by "
        "p2_anchor_path_not_excluded; evidence: anchor path Literacy t61 -> "
        "The Republic t69 is the fork modal path 100/100 and 99/100, p0 held "
        "99 banked Literacy bulbs at t60 making tech #1 legitimately "
        "deterministic; commit 779c06e0")


def test_gate_revision_log_verbatim():
    assert vr.GATE_REVISION_LOG == [REV1, REV2]


# ---------------------------------------------------------------------------
# (a) pilot gate verdicts
# ---------------------------------------------------------------------------

def _verdict(overall="PASS", gates=None):
    gates = gates if gates is not None else [
        {"gate": "p1_frozen", "status": "PASS", "value": "10/10",
         "threshold": "10/10 worlds with frozen=0", "detail": ""}]
    n_fail = sum(1 for g in gates if g["status"] == "FAIL")
    return {"overall": overall, "n_pass": len(gates) - n_fail,
            "n_fail": n_fail, "n_missing": 0,
            "inputs": {"p1_dir": "x"}, "gates": gates}


def test_load_pilot_verdicts(tmp_path):
    (tmp_path / "p1.json").write_text(json.dumps(_verdict()))
    (tmp_path / "p2.json").write_text(json.dumps(_verdict(gates=[
        {"gate": "p2_calibration", "status": "PASS", "value": "0.004",
         "threshold": "pooled |obs - p_mc| < 0.05", "detail": ""}])))
    out = vr.load_pilot_verdicts(str(tmp_path / "p1.json"),
                                 str(tmp_path / "p2.json"),
                                 str(tmp_path / "nope.json"))
    assert out["p1"]["overall"] == "PASS" and not out["p1"]["missing"]
    assert out["p1"]["gates"][0]["gate"] == "p1_frozen"
    assert out["p1_p2"]["gates"][0]["gate"] == "p2_calibration"
    assert out["pre_revision_snapshot"]["missing"] is True
    assert out["pre_revision_snapshot"]["overall"] is None


# ---------------------------------------------------------------------------
# (b) world health
# ---------------------------------------------------------------------------

def _world(frozen_pid1=False, coverage=1.0):
    events = [{"turn": 61, "type": "tech_discovered", "player_id": 0,
               "description": "Benin discovered Alphabet"},
              {"turn": 70, "type": "wonder_completed", "player_id": 0,
               "description": "Pyramids"}]
    if not frozen_pid1:
        events.append({"turn": 62, "type": "tech_discovered", "player_id": 1,
                       "description": "Jolof discovered Pottery"})
    return {"events": events,
            "time_series": {"techs_known": {"60": {"0": 5, "1": 4}}},
            "diplomacy": {"relations": {"0-1": "war"}},
            "metadata": {"savegame_coverage": coverage}}


def test_world_health_table(tmp_path):
    with gzip.open(tmp_path / "seedA_data.json.gz", "wt") as f:
        json.dump(_world(), f)
    with gzip.open(tmp_path / "seedB_data.json.gz", "wt") as f:
        json.dump(_world(frozen_pid1=True, coverage=0.5), f)
    rows = {r["world"]: r for r in vr.world_health_table(str(tmp_path))}
    assert set(rows) == {"seedA", "seedB"}
    a = rows["seedA"]
    assert a["publish"] is True and a["reasons"] == []
    assert a["frozen_civs"] == 0 and a["tech_covers_all_civs"] is True
    assert a["n_civs"] == 2 and a["tech_events"] == 2
    assert a["wonder_completed_events"] == 1 and a["diplomacy_pairs"] == 1
    assert a["savegame_coverage"] == 1.0
    b = rows["seedB"]
    assert b["publish"] is False
    assert b["frozen_civs"] == 1 and b["tech_covers_all_civs"] is False
    assert b["savegame_coverage"] == 0.5
    assert any("frozen_civs=1" in r for r in b["reasons"])
    assert any("savegame_coverage=0.5" in r for r in b["reasons"])


# ---------------------------------------------------------------------------
# (c) fork exchangeability (pilot_gates P2 reuse)
# ---------------------------------------------------------------------------

def _fork_world(p0_seq=("Literacy", "The Republic"), wonder_turn=70):
    events = [{"turn": 61 + 8 * i, "type": "tech_discovered", "player_id": 0,
               "description": f"Benin discovered {t}",
               "metadata": {"tech_name": t}}
              for i, t in enumerate(p0_seq)]
    events += [{"turn": 63, "type": "tech_discovered", "player_id": 1,
                "description": "Jolof discovered Pottery"},
               {"turn": wonder_turn, "type": "wonder_completed",
                "player_id": 0, "description": "Pyramids"}]
    return {"events": events,
            "time_series": {"techs_known": {"60": {"0": 5, "1": 4}}},
            "diplomacy": {"relations": {"0-1": "peace"}},
            "metadata": {"savegame_coverage": 1.0}}


def test_fork_exchangeability_section(tmp_path):
    forks = tmp_path / "forks"
    forks.mkdir()
    for fid in ("f000", "f001"):
        with gzip.open(forks / f"{fid}_data.json.gz", "wt") as f:
            json.dump(_fork_world(), f)
    anchor = tmp_path / "seed1000_data.json"
    anchor.write_text(json.dumps(_fork_world()))
    bank = tmp_path / "anchor_bank.json"
    bank.write_text(json.dumps({"q1": {"obs": 1, "p_mc": 0.98},
                                "q2": {"obs": 0, "p_mc": 0.03}}))

    sec = vr.fork_exchangeability(str(forks), str(anchor), None, None,
                                  str(bank), fork_turn=60, window_end=90,
                                  expected_forks=2)
    assert sec["anchor"] == "seed1000"
    by = {g["gate"]: g for g in sec["gates"]}
    assert by["p2_frozen"]["status"] == "PASS"
    assert by["p2_wonder_window"]["status"] == "PASS"
    # anchor's 2-step p0 path is the fork modal path here -> PASS
    assert by["p2_anchor_path_not_excluded"]["status"] == "PASS"
    # pooled |mean(obs) - mean(p_mc)| = |0.5 - 0.505| < 0.05
    assert by["p2_calibration"]["status"] == "PASS"
    # savegame-based gates have no inputs in this synthetic pilot
    for g in ("p2_t60_research_identical", "p2_t60_diplomacy_identical",
              "p2_seed_pairs", "p2_goal_not_unset", "p2_anchor_in_ensemble",
              "p2_settings_identical"):
        assert by[g]["status"] == "MISSING"
    assert sec["n_pass"] == 4 and sec["n_fail"] == 0 and sec["n_missing"] == 6


# ---------------------------------------------------------------------------
# (d) truth quality: split-half reliability per horizon
# ---------------------------------------------------------------------------

def _cell(hz, rt, py_a, pyx_a, d_a, py_b, pyx_b, d_b):
    return {"horizon": hz, "resolution_turn": rt,
            "half_a": {"p_y": py_a, "p_yx": pyx_a, "delta": d_a},
            "half_b": {"p_y": py_b, "p_yx": pyx_b, "delta": d_b}}


def test_truth_quality_split_half_reliability():
    cells = [
        # H2: p_y and p_yx replicate exactly (r=1); delta anti-replicates
        _cell("H2", 120, 0.2, 0.5, 0.1, 0.2, 0.5, -0.1),
        _cell("H2", 120, 0.4, 0.6, 0.2, 0.4, 0.6, -0.2),
        _cell("H2", 120, 0.8, 0.9, 0.3, 0.8, 0.9, -0.3),
        # H4 (t180): single cell -> r undefined
        _cell("H4", 180, 0.3, 0.4, 0.1, 0.35, 0.45, 0.1),
        # H5 (t210): zero variance in half A -> r undefined
        _cell("H5", 210, 0.5, 0.5, 0.0, 0.4, 0.6, 0.2),
        _cell("H5", 210, 0.5, 0.5, 0.0, 0.6, 0.7, 0.1),
        # a cell missing halves (v1 schema) is skipped, not fatal
        {"horizon": "H1", "p_y": 0.5},
    ]
    tq = vr.truth_quality(cells)
    assert set(tq) == {"H2", "H4", "H5"}
    assert tq["H2"] == {"n_cells": 3, "resolution_turns": [120],
                        "r_p_y": 1.0, "r_p_yx": 1.0, "r_delta": -1.0}
    assert tq["H4"]["n_cells"] == 1
    assert tq["H4"]["r_p_y"] is None and tq["H4"]["resolution_turns"] == [180]
    assert tq["H5"]["r_p_y"] is None  # degenerate margin
    assert tq["H5"]["resolution_turns"] == [210]


def test_pearson():
    assert vr.pearson([1, 2, 3], [2, 4, 6]) == 1.0
    assert vr.pearson([1, 2, 3], [3, 2, 1]) == -1.0
    assert vr.pearson([1], [1]) is None
    assert vr.pearson([1, 1, 1], [1, 2, 3]) is None
    r = vr.pearson([1, 2, 3, 4], [1.1, 1.9, 3.2, 3.8])
    assert 0.99 < r < 1.0


# ---------------------------------------------------------------------------
# (e) provenance
# ---------------------------------------------------------------------------

def test_bundle_hashes(tmp_path):
    (tmp_path / "a.txt").write_bytes(b"hello")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "b.bin").write_bytes(b"\x00\x01")
    hashes = vr.bundle_hashes(str(tmp_path))
    assert hashes == {
        "a.txt": hashlib.sha256(b"hello").hexdigest(),
        "sub/b.bin": hashlib.sha256(b"\x00\x01").hexdigest(),
    }


# ---------------------------------------------------------------------------
# assembly + markdown framing
# ---------------------------------------------------------------------------

def test_assemble_report_carries_verbatim_revision_log():
    report = vr.assemble_report()
    prov = report["sections"]["provenance"]
    assert prov["gate_revision_log"] == [REV1, REV2]
    # provenance section exists (with the revision log) even with no bundle
    assert "sha256" not in prov


def test_render_markdown_gates_before_evidence():
    pilot = {"p1": {"path": "p1.json", "missing": False, "overall": "PASS",
                    "n_pass": 1, "n_fail": 0, "n_missing": 0, "inputs": {},
                    "gates": [{"gate": "p1_frozen", "status": "PASS",
                               "value": "10/10",
                               "threshold": "10/10 worlds with frozen=0",
                               "detail": ""}]},
             "p1_p2": {"path": "p2.json", "missing": True, "overall": None,
                       "gates": []},
             "pre_revision_snapshot": {"path": "prerev.json", "missing": True,
                                       "overall": None, "gates": []}}
    health = [{"world": "seedA", "frozen_civs": 0, "n_civs": 2,
               "tech_events": 500, "tech_covers_all_civs": True,
               "wonder_completed_events": 12, "diplomacy_pairs": 10,
               "savegame_coverage": 1.0, "publish": True, "reasons": []}]
    truth = {"H4": {"n_cells": 5, "resolution_turns": [180],
                    "r_p_y": 0.9, "r_p_yx": 0.8, "r_delta": 0.7}}
    report = vr.assemble_report(
        pilot=pilot, world_health=health, truth=truth,
        provenance={"bundle_dir": "b", "sha256": {"x.json": "ab" * 32}})
    md = vr.render_markdown(report)
    # framing: registered gates + verbatim revision log precede all evidence
    i_gates = md.index("## 1. Pre-registered gates")
    i_rev = md.index("## 2. Gate revision log (verbatim)")
    i_evidence = md.index("## 3. Evidence")
    assert i_gates < i_rev < i_evidence
    assert REV1 in md and REV2 in md
    # registered threshold surfaces in Section 1, before any status
    assert md.index("10/10 worlds with frozen=0") < md.index("Overall: **PASS**")
    # evidence rows render
    assert "| seedA | 0 | 500 | True | 12 | 10 | 1.0 | PUBLISH |" in md
    assert "| H4 | 180 | 5 | 0.9 | 0.8 | 0.7 |" in md
    assert "x.json" in md and "ab" * 32 in md
    # missing sections are marked, not dropped
    assert "MISSING: p2.json" in md
    assert "Not assembled (--forks-dir not provided)." in md


def test_main_end_to_end(tmp_path, capsys):
    (tmp_path / "p1.json").write_text(json.dumps(_verdict()))
    worlds = tmp_path / "worlds"
    worlds.mkdir()
    with gzip.open(worlds / "seedA_data.json.gz", "wt") as f:
        json.dump(_world(), f)
    cells = tmp_path / "cells.json"
    cells.write_text(json.dumps([
        _cell("H2", 120, 0.2, 0.5, 0.1, 0.25, 0.5, 0.12),
        _cell("H2", 120, 0.6, 0.8, 0.2, 0.55, 0.85, 0.22)]))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "bank.json").write_text("{}")
    out_json = tmp_path / "report" / "verification_report.json"

    rc = vr.main(["--p1-verdict", str(tmp_path / "p1.json"),
                  "--p2-verdict", str(tmp_path / "absent.json"),
                  "--prerev-verdict", str(tmp_path / "absent2.json"),
                  "--worlds-dir", str(worlds),
                  "--cells", str(cells),
                  "--bundle-dir", str(bundle),
                  "--out-json", str(out_json)])
    assert rc == 0
    report = json.loads(out_json.read_text())
    md = (out_json.parent / "verification_report.md").read_text()
    sections = report["sections"]
    assert sections["pilot_gate_verdicts"]["p1"]["overall"] == "PASS"
    assert sections["world_health"][0]["world"] == "seedA"
    assert sections["fork_exchangeability"] is None  # no --forks-dir
    assert sections["truth_quality"]["H2"]["n_cells"] == 2
    assert sections["provenance"]["sha256"] == {
        "bank.json": hashlib.sha256(b"{}").hexdigest()}
    assert sections["provenance"]["gate_revision_log"] == [REV1, REV2]
    assert REV1 in md and REV2 in md
    assert md.index("## 1. Pre-registered gates") < md.index("## 3. Evidence")
    out = capsys.readouterr().out
    assert "verification report ->" in out
