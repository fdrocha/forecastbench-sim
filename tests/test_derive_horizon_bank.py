"""Unit tests for the deterministic H2/H3 bank derivation and truth helpers."""
import copy
import json

from conftest import load_script

derive = load_script("scripts/uplift_v2/derive_horizon_bank.py")
truth = load_script("scripts/uplift_v2/mine_natcond_truth.py")


def _bank():
    return {
        "game_id": "w", "snapshot_turn": 60, "game_max_turn": 302,
        "questions": [
            {"question_id": "q0007", "template_id": "tech_comparative",
             "resolution_turn": 90, "horizon": "H1",
             "parameters": {"civ_a": "Benin", "civ_b": "Jolof",
                            "resolution_turn": 90},
             "question_text": "Will Benin have more technologies than Jolof at turn 90?",
             "resolution": {"answer": True, "value_a": 26, "value_b": 20}},
            {"question_id": "q0010", "template_id": "wonder_completed",
             "resolution_turn": 90, "horizon": "H1",
             "parameters": {"wonder_name": "Pyramids", "snapshot_turn": 60,
                            "resolution_turn": 90},
             "question_text": "Will any civilization complete Pyramids by turn 90?",
             "resolution": {"answer": False}},
            # organic non-H1 question: must pass through untouched, underived
            {"question_id": "q9000", "template_id": "score_rank_1",
             "resolution_turn": 300, "horizon": "H7",
             "parameters": {"resolution_turn": 300},
             "question_text": "Will Benin lead at turn 300?"},
        ],
    }


def test_qid_scheme_and_family_key():
    out = derive.derive_bank(_bank())
    qids = [q["question_id"] for q in out["questions"]]
    assert qids == ["q0007", "q0010", "q9000",
                    "q0007_h2", "q0010_h2", "q0007_h3", "q0010_h3"]
    assert derive.family_qid("q0007_h2") == "q0007"
    assert derive.family_qid("q0007_h3") == "q0007"
    assert derive.family_qid("q0007") == "q0007"
    assert derive.family_qid("q9000") == "q9000"


def test_derivation_rewrites_turn_and_drops_base_resolution():
    out = derive.derive_bank(_bank())
    by_id = {q["question_id"]: q for q in out["questions"]}
    h2 = by_id["q0007_h2"]
    assert h2["horizon"] == "H2" and h2["resolution_turn"] == 120
    assert h2["question_text"] == (
        "Will Benin have more technologies than Jolof at turn 120?")
    assert h2["parameters"]["resolution_turn"] == 120
    assert h2["parameters"]["civ_a"] == "Benin"  # same target/subject civs
    assert h2["template_id"] == "tech_comparative"
    assert "resolution" not in h2  # base game's H1 answer never leaks
    h3 = by_id["q0010_h3"]
    assert h3["resolution_turn"] == 150 and h3["horizon"] == "H3"
    assert "by turn 150?" in h3["question_text"]
    assert h3["parameters"]["snapshot_turn"] == 60  # non-resolution turns kept
    # H1 originals untouched, organic H7 untouched
    src = _bank()
    assert by_id["q0007"] == src["questions"][0]
    assert by_id["q9000"] == src["questions"][2]


def test_deterministic_and_idempotent():
    b = _bank()
    once = derive.derive_bank(b)
    again = derive.derive_bank(copy.deepcopy(b))
    assert once == again  # deterministic
    twice = derive.derive_bank(copy.deepcopy(once))
    assert twice == once  # idempotent: derived questions are regenerated
    assert json.dumps(twice, sort_keys=True) == json.dumps(once, sort_keys=True)


def test_dead_derived_qids_guard():
    answers = {
        "q0007_h2": [{"tag": "s1", "answer": True}, {"tag": "s2", "answer": None}],
        "q0010_h2": [{"tag": "s1", "answer": None}, {"tag": "s2", "answer": None}],
        "q0007_h3": [{"tag": "s1", "answer": False}],
        "q0007": [{"tag": "s1", "answer": None}],  # H1 is never in scope
    }
    dead = truth.dead_derived_qids(
        answers, {"q0007_h2", "q0010_h2", "q0007_h3", "q9999_h2"})
    assert dead == ["q0010_h2"]
    assert "q0010_h2" in answers  # reported, not deleted (resume consistency)


def test_load_questions_returns_bank_civs(tmp_path):
    cells_v2 = load_script("scripts/uplift_v2/natcond_cells_v2.py")
    bank = _bank()
    bank["civilizations"] = {"0": {"name": "Benin", "nation_id": 1},
                             "1": {"name": "Jolof", "nation_id": 2},
                             "5": {"name": "Pirate", "nation_id": 99}}
    p = tmp_path / "questions.json"
    p.write_text(json.dumps(bank))
    qinfo, civs = cells_v2.load_questions(str(p), {"H1"})
    assert set(qinfo) == {"q0007", "q0010"}
    assert civs == {"Benin", "Jolof", "Pirate"}  # Pirate filtered later
    # bank without a civilizations table -> empty set (harvest fallback)
    p2 = tmp_path / "q2.json"
    p2.write_text(json.dumps(_bank()))
    _, civs2 = cells_v2.load_questions(str(p2), {"H1"})
    assert civs2 == set()


def test_cells_auto_use_arm_questions(tmp_path):
    cells_v2 = load_script("scripts/uplift_v2/natcond_cells_v2.py")
    arm = tmp_path / "w"
    arm.mkdir()
    assert cells_v2.world_questions_path(
        str(arm), "data/questions_mc/{game_id}/questions.json", "w") == (
        "data/questions_mc/w/questions.json")
    (arm / "questions.json").write_text("{}")
    assert cells_v2.world_questions_path(
        str(arm), "data/questions_mc/{game_id}/questions.json", "w") == (
        str(arm / "questions.json"))
