"""Unit tests for natcond_elicit_v2.py prompt construction (no API calls)."""
from conftest import load_script

elicit = load_script("scripts/uplift_v2/natcond_elicit_v2.py")

CELL = {"qid": "q0007", "event_id": "sp_1", "event_kind": "specific",
        "event_desc": "Benin discovered Feudalism", "window": [75, 90],
        "question": "Will Benin control the most cities at turn 120?",
        "horizon": "H2", "resolution_turn": 120}


def test_turn1_prompt_uses_closeness_scoring_not_brier():
    prompt = elicit.build_base_prompt("REPORT BODY", CELL["question"], 120)
    assert "Estimate the probability of YES as accurately as you can." in prompt
    assert "We have re-run this world many times" in prompt
    assert "closer is strictly better, from either side" in prompt
    assert ("saying 1% when the truth is 10% is penalized far more than "
            "saying 30% when the truth is 39%") in prompt
    assert "Do not round to 0 or 1." in prompt
    assert "brier" not in prompt.lower()
    assert "MAXIMIZE" not in prompt
    # skeleton and per-cell resolution turn survive
    assert "Question Title: " + CELL["question"] in prompt
    assert "at turn 120" in prompt
    assert "REPORT BODY" in prompt
    assert "<probability> </probability>" in prompt


def test_reveal_turn_wording():
    msg = elicit.build_reveal_message(CELL)
    assert msg == (
        "The game has continued. One fact about turns 75-90 has been revealed "
        "to you: Benin discovered Feudalism. Given this news and everything "
        "you already knew, provide an updated forecast for the SAME question: "
        "Will Benin control the most cities at turn 120? The same scoring "
        "applies: get as close as you can to the true probability given this "
        "information. End with your updated probability in "
        "<probability> </probability> tags.")
    assert "suppose" not in msg.lower()  # v1 hypothetical wording is gone


def test_single_turn_prompt_contains_same_final_info():
    prompt = elicit.build_single_turn_prompt("REPORT BODY", CELL, 120)
    assert ("The game has continued. One fact about turns 75-90 has been "
            "revealed to you: Benin discovered Feudalism.") in prompt
    assert "Question Title: " + CELL["question"] in prompt
    assert "REPORT BODY" in prompt
    assert "closer is strictly better, from either side" in prompt
    assert "at turn 120" in prompt
    assert "suppose" not in prompt.lower()
    # the reveal is additional background, after the report
    assert prompt.index("REPORT BODY") < prompt.index("The game has continued.")


def test_parse_prob():
    assert elicit.parse_prob("blah <probability>0.42</probability>") == 0.42
    assert elicit.parse_prob("<probability> 0.07 </probability>") == 0.07
    assert elicit.parse_prob("p=0.4, no tags") is None
    assert elicit.parse_prob("<probability>1.5</probability>") is None
    assert elicit.parse_prob("<probability>1.005</probability>") == 1.0
    assert elicit.parse_prob(
        "<think><probability>0.9</probability></think>"
        "after thought <probability>0.2</probability>") == 0.2
    # takes the last tag
    assert elicit.parse_prob(
        "<probability>0.1</probability> ... <probability>0.3</probability>") == 0.3
