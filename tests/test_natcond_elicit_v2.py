"""Unit tests for natcond_elicit_v2.py prompt construction (no API calls)."""
from conftest import load_script

elicit = load_script("scripts/uplift_v2/natcond_elicit_v2.py")

CELL = {"qid": "q0007", "event_id": "sp_1", "event_kind": "specific",
        "event_desc": "Benin discovered Feudalism", "window": [75, 90],
        "question": "Will Benin control the most cities at turn 120?",
        "horizon": "H2", "resolution_turn": 120}


def test_turn1_prompt_uses_log_loss_scoring_not_brier():
    prompt = elicit.build_base_prompt("REPORT BODY", CELL["question"], 120)
    assert ("Estimate the probability of YES as accurately as you can. "
            "We have re-run this world many times, so the true probability "
            "of this event is known. Your answer is scored by log loss "
            "against that true probability. Getting the odds right matters: "
            "errors at extreme odds are penalized more heavily. "
            "Do not report 0 or 1.") in prompt
    # v3 removed the perfect-forecaster floor clause and the concrete
    # 1%/10% example (anchoring)
    assert "floor" not in prompt.lower()
    assert "10%" not in prompt and "39%" not in prompt
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
        "applies: log loss against the true probability given "
        "this information. End with your updated probability in "
        "<probability> </probability> tags.")
    assert "suppose" not in msg.lower()  # v1 hypothetical wording is gone


def test_single_turn_prompt_contains_same_final_info():
    prompt = elicit.build_single_turn_prompt("REPORT BODY", CELL, 120)
    assert ("The game has continued. One fact about turns 75-90 has been "
            "revealed to you: Benin discovered Feudalism.") in prompt
    assert "Question Title: " + CELL["question"] in prompt
    assert "REPORT BODY" in prompt
    assert "scored by log loss against that true probability" in prompt
    assert "at turn 120" in prompt
    assert "suppose" not in prompt.lower()
    # the reveal is additional background, after the report
    assert prompt.index("REPORT BODY") < prompt.index("The game has continued.")


def test_resolution_criteria_render_per_horizon_turn():
    # derived-qid cells carry their own resolution_turn; the rc string and
    # single-turn prompt must state that turn, not t90 (H4/H5 = the
    # production-fleet t180/t210 horizons)
    for rt, hz in ((90, "H1"), (120, "H2"), (150, "H3"),
                   (180, "H4"), (210, "H5")):
        cell = dict(CELL, qid=f"q0007_h{hz[1]}" if hz != "H1" else "q0007",
                    horizon=hz, resolution_turn=rt,
                    question=f"Will Benin lead at turn {rt}?")
        base = elicit.build_base_prompt("R", cell["question"], rt)
        single = elicit.build_single_turn_prompt("R", cell, rt)
        want = (f"Resolves YES if the answer to the question is affirmative "
                f"in the simulation state at turn {rt}")
        assert want in base and want in single
        assert f"at turn {90 if rt != 90 else 150}" not in base.split(
            "Question Title:")[0]  # no stray other-horizon turn in the header


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
