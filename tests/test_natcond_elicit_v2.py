"""Unit tests for natcond_elicit_v2.py prompt construction (no API calls).

The v4 prompt strings are APPROVED wording: the pinned full-text tests here
are conformance tests — do not loosen them to make an edit pass; a wording
change needs an approved revision and a matching pin update.
"""
from conftest import load_script

elicit = load_script("scripts/uplift_v2/natcond_elicit_v2.py")

CELL = {"qid": "q0007", "event_id": "sp_1", "event_kind": "specific",
        "event_desc": "Benin discovered Feudalism", "window": [75, 90],
        "question": "Will Benin control the most cities at turn 120?",
        "horizon": "H2", "resolution_turn": 120}

CELL_TPL = dict(
    CELL,
    question="Will Benin have more cities than Jolof at turn 120?",
    template_id="city_count_comparative",
    parameters={"civ_a": "Benin", "civ_b": "Jolof", "player_id_a": 0,
                "player_id_b": 1, "resolution_turn": 120})


def test_base_prompt_pinned_exactly():
    # generic-criteria path (cell carries no template_id/parameters)
    prompt = elicit.build_base_prompt("REPORT BODY", CELL["question"], 120)
    assert prompt == (
        "You will be asked a binary forecasting question about a simulation "
        "in progress. Your task is to estimate the probability that the "
        "event described in the question occurs. Please provide your "
        "reasoning before stating your final probability.\n"
        "\n"
        "Question Title: Will Benin control the most cities at turn 120?\n"
        "Question Background: This is a partial report on a FreeCiv game "
        "simulation in progress, observed at turn 60. Five AI civilizations "
        "are competing.\n"
        "\n"
        "REPORT BODY\n"
        "\n"
        "Resolution Criteria: Resolves YES if the answer to the question is "
        "affirmative in the simulation state at turn 120, as determined by "
        "the game's recorded metrics.\n"
        "\n"
        "Think step by step about the information provided, reason about "
        "uncertainty, and put your final confidence that the question "
        "resolves YES in <probability> </probability> tags. The probability "
        "should be a number between 0 and 1.\n"
        "\n"
        "Your answer will be evaluated with a proper scoring rule, so your "
        "best strategy is to report your honest probability estimate.\n"
        "\n"
        "Your final answer should be the probability that the event resolves "
        "YES and your response SHOULD STRICTLY END with "
        "<probability> </probability> tags.")


def test_base_prompt_has_no_prior_scoring_paragraphs():
    prompt = elicit.build_base_prompt("REPORT BODY", CELL["question"], 120)
    low = prompt.lower()
    # v1 Brier paragraph and the v2/v3 log-loss paragraph are both gone
    assert "brier" not in low
    assert "log loss" not in low
    assert "maximize" not in low
    assert "re-run" not in low
    assert "do not report 0 or 1" not in low
    assert "true probability" not in low
    # replaced by the proper-scoring-rule honesty paragraph
    assert ("Your answer will be evaluated with a proper scoring rule, so "
            "your best strategy is to report your honest probability "
            "estimate.") in prompt


def test_base_prompt_uses_per_template_criteria_when_cell_has_them():
    prompt = elicit.build_base_prompt(
        "REPORT BODY", CELL_TPL["question"], 120,
        CELL_TPL["template_id"], CELL_TPL["parameters"])
    assert ("Resolution Criteria: Resolves YES if Benin's recorded number of "
            "cities is strictly greater than Jolof's at exactly turn 120") in prompt
    assert "An exact tie resolves NO." in prompt
    assert prompt.count("Resolution Criteria:") == 1


def test_reveal_turn_pinned_exactly():
    msg = elicit.build_reveal_message(CELL)
    assert msg == (
        "The game has continued. One fact about turns 75-90 has been "
        "revealed to you: Benin discovered Feudalism. Given this news and "
        "everything you already knew, provide an updated forecast for the "
        "SAME question: Will Benin control the most cities at turn 120?. "
        "Report your honest updated probability. Your response SHOULD "
        "STRICTLY END with <probability> </probability> tags.")
    assert "suppose" not in msg.lower()  # v1 hypothetical wording is gone
    assert "log loss" not in msg.lower()  # v2/v3 scoring sentence is gone


def test_nonews_turn_pinned_exactly():
    msg = elicit.build_nonews_message(CELL["question"], (60, 90))
    assert msg == (
        "The game has continued. No new information about turns 60-90 is "
        "available. If you wish, revise your forecast for the SAME "
        "question: Will Benin control the most cities at turn 120?. Report "
        "your honest probability. Your response SHOULD STRICTLY END with "
        "<probability> </probability> tags.")


def test_family_window_spans_the_family_reveal_windows():
    fam = [dict(CELL, window=[60, 75]), dict(CELL, window=[75, 90])]
    assert elicit.family_window(fam) == (60, 90)
    assert elicit.family_window([CELL]) == (75, 90)


def test_single_turn_reveal_matches_turn2_opening_sentence():
    # conformance: the single-turn insert IS the turn-2 opening, verbatim —
    # future edits must not diverge them
    assert elicit.REVEAL_TEMPLATE.startswith(elicit.SINGLE_TURN_REVEAL)
    rendered = elicit.SINGLE_TURN_REVEAL.format(
        a=75, b=90, event=CELL["event_desc"])
    assert elicit.build_reveal_message(CELL).startswith(rendered)
    assert rendered in elicit.build_single_turn_prompt("REPORT BODY", CELL, 120)


def test_single_turn_prompt_inserts_reveal_between_report_and_criteria():
    prompt = elicit.build_single_turn_prompt("REPORT BODY", CELL, 120)
    assert ("REPORT BODY\n"
            "\n"
            "The game has continued. One fact about turns 75-90 has been "
            "revealed to you: Benin discovered Feudalism.\n"
            "\n"
            "Resolution Criteria:") in prompt
    assert "Question Title: " + CELL["question"] in prompt
    assert "suppose" not in prompt.lower()
    # apart from the inserted line, the prompt is the base prompt verbatim
    base = elicit.build_base_prompt("REPORT BODY", CELL["question"], 120)
    insert = ("\n\nThe game has continued. One fact about turns 75-90 has "
              "been revealed to you: Benin discovered Feudalism.")
    assert prompt == base.replace("REPORT BODY", "REPORT BODY" + insert)


def test_single_turn_prompt_uses_cell_template_criteria():
    prompt = elicit.build_single_turn_prompt("REPORT BODY", CELL_TPL, 120)
    assert "strictly greater" in prompt and "An exact tie resolves NO." in prompt


def test_resolution_criteria_render_per_horizon_turn():
    # derived-qid cells carry their own resolution_turn; the criteria and
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
        # per-template path states the derived turn too
        tpl = dict(CELL_TPL, resolution_turn=rt)
        tpl_prompt = elicit.build_single_turn_prompt("R", tpl, rt)
        assert f"at exactly turn {rt}" in tpl_prompt


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
