"""Tests for scripts/uplift_v2/resolution_criteria.py.

Two layers:
  1. pinned snapshots — one exact rendered string per binary template, so any
     wording drift is a deliberate, reviewed change;
  2. resolver cross-checks — the claims the criteria make (strict >, tie->NO,
     inclusive/exclusive window bounds, changed-and-changed-back, tie-for-
     first) are executed against the REAL fbsim_core resolver with the
     freeciv TemplateRegistry on constructed game data. The criteria follow
     the resolver; if a cross-check fails, fix the criteria, not the test.
"""
import pytest
from conftest import load_script

from fbsim_core.questions.resolver import QuestionResolver
from fbsim_core.questions.schema import QuestionInstance
from freeciv_world.world_reports.questions import templates as fc_templates

rc = load_script("scripts/uplift_v2/resolution_criteria.py")
dh = load_script("scripts/uplift_v2/derive_horizon_bank.py")

RESOLVER = QuestionResolver(fc_templates.REGISTRY)

COMP_PARAMS = {"civ_a": "Benin", "civ_b": "Jolof", "player_id_a": 0,
               "player_id_b": 1, "resolution_turn": 90}


def resolve(template_id, parameters, rt, game_data, snapshot_turn=60):
    q = QuestionInstance(
        question_id="qx", template_id=template_id, resolution_turn=rt,
        horizon="H1", parameters=parameters, question_text="?")
    return RESOLVER.resolve(q, game_data, snapshot_turn).answer


# ---------------------------------------------------------------- snapshots

PINNED = {
    ("tech_comparative", 90): (
        "Resolves YES if Benin's recorded count of known technologies is "
        "strictly greater than Jolof's at exactly turn 90 (both values are "
        "read at exactly that turn; values at earlier or later turns do not "
        "matter). An exact tie resolves NO. If either civilization's value "
        "is missing from the record at turn 90, the question resolves NO. "
        "Resolution is as determined by the game's recorded metrics."),
    ("score_comparative", 90): (
        "Resolves YES if Benin's recorded game score is strictly greater "
        "than Jolof's at exactly turn 90 (both values are read at exactly "
        "that turn; values at earlier or later turns do not matter). An "
        "exact tie resolves NO. If either civilization's value is missing "
        "from the record at turn 90, the question resolves NO. Resolution "
        "is as determined by the game's recorded metrics."),
    ("population_comparative", 90): (
        "Resolves YES if Benin's recorded total population is strictly "
        "greater than Jolof's at exactly turn 90 (both values are read at "
        "exactly that turn; values at earlier or later turns do not "
        "matter). An exact tie resolves NO. If either civilization's value "
        "is missing from the record at turn 90, the question resolves NO. "
        "Resolution is as determined by the game's recorded metrics."),
    ("city_count_comparative", 90): (
        "Resolves YES if Benin's recorded number of cities is strictly "
        "greater than Jolof's at exactly turn 90 (both values are read at "
        "exactly that turn; values at earlier or later turns do not "
        "matter). An exact tie resolves NO. If either civilization's value "
        "is missing from the record at turn 90, the question resolves NO. "
        "Resolution is as determined by the game's recorded metrics."),
    ("territory_comparative", 90): (
        "Resolves YES if Benin's recorded number of map tiles controlled is "
        "strictly greater than Jolof's at exactly turn 90 (both values are "
        "read at exactly that turn; values at earlier or later turns do not "
        "matter). An exact tie resolves NO. If either civilization's value "
        "is missing from the record at turn 90, the question resolves NO. "
        "Resolution is as determined by the game's recorded metrics."),
    ("treasury_comparative", 90): (
        "Resolves YES if Benin's recorded treasury (gold reserve) is "
        "strictly greater than Jolof's at exactly turn 90 (both values are "
        "read at exactly that turn; values at earlier or later turns do not "
        "matter). An exact tie resolves NO. If either civilization's value "
        "is missing from the record at turn 90, the question resolves NO. "
        "Resolution is as determined by the game's recorded metrics."),
    ("score_rank_1", 120): (
        "Resolves YES if Benin holds rank #1 in the game's recorded score "
        "rankings at exactly turn 120 (the ranking is read at exactly that "
        "turn). Exactly one civilization holds rank #1 even when the top "
        "score is tied: the recorded ranking breaks ties by the order in "
        "which players appear in the game record, so Benin being tied for "
        "the top score resolves YES only if the record ranks Benin #1. If "
        "Benin has no recorded rank at turn 120, the question resolves NO. "
        "Resolution is as determined by the game's recorded metrics."),
    ("tech_discovered", 90): (
        "Resolves YES if the game record shows Benin discovering Feudalism "
        "at any turn up to and including turn 90; a discovery recorded at "
        "any earlier point in the game counts, including turns before the "
        "report's snapshot. Resolves NO if no such discovery by Benin is "
        "recorded by turn 90. Resolution is as determined by the game's "
        "recorded metrics."),
    ("wonder_completed", 90): (
        "Resolves YES if the game record shows Great Wall being completed "
        "by any civilization at any turn up to and including turn 90, "
        "counting only completions strictly after turn 60 (a completion "
        "recorded at or before turn 60 does not count). Resolves NO "
        "otherwise. Resolution is as determined by the game's recorded "
        "metrics."),
    ("government_at", 90): (
        "Resolves YES if Benin's government at exactly turn 90 is Monarchy. "
        "The government at turn 90 is the destination of the most recent "
        "government change recorded for Benin at any turn up to and "
        "including turn 90; changing into Monarchy and back out again "
        "before turn 90 does not count — only the government in effect at "
        "turn 90 matters. If no government change for Benin is recorded by "
        "turn 90, the question resolves NO regardless of Benin's starting "
        "government. Resolution is as determined by the game's recorded "
        "metrics."),
}

PARAMS = {
    "score_rank_1": {"civ": "Benin", "player_id": 0, "resolution_turn": 120},
    "tech_discovered": {"civ": "Benin", "player_id": 0,
                        "tech_name": "Feudalism", "tech_id": "20",
                        "resolution_turn": 90},
    "wonder_completed": {"wonder_name": "Great Wall", "wonder_id": 47,
                         "snapshot_turn": 60, "resolution_turn": 90},
    "government_at": {"civ": "Benin", "player_id": 0,
                      "government_type": "Monarchy", "resolution_turn": 90},
}


@pytest.mark.parametrize("tid,rt", sorted(PINNED))
def test_pinned_snapshot(tid, rt):
    params = PARAMS.get(tid, COMP_PARAMS)
    assert rc.render_criteria(tid, params, rt) == PINNED[(tid, rt)]


def test_covers_exactly_the_active_binary_templates():
    active = {t.template_id for t in fc_templates.ALL_TEMPLATES}
    assert set(rc.BINARY_TEMPLATE_IDS) == active
    assert len(rc.BINARY_TEMPLATE_IDS) == 10


def test_every_criteria_string_ends_with_the_required_phrase():
    for tid in rc.BINARY_TEMPLATE_IDS:
        params = PARAMS.get(tid, COMP_PARAMS)
        out = rc.render_criteria(tid, params, 90)
        assert out.endswith("as determined by the game's recorded metrics."), tid
    assert rc.GENERIC_CRITERIA.format(rt=90).endswith(
        "as determined by the game's recorded metrics.")


def test_derived_horizon_variants_reuse_base_template_ids():
    # derive_horizon_bank re-emits H1 questions at t120..t210 with the SAME
    # template_id and a rewritten parameters.resolution_turn; the renderer
    # must state the derived turn (and keep the t60 wonder window bound)
    q_comp = {"question_id": "q0000", "template_id": "tech_comparative",
              "horizon": "H1", "resolution_turn": 90,
              "question_text": "Will Benin have more technologies than "
                               "Jolof at turn 90?",
              "parameters": dict(COMP_PARAMS)}
    q_wonder = {"question_id": "q0044", "template_id": "wonder_completed",
                "horizon": "H1", "resolution_turn": 90,
                "question_text": "Will Great Wall be completed by any "
                                 "civilization by turn 90?",
                "parameters": dict(PARAMS["wonder_completed"])}
    for hz, rt in sorted(dh.DERIVED_HORIZONS.items()):
        d = dh.derive_question(q_comp, hz, rt)
        out = rc.render_criteria(d["template_id"], d["parameters"],
                                 d["resolution_turn"])
        assert f"at exactly turn {rt}" in out and "turn 90" not in out
        w = dh.derive_question(q_wonder, hz, rt)
        wout = rc.render_criteria(w["template_id"], w["parameters"],
                                  w["resolution_turn"])
        assert f"up to and including turn {rt}" in wout
        assert "strictly after turn 60" in wout  # snapshot bound stays t60


def test_h0_prefix_resolves_through_base_template():
    assert rc.render_criteria("h0_tech_comparative", COMP_PARAMS, 90) == \
        rc.render_criteria("tech_comparative", COMP_PARAMS, 90)


def test_unknown_template_raises_and_fallback_is_generic():
    with pytest.raises(ValueError):
        rc.render_criteria("at_war_dyad", {"civ_a": "A", "civ_b": "B"}, 90)
    with pytest.raises(ValueError):
        rc.render_criteria("tech_comparative", {"civ_a": "A"}, 90)  # no civ_b
    generic = ("Resolves YES if the answer to the question is affirmative in "
               "the simulation state at turn 90, as determined by the game's "
               "recorded metrics.")
    assert rc.criteria_or_generic(None, None, 90) == generic
    assert rc.criteria_or_generic("no_such_template", {}, 90) == generic
    assert rc.criteria_or_generic("tech_comparative", None, 90) == generic
    assert rc.criteria_or_generic("tech_comparative", COMP_PARAMS, 90) == \
        PINNED[("tech_comparative", 90)]


# ------------------------------------------------- resolver cross-checks


def test_comparative_is_strict_and_tie_resolves_no():
    tie = {"time_series": {"techs_known": {"90": {"0": 5, "1": 5}}}}
    ahead = {"time_series": {"techs_known": {"90": {"0": 6, "1": 5}}}}
    behind = {"time_series": {"techs_known": {"90": {"0": 5, "1": 6}}}}
    assert resolve("tech_comparative", COMP_PARAMS, 90, tie) is False
    assert resolve("tech_comparative", COMP_PARAMS, 90, ahead) is True
    assert resolve("tech_comparative", COMP_PARAMS, 90, behind) is False
    out = rc.render_criteria("tech_comparative", COMP_PARAMS, 90)
    assert "strictly greater" in out and "An exact tie resolves NO." in out


def test_comparative_reads_exactly_the_resolution_turn():
    decoys = {"time_series": {"techs_known": {
        "89": {"0": 99, "1": 0},   # decoy either side of T
        "90": {"0": 5, "1": 5},
        "91": {"0": 99, "1": 0}}}}
    assert resolve("tech_comparative", COMP_PARAMS, 90, decoys) is False
    no_t = {"time_series": {"techs_known": {"89": {"0": 9, "1": 5}}}}
    assert resolve("tech_comparative", COMP_PARAMS, 90, no_t) is False
    missing_b = {"time_series": {"techs_known": {"90": {"0": 6}}}}
    assert resolve("tech_comparative", COMP_PARAMS, 90, missing_b) is False
    out = rc.render_criteria("tech_comparative", COMP_PARAMS, 90)
    assert "at exactly turn 90" in out
    assert "missing from the record at turn 90, the question resolves NO" in out


def _tech_event(turn):
    return {"type": "tech_discovered", "player_id": 0, "turn": turn,
            "metadata": {"tech_name": "Feudalism", "tech_id": "20"}}


def test_tech_discovered_cumulative_and_inclusive_at_t():
    p = PARAMS["tech_discovered"]
    assert resolve("tech_discovered", p, 90, {"events": [_tech_event(90)]}) is True
    assert resolve("tech_discovered", p, 90, {"events": [_tech_event(91)]}) is False
    # cumulative over the whole game: a pre-snapshot discovery counts
    assert resolve("tech_discovered", p, 90, {"events": [_tech_event(10)]}) is True
    assert resolve("tech_discovered", p, 90, {"events": []}) is False
    out = rc.render_criteria("tech_discovered", p, 90)
    assert "up to and including turn 90" in out
    assert "including turns before the report's snapshot" in out


def _wonder_event(turn):
    return {"type": "wonder_completed", "player_id": 2, "turn": turn,
            "metadata": {"wonder_name": "Great Wall", "wonder_id": 47}}


def test_wonder_completed_window_is_snapshot_exclusive_t_inclusive():
    # RESOLVER FACT (differs from tech_discovered): the event window is
    # (snapshot_turn, resolution_turn] — a completion at or before the
    # snapshot turn does NOT count. The criteria must say so.
    p = PARAMS["wonder_completed"]
    assert resolve("wonder_completed", p, 90, {"events": [_wonder_event(90)]}) is True
    assert resolve("wonder_completed", p, 90, {"events": [_wonder_event(61)]}) is True
    assert resolve("wonder_completed", p, 90, {"events": [_wonder_event(60)]}) is False
    assert resolve("wonder_completed", p, 90, {"events": [_wonder_event(40)]}) is False
    assert resolve("wonder_completed", p, 90, {"events": [_wonder_event(91)]}) is False
    out = rc.render_criteria("wonder_completed", p, 90)
    assert "up to and including turn 90" in out
    assert "strictly after turn 60" in out
    assert "any civilization" in out  # completion by anyone counts


def _gov_event(turn, to):
    return {"type": "government_change", "player_id": 0, "turn": turn,
            "metadata": {"to": to}}


def test_government_at_changed_and_changed_back_before_t():
    p = PARAMS["government_at"]  # target Monarchy, rt 90
    p_desp = dict(p, government_type="Despotism")
    flip_flop = {"events": [_gov_event(70, "Monarchy"),
                            _gov_event(85, "Despotism")]}
    assert resolve("government_at", p, 90, flip_flop) is False
    assert resolve("government_at", p_desp, 90, flip_flop) is True
    assert resolve("government_at", p, 90,
                   {"events": [_gov_event(90, "Monarchy")]}) is True
    assert resolve("government_at", p, 90,
                   {"events": [_gov_event(91, "Monarchy")]}) is False
    out = rc.render_criteria("government_at", p, 90)
    assert "most recent government change" in out
    assert "back out again before turn 90 does not count" in out


def test_government_at_no_recorded_change_resolves_no_even_for_start_gov():
    # RESOLVER FACT: with no recorded government_change the resolver answers
    # NO for EVERY government, including the civ's actual starting one.
    p_desp = dict(PARAMS["government_at"], government_type="Despotism")
    assert resolve("government_at", p_desp, 90, {"events": []}) is False
    out = rc.render_criteria("government_at", p_desp, 90)
    assert ("If no government change for Benin is recorded by turn 90, the "
            "question resolves NO regardless of Benin's starting "
            "government.") in out


def test_score_rank_1_tie_for_first_is_broken_by_record_order():
    # RESOLVER FACT: exactly one civ holds rank #1 even on a tied top score;
    # the stable descending sort breaks ties by player order in the record.
    p0 = {"civ": "Benin", "player_id": 0, "resolution_turn": 90}
    p1 = {"civ": "Jolof", "player_id": 1, "resolution_turn": 90}
    tie = {"time_series": {"scores": {"90": {"0": 100, "1": 100, "2": 50}}}}
    assert resolve("score_rank_1", p0, 90, tie) is True
    assert resolve("score_rank_1", p1, 90, tie) is False
    flipped = {"time_series": {"scores": {"90": {"1": 100, "0": 100, "2": 50}}}}
    assert resolve("score_rank_1", p0, 90, flipped) is False
    assert resolve("score_rank_1", p1, 90, flipped) is True
    # recorded snapshot rankings take precedence and their rank field is
    # trusted as-is
    snap = {"snapshots": {"90": {"rankings": [
        {"rank": 1, "player_id": 1, "score": 100},
        {"rank": 2, "player_id": 0, "score": 100}]}}}
    assert resolve("score_rank_1", p1, 90, snap) is True
    assert resolve("score_rank_1", p0, 90, snap) is False
    # absent civ -> NO
    assert resolve("score_rank_1", {"civ": "X", "player_id": 9,
                                    "resolution_turn": 90}, 90, tie) is False
    out = rc.render_criteria("score_rank_1", p0, 90)
    assert "Exactly one civilization holds rank #1" in out
    assert ("being tied for the top score resolves YES only if the record "
            "ranks Benin #1") in out


def test_score_rank_1_reads_exactly_the_resolution_turn():
    p0 = {"civ": "Benin", "player_id": 0, "resolution_turn": 90}
    other_turn = {"time_series": {"scores": {"89": {"0": 100, "1": 50}}}}
    assert resolve("score_rank_1", p0, 90, other_turn) is False
    at_t = {"time_series": {"scores": {"90": {"0": 100, "1": 50}}}}
    assert resolve("score_rank_1", p0, 90, at_t) is True
    assert "at exactly turn 90" in rc.render_criteria("score_rank_1", p0, 90)
