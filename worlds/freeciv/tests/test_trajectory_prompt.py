"""
Tests for trajectory (per-time-series) question format.

The trajectory format groups questions by template and presents all horizons
together, rather than interleaving templates. The response format stays flat
(Q1, Q2, ... QN) so existing parsing works unchanged.
"""

import pytest
from freeciv_world.evaluation.parallel_evaluator import parse_batch_percentiles


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_question(template_id: str, resolution_turn: int, civ: str = "Pontic") -> dict:
    """Build a minimal question dict matching the real data schema."""
    text_templates = {
        "population_continuous": f"What will {civ}'s population be at turn {resolution_turn}?",
        "territory_continuous": f"How many tiles will {civ} control at turn {resolution_turn}?",
        "treasury_continuous": f"How much gold will {civ} have at turn {resolution_turn}?",
        "cities_count_continuous": f"How many cities will {civ} have at turn {resolution_turn}?",
    }
    return {
        "question_id": f"{template_id}_t{resolution_turn}",
        "template_id": template_id,
        "resolution_turn": resolution_turn,
        "question_type": "continuous",
        "question_text": text_templates[template_id],
        "resolution": {"value_at_resolution": 100},
    }


DISRUPTABLE_TEMPLATES = [
    "population_continuous",
    "territory_continuous",
    "treasury_continuous",
    "cities_count_continuous",
]
HORIZONS = [90, 120, 150, 180, 210, 240]


@pytest.fixture
def interleaved_questions():
    """Questions in arbitrary/interleaved order (current baseline format)."""
    questions = []
    for turn in HORIZONS:
        for tmpl in DISRUPTABLE_TEMPLATES:
            questions.append(_make_question(tmpl, turn))
    return questions


@pytest.fixture
def trajectory_questions():
    """Questions grouped by template, sorted by horizon within each group."""
    questions = []
    for tmpl in DISRUPTABLE_TEMPLATES:
        for turn in HORIZONS:
            questions.append(_make_question(tmpl, turn))
    return questions


# ── Tests: group_questions_by_template ────────────────────────────────────────


class TestGroupQuestionsByTemplate:
    """Tests for the grouping function that reorders questions into trajectory format."""

    def test_groups_by_template(self, interleaved_questions):
        grouped = group_questions_by_template(interleaved_questions)
        # Should produce one group per template
        assert len(grouped) == 4

    def test_each_group_sorted_by_resolution_turn(self, interleaved_questions):
        grouped = group_questions_by_template(interleaved_questions)
        for tmpl, questions in grouped:
            turns = [q["resolution_turn"] for q in questions]
            assert turns == sorted(turns), f"{tmpl} not sorted by turn: {turns}"

    def test_preserves_all_questions(self, interleaved_questions):
        grouped = group_questions_by_template(interleaved_questions)
        total = sum(len(qs) for _, qs in grouped)
        assert total == len(interleaved_questions)

    def test_no_cross_template_contamination(self, interleaved_questions):
        grouped = group_questions_by_template(interleaved_questions)
        for tmpl, questions in grouped:
            for q in questions:
                assert q["template_id"] == tmpl


# ── Tests: build_trajectory_continuous_prompt ─────────────────────────────────


class TestBuildTrajectoryContinuousPrompt:
    """Tests for trajectory prompt construction."""

    def test_questions_numbered_sequentially(self, interleaved_questions):
        prompt = build_trajectory_continuous_prompt(interleaved_questions, "fake report")
        # Should contain Q1 through Q24
        for i in range(1, len(interleaved_questions) + 1):
            assert f"{i}." in prompt, f"Missing question number {i}"

    def test_questions_grouped_in_prompt(self, interleaved_questions):
        prompt = build_trajectory_continuous_prompt(interleaved_questions, "fake report")
        # All population questions should appear before all territory questions, etc.
        # Find positions of first question of each template
        positions = {}
        for tmpl in DISRUPTABLE_TEMPLATES:
            sample_text = _make_question(tmpl, 90)["question_text"]
            pos = prompt.find(sample_text)
            assert pos != -1, f"Missing question text for {tmpl}"
            positions[tmpl] = pos

        # Verify ordering: all questions of template N appear before template N+1
        template_positions = [positions[t] for t in DISRUPTABLE_TEMPLATES]
        assert template_positions == sorted(template_positions)

    def test_horizons_ordered_within_group(self, interleaved_questions):
        prompt = build_trajectory_continuous_prompt(interleaved_questions, "fake report")
        # For each template, check that turn 90 appears before turn 240
        for tmpl in DISRUPTABLE_TEMPLATES:
            early = _make_question(tmpl, 90)["question_text"]
            late = _make_question(tmpl, 240)["question_text"]
            assert prompt.find(early) < prompt.find(late), (
                f"{tmpl}: turn 90 should appear before turn 240"
            )

    def test_contains_trajectory_headers(self, interleaved_questions):
        prompt = build_trajectory_continuous_prompt(interleaved_questions, "fake report")
        # Should have some kind of grouping header for each template
        assert "population" in prompt.lower()
        assert "treasury" in prompt.lower()

    def test_contains_world_report(self, interleaved_questions):
        prompt = build_trajectory_continuous_prompt(interleaved_questions, "my world report data")
        assert "my world report data" in prompt

    def test_contains_percentile_instructions(self, interleaved_questions):
        prompt = build_trajectory_continuous_prompt(interleaved_questions, "fake report")
        assert "<<<PERCENTILES>>>" in prompt
        assert "<<<END>>>" in prompt

    def test_question_count_matches(self, interleaved_questions):
        prompt = build_trajectory_continuous_prompt(interleaved_questions, "fake report")
        assert f"all {len(interleaved_questions)} questions" in prompt


# ── Tests: parsing trajectory responses ───────────────────────────────────────


class TestParseTrajectoryResponses:
    """Verify that existing parse_batch_percentiles handles trajectory-format responses.

    The response format is unchanged (flat Q1-QN lines), so this is really a
    confirmation that the existing parser works with the expected output.
    """

    def test_parses_full_trajectory_response(self):
        """24 questions (4 templates × 6 horizons) in standard delimiter format."""
        lines = []
        for i in range(1, 25):
            lines.append(
                f"Q{i}: p10={i*10}, p25={i*20}, p50={i*30}, p75={i*40}, p90={i*50}"
            )
        response = (
            "Here is my analysis...\n\n"
            "<<<PERCENTILES>>>\n"
            + "\n".join(lines)
            + "\n<<<END>>>"
        )
        results = parse_batch_percentiles(response, 24)
        assert len(results) == 24
        assert all(r is not None for r in results)
        # Spot check first and last
        assert results[0] == {"p10": 10, "p25": 20, "p50": 30, "p75": 40, "p90": 50}
        assert results[23] == {"p10": 240, "p25": 480, "p50": 720, "p75": 960, "p90": 1200}

    def test_parses_without_q_prefix(self):
        """Some models omit the Q prefix — just numbered lines."""
        lines = []
        for i in range(1, 25):
            lines.append(f"{i}. p10={i*5}, p25={i*10}, p50={i*15}, p75={i*20}, p90={i*25}")
        response = "<<<PERCENTILES>>>\n" + "\n".join(lines) + "\n<<<END>>>"
        results = parse_batch_percentiles(response, 24)
        assert len(results) == 24
        assert all(r is not None for r in results)

    def test_parses_partial_response(self):
        """If model only outputs 20 of 24 lines, remaining should be None."""
        lines = []
        for i in range(1, 21):
            lines.append(f"Q{i}: p10={i}, p25={i*2}, p50={i*3}, p75={i*4}, p90={i*5}")
        response = "<<<PERCENTILES>>>\n" + "\n".join(lines) + "\n<<<END>>>"
        results = parse_batch_percentiles(response, 24)
        assert len(results) == 24
        assert all(r is not None for r in results[:20])
        assert all(r is None for r in results[20:])

    def test_parses_response_with_group_headers_mixed_in(self):
        """Model might echo group headers back — parser should skip non-data lines."""
        response = """<<<PERCENTILES>>>
Population trajectory:
Q1: p10=10, p25=20, p50=30, p75=40, p90=50
Q2: p10=15, p25=25, p50=35, p75=45, p90=55
Territory trajectory:
Q3: p10=100, p25=200, p50=300, p75=400, p90=500
Q4: p10=150, p25=250, p50=350, p75=450, p90=550
<<<END>>>"""
        results = parse_batch_percentiles(response, 4)
        assert len(results) == 4
        assert all(r is not None for r in results)
        assert results[0]["p50"] == 30
        assert results[2]["p50"] == 300


# ── Import hook: these will fail until implementation exists ───────────────────

try:
    from freeciv_world.evaluation.trajectory_prompt import (
        group_questions_by_template,
        build_trajectory_continuous_prompt,
    )
except ImportError:
    # Mark grouping/prompt tests as expected failures until implemented
    for cls in [TestGroupQuestionsByTemplate, TestBuildTrajectoryContinuousPrompt]:
        for name in dir(cls):
            if name.startswith("test_"):
                method = getattr(cls, name)
                setattr(cls, name, pytest.mark.xfail(
                    reason="trajectory_prompt module not yet implemented",
                    raises=(NameError, ImportError),
                )(method))
