"""Unit tests for the batched single-city eval.

Covers the batch prompt builder, the batch response parser, and the cache
layout helpers. Nothing here calls a model or writes to disk; the only I/O is
reading a stored real model response from fixtures/ — except the questions
sort tests, which build a small corpus and so run the simulator.
"""

from pathlib import Path

import pytest

from micropolis_world.config import QUESTIONS_SORT_METRIC
from micropolis_world.scenarios import (
    DEFAULT_EPILOGUE_PATH,
    PERCENTILE_KEYS,
    build_batch_prompt_continuous,
    build_corpus,
    get_single_city_base_scenarios,
    parse_batch_percentiles,
)
from micropolis_world.single_city import batch_id_for, response_path
from micropolis_world.templates import ALL_TEMPLATES

FIXTURES_DIR = Path(__file__).parent / "fixtures"

REPORT = "city report: population 1234, funds 5000"

QUESTIONS = [
    {
        "question_id": f"bruce_s42_T240_H48_metric{i}",
        "question_text": f"What will metric {i} be at turn 288?",
        "scenario_id": "bruce_s42",
        "snapshot_turn": 240,
    }
    for i in range(1, 4)
]
LABELS = [f"test-model {q['question_id']}" for q in QUESTIONS]

SET1 = {"p10": 5.0, "p25": 10.0, "p50": 15.0, "p75": 20.0, "p90": 25.0}
SET2 = {"p10": 100.0, "p25": 200.0, "p50": 300.0, "p75": 400.0, "p90": 500.0}
SET3 = {"p10": 1.0, "p25": 2.0, "p50": 3.0, "p75": 4.0, "p90": 5.0}


def as_line(s: dict) -> str:
    return ", ".join(f"{k}={v:g}" for k, v in s.items())


def delimited(*lines: str) -> str:
    """A response in the requested format, with reasoning above the block."""
    return (
        "Considering the report, p50 is probably around 7 for most of these.\n"
        "<<<PERCENTILES>>>\n" + "\n".join(lines) + "\n<<<END>>>"
    )


class TestParseBatchPercentiles:
    def test_q_numbered_lines(self):
        response = delimited(
            f"Q1: {as_line(SET1)}", f"Q2: {as_line(SET2)}", f"Q3: {as_line(SET3)}"
        )
        assert parse_batch_percentiles(response, LABELS) == [SET1, SET2, SET3]

    def test_dot_and_paren_numbered_lines(self):
        response = delimited(
            f"1. {as_line(SET1)}", f"2) {as_line(SET2)}", f"3: {as_line(SET3)}"
        )
        assert parse_batch_percentiles(response, LABELS) == [SET1, SET2, SET3]

    def test_plain_lines_fall_back_to_positional(self):
        response = delimited(as_line(SET1), as_line(SET2), as_line(SET3))
        assert parse_batch_percentiles(response, LABELS) == [SET1, SET2, SET3]

    def test_numbered_lines_ignore_order_and_skips(self):
        # Q2 is skipped and the others arrive out of order: the stated numbers,
        # not the positions, decide where each answer lands.
        response = delimited(f"Q3: {as_line(SET3)}", f"Q1: {as_line(SET1)}")
        assert parse_batch_percentiles(response, LABELS, quiet=True) == [
            SET1,
            None,
            SET3,
        ]

    def test_numbered_bare_numbers(self):
        response = delimited("Q1: 5, 10, 15, 20, 25", f"Q2: {as_line(SET2)}")
        assert parse_batch_percentiles(response, LABELS, quiet=True) == [
            SET1,
            SET2,
            None,
        ]

    def test_non_monotonic_set_dropped_alone(self):
        response = delimited(
            f"Q1: {as_line(SET1)}",
            "Q2: p10=500, p25=400, p50=300, p75=200, p90=100",
            f"Q3: {as_line(SET3)}",
        )
        assert parse_batch_percentiles(response, LABELS, quiet=True) == [
            SET1,
            None,
            SET3,
        ]

    def test_empty_response(self):
        assert parse_batch_percentiles(None, LABELS, quiet=True) == [None] * 3
        assert parse_batch_percentiles("", LABELS, quiet=True) == [None] * 3

    def test_no_block_still_parses_numbered_lines(self):
        response = f"Q1: {as_line(SET1)}\nQ2: {as_line(SET2)}\nQ3: {as_line(SET3)}"
        assert parse_batch_percentiles(response, LABELS) == [SET1, SET2, SET3]

    def test_single_question_uses_single_format(self):
        response = delimited(as_line(SET1))
        assert parse_batch_percentiles(response, LABELS[:1]) == [SET1]

    def test_prose_numbers_are_not_answers(self):
        # A numbered list in the reasoning must not be read as answers when the
        # real answers never come.
        response = (
            "1. The population will likely grow.\n"
            "2. Funds are stable near 5000, 6000, 7000, 8000, 9000 range.\n"
        )
        parsed = parse_batch_percentiles(response, LABELS, quiet=True)
        assert parsed == [None, None, None]

    def test_real_model_response_fixture(self):
        # A real batched response, captured verbatim from claude-haiku-4-5 on
        # the bruce_nodisasters_seed42_T240 batch (9 questions, all answered).
        fixture = FIXTURES_DIR / "batch_response_9_questions.txt"
        labels = [f"fixture q{i}" for i in range(1, 10)]
        parsed = parse_batch_percentiles(fixture.read_text(), labels)
        assert all(p is not None for p in parsed)
        for p in parsed:
            assert sorted(p) == sorted(PERCENTILE_KEYS)


class TestBuildBatchPrompt:
    def test_multi_question_prompt(self):
        prompt = build_batch_prompt_continuous(REPORT, QUESTIONS)
        assert prompt.count(REPORT) == 1
        assert "## Questions" in prompt
        for i, q in enumerate(QUESTIONS, 1):
            assert f"{i}. {q['question_text']}" in prompt
        assert "each of the 3 questions" in prompt
        assert "Q1: p10=5" in prompt  # the answer-format example

    def test_single_question_prompt(self):
        # A one-question batch takes the same numbered form as any other, so
        # there is a single prompt shape for the parser to read back.
        prompt = build_batch_prompt_continuous(REPORT, QUESTIONS[:1])
        assert "## Questions" in prompt
        assert f"1. {QUESTIONS[0]['question_text']}" in prompt
        assert "each of the 1 questions" in prompt

    def test_epilogue_path_overrides_the_default(self, tmp_path):
        epilogue = tmp_path / "epilogue-test.txt"
        epilogue.write_text("Answer all {n} of them. {n} lines, please.")
        prompt = build_batch_prompt_continuous(
            REPORT, QUESTIONS, epilogue_path=epilogue
        )
        assert prompt.endswith("Answer all 3 of them. 3 lines, please.")
        assert "<<<PERCENTILES>>>" not in prompt  # the default epilogue is gone

    def test_epilogue_without_placeholder_is_used_verbatim(self, tmp_path):
        epilogue = tmp_path / "epilogue-no-n.txt"
        epilogue.write_text('Reply as {"p50": 1}.')
        prompt = build_batch_prompt_continuous(
            REPORT, QUESTIONS, epilogue_path=epilogue
        )
        assert prompt.endswith('Reply as {"p50": 1}.')

    def test_default_epilogue_file_matches_the_prompt(self):
        # The shipped default is what a config gets when it names no
        # epilogue_path, so the prompt must end with it.
        assert build_batch_prompt_continuous(REPORT, QUESTIONS).endswith(
            DEFAULT_EPILOGUE_PATH.read_text(encoding="utf-8").replace("{n}", "3")
        )


class TestQuestionsSort:
    """The two orders build_corpus can number a batch's questions in."""

    HORIZONS = [12, 24]

    def _corpus(self, **kw):
        scenarios = get_single_city_base_scenarios(
            seed=42, cities=["bruce"], disasters=[False]
        )
        return build_corpus(
            scenarios,
            [240],
            self.HORIZONS,
            history_freq=12,
            label="sorttest",
            **kw,
        )

    def test_turn_order_is_the_default(self):
        pairs = [(q["horizon"], q["metric"]) for q in self._corpus()]
        assert pairs == [
            (h, t.signal_name) for h in self.HORIZONS for t in ALL_TEMPLATES
        ]

    def test_metric_order_groups_horizons_per_metric(self):
        corpus = self._corpus(questions_sort=QUESTIONS_SORT_METRIC)
        pairs = [(q["horizon"], q["metric"]) for q in corpus]
        assert pairs == [
            (h, t.signal_name) for t in ALL_TEMPLATES for h in self.HORIZONS
        ]

    def test_both_orders_ask_the_same_questions(self):
        # Only the numbering changes, so a reordered run must resolve to the
        # same values — otherwise the two orders would not be comparable.
        by_turn = self._corpus()
        by_metric = self._corpus(questions_sort=QUESTIONS_SORT_METRIC)
        assert {q["question_id"]: q["value"] for q in by_turn} == {
            q["question_id"]: q["value"] for q in by_metric
        }

    def test_unknown_sort_is_rejected(self):
        with pytest.raises(ValueError, match="questions_sort"):
            self._corpus(questions_sort="sideways")


class TestBatchCacheHelpers:
    def test_batch_id_groups_by_scenario_and_snapshot(self):
        assert batch_id_for(QUESTIONS[0]) == "bruce_s42_T240"
        other_snapshot = dict(QUESTIONS[0], snapshot_turn=480)
        other_scenario = dict(QUESTIONS[0], scenario_id="kobe_s42")
        assert batch_id_for(other_snapshot) == "bruce_s42_T480"
        assert batch_id_for(other_scenario) == "kobe_s42_T240"

    def test_response_path_sanitizes_model_id(self):
        path = response_path("bruce_s42_T240", "openai/gpt-4o-2024-05-13", "abcd1234")
        assert path.name == "response-openai_gpt-4o-2024-05-13-abcd1234.txt"
        assert path.parent.name == "bruce_s42_T240"
