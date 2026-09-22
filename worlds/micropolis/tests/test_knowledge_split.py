"""The knowledge test's statement set and its split into prompts.

A true statement and its false twin must never share a prompt, every statement
must be asked exactly once, and a reply to a half must land back on the right
statements. Nothing here calls a model or touches disk.
"""

from micropolis_world.knowledge_eval import statements as st
from micropolis_world.knowledge_eval.runner import (
    HALVES,
    Answer,
    build_prompts,
    merge_answers,
    parse_response,
    split_halves,
    statements,
)


def test_every_statement_is_asked_exactly_once():
    positions = sorted(p for half in split_halves(statements) for p in half)
    assert positions == list(range(len(statements)))


def test_pair_members_never_share_a_prompt():
    halves = split_halves(statements)
    where = {}
    for index, half in enumerate(halves):
        for pos in half:
            s = statements[pos]
            if s.pair is not None:
                where.setdefault(s.pair, set()).add(index)
    assert len(where) == len(st.PAIRS)
    assert all(len(indices) == 2 for indices in where.values())


def test_halves_are_balanced():
    sizes = [len(h) for h in split_halves(statements)]
    assert len(sizes) == HALVES
    assert max(sizes) - min(sizes) <= 1


def test_pairs_have_opposite_truth_and_distinct_text():
    by_pair = {}
    for s in statements:
        if s.pair is not None:
            by_pair.setdefault(s.pair, []).append(s)
    for members in by_pair.values():
        assert sorted(m.is_true for m in members) == [False, True]
        assert members[0].text != members[1].text
    assert len({s.text for s in statements}) == len(statements)


def test_honeypots_are_false_and_unpaired():
    honeypots = [s for s in statements if s.is_honeypot]
    assert len(honeypots) == len(st.HONEYPOTS)
    assert all(not s.is_true and s.pair is None for s in honeypots)


def test_every_topic_is_present():
    assert {s.topic for s in statements} == {st.ENGINE, st.CITIES, st.DYNAMICS}


def test_prompts_number_from_one_and_merge_back_in_order():
    parts = {}
    for prompt in build_prompts():
        assert (
            prompt.text.rstrip()
            .splitlines()[-1]
            .startswith(f" {len(prompt.positions)}: ")
        )
        reply = "\n".join(
            f"{i}: {'True' if statements[pos].is_true else 'False'}"
            for i, pos in enumerate(prompt.positions, start=1)
        )
        parts[prompt.index] = parse_response(reply, len(prompt.positions))
    merged = merge_answers(parts)
    assert all((a is Answer.TRUE) == s.is_true for a, s in zip(merged, statements))


def test_a_missing_half_is_unparseable_not_wrong():
    prompts = build_prompts()
    only_first = {prompts[0].index: [Answer.UNKNOWN] * len(prompts[0].positions)}
    merged = merge_answers(only_first)
    assert merged.count(Answer.UNKNOWN) == len(prompts[0].positions)
    assert merged.count(Answer.UNPARSEABLE) == len(statements) - len(
        prompts[0].positions
    )


def test_parse_ignores_numbers_beyond_the_half():
    answers = parse_response("1: True\n2: False\n99: True", 2)
    assert answers == [Answer.TRUE, Answer.FALSE]
