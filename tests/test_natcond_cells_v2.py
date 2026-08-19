"""Unit tests for natcond_cells_v2.py split-half certification."""
import gzip
import json
import math
import random

from conftest import load_script

cells_v2 = load_script("scripts/uplift_v2/natcond_cells_v2.py")


def test_split_tags_fixed_halves_by_rollout_index():
    tags = [f"s{k}" for k in range(1, 1001)]
    a, b = cells_v2.split_tags(tags)
    assert len(a) == len(b) == 500
    assert a.isdisjoint(b) and a | b == set(tags)
    # deterministic regardless of input order
    shuffled = tags[:]
    random.Random(3).shuffle(shuffled)
    a2, b2 = cells_v2.split_tags(shuffled)
    assert (a2, b2) == (a, b)
    # numeric-aware sort: s999 precedes s1000
    assert ("s999" in a) != ("s1000" in a)
    a3, _ = cells_v2.split_tags(["s2", "s10", "s1"])  # order s1, s2, s10
    assert a3 == {"s1", "s10"}


def test_reserved_tag_deterministic_and_excluded_from_halves():
    assert cells_v2.reserved_tag(["s2", "s10", "s1"]) == "s1"
    tags = [f"s{k}" for k in range(1, 1001)]
    shuffled = tags[:]
    random.Random(5).shuffle(shuffled)
    assert cells_v2.reserved_tag(shuffled) == "s1"
    # halves partition the REMAINING rollouts; the reserved draw is in neither
    a, b = cells_v2.split_tags(set(tags) - {"s1"})
    assert len(a) == 500 and len(b) == 499
    assert "s1" not in a and "s1" not in b
    assert a | b == set(tags) - {"s1"} and a.isdisjoint(b)


def _write_fixture(tmp_path):
    """40 rollouts s1..s40; conditioning event = 'Benin discovered Alphabet'
    in window (60, 75] for members s1..s30 (15 per half). Questions:
      q0000 H1: strong effect (answer == membership), one null answer (s40)
      q0001 H2: exact independence pattern (seed % 3 == 0) -> placebo
      q0002 H3: strong effect again (checks horizon plumbing)
      q0003 H7: must be filtered out by the horizon set
    """
    arm = tmp_path / "w_test"
    (arm / "rollouts").mkdir(parents=True)
    tags = [f"s{k}" for k in range(1, 41)]
    members = {f"s{k}" for k in range(1, 31)}
    for tag in tags:
        seed = int(tag[1:])
        turn = 70 if tag in members else 80  # 80 is outside (60, 75]
        events = [
            {"turn": 61, "type": "city_founded", "description": f"city ({tag})"},
            {"turn": turn, "type": "tech_discovered",
             "description": "Benin discovered Alphabet"},
        ]
        with gzip.open(arm / "rollouts" / f"{tag}.json.gz", "wt") as f:
            json.dump({"metadata": {"turn": 150},
                       "civilizations": {"0": {"name": "Benin"},
                                         "1": {"name": "Jolof"},
                                         "5": {"name": "Pirate"},
                                         "6": {"name": "Barbarian"}},
                       "events": events}, f)

    def rec(tag, answer):
        return {"tag": tag, "answer": answer}

    answers = {
        "q0000": [rec(t, None if t == "s40" else t in members) for t in tags],
        "q0001": [rec(t, int(t[1:]) % 3 == 0) for t in tags],
        "q0002": [rec(t, t in members) for t in tags],
        "q0003": [rec(t, True) for t in tags],
    }
    (arm / "manifest.json").write_text(json.dumps(
        {"config": {"game_id": "w_test", "snapshot_turn": 60,
                    "resolution_turns": [90, 120, 150]},
         "answers": answers}))

    questions = [
        {"question_id": "q0000", "template_id": "tech_comparative",
         "horizon": "H1", "resolution_turn": 90,
         "question_text": "Will A beat B at turn 90?"},
        {"question_id": "q0001", "template_id": "tech_within",
         "horizon": "H2", "resolution_turn": 120,
         "question_text": "Will C discover D by turn 120?"},
        {"question_id": "q0002", "template_id": "tech_comparative",
         "horizon": "H3", "resolution_turn": 150,
         "question_text": "Will A beat B at turn 150?"},
        {"question_id": "q0003", "template_id": "tech_comparative",
         "horizon": "H7", "resolution_turn": 300,
         "question_text": "Out-of-scope horizon."},
    ]
    qpath = tmp_path / "questions.json"
    qpath.write_text(json.dumps({"questions": questions}))
    return arm, qpath, members


def _expected_half(amap, half, members):
    tags = [t for t in half if t in amap]
    p_y = sum(amap[t] for t in tags) / len(tags)
    sub = [amap[t] for t in tags if t in members]
    p_yx = sum(sub) / len(sub)
    se = math.sqrt(max(p_yx * (1 - p_yx), 0.25 / len(sub)) / len(sub)
                   + p_y * (1 - p_y) / len(tags))
    return {"n": len(tags), "n_x": len(sub), "p_y": round(p_y, 4),
            "p_yx": round(p_yx, 4), "delta": round(p_yx - p_y, 4),
            "se_delta": round(se, 4)}


def test_mine_world_split_half_certification(tmp_path):
    arm, qpath, members = _write_fixture(tmp_path)
    cells = cells_v2.mine_world(
        "w_test", str(arm), str(qpath), {"H1", "H2", "H3"},
        [(60, 75)], 0.05, 0.95, 6, 6)

    by_q = {c["qid"]: c for c in cells}
    assert set(by_q) == {"q0000", "q0001", "q0002"}  # H7 filtered out
    assert all(c["event_desc"] == "Benin discovered Alphabet" for c in cells)
    assert all(c["window"] == [60, 75] for c in cells)

    tags = [f"s{k}" for k in range(1, 41)]
    res = cells_v2.reserved_tag(tags)
    assert res == "s1"
    half_a, half_b = cells_v2.split_tags(set(tags) - {res})
    assert len(half_a) == 20 and len(half_b) == 19
    assert res not in half_a and res not in half_b

    manifest = json.loads((arm / "manifest.json").read_text())
    for qid, cell in by_q.items():
        amap = {r["tag"]: r["answer"] for r in manifest["answers"][qid]
                if r["answer"] is not None}
        want_a = _expected_half(amap, half_a, members)
        want_b = _expected_half(amap, half_b, members)
        for k, v in want_a.items():
            assert cell["half_a"][k] == v, (qid, k)
        for k, v in want_b.items():
            assert cell["half_b"][k] == v, (qid, k)
        # v1 top-level fields carry the half-A (certification) values
        for k in ("n_x", "p_y", "p_yx", "delta", "se_delta"):
            assert cell[k] == cell["half_a"][k]
        assert cell["freq"] == cell["half_a"]["freq"]
        # certification on half A only
        want_cls = ("effect" if abs(want_a["delta"]) > 2 * want_a["se_delta"]
                    else "placebo")
        assert cell["cls"] == want_cls
        # shrinkage diagnostic
        want_shrunk = round(math.sqrt(max(
            cell["delta"] ** 2 - cell["se_delta"] ** 2, 0.0)), 4)
        assert cell["shrunk_abs_delta"] == want_shrunk

    # the two designed-strong cells certify as effects, the null as placebo
    assert by_q["q0000"]["cls"] == "effect"
    assert by_q["q0002"]["cls"] == "effect"
    assert by_q["q0001"]["cls"] == "placebo"
    assert by_q["q0001"]["shrunk_abs_delta"] == 0.0  # |delta| < SE -> shrunk to 0

    # designed values for the strong H1 cell (members all yes in both halves).
    # s1 is reserved -> A = even seeds s2..s40 (20), B = odd s3..s39 (19).
    c = by_q["q0000"]
    assert c["half_a"] == {**c["half_a"], "n": 19,  # s40's null answer dropped
                           "n_x": 15, "p_y": round(15 / 19, 4), "p_yx": 1.0}
    assert c["freq"] == 0.75  # 15 of 20 half-A rollouts are members
    assert c["half_b"] == {**c["half_b"], "n": 19, "n_x": 14, "p_yx": 1.0,
                           "p_y": round(14 / 19, 4)}
    # reserved continuation: excluded above, emitted as the 0/1 Brier option
    assert all(cell["resolution_rollout_id"] == "s1" for cell in cells)
    assert by_q["q0000"]["resolution_outcome"] == 1  # s1 is a member -> True
    assert by_q["q0001"]["resolution_outcome"] == 0  # 1 % 3 != 0
    assert by_q["q0002"]["resolution_outcome"] == 1
    # horizon plumbing
    assert (by_q["q0000"]["horizon"], by_q["q0000"]["resolution_turn"]) == ("H1", 90)
    assert (by_q["q0001"]["horizon"], by_q["q0001"]["resolution_turn"]) == ("H2", 120)
    assert (by_q["q0002"]["horizon"], by_q["q0002"]["resolution_turn"]) == ("H3", 150)

    rows = cells_v2.summarize(cells)
    assert {(r["template_id"], r["horizon"], r["cls"], r["cells"]) for r in rows} == {
        ("tech_comparative", "H1", "effect", 1),
        ("tech_within", "H2", "placebo", 1),
        ("tech_comparative", "H3", "effect", 1),
    }


def test_civs_from_game_data_table_not_just_tech_events(tmp_path):
    arm, _, _ = _write_fixture(tmp_path)
    rollouts, civs = cells_v2.load_rollout_events(str(arm))
    assert len(rollouts) == 40
    # Jolof never techs in the events, but is in the civilizations table
    assert civs == {"Benin", "Jolof", "Pirate", "Barbarian"}


def test_lane_sharded_arm_matches_single_arm(tmp_path):
    """A world split across laneNN/ shards (the mined-fleet layout) mines to
    exactly the same cells as the equivalent single arm."""
    arm, qpath, members = _write_fixture(tmp_path)
    single = cells_v2.mine_world(
        "w_test", str(arm), str(qpath), {"H1", "H2", "H3"},
        [(60, 75)], 0.05, 0.95, 6, 6)

    manifest = json.loads((arm / "manifest.json").read_text())
    sharded = tmp_path / "w_sharded"
    lanes = [[f"s{k}" for k in range(1, 15)],
             [f"s{k}" for k in range(15, 29)],
             [f"s{k}" for k in range(29, 41)]]
    for i, lane_tags in enumerate(lanes):
        lane = sharded / f"lane{i:02d}"
        (lane / "rollouts").mkdir(parents=True)
        lane_answers = {
            qid: [r for r in recs if r["tag"] in lane_tags]
            for qid, recs in manifest["answers"].items()}
        (lane / "manifest.json").write_text(json.dumps(
            {"config": {"game_id": "w_test", "rng_seeds": lane_tags},
             "answers": lane_answers}))
        for tag in lane_tags:
            src = arm / "rollouts" / f"{tag}.json.gz"
            (lane / "rollouts" / f"{tag}.json.gz").write_bytes(src.read_bytes())

    assert cells_v2.manifest_paths(str(sharded)) == sorted(
        sharded.glob("lane*/manifest.json"))
    merged = cells_v2.load_answers(str(sharded))
    assert merged == cells_v2.load_answers(str(arm))
    sharded_cells = cells_v2.mine_world(
        "w_test", str(sharded), str(qpath), {"H1", "H2", "H3"},
        [(60, 75)], 0.05, 0.95, 6, 6)
    assert sharded_cells == single and len(single) == 3


def test_mine_world_enforces_nx_floor_both_halves(tmp_path):
    """An event with too-few conditioning rollouts in either half emits no cells."""
    arm, qpath, members = _write_fixture(tmp_path)
    # rewrite rollouts: only 8 members (4 per half) < min_nx floor of 12
    few = {f"s{k}" for k in range(1, 9)}
    for k in range(1, 41):
        tag = f"s{k}"
        turn = 70 if tag in few else 80
        events = [{"turn": turn, "type": "tech_discovered",
                   "description": "Benin discovered Alphabet"}]
        with gzip.open(arm / "rollouts" / f"{tag}.json.gz", "wt") as f:
            json.dump({"events": events}, f)
    cells = cells_v2.mine_world(
        "w_test", str(arm), str(qpath), {"H1", "H2", "H3"},
        [(60, 75)], 0.0, 1.0, 6, 6)
    assert cells == []
