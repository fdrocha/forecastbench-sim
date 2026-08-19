"""Unit tests for mine_lowprob_corpus.py support counting and split-half mode."""
from conftest import load_script

miner = load_script("worlds/freeciv/scripts/mine_lowprob_corpus.py")


def make_records(n_games, per_game, yes_flags, template="tech_within",
                 target="tech_name=Feudalism", horizon="H2"):
    """One class: n_games games x per_game correlated instances.

    yes_flags maps (game_index, instance_index) -> True for yes instances.
    """
    recs = []
    for g in range(n_games):
        for i in range(per_game):
            recs.append({
                "game_id": f"g{g:03d}",
                "question_id": f"q_{g}_{i}",
                "template_id": template, "target": target, "horizon": horizon,
                "ground_truth": bool(yes_flags.get((g, i), False)),
            })
    return recs


def test_game_level_support_reported_and_selected():
    # 3 games x 10 instances, 1 yes -> rate 1/30 in the 0.005-0.05 band
    recs = make_records(3, 10, {(0, 0): True})
    rows = miner.summarize_classes(recs)
    assert len(rows) == 1
    row = rows[0]
    assert row["n"] == 30 and row["yes"] == 1
    assert abs(row["base_rate"] - 1 / 30) < 1e-12
    assert row["n_games"] == 3  # distinct games, not instances

    # legacy instance support: 30 instances pass min_n=10
    assert miner.select_classes(rows, 0.005, 0.05, 10, "instances", False) == [row]
    # game-level support: only 3 games -> rejected at min_n=10, kept at 3
    assert miner.select_classes(rows, 0.005, 0.05, 10, "games", False) == []
    assert miner.select_classes(rows, 0.005, 0.05, 3, "games", False) == [row]


def test_rate_band_still_applies_with_game_support():
    recs = make_records(30, 5, {(g, 0): True for g in range(15)})  # rate 0.1
    rows = miner.summarize_classes(recs)
    assert rows[0]["n_games"] == 30
    assert miner.select_classes(rows, 0.005, 0.05, 10, "games", False) == []


def test_split_half_deterministic_and_seedable():
    ids = [f"g{k:03d}" for k in range(200)]
    halves0 = {g: miner.split_half_of(g, 0) for g in ids}
    assert set(halves0.values()) == {"A", "B"}
    # stable across repeated calls
    assert all(miner.split_half_of(g, 0) == h for g, h in halves0.items())
    # a different seed redraws the partition
    halves7 = {g: miner.split_half_of(g, 7) for g in ids}
    assert halves0 != halves7


def test_split_half_class_rows_partition_instances():
    recs = make_records(40, 5, {(g, i): True for g in range(40)
                                for i in range(5) if (g + i) % 7 == 0})
    rows = miner.summarize_classes(recs, split_seed=0)
    row = rows[0]
    assert row["n_a"] + row["n_b"] == row["n"] == 200
    assert row["yes_a"] + row["yes_b"] == row["yes"]
    assert row["n_games_a"] + row["n_games_b"] == row["n_games"] == 40
    # per-half rates recompute from the deterministic hash split
    for half in "ab":
        want_games = {r["game_id"] for r in recs
                      if miner.split_half_of(r["game_id"], 0) == half.upper()}
        sub = [r for r in recs if r["game_id"] in want_games]
        assert row[f"n_{half}"] == len(sub)
        assert row[f"rate_{half}"] == sum(
            r["ground_truth"] for r in sub) / len(sub)


def test_split_half_selection_uses_half_a_band_and_emits_half_b_truth():
    n_games, per_game, seed = 60, 5, 0
    games_a = [g for g in range(n_games)
               if miner.split_half_of(f"g{g:03d}", seed) == "A"]
    games_b = [g for g in range(n_games) if g not in games_a]
    assert games_a and games_b
    # half A: exactly 2 yes instances (in band); half B: many more (off band)
    yes = {(games_a[0], 0): True, (games_a[1], 0): True}
    yes.update({(g, i): True for g in games_b[:6] for i in range(5)})
    rows = miner.summarize_classes(make_records(n_games, per_game, yes),
                                   split_seed=seed)
    row = rows[0]
    rate_a = 2 / (len(games_a) * per_game)
    rate_b = 30 / (len(games_b) * per_game)
    assert abs(row["rate_a"] - rate_a) < 1e-12
    assert abs(row["rate_b"] - rate_b) < 1e-12

    lo, hi = 0.005, 0.05
    assert lo <= rate_a <= hi < rate_b  # selection band would reject pooled B
    picked = miner.select_classes(rows, lo, hi, 5, "games", True)
    assert picked == [row]  # selected on half A despite half-B rate off band
    # and the held-out truth label rides along
    assert picked[0]["rate_b"] == rate_b
    # support threshold applies to half-A games only
    assert miner.select_classes(rows, lo, hi, len(games_a) + 1, "games", True) == []
