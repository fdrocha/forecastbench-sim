"""Unit tests for natcond_score_v2.py: KL-bits loss, clipping, CUS, direction,
and the brier-single mode against the reserved resolution continuation."""
import math

import pytest
from conftest import load_script

scorer = load_script("scripts/uplift_v2/natcond_score_v2.py")


def test_kl_bits_known_values():
    assert scorer.kl_bits(0.5, 0.5) == 0.0
    assert scorer.kl_bits(0.037, 0.037) < 1e-12
    want = 0.5 * math.log2(0.5 / 0.25) + 0.5 * math.log2(0.5 / 0.75)
    assert abs(scorer.kl_bits(0.5, 0.25) - want) < 1e-12
    # symmetric-direction sanity: KL is asymmetric but always >= 0
    assert scorer.kl_bits(0.25, 0.5) > 0
    assert abs(scorer.kl_bits(0.9, 0.5) - scorer.kl_bits(0.1, 0.5)) < 1e-12


def test_kl_bits_handles_degenerate_truth():
    # q may be exactly 0 or 1 (empirical half-B rates); 0*log0 := 0
    assert abs(scorer.kl_bits(0.0, 0.5) - 1.0) < 1e-12
    assert abs(scorer.kl_bits(1.0, 0.5) - 1.0) < 1e-12
    assert math.isfinite(scorer.kl_bits(0.0, scorer.clip(0.0)))
    assert math.isfinite(scorer.kl_bits(1.0, scorer.clip(1.0)))


def test_kl_penalizes_tail_odds_as_promised_in_prompt():
    # "saying 1% when the truth is 10% costs far more than saying 30% when
    #  the truth is 39%"
    assert scorer.kl_bits(0.10, 0.01) > 5 * scorer.kl_bits(0.39, 0.30)


def test_clip_bounds():
    assert scorer.clip(0.0) == 0.001
    assert scorer.clip(1.0) == 0.999
    assert scorer.clip(-3.0) == 0.001
    assert scorer.clip(0.42) == 0.42


def _cells():
    def cell(qid, event_id, cls, horizon, p_y_b, p_yx_b):
        return {"qid": qid, "event_id": event_id, "cls": cls,
                "horizon": horizon, "question": "?", "game_id": "w",
                "half_b": {"p_y": p_y_b, "p_yx": p_yx_b,
                           "delta": round(p_yx_b - p_y_b, 4)}}
    return [
        cell("q1", "e1", "effect", "H1", 0.30, 0.60),   # delta_b > 0
        cell("q1", "e2", "placebo", "H1", 0.30, 0.31),
        cell("q2", "e1", "effect", "H2", 0.80, 0.50),   # delta_b < 0
    ]


def _payload(rows):
    return {"mode": "two-turn", "tag": "t", "model": "m", "results": rows}


def _rows(preds):
    """preds: {(qid, event_id or None): p}; event None -> base stage."""
    out = []
    for (qid, event_id), p in preds.items():
        stage = "base" if event_id is None else "cond"
        out.append({"qid": qid, "sample": 0, "stage": stage,
                    "event_id": event_id, "p": p})
    return out


def test_perfect_update_scores_cus_one():
    cells = _cells()
    preds = {("q1", None): 0.30, ("q2", None): 0.80,
             ("q1", "e1"): 0.60, ("q1", "e2"): 0.31, ("q2", "e1"): 0.50}
    report = scorer.score(cells, _payload(_rows(preds)), bootstrap=0)
    overall = report["bands"]["overall"]
    assert overall["n"] == 3
    assert overall["kl_bits"]["loss_updated"] < 1e-12
    assert overall["kl_bits"]["cus"] == 1.0
    assert overall["sq"]["cus"] == 1.0
    assert report["bands"]["overall"]["direction"]["accuracy"] == 1.0


def test_no_update_scores_cus_zero_and_direction_incorrect():
    cells = _cells()
    preds = {("q1", None): 0.45, ("q2", None): 0.45,
             ("q1", "e1"): 0.45, ("q1", "e2"): 0.45, ("q2", "e1"): 0.45}
    report = scorer.score(cells, _payload(_rows(preds)), bootstrap=0)
    overall = report["bands"]["overall"]
    assert abs(overall["kl_bits"]["cus"]) < 1e-12
    assert abs(overall["sq"]["cus"]) < 1e-12
    # zero updates never count as directionally correct
    assert overall["direction"] == {"correct": 0, "total": 2, "accuracy": 0.0}


def test_direction_against_half_b_delta():
    cells = _cells()
    preds = {("q1", None): 0.40, ("q2", None): 0.60,
             ("q1", "e1"): 0.50,   # up, delta_b > 0 -> correct
             ("q1", "e2"): 0.40,
             ("q2", "e1"): 0.70}   # up, delta_b < 0 -> incorrect
    report = scorer.score(cells, _payload(_rows(preds)), bootstrap=0)
    assert report["bands"]["overall"]["direction"] == {
        "correct": 1, "total": 2, "accuracy": 0.5}


def test_losses_use_clipped_predictions_and_match_hand_computation():
    cells = _cells()[:1]  # q1/e1: truth p_yx_b = 0.60
    preds = {("q1", None): 0.0, ("q1", "e1"): 1.0}  # extreme, must clip
    report = scorer.score(cells, _payload(_rows(preds)), bootstrap=0)
    m = report["bands"]["overall"]
    assert abs(m["kl_bits"]["loss_updated"] - scorer.kl_bits(0.60, 0.999)) < 1e-12
    assert abs(m["kl_bits"]["loss_baseline"] - scorer.kl_bits(0.60, 0.001)) < 1e-12
    assert abs(m["sq"]["loss_updated"] - (0.999 - 0.60) ** 2) < 1e-12
    assert abs(m["sq"]["loss_baseline"] - (0.001 - 0.60) ** 2) < 1e-12
    # baseline calibration is against half-B p_y
    assert abs(m["baseline_calibration_kl_bits"]
               - scorer.kl_bits(0.30, 0.001)) < 1e-12


def test_bands_split_by_horizon_and_cls_and_missing_counted():
    cells = _cells()
    preds = {("q1", None): 0.30,  # q2 base missing -> its cells unscorable
             ("q1", "e1"): 0.55, ("q1", "e2"): 0.32}
    report = scorer.score(cells, _payload(_rows(preds)), bootstrap=0)
    assert report["n_scored"] == 2
    assert report["missing"] == {"no_base": 1, "no_cond": 0}
    assert set(report["bands"]) >= {"overall", "effect", "placebo", "H1"}
    assert "H2" not in report["bands"]  # its only cell was unscorable
    assert report["bands"]["effect"]["n"] == 1
    assert report["bands"]["placebo"]["n"] == 1


def _cells_with_resolution():
    cells = _cells()
    outcomes = {("q1", "e1"): 1, ("q1", "e2"): 0, ("q2", "e1"): None}
    for c in cells:
        c["resolution_rollout_id"] = "s1"
        c["resolution_outcome"] = outcomes[(c["qid"], c["event_id"])]
    return cells


def test_brier_single_math_and_unresolved_cells_skipped():
    cells = _cells_with_resolution()
    preds = {("q1", None): 0.30, ("q2", None): 0.80,
             ("q1", "e1"): 0.60, ("q1", "e2"): 0.20, ("q2", "e1"): 0.50}
    report = scorer.score(cells, _payload(_rows(preds)), bootstrap=0,
                          loss="brier-single")
    assert report["loss_mode"] == "brier-single"
    assert report["clip"] is None
    assert report["missing"]["no_resolution"] == 1  # q2/e1 unresolved in s1
    m = report["bands"]["overall"]
    assert m["n"] == 2
    want_upd = ((0.60 - 1) ** 2 + (0.20 - 0) ** 2) / 2
    want_base = ((0.30 - 1) ** 2 + (0.30 - 0) ** 2) / 2
    assert abs(m["brier_single"]["loss_updated"] - want_upd) < 1e-12
    assert abs(m["brier_single"]["loss_baseline"] - want_base) < 1e-12
    assert abs(m["brier_single"]["cus"] - (1 - want_upd / want_base)) < 1e-12
    # direction stays against the half-B delta (mode-independent)
    assert m["direction"]["total"] == 1  # only q1/e1 survives the filter


def test_brier_single_uses_unclipped_predictions():
    cells = _cells_with_resolution()[:1]  # q1/e1, outcome 1
    preds = {("q1", None): 0.50, ("q1", "e1"): 1.0}
    report = scorer.score(cells, _payload(_rows(preds)), bootstrap=0,
                          loss="brier-single")
    # (1.0 - 1)^2 == 0 exactly; a clipped 0.999 would give 1e-6
    assert report["bands"]["overall"]["brier_single"]["loss_updated"] == 0.0


def test_brier_single_requires_migrated_cells():
    cells = _cells()  # mined without the reserved continuation
    preds = {("q1", None): 0.30, ("q2", None): 0.80,
             ("q1", "e1"): 0.60, ("q1", "e2"): 0.20, ("q2", "e1"): 0.50}
    with pytest.raises(RuntimeError, match="resolution_outcome"):
        scorer.score(cells, _payload(_rows(preds)), bootstrap=0,
                     loss="brier-single")


def test_kl_mode_ignores_resolution_fields():
    cells = _cells_with_resolution()
    preds = {("q1", None): 0.30, ("q2", None): 0.80,
             ("q1", "e1"): 0.60, ("q1", "e2"): 0.20, ("q2", "e1"): 0.50}
    report = scorer.score(cells, _payload(_rows(preds)), bootstrap=0)
    assert report["loss_mode"] == "kl"
    assert report["bands"]["overall"]["n"] == 3  # unresolved cell still scored
    assert "no_resolution" not in report["missing"]


def test_samples_average_and_bootstrap_deterministic():
    cells = _cells()
    rows = _rows({("q1", None): 0.20, ("q2", None): 0.80,
                  ("q1", "e1"): 0.50, ("q1", "e2"): 0.30, ("q2", "e1"): 0.60})
    rows += [{"qid": "q1", "sample": 1, "stage": "base", "event_id": None,
              "p": 0.40}]  # second sample -> mean base 0.30
    r1 = scorer.score(cells, _payload(rows), bootstrap=50, seed=7)
    r2 = scorer.score(cells, _payload(rows), bootstrap=50, seed=7)
    assert r1 == r2
    ci = r1["bands"]["overall"]["kl_bits"]["cus_95pct_ci"]
    assert ci is not None and ci[0] <= ci[1]
    base, _ = scorer.load_predictions(_payload(rows))
    assert abs(base["q1"] - 0.30) < 1e-12
