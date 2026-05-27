# FBSim → ForecastBench Arena: Output Format Proposal

**Author:** Nick Merrill  
**Date:** 2026-05-26  
**Status:** Draft — for Houtan's review

---

## Goal

This document proposes the output format FBSim should produce so that it can be ingested by the ForecastBench Arena submission pipeline as a first-class sub-benchmark ("ForecastBench Simulated").

The design goal is: **align with the existing ForecastBench format as closely as possible**, adding only what simulation uniquely requires.

---

## Background: ForecastBench's existing format

The ForecastBench pipeline already defines three artifact types per submission round:

| Artifact | File | Contents |
|---|---|---|
| Question set | `question_sets/{date}-llm.json` | Questions for forecasters to answer |
| Forecast set | `forecast_sets/{date}/{file}.json` | Model or human forecasts, one file per forecaster |
| Resolution set | `resolution_sets/{date}_resolution_set.json` | Ground-truth resolutions |

A **processed forecast set** and a **leaderboard CSV/HTML** are derived from these.

FBSim needs to produce these same three artifacts, in the same format, so the existing pipeline can handle them without modification.

---

## FBSim artifacts

### 1. Question set

Published at the start of each submission window (daily). Resolutions are **not included**.

```json
{
  "forecast_due_date": "2026-05-27",
  "question_set": "2026-05-27-fbsim.json",
  "questions": [
    {
      "id": "a3f9e2c1b84d...",
      "source": "fbsim",
      "question": "At turn 90, will Uruguayan have more cities than Hungarian?",
      "resolution_criteria": "Resolves YES if Uruguayan civilization has strictly more cities than Hungarian civilization at turn 90 of the FreeCiv simulation.",
      "background": "<world report text at snapshot turn>",
      "source_intro": "You are forecasting the outcome of a FreeCiv civilization simulation. The world report in 'background' describes the current game state at turn 60. Forecast what will have happened by the resolution turn.",
      "freeze_datetime": "2026-05-27T00:00:00Z",
      "freeze_datetime_value": null,
      "freeze_datetime_value_explanation": "N/A — resolution is determined by simulation, not a market.",
      "resolution_dates": "2026-05-27",
      "combination_of": "N/A",

      "fbsim": {
        "game_id": "seed1000",
        "question_id": "q0023",
        "template_id": "city_count_comparative",
        "question_type": "binary",
        "horizon": "H2",
        "snapshot_turn": 60,
        "resolution_turn": 90,
        "tail_risk_seed": false
      }
    }
  ]
}
```

**Notes:**

- `id` — `sha256(game_id + ":" + question_id)[:20]`. Deterministic and stable; same question from the same seed always gets the same ID across batches.
- `background` — the world report text at `snapshot_turn`. This is the context the forecaster uses to answer the question.
- `fbsim` namespace — FBSim-specific metadata carried through for stratified scoring. Submitters don't need to read or return it; it's for the scoring pipeline only.
- `question_type` — `"binary"` or `"continuous"`. Binary questions resolve to 0.0/1.0 and are scored with Brier score. Continuous questions resolve to a float and are scored with CRPS.
- `tail_risk_seed` — true if this seed's game eventually collapses (population crash, etc.). Enables reporting a separate tail-risk leaderboard, which is the core FBSim research claim.

### 2. Forecast set

Returned by submitters within the 24-hour window. **This is the existing ForecastBench format — no changes required of submitters.**

```json
{
  "organization": "MIT CSAIL",
  "model": "ForecastBot-v2",
  "model_organization": "MIT CSAIL",
  "forecast_due_date": "2026-05-27",
  "question_set": "2026-05-27-fbsim.json",
  "leaderboard_eligible": true,
  "forecasts": [
    {
      "id": "a3f9e2c1b84d...",
      "source": "fbsim",
      "forecast": 0.73,
      "resolution_date": "2026-05-27",
      "reasoning": "The Uruguayan civilization had more cities at the snapshot turn and appeared to be growing faster.",
      "direction": null
    }
  ]
}
```

**For continuous questions**, `forecast` is a quantile dict instead of a scalar probability:

```json
"forecast": {"p10": 5, "p25": 9, "p50": 14, "p75": 19, "p90": 25}
```

The quantile keys used here should match whatever ForecastBench Quantile (Simas) adopts — this format should be shared, not FBSim-specific.

### 3. Resolution set

Generated at batch creation time (FBSim knows all answers deterministically from the seed). **Kept private during the 24-hour submission window.** Published after the window closes; the static site rebuilds and the leaderboard updates.

```json
{
  "forecast_due_date": "2026-05-27",
  "question_set": "2026-05-27-fbsim.json",
  "resolutions": [
    {
      "id": "a3f9e2c1b84d...",
      "source": "fbsim",
      "direction": null,
      "resolution_date": "2026-05-27",
      "resolved_to": 1.0,
      "resolved": true
    }
  ]
}
```

For continuous questions, `resolved_to` is the actual simulation value (e.g., `14` for 14 cities), not 0.0/1.0.

---

## Batch structure

Each daily batch is a **stratified sample across N seeds**. Concretely, a batch might be:

- 20 seeds × ~25 questions per seed (stratified across templates and horizons) = ~500 questions
- Questions span H0 (no lookahead) through H4 (longest horizon)
- Seeds sampled to include a mix of stable and volatile (tail-risk) games
- Submitters see a flat question list — they do not know which seed produced which question

This matches the ForecastBench model: submitters receive a question set file, respond within 24 hours, and submit a forecast set file. No per-seed structure is visible to the submitter.

---

## Leaderboard

The FBSim leaderboard follows the ForecastBench leaderboard format (rank, organization, model, score, N, 95% CI) and adds FBSim-specific columns for stratified reporting:

| Column | Description |
|---|---|
| Overall Brier / CRPS | Aggregate score across all questions |
| By horizon (H0–H4) | Score at each time horizon |
| Binary Brier | Score on binary questions only |
| Continuous CRPS | Score on continuous questions only |
| Tail-risk Brier | Score on crash-seed questions — the core FBSim measure |
| Template category | Breakdown by economic / military / research / territorial |

The first four columns are sufficient for a v1 leaderboard. Tail-risk and template columns can follow.

---

## What FBSim needs to implement

One new pipeline step: `scripts/export_arena_batch.py`. It takes a set of pre-generated seeds and outputs the three artifact files above. The underlying data is already available — this is a translation layer, not new computation.

The main sub-tasks:
1. ID generation (`sha256(game_id + ":" + question_id)[:20]`)
2. Question-set assembly (stratified sampling across seeds, horizons, templates)
3. World report → `background` field (already at `data/questions/{seed}/world_report/`)
4. Resolution set generation (deterministic from seed data; kept private until window closes)
5. Continuous question `forecast` format — coordinate with ForecastBench Quantile format
6. `tail_risk_seed` flag from `data/seed_classifications.json`

---

## Open questions for Houtan

1. **Quantile format:** What keys should continuous `forecast` use? Options: `{p10, p25, p50, p75, p90}`, `{p05, p25, p50, p75, p95}`, or a list of `[{"q": 0.1, "v": 5}, ...]`. This should be shared with ForecastBench Quantile — recommend deciding now before either ships.

2. **Background field:** World reports can be several thousand tokens. Options: (a) include inline in the question set — simple, matches how FB binary works today, but produces large files; (b) reference a separate file via `world_report_url` — cleaner pipeline, adds a download step for submitters. Leaning toward (a) for v1.

3. **Submission window:** Is 24 hours the right window for FBSim given daily freshness, or would a shorter window (e.g., same-day) be preferable?

4. **Leaderboard aggregation:** For the FBI, how should FBSim's score be weighted against binary and quantile — equal weight per question, or equal weight per benchmark?

5. **Naming:** Should the question set file follow the existing pattern (`{date}-fbsim.json`) to match `-llm.json` / `-human.json`?
