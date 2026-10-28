# FreeCiv v3

The FreeCiv pipeline behind the ForecastBench-Sim paper (Jaeho Lee, September 2026): eight anchor games played to
turn 60, 1,000 replays of each from turn 60, question sets whose truth is the frequency or distribution over the
replays, elicitation through OpenRouter and scoring. Copied from the `freeciv-v3` branch at `daca5a15`; sources,
hashes and changes are in `PROVENANCE.json`.

| folder | contents |
|---|---|
| `fleet/` | RunPod scripts for the replay fleet; `make_bundle.sh` assembles what `bringup.sh` sends to each pod |
| `draw/` | corpus consolidation, candidate pools (`bank_v1.py`), the structural draw (`draw_v1.py`), resolution criteria, natural conditionals, report regeneration, QC |
| `run/` | elicitation (`elicit_v1.py` one question per prompt, `elicit_v2.py` batched, `elicit_natcond_v1.py`, `effort_check_v2.py`), scoring (`score_v1.py`, `score_v2.py`, `score_selftest.py`), row rebuild and reparse, selection fractions |

Inputs live outside this folder:

- `FBSIM_FREECIV_V3_DATA`: a folder with `sets/draw_v1/`, `reports/` and `run/models_v*.csv`, such as
  `forecastbench-sim-paper/production/freeciv` (the paper's question sets, reports and model panels).
- `FBSIM_V3_CORPUS`: the working corpus of a replay run (`raw/`, `consol_v4/`, `bank_v1/`), e.g.
  `<repo>/data/freeciv/fbsim_v3/freeciv_1p`.
- `OPENROUTER_API_KEY`, `RUNPOD_API_KEY`: exported, or in the repository's `credentials/.env`; never stored here.
  The fleet's ssh keys are `credentials/freeciv/{runpod_key,archive_key}` and its run state (pod ledger, queues,
  anchors, logs) is `data/freeciv/fbsim_v3/fleet_state/` (both set in `fleet/paths.sh`; see `AGENTS.md`).

Launch scripts, run records and the paper's analysis are in `forecastbench-sim-paper` (`production/freeciv`,
`src/fbsim_paper/assets/freeciv`). Raw responses and scores are in the private data archive.
