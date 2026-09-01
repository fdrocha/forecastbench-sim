# Micropolis world

Validates simulated forecasting against real-world forecasting ability: LLMs forecast city
metrics from a text report of a Micropolis run, and their CRPS/skill is correlated against
external benchmarks (ECI, ForecastBench) held in `model_scores.csv`.

## Layout

`micropolis_world/` is the importable package (`uv run` resolves it via the root uv
workspace); `scripts/` holds the executables — not a package, so no `__init__.py`, and
`#!/usr/bin/env -S uv run python3` + `chmod +x`.

The simulator itself is **external**: a MicropolisCore checkout at `$MICROPOLIS_CORE_PATH`
(read from `worlds/micropolis/.env` at `module_globals` import time), driven as
`pnpm run run-sim` by `city_sim.CitySimulation`. Provider API keys come from `.env` or, when
absent, `module_globals.ensure_api_keys()` (GCP Secret Manager).

## Pipeline

`CitySimulation` (JSONL log/events cached per city+seed+disasters) → `report.gen_world_report`
(the model-facing text) → `scenarios.build_corpus` (one question per scenario × snapshot_turn
× horizon × metric, each with its resolved value) → `continuous_eval.group_into_batches` →
`scenarios.build_batch_prompt_continuous` → `prompting.run_prompts` (async fan-out,
per-provider semaphores, own retry/backoff) → `scenarios.parse_batch_percentiles[_semantic]`
→ `data.json` → the `analyze_*` / `plot_*` scripts.

From fbsim-core: `QuestionTemplate`/`TemplateRegistry`/`QuestionResolver`, `compute_crps`,
`evaluation.models.get_models`. Each scenario is a single city, always entity `0`, in the
turn-major world schema built by `city_sim.to_world`.

## Conventions

- **Configs, not flags.** Every script takes a JSON5 config as its first positional arg
  (`config.add_config_args` / `load_config(s)`; wrap `main` in `@main_with_config`).
  Behavior toggles (`--dry-run`, `--no-plot`, …) stay on the CLI; only
  `--seed/--cities/--disasters/--models/--label` override config values. Read new keys with
  `get_*_or(...)` so configs written earlier keep working. One config can carry the union of
  every script's keys.
- **A prompt variant is a config plus a text file, not a code change** — see
  `configs/preamble*.txt`, `epilogue*.txt`, `prompt-*.json5`.
- **The response cache is content-addressed and never invalidated.** Prompts, raw responses
  and usage sidecars live in `continuous/cache/{batch_id}/…-{prompt_hash}.txt|json`, shared
  across labels; a new variant adds files beside the old ones. A cached response is never
  re-fetched, so its cost must be written when first paid. Per-label outputs (`data.json`,
  plots, reports) go under `continuous/{label}/`.
- **Analysis is offline.** The `analyze_*`/`plot_*` scripts read only `data.json` and cached
  sim logs — no prompting, no re-simulating, no API keys. Keep it that way; it makes them
  free to re-run. `select_for_config` narrows a gathered dataset to a config's slice and
  *errors* on anything missing rather than silently reporting less.
- **litellm is imported lazily** (inside functions in `module_globals`, `usage`, `prompting`)
  so simulation- and scoring-only code never pulls it in. Tests monkeypatch through those
  lazy imports.
- **Output goes to a Markdown report**, accumulated via `continuous_eval.MdReport` and stamped
  with both this repo's and the engine checkout's commit; stdout gets only paths.
- Model ids are `provider/name`; filenames slugify `/` → `_`. External scores join on
  `LiteLLMSlug` in `model_scores.csv` — a blank slug means the model is excluded, and
  near-miss slugs must never be guessed at.
- Scoring choices that must not be reinvented per script: horizon 0 is a read-off, not a
  forecast, and is excluded from aggregates; `totalFunds` is excluded from |actual|-normalized
  CRPS; skill (`CRPS_model / CRPS_baseline`) is aggregated as a geometric mean with t-based
  CIs clustered on (scenario, snapshot turn). `analyze_skill_by_config.py` →
  `analyze_baseline_skill.py` → `analyze_continuous.py` import each other via
  `sys.path.insert` for exactly this reason.

## Scripts

Simulation / inspection:
- `run_sim.py` — run the engine for the config's cities × disasters, plus a plot each.
- `gen_report.py` — print the model-facing world report for already-run sims.
- `gen_corpus.py` — build and dump the question corpus.
- `generate_prompt.py` — print the first batch prompt a config would send (stdout = bare
  prompt, status to stderr), for eyeballing or diffing variants.
- `check_determinism.py` — repeat a scenario at one seed and diff the outputs.

Continuous forecasting eval:
- `run_eval_continuous.py` — the only script here that prompts models; writes `data.json`.
- `analyze_continuous.py` — CRPS tables/figures, normalized by |actual|.
- `analyze_baseline_skill.py` — same forecasts scored against a naive (`plain`/`sigma`)
  no-change baseline, so 1.0 is the meaningful zero point.
- `analyze_skill_by_config.py` — that skill compared across several configs (many-config
  arg form), typically the prompt variants.
- `plot_forecasts.py` — trajectories with forecast quantiles overlaid.

Domain-knowledge eval (`micropolis_world/knowledge_eval/`, True/False/Unknown statements
about the engine, own cache under `data/micropolis/knowledge_eval/`):
- `run_eval_knowledge.py` — gather answers (prompts models; defaults to
  `configs/knowledge_eval.json5`).
- `analyze_knowledge.py` — score them and correlate with ECI.

Cost accounting: `analyze_usage.py` — sums the usage sidecars, either for what given configs
imply or for a `--glob` of sidecar paths.

## Tests

`uv run pytest` from `worlds/micropolis`. Tests import sibling test modules by bare name
(pytest rootdir path insertion), and load analysis scripts through
`importlib.util.spec_from_file_location` since `scripts/` isn't importable. Nothing in the
suite calls a model or needs a key.

When testing changes avoid doing API calls to models unlessnecessary.
If API calls are needed, use the config worlds/micropolis/micropolis_world/configs/testing.json5 with appropriate overrides or something derived from it.
