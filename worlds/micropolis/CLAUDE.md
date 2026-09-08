# Micropolis world

Validates simulated forecasting against real-world forecasting ability: LLMs forecast from a
text report of a Micropolis run, and their skill is correlated against external benchmarks
(ECI, ForecastBench) held in `micropolis_world/datafiles/model_scores.csv`.
Two evals share that report: the **continuous** one asks for p10–p90 percentiles
of city metrics (scored by CRPS), the **binary** one asks for a single P(Yes)
on 27 yes/no questions (to be scored by Brier).

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
(the model-facing text) → a corpus builder → `gather.gather_raw_responses` →
a parser → `data.json` → the `analyze_*` / `plot_*` scripts.

`gather.py` is the half both evals share: `group_into_batches` (questions sharing a scenario
and snapshot turn share a report, so they are asked in one numbered prompt),
`EvalPaths` (the content-addressed cache layout, instantiated per eval as
`continuous_eval.PATHS` / `binary_eval.PATHS`), `gather_raw_responses` (cache-hit split,
`prompting.run_prompts` async fan-out under one global concurrency cap, write-on-landing, cost
accounting) and `write_dataset`. It stops at raw response text; **parsing is per eval**,
which is where the two genuinely differ. Do not import `continuous_eval` from `gather` —
that cycles.

Per eval, the pieces around that seam:
- **Continuous** — `scenarios.build_corpus` (one question per scenario × snapshot_turn ×
  horizon × metric, resolved through fbsim-core's `QuestionResolver`) →
  `scenarios.build_batch_prompt_continuous` → `scenarios.parse_batch_percentiles[_semantic]`
  → `continuous_eval.save_dataset` (`"percentiles"` per forecast).
- **Binary** — `binary_questions.build_corpus_binary` (27 questions × snapshot_turn ×
  horizon, resolved locally — see below) → `scenarios.build_batch_prompt_binary` →
  `scenarios.parse_batch_probabilities` → `binary_eval.save_dataset_binary`
  (`"probability"` per forecast, `"answer"` bool per question).

From fbsim-core: `QuestionTemplate`/`TemplateRegistry`/`QuestionResolver`, `compute_crps`,
`evaluation.models.get_models` (`compute_brier_score` is there for the binary analysis when
it lands). Each scenario is a single city, always entity `0`, in the turn-major world schema
built by `city_sim.to_world`.

## Binary questions

`binary_questions.py` implements `binary_forecasts.md`, which is the normative spec —
question texts (§3), the messageNum/messageText table (§2.3), the reference resolver (§4)
and the structural constraints (§5). Read it before touching resolution.

- `QUESTIONS` is the single table: each `Question(qid, text, resolution)` holds its id
  (A1–A16 mid-range, target P(Yes) ≈ 10–90%; B1–B11 tail, ≈ 0.5–5%), its `{HORIZON}`-templated
  text and its criterion as a lambda over a `Window` — `w.n(msg)`, `w.at(t)`, `w.pop(t)`,
  `w.yc(a, b)` — so a question's wording and its resolution never drift apart.
- **Turn convention is this codebase's, not the doc's**: `state_at(T) = log_data[T]` and an
  event's turn is `tick // 16` (`city_sim.turn_of`), a uniform one-turn relabel of §4's
  `rows[T-1]`. Windows are half-open `(NOW, H]` — strictly after the snapshot — and the
  question texts say "between the current turn and turn {HORIZON}" because models otherwise
  count events already listed in the report.
- Events are referenced by `Message(num, text)` constants (`MSG_EARTHQUAKE`, …), never bare
  numbers. `RunIndex.from_sim` raises if a messageNum's text disagrees with the table — the
  §2.3 drift guard, since the engine's enum is position-implied. Message 27 (helicopter) is
  guarded but counted by no question: it co-fires with 24 for one plane crash.
- `check_structural_constraints` runs on every (sim, window) during corpus building: a Yes on
  a question the city's map makes impossible (§5) means a resolver or table bug, not a rare
  event.

## Conventions

- **Configs, not flags.** Every script takes a JSON5 config as its first positional arg
  (`config.add_config_args` / `load_config(s)`; wrap `main` in `@main_with_config`).
  Behavior toggles (`--dry-run`, `--no-plot`, …) stay on the CLI; only
  `--seed/--cities/--disasters/--models/--label` override config values. Read new keys with
  `get_*_or(...)` so configs written earlier keep working. One config can carry the union of
  every script's keys.
- **A prompt variant is a config plus a text file, not a code change** — see
  `micropolis_world/datafiles/preamble*.txt`, `micropolis_world/datafiles/epilogue*.txt`,
  `micropolis_world/configs/prompt-*.json5`.
  A config's `preamble_path`/`epilogue_path` is resolved against
  `micropolis_world/datafiles/`, not against the config's own directory.
- **The response cache is content-addressed and never invalidated.** Prompts, raw responses
  and usage sidecars live in `{continuous,binary}/cache/{batch_id}/…-{prompt_hash}.txt|json`,
  shared across labels; a new variant adds files beside the old ones. A cached response is
  never re-fetched, so its cost must be written when first paid. Per-label outputs
  (`data.json`, plots, reports) go under `{continuous,binary}/{label}/`. **Anything that
  changes the prompt text — including a report default such as `report_census` — re-hashes
  every batch and re-prompts from scratch.** Before running a real config to "check the
  cache", confirm the prompts are unchanged or use `--dry-run`; a full continuous config is
  ~$3 to re-gather.
- **Both evals share one world report.** `gen_world_report`'s defaults are what the
  continuous corpus gets (it passes no report flags), so a default change moves both evals
  together — which is the point: the two are meant to be comparable at the same snapshot.
  `report_census` (rubble/fire/road tile counts, on by default) exists because the binary
  A13–A15 resolve on those counts.
- **Analysis is offline.** The `analyze_*`/`plot_*` scripts read only `data.json` and cached
  sim logs — no prompting, no re-simulating, no API keys. Keep it that way; it makes them
  free to re-run. `select_for_config` narrows a gathered dataset to a config's slice and
  *errors* on anything missing rather than silently reporting less. `--incomplete` (on every
  `analyze_*`/`plot_forecasts` script) downgrades that to a warning and keeps every
  (question, model) pair that was gathered, listing each model's coverage. **A testing aid
  for exercising the scripts before a gather finishes, not a reporting mode**: the selection
  is ragged, so per-model figures cover different question sets and are not comparable with
  each other. It does not relax the coverage check — a model, city or horizon absent from the
  dataset is still an error, as is a selected model with no forecasts at all.
- **One import boundary for the LLM client: `llm_backend`.** It re-exports
  `completion`/`acompletion`/`completion_cost`, the five transient-error classes the retry
  loop catches, and `to_model_id` (slug → the underlying model id). Callers pass the slug
  as `model`; the backend translates internally. OpenRouter
  (`openrouter_completion`, whose per-model routing lives in `model_specs.json5`) is live;
  litellm is a commented block in the same file, so switching back is flipping which block
  is uncommented — never import either client directly. It is still imported lazily (inside
  functions in `module_globals`, `usage`, `prompting`) so simulation- and scoring-only code
  never pulls it in, and tests monkeypatch `llm_backend.acompletion`/`completion` through
  those lazy imports. `tests/conftest.py` blocks httpx outright, so a stub aimed at the wrong
  target fails instead of quietly calling the real API.
- **No caller sets `max_tokens`.** The output cap is a per-endpoint limit, so it belongs to
  the model's `model_specs.json5` entry; configs and `PromptJob` have no such key. The same
  goes for reasoning effort: a caller picks it by naming a suffixed slug, never by a kwarg.
- **Output goes to a Markdown report**, accumulated via `continuous_eval.MdReport` and stamped
  with both this repo's and the engine checkout's commit; stdout gets only paths.
- **Warnings and errors go to stderr, colored, via `messages.warn`/`error`/`plain`** — never
  `print`. A prompting run scrolls hundreds of progress lines, so a failed call has to stand
  out and has to be separable from the report by redirection. Warnings are yellow and errors
  bold red, told apart at a glance rather than by reading the prefix: a warning means the run
  went on, an error means something did not happen. `plain` is for the detail lines under a
  summary and takes the block's color (red by default). Color is dropped when stderr is not a
  tty or `NO_COLOR` is set, so a redirected log carries no escape sequences.
- Models are named by **slug**: a bare OpenRouter model id (`provider/name`, e.g.
  `anthropic/claude-haiku-4.5` — no `openrouter/` prefix, no dated aliases) optionally
  followed by `:suffix` (`openai/o3:lowef`). The suffix selects a different
  `model_specs.json5` entry for the same underlying model, e.g. a reasoning effort; the
  **model id** is the part before the colon (`model_ids.to_model_id`), and it is used in
  exactly two places: the `model` field of the request, inside `openrouter_completion`, and
  the join against `model_scores.csv`, whose `slug` column holds model ids because the
  leaderboards do not score effort settings. Everywhere else — configs, spec keys, cache
  filenames, `data.json`, labels — the slug is the canonical id. A suffixed slug with no spec
  entry is an error, not a fallback to the base model. Filenames use
  `model_ids.filename_slug` (`/` → `_`, `:` → `+`), shared by both response caches. A blank
  `slug` cell in `model_scores.csv` means the model is excluded, and near-miss slugs must
  never be guessed at. The dormant litellm block would have to map back on its own side
  (strip the suffix, passthrough prefix, dated aliases, `gemini/` for `google/`).
- Scoring choices that must not be reinvented per script: horizon 0 is a read-off, not a
  forecast, and is excluded from aggregates; `totalFunds` has no scale and so is excluded
  from normalized CRPS; skill (`CRPS_model / CRPS_baseline`) is aggregated as a geometric mean with t-based
  CIs clustered on (scenario, snapshot turn). `analyze_skill_by_config.py` →
  `analyze_baseline_skill.py` → `analyze_continuous.py` import each other via
  `sys.path.insert` for exactly this reason.

## Scripts

Simulation / inspection:
- `run_sim.py` — run the engine for the config's cities × disasters, plus a plot each.
- `gen_report.py` — print the model-facing world report for already-run sims.
- `gen_corpus.py` — build and dump the continuous question corpus.
- `generate_prompt.py` — print the first continuous batch prompt a config would send
  (stdout = bare prompt, status to stderr), for eyeballing or diffing variants. It has no
  binary equivalent yet; binary prompts are written to the cache by a real
  `run_eval_binary.py` run, not by its `--dry-run`.
- `check_determinism.py` — repeat a scenario at one seed and diff the outputs.

Continuous forecasting eval (percentiles, `data/micropolis/continuous/`). The first three
default to `micropolis_world/configs/continuous.json5`; `analyze_skill_by_config.py` requires
its configs and `plot_forecasts.py` has its own:
- `run_eval_continuous.py` — prompts models; writes `data.json`.
- `analyze_continuous.py` — CRPS tables/figures, normalized by a per-metric scale.
  `--norm global` (the default) divides by a fixed scale per metric
  (`continuous_eval.GLOBAL_SCALES`), so a cell is comparable across scenarios,
  snapshots and horizons; `--norm local`/`baseline` are named but not implemented.
- `analyze_baseline_skill.py` — same forecasts scored against a naive (`plain`/`sigma`)
  no-change baseline, so 1.0 is the meaningful zero point.
- `analyze_skill_by_config.py` — that skill compared across several configs (many-config
  arg form), typically the prompt variants.
- `plot_forecasts.py` — trajectories with forecast quantiles overlaid.

Binary forecasting eval (P(Yes), `data/micropolis/binary/`, spec in `binary_forecasts.md`):
- `run_eval_binary.py` — prompts models; writes `data.json`. Defaults to
  `micropolis_world/configs/binary.json5`
  (19 eligible cities, snapshots 960/1440, horizons +240/+480, disasters on).
  `--dry-run` builds the corpus and prints per-question Yes counts, for eyeballing resolution
  against the spec's P(Yes) ranges. No analysis script yet.
- `extract_ground_truth.py` — pipes the engine's `run_continuations.js` (trunk to the snapshot,
  then `branch_nseeds` reseeded continuations from a byte copy of its state) through
  `ground_truth.consume_stream`, tallying per-question Yes counts and per-metric values at
  every horizon. One JSONL per (scenario, snapshot) under `data/micropolis/ground_truth/`,
  one line per horizon. Branches at `S+1` so continuations start from the row the report
  showed; cross-checks the trunk against the cached `runs/` log; skips files that already
  cover the config unless `--force-regen`.

Domain-knowledge eval (`micropolis_world/knowledge_eval/`, True/False/Unknown statements
about the engine, own cache under `data/micropolis/knowledge_eval/`):
- `run_eval_knowledge.py` — gather answers (prompts models; defaults to
  `micropolis_world/configs/knowledge_eval.json5`).
- `analyze_knowledge.py` — score them and correlate with ECI.

Cost accounting: `analyze_usage.py` — sums the usage sidecars, either for what given configs
imply or for a `--glob` of sidecar paths. Each sidecar also records `provider`, the upstream
endpoint the gateway routed to (null when unreported), and the report warns when one model id
was served by more than one — those endpoints can differ in quantization and speed, so the
calls may not be comparable. Pin one with an `endpoint` in `model_specs.json5`.

## Tests

`uv run pytest` from `worlds/micropolis`. Tests import sibling test modules by bare name
(pytest rootdir path insertion), and load analysis scripts through
`importlib.util.spec_from_file_location` since `scripts/` isn't importable. Nothing in the
suite calls a model or needs a key. The gather loop itself is not covered (it needs API
mocking), so a change to it is verified by regenerating a `data.json` from cache before and
after and diffing.

When testing changes avoid doing API calls to models unless necessary.
If API calls are needed, use `micropolis_world/configs/testing.json5` (continuous) or
`micropolis_world/configs/binary-testing.json5` (binary) — two cheap models, a couple of
cities — or something derived from them, never a production config.
