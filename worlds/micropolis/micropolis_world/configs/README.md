# Micropolis run configs

Every script in `worlds/micropolis/scripts/` takes an optional config file as
its first positional argument and reads all of its parameters from there. Their
shebang runs them under `uv`, so run them directly, from the `worlds/micropolis`
directory:

```
scripts/run_single_city_eval.py                        # default.json5
scripts/run_single_city_eval.py my_config.json5        # a custom config
scripts/run_single_city_eval.py my_config.json5 --seed 7   # seed override
```

The parameters that most often vary run to run also have a flag, so a one-off
variation needs no new config file: `--seed`, `--cities`, `--disasters`,
`--models` and `--label`. Each overrides the config's key of the same name and
is validated the same way, so `--cities notacity` fails exactly as a typo in the
file would. `--disasters` takes `true`/`false` words: `--disasters false` runs
one variant of every city, `--disasters true false` runs both.

```
scripts/run_single_city_eval.py --cities kyoto --disasters false
scripts/analyze_single_city.py --models openai/gpt-5.6-sol --label kyoto_only
```

A missing parameter is a hard error naming the key and the file. Extra
parameters are ignored, so one config file can serve every script.

## Reporting on a slice of a run

`analyze_single_city.py` and `plot_forecasts.py` read the dataset
`run_single_city_eval.py` wrote, and their config says which slice of it to
report on. So one expensive gathering run can be viewed many ways — fewer
models for a readable figure, one city, the near horizons only — without
prompting anything again:

```
scripts/run_single_city_eval.py                  # gather everything, once
scripts/analyze_single_city.py three_models.json5
scripts/plot_forecasts.py one_city.json5
```

Naming something the dataset doesn't have — a model that wasn't prompted, a
city that wasn't simulated, a horizon that wasn't asked — is an error listing
what is missing and what the dataset holds, rather than a quietly smaller
table. Which dataset they read and where the figures go are both fixed by
`label` (or `--label`) — one of each per label, so a table and a plot always
describe the same run.

## Checking that a seed reproduces

`check_determinism.py` runs each `(city, disasters)` scenario in the config
several times at the same seed and compares the log and events files across
runs, reporting the first turn and field where any two differ:

```
scripts/check_determinism.py                        # default.json5, 5 repeats
scripts/check_determinism.py one_city.json5 --repeats 8 --turns 100
scripts/check_determinism.py one_city.json5 --plot  # overlay the repeats
```

`--plot` also writes one figure per scenario into
`data/micropolis/determinism/`, overlaying every repeat on the same panels
`run_sim.py`'s figures use, so the run-to-run spread is visible rather than only
tabulated. Scenarios with disasters enabled are skipped there: the figure draws
no disaster lines, so a strike hitting one run and not another would be
indistinguishable from the RNG divergence the figure is about.

It reads only `seed`, `cities`, and `disasters`, and takes its run length from
`--turns` rather than the config's `turns`, so a check can be much shorter than
a full run. Because the engine writes each run to a path fixed by
`(city, seed, disasters)`, the repeats overwrite the cached runs in
`data/micropolis/runs/`; the originals are saved first and put back afterwards,
so a check leaves that directory as it found it.

## Format

Configs are parsed as **JSON5**, which adds `//` and `/* */` comments, trailing
commas, and unquoted keys to plain JSON. Plain JSON is a subset, so a `.json`
file still loads unchanged — but prefer the `.json5` extension, or editors will
flag the comments as syntax errors.

Comments are the reason for the format: they let you park unused cities or
models in place rather than deleting them.

```json5
{
  "cities": [
    "haight",
    // "deadwood",  // crashes the engine when disasters are on
  ],
  "max_tokens": 4000,  // trailing commas are fine
}
```

Only behavior toggles stay on the command line — `--dry-run`, `--quiet`,
`--plot` — along with the five parameter overrides above (`--seed`, `--cities`,
`--disasters`, `--models`, `--label`). Everything else comes from the config.

In the table below, "reporting" means `analyze_single_city.py` and
`plot_forecasts.py`, which use these keys to pick a slice of an existing
dataset rather than to run anything.

## Parameters

| Key | Type | Used by | Meaning |
| --- | --- | --- | --- |
| `seed` | int | all | RNG seed for the simulations. Overridable with `--seed`. Part of a scenario's identity, so reporting on a seed that wasn't gathered is an error. |
| `label` | str | `run_single_city_eval.py`, reporting | Names the output directory under `data/micropolis/single_city/`. Defaults to the config file's own name, so every config gets a distinct one for free. Overridable with `--label`. |
| `cities` | list[str] | all | Micropolis cities to run. Must be names from `module_globals.CITY_CHOICES`. Reporting selects on them. Overridable with `--cities`. |
| `disasters` | list[bool] | all | Disaster settings to run each city under. `[false, true]` runs both variants; `[false]` runs only one. Combined with `cities` as a cross product. Reporting selects on them. Overridable with `--disasters`. |
| `turns` | int | `run_sim.py` | How many turns to simulate. The corpus scripts ignore this and derive their own length from `snapshot_turns` + `horizons`. |
| `snapshot_turns` | list[int] | `gen_corpus.py`, `run_single_city_eval.py`, reporting | Turns at which a world report is generated and questions are asked. 48 turns per year. |
| `horizons` | list[int] | `gen_corpus.py`, `run_single_city_eval.py`, reporting | Forecast horizons past each snapshot, in turns. |
| `report_turn` | int | `gen_report.py` | Turn to print the report for; negative counts back from the last logged turn. |
| `history_freq` | int | `gen_report.py` | The report includes data every this many turns. |
| `history_length` | int | `gen_report.py`, `gen_corpus.py`, `run_single_city_eval.py` | Cap the HISTORY table at this many rows, keeping the most recent ones. Optional; `-1` (the default) keeps the whole history. |
| `report_effectiveness` | bool | `gen_report.py`, `gen_corpus.py`, `run_single_city_eval.py` | Include the road, police and fire funding effectiveness lines in the report's CURRENT STATE block. Optional; `false` (the default) drops all three. |
| `preamble_path` | str | `run_single_city_eval.py`, `analyze_usage.py` | File holding the forecasting prompt's preamble, relative to the config file's directory. Must contain a `{sources}` placeholder, filled in with the report sections the variant has. Optional; `preamble1.txt` is the default. Part of the prompt, so changing it misses the response cache rather than mixing variants. |
| `epilogue_path` | str | `run_single_city_eval.py`, `analyze_usage.py` | File holding the answer format instructions that follow the questions, relative to the config file's directory. May contain an `{n}` placeholder, replaced with the number of questions in the batch; an epilogue without it is used as written. Optional; `epilogue1.txt` is the default. Part of the prompt, so changing it misses the response cache rather than mixing variants. |
| `models` | list[str] | `run_single_city_eval.py`, reporting | Model ids to prompt, in `provider/name` form. Reporting selects on them, so trimming this list is how you get a readable figure from a large run. Overridable with `--models`. |
| `max_tokens` | int | `run_single_city_eval.py` | Response token cap per model call. For a reasoning model this covers thinking as well as the answer, so too low a cap yields an empty reply; the script warns when one is hit. |
| `provider_concurrency` | dict[str, int] | `run_single_city_eval.py`, `run_knowledge_eval.py` | Max concurrent API calls per provider, e.g. `{ "AnthropicProvider": 1 }`. Merged over the defaults in `micropolis_world/prompting.py`; optional — omit to use the defaults. |

## Model ids

The default config uses `openai/gpt-3.5-turbo-1106` — currently the cheapest
model, for testing. The sets used elsewhere in ForecastBench, to paste into a
config's `models` list:

```jsonc
// ForecastBench models
"anthropic/claude-3-7-sonnet-20250219",
"anthropic/claude-opus-4-1-20250805",
"anthropic/claude-sonnet-4-20250514",
"openai/o3-2025-04-16",
"openai/gpt-4.1-2025-04-14",
"openai/gpt-5-2025-08-07",
"openai/gpt-5-mini-2025-08-07",
"google/gemini-2.5-pro",
"google/gemini-2.5-flash",
"together/DeepSeek-V3.1",
"together/Qwen3-235B-A22B-fp8-tput",
"together/Kimi-K2-Instruct",
"together/GLM-4.5-Air-FP8",
"mistral/mistral-large-2411",

// Frontier models
"anthropic/claude-opus-4-5-20251101",
"anthropic/claude-sonnet-4-5-20250929",
"google/gemini-3-pro-preview",
"openai/gpt-5.1-2025-11-13",

// Pandemic models
"deepinfra/meta-llama/Meta-Llama-3.1-8B-Instruct",
"deepinfra/meta-llama/Meta-Llama-3.1-70B-Instruct",
"deepinfra/Qwen/Qwen2.5-72B-Instruct",

// "openai/gpt-5.6-luna" — note this one does not support temperature=0
```
