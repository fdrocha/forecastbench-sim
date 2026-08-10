# Micropolis run configs

Every script in `worlds/micropolis/scripts/` takes an optional config file as
its first positional argument and reads all of its parameters from there:

```
uv run python scripts/run_single_city_eval.py                        # default.json5
uv run python scripts/run_single_city_eval.py my_config.json5        # a custom config
uv run python scripts/run_single_city_eval.py my_config.json5 --seed 7   # seed override
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
uv run python scripts/run_single_city_eval.py                  # gather everything, once
uv run python scripts/analyze_single_city.py three_models.json5
uv run python scripts/plot_forecasts.py one_city.json5
```

Naming something the dataset doesn't have — a model that wasn't prompted, a
city that wasn't simulated, a horizon that wasn't asked — is an error listing
what is missing and what the dataset holds, rather than a quietly smaller
table. Use `--data` to point at a dataset other than the default.

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

Only behavior toggles stay on the command line: `--dry-run`, `--quiet`,
`--seed` (which overrides the config's `seed`), and the reporting scripts'
`--data` and `--outdir`. Everything else comes from the config.

In the table below, "reporting" means `analyze_single_city.py` and
`plot_forecasts.py`, which use these keys to pick a slice of an existing
dataset rather than to run anything.

## Parameters

| Key | Type | Used by | Meaning |
| --- | --- | --- | --- |
| `seed` | int | all | RNG seed for the simulations. Overridable with `--seed`. Part of a scenario's identity, so reporting on a seed that wasn't gathered is an error. |
| `cities` | list[str] | all | Micropolis cities to run. Must be names from `module_globals.CITY_CHOICES`. Reporting selects on them. |
| `disasters` | list[bool] | all | Disaster settings to run each city under. `[false, true]` runs both variants; `[false]` runs only one. Combined with `cities` as a cross product. Reporting selects on them. |
| `turns` | int | `run_sim.py` | How many turns to simulate. The corpus scripts ignore this and derive their own length from `snapshot_turns` + `horizons`. |
| `snapshot_turns` | list[int] | `gen_corpus.py`, `run_single_city_eval.py`, reporting | Turns at which a world report is generated and questions are asked. 48 turns per year. |
| `horizons` | list[int] | `gen_corpus.py`, `run_single_city_eval.py`, reporting | Forecast horizons past each snapshot, in turns. |
| `report_turn` | int | `gen_report.py` | Turn to print the report for; negative counts back from the last logged turn. |
| `history_freq` | int | `gen_report.py` | The report includes data every this many turns. |
| `models` | list[str] | `run_single_city_eval.py`, reporting | Model ids to prompt, in `provider/name` form. Reporting selects on them, so trimming this list is how you get a readable figure from a large run. |
| `max_tokens` | int | `run_single_city_eval.py` | Response token cap per model call. For a reasoning model this covers thinking as well as the answer, so too low a cap yields an empty reply; the script warns when one is hit. |

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
