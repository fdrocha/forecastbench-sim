# Binary forecast questions for Micropolis no-mayor runs

Specification for a harness that (a) drives `run_sim.js` simulations, (b) asks LLMs the
binary forecast questions below at fixed snapshot turns, and (c) resolves each question
from the simulator's output files. This document is **self-contained**: everything needed
to implement resolution is here — no access to the MicropolisCore source is required,
only the ability to run `run_sim.js` and read its two output files.

Context: these are "no-mayor" runs — a builtin city is loaded and simulated with zero
player input (no building, zoning, budget or tax changes). Random disasters stay enabled.

| | |
|:--|:--|
| Questions | 16 mid-range (A1–A16, target ≈10–90% Yes) + 9 tail (B1–B9, target ≈0.5–5% Yes) |
| Snapshots | game-year 20 (turn 960) and game-year 30 (turn 1440) |
| Horizons | snapshot + 240 turns (5 game-years) and + 480 turns (10 game-years) |
| Written | 2026-08-31, against engine/runner state of branch `exploration` |

---

## 1. Running the simulator

From the `apps/micropolis` directory of the MicropolisCore checkout:

```sh
pnpm tsx cli/run_sim.js --city <name> --turns <n> --seed <s> --output-base-dir <dir> [--no-disasters]
```

- **`--turns N`** — run N in-game turns (1 turn = 16 engine ticks = 1 `cityTime` step;
  **48 turns = 1 game year**). To resolve the year-30 snapshot at the 10-year horizon you
  need `--turns 1920` (40 years). Prefer `--turns` over the older `--ticks`.
- **`--seed S`** — integer RNG seed (default 42). A run is fully reproducible given
  (city, seed, turns). Use disjoint seed sets when you need independent Monte Carlo runs.
- **`--output-base-dir DIR`** — output root (default `./sim-runs/`).
- **`--no-disasters`** — turns random disasters off. The benchmark uses disasters **on**
  (the default).
- Do not pass `--log-every-tick` or `--verbose-log`; the defaults produce exactly the
  format described below.

Cost: ≈2 s for a 40-year run of a mid-size city on a laptop (bigger cities a few times
more); one process per run. Output streams are only guaranteed complete when the process
exits.

**Eligible cities** (19 — populated, non-scenario builtins):

```
badnews bruce deadwood freds haight happisle joffburg kamakura kobe kowloon
kyoto linecity med_isle ndulls radial senri southpac wetcity yokohama
```

Excluded: `about`, `bluebird`, `finnigan`, `neatmap`, `splats` (zero-population maps) and
all `scenario_*` maps (scripted win/lose conditions and forced disasters).

Note: `happisle` is the only builtin saved at game level 1 (medium); its random-disaster
rates are ≈2× the others'. All other builtins load at level 0.

### Output files

Each run writes two JSONL files:

```
<output-base-dir>/<city>/log-seed<seed>-disasters.jsonl      # stats: one JSON row per turn
<output-base-dir>/<city>/events-seed<seed>-disasters.jsonl   # events: one JSON row per engine callback
```

(the suffix becomes `-nodisasters` if disasters were disabled).

---

## 2. Reading the outputs

### 2.1 Turn arithmetic (read this first)

`cityTime` is the engine's absolute clock and is **restored from the city save file**, so
it does not start at 0 (e.g. kamakura's first row has `cityTime = 4467`). Never use
absolute `cityTime` or `cityYear`; work in **elapsed turns since run start**:

- The stats file has exactly one row per turn, with consecutive `cityTime` values.
  With 0-based row index `k`, **row `k` is the state at the end of elapsed turn `k+1`.**
  So `state_at(T) = rows[T-1]` for elapsed turn `T ≥ 1`.
- For an event `e`: `elapsed_turn(e) = e.cityTime - rows[0].cityTime + 1`. Events with
  `elapsed_turn ≤ 0` are load-time noise; ignore them.
- **Yearly checkpoints** are elapsed turns 48, 96, 144, … (`rows[47]`, `rows[95]`, …).
- Benchmark snapshot turns: `NOW = 960` (year 20) or `1440` (year 30);
  `HORIZON = NOW + 240` or `NOW + 480`.
- Every "by turn {HORIZON}" question is evaluated over the half-open window
  `(NOW, HORIZON]` — strictly after the snapshot turn, up to and including the horizon
  turn.

### 2.2 Stats rows (`log-seed*.jsonl`)

One JSON object per turn. Fields used by the questions (others exist; ignore them):

| Field | Meaning |
|:--|:--|
| `cityTime` | absolute in-game clock; use for turn arithmetic as above |
| `cityPop` | headline population (recomputed at each yearly evaluation; constant between evaluations) |
| `cityClass` | size class ordinal: 0 Village (<2k), 1 Town (2k–10k), 2 City (10k–50k), 3 Capital (50k–100k), 4 Metropolis (100k–500k), 5 Megalopolis (>500k) |
| `cityScore` | overall evaluation score (0–1000) |
| `pollutionAverage` | citywide average pollution (0–255) |
| `unpoweredZoneCount` | zone tiles currently lacking power |
| `census` | object of map-wide tile counts, recomputed every turn (see below) |

`census` keys: `water, shore, tree, rubble, flood, radioactive, fire, road, wire, rail,
zoneCenters`. The questions use `census.rubble` (rubble tiles), `census.fire` (tiles
actively burning this turn), and `census.road` (road tiles).

The first row already contains a real `cityPop` (the engine runs its first evaluation
before the first row is logged). Defensive rule anyway: population scans specified below
start at the first **yearly checkpoint** (elapsed turn 48), never at row 0.

### 2.3 Event rows (`events-seed*.jsonl`)

One JSON object per engine callback. The only event type needed for resolution is
`sendMessage`:

```json
{"tick": 295, "event": "sendMessage", "cityTime": 4485, "cityYear": 1993, "cityMonth": 5,
 "messageNum": 41, "x": 98, "y": 53, "pictureFlag": true, "important": false,
 "messageText": "Heavy Traffic reported."}
```

Filter on `event == "sendMessage"`, map `cityTime` to an elapsed turn, and match
`messageNum`. Every row also carries `messageText`; **match both number and text** as a
drift guard — if they ever disagree with the table below, stop and investigate rather
than trusting either.

Message numbers used by the questions:

| # | `messageText` | Used by |
|--:|:--|:--|
| 15 | `Blackouts reported. Check power map.` | A11 |
| 20 | `Fire reported!` | B4 |
| 21 | `A monster has been sighted!` | A4, B7 |
| 22 | `Tornado reported!` | A2, B3 |
| 23 | `Major earthquake reported!` | A1, B2 |
| 24 | `A plane has crashed!` | A5 |
| 25 | `Shipwreck reported!` | A6 |
| 26 | `A train crashed!` | B5 |
| 42 | `Flooding reported!` | A3, B6 |
| 43 | `A Nuclear Meltdown has occurred!` | B1 |

Known double-signals — count **one**, never both:

- Message **27** (`A helicopter crashed!`) is the same mid-air collision as message 24,
  emitted on the same turn. Resolve plane-crash questions on 24 only.
- A `startEarthquake` event fires on exactly the same turns as message 23. Resolve
  earthquake questions on message 23 only.

Messages that never occur in no-mayor runs (useful as negative tests for your parser):
**30** (`Firebombing reported!`) and **44** (`They're rioting in the streets!`) — zero
occurrences across 1,900 fifty-year reference runs.

One flood emits exactly one message 42 (verified over 380 runs / 238 flood messages:
consecutive 42s are never closer than 20 turns), so raw message counts are safe for B6.

---

## 3. The questions

Presentation to the model: replace `{HORIZON}` with the absolute elapsed-turn number
(e.g. 1200), always in the run's future. The approximate Yes-probability ranges are the
spread across the 19 cities × 4 snapshot/horizon windows, from ≈1,900 reference runs plus
a dedicated 19-city × 20-seed × 40-year sweep; use them for sanity-checking your
resolver and scoring pipeline, not as ground truth per city.

Notation: `msgs(m, a, b]` = count of `sendMessage` events with `messageNum == m` and
elapsed turn in `(a, b]`. `pop(T)`, `class(T)`, `score(T)`, `poll(T)`, `rubble(T)`,
`fire(T)`, `road(T)` = the corresponding stats-row fields at elapsed turn `T`
(`rows[T-1]`). `NOW` = snapshot turn, `H` = horizon turn. `YC(a, b]` = yearly checkpoints
(multiples of 48) in `(a, b]`.

### Part A — mid-range (target ≈10–90%)

| ID | Question text | Resolution (Yes iff) | ≈ P(Yes) range |
|:--|:--|:--|:--|
| A1 | Will at least one earthquake be reported by turn {HORIZON}? | `msgs(23, NOW, H] ≥ 1` | 5% (5 yr) – 20% (10 yr, happisle); median ≈8% |
| A2 | Will at least one tornado be sighted by turn {HORIZON}? | `msgs(22, NOW, H] ≥ 1` | same as A1 (statistically identical process) |
| A3 | Will at least one flood be reported by turn {HORIZON}? | `msgs(42, NOW, H] ≥ 1` | 0 in 5 cities (see §5); 8–20% elsewhere |
| A4 | Will a monster be sighted by turn {HORIZON}? | `msgs(21, NOW, H] ≥ 1` | 0.5–35%; scales with pollution staying > 60 |
| A5 | Will an airplane crash by turn {HORIZON}? | `msgs(24, NOW, H] ≥ 1` | 0 in 8 airport-less cities; 0–85% elsewhere (hinges on airport survival) |
| A6 | Will a shipwreck be reported by turn {HORIZON}? | `msgs(25, NOW, H] ≥ 1` | ≈0 in ~15 cities; 40–75% (freds, ndulls), 8–16% (wetcity) |
| A7 | Will the city's population at turn {HORIZON} be lower than it is at the current turn? | `pop(H) < pop(NOW)` | 30–85%; median ≈65% |
| A8 | Will the city's population at turn {HORIZON} be less than half of its current value? | `pop(H) < 0.5 * pop(NOW)` | 0–80%; median ≈15% |
| A9 | Will the city's population read zero at any yearly checkpoint by turn {HORIZON}? | `min over T in YC(NOW, H] of pop(T) == 0` | 0 for most; up to 65% (kowloon, linecity late windows) |
| A10 | Will the city's classification at turn {HORIZON} be lower than it is now? | `class(H) < class(NOW)` | 0–70%; median ≈15% |
| A11 | Will a "Blackouts reported" advisory appear by turn {HORIZON}? | `msgs(15, NOW, H] ≥ 1` | 5–100%; median ≈45% |
| A12 | Will the citywide average pollution level exceed 60 at turn {HORIZON}? | `poll(H) > 60` | 0–100%; median ≈55% |
| A13 | Will the map hold at least 50 more rubble tiles at turn {HORIZON} than it does now? | `rubble(H) - rubble(NOW) >= 50` | 0–95%; median ≈25% |
| A14 | Will at least one map tile be actively burning at turn {HORIZON}? | `fire(H) > 0` | 0–95%; median ≈25% |
| A15 | Will the map contain fewer road tiles at turn {HORIZON} than it does now? | `road(H) < road(NOW)` | 0–100%; median ≈50% |
| A16 | Will the city's evaluation score at turn {HORIZON} be higher than it is now? | `score(H) > score(NOW)` | 25–90%; median ≈45% |

Wording note for A10: append the example "(e.g. Metropolis → Capital)" and, if the game
report does not already define classes, the ordinal table from §2.2.

### Part B — tail (target ≈0.5–5%)

| ID | Question text | Resolution (Yes iff) | ≈ P(Yes) range |
|:--|:--|:--|:--|
| B1 | Will a nuclear meltdown occur by turn {HORIZON}? | `msgs(43, NOW, H] ≥ 1` | 0 in 9 plant-less cities; 0.8–1.6% single plant; up to ~15% (deadwood/badnews, 8 plants, 10 yr) |
| B2 | Will two or more earthquakes be reported by turn {HORIZON}? | `msgs(23, NOW, H] ≥ 2` | 0.15% (5 yr) – 2.2% (10 yr happisle) |
| B3 | Will two or more tornadoes be sighted by turn {HORIZON}? | `msgs(22, NOW, H] ≥ 2` | same as B2 |
| B4 | Will a "Fire reported!" disaster strike by turn {HORIZON}? | `msgs(20, NOW, H] ≥ 1` | 0.2–7%; median ≈1.5% |
| B5 | Will a train crash by turn {HORIZON}? | `msgs(26, NOW, H] ≥ 1` | 0 without rail traffic; ≈0.2–1.6% with |
| B6 | Will two or more separate floods be reported by turn {HORIZON}? | `msgs(42, NOW, H] ≥ 2` | 0 in the 5 unfloodable cities; 0.6–2.1% elsewhere |
| B7 | Will the monster be sighted two or more times by turn {HORIZON}? | `msgs(21, NOW, H] ≥ 2` | ≈0 clean cities; 1–6% dirty cities |
| B8 | Will the city's population reach a new all-time high at any yearly checkpoint by turn {HORIZON}? | `max over T in YC(NOW, H] of pop(T) > max over T in YC(0, NOW] of pop(T)` | ≈0–2% for most; 10–40% for still-growing cities (med_isle, deadwood, senri, ndulls) |
| B9 | Will the city's classification at turn {HORIZON} be higher than it is now? | `class(H) > class(NOW)` | ≈0 mostly; up to 20% (kyoto, kamakura near a boundary) |

---

## 4. Reference resolver

Self-contained Python; resolves every question for one run. Treat this as the normative
definition where prose and code could be read differently.

```python
import json


def load_run(stats_path, events_path):
    rows = [json.loads(l) for l in open(stats_path) if l.strip()]
    t0 = rows[0]["cityTime"]  # first row = end of elapsed turn 1
    msgs = []  # (elapsed_turn, messageNum) pairs
    for l in open(events_path):
        e = json.loads(l)
        if e.get("event") == "sendMessage":
            msgs.append((e["cityTime"] - t0 + 1, e["messageNum"]))
    return rows, msgs


def resolve(rows, msgs, now, h):
    """now, h: elapsed turns (e.g. 960 and 1200). Requires len(rows) >= h."""
    at = lambda t: rows[t - 1]  # state at end of elapsed turn t
    n = lambda m: sum(1 for t, mm in msgs if mm == m and now < t <= h)
    yc = lambda a, b: range(
        48 * (a // 48 + 1), b + 1, 48
    )  # yearly checkpoints in (a, b]
    pop = lambda t: at(t)["cityPop"]
    return {
        "A1": n(23) >= 1,
        "A2": n(22) >= 1,
        "A3": n(42) >= 1,
        "A4": n(21) >= 1,
        "A5": n(24) >= 1,
        "A6": n(25) >= 1,
        "A7": pop(h) < pop(now),
        "A8": pop(h) < 0.5 * pop(now),
        "A9": any(pop(t) == 0 for t in yc(now, h)),
        "A10": at(h)["cityClass"] < at(now)["cityClass"],
        "A11": n(15) >= 1,
        "A12": at(h)["pollutionAverage"] > 60,
        "A13": at(h)["census"]["rubble"] - at(now)["census"]["rubble"] >= 50,
        "A14": at(h)["census"]["fire"] > 0,
        "A15": at(h)["census"]["road"] < at(now)["census"]["road"],
        "A16": at(h)["cityScore"] > at(now)["cityScore"],
        "B1": n(43) >= 1,
        "B2": n(23) >= 2,
        "B3": n(22) >= 2,
        "B4": n(20) >= 1,
        "B5": n(26) >= 1,
        "B6": n(42) >= 2,
        "B7": n(21) >= 2,
        "B8": max(pop(t) for t in yc(now, h)) > max(pop(t) for t in yc(0, now)),
        "B9": at(h)["cityClass"] > at(now)["cityClass"],
    }
```

Snapshot/horizon grid: `(now, h)` in `(960, 1200)`, `(960, 1440)`, `(1440, 1680)`,
`(1440, 1920)` — i.e. `--turns 1920` covers all four.

---

## 5. Structural facts for validating your pipeline

These hold **deterministically** — decided by the city's map before the first tick. Use
them as assertions on resolver output; a violation means a parsing/off-by-one bug (the
message-number table is the classic failure: 42/43/44 shifted by one files meltdowns as
floods).

| Constraint | Cities |
|:--|:--|
| Flood (42) can NEVER fire (no river-edge tiles) | badnews, haight, happisle, kowloon, linecity |
| Meltdown (43) can NEVER fire (no nuclear plant) | kamakura, kobe, kowloon, kyoto, linecity, ndulls, radial, southpac, wetcity |
| Plane crash (24) / helicopter (27) can NEVER fire (no airport) | bruce, freds, linecity, med_isle, ndulls, radial, senri, southpac |
| Riots (44) and firebombing (30) never fire anywhere | all 19 |
| Messages 24 and 27 co-occur on the same turn when they fire | all others |
| `startEarthquake` events and message 23 land on identical turns | all |

Nuclear-plant counts at start (for B1 sanity): badnews 8, deadwood 8, haight 6,
happisle 5, joffburg 3, bruce/freds/med_isle/senri/yokohama 1 each.

Coarse expected behavior for smoke tests (100+ seeds): earthquake and tornado rates are
map-independent (≈1.1%/yr at level 0, ≈2.2%/yr for happisle); kobe/kowloon runs usually
show large rubble growth and active fires at late snapshots; haight shows almost none.

## 6. Gotchas

1. **Never hand-derive message numbers** from any Micropolis source you may have seen —
   this engine's enum is position-implied and versions differ. Use the table in §2.3 and
   the `messageText` cross-check.
2. **`cityPop` vs turn 0:** always anchor population scans on yearly checkpoints
   (elapsed turn ≥ 48), as the resolver does. `cityPop` is also only recomputed at
   yearly evaluations, so mid-year rows repeat the last evaluated value — checkpoint
   sampling loses nothing.
3. **`cityTime` is absolute** (restored from the save file). All window arithmetic must
   subtract `rows[0].cityTime` as in §2.1.
4. **One run = one process.** Don't reuse a process for many sequential engine
   instances; the WASM engine can crash after several disaster-heavy runs in one
   process.
5. **Money is causally inert in this port** (a known engine quirk: underfunding never
   degrades services). Don't be tempted to add treasury-based questions; they were
   deliberately excluded.
6. If the LLM-facing game report quotes turns in a different unit (ticks, years),
   convert: 16 ticks = 1 turn, 48 turns = 1 game year. `{HORIZON}` in the question text
   is an elapsed-turn number as defined in §2.1.
