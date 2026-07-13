# Human Subjects Study

Materials for the TailRiskBench human forecasting study (pilot).

## Design Summary

- **N = 30** participants
- **Binned elicitation**: 10-bin distributional forecasts → CRPS
- **Horizons**: H1, H3, H4, H6 (2 near, 2 far)
- **Templates per world**: 3 (2 disruptable + 1 non-disruptable) × 1 civ × 4 horizons = 12 questions
- **Worlds per participant**: 2
- **Questions per participant**: 24
- **Estimated time**: ~45-55 minutes

### Research Questions

1. How hard are these questions for humans? (absolute CRPS with CIs)
2. Human performance relative to LLMs (CRPS comparison)
3. Are disruptable templates harder than non-disruptable? (main effect)
4. Are longer horizons harder? (main effect)
5. Does the disruptable gap grow at far horizons? (interaction, exploratory)

### World Assignment (Balanced Incomplete Block Design)

4 survey worlds, 6 possible pairs, 5 participants per pair, 15 observations per world.

| Pair | World A    | World B    | Participants |
|------|-----------|-----------|--------------|
| 1    | seed1976  | seed1515  | 5            |
| 2    | seed1976  | seed1327  | 5            |
| 3    | seed1976  | seed1525  | 5            |
| 4    | seed1515  | seed1327  | 5            |
| 5    | seed1515  | seed1525  | 5            |
| 6    | seed1327  | seed1525  | 5            |

### Power Analysis

At N=30, 2 worlds/person, 12 questions/world (3 templates × 1 civ × 4 horizons):
- **Horizon effect** (near vs far): ~100% power
- **Disruptability effect**: ~72% power
- **Interaction**: exploratory (underpowered by design — fine for pilot)

See `power_analysis.py` for simulation details calibrated from LLM CRPS data.

## Example Worlds

Two worlds from seeds 0-10 (which have full recordings). Shown as completed-game
examples so participants understand world dynamics before encountering survey questions.

- **seed10** — 10 civs, close two-horse race (Assamese 496 vs Burgundian 486)
- **seed1** — 11 civs, spread field (Burgundic 285, Miao 208, ... Scythian 127)

Full end-turn (turn 301) world reports in `examples/seed10/` and `examples/seed1/`.

## Survey Worlds

Four worlds from the 1000-world corpus. Participants see turn-60 world reports
and forecast outcomes at H1, H3, H4, H6.

| World      | Active @t60 | End-game pattern              | Top final scores           |
|-----------|------------|-------------------------------|---------------------------|
| seed1976  | 4          | Tight 4-way race (spread=22)  | 306, 284, 284, 283        |
| seed1515  | 4          | Tight 2-way race (spread=13)  | 283, 270, 211, 194        |
| seed1327  | 3          | One dominant + cluster (274)  | 527, 253, 253, 251        |
| seed1525  | 3          | Extreme runaway (spread=504)  | 839, 335, 208, 183        |

Turn-60 world reports and questions in `worlds/seed*/`.

## Template Selection

3 templates per world (same across all worlds):

**Disruptable:**
- `treasury_continuous` — 91% crash rate, CV=1.43, paper's showcase for anti-g
- `cities_count_continuous` — 48% crash rate, CV=0.82, intuitive small integers

**Non-disruptable:**
- `techs_continuous` — 0% crash rate, CV=0.12, monotonically increasing, stays pro-g

## Civ Selection

Player 0 per world (matches the intervention study methodology — systematic, no cherry-picking):

| World    | Player 0       | Score @t60 | Score @t301 |
|----------|---------------|-----------|------------|
| seed1976 | Afghani       | 42        | 283        |
| seed1515 | Micronesian   | 38        | 211        |
| seed1327 | Italian Greek | 39        | 253        |
| seed1525 | Maori         | 37        | 183        |

## Bin Boundaries

Corpus deciles: ground-truth outcomes pooled across all 1,019 worlds and H1-H6,
divided into 10 equal-frequency bins (~10% of actual outcomes per bin).

**Treasury (gold):**
`[0] [1-98] [99-167] [168-236] [237-330] [331-468] [469-647] [648-889] [890-1274] [1275+]`

**Cities:**
`[0-1] [2-5] [6-8] [9-10] [11-13] [14-16] [17-20] [21-24] [25-29] [30+]`

**Techs:**
`[0-24] [25-31] [32-36] [37-40] [41-44] [45-47] [48-49] [50-52] [53-56] [57+]`

Methodology matches the LLM binned elicitation study (Section 6 appendix).
Techs bins are narrow (3-5 techs each) because techs is highly predictable —
this tests whether humans can calibrate precisely on a stable metric.

## Open Questions for Zach

- **Demographics**: Target specific demographics in the final study? (affects recruit cost)
