# Survey Screens — CivBench Human Subjects Pilot

Screen-by-screen layout for Nik to build in Quorum.

## Flow

Every participant sees screens 01–04 (common), then two worlds (randomized via BIBD), then 13–14.

```
01  Consent
02  Instructions (FreeCiv background)
03  Example world (completed game — seed10)
04  Task intro (how bins work + practice question)
    --- World 1 (randomized) ---
05  World report (turn 60 game state + territory maps)
06  Treasury questions (4 horizons × 5 bins each)
07  Cities questions (4 horizons × 5 bins each)
08  Techs questions (4 horizons × 5 bins each)
    --- World 2 (randomized) ---
09  World report
10  Treasury questions
11  Cities questions
12  Techs questions
    ---
13  Demographics
14  Debrief
```

Total questions per participant: 3 templates × 4 horizons × 2 worlds = 24 questions.
Each question = 5 bins summing to 100%.

## Randomization (Balanced Incomplete Block Design)

4 worlds, 6 possible pairs, 5 participants per pair = 30 total.

| Pair | World A    | World B    | N |
|------|-----------|-----------|---|
| 1    | seed1976  | seed1515  | 5 |
| 2    | seed1976  | seed1327  | 5 |
| 3    | seed1976  | seed1525  | 5 |
| 4    | seed1515  | seed1327  | 5 |
| 5    | seed1515  | seed1525  | 5 |
| 6    | seed1327  | seed1525  | 5 |

Each world seen by 15 participants. Order within pair is also randomized (A-B vs B-A).

For a quick pilot: skip randomization, everyone sees seed1976 only (screens 05–08).

## Bin Parameters

5 bins per question: 4 uniform-width bins + 1 open-ended top bin.

| Template | min_max_range | Bin width | Bins | Unit suffix |
|----------|--------------|-----------|------|-------------|
| Treasury | [0, 2000]    | 500       | [0–500] [500–1000] [1000–1500] [1500–2000] [2000+] | gold |
| Cities   | [0, 40]      | 10        | [0–10] [10–20] [20–30] [30–40] [40+] | cities |
| Techs    | [0, 60]      | 15        | [0–15] [15–30] [30–45] [45–60] [60+] | techs |

Ranges derived from 2nd–98th percentile of 6,095 ground-truth observations (1,019 worlds × 6 horizons), rounded to clean integers.

## Files

- `common/` — screens shown to all participants (01–04, 13–14)
- `worlds/{seed}/` — per-world screens (05–08, reused as 09–12 for world 2)
- `questions_json/` — question data in Nik's JSON format
