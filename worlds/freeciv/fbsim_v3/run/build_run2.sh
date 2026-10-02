#!/bin/bash
# build_run2.sh OUT_RUN BATCHED_DIR NATCOND_DIR — score run 2 and build the paper's result files from it.
#   BATCHED_DIR: <model>/results.jsonl + calls.jsonl from elicit_v2.py (arm t1)
#   NATCOND_DIR: from_run1/results.jsonl and repinned/results.jsonl (arms t1nc, t2, nonews)
#   OUT_RUN (under ../results/) receives the layout the paper generators expect (paper/_common.py):
#     scores_v1/ (score_v2.py: SCORES.md, score_items.csv.gz, score_summary.csv), results_v1/ (results_table_v2.py:
#     wide, long, binary, continuous CSVs and the markdown table), family_horizon_scores.csv, reliability_bands.csv,
#     model_scores.csv, model_scores_with_slugs.csv.
#   bash build_run2.sh run2_paper ../results/run2b_2026-09-20/batched ../results/run2_2026-09-20/natcond
#   then: FBSIM_RUN=run2_paper FBSIM_PAPER_ROOT=<paper> python ../paper/<generator>.py
#   BANK_WEIGHTS=<bank_weights.csv> weights every bank aggregate by inverse selection probability (draw/bank_weights_v1.py),
#   TAILS_WEIGHTS=<tails_weights.csv> every tail aggregate (draw/tails_weights_v1.py):
#   BANK_WEIGHTS=../sets/draw_v1/bank_weights.csv bash build_run2.sh run2_paper_weighted ../results/run2b_2026-09-20/batched ../results/run2_2026-09-20/natcond
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; RUN="$HERE/../results/$1"; BATCHED="$2"; NATCOND="$3"; PY="${PYTHON:-python3}"
WARGS=(); [ -n "${BANK_WEIGHTS:-}" ] && WARGS+=(--bank-weights "$BANK_WEIGHTS"); [ -n "${TAILS_WEIGHTS:-}" ] && WARGS+=(--tails-weights "$TAILS_WEIGHTS")
[ -d "$BATCHED" ] && [ -d "$NATCOND" ] || { echo "missing $BATCHED or $NATCOND"; exit 1; }
mkdir -p "$RUN"
"$PY" "$HERE/score_v2.py" "$BATCHED" "$NATCOND" --out "$RUN/scores_v1" --boot 1000 --seed 0 ${WARGS[@]+"${WARGS[@]}"}
gzip -9 -f "$RUN/scores_v1/score_items.csv"
"$PY" "$HERE/results_table_v2.py" "$BATCHED" "$NATCOND" --out "$RUN/results_v1" ${WARGS[@]+"${WARGS[@]}"} --natcond-cost-from "$HERE/../results/run1_2026-09-09/results_v1/freeciv_results_wide.csv"
"$PY" "$HERE/make_freeciv_aggregates.py" "$RUN/scores_v1/score_items.csv.gz" "$RUN"
cp "$HERE/../results/run1_2026-09-09/model_scores.csv" "$HERE/../results/run1_2026-09-09/model_scores_with_slugs.csv" "$RUN/"
printf 'batched: %s\nnatcond: %s\nbuilt: %s\n' "$BATCHED" "$NATCOND" "$(date -u +%FT%TZ)" > "$RUN/SOURCES.txt"
[ -n "${BANK_WEIGHTS:-}" ] && printf 'bank weights: %s\n' "$BANK_WEIGHTS" >> "$RUN/SOURCES.txt"
[ -n "${TAILS_WEIGHTS:-}" ] && printf 'tails weights: %s\n' "$TAILS_WEIGHTS" >> "$RUN/SOURCES.txt"
echo "built $RUN"
