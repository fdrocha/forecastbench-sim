#!/usr/bin/env bash
# mine_one_world.sh SEED [MAX_TURNS] — run one all-AI world, serialize it with
# full savegame coverage, and publish ONLY through the publish gate.
#
# Publish contract (never violated, including by any rescue caller):
#   - run_world.py exit code gated (crashed games are FAILED, not published)
#   - serialization runs with FBSIM_REQUIRE_SAVEGAME_COVERAGE=1 from a root
#     that contains the per-turn savegames (single recording root)
#   - bench/world_publish_gate.py must PASS (frozen==0, wonders>0, diplomacy
#     non-empty, savegame_coverage==1.0) before anything lands in out/worlds
#   - bench/observability_gate.py must PASS (per-player metrics sourced from
#     the omniscient savegame, no -1 scores)
#   - ALL per-turn savegames are published next to the data file
#   - anything else lands in out/failed/<seed>.FAILED with a reason
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${BUNDLE_DIR}/pod_env.sh"

SEED="${1:?usage: mine_one_world.sh SEED [MAX_TURNS]}"
MAX_TURNS="${2:-301}"
REC_DIR="${FBSIM_LOG_ROOT}/recordings/seed${SEED}"
OUT_WORLDS="${FBSIM_OUT_DIR}/worlds"
OUT_FAILED="${FBSIM_OUT_DIR}/failed"
SER_DIR="${FBSIM_OUT_DIR}/serialize_tmp/seed${SEED}"
mkdir -p "${OUT_WORLDS}" "${OUT_FAILED}" "${SER_DIR}"

fail() {
  echo "seed${SEED} FAILED: $*"
  { echo "seed=${SEED}"; echo "reason: $*"; date -u; } \
    > "${OUT_FAILED}/seed${SEED}.FAILED"
  exit 1
}

echo "== seed${SEED}: run_world (${MAX_TURNS} turns) =="
python "${BUNDLE_DIR}/worlds/freeciv/scripts/run_world.py" \
  --seed "${SEED}" --max_turns "${MAX_TURNS}" --quiet
RC=$?
# Exit-code gate: a crashed game must never be serialized as a world.
# (Legitimate early in-game victories end via the normal terminated path
# and still exit 0; they carry fewer turns and are documented separately.)
[ "${RC}" -eq 0 ] || fail "run_world exited ${RC}"

echo "== seed${SEED}: savegame completeness precheck =="
[ -d "${REC_DIR}/savegames" ] || fail "no savegames/ under ${REC_DIR} (split-root?)"
N_SAVES=$(ls "${REC_DIR}/savegames" | grep -c "seed${SEED}_T" || true)
MIN_SAVES=$(( MAX_TURNS * 9 / 10 ))
[ "${N_SAVES}" -ge "${MIN_SAVES}" ] || \
  fail "only ${N_SAVES} per-turn savegames (<${MIN_SAVES}); serialization would be savegame-blind"

echo "== seed${SEED}: serialize (coverage-required) =="
export FBSIM_REQUIRE_SAVEGAME_COVERAGE=1
python "${BUNDLE_DIR}/worlds/freeciv/scripts/generate_data_batch.py" \
  --input-dir "${FBSIM_LOG_ROOT}/recordings" \
  --output-dir "${SER_DIR}" \
  --filter "seed${SEED}" --force --workers 1 \
  || fail "serialization failed (coverage or extraction error)"
DATA_JSON="${SER_DIR}/seed${SEED}_data.json"
[ -s "${DATA_JSON}" ] || fail "serializer produced no ${DATA_JSON}"

echo "== seed${SEED}: publish gate =="
python "${BUNDLE_DIR}/bench/world_publish_gate.py" "${DATA_JSON}" \
  || fail "publish gate flagged the world"

echo "== seed${SEED}: observability gate (savegame-sourced metrics) =="
python "${BUNDLE_DIR}/bench/observability_gate.py" "${DATA_JSON}" \
  --recording-dir "${REC_DIR}" --username "seed${SEED}" \
  || fail "observability gate flagged the world (client-view metrics / -1 scores / coverage)"

echo "== seed${SEED}: publish =="
gzip -c "${DATA_JSON}" > "${OUT_WORLDS}/seed${SEED}_data.json.gz" \
  || fail "gzip publish failed"
# Keep ALL per-turn savegames (downstream metrics/gates re-read them)
KEEP_DIR="${OUT_WORLDS}/savegames/seed${SEED}"
mkdir -p "${KEEP_DIR}"
cp "${REC_DIR}/savegames/seed${SEED}_T"*.sav.* "${KEEP_DIR}/" \
  || fail "savegame publish copy failed"
N_KEPT=$(ls "${KEEP_DIR}" | grep -c "seed${SEED}_T" || true)
[ "${N_KEPT}" -ge "${MIN_SAVES}" ] || fail "published only ${N_KEPT} savegames (<${MIN_SAVES})"
rm -f "${OUT_FAILED}/seed${SEED}.FAILED" 2>/dev/null
echo "seed${SEED} PUBLISHED"

# Rolling cleanup (recordings are 150-300MB/world)
rm -rf "${SER_DIR}"
rm -rf "${REC_DIR}"
exit 0
