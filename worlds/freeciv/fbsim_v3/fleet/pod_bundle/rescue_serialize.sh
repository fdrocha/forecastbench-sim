#!/usr/bin/env bash
# rescue_serialize.sh SEED [MAX_TURNS] — serialize an ALREADY-FINISHED game
# whose publish did not complete (driver crash, shipping failure, ...).
#
# HARD RULE (dark-world incident): a rescue may publish ONLY from a recording
# root that contains the per-turn savegames. If they are absent it tries ONCE
# to download them from the game container; if they are still absent the world
# is marked FAILED for re-mining. A rescue must NEVER serialize blind.
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${BUNDLE_DIR}/pod_env.sh"

SEED="${1:?usage: rescue_serialize.sh SEED [MAX_TURNS]}"
MAX_TURNS="${2:-301}"
REC_DIR="${FBSIM_LOG_ROOT}/recordings/seed${SEED}"
OUT_FAILED="${FBSIM_OUT_DIR}/failed"
mkdir -p "${OUT_FAILED}"

fail() {
  echo "seed${SEED} RESCUE-FAILED: $*"
  { echo "seed=${SEED}"; echo "reason (rescue): $*"; date -u; } \
    > "${OUT_FAILED}/seed${SEED}.FAILED"
  exit 1
}

[ -d "${REC_DIR}" ] || fail "no recording dir ${REC_DIR}"

count_saves() {
  ls "${REC_DIR}/savegames" 2>/dev/null | grep -c "seed${SEED}_T" || true
}

MIN_SAVES=$(( MAX_TURNS * 9 / 10 ))
if [ "$(count_saves)" -lt "${MIN_SAVES}" ]; then
  echo "== seed${SEED}: savegames incomplete; attempting docker download =="
  python - "${SEED}" "${REC_DIR}" <<'PY'
import sys
from freeciv_world.world_reports.utils.savegame_parser import (
    download_all_savegames_from_docker)
seed, rec_dir = sys.argv[1], sys.argv[2]
d, s, f = download_all_savegames_from_docker(f'seed{seed}', rec_dir)
print(f'downloaded={d} skipped={s} failed={f}')
PY
fi

N_SAVES=$(count_saves)
[ "${N_SAVES}" -ge "${MIN_SAVES}" ] || \
  fail "per-turn savegames unrecoverable (${N_SAVES}<${MIN_SAVES}); re-mine this seed"

# From here the flow is identical to the normal publish path — including the
# coverage requirement and the publish gate. Reuse it rather than fork it.
echo "== seed${SEED}: savegames present; running gated serialize+publish =="
SER_DIR="${FBSIM_OUT_DIR}/serialize_tmp/seed${SEED}"
mkdir -p "${SER_DIR}"
export FBSIM_REQUIRE_SAVEGAME_COVERAGE=1
python "${BUNDLE_DIR}/worlds/freeciv/scripts/generate_data_batch.py" \
  --input-dir "${FBSIM_LOG_ROOT}/recordings" \
  --output-dir "${SER_DIR}" \
  --filter "seed${SEED}" --force --workers 1 \
  || fail "serialization failed (coverage or extraction error)"
DATA_JSON="${SER_DIR}/seed${SEED}_data.json"
[ -s "${DATA_JSON}" ] || fail "serializer produced no ${DATA_JSON}"

python "${BUNDLE_DIR}/bench/world_publish_gate.py" "${DATA_JSON}" \
  || fail "publish gate flagged the rescued world"
python "${BUNDLE_DIR}/bench/observability_gate.py" "${DATA_JSON}" \
  --recording-dir "${REC_DIR}" --username "seed${SEED}" \
  || fail "observability gate flagged the rescued world"

OUT_WORLDS="${FBSIM_OUT_DIR}/worlds"
mkdir -p "${OUT_WORLDS}"
gzip -c "${DATA_JSON}" > "${OUT_WORLDS}/seed${SEED}_data.json.gz" \
  || fail "gzip publish failed"
KEEP_DIR="${OUT_WORLDS}/savegames/seed${SEED}"
mkdir -p "${KEEP_DIR}"
# Keep ALL per-turn savegames (same contract as mine_one_world.sh)
cp "${REC_DIR}/savegames/seed${SEED}_T"*.sav.* "${KEEP_DIR}/" \
  || fail "savegame publish copy failed"
rm -f "${OUT_FAILED}/seed${SEED}.FAILED" 2>/dev/null
rm -rf "${SER_DIR}"
echo "seed${SEED} RESCUE-PUBLISHED (gated)"
exit 0
