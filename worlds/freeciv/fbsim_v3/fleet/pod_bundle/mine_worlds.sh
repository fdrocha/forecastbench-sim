#!/usr/bin/env bash
# mine_worlds.sh SEED_START SEED_END [CONCURRENCY] [MAX_TURNS]
# Fleet driver: mines a seed range with bounded concurrency, one retry per
# failed seed, and a W3-compliant rescue pass (rescue can only publish through
# the same savegame-complete, gated path — never blind).
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${BUNDLE_DIR}/pod_env.sh"

SEED_START="${1:?usage: mine_worlds.sh SEED_START SEED_END [CONCURRENCY] [MAX_TURNS]}"
SEED_END="${2:?}"
CONCURRENCY="${3:-25}"
MAX_TURNS="${4:-301}"
DRIVER_LOG_DIR="${FBSIM_OUT_DIR}/driver_logs"
mkdir -p "${DRIVER_LOG_DIR}"

mine_seed() {
  local seed="$1"
  bash "${BUNDLE_DIR}/mine_one_world.sh" "${seed}" "${MAX_TURNS}" \
    > "${DRIVER_LOG_DIR}/seed${seed}.log" 2>&1
}

echo "Wave 1: seeds ${SEED_START}-${SEED_END} at ${CONCURRENCY}-way"
active=0
for seed in $(seq "${SEED_START}" "${SEED_END}"); do
  mine_seed "${seed}" &
  active=$((active + 1))
  if [ "${active}" -ge "${CONCURRENCY}" ]; then
    wait -n || true
    active=$((active - 1))
  fi
done
wait || true

echo "Retry pass: re-running failed seeds once"
for f in "${FBSIM_OUT_DIR}"/failed/seed*.FAILED; do
  [ -e "$f" ] || continue
  seed=$(basename "$f" .FAILED | sed 's/seed//')
  if [ "${seed}" -ge "${SEED_START}" ] && [ "${seed}" -le "${SEED_END}" ]; then
    echo "  retrying seed${seed}"
    mine_seed "${seed}" || true
  fi
done

echo "Rescue pass: gated rescue for finished-but-unpublished recordings"
for rec in "${FBSIM_LOG_ROOT}"/recordings/seed*; do
  [ -d "${rec}" ] || continue
  seed=$(basename "${rec}" | sed 's/seed//')
  case "${seed}" in (*[!0-9]*) continue;; esac
  if [ "${seed}" -ge "${SEED_START}" ] && [ "${seed}" -le "${SEED_END}" ] \
     && [ ! -s "${FBSIM_OUT_DIR}/worlds/seed${seed}_data.json.gz" ]; then
    echo "  rescuing seed${seed} (gated — publishes only with per-turn savegames)"
    bash "${BUNDLE_DIR}/rescue_serialize.sh" "${seed}" "${MAX_TURNS}" \
      >> "${DRIVER_LOG_DIR}/seed${seed}.log" 2>&1 || true
  fi
done

N_PUB=$(ls "${FBSIM_OUT_DIR}"/worlds/seed*_data.json.gz 2>/dev/null | wc -l)
N_FAIL=$(ls "${FBSIM_OUT_DIR}"/failed/seed*.FAILED 2>/dev/null | wc -l)
echo "mine_worlds done: ${N_PUB} published, ${N_FAIL} failed (re-mine elsewhere)"
grep -l "zstd" "${DRIVER_LOG_DIR}"/seed*.log 2>/dev/null \
  && echo "WARNING: zstd mentions in driver logs above — investigate" \
  || echo "No zstd errors in driver logs"
