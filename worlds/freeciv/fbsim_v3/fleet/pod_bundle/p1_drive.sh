#!/usr/bin/env bash
# p1_drive.sh SEED [SEED...] — pilot P1 driver: mine an explicit seed list with
# bounded concurrency, one retry per failed seed, and a gated rescue pass.
# Identical publish semantics to mine_worlds.sh (which takes a contiguous
# range); the pilot's dark seeds are not contiguous.
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${BUNDLE_DIR}/pod_env.sh"

CONCURRENCY="${P1_CONCURRENCY:-5}"
MAX_TURNS="${P1_MAX_TURNS:-301}"
SEEDS=("$@")
[ "${#SEEDS[@]}" -ge 1 ] || { echo "usage: p1_drive.sh SEED [SEED...]"; exit 2; }

DRIVER_LOG_DIR="${FBSIM_OUT_DIR}/driver_logs"
mkdir -p "${DRIVER_LOG_DIR}"

mine_seed() {
  local seed="$1"
  bash "${BUNDLE_DIR}/mine_one_world.sh" "${seed}" "${MAX_TURNS}" \
    > "${DRIVER_LOG_DIR}/seed${seed}.log" 2>&1
}

echo "P1 wave 1: ${#SEEDS[@]} seeds (${SEEDS[*]}) at ${CONCURRENCY}-way"
active=0
for seed in "${SEEDS[@]}"; do
  mine_seed "${seed}" &
  active=$((active + 1))
  if [ "${active}" -ge "${CONCURRENCY}" ]; then
    wait -n || true
    active=$((active - 1))
  fi
done
wait || true

echo "P1 retry pass: re-running failed seeds once"
for seed in "${SEEDS[@]}"; do
  if [ -e "${FBSIM_OUT_DIR}/failed/seed${seed}.FAILED" ]; then
    echo "  retrying seed${seed}"
    mine_seed "${seed}" || true
  fi
done

echo "P1 rescue pass: gated rescue for finished-but-unpublished recordings"
for seed in "${SEEDS[@]}"; do
  if [ -d "${FBSIM_LOG_ROOT}/recordings/seed${seed}" ] \
     && [ ! -s "${FBSIM_OUT_DIR}/worlds/seed${seed}_data.json.gz" ]; then
    echo "  rescuing seed${seed} (gated)"
    bash "${BUNDLE_DIR}/rescue_serialize.sh" "${seed}" "${MAX_TURNS}" \
      >> "${DRIVER_LOG_DIR}/seed${seed}.log" 2>&1 || true
  fi
done

N_PUB=0
for seed in "${SEEDS[@]}"; do
  [ -s "${FBSIM_OUT_DIR}/worlds/seed${seed}_data.json.gz" ] && N_PUB=$((N_PUB + 1))
done
echo "p1_drive done: ${N_PUB}/${#SEEDS[@]} published"
grep -l "zstd" "${DRIVER_LOG_DIR}"/seed*.log 2>/dev/null \
  && echo "WARNING: zstd mentions in driver logs above — investigate" \
  || echo "No zstd errors in driver logs"
