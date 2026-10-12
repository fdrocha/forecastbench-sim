#!/usr/bin/env bash
# mine_forks.sh ANCHOR_SEED TASKS_FILE [N_LANES] [END_TURN]
# P2 lane driver. TASKS_FILE lines: "<fork_name> <rng_seed>". Tasks are dealt
# round-robin to N_LANES lanes; each lane runs its forks sequentially via
# fork_one.sh (subprocess isolation per fork; ports via the FileLock
# allocator). One retry pass at the end for failed forks.
set -uo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${BUNDLE_DIR}/pod_env.sh"

ANCHOR="${1:?usage: mine_forks.sh ANCHOR_SEED TASKS_FILE [N_LANES] [END_TURN]}"
TASKS="${2:?}"
N_LANES="${3:-10}"
END_TURN="${4:-90}"

DRIVER_LOG_DIR="${FBSIM_OUT_DIR}/fork_driver_logs"
LANE_DIR="${FBSIM_OUT_DIR}/fork_lanes"
mkdir -p "${DRIVER_LOG_DIR}" "${LANE_DIR}"
rm -f "${LANE_DIR}"/lane*.tasks

i=0
while read -r name rseed; do
  [ -n "${name}" ] || continue
  echo "${name} ${rseed}" >> "${LANE_DIR}/lane$((i % N_LANES)).tasks"
  i=$((i + 1))
done < "${TASKS}"
echo "Dealt ${i} forks to ${N_LANES} lanes"

run_lane() {
  local lane_file="$1"
  while read -r name rseed; do
    bash "${BUNDLE_DIR}/fork_one.sh" "${ANCHOR}" "${name}" "${rseed}" "${END_TURN}" \
      > "${DRIVER_LOG_DIR}/${name}.log" 2>&1 || true
  done < "${lane_file}"
}

for lf in "${LANE_DIR}"/lane*.tasks; do
  run_lane "${lf}" &
done
wait || true

echo "Retry pass: re-running failed forks once"
while read -r name rseed; do
  [ -n "${name}" ] || continue
  if [ -e "${FBSIM_OUT_DIR}/forks_failed/${name}.FAILED" ]; then
    echo "  retrying fork ${name}"
    bash "${BUNDLE_DIR}/fork_one.sh" "${ANCHOR}" "${name}" "${rseed}" "${END_TURN}" \
      >> "${DRIVER_LOG_DIR}/${name}.log" 2>&1 || true
  fi
done < "${TASKS}"

N_PUB=$(ls -d "${FBSIM_OUT_DIR}"/forks/*/ 2>/dev/null | wc -l)
N_FAIL=$(ls "${FBSIM_OUT_DIR}"/forks_failed/*.FAILED 2>/dev/null | wc -l)
echo "mine_forks done: ${N_PUB} fork dirs published, ${N_FAIL} failed"
