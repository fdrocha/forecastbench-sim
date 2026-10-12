#!/usr/bin/env bash
# pod_setup.sh — bring a fresh civrealm/freeciv-web:1.4 pod to producing state.
#
# Rebuilt 2026-08-23 from the bench-prep-v2 fix bundle (the original bundle
# scripts were reaped by tmp cleanup; behavior reconstructed from
# worlds_mining/BENCHMARK.md and the root-cause fix plan). Includes the
# post-incident hardening:
#   - zstd + xz binaries installed (savegame-blind serialization fix; the
#     python `zstandard` fallback in savegame_parser covers any host where
#     this somehow fails)
#   - fc_logger packet logs disabled via FBSIM_FC_LOG_LEVEL (37 GB/pod ENOSPC)
#   - debug.logging_path pinned so observation states and savegames share ONE
#     recording root (split-root bug published dark worlds)
#   - FBSIM_REQUIRE_SAVEGAME_COVERAGE=1 for every serialization
#   - clean-pod asserts: no zombie civservers, no stale savegames at boot
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_ROOT="${BUNDLE_DIR}/worlds/freeciv/logs"
ENV_FILE="${BUNDLE_DIR}/pod_env.sh"

echo "== pod_setup: apt deps (incl. zstd) =="
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
# zstd is REQUIRED: fork servers historically wrote .sav.zst autosaves and a
# missing zstd binary silently blinded serialization. We both install the
# binary AND force compresstype XZ after /load, AND ship a python fallback.
apt-get install -y -qq zstd xz-utils rsync curl >/dev/null
command -v zstd >/dev/null || { echo "FATAL: zstd install failed"; exit 1; }

echo "== pod_setup: docker shim (pod IS the freeciv-web container) =="
# The vendored client code addresses savegames via `docker exec freeciv-web`
# and `docker cp ... freeciv-web:...`. On a RunPod pod the image runs AS the
# container, so a shim translates those into local commands.
if ! command -v docker >/dev/null; then
  cat > /usr/local/bin/docker <<'SHIM'
#!/usr/bin/env bash
# docker shim: this pod IS the freeciv-web container. Container-side commands
# must run as the image's service user ("docker") so that directories created
# for civserver (savegames/) stay writable by it; the pilot client runs as
# root, and a root-owned savegames dir broke `save` server-side.
cmd="${1:-}"; shift || true
case "$cmd" in
  exec)
    while [ "$#" -gt 0 ] && [ "${1#-}" != "$1" ]; do shift; done
    shift   # container name
    if [ "$(id -u)" = "0" ]; then
      exec sudo -n -u docker -- "$@"
    fi
    exec "$@"
    ;;
  cp)
    src="${1:?}"; dst="${2:?}"
    case "$src" in *:*) src="${src#*:}";; esac
    case "$dst" in *:*) dst="${dst#*:}";; esac
    cp "$src" "$dst" || exit 1
    if [ "$(id -u)" = "0" ]; then
      chown docker:docker "$dst" 2>/dev/null || true
      chmod a+r "$dst" 2>/dev/null || true
    fi
    exit 0
    ;;
  *)
    echo "docker-shim: unsupported subcommand: $cmd $*" >&2
    exit 1
    ;;
esac
SHIM
  chmod 755 /usr/local/bin/docker
  echo "installed /usr/local/bin/docker shim"
fi
docker exec freeciv-web true || { echo "FATAL: docker(-shim) exec failed"; exit 1; }

echo "== pod_setup: uv + python 3.12 =="
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
cd "${BUNDLE_DIR}"
uv venv --clear --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e packages/fbsim-core -e worlds/freeciv

echo "== pod_setup: pin single recording root =="
mkdir -p "${LOG_ROOT}"
# Pin debug.logging_path explicitly (belt and braces: the shipped
# logging_config default already resolves here, but drivers must never again
# depend on path math lining up).
SETTINGS="${BUNDLE_DIR}/worlds/freeciv/freeciv_world/configs/default_settings.yaml"
sed -i "s#^  logging_path: null#  logging_path: ${LOG_ROOT}#" "${SETTINGS}"
grep -q "logging_path: ${LOG_ROOT}" "${SETTINGS}" \
  || { echo "FATAL: failed to pin logging_path"; exit 1; }

echo "== pod_setup: persistent env =="
cat > "${ENV_FILE}" <<EOF
export PATH="\$HOME/.local/bin:\$PATH"
source "${BUNDLE_DIR}/.venv/bin/activate"
export FREECIV_LOCAL_MODE=1
export CIVREALM_HOST_PORT=80
# Silence fc_logger packet-debug logs (37 GB/pod ENOSPC incident).
export FBSIM_FC_LOG_LEVEL=WARNING
# Serialization hard-fails unless EVERY turn's savegame parsed.
export FBSIM_REQUIRE_SAVEGAME_COVERAGE=1
export FBSIM_LOG_ROOT="${LOG_ROOT}"
export FBSIM_OUT_DIR="\${FBSIM_OUT_DIR:-/workspace/out}"
EOF
# shellcheck disable=SC1090
source "${ENV_FILE}"
mkdir -p "${FBSIM_OUT_DIR}/worlds" "${FBSIM_OUT_DIR}/failed"

echo "== pod_setup: clean-pod asserts =="
ZOMBIES=$(pgrep -c -f '[c]ivserver' || true)
if [ "${ZOMBIES:-0}" -gt 40 ]; then
  # The image's own pool is ~35 servers; anything much above that is stale
  # pre-baked state from a dirty template.
  echo "FATAL: ${ZOMBIES} civservers at boot (stale pod state)"; exit 1
fi
STALE_SAVES=$(docker exec freeciv-web sh -c \
  'du -sm /var/lib/tomcat10/webapps/data/savegames 2>/dev/null | cut -f1' \
  2>/dev/null || echo 0)
if [ "${STALE_SAVES:-0}" -gt 100 ]; then
  echo "WARN: ${STALE_SAVES}MB stale savegames at boot; clearing"
  docker exec freeciv-web sh -c \
    'rm -rf /var/lib/tomcat10/webapps/data/savegames/*' || true
fi

echo "== pod_setup: smoke (imports + gates present) =="
python - <<'PY'
import zstandard
from freeciv_world.forking.savegame_modifier import SavegameModifier
from freeciv_world.world_reports.utils.savegame_parser import (
    decompress_savegame_content, compute_savegame_coverage)
assert hasattr(SavegameModifier, 'set_rng_from_seed')
assert hasattr(SavegameModifier, 'clear_rng_state')
print('bundle smoke OK')
PY
test -f "${BUNDLE_DIR}/bench/world_publish_gate.py"
test -f "${BUNDLE_DIR}/bench/pilot_gates.py"

echo "pod_setup complete. 'source ${ENV_FILE}' in every session."
