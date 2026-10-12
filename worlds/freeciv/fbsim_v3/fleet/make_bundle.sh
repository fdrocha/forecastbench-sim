#!/usr/bin/env bash
# make_bundle.sh OUT_DIR — assemble the pod bundle that bringup.sh rsyncs to each pod: pod_bundle/ plus this
# repository's packages/fbsim-core and worlds/freeciv, under a uv workspace. pod_setup.sh builds the venv on the pod.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../../../.." && pwd)"
OUT="${1:?OUT_DIR}"; mkdir -p "$OUT/packages" "$OUT/worlds"
X=(--exclude .venv --exclude __pycache__ --exclude '*.pyc' --exclude '*.egg-info')
rsync -a --delete "${X[@]}" "$HERE/pod_bundle/" "$OUT/" --exclude packages --exclude worlds --exclude pyproject.toml
rsync -a --delete "${X[@]}" "$REPO/packages/fbsim-core/" "$OUT/packages/fbsim-core/"
rsync -a --delete "${X[@]}" --exclude fbsim_v3 --exclude logs "$REPO/worlds/freeciv/" "$OUT/worlds/freeciv/"
printf '[tool.uv.workspace]\nmembers = ["packages/*", "worlds/*"]\n' > "$OUT/pyproject.toml"
echo "bundle in $OUT"
