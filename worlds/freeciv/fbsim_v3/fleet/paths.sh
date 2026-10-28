# paths.sh — sourced by the fleet scripts once HERE is set: where the run state and the keys live (outside the checkout,
# through the repository's data/ and credentials/ symlinks; see scripts/setup_worktree.sh).
REPO="$(cd "$HERE/../../../.." && pwd)"
STATE="$REPO/data/freeciv/fbsim_v3/fleet_state"   # pods.tsv ledger, queues, anchors, bringup/launch logs
CRED="$REPO/credentials"                          # .env (RUNPOD_API_KEY), freeciv/runpod_key, freeciv/archive_key
SSHKEY="${FBSIM_SSH_KEY:-$CRED/freeciv/runpod_key}"
ARCHKEY="$CRED/freeciv/archive_key"
SSHO=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR)
# podsh PORT root@IP [CMD...] — ssh to a pod; safe when the checkout path has spaces. For rsync -e, quote the key
# inside the string instead ("ssh -i '$SSHKEY' ..."): rsync splits that string honoring quotes.
podsh() { local port="$1"; shift; ssh -i "$SSHKEY" -p "$port" "${SSHO[@]}" "$@"; }
