Read the repository CLAUDE.md and this folder's README.md. Use uv run.

## Fleet state and keys

`fleet/paths.sh` (sourced by every fleet script) puts everything outside the checkout, through the repository's
`data/` and `credentials/` symlinks (`scripts/setup_worktree.sh` creates them in every worktree):

| path | contents |
|---|---|
| `credentials/.env` | `RUNPOD_API_KEY` (read by `rp.sh` when the variable is unset), `OPENROUTER_API_KEY` (the `run/` elicitation scripts) |
| `credentials/freeciv/runpod_key{,.pub}` | ssh key for every pod; `rp.sh create` passes the public key, override with `FBSIM_SSH_KEY` |
| `credentials/freeciv/archive_key{,.pub}` | ssh key the workers use to rsync to the archive pod |
| `data/freeciv/fbsim_v3/fleet_state/` | `pods.tsv`, `archive_pods.txt`, `addr_cache.tsv`, `queue/`, `anchors/`, `bundle/`, logs |

All worktrees share one fleet state and so one pod ledger: run one fleet at a time.

On a machine without the keys yet, from the repository root:

```bash
C=credentials/freeciv; mkdir -p "$C" && chmod 700 "$C"
ssh-keygen -t ed25519 -N '' -C fbsim3-runpod -f "$C/runpod_key"
ssh-keygen -t ed25519 -N '' -C fbsim3-archive -f "$C/archive_key"
```

Never copy keys or run state into the repository.
