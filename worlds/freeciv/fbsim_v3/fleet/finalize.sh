#!/usr/bin/env bash
# finalize.sh ARCHIVE_POD [PULL_POD] — after all workers are gone: archive counts -> pull -> QC -> consolidate_v4 -> sweep_v4.
# The corpus goes to FBSIM_V3_CORPUS. Anchors: FBSIM_ANCHORS (default: the seeds in state/anchors/);
# FBSIM_N_PLAYERS (default 5) is passed to consolidate_v4 --n-main; the ruleset comes from the pulled anchors.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
C="${FBSIM_V3_CORPUS:?set FBSIM_V3_CORPUS}"; ARCH="${1:?archive pod id}"; PULLPOD="${2:-$1}"
ANCH="${FBSIM_ANCHORS:-$(ls -d state/anchors/seed* 2>/dev/null | sed 's/.*seed//' | sort -n | tr '\n' ' ')}"; [ -n "${ANCH// /}" ] || { echo "no anchors"; exit 1; }
echo "== archive per-anchor counts (forks / sgtables / truncated)"
bash rp.sh ssh "$ARCH" "for a in $ANCH; do echo \"a\$a \$(ls -d /vol/fbsim_v3/*/a\$a/forks/rng* 2>/dev/null | wc -l) \$(ls /vol/fbsim_v3/*/a\$a/forks/rng*/*_sgtables.json.gz 2>/dev/null | wc -l) \$(ls /vol/fbsim_v3/*/a\$a/forks/rng*/TRUNCATED 2>/dev/null | wc -l)\"; done; du -sh /vol/fbsim_v3" </dev/null | tee state/archive_counts.txt
echo "== pull (incremental, parallel)"; bash pull2.sh "$PULLPOD" 2>&1 | tail -n 3
echo "== corpus QC"; uv run python ../draw/corpus_qc.py $C/raw | tee state/corpus_qc.txt
RS="${FBSIM_RULESET:-$(ls $C/raw/*/anchors/seed*/ruleset.json 2>/dev/null | head -n 1)}"; [ -s "$RS" ] || { echo "no ruleset.json (set FBSIM_RULESET)"; exit 1; }
echo "== consolidate_v4"; rm -rf $C/consol_v4; uv run python ../draw/consolidate_v4.py $C/raw $C/consol_v4 --workers 4 --anchors $ANCH --n-main "${FBSIM_N_PLAYERS:-5}" --ruleset "$RS" 2>&1 | tail -n 10
echo "== sweep_v4"; uv run python ../draw/sweep_v4.py $C/consol_v4 $C/sweep_v4.json 2>&1 | tail -n 3
cp CHANNEL_READINESS.md state/RUN_LOG.md state/archive_counts.txt state/corpus_qc.txt $C/ 2>/dev/null
echo "== done $(date -u +%FT%TZ)"; ls -la $C | head -20
