#!/usr/bin/env bash
# finalize.sh ARCHIVE_POD [PULL_POD] — after all workers are gone: archive counts -> pull -> QC -> consolidate_v4 -> sweep_v4.
# The corpus goes to FBSIM_V3_CORPUS.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
C="${FBSIM_V3_CORPUS:?set FBSIM_V3_CORPUS}"; ARCH="${1:?archive pod id}"; PULLPOD="${2:-$1}"; ANCH="7001 7003 7005 7008 7010 7011 7014 7022"
echo "== archive per-anchor counts (forks / sgtables / truncated)"
bash rp.sh ssh "$ARCH" "for a in $ANCH 7016; do echo \"a\$a \$(ls -d /vol/fbsim_v3/*/a\$a/forks/rng* 2>/dev/null | wc -l) \$(ls /vol/fbsim_v3/*/a\$a/forks/rng*/*_sgtables.json.gz 2>/dev/null | wc -l) \$(ls /vol/fbsim_v3/*/a\$a/forks/rng*/TRUNCATED 2>/dev/null | wc -l)\"; done; du -sh /vol/fbsim_v3" </dev/null | tee state/archive_counts.txt
echo "== pull (incremental, parallel)"; bash pull2.sh "$PULLPOD" 2>&1 | tail -n 3
echo "== corpus QC"; uv run python ../draw/corpus_qc.py $C/raw | tee state/corpus_qc.txt
echo "== consolidate_v4"; rm -rf $C/consol_v4; uv run python ../draw/consolidate_v4.py $C/raw $C/consol_v4 --workers 4 --anchors $ANCH 2>&1 | tail -n 10
echo "== sweep_v4"; uv run python ../draw/sweep_v4.py $C/consol_v4 $C/sweep_v4.json 2>&1 | tail -n 3
cp CHANNEL_READINESS.md state/RUN_LOG.md state/archive_counts.txt state/corpus_qc.txt $C/ 2>/dev/null
echo "== done $(date -u +%FT%TZ)"; ls -la $C | head -20
