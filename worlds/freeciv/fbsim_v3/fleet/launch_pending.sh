#!/usr/bin/env bash
# launch_pending.sh — launch lanes on any worker pod whose bringup finished but has no launch log yet.
# Workers sync to the archive pods in $STATE/archive_pods.txt (one ID per line), dealt round-robin.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
. "$HERE/paths.sh"
i=0; ARCHS=($(grep -v '^#' "$STATE"/archive_pods.txt | grep .)); [ ${#ARCHS[@]} -gt 0 ] || { echo "no archive pods in $STATE/archive_pods.txt"; exit 1; }
while IFS=$'\t' read -r ts id name flavor vcpu cloud dc cost status; do
  case "$ts" in \#*) continue;; esac
  [ "$status" = deleted ] && continue
  grep -qx "$id" "$STATE"/archive_pods.txt 2>/dev/null && continue
  [ -f "$STATE/launch_$id.log" ] && continue
  [ -d "$STATE/queue/$id/todo" ] || { echo "$id: no queue slice"; continue; }
  if grep -q "bringup done" "$STATE/bringup_$id.log" 2>/dev/null && grep -q "pod_setup_rc=0" "$STATE/bringup_$id.log"; then
    arch=${ARCHS[$((i % ${#ARCHS[@]}))]}; i=$((i+1))
    lanes=$((vcpu*3/4)); nohup bash launch_workers.sh $arch $id:$lanes > "$STATE/launch_$id.log" 2>&1 &
    echo "launched $id ($name) lanes=$lanes -> $arch"
  else echo "pending $id ($name): $(tail -n 1 "$STATE"/bringup_$id.log 2>/dev/null | cut -c1-60)"; fi
done < "$STATE"/pods.tsv
