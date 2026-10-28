#!/usr/bin/env bash
# fleet_ssh.sh [--workers] CMD — run CMD on every live pod (parallel), print "id name: output". --workers excludes the pods in $STATE/archive_pods.txt.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "$HERE/paths.sh"
only_workers=0; [ "${1:-}" = "--workers" ] && { only_workers=1; shift; }
[ "$only_workers" = 1 ] && ! [ -s "$STATE/archive_pods.txt" ] && { echo "no archive pods in $STATE/archive_pods.txt"; exit 1; }
CMD="$1"; tmp="$(mktemp -d)"
while IFS=$'\t' read -r ts id name flavor vcpu cloud dc cost status; do
  case "$ts" in \#*) continue;; esac; [ "$status" = deleted ] && continue
  if [ "$only_workers" = 1 ] && grep -qx "$id" "$STATE/archive_pods.txt" 2>/dev/null; then continue; fi
  ( out="$(bash "$HERE/rp.sh" ssh "$id" "$CMD" 2>&1)" || out="UNREACHABLE: $out"; printf '%s %s: %s\n' "$id" "$name" "$(echo "$out" | tr '\n' '|' | cut -c1-300)" > "$tmp/$id" ) &
done < "$STATE/pods.tsv"
wait; cat "$tmp"/* | sort -k2; rm -rf "$tmp"
