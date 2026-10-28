#!/usr/bin/env bash
# requeue_slice.sh SRC_POD_ID — deal a dead pod's remaining local todo slice round-robin to the other live workers and ship it.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"; SRC="${1:?}"
. "$HERE/paths.sh"
[ -s "$STATE"/archive_pods.txt ] || { echo "no archive pods in $STATE/archive_pods.txt"; exit 1; }
targets=$(awk -F'\t' -v src="$SRC" 'NR==FNR{arch[$1]=1; next} $1!~/^#/ && $9!="deleted" && !($2 in arch) && $2!=src{print $2}' "$STATE"/archive_pods.txt "$STATE"/pods.tsv)
n=$(echo "$targets" | wc -w); i=0; mkdir -p "$STATE"/requeue
for t in $targets; do rm -rf "$STATE/requeue/$t"; mkdir -p "$STATE/requeue/$t"; done
for f in "$STATE"/queue/$SRC/todo/*; do i=$((i+1)); t=$(echo $targets | awk -v k=$(( (i-1) % n + 1 )) '{print $k}'); cp "$f" "$STATE/requeue/$t/"; done
echo "dealt $i tasks from $SRC to $n pods"
for t in $targets; do ( read -r IP PORT <<<"$(bash rp.sh addr $t)"; rsync -az --ignore-existing -e "ssh -i '$SSHKEY' -p $PORT -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o BatchMode=yes -o LogLevel=ERROR" "$STATE/requeue/$t/" "root@$IP:/workspace/fleet/queue/todo/" && cp "$STATE"/requeue/$t/* "$STATE"/queue/$t/todo/ && echo "$t +$(ls "$STATE"/requeue/$t | wc -l)" ) & done; wait
mv "$STATE"/queue/$SRC "$STATE"/queue/_requeued_$SRC
