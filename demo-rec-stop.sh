#!/usr/bin/env bash
# demo-rec-stop.sh — tear down exactly what demo-rec-start.sh created.
# Unloads the tracked module IDs, then sweeps any stray module that
# references the demo_rec sink (e.g. after a crashed run). Never does a
# blanket unload-by-name, so the AEC plumbing and any other loopbacks
# on the system are left untouched.
#
# No hardware endpoint names appear here (only the demo_rec null-sink name
# and the tracked module IDs), so the headset-rename refactor needs no
# change to this file.
set -uo pipefail

STATE=/tmp/demo-rec.modules
removed=0

if [[ -f $STATE ]]; then
    while read -r id; do
        if [[ -n ${id:-} ]] && pactl unload-module "$id" 2>/dev/null; then
            removed=$((removed+1))
        fi
    done < "$STATE"
    rm -f "$STATE"
fi

# Sweep strays: the null sink (sink_name=demo_rec) and both loopbacks
# (sink=demo_rec) all carry "demo_rec" in their argument string.
pactl list short modules | awk '/demo_rec/ {print $1}' | while read -r id; do
    pactl unload-module "$id" 2>/dev/null && echo "[demo-rec] swept stray module $id"
done

echo "[demo-rec] removed $removed tracked module(s); recording tap is gone."
