#!/usr/bin/env bash
# demo-rec-start.sh (v4) — demo-recording tap for Athena
#
# Topology:
#
#   aplay ──────────────► headset sink ──► your ears
#                              │ .monitor
#   headset mic ──► talk-llama │
#        │                     ▼
#        └──────────────► demo_rec (null sink) ──► SSR records its monitor
#
# Taps the BARE hardware endpoints, so it works identically whether the
# launcher runs in direct mode (default) or aec mode — in aec mode her
# audio still lands on the headset sink via the canceller's forward.
# No virtual-sink volume stage exists anywhere in the recorded path.
#
# v4: the headset endpoints are RESOLVED AT RUNTIME instead of hardcoded.
# The profile suffix drifts between ACP modes (analog-stereo <-> stereo-
# fallback) while the all-zero serial stays put, which silently broke the
# old pinned names; matching on driver (alsa_output/alsa_input) + vendor
# (logi) survives that. The mic line excludes .monitor so it can never bind
# a monitor source. Override either explicitly if you run more than one
# Logitech card:   SINK=... MIC=... ./demo-rec-start.sh
#
# Other hardening (unchanged): 40 ms loopback latency (1 ms forced a tiny
# PipeWire quantum onto the USB device and starved capture), stream volumes
# force-pinned (stream-restore silently brings back saved low volumes), and
# ID-tracked teardown (demo-rec-stop.sh) instead of blanket unload-by-name.
#
# Optional mix balance:  ATHENA_GAIN=150% VOICE_GAIN=80% ./demo-rec-start.sh
set -euo pipefail

STATE=/tmp/demo-rec.modules
command -v pactl >/dev/null || { echo "pactl not found (sudo apt install pulseaudio-utils)"; exit 1; }

# Resolve the headset endpoints (an explicit SINK=/MIC= override wins). The
# trailing ' || true' keeps set -e from aborting if pactl can't reach the
# server or awk's early exit SIGPIPEs it — an empty result is reported below.
SINK=${SINK:-}
MIC=${MIC:-}
[[ -n $SINK ]] || SINK=$(pactl list short sinks   | awk '/alsa_output.*[Ll]ogi/{print $2; exit}') || true
[[ -n $MIC  ]] || MIC=$(pactl list short sources  | awk '/alsa_input.*[Ll]ogi/ && $2 !~ /\.monitor$/{print $2; exit}') || true

if [[ -z $SINK ]]; then
    echo "[demo-rec] ERROR: Logitech headset sink not found"
    echo "[demo-rec] available sinks:"
    pactl list short sinks | awk '{print "    " $2}'
    exit 1
fi
if [[ -z $MIC ]]; then
    echo "[demo-rec] ERROR: Logitech headset source not found"
    echo "[demo-rec] available sources:"
    pactl list short sources | awk '$2 !~ /\.monitor$/ {print "    " $2}'
    exit 1
fi

# Clean leftovers from a previous run so the script is idempotent.
if [[ -f $STATE ]]; then
    echo "[demo-rec] cleaning previous instance..."
    xargs -r -n1 pactl unload-module < "$STATE" 2>/dev/null || true
    rm -f "$STATE"
fi

DEF_SRC=$(pactl get-default-source)
DEF_SINK=$(pactl get-default-sink)

M1=$(pactl load-module module-null-sink sink_name=demo_rec sink_properties=device.description=Demo_Recording_Sink)
echo "$M1" >> "$STATE"
M2=$(pactl load-module module-loopback source="$SINK.monitor" sink=demo_rec latency_msec=40)
echo "$M2" >> "$STATE"
M3=$(pactl load-module module-loopback source="$MIC" sink=demo_rec latency_msec=40)
echo "$M3" >> "$STATE"

# WirePlumber sometimes auto-switches defaults to a new sink — undo that
# so Athena's playback and capture are untouched by the recording rig.
pactl set-default-sink   "$DEF_SINK"
pactl set-default-source "$DEF_SRC"
pactl set-sink-volume demo_rec 100%
pactl set-sink-mute   demo_rec 0

# Pin the streams owned by a module to explicit volumes (stream-restore
# otherwise brings back whatever was last saved for them).
pin_streams() { # $1=module-id  $2=source-output gain  $3=sink-input gain
    local mid=$1 sog=$2 sig=$3 id matched=0
    for id in $(pactl list source-outputs | awk -v m="$mid" \
        '/^Source Output #/{i=substr($3,2)} $1=="Owner" && $2=="Module:" && $3==m {print i}'); do
        pactl set-source-output-volume "$id" "$sog" && matched=1
    done
    for id in $(pactl list sink-inputs | awk -v m="$mid" \
        '/^Sink Input #/{i=substr($3,2)} $1=="Owner" && $2=="Module:" && $3==m {print i}'); do
        pactl set-sink-input-volume "$id" "$sig" && matched=1
    done
    [[ $matched -eq 1 ]] || echo "[demo-rec]   note: no streams matched module $mid; volumes left as created"
}
pin_streams "$M2" "${ATHENA_GAIN:-100%}" "100%"
pin_streams "$M3" "${VOICE_GAIN:-100%}"  "100%"

echo "[demo-rec] athena leg : $SINK.monitor  (gain ${ATHENA_GAIN:-100%})"
echo "[demo-rec] voice leg  : $MIC  (gain ${VOICE_GAIN:-100%})"
echo "[demo-rec] module ids : $(tr '\n' ' ' < "$STATE")"
echo "[demo-rec] In SimpleScreenRecorder pick:  Monitor of Demo_Recording_Sink"
echo "[demo-rec] Tear down with demo-rec-stop.sh."
