#!/usr/bin/env bash
# speak-daemon.sh — talk-llama --speak wrapper for orpheus-speak --watch mode
#
# LEGACY/FALLBACK PATH. launch-athena-397b.sh runs talk-llama with --stream-file,
# so talk-llama streams sentences to the trigger file itself and this -s handler
# never runs. It is kept only as the non-streaming batch fallback. The path
# variables below stay in sync with launch-athena-397b.sh (SPEAK_FILE /
# TRIGGER_FILE / DONE_FILE) because both DERIVE the same ATHENA_DIR from their own
# location — talk-llama's -sf writes the speak file, this script reads it, and
# orpheus-speak watches the trigger, all under the same repo dir.
#
# Manual setup (three terminals), reflecting the current ATHENA/ layout:
#
#   Terminal 1 (llama-server — Orpheus TTS backend, start first, leave running):
#     $ATHENA_DIR/llama.cpp/build/bin/llama-server -m $ATHENA_DIR/models/orpheus-3b-0.1-ft-UD-Q4_K_XL.gguf -c 13824 -np 4 -ngl 99 --host 127.0.0.1 --port 8080 --cache-type-k f16 --cache-type-v f16 -fa on
#
#   Terminal 2 (daemon — start second, leave running):
#     $ATHENA_DIR/orpheus/build/orpheus-speak --watch $ATHENA_DIR/speak_tts.txt --snac $ATHENA_DIR/orpheus/snac24_dynamic_fp16.onnx --play "aplay -q" -v
#
#   Terminal 3 (talk-llama — STT + Qwen3.5-397B):
#     $ATHENA_DIR/whisper.cpp/build/bin/whisper-talk-llama -ml $ATHENA_DIR/models/Qwen3.5-397B-A17B-UD-Q3_K_XL-00001-of-00005.gguf -mw $ATHENA_DIR/models/ggml-small.en.bin --cpu-moe --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.00 -t 16 -ngl 99 -s $ATHENA_DIR/speak-daemon.sh -sf $ATHENA_DIR/speakfile.temp -p Igor -bn Athena -mt 128 -vms 15000 --presence-penalty 1.5 --repeat-penalty 1.0 -ctk bf16 -ctv bf16 -fa; rm $ATHENA_DIR/speakfile.temp
#
# MUCH faster than speak.sh because the ONNX session stays warm.

# Repo dir, resolved through symlinks — the same ATHENA_DIR the launcher derives
# (and exports), so these speak_tts.* paths match what orpheus-speak watches.
ATHENA_DIR="${ATHENA_DIR:-$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)}"
SPEAK_FILE="$ATHENA_DIR/speakfile.temp"
TRIGGER_FILE="$ATHENA_DIR/speak_tts.txt"
DONE_FILE="$ATHENA_DIR/speak_tts.done"

if [ ! -f "$SPEAK_FILE" ]; then
    exit 0
fi

# Remove any stale done flag
rm -f "$DONE_FILE"

# Sanitize and write to trigger file (atomic via temp + mv)
TMPF=$(mktemp)
sed -E \
    -e 's/\*\*([^*]*)\*\*/\1/g' \
    -e 's/\*([^*]*)\*/\1/g' \
    -e 's/`[^`]*`//g' \
    -e 's/^#{1,6} //g' \
    -e 's/https?:\/\/[^ ]*//g' \
    -e 's/\[([^]]*)\]\([^)]*\)/\1/g' \
    -e 's/<\|[a-z_]*\|>//g' \
    "$SPEAK_FILE" > "$TMPF"

if [ ! -s "$TMPF" ]; then
    rm -f "$TMPF"
    exit 0
fi

mv -f "$TMPF" "$TRIGGER_FILE"

# Wait for daemon to finish (it creates .done after playback)
# Timeout after 1800s (30 min) to avoid hanging forever (long stories need time)
for i in $(seq 1 36000); do
    if [ -f "$DONE_FILE" ]; then
        rm -f "$DONE_FILE"
        exit 0
    fi
    sleep 0.05
done

echo "[speak-daemon.sh] WARNING: daemon did not signal completion" >&2
