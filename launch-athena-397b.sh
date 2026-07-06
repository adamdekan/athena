#!/usr/bin/env bash
# launch-athena.sh — One-click launcher for the Athena voice assistant
#
# Starts all three processes (llama-server, orpheus-speak, talk-llama),
# waits for Ctrl+C, then shuts everything down and cleans up temp files.
#
# Suitable for launching from a desktop shortcut.
#
# Hardware: ThinkPad P16 Gen 3
#   CPU:  Intel Core Ultra 9 285HX (8P + 16E = 24 cores / 24 threads — no SMT)
#   RAM:  192 GB DDR5-4000
#   GPU:  NVIDIA RTX PRO 5000 Blackwell Laptop 24 GB GDDR7 (896 GB/s)
#   LLM:  Qwen3.5-397B-A17B UD-Q3_K_XL (routed experts on CPU via --cpu-moe)
#
# ─────────────────────────────────────────────────────────────────────────────

set -e
set -E   # so ERR traps are inherited by shell functions
trap 'echo "[launch-athena] ERROR at line $LINENO: \"$BASH_COMMAND\" exited with $?" >&2' ERR

# ── Paths (auto-derived — the repo is relocatable) ────────────────────────────
# ATHENA_DIR is the directory holding this script, resolved THROUGH any symlink and
# independent of the caller's working directory — so the .desktop launcher, a symlink
# in ~/bin, cron, and a plain `./launch-athena-397b.sh` all resolve the same repo.
# Everything below is relative to it. Export ATHENA_DIR before launch only if the
# script lives apart from the models/binaries.
ATHENA_DIR="${ATHENA_DIR:-$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)}"
export ATHENA_DIR   # children (speak-daemon.sh, the monitor) inherit it

# llama-server is used ONLY for the Orpheus TTS backend (:8080); the conversational
# model runs in-process inside whisper-talk-llama, not over HTTP. The system binary
# (b9355) already runs on this GPU, so we reuse it and skip building llama.cpp in-tree.
# Override with LLAMA_SERVER=/path/to/llama-server if you build one yourself.
LLAMA_SERVER="${LLAMA_SERVER:-$(command -v llama-server 2>/dev/null || echo "$ATHENA_DIR/llama.cpp/build/bin/llama-server")}"
ORPHEUS_SPEAK="$ATHENA_DIR/orpheus/build/orpheus-speak"
TALK_LLAMA="$ATHENA_DIR/whisper.cpp/build/bin/whisper-talk-llama"

ORPHEUS_MODEL="$ATHENA_DIR/models/orpheus-3b-0.1-ft-UD-Q4_K_XL.gguf"
SNAC_MODEL="$ATHENA_DIR/orpheus/snac24_dynamic_fp16.onnx"
# Conversational model retargeted for THIS device (16 GB VRAM / 62 GB RAM): the
# Qwen3.6-35B-A3B MoE (Q4_K_XL, single-file GGUF), using the optimized inference
# settings from ~/qwen_t8.sh. The 397B needed 192 GB RAM and does not fit here.
# Attention/dense layers run on the GPU (-ngl 99); most routed experts on CPU, with a
# few layers' experts offloaded to the GPU for speed (see ATHENA_N_CPU_MOE below).
QWEN_MODEL="${QWEN_MODEL:-$HOME/.models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
QWEN_SHARDS=1   # single-file GGUF (not a split model)
# STT model. Upgraded small.en -> large-v3-turbo to spend VRAM headroom on transcription
# accuracy (the front of the pipeline; STT errors cascade into wrong replies). turbo is
# multilingual, so language is pinned to English in the talk-llama invocation (-l en).
WHISPER_MODEL="${WHISPER_MODEL:-$ATHENA_DIR/models/ggml-large-v3-turbo.bin}"
VAD_MODEL="$ATHENA_DIR/models/ggml-silero-v6.2.0.bin"   # Silero streaming end-of-turn VAD (./models/download-vad-model.sh silero-v6.2.0)

# MoE expert placement. Qwen3.6-35B-A3B has 40 MoE layers (~490 MiB experts each).
# ATHENA_N_CPU_MOE = number of layers whose experts stay in system RAM; the remaining
# (40 - N) layers' experts run on the GPU for faster prefill/decode. Default 34 → 6 layers
# on GPU (~3 GB), measured full-pipeline peak 14.3/16.3 GB (~2 GB margin for the desktop
# compositor, which itself uses ~1.2 GB here). Lower N = faster + more VRAM; 0 = all on CPU
# (safest). Above ~32 the pipeline risks OOM if the desktop GPU load spikes.
ATHENA_N_CPU_MOE="${ATHENA_N_CPU_MOE:-34}"
SPEAK_DAEMON="$ATHENA_DIR/speak-daemon.sh"

# emotion2vec speech-emotion tagging (ONNX, runs inline in talk-llama on GPU 0).
# Adjust the path to wherever you placed the exported model. Comment the export
# (or remove the file) to disable; set ATHENA_EMOTION_CPU=1 to force the CPU EP.
EMOTION_MODEL="$ATHENA_DIR/models/emotion2vec_plus_large.onnx"
[ -f "$EMOTION_MODEL" ] && export ATHENA_EMOTION_ONNX="$EMOTION_MODEL"
#export ATHENA_EMOTION_CPU=1          # <-- add this line to run emotion detection on CPU
export ATHENA_EMOTION_DEBUG=1

# ── Emotion profile ───────────────────────────────────────────────────────────
# Two profiles, selected by ATHENA_EMOTION_CALIBRATION (default = production).
#
# PRODUCTION (default): negative-collapse ON. The v2 calibration (5 reps/class)
# showed emotion2vec only resolves coarse VALENCE on this model+mic — {angry,
# disgusted, fearful, sad} are acoustically indistinguishable here, and natural
# (non-shouted) delivery of most things collapses to neutral. The tagger folds
# those four to one [emotion: negative] (ATHENA_EMOTION_COLLAPSE_NEGATIVE), giving
# an honest happy / surprised / negative signal. Floors (revised after the
# 2026-06-30 two-session demo — see CHANGES.MD §8):
#   • happy = 0.50  LOWERED from 0.95 for the demo. 0.95 sat above a fear-leak
#                   (fear-induced happy ~0.916) and below true happy (0.992) at
#                   calibration — but in the demo genuine happy peaked at only 0.58
#                   (BELOW the fear confuser), so no floor cleanly separates them.
#                   0.50 lets a clearly-bright take fire, accepting that fearful
#                   speech ~0.92 could mislabel as happy (it did NOT in either run —
#                   negatives read happy=0.000). Raise back to 0.95 for production
#                   if upset speech starts tagging happy.
#   • sad = 0.60    RAISED from the 0.50 base. sad is the negative "sink"; at 0.50 it
#                   caught a warm/playful beat (0.548) as negative. 0.60 trims those
#                   marginal false-positives while keeping every genuine negative
#                   (all observed true sads were >=0.92). Costs a little negative
#                   recall in the 0.50–0.60 band (none seen in the demo).
#   • disgusted rides the 0.50 base; folded to "negative", confusion among the
#     negatives costs nothing.
#   • angry / fearful / surprised also ride base (they essentially never fire —
#     surprised never exceeded 0.008 across the whole demo: a model limit, not a
#     floor one, so no threshold recovers it).
#
# CALIBRATION (ATHENA_EMOTION_CALIBRATION=1): collapse OFF + per-class precision
# floors (happy/sad/disgusted = 0.95/0.90/0.90), for measuring individual classes
# from the debug distribution. The full 9-class line is logged either way, but the
# EMITTED label stays specific here.
#
#   production:   ./launch-athena-397b.sh
#   calibration:  ATHENA_EMOTION_CALIBRATION=1 ./launch-athena-397b.sh
# talk-llama echoes the effective floors at startup ("emotion: floors ...").
ATHENA_EMOTION_CALIBRATION="${ATHENA_EMOTION_CALIBRATION:-0}"
if [ "$ATHENA_EMOTION_CALIBRATION" = "1" ]; then
    # Per-class calibration: specific labels, precision floors.
    unset ATHENA_EMOTION_COLLAPSE_NEGATIVE
    export ATHENA_EMOTION_MIN=0.50
    export ATHENA_EMOTION_MIN_HAPPY=0.95
    export ATHENA_EMOTION_MIN_SAD=0.90
    export ATHENA_EMOTION_MIN_DISGUSTED=0.90
else
    # Production: fold the negative classes to [emotion: negative].
    export ATHENA_EMOTION_COLLAPSE_NEGATIVE=1
    export ATHENA_EMOTION_MIN=0.50
    export ATHENA_EMOTION_MIN_HAPPY=0.40    # was 0.95 (unreachable: true happy peaked 0.58) — demo lever, see note & CHANGES.MD §8
    export ATHENA_EMOTION_MIN_SAD=0.60      # was base 0.50; trims marginal false-positives (warm/playful ~0.55) while keeping true sad (>=0.92)
    unset ATHENA_EMOTION_MIN_DISGUSTED
fi

# Production GPU env — defaulted HERE, not only in the desktop icon (CHANGES.MD §16
# Fix 4). Previously only `Athena 397B.desktop` set these, so a direct shell launch
# silently reverted to pinned host buffers + no-MPS — a different (and less tested)
# memory configuration than every production run. Overridable per-run as usual.
#   GGML_CUDA_NO_PINNED=1  all GPU<->host staging on pageable memory (no cudaMallocHost)
#   ATHENA_MPS=1           single shared CUDA context for the three GPU clients
export GGML_CUDA_NO_PINNED="${GGML_CUDA_NO_PINNED:-1}"
ATHENA_MPS="${ATHENA_MPS:-1}"

# Temp files
SPEAK_FILE="$ATHENA_DIR/speakfile.temp"
TRIGGER_FILE="$ATHENA_DIR/speak_tts.txt"
DONE_FILE="$ATHENA_DIR/speak_tts.done"
STOP_FILE="$ATHENA_DIR/speak_tts.stop"   # barge-in: talk-llama -> orpheus-speak abort request
WAV_FILE="/dev/shm/orpheus_tts.wav"
OPT_CACHE="${SNAC_MODEL}.optimized"

# Cross-session memory + long-term personality. talk-llama reads its two injected
# text files (memory.txt, personality.txt) and two sidecars (memory.state.tsv,
# personality.ledger) plus a "meta" timestamp here. Consolidation writes them
# only on a graceful "Goodbye Athena" (a Ctrl+C is SIGTERM'd before that point).
# Comment this out (or set empty) to run with memory fully disabled — byte-for-
# byte the old behavior.
MEMORY_DIR="$ATHENA_DIR/athena_memory"

# PIDs for cleanup
PID_LLAMA=""
PID_ORPHEUS=""
PID_TALK=""
PID_WATCHDOG=""          # llama-server health-watchdog subshell (CHANGES.MD §12)
PID_KEEPALIVE=""         # llama-server idle-keepalive subshell

# ── llama-server watchdog / keepalive / coredumps (all A/B-overridable) ──────────
# The 20260701-123219 run proved llama-server can die a CONTAINED userspace libcuda
# GPF while MPS keeps the brain + orpheus-speak healthy — a restart restores TTS in
# seconds at ZERO steady-state cost (acts only on failure).
ATHENA_WATCHDOG="${ATHENA_WATCHDOG:-1}"                 # 0 disables auto-restart
ATHENA_KEEPALIVE="${ATHENA_KEEPALIVE:-1}"               # 0 disables the idle 1-token poke
# DEFAULT 0 since CHANGES.MD §19: on this corrupting box a client's core dump is a
# large ext4 write whose folio allocations draw poisoned pages — the 20260702-124334
# fatal oops was INSIDE the coredump path (do_coredump→elf_core_dump→...→folio_alloc),
# i.e. the dump turned a survivable client abort (Xid→watchdog restart) into a reboot.
# We already have the libcuda-GPF backtraces from prior runs. Set =1 to re-enable for
# forensics on a NON-corrupting kernel.
ATHENA_COREDUMP="${ATHENA_COREDUMP:-0}"                 # 1 re-enables llama/orpheus core dumps
LLAMA_PIDFILE="/dev/shm/athena-llama-server.$$.pid"     # live llama-server PID (survives restarts)
ORPHEUS_PIDFILE="/dev/shm/athena-orpheus-speak.$$.pid"  # live orpheus-speak PID (survives restarts, §20)
LLAMA_WEDGE="/dev/shm/athena-llama-wedge.$$"            # keepalive-detected decode wedge (§20: /health is
                                                        # hardcoded "ok" — it cannot see a stuck decode)
LLAMA_WD_STOP="/dev/shm/athena-wd-stop.$$"              # teardown sentinel (blocks respawn/poke, both watchdogs)
PID_ORPHWD=""                                           # orpheus-speak watchdog subshell (§20)
WD_INTERVAL="${ATHENA_WD_INTERVAL:-2}"                  # health-poll cadence (s)
WD_FAIL_THRESHOLD="${ATHENA_WD_FAIL_THRESHOLD:-2}"      # consecutive /health misses ⇒ wedged
WD_MAX_RESTARTS="${ATHENA_WD_MAX_RESTARTS:-5}"          # cap within …
WD_WINDOW="${ATHENA_WD_WINDOW:-300}"                    # … rolling window (s) before giving up
CORE_ULIMIT=0            # raised to 'unlimited' for llama/orpheus ONLY (never the 161 GiB brain)
CORE_PATTERN_SAVE=""     # original /proc/sys/kernel/core_pattern, to restore on exit
PID_GPUMON=""

# ── Diagnostics ───────────────────────────────────────────────────────────────
# DEFAULT 0 (clean-system deployment — no forensic instrumentation). When set to 1,
# ATHENA_DIAG captures per-run diagnostics into a timestamped dir AND spawns the
# GPU/Xid sampler (athena-gpu-monitor.sh, which READS THE KERNEL LOG via dmesg/kmsg)
# AND runs the prior-crash pstore harvest + nvidia-bug-report + pstore-clear below.
# Contents when on:
#   orpheus-server.log  llama-server's own stdout/stderr (else /dev/null)
#   gpu.csv             2 Hz GPU clocks/power/temp/util/VRAM sample
#   gpu-proc.log        1 Hz per-process VRAM (shows WHICH process's memory drops)
#   gpu-throttle.log    1 Hz active clock-event/throttle reasons
#   xid.log             kernel Xid/NVRM faults (GPU errors invisible to the apps)
# Set ATHENA_DIAG=1 to re-enable (e.g. one bring-up run to capture the server log).
ATHENA_DIAG="${ATHENA_DIAG:-0}"

# DEFAULT 0 (clean-system deployment). When set to 1, orpheus-speak dumps, per TTS
# session, the raw model tokens (.tokens), parsed audio codes (.codes), the live
# decoded audio (.wav) and a summary (.meta) into <run-dir>/tts-capture — the data
# to root-cause TTS garbling (model output vs SSE parse vs on-GPU SNAC decode).
# Re-decode a captured .codes offline with: orpheus-speak --snac <model> --snac-cpu
#   --decode-codes <file.codes>  (writes <file>.redecode.wav for an A/B vs the .wav).
# Set ATHENA_TTS_CAPTURE=1 to re-enable (each captured session writes a few MB of WAV).
ATHENA_TTS_CAPTURE="${ATHENA_TTS_CAPTURE:-0}"
GPU_MONITOR="$ATHENA_DIR/athena-gpu-monitor.sh"
DIAG_DIR="${ATHENA_DIAG_DIR:-$ATHENA_DIR/athena-diag}"
RUN_DIR="$DIAG_DIR/$(date +%Y%m%d-%H%M%S)"
ORPHEUS_SERVER_LOG="/dev/null"
if [ "$ATHENA_DIAG" != "0" ]; then
    mkdir -p "$RUN_DIR" 2>/dev/null && ORPHEUS_SERVER_LOG="$RUN_DIR/orpheus-server.log" \
        || { RUN_DIR=""; echo "[launch-athena] WARNING: could not create $DIAG_DIR — diagnostics off"; ATHENA_DIAG=0; }
else
    # Diagnostics off: no run dir is created, so leave RUN_DIR empty. Consumers
    # already handle empty (watchdogs fall back to /dev/shm/watchdog.log; MPS log
    # to /tmp; core-dump and sampler blocks skip) — a non-empty path to a dir that
    # was never mkdir'd caused "watchdog.log: No such file or directory" spam.
    RUN_DIR=""
fi

# Prior-crash harvest: if the last boot ended in a kernel panic, its RIP+trace was
# written to efi-pstore (armed by apply-crash-capture.sh) and survives into this boot.
# Pull it — plus a full nvidia-bug-report — into athena-diag/crash-<ts>/ for the
# #1064/#1111 report, then CLEAR pstore (the EFI var store is tiny; a full one blocks
# the next dump). Runs once per launch, before we touch the GPU. Never fatal.
# Two sources, because systemd-pstore.service RACES us: at boot it moves the EFI
# records out of /sys/fs/pstore into /var/lib/systemd/pstore/<epoch>/ before this
# preflight ever runs (which is why crash-<ts>/ dirs never appeared for the earlier
# panics — the dumps were all sitting in the systemd archive). Harvest BOTH: any raw
# records still in /sys/fs/pstore, and any systemd-archived dirs newer than our marker.
if [ "$ATHENA_DIAG" != "0" ]; then
    PSTORE_MARK="$DIAG_DIR/.pstore-harvest-marker"
    PSTORE_RAW=""; PSTORE_ARCH=""
    ls /sys/fs/pstore/* >/dev/null 2>&1 && PSTORE_RAW=1
    if [ -d /var/lib/systemd/pstore ]; then
        if [ -e "$PSTORE_MARK" ]; then
            [ -n "$(find /var/lib/systemd/pstore -mindepth 1 -maxdepth 1 -newer "$PSTORE_MARK" -print -quit 2>/dev/null)" ] && PSTORE_ARCH=1
        else
            [ -n "$(find /var/lib/systemd/pstore -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ] && PSTORE_ARCH=1
        fi
    fi
    if [ -n "$PSTORE_RAW$PSTORE_ARCH" ]; then
        CRASH_DIR="$DIAG_DIR/crash-$(date +%Y%m%d-%H%M%S)"
        if mkdir -p "$CRASH_DIR" 2>/dev/null; then
            echo "[launch-athena] prior crash dump(s) found — harvesting to $CRASH_DIR"
            [ -n "$PSTORE_RAW" ] && { sudo -n cp -a /sys/fs/pstore/. "$CRASH_DIR"/raw-pstore/ 2>/dev/null \
                || echo "  (could not copy /sys/fs/pstore — need sudo)"; }
            if [ -n "$PSTORE_ARCH" ]; then
                mkdir -p "$CRASH_DIR/systemd-pstore"
                sudo -n cp -a /var/lib/systemd/pstore/. "$CRASH_DIR"/systemd-pstore/ 2>/dev/null \
                    || echo "  (could not copy systemd pstore archive — need sudo)"
            fi
            if command -v nvidia-bug-report.sh >/dev/null 2>&1; then
                sudo -n nvidia-bug-report.sh --output-file "$CRASH_DIR/nvidia-bug-report" >/dev/null 2>&1 \
                    && echo "  nvidia-bug-report saved" || echo "  (nvidia-bug-report skipped)"
            fi
            sudo -n chown -R "$(id -u):$(id -g)" "$CRASH_DIR" 2>/dev/null || true
            [ -n "$PSTORE_RAW" ] && sudo -n rm -f /sys/fs/pstore/* 2>/dev/null && echo "  /sys/fs/pstore cleared for the next dump"
            touch "$PSTORE_MARK" 2>/dev/null
            echo "  panic backtraces: look for dmesg.txt under $CRASH_DIR/systemd-pstore/<epoch>/"
        fi
    fi
fi

# Line timestamper for captured child logs: moreutils ts if present (matches the
# [YYYY-MM-DD HH:MM:SS.ffffff] format of this script's own log), else passthrough.
_ts() { if command -v ts >/dev/null 2>&1; then ts '[%Y-%m-%d %H:%M:%.S]'; else cat; fi; }

# Track whether GPU clocks were successfully locked (for cleanup)
GPU_CLOCKS_LOCKED=false

# Track whether THIS launcher started a CUDA MPS daemon, so cleanup only quits
# one we own. Values: false | true (we started it) | reused (pre-existing daemon).
MPS_STARTED=false

# Power/EPP state captured at launch so the stop path can revert it exactly.
EPP_SAVE=""           # temp file: per-core original energy_performance_preference
PPD_WAS_ACTIVE=false  # power-profiles-daemon was running and we stopped it
TUNED_SAVE=""         # original tuned profile name, if we switched it
PLATFORM_PROFILE_SAVE=""  # original ACPI platform_profile, if we changed it

# ── GPU performance setup ─────────────────────────────────────────────────────

setup_gpu_performance() {
    echo "[launch-athena] configuring GPU for maximum performance..."

    if ! command -v nvidia-smi &>/dev/null; then
        echo "[launch-athena] WARNING: nvidia-smi not found, skipping GPU tuning"
        return
    fi

    # Enable persistence mode
    sudo nvidia-smi -pm 1 2>/dev/null && echo "  persistence mode: enabled"

    # Dynamic Boost / nvidia-powerd: NOT auto-started by default (ATHENA_START_POWERD=0).
    # On the original Blackwell box the GPU<->SBIOS handshake wants nvidia-powerd driving
    # the ~95 W budget (unmanaged Dynamic Boost was a suspected Xid 69 trigger), so the
    # launcher used to start it if inactive. That's now opt-in — the launcher no longer
    # starts this system service on its own. Set ATHENA_START_POWERD=1 to restore the old
    # start-if-inactive behavior. Either way we only REPORT its state; we never stop it.
    ATHENA_START_POWERD="${ATHENA_START_POWERD:-0}"
    if command -v systemctl &>/dev/null && systemctl list-unit-files nvidia-powerd.service &>/dev/null; then
        if systemctl is-active --quiet nvidia-powerd; then
            echo "  nvidia-powerd: active (Dynamic Boost managed)"
        elif [ "$ATHENA_START_POWERD" = "1" ]; then
            echo "  nvidia-powerd: NOT active — attempting to start (ATHENA_START_POWERD=1)"
            sudo systemctl start nvidia-powerd 2>/dev/null
            if systemctl is-active --quiet nvidia-powerd; then
                echo "  nvidia-powerd: started"
            else
                echo "  WARNING: nvidia-powerd could not be started — GPU power management is degraded"
            fi
        else
            echo "  nvidia-powerd: NOT active — leaving as-is (ATHENA_START_POWERD=0 default; set =1 to auto-start)"
        fi
    fi

    # Graphics clock: deliberately NOT locked. Sweep 2026-06 (orpheus-powertune.sh
    # T2): unlocked = 167.2 tok/s vs ~158-165 for every fixed lock (900-3090).
    # Under the 95 W platform cap the boost governor beats any fixed V/F point.
    #
    # Memory clock: DEFAULT IS NOW 0 (driver-managed, no lock) — reverted CHANGES.MD §12.
    # The §10 D2 floor mode [13000,max] was a NO-OP in this passthrough container: the
    # 20260701-123219 gpu.csv shows memclk still dropped to 9001 in 45% of samples
    # despite the lock (the dead SBIOS handshake makes -lmc's floor unenforceable, same
    # root cause as -pl being rejected). Under load the driver runs 13801 either way, so
    # the floor recovered nothing and only logged a "floored at 13000" success the
    # telemetry falsifies. Default 0 = tell the truth; the levers stay for A/B:
    #   ATHENA_LOCK_MEMCLK=0        driver-managed, no lock (DEFAULT)
    #   ATHENA_LOCK_MEMCLK=1        rigid pin to max 14001 (Xid-69 suspect; A/B only)
    #   ATHENA_LOCK_MEMCLK=<floor>  lock range [<floor>, max] (floor-mode A/B)
    ATHENA_LOCK_MEMCLK="${ATHENA_LOCK_MEMCLK:-0}"
    local max_mem
    max_mem=$(nvidia-smi --query-gpu=clocks.max.memory --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')

    if [[ "$ATHENA_LOCK_MEMCLK" == "1" && -n "$max_mem" && "$max_mem" != "[N/A]" ]]; then
        if sudo nvidia-smi -lmc "$max_mem" 2>/dev/null; then
            GPU_CLOCKS_LOCKED=true
            echo "  memory clocks PINNED to ${max_mem} MHz (ATHENA_LOCK_MEMCLK=1 — Xid 69 suspect)"
        fi
    elif [[ "$ATHENA_LOCK_MEMCLK" != "0" && "$ATHENA_LOCK_MEMCLK" != "1" ]]; then
        # Floor mode: lock a range [floor, max] so the clock can still float down for
        # power but never drops to the unstable low state during power transitions.
        if [[ -n "$max_mem" && "$max_mem" != "[N/A]" ]] && \
           sudo nvidia-smi -lmc "${ATHENA_LOCK_MEMCLK},${max_mem}" 2>/dev/null; then
            GPU_CLOCKS_LOCKED=true
            echo "  memory clocks floored at ${ATHENA_LOCK_MEMCLK} MHz (range ${ATHENA_LOCK_MEMCLK}-${max_mem})"
        fi
    else
        echo "  memory clocks: left to the driver (not pinned) — default; set ATHENA_LOCK_MEMCLK=1 to A/B test"
    fi

    # ── Power ceiling: DO NOT touch -pl by default (reverted, CHANGES.MD §12) ──
    # History: §10 D2 raised -pl toward max; §11 F3 lowered it ~10 W below default.
    # The 20260701-123219 run PROVED both are pointless HERE: the SBIOS power
    # handshake is dead in this passthrough container ('PlatformRequestHandler failed
    # to get target temp/platform power mode from SBIOS' at boot), so nvidia-powerd
    # owns the ceiling outright and 'nvidia-smi -pl' is rejected ('not supported in
    # current scope'; readback stayed 105.35 W vs an 85 W request). And it wasn't
    # needed: that run rode the SAME ~95 W wall as the 074343 crash yet threw NO Xid —
    # stability came from S0ix-off (D0 pinning), not from any clock/power tuning. So by
    # default we issue NO -pl at all (a no-op here, a perf tax anywhere it IS honored).
    #   ATHENA_GPU_POWER_LIMIT=<W>  explicit cap, opt-in A/B only (perf risk if honored)
    #   ATHENA_GPU_POWER_MAX=1      opt-in: raise to board max
    local max_pl
    max_pl=$(nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' | cut -d. -f1)
    [[ "${ATHENA_GPU_POWER_MAX:-0}" == "1" ]] && ATHENA_GPU_POWER_LIMIT="${ATHENA_GPU_POWER_LIMIT:-$max_pl}"
    if [[ -n "${ATHENA_GPU_POWER_LIMIT:-}" && "$ATHENA_GPU_POWER_LIMIT" != "[N/A]" ]]; then
        if sudo nvidia-smi -pl "$ATHENA_GPU_POWER_LIMIT" 2>/dev/null; then
            local now_pl
            now_pl=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
            echo "  power limit set to ${ATHENA_GPU_POWER_LIMIT} W (now reads: ${now_pl} W — if unchanged, nvidia-powerd holds its own ceiling and ignored -pl; try ATHENA_GPU_CLOCK_CAP)"
        else
            echo "  WARNING: power limit ${ATHENA_GPU_POWER_LIMIT} W rejected by the GPU"
        fi
    fi
    # Escalation lever only (unset by default): cap the graphics clock if -pl is
    # ignored and you need to force the GPU out of the power-cap regime for an A/B.
    #   ATHENA_GPU_CLOCK_CAP=1100   cap max graphics clock at 1100 MHz
    if [[ -n "${ATHENA_GPU_CLOCK_CAP:-}" ]]; then
        if sudo nvidia-smi -lgc "0,${ATHENA_GPU_CLOCK_CAP}" 2>/dev/null; then
            GPU_CLOCKS_LOCKED=true
            echo "  graphics clock capped at ${ATHENA_GPU_CLOCK_CAP} MHz (lower this until 'SW Power Cap' stops appearing in gpu-throttle.log)"
        else
            echo "  WARNING: graphics clock cap ${ATHENA_GPU_CLOCK_CAP} MHz rejected"
        fi
    fi

    echo "[launch-athena] GPU configured."
}

restore_gpu_performance() {
    echo "  restoring GPU to normal..."

    if ! command -v nvidia-smi &>/dev/null; then return; fi

    if $GPU_CLOCKS_LOCKED; then
        sudo nvidia-smi -rmc 2>/dev/null && echo "    memory clocks: reset"
        sudo nvidia-smi -rgc 2>/dev/null   # harmless even though gfx is no longer locked
    fi

    # Restore the default board power limit if a test cap was applied this run.
    if [[ -n "${ATHENA_GPU_POWER_LIMIT:-}" ]]; then
        local def_pl
        def_pl=$(nvidia-smi --query-gpu=power.default_limit --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
        [[ -n "$def_pl" && "$def_pl" != "[N/A]" ]] && sudo nvidia-smi -pl "$def_pl" 2>/dev/null && echo "    power limit: restored to ${def_pl} W"
    fi

    sudo nvidia-smi -pm 0 2>/dev/null && echo "    persistence mode: disabled"
}

# ── CUDA MPS (diagnostic: merge the per-process GPU contexts) ──────────────────
# ATHENA_MPS=1 routes llama-server, talk-llama and orpheus-speak through a single
# CUDA MPS server context instead of letting them time-slice the GPU as three
# separate contexts — the no-rebuild A/B for the Xid 69 multi-context hypothesis.
# Unset (default) exports nothing and starts no daemon: byte-identical behavior.
#
# Compute mode is deliberately left at DEFAULT. Do NOT set EXCLUSIVE_PROCESS: if
# the display is driven by the dGPU it would lose its context and the desktop
# would freeze. In DEFAULT mode MPS still serves the Athena clients while the
# compositor keeps its own context.
start_mps() {
    [ "${ATHENA_MPS:-0}" = "1" ] || return 0

    if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
        echo "[launch-athena] ATHENA_MPS=1 but nvidia-cuda-mps-control not found — continuing WITHOUT MPS"
        return 0
    fi

    # The MPS *server* ships in /usr/sbin (nvidia-cuda-mps-server), but Debian omits
    # /usr/sbin from the user PATH — so the control daemon (in /usr/bin) can't exec
    # the server: every server dies "exited with status 1" (empty server.log) and
    # clients fail with "MPS client failed to connect to the MPS control daemon or
    # the MPS server". Ubuntu keeps /usr/sbin on PATH, which is why MPS worked there.
    # Put it on PATH so the daemon we spawn below inherits it and can find the server.
    case ":$PATH:" in *":/usr/sbin:"*) : ;; *) export PATH="/usr/sbin:$PATH" ;; esac

    # Honor pre-set pipe/log dirs; otherwise default the pipe to the conventional
    # /tmp path and put the server log under the per-run diag dir when one exists
    # (so it lands next to the other run artifacts). These exports persist in the
    # launcher's shell, so every GPU process started below inherits them.
    export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
    if [ -z "${CUDA_MPS_LOG_DIRECTORY:-}" ]; then
        if [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ]; then
            export CUDA_MPS_LOG_DIRECTORY="$RUN_DIR/mps-log"
        else
            export CUDA_MPS_LOG_DIRECTORY="/tmp/nvidia-mps-log"
        fi
    fi
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" 2>/dev/null || true

    # A pre-existing control endpoint does NOT prove a daemon is alive — it may be
    # a stale pipe left by a daemon that died without cleanup (a SIGKILLed
    # talk-llama, or a prior launch that "reused" it and never stopped it). The
    # bare `-e` test used to false-positive on such a pipe, silently skipping MPS
    # and letting the three GPU clients time-slice one context. Probe it instead:
    # a live daemon answers a status query; a stale/dead pipe errors or hangs
    # (hence `timeout`, which also covers a FIFO open-for-write block).
    if [ -e "$CUDA_MPS_PIPE_DIRECTORY/control" ]; then
        if echo "get_default_active_thread_percentage" \
             | timeout 5 nvidia-cuda-mps-control >/dev/null 2>&1; then
            MPS_STARTED=reused
            echo "[launch-athena] MPS: reusing LIVE daemon at $CUDA_MPS_PIPE_DIRECTORY (will NOT stop it on exit)"
            return 0
        fi
        echo "[launch-athena] MPS: stale control pipe at $CUDA_MPS_PIPE_DIRECTORY (no daemon responding) — clearing and starting fresh"
        # pkill -f (not -x): the executable names exceed 15 chars, so `comm`
        # truncates and `pkill -x nvidia-cuda-mps-server` never matches (silent no-op).
        pkill -TERM -f nvidia-cuda-mps-server  2>/dev/null || true
        pkill -TERM -f nvidia-cuda-mps-control 2>/dev/null || true
        rm -rf "$CUDA_MPS_PIPE_DIRECTORY"
        mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" 2>/dev/null || true
    fi

    # Create the log dir only now, when WE are about to own a daemon. Doing it
    # earlier (or on the reuse path) leaves an empty mps-log/ that misleadingly
    # reads as "MPS ran" — a reused daemon logs to its own original dir, not this.
    mkdir -p "$CUDA_MPS_LOG_DIRECTORY" 2>/dev/null || true
    if nvidia-cuda-mps-control -d 2>/dev/null; then
        MPS_STARTED=true
        # The control daemon starting is NOT proof MPS works — the per-user server
        # still has to spawn and init CUDA. Force one and confirm it survives (a
        # broken server exits status 1 and get_server_list stays empty). On failure
        # tear MPS down and run WITHOUT it, so a bad MPS degrades to the known-good
        # direct-GPU path instead of bricking every CUDA client (orpheus-speak died
        # here before this guard existed).
        echo "start_server -uid $(id -u)" | timeout 5 nvidia-cuda-mps-control >/dev/null 2>&1 || true
        if echo "get_server_list" | timeout 5 nvidia-cuda-mps-control 2>/dev/null | grep -q '[0-9]'; then
            echo "[launch-athena] MPS: daemon + server up — pipe=$CUDA_MPS_PIPE_DIRECTORY log=$CUDA_MPS_LOG_DIRECTORY"
            echo "[launch-athena] MPS: llama-server / orpheus-speak / talk-llama will share one GPU context"
        else
            echo "[launch-athena] WARNING: MPS server won't start (server exited; unsupported here?) — continuing WITHOUT MPS"
            echo quit | timeout 5 nvidia-cuda-mps-control >/dev/null 2>&1 || true
            unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
            MPS_STARTED=false
        fi
    else
        echo "[launch-athena] WARNING: failed to start MPS daemon — continuing WITHOUT MPS"
    fi
}

# Graceful MPS shutdown. Only quits a daemon THIS launcher started (MPS_STARTED
# == true); a reused daemon is left running. Called from cleanup() after the GPU
# client processes are already killed, so 'quit' has nothing to drain and returns
# fast. Runs under cleanup's `set +e`, so a wedged quit can't abort teardown.
stop_mps() {
    [ "${MPS_STARTED:-false}" = "true" ] || return 0
    command -v nvidia-cuda-mps-control >/dev/null 2>&1 || return 0

    echo "  stopping MPS daemon..."
    # 'quit -t N' tells the daemon to shut its servers down and, crucially, to
    # FORCE any server still up after N seconds. Bare 'quit' instead WAITS on
    # clients and hangs with exponential backoff — the exact 230402 wedge, where a
    # SIGKILLed client left "Server 3039 has 1 active worker thread ... will not
    # shutdown" and the plain `quit` timed out (line-382 exit-124). timeout may be
    # absent (fallback). All returns are swallowed (`|| true`): teardown is best-
    # effort and the ERR trap is already cleared in cleanup().
    if command -v timeout >/dev/null 2>&1; then
        echo "quit -t 5" | timeout 8 nvidia-cuda-mps-control >/dev/null 2>&1 || true
    else
        echo "quit -t 5" | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    fi
    # If the daemon or a server survived the bounded quit, force it down and clear
    # the pipe dir so the NEXT launch's stale-pipe probe (start_mps) starts clean.
    # NOTE: match with pkill -f (full cmdline): the executable names are >15 chars,
    # so `comm` truncates to "nvidia-cuda-mps" and the old `pkill -x
    # nvidia-cuda-mps-control` NEVER matched (procps warns ">15 chars = zero
    # matches") — it was a silent no-op that never actually killed a wedged daemon.
    if pgrep -f nvidia-cuda-mps-control >/dev/null 2>&1 || \
       pgrep -f nvidia-cuda-mps-server  >/dev/null 2>&1; then
        echo "    MPS wedged after quit — forcing teardown (stuck worker thread)"
        pkill -TERM -f nvidia-cuda-mps-control 2>/dev/null
        pkill -TERM -f nvidia-cuda-mps-server  2>/dev/null
        sleep 1
        pkill -KILL -f nvidia-cuda-mps-control 2>/dev/null
        pkill -KILL -f nvidia-cuda-mps-server  2>/dev/null
        [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] && rm -rf "$CUDA_MPS_PIPE_DIRECTORY" 2>/dev/null
    fi
    echo "    MPS stopped"
}

# Gracefully detach one MPS client's CUDA context via the documented
# terminate_client control command BEFORE we SIGKILL the process, so the server's
# per-client worker thread DRAINS instead of wedging. In 230402, SIGKILLing
# orpheus-speak (3481) while it still held a live MPS-client context left the
# server with a stuck worker thread ("... will not shutdown"). Only acts on a
# daemon THIS launcher owns; all control I/O is timeout-bounded so a wedged server
# can't hang teardown. (rs:mps-fault-isolation §4; quit -t in stop_mps is the
# guaranteed backstop if the context is already reset/poisoned and won't drain.)
mps_terminate_client() {            # $1 = client pid
    [ "${MPS_STARTED:-false}" = "true" ] || return 0
    command -v nvidia-cuda-mps-control >/dev/null 2>&1 || return 0
    local cpid="$1" srv
    [ -n "$cpid" ] || return 0
    for srv in $(echo get_server_list | timeout 5 nvidia-cuda-mps-control 2>/dev/null); do
        if echo "get_client_list $srv" | timeout 5 nvidia-cuda-mps-control 2>/dev/null \
             | tr -s ' ' '\n' | grep -qx "$cpid"; then
            echo "terminate_client $srv $cpid" | timeout 5 nvidia-cuda-mps-control >/dev/null 2>&1 || true
        fi
    done
}

# ── CPU / system performance setup ────────────────────────────────────────────

setup_system_performance() {
    echo "[launch-athena] configuring system for maximum performance..."

    # ── Power-profile manager: the silent HWP trap ────────────────────────────
    # power-profiles-daemon (default on GNOME/KDE/Ubuntu desktops) can pin a
    # DEGRADED hardware P-state at the MSR level even while scaling_governor and
    # EPP both read "performance" — a documented 20-30% TG loss that varies
    # between boots and is invisible to every sysfs check. Take whichever manager
    # is active out of the loop for the session (so the governor/EPP writes below
    # actually hold); the stop path restores it. No-op if neither is running.
    if command -v systemctl &>/dev/null && systemctl is-active --quiet power-profiles-daemon 2>/dev/null; then
        if sudo systemctl stop power-profiles-daemon 2>/dev/null; then
            PPD_WAS_ACTIVE=true
            echo "  power-profiles-daemon: stopped for session (was free to override HWP)"
        else
            echo "  power-profiles-daemon: active but stop failed (continuing)"
        fi
    elif command -v tuned-adm &>/dev/null && systemctl is-active --quiet tuned 2>/dev/null; then
        TUNED_SAVE="$(tuned-adm active 2>/dev/null | sed -n 's/^Current active profile: //p')"
        if [ -n "$TUNED_SAVE" ] && [ "$TUNED_SAVE" != "throughput-performance" ]; then
            if sudo tuned-adm profile throughput-performance 2>/dev/null; then
                echo "  tuned: throughput-performance (was $TUNED_SAVE)"
            else
                echo "  tuned: could not switch profile (was $TUNED_SAVE)"
                TUNED_SAVE=""
            fi
        else
            TUNED_SAVE=""   # already optimal or unreadable — nothing to revert
        fi
    fi

    # ── ACPI platform profile → performance ──────────────────────────────────
    # The EC-level budget knob (Lenovo: low-power/balanced/performance). It caps
    # SUSTAINED package power and the dGPU TGP share — governor/EPP writes below
    # cannot override the EC. With PPD stopped above nothing manages it, and
    # stuck at low-power the EC clamps the GPU under combined Qwen+Orpheus load
    # (measured 2026-07-04 15:24 run: Orpheus 47-66 tok/s vs the 82 tok/s
    # realtime floor → 830 stalls / 33 s injected silence). Must run AFTER the
    # PPD stop so PPD can't immediately re-assert its own profile. Restored on
    # exit. ATHENA_PLATFORM_PROFILE=<name> picks another profile; ="" skips.
    ATHENA_PLATFORM_PROFILE="${ATHENA_PLATFORM_PROFILE-performance}"
    local pp=/sys/firmware/acpi/platform_profile
    if [ -z "$ATHENA_PLATFORM_PROFILE" ]; then
        :   # explicitly skipped
    elif [ ! -f "$pp" ]; then
        echo "  platform profile: interface not present (skipped)"
    else
        local pp_cur pp_choices
        pp_cur="$(cat "$pp" 2>/dev/null)"
        pp_choices="$(cat "${pp}_choices" 2>/dev/null)"
        if [ "$pp_cur" = "$ATHENA_PLATFORM_PROFILE" ]; then
            echo "  platform profile: already $pp_cur"
        elif ! printf '%s' " $pp_choices " | grep -qF " $ATHENA_PLATFORM_PROFILE "; then
            echo "  platform profile: '$ATHENA_PLATFORM_PROFILE' not in choices ($pp_choices) — skipped"
        elif echo "$ATHENA_PLATFORM_PROFILE" | sudo tee "$pp" >/dev/null 2>&1; then
            PLATFORM_PROFILE_SAVE="$pp_cur"
            echo "  platform profile: $ATHENA_PLATFORM_PROFILE (was $pp_cur)"
        else
            echo "  WARNING: platform profile write failed (still $pp_cur — GPU may be EC-clamped under sustained load)"
        fi
    fi

    # CPU governor → performance
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo performance | sudo tee "$gov" >/dev/null 2>&1 || true
    done
    echo "  CPU governor: performance"

    # EPP (energy_performance_preference) → performance. On intel_pstate the
    # governor alone does NOT pin EPP; a non-performance EPP silently caps turbo.
    # This is the half of the fix the launcher was missing. Save each core's
    # original so the stop path reverts exactly (intel_pstate default is usually
    # balance_performance). Files are world-readable, so the read needs no sudo;
    # the write does.
    EPP_SAVE="$(mktemp 2>/dev/null || echo "/tmp/athena_epp.$$")"
    : > "$EPP_SAVE" 2>/dev/null || EPP_SAVE=""
    local epp epp_any=false
    for epp in /sys/devices/system/cpu/cpu*/cpufreq/energy_performance_preference; do
        [ -f "$epp" ] || continue
        [ -n "$EPP_SAVE" ] && printf '%s\t%s\n' "$epp" "$(cat "$epp" 2>/dev/null)" >> "$EPP_SAVE"
        echo performance | sudo tee "$epp" >/dev/null 2>&1 || true
        epp_any=true
    done
    if $epp_any; then
        echo "  EPP: performance (now: $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null))"
    else
        echo "  EPP: not exposed by this CPU/driver (skipped)"
    fi

    # Disable deep C-states (best effort — some sysfs files are read-only
    # depending on CPU/kernel; e.g. Arrow Lake-HX rejects writes to state0)
    for state in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
        echo 1 | sudo tee "$state" >/dev/null 2>&1 || true
    done
    echo "  C-states: disabled (best effort)"

    # Enable transparent huge pages
    echo always | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null 2>&1 || true
    echo always | sudo tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null 2>&1 || true
    echo "  transparent huge pages: always"

    # Disable NUMA balancing
    echo 0 | sudo tee /proc/sys/kernel/numa_balancing >/dev/null 2>&1 || true
    echo "  NUMA balancing: disabled"

    echo "[launch-athena] system configured."
}

restore_system_performance() {
    echo "  restoring system to normal..."

    # CPU governor → powersave (default on most Linux distros)
    for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
        echo powersave | sudo tee "$gov" >/dev/null 2>&1 || true
    done
    echo "    CPU governor: powersave"

    # Re-enable C-states (best effort — see note in setup_system_performance)
    for state in /sys/devices/system/cpu/cpu*/cpuidle/state*/disable; do
        echo 0 | sudo tee "$state" >/dev/null 2>&1 || true
    done
    echo "    C-states: re-enabled"

    # THP → madvise (default on most Linux distros)
    echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/enabled >/dev/null 2>&1 || true
    echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag >/dev/null 2>&1 || true
    echo "    transparent huge pages: madvise"

    # Re-enable NUMA balancing
    echo 1 | sudo tee /proc/sys/kernel/numa_balancing >/dev/null 2>&1 || true
    echo "    NUMA balancing: enabled"

    # Restore each core's original EPP (saved at launch). If the save file is
    # missing (mktemp failed, or EPP unsupported) this is skipped rather than
    # guessing a value.
    if [ -n "${EPP_SAVE:-}" ] && [ -f "$EPP_SAVE" ]; then
        local epp orig tab; tab=$(printf '\t')
        while IFS="$tab" read -r epp orig; do
            [ -n "$epp" ] && [ -n "$orig" ] && [ -f "$epp" ] || continue
            echo "$orig" | sudo tee "$epp" >/dev/null 2>&1 || true
        done < "$EPP_SAVE"
        rm -f "$EPP_SAVE" 2>/dev/null || true
        echo "    EPP: restored"
    fi

    # Restore the ACPI platform profile BEFORE handing control back to PPD (a
    # restarted PPD re-asserts its own persisted profile anyway, which then
    # wins — but when no manager is installed this write is the only restore).
    if [ -n "${PLATFORM_PROFILE_SAVE:-}" ] && [ -f /sys/firmware/acpi/platform_profile ]; then
        echo "$PLATFORM_PROFILE_SAVE" | sudo tee /sys/firmware/acpi/platform_profile >/dev/null 2>&1 \
            && echo "    platform profile: restored to $PLATFORM_PROFILE_SAVE"
        PLATFORM_PROFILE_SAVE=""
    fi

    # Hand the power-profile manager back (restart PPD / restore tuned profile).
    if $PPD_WAS_ACTIVE; then
        sudo systemctl start power-profiles-daemon 2>/dev/null && echo "    power-profiles-daemon: restarted"
        PPD_WAS_ACTIVE=false
    fi
    if [ -n "${TUNED_SAVE:-}" ]; then
        sudo tuned-adm profile "$TUNED_SAVE" 2>/dev/null && echo "    tuned: restored to $TUNED_SAVE"
        TUNED_SAVE=""
    fi
}

# ── Cleanup function ──────────────────────────────────────────────────────────

# ── Audio routing ─────────────────────────────────────────────────────────────
# Streams are pinned per-process via PULSE_SOURCE / PULSE_SINK exported to
# the children (SDL capture and aplay both honor them), NOT by flipping
# system defaults. Field finding (pactl dump, 2026-06-12): default-flipping
# silently failed — pipewire registers the AEC nodes asynchronously, so
# set-default raced them (error swallowed by 2>/dev/null), and WirePlumber's
# stream-restore re-routed aplay / talk-llama to their remembered raw
# devices anyway. Both streams ran on the bare headset the whole time while
# the AEC module sat loaded and IDLE. An explicit per-stream target beats
# stream-restore and cannot land on an unregistered node.
#
#   ATHENA_AUDIO=direct  (default) mic and playback on the bare headset.
#                        With a headset, physical isolation replaces echo
#                        cancellation — no DSP in the path. Validated by the
#                        2026-06-11 20:11 field session (6 clean barges).
#   ATHENA_AUDIO=aec     route both streams through module-echo-cancel
#                        (webrtc). For open speakers + mic, where her voice
#                        would pollute the VAD floor and barge detector.
# Bose Revolve SoundLink over Bluetooth. "bluez" matches both the A2DP output
# (bluez_output.<MAC>.*) and the HFP input (bluez_input.<MAC>) PipeWire nodes.
# NOTE on Bluetooth duplex: a single BT speaker cannot do hi-fi A2DP output AND mic input
# at once. talk-llama holds the mic open continuously, so setup_audio forces the Bose into
# HFP/mSBC duplex (16 kHz mono both ways) — "phone-call" quality, not A2DP.
HEADSET_PATTERN="${HEADSET_PATTERN:-bluez}"

# Barge-in: talk over Athena to interrupt her mid-reply. Needs the mic open during playback
# (HFP duplex gives that) AND echo cancellation, or she hears her own voice and interrupts
# herself. So when barge-in is on with a Bluetooth speaker, default the audio path to AEC.
ATHENA_BARGE="${ATHENA_BARGE:-1}"
if [ -z "${ATHENA_AUDIO:-}" ]; then
    if [ "$ATHENA_BARGE" != "0" ] && [[ "$HEADSET_PATTERN" == *bluez* ]]; then
        ATHENA_AUDIO=aec     # echo cancel so barge-in doesn't self-trigger on the Bose
    else
        ATHENA_AUDIO=direct
    fi
fi
#   ATHENA_AUDIO=direct  mic + playback on the bare device, no DSP. Fine for a headset
#                        (physical isolation); on the Bose, barge-in will self-interrupt.
#   ATHENA_AUDIO=aec     route both streams through module-echo-cancel (webrtc), bound to
#                        the Bose HFP nodes — removes Athena's voice from the mic. Needed
#                        for reliable barge-in on an open BT speaker + mic.

# TTS playback command for orpheus-speak's streaming path (--play-raw). orpheus-speak
# pipes raw mono S16_LE PCM to this command's stdin AND appends aplay-style "-r <rate> -"
# (sample rate + read-from-stdin), so the command MUST be aplay-compatible. We route ALSA
# to the Bose via the "pulse" ALSA plugin (-D pulse), which honors the PULSE_SINK that
# setup_audio exports — no device hardcoded. NB: a pacat/pw-cat command does NOT work here,
# because orpheus appends "-r" and pacat reads that as --record (this was the "no audio"
# bug). The bare "aplay" default also fails: raw ALSA cannot reach the PipeWire BT sink.
TTS_PLAY_CMD="${TTS_PLAY_CMD:-aplay -q -t raw -f S16_LE -c 1 --buffer-time=300000 -D pulse}"
AEC_MODULE_ID=""
BT_CARD_SAVE=""       # Bluetooth card whose profile setup_audio forced to HFP (restored on exit)
BT_PROFILE_SAVE=""    # its pre-launch active profile (e.g. a2dp-sink)
BT_SINK=""            # Bose HFP sink node   (set by force_bt_hfp)
BT_SRC=""             # Bose HFP source node (set by force_bt_hfp)

# force_bt_hfp: put the matched Bluetooth card into HFP so mic + playback coexist (a single
# BT speaker can't do A2DP output + mic input at once). Sets BT_SINK/BT_SRC and saves the
# prior profile for restore_audio. No-op for a non-bluez HEADSET_PATTERN (USB headsets).
force_bt_hfp() {
    [ "${ATHENA_BT_FORCE_HFP:-1}" = "0" ] && return
    local bt_card
    bt_card=$(pactl list short cards | awk -v p="$HEADSET_PATTERN" '$2 ~ p {print $2; exit}')
    [ -z "$bt_card" ] && return
    BT_PROFILE_SAVE=$(pactl list cards | awk -v c="Name: $bt_card" '
        index($0,c){f=1} f && /Active Profile:/{print $3; exit}')
    BT_CARD_SAVE="$bt_card"
    if pactl set-card-profile "$bt_card" headset-head-unit 2>/dev/null; then
        echo "  Bluetooth: $bt_card -> headset-head-unit (HFP/mSBC duplex; was ${BT_PROFILE_SAVE:-?})"
    elif pactl set-card-profile "$bt_card" headset-head-unit-cvsd 2>/dev/null; then
        echo "  Bluetooth: $bt_card -> headset-head-unit-cvsd (HFP/CVSD duplex; was ${BT_PROFILE_SAVE:-?})"
    else
        echo "  WARNING: could not set an HFP profile on $bt_card — mic + playback may not coexist"
        BT_CARD_SAVE=""; return
    fi
    sleep 1   # let PipeWire tear down the A2DP node and expose the HFP sink/source
    BT_SINK=$(pactl list short sinks   | awk -v p="$HEADSET_PATTERN" '$2 ~ p {print $2; exit}')
    BT_SRC=$(pactl  list short sources | awk -v p="$HEADSET_PATTERN" '$2 ~ p && $2 !~ /\.monitor$/ {print $2; exit}')
}

setup_audio() {
    if ! command -v pactl &>/dev/null; then
        echo "[launch-athena] WARNING: pactl not found (sudo apt install pulseaudio-utils) — using system default audio devices"
        return
    fi

    echo "[launch-athena] audio: $ATHENA_AUDIO mode"

    # Force the Bose into HFP first (mic + playback must share the HSP/HFP profile — A2DP
    # has no mic). Runs for BOTH aec and direct so BT_SINK/BT_SRC are the stable HFP nodes.
    force_bt_hfp

    if [ "$ATHENA_AUDIO" = "aec" ]; then
        # Echo cancellation: remove Athena's own voice from the mic so barge-in does not
        # self-interrupt on a single BT speaker+mic. Bind the canceller's master to the Bose
        # HFP nodes (BT_SINK/BT_SRC); without a BT match it falls back to system defaults.
        local master_args=""
        [ -n "$BT_SRC" ] && [ -n "$BT_SINK" ] && master_args="source_master=$BT_SRC sink_master=$BT_SINK"
        AEC_MODULE_ID=$(pactl load-module module-echo-cancel aec_method=webrtc \
            $master_args source_name=athena_aec_src sink_name=athena_aec_sink) || AEC_MODULE_ID=""
        if [ -n "$AEC_MODULE_ID" ]; then
            local i   # nodes register asynchronously — wait, or targets land on nothing
            for i in $(seq 1 40); do
                pactl list short sources | grep -q "athena_aec_src" && break
                sleep 0.05
            done
            if pactl list short sources | grep -q "athena_aec_src"; then
                export PULSE_SOURCE=athena_aec_src
                export PULSE_SINK=athena_aec_sink
                export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-pulseaudio}"
                echo "  mic      <- athena_aec_src (webrtc echo cancel${BT_SRC:+ on $BT_SRC})"
                echo "  playback -> athena_aec_sink${BT_SINK:+ -> $BT_SINK}"
                return
            fi
            echo "  WARNING: AEC nodes never registered — falling back to direct"
            pactl unload-module "$AEC_MODULE_ID" 2>/dev/null || true
            AEC_MODULE_ID=""
        else
            echo "  WARNING: module-echo-cancel failed to load — falling back to direct"
        fi
    fi

    # Direct mode: route talk-llama's mic + TTS straight at the Bose HFP nodes. No echo
    # cancellation, so barge-in (if enabled) can self-trigger on Athena's own voice — use
    # ATHENA_AUDIO=aec for reliable barge-in on the Bose.
    local sink="$BT_SINK" src="$BT_SRC"
    [ -z "$sink" ] && sink=$(pactl list short sinks   | awk -v p="$HEADSET_PATTERN" '$2 ~ p {print $2; exit}')
    [ -z "$src" ]  && src=$(pactl  list short sources | awk -v p="$HEADSET_PATTERN" '$2 ~ p && $2 !~ /\.monitor$/ {print $2; exit}')
    if [ -n "$sink" ] && [ -n "$src" ]; then
        export PULSE_SINK="$sink"
        export PULSE_SOURCE="$src"
        # Force SDL (talk-llama mic capture) onto the PulseAudio backend so it honors
        # PULSE_SOURCE. The TTS player (aplay -D pulse) honors PULSE_SINK the same way.
        export SDL_AUDIODRIVER="${SDL_AUDIODRIVER:-pulseaudio}"
        echo "  playback -> $sink"
        echo "  mic      <- $src"
        echo "  SDL_AUDIODRIVER=$SDL_AUDIODRIVER (mic honors PULSE_SOURCE)"
    else
        echo "  WARNING: no devices matching '$HEADSET_PATTERN' — using system defaults. Available:"
        pactl list short sinks   | awk '{print "    sink   " $2}'
        pactl list short sources | awk '$2 !~ /\.monitor$/ {print "    source " $2}'
    fi
}

restore_audio() {
    if [ -n "$AEC_MODULE_ID" ]; then
        pactl unload-module "$AEC_MODULE_ID" 2>/dev/null && echo "  echo cancel: unloaded"
        AEC_MODULE_ID=""
    fi
    # Restore the Bluetooth card to its pre-launch profile (typically a2dp-sink) so the
    # Bose returns to hi-fi playback for other apps after ATHENA exits.
    if [ -n "$BT_CARD_SAVE" ] && [ -n "$BT_PROFILE_SAVE" ]; then
        pactl set-card-profile "$BT_CARD_SAVE" "$BT_PROFILE_SAVE" 2>/dev/null \
            && echo "  Bluetooth: $BT_CARD_SAVE -> $BT_PROFILE_SAVE (restored)"
        BT_CARD_SAVE=""
    fi
}

cleanup() {
    set +e     # Don't exit on error during cleanup
    trap - ERR # `set +e` stops errexit but NOT the ERR trap (it is inherited via
               # `set -E`, line 18), so every EXPECTED non-zero return in teardown
               # (bounded MPS `quit` timing out = 124, pkill matching nothing = 1)
               # printed a spurious "[launch-athena] ERROR at line N" — the
               # misleading line-382/387 noise in the 230402 shutdown. Drop it here;
               # the global trap (line 19) still catches a REAL crash before cleanup.
    echo ""
    echo "[launch-athena] shutting down..."

    # Stop the diagnostics sampler last-started/first-stopped, but give it a beat
    # so it records the shutdown (and any abort the kills below surface). The
    # monitor traps TERM and tears down its own nvidia-smi/dmesg children.
    if [ -n "$PID_GPUMON" ] && kill -0 "$PID_GPUMON" 2>/dev/null; then
        kill "$PID_GPUMON" 2>/dev/null
        echo "  stopped GPU/Xid sampler ($PID_GPUMON)"
    fi
    [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR" ] && echo "  diagnostics saved to $RUN_DIR"

    # Stop the llama-server watchdog + keepalive FIRST so they cannot respawn/poke the
    # backend mid-teardown. The sentinel blocks an in-flight respawn; TERM trips their
    # traps (exit 0), which deliberately leave llama-server for the ordered _stop_pid
    # below — preserving the load-bearing kill ORDER (orpheus → llama → talk-llama).
    local _wp _p _i
    touch "$LLAMA_WD_STOP" 2>/dev/null
    for _wp in PID_WATCHDOG PID_KEEPALIVE PID_ORPHWD; do
        _p="${!_wp}"
        if [ -n "$_p" ] && kill -0 "$_p" 2>/dev/null; then
            kill "$_p" 2>/dev/null
            for _i in $(seq 1 20); do kill -0 "$_p" 2>/dev/null || break; sleep 0.1; done
            kill -0 "$_p" 2>/dev/null && kill -9 "$_p" 2>/dev/null
            echo "  stopped $_wp ($_p)"
        fi
    done

    # ── Ordered teardown (avoids the MPS page-accounting kernel crash) ─────────
    # We hit a kernel "BUG: Bad page state ... page still charged to cgroup" in
    # exit_mmap during shutdown: a GPU client's address space was torn down while
    # GPU work was still in flight under MPS, and the driver left a mapped/pinned
    # page mis-accounted, corrupting kernel memory and locking the machine.
    # talk-llama runs a final ~20-30s Qwen "memory consolidation" generation at
    # exit, so it MUST stop LAST and be given enough grace to finish on its own —
    # a SIGKILL mid-generation, with its ~161 GiB pinned host allocation, is
    # exactly the forced teardown that corrupts page state. The auxiliary GPU
    # clients (orpheus-speak, llama-server) are idle once the goodbye is spoken,
    # so stop them FIRST and let them fully release their MPS contexts before
    # talk-llama's consolidation teardown, so nothing tears down concurrently.
    _stop_pid() {                 # $1 = pid var name   $2 = grace in 0.1s units
        local pid="${!1}" grace="$2" i
        [ -z "$pid" ] && return 0
        kill -0 "$pid" 2>/dev/null || return 0
        kill "$pid" 2>/dev/null
        for ((i=0; i<grace; i++)); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
        if kill -0 "$pid" 2>/dev/null; then
            echo "  WARNING: $1 ($pid) still alive after $((grace/10))s — detaching MPS context, then SIGKILL"
            echo "           (a forced GPU-client teardown under MPS can corrupt kernel page state;"
            echo "            terminate_client first drains the server's per-client worker thread)"
            mps_terminate_client "$pid"   # documented graceful detach before the kill (no-op without MPS)
            kill -9 "$pid" 2>/dev/null
        fi
        echo "  stopped $1 ($pid)"
    }

    # The orpheus watchdog may have restarted the daemon → live PID differs from launch.
    [ -r "$ORPHEUS_PIDFILE" ] && PID_ORPHEUS=$(cat "$ORPHEUS_PIDFILE" 2>/dev/null)
    _stop_pid PID_ORPHEUS 50      # 5s  — TTS playback finished after the goodbye
    # The watchdog may have restarted llama-server → the live PID differs from launch.
    [ -r "$LLAMA_PIDFILE" ] && PID_LLAMA=$(cat "$LLAMA_PIDFILE" 2>/dev/null)
    _stop_pid PID_LLAMA   50      # 5s  — Orpheus server not used by consolidation
    # talk-llama last, with a long grace so its consolidation generation completes
    # and the process exits cleanly instead of being force-killed mid-decode.
    _stop_pid PID_TALK "${ATHENA_SHUTDOWN_GRACE_DS:-600}"   # 60s default

    # Remove temp files
    rm -f "$SPEAK_FILE" "$TRIGGER_FILE" "$DONE_FILE" "$STOP_FILE" "$WAV_FILE" \
          "$LLAMA_PIDFILE" "$ORPHEUS_PIDFILE" "$LLAMA_WEDGE" "$LLAMA_WD_STOP"
    echo "  cleaned up temp files"

    # Restore the kernel core_pattern if the core-dump setup changed it.
    if [ -n "$CORE_PATTERN_SAVE" ]; then
        printf '%s\n' "$CORE_PATTERN_SAVE" | sudo tee /proc/sys/kernel/core_pattern >/dev/null 2>&1 \
            && echo "  core_pattern: restored"
    fi

    # Restore audio routing, system and GPU to pre-launch state. MPS is stopped
    # after the GPU client processes above are down and before the GPU clocks /
    # power limit are reverted.
    restore_audio
    restore_system_performance
    stop_mps
    restore_gpu_performance

    echo "[launch-athena] done."
}

trap cleanup EXIT INT TERM

# ── GPU performance tuning ────────────────────────────────────────────────────

setup_gpu_performance

# ── CUDA MPS (diagnostic, ATHENA_MPS=1) ───────────────────────────────────────
# Must run before any GPU process so llama-server, orpheus-speak and talk-llama
# all inherit the exported CUDA_MPS_* env and share one MPS context. No-op unless
# ATHENA_MPS=1.
start_mps

# ── CPU / system performance tuning ───────────────────────────────────────────

setup_system_performance

# ── Audio routing ──────────────────────────────────────────────────────────────

setup_audio

# ── Preflight checks ──────────────────────────────────────────────────────────

check_file() {
    if [ ! -f "$1" ]; then
        echo "[launch-athena] ERROR: $2 not found: $1"
        exit 1
    fi
}

check_file "$LLAMA_SERVER"  "llama-server binary"
check_file "$ORPHEUS_SPEAK" "orpheus-speak binary"
check_file "$TALK_LLAMA"    "whisper-talk-llama binary"
check_file "$ORPHEUS_MODEL" "Orpheus GGUF model"
check_file "$SNAC_MODEL"    "SNAC ONNX decoder"
# Qwen model check. QWEN_SHARDS=1 → single-file GGUF (current 35B). For a split
# GGUF (QWEN_SHARDS>1) llama.cpp opens shard 1 and pulls in the rest, so we verify
# every shard up front — a missing later shard would otherwise only fail minutes in.
for i in $(seq 1 "$QWEN_SHARDS"); do
    if [ "$QWEN_SHARDS" -eq 1 ]; then
        check_file "$QWEN_MODEL" "Qwen GGUF model"
    else
        check_file "${QWEN_MODEL/00001-of/0000${i}-of}" "Qwen GGUF shard ${i}/${QWEN_SHARDS}"
    fi
done
check_file "$WHISPER_MODEL" "Whisper model"
check_file "$VAD_MODEL"     "Silero VAD model"
check_file "$SPEAK_DAEMON"  "speak-daemon.sh"

# Clean stale temp files from previous runs
rm -f "$SPEAK_FILE" "$TRIGGER_FILE" "$DONE_FILE" "$STOP_FILE"

# Remove stale ONNX optimized graph cache — a GPU-optimized cache will cause
# VRAM allocation even when running SNAC on CPU via --snac-cpu
rm -f "$OPT_CACHE"

# ── Terminal 1: llama-server (Orpheus TTS) ────────────────────────────────────

# ── Diagnostics: GPU / Xid telemetry sampler ──────────────────────────────────
# Start before the heavy model loads so VRAM growth, clock locks, and any load-time
# fault are captured. Best-effort: a missing script just skips it.
if [ "$ATHENA_DIAG" != "0" ] && [ -n "$RUN_DIR" ]; then
    if [ -x "$GPU_MONITOR" ]; then
        # 1000 ms cadence (was 500): halves the highest-rate nvidia-smi spawner — the
        # process that drew the "Bad page state" poisoned page on 20260701-123219 — at
        # near-zero diagnostic loss (clocks/power move slowly; fault capture is
        # event-driven via athena-kmsg-logger). Override: "$GPU_MONITOR" <dir> <ms>.
        "$GPU_MONITOR" "$RUN_DIR" "${ATHENA_GPU_SAMPLE_MS:-1000}" &
        PID_GPUMON=$!
        echo "[launch-athena] diagnostics -> $RUN_DIR (gpu/xid sampler pid $PID_GPUMON)"
    else
        echo "[launch-athena] NOTE: $GPU_MONITOR not executable — GPU sampler skipped (server log still captured)"
    fi
fi

echo "[launch-athena] starting llama-server (Orpheus TTS)..."

# Bench-verified across two runs (orpheus-bench.sh, 2026-06):
#   GGML_CUDA_GRAPH_OPT=1 -> no measurable benefit (run 2: 158.8 vs 159.7 off).
#     NOW UNSET (see CHANGES.MD §10): it is an *experimental* multi-stream/
#     buffer-reuse CUDA path (llama.cpp PR #16991, off-by-default, ~1-9% TG only
#     at batch-1) whose author flags a buffer-reuse race — exactly the class that
#     surfaces as intermittent "unspecified launch failure". Base CUDA graphs
#     (-DGGML_CUDA_GRAPHS=ON) + kernel fusion stay ON and self-disable at batch>1,
#     so retiring GRAPH_OPT retires extra concurrent streams at ~0 perf cost.
#   KV f16 vs q8_0 -> within power-hunting noise (each "won" one run by ~5%).
#     q8_0 now kept: bench-neutral, and frees ~0.75 GiB for the 397B's budget.
# Real bottleneck: ~94 W power wall collapses SM clock to ~870 MHz under
# sustained decode (see orpheus-powertune.sh). Single-stream ceiling ~160 tok/s.
# -np 2 (was 4): ATHENA is single-user, so np>1 gives ZERO per-stream latency
# benefit (Orpheus needs 82 tok/s/stream for real-time; single-stream is ~160).
# np only fans the client's sentence-chunk pipeline across slots. At np=2 each
# stream holds ~122 tok/s (1.49x RT) vs ~87 (1.07x) at np=4 — the real-time
# margin GROWS while peak concurrent-decode stress (and shared MPS/GSP stream
# count) HALVES. Per-slot ctx 13824/2=6912 still fits the largest chunk (~1,800
# tok). The 230402 fatal fault hit with 3 slots decoding concurrently.
# stdbuf -oL -eL forces line buffering so the final lines before an abort (the
# CUDA error / assertion / OOM reason) are flushed rather than lost in glibc's
# block buffer. Process substitution keeps $! = llama-server's PID for the health
# checks and cleanup below. With ATHENA_DIAG=0, ORPHEUS_SERVER_LOG is /dev/null.
# Core dumps for the SMALL GPU clients only (llama-server, orpheus-speak) so a libcuda
# GPF (20260701-123219) leaves a gdb-inspectable backtrace. NEVER talk-llama: its
# ~161 GiB mlock'd RSS would write a catastrophic core. (CHANGES.MD §12)
if [ "$ATHENA_COREDUMP" != "0" ] && [ -n "$RUN_DIR" ]; then
    CORE_PATTERN_SAVE="$(cat /proc/sys/kernel/core_pattern 2>/dev/null)"
    if printf '%s\n' "$RUN_DIR/core.%e.%p" | sudo tee /proc/sys/kernel/core_pattern >/dev/null 2>&1; then
        CORE_ULIMIT=unlimited
        echo "[launch-athena] core dumps ON for llama-server/orpheus-speak -> $RUN_DIR/core.* (ATHENA_COREDUMP=0 to disable)"
    else
        CORE_PATTERN_SAVE=""   # nothing changed → nothing to restore
        echo "[launch-athena] NOTE: could not set core_pattern (need sudo) — core dumps off"
    fi
fi

# start_llama_server: launch llama-server EXACTLY as configured and record its live PID
# in PID_LLAMA and $LLAMA_PIDFILE (the values cleanup() and the watchdog trust). The
# ( ulimit -c …; exec … ) wrapper sets the core limit for THIS client only; exec keeps
# $! == llama-server's PID. stdbuf -oL -eL forces line buffering so the last lines
# before an abort survive; the process substitution preserves $! (see 230402 notes).
start_llama_server() {
    ( ulimit -c "$CORE_ULIMIT"
      exec stdbuf -oL -eL "$LLAMA_SERVER" \
          -m "$ORPHEUS_MODEL" \
          -c 13824 -np 2 -ngl 99 \
          --host 127.0.0.1 --port 8080 \
          --cache-type-k f16 --cache-type-v f16 --cache-ram 0 \
          -fa on -t 0
    ) > >(_ts >> "$ORPHEUS_SERVER_LOG") 2>&1 &
    PID_LLAMA=$!
    echo "$PID_LLAMA" > "$LLAMA_PIDFILE" 2>/dev/null
}

# llama_watchdog: background health-watchdog + auto-restart. Scoped to the userspace-
# death case (SIGSEGV/GPF/OOM in llama-server, GPU still healthy, MPS not poisoned) —
# exactly the 20260701-123219 failure. It CANNOT save a device Xid (that poisons all
# MPS clients and needs a full-stack relaunch); it caps restarts and gives up if so.
llama_watchdog() {
    set +e; trap - ERR
    trap 'exit 0' TERM INT
    local WD_LOG="${RUN_DIR:-/dev/shm}/watchdog.log"
    _wlog() { echo "[$(date '+%F %T.%3N')] watchdog[$BASHPID]: $*" >> "$WD_LOG" 2>/dev/null; }
    _wlog "armed — guarding llama-server :8080 (interval ${WD_INTERVAL}s, cap ${WD_MAX_RESTARTS}/${WD_WINDOW}s)"
    local -a stamps=()
    local fails=0 i t pid down cutoff ok backoff=2
    while :; do
        [ -f "$LLAMA_WD_STOP" ] && { _wlog "stop sentinel — exiting"; exit 0; }
        sleep "$WD_INTERVAL"
        [ -f "$LLAMA_WD_STOP" ] && { _wlog "stop sentinel — exiting"; exit 0; }
        pid=$(cat "$LLAMA_PIDFILE" 2>/dev/null); down=0
        if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then
            down=1                                           # GPF/exit — definitive fast path
        elif [ -f "$LLAMA_WEDGE" ]; then
            rm -f "$LLAMA_WEDGE" 2>/dev/null
            down=1                                           # keepalive-flagged decode wedge (§20):
            _wlog "decode WEDGE flagged by keepalive (health green, completions dead) — forcing restart"
        elif ! curl -sf --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
            fails=$((fails+1)); [ "$fails" -ge "$WD_FAIL_THRESHOLD" ] && down=1   # wedged/half-alive
        else
            fails=0
        fi
        [ "$down" -eq 0 ] && continue
        fails=0
        cutoff=$(( SECONDS - WD_WINDOW ))
        local -a kept=()
        for t in "${stamps[@]}"; do [ "$t" -ge "$cutoff" ] && kept+=("$t"); done
        stamps=("${kept[@]}")
        if [ "${#stamps[@]}" -ge "$WD_MAX_RESTARTS" ]; then
            _wlog "GIVING UP: ${#stamps[@]} restarts within ${WD_WINDOW}s — backend hard-down (likely a device-level fault; TTS offline, brain unaffected)"
            exit 0
        fi
        stamps+=("$SECONDS")
        _wlog "llama-server DOWN (pid=${pid:-none}); restart #${#stamps[@]}/${WD_MAX_RESTARTS}"
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then    # wedged: force it gone before rebind
            mps_terminate_client "$pid" 2>/dev/null
            kill -9 "$pid" 2>/dev/null
            for i in $(seq 1 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
        fi
        for i in $(seq 1 25); do curl -s --max-time 1 http://127.0.0.1:8080/health >/dev/null 2>&1 || break; sleep 0.2; done
        [ -f "$LLAMA_WD_STOP" ] && { _wlog "stop before respawn — exiting"; exit 0; }
        start_llama_server
        _wlog "respawned pid $PID_LLAMA — polling /health"
        ok=0
        for i in $(seq 1 120); do
            [ -f "$LLAMA_WD_STOP" ] && exit 0
            if curl -sf --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then ok=1; break; fi
            kill -0 "$PID_LLAMA" 2>/dev/null || break        # died during load
            sleep 0.5
        done
        if [ "$ok" -eq 1 ]; then
            _wlog "HEALTHY again (pid $PID_LLAMA) — TTS restored; orpheus-speak reconnects on its next request"
            backoff=2
        else
            _wlog "respawn not healthy (pid $PID_LLAMA); backing off ${backoff}s"
            sleep "$backoff"; backoff=$(( backoff*2 )); [ "$backoff" -gt 30 ] && backoff=30
        fi
    done
}

# llama_keepalive: idle 1-token poke so the CUDA context/graphs never go fully cold
# (the 123219 GPF hit 13 ms after a 67 s-idle re-entry). ~8 ms GPU / 30 s; off the
# user-latency path. Harmless while llama-server is momentarily down (curl just fails).
llama_keepalive() {
    set +e; trap - ERR; trap 'exit 0' TERM INT
    local util kfail=0 csv="${RUN_DIR:+$RUN_DIR/gpu.csv}"
    while :; do
        [ -f "$LLAMA_WD_STOP" ] && exit 0
        sleep "${ATHENA_KEEPALIVE_INTERVAL:-30}"
        [ -f "$LLAMA_WD_STOP" ] && exit 0
        # IDLE-GATE (CHANGES.MD §15): poke ONLY when the GPU is idle. When busy the
        # CUDA context is already warm, so the poke buys nothing — and it injects a
        # SECOND MPS client's decode on top of the brain's. That concurrency-at-the-
        # power-wall pattern is present at both Xid-69 faults (074343: 3 TTS slots
        # decoding; 174424: the poke came due within ~100 ms of the Xid while the
        # brain decoded at 88% util on the 94 W cap). gpu.csv col 7 (utilization.gpu)
        # is refreshed ~1/s by the persistent sampler; if it is unavailable
        # (ATHENA_DIAG=0), fail open and poke — old behavior.
        if [ -n "$csv" ] && [ -r "$csv" ]; then
            util=$(tail -1 "$csv" 2>/dev/null | awk -F',' 'NF>=7{gsub(/ /,"",$7); printf "%d", $7+0}')
            if [ -n "$util" ] && [ "$util" -ge "${ATHENA_KEEPALIVE_MAX_UTIL:-15}" ]; then
                continue    # GPU busy → context warm → skip this poke
            fi
        fi
        # Deep probe (§20): /health is a hardcoded "ok" served by an independent HTTP
        # thread pool — a llama-server whose decode loop is wedged inside a CUDA call
        # answers /health forever while every /completion hangs, and the watchdog's
        # health poll is blind to it. This poke IS a real decode, so report it:
        # 3 consecutive failures WITH /health still green = the wedge signature →
        # flag the watchdog. A down/restarting server (health also failing) is the
        # watchdog fast path's job, not a wedge.
        if curl -sf -m 10 http://127.0.0.1:8080/completion \
             -d '{"prompt":" ","n_predict":1,"temperature":0,"cache_prompt":false}' >/dev/null 2>&1; then
            kfail=0; rm -f "$LLAMA_WEDGE" 2>/dev/null
        elif curl -sf --max-time 2 http://127.0.0.1:8080/health >/dev/null 2>&1; then
            kfail=$((kfail+1))
            if [ "$kfail" -ge 3 ]; then touch "$LLAMA_WEDGE" 2>/dev/null; kfail=0; fi
        else
            kfail=0
        fi
    done
}

# orpheus_watchdog (§20): supervise the TTS daemon the same way llama_watchdog
# supervises the backend — run 20260702-134528 proved a dead orpheus-speak freezes
# the brain (silent .done wait) with no recovery path. Detection is kill -0 only
# (the daemon has no HTTP surface); recovery is a byte-identical respawn. The
# respawned binary's own startup-recovery consumes a stale COMPLETED trigger and
# writes the .done that releases a blocked brain — without that binary fix a
# respawn alone would NOT unblock it (a completed trigger never fires inotify
# again). Rolling cap and stop sentinel shared with llama_watchdog.
orpheus_watchdog() {
    set +e; trap - ERR
    trap 'exit 0' TERM INT
    local WD_LOG="${RUN_DIR:-/dev/shm}/watchdog.log"
    _olog() { echo "[$(date '+%F %T.%3N')] orpheus-wd[$BASHPID]: $*" >> "$WD_LOG" 2>/dev/null; }
    _olog "armed — guarding orpheus-speak (interval ${WD_INTERVAL}s, cap ${WD_MAX_RESTARTS}/${WD_WINDOW}s)"
    local -a stamps=()
    local pid t cutoff
    while :; do
        [ -f "$LLAMA_WD_STOP" ] && { _olog "stop sentinel — exiting"; exit 0; }
        sleep "$WD_INTERVAL"
        [ -f "$LLAMA_WD_STOP" ] && { _olog "stop sentinel — exiting"; exit 0; }
        pid=$(cat "$ORPHEUS_PIDFILE" 2>/dev/null)
        [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && continue
        cutoff=$(( SECONDS - WD_WINDOW ))
        local -a kept=()
        for t in "${stamps[@]}"; do [ "$t" -ge "$cutoff" ] && kept+=("$t"); done
        stamps=("${kept[@]}")
        if [ "${#stamps[@]}" -ge "$WD_MAX_RESTARTS" ]; then
            _olog "GIVING UP: ${#stamps[@]} restarts within ${WD_WINDOW}s — TTS daemon hard-down (brain unaffected; TTS offline)"
            exit 0
        fi
        stamps+=("$SECONDS")
        _olog "orpheus-speak DOWN (pid=${pid:-none}); restart #${#stamps[@]}/${WD_MAX_RESTARTS}"
        [ -f "$LLAMA_WD_STOP" ] && { _olog "stop before respawn — exiting"; exit 0; }
        start_orpheus_speak
        _olog "respawned pid $PID_ORPHEUS — binary startup-recovery releases any waiting brain"
    done
}

start_llama_server

# Wait for llama-server to be ready
echo "[launch-athena] waiting for llama-server to start..."
for i in $(seq 1 120); do
    if curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
        echo "[launch-athena] llama-server ready."
        break
    fi
    if ! kill -0 "$PID_LLAMA" 2>/dev/null; then
        echo "[launch-athena] ERROR: llama-server exited unexpectedly"
        exit 1
    fi
    sleep 0.5
done

if ! curl -s http://127.0.0.1:8080/health >/dev/null 2>&1; then
    echo "[launch-athena] ERROR: llama-server did not start within 60s"
    exit 1
fi

# Arm the watchdog + keepalive now that llama-server is healthy (this also covers the
# multi-minute talk-llama model load that follows). Backgrounded ⇒ never trips set -e
# or the ERR trap; the restart path uses its OWN bounded re-poll, never exit 1.
rm -f "$LLAMA_WD_STOP" 2>/dev/null
if [ "$ATHENA_WATCHDOG" != "0" ]; then
    llama_watchdog & PID_WATCHDOG=$!
    echo "[launch-athena] llama-server watchdog armed (pid $PID_WATCHDOG; ATHENA_WATCHDOG=0 to disable)"
fi
if [ "$ATHENA_KEEPALIVE" != "0" ]; then
    llama_keepalive & PID_KEEPALIVE=$!
    echo "[launch-athena] llama-server keepalive armed (pid $PID_KEEPALIVE; ATHENA_KEEPALIVE=0 to disable)"
fi

# ── Terminal 2: orpheus-speak daemon (SNAC decoder) ───────────────────────────

echo "[launch-athena] starting orpheus-speak daemon..."

# --diag (gated on ATHENA_DIAG) enables the real-time device-fill heartbeat and
# stall-run markers in athena.log, so a developing underrun is visible live.
DIAG_FLAG=()
[ "$ATHENA_DIAG" != "0" ] && DIAG_FLAG=(--diag)

# --capture-dir (gated on ATHENA_TTS_CAPTURE) dumps per-session tokens/codes/wav/meta
# for garble diagnosis. Land it under the per-run diag dir when one exists, else a
# standalone dir, so captures sit next to gpu.csv / orpheus-server.log for the run.
CAPTURE_FLAG=()
if [ "$ATHENA_TTS_CAPTURE" != "0" ]; then
    TTS_CAPTURE_DIR="${RUN_DIR:-$DIAG_DIR}/tts-capture"
    if mkdir -p "$TTS_CAPTURE_DIR" 2>/dev/null; then
        CAPTURE_FLAG=(--capture-dir "$TTS_CAPTURE_DIR")
        echo "[launch-athena] TTS capture ON -> $TTS_CAPTURE_DIR (ATHENA_TTS_CAPTURE=0 to disable)"
    else
        echo "[launch-athena] WARNING: could not create $TTS_CAPTURE_DIR — TTS capture off"
    fi
fi

# start_orpheus_speak: launch the TTS daemon EXACTLY as configured and record its
# live PID in PID_ORPHEUS and $ORPHEUS_PIDFILE (the values cleanup() and
# orpheus_watchdog trust). exec keeps $! == the daemon's PID (§12 pattern).
start_orpheus_speak() {
    ( ulimit -c "$CORE_ULIMIT"
      exec "$ORPHEUS_SPEAK" \
          --watch "$TRIGGER_FILE" \
          --snac "$SNAC_MODEL" \
          --snac-cpu \
          --play-raw "$TTS_PLAY_CMD" \
          --stream-tts \
          "${DIAG_FLAG[@]}" \
          "${CAPTURE_FLAG[@]}" \
          -v
    ) &
    PID_ORPHEUS=$!
    echo "$PID_ORPHEUS" > "$ORPHEUS_PIDFILE" 2>/dev/null
}

start_orpheus_speak

# Give it a moment to load the ONNX model
sleep 2

if ! kill -0 "$PID_ORPHEUS" 2>/dev/null; then
    echo "[launch-athena] ERROR: orpheus-speak exited unexpectedly"
    exit 1
fi

# Arm the orpheus-speak watchdog (§20): a TTS-daemon death no longer strands the
# brain — the respawned binary's startup-recovery releases any pending .done wait.
if [ "$ATHENA_WATCHDOG" != "0" ]; then
    orpheus_watchdog & PID_ORPHWD=$!
    echo "[launch-athena] orpheus-speak watchdog armed (pid $PID_ORPHWD; ATHENA_WATCHDOG=0 to disable)"
fi

# ── Terminal 3: talk-llama (voice assistant) ──────────────────────────────────

echo "[launch-athena] starting talk-llama..."

# Assemble --memory args only when MEMORY_DIR is set, so disabling is a one-line
# comment. An empty array splat expands to nothing (safe under set -u in bash 4.4+).
MEMORY_ARGS=()
if [ -n "${MEMORY_DIR:-}" ]; then
    mkdir -p "$MEMORY_DIR"
    MEMORY_ARGS=(--memory "$MEMORY_DIR" --memory-words 2048 --time-refresh-min 15 --personality-reflect-every 1) #temp edit - time-refresh should normally be 15; reflect-every 1 lowers personality-revision threshold to 10 for the demo (revert to drop the flag)
    echo "[launch-athena] memory enabled: $MEMORY_DIR"
fi
echo "[launch-athena] ═══════════════════════════════════════════════════"
echo "[launch-athena] Athena is ready. Speak into your microphone."
echo "[launch-athena] Press Ctrl+C to quit."
echo "[launch-athena] ═══════════════════════════════════════════════════"
echo ""

export SDL_AUDIODRIVER=pulse

# boost the Logi headset mic so barge-ins clear the 0.0020 trigger
# pactl set-source-volume "$(pactl list short sources | grep -i logi | grep -iv monitor | head -1 | cut -f2)" 250%

# ── Qwen3.5-397B-A17B UD-Q3_K_XL settings (researched + MEASURED 2026-06) ──
#
# Sampling = OFFICIAL preset for Instruct (non-thinking) mode, "Reasoning
# tasks" column — Unsloth Qwen3.5 docs (unsloth.ai/docs/models/qwen3.5):
#   temp=1.0, top_p=0.95, top_k=20, min_p=0.0,
#   presence_penalty=1.5, repeat_penalty=1.0 (disabled)
# Instruct mode applies because talk-llama feeds a raw transcript (no chat
# template -> no <think> blocks; a think block would be spoken aloud anyway).
# Changes vs the 122B line: top-p 0.8->0.95, presence 1.2->1.5.
# Unsloth caveat on record: presence_penalty >0 "may result in slight
# decrease in performance" — but 1.5 is the official preset; A/B if speech
# style degrades.
#
# KV CACHE: bf16. q8_0 was tried first (would allow full 262144 ctx) and
# produced degraded output in the field — including a literal <think> token
# emitted into the transcript by turn 3 — matching Unsloth's documented
# quantized-KV symptom for this family ("if gibberish, try bf16 KV").
#
# CONTEXT 131072: sized from the MEASURED ledger of the 2026-06-10 load
# (19,186 MiB free at Qwen load with Orpheus q8_0 + Whisper + SNAC resident):
#   dense weights (CUDA0 model buffer)   9,129.7 MiB   measured
#   KV bf16 @131072 (15 full-attn lyrs)  3,840.0 MiB   = 15 x 131072 x 1024 x 2 B
#   GDN recurrent state                    186.3 MiB   measured
#   compute buffer (worst observed)      2,498.7 MiB   measured (reserve 1832 grew +666)
#   total ~15,656 MiB -> ~3.5 GiB headroom
# bf16 @262144 totals 19,496 MiB vs 19,186 free -> over by ~310 MiB: does NOT
# fit. @196608 fits with only ~1.6 GiB margin — too thin given the compute
# buffer's observed growth. 131072 is the robust setting.
#
# --no-mmap: 165 GB split GGUF; read() into anonymous memory + --mlock avoids
# double residency in page cache. Experts measured: 161,318.6 MiB CUDA_Host.
#
# Speed: routed reads ~3.07 GB/token (10x60 experts @ ~3.25 bpw) — half the
# 122B-Q6's 6.2 GB/token (measured 7-10 tok/s). Expect ~9-14 tok/s.
# Adopted after field A/B (2026-06-12): 0-15 / -t 16 measured ~+30% decode
# vs 0-7 / -t 8 on the bandwidth-bound MoE expert reads (4.8 -> ~6.4+ tok/s,
# near the DDR5-4000 ceiling). Cores 16-23 stay free to absorb audio/daemon
# wakeups so llama.cpp's per-op barrier never waits on a preempted thread.
# -t 24 expected flat-to-negative; re-measure via chunk-gap pacing if tried.
#
# BARGE-IN: speak over Athena and she stops within ~0.5 s, listens, then
# either pivots (real interruption -> LLM state rollback, only the words she
# got out stay in the transcript, closed with an em-dash) or resumes the
# sentence (false alarm: cough, door, nothing transcribed). Knobs:
# Values below are headset-tuned from the 2026-06-12 18:16 field session
# (5 missed barges analyzed): soft first-attempt speech measured
# 0.0017-0.0038 RMS, earcup BLEED holds the EMA floor at 0.0005-0.0008
# while she speaks, and silence floor is ~0.0002.
#   --barge-rms 0.0020       absolute MINIMUM threshold (high-passed 300 ms
#                            RMS). 2026-06-13 demo: your intentional barges
#                            ran 0.0024-0.0059, but soft BACKCHANNELS ("sure",
#                            "absolutely") landed 0.0019-0.0023 and cut her
#                            off. 0.0020 gates the softest of those at the
#                            energy stage; the rest are caught by content
#                            (is_backchannel) which resumes her instead of
#                            derailing. Miss a soft deliberate barge -> 0.0015
#                            (the filter still prevents backchannel derail);
#                            still cut off mid-thought -> 0.0024.
#   --barge-ratio 1.5        a barge must also exceed this multiple of the
#                            self-measured playback-time ambient. On a
#                            headset that ambient IS earcup bleed: at the
#                            old 4.0 the bleed-fed floor (0.0008) raised the
#                            bar to 0.0032 and beat the absolute — misses.
#                            1.5 keeps the absolute in charge (1.5 x 0.0008
#                            = 0.0012 < 0.0015) while still scaling if room
#                            noise genuinely jumps. OPEN SPEAKERS: go back
#                            to 4.0+ — there the ratio arm is the defense.
#   --barge-ms 150           sustained energy required (consecutive polls;
#                            one cold poll resets the run). Two misses were
#                            hot at 0.0035-0.0038 but never held 300 ms.
#                            Each poll already averages a 300 ms window, so
#                            150 latches one-word interjections. Breath/
#                            plosive false triggers -> raise to 200 first.
#   --barge-blackout-ms 200  arm delay after the first sentence flush. Her
#                            voice cannot reach a headset mic, so the old
#                            700 ms (first-audio + AEC settle) is dead time;
#                            calibration on room ambience is valid anytime.
#                            OPEN SPEAKERS: restore 700.
# With -pe set, a once-per-second "barge monitor: rms/peak/floor/threshold"
# line shows your live levels — calibrate from those numbers, not by feel.
# A false trigger costs a ~1-2 s hiccup (validation -> false-alarm resume);
# a missed barge costs the feature. Tune aggressive.
# Cost when idle: one state snapshot per turn (~0.5 GB host RAM reused,
# ~30-90 ms typical; grows ~30 KB/token of context + 186 MB GDN state, so
# ~4 GB / a few hundred ms at the full 131K window).
#
# ── prosodic endpointing (--endpoint) ────────────────────────────────────────
# Replaces the single fixed end-of-turn wait (--vad-last-ms, 400 ms here) with
# two, chosen from the pitch/energy contour of the speech just before the pause:
#   --endpoint-short-ms 800   turn-final FALL/trail-off (NOW == long: short path
#                             collapsed, §23.7 — was 350, which cut mid-thought pauses)
#   --endpoint-long-ms  800   a flat/rising "not done yet" pause -> keep listening
# This is the right shape for an already-snappy 400 ms setup: the win is the long
# branch (stop cutting off mid-thought pauses), with a slightly faster turn-final.
# --vad-last-ms stays the fallback when --endpoint is absent. A wrong "final" call
# is caught by the barge-in path, so the downside is bounded by the old behavior.
# Decision thresholds (tune from the -pe log, which now prints one
# "endpoint f0_slope=.. e_slope=.. -> turn-final/continue" line per turn):
#   --endpoint-f0-fall 60       Hz/s of F0 fall that counts as falling
#   --endpoint-energy-decay 4.0 log-energy slope (/s) that counts as trailing off
# Read f0_slope/e_slope on turns you KNOW were final vs mid-thought, then set the
# thresholds where they separate cleanly. Loosen (lower) to catch more finals;
# tighten (raise) if it answers over your pauses.
#
# ── Silero streaming VAD (--vad-engine silero, the default) ──────────────────
# The neural Silero VAD streams per-frame speech probabilities and ends the turn
# on sustained REAL silence — fixing the 10-30 s hangovers the energy vad_simple
# hit once speech scrolled out of its relative window. --endpoint still applies:
# prosody picks the silence TARGET per turn (--endpoint-short/long below) and
# Silero detects the silence. If the model is missing it falls back to the energy
# path (--vad-last-ms 400 + the absolute-floor backstop). Silero knobs (defaults):
#   --silero-threshold 0.5      speech-probability cutoff
#   --silero-min-run-ms 100     HYSTERESIS: consecutive speech (ms) needed to count
#                               as speech. Single-frame Silero spikes (noise/breath)
#                               are shorter and so can't reset the silence clock —
#                               this is the fix for "turn never ended" flapping.
#   --silero-silence-ms 700     turn-end silence used only when --endpoint is OFF
#   --silero-min-speech-ms 120  ignore sub-utterance blips (cumulative)
#   --silero-poll-ms 100        capture + inference cadence
#   --silero-debug              per-poll [silero-dbg] trace (prob + accumulators)
# NOTE: SHORT-PATH COLLAPSED — short-ms == long-ms == 800 (CHANGES.MD §23.7).
# The prosodic turn-final decision (talk-llama.cpp:1768) is an OR of f0-fall/energy-
# decay and there is NO minimum-silence floor on the Silero path, so a sincere
# falling clause + a >350 ms dramatic pause fired the SHORT (350 ms) target and cut
# beat-4's final sentence (run 20260704-222137, silence=352ms). Setting short==long
# makes prosody's verdict irrelevant to timing: every turn now needs a full 800 ms of
# trailing silence, structurally eliminating the premature-fire class. 800 = ~30-95 ms
# above this run's genuine turn-ends (704-768 ms) and above the ~640 ms within-turn
# pause median (research), so dramatic pauses up to 800 ms are safe. Snappier turn-
# taking: drop both to 700 (proven 23/23 clean, no margin). More margin: raise both to
# 850. Do NOT re-split them without adding a min-silence floor in silero-turn-state.h.
# (Original: 350/1200; long-ms had been lowered to 700 because the pitch detector
# reads voiced≈0 on quiet post-speech audio, so the long target already fired nearly
# every turn.) --vad-last-ms is a red herring here (Silero path ignores it).
# --silero-debug is ON for tuning — delete that line once dialled in.
#
# PROSODY-TO-EXTEND (CHANGES.MD §23.8): the robust use of prosody — grant MORE time
# when the acoustics say "not done", NEVER less (the §23.7 800 ms floor still holds;
# extend only raises the target via max()). --endpoint-extend-ms 1200: on a "not-done"
# contour, wait 1200 ms instead of 800. "Continuing" = ProsodicEndpointer.continuing:
# !turn_final AND (F0 slope > +f0-rise, voiced) AND (log-energy slope > +energy-rise,
# voiced) — BOTH a pitch rise AND rising/held energy required (an AND, not OR).
#   WHY AND (review C1/C2): a wrong EXTEND is UNRECOVERABLE — Athena answers LATE while
#   the user, done, stays silent, so no barge-in recovers it (a wrong SHORT is recovered
#   by a barge). A genuinely turn-final yes/no QUESTION rises in pitch but its energy
#   TAPERS into the pause -> energy-rise fails -> NOT extended. A real held continuation
#   ("...and,") both rises and sustains energy -> extended. extend capped at base+400 to
#   bound the dead-air of any residual false-positive.
#   DIALS: still over-extending questions -> raise --endpoint-f0-rise or set
#   --endpoint-extend-ms 0 (off, reverts to flat 800). Want more coverage (accept the
#   question risk) -> lower --endpoint-f0-rise / switch continuing to || in the source.
#   --endpoint-f0-fall/-energy-decay still select short-vs-long, inert since
#   short==long==800; extend is layered on top. Watch the [silero] "-> not-done/extend
#   (1200 ms)" debug lines: they should fire on held mid-thought rises, NOT genuine ends.
# NEVER core-dump the brain: on the reference machine its mlock'd RSS would write a
# catastrophic core (CHANGES.MD §12); harmless here (35B ~ 23 GiB) but kept for safety.
# Device retarget (16 GB VRAM / 62 GB RAM, ~/qwen_t8.sh): --ctx-size 32768, -t 8 on the
# P-cores (taskset 0-15). KV cache is bf16 (NOT qwen_t8's q4_0): with ~11 GB VRAM free at
# 32k there is no need to quantize KV, and bf16 avoids the Qwen output degradation seen
# with quantized KV. Barge-in (talk over Athena to interrupt) is set by ATHENA_BARGE below.
# Expert placement: partial GPU offload of MoE experts (ATHENA_N_CPU_MOE, see top). N>0
# keeps the first N layers' experts in RAM and runs the rest on the GPU; 0 = all on CPU.
if [ "${ATHENA_N_CPU_MOE:-0}" -gt 0 ] 2>/dev/null; then
    MOE_FLAG=(--n-cpu-moe "$ATHENA_N_CPU_MOE")
else
    MOE_FLAG=(--cpu-moe)
fi
# Barge-in (only when ATHENA_BARGE != 0). The mic stays open during playback via HFP duplex;
# AEC (ATHENA_AUDIO=aec, defaulted on for the Bose) keeps Athena from hearing herself and
# self-interrupting. Thresholds are the field-tuned ATHENA values — raise --barge-rms /
# --barge-ms if she still cuts herself off, lower them if she's hard to interrupt.
if [ "${ATHENA_BARGE:-1}" != "0" ]; then
    BARGE_FLAG=(--barge-in --barge-rms 0.0020 --barge-ratio 1.5 --barge-ms 150 --barge-blackout-ms 200)
    [ "$ATHENA_AUDIO" != "aec" ] && echo "[launch-athena] NOTE: barge-in ON without AEC (ATHENA_AUDIO=$ATHENA_AUDIO) — on an open speaker Athena may interrupt herself; use ATHENA_AUDIO=aec"
else
    BARGE_FLAG=()
fi
ulimit -c 0
taskset -c 0-15 "$TALK_LLAMA" \
    -ml "$QWEN_MODEL" \
    -mw "$WHISPER_MODEL" \
    -l en \
    --ctx-size 32768 \
    --mlock \
    --no-mmap \
    "${MOE_FLAG[@]}" \
    -t 8 -ngl 99 \
    -ctk bf16 -ctv bf16 -fa \
    --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.00 \
    --presence-penalty 1.5 --repeat-penalty 1.0 \
    --reasoning off \
    -s "$SPEAK_DAEMON" -sf "$SPEAK_FILE" \
    --stream-file "$TRIGGER_FILE" \
    "${BARGE_FLAG[@]}" \
    "${MEMORY_ARGS[@]}" \
    -p Igor -bn Athena \
    -mt 256 -vms 25000 \
    --vad-engine silero \
    --vad-model "$VAD_MODEL" \
    --vad-last-ms 400 \
    --vad-window-ms 700 \
    --endpoint \
    --endpoint-short-ms 800 \
    --endpoint-long-ms 800 \
    --endpoint-f0-fall 60 \
    --endpoint-energy-decay 4.0 \
    --endpoint-extend-ms 1200 \
    --endpoint-f0-rise 50 \
    --endpoint-energy-rise 0.5 \
    --silero-min-run-ms 100 \
    --silero-debug \
    -pe

# If talk-llama exits on its own, cleanup runs via the trap
