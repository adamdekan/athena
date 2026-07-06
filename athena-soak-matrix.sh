#!/usr/bin/env bash
# athena-soak-matrix.sh — attribute the kernel-memory corruption to a component or
# interaction, empirically (CHANGES.MD §21).
#
# WHY: 8 crashes/hangs share one disease — single-byte scribbles in KERNEL memory
# (freed-page poison overwrites, folio->memcg pointers, XArray nodes, PTEs) under GPU
# load. Userspace code CANNOT write those addresses, so no talk-llama/orpheus/server
# code audit can convict or acquit the app INTERACTION — only isolation can: run each
# load shape alone, count corruption events, compare. Falsified so far: DRAM (memtest
# clean x2), driver version (571..610 all affected), DMA (IOMMU translated, 0 faults),
# kernel version (6.12.94 + 7.0.12), page hygiene, S0ix/RTD3, PreserveVMA, huge-vmalloc.
#
# PHASES (run one, several, or 'all'; ATHENA must NOT be running):
#   A  idle         GPU idle control — is the box corrupting with no GPU load at all?
#   B  smi-churn    4 parallel `nvidia-smi` spawn loops (UVM va-space churn repro) —
#                   does ANY trivial UVM client suffice, no ATHENA involved?
#   C  orpheus      llama-server (Orpheus GGUF, port 8181) + scripted completions, NO MPS
#   D  orpheus-mps  same load under MPS — does MPS change the rate for ONE client?
#   E  qwen         sustained Qwen decode via llama-cli (--cpu-moe -ngl 99, production
#                   flags, ~161 GiB mlock) — the "brain" load alone, NO MPS
#   F  dual-mps     C-load + E-load CONCURRENTLY under MPS — the two-client interaction
#                   (the audio-free analog of a live ATHENA session)
#
# READING THE TABLE (events are deltas of kernel log signatures per phase):
#   A>0            → platform corrupts without the GPU: kernel/CPU/IMC/power — not ATHENA.
#   B>0            → any UVM client triggers it: driver kernel-side bug, trivial upstream repro.
#   E>0, C≈0       → the big-model/cpu-moe profile is the trigger (bus+GPU concurrency).
#   D≫C or F≫C+E   → MPS / multi-client INTERACTION is the trigger → running without MPS
#                    (or serializing clients) becomes a legitimate mitigation lever.
#   all ≈0 but live ATHENA still corrupts → the remainder (audio stack, whisper, emotion
#                    ONNX, barge/snapshot traffic) — next matrix level.
#
# Usage:  ./athena-soak-matrix.sh all            # every phase, default 600 s each
#         SOAK_SECS=1200 ./athena-soak-matrix.sh B F
#         ./athena-soak-matrix.sh C              # single phase
# Qwen phases (E/F) additionally allow QWEN_LOAD_ALLOW (default 600 s) for model load.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOAK_SECS="${SOAK_SECS:-600}"
QWEN_LOAD_ALLOW="${QWEN_LOAD_ALLOW:-600}"
ORPHEUS_MODEL="$HERE/models/orpheus-3b-0.1-ft-UD-Q4_K_XL.gguf"
QWEN_MODEL="$HERE/models/Qwen3.5-397B-A17B-UD-Q3_K_XL-00001-of-00005.gguf"
LLAMA_SERVER="$HERE/llama.cpp/build/bin/llama-server"
LLAMA_CLI="$HERE/llama.cpp/build/bin/llama-cli"
PORT=8181
MPS_DIR="/tmp/athena-soak-mps"
OUT="$HERE/athena-diag/soak-$(date +%Y%m%d-%H%M%S)"
SIGS='pagealloc: memory corruption|BUG: Bad page|Bad page map|NVRM: Xid|general protection|Oops'

mkdir -p "$OUT"
declare -a RESULT_ROWS=()
declare -a CHILD_PIDS=()

die() { echo "ERROR: $*" >&2; exit 1; }

# ── guards ────────────────────────────────────────────────────────────────────
pgrep -f whisper-talk-llama >/dev/null && die "ATHENA is running — stop it first"
curl -s --max-time 1 http://127.0.0.1:8080/health >/dev/null 2>&1 && die "something is on :8080 — stop ATHENA first"
[ -x "$LLAMA_SERVER" ] || die "$LLAMA_SERVER missing"
# dmesg is root-only on this boot (dmesg_restrict): authenticate sudo ONCE here,
# then refresh the timestamp before each count so multi-phase runs never re-prompt.
sudo -v || die "sudo required (kernel log is root-only)"
sudo -n dmesg >/dev/null 2>&1 || die "dmesg not readable even via sudo"

count_sigs() { sudo -n dmesg 2>/dev/null | grep -acE "$SIGS"; }

reap() {
    local p
    for p in "${CHILD_PIDS[@]:-}"; do
        [ -n "$p" ] && kill "$p" 2>/dev/null
    done
    sleep 1
    for p in "${CHILD_PIDS[@]:-}"; do
        [ -n "$p" ] && kill -9 "$p" 2>/dev/null
    done
    CHILD_PIDS=()
    pkill -f "llama-server.*--port $PORT" 2>/dev/null
    pkill -f "athena-soak-curl-load" 2>/dev/null
    sleep 2
}
trap 'reap; mps_stop 2>/dev/null; echo "[soak] interrupted"; exit 130' INT TERM

mps_start() {
    export PATH="/usr/sbin:$PATH"
    export CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR" CUDA_MPS_LOG_DIRECTORY="$OUT/mps-log"
    mkdir -p "$MPS_DIR" "$OUT/mps-log"
    nvidia-cuda-mps-control -d >/dev/null 2>&1
    echo "start_server -uid $(id -u)" | nvidia-cuda-mps-control >/dev/null 2>&1
    sleep 1
    echo "[soak]   MPS daemon up ($MPS_DIR)"
}
mps_stop() {
    [ -d "$MPS_DIR" ] || return 0
    echo quit | timeout 10 env CUDA_MPS_PIPE_DIRECTORY="$MPS_DIR" nvidia-cuda-mps-control >/dev/null 2>&1
    pkill -f nvidia-cuda-mps 2>/dev/null
    rm -rf "$MPS_DIR"
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
    sleep 1
}

start_orpheus_server() {          # $1 = log tag
    "$LLAMA_SERVER" -m "$ORPHEUS_MODEL" -c 13824 -np 2 -ngl 99 \
        --host 127.0.0.1 --port "$PORT" \
        --cache-type-k f16 --cache-type-v f16 --cache-ram 0 -fa on -t 0 \
        > "$OUT/$1-server.log" 2>&1 &
    CHILD_PIDS+=($!)
    local i
    for i in $(seq 1 240); do
        curl -s --max-time 1 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 "${CHILD_PIDS[-1]}" 2>/dev/null || die "orpheus llama-server died during load (see $OUT/$1-server.log)"
        sleep 0.5
    done
    die "orpheus llama-server not healthy in 120 s"
}
start_curl_load() {               # continuous ~512-token completions
    ( exec -a athena-soak-curl-load bash -c '
        P='"$PORT"'
        while :; do
            curl -s -m 120 "http://127.0.0.1:$P/completion" \
                 -d "{\"prompt\":\"<|audio|>tara: The quick brown fox and a very long tale about mountains<|eot_id|>\",\"n_predict\":512,\"temperature\":0.6}" >/dev/null 2>&1
        done' ) &
    CHILD_PIDS+=($!)
}
start_qwen_decode() {             # $1 = log tag — production-shaped brain load
    taskset -c 0-15 "$LLAMA_CLI" -m "$QWEN_MODEL" \
        --cpu-moe -ngl 99 -t 16 -c 8192 -ctk bf16 -ctv bf16 -fa on \
        --mlock --no-mmap --temp 0.7 --top-p 0.8 --top-k 20 \
        -n 1000000 --ignore-eos --no-display-prompt \
        -p "Write an extremely long, winding story about a lighthouse keeper." \
        > "$OUT/$1-qwen.log" 2>&1 &
    CHILD_PIDS+=($!)
}

run_phase() {                     # $1=name  $2=setup-fn  $3=extra-secs
    local name="$1" fn="$2" extra="${3:-0}"
    local before after t0 dur
    echo ""
    echo "[soak] ── phase $name — $(date '+%H:%M:%S'), soak ${SOAK_SECS}s ──"
    sudo -v 2>/dev/null   # refresh the sudo timestamp for this phase's counts
    before=$(count_sigs)
    sudo -n dmesg 2>/dev/null | tail -1 > "$OUT/$name-dmesg-mark"
    t0=$SECONDS
    $fn
    sleep "$(( SOAK_SECS + extra ))"
    reap
    mps_stop
    sudo -v 2>/dev/null
    after=$(count_sigs)
    dur=$(( SECONDS - t0 ))
    sudo -n dmesg 2>/dev/null | grep -aE "$SIGS" | tail -20 > "$OUT/$name-signatures.log"
    printf '[soak]   phase %-12s events: %d  (%d → %d, %ds)\n' "$name" "$((after-before))" "$before" "$after" "$dur"
    RESULT_ROWS+=("$(printf '%-14s %8d %8ds' "$name" "$((after-before))" "$dur")")
}

# ── phase bodies ──────────────────────────────────────────────────────────────
phase_A() { :; }                                              # idle control
phase_B() {
    local i
    for i in 1 2 3 4; do
        ( exec -a athena-soak-curl-load bash -c 'while :; do nvidia-smi >/dev/null 2>&1; done' ) &
        CHILD_PIDS+=($!)
    done
}
phase_C() { start_orpheus_server C; start_curl_load; }
phase_D() { mps_start; start_orpheus_server D; start_curl_load; }
phase_E() { start_qwen_decode E; }
# G: GPU-IDLE control — the same Qwen decode with -ngl 0 (pure CPU + DDR5 traffic,
# GPU untouched). Discriminates "heavy CPU/RAM load corrupts" (G>0: memtest-style
# load was never the right test) from "corruption requires GPU power draw" (G≈0
# while C/E>0 — the phase C+E result pattern → platform power/electrical under GPU
# load, ATHENA and the CPU/RAM acquitted).
start_qwen_decode_cpu() {          # $1 = log tag
    taskset -c 0-15 "$LLAMA_CLI" -m "$QWEN_MODEL" \
        --cpu-moe -ngl 0 -t 16 -c 8192 -fa on \
        --mlock --no-mmap --temp 0.7 --top-p 0.8 --top-k 20 \
        -n 1000000 --ignore-eos --no-display-prompt \
        -p "Write an extremely long, winding story about a lighthouse keeper." \
        > "$OUT/$1-qwen-cpu.log" 2>&1 &
    CHILD_PIDS+=($!)
}
phase_G() { start_qwen_decode_cpu G; }
phase_F() { mps_start; start_orpheus_server F; start_curl_load; start_qwen_decode F; }

# ── main ──────────────────────────────────────────────────────────────────────
[ $# -ge 1 ] || die "usage: $0 all | A B C D E F G  (env: SOAK_SECS=$SOAK_SECS QWEN_LOAD_ALLOW=$QWEN_LOAD_ALLOW)"
PHASES="$*"; [ "$1" = "all" ] && PHASES="A B C D E F G"
echo "[soak] phases: $PHASES | soak ${SOAK_SECS}s each | results -> $OUT"
echo "[soak] boot signature baseline: $(count_sigs)"

for ph in $PHASES; do
    case "$ph" in
        A) run_phase A phase_A 0 ;;
        B) run_phase B phase_B 0 ;;
        C) run_phase C phase_C 0 ;;
        D) run_phase D phase_D 0 ;;
        E) run_phase E phase_E "$QWEN_LOAD_ALLOW" ;;
        G) run_phase G phase_G "$QWEN_LOAD_ALLOW" ;;
        F) run_phase F phase_F "$QWEN_LOAD_ALLOW" ;;
        *) echo "[soak] unknown phase '$ph' — skipped" ;;
    esac
done

echo ""
echo "[soak] ═══ SUMMARY (kernel corruption signatures per phase) ═══"
printf '%-14s %8s %9s\n' "phase" "events" "duration"
for row in "${RESULT_ROWS[@]}"; do echo "$row"; done
echo ""
echo "[soak] interpretation guide is in the header of this script."
echo "[soak] per-phase signature excerpts: $OUT/<phase>-signatures.log"
