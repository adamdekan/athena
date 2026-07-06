#!/usr/bin/env bash
# athena-gpu-monitor.sh — lightweight GPU + kernel telemetry sampler for diagnosing
# Athena's TTS stalls and llama-server aborts. Writes timestamped logs to an output
# directory and tears down all child samplers cleanly on TERM/INT/EXIT.
#
#   Usage: athena-gpu-monitor.sh <output_dir> [interval_ms=500]
#
# Outputs (in <output_dir>):
#   gpu.csv           GPU clocks/power/temp/util/VRAM at <interval_ms> cadence.
#                     The first CSV column is nvidia-smi's own timestamp.
#   gpu-proc.log      1 Hz per-process compute VRAM — shows WHICH process's memory
#                     drops when one dies (e.g. the ~3.9 GiB Orpheus free on abort).
#   gpu-throttle.log  1 Hz active clock-event/throttle reasons (power vs thermal vs
#                     reliability cap). Field name differs by driver — both tried.
#   xid.log           kernel Xid / NVRM faults (GPU errors invisible to the apps).
#                     Needs dmesg read access; the launcher already uses sudo.
set -u

OUT="${1:?usage: athena-gpu-monitor.sh <out_dir> [interval_ms]}"
IVL_MS="${2:-1000}"   # 1 Hz default (was 500ms): lower nvidia-smi churn — see CHANGES.MD §12
mkdir -p "$OUT" 2>/dev/null || { echo "[gpu-monitor] cannot create $OUT" >&2; exit 1; }

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "[gpu-monitor] nvidia-smi not found — nothing to sample" >&2
    exit 0
fi

IVL_S="$(awk "BEGIN{printf \"%.3f\", ${IVL_MS}/1000.0}")"
GPU_CSV="$OUT/gpu.csv"
PROC_LOG="$OUT/gpu-proc.log"
THR_LOG="$OUT/gpu-throttle.log"
XID_LOG="$OUT/xid.log"
PROC_FIFO="$OUT/.gpu-proc.fifo"       # feeds the gpu-proc [wall] prefixer (see §2)
THR_FIFO="$OUT/.gpu-throttle.fifo"    # feeds the throttle  [wall] prefixer (see §3)

pids=()
cleanup() {
    trap - TERM INT EXIT
    for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null; done  # each supervisor's TERM trap kills its nvidia-smi
    rm -f "$PROC_FIFO" "$THR_FIFO" 2>/dev/null
}
trap cleanup TERM INT EXIT

now() { date '+[%Y-%m-%d %H:%M:%S.%6N]'; }

# One-time snapshot of the power-management state this run depends on: whether
# Dynamic Boost is actually being managed (nvidia-powerd) and what the driver
# thinks the platform power/clock limits are. The recurring Xid 69 correlates
# with a failing GPU<->SBIOS power handshake, so capture it up front.
{
    echo "$(now) ===== power-management snapshot ====="
    echo "--- systemctl status nvidia-powerd ---"
    systemctl status nvidia-powerd 2>&1 | head -20
    echo "--- /proc/driver/nvidia/gpus/*/power ---"
    for p in /proc/driver/nvidia/gpus/*/power; do echo "# $p"; cat "$p" 2>/dev/null; done
    echo "--- nvidia-smi -q (POWER/CLOCK/PERF/VOLTAGE) ---"
    nvidia-smi -q -d POWER,CLOCK,PERFORMANCE,VOLTAGE 2>/dev/null \
        | grep -iE "Power Limit|Default Power|Enforced|Power Draw|Clocks Event|Clock Throttle|Performance State|SW Power|HW Power|Slowdown|Graphics Clock|Memory Clock|Voltage|GPU Operation" | head -60
} >> "$OUT/power-state.log" 2>&1

# PERSISTENT SAMPLERS (CHANGES.MD §13). Each stream is ONE long-lived nvidia-smi that
# samples internally via --loop-ms, instead of a fresh nvidia-smi per tick. Rationale:
# every nvidia-smi open does a UVM_INITIALIZE / uvm_va_space_create+destroy, and ~3
# spawns/sec = ~thousands of UVM create/destroy cycles/run — the churn that surfaced the
# kernel memcg/huge-vmalloc use-after-free (bad_page / page_poison "memory corruption").
# One persistent process per stream = UVM va-space created ONCE (~500x fewer), lower CPU,
# zero telemetry loss. A supervisor subshell holds the REAL nvidia-smi PID (via a FIFO,
# not a hidden pipeline), traps TERM to kill it (no orphan), and auto-restarts on death.

# 1) GPU telemetry CSV. nvidia-smi's own `timestamp` is column 1, so --loop-ms rows are
#    byte-identical to the old per-shot output. stderr -> .err sidecar (keeps CSV pristine).
QFIELDS="timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,power.limit,utilization.gpu,utilization.memory,memory.used,memory.total,pstate"
echo "$QFIELDS" > "$GPU_CSV"
(
    smi=""
    trap '[ -n "${smi:-}" ] && kill "$smi" 2>/dev/null; exit 0' TERM
    while :; do
        nvidia-smi --query-gpu="$QFIELDS" --format=csv,noheader,nounits \
                   --loop-ms="$IVL_MS" >> "$GPU_CSV" 2>>"$GPU_CSV.err" & smi=$!
        wait "$smi"; rc=$?
        printf '%s gpu.csv sampler (pid %s) exited rc=%s — restarting in 1s\n' \
               "$(now)" "$smi" "$rc" >> "$GPU_CSV.err"
        sleep 1
    done
) &
pids+=($!)

# 2) Per-process compute VRAM @ 1 Hz. One persistent nvidia-smi loops internally; rows
#    are piped through the FIFO + `while read` to re-add the [wall] prefix (same
#    "[ts] pid, mem, name" format the old sed produced).
mkfifo "$PROC_FIFO" 2>/dev/null
touch "$PROC_LOG"   # guarantee the file exists even before the first compute-app row
(
    smi=""
    trap '[ -n "${smi:-}" ] && kill "$smi" 2>/dev/null; exit 0' TERM
    while :; do
        nvidia-smi --query-compute-apps=pid,used_memory,process_name \
                   --format=csv,noheader,nounits --loop-ms=1000 2>/dev/null > "$PROC_FIFO" & smi=$!
        while IFS= read -r line; do printf '%s %s\n' "$(now)" "$line" >> "$PROC_LOG"; done < "$PROC_FIFO"
        wait "$smi" 2>/dev/null; rc=$?
        printf '%s [gpu-proc sampler (pid %s) exited rc=%s — restarting in 1s]\n' "$(now)" "$smi" "$rc" >> "$PROC_LOG"
        sleep 1
    done
) &
pids+=($!)

# 3) Active clock-event / throttle reasons @ 1 Hz. Probe the driver's field name ONCE
#    (modern clocks_event_reasons.active, else legacy clocks_throttle_reasons.active)
#    instead of double-querying every sample, then one persistent nvidia-smi + FIFO prefix.
if nvidia-smi --query-gpu=clocks_event_reasons.active --format=csv,noheader >/dev/null 2>&1; then
    THR_FIELD="clocks_event_reasons.active"
else
    THR_FIELD="clocks_throttle_reasons.active"
fi
mkfifo "$THR_FIFO" 2>/dev/null
(
    smi=""
    trap '[ -n "${smi:-}" ] && kill "$smi" 2>/dev/null; exit 0' TERM
    while :; do
        nvidia-smi --query-gpu="$THR_FIELD" --format=csv,noheader \
                   --loop-ms=1000 2>/dev/null > "$THR_FIFO" & smi=$!
        while IFS= read -r line; do printf '%s %s\n' "$(now)" "${line:-n/a}" >> "$THR_LOG"; done < "$THR_FIFO"
        wait "$smi" 2>/dev/null; rc=$?
        printf '%s [gpu-throttle sampler (pid %s) exited rc=%s — restarting in 1s]\n' "$(now)" "$smi" "$rc" >> "$THR_LOG"
        sleep 1
    done
) &
pids+=($!)

# 4) Kernel Xid / NVRM faults + FULL unfiltered kernel log. dmesg needs privilege on
#    Debian/Ubuntu (dmesg_restrict); sudo -n avoids hanging if it isn't passwordless.
#    Snapshot the NVRM/Xid backlog first, then follow.
#
#    PREFER the C++ /dev/kmsg logger (athena-kmsg-logger, built by apply-crash-capture.sh):
#    it keeps an UNFILTERED durable copy in kmsg-full.log and fdatasync()s every line,
#    so a kernel oops/panic backtrace (BUG:/Oops:/RIP:/registers/Call Trace:) survives a
#    hard lock. The old `dmesg -w | grep -iE "NVRM|Xid|nvidia"` follow discarded those
#    lines (they match none of those tokens) and never fsync'd — which is why the
#    20260701 panic left no trace. If the binary is missing, fall back to an UNFILTERED
#    bash follow (still fixes the data loss; just lacks the per-line fsync).
{
    echo "$(now) --- dmesg snapshot (NVRM/Xid) ---"
    sudo -n dmesg 2>/dev/null | grep -iE "NVRM|Xid" | tail -50
    echo "$(now) --- following dmesg ---"
} >> "$XID_LOG" 2>/dev/null

KMSG_LOGGER="$(dirname "$0")/athena-kmsg-logger"
if [ -x "$KMSG_LOGGER" ]; then
    sudo -n "$KMSG_LOGGER" "$OUT" &
    pids+=($!)
else
    echo "[gpu-monitor] athena-kmsg-logger not built — using unfiltered bash fallback (no fsync); run ./apply-crash-capture.sh for durable capture" >&2
    (
        sudo -n dmesg -w 2>/dev/null | stdbuf -oL cat | while IFS= read -r line; do
            printf '%s %s\n' "$(now)" "$line" >> "$OUT/kmsg-full.log"
            case "$line" in *NVRM*|*Xid*|*xid*|*nvidia*|*GSP*) printf '%s %s\n' "$(now)" "$line" >> "$XID_LOG";; esac
        done
    ) &
    pids+=($!)
fi

echo "[gpu-monitor] sampling -> $OUT (interval ${IVL_MS}ms, pid $$)" >&2

# 5) nvidia-powerd liveness @ 10 s. If Dynamic Boost management drops mid-run,
#    this pins the moment so it can be correlated against any Xid in xid.log.
(
    last=""
    while :; do
        s="$(systemctl is-active nvidia-powerd 2>/dev/null || echo unknown)"
        if [ "$s" != "$last" ]; then
            echo "$(now) nvidia-powerd: $s" >> "$OUT/power-state.log"
            last="$s"
        fi
        sleep 10
    done
) &
pids+=($!)

wait
