#!/usr/bin/env bash
# apply-crash-capture.sh — make the NEXT ATHENA GPU hard-lock actually capturable.
#
# WHY: the 20260701-074343 kernel panic left no backtrace. This wires up three
# Bash/C++-only fixes (NO Python):
#   1. builds athena-kmsg-logger (C++) — the monitor uses it to keep an UNFILTERED,
#      fdatasync'd copy of the kernel log (kmsg-full.log) so an oops/panic body is
#      preserved instead of grep'd away and lost in the page cache.
#   2. installs sysctls (99-athena-crash.conf) that escalate a silent lock to a real
#      panic() → kmsg_dump writes RIP+trace into efi-pstore (survives the cold reboot).
#   3. installs zzz-athena-gsp-diag.conf so NVreg_EnableGpuFirmwareLogs=1 routes GSP
#      firmware logs into dmesg for the #1111 bug report.
# After the next crash, launch-athena-397b.sh auto-harvests /sys/fs/pstore +
# nvidia-bug-report.sh into athena-diag/crash-<ts>/ on the following launch.
#
# Usage:  ./apply-crash-capture.sh            # build + install + apply live
#         ./apply-crash-capture.sh --revert   # remove sysctls + diag conf, reset live
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SYSCTL_SRC="$HERE/athena-crash-capture.sysctl.conf"
SYSCTL_DST="/etc/sysctl.d/99-athena-crash.conf"
DIAG_SRC="$HERE/zzz-athena-gsp-diag.conf"
DIAG_DST="/etc/modprobe.d/zzz-athena-gsp-diag.conf"
LOGGER_SRC="$HERE/athena-kmsg-logger.cpp"
LOGGER_BIN="$HERE/athena-kmsg-logger"

# Debian omits /usr/sbin from the user PATH (same trap that broke the MPS server) —
# sysctl lives there, so resolve it explicitly.
SYSCTL_BIN="$(command -v sysctl || true)"; [ -n "$SYSCTL_BIN" ] || SYSCTL_BIN=/usr/sbin/sysctl

if [[ "${1:-}" == "--revert" ]]; then
    echo "[crash-capture] reverting..."
    sudo rm -f "$SYSCTL_DST" "$DIAG_DST"
    # Reset the live knobs to stock (panic-on-oops off, no auto-reboot).
    sudo "$SYSCTL_BIN" -w kernel.panic_on_oops=0 kernel.panic=0 kernel.hardlockup_panic=0 >/dev/null 2>&1 || true
    echo "[crash-capture] removed $SYSCTL_DST and $DIAG_DST; live panic sysctls reset."
    echo "[crash-capture] (NVreg_EnableGpuFirmwareLogs clears on next reboot.)"
    exit 0
fi

# 1) Build the C++ /dev/kmsg logger (rebuild if source is newer).
echo "[crash-capture] building athena-kmsg-logger..."
if command -v g++ >/dev/null 2>&1; then
    if [ ! -x "$LOGGER_BIN" ] || [ "$LOGGER_SRC" -nt "$LOGGER_BIN" ]; then
        g++ -O2 -o "$LOGGER_BIN" "$LOGGER_SRC" && echo "  built $LOGGER_BIN"
    else
        echo "  up to date: $LOGGER_BIN"
    fi
else
    echo "  WARNING: g++ not found — the monitor will fall back to an unfiltered (but"
    echo "           un-fsync'd) bash dmesg follow; install g++ and re-run for durability"
fi

# 2) Install + apply the panic-escalation sysctls.
echo "[crash-capture] installing $SYSCTL_DST ..."
sudo install -m 0644 "$SYSCTL_SRC" "$SYSCTL_DST"
echo "[crash-capture] applying sysctls live via $SYSCTL_BIN ..."
sudo "$SYSCTL_BIN" --system >/dev/null 2>&1 || sudo "$SYSCTL_BIN" -p "$SYSCTL_DST" >/dev/null 2>&1 || \
    echo "  WARNING: could not apply sysctls live (they still apply on next boot)"
for k in kernel.panic_on_oops kernel.panic kernel.hardlockup_panic; do
    v="$(sudo "$SYSCTL_BIN" -n "$k" 2>/dev/null || echo '?')"; echo "  $k = $v"
done

# 3) Install the GSP-firmware-log modprobe conf (applies next boot).
echo "[crash-capture] installing $DIAG_DST ..."
sudo install -m 0644 "$DIAG_SRC" "$DIAG_DST"

# 4) Verify the efi-pstore backend is actually present (the durable sink).
echo "[crash-capture] checking efi-pstore backend..."
if [ -d /sys/firmware/efi ]; then echo "  ok: UEFI boot"; else echo "  WARNING: no /sys/firmware/efi — efi-pstore unavailable"; fi
if mount 2>/dev/null | grep -q 'type pstore'; then echo "  ok: /sys/fs/pstore mounted"; else echo "  WARNING: /sys/fs/pstore not mounted — run: sudo mount -t pstore pstore /sys/fs/pstore"; fi
if lsmod 2>/dev/null | grep -q '^efi_pstore'; then echo "  ok: efi_pstore loaded"; else echo "  NOTE: efi_pstore not shown in lsmod (may be built-in)"; fi

cat <<EOF

[crash-capture] DONE.
  • Live now: a kernel oops/hard-lockup will escalate to panic() and dump to efi-pstore.
  • Next boot: NVreg_EnableGpuFirmwareLogs=1 adds GSP firmware detail to dmesg
    (reboot is YOUR job). No initramfs rebuild needed (nvidia is modeset=0).
  • Next launch after a crash: launch-athena-397b.sh copies /sys/fs/pstore +
    nvidia-bug-report.sh into athena-diag/crash-<ts>/, then clears pstore.

  Reboot to activate the GSP firmware logs:  sudo reboot
  Revert everything:                         ./apply-crash-capture.sh --revert
EOF
