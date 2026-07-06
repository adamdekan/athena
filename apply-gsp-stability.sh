#!/usr/bin/env bash
# apply-gsp-stability.sh — stage the Blackwell GSP-fault mitigation, then reboot.
#
# WHY: the 20260630 crash began with "NVRM: GspMsgQueueReceiveStatus: Bad checksum"
# (GSP firmware RPC corruption) — a known, unresolved Blackwell open-driver bug
# (open-gpu-kernel-modules #1064/#1111) that correlates with runtime GPU power-state
# transitions (D3cold<->D0, S0ix). This installs nvidia-athena-gsp-stability.conf,
# which keeps the GPU resident in D0 (removing those transitions). Compute perf is
# unchanged; only idle power rises slightly. GSP cannot be disabled on Blackwell.
#
# SAFE + IDEMPOTENT: verifies the module params exist first, backs nothing up it
# doesn't need to, and does NOT touch GRUB automatically (it prints the optional
# pcie_aspm=off step for you to do by hand). Run it, then reboot.
#
# Usage:  ./apply-gsp-stability.sh          # applies + rebuilds initramfs
#         ./apply-gsp-stability.sh --revert # removes the conf + rebuilds initramfs
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$HERE/zzz-nvidia-athena-gsp-stability.conf"
DST="/etc/modprobe.d/zzz-nvidia-athena-gsp-stability.conf"
LEGACY_DST="/etc/modprobe.d/nvidia-athena-gsp-stability.conf"  # pre-fix name (sorted too early)
UVM_SRC="$HERE/athena-uvm-hardening.conf"                       # §16 Fix 1: uvm_disable_hmm/ats
UVM_DST="/etc/modprobe.d/athena-uvm-hardening.conf"

rebuild_initramfs() {
    if   command -v update-initramfs >/dev/null 2>&1; then sudo update-initramfs -u
    elif command -v dracut           >/dev/null 2>&1; then sudo dracut -f
    else echo "  WARNING: no update-initramfs/dracut found — rebuild your initramfs manually before reboot"; fi
}

if [[ "${1:-}" == "--revert" ]]; then
    echo "[gsp-stability] reverting..."
    sudo rm -f "$DST" "$LEGACY_DST" "$UVM_DST"
    rebuild_initramfs
    echo "[gsp-stability] removed $DST + $UVM_DST (and any legacy copy) — REBOOT to return to stock."
    exit 0
fi

[ -f "$SRC" ] || { echo "ERROR: $SRC not found (run from the ATHENA dir)"; exit 1; }

# 1) Best-effort confirm the driver accepts these params (guards a typo / drift).
#    Prefer modinfo; fall back to sysfs; if neither is available, proceed anyway
#    (unknown nvidia module options are IGNORED by the driver, never fatal).
echo "[gsp-stability] checking nvidia module params..."
param_known() {  # $1 = NVreg_ name
    if command -v modinfo >/dev/null 2>&1; then
        modinfo nvidia 2>/dev/null | grep -q "parm:.*$1" && return 0 || return 1
    elif [ -e "/sys/module/nvidia/parameters/$1" ]; then
        return 0
    fi
    return 2   # can't verify
}
if ! command -v modinfo >/dev/null 2>&1 && [ ! -d /sys/module/nvidia/parameters ]; then
    echo "  (cannot verify param names on this host — modinfo absent and sysfs params not exposed;"
    echo "   proceeding: the three options are documented NVIDIA open-module params, and any the"
    echo "   driver does not recognize are silently ignored, not fatal)"
else
    for p in NVreg_DynamicPowerManagement NVreg_EnableS0ixPowerManagement NVreg_PreserveVideoMemoryAllocations; do
        param_known "$p"; case $? in
            0) echo "  ok: $p" ;;
            1) echo "  WARNING: $p not advertised by this driver — will be ignored (harmless); review before relying on it" ;;
            2) echo "  (unverifiable: $p — proceeding)" ;;
        esac
    done
fi

# 2) Install the conf, and remove the pre-fix legacy copy (which sorted BEFORE
#    nvidia.conf and let nvidia.conf's NVreg_EnableS0ixPowerManagement=1 override us).
echo "[gsp-stability] installing $DST ..."
sudo install -m 0644 "$SRC" "$DST"
if [ -e "$LEGACY_DST" ]; then
    echo "[gsp-stability] removing legacy $LEGACY_DST (sorted too early — was being overridden)"
    sudo rm -f "$LEGACY_DST"
fi

# §16 Fix 1: UVM hardening (uvm_disable_hmm=1 uvm_ats_mode=0 — shrink the folio-
# lifecycle machinery neighboring the live-struct-page corruption; zero cost for
# ATHENA's cudaMalloc-only stack). No sort-order concern: nothing else sets
# nvidia_uvm options.
if [ -f "$UVM_SRC" ]; then
    echo "[gsp-stability] installing $UVM_DST ..."
    sudo install -m 0644 "$UVM_SRC" "$UVM_DST"
else
    echo "  WARNING: $UVM_SRC missing — UVM hardening skipped"
fi

# 2b) Conflict check: for each param we set, report EVERY `options nvidia` line
#     across /etc/modprobe.d in C-locale read order and flag which file wins
#     (kernel module params are last-assignment-wins). Our file must be last.
echo "[gsp-stability] checking modprobe.d precedence (last line wins)..."
mapfile -t CONF_ORDER < <(ls /etc/modprobe.d/*.conf 2>/dev/null | LC_ALL=C sort)
for p in NVreg_DynamicPowerManagement NVreg_EnableS0ixPowerManagement NVreg_PreserveVideoMemoryAllocations; do
    winner=""; wval=""
    for f in "${CONF_ORDER[@]}"; do
        line="$(grep -E "^[[:space:]]*options[[:space:]]+nvidia\b.*${p}=" "$f" 2>/dev/null | tail -1 || true)"
        if [ -n "$line" ]; then winner="$f"; wval="$(sed -E "s/.*${p}=([^[:space:]]+).*/\1/" <<<"$line")"; fi
    done
    if [ -z "$winner" ]; then echo "  $p: (not set anywhere)"; continue; fi
    if [ "$winner" = "$DST" ]; then echo "  ok: $p=$wval  (wins, from ${winner##*/})"
    else echo "  WARNING: $p=$wval is set LAST by ${winner##*/} and OVERRIDES this conf — rename that file or remove its line"; fi
done

# 3) Rebuild the initramfs so the option is present at next boot.
echo "[gsp-stability] rebuilding initramfs..."
rebuild_initramfs

# 4) Optional PCIe-link belt-and-suspenders (manual — we don't auto-edit GRUB).
cat <<'EOF'

[gsp-stability] DONE (staged). One more OPTIONAL step, by hand:
  The GSP RPC ring rides the PCIe link; disabling ASPM link-power transitions can
  further stabilize it (no compute cost). If you want it, add pcie_aspm=off:
    - edit /etc/default/grub -> GRUB_CMDLINE_LINUX_DEFAULT="... pcie_aspm=off"
    - sudo update-grub

REBOOT to activate:  sudo reboot

After reboot, verify it took (use /proc — /sys/module/nvidia is virtualized away in
the GPU-passthrough container; nvidia_uvm params ARE visible in /sys):
  grep -iE 'EnableS0ix|DynamicPower' /proc/driver/nvidia/params
      # expect  EnableS0ixPowerManagement: 0   and  DynamicPowerManagement: 0
  cat /proc/driver/nvidia/gpus/*/power | grep -iE 'S0ix|Runtime D3|Status'
      # expect  Runtime D3 status: Disabled   and  S0ix ... Status: Disabled
  cat /sys/module/nvidia_uvm/parameters/uvm_disable_hmm   # expect Y or 1 (§16)
  cat /sys/module/nvidia_uvm/parameters/uvm_ats_mode      # expect 0      (§16)
Then run ATHENA as usual and watch athena-diag/<ts>/xid.log for a recurrence of
"GspMsgQueueReceiveStatus: Bad checksum" or "Xid". Revert any time: ./apply-gsp-stability.sh --revert
EOF
