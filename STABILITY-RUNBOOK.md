# ATHENA Stability Runbook — kernel swap + UVM hardening (CHANGES.MD §16)

**Date:** 2026-07-02 · **Context:** six crashes root-caused to in-place single-byte corruption
of live kernel page metadata by CPU-side kernel code (CVE-2026-43303 class, exercised by
NVIDIA driver/UVM kernel paths; Xid 69 is a co-symptom). The fix is a patched kernel —
**6.12.94+deb13 (trixie-security) is already in `/boot`** and carries it. Full evidence:
`CHANGES.MD` §15–§16, panic dumps in `athena-diag/pstore-archive/`.

**Environment facts this runbook relies on (verified 2026-07-02):**
- Driver = NVIDIA CUDA apt repo Debian packaging, `nvidia-open` **595.71.05-1**, DKMS-managed
  (`/usr/sbin/dkms`; modules in `/lib/modules/<kernel>/updates/dkms/`).
- Kernel 6.12.94+deb13: `vmlinuz` + `initrd` present in `/boot`, but **no headers and no
  NVIDIA module built yet** — booting it before step A2 = no GPU driver.
- `page_poison=1 nohugevmalloc` are in `GRUB_CMDLINE_LINUX_DEFAULT` → they apply to every
  GRUB menu entry automatically (tripwires stay on through the swap).
- `update-grub` is not available in the working container — run it on the **host**, as done
  for the previous GRUB changes.

---

## Phase A — UVM hardening + kernel swap (the fix)

### A1. Install the UVM hardening conf

```bash
cd ~/ATHENA
./apply-gsp-stability.sh
```

Expect: the precedence report (all `ok:` lines) plus
`installing /etc/modprobe.d/athena-uvm-hardening.conf`
(`uvm_disable_hmm=1 uvm_ats_mode=0` — zero cost for ATHENA's cudaMalloc-only stack).
Ignore an initramfs warning if one appears — `nvidia` loads post-initramfs (`modeset=0`),
so the conf applies at next boot regardless.

### A2. Build the NVIDIA module for 6.12.94 — BEFORE booting it

```bash
sudo apt update
sudo apt install linux-headers-6.12.94+deb13-amd64
```

Installing the headers triggers Debian's DKMS hook, which auto-builds `nvidia/595.71.05`
for 6.12.94 (~2–4 min on 24 cores). **Verify it built — this is the load-bearing check:**

```bash
/usr/sbin/dkms status
# expect BOTH lines:
#   nvidia/595.71.05, 7.0.12+deb13-amd64, x86_64: installed
#   nvidia/595.71.05, 6.12.94+deb13-amd64, x86_64: installed
ls /lib/modules/6.12.94+deb13-amd64/updates/dkms/
# expect: nvidia.ko.xz  nvidia-uvm.ko.xz  nvidia-modeset.ko.xz  nvidia-drm.ko.xz
```

If the second dkms line is missing, build explicitly:

```bash
sudo /usr/sbin/dkms autoinstall -k 6.12.94+deb13-amd64
```

### A3. Boot into 6.12.94

**One-time (recommended for the first soak):** reboot; at the GRUB menu (5 s timeout —
press/hold `Esc` or an arrow key if hidden) select
**Advanced options for Debian GNU/Linux → Debian GNU/Linux, with Linux 6.12.94+deb13-amd64**.
No config change needed — the entry already exists and inherits the tripwire cmdline.

**Make it the default (after the first clean soak):**

```bash
sudo sed -i 's/^GRUB_DEFAULT=0/GRUB_DEFAULT="Advanced options for Debian GNU\/Linux>Debian GNU\/Linux, with Linux 6.12.94+deb13-amd64"/' /etc/default/grub
grep GRUB_DEFAULT /etc/default/grub    # verify the quoting survived
# then on the HOST:  sudo update-grub
```

Revert later with `GRUB_DEFAULT=0` + host `update-grub` when the fixed 7.0.13 backport ships.

### A4. Post-boot verification — all must pass before the demo

```bash
uname -r                                                   # 6.12.94+deb13-amd64
cat /sys/module/nvidia_uvm/parameters/uvm_disable_hmm      # Y   (§16 UVM hardening)
cat /sys/module/nvidia_uvm/parameters/uvm_ats_mode         # 0
grep -oE 'page_poison=1|nohugevmalloc' /proc/cmdline       # both (tripwires stay ON)
nvidia-smi --query-gpu=driver_version --format=csv,noheader  # 595.71.05 (unchanged — one variable at a time)
grep -iE 'EnableS0ix|DynamicPower' /proc/driver/nvidia/params # still 0 / 0
```

### A5. Run the full demo, then measure

```bash
D=$(ls -dt ~/ATHENA/athena-diag/2*/ | head -1); echo "run: $D"
grep -c 'pagealloc: memory corruption' "$D/kmsg-full.log"   # expect 0
grep -c 'page still charged'           "$D/kmsg-full.log"   # expect 0
grep -ci 'NVRM: Xid'                   "$D/kmsg-full.log"   # expect 0
grep -c 'watchdog\|restart'            "$D/watchdog.log"    # expect 1 (armed only)
```

**Attribution note:** Phase A changes two things at once (kernel + UVM conf) — deliberate,
since the goal is demo stability, not attribution purity. For clean attribution, run the
kernel swap without A1 first.

**Two clean full-demo soaks = root cause confirmed** → then remove
`page_poison=1 nohugevmalloc` from `GRUB_CMDLINE_LINUX_DEFAULT` (host `update-grub` +
reboot) and reclaim the tripwire overhead.

---

## Phase B — NVIDIA driver update (deferred; separate variable)

Current honest state: the configured NVIDIA CUDA apt repo
(`developer.download.nvidia.com/.../debian13`) tops out at **595.71.05** on this branch —
**595.84 is not published there yet**. `nvidia-driver-pinning-595` correctly blocks the
610 feature branch. Do this only **after** Phase A's soaks.

**Check periodically** (read-only):

```bash
sudo apt update && apt-cache policy nvidia-kernel-open-dkms | head -8
```

When a `595.8x-1` line appears above `*** 595.71.05-1`, **upgrade in-place** (DKMS rebuilds
for every installed kernel; `firmware-nvidia-gsp` — the GSP firmware that matters here —
upgrades in lockstep):

```bash
sudo apt-mark unhold nvidia-open        # only if apt refuses (nvidia-open is on hold)
sudo apt install nvidia-open            # pulls the whole matched 595.8x stack
/usr/sbin/dkms status                   # expect nvidia/595.8x installed for BOTH kernels
sudo reboot
nvidia-smi --query-gpu=driver_version --format=csv,noheader   # 595.8x
sudo apt-mark hold nvidia-open          # re-hold if desired
```

**Do NOT** install the `.run` from nvidia.com over this deb-managed setup (mixing corrupts
package state), and **do NOT** jump to the 610.43 feature branch.

---

## Phase C — housekeeping

- **Kernel watch** (until `7.0.13-1~bpo13+1` lands in trixie-backports, returning you to the
  7.0 series, fixed): `cd ~/ATHENA && ./check-kernel-fix.sh` every day or two.
- **Optional auto-relaunch after a crash:**
  `cp ~/ATHENA/athena-autostart.desktop ~/.config/autostart/` — read its header first: the
  launcher's sudo prompts block truly unattended relaunch unless targeted NOPASSWD rules are
  added (`/etc/sudoers.d/athena`).
- **Off-hours memtest** (closes the RAM question for the upstream report; expected negative —
  the corruption signature is byte-writes, not bit-flips): boot memtest86+ from GRUB entry or
  USB, one full pass overnight.
- **Panic dumps are now self-harvesting:** each launch sweeps `/sys/fs/pstore` *and*
  `/var/lib/systemd/pstore` into `athena-diag/crash-<ts>/` — no more manual copies.

---

## Do NOT (falsified or superseded — see CHANGES.MD §16)

- Do not add `intel_iommu=on` (no-op — the dGPU already sits in a translated `DMA-FQ` domain)
  or `iommu.strict=1`.
- Do not treat `page_poison=1`/`nohugevmalloc` as fixes — tripwires only, until the fixed
  kernel + clean soaks.
- Do not build Xid-triggered defenses (2 of 5 fatal runs had no Xid — it's a co-symptom).
- Do not replace RAM preemptively; run the free memtest instead.
- Do not chase further pinned-memory removal (`GGML_CUDA_NO_PINNED=1` is now defaulted in the
  launcher; zero app-level pinned registrations remain).

**Critical path: A1 → A2 (verify the dkms line!) → A3 one-time boot → A4 → demo → A5 counters.**
