#!/usr/bin/env bash
# check-kernel-fix.sh — is the CVE-2026-43303-fixed kernel installable yet?
#
# The root cause of ATHENA's panics (CHANGES.MD §15/§16) is a kernel live-struct-page
# corruption in the CVE-2026-43303 class ("mm/page_alloc: clear page->private in
# free_pages_prepare", upstream ac1ea219590c). Fixed in: 6.12.94-1 (trixie-security,
# ALREADY in /boot on this box) and 7.0.13-1 (sid/testing). The running
# 7.0.12-2~bpo13+1 is vulnerable; the backports vehicle will be 7.0.13-1~bpo13+1.
# Run this daily-ish (needs network): it tells you the moment the fix ships.
set -u

echo "== running kernel =="
uname -r
echo ""
echo "== refreshing package lists (needs network + sudo) =="
sudo apt update -qq 2>&1 | tail -2
echo ""
echo "== newer kernel available? =="
apt list --upgradable 2>/dev/null | grep -E 'linux-image' \
    && echo ">>> UPGRADE AVAILABLE — check its changelog for CVE-2026-43303, then: sudo apt install linux-image-amd64 && sudo reboot" \
    || echo "no linux-image upgrade offered yet (backports still 7.0.12-2~bpo13+1)"
echo ""
echo "== candidate vs installed =="
apt-cache policy linux-image-amd64 | head -4
echo ""
echo "== fallback available TODAY (already in /boot): the FIXED trixie-security kernel =="
ls /boot/vmlinuz-6.12.94+deb13-amd64 2>/dev/null \
    && echo ">>> 6.12.94+deb13 is installed and carries the CVE fix. To test-boot it: pick it in the GRUB 'Advanced options' menu (DKMS rebuilds the NVIDIA module automatically). One clean full-demo soak on it = root-cause confirmation." \
    || echo "(6.12.94 image not present)"
