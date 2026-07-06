# ATHENA — Session Changes (2026-06-30)

Summary of all code/config changes made this session, with current on-disk state.
**No system state (GPU compute mode, CPU governor, MPS) was left modified — a reboot is clean.**

> **Platform note.** §1–§22 ran on the reference machine's original **Debian 13 (trixie)** environment — where the kernel-corruption crash-forensics saga unfolded. **§23 onward is the current Ubuntu 24.04.4 LTS redeploy, the tested and shipped platform.** The user-facing docs (README, CLAUDE.md) describe Ubuntu 24.04.4; this changelog and `STABILITY-RUNBOOK.md` deliberately keep the Debian history intact as the forensic record.

---

## 1. Documentation

### `CLAUDE.md` *(new file)*
Repo guide for future work: STT→LLM→TTS architecture, build/run commands, the
file-based TTS handoff protocol, GPU/model assumptions, and gotchas. Fact-checked
against the source.

---

## 2. Bug fixes (live)

### `launch-athena-397b.sh:29` — Qwen model path
The launcher pointed `QWEN_MODEL` at a nonexistent `models/QWEN-3.5-397B/` subdir,
so **preflight aborted every launch**. Repointed to the shards' real flat location:
```
QWEN_MODEL="/home/user/ATHENA/models/Qwen3.5-397B-A17B-UD-Q3_K_XL-00001-of-00005.gguf"
```

### `speak-daemon.sh` — stale paths
`SPEAK_FILE`/`TRIGGER_FILE`/`DONE_FILE` were missing the `/ATHENA/` path segment
(would have made the legacy fallback silently no-op). Now correct (lines 24–26).
The comment block's manual-setup examples were also refreshed from the old
`QWEN-3.5-9B` / pre-`ATHENA/` layout to the current 397B layout.

### `athena-demo-script.txt` — stale memory paths
Four references (lines 25, 56, 257, 267) corrected to `/home/user/ATHENA/athena_memory`
(the launcher's real `MEMORY_DIR`), including the "fake the time gap" `meta` command.

---

## 3. Extractor role-leak fix (live — built into whisper-talk-llama)

Fixes a ChatML role-word leak (`"NONEuser"`) in the end-of-session memory
consolidation. **Verified working in the two-session test** (clean `IMP | gist`
extraction, no leak).

### `whisper.cpp/examples/talk-llama/talk-llama.cpp`
- `mem_generate()`: added a `<|im_start|>` turn-opener stop (resolve the token,
  break on it) so the consolidation generator stops at the true turn boundary —
  mirrors the main dialogue loop. Protects extract/compact/personality passes.
- Added `strip_role_leak()` and applied it to all three consolidation outputs.

### `whisper.cpp/examples/talk-llama/athena_memory.h:516`
- `parse_extracted()` NONE sentinel hardened: `line == "NONE"` → `line.rfind("NONE", 0) == 0`
  (tolerates a leaked suffix).

> ⚠️ These two files live under `whisper.cpp/examples/talk-llama/` (copied in at
> build time). **Copy them to your portable drive** — a clean rebuild overwrites them.

---

## 4. Demo support (live)

### `launch-athena-397b.sh:777` — personality threshold
Added `--personality-reflect-every 1` to `MEMORY_ARGS` (lowers the personality-revision
threshold from 20 → 10 so a single demo session can fire a revision). Marked as a
temp/demo edit; drop the flag to revert.

---

## 5. Garble-diagnosis instrumentation (live — built into orpheus-speak)

Pure C++ (no Python). Lets the intermittent TTS garbling be root-caused from data.

### `orpheus/orpheus-speak.cpp` *(new functionality)*
- `--capture-dir DIR` — per TTS session, dumps `.tokens` (raw model output),
  `.codes` (parsed audio codes, re-decodable), `.wav` (live decoded audio), `.meta`.
- `--decode-codes FILE` — offline: re-decode a captured `.codes` to `FILE.redecode.wav`
  (use `--snac-cpu` for a contention-free reference).
- `--compare-wav A B` — prints correlation / max-diff / RMS verdict between two WAVs.
  Clean decodes correlate ~0.999 (MATCH); garble craters it (MISMATCH).
- Supporting: `read_wav()`, `write_capture()`, `decode_codes_file()`, `compare_wavs()`,
  `StreamChunk` capture buffers, capture hooks, Config/arg/usage/main wiring, `#include <ctime>`.

### `launch-athena-397b.sh` — capture wired ON
- `ATHENA_TTS_CAPTURE` (default **1**) added near `ATHENA_DIAG` (line 126).
- Builds `CAPTURE_FLAG=--capture-dir <run-dir>/tts-capture` and passes it to
  orpheus-speak (lines ~736–757). Set `ATHENA_TTS_CAPTURE=0` to disable.

### Diagnostic workflow (when garble recurs)
```bash
ls athena-diag/<run>/tts-capture/                                  # find the session
orpheus-speak --snac models/.../snac24_dynamic_fp16.onnx --snac-cpu \
              --decode-codes <sess>.codes                          # CPU reference
orpheus-speak --compare-wav <sess>.wav <sess>.codes.redecode.wav   # verdict
```
MATCH ⇒ codes/model fault; MISMATCH ⇒ on-GPU SNAC decode corrupted it under contention.

---

## 6. Tier-1 Orpheus stop-token fix — ATTEMPTED then REVERTED (net zero)

Hypothesis: a wrong stop token (`<|eot_id|>` vs the real audio-end `<custom_token_2>`)
caused runaway generation → garble/crash. A live A/B `/completion` test **refuted it**
(the model stops ~2 tokens after the audio-end marker; high token counts are
legitimate sentence lengths). The `max_tokens` cap would have clipped long sentences.
**Both changes were fully reverted.** Current `orpheus-speak.cpp` state:
`max_tokens = 2500`, `"stop":["<|eot_id|>"]` (original).

> Still open: the garble (a content-corruption, likely on-GPU SNAC decode under
> contention) and the **real S2 crash** — `llama-server` (Orpheus) segfaulted at the
> end of the S2 story (server log stops mid-task; HTTP 0 afterward; no clean exit).
> Suggested next step (not yet done): a health-watchdog/auto-restart for llama-server.

---

## 7. MPS stale-pipe fix (live — `launch-athena-397b.sh`)

Found while analyzing the two demo runs (`athena1.log` / `athena2.log`): both were
launched from the desktop icon (so `ATHENA_MPS=1` **was** set and honored), yet
**neither actually ran under MPS** — all three GPU clients time-sliced one context.

**Evidence:** both logs, line 8: `MPS: reusing daemon already at /tmp/nvidia-mps`;
`gpu-proc.log` shows **0** `nvidia-cuda-mps-server` processes and three independent
clients (`llama-server`, `orpheus-speak`, `whisper-talk-llama`) directly on the GPU;
both runs' `mps-log/` dirs are empty.

**Root cause:** `start_mps()` inferred "a daemon is up" from a bare
`[ -e "$CUDA_MPS_PIPE_DIRECTORY/control" ]` existence test. A **stale**
`/tmp/nvidia-mps/control` — left by a prior daemon that died without cleanup (a
SIGKILLed `talk-llama`, or an earlier launch that took the reuse path, which by
design "will NOT stop it on exit") — makes that test a permanent false positive:
every launch prints "reusing daemon," returns early, and never starts a real one.
Self-perpetuating, since nothing ever removes the pipe.

**Fix (start_mps):**
- Replace the `-e` test with a **liveness probe** —
  `echo get_default_active_thread_percentage | timeout 5 nvidia-cuda-mps-control`.
  Responds → reuse the LIVE daemon (unchanged). Errors/hangs (dead pipe; `timeout`
  also covers the FIFO open-for-write block) → `pkill` any lingering
  `nvidia-cuda-mps-server`/`-control`, `rm -rf` the pipe dir, and start fresh.
  A daemon started this way sets `MPS_STARTED=true`, so `cleanup()`/`stop_mps`
  properly owns and stops it on exit.
- **Moved the `mps-log/` `mkdir`** out of the early unconditional call and into the
  start-daemon branch (right before `nvidia-cuda-mps-control -d`); the early call now
  creates only the pipe dir. Net: no empty `mps-log/` is left on the reuse-live /
  MPS-off paths, so **non-empty `mps-log/` ⇒ this launcher started MPS and it ran** —
  a reliable signal.

Bash-only; no binary rebuild. `bash -n` clean. The desktop icon's `ATHENA_MPS=1`
now genuinely engages MPS.

> Relevance to the open garble (§5): both demo runs were **unintentionally no-MPS**,
> i.e. the worst-case contended, time-sliced GPU — exactly the condition under which
> the ~16:10 S1 TTS garble occurred. With MPS now actually engaging **and** the §5
> `--capture-dir` instrumentation, the MPS-vs-no-MPS A/B (with a `--compare-wav`
> MATCH/MISMATCH verdict) is finally runnable. (Note: the §5 capture postdates these
> two runs, so the 16:10 garble itself was never captured.)

---

## 8. Emotion floor retune (live — `launch-athena-397b.sh:80-83`, production profile)

From the two-run emotion scorecard: of the 5 designated checkpoints, 3 hit (the
negative-collapse beats S1-7 sad=0.993, S1-8 angry=1.000, S1-9 sad=0.923) and 2
missed (S1-3 happy, S2-6 surprised). Retuned the two floors that are actually
threshold-fixable; the floors are read from env at startup (`talk-llama.cpp:1483`),
so this is **runtime-only, no rebuild**.

**Changes (production profile only; calibration profile untouched):**
- `ATHENA_EMOTION_MIN_HAPPY` **0.95 → 0.50**. The 0.95 was unreachable — genuine
  happy peaked at **0.58** (S1-1) / **0.309** (the S1-3 delight beat), so happy
  never fired. Caveat (documented inline): 0.95 was set *above* a fear-leak
  (fear-induced happy ~0.916) and *below* true happy (0.992) from calibration; since
  demo-genuine happy (0.58) reads **below** the fear confuser, no floor cleanly
  separates them. 0.50 lets a clearly-bright take fire and accepts a fear-leak
  mislabel risk that **did not occur** in either run (all negatives read
  happy=0.000). **User-selected demo lever;** revert to 0.95 for production if upset
  speech starts tagging happy.
- `ATHENA_EMOTION_MIN_SAD` **0.50 (base) → 0.60**. Cleanly drops the one marginal
  false-positive — S1-12 "Name her" (playful/warm) read sad=0.548 → `[negative]` —
  while keeping every true negative (all observed true sads ≥0.92). Small negative-
  recall cost in the 0.50–0.60 band (none seen in the demo).

**Explicitly NOT changed (not threshold-fixable):**
- **S2-6 surprised** — raw surprised **0.004** (max **0.008** anywhere). A dead
  channel; no floor recovers it. Needs re-delivery as a *strong, standalone*
  (non-barge-in) beat, or it's an emotion2vec limit for this class. (It was also
  delivered as a barge-in, truncating the scored clip.)
- **High-confidence negative false-positives** — S1-11 (0.955), S2-4 (0.940), S2-5
  (0.989) read sad *higher* than a true-sad beat (S1-9 0.923); they interleave, so
  no threshold separates them from genuine sad. A model discrimination limit (firm/
  earnest low-energy speech ≈ sad acoustically) — the reason the collapse exists.

> Demo note: the two designated misses still need performance fixes, not just floors
> — deliver S1-3 unmistakably bright, and S2-6 as a clean standalone surprised beat.

---

## 9. MPS works now — Debian `/usr/sbin` PATH + server self-check + SNAC CPU fallback

§7 made MPS actually start for the first time, which immediately broke a run: SNAC
failed to launch (`[orpheus-speak] SNAC decoder init failed`) and the launch aborted.
Diagnosis from `athena.log` + `athena-diag/20260630-203420/`:

**Root cause — NOT the instrumentation.** Every CUDA client failed with
`ggml_cuda_init: ... MPS client failed to connect to the MPS control daemon or the
MPS server` (llama-server too, which has no instrumentation). `mps-log/control.log`
showed the control daemon up but the per-user server dying instantly —
`Starting new server N` → `Server N exited with status 1`, ~12×, `server.log` empty.
The server binary is `/usr/sbin/nvidia-cuda-mps-server`, and **Debian omits
`/usr/sbin` from the user PATH**, so the control daemon (in `/usr/bin`) couldn't exec
it. Confirmed by direct test: adding `/usr/sbin` to PATH → server spawns (live PID,
`server.log` populated). Same hardware worked under Ubuntu because Ubuntu keeps
`/usr/sbin` on PATH. Consequence: llama-server fell back to CPU (survived); the
SNAC ONNX CUDA-EP `CreateSession` failed and orpheus-speak **exited** (fatal). The
capture instrumentation is not in the SNAC init path — fully exonerated.

**Fixes:**
- `launch-athena-397b.sh` `start_mps` — prepend `/usr/sbin` to PATH before starting
  the daemon so it can exec the server. **This is the fix that makes MPS work here.**
- `launch-athena-397b.sh` `start_mps` — after `-d`, force a server
  (`start_server -uid $(id -u)`) and confirm via `get_server_list`; if it won't
  spawn, quit MPS, `unset CUDA_MPS_*`, continue WITHOUT MPS. A broken MPS now
  degrades to the known-good direct-GPU path instead of bricking every CUDA client.
- `orpheus/orpheus-speak.cpp` `SnacDecoder::init` — the CUDA `CreateSession` status
  was discarded ("init failed", no reason). Now prints the ORT error and, if the
  CUDA EP was active, rebuilds options without CUDA and retries on **CPU** (clears
  `has_cuda_ep` so the GPU-arena RunConfig is skipped). SNAC survives a CUDA/MPS
  hiccup instead of killing the daemon. **Rebuilt.**

Verified: PATH + self-check integration test brings daemon+server up (live PID,
877-byte `server.log`) and detects failure→fallback; orpheus-speak rebuilt clean.

---

## 10. Stability hardening after the 20260630-230402 GSP-fault crash

First full MPS run (§9 fixes held: `MPS: daemon + server up`, SNAC on CUDA, 8
utterances, 0 stalls, **audio 100% clean** — the original garble did not recur).
It ended in a crash. Empirical diagnosis (multi-agent analysis + adversarial
verify + online research; all evidence in `athena-diag/20260630-230402/`):

**Fault chain (proven):** `23:18:05` `xid.log` — `NVRM: GspMsgQueueReceiveStatus:
Bad checksum` (GSP firmware RPC-ring corruption) → `23:19:59` llama-server (Orpheus,
PID 3375) hit a fatal GPU fault mid-`llama_decode` (3 slots decoding, `-np 4`) →
**MPS escalated it**: single shared context on one GPU = one fault domain, so
`mps-log/server.log` shows *"devices will be reset: 0"* and a sticky-error poison of
**all three** clients → `23:20:52` talk-llama (mostly CPU via `--cpu-moe`, idle-on-GPU
at the fault) hit the sticky error at its next GPU sync — the per-turn `state_write` —
and SIGABRT'd → the foreground death tripped the ERR trap; teardown then force-
SIGKILLed orpheus-speak on a live context and the MPS server wedged.

**Root cause = a known, unresolved Blackwell open-driver GSP firmware bug**
(open-gpu-kernel-modules #1064: faults on runtime-PM D3↔D0 transitions; #1111: GSP
halt under sustained llama.cpp inference — ATHENA's exact workload; CachyOS report on
595.71.05). GSP cannot be disabled on Blackwell. `ev:power-clock` ruled out thermal
(79 °C peak, HW-thermal/brake counters 0 µs); the GPU sat pinned at the ~99.6 W SW
power cap (78–90 % of loaded samples). MPS did **not** cause the fault (single-GPU MPS
has no per-client fault isolation *by design*) and is kept — it is exactly the
"cooperative single application" MPS is recommended for; dropping it would serialize
TTS/STT and cost latency without preventing a device-level GSP fault.

Strategy: **reduce the trigger + survive it gracefully**, no performance cost.

**L1 · Teardown + MPS recovery** (`launch-athena-397b.sh`, shutdown-only):
- `cleanup()` now does `trap - ERR` — `set +e` never silenced the ERR trap (it is
  inherited via `set -E`), which is why teardown printed spurious `ERROR at line
  382/387` (bounded `quit` timeout=124, `pkill` no-match=1).
- `stop_mps` uses `quit -t 5` (daemon force-stops a server stuck on a SIGKILLed
  client's worker thread; bare `quit` hung with exponential backoff — the wedge) and,
  if still up, force-kills + clears the pipe dir. Fixed the **latent no-op**: the old
  `pkill -x nvidia-cuda-mps-control` never matched (name >15 chars → `comm` truncates);
  now `pkill -f`. Same `-x`→`-f` fix in the `start_mps` stale-pipe cleanup.
- New `mps_terminate_client` — `_stop_pid`'s force branch now issues the documented
  `terminate_client` to drain a client's context **before** SIGKILL (avoids the
  "corrupt kernel page state" path).

**C1 · llama-config** (`launch-athena-397b.sh`, perf-neutral/positive):
- Dropped `GGML_CUDA_GRAPH_OPT=1` from **both** llama-server and talk-llama — it is an
  experimental multi-stream/buffer-reuse path (PR #16991, off-by-default, ~1–9 % TG at
  batch-1) with an author-flagged buffer-reuse race (the "unspecified launch failure"
  class); base CUDA graphs + fusion stay on. Bench-neutral (159.7 off vs 158.8 on).
- Orpheus `-np 4 → 2`: single-user gets zero latency benefit from >1 slots; per-stream
  real-time margin *grows* 1.07×→1.49× while peak concurrent decode (shared MPS/GSP
  stream count) halves. The fault hit with 3 slots live.

**L2 · Brain-resilience** (`whisper.cpp/examples/talk-llama/talk-llama.cpp`, **rebuilt**):
- New `gpu_context_healthy()` — resolves `cudaPeekAtLastError`/`cudaDeviceSynchronize`
  from the already-loaded libcudart via `dlsym` (no CUDA header/link change; CPU build
  → "healthy"). These *return* the sticky MPS error instead of aborting.
- Gated the first-reachable GPU ops so a poisoned context ends the session **cleanly**
  (exit 0, cross-session memory timestamp preserved) instead of SIGABRT: `whisper_full`
  (STT), `snapshot_take`/`snapshot_restore` (the proven `state_write` crash site),
  `decode_tokens`, and the response-generation `llama_decode`. Master `g_gpu_lost` flag
  + a loop-top break route to a teardown tail that skips GPU consolidation/frees
  (`llama_free`→`cudaFree` would re-abort) and `_exit(0)`. Cannot keep serving after a
  single-GPU fault (client must exit), but no more abort / ERR-trap / lost turn.

**D2 · Clock/power** (`launch-athena-397b.sh`; A/B-experimental, reversible):
- Power limit now defaults to board **max (110 W)** as attempt+readback (verified
  `power.max_limit=110`, `default=95`): capped 78–90 % of loaded time at only 79 °C, so
  raising the ceiling is perf-positive if honored, no-op if nvidia-powerd overrides it.
- Memclk default → **floor mode `[13000, 14001]`** (was unlocked). The driver already
  runs the top bin (13801) under load regardless, so this changes nothing at load —
  it only stops the idle drop to 9001 between chunks (removes the churn, recovers
  ~3–4 %). Deliberately NOT a rigid pin to the never-driver-selected 14001, which is
  the Xid-69 suspect. Revert with `ATHENA_LOCK_MEMCLK=0`.

**D1 · Driver GSP mitigation** (staged, **needs reboot — user applies**):
`nvidia-athena-gsp-stability.conf` + `apply-gsp-stability.sh` disable Runtime-D3 /
S0ix (`NVreg_DynamicPowerManagement=0x00`, `NVreg_EnableS0ixPowerManagement=0`,
`NVreg_PreserveVideoMemoryAllocations=1`) so the GPU stays in D0 — removing the
D3↔D0 transitions #1064 ties to the GSP corruption. Compute cost: none (idle power
only). Not a vendor-confirmed cure; residual faults still captured in `xid.log`.

The watchdog/auto-restart supervisor (restart the GPU-client set on a fault so the
demo continues) was designed and verified but **deferred** — it is a larger launcher
refactor (backgrounds talk-llama) held for a dedicated pass.

---

## 11. Post-`20260701-074343` crash: S0ix-override fix + panic-capture + power-wall backoff

Second full-MPS run ended in **Xid 69 + a kernel panic** (hard reboot). Multi-agent
forensics + adversarial verification (all evidence in `athena-diag/20260701-074343/`):

**Fault chain (proven):** GPU pinned at the **SW Power Cap** (`gpu-throttle.log` reason
`0x4`) continuously ~5 min at ~94 W / 79 °C → `07:52:59` **Xid 69** graphics/compute
engine class error; `llama-server` (Orpheus, MPS PID 2686) hit `unspecified launch
failure` mid-`llama_decode` → `ggml_abort` → **MPS reset device 0 and poisoned all 3
clients** → app limped ~42 s draining audio → system hard-locked ~`07:54:07` → **kernel
panic** (backtrace **not captured**). Distinct from the 230402 `GspMsgQueueReceiveStatus:
Bad checksum` signature — this is the chronic **Xid 69 at the ~95 W power wall**.

**Root-contributor found (the headline): the §10 D1 S0ix mitigation was silently
defeated.** `apply-gsp-stability.sh` installed `nvidia-athena-gsp-stability.conf`
(`NVreg_EnableS0ixPowerManagement=0`), but the distro's `/etc/modprobe.d/nvidia.conf`
sets `=1` and — because modprobe reads C-locale order and kernel params are
**last-assignment-wins** — `nvidia.conf` sorted *after* our file and won. **S0ix ran
ENABLED** this boot (confirmed in `power-state.log`), so the mitigation's headline
directive never took effect. Only `NVreg_DynamicPowerManagement=0x00` (RTD3-off, no
conflicting line) actually applied.

**Not culpable (adversarially cleared):** memclk never left 13801 (never the 14001
suspect); the D2 power-raise did not create a hotter regime (98.3 W ≈ prior 99.6 W);
`GGML_CUDA_GRAPH_OPT`-off / `-np 2` are protective; `PreserveVideoMemoryAllocations=1`
predates this work (set by `nvidia.conf` since April), so it is not a new delta.

**Fixes (live/staged):**

**F1 · S0ix override** — renamed `nvidia-athena-gsp-stability.conf` →
**`zzz-nvidia-athena-gsp-stability.conf`** so it sorts *after* `nvidia.conf` and its
`S0ix=0` wins. `apply-gsp-stability.sh` now removes the legacy copy, prints a
per-param **precedence report** (warns if any modprobe.d file overrides us), and its
verify hint checks the S0ix param too. *User re-runs `apply-gsp-stability.sh` + reboot.*

**F2 · Panic capture** (so the next hard-lock is diagnosable — the current diag lost
the backtrace twice: the `dmesg -w | grep "NVRM|Xid|nvidia"` filter discarded the oops
body, and no `fsync` lost ~68 s in the page cache):
- `athena-kmsg-logger.cpp` *(new, C++/POSIX, no deps)* — reads `/dev/kmsg`, appends
  every record **unfiltered** to `kmsg-full.log` + the NVRM/Xid/GSP subset to `xid.log`,
  `fdatasync()` per line. Built by `apply-crash-capture.sh`; used by the monitor.
- `athena-gpu-monitor.sh` §4 — prefers `athena-kmsg-logger`; falls back to an
  **unfiltered** bash follow (keeps the backtrace) if the binary is absent.
- `athena-crash-capture.sysctl.conf` → `/etc/sysctl.d/99-athena-crash.conf`
  (`panic_on_oops=1`, `hardlockup_panic=1`, `panic=30`) escalates a silent wedge to a
  real `panic()` → dumps RIP+trace into **efi-pstore** (already armed; survives cold
  reboot). `softlockup_panic=0` (avoid false-fire under mlock).
- `zzz-athena-gsp-diag.conf` — `NVreg_EnableGpuFirmwareLogs=1` routes GSP firmware
  logs into dmesg for the #1111 report (applies next boot).
- `launch-athena-397b.sh` preflight — harvests `/sys/fs/pstore` + `nvidia-bug-report.sh`
  into `athena-diag/crash-<ts>/` on the next launch, then clears pstore.
- `apply-crash-capture.sh` *(new)* — builds the logger, installs+applies the sysctls
  (via `/usr/sbin/sysctl` — Debian omits it from PATH), installs the GSP-diag conf,
  verifies the efi-pstore backend. `--revert` supported.

**F3 · Power-wall backoff** (`launch-athena-397b.sh` `setup_gpu_performance`, reversible)
— the D2 "raise to board max" default is **reversed**: since the GPU sat pinned on the
SW power cap and that regime correlates with the chronic Xid 69, the default now caps
**~10 W below the board default** (95→**85 W** here) to pull DVFS off the wall. Trade:
a few % TTS throughput (absorbed by the ~1.5× real-time margin at `-np 2`). Revert with
`ATHENA_GPU_POWER_MAX=1` (old raise-to-max) or `ATHENA_GPU_POWER_LIMIT=<W>`.

## 12. Post-`20260701-123219`: watchdog + driver A/B + revert the container no-op levers

Third full run with §11 mitigations live (S0ix `Disabled` confirmed in `/proc/driver/nvidia/params`).
**No Xid 69, no GSP fault, no MPS poison, no kernel panic** — through the *same* ~95 W
power wall that preceded the 074343 crash. It ran the whole session and shut down
clean **with memory preserved**. Multi-agent forensics + adversarial verify (evidence
in `athena-diag/20260701-123219/`; crash-capture from §11 F2 caught the backtraces):

**What failed (both non-fatal, both captured):**
- `12:40:17` **`BUG: Bad page state … page still charged to cgroup`** (victim `nvidia-smi`) —
  a kernel memcg/page-accounting corruption; `bad_page()` discards the page, kernel
  tainted, **continues**. Call trace has **zero NVIDIA frames** (generic buddy allocator),
  so attribution to the driver is inferential (V5/V7). Same signature the shutdown
  `cleanup()` ordering already guards against — but this one fired mid-run.
- `12:47:52` **`llama-server[2736] general protection fault … in libcuda.so.595.71.05`** —
  a **userspace** GPF (not a device Xid) 13 ms after a new task following 67 s idle. MPS
  logged a **normal client exit** — no `fatal GPU error`, no `devices will be reset`, no
  sticky-poison; the brain (2886) and orpheus-speak (2822) kept running (V1). "TTS fell
  down" = orpheus-speak retried a dead `:8080` (`HTTP 0, retry`) for captures 004-006.

**Key insight (V3/V4, verified):** every `nvidia-smi` clock/power *set* is a **no-op in
this passthrough container** — the boot NVRM log shows the SBIOS power handshake is dead
(`PlatformRequestHandler failed to get target temp/platform power mode from SBIOS`), so
`nvidia-powerd` owns the ceiling and `-pl 85` was *rejected* (readback 105.35 W) and the
memclk floor didn't hold (9001 MHz in 45 % of samples). **Stability came entirely from
S0ix-off / RTD3-off D0-pinning (zero compute cost) — not from any clock/power tuning.**

**Changes (all Bash/C++, zero-perf-cost):**

**W1 · llama-server watchdog + auto-restart** (`launch-athena-397b.sh`) — the recoverable
userspace-death case (this run) now self-heals in seconds. `start_llama_server()` factored
out; `/dev/shm` pidfile is the single source of truth (survives restarts); `llama_watchdog()`
detects death (`kill -0` fast path + bounded `curl /health` for wedges), enforces a rolling
cap (`WD_MAX_RESTARTS=5`/`WD_WINDOW=300`), and respawns the byte-identical command. `cleanup()`
stops it FIRST (sentinel + TERM) and re-syncs `PID_LLAMA` from the pidfile. `set -e`/ERR-safe
(backgrounded). MPS auto-reconnects the fresh client; orpheus-speak auto-heals on its next
request. Scope: cannot save a device Xid (needs full-stack relaunch) — it gives up at the cap.
`ATHENA_WATCHDOG=0` disables. **Functionally tested** (mock-backend kill→restart→sentinel).

**W2 · idle keepalive** (`launch-athena-397b.sh`) — `llama_keepalive()` pokes `:8080` with a
1-token completion every 30 s so the CUDA context/graphs never go fully cold (the GPF hit a
67 s-idle re-entry). ~8 ms GPU/30 s, off the latency path. `ATHENA_KEEPALIVE=0` disables.

**W3 · scoped core dumps** (`launch-athena-397b.sh`) — `core_pattern` → `<run>/core.%e.%p`;
llama-server + orpheus-speak launch in `( ulimit -c unlimited; exec … )` (exec preserves `$!`),
**talk-llama forced `ulimit -c 0`** (its ~161 GiB mlock'd RSS must never core). Restored on exit.
`ATHENA_COREDUMP=0` disables. Next libcuda GPF → `gdb <bin> <core> -ex bt`.

**W4 · revert the container no-op levers** (truth-in-logging, `launch-athena-397b.sh`):
- **power:** default no longer issues `nvidia-smi -pl` at all (§11 F3's 85 W was rejected here
  and would be a perf tax where honored). `ATHENA_GPU_POWER_LIMIT=<W>` / `ATHENA_GPU_POWER_MAX=1`
  stay opt-in.
- **memclk:** default `ATHENA_LOCK_MEMCLK=0` (the `[13000,max]` floor didn't hold — 9001 MHz at
  idle anyway). `=1`/`=<floor>` stay opt-in.
- **sampler cadence** `500→1000 ms` (`athena-gpu-monitor.sh` + launcher call) — halves the
  highest-rate `nvidia-smi` spawner (the bad-page canary), near-zero diagnostic loss.

**W5 · driver A/B `NVreg_PreserveVideoMemoryAllocations=1→0`** (`zzz-nvidia-athena-gsp-stability.conf`)
— with RTD3/S0ix off it has ~zero runtime benefit and its host-page save/restore path is the
leading suspect for the bad-page memcg leak. **Staged — user re-runs `apply-gsp-stability.sh` +
reboot**, then verify `grep -i PreserveVideoMemory /proc/driver/nvidia/params` → 0.

**Staged (user, host GRUB — temporary diagnostic):** add `page_poison=1` to
`GRUB_CMDLINE_LINUX_DEFAULT` (`page_owner`/`DEBUG_VM` are compiled out of this kernel;
`CONFIG_PAGE_POISONING=y` is set) to catch the next mm corruption earlier; remove after diagnosis.
Note: `NVreg_EnableGpuFirmwareLogs=1` (§11) is a confirmed **no-op** — the `gsp_log_*.bin` decoder
isn't in the public 595.71.05 firmware; rely on `athena-kmsg-logger` + `nvidia-bug-report.sh`.

**Kept (validated this run):** MPS (positively proved fault-containment — a userspace GPF stayed
contained), `-np 2`, `GGML_CUDA_GRAPH_OPT` off, and the §11 F2 panic sysctls (correctly did NOT
over-escalate the non-fatal bad-page / userspace GPF).

## 13. Post-`20260701-144239`: the recurring bad-page is a KERNEL UAF, not a GPU/ATHENA bug

Fourth run with the §11/§12 stack live (S0ix off, watchdog, PreserveVMA=0, `page_poison=1`).
**Stable: zero Xid / GSP / libcuda GPF / kernel panic; TTS ran to the end; the watchdog
armed but never fired; user-interrupted (no crash).** But `page_poison=1` exposed a real,
ongoing memory corruption. Deep multi-agent forensics + adversarial verify + online research
(evidence in `athena-diag/20260701-144239/kmsg-full.log`):

**Root cause (RESEARCH-BACKED): a Linux kernel memcg / huge-vmalloc `split_page` /
`page->private` use-after-free**, *triggered* by NVIDIA UVM's large `kvmalloc→vmalloc`
allocations — **not** an NVIDIA-UVM code bug and **not** ATHENA's C++. The identical
`page still charged to cgroup` signature reproduces with a DIFFERENT out-of-tree driver
(OpenZFS, [zfs#16482](https://github.com/openzfs/zfs/issues/16482)) and historically with
`nvidia_uvm` ([dragon#8](https://github.com/pakmarkthub/dragon/issues/8)) → kernel-side,
driver-agnostic. Most relevant fix: [CVE-2026-43303](https://nvd.nist.gov/vuln/detail/CVE-2026-43303)
("clear `page->private` in `free_pages_prepare()`"; fixed <6.18.16 / <6.19.6) + commit
`3b8000ae185c` (huge-vmalloc split, ~v5.18).

**Empirical facts (adversarially verified):**
- **34 corruption events / 8.5 min** (31 `page_poison` write-after-free + 3 memcg `bad_page`).
  Non-fatal; ~16 KiB/min leaked (~9.6 MB over 10 h — harmless on memory grounds); steady.
- **`page_poison=1` is the ONLY containment.** Each poisoned page is discarded, so the
  write-after-free never reaches a new owner. Without it the UAF is **silent + uncontained**
  → the likely substrate under the 074343 panic / 123219 GPF (both ran without page_poison).
  **Do NOT remove `page_poison` until a run confirms 0 events.**
- **The corruptor is unidentifiable from the logs** (poison/bad_page fire at *allocation* — every
  trace is the victim that drew the polluted page). Non-GPU victims (khugepaged, the screen
  recorder) prove it's the **global** page freelist. Strong-inference (not proof): the freer is
  `nvidia_uvm`'s va-space path.
- **Trigger = per-sample `nvidia-smi` UVM churn.** Confirmed by fd-inspection that even
  `nvidia-smi --query-gpu=clocks.sm` opens `/dev/nvidia-uvm` (2 fds) and does `UVM_INITIALIZE`
  (`uvm_va_space_create`, `RSI=0x30000001`); the monitor spawned ~3/sec ≈ ~2500 create/destroy
  cycles/run.
- **`PreserveVideoMemoryAllocations=0` REFUTED** (§12 W5): the memcg leak recurred with it off.
- **ATHENA's C++ is churn-neutral:** all 3 GPU procs hold ONE CUDA context/UVM va-space for the
  run; orpheus-speak forks `aplay` **once per turn** (`PcmSink`), not per chunk.

**Fixes (implemented, zero-perf-cost):**

**13a · Persistent `nvidia-smi` in `athena-gpu-monitor.sh`** — the 3 spawn-per-sample loops
(gpu.csv / gpu-proc / throttle) are now ONE long-lived `nvidia-smi … --loop-ms=…` each: UVM
va-space created **once** per stream instead of per tick (**~500× fewer** create/destroy cycles,
lower CPU, byte-identical telemetry). A supervisor subshell holds the real nvidia-smi PID via a
FIFO (clean teardown, no orphan) and auto-restarts on death; throttle field is probed once. The
`nvidia-powerd` liveness loop (§5) and the kmsg logger (§4) are unchanged (they touch no UVM).

**13b · `orpheus-speak.cpp` `fork()+execvp` → `posix_spawnp()`** (both player sites: `PcmSink::start`
active per-turn, `play_audio` dormant batch) — glibc `posix_spawn` uses `clone(CLONE_VM|CLONE_VFORK)`,
so it does **not** `dup_mmap`/`pte_alloc`-copy this process's CUDA/ONNX address space, removing
orpheus-speak from the exact fork path that took a `bad_page`. Byte-identical behaviour (pipe→stdin,
stderr→/dev/null, same `waitpid` drain contract, verified). **Rebuilt.**

**User-side (staged — reboot/kernel/experiment):**
- **`nohugevmalloc` boot param** — cheapest decisive test: forces small-page vmalloc, sidesteps the
  huge-vmalloc/`split_page` bug class. If the corruption vanishes → root cause confirmed + permanent
  zero-cost workaround.
- **Check/patch the kernel for CVE-2026-43303** (`7.0.12+deb13` may already carry it).
- **Keep `page_poison=1`** until a confirmation run shows `grep -c "pagealloc: memory corruption"` = 0
  and `"page still charged to cgroup"` = 0; only then remove it (it's a temporary diagnostic).
- **Isolation experiment** (proves the trigger, and is the upstream minimal repro): on an idle box,
  `while :; do nvidia-smi >/dev/null; done` with `page_poison=1`, watch `dmesg` for the two signatures
  with NO ATHENA/MPS/TTS. **File upstream** (open-gpu-kernel-modules + kernel mm) with both signatures.
- Optional: `nvidia-persistenced --uvm-persistence-mode` (keeps UVM state resident across process exits).

## 14. Post-`20260701-163723`: kernel corruption CONFIRMED FIXED; "peak Igor:" antiprompt truncation fixed

Fifth run, §13a/13b live. **The §13 fix is confirmed: 0 corruption events** (was 34) —
`pagealloc: memory corruption` 0/31, `page still charged to cgroup` 0/3, BAD_PAGE taint 0,
zero Xid/GSP/GPF/panic, watchdog armed-only, exactly 3 persistent nvidia-smi. The kernel-UAF
trigger is eliminated by code alone; the `nohugevmalloc`/kernel-patch steps were not needed.
`page_poison=1` can now be REMOVED from GRUB after one more confirming run (user's call).

**New failure found + fixed: mid-sentence antiprompt truncation.** Athena's reply cut off at
"…This is peak" and every re-ask replayed the same text verbatim and cut at the same word.
Empirical chain (all in `athena-diag/20260701-163723/` + athena.log):
- tts-capture metas: sessions 002/003/004 all `COMPLETE` with **identical** chunk structure
  (61/19/223/12 chars) — orpheus-speak played everything it was given and saw a clean end →
  **TTS/barge-in/llama-server all exonerated**; the *brain* stopped generating.
- athena.log:3142: `ctx_tail=<<< take notes. This is peak Igor:>>>` — the model generated the
  sentence "…This is peak **Igor:** …" ("peak Igor" + colon), and the dialogue **antiprompt —
  the unanchored string `"Igor:"`** (`person + chat_symb`, talk-llama.cpp) — matched
  MID-SENTENCE, truncating the turn and leaving a user tag at the context tail
  (`ends_with_user_tag=1`).
- The re-ask repetition: the truncated turn stayed in context; with `--repeat-penalty 1.0`
  (disabled) the model verbatim copy-replayed its own previous turn (induction attractor) and
  hit the same "Igor:" again → deterministic loop (identical ctx_tail at athena.log:3372).
- **Latent bug, not a §13 regression**: the unanchored antiprompt predates everything; the
  trigger was purely content (the model addressing Igor by name + colon for the first time).

**Fix (`whisper.cpp/examples/talk-llama/talk-llama.cpp`, rebuilt):**
- `antiprompts[0]`: `person + chat_symb` → **`"\n" + person + chat_symb`** (newline-anchored).
  Every REAL turn boundary is written with a preceding newline (few-shot prompt lines, all
  `"\n" + person + chat_symb` injections, and the healthy-turn evidence `…digging?\nIgor:` at
  athena.log:2957) — so anchored matching keeps turn detection intact while making mid-sentence
  name+colon mentions inert. The bot tag (`"\nAthena:"`) was ALWAYS anchored this way; the fix
  restores symmetry.
- **Antiprompt detection HOISTED above the sentence flusher** (adversarial patch review
  caught a pre-existing hole and reproduced it: the flusher ran *before* the match, and the
  first-flush clause-boundary fallback splits at `",;:"` — *including a tag's own colon* — so
  an 80+ char unpunctuated opening could flush a spoken "Igor:" before the strip ran). Now a
  per-token pre-check sets `antiprompt_hit` *before* the flush, the flush is gated on it, and
  a consumer block after the flush applies done + strip. A completed tag can never reach TTS.
- Strip refined: anchored replace first; a **bare**-tag tail-strip only when the anchored form
  wasn't found in that buffer (newline already flushed) — a legitimate name mention immediately
  before a real turn tag is never double-stripped, and mid-sentence mentions stay exact in the
  transcript/memory.
- Verified: adversarial review (V1 root cause SUPPORTED-high; V3 latent-not-regression
  SUPPORTED-high; upstream research confirms unanchored matching is inherited from llama.cpp
  `main.cpp`/talk-llama and newline-anchored stops are the standard remedy — same bug class as
  SillyTavern #3302). Offline pipeline test: old-order leak reproduced / new-order no-flush /
  no remnant / legit tail mention preserved / bare strip in flushed-newline state — all pass.
- Accepted tradeoff (documented): a model-emitted *space*-preceded turn-yield (`" Igor:"`)
  would no longer stop generation — backstopped by `"\nAthena:"`, `<|im_start|>`, EOS, and
  `n_gen_max`; never observed in any log.

> ⚠️ Per §3: `talk-llama.cpp` lives in the build tree — **re-copy it to the portable drive**
> (a clean rebuild overwrites it).

## 15. Post-`20260701-174424` panic: the pstore dumps unify EVERY crash — one kernel-corruption disease

Run 174424 (first boot WITHOUT `page_poison`) panicked at 17:54 with a 30 s auto-reboot (§11
`kernel.panic=30` working as designed). The §12/13 stacks held (0 mm events logged, watchdog
armed-only, 3 persistent nvidia-smi). Recovered the **systemd-pstore panic archives** — five
kernel dumps including every previously "lost" panic — and they rewrite the whole story:

**All crashes are ONE disease: kernel memory corruption** (the nvidia_uvm-triggered
huge-vmalloc/memcg UAF class, §13/CVE-2026-43303 family), seen as corrupted lruvec/memcg stats
(`lruvec_stat_mod_folio` GPF in THREE crashes), corrupted page-cache XArray
(`filemap_get_folios_tag`, today), corrupted **PTEs in CUDA driver threads**
(`Bad page map … cuda00001800007`, Jun 30), RCU state, rss counters. **The Xid 69s are best
read as the GPU-side symptom of the same corruption** (GPU walking corrupted UVM/host
structures → engine Class Error), not an independent GPU/GSP hardware fault:
- 074343: `Bad page state` in a CUDA thread **13.5 s BEFORE** the Xid; panic = GPF in
  interrupt on swapper.
- **Hidden panic discovered:** run 144239 (page_poison ON, 34 contained events) ALSO panicked
  at its teardown (~15:00:38, auto-reboot) — GPF in `lruvec_stat_mod_folio`, **no Xid at all**.
- Today: Xid 69 → poisoned llama-server exits → exit-path ext4 writeback walks a corrupted
  XArray (non-canonical `ff92…`) → GPF → panic_on_oops.
- Timing note: today's Xid landed within ~0.1 s of the (then-unconditional) §12 keepalive
  poke while the brain decoded at the 94 W SW-power-cap — concurrency at the wall remains a
  plausible *stressor*, but the corruption is the disease.

**Consequences / fixes:**
- **§13's churn fix lowered the corruption RATE (163723 fully clean) but did NOT remove the
  kernel bug** — the CUDA clients' own driver threads exercise the same path. "Steps 3/4 not
  needed" (§14) was WRONG.
- **F1 (user, GRUB): `page_poison=1` RESTORED + `nohugevmalloc` added** — poison contains
  realloc-time reuse (real protective work; its removal preceded today's cascade), and
  `nohugevmalloc` sidesteps the huge-vmalloc/`split_page` class entirely (near-zero cost —
  small-page vmalloc only affects driver alloc TLB, not inference). End state after a clean
  confirmation run: keep `nohugevmalloc`, drop `page_poison` (wins the perf back).
- **F2: kernel update check** (`apt list --upgradable | grep linux-image`) for a 7.0.x
  carrying the CVE-2026-43303 `free_pages_prepare` fix — the true fix.
- **F3 (implemented): keepalive IDLE-GATED** (`launch-athena-397b.sh`) — pokes only when
  gpu.csv utilization < `ATHENA_KEEPALIVE_MAX_UTIL` (15%); a busy GPU has a warm context, so
  the poke bought nothing and injected a second MPS client's decode at peak load.
- **F4 (implemented): pstore harvest fixed** — `systemd-pstore.service` RACES the §11
  preflight and moves records to `/var/lib/systemd/pstore/` at boot before we looked (why no
  `crash-<ts>/` ever appeared). The launcher now harvests BOTH locations with a marker file;
  panic backtraces land in `crash-<ts>/systemd-pstore/<epoch>/dmesg.txt`.
- **F5: upstream report now fully armed** — five dumps, the `while :; do nvidia-smi; done`
  repro, page_poison byte-dumps, and the corrupted-PTE evidence tie nvidia_uvm to the kernel
  mm bug for both the kernel-mm list and open-gpu-kernel-modules.

**Research corroboration (verified online):**
- Xid-69 payload decoded: Class `0000cec0` = **BLACKWELL_COMPUTE_B** (sm_120 CUDA compute class,
  `clcec0.h`); **ErrorCode 0x4 ≈ INVALID_VALUE / 0xc ≈ INVALID_BITFIELD** (nouveau GR class-error
  enum, stable ~15 yrs) — i.e. **the GPU front-end was fed an invalid method value: data
  corruption reaching the command stream**, exactly the GPU-side symptom the pstore dumps imply.
  NVIDIA buckets Xid 69 as app/driver-side ("RESTART_APP"), not hardware.
- open-gpu-kernel-modules **#1111 still open** (no assignee, no fix version); NVIDIA staff
  acknowledge a "known cluster" of Blackwell GSP patterns; an RTX PRO 5000 **Laptop** sibling
  report (kernel-mode access violations under sustained CUDA, 595.79/.97) is unresolved. No
  public MPS+power-cap→Xid-69 report exists — this box may be first.
- **Lenovo BIOS 1.30 (N4FET49W) is CURRENT** for P16 Gen 3 — no newer BIOS; changelog has no
  GPU/thermal fixes → nothing to chase there. The boot `PlatformRequestHandler` asserts are a
  known cross-vendor 595.x-laptop `_DSM` handshake failure (4 open GH issues); consequence is
  VBIOS-default limits / degraded Dynamic Boost (a perf nuance), NOT the crash cause; the
  asserts come from the kernel module, so stopping nvidia-powerd would not silence them.
- Keepalive-phase verification: cadence 30.021 s ± 0.01, next poke due 17:54:16.743, Xid at
  ≤17:54:16.7685 — **0-45 ms gap** (~0.2-0.3 % null probability), BUT 3 of 4 peak-load pokes did
  NOT fault → the poke is at most a **probabilistic stressor**; idle-gating (F3) stays justified,
  causation unproven. Also corrected: power was at the wall only for the final ~11 s (17:53 had
  an idle trough), and llama-server was mid-SIGABRT (`tgkill` sig 6 — the CUDA sticky-error
  abort) when its exit-path writeback GPF'd on the corrupted XArray.

## 16. Post-`20260702-083108` (6th crash): DMA hypothesis REFUTED; root cause narrowed to a
## kernel live-struct-page corruption (CVE-2026-43303 class); exposure-shrinking fixes staged

Sixth crash, Prompt-#5 story, **with `page_poison=1 nohugevmalloc` verified on the cmdline** —
kernel panic + 30 s auto-reboot at 08:47:00, ~16 min in, mid-story at the ~94 W cap, **no Xid**.
Full ultracode pass (3 adversarial skeptics + ggml-CUDA code dive + driver/IOMMU/kernel research
+ synthesis; pstore dumps for ALL six crashes now harvested and cross-tabulated):

**Root cause (PROVEN class, V1-high/V2-medium): in-place SINGLE-BYTE corruption of live kernel
page metadata.** 4 of 5 terminal oopses GPF at `lruvec_stat_mod_folio+0x3f/+0x54` dereferencing a
garbage memcg pointer loaded from a live folio's `memcg_data` (instruction decode + fault-address
arithmetic proven: today RAX `ffff…`→`ffa0…`, ONE byte scribbled); the 5th is the same class via a
corrupted page-cache XArray. Victims arbitrary (swapper-in-interrupt, CUDA thread, nvidia-smi,
`date` mapping libc). Falsified TODAY: huge-vmalloc as sole variant; alloc-time hygiene as
sufficient (poison caught 2 free-window scribbles — one in the NVIDIA `rpcRmApiControl_GSP`
kmalloc path — but the kill shot hit never-freed vmemmap).

**REFUTED (V3-high): errant device DMA — and my earlier "IOMMU is off" claim was WRONG.** The
dGPU sat in a translated `DMA-FQ` domain all along (Debian `CONFIG_INTEL_IOMMU_DEFAULT_ON_INTGPU_OFF=y`;
the "Default domain type: Translated" line appears late in boot), with ZERO DMAR faults; the
corrupted regions (vmemmap, PTEs, slab) are never IOVA-mapped. **The writer is CPU-side kernel
code.** Leading hypothesis: CVE-2026-43303-class stale-struct-page bug exercised by driver/UVM
kernel paths under load. **Xid 69 is a co-symptom** (2 of 5 fatal runs had none; the GPU consumes
the same corrupted host memory → INVALID_VALUE method errors). Robust correlate = UVM/driver
activity, NOT the power wall (one poison catch at idle 22 W).

**Fixes (staged/implemented):**
- **16.1 UVM hardening** (`athena-uvm-hardening.conf`, installed by `apply-gsp-stability.sh`):
  `uvm_disable_hmm=1 uvm_ats_mode=0` — drops the ZONE_DEVICE struct-page array + mmu_notifier/
  migrate_vma/devmem-folio-free hooks (the folio-lifecycle code neighboring the corruption) and
  the Blackwell ATS prefetcher hazard. **Zero cost** (cudaMalloc-only stack; MPS clients can't
  use HMM). Params verified present on this 595.71.05 build (were N/1). *User re-runs
  `apply-gsp-stability.sh` + reboot.*
- **16.2 Driver 595.71.05 → 595.84** (user): same production branch, 2 point releases newer,
  ships newer GSP firmware + RTD3 fix. Source-diffed: no targeted UVM fix — variable change,
  not a known cure. Do NOT jump to 610.43.02.
- **16.3 Kernel = the actual fix** (`check-kernel-fix.sh` helper): CVE-2026-43303 fixed in
  **6.12.94-1 (trixie-security — ALREADY in /boot!)** and 7.0.13-1 (sid/testing); backports
  vehicle 7.0.13-1~bpo13+1 imminent. **Fallback available today: boot 6.12.94+deb13 from GRUB
  Advanced options** (DKMS rebuilds the module) — one clean soak on it = root-cause confirmation.
- **16.4 Launcher env hygiene**: `GGML_CUDA_NO_PINNED=1` + `ATHENA_MPS=1` now DEFAULTED in
  `launch-athena-397b.sh` (were desktop-icon-only, so shell launches silently reverted to pinned
  buffers). Code-dive verified: with NO_PINNED there are ZERO app-level pinned registrations left.
- **16.5 Demo resilience** (`athena-autostart.desktop`, staged OPTIONAL): auto-relaunch at login
  after a panic reboot; worst-case unattended downtime ≈ 9 min (30 s panic + 60 s boot + 7.3 min
  model load); per-turn snapshots + memory bound loss to <1 turn. Caveat: launcher sudo prompts
  block unattended use unless targeted NOPASSWD is granted.
- **16.6 DRAM discrimination** (user, off-hours): one overnight memtest86+ pass (expected
  negative — structured pointer scribbles, not bit flips).

**Keep as tripwires (not fixes): `page_poison=1 nohugevmalloc`** — poison converted 2 would-be
silent corruptions into contained catches this run; remove only after a CVE-fixed kernel + clean
soaks. **Do NOT:** add `intel_iommu=on` (no-op, already translated) or `iommu.strict=1`; build
Xid-triggered defenses; move to 610.43; replace RAM preemptively; chase further pinned-memory
removal (surface exhausted).

**Upstream filings (evidence complete):** Debian BTS vs `linux-image-7.0.12+deb13` citing
CVE-2026-43303 with the six pstore dump sets + the lruvec unification table + poison byte-dumps +
corrupted PTE, leading with "persists under nohugevmalloc"; NVIDIA open-gpu-kernel-modules with
the GSP `Bad checksum`→Xid-69 decode (BLACKWELL_COMPUTE_B, INVALID_VALUE) + the
`rpcRmApiControl_GSP` poison catch + the Xid-decoupling facts.

## 17. README.md full reconciliation (2026-07-02)

Complete rewrite to a current, comprehensive setup guide. All URLs live-verified (HTTP-checked)
and all flags/versions taken from the running system, not prior docs:
- **Versions corrected:** ONNX Runtime 1.26.0→**1.27.0** (asset `onnxruntime-linux-x64-gpu_cuda12-1.27.0.tgz`
  — new naming since 1.27.0, extracts to the exact dir the builds reference); CUDA 12.8→**12.9
  (V12.9.86 = cuda-toolkit-12-9 12.9.1)**; gcc-12→**GCC 14.2.0**; cuDNN 9.22→**9.23.2.1-1**;
  ctx 100000→**80000**; `-np 4`→**2**; endpoint-long 1200→**700**; emotion floors per §8;
  memory dir + all paths → real `/home/user/ATHENA/` layout; desktop filename (space) fixed;
  `LLAMA_SET_ROWS`/`GGML_CUDA_GRAPH_OPT` references removed.
- **New sections:** NVIDIA driver + CUDA install (deb/DKMS via the CUDA apt repos — incl. the
  verified quirk that **CUDA 12.9 + cuDNN live in the debian12 repo**, debian13 carries only
  CUDA 13.x toolkits; `nvidia-driver-pinning-595` documented); **Silero VAD v6.2 download**
  (`ggml-org/whisper-vad`, was missing entirely); **emotion2vec → ONNX conversion** via
  `models/emotion2vec-reexport.py` (one-time offline Python; dynamic-axis export validated vs
  FunASR; the script self-downloads `iic/emotion2vec_plus_large`); **Stability, diagnostics &
  crash capture** (kernel CVE-2026-43303 requirement, apply scripts, watchdog/keepalive/coredumps,
  diag-dir layout, RUNBOOK pointer); launcher env-var reference; repository-layout map.
- **Download fixes:** Qwen shard URLs need the **`UD-Q3_K_XL/` subfolder** (the old URLs 404);
  all model URLs byte-size-verified against the on-disk files (SNAC = onnx-community
  `decoder_model_fp16.onnx`, exact match). No HF tooling anywhere — plain wget.
- CLAUDE.md's "documentation vs reality" gotcha updated to reflect the reconciliation.

## 18. `patches/talk-llama/` + from-scratch source instructions (2026-07-02)

The ATHENA↔whisper.cpp patch boundary is now explicit and version-portable:

- **Discovery: the true patch set is FIVE files, not the two previously documented.**
  Verified against upstream (pinned commit `afa2ea54` and current master, via the GitHub API):
  `silero-endpointer.h`, `silero-turn-state.h` (both `#include`d by `talk-llama.cpp`) and the
  example `CMakeLists.txt` (carries the ONNX Runtime/`ATHENA_EMOTION_ORT` wiring) were always
  ATHENA-custom, alongside `talk-llama.cpp` + `athena_memory.h`. (`llama-ext.h` is upstream.)
- **New `patches/talk-llama/` dir** = source of truth for all five, populated from the live
  build tree (fixes the "portable drive" fragility — a clean rebuild no longer risks losing
  the custom sources).
- **CMakeLists unified for pinned + current trees:** current whisper.cpp master ships
  `llama-kv-cache-dsa.cpp/.h` (DeepSeek Sparse Attention) in `examples/talk-llama` and lists
  the `.cpp` in its build; ATHENA's pre-DSA CMakeLists would fail to link there, and
  upstream's own CMakeLists drops emotion2vec. The patched file adds a conditional
  `if(EXISTS …llama-kv-cache-dsa.cpp)` source guard — verified: reconfigures + rebuilds clean
  on the pinned tree (ORT ENABLED, guard no-ops).
- **README**: new "Get the sources" section — verified `git clone` + pinned checkouts
  (llama.cpp tag `b9253` confirmed via GitHub refs API; whisper.cpp commit `afa2ea54…`),
  current-master option with honest API-drift caveat, the 5-file patch table, and the DSA/
  re-patch-after-sync notes. CLAUDE.md's 2-file/"no CMake edit" claims corrected.

## 19. Crash on the FIXED kernel (20260702-115326) → demo-mode panic sysctls; kernel-only theory falsified

Run 20260702-115326 crashed on **6.12.94+deb13 (the CVE-2026-43303-fixed kernel), with
`page_poison=1 nohugevmalloc` on the cmdline and the §16 UVM hardening applied**:
1× `pagealloc: memory corruption` (poison catch), then a GPF at `xas_start` (XArray walk)
on another byte-scribbled non-canonical pointer — victim `curl` (a watchdog health probe),
**task context** — which `panic_on_oops=1` escalated into a panic + 30 s reboot.

**Consequences:**
- **The "fixed kernel = cure" prediction (§16) is FALSIFIED.** The corruption reproduces on
  two kernel majors (6.12.94 and 7.0.12) with every kernel-side mitigation active. Remaining
  suspects: the NVIDIA open driver 595.71.05's kernel-side code (the one software constant),
  and **hardware** — the corruption shape (multiple bits within ONE byte, e.g. `ff→a0`) is
  characteristic of a marginal DRAM x8 device / byte-lane under sustained load. **memtest86+
  overnight is now the top diagnostic**, then driver 595.84 when the repo publishes it.
- **Demo-mode sysctls** (`athena-crash-capture.sysctl.conf` edited; user re-runs
  `./apply-crash-capture.sh`): `panic_on_oops=0`, `hardlockup_panic=0` — every recent fatal
  was a task-context oops on a throwaway victim (curl/date), i.e. survivable kills the old
  settings needlessly escalated into reboots. `panic=30` KEPT (interrupt-context oopses panic
  unconditionally; the box still self-recovers + captures). Trade documented in the conf;
  forensics mode = flip the two back to 1 + re-run the script.
- Capture stack (kmsg logger, pstore harvest, watchdog) unchanged and still armed.

### 19a. Run 20260702-124334 → core dumps default OFF (they amplify the crash)

Next demo-mode run crashed differently, and the stack named the mechanism. Timeline:
`12:49:14` llama-server `GGML_ASSERT(stat==cudaSuccess)` (CUDA error from the corruption) →
`12:49:16` **watchdog restarted it, HEALTHY in ~1.5 s** (the "Orpheus fell down" the user saw —
it recovered) → corruption accumulated (**19 `pagealloc` poison catches**, 1 `Bad page` in curl,
victims incl. nvidia-powerd ×7) → `12:56:19.335` **Xid 69** poisons the MPS context → a GPU client
aborts → **the fatal kernel stack is a CORE DUMP in flight**:
`do_coredump → elf_core_dump → dump_user_range → __kernel_write_iter → ext4_da_write_begin →
__filemap_get_folio → folio_alloc → __alloc_pages` — the fault is INSIDE the coredump's own folio
allocation. The §12-W3 core dump (a large ext4 write) draws poisoned pages precisely because it
fires when memory is corrupt, converting a survivable client abort into a reboot. Demo mode
couldn't help (fault in an alloc/panic path, not a plain task oops).

**Fix: `ATHENA_COREDUMP` default 1 → 0** (`launch-athena-397b.sh`). We already have the libcuda-GPF
backtraces from prior runs; on a corrupting kernel the dump is pure liability. Set `=1` to re-enable
for forensics once on a non-corrupting kernel. Watchdog + kmsg/pstore capture stay on.
**Escalating-rate note:** 19 catches this run (vs 1–2 typical) on the FIXED kernel with all
mitigations on — reinforces §19's memtest86+-first / hardware-suspect direction.

## 20. The 20260702-134528 "hang" (not a crash): dead-TTS freeze class — found and closed

Run on 595.84 + `pcie_aspm=off` + demo mode + `ATHENA_COREDUMP=0`. No panic, no Xid — the
OS-layer containment worked. What the user saw ("voice died, then Athena went down, timestamps
stopped, system fine") was an **app-layer freeze**, adversarially root-caused (V1/V2 SUPPORTED-high,
four deep code reviews, old-repo comparison):

**Chain:** `13:54:47` orpheus-speak GPF'd in `libonnxruntime.so` (disassembly: non-canonical
byte-scribbled pointer in a shared_ptr release walk = the background corruption's victim, not an
ATHENA race — the only latent race found is unreachable in the SSE config) → voice died, `.done`
can never come → `13:54:53` turn ended, brain entered the `.done` wait with `stop_sent=false` →
**`max_polls` frozen at 36000 (30 min)** → `13:54:55.6` barge fired MID-LOOP (site 2), set
`stop_sent=true` but the deadline never rescaled AND the `!stop_sent` gate silenced all further
output → silent 50 ms poll loop = "Athena went down." Physical proof: the barge's stop file was
still on disk, unconsumed. Even after 30 min it would NOT recover (fabricated COMPLETE → false-alarm
resume → next turn re-enters a fresh 30-min wait with the barge gate disarmed — infinite cycles).

**Old-version comparison (github.com/igorbarshteyn/Athena):** the rock-solid old build had the
SAME 30-min `.done` wait (`speak-daemon.sh:49`) — it was solid because the daemon never died in
that era, not because death was handled. The hang class is design debt, fixable with the features
intact; the process deaths remain the environmental corruption.

**Fixes (all implemented, tested, zero perf):**
- **20.1 `talk-llama.cpp`:** mid-loop barge now RESCALES the wait deadline
  (`max_polls = min(max_polls, i+100)` → 5 s from the stop, not 30 min from loop entry).
- **20.2 `talk-llama.cpp`:** `stream_tts_wait_done` (goodbye/heard_ok path) bound 30 min → **120 s**
  (callers only ever wait on short one-shot utterances; a dead daemon no longer blocks memory
  consolidation behind a half-hour wall). Rebuilt; `patches/talk-llama/` synced.
- **20.3 `orpheus-speak.cpp` startup-recovery:** a restarted daemon now checks for a stale trigger:
  contains `---END---` → consume it + write `.done COMPLETE` (releases a blocked brain — without
  this, a completed trigger never fires inotify and a respawn alone cannot unblock); no `---END---`
  → leave it (the brain's next sentence append replays the turn normally). **End-to-end tested
  both cases with the real binary.** Rebuilt.
- **20.4 launcher `orpheus_watchdog`:** the TTS daemon is now supervised like llama-server
  (pidfile + `kill -0` @2 s, byte-identical respawn, rolling cap 5/300 s, shared stop sentinel,
  cleanup() resyncs `PID_ORPHEUS` from the pidfile). **Mock-tested: detect→respawn→sentinel.**
  Recovery story: daemon dies → respawn ≈2-4 s → startup-recovery releases the brain → next turn
  speaks normally. Voice outage ≈ seconds, brain never freezes.
- **20.5 launcher keepalive → deep probe + `llama_watchdog` wedge branch:** `/health` is a
  hardcoded "ok" on an independent HTTP thread pool (verified in server source) — a llama-server
  wedged inside a CUDA call answers /health forever while completions hang, invisible to the old
  watchdog. The idle-gated keepalive (already a real 1-token decode) now REPORTS: 3 consecutive
  completion failures with /health green → `LLAMA_WEDGE` flag → watchdog force-restarts. Zero new
  GPU traffic.

**Corruption status:** 7 poison catches this run — 595.84 + `pcie_aspm=off` did NOT stop the
disease either (user reports memtest86+ clean ~1 month ago; discriminators left: extended memtest
under row-hammer/load patterns, a no-GPU soak, and a no-MPS single-client soak). The app layer is
now built to ride through single-client deaths.

## 21. Isolation matrix (`athena-soak-matrix.sh`) — attributing the corruption empirically

User falsified two more suspects: **memtest86+ clean (again)** and **driver versions 571→610 all
affected**. Honest audit state: the code reviews validated the app-level interaction (found and
fixed the §14 truncation, §19a coredump amplifier, §20 hang class) — but no userspace audit can
convict or acquit the interaction for the KERNEL-memory scribbles (userspace cannot write freed
kernel pages / folio metadata / PTEs; the apps can only matter as ioctl/copy-shaped TRIGGERS of a
driver kernel-side bug). What was never run is the **isolation matrix** — so it now exists:

`athena-soak-matrix.sh` — phases: A (idle control), B (nvidia-smi UVM churn, no ATHENA),
C (Orpheus llama-server + scripted load, no MPS), D (same under MPS), E (sustained Qwen
cpu-moe decode via llama-cli, production flags, no MPS), F (C+E concurrently under MPS — the
audio-free analog of a live session). Each phase counts kernel corruption signatures via
`sudo dmesg` deltas and writes per-phase excerpts to `athena-diag/soak-<ts>/`. Interpretation
table in the script header: A>0 = platform corrupts without the GPU; B>0 = any UVM client
suffices (trivial upstream repro); F≫C+E = the multi-client/MPS interaction is the trigger →
running without MPS becomes a legitimate mitigation lever; all ≈0 while live ATHENA still
corrupts = the audio/whisper/emotion/snapshot paths are implicated next. Guards against running
while ATHENA is up; one sudo prompt at start.

## 22. Soak-matrix verdict: ATHENA ACQUITTED — the platform corrupts under bare GPU compute

Phases C and E both reproduced the disease **with zero ATHENA code in the loop**:
- **C** (llama-server + scripted curl load, NO MPS, no audio/brain): system-wide carnage —
  arbitrary victims including the Claude Code process segfaulting; box down mid-phase
  (pstore batch ~19:44).
- **E** (bare `llama-cli`, Qwen cpu-moe, production flags, single CUDA client, NO MPS, NO
  server): CUDA error → `ggml_abort` inside plain `llama_decode`
  (`soak-20260702-195957/E-qwen.log`), box panicked minutes later (pstore 20:03–20:06).
- Previously falsified by the user: **memtest86+ clean (twice)**; **every driver 571→610
  affected**; two kernel majors affected; IOMMU translated with zero DMAR faults.

**Conclusion:** the corruption requires only *sustained GPU compute on this machine*. Every
ATHENA component, MPS, the audio stack, the monitor, and the app-level handshakes are
eliminated as causes (they were victims/amplifiers, now hardened). The remaining common
factors are the GPU board itself and the **platform power delivery/coordination** — on a
machine whose SBIOS power handshake (`PlatformRequestHandler … NV_ERR_INVALID_DATA`) fails
at every boot on every driver. Leading model: electrical/power-integrity corruption of host
memory writes under combined GPU(+CPU) load — memtest passes because the GPU is idle during
it.

**New: phase G** added to `athena-soak-matrix.sh` — the same Qwen decode with `-ngl 0`
(GPU fully idle; pure CPU+DDR5 load). G≈0 while C/E>0 completes the isolation: corruption
requires GPU power draw, not memory traffic per se.

**Discriminators queued:** (1) phase C/E **on battery** (EC hard-caps dGPU TGP — if the
corruption collapses at low GPU wattage, power delivery is confirmed); (2) phase G; (3) a
Lenovo service/warranty case — the evidence package (8 pstore panic sets, poison byte-dumps,
the failing ACPI handshake, bare-upstream repro commands) is complete and damning.

## Build status (all current on-disk artifacts)

| Artifact | State |
|---|---|
| `STABILITY-RUNBOOK.md` *(new, §16)* | user-facing step-by-step: UVM conf → DKMS build for 6.12.94 → kernel swap → verification → driver-update path → housekeeping |
| `whisper.cpp/build/bin/whisper-talk-llama` | **rebuilt** (Jul 2 19:02) — §3/§10/§14 + **§20.1 mid-loop deadline rescale + §20.2 120 s goodbye wait** live |
| `patches/talk-llama/` *(new, §18)* | source of truth: `talk-llama.cpp`, `athena_memory.h`, `silero-endpointer.h`, `silero-turn-state.h`, `CMakeLists.txt` — copy into `whisper.cpp/examples/talk-llama/` before configure |
| `orpheus/build/orpheus-speak` | **rebuilt** (Jul 2 19:02) — §9 + §13b + **§20.3 startup-recovery** (stale-trigger consume + .done release; end-to-end tested) |
| `llama.cpp/build/bin/llama-server` | unchanged (runtime flags changed in launcher: no `GGML_CUDA_GRAPH_OPT`, `-np 2`) |
| `launch-athena-397b.sh` *(bash — no build)* | modified — §7–§11 as before, **plus §12**: llama-server **watchdog + auto-restart** (W1), idle keepalive (W2), scoped **core dumps** (W3, never talk-llama), and reverted the container no-op levers — no `-pl` by default, `ATHENA_LOCK_MEMCLK=0`, sampler 1 Hz (W4) |
| `zzz-nvidia-athena-gsp-stability.conf` + `apply-gsp-stability.sh` | **staged, not applied** — RTD3/S0ix mitigation (§10 D1); §11 F1 `zzz-` rename so S0ix=0 wins; **§12 W5 flips `PreserveVideoMemoryAllocations`→0** (A/B for the bad-page leak); user re-runs `apply-gsp-stability.sh` + reboot |
| `athena-kmsg-logger` (`athena-kmsg-logger.cpp`) *(new, C++)* | **built** (Jul 1) — durable unfiltered `/dev/kmsg` capture (§11 F2); **caught the 123219 backtraces** |
| `apply-crash-capture.sh` + `athena-crash-capture.sysctl.conf` + `zzz-athena-gsp-diag.conf` *(new)* | **applied** — panic sysctls live + efi-pstore armed (§11 F2). Note: `EnableGpuFirmwareLogs=1` is a confirmed no-op (decoder `.bin` absent) |
| `athena-gpu-monitor.sh` *(bash — no build)* | modified — §11 F2 kmsg logger; §12 W4 cadence 1 Hz; **§13a persistent `nvidia-smi --loop-ms` per stream** (FIFO + auto-restart) → ~500× fewer UVM va-space cycles |

---

# ATHENA — Session Changes (2026-07-04, NEW BOX)

> The Build status table above is the **old Debian-trixie box** as of 2026-07-02. Everything
> below happened on the **Ubuntu redeploy** and supersedes per-box state, not the history.

## 23. Ubuntu 24.04 redeploy: forensics-off defaults; ORT symlink fix; whisper.cpp master
## STT regression → re-pinned; choppy TTS root-caused to EC `platform_profile` (launcher fix)

**Context.** Worktree redeployed onto a new build: Ubuntu 24.04.4 LTS, kernel
`6.17.0-1028-oem` (Lenovo OEM), NVIDIA 595.71.05 **compute-only** (no `nvidia-powerd` shipped),
CUDA 12.9 (nvcc V12.9.86), cuDNN 9.24 (ubuntu2404 local repo), fresh clones of **current
master** llama.cpp (`ef2d77011`) + whisper.cpp (`6fc7c33b`); 5 Qwen shards copied over
(complete, ~179 GB). None of the old box's staged mitigations (zzz-modprobe confs, panic
sysctls, GRUB tripwires) exist here — verified clean. The §22 corruption disease is a
property of the old platform; nothing here assumes it.

### 23.1 No-forensics mandate (user decision) — defaults flipped in the launcher
- `ATHENA_DIAG` **1 → 0** and `ATHENA_TTS_CAPTURE` **1 → 0** (launcher defaults; env
  override preserved — `ATHENA_DIAG=1` still works for a one-off). DIAG=0 disables the GPU/Xid
  sampler (which reads the kernel log), the prior-crash pstore harvest + `nvidia-bug-report`
  + pstore-clear, and routes llama-server output to /dev/null. `ATHENA_COREDUMP` already 0.
- New **`ATHENA_START_POWERD`** (default **0**): the launcher no longer
  `systemctl start nvidia-powerd` on its own — reports state only, never stops it. (=1
  restores start-if-inactive. Moot on this box: the service is `not-found` in the
  compute-only packaging.)
- `athena-kmsg-logger` rebuilt natively (bare `g++ -O2`, **without** running
  `apply-crash-capture.sh` — no sysctls/pstore/modprobe installs). Inert while DIAG=0.
- The launcher itself installs nothing persistent (never calls `apply-*.sh`) — audited.

### 23.2 Redeploy/build fixes
- **ORT symlinks were IntxLNK stubs.** The redeploy copy path (Windows/WSL-style symlink
  serialization) materialized `lib/libonnxruntime.so` and `.so.1` as 46/56-byte files starting
  `IntxLNK\1` → `ld: file format not recognized` at the `whisper-talk-llama` link. Recreated
  `libonnxruntime.so → .so.1 → .so.1.27.0` (SONAME chain — fixes link AND runtime). Tree-wide
  sweep: only those 2 files were mangled. Re-copy the tree only with symlink-preserving tools
  (`rsync -a` / `cp -a` / Linux tar).
- CUDA arch stays **`"86;120a"`** — nvcc 12.9 compile-tested for `sm_120a`; cmake logs
  `Replacing 120-real … with 120a-real`. Keep the `a` (Blackwell arch-specific kernels).
- llama.cpp master configure warns `OpenSSL not found, HTTPS support disabled`
  (new `LLAMA_OPENSSL=ON` default) — **benign**: ATHENA's llama-server is loopback HTTP only.
  Optional silence: `-DLLAMA_OPENSSL=OFF`.

### 23.3 whisper.cpp master STT regression → REVERTED TO PIN `afa2ea544`
On master (286 commits past the pin; **34 CUDA-backend commits**; whisper core itself only 4,
all VAD-API, off by default), run 20260704-144327: Silero fired fine (`maxp=1.00`, ENDPOINT
lines) but `whisper_full` returned **empty text for all normal speech** → silent discard at
the empty-transcription gate → **0 turns**; only the loud "Goodbye Athena" transcribed. Mic
ruled out (100% / 0.00 dB / RUNNING); energy-gate direction wrong for the symptom. Same
patches, same box, **pin build works** (run 20260704-152425 held a full conversation). Verdict:
ggml/CUDA numerics drift on sm_120a vs ATHENA's deliberately strict no-fallback decode
thresholds (`temperature_inc=0`, tightened entropy/logprob) — the documented "version-sensitive
on Blackwell" class. **Keep whisper.cpp pinned; llama.cpp master is fine** (llama-server hit
136 tok/s on a fresh boot). Procedure: `git checkout -f afa2ea544…` → re-copy the 5
`patches/talk-llama/` files → full reconfigure (delete `build/`) → rebuild.

### 23.4 Choppy TTS (run 20260704-152425): ONNX warnings ACQUITTED; root cause =
### EC `platform_profile=low-power`; launcher now sets `performance`
The four early-launch ONNX Runtime messages were investigated online and are **benign,
one-time, startup-only** (all present identically in clean runs):
- `transformer_memcpy … 9 Memcpy nodes` (SNAC; 291 for emotion2vec) — CPU-assigned
  shape-subgraph boundaries get MemcpyToHost/FromHost pairs; µs-scale scalar copies at
  ~11.7 Runs/s; CUDA-graph note moot (not enabled; `do_copy_in_default_stream=1`).
- `Some nodes were not assigned to the preferred EP` — deliberate ORT policy (shape ops on
  CPU); the conv stacks are on CUDA (1.7 GB VRAM, 509 ms warmup).
- `Serializing optimized model … NchwcTransformer` — the `.optimized` cache write at
  ORT_ENABLE_ALL; non-portable by design; launcher already `rm -f`s it each start. Expected.
- `ScatterND reduction=='none'` ×24 — **emotion2vec only**, once per user turn, off the audio path.

**What the log proved instead:** Orpheus token generation ran **below the 82.05 tok/s
realtime floor** (7 tokens = 85.3 ms audio): session 1 ≈ 47–66 tok/s → **830 stalls / 33.2 s
injected silence**, low-water −125 ms; session 2 58.9 tok/s; even the post-load goodbye
61.8 tok/s — vs **136.3 tok/s** for the fresh-boot goodbye and 160–167 documented. Chunk
first-audio degraded 318 ms → 2.6 s → 13.2 s (token starvation queueing, not SNAC decode).
**Root cause:** `/sys/firmware/acpi/platform_profile` sat at **`low-power`** — the EC-level
sustained CPU+dGPU budget clamp. The launcher stops power-profiles-daemon (so nothing manages
the profile) and `nvidia-powerd` doesn't exist here → profile frozen; bursts boost, sustained
combined Qwen+Orpheus load clamps progressively.
**Fix (live):** `setup_system_performance` now saves + sets `platform_profile=performance`
**after** the PPD stop (so PPD can't re-assert); validates against `platform_profile_choices`;
`restore_system_performance` restores the saved value **before** PPD restart. Env:
`ATHENA_PLATFORM_PROFILE=<name>` overrides, `=""` skips. Decision tree simulated against live
sysfs (default/skip/alternate/invalid/no-op paths). **VALIDATED — run 20260704-172530**
(pin STT + `performance` profile; 11 TTS sessions over ~16 min of sustained conversation):
llama-server **114–129 tok/s** (1.39–1.57× the 82.05 floor, **flat across the session** —
no thermal/power decay), `stall silence: 0 ms in 0 stalls` in 10/11 sessions (lone
exception: 320 ms across 8 micro-dips inside a rapid barge-pivot session), low water
≥ +488 ms everywhere else (clamped run: −125 ms). Caveat: the profile was ALREADY
`performance` at this launch (PPD re-assert after the prior cleanup, or a manual set), so
the block exercised its verify/no-op path; the write path remains simulation-tested only.
- Also fixed: with DIAG=0, `RUN_DIR` stays **empty** (was a never-created path → per-launch
  `watchdog.log: No such file or directory` spam; watchdogs now fall back to `/dev/shm`).

### 23.5 State on this box (2026-07-04, updated through §23.6)
| Artifact | State |
|---|---|
| `llama.cpp` tree + `build/bin/llama-server` | **master `ef2d77011`**, built Jul 4 10:30 — works (TTS served both runs) |
| `orpheus/build/orpheus-speak` | rebuilt Jul 4 10:33 — SNAC on CUDA EP, warmup ~509 ms, healthy |
| `whisper.cpp` tree | **re-pinned to `afa2ea544`**, `whisper-talk-llama` rebuilt **16:43** (4,113,704 B — pin-build size; the 15:45 master drift was transient and never run). STT validated in run 20260704-172530: 7 endpoints, full transcriptions incl. a ~60-word turn, **14 turns captured**, goodbye consolidation ran (7 candidates, personality updated, 13 memories retained). |
| `launch-athena-397b.sh` | modified — §23.1 defaults (DIAG/TTS_CAPTURE/START_POWERD), §23.4 platform_profile save/set/restore + RUN_DIR fix, §23.6 `-vms 25000 -mt 256` |
| `athena-kmsg-logger` | rebuilt natively Jul 4 10:05; inert (DIAG=0) |
| `models/` | all 5 Qwen shards present; Orpheus/Whisper/Silero/emotion2vec in place |
| Old-box apparatus (`apply-*.sh`, zzz confs, runbook) | **not applied here**; STABILITY-RUNBOOK.md is Debian-specific and does not map to this Ubuntu box |

### 23.6 STT truncation on long read turns — diagnosed against the demo script; capture
### window widened (`-vms 15000→25000`, `-mt 128→256`)

User report after run 20260704-172530: transcripts "cutting inputs off towards the later
turns" while reading `athena-demo-script.txt` aloud. Ground-truth diff of every
endpoint-path turn against the script ACQUITS Whisper of progressive degradation:
5 of 7 turns essentially verbatim — including turn 6-of-7 at 17:42, mid-late session,
perfect — with constant word-level lexicon noise only ("steelman" → "steal me on").
Prosodic endpointing also acquitted: **23/23 decisions chose the long 700 ms target**;
the 350 ms short path never fired once. The two real losses are DISCRETE mechanisms:

- **Turn S1-2 (13.0 s read) beheaded** ("See, you just drew a line…" → "There is a
  line…"): fixed-trailing-window arithmetic. `audio.get(voice_ms)` runs **~1.4–1.7 s
  after the last word** (704 ms Silero-confirmed silence + ~0.7–1.0 s silence-confirm
  hangover + processing), so `-vms 15000` covers only **~13.3 s of actual speech** — a
  13.0 s turn loses its first ~0.5–1 s. Longest-turn artifact, not latest-turn.
- **Turn S1-9 first sentence gone** ("No — no, I'm fine." → only the trailing
  "This is great. I love where this is going.", speech=2240 ms — exact length of the
  survivors): turn-taking collision, BY DESIGN. The script's own [DELIVER: flat,
  deflated] read began during the tail of an 80 s reply, sat **below
  `--barge-rms 0.0020`** (monitor floor read 0.0000) → no barge → the deliberate
  post-playback `audio.clear()` (talk-llama.cpp:3594 — the anti-bleed/stale-audio
  guard) wiped everything said during playback.
- The "increasing over later turns" PERCEPTION was interaction dynamics, not decay:
  replies grew 55–160 s, so later script beats (6/8/10/11/12) were delivered as
  BARGE-INS (they never entered the endpoint path at all) and the ending compressed
  to barge → barge → goodbye. System dials were flat all session.

**Fix (live, launcher-only — runtime flags, no rebuild):** `-vms 15000 → 25000`
(30 s ring; effective speech coverage now ~23.3 s — the longest script read fits with
~10 s margin) **paired with `-mt 128 → 256`** (25 s of dense speech ≈ 110–140 Whisper
tokens would have hit the old 128 cap and traded head-truncation for a NEW
tail-truncation). README flag table + CLAUDE.md command block reconciled in the same
change — zero stale `-mt 128`/`-vms 15000` references remain.

**Watch-item:** the 0.003 whole-window RMS energy gate now averages over a wider
buffer — more silence dilution. If a short, quietly-spoken turn is ever silently
skipped (endpoint fires, no transcript, no response), move the gate to a
peak/trailing-window basis in talk-llama.cpp. **Not changed (by design):** the
collision class — remedies are a firm barge (the firm ones all worked: beats 6/8) or
an A/B of `--barge-rms 0.0020 → ~0.0012` (false-trigger risk from breaths). **Queued
structural fix:** anchor capture at Silero's tracked speech-START
(`audio.get(now − t₀ + margin)`) instead of a fixed trailing window — eliminates the
window-arithmetic class for any turn length up to the ring.

### 23.7 STT premature termination — prosodic short-target collapsed
### (`--endpoint-short-ms 350→800`, `--endpoint-long-ms 700→800`)

Run 20260704-222137 (pin STT + platform_profile=performance + §23.6 `-vms 25000`):
user reports inputs still cut off mid-read on the demo script. Ground-truth diff of all
4 endpoint-path turns vs `athena-demo-script.txt`: 3 fired correctly on the LONG target
(silence 704–768 ms, transcripts complete). **Endpoint 4 (beat 4, "the late nights")
fired at `silence=352ms`** — the FIRST time the prosodic SHORT (350 ms) target ever
engaged — and cut the payload sentence: user said "…I didn't know how to answer. **This
matters to me more than I expected it to.**"; Whisper got only through "…how to answer."

**Root cause (independently read + 3-agent workflow, adversarially verified).** On the
active Silero engine, `target_provider` (talk-llama.cpp:2602-2612) is called once at
`entered_trailing` and returns `endpoint_short_ms` when `ProsodicEndpointer` sets
`turn_final`. `turn_final` (talk-llama.cpp:1768) is `analyzed && (f0_falling ||
f0_falling_strong || e_decaying)` — an **OR**; `f0_falling` ALONE sufficed
(f0_slope=−72.8 < −60, voiced=0.38 ≥ 0.25). Crucially the Silero trailing gate
(silero-turn-state.h:127) is `silence_ms_ >= target_ms_` with **NO minimum-silence
floor** — the absolute-RMS floor `endpoint_silence_rms` lives only in the energy
`vad_simple` branch, guarded by `vad_engine != "silero"` (talk-llama.cpp:2633) and never
runs under Silero. So prosody wrote 350 → turn ended after exactly 11×32 ms = 352 ms,
floorless. This is structural: a sincere clause-final falling intonation is acoustically
identical to a true turn-final fall (research: within-turn pauses median ~640 ms —
they EXCEED between-turn gaps; declination marks every intonational phrase), so a 350 ms
short target on emotional/deliberate delivery is guaranteed to misfire.

**Fix (live, flag-only, no rebuild):** collapse the short path —
`--endpoint-short-ms 350 → 800` and `--endpoint-long-ms 700 → 800` (equal). With
short==long, prosody's verdict can no longer commit a target below the long value, so
the floorless gate demands a full 800 ms of trailing silence on EVERY turn — the exact
regime that ran 23/23 clean, plus margin. 800 = ~30–95 ms over this run's genuine
turn-ends (704–768 ms) and above the ~640 ms within-turn median, so dramatic pauses up
to 800 ms are safe. Latency cost is ~invisible against 5–13 s LLM turn latency. **Dial:**
700 = snappier (proven clean, no margin); 850 = more margin. CLAUDE.md + README flag
table reconciled; zero stale 350/700 endpoint values remain.

**Verified NOT the fix (workflow do-not list):** raising `--endpoint-long-ms` alone
(the SHORT path fired — long-ms is irrelevant when turn_final==true); `--vad-last-ms`
(red herring — unused on the Silero path); `--endpoint-f0-fall-strong` (bypasses the
voicing gate → MORE aggressive on sparse-voicing emotional endings — keep 0);
`--endpoint-f0-fall` alone (brittle: the independent energy-decay OR-branch still fires;
and moot once short==long). Code options (min-silence floor in silero-turn-state.h;
conjunctive `f0_falling && e_decaying`) rejected as first move: functionally identical
to the flag fix on the active Silero path and needs a rebuild (risks the sm_120a build).

**Queued (research-endorsed):** `--endpoint-f0-fall`/`--endpoint-energy-decay` are now
inert on timing but kept wired for a future prosody-to-**EXTEND** path — a flat/rising
"not-done" contour should wait LONGER than long-ms (extend), which is the robust use of
prosody, rather than shortening on a fall. Pairs with the §23.6 speech-start-anchored
capture as the two structural STT-robustness improvements.

### 23.8 STT robustness upgrade — prosody-to-EXTEND (code + rebuild)

The §23.7 collapse (short==long==800) removed the premature-*shorten* class but left prosody
inert. §23.8 implements the research-endorsed **robust** use of prosody: use the pre-pause
contour to wait **LONGER** when the speaker is clearly mid-thought — never shorter. Additive to
the 800 ms floor; cannot regress §23.7.

**Code (patches/talk-llama/talk-llama.cpp, final build 23:33 on pin afa2ea544, ORT still linked):**
- `ProsodicEndpointer`: new fields `f0_rise_hz_s=50`, `energy_rise=0.5`; `Result` gains
  `bool continuing`. Computed after `turn_final`:
  `continuing = analyzed && !turn_final && (voiced≥0.25 && f0_slope > +f0_rise) && (voiced≥0.25 && energy_slope > +energy_rise)`
  — **an AND of pitch-rise AND energy-rise** (see the review finding below), POSITIVE slopes +
  voicing, suppressed when `turn_final`.
- `whisper_params`: `endpoint_extend_ms=0` (off), `endpoint_f0_rise=50`, `endpoint_energy_rise=0.5`
  + arg parsing (`--endpoint-extend-ms`/`--endpoint-f0-rise`/`--endpoint-energy-rise`) + help.
- Both target-provider sites (active Silero ~2637 and vad_simple fallback ~2679):
  `if (endpoint_extend_ms > tgt && ep.continuing) tgt = endpoint_extend_ms;` — **upward only**
  (`max` semantics via the `>` guard), debug label `not-done/extend`.
- Off by default (`endpoint_extend_ms=0` → `0 > tgt` false → byte-identical prior behavior).

**Launcher (live):** `--endpoint-extend-ms 1200 --endpoint-f0-rise 50 --endpoint-energy-rise 0.5`.
A "not-done" contour grants up to 1200 ms (vs the 800 ms floor) for the user to resume. Genuine
turn-ends (falling/flat) stay at 800. CLAUDE.md + README reconciled.

**Adversarial review changed the design (and it matters).** A first cut used `f0_rising ||
energy_rising`, extend 1400, f0-rise 30. The review (self + agent) found:
- **C1** — a genuinely turn-final yes/no QUESTION rises in pitch (`f0_slope` ≫ +30, never
  `turn_final`), so the OR fired `continuing` and over-extended EVERY question turn — the most
  latency-sensitive kind, and questions dominate this Q&A demo.
- **C2 (decisive)** — a wrong EXTEND is **UNRECOVERABLE**: Athena answers LATE while the user,
  being done, stays silent, so the barge-in path (which recovers a wrong SHORT) never triggers;
  the full extend elapses as dead air. The original prosody design's "wrong final call is
  recoverable" premise does NOT hold for extend.
- **C3** — `f0_rise=30` (~13.5 Hz over the 450 ms window) leaked on flat declaratives.
**Fixes applied:** `||` → **`&&`** (a final question rises in pitch but its energy TAPERS into
the pause → energy-rise fails → NOT extended; a real held continuation both rises and sustains
energy → extended); `f0_rise 30 → 50`; extend `1400 → 1200` (base+400) to **bound** any residual
false-positive's dead-air. Verified: (a) build clean, flags exposed, ORT linked; (b) off-by-default
byte-identical (`0 > tgt` short-circuits); (c) strictly upward — §23.7 800 ms floor preserved;
(d) aggregate-init arity correct (7 members / 7 inits); (e) no adverse hysteresis interaction
(target re-established per pause). Residual risk: a question that rises in BOTH pitch and energy
still over-extends — rare; dial via `--endpoint-f0-rise` up or `--endpoint-extend-ms 0`.
**Validation pending the next run** (watch `[silero] ... -> not-done/extend (1200 ms)`: should
fire on held mid-thought rises, NOT on genuine ends — especially NOT on question-ending turns).

**Still queued (the OTHER structural item, NOT done here):** §23.6 speech-start-anchored capture
(`audio.get(now − t₀ + margin)`) — eliminates the fixed-trailing-window head-truncation class.

### 23.9 Qwen decode perf A/B (llama-bench) — flag surface confirmed tapped out; keep `-t 16 --cpu-moe`

Measured decode (`tg128`, bf16 KV, `-ngl 99`, `--no-mmap`, 5 reps) on the pinned build:
- **A** `-t 16 --cpu-moe` (`-ncmoe 60`): **3.74 ± 0.01 tok/s** (baseline)
- **B** `-t 22 --cpu-moe` (2 cores reserved): **2.86 ± 0.02 (−23.5%)** — CONFOUNDED (bench ran
  `governor=powersave` + C-states on → E-cores at 3.4 GHz not 4.7; penalty inflated). Mechanism
  (per-layer matmul barriers wait on the slower E-cores + memory-controller contention on a
  latency-bound scattered gather) means best-case-tuned is ~flat, not a win. **Do not raise `-t`.**
- **C** `-t 16 --n-cpu-moe 59` (1 layer's experts → GPU): **3.80 ± 0.03 (+1.6%)** — clean (same CPU
  conditions as A), but marginal. Scales ~+1.6%/layer; production's ~5 GB VRAM headroom caps it at
  ~1–2 layers (~+1.6–3.2%), with OOM-under-MPS risk. Not worth it.

Absolute t/s is ~half production's 6.4 because the standalone bench skipped the launcher's system
tuning (governor=performance + C-states off + platform_profile ≈ 1.7×) — an accidental measurement
of how much that tuning is worth. **Verdict: keep `-t 16 --cpu-moe`.** The ceiling is DDR5
scattered-expert-gather bandwidth (~3.07 GB/token, 10-of-512 experts); no launch flag moves it. GPU
clock/power flags (`ATHENA_LOCK_MEMCLK`/`_POWER_LIMIT`/`_CLOCK_CAP`) are the wrong subsystem (GPU
GDDR7 ≠ system DDR5) and stay off. The only real lever is hardware (RAM speed — already at the
laptop's rated DDR5-4000). Re-run script: `~/.claude/jobs/*/tmp/qwen-bench.sh` (apply
governor=performance + C-states off FIRST or the E-core numbers are confounded).

## 24. GitHub deployment prep — `install.sh` + `.gitignore` + README quick-start (2026-07-04)

Prepared the repo to be published to GitHub with everything EXCEPT models. Three artifacts.

**`.gitignore` (new).** Commits source + scripts + docs + confs + the ATHENA patches +
the emotion2vec conversion script; excludes everything `install.sh` produces: `models/*`
(~179 GB, keep `.gitkeep` + `emotion2vec-reexport.py`), `orpheus/snac24_dynamic_fp16.onnx`
+ `*.optimized`, `onnxruntime-linux-*/` + tarball, `/llama.cpp/` + `/whisper.cpp/`,
`orpheus/build/`, the `athena-kmsg-logger` binary (source `.cpp` stays), `athena-diag/`,
`athena_memory/*` (keep `.gitkeep`), logs, trigger files. Verified against a real git
index: **0 large/generated paths leak; 32 files tracked; all must-ship source present.**

**`install.sh` (new, ~470 lines).** Assumes the GPU stack (driver/CUDA/cuDNN) is present;
does NOT install it. Prints the system-requirements block, fail-fast preflight, then:
(1) models → `models/` (+ SNAC → `orpheus/`) + optional emotion2vec ONNX conversion;
(2) ONNX Runtime 1.27.0 download/extract (+ SONAME-symlink repair, §23.2); (3) llama.cpp
master → `llama-server` (`-DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF`, arch `86;120a`);
(4) whisper.cpp pinned `afa2ea544` + patches → `whisper-talk-llama`; (4b) `orpheus-speak`
(the user's step list omitted it — required, added); (5) integrity check → "ATHENA is
ready to run". Downloads are **Content-Length-verified + resumable** (`wget -c`), builds
are **idempotent** (skip when the binary exists AND its libs resolve), fully gated
(`--skip-*`/`--force`/`--check-only`/`--jobs`/`-y`). Design: no surprise sudo (only
ldconfig, degrades gracefully); warns loudly if the repo isn't at `/home/user/ATHENA`
(the launcher hard-codes it 17×).

**Adversarially reviewed (3-agent workflow: bash-correctness / edge-failure / fidelity)**
before shipping — the reviewers empirically reproduced real bugs on bash 5.2, all fixed:
- **`set -Eeuo pipefail` + ERR-trap turned every best-effort `var="$(cmd|pipe)"` probe into
  an install-killer** — a transient `curl` HEAD or `nvidia-smi`/`free`/`df` blip aborted the
  whole multi-hour download. Fixed: `remote_size` ends `|| true`; every preflight probe `|| true`.
- **Truncation/error-page holes:** unknown-Content-Length partials were blindly kept; the
  HTML guard only fired <1 MB and missed HF's JSON 401 bodies. Fixed: `verify_container()`
  (positive GGUF/gzip magic — authoritative, no false text-sniff on GGUF chat-template
  metadata; negative HTML/JSON sniff otherwise) runs at any size; `wget -c` always finishes
  an unknown-size file; added `--retry-on-http-error=429,500,502,503,504`.
- **Stale/broken build skipped:** `[ -x ]` passed an executable-but-broken binary. Fixed:
  `bin_ok()` also requires `ldd` to resolve all libs; `--force`/source-change `rm -rf build`.
- **Pin not enforced when a binary existed:** an existing whisper.cpp master checkout
  (§23.3 STT regression) was skipped. Fixed: pin reconcile + patch re-copy moved BEFORE the
  build-skip gate; `need_build` forces a clean rebuild on any checkout change.
- **Emotion step could abort the install** (subshell left `set -u` on; dead `pip --user`
  fallback). Fixed: `( set +eu; trap - ERR … ) || true`; `--user --break-system-packages`
  (PEP-668 on 24.04). Plus: top-level `read || true` (clean Ctrl-D), `--jobs` numeric
  validation, integrity now also checks `speak-daemon.sh` (launcher preflight hard-requires it).

**Fidelity: zero drift** — the 6 download URLs, the whisper pin, and all build flags match
README §"Download components"/"Build …" + CHANGES §23 exactly (only cosmetic arch-order
`86;120a` vs the README blocks' `120a;86`, functionally identical). **Tested on this box:**
`--check-only`, preflight, `fetch()` size-verify/skip on the real 179 GB tree, `--help`,
bad-flag, and `--jobs` validation all pass; `bash -n` clean. README gained a Quick-start
section. Not committed to git here (env is not a repo) — the user runs `git init` + push.

## 25. Path portability — the repo is now relocatable (2026-07-04)

Before this, `launch-athena-397b.sh` hard-coded `/home/user/ATHENA` in **17 places** and
`speak-daemon.sh` in **3** — so a fresh GitHub clone only ran if placed at exactly that
path. Made the whole tree self-locating so it runs from ANY clone location.

**Scripts self-locate.** Both `launch-athena-397b.sh` and `speak-daemon.sh` now derive
```
ATHENA_DIR="${ATHENA_DIR:-$(cd -- "$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")" && pwd)}"
```
at the top and build every path from it; the launcher `export`s it so children inherit it.
`readlink -f` resolves through symlinks, and the `cd/pwd` makes it independent of the
caller's working directory. **Verified across all invocation modes** — direct absolute
path, symlink (the `~/bin`/`.desktop` case), relative `./`, `bash script`, and an
`ATHENA_DIR=…` env override — plus a full **relocation test**: the real derivation +
path-assignments run from `/tmp/athena-reloc` resolve every binary/model/sentinel to the
relocated dir.

**`.desktop` launchers are generated, not committed.** `Athena 397B.desktop` and
`athena-autostart.desktop` (whose `Exec`/`Icon`/`Path` can't hold shell vars) became
`*.desktop.in` templates with an `@ATHENA_DIR@` placeholder; `install.sh`'s new
`gen_desktop` step sed-fills them with the real path. The generated `.desktop` are
gitignored; the `.in` templates are committed.

**`install.sh`** dropped the now-obsolete `EXPECTED_HOME` warning and the "move the repo
to /home/user/ATHENA" reminder; preflight now prints `repo location: <dir> (relocatable)`
and warns only if the path contains a space.

**Comment-only machine paths genericized** in `orpheus/orpheus-speak.cpp`,
`patches/talk-llama/CMakeLists.txt`, `orpheus/CMakeLists.txt`. Docs updated: README
quick-start is now "clone ANYWHERE — paths auto-derive" (deep manual examples keep the
reference path, disclaimed); CLAUDE.md's repo-root note marks the tree relocatable; the
v1 demo's `mv /home/user/athena_memory …` setup was fixed to a repo-relative path — which
also corrected a latent bug (it had been moving the wrong directory, not the launcher's
actual `athena_memory`).

**Adversarially verified** (workflow: completeness + did-it-break). Confirmed sound: all
paths resolve; the launcher's `DONE_FILE`/`STOP_FILE` still agree with what talk-llama and
orpheus-speak derive by extension-stripping the trigger (holds even with dots/spaces);
speak-daemon inherits the exported dir; `install.sh` bakes an absolute `ONNXRUNTIME_ROOT`
RUNPATH at build time (moving the repo AFTER building breaks it — expected, and `bin_ok`'s
ldd check catches it). The review also caught **two real space-in-clone-path bugs**, both
fixed and re-tested: (a) the launcher's `--capture-dir`/`--diag` flags were built as
space-joined strings and expanded unquoted — converted to **bash arrays**
(`"${CAPTURE_FLAG[@]}"`); (b) the `.desktop` `bash -c` payload left the launch path
unquoted — now **single-quoted** inside the payload. A spaced clone path now works
end-to-end (proven: the capture path stays one arg; the desktop path stays one token).

**Not portable-after-build:** the ONNX Runtime RUNPATH and the CUDA `sm_120a` binaries are
baked at build time, so *clone → build → run* works anywhere, but *moving the tree after
building* requires a rebuild (README-documented).
