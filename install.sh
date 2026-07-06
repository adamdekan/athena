#!/usr/bin/env bash
# ==============================================================================
#  ATHENA — installer
#  Fetches models, ONNX Runtime, and upstream sources; builds the three binaries;
#  verifies component integrity. Assumes the SYSTEM-LEVEL GPU stack is already in
#  place (NVIDIA driver, CUDA toolkit, cuDNN) — this script does NOT install those.
#
#  Usage:   ./install.sh [options]
#  Run  ./install.sh --help  for the full option list.
#
#  Design notes:
#   - Idempotent & resumable: re-run any time. Completed downloads (verified by
#     server Content-Length) and existing binaries are skipped unless --force.
#   - Fail-fast preflight: every prerequisite is checked before any work begins;
#     unfixable gaps (GPU stack, RAM, disk) abort with the exact remedy.
#   - No surprise sudo: only ldconfig (for the ONNX Runtime lib path) uses sudo,
#     and it degrades gracefully (the binaries carry an RUNPATH regardless).
# ==============================================================================
set -Eeuo pipefail

# ── Resolve our own location; ATHENA_DIR is the repo root ─────────────────────
SELF="$(readlink -f "${BASH_SOURCE[0]}")"
ATHENA_DIR="$(cd "$(dirname "$SELF")" && pwd)"
cd "$ATHENA_DIR"

# ── Pins (must match README §"Verified build environment" / CHANGES.MD §23) ───
ORT_VER="1.27.0"
ORT_DIR="onnxruntime-linux-x64-gpu_cuda12-${ORT_VER}"
ORT_TGZ="${ORT_DIR}.tgz"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/${ORT_TGZ}"
ORT_ROOT="${ATHENA_DIR}/${ORT_DIR}"

WHISPER_PIN="afa2ea544fb4b0448916b4a31ecd33c8685bd482"   # tested commit (STT master regresses on sm_120a — CHANGES.MD §23.3)
LLAMA_REPO="https://github.com/ggml-org/llama.cpp.git"    # built from latest master (works — §23.5)
WHISPER_REPO="https://github.com/ggml-org/whisper.cpp.git"

CUDA_ARCH="86;120a"          # Blackwell sm_120a (arch-specific) + Ampere fallback
# (repo is relocatable — the launcher derives its own dir; no fixed home required)

# ── Requirements shown to the user (targets THIS reference machine) ───────────
REQ_OS="Ubuntu 24.04.4 LTS"
REQ_GPU_MIB=23000            # RTX PRO 5000 Blackwell reports ~24463 MiB
REQ_RAM_GB=160               # ~161 GiB resident under --cpu-moe (192 GB machine)
REQ_DISK_GB=210              # ~179 GB models + 0.24 GB ORT + ~30 GB build/clone headroom
REQ_CUDA="12.9.x  (nvcc V12.9.86 tested)"
REQ_DRIVER="NVIDIA 595.x production branch  (595.71.05 tested, open kernel modules)"
REQ_CUDNN="cuDNN 9.x  (9.23.2.1 tested — cudnn9-cuda-12)"

# ── Options ───────────────────────────────────────────────────────────────────
DO_MODELS=1 DO_EMOTION=1 DO_ORT=1 DO_LLAMA=1 DO_WHISPER=1 DO_ORPHEUS=1
FORCE=0 ASSUME_YES=0 CHECK_ONLY=0 JOBS="$(nproc 2>/dev/null || echo 4)"

usage() {
    cat <<EOF
ATHENA installer — download models + ONNX Runtime, build binaries, verify.

Usage: ./install.sh [options]

  --check-only        Run only the final integrity check (Step 5) and exit.
  --force             Rebuild binaries / re-extract ORT even if present.
                      (Does NOT re-download models — delete a model file to re-fetch it;
                       fetch() already verifies size + container on every run.)
  --jobs N            Parallel build jobs (default: nproc = ${JOBS}).
  -y, --yes           Non-interactive; assume "yes" to prompts.

  --skip-models       Skip model downloads (Step 1).
  --skip-emotion      Skip only the emotion2vec ONNX conversion (Step 1c).
  --skip-ort          Skip ONNX Runtime download/extract (Step 2).
  --skip-llama        Skip llama.cpp clone/build (Step 3).
  --skip-whisper      Skip whisper.cpp clone/patch/build (Step 4).
  --skip-orpheus      Skip orpheus-speak build (Step 4b).
  -h, --help          Show this help.

Assumes the GPU stack (driver, CUDA 12.9, cuDNN 9.x) is already installed.
Models are NOT bundled — this fetches ~179 GB into models/.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --check-only)  CHECK_ONLY=1 ;;
        --force)       FORCE=1 ;;
        --jobs)        JOBS="${2:?--jobs needs a number}"; [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be a positive integer, got '$JOBS'" >&2; exit 2; }; shift ;;
        -y|--yes)      ASSUME_YES=1 ;;
        --skip-models) DO_MODELS=0 ;;
        --skip-emotion) DO_EMOTION=0 ;;
        --skip-ort)    DO_ORT=0 ;;
        --skip-llama)  DO_LLAMA=0 ;;
        --skip-whisper) DO_WHISPER=0 ;;
        --skip-orpheus) DO_ORPHEUS=0 ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage; exit 2 ;;
    esac
    shift
done

# ── Pretty output (color only on a TTY) ───────────────────────────────────────
if [ -t 1 ]; then B=$'\e[1m'; R=$'\e[31m'; G=$'\e[32m'; Y=$'\e[33m'; C=$'\e[36m'; Z=$'\e[0m'
else B='' R='' G='' Y='' C='' Z=''; fi
log()  { printf '%s\n' "${C}${B}==>${Z} $*"; }
sub()  { printf '    %s\n' "$*"; }
ok()   { printf '    %s✓%s %s\n' "$G" "$Z" "$*"; }
warn() { printf '    %s! %s%s\n' "$Y" "$*" "$Z" >&2; }
bad()  { printf '    %s✗ %s%s\n' "$R" "$*" "$Z" >&2; }
die()  { printf '\n%s✗ FATAL:%s %s\n' "$R$B" "$Z" "$*" >&2; exit 1; }
hr()   { printf '%s\n' "────────────────────────────────────────────────────────────────────────"; }
trap 'die "aborted at line $LINENO (command: $BASH_COMMAND)"' ERR

confirm() {  # confirm "question" — returns 0 for yes
    [ "$ASSUME_YES" = 1 ] && return 0
    [ -t 0 ] || { warn "non-interactive, no --yes → treating as NO: $1"; return 1; }
    local a; read -r -p "    $1 [y/N] " a; [[ "$a" =~ ^[Yy] ]]
}

human() { awk -v b="$1" 'BEGIN{s="B KB MB GB TB";split(s,u," ");i=1;while(b>=1024&&i<5){b/=1024;i++}printf "%.1f %s",b,u[i]}'; }

# server Content-Length for a URL (follows redirects); empty if unknown.
# Terminates with `|| true` so a HEAD failure (timeout/DNS/5xx) NEVER aborts the
# install — an unknown size just falls through to wget -c + container validation.
remote_size() {
    { curl -sIL --max-time 30 "$1" 2>/dev/null \
        | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{if(v)print v}'; } || true
}

# Reject an error page (HTML/JSON) or wrong container type, at ANY size. Positive
# magic where we know it (GGUF, gzip); negative sniff otherwise. Returns non-zero
# if the file is NOT what we expect. Catches HF 401/403/429 bodies saved as models.
verify_container() {
    local f="$1"
    # Positive magic is AUTHORITATIVE where we know it — do not also text-sniff (a valid
    # GGUF's metadata can legitimately contain words like "message" in the chat template).
    case "$f" in
        *.gguf) [ "$(head -c4 "$f" 2>/dev/null)" = "GGUF" ] && return 0 || return 1 ;;               # GGUF magic
        *.tgz)  [ "$(head -c2 "$f" 2>/dev/null | od -An -tx1 | tr -d ' ')" = "1f8b" ] && return 0 || return 1 ;;  # gzip
    esac
    # No positive magic (.onnx/.bin/…): reject an HTML page or a JSON error body.
    if head -c 512 "$f" 2>/dev/null | grep -qiE '<html|<!doctype|"error":|"message":|access to model|is restricted|externally-managed'; then
        return 1
    fi
    return 0
}

# fetch URL DEST — resumable, size- AND container-verified. Idempotent.
fetch() {
    local url="$1" dest="$2" name; name="$(basename "$dest")"
    local want; want="$(remote_size "$url")"       # empty if unknown; never aborts
    mkdir -p "$(dirname "$dest")"
    if [ -f "$dest" ]; then
        local have; have="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
        if [ -n "$want" ] && [ "$have" = "$want" ]; then
            if verify_container "$dest"; then ok "$name — present & complete ($(human "$have"))"; return 0
            else warn "$name present at expected size but fails validation (error page?) — re-fetching"; rm -f "$dest"; fi
        elif [ -n "$want" ]; then
            sub "$name — resuming ($(human "$have") of $(human "$want"))"
        else
            # unknown server size: NEVER trust a partial blindly — let wget -c finish it
            sub "$name — server size unknown; verifying/continuing via wget -c"
        fi
    else
        sub "$name — downloading ${want:+($(human "$want"))}"
    fi
    # wget -c resumes; a fully-present file yields HTTP 416 (treated as done).
    # --retry-on-http-error covers HF 429/5xx (wget treats 4xx/5xx as fatal otherwise).
    wget -c -q --show-progress --tries=6 --retry-connrefused \
         --retry-on-http-error=429,500,502,503,504 --timeout=60 --waitretry=10 \
         -O "$dest" "$url" || die "download failed (network/HTTP) for $name: $url  — re-run to resume."
    if [ -n "$want" ]; then
        local got; got="$(stat -c%s "$dest" 2>/dev/null || echo 0)"
        [ "$got" = "$want" ] || die "$name truncated: got $(human "$got") of $(human "$want"). Re-run to resume."
    fi
    verify_container "$dest" || { rm -f "$dest"; die "$name is not a valid file (HTML/JSON error page or wrong type). Check the URL / model access."; }
    ok "$name — $(human "$(stat -c%s "$dest")")"
}

need_cmd() { command -v "$1" >/dev/null 2>&1; }
# a binary is "ok" only if it exists, is executable, AND all its shared libs resolve
# (catches a build linked to an ORT dir that later moved — the realistic broken-build case).
bin_ok() { [ -x "$1" ] && ! ldd "$1" 2>/dev/null | grep -q 'not found'; }

# ==============================================================================
#  BANNER + SYSTEM REQUIREMENTS
# ==============================================================================
print_requirements() {
    hr
    printf '%s\n' "${B} ATHENA — fully-offline voice assistant · installer${Z}"
    hr
    cat <<EOF
 ${B}SYSTEM REQUIREMENTS${Z} (the GPU stack below must already be installed):

   • OS            ${REQ_OS}
   • GPU           NVIDIA Blackwell, ≥ 24 GB VRAM  (sm_120a; RTX PRO 5000 tested)
   • System RAM    ≥ ${REQ_RAM_GB} GB free  (≈161 GiB resident for the MoE experts)
   • Free disk     ≥ ${REQ_DISK_GB} GB  (models ≈179 GB + ORT + build trees)
   • NVIDIA driver ${REQ_DRIVER}
   • CUDA toolkit  ${REQ_CUDA}
   • cuDNN         ${REQ_CUDNN}
   • Build deps    cmake ≥ 3.24 · g++ (C++17) · git · wget · curl · tar ·
                   libsdl2-dev · libcurl4-openssl-dev
                   (optional: moreutils, pulseaudio-utils, python3 for emotion2vec)

 This installer downloads models + ONNX Runtime, clones & builds llama.cpp and
 whisper.cpp (+ the ATHENA patches) and orpheus-speak, then verifies everything.
 It does ${B}not${Z} install the driver/CUDA/cuDNN — see README "Install the NVIDIA driver".
EOF
    hr
}

# ==============================================================================
#  PREFLIGHT — check everything, fail fast on the unfixable
# ==============================================================================
preflight() {
    log "Preflight — verifying prerequisites"
    local fatal=0

    # Repo location: the launcher derives its dir at runtime, so any path works.
    ok "repo location: $ATHENA_DIR (relocatable)"
    case "$ATHENA_DIR" in *[[:space:]]*) warn "path contains a space — supported, but some external tools dislike it";; esac

    # OS
    if [ -r /etc/os-release ]; then
        . /etc/os-release
        if [ "${VERSION_ID:-}" = "24.04" ]; then ok "OS: ${PRETTY_NAME:-Ubuntu 24.04}"
        else warn "OS is '${PRETTY_NAME:-unknown}' — tested only on ${REQ_OS} (continuing)"; fi
    else warn "cannot read /etc/os-release — OS unverified"; fi

    # NVIDIA driver + GPU + VRAM
    if need_cmd nvidia-smi; then
        local drv vram gpu
        drv="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
        vram="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc 0-9 || true)"
        gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)"
        ok "GPU: ${gpu:-?} (${vram:-?} MiB, driver ${drv:-?})"
        [ -n "$vram" ] && [ "$vram" -lt "$REQ_GPU_MIB" ] && { bad "VRAM ${vram} MiB < ${REQ_GPU_MIB} MiB required (~19 GB used at runtime)"; fatal=1; }
        case "$drv" in 595.*) ;; *) warn "driver ${drv:-?} is not the tested 595.x production branch" ;; esac
    else bad "nvidia-smi not found — NVIDIA driver missing (system-level prereq)"; fatal=1; fi

    # CUDA toolkit (nvcc)
    local NVCC; NVCC="$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)"
    if [ -x "$NVCC" ]; then
        local cv; cv="$("$NVCC" --version 2>/dev/null | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}' || true)"
        if [ "$cv" = "12.9" ]; then ok "CUDA toolkit: nvcc $cv ($NVCC)"
        else warn "nvcc reports CUDA $cv — tested on 12.9 (the Qwen3.5/sm_120a path is version-sensitive)"; fi
        export PATH="$(dirname "$NVCC"):$PATH"
    else bad "nvcc not found — install CUDA 12.9 toolkit (system-level prereq)"; fatal=1; fi

    # cuDNN (ONNX Runtime CUDA EP needs it)
    if ldconfig -p 2>/dev/null | grep -q 'libcudnn.so.9' || ls /usr/lib/x86_64-linux-gnu/libcudnn.so.9* >/dev/null 2>&1; then
        local cd; cd="$(dpkg-query -W -f='${Version}' cudnn9-cuda-12 2>/dev/null || echo '')"
        ok "cuDNN 9 present${cd:+ ($cd)}"
        [ -n "$cd" ] && [[ "$cd" != 9.23.* ]] && warn "cuDNN $cd — tested on 9.23.2.1 (any 9.x should work)"
    else warn "libcudnn.so.9 not found via ldconfig — the CUDA EP for SNAC/emotion2vec may fall back to CPU. Ensure cuDNN 9.x is installed and 'sudo ldconfig' has run."; fi

    # RAM
    local ram_gb; ram_gb="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || true)"
    if [ -n "$ram_gb" ]; then
        if [ "$ram_gb" -ge "$REQ_RAM_GB" ]; then ok "RAM: ${ram_gb} GB total"
        else bad "RAM ${ram_gb} GB < ${REQ_RAM_GB} GB required (MoE experts are mlock'd in host RAM)"; fatal=1; fi
    else warn "cannot read RAM size"; fi

    # Disk (free space in the repo filesystem)
    local free_gb; free_gb="$(df -PBG "$ATHENA_DIR" 2>/dev/null | awk 'NR==2{gsub("G","",$4);print $4}' || true)"
    if [ -n "$free_gb" ]; then
        if [ "$free_gb" -ge "$REQ_DISK_GB" ]; then ok "disk: ${free_gb} GB free"
        else bad "only ${free_gb} GB free < ${REQ_DISK_GB} GB needed (models are ~179 GB). Free space or --skip-models."; fatal=1; fi
    else warn "cannot read free disk space"; fi

    # Build tool-chain (fixable via apt — report exact command, do not auto-sudo)
    local miss_cmd=() ; local c
    for c in cmake g++ gcc git make wget curl tar; do need_cmd "$c" || miss_cmd+=("$c"); done
    # dev headers
    local miss_dev=()
    pkg-config --exists sdl2 2>/dev/null   || ls /usr/include/SDL2/SDL.h >/dev/null 2>&1 || miss_dev+=("libsdl2-dev")
    pkg-config --exists libcurl 2>/dev/null || ls /usr/include/curl/curl.h >/dev/null 2>&1 || miss_dev+=("libcurl4-openssl-dev")
    if [ ${#miss_cmd[@]} -gt 0 ] || [ ${#miss_dev[@]} -gt 0 ]; then
        bad "missing build prerequisites: ${miss_cmd[*]} ${miss_dev[*]}"
        # map bare commands to apt packages
        local pkgs=("${miss_dev[@]}")
        for c in "${miss_cmd[@]}"; do case "$c" in g++|gcc|make) pkgs+=("build-essential");; cmake) pkgs+=("cmake");; *) pkgs+=("$c");; esac; done
        # dedupe
        local uniq; uniq="$(printf '%s\n' "${pkgs[@]}" | sort -u | tr '\n' ' ')"
        warn "install with:  sudo apt update && sudo apt install -y $uniq"
        fatal=1
    else ok "build tool-chain: cmake $(cmake --version | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+'), $(g++ --version | head -1)"; fi

    # cmake >= 3.24 (needed for the 120a arch suffix)
    if need_cmd cmake; then
        local cmv; cmv="$(cmake --version 2>/dev/null | head -1 | grep -oE '[0-9]+\.[0-9]+' | head -1 || true)"
        awk -v v="$cmv" 'BEGIN{split(v,a,".");exit !(a[1]>3||(a[1]==3&&a[2]>=24))}' \
            || { bad "cmake $cmv < 3.24 — the '120a' CUDA arch suffix needs cmake ≥ 3.24"; fatal=1; }
    fi

    [ "$fatal" = 1 ] && die "preflight failed — resolve the ✗ items above and re-run."
    ok "preflight passed"
}

# ==============================================================================
#  STEP 1 — models
# ==============================================================================
step_models() {
    log "Step 1 — models  → models/ (+ SNAC → orpheus/)"
    mkdir -p models orpheus
    local HF="https://huggingface.co"

    # 1a. Qwen3.5-397B-A17B — 5 shards (~179 GB). Note the UD-Q3_K_XL/ subfolder.
    local i n
    for i in 1 2 3 4 5; do
        n="$(printf '0000%d-of-00005' "$i")"
        fetch "${HF}/unsloth/Qwen3.5-397B-A17B-GGUF/resolve/main/UD-Q3_K_XL/Qwen3.5-397B-A17B-UD-Q3_K_XL-${n}.gguf" \
              "models/Qwen3.5-397B-A17B-UD-Q3_K_XL-${n}.gguf"
    done

    # 1b. Orpheus 3B TTS, Whisper small.en, Silero VAD, SNAC decoder
    fetch "${HF}/unsloth/orpheus-3b-0.1-ft-GGUF/resolve/main/orpheus-3b-0.1-ft-UD-Q4_K_XL.gguf" "models/orpheus-3b-0.1-ft-UD-Q4_K_XL.gguf"
    fetch "${HF}/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin"                          "models/ggml-small.en.bin"
    fetch "${HF}/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin"                      "models/ggml-silero-v6.2.0.bin"
    fetch "${HF}/onnx-community/snac_24khz-ONNX/resolve/main/onnx/decoder_model_fp16.onnx"      "orpheus/snac24_dynamic_fp16.onnx"

    # 1c. emotion2vec → ONNX (optional; heavy Python conversion; MUST NOT abort the install)
    if [ "$DO_EMOTION" = 1 ]; then step_emotion || warn "emotion2vec step failed — continuing without emotion tags (non-fatal)."
    else warn "Step 1c (emotion2vec) skipped — Athena runs without emotion tags."; fi
}

step_emotion() {
    log "Step 1c — emotion2vec → ONNX (optional)"
    local out="models/emotion2vec_plus_large.onnx"
    if [ -f "$out" ] && [ "$FORCE" != 1 ]; then ok "$out present — skipping conversion (--force to redo)"; return 0; fi
    if ! need_cmd python3; then warn "python3 not found — skipping emotion2vec (Athena runs without it)"; return 0; fi
    if ! confirm "Convert emotion2vec now? (installs torch/funasr in a venv, ~5 GB, downloads a checkpoint)"; then
        warn "emotion2vec conversion declined — run 'python3 models/emotion2vec-reexport.py' later to enable tags."
        return 0
    fi
    local venv="models/.emotion-venv"
    ( set +eu; trap - ERR   # fully non-fatal: emotion tagging is optional (nounset OFF too)
      PIP=(python3 -m pip); userflag=""
      if python3 -m venv "$venv" 2>/dev/null && [ -f "$venv/bin/activate" ]; then
          # shellcheck disable=SC1091
          . "$venv/bin/activate"
      else
          warn "python venv unavailable (apt install python3-venv) — isolated --user install"
          userflag="--user --break-system-packages"    # PEP-668 on Ubuntu 24.04
      fi
      # shellcheck disable=SC2086
      "${PIP[@]}" install $userflag -U -q funasr torch torchaudio modelscope onnx onnxruntime \
          || { warn "pip install for emotion2vec failed — skipping (non-fatal)"; exit 0; }
      ( cd models && python3 emotion2vec-reexport.py 2>&1 | tee emotion2vec-reexport.out ) \
          || warn "conversion script errored — see models/emotion2vec-reexport.out"
    ) || true
    if [ -f "$out" ]; then ok "emotion2vec_plus_large.onnx created ($(human "$(stat -c%s "$out")"))"
    else warn "emotion2vec ONNX not produced — Athena will run without emotion tags (non-fatal)."; fi
}

# ==============================================================================
#  STEP 2 — ONNX Runtime 1.27.0
# ==============================================================================
step_ort() {
    log "Step 2 — ONNX Runtime ${ORT_VER} (CUDA 12 GPU)"
    if [ -f "${ORT_ROOT}/lib/libonnxruntime.so" ] && [ "$FORCE" != 1 ]; then
        ok "${ORT_DIR}/ already extracted (--force to re-extract)"
    else
        fetch "$ORT_URL" "$ATHENA_DIR/$ORT_TGZ"
        sub "extracting ${ORT_TGZ}"
        tar xzf "$ATHENA_DIR/$ORT_TGZ" -C "$ATHENA_DIR"
        [ -f "${ORT_ROOT}/lib/libonnxruntime.so" ] || die "ORT extract did not produce ${ORT_DIR}/lib/libonnxruntime.so"
        # Repair the SONAME symlink chain if the archive/copy flattened it (CHANGES.MD §23.2).
        ( cd "${ORT_ROOT}/lib"
          for l in libonnxruntime.so libonnxruntime.so.1; do
              if [ -f "$l" ] && ! [ -L "$l" ] && head -c7 "$l" | grep -q IntxLNK; then rm -f "$l"; fi
          done
          [ -e libonnxruntime.so.1 ] || ln -sf "$(basename "$(ls libonnxruntime.so.1.* 2>/dev/null | head -1)")" libonnxruntime.so.1 2>/dev/null || true
          [ -e libonnxruntime.so ]   || ln -sf libonnxruntime.so.1 libonnxruntime.so 2>/dev/null || true ) || true
        ok "extracted → ${ORT_DIR}/"
        rm -f "$ATHENA_DIR/$ORT_TGZ"
    fi
    # Make the loader aware of the ORT lib dir (belt-and-suspenders; the binaries
    # also carry an RUNPATH). Needs sudo; non-fatal if unavailable.
    if [ -w /etc/ld.so.conf.d ] || sudo -n true 2>/dev/null; then
        echo "${ORT_ROOT}/lib" | sudo tee /etc/ld.so.conf.d/athena-onnxruntime.conf >/dev/null 2>&1 \
            && sudo ldconfig 2>/dev/null && ok "registered ORT lib path with ldconfig" \
            || warn "could not run ldconfig (RUNPATH still covers runtime resolution)"
    else warn "skipped ldconfig (no sudo) — binaries carry an RUNPATH, so this is usually fine"; fi
}

# ==============================================================================
#  STEP 3 — llama.cpp (latest master) → llama-server (Orpheus TTS backend)
# ==============================================================================
step_llama() {
    log "Step 3 — llama.cpp (master) → llama-server"
    local bin="llama.cpp/build/bin/llama-server"
    if bin_ok "$bin" && [ "$FORCE" != 1 ]; then ok "llama-server present & links resolve — skipping build (--force to rebuild)"; return 0; fi
    [ -d llama.cpp/.git ] || { sub "cloning llama.cpp"; git clone --depth 1 "$LLAMA_REPO" llama.cpp; }
    [ "$FORCE" = 1 ] && rm -rf llama.cpp/build   # a stale CMakeCache (wrong arch) survives otherwise
    sub "configuring (CUDA arch ${CUDA_ARCH})"
    cmake -S llama.cpp -B llama.cpp/build \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DGGML_CUDA_FORCE_CUBLAS=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_F16=ON \
        -DGGML_SCHED_MAX_COPIES=1 -DGGML_NATIVE=ON -DGGML_LTO=ON -DGGML_CUDA_GRAPHS=ON \
        -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_UI=OFF -DLLAMA_OPENSSL=OFF >/dev/null
    sub "building (-j ${JOBS}) — this takes a while"
    cmake --build llama.cpp/build -j "$JOBS" --target llama-server
    [ -x "$bin" ] || die "llama-server was not produced"
    ok "built $bin"
}

# ==============================================================================
#  STEP 4 — whisper.cpp (pinned) + ATHENA patches → whisper-talk-llama (the brain)
# ==============================================================================
step_whisper() {
    log "Step 4 — whisper.cpp (pinned ${WHISPER_PIN:0:8}) + patches → whisper-talk-llama"
    local bin="whisper.cpp/build/bin/whisper-talk-llama"
    [ -d "$ORT_ROOT/lib" ] || die "ONNX Runtime not found at $ORT_ROOT — run Step 2 first (or drop --skip-ort)."

    # 1. Ensure the repo exists AND sits at the tested pin — enforced even when a
    #    binary already exists (an existing master checkout STT-regresses, §23.3).
    local need_build=0
    if [ ! -d whisper.cpp/.git ]; then
        sub "cloning whisper.cpp"; git clone "$WHISPER_REPO" whisper.cpp; need_build=1
    fi
    local head; head="$(git -C whisper.cpp rev-parse HEAD 2>/dev/null || echo '')"
    if [ "$head" != "$WHISPER_PIN" ]; then
        warn "whisper.cpp at ${head:0:8} ≠ tested pin — checking out the pin (STT regresses on master, §23.3)"
        git -C whisper.cpp fetch -q origin "$WHISPER_PIN" 2>/dev/null || git -C whisper.cpp fetch -q --unshallow 2>/dev/null || true
        git -C whisper.cpp checkout -q -f "$WHISPER_PIN"; need_build=1
    fi
    # 2. Always (re)apply the FIVE ATHENA files BEFORE configure (a checkout wipes them).
    sub "applying patches/talk-llama/ → whisper.cpp/examples/talk-llama/"
    cp patches/talk-llama/* whisper.cpp/examples/talk-llama/
    grep -q 'gpu_context_healthy\|silero-dbg' whisper.cpp/examples/talk-llama/talk-llama.cpp \
        || die "patch copy failed — examples/talk-llama/talk-llama.cpp is not the ATHENA version"
    # 3. Skip the build only if the binary is present, links resolve, AND nothing changed.
    if bin_ok "$bin" && [ "$need_build" = 0 ] && [ "$FORCE" != 1 ]; then
        ok "whisper-talk-llama present, at the pin, links resolve — skipping build (--force to rebuild)"; return 0
    fi
    { [ "$FORCE" = 1 ] || [ "$need_build" = 1 ]; } && rm -rf whisper.cpp/build   # sources/arch changed → clean
    sub "configuring (CUDA arch ${CUDA_ARCH}, ONNXRUNTIME_ROOT set, SDL2 on)"
    cmake -S whisper.cpp -B whisper.cpp/build \
        -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
        -DONNXRUNTIME_ROOT="$ORT_ROOT" \
        -DGGML_CUDA_FORCE_CUBLAS=OFF -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_F16=ON \
        -DGGML_SCHED_MAX_COPIES=1 -DGGML_NATIVE=ON -DGGML_LTO=ON -DWHISPER_SDL2=ON \
        -DGGML_CUDA_GRAPHS=ON -DCMAKE_BUILD_TYPE=Release >/dev/null
    sub "building (-j ${JOBS}) — this takes a while"
    cmake --build whisper.cpp/build -j "$JOBS" --target whisper-talk-llama
    [ -x "$bin" ] || die "whisper-talk-llama was not produced"
    ldd "$bin" 2>/dev/null | grep -q onnxruntime || warn "binary does not link onnxruntime — emotion2vec may be compiled out"
    ok "built $bin"
}

# ==============================================================================
#  STEP 4b — orpheus-speak (ATHENA's own TTS engine)
# ==============================================================================
step_orpheus() {
    log "Step 4b — orpheus-speak (ATHENA TTS engine)"
    local bin="orpheus/build/orpheus-speak"
    if bin_ok "$bin" && [ "$FORCE" != 1 ]; then ok "orpheus-speak present & links resolve — skipping build (--force to rebuild)"; return 0; fi
    [ -d "$ORT_ROOT/lib" ] || die "ONNX Runtime not found at $ORT_ROOT — run Step 2 first."
    [ "$FORCE" = 1 ] && rm -rf orpheus/build
    sub "configuring + building (-j ${JOBS})"
    cmake -S orpheus -B orpheus/build -DONNXRUNTIME_ROOT="$ORT_ROOT" -DCMAKE_BUILD_TYPE=Release >/dev/null
    cmake --build orpheus/build -j "$JOBS"
    [ -x "$bin" ] || die "orpheus-speak was not produced"
    ok "built $bin"
}

# ==============================================================================
#  Desktop launchers — generate from *.desktop.in with the real repo path
# ==============================================================================
gen_desktop() {
    local t
    for t in "Athena 397B.desktop" "athena-autostart.desktop"; do
        [ -f "$t.in" ] || continue
        if sed "s|@ATHENA_DIR@|$ATHENA_DIR|g" "$t.in" > "$t" 2>/dev/null; then
            sub "desktop launcher: $t  (paths → $ATHENA_DIR)"
        fi
    done
}

# ==============================================================================
#  STEP 5 — integrity check
# ==============================================================================
integrity() {
    log "Step 5 — component integrity check"
    local miss=0 warn_n=0

    # binary must exist, be executable, AND have all shared libs resolve (bin_ok)
    check_bin() { if bin_ok "$1"; then ok "binary: $1"; elif [ -x "$1" ]; then bad "$1 exists but a shared lib is UNRESOLVED (ldd 'not found') — rebuild / re-run Step 2 (ldconfig)"; miss=1; else bad "MISSING binary: $1"; miss=1; fi; }
    # model must be non-empty AND pass the container sniff (not an HTML/JSON error page)
    check_file() { if [ ! -s "$1" ]; then bad "MISSING $2: $1"; miss=1; elif verify_container "$1"; then ok "$(printf '%-9s' "$2") $1 ($(human "$(stat -c%s "$1")"))"; else bad "$2 corrupt/not-a-model (error page or wrong type): $1 — delete it and re-run"; miss=1; fi; }
    check_opt() { if [ -s "$1" ]; then ok "$(printf '%-9s' "$2") $1"; else warn "optional $2 absent: $1"; warn_n=$((warn_n+1)); fi; }

    check_bin "llama.cpp/build/bin/llama-server"
    check_bin "whisper.cpp/build/bin/whisper-talk-llama"
    check_bin "orpheus/build/orpheus-speak"
    # launcher preflight hard-requires speak-daemon.sh (exit 1 if absent), though it is
    # dormant at runtime. Plain presence check — it is a repo script, not a container.
    if [ -s "speak-daemon.sh" ]; then ok "$(printf '%-9s' 'speak-dmn') speak-daemon.sh"; else bad "MISSING speak-daemon.sh (launcher preflight requires it)"; miss=1; fi
    [ -f "${ORT_ROOT}/lib/libonnxruntime.so" ] && ok "ONNX RT:  ${ORT_DIR}/" || { bad "MISSING ONNX Runtime: ${ORT_DIR}/"; miss=1; }

    local i n
    for i in 1 2 3 4 5; do
        n="$(printf '0000%d-of-00005' "$i")"
        check_file "models/Qwen3.5-397B-A17B-UD-Q3_K_XL-${n}.gguf" "Qwen s${i}"
    done
    check_file "models/orpheus-3b-0.1-ft-UD-Q4_K_XL.gguf" "Orpheus"
    check_file "models/ggml-small.en.bin"                 "Whisper"
    check_file "models/ggml-silero-v6.2.0.bin"            "Silero"
    check_file "orpheus/snac24_dynamic_fp16.onnx"         "SNAC"
    check_opt  "models/emotion2vec_plus_large.onnx"       "emotion2vec"

    # runtime linkage: whisper-talk-llama must resolve libonnxruntime (RUNPATH/ldconfig)
    if [ -x whisper.cpp/build/bin/whisper-talk-llama ]; then
        if ldd whisper.cpp/build/bin/whisper-talk-llama 2>/dev/null | grep -q 'onnxruntime.*not found'; then
            bad "whisper-talk-llama cannot resolve libonnxruntime at runtime — re-run Step 2 (ldconfig) or rebuild."; miss=1
        fi
    fi

    hr
    if [ "$miss" = 0 ]; then
        printf '%s\n' "${G}${B} ✓ ATHENA is ready to run.${Z}"
        [ "$warn_n" -gt 0 ] && sub "(${warn_n} optional component(s) absent — see warnings above)"
        cat <<EOF

   Launch:   ./launch-athena-397b.sh
   (production defaults; forensics OFF, platform_profile=performance — CHANGES.MD §23)
EOF
        return 0
    else
        die "integrity check found missing components (see ✗ above). Re-run the relevant step (nothing already-done is repeated)."
    fi
}

# ==============================================================================
#  MAIN
# ==============================================================================
print_requirements
if [ "$CHECK_ONLY" = 1 ]; then integrity; exit $?; fi

if [ "$ASSUME_YES" != 1 ] && [ -t 0 ]; then
    read -r -p "$(printf '    %sProceed with install?%s [Y/n] ' "$B" "$Z")" a || true   # Ctrl-D → clean abort, not FATAL
    [[ "${a:-y}" =~ ^[Nn] ]] && { echo "aborted."; exit 0; }
fi

# Dispatch with plain if/fi — NOT `f && g || h`: a function invoked in an &&/||
# test context runs with `set -e` suppressed inside it, so a bare failing command
# would not abort. if/then keeps errexit + the ERR trap live within each step.
preflight
if [ "$DO_MODELS"  = 1 ]; then step_models;  else warn "Step 1 (models) skipped"; fi
if [ "$DO_ORT"     = 1 ]; then step_ort;      else warn "Step 2 (ONNX Runtime) skipped"; fi
if [ "$DO_LLAMA"   = 1 ]; then step_llama;    else warn "Step 3 (llama.cpp) skipped"; fi
if [ "$DO_WHISPER" = 1 ]; then step_whisper;  else warn "Step 4 (whisper.cpp) skipped"; fi
if [ "$DO_ORPHEUS" = 1 ]; then step_orpheus;  else warn "Step 4b (orpheus-speak) skipped"; fi
gen_desktop
integrity
