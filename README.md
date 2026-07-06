# Athena

A fully offline, privacy-first voice assistant. It runs speech-to-text, a large language model, and neural text-to-speech entirely in C++ on one consumer GPU plus system RAM. No cloud, no telemetry, no API keys, and no Python at runtime.

Athena listens on the microphone, transcribes with Whisper, replies with a Qwen mixture-of-experts model, and speaks the reply through the Orpheus neural TTS voice. She reads the emotion in your voice, remembers across sessions, and can be interrupted mid-sentence (talk over her and she stops).

This build is configured for a desktop with an RTX 5080 (16 GB) and a Bose SoundLink over Bluetooth. It replaces the original 397B reference setup, whose 192 GB RAM requirement does not fit here. See `CHANGES.MD` section 26 for the retarget details.

## Demo

<p align="center">
  <a href="https://www.youtube.com/watch?v=8HuRUpJ4_as&t=237s">
    <img src="https://img.youtube.com/vi/8HuRUpJ4_as/maxresdefault.jpg" alt="ATHENA — live demo video" width="720">
  </a>
</p>
<p align="center"><em>Watch the demo on YouTube</em>, a live voice-to-voice session (offline, single GPU).</p>

## This build

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 5080, 16 GB (Blackwell, `sm_120`) |
| RAM | 62 GB |
| CUDA | 13.3 toolkit, built with the `gcc-14` host compiler |
| OS / audio | Arch-based (CachyOS), PipeWire |
| Speaker + mic | Bose Revolve SoundLink over Bluetooth (HFP/mSBC duplex) |

| Role | Model | Location |
|---|---|---|
| LLM | Qwen3.6-35B-A3B UD-Q4_K_XL (~22 GB) | `~/.models/` |
| TTS | Orpheus 3B UD-Q4_K_XL | `models/` |
| TTS vocoder | SNAC 24 kHz ONNX (fp16) | `orpheus/` |
| STT | Whisper large-v3-turbo | `models/` |
| VAD | Silero v6.2 | `models/` |

Peak VRAM with the defaults is about 14.3 GB of 16.3 GB (Qwen attention layers on the GPU, its experts in system RAM via `--cpu-moe`, six expert layers offloaded to the GPU for speed, whisper and the Orpheus backend alongside, SNAC on the CPU). That leaves roughly 2 GB for the desktop compositor.

## How it works

Three processes run at once, all started by `launch-athena-397b.sh`:

1. **whisper-talk-llama** captures the mic, transcribes with Whisper, and runs the Qwen chat model in-process. It also does voice-activity detection, prosodic endpointing, cross-session memory, and barge-in. It streams each finished sentence to a trigger file.
2. **llama-server** hosts the Orpheus TTS model on `127.0.0.1:8080`. This build reuses the system `llama-server` binary, so llama.cpp is not built in-tree.
3. **orpheus-speak** watches the trigger file, asks `llama-server` for SNAC audio tokens over HTTP, decodes them to PCM, and plays them to the speaker.

Two models run concurrently: Orpheus serves TTS on port 8080, and the conversational Qwen model runs inside `whisper-talk-llama`, not over HTTP.

## Setup

### Dependencies

Install the NVIDIA driver plus the CUDA 13.3 toolkit at `/opt/cuda`. CUDA 13.3 rejects host compilers newer than `gcc-15`, so the builds use `gcc-14`. On CachyOS / Arch:

```bash
sudo pacman -S cuda gcc14 sdl2 curl cmake git wget \
               pipewire pipewire-pulse pipewire-alsa alsa-utils
```

The `pipewire-alsa` package provides the ALSA `pulse` plugin that TTS playback routes through.

### Download models and sources

```bash
export ATHENA="$PWD"          # run from the repo root

# ONNX Runtime (SNAC decode), CUDA 12 build
wget https://github.com/microsoft/onnxruntime/releases/download/v1.27.0/onnxruntime-linux-x64-gpu_cuda12-1.27.0.tgz
tar xzf onnxruntime-linux-x64-gpu_cuda12-1.27.0.tgz

# whisper.cpp (current master; the ATHENA CMake patch builds against it)
git clone https://github.com/ggml-org/whisper.cpp.git

# TTS + STT + VAD models
wget -O models/orpheus-3b-0.1-ft-UD-Q4_K_XL.gguf https://huggingface.co/unsloth/orpheus-3b-0.1-ft-GGUF/resolve/main/orpheus-3b-0.1-ft-UD-Q4_K_XL.gguf
wget -O models/ggml-large-v3-turbo.bin           https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
wget -O models/ggml-silero-v6.2.0.bin            https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin
wget -O orpheus/snac24_dynamic_fp16.onnx         https://huggingface.co/onnx-community/snac_24khz-ONNX/resolve/main/onnx/decoder_model_fp16.onnx
```

Place the Qwen3.6-35B-A3B UD-Q4_K_XL GGUF (unsloth) at `~/.models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf`, or point `QWEN_MODEL` at it. The launcher uses the system `llama-server` for the Orpheus backend; install a recent llama.cpp `llama-server` (b9355 or newer) on `PATH`, or build llama.cpp and set `LLAMA_SERVER`.

### Build

```bash
export ATHENA="$PWD"
export ORT="$ATHENA/onnxruntime-linux-x64-gpu_cuda12-1.27.0"
export PATH=/opt/cuda/bin:$PATH CUDACXX=/opt/cuda/bin/nvcc

# whisper-talk-llama (STT + in-process Qwen). Copy the five ATHENA patch files first.
cd "$ATHENA/whisper.cpp"
cp ../patches/talk-llama/* examples/talk-llama/
CC=gcc-14 CXX=g++-14 cmake -B build \
  -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DCMAKE_CUDA_HOST_COMPILER=g++-14 -DCMAKE_C_COMPILER=gcc-14 -DCMAKE_CXX_COMPILER=g++-14 \
  -DCUDAToolkit_ROOT=/opt/cuda -DONNXRUNTIME_ROOT="$ORT" \
  -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_CUDA_F16=ON -DGGML_NATIVE=ON -DGGML_LTO=ON \
  -DWHISPER_SDL2=ON -DGGML_CUDA_GRAPHS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target whisper-talk-llama -j

# orpheus-speak (SNAC decoder + player)
cd "$ATHENA/orpheus"
CC=gcc-14 CXX=g++-14 cmake -B build -DONNXRUNTIME_ROOT="$ORT" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The five files in `patches/talk-llama/` are the ATHENA-custom `talk-llama.cpp`, memory and Silero headers, and `CMakeLists.txt`. Copy them into `whisper.cpp/examples/talk-llama/` before configuring, and re-copy after any `git checkout` in that tree. If you edit the build-tree copies, sync them back to `patches/`.

## Run

Connect the Bose speaker, then:

```bash
./launch-athena-397b.sh
```

The launcher tunes the GPU and CPU (it uses `sudo` and will prompt for a password), forces the Bose into HFP duplex, starts the three processes, and blocks on `whisper-talk-llama`. Say "Goodbye Athena" to exit cleanly, which triggers a memory-consolidation pass. Ctrl+C does not write memory.

Startup prints `audio: aec mode` and the resolved sink and source, so you can confirm the Bose is wired before speaking.

## Configuration

All modes are environment variables read near the top of the launcher, not command-line flags.

| Variable | Default | Effect |
|---|---|---|
| `ATHENA_N_CPU_MOE` | `34` | Layers (of 40) whose experts stay in RAM. Lower = more experts on the GPU, faster, more VRAM. `0` = all experts on CPU (safest). Above ~32 risks OOM if the desktop GPU load spikes. |
| `ATHENA_BARGE` | `1` | Barge-in. Talk over Athena to stop her mid-reply. `0` disables it. |
| `ATHENA_AUDIO` | `aec` when barge-in is on with a Bluetooth speaker, else `direct` | `aec` routes both streams through webrtc echo cancellation bound to the Bose, so she does not interrupt herself. `direct` skips it. |
| `HEADSET_PATTERN` | `bluez` | PipeWire node-name pattern for the speaker and mic. |
| `ATHENA_BT_FORCE_HFP` | `1` | Force the Bluetooth card into HFP duplex at startup (needed so the mic and speaker coexist). Restored on exit. |
| `QWEN_MODEL` | `~/.models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` | Conversational model path. |
| `WHISPER_MODEL` | `models/ggml-large-v3-turbo.bin` | STT model path. |
| `LLAMA_SERVER` | system `llama-server` | Orpheus TTS backend binary. |
| `ATHENA_DIAG` | `1` | Per-run GPU and kernel diagnostics under `athena-diag/`. |
| `ATHENA_TTS_CAPTURE` | `1` | Per-session TTS token/audio dump for garble diagnosis. |

## Notes

**Bluetooth audio is phone-quality.** A single Bluetooth speaker cannot do hi-fi A2DP output and microphone input at the same time. Athena holds the mic open continuously, so the Bose runs in HFP/mSBC duplex: 16 kHz mono in both directions.

**Barge-in tuning.** Echo cancellation over Bluetooth is imperfect. If Athena cuts herself off, raise the thresholds in the `BARGE_FLAG` line of the launcher (for example `--barge-rms 0.004 --barge-ms 300`). If she is hard to interrupt, lower them. A wired headset (`ATHENA_AUDIO=direct`, `HEADSET_PATTERN=<usb-name>`) gives the most reliable barge-in through physical isolation, no echo cancellation needed.

**Emotion detection** (emotion2vec) is optional and off by default here. It activates when an `emotion2vec_plus_large.onnx` model is present in `models/` and `ATHENA_EMOTION_ONNX` points at it.

## License

MIT licensed. Orpheus weights: Llama 3.2 community license. SNAC codec: MIT. Qwen: Apache-2.0.
