// silero-endpointer.h — streaming end-of-turn detector backed by whisper.cpp's
// built-in Silero VAD (whisper_vad_* API). Owns a whisper_vad_context, feeds
// captured audio frame-aligned through whisper_vad_detect_speech_no_reset()
// (LSTM state preserved across calls), reads the per-frame probability stream,
// and drives the pure SileroTurnState machine to decide turn-end.
//
// Verified against whisper.cpp (include/whisper.h, src/whisper.cpp):
//   - whisper_vad_detect_speech_no_reset(ctx, samples, n): streaming inference
//   - whisper_vad_reset_state(ctx): clear LSTM between utterances
//   - whisper_vad_n_probs(ctx) / whisper_vad_probs(ctx): per-call frame probs
//     (probs.resize(n_chunks) each call -> per-call, not cumulative)
//   - n_probs == ceil(n_samples / n_window); Silero @16kHz uses n_window=512
//     (32 ms/frame). We feed exact multiples of the window (carry remainder)
//     so the last frame is never zero-padded; the actual window is re-derived
//     at runtime and a mismatch is reported.
//
// Depends only on whisper.h (+ the pure header). The audio source and the
// SDL poll are injected as template params so this header stays decoupled from
// common-sdl.

#pragma once

#include "silero-turn-state.h"

#include "whisper.h"

#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace athena {

// Quiet whisper's INFO/DEBUG chatter (whisper_vad_detect_speech_no_reset emits
// an INFO line every call — a flood at streaming cadence). WARN/ERROR always
// pass; INFO/DEBUG only when verbose. Installed once at init().
inline bool& silero_log_verbose() { static bool v = false; return v; }
inline void  silero_log_cb(ggml_log_level level, const char* text, void* /*ud*/) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN || silero_log_verbose()) {
        fputs(text, stderr);
    }
}

class SileroEndpointer {
public:
    struct Config {
        std::string      model_path;        // ggml-silero-v6.2.0.bin (required)
        bool             use_gpu    = false; // Silero runs on CPU: whisper.cpp defaults
                                            // VAD to CPU and the GPU graph won't schedule
                                            // (the model is 0.88 MB / a 128-unit LSTM, so
                                            // CPU is microseconds per frame).
        int              gpu_device = 0;
        int              n_threads  = 4;
        SileroTurnConfig turn;              // threshold / min_speech_ms / silence_ms
        int              poll_ms    = 100;  // capture + inference cadence
        bool             verbose    = false; // pass whisper VAD INFO logs through the filter
        bool             install_log_filter = true; // silence per-call streaming INFO spam
        bool             debug      = false; // per-poll [silero-dbg] trace (prob + accumulators)
    };

    SileroEndpointer() = default;
    ~SileroEndpointer() { free(); }
    SileroEndpointer(const SileroEndpointer&)            = delete;
    SileroEndpointer& operator=(const SileroEndpointer&) = delete;

    // Returns false if no model path or the model fails to load — caller should
    // then fall back to the energy VAD.
    bool init(const Config& cfg) {
        free();
        cfg_ = cfg;
        if (cfg_.model_path.empty()) return false;

        whisper_vad_context_params cp = whisper_vad_default_context_params();
        cp.use_gpu    = cfg_.use_gpu;   // CPU by default (see Config) -- GPU VAD aborts
        cp.gpu_device = cfg_.gpu_device;
        cp.n_threads  = cfg_.n_threads;

        vctx_ = whisper_vad_init_from_file_with_params(cfg_.model_path.c_str(), cp);
        if (!vctx_) return false;

        // Installed AFTER the load so the one-time model banner prints, but the
        // per-call INFO from whisper_vad_detect_speech_no_reset is suppressed
        // during streaming (WARN/ERROR always pass; INFO/DEBUG only if verbose).
        if (cfg_.install_log_filter) {
            silero_log_verbose() = cfg_.verbose;
            whisper_log_set(silero_log_cb, nullptr);
        }

        state_       = SileroTurnState(cfg_.turn);
        carry_.clear();
        win_         = kSileroWindow;
        win_ms_      = 1000.0f * (float) kSileroWindow / (float) WHISPER_SAMPLE_RATE;
        win_checked_ = false;
        return true;
    }

    bool ready() const { return vctx_ != nullptr; }

    void free() {
        if (vctx_) { whisper_vad_free(vctx_); vctx_ = nullptr; }
        carry_.clear();
    }

    // Feed one captured audio chunk (must be WHISPER_SAMPLE_RATE mono f32).
    // Buffers a sub-frame remainder so successive chunks tile the window grid
    // exactly. Returns the aggregate turn-state step for the frames consumed.
    SileroTurnState::Step feed(const float* samples, int n) {
        SileroTurnState::Step none{ state_.phase() == SileroTurnState::Phase::InSpeech, false, false };
        if (!vctx_ || n <= 0) return none;

        carry_.insert(carry_.end(), samples, samples + n);
        const int nframes = (int) carry_.size() / win_;
        if (nframes <= 0) return none;

        const int feed_n = nframes * win_;
        if (!whisper_vad_detect_speech_no_reset(vctx_, carry_.data(), feed_n)) {
            carry_.erase(carry_.begin(), carry_.begin() + feed_n);
            return none;
        }

        const int    np = whisper_vad_n_probs(vctx_);
        const float* p  = whisper_vad_probs(vctx_);

        if (!win_checked_ && np > 0) {
            const int obs = feed_n / np;                 // actual samples/frame
            if (obs != win_) {
                fprintf(stderr, "[silero] model window %d samples != assumed %d; "
                                "using %d (%.1f ms/frame)\n",
                        obs, win_, obs, 1000.0f * obs / (float) WHISPER_SAMPLE_RATE);
                win_ms_ = 1000.0f * (float) obs / (float) WHISPER_SAMPLE_RATE;
                // NOTE: carry was aligned to the assumed window; this is exact
                // when obs divides the assumed window (e.g. 256|512). A window
                // that is not a divisor would need re-alignment — not the case
                // for Silero @16kHz (512).
            }
            win_checked_ = true;
        }

        last_max_prob_ = 0.0f;
        for (int i = 0; i < np; ++i) if (p[i] > last_max_prob_) last_max_prob_ = p[i];

        SileroTurnState::Step agg = (np > 0 && p) ? state_.push_frames(p, np, win_ms_) : none;
        carry_.erase(carry_.begin(), carry_.begin() + feed_n);
        return agg;
    }

    // Block until the user's utterance ends (true) or shutdown/Ctrl-C (false).
    //   audio      : object with .get(ms, std::vector<float>&) (e.g. audio_async)
    //   is_running : refreshed from poll(); loop exits when it goes false
    //   poll       : pump UI events, return is_running (e.g. sdl_poll_events)
    //   on_trailing: optional; called once when speech first turns to silence,
    //                returns the silence target (ms) for this utterance
    //                (prosody hook). Return <=0 to keep the configured default.
    template <class AudioT, class PollFn>
    bool wait_for_endpoint(AudioT& audio, bool& is_running, PollFn poll,
                           const std::function<int()>& on_trailing = {}) {
        if (!vctx_) return false;

        state_.reset();
        whisper_vad_reset_state(vctx_);
        carry_.clear();

        std::vector<float> chunk;
        while (is_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(cfg_.poll_ms));
            is_running = poll();
            if (!is_running) return false;

            audio.get(cfg_.poll_ms, chunk);
            if (chunk.empty()) continue;

            const SileroTurnState::Step s = feed(chunk.data(), (int) chunk.size());

            if (cfg_.debug) {
                fprintf(stderr, "main: [silero-dbg] %-8s maxp=%.2f speech=%.0f sil=%.0f run=%.0f tgt=%d\n",
                        SileroTurnState::phase_name(state_.phase()), last_max_prob_,
                        state_.speech_ms(), state_.silence_ms(), state_.run_ms(),
                        state_.silence_target_ms());
            }

            if (s.entered_trailing && on_trailing) {
                state_.set_silence_target_ms(on_trailing());
            }
            if (s.endpoint) {
                if (cfg_.debug)
                    fprintf(stderr, "main: [silero-dbg] ENDPOINT (speech=%.0fms silence=%.0fms)\n",
                            state_.speech_ms(), state_.silence_ms());
                return true;
            }
        }
        return false;
    }

    const SileroTurnState& state() const { return state_; }

private:
    // Silero v4/v5/v6 process 512-sample windows at 16 kHz (32 ms/frame);
    // re-derived at runtime in feed() and reported if a model differs.
    static constexpr int kSileroWindow = 512;

    Config              cfg_{};
    whisper_vad_context* vctx_ = nullptr;
    SileroTurnState     state_{};
    std::vector<float>  carry_;
    float               last_max_prob_ = 0.0f; // max prob of the last chunk (debug)
    int                 win_         = kSileroWindow;
    float               win_ms_      = 1000.0f * (float) kSileroWindow / 16000.0f;
    bool                win_checked_ = false;
};

} // namespace athena
