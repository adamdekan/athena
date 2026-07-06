// silero-turn-state.h — pure end-of-turn state machine for streaming VAD.
//
// This header has NO dependencies (no whisper, no ggml, no SDL) so it can be
// unit-tested in isolation. It consumes a stream of per-frame speech
// probabilities (e.g. from Silero) plus each frame's duration in ms, and
// decides when an utterance has ended: speech, followed by a configurable run
// of trailing silence. Prosody (or anything else) may override the trailing
// target per-utterance via set_silence_target_ms().
//
// Why this exists: the legacy energy VAD (vad_simple) measured silence
// *relative* to a sliding window, which degenerates on sustained quiet. Silero
// gives an absolute per-frame speech probability, so silence is just
// "probability below threshold" — and turn-end is "that, sustained for N ms".
//
// Frame accounting is in milliseconds (frame_ms), not frame counts, so the
// caller can feed variable-size blocks; the only requirement is that each
// captured frame is pushed exactly once (the wrapper guarantees this).

#pragma once

namespace athena {

struct SileroTurnConfig {
    float threshold         = 0.5f; // speech if prob >= threshold (Silero default 0.5)
    int   min_speech_run_ms = 100;  // HYSTERESIS: consecutive speech required to confirm
                                    // a speech onset/resume. Single-frame Silero spikes
                                    // (noise, breath) are shorter than this and so cannot
                                    // reset the trailing-silence clock. Mirrors Silero's
                                    // own min_speech_duration_ms (250). 0 = no hysteresis.
    int   min_speech_ms     = 120;  // total cumulative speech in the utterance before an
                                    // endpoint may fire (a second, weaker blip guard)
    int   silence_ms        = 700;  // trailing silence (ms) that ends a turn, unless
                                    // overridden per-utterance (e.g. by prosody)
};

class SileroTurnState {
public:
    enum class Phase { WaitingForSpeech, InSpeech, Trailing };

    struct Step {
        bool in_speech;        // currently inside a speech run
        bool entered_trailing; // this step crossed speech -> silence (set target now)
        bool endpoint;         // utterance just ended (sustained trailing silence)
    };

    explicit SileroTurnState(SileroTurnConfig cfg = {}) : cfg_(cfg) { reset(); }

    // Begin a fresh utterance.
    void reset() {
        phase_      = Phase::WaitingForSpeech;
        speech_ms_  = 0.0f;
        silence_ms_ = 0.0f;
        run_ms_     = 0.0f;
        last_prob_  = 0.0f;
        target_ms_  = cfg_.silence_ms;
    }

    // Override the trailing-silence target for the current utterance. Intended
    // to be called by the orchestrator right after a Step reports
    // entered_trailing (e.g. prosody: shorter on a turn-final fall, longer on a
    // mid-thought rise). A value <= 0 is ignored (keeps the default).
    void set_silence_target_ms(int ms) { if (ms > 0) target_ms_ = ms; }

    Phase phase()             const { return phase_; }
    float speech_ms()         const { return speech_ms_; }
    float silence_ms()        const { return silence_ms_; }
    float run_ms()            const { return run_ms_; }
    float last_prob()         const { return last_prob_; }
    int   silence_target_ms() const { return target_ms_; }
    const SileroTurnConfig& config() const { return cfg_; }

    static const char* phase_name(Phase p) {
        switch (p) {
            case Phase::WaitingForSpeech: return "wait";
            case Phase::InSpeech:         return "speech";
            case Phase::Trailing:         return "trailing";
        }
        return "?";
    }

    // Advance by a single VAD frame.
    Step push_frame(float prob, float frame_ms) {
        last_prob_ = prob;
        const bool speech = prob >= cfg_.threshold;

        // Consecutive-speech run (hysteresis). A speech frame extends the run; any
        // silence frame breaks it. Onset and resume require run >= min_speech_run_ms
        // so isolated Silero spikes (noise/breath, typically 1-2 frames) are not
        // treated as speech and cannot reset the silence clock.
        if (speech) run_ms_ += frame_ms;
        else        run_ms_  = 0.0f;
        const bool confirmed = run_ms_ >= (float) cfg_.min_speech_run_ms;

        Step s{false, false, false};
        switch (phase_) {
            case Phase::WaitingForSpeech:
                if (confirmed) {
                    phase_      = Phase::InSpeech;
                    speech_ms_ += run_ms_;            // count the whole confirmed run once
                }
                break;

            case Phase::InSpeech:
                if (speech) {
                    speech_ms_ += frame_ms;
                } else {
                    // First silence frame starts the trailing clock promptly.
                    phase_             = Phase::Trailing;
                    silence_ms_        = frame_ms;
                    target_ms_         = cfg_.silence_ms; // default; orchestrator may override
                    s.entered_trailing = true;
                }
                break;

            case Phase::Trailing:
                if (confirmed) {
                    // Sustained speech resumed -> the pause was within the utterance.
                    phase_      = Phase::InSpeech;
                    speech_ms_ += run_ms_;
                    silence_ms_ = 0.0f;
                } else if (speech) {
                    // Unconfirmed blip (< min_speech_run_ms): hold the silence clock
                    // -- neither reset nor advance -- until we know if it is real
                    // speech. This is what stops noise/breath from resetting silence.
                } else {
                    silence_ms_ += frame_ms;
                    if (silence_ms_ >= (float) target_ms_ &&
                        speech_ms_  >= (float) cfg_.min_speech_ms) {
                        s.endpoint = true;
                    }
                }
                break;
        }

        s.in_speech = (phase_ == Phase::InSpeech);
        return s;
    }

    // Advance by a block of frames (all the same duration). The aggregate
    // reports in_speech of the final frame, OR of entered_trailing across the
    // block, and endpoint if any frame in the block ended the turn.
    Step push_frames(const float* probs, int n, float frame_ms) {
        Step agg{false, false, false};
        for (int i = 0; i < n; ++i) {
            const Step s = push_frame(probs[i], frame_ms);
            agg.in_speech         = s.in_speech;
            agg.entered_trailing |= s.entered_trailing;
            agg.endpoint         |= s.endpoint;
        }
        return agg;
    }

private:
    SileroTurnConfig cfg_;
    Phase phase_;
    float speech_ms_;
    float silence_ms_;
    float run_ms_;      // current consecutive-speech run (hysteresis)
    float last_prob_;   // most recent frame probability (debug)
    int   target_ms_;
};

} // namespace athena
