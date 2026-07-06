// athena_memory.h
// ─────────────────────────────────────────────────────────────────────────────
// Athena cross-session memory + long-term personality (header-only).
//
// Header-only by design: talk-llama.cpp is patched into whisper.cpp/examples and
// built by their CMakeLists, which lists .cpp SOURCES explicitly but NOT headers.
// So a header placed next to talk-llama.cpp compiles with zero CMake changes —
// only an extra `cp athena_memory.h examples/talk-llama/` at build time. Every
// function is static/inline; this header is included by exactly one TU
// (talk-llama.cpp), so there are no ODR concerns.
//
// Everything here is PURE (std-only: file I/O, strings, time, math). It has no
// dependency on llama.h. The model-driven steps (extract / compact / personality
// integrate) are split in two: this header BUILDS their prompts and PARSES their
// output (both pure + unit-tested); talk-llama.cpp owns the thin glue that runs
// Qwen, reusing the generation patterns already proven in that file.
//
// Design grounding (see the design discussion for citations):
//   * Ebbinghaus forgetting curve  R = exp(-Δt / strength)  (MemoryBank) — older
//     memories fade, recall reinforces strength and resets the clock.
//   * Emotional salience (amygdala/flashbulb) — high-arousal memories get a
//     higher effective strength and resist pruning. Athena's emotion2vec tags
//     are the arousal signal, already present in the transcript.
//   * Episodic→semantic "semanticization" (systems consolidation) — recent
//     memories stay vivid/verbatim; old ones compact to generalized gist.
//   * Personality = McAdams' 3 levels, evolving SLOWLY and only on subjectively
//     impactful evidence (the personality.ledger threshold).
// ─────────────────────────────────────────────────────────────────────────────
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace amem {

// ── Tunable constants (all documented; safe defaults) ────────────────────────
// Decay timescale. R = exp(-Δdays / (BASE_TAU_DAYS * eff_strength)). With a
// fresh, low-salience memory (eff≈1) and BASE_TAU_DAYS=5: R≈0.82 after a day,
// 0.25 after a week, 0.05 after two. Salience multiplies eff_strength, so
// emotional memories decay several times slower (the flashbulb effect).
static constexpr double BASE_TAU_DAYS = 5.0;
static constexpr double R_FLOOR       = 0.15; // below this (and low salience) → prune
static constexpr int    SAL_KEEP      = 7;    // salience ≥ this is "flashbulb": never auto-pruned
static constexpr double VIVID_R       = 0.55; // R ≥ this renders under "Still vivid"
static constexpr double COMPACT_AGE_DAYS = 14.0; // older + faded → compaction candidate

// ── A single memory ──────────────────────────────────────────────────────────
struct MemEntry {
    std::string id;
    long   born        = 0;  // epoch seconds, when first recorded
    long   last_recall = 0;  // epoch seconds, last time reinforced/recalled
    int    S           = 1;  // Ebbinghaus discrete strength (≥1; +1 per recall)
    int    salience    = 5;  // 0..10 (importance blended with vocal arousal)
    std::string emotion;     // emotion2vec label at creation ("" if none)
    std::string gist;        // first-person memory text
};

// ── small helpers ─────────────────────────────────────────────────────────────
static inline std::string trim_copy(const std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static inline std::string flatten_ws(std::string s) {
    for (char &c : s) if (c == '\t' || c == '\n' || c == '\r') c = ' ';
    return s;
}

static inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

static inline size_t word_count(const std::string &s) {
    std::istringstream is(s);
    std::string w; size_t n = 0;
    while (is >> w) ++n;
    return n;
}

// ── salience: blend an LLM importance (1..10) with the vocal-affect tag ───────
// High-arousal emotions (anger/fear/surprise) move memory more than low-arousal
// (happy/sad); any non-neutral tag is at least a small bump. emotion2vec already
// dropped neutral/other/unk before tagging, so a non-empty tag means affect.
static inline int emotion_bonus(const std::string &emotion) {
    if (emotion.empty()) return 0;
    if (emotion == "angry" || emotion == "fearful" || emotion == "surprised") return 3;
    return 2; // happy / sad / disgusted
}

static inline int blend_salience(int importance, const std::string &emotion) {
    return clampi(importance + emotion_bonus(emotion), 0, 10);
}

// ── Ebbinghaus retention ──────────────────────────────────────────────────────
static inline double days_between(long a, long b) {
    return (double)(b - a) / 86400.0;
}

static inline double retention(const MemEntry &e, long now) {
    const double dt  = std::max(0.0, days_between(e.last_recall, now));
    const double eff = (double)std::max(1, e.S) * (1.0 + (double)e.salience / 5.0);
    return std::exp(-dt / (BASE_TAU_DAYS * eff));
}

// recall reinforcement: +1 strength, reset the decay clock (spacing effect)
static inline void reinforce(MemEntry &e, long now) {
    e.S += 1;
    e.last_recall = now;
}

// ── Decay + prune ─────────────────────────────────────────────────────────────
// Drop faded, low-salience memories. High-salience (flashbulb) memories are
// exempt no matter how old. Returns survivors; does not mutate input.
static inline std::vector<MemEntry> prune(const std::vector<MemEntry> &in, long now) {
    std::vector<MemEntry> out;
    out.reserve(in.size());
    for (const auto &e : in) {
        if (e.salience >= SAL_KEEP) { out.push_back(e); continue; }
        if (retention(e, now) >= R_FLOOR) out.push_back(e);
    }
    return out;
}

// Indices of entries that are old + faded + not flashbulb → compaction fodder.
// (talk-llama feeds these gists to Qwen, which synthesizes a single semantic
// memory that replaces them — episodic→semantic.)
static inline std::vector<size_t> compaction_candidates(const std::vector<MemEntry> &in, long now) {
    std::vector<size_t> idx;
    for (size_t i = 0; i < in.size(); ++i) {
        const auto &e = in[i];
        if (e.salience >= SAL_KEEP) continue;
        const bool old_enough = days_between(e.born, now) >= COMPACT_AGE_DAYS;
        const bool faded      = retention(e, now) < VIVID_R;
        if (old_enough && faded) idx.push_back(i);
    }
    return idx;
}

// If the rendered memory would exceed the word budget, return indices to compact,
// oldest/faintest first, until the estimate fits. Pure: caller does the merging.
static inline std::vector<size_t> over_budget_candidates(const std::vector<MemEntry> &in,
                                                         long now, size_t word_budget) {
    size_t total = 0;
    for (const auto &e : in) total += word_count(e.gist);
    if (total <= word_budget) return {};
    // rank non-flashbulb entries by ascending retention (faintest first)
    std::vector<size_t> order;
    for (size_t i = 0; i < in.size(); ++i)
        if (in[i].salience < SAL_KEEP) order.push_back(i);
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return retention(in[a], now) < retention(in[b], now);
    });
    std::vector<size_t> pick;
    for (size_t i : order) {
        if (total <= word_budget) break;
        pick.push_back(i);
        // a compacted cluster averages ~12 words; approximate the reclaim
        size_t w = word_count(in[i].gist);
        total -= (w > 12 ? w - 12 : 0);
    }
    return pick;
}

// ── Time formatting (all pure; localtime-based) ──────────────────────────────
static inline const char *part_of_day(int hour) {
    if (hour < 12) return "morning";
    if (hour < 17) return "afternoon";
    if (hour < 21) return "evening";
    return "night";
}

static inline std::string fmt_clock(long epoch) {
    time_t t = (time_t)epoch; struct tm tmv; localtime_r(&t, &tmv);
    char buf[32];
    strftime(buf, sizeof(buf), "%-I:%M %p", &tmv); // e.g. "2:47 PM"
    return buf;
}

// "Thursday afternoon, June 18, 2026 — 2:47 PM"
static inline std::string fmt_datetime(long epoch) {
    time_t t = (time_t)epoch; struct tm tmv; localtime_r(&t, &tmv);
    char day[16], date[48];
    strftime(day,  sizeof(day),  "%A",          &tmv);
    strftime(date, sizeof(date), "%B %-d, %Y",  &tmv);
    return std::string(day) + " " + part_of_day(tmv.tm_hour) + ", " + date +
           " — " + fmt_clock(epoch);
}

static inline bool same_calendar_day(long a, long b) {
    time_t ta = (time_t)a, tb = (time_t)b; struct tm va, vb;
    localtime_r(&ta, &va); localtime_r(&tb, &vb);
    return va.tm_year == vb.tm_year && va.tm_yday == vb.tm_yday;
}

static inline bool is_yesterday(long then, long now) {
    time_t tn = (time_t)now; struct tm vn; localtime_r(&tn, &vn);
    vn.tm_mday -= 1; time_t y = mktime(&vn);
    return same_calendar_day(then, (long)y);
}

// Natural, fuzzy "how long since we last spoke". Pure; testable with fixed epochs.
static inline std::string humanize_elapsed(long then, long now) {
    if (then <= 0 || now <= then) return "";
    const long secs = now - then;
    const long mins = secs / 60;
    if (mins < 1)               return "less than a minute ago";
    if (mins < 45)              return "about " + std::to_string(mins) + (mins == 1 ? " minute ago" : " minutes ago");
    if (mins < 90)              return "about an hour ago";
    if (same_calendar_day(then, now)) {
        const long hours = (mins + 30) / 60;   // round to nearest hour
        return "about " + std::to_string(hours) + (hours == 1 ? " hour ago" : " hours ago");
    }
    if (is_yesterday(then, now)) {
        time_t tt = (time_t)then; struct tm vt; localtime_r(&tt, &vt);
        return std::string("yesterday ") + part_of_day(vt.tm_hour);
    }
    const long days = secs / 86400;
    if (days < 7)  return std::to_string(days) + " days ago";
    if (days < 11) return "about a week ago";
    const long weeks = days / 7;
    if (days < 45) return "about " + std::to_string(weeks) + " weeks ago";
    const long months = days / 30;
    if (months < 18) return "a while back — about " + std::to_string(std::max(1L, months)) + " months ago";
    return "a long time ago";
}

// ── Sidecar (machine-only) TSV I/O ───────────────────────────────────────────
// Format, one record per line (gist last; tabs/newlines flattened on write):
//   id \t born \t last_recall \t S \t salience \t emotion \t gist
static inline std::vector<MemEntry> load_state(const std::string &path) {
    std::vector<MemEntry> out;
    std::ifstream f(path);
    if (!f) return out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::vector<std::string> col;
        size_t start = 0;
        for (int i = 0; i < 6; ++i) {
            size_t tab = line.find('\t', start);
            if (tab == std::string::npos) { col.clear(); break; }
            col.push_back(line.substr(start, tab - start));
            start = tab + 1;
        }
        if (col.size() != 6) continue;       // malformed line → skip defensively
        MemEntry e;
        e.id          = col[0];
        e.born        = strtol(col[1].c_str(), nullptr, 10);
        e.last_recall = strtol(col[2].c_str(), nullptr, 10);
        e.S           = clampi((int)strtol(col[3].c_str(), nullptr, 10), 1, 1000000);
        e.salience    = clampi((int)strtol(col[4].c_str(), nullptr, 10), 0, 10);
        e.emotion     = col[5];
        e.gist        = line.substr(start);  // everything after the 6th tab
        if (!e.gist.empty()) out.push_back(e);
    }
    return out;
}

// atomic: write temp in the same dir, fsync, rename over the target.
static inline bool atomic_write(const std::string &path, const std::string &content) {
    const std::string tmp = path + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        if (!f) return false;
        f << content;
        f.flush();
        if (!f.good()) return false;
    }
    return std::rename(tmp.c_str(), path.c_str()) == 0;
}

static inline bool save_state(const std::string &path, const std::vector<MemEntry> &v) {
    std::ostringstream os;
    for (const auto &e : v) {
        os << e.id << '\t' << e.born << '\t' << e.last_recall << '\t'
           << e.S << '\t' << e.salience << '\t' << flatten_ws(e.emotion) << '\t'
           << flatten_ws(e.gist) << '\n';
    }
    return atomic_write(path, os.str());
}

// id generator: stable, sortable, unique within a process run
static inline std::string make_id(long born) {
    static long seq = 0;
    return "m" + std::to_string(born) + "_" + std::to_string(seq++);
}

// ── Rendering the injected memory bullets (newest-first, vivid vs faded) ──────
// This produces the PERSISTENT memory.txt body (no temporal header — that is
// computed fresh at startup). Used by the consolidation pass.
static inline std::string render_memory_body(std::vector<MemEntry> v, long now) {
    // newest activity first
    std::sort(v.begin(), v.end(), [](const MemEntry &a, const MemEntry &b) {
        return std::max(a.born, a.last_recall) > std::max(b.born, b.last_recall);
    });
    std::ostringstream vivid, faded;
    int nv = 0, nf = 0;
    for (const auto &e : v) {
        if (e.gist.empty()) continue;
        if (retention(e, now) >= VIVID_R || e.salience >= SAL_KEEP) {
            vivid << "- " << trim_copy(e.gist) << "\n"; ++nv;
        } else {
            faded << "- " << trim_copy(e.gist) << "\n"; ++nf;
        }
    }
    std::ostringstream os;
    if (nv) os << "## Still vivid\n" << vivid.str();
    if (nf) os << (nv ? "\n" : "") << "## Settled into the background\n" << faded.str();
    return os.str();
}

// ── The startup injection block ({6} content) ────────────────────────────────
// Composes the third-person awareness instruction + the first-person personality
// and memory, with a FRESHLY computed temporal header. Pure apart from file
// reads + the wall clock. Returns "" only if memory is disabled (dir empty).
static inline std::string read_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

static inline std::string join_path(const std::string &dir, const std::string &name) {
    if (dir.empty()) return name;
    return dir.back() == '/' ? dir + name : dir + "/" + name;
}

// dir       : --memory directory (already validated non-empty by caller)
// now       : epoch seconds (current)
// last_sess : epoch of previous session end (0 if first run)
//
// The prompt above this block is a "never ending dialog" whose few-shot examples
// are indistinguishable from real conversation history. Without a hard boundary
// the model reads those examples AS its memory and opens by calling back to them
// — the cats / free-will / "semicolon victory" confabulation, and the verbatim
// re-recital every turn, both seen in live testing. So this block (a) explicitly
// disowns everything above as demonstration that never happened, (b) presents the
// REAL personality + memory addressed to {person} as "you" (so recall in
// conversation says "you're allergic to shellfish", not "he is"), (c) explains
// the [time:] side-channel so the clock isn't confabulated into the dialogue, and
// (d) hands off to the live turn with an explicit "respond to what they actually
// say next; don't recite memory or the examples."
static inline std::string build_injection_block(const std::string &dir, long now,
                                                long last_sess,
                                                const std::string &bot,
                                                const std::string &person) {
    const std::string personality = trim_copy(read_file(join_path(dir, "personality.txt")));
    const std::string memory_body = trim_copy(read_file(join_path(dir, "memory.txt")));
    const std::string rule(60, '_');

    std::ostringstream os;
    os << "\n\n" << rule << "\n";
    os << "Everything above this line — including the brief \"let's start a fresh "
          "conversation\" exchange — is ONLY a demonstration of how " << bot
       << " talks and thinks. None of it actually happened: " << bot << " does not "
          "remember any of it and never refers back to it as if it were real. What "
          "follows is " << bot << "'s genuine, lasting self — her real memory of past "
          "conversations with " << person << " and who she has become. It is written "
          "to " << person << ", and speaks to " << person << " directly as \"you.\"\n";
    os << rule << "\n";

    // ── personality (who she's become) ──
    os << "\nWHO " << bot << " HAS BECOME (her personality — it shifts only slowly):\n";
    if (!personality.empty()) os << personality << "\n";
    else os << "(Still early — so far this is just her core temperament.)\n";

    // ── memory, with a fresh temporal header ──
    os << "\nWHAT " << bot << " REMEMBERS (real past conversations with " << person << "):\n";
    os << "It's " << fmt_datetime(now) << ".";
    const std::string elapsed = humanize_elapsed(last_sess, now);
    if (!elapsed.empty()) {
        os << " The last time you and " << bot << " talked was " << elapsed;
        // Absolute anchor next to the relative phrase: the relative figure is
        // computed once here and then frozen, so it drifts as this session runs;
        // the absolute wall-clock/date of the last session does not.
        if (same_calendar_day(last_sess, now)) os << " (around " << fmt_clock(last_sess) << ")";
        else                                    os << " (" << fmt_datetime(last_sess) << ")";
        os << ".";
    }
    os << "\n";
    if (!memory_body.empty()) os << memory_body << "\n";
    else os << "(You and " << bot << " are just getting to know each other; "
            << "she has no memories of past conversations yet.)\n";

    // ── handoff into the live turn ──
    os << rule << "\n";
    os << bot << " draws on the real memories above the way you recall shared history "
          "with an old friend — naturally, never reciting them and never saying she is "
          "\"checking her memory.\" If a line from " << person << " ends with a tag like "
          "[time: 3:47 PM], that is simply the current clock so she stays oriented in "
          "time; she uses it only if it matters and never reads it aloud. The live "
          "conversation with " << person << " starts now: " << bot << " just responds to "
          "whatever " << person << " actually says next — she does not open by reciting "
          "memories, listing what she knows, or mentioning the examples above.\n";
    os << rule << "\n\n";
    return os.str();
}

// ── In-session "now" cue (appended to user turns, like the emotion tag) ───────
// The injection block already states the current time at session start, so the
// FIRST turn is seeded silently — emitting a [time:] tag there is redundant and,
// in testing, gave the model a concrete clock to confabulate a fake earlier time
// from. After that, returns " [time: 2:47 PM]" whenever wall-clock has advanced
// past refresh_min since the last cue; "" otherwise. Stateful struct.
struct TimeCue {
    long last_cue = 0;
    bool seeded   = false;
    std::string maybe(long now, int refresh_min) {
        if (!seeded) { seeded = true; last_cue = now; return ""; } // turn 1: time already in the injected header
        if ((now - last_cue) >= (long)refresh_min * 60) {
            last_cue = now;
            return " [time: " + fmt_clock(now) + "]";
        }
        return "";
    }
};

// ── meta (last-session timestamp) ────────────────────────────────────────────
static inline long read_last_session(const std::string &dir) {
    const std::string s = trim_copy(read_file(join_path(dir, "meta")));
    if (s.empty()) return 0;
    return strtol(s.c_str(), nullptr, 10);
}
static inline bool write_last_session(const std::string &dir, long epoch) {
    return atomic_write(join_path(dir, "meta"), std::to_string(epoch) + "\n");
}

// ═════════════════════════════════════════════════════════════════════════════
// Consolidation prompt BUILDERS + output PARSERS (pure; used by Part-2 glue).
// Kept here so all the fiddly formatting/parsing is unit-tested ahead of the
// thin model-running code in talk-llama.cpp.
// ═════════════════════════════════════════════════════════════════════════════

// A captured turn in the just-finished session.
struct Turn {
    std::string speaker; // person name or bot name
    std::string text;    // includes any [emotion: ..] / [time: ..] tags inline
};

static inline std::string transcript_text(const std::vector<Turn> &turns, size_t char_cap) {
    std::string out;
    for (const auto &t : turns) {
        out += t.speaker + ": " + t.text + "\n";
    }
    if (char_cap && out.size() > char_cap) out = out.substr(out.size() - char_cap); // keep the END (most recent)
    return out;
}

// EXTRACT: ask Qwen for memorable moments, one per line, fixed parseable format.
static inline std::string build_extract_prompt(const std::string &bot,
                                               const std::string &person,
                                               const std::vector<Turn> &turns,
                                               size_t char_cap) {
    std::ostringstream os;
    os << "You are " << bot << "'s memory-consolidation process, running after a "
          "voice conversation with " << person << " has ended. Read the transcript and "
          "write down the moments worth remembering long-term — things that matter to "
       << person << ", facts about " << person << " and their life, decisions, "
          "emotional moments, and anything that would make the next conversation feel "
          "continuous. Ignore small talk and anything trivial.\n\n"
          "Write each memory on its own line, in the first person as " << bot
       << ", speaking TO " << person << " as \"you\" — write \"I helped you ...\" or "
          "\"you decided ...\", never \"I helped " << person << " ...\", \"" << person
       << " decided ...\", or \"he ...\". One short sentence per memory, in this exact "
          "format:\n"
          "IMPORTANCE | MEMORY\n"
          "where IMPORTANCE is an integer 1-10 (10 = life-changing/deeply emotional, "
          "1 = minor). Inline tags like [emotion: sad] in the transcript reflect "
       << person << "'s tone of voice — weight emotional moments higher. Output only "
          "the lines, no preamble. If nothing is worth remembering, output exactly: NONE\n\n"
          "TRANSCRIPT:\n"
       << transcript_text(turns, char_cap)
       << "\nMEMORIES:\n";
    return os.str();
}

// Parse "IMP | gist" lines. Returns entries with born/last_recall=now, S=1,
// salience = blend(importance, emotion). emotion is detected from a [emotion: x]
// tag if the model echoed one into the gist (then stripped from the gist text).
static inline std::string extract_emotion_tag(std::string &gist) {
    const std::string key = "[emotion:";
    size_t p = gist.find(key);
    if (p == std::string::npos) return "";
    size_t e = gist.find(']', p);
    if (e == std::string::npos) return "";
    std::string inside = trim_copy(gist.substr(p + key.size(), e - (p + key.size())));
    gist = trim_copy(gist.substr(0, p) + gist.substr(e + 1));
    return inside;
}

static inline std::vector<MemEntry> parse_extracted(const std::string &model_out_raw, long now) {
    // Defense-in-depth: strip a leading <think>...</think> block if one leaks
    // through (sampler suppression should prevent it, but a spelled-out tag must
    // never poison parsing). A complete block is dropped; an unterminated one
    // means the answer never arrived → yield nothing (the raw-output log shows it).
    std::string model_out = model_out_raw;
    {
        size_t ts = model_out.find("<think>");
        if (ts != std::string::npos) {
            size_t te = model_out.find("</think>", ts);
            model_out = (te != std::string::npos) ? model_out.substr(te + 8) : std::string();
        }
    }
    std::vector<MemEntry> out;
    std::istringstream is(model_out);
    std::string line;
    while (std::getline(is, line)) {
        line = trim_copy(line);
        if (line.empty()) continue;
        if (line.rfind("NONE", 0) == 0) break;             // tolerate a leaked suffix, e.g. "NONEuser"
        size_t bar = line.find('|');
        if (bar == std::string::npos) continue;            // not in IMP|gist form → skip
        const std::string impstr = trim_copy(line.substr(0, bar));
        std::string gist         = trim_copy(line.substr(bar + 1));
        // strip a leading "- " or "* " bullet the model may add
        if (gist.size() > 2 && (gist[0] == '-' || gist[0] == '*') && gist[1] == ' ')
            gist = trim_copy(gist.substr(2));
        if (gist.empty()) continue;
        int importance = clampi((int)strtol(impstr.c_str(), nullptr, 10), 1, 10);
        if (importance == 0) importance = 5;               // unparseable → neutral
        std::string emo = extract_emotion_tag(gist);
        MemEntry e;
        e.born = e.last_recall = now;
        e.id = make_id(now);
        e.S = 1;
        e.salience = blend_salience(importance, emo);
        e.emotion = emo;
        e.gist = gist;
        out.push_back(e);
    }
    return out;
}

// COMPACT: synthesize a faded cluster into one generalized (semantic) memory.
static inline std::string build_compact_prompt(const std::string &bot,
                                               const std::vector<MemEntry> &cluster) {
    std::ostringstream os;
    os << "You are " << bot << "'s memory consolidating older recollections, the way "
          "human memory turns specific episodes into general knowledge over time. "
          "Combine the related older memories below into ONE shorter, more general "
          "first-person memory (as " << bot << ", \"I ...\", still speaking to the same "
          "person as \"you\") that keeps the lasting gist and drops specifics. Output "
          "only that single sentence.\n\nOLDER MEMORIES:\n";
    for (const auto &e : cluster) os << "- " << e.gist << "\n";
    os << "\nGENERALIZED MEMORY:\n";
    return os.str();
}

static inline std::string parse_single_line(const std::string &model_out) {
    std::istringstream is(model_out);
    std::string line;
    while (std::getline(is, line)) {
        line = trim_copy(line);
        if (line.empty()) continue;
        if (line.size() > 2 && (line[0] == '-' || line[0] == '*') && line[1] == ' ')
            line = trim_copy(line.substr(2));
        return line;
    }
    return "";
}

// ── Personality ledger (sidecar): impact evidence accumulates until threshold ─
struct LedgerRow {
    std::string session; // YYYY-MM-DD
    int    weight = 0;    // accumulated impact (sum of salience of character-relevant moments)
    std::string axis;     // "agency" | "communion" | "" (McAdams' two axes)
    std::string evidence; // short note
};

static inline std::vector<LedgerRow> load_ledger(const std::string &path) {
    std::vector<LedgerRow> out;
    std::ifstream f(path); if (!f) return out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        size_t t1 = line.find('\t'); if (t1 == std::string::npos) continue;
        size_t t2 = line.find('\t', t1 + 1); if (t2 == std::string::npos) continue;
        size_t t3 = line.find('\t', t2 + 1); if (t3 == std::string::npos) continue;
        LedgerRow r;
        r.session  = line.substr(0, t1);
        r.weight   = (int)strtol(line.substr(t1 + 1, t2 - t1 - 1).c_str(), nullptr, 10);
        r.axis     = line.substr(t2 + 1, t3 - t2 - 1);
        r.evidence = line.substr(t3 + 1);
        out.push_back(r);
    }
    return out;
}

static inline bool save_ledger(const std::string &path, const std::vector<LedgerRow> &v) {
    std::ostringstream os;
    for (const auto &r : v)
        os << r.session << '\t' << r.weight << '\t' << flatten_ws(r.axis) << '\t'
           << flatten_ws(r.evidence) << '\n';
    return atomic_write(path, os.str());
}

static inline int ledger_weight(const std::vector<LedgerRow> &v) {
    int w = 0; for (const auto &r : v) w += r.weight; return w;
}

// PERSONALITY INTEGRATION prompt: conservative, slow edit, McAdams' 3 layers.
static inline std::string build_personality_prompt(const std::string &bot,
                                                   const std::string &person,
                                                   const std::string &current_personality,
                                                   const std::vector<LedgerRow> &evidence) {
    std::ostringstream os;
    os << "You are slowly updating " << bot << "'s sense of who she is. Personality "
          "changes GRADUALLY and only from genuinely impactful experience — make a "
          "small, conservative revision, not a rewrite. Keep her core temperament. "
          "Write in the first person as " << bot << ", speaking to " << person
       << " directly as \"you\" (write \"you push for rigor,\" never \"" << person
       << " pushes\" or \"he pushes\"). Use three short sections with these exact "
          "headers:\n"
          "## Temperament (rarely shifts)\n"
          "## What I've come to value, and how I tend to engage (shifts slowly)\n"
          "## My story with you (the throughline)\n"
          "Keep the whole thing under 250 words. Output only the revised "
          "description.\n\n"
          "CURRENT SELF-DESCRIPTION:\n"
       << (current_personality.empty() ? "(none yet)\n" : current_personality + "\n")
       << "\nRECENT IMPACTFUL EVIDENCE (from memorable conversations):\n";
    for (const auto &r : evidence)
        os << "- (" << (r.axis.empty() ? "general" : r.axis) << ", weight " << r.weight
           << ") " << r.evidence << "\n";
    os << "\nREVISED SELF-DESCRIPTION:\n";
    return os.str();
}

} // namespace amem
