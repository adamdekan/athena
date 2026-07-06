// orpheus-speak.cpp
// ─────────────────────────────────────────────────────────────────────────────
// Minimal C++ CLI that turns text into speech via:
//   1. llama-server  (Orpheus 3B GGUF on GPU)  → SNAC audio tokens
//   2. ONNX Runtime  (SNAC 24 kHz decoder)      → 24 kHz PCM → WAV file
//
// Zero Python dependencies.  Requires libcurl + onnxruntime C API.
//
// Usage:
//   orpheus-speak [options] "Text to speak"
//   orpheus-speak [options] -f input.txt
//   orpheus-speak [options] -f input.txt -o output.wav
//
// Designed to be used as the --speak command for whisper.cpp/talk-llama.
// ─────────────────────────────────────────────────────────────────────────────

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <fstream>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <libgen.h>
#include <signal.h>
#include <sys/inotify.h>
#include <sys/stat.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <curl/curl.h>
#include <onnxruntime_c_api.h>

// posix_spawn's environ (CHANGES.MD §13): the player is launched with posix_spawnp
// instead of fork()+execvp so glibc uses clone(CLONE_VM|CLONE_VFORK) and does NOT
// dup_mmap/pte_alloc-copy this process's CUDA/ONNX-laden address space — keeping
// orpheus-speak out of the kernel bad_page fork path the UVM churn poisons.
extern char **environ;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration defaults (override via CLI flags)
// ─────────────────────────────────────────────────────────────────────────────
struct Config {
    std::string api_url       = "http://127.0.0.1:8080/completion";
    std::string snac_model    = "snac24_decoder.onnx";  // SNAC ONNX decoder path
    std::string voice         = "tara";
    std::string output_wav    = "/dev/shm/orpheus_tts.wav"; // RAM-backed tmpfs — no disk IO
    std::string input_file;                              // -f <file>
    std::string text;                                    // positional text
    std::string play_cmd      = "";                      // e.g. "aplay" or "ffplay -nodisp -autoexit"
    std::string watch_file;                              // --watch <file>: persistent daemon mode
    float       temperature   = 0.6f;
    float       top_p         = 0.9f;
    float       rep_penalty   = 1.1f;
    int         max_tokens    = 2500;
    int         sample_rate   = 24000;
    bool        verbose       = false;
    bool        diag          = false;  // real-time pipeline diagnostics: per-second
                                        // device-fill heartbeat + stall-run markers
                                        // (for tracing TTS underruns / server stalls)
    bool        snac_cpu      = false;  // force SNAC decoder to CPU (saves ~1.7 GB VRAM)
    // ── SSE streaming TTS (--stream-tts) ─────────────────────────────────────
    bool        stream_tts    = false;  // stream tokens via SSE + incremental SNAC decode
    std::string play_raw_cmd  = "aplay -q -t raw -f S16_LE -c 1 --buffer-time=300000";
                                        // persistent raw-PCM sink; "-r <rate> -" appended
    int         prebuffer_ms  = 350;    // startup watermark: buffer this much audio before
                                        // the FIRST write of a session (absorbs the
                                        // turn-start generation deficit while Qwen is
                                        // contending for the GPU; never applies mid-turn)

    // ── Garble-diagnosis instrumentation (off by default) ───────────────────
    std::string capture_dir;            // --capture-dir DIR: per-session dump of raw tokens,
                                        // parsed codes, and the live decoded WAV (empty = off)
    std::string decode_codes;           // --decode-codes FILE: offline — re-decode a captured
                                        // .codes file to WAV (the discriminator), then exit
    std::string compare_a, compare_b;   // --compare-wav A B: print correlation/RMS verdict, exit
};

// ─────────────────────────────────────────────────────────────────────────────
// Orpheus special token IDs
// ─────────────────────────────────────────────────────────────────────────────
// The Orpheus tokenizer maps audio codes to custom_token_N where
// N = AUDIO_OFFSET + codebook_layer * 4096 + code_index
// custom_token_0 through custom_token_9 are special (SOH, EOT, etc.)
// Audio codes start at custom_token_10.
// See: https://github.com/canopyai/Orpheus-TTS
static constexpr int ORPHEUS_TOKEN_BASE   = 128266;  // vocab ID of first audio token
static constexpr int ORPHEUS_CODEBOOK_SZ  = 4096;
static constexpr int ORPHEUS_FRAME_TOKENS = 7;       // tokens per SNAC frame
static constexpr int ORPHEUS_AUDIO_OFFSET = 10;      // audio starts at <custom_token_10>

// Special framing tokens (not audio)
static constexpr int TOKEN_SOH = 128259;  // start-of-human
static constexpr int TOKEN_EOT = 128009;  // end-of-text
static constexpr int TOKEN_EOH = 128260;  // end-of-human
static constexpr int TOKEN_SOA = 128261;  // start-of-audio  (custom_token_5 sometimes)

// ─────────────────────────────────────────────────────────────────────────────
// WAV writer  (16-bit PCM, mono)
// ─────────────────────────────────────────────────────────────────────────────
static bool write_wav(const std::string &path, const std::vector<float> &pcm, int sr) {
    const int num_samples  = (int)pcm.size();
    const int bits         = 16;
    const int byte_rate    = sr * bits / 8;
    const int data_bytes   = num_samples * (bits / 8);
    const int file_size    = 36 + data_bytes;

    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    // RIFF header
    f.write("RIFF", 4);
    auto write32 = [&](uint32_t v) { f.write(reinterpret_cast<const char*>(&v), 4); };
    auto write16 = [&](uint16_t v) { f.write(reinterpret_cast<const char*>(&v), 2); };

    write32(file_size);
    f.write("WAVE", 4);

    // fmt chunk
    f.write("fmt ", 4);
    write32(16);           // chunk size
    write16(1);            // PCM
    write16(1);            // mono
    write32(sr);           // sample rate
    write32(byte_rate);    // byte rate
    write16(bits / 8);     // block align
    write16(bits);         // bits per sample

    // data chunk
    f.write("data", 4);
    write32(data_bytes);

    for (float s : pcm) {
        float clamped = std::max(-1.0f, std::min(1.0f, s));
        int16_t i16   = static_cast<int16_t>(clamped * 32767.0f);
        f.write(reinterpret_cast<const char*>(&i16), 2);
    }

    return f.good();
}

// WAV reader (16-bit PCM; multi-channel → channel 0). Returns samples in -1..1.
static bool read_wav(const std::string &path, std::vector<float> &out, int &sr) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char tag[4];
    auto rd32 = [&](uint32_t &v) { f.read(reinterpret_cast<char*>(&v), 4); };
    auto rd16 = [&](uint16_t &v) { f.read(reinterpret_cast<char*>(&v), 2); };
    if (!f.read(tag, 4) || std::strncmp(tag, "RIFF", 4)) return false;
    uint32_t riff_sz; rd32(riff_sz);
    if (!f.read(tag, 4) || std::strncmp(tag, "WAVE", 4)) return false;
    uint16_t channels = 1, bits = 16; uint32_t srate = 24000; bool have_fmt = false;
    while (f.read(tag, 4)) {
        uint32_t csz; rd32(csz);
        if (!std::strncmp(tag, "fmt ", 4)) {
            uint16_t fmt, balign; uint32_t byterate;
            rd16(fmt); rd16(channels); rd32(srate); rd32(byterate); rd16(balign); rd16(bits);
            if (csz > 16) f.seekg(csz - 16, std::ios::cur);   // skip any fmt extension
            have_fmt = true;
        } else if (!std::strncmp(tag, "data", 4)) {
            if (!have_fmt || bits != 16 || channels == 0) return false;
            size_t nsamp = csz / 2;
            std::vector<int16_t> raw(nsamp);
            f.read(reinterpret_cast<char*>(raw.data()), (std::streamsize)nsamp * 2);
            size_t got = (size_t)f.gcount() / 2;
            out.clear(); out.reserve(got / channels);
            for (size_t i = 0; i + channels <= got; i += channels)
                out.push_back(raw[i] / 32768.0f);            // channel 0
            sr = (int)srate;
            return true;
        } else {
            f.seekg(csz + (csz & 1), std::ios::cur);          // skip unknown chunk (pad to even)
        }
    }
    return false;
}

// ── --compare-wav A B: WAV similarity verdict (garble discriminator scoring) ──
// Pure C++ replacement for ad-hoc analysis: prints correlation, max diff, and RMS
// diff between two WAVs (e.g. a session's live .wav vs its --decode-codes .redecode.wav).
// Two clean SNAC decodes of the same codes correlate ~0.999; garble (structurally
// wrong audio) craters the correlation. A small lag search rules out a trivial
// sample misalignment masquerading as a mismatch.
static int compare_wavs(const std::string &pa, const std::string &pb) {
    std::vector<float> a, b; int sra = 0, srb = 0;
    if (!read_wav(pa, a, sra)) { std::cerr << "[orpheus-speak] cannot read WAV: " << pa << "\n"; return 1; }
    if (!read_wav(pb, b, srb)) { std::cerr << "[orpheus-speak] cannot read WAV: " << pb << "\n"; return 1; }
    std::cerr << "[orpheus-speak] compare-wav:\n"
              << "  A " << pa << "  (" << a.size() << " samp, " << sra << " Hz, "
              << (sra ? (float)a.size() / sra : 0) << " s)\n"
              << "  B " << pb << "  (" << b.size() << " samp, " << srb << " Hz, "
              << (srb ? (float)b.size() / srb : 0) << " s)\n";
    if (a.empty() || b.empty()) { std::cerr << "  empty signal\n"; return 1; }
    if (a.size() != b.size())
        std::cerr << "  NOTE: length differs by " << (long)a.size() - (long)b.size()
                  << " samples (truncation/abort?)\n";

    auto corr_at = [&](int lag) -> double {
        size_t ia = lag >= 0 ? 0 : (size_t)(-lag);
        size_t ib = lag >= 0 ? (size_t)lag : 0;
        size_t n = std::min(a.size() - ia, b.size() - ib);
        if (n < 16) return -2.0;
        double sa = 0, sb = 0;
        for (size_t i = 0; i < n; i++) { sa += a[ia + i]; sb += b[ib + i]; }
        double ma = sa / n, mb = sb / n, num = 0, da = 0, db = 0;
        for (size_t i = 0; i < n; i++) {
            double x = a[ia + i] - ma, y = b[ib + i] - mb;
            num += x * y; da += x * x; db += y * y;
        }
        return (da > 0 && db > 0) ? num / std::sqrt(da * db) : 0.0;
    };
    int best_lag = 0; double best = -2.0;
    for (int lag = -32; lag <= 32; ++lag) { double c = corr_at(lag); if (c > best) { best = c; best_lag = lag; } }

    size_t ia = best_lag >= 0 ? 0 : (size_t)(-best_lag);
    size_t ib = best_lag >= 0 ? (size_t)best_lag : 0;
    size_t n = std::min(a.size() - ia, b.size() - ib);
    double sumd2 = 0, suma2 = 0; int maxd = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[ia + i] - b[ib + i];
        sumd2 += d * d; suma2 += (double)a[ia + i] * a[ia + i];
        int di = (int)(std::fabs(d) * 32768.0 + 0.5); if (di > maxd) maxd = di;
    }
    double rms = std::sqrt(sumd2 / n), srms = std::sqrt(suma2 / n);
    std::cerr << "  overlap     : " << n << " samp (best lag " << best_lag << ")\n"
              << "  correlation : " << best << "\n"
              << "  max |diff|  : " << maxd << " / 32768\n"
              << "  rms diff    : " << (srms > 0 ? 100.0 * rms / srms : 0.0) << "% of signal\n"
              << "  verdict     : "
              << (best >= 0.995 ? "MATCH — within SNAC decode noise (clean)"
                : best >= 0.95  ? "PARTIAL divergence — inspect (localized corruption?)"
                                : "MISMATCH — structurally different (garble: bad codes or on-GPU decode)")
              << "\n  (baseline: two clean decodes correlate ~0.999; garble drops it hard)\n";
    return 0;
}

// ─────────────────────────────────────────────────────────────────────────────
// libcurl helpers
// ─────────────────────────────────────────────────────────────────────────────
static size_t curl_write_cb(char *data, size_t size, size_t nmemb, void *userp) {
    auto *buf = static_cast<std::string *>(userp);
    buf->append(data, size * nmemb);
    return size * nmemb;
}

// Build the Orpheus prompt format:
//   <custom_token_3><|begin_of_text|>{voice}: {text}<|eot_id|>
//   <custom_token_4><custom_token_5><custom_token_1>
//
// These map to token IDs in the Orpheus/Llama tokenizer:
//   <custom_token_3> = 128259 (start-of-human)
//   <|begin_of_text|> = standard Llama BOS
//   <|eot_id|>       = 128009 (end-of-text)
//   <custom_token_4> = 128260 (end-of-human)
//   <custom_token_5> = 128261 (start-of-audio)
//   <custom_token_1> = 128257 (audio generation marker)
static std::string build_orpheus_prompt(const std::string &voice, const std::string &text) {
    return "<custom_token_3><|begin_of_text|>" + voice + ": " + text +
           "<|eot_id|><custom_token_4><custom_token_5><custom_token_1>";
}

// Build the JSON body for /completion
static std::string build_request_json(const Config &cfg, const std::string &prompt,
                                      bool stream = false) {
    // Manual JSON construction to avoid external dependency.
    // Escape any quotes/backslashes in the prompt.
    std::string escaped;
    for (char c : prompt) {
        if (c == '"')       escaped += "\\\"";
        else if (c == '\\') escaped += "\\\\";
        else if (c == '\n') escaped += "\\n";
        else if (c == '\r') escaped += "\\r";
        else if (c == '\t') escaped += "\\t";
        else                escaped += c;
    }

    std::ostringstream js;
    js << "{"
       << "\"prompt\":\"" << escaped << "\","
       << "\"n_predict\":" << cfg.max_tokens << ","
       << "\"temperature\":" << cfg.temperature << ","
       << "\"top_p\":" << cfg.top_p << ","
       << "\"repeat_penalty\":" << cfg.rep_penalty << ","
       << "\"stop\":[\"<|eot_id|>\"],"
       << "\"parse_special\":true,"
       << "\"stream\":" << (stream ? "true" : "false")
       << "}";
    return js.str();
}

// POST to llama-server and return the raw response body.
// Uses thread_local CURL handle — allocated once per thread, reused across
// calls.  Keeps TCP connection alive to llama-server and avoids per-request
// handle allocation (~0.1-0.2ms savings per call, compounds over 10+ chunks).
static bool http_post(const std::string &url, const std::string &body,
                      std::string &response, bool verbose) {
    thread_local CURL *curl = nullptr;
    thread_local struct curl_slist *headers = nullptr;

    if (!curl) {
        curl = curl_easy_init();
        if (!curl) { std::cerr << "[orpheus-speak] curl_easy_init failed\n"; return false; }
        headers = curl_slist_append(nullptr, "Content-Type: application/json");
        headers = curl_slist_append(headers, "Connection: keep-alive");
    } else {
        curl_easy_reset(curl);
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 120L);
    curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);      // disable Nagle — lower latency
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);     // keep TCP connection alive

    if (verbose) {
        std::cerr << "[orpheus-speak] POST " << url << "\n";
        std::cerr << "[orpheus-speak] body: " << body.substr(0, 200) << "...\n";
    }

    CURLcode res = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    if (res != CURLE_OK) {
        std::cerr << "[orpheus-speak] curl error: " << curl_easy_strerror(res) << "\n";
        return false;
    }

    if (http_code != 200) {
        if (verbose)
            std::cerr << "[orpheus-speak] HTTP " << http_code
                      << " (response: " << response.substr(0, 120) << ")\n";
        response.clear();  // treat non-200 as empty
        return true;        // curl succeeded, but server rejected
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Async HTTP generation (thread-safe — no SNAC, no shared state)
// ─────────────────────────────────────────────────────────────────────────────
struct HttpResult {
    std::string response;
    float       server_ms;
    bool        ok;
};

static HttpResult http_generate(const std::string &api_url,
                                const std::string &voice,
                                const std::string &chunk_text,
                                const Config      &cfg,
                                bool               verbose) {
    std::string prompt = build_orpheus_prompt(voice, chunk_text);
    std::string body   = build_request_json(cfg, prompt);

    // Retry loop — server may return 503 when slots are momentarily busy
    // during concurrent submission.  Backoff gives the server time to
    // finish a prior request and free a slot.
    constexpr int max_retries = 3;

    for (int attempt = 0; attempt < max_retries; attempt++) {
        std::string response;

        auto t0 = std::chrono::high_resolution_clock::now();
        bool ok  = http_post(api_url, body, response, verbose && attempt == 0);
        auto t1 = std::chrono::high_resolution_clock::now();

        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        if (ok && !response.empty()) {
            return { std::move(response), ms, true };
        }

        // Failed — retry with backoff (500ms, 1000ms)
        if (attempt + 1 < max_retries) {
            int backoff = 500 * (attempt + 1);
            std::cerr << "[orpheus-speak] slot busy, retry " << attempt + 1
                      << "/" << max_retries << " in " << backoff << "ms\n";
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff));
        }
    }

    return { "", 0, false };
}

// ─────────────────────────────────────────────────────────────────────────────
// Token extraction
// ─────────────────────────────────────────────────────────────────────────────
// Extract custom_token IDs from the LLM response text.
// Orpheus outputs tokens like: <custom_token_28631> <custom_token_5043> ...
// We extract the integer N from each <custom_token_N>.
static std::vector<int> extract_audio_tokens(const std::string &text, bool verbose) {
    std::vector<int> tokens;
    tokens.reserve(512);  // typical response size

    // Hand-rolled scanner — much faster than std::regex for large responses.
    // Matches: <custom_token_DIGITS>
    static const char prefix[] = "<custom_token_";
    static const size_t plen = sizeof(prefix) - 1;

    size_t pos = 0;
    while (pos < text.size()) {
        pos = text.find(prefix, pos);
        if (pos == std::string::npos) break;
        pos += plen;

        // Parse digits
        int tok = 0;
        bool has_digits = false;
        while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') {
            tok = tok * 10 + (text[pos] - '0');
            has_digits = true;
            pos++;
        }

        // Must end with '>'
        if (!has_digits || pos >= text.size() || text[pos] != '>') continue;
        pos++; // skip '>'

        // Audio tokens start at <custom_token_10> (tokens 0-9 are special)
        if (tok >= ORPHEUS_AUDIO_OFFSET) {
            int code = tok - ORPHEUS_AUDIO_OFFSET;
            if (code < ORPHEUS_FRAME_TOKENS * ORPHEUS_CODEBOOK_SZ) {
                tokens.push_back(code);
            }
        }
    }

    if (verbose) {
        std::cerr << "[orpheus-speak] extracted " << tokens.size()
                  << " audio tokens from response\n";
    }
    return tokens;
}

// Extract the "content" or "text" field value from a JSON response.
// Handles both native /completion format: {"content":"..."}
// and OAI-compatible format: {"choices":[{"text":"..."}]}
static std::string extract_text_from_json(const std::string &json) {
    // Find "content": first (native /completion endpoint), then "text": (OAI compat)
    auto pos = json.find("\"content\":");
    if (pos == std::string::npos) {
        pos = json.find("\"text\":");
        if (pos == std::string::npos) return "";
    }

    // Find the colon after the key, then the opening quote of the value
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";
    pos++; // skip opening quote

    // Read until unescaped closing quote
    std::string result;
    for (size_t i = pos; i < json.size(); i++) {
        if (json[i] == '\\' && i + 1 < json.size()) {
            i++;
            switch (json[i]) {
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                case 'n':  result += '\n'; break;
                case 'r':  result += '\r'; break;
                case 't':  result += '\t'; break;
                default:   result += '\\'; result += json[i]; break;
            }
        } else if (json[i] == '"') {
            break;
        } else {
            result += json[i];
        }
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// Sentence splitter for chunked TTS
// ─────────────────────────────────────────────────────────────────────────────
// Split text at sentence boundaries so each sentence can be sent to Orpheus
// independently. This avoids attention skip issues on long text and keeps
// each request within the context window.
static std::vector<std::string> split_sentences(const std::string &text) {
    std::vector<std::string> sentences;
    std::string current;

    for (size_t i = 0; i < text.size(); i++) {
        current += text[i];

        bool is_terminal = (text[i] == '.' || text[i] == '?' || text[i] == '!');
        bool at_end      = (i + 1 >= text.size());
        bool next_space  = (!at_end && (text[i + 1] == ' ' || text[i + 1] == '\n'));
        bool at_boundary = is_terminal && (at_end || next_space);

        // Don't split fragments shorter than ~3 words
        if (at_boundary && current.size() > 15) {
            size_t start = current.find_first_not_of(" \n\r\t");
            if (start != std::string::npos) {
                sentences.push_back(current.substr(start));
            }
            current.clear();
        }
    }

    // Remaining text
    if (!current.empty()) {
        size_t start = current.find_first_not_of(" \n\r\t");
        if (start != std::string::npos) {
            std::string remainder = current.substr(start);
            if (!sentences.empty() && remainder.size() < 15) {
                sentences.back() += " " + remainder;
            } else {
                sentences.push_back(remainder);
            }
        }
    }

    if (sentences.empty() && !text.empty()) {
        sentences.push_back(text);
    }

    return sentences;
}

// ─────────────────────────────────────────────────────────────────────────────
// SNAC token deinterleaving
// ─────────────────────────────────────────────────────────────────────────────
// Orpheus produces 7 tokens per SNAC frame in this interleaved order:
//   [L0, L1a, L2a, L2b, L1b, L2c, L2d]
//
// We deinterleave into 3 codebook layers:
//   codes0 (L0):  1 code  per frame  → length = num_frames
//   codes1 (L1):  2 codes per frame  → length = num_frames * 2
//   codes2 (L2):  4 codes per frame  → length = num_frames * 4
//
// Each code value = raw_token_id % 4096  (the codebook index)
// The layer is encoded as raw_token_id / 4096.
struct SnacCodes {
    std::vector<int64_t> codes0;  // [num_frames]
    std::vector<int64_t> codes1;  // [num_frames * 2]
    std::vector<int64_t> codes2;  // [num_frames * 4]
};

static SnacCodes deinterleave_tokens(const std::vector<int> &tokens, bool verbose) {
    SnacCodes codes;
    int num_frames = (int)tokens.size() / ORPHEUS_FRAME_TOKENS;

    if (verbose) {
        std::cerr << "[orpheus-speak] " << tokens.size() << " tokens → "
                  << num_frames << " SNAC frames ("
                  << (float)num_frames / 11.71875f << "s audio)\n";   // 24000/2048 = 11.72 frames/s
    }

    for (int i = 0; i < num_frames; i++) {
        int base = i * ORPHEUS_FRAME_TOKENS;

        // Layer 0: 1 code per frame
        codes.codes0.push_back(tokens[base + 0] % ORPHEUS_CODEBOOK_SZ);

        // Layer 1: 2 codes per frame (positions 1, 4)
        codes.codes1.push_back((tokens[base + 1] - 1 * ORPHEUS_CODEBOOK_SZ) % ORPHEUS_CODEBOOK_SZ);
        codes.codes1.push_back((tokens[base + 4] - 4 * ORPHEUS_CODEBOOK_SZ) % ORPHEUS_CODEBOOK_SZ);

        // Layer 2: 4 codes per frame (positions 2, 3, 5, 6)
        codes.codes2.push_back((tokens[base + 2] - 2 * ORPHEUS_CODEBOOK_SZ) % ORPHEUS_CODEBOOK_SZ);
        codes.codes2.push_back((tokens[base + 3] - 3 * ORPHEUS_CODEBOOK_SZ) % ORPHEUS_CODEBOOK_SZ);
        codes.codes2.push_back((tokens[base + 5] - 5 * ORPHEUS_CODEBOOK_SZ) % ORPHEUS_CODEBOOK_SZ);
        codes.codes2.push_back((tokens[base + 6] - 6 * ORPHEUS_CODEBOOK_SZ) % ORPHEUS_CODEBOOK_SZ);
    }

    return codes;
}

// ─────────────────────────────────────────────────────────────────────────────
// ONNX Runtime: SNAC decode  (codes → PCM float32)
// ─────────────────────────────────────────────────────────────────────────────
#define ORT_CHECK(expr) do {                                             \
    OrtStatus *_s = (expr);                                              \
    if (_s) {                                                            \
        const char *msg = g_ort->GetErrorMessage(_s);                    \
        std::cerr << "[orpheus-speak] ORT error: " << msg << "\n";       \
        g_ort->ReleaseStatus(_s);                                        \
        return {};                                                       \
    }                                                                    \
} while(0)

// ─────────────────────────────────────────────────────────────────────────────
// Persistent SNAC decoder — init once, decode many times
// ─────────────────────────────────────────────────────────────────────────────
struct SnacDecoder {
    const OrtApi *g_ort = nullptr;
    OrtEnv *env = nullptr;
    OrtSessionOptions *opts = nullptr;
    OrtSession *session = nullptr;
    OrtMemoryInfo *mem_info = nullptr;
    OrtAllocator *allocator = nullptr;
    OrtRunOptions *run_opts = nullptr;      // Fix 1: arena shrinkage per-Run

    const char *in_names[3] = {};
    const char *out_names[1] = {};
    bool is_dynamic = false;
    bool has_cuda_ep = false;
    bool ok = false;

    // Fix 3: pre-allocated padded buffers for dynamic model.
    // Each SNAC codes0 frame produces 2048 audio samples at 24 kHz.
    // MAX_C0=1024 covers ~42 seconds of audio per decode call — well beyond
    // any chunked utterance.  If exceeded, falls back to fresh allocation.
    static constexpr int64_t MAX_C0 = 1024;
    static constexpr int64_t MAX_C1 = MAX_C0 * 2;
    static constexpr int64_t MAX_C2 = MAX_C0 * 4;
    static constexpr int SAMPLES_PER_FRAME = 2048;
    std::vector<int64_t> buf_c0, buf_c1, buf_c2;

    void init(const Config &cfg) {
        g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

        if (g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "orpheus-speak", &env)) return;
        if (g_ort->CreateSessionOptions(&opts)) return;

        g_ort->SetIntraOpNumThreads(opts, 0);
        g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);

        // Fix 2: disable CPU memory arena — prevents unbounded growth from
        // dynamic input shapes.  Uses raw malloc/free per tensor instead.
        // The SNAC model is small (~50 MB) and tensors are tiny, so the
        // per-call malloc overhead is negligible vs. compute time.
        g_ort->DisableCpuMemArena(opts);

        // Cache the optimized graph to disk
        std::string opt_path = cfg.snac_model + ".optimized";
        g_ort->SetOptimizedModelFilePath(opts, opt_path.c_str());

        // CUDA EP (skip if --snac-cpu requested)
        if (!cfg.snac_cpu) {
            OrtCUDAProviderOptions cuda_opts{};
            cuda_opts.device_id = 0;
            // kSameAsRequested (1) prevents power-of-two over-allocation.
            // Combined with Fix 3 (fixed shapes), the GPU arena stabilizes
            // after the first Run() and never grows.
            cuda_opts.arena_extend_strategy = 1;
            cuda_opts.do_copy_in_default_stream = 1;
            OrtStatus *cs = g_ort->SessionOptionsAppendExecutionProvider_CUDA(opts, &cuda_opts);
            if (cs) {
                if (cfg.verbose) std::cerr << "[orpheus-speak] CUDA EP unavailable, using CPU\n";
                g_ort->ReleaseStatus(cs);
            } else {
                has_cuda_ep = true;
                if (cfg.verbose) std::cerr << "[orpheus-speak] SNAC using CUDA EP\n";
            }
        } else {
            std::cerr << "[orpheus-speak] SNAC using CPU (--snac-cpu)\n";
        }

        OrtStatus *css = g_ort->CreateSession(env, cfg.snac_model.c_str(), opts, &session);
        if (css) {
            // Previously this discarded the status and returned "SNAC decoder init
            // failed" with no reason. Surface the actual ORT/CUDA error...
            std::cerr << "[orpheus-speak] SNAC CreateSession failed: "
                      << g_ort->GetErrorMessage(css) << "\n";
            g_ort->ReleaseStatus(css);
            session = nullptr;
            // ...and if the CUDA EP was in play, retry on CPU rather than killing
            // the daemon. A CUDA-EP session failure is almost always a context/MPS
            // problem (e.g. the MPS server down); CPU SNAC is slower but keeps TTS
            // alive. has_cuda_ep is cleared so the RunOptions below don't request a
            // GPU memory arena that no longer exists.
            if (!has_cuda_ep) return;
            std::cerr << "[orpheus-speak] SNAC: falling back to CPU EP\n";
            has_cuda_ep = false;
            if (opts) { g_ort->ReleaseSessionOptions(opts); opts = nullptr; }
            if (g_ort->CreateSessionOptions(&opts)) return;
            g_ort->SetIntraOpNumThreads(opts, 0);
            g_ort->SetSessionGraphOptimizationLevel(opts, ORT_ENABLE_ALL);
            g_ort->DisableCpuMemArena(opts);
            OrtStatus *css2 = g_ort->CreateSession(env, cfg.snac_model.c_str(), opts, &session);
            if (css2) {
                std::cerr << "[orpheus-speak] SNAC CPU CreateSession also failed: "
                          << g_ort->GetErrorMessage(css2) << "\n";
                g_ort->ReleaseStatus(css2);
                session = nullptr;
                return;
            }
            std::cerr << "[orpheus-speak] SNAC running on CPU EP (fallback)\n";
        }

        // Fix 2: use OrtDeviceAllocator (raw malloc/free) instead of
        // OrtArenaAllocator which grows monotonically with dynamic shapes
        if (g_ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mem_info)) return;
        if (g_ort->GetAllocatorWithDefaultOptions(&allocator)) return;

        // Fix 1: create RunOptions with arena shrinkage for any active arena.
        // CPU arena is disabled by Fix 2, so only request GPU shrinkage when
        // CUDA EP is active.  Requesting shrinkage on a non-existent arena
        // causes ORT to error on every Run() call.
        if (g_ort->CreateRunOptions(&run_opts)) return;
        if (has_cuda_ep) {
            g_ort->AddRunConfigEntry(run_opts,
                "memory.enable_memory_arena_shrinkage", "gpu:0");
        }

        // Read input/output names from model
        char *n0, *n1, *n2, *no;
        g_ort->SessionGetInputName(session, 0, allocator, &n0);
        g_ort->SessionGetInputName(session, 1, allocator, &n1);
        g_ort->SessionGetInputName(session, 2, allocator, &n2);
        g_ort->SessionGetOutputName(session, 0, allocator, &no);
        in_names[0] = n0; in_names[1] = n1; in_names[2] = n2;
        out_names[0] = no;

        // Detect static vs dynamic
        OrtTypeInfo *ti = nullptr;
        g_ort->SessionGetInputTypeInfo(session, 0, &ti);
        const OrtTensorTypeAndShapeInfo *si = nullptr;
        g_ort->CastTypeInfoToTensorInfo(ti, &si);
        size_t dc = 0; g_ort->GetDimensionsCount(si, &dc);
        std::vector<int64_t> dims(dc);
        g_ort->GetDimensions(si, dims.data(), dc);
        g_ort->ReleaseTypeInfo(ti);
        is_dynamic = (dc < 2 || dims[1] <= 0);

        // Fix 3: pre-allocate padded buffers for dynamic model
        if (is_dynamic) {
            buf_c0.resize(MAX_C0, 0);
            buf_c1.resize(MAX_C1, 0);
            buf_c2.resize(MAX_C2, 0);
        }

        if (cfg.verbose) {
            std::cerr << "[orpheus-speak] SNAC model: "
                      << (is_dynamic ? "dynamic" : "static")
                      << ", inputs: " << n0 << ", " << n1 << ", " << n2 << "\n";
        }

        ok = true;
    }

    // Serializes ORT Run() calls.  The batch path was single-threaded by
    // construction; the SSE streaming path decodes from multiple chunk
    // threads concurrently, so all decodes funnel through this mutex.
    std::mutex run_mutex;

    // Core decode: run the session on exact buffers, append PCM to `out`.
    // Returns false on ORT error.
    bool run_decode(const Config &cfg, int64_t *d0, int64_t n0, int64_t *d1, int64_t n1,
                    int64_t *d2, int64_t n2, std::vector<float> &out) {
        std::lock_guard<std::mutex> lk(run_mutex);
        int64_t s0[] = {1, n0}, s1[] = {1, n1}, s2[] = {1, n2};
        OrtValue *inputs[3] = {};
        OrtStatus *st;
        st = g_ort->CreateTensorWithDataAsOrtValue(mem_info, d0, n0*8, s0, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[0]);
        if (st) g_ort->ReleaseStatus(st);
        st = g_ort->CreateTensorWithDataAsOrtValue(mem_info, d1, n1*8, s1, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[1]);
        if (st) g_ort->ReleaseStatus(st);
        st = g_ort->CreateTensorWithDataAsOrtValue(mem_info, d2, n2*8, s2, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[2]);
        if (st) g_ort->ReleaseStatus(st);

        OrtValue *output = nullptr;
        OrtStatus *rs = g_ort->Run(session, run_opts, in_names, (const OrtValue *const *)inputs, 3, out_names, 1, &output);
        if (rs) {
            if (cfg.verbose) std::cerr << "[orpheus-speak] ORT Run error: " << g_ort->GetErrorMessage(rs) << "\n";
            g_ort->ReleaseStatus(rs);
            for (auto &v : inputs) if (v) g_ort->ReleaseValue(v);
            return false;
        }

        float *pcm = nullptr;
        g_ort->GetTensorMutableData(output, (void**)&pcm);
        OrtTensorTypeAndShapeInfo *info = nullptr;
        g_ort->GetTensorTypeAndShape(output, &info);
        size_t ne = 0; g_ort->GetTensorShapeElementCount(info, &ne);
        g_ort->ReleaseTensorTypeAndShapeInfo(info);

        out.insert(out.end(), pcm, pcm + ne);
        g_ort->ReleaseValue(output);
        for (auto &v : inputs) g_ort->ReleaseValue(v);
        return true;
    }

    // Decode codes at their EXACT shape (no MAX_C0 padding).  Used by the
    // SSE streaming path, where windows are a constant 4 frames — a stable
    // tiny shape that ORT caches after the first call, making each decode
    // a few ms instead of the padded ~300-400 ms.
    std::vector<float> decode_exact(const Config &cfg, const SnacCodes &codes) {
        if (!ok) return {};
        std::vector<float> out;
        std::vector<int64_t> c0(codes.codes0), c1(codes.codes1), c2(codes.codes2);
        if (c0.empty()) return out;
        run_decode(cfg, c0.data(), (int64_t)c0.size(), c1.data(), (int64_t)c1.size(),
                   c2.data(), (int64_t)c2.size(), out);
        return out;
    }

    std::vector<float> decode(const Config &cfg, const SnacCodes &codes) {
        if (!ok) return {};
        std::vector<float> all_pcm;

        auto run_one = [&](int64_t *d0, int64_t n0, int64_t *d1, int64_t n1,
                           int64_t *d2, int64_t n2) {
            run_decode(cfg, d0, n0, d1, n1, d2, n2, all_pcm);
        };

        if (is_dynamic) {
            int64_t actual_n0 = (int64_t)codes.codes0.size();
            int64_t actual_n1 = (int64_t)codes.codes1.size();
            int64_t actual_n2 = (int64_t)codes.codes2.size();

            if (actual_n0 <= MAX_C0) {
                // Fix 3: use pre-allocated padded buffers — the arena sees
                // the same tensor shape [1, MAX_C0/C1/C2] every time, so no
                // new allocations and no growth.  Zero-pad unused slots;
                // truncate output PCM to match actual input length.
                std::memset(buf_c0.data(), 0, MAX_C0 * sizeof(int64_t));
                std::memset(buf_c1.data(), 0, MAX_C1 * sizeof(int64_t));
                std::memset(buf_c2.data(), 0, MAX_C2 * sizeof(int64_t));
                std::memcpy(buf_c0.data(), codes.codes0.data(), actual_n0 * sizeof(int64_t));
                std::memcpy(buf_c1.data(), codes.codes1.data(), actual_n1 * sizeof(int64_t));
                std::memcpy(buf_c2.data(), codes.codes2.data(), actual_n2 * sizeof(int64_t));
                run_one(buf_c0.data(), MAX_C0, buf_c1.data(), MAX_C1, buf_c2.data(), MAX_C2);
                // Truncate: SNAC produces SAMPLES_PER_FRAME samples per codes0 frame
                int64_t actual_samples = actual_n0 * SAMPLES_PER_FRAME;
                if ((int64_t)all_pcm.size() > actual_samples)
                    all_pcm.resize(actual_samples);
            } else {
                // Fallback for extremely long audio (>42s per chunk)
                std::vector<int64_t> c0(codes.codes0), c1(codes.codes1), c2(codes.codes2);
                run_one(c0.data(), c0.size(), c1.data(), c1.size(), c2.data(), c2.size());
            }
        } else {
            static constexpr int W0 = 12, W1 = 24, W2 = 48, SPW = 24576;
            int total = (int)codes.codes0.size();
            int nwin = (total + W0 - 1) / W0;
            all_pcm.reserve(nwin * SPW);
            for (int w = 0; w < nwin; w++) {
                std::vector<int64_t> c0(W0, 0), c1(W1, 0), c2(W2, 0);
                int o0 = w*W0, o1 = w*W1, o2 = w*W2;
                int n0 = std::min(W0, total - o0);
                int n1 = std::min(W1, (int)codes.codes1.size() - o1);
                int n2 = std::min(W2, (int)codes.codes2.size() - o2);
                for (int i = 0; i < n0; i++) c0[i] = codes.codes0[o0+i];
                for (int i = 0; i < n1; i++) c1[i] = codes.codes1[o1+i];
                for (int i = 0; i < n2; i++) c2[i] = codes.codes2[o2+i];
                size_t prev = all_pcm.size();
                run_one(c0.data(), W0, c1.data(), W1, c2.data(), W2);
                if (w == nwin - 1 && n0 < W0)
                    all_pcm.resize(prev + (int)((float)n0 / W0 * SPW));
            }
        }

        if (cfg.verbose) {
            std::cerr << "[orpheus-speak] SNAC decoded " << all_pcm.size()
                      << " samples (" << (float)all_pcm.size() / cfg.sample_rate
                      << "s)" << (is_dynamic ? " in 1 pass" : "") << "\n";
        }
        return all_pcm;
    }

    void cleanup() {
        if (!g_ort) return;
        if (allocator) {
            for (auto n : in_names) if (n) g_ort->AllocatorFree(allocator, (void*)n);
            if (out_names[0]) g_ort->AllocatorFree(allocator, (void*)out_names[0]);
        }
        if (run_opts) g_ort->ReleaseRunOptions(run_opts);
        if (mem_info) g_ort->ReleaseMemoryInfo(mem_info);
        if (session)  g_ort->ReleaseSession(session);
        if (opts)     g_ort->ReleaseSessionOptions(opts);
        if (env)      g_ort->ReleaseEnv(env);
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// SSE streaming TTS subsystem (--stream-tts)
// ─────────────────────────────────────────────────────────────────────────────
// Replaces "generate whole chunk → decode whole chunk → WAV → aplay" with:
//   1. llama-server /completion with "stream":true (SSE) — tokens arrive as
//      they are generated (~160 tok/s = ~2x the 82 tok/s real-time rate)
//   2. a 4-frame sliding-window SNAC decode — to emit frame e, decode frames
//      [e-1 .. e+2] and keep the middle 2048-sample slice; head emits frames
//      0-1 from the first window, tail emits the remainder from the last
//      window.  Constant tiny ONNX shape → a few ms per decode, and no frame
//      is ever dropped (unlike the reference impl, which loses head + tail).
//   3. a per-session raw-PCM player (aplay reading stdin) — stdin close gives
//      exact drain semantics, eliminating WAV files and the drain-guard
//      heuristic.  Stall gaps are fed with silence to avoid underrun clicks.
//
// First audio per chunk ≈ time to 4 frames (28 tokens ≈ 0.2-0.4 s of
// generation) instead of the full chunk + a ~300-400 ms padded decode.

// Incrementally scan `tail` for complete <custom_token_N> matches, appending
// code values (N - ORPHEUS_AUDIO_OFFSET) to `out`.  Consumed text is erased;
// a trailing partial match (e.g. "<custom_tok") is preserved for next call.
static void scan_tokens_streaming(std::string &tail, std::vector<int> &out) {
    static const char prefix[] = "<custom_token_";
    static const size_t plen = sizeof(prefix) - 1;

    size_t pos = 0;
    size_t keep_from = std::string::npos;   // start of an incomplete match
    while (pos < tail.size()) {
        size_t lt = tail.find('<', pos);
        if (lt == std::string::npos) { pos = tail.size(); break; }

        // Is this the start of a (possibly partial) prefix?
        size_t avail = tail.size() - lt;
        size_t cmp = std::min(avail, plen);
        if (tail.compare(lt, cmp, prefix, cmp) != 0) { pos = lt + 1; continue; }
        if (avail < plen) { keep_from = lt; break; }    // partial prefix at end

        // Parse digits
        size_t p = lt + plen;
        int tok = 0; bool has_digits = false;
        while (p < tail.size() && tail[p] >= '0' && tail[p] <= '9') {
            tok = tok * 10 + (tail[p] - '0'); has_digits = true; p++;
        }
        if (p >= tail.size()) { keep_from = lt; break; }  // digits may continue
        if (!has_digits || tail[p] != '>') { pos = lt + 1; continue; }
        p++;  // consume '>'

        if (tok >= ORPHEUS_AUDIO_OFFSET) {
            int code = tok - ORPHEUS_AUDIO_OFFSET;
            if (code < ORPHEUS_FRAME_TOKENS * ORPHEUS_CODEBOOK_SZ) out.push_back(code);
        }
        pos = p;
    }
    if (keep_from != std::string::npos) tail.erase(0, keep_from);
    else                                tail.clear();
}

// ── Persistent raw-PCM sink ──────────────────────────────────────────────────
// One player process per session.  float PCM → S16_LE over a pipe; blocking
// writes provide backpressure; closing stdin makes the player drain and exit,
// so finish() returns exactly when the last sample has been handed to ALSA.
struct PcmSink {
    pid_t pid = -1;
    int   fd  = -1;
    int   sample_rate = 24000;

    bool start(const std::string &raw_cmd, int rate) {
        sample_rate = rate;
        int p[2];
        if (pipe(p) != 0) { perror("[orpheus-speak] pipe"); return false; }

        std::vector<std::string> args;
        std::istringstream iss(raw_cmd);
        std::string tok;
        while (iss >> tok) args.push_back(tok);
        args.push_back("-r"); args.push_back(std::to_string(rate));
        args.push_back("-");

        std::vector<char *> argv;
        for (auto &a : args) argv.push_back(&a[0]);
        argv.push_back(nullptr);

        // posix_spawnp (glibc clone(CLONE_VM|CLONE_VFORK)) instead of fork()+execvp:
        // no dup_mmap/pte_alloc over this process's CUDA/ONNX address space (CHANGES.MD
        // §13). Behaviour is identical — stdin<-pipe read end, stderr->/dev/null, PATH
        // search, and pid is a normal child waited by finish()/abort_now() below.
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_adddup2(&fa, p[0], STDIN_FILENO);
        posix_spawn_file_actions_addclose(&fa, p[0]);
        posix_spawn_file_actions_addclose(&fa, p[1]);
        posix_spawn_file_actions_addopen(&fa, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
        int rc = posix_spawnp(&pid, argv[0], &fa, nullptr, argv.data(), environ);
        posix_spawn_file_actions_destroy(&fa);
        close(p[0]);
        if (rc != 0) { close(p[1]); pid = -1; errno = rc; perror("[orpheus-speak] posix_spawnp"); return false; }
        fd = p[1];
        return true;
    }

    bool write_all(const int16_t *buf, size_t n) {
        const char *ptr = (const char *)buf;
        size_t left = n * sizeof(int16_t);
        while (left > 0) {
            ssize_t w = ::write(fd, ptr, left);
            if (w < 0) {
                if (errno == EINTR) continue;
                return false;   // EPIPE → player died
            }
            ptr += w; left -= (size_t)w;
        }
        return true;
    }

    bool write_pcm(const float *pcm, size_t n) {
        if (fd < 0) return false;
        std::vector<int16_t> s16(n);
        for (size_t i = 0; i < n; i++) {
            float v = pcm[i] * 32767.0f;
            if (v >  32767.0f) v =  32767.0f;
            if (v < -32768.0f) v = -32768.0f;
            s16[i] = (int16_t)v;
        }
        return write_all(s16.data(), n);
    }

    bool write_silence_ms(int ms) {
        if (fd < 0) return false;
        size_t n = (size_t)((int64_t)sample_rate * ms / 1000);
        std::vector<int16_t> z(n, 0);
        return write_all(z.data(), n);
    }

    // Close stdin → player drains its buffer and exits.  Returns when audio
    // has fully played (the exact .done timing talk-llama waits on).
    void finish() {
        if (fd >= 0) { close(fd); fd = -1; }
        if (pid > 0) { int st; waitpid(pid, &st, 0); pid = -1; }
    }

    // Hard stop for barge-in: kill the player without draining.  Closing the
    // PCM device (process death) halts output within one ALSA period — this
    // is what makes an interruption feel immediate.
    void abort_now() {
        if (fd >= 0) { close(fd); fd = -1; }
        if (pid > 0) { kill(pid, SIGKILL); int st; waitpid(pid, &st, 0); pid = -1; }
    }
};

// ── 4-frame sliding-window decoder ───────────────────────────────────────────
// Maintains the code stream for one chunk; advances emission one frame at a
// time with 1 frame of left context and 2 of lookahead (the canonical Orpheus
// streaming scheme), decoding a constant [1,4]/[1,8]/[1,16] shape.
struct SlidingWindowDecoder {
    static constexpr int W = 4;             // window frames
    std::vector<int> codes;                 // raw code stream (7 per frame)
    int emitted = 0;                        // frames already emitted

    int frames() const { return (int)codes.size() / ORPHEUS_FRAME_TOKENS; }

    void feed(const std::vector<int> &more) {
        codes.insert(codes.end(), more.begin(), more.end());
    }

    // Build SnacCodes for frames [f0, f0+nf)
    SnacCodes window(int f0, int nf) const {
        std::vector<int> sub(codes.begin() + (size_t)f0 * ORPHEUS_FRAME_TOKENS,
                             codes.begin() + (size_t)(f0 + nf) * ORPHEUS_FRAME_TOKENS);
        return deinterleave_tokens(sub, false);
    }

    // Decode whatever is decodable; append emitted samples to `out`.
    // final=true flushes the tail (and handles chunks shorter than W).
    void drain(SnacDecoder &snac, const Config &cfg, std::vector<float> &out, bool final) {
        constexpr int SPF = SnacDecoder::SAMPLES_PER_FRAME;
        int F = frames();

        // Head: first window emits frames 0-1
        if (emitted == 0) {
            if (F >= W) {
                std::vector<float> pcm = snac.decode_exact(cfg, window(0, W));
                if ((int)pcm.size() >= 2 * SPF) out.insert(out.end(), pcm.begin(), pcm.begin() + 2 * SPF);
                emitted = 2;
            } else if (final && F > 0) {
                std::vector<float> pcm = snac.decode_exact(cfg, window(0, F));
                size_t want = (size_t)F * SPF;
                if (pcm.size() > want) pcm.resize(want);
                out.insert(out.end(), pcm.begin(), pcm.end());
                emitted = F;
                return;
            } else {
                return;
            }
        }

        // Steady state: emit frame e from window [e-1, e+2]
        while (emitted >= 2 && emitted + 3 <= F) {
            std::vector<float> pcm = snac.decode_exact(cfg, window(emitted - 1, W));
            if ((int)pcm.size() >= 2 * SPF)
                out.insert(out.end(), pcm.begin() + SPF, pcm.begin() + 2 * SPF);
            emitted++;
        }

        // Tail: flush remaining frames from the last full window
        if (final && emitted < F) {
            int f0 = std::max(0, F - W);
            std::vector<float> pcm = snac.decode_exact(cfg, window(f0, F - f0));
            int off = (emitted - f0) * SPF;
            if (off >= 0 && off < (int)pcm.size())
                out.insert(out.end(), pcm.begin() + off, pcm.end());
            emitted = F;
        }
    }
};

// ── Offline re-decode (garble discriminator) ─────────────────────────────────
// Re-decode a captured .codes file with the SAME SnacDecoder + sliding-window
// logic, deterministically. Compare this WAV against the session's live .wav:
//   clean offline + garbled live  -> the contended on-GPU decode corrupted it
//   both garbled                  -> the model emitted bad audio codes
// Run with --snac-cpu for a contention-free reference decode.
static int decode_codes_file(const Config &cfg, SnacDecoder &snac) {
    std::ifstream f(cfg.decode_codes);
    if (!f) { std::cerr << "[orpheus-speak] cannot open codes file: " << cfg.decode_codes << "\n"; return 1; }
    std::vector<float> all;
    std::string line; std::vector<int> cur; int chunk = 0;
    auto flush = [&]() {
        if (cur.empty()) return;
        SlidingWindowDecoder d; d.feed(cur);
        std::vector<float> out; d.drain(snac, cfg, out, /*final=*/true);
        all.insert(all.end(), out.begin(), out.end());
        std::cerr << "[orpheus-speak] re-decode chunk " << chunk << ": "
                  << cur.size() << " codes -> " << out.size() << " samples\n";
        cur.clear(); ++chunk;
    };
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') { flush(); continue; }   // chunk-delimiter line
        std::istringstream is(line); int v; while (is >> v) cur.push_back(v);
    }
    flush();
    const std::string out_path = cfg.decode_codes + ".redecode.wav";
    if (!write_wav(out_path, all, cfg.sample_rate)) { std::cerr << "[orpheus-speak] WAV write failed\n"; return 1; }
    std::cerr << "[orpheus-speak] re-decode wrote " << out_path << " ("
              << all.size() << " samples, " << (float)all.size() / cfg.sample_rate << " s)\n";
    return 0;
}

// ── Per-chunk streaming state ────────────────────────────────────────────────
struct SessionSync {
    std::mutex m;
    std::condition_variable cv;
};

// One trigger-file line as it sits inside a chunk's text. Kept so an abort
// position (samples played) can be mapped back to talk-llama's line indexing
// without fuzzy text matching.
struct ChunkLine {
    size_t line_idx;   // global line number within the session (talk-llama's write order)
    size_t off;        // char offset of this line inside the chunk text
    size_t len;        // line length in chars
};

struct StreamChunk {
    std::string text;
    size_t index = 0;

    // SSE parse state (touched only by the chunk's curl thread)
    std::string sse_buf;       // unparsed SSE bytes
    std::string tok_tail;      // partial <custom_token_ text
    SlidingWindowDecoder dec;

    // Shared with playback thread (guarded by SessionSync::m)
    std::deque<float> pcm;
    bool done   = false;
    bool failed = false;
    int  tokens = 0;
    float server_ms = 0;
    size_t pcm_total = 0;          // total samples this chunk has produced
    std::vector<ChunkLine> lines;  // constituent trigger-file lines

    std::atomic<bool> *abort_flag = nullptr;   // session abort (barge-in)

    SessionSync *sync = nullptr;
    SnacDecoder *snac = nullptr;
    const Config *cfg = nullptr;
    std::chrono::high_resolution_clock::time_point t_post;
    bool first_pcm_logged = false;

    // capture-mode buffers (populated only when cfg->capture_dir is set)
    std::string cap_tokens;            // verbatim SSE content (model output) — for parser checks
    std::vector<float> cap_pcm;        // preserved copy of decoded PCM (playback drains c->pcm)
};

// SSE write callback: parse events, scan tokens, advance the window decoder,
// publish PCM.  Runs on the chunk's curl thread; decodes serialize through
// SnacDecoder::run_mutex.
static size_t sse_write_cb(char *data, size_t size, size_t nmemb, void *userp) {
    auto *c = static_cast<StreamChunk *>(userp);
    if (c->abort_flag && c->abort_flag->load()) return 0;   // short-write → curl aborts the transfer
    size_t total = size * nmemb;
    c->sse_buf.append(data, total);

    std::vector<int> new_codes;
    size_t nl;
    while ((nl = c->sse_buf.find('\n')) != std::string::npos) {
        std::string line = c->sse_buf.substr(0, nl);
        c->sse_buf.erase(0, nl + 1);
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("data: ", 0) != 0) continue;     // ignore non-data lines
        std::string payload = line.substr(6);
        if (payload.empty() || payload == "[DONE]") continue;
        // Defensive: never scan the final event.  Token events carry
        // "stop":false; some llama-server builds repeat the accumulated
        // content in the "stop":true event, which would duplicate audio.
        if (payload.find("\"stop\":true") != std::string::npos) continue;
        std::string content = extract_text_from_json(payload);
        if (content.empty()) continue;
        c->tok_tail += content;
        if (!c->cfg->capture_dir.empty()) c->cap_tokens += content;
        scan_tokens_streaming(c->tok_tail, new_codes);
    }

    if (!new_codes.empty()) {
        c->tokens += (int)new_codes.size();
        c->dec.feed(new_codes);
        std::vector<float> out;
        c->dec.drain(*c->snac, *c->cfg, out, /*final=*/false);
        if (!out.empty()) {
            if (c->cfg->verbose && !c->first_pcm_logged) {
                auto dt = std::chrono::duration<float, std::milli>(
                    std::chrono::high_resolution_clock::now() - c->t_post).count();
                std::cerr << "[orpheus-speak]   chunk " << c->index + 1
                          << ": first audio in " << (int)dt << " ms\n";
                c->first_pcm_logged = true;
            }
            std::lock_guard<std::mutex> lk(c->sync->m);
            c->pcm.insert(c->pcm.end(), out.begin(), out.end());
            if (!c->cfg->capture_dir.empty()) c->cap_pcm.insert(c->cap_pcm.end(), out.begin(), out.end());
            c->pcm_total += out.size();
            c->sync->cv.notify_all();
        }
    }
    return total;
}

// Streaming counterpart of http_generate: SSE POST with retry-on-503.
// Retries only if no frames were received (a 503 body carries no tokens).

// Barge-in: libcurl polls this between transfer events; returning nonzero
// cancels the SSE request, which also frees the llama-server slot for the
// pivot turn that is about to need the GPU.
static int sse_abort_cb(void *userp, curl_off_t, curl_off_t, curl_off_t, curl_off_t) {
    auto *c = static_cast<StreamChunk *>(userp);
    return (c->abort_flag && c->abort_flag->load()) ? 1 : 0;
}

static void http_stream_generate(StreamChunk *c) {
    const Config &cfg = *c->cfg;
    std::string prompt = build_orpheus_prompt(cfg.voice, c->text);
    std::string body   = build_request_json(cfg, prompt, /*stream=*/true);

    // One chunk = one std::async thread, so a thread_local handle would leak
    // one CURL handle per chunk over a long daemon run.  Local + cleanup.
    CURL *curl = curl_easy_init();
    if (!curl) {
        std::lock_guard<std::mutex> lk(c->sync->m);
        c->done = true; c->failed = true;
        c->sync->cv.notify_all();
        return;
    }
    struct curl_slist *headers = curl_slist_append(nullptr, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Connection: keep-alive");
    headers = curl_slist_append(headers, "Accept: text/event-stream");

    constexpr int max_retries = 3;
    bool ok = false;

    for (int attempt = 0; attempt < max_retries && !ok; attempt++) {
        curl_easy_reset(curl);
        curl_easy_setopt(curl, CURLOPT_URL, cfg.api_url.c_str());
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, body.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, sse_write_cb);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, c);
        curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
        curl_easy_setopt(curl, CURLOPT_TCP_NODELAY, 1L);
        curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
        curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 0L);
        curl_easy_setopt(curl, CURLOPT_XFERINFOFUNCTION, sse_abort_cb);
        curl_easy_setopt(curl, CURLOPT_XFERINFODATA, c);

        c->t_post = std::chrono::high_resolution_clock::now();
        CURLcode res = curl_easy_perform(curl);
        auto t1 = std::chrono::high_resolution_clock::now();
        c->server_ms = std::chrono::duration<float, std::milli>(t1 - c->t_post).count();

        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        if (res == CURLE_OK && http_code == 200) {
            ok = true;
        } else if (c->abort_flag && c->abort_flag->load()) {
            break;  // barge-in abort: no retries, no tail flush — just mark done
        } else if (c->dec.frames() == 0 && attempt + 1 < max_retries) {
            int backoff = 500 * (attempt + 1);
            std::cerr << "[orpheus-speak] chunk " << c->index + 1
                      << " HTTP " << http_code << ", retry in " << backoff << "ms\n";
            c->sse_buf.clear(); c->tok_tail.clear();
            std::this_thread::sleep_for(std::chrono::milliseconds(backoff));
        } else {
            break;  // mid-stream failure: keep what we have, give up
        }
    }

    // Final flush of the window tail, then mark done
    const bool aborted = c->abort_flag && c->abort_flag->load();
    std::vector<float> out;
    if (!aborted) c->dec.drain(*c->snac, cfg, out, /*final=*/true);
    {
        std::lock_guard<std::mutex> lk(c->sync->m);
        if (!out.empty()) {
            c->pcm.insert(c->pcm.end(), out.begin(), out.end());
            if (!cfg.capture_dir.empty()) c->cap_pcm.insert(c->cap_pcm.end(), out.begin(), out.end());
            c->pcm_total += out.size();
        }
        c->done = true;
        c->failed = !ok && c->dec.frames() == 0 && !aborted;
        c->sync->cv.notify_all();
    }
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
}

// ── Per-session capture (garble diagnosis) ───────────────────────────────────
// After a finished SSE session, dump: raw model output (.tokens), parsed audio
// codes — chunk-delimited and re-decodable (.codes), the live decoded audio that
// actually played (.wav), and a summary (.meta). Off unless --capture-dir is set;
// runs after playback.join(), off the realtime path.
static void write_capture(const Config &cfg,
                          const std::deque<std::unique_ptr<StreamChunk>> &chunks,
                          const std::string &report) {
    static std::atomic<int> sess{0};
    char base[600];
    std::snprintf(base, sizeof(base), "%s/tts_%ld_%03d",
                  cfg.capture_dir.c_str(), (long)time(nullptr), sess.fetch_add(1));
    const std::string b = base;

    { std::ofstream f(b + ".codes");
      for (auto &c : chunks) {
          f << "# chunk " << c->index << " (" << c->dec.codes.size() << " codes) \""
            << c->text << "\"\n";
          for (size_t i = 0; i < c->dec.codes.size(); ++i)
              f << c->dec.codes[i] << (((i + 1) % ORPHEUS_FRAME_TOKENS == 0) ? '\n' : ' ');
          f << "\n";
      } }

    { std::ofstream f(b + ".tokens");
      for (auto &c : chunks)
          f << "=== chunk " << c->index << " \"" << c->text << "\" ===\n"
            << c->cap_tokens << "\n\n"; }

    { std::vector<float> all;
      for (auto &c : chunks) all.insert(all.end(), c->cap_pcm.begin(), c->cap_pcm.end());
      write_wav(b + ".wav", all, cfg.sample_rate); }

    { std::ofstream f(b + ".meta");
      f << "epoch\t" << (long)time(nullptr) << "\nresult\t" << report
        << "\nchunks\t" << chunks.size() << "\n"
        << "idx\tchars\tcodes\tframes\tsamples\tserver_ms\n";
      for (auto &c : chunks)
          f << c->index << "\t" << c->text.size() << "\t" << c->dec.codes.size() << "\t"
            << (c->dec.codes.size() / ORPHEUS_FRAME_TOKENS) << "\t"
            << c->cap_pcm.size() << "\t" << c->server_ms << "\n"; }

    std::cerr << "[orpheus-speak] capture: " << b << ".{tokens,codes,wav,meta} ("
              << chunks.size() << " chunks)\n";
}

// ── SSE streaming session (replaces FILL/PLAY/REFILL when --stream-tts) ─────
// Returns the completion report written to the .done file:
//   "COMPLETE"            — the whole session played out
//   "INTERRUPTED <L> <C>" — barge-in: L lines fully spoken, C chars
//                           (word-snapped) spoken into line L
static std::string run_sse_session(const Config &cfg, SnacDecoder &snac) {
    if (cfg.verbose) std::cerr << "[orpheus-speak] session started (SSE streaming)\n";

    static constexpr int max_inflight = 5;
    static constexpr int CHUNK_CHAR_LIMIT = 300;

    // ── barge-in abort channel ──
    // talk-llama requests an abort by creating <base>.stop next to the
    // trigger file. Checked in the 20 ms session loop; the playback thread,
    // SSE transfers, and the sink all key off one atomic.
    std::string stop_path = cfg.watch_file;
    {
        auto dot = stop_path.rfind('.');
        if (dot != std::string::npos) stop_path = stop_path.substr(0, dot);
        stop_path += ".stop";
    }
    std::remove(stop_path.c_str());     // a stale stop must never pre-abort a session
    std::atomic<bool> session_abort{false};

    SessionSync sync;
    std::deque<std::unique_ptr<StreamChunk>> chunks;   // stable addresses
    std::vector<std::future<void>> tasks;
    size_t next_submit = 0;
    std::string pending_chunk;
    size_t file_pos = 0;
    bool end_received = false;

    size_t line_counter = 0;             // global line index (talk-llama's write order)
    std::vector<ChunkLine> pending_lines;

    auto t0 = std::chrono::high_resolution_clock::now();
    

    auto flush_pending = [&]() {
        if (pending_chunk.empty()) return;
        auto c = std::make_unique<StreamChunk>();
        c->text = pending_chunk;
        c->index = chunks.size();
        c->sync = &sync; c->snac = &snac; c->cfg = &cfg;
        c->abort_flag = &session_abort;
        c->lines = std::move(pending_lines);
        pending_lines.clear();
        {
            std::lock_guard<std::mutex> lk(sync.m);
            chunks.push_back(std::move(c));
        }
        pending_chunk.clear();
    };

    auto read_new_lines = [&]() {
        if (end_received) return;
        std::ifstream ifs(cfg.watch_file);
        if (!ifs) return;
        ifs.seekg(file_pos);
        std::string line;
        while (std::getline(ifs, line)) {
            file_pos = ifs.tellg();
            while (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.empty()) continue;
            if (line == "---END---") {
                flush_pending();
                end_received = true;
                return;
            }
            if (!pending_chunk.empty() &&
                (int)(pending_chunk.size() + 1 + line.size()) > CHUNK_CHAR_LIMIT) {
                flush_pending();
            }
            // record where this trigger-file line lands inside the chunk text
            // (offset accounts for the joining space added below)
            pending_lines.push_back({line_counter++,
                                     pending_chunk.empty() ? 0 : pending_chunk.size() + 1,
                                     line.size()});
            if (!pending_chunk.empty()) pending_chunk += " ";
            pending_chunk += line;
        }
    };

    auto inflight = [&]() -> size_t {
        std::lock_guard<std::mutex> lk(sync.m);
        size_t n = 0;
        for (size_t i = 0; i < next_submit; i++)
            if (!chunks[i]->done) n++;
        return n;
    };

    auto all_submitted_done = [&]() -> bool {
        std::lock_guard<std::mutex> lk(sync.m);
        for (size_t i = 0; i < next_submit; i++)
            if (!chunks[i]->done) return false;
        return true;
    };

    // ── Playback thread: consume chunk PCM in order into the raw sink ──
    std::atomic<bool> session_over{false};
    size_t total_samples = 0;
    float  first_audio_ms = 0;
    float  startup_buffer_ms = 0;     // audio buffered when playback began
    float  startup_low_water_ms = -1; // lowest device-fill estimate seen
    int    silence_ms_total = 0;      // injected stall-silence (device near-dry only)
    int    silence_events = 0;

    // barge-in: where the audio actually stopped, filled by the playback
    // thread at abort time (sample domain; mapped to text after join)
    int    abort_chunk_idx    = -1;   // chunk containing the last audible sample
    size_t abort_chunk_played = 0;    // samples of that chunk's PCM that played

    std::thread playback([&]() {
        PcmSink sink;
        bool sink_ok = sink.start(cfg.play_raw_cmd, cfg.sample_rate);
        bool wrote_any = false;
        bool started = false;         // startup watermark gate (once per session)
        const size_t prebuf_target =
            (size_t)((int64_t)cfg.sample_rate * std::max(0, cfg.prebuffer_ms) / 1000);
        std::vector<float> buf;
        size_t play_idx = 0;

        // ── Device-fill write clock ──────────────────────────────────────
        // The device consumes at exactly real-time once started, so
        //   buffered ≈ (samples written) − (wall time since first write).
        // Silence is injected ONLY when this estimate nears dry (<60 ms) —
        // NOT when the in-process queue is momentarily empty between 85 ms
        // frame arrivals (rev-2 bug: that fired whenever supply < 1.71× RT,
        // chopping the first utterance while ~400 ms of real audio sat
        // unplayed in aplay's buffer).
        std::chrono::high_resolution_clock::time_point t_play0;
        size_t written_samples = 0;
        float  min_buffered = 1e9f;
        auto buffered_ms_now = [&]() -> float {
            float el = std::chrono::duration<float, std::milli>(
                std::chrono::high_resolution_clock::now() - t_play0).count();
            return (float)written_samples * 1000.0f / cfg.sample_rate - el;
        };
        constexpr float LOW_WATER_MS = 60.0f;

        // barge-in: ordered record of what was written to the device —
        // (chunk index, samples) for PCM, (-1, samples) for injected silence.
        // At abort, real time pins the played position; this maps it back to
        // a chunk-relative sample count.
        std::vector<std::pair<int, size_t>> timeline;
        bool aborted_local = false;

        // ── diag (--diag): real-time underrun/stall tracing ──────────────────
        // A 1 Hz heartbeat of the device-fill estimate + cumulative stalls makes
        // a developing starvation visible as it happens (the session-end summary
        // is too late to correlate with the server log / GPU telemetry). A
        // stall-RUN marker (rate-limited) flags each acute dry spell. All emitted
        // to stderr, which the launcher's ts wrapper timestamps into athena.log.
        auto   diag_hb_t       = std::chrono::high_resolution_clock::now();
        auto   diag_stall_log_t= diag_hb_t - std::chrono::seconds(1);
        bool   diag_stalling   = false;
        float  diag_run_minfill= 1e9f;
        int    diag_run_inj    = 0;

        while (true) {
            buf.clear();
            if (cfg.diag && wrote_any) {
                auto now = std::chrono::high_resolution_clock::now();
                if (std::chrono::duration<float, std::milli>(now - diag_hb_t).count() >= 1000.0f) {
                    std::cerr << "[orpheus-speak] diag: chunk=" << play_idx
                              << " fill=" << (long)buffered_ms_now() << "ms"
                              << " played=" << (long)((float)written_samples * 1000.0f / cfg.sample_rate) << "ms"
                              << " stalls=" << silence_events
                              << " silence=" << silence_ms_total << "ms\n";
                    diag_hb_t = now;
                }
            }
            bool chunk_done = false, have_chunk = false;
            {
                std::unique_lock<std::mutex> lk(sync.m);
                sync.cv.wait_for(lk, std::chrono::milliseconds(50), [&] {
                    if (session_abort.load()) return true;
                    if (session_over.load() && play_idx >= chunks.size()) return true;
                    return play_idx < chunks.size() &&
                           (!chunks[play_idx]->pcm.empty() || chunks[play_idx]->done);
                });
                if (play_idx < chunks.size()) {
                    have_chunk = true;
                    auto &c = *chunks[play_idx];
                    chunk_done = c.done;
                    // Startup watermark: hold the session's first write until
                    // enough audio is buffered to ride out the turn-start
                    // generation deficit (Qwen still contending for the GPU).
                    // Chunk completion or session end starts playback with
                    // whatever exists (tiny chunks must not wait forever).
                    if (!started &&
                        (c.pcm.size() >= prebuf_target || c.done || session_over.load())) {
                        started = true;
                        startup_buffer_ms = (float)c.pcm.size() * 1000.0f / cfg.sample_rate;
                    }
                    if (started && !c.pcm.empty()) {
                        // Pull at most ~200 ms per pass. write_pcm blocks while
                        // the pipe is full, and generation runs >1.3x real-time,
                        // so handing a fully-decoded chunk to one write() pinned
                        // this thread for seconds (field: 4.5 s barge-in abort
                        // latency). Slicing bounds the block, so the abort
                        // check below runs every couple hundred ms of audio.
                        const size_t slice = (size_t) (cfg.sample_rate / 5);
                        const size_t n = std::min(c.pcm.size(), slice);
                        buf.assign(c.pcm.begin(), c.pcm.begin() + n);
                        c.pcm.erase(c.pcm.begin(), c.pcm.begin() + n);
                    }
                }
            }

            // ── barge-in abort: pin the played position, kill the player ──
            // The main session loop stops polling the stop file once every
            // chunk is submitted and decoded — which, at >1.3x real-time
            // generation, is long before playback finishes. The drain tail
            // must watch for it here (field: a barge in a story's tail went
            // unseen and 8.6 s of audio played to natural completion).
            if (!session_abort.load()) {
                struct stat st_stop;
                if (stat(stop_path.c_str(), &st_stop) == 0) {
                    session_abort.store(true);
                    if (cfg.verbose) std::cerr << "[orpheus-speak] abort requested (barge-in)\n";
                }
            }
            if (session_abort.load()) {
                if (wrote_any) {
                    const float el = std::chrono::duration<float, std::milli>(
                        std::chrono::high_resolution_clock::now() - t_play0).count();
                    size_t played = (size_t)std::max(0.0f, el * (float)cfg.sample_rate / 1000.0f);
                    played = std::min(played, written_samples);
                    size_t remaining = played;
                    std::vector<size_t> per;   // per-chunk samples played
                    for (const auto &seg : timeline) {
                        if (remaining == 0) break;
                        const size_t take = std::min(seg.second, remaining);
                        remaining -= take;
                        if (seg.first >= 0 && take > 0) {
                            if ((size_t)seg.first >= per.size()) per.resize(seg.first + 1, 0);
                            per[seg.first] += take;
                            abort_chunk_idx    = seg.first;
                            abort_chunk_played = per[seg.first];
                        }
                    }
                }
                aborted_local = true;
                break;
            }

            if (!buf.empty()) {
                if (!wrote_any) {
                    first_audio_ms = std::chrono::duration<float, std::milli>(
                        std::chrono::high_resolution_clock::now() - t0).count();
                    t_play0 = std::chrono::high_resolution_clock::now();
                    wrote_any = true;
                }
                written_samples += buf.size();
                total_samples += buf.size();
                timeline.emplace_back((int)play_idx, buf.size());
                if (sink_ok && !sink.write_pcm(buf.data(), buf.size())) sink_ok = false;
                continue;   // drain more immediately
            }

            if (started && have_chunk && chunk_done) { play_idx++; continue; }

            bool over = session_over.load() && play_idx >= [&]{
                std::lock_guard<std::mutex> lk(sync.m); return chunks.size(); }();
            if (over) break;

            // Anti-underrun guard: feed silence only when the DEVICE is
            // nearly dry — a genuine gap (waiting on the next chunk's text),
            // not the inter-frame arrival jitter of normal streaming.
            if (wrote_any && sink_ok) {
                float b = buffered_ms_now();
                if (b < min_buffered) min_buffered = b;
                if (b < LOW_WATER_MS) {
                    sink.write_silence_ms(40);
                    written_samples += (size_t)((int64_t)cfg.sample_rate * 40 / 1000);
                    timeline.emplace_back(-1, (size_t)((int64_t)cfg.sample_rate * 40 / 1000));
                    silence_ms_total += 40;
                    silence_events++;
                    if (cfg.diag) {
                        diag_run_inj++;
                        if (b < diag_run_minfill) diag_run_minfill = b;
                        // Mark the START of a contiguous dry spell (rate-limited so a
                        // choppy run can't flood). Recovery is logged on exit below.
                        auto now = std::chrono::high_resolution_clock::now();
                        if (!diag_stalling &&
                            std::chrono::duration<float, std::milli>(now - diag_stall_log_t).count() >= 250.0f) {
                            std::cerr << "[orpheus-speak] diag: STALL begin chunk=" << play_idx
                                      << " fill=" << (long)b << "ms\n";
                            diag_stall_log_t = now;
                        }
                        diag_stalling = true;
                    }
                } else if (cfg.diag && diag_stalling) {
                    std::cerr << "[orpheus-speak] diag: STALL end chunk=" << play_idx
                              << " injected=" << diag_run_inj << " (" << diag_run_inj * 40 << "ms)"
                              << " minfill=" << (long)diag_run_minfill << "ms\n";
                    diag_stalling    = false;
                    diag_run_inj     = 0;
                    diag_run_minfill = 1e9f;
                }
            }
        }
        startup_low_water_ms = (min_buffered > 1e8f) ? -1.0f : min_buffered;
        if (aborted_local) {
            sink.abort_now();   // device closes → silence within one ALSA period
        } else {
            sink.finish();      // blocks until the last sample has played
        }
    });

    // ── Main loop: read text, submit streaming requests ──
    while (true) {
        // barge-in: talk-llama dropped a stop file → abort the session
        if (!session_abort.load()) {
            struct stat st;
            if (stat(stop_path.c_str(), &st) == 0) {
                session_abort.store(true);
                sync.cv.notify_all();
                if (cfg.verbose) std::cerr << "[orpheus-speak] abort requested (barge-in)\n";
            }
        }
        if (session_abort.load()) break;   // stop reading and submitting

        read_new_lines();
        // Pipeline starved → push the partial sentence now (chunk-1 latency)
        if (!pending_chunk.empty() && next_submit == chunks.size() && all_submitted_done())
            flush_pending();

        size_t total;
        { std::lock_guard<std::mutex> lk(sync.m); total = chunks.size(); }
        while (next_submit < total && inflight() < (size_t)max_inflight) {
            StreamChunk *cp;
            { std::lock_guard<std::mutex> lk(sync.m); cp = chunks[next_submit].get(); }
            if (cfg.verbose) {
                std::cerr << "[orpheus-speak]   chunk " << next_submit + 1
                          << ": \"" << cp->text.substr(0, 60)
                          << (cp->text.size() > 60 ? "..." : "") << "\"\n";
            }
            tasks.push_back(std::async(std::launch::async, http_stream_generate, cp));
            next_submit++;
        }

        if (end_received && next_submit == total && all_submitted_done()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    for (auto &t : tasks) t.wait();
    session_over = true;
    sync.cv.notify_all();
    playback.join();

    // ── completion report ──
    // Map the abort position (chunk, samples played) back to talk-llama's
    // line indexing. Chars heard ≈ proportional position in the chunk's
    // audio; the playing chunk is almost always fully decoded (generation
    // runs ≥1.3× ahead of playback), so the mapping is usually exact. If it
    // is still streaming, estimate its total from the session's measured
    // samples-per-char.
    std::string report = "COMPLETE";
    if (session_abort.load()) {
        size_t rep_line = 0, rep_char = 0;
        std::lock_guard<std::mutex> lk(sync.m);
        if (abort_chunk_idx >= 0 && (size_t)abort_chunk_idx < chunks.size()) {
            const auto &c = *chunks[abort_chunk_idx];
            size_t total = c.pcm_total;
            if (!c.done || total == 0) {
                double spc = 2600.0;   // fallback ≈9 chars/s of speech at 24 kHz
                size_t s_sum = 0, ch_sum = 0;
                for (const auto &q : chunks)
                    if (q->done && q->pcm_total > 0) { s_sum += q->pcm_total; ch_sum += q->text.size(); }
                if (ch_sum > 0) spc = (double)s_sum / (double)ch_sum;
                total = std::max(total, (size_t)((double)c.text.size() * spc));
            }
            size_t pos = total ? (size_t)((double)c.text.size() * (double)abort_chunk_played / (double)total) : 0;
            pos = std::min(pos, c.text.size());

            bool placed = false;
            for (const auto &ln : c.lines) {
                if (pos < ln.off) {                  // in the joining gap → this line not started
                    rep_line = ln.line_idx; rep_char = 0; placed = true; break;
                }
                if (pos < ln.off + ln.len) {         // inside this line
                    rep_line = ln.line_idx;
                    size_t cc = pos - ln.off;
                    if (cc > 0 && cc < ln.len) {     // snap back to a word boundary
                        const std::string lt = c.text.substr(ln.off, ln.len);
                        const size_t sp = lt.rfind(' ', cc);
                        cc = (sp == std::string::npos) ? 0 : sp;
                    }
                    rep_char = cc; placed = true; break;
                }
            }
            if (!placed) {                           // past the chunk's last line
                rep_line = c.lines.empty() ? 0 : c.lines.back().line_idx + 1;
                rep_char = 0;
            }
        }
        report = "INTERRUPTED " + std::to_string(rep_line) + " " + std::to_string(rep_char);
        std::remove(stop_path.c_str());
        if (cfg.verbose)
            std::cerr << "[orpheus-speak] aborted at line " << rep_line
                      << ", char " << rep_char << "\n";
    }

    if (cfg.verbose && total_samples > 0) {
        auto total_ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        float server_ms = 0; int tok = 0; size_t nch;
        { std::lock_guard<std::mutex> lk(sync.m);
          nch = chunks.size();
          for (auto &c : chunks) { server_ms += c->server_ms; tok += c->tokens; } }
        std::cerr << "[orpheus-speak] streaming timing (SSE):\n"
                  << "  llama-server:  " << server_ms << " ms (" << tok << " tokens)\n"
                  << "  first audio:   " << first_audio_ms << " ms after session start\n"
                  << "  startup buf:   " << startup_buffer_ms << " ms (watermark "
                  << cfg.prebuffer_ms << " ms)\n"
                  << "  stall silence: " << silence_ms_total << " ms in "
                  << silence_events << " stalls\n"
                  << "  low water:     " << startup_low_water_ms << " ms min device fill\n"
                  << "  total wall:    " << total_ms << " ms\n"
                  << "  audio length:  " << (float)total_samples / cfg.sample_rate * 1000.0f << " ms\n"
                  << "  chunks:        " << nch << "\n"
                  << "  result:        " << report << "\n"
                  << "  pipeline:      SSE + 4-frame sliding-window decode (max "
                  << max_inflight << " in-flight)\n";
    }

    if (!cfg.capture_dir.empty()) write_capture(cfg, chunks, report);
    return report;
}

// ─────────────────────────────────────────────────────────────────────────────
// CLI
// ─────────────────────────────────────────────────────────────────────────────
static void usage(const char *argv0) {
    std::cerr
        << "Usage: " << argv0 << " [options] \"text to speak\"\n"
        << "       " << argv0 << " [options] -f input.txt\n"
        << "       " << argv0 << " [options] --watch /tmp/speak.txt  (daemon mode)\n\n"
        << "Options:\n"
        << "  -f FILE        Read input text from FILE\n"
        << "  -o FILE        Output WAV path        [/dev/shm/orpheus_tts.wav]\n"
        << "  --voice NAME   Orpheus voice           [tara]\n"
        << "  --api URL      llama-server URL        [http://127.0.0.1:8080/completion]\n"
        << "  --snac PATH    SNAC ONNX decoder path  [snac24_decoder.onnx]\n"
        << "  --play CMD     Play command after gen   (e.g. 'aplay')\n"
        << "  --watch FILE   Daemon mode: watch FILE for changes, speak each update\n"
        << "                 SNAC session stays alive — much faster after first utterance\n"
        << "  --temp F       Temperature              [0.6]\n"
        << "  --top-p F      Top-p                    [0.9]\n"
        << "  --rep-pen F    Repetition penalty        [1.1]\n"
        << "  --max-tokens N Max tokens to generate   [2500]\n"
        << "  --snac-cpu     Force SNAC decoder to CPU (frees ~1.7 GB VRAM)\n"
        << "  --stream-tts   Watch mode: SSE-stream tokens from llama-server and decode\n"
        << "                 incrementally (4-frame sliding window) into a persistent\n"
        << "                 raw-PCM player — first audio ~0.5s after chunk submission\n"
        << "  --play-raw CMD Raw-PCM player for --stream-tts ['aplay -q -t raw -f S16_LE -c 1 --buffer-time=300000']\n"
        << "                 ('-r <rate> -' is appended automatically)\n"
        << "  --capture-dir DIR Garble diagnosis: per-session dump of .tokens/.codes/.wav/.meta\n"
        << "  --decode-codes F  Offline: re-decode a captured .codes file to F.redecode.wav, exit\n"
        << "  --compare-wav A B Print correlation/RMS verdict between two WAVs (garble scoring), exit\n"
        << "  --prebuffer-ms N  Startup watermark for --stream-tts [350]: buffer N ms of\n"
        << "                 audio before a session's first write (turn-start chop guard)\n"
        << "  -v, --verbose  Verbose logging\n"
        << "  --diag         Real-time pipeline diagnostics: 1 Hz device-fill heartbeat\n"
        << "                 + stall-run markers (trace TTS underruns / server stalls)\n"
        << "  -h, --help     This message\n\n"
        << "Voices: tara, leah, jess, leo, dan, mia, zac, zoe\n"
        << "Emotion tags: <laugh> <chuckle> <sigh> <cough> <sniffle> <groan> <yawn> <gasp>\n";
}

static Config parse_args(int argc, char **argv) {
    Config cfg;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if ((a == "-f") && i + 1 < argc)             cfg.input_file = argv[++i];
        else if ((a == "-o") && i + 1 < argc)        cfg.output_wav = argv[++i];
        else if ((a == "--voice") && i + 1 < argc)    cfg.voice      = argv[++i];
        else if ((a == "--api") && i + 1 < argc)      cfg.api_url    = argv[++i];
        else if ((a == "--snac") && i + 1 < argc)     cfg.snac_model = argv[++i];
        else if ((a == "--play") && i + 1 < argc)     cfg.play_cmd   = argv[++i];
        else if ((a == "--watch") && i + 1 < argc)    cfg.watch_file = argv[++i];
        else if ((a == "--temp") && i + 1 < argc)     cfg.temperature= std::stof(argv[++i]);
        else if ((a == "--top-p") && i + 1 < argc)    cfg.top_p      = std::stof(argv[++i]);
        else if ((a == "--rep-pen") && i + 1 < argc)  cfg.rep_penalty= std::stof(argv[++i]);
        else if ((a == "--max-tokens") && i + 1 < argc) cfg.max_tokens= std::stoi(argv[++i]);
        else if (a == "-v" || a == "--verbose")       cfg.verbose     = true;
        else if (a == "--diag")                       cfg.diag        = true;
        else if (a == "--snac-cpu")                   cfg.snac_cpu    = true;
        else if (a == "--stream-tts")                 cfg.stream_tts  = true;
        else if (a == "--no-stream-tts")              cfg.stream_tts  = false;
        else if ((a == "--play-raw") && i + 1 < argc) cfg.play_raw_cmd = argv[++i];
        else if ((a == "--prebuffer-ms") && i + 1 < argc) cfg.prebuffer_ms = std::stoi(argv[++i]);
        else if ((a == "--capture-dir") && i + 1 < argc)  cfg.capture_dir  = argv[++i];
        else if ((a == "--decode-codes") && i + 1 < argc) cfg.decode_codes = argv[++i];
        else if ((a == "--compare-wav") && i + 2 < argc) { cfg.compare_a = argv[++i]; cfg.compare_b = argv[++i]; }
        else if (a == "-h" || a == "--help")          { usage(argv[0]); exit(0); }
        else if (a[0] != '-')                         cfg.text        = a;
        else { std::cerr << "Unknown option: " << a << "\n"; usage(argv[0]); exit(1); }
    }
    return cfg;
}

// ─────────────────────────────────────────────────────────────────────────────
// Audio playback via fork/exec (replaces system() to avoid shell overhead)
// ─────────────────────────────────────────────────────────────────────────────
// Splits play_cmd into argv, appends wav_path, forks and execs directly.
// Saves ~5ms per chunk vs system() by avoiding shell process spawning.
// Redirects child stderr to /dev/null (equivalent to 2>/dev/null).
static int play_audio(const std::string &play_cmd, const std::string &wav_path) {
    // Tokenize play_cmd (e.g. "aplay -q -D pulse" → ["aplay", "-q", "-D", "pulse"])
    std::vector<std::string> args;
    std::istringstream iss(play_cmd);
    std::string tok;
    while (iss >> tok) args.push_back(tok);
    args.push_back(wav_path);

    // Build null-terminated argv array for execvp
    std::vector<char *> argv;
    for (auto &a : args) argv.push_back(&a[0]);
    argv.push_back(nullptr);

    // posix_spawnp instead of fork()+execvp (CHANGES.MD §13 — no dup_mmap over the
    // CUDA/ONNX address space). Legacy batch path; behaviour identical.
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_addopen(&fa, STDERR_FILENO, "/dev/null", O_WRONLY, 0);
    pid_t pid = -1;
    int rc = posix_spawnp(&pid, argv[0], &fa, nullptr, argv.data(), environ);
    posix_spawn_file_actions_destroy(&fa);
    if (rc != 0) { errno = rc; return -1; }   // spawn failed
    int status;
    waitpid(pid, &status, 0);
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline: text → (chunked) API → tokens → SNAC → WAV → play
// ─────────────────────────────────────────────────────────────────────────────
// For short text (1-2 sentences): single Orpheus request, same as before.
// For long text (3+ sentences): parallel chunked processing:
//   - All-ahead submission: all chunks submitted as staggered concurrent HTTP requests
//   - Retry with backoff on HTTP 503 (slot busy)
//   - llama-server continuous batching fills up to -np slots in parallel
//   - Background playback: play each chunk via double-buffered WAV as it arrives
// This avoids Orpheus attention-skip on long text and stays within context.
static bool speak_text(const Config &cfg, SnacDecoder &snac, const std::string &text) {
    if (text.empty()) return false;

    auto sentences = split_sentences(text);
    bool use_chunked = (int)text.size() > 300;

    if (cfg.verbose) {
        std::cerr << "[orpheus-speak] voice=" << cfg.voice
                  << " text=\"" << text.substr(0, 80) << (text.size() > 80 ? "..." : "")
                  << "\"" << (use_chunked ? " [chunked: " + std::to_string(sentences.size()) + " sentences]" : "")
                  << "\n";
    }

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<float> all_pcm;

    if (use_chunked) {
        // ── Chunked mode: character-based grouping ─────────────────────
        // Orpheus-3B reliably vocalizes up to ~430 chars per request.
        // Beyond that it probabilistically hits <eot_id> before finishing.
        // Group sentences until adding the next would exceed the limit.
        static constexpr int CHUNK_CHAR_LIMIT = 300;  // safe for 2304 tokens/slot

        std::vector<std::string> chunks;
        std::string current;
        for (const auto &s : sentences) {
            if (!current.empty()
                && (int)(current.size() + 1 + s.size()) > CHUNK_CHAR_LIMIT) {
                chunks.push_back(current);
                current.clear();
            }
            if (!current.empty()) current += " ";
            current += s;
        }
        if (!current.empty()) chunks.push_back(current);

        // ── Parallel generation + pipelined playback ─────────────────
        //
        // Architecture:
        //   - All-ahead submission: submit ALL chunks as concurrent HTTP
        //     requests with a small stagger (200ms) between each.  With
        //     -np N, llama-server's continuous batching processes up to N
        //     in parallel on the GPU.
        //   - Retry with backoff: if a slot is momentarily busy (HTTP 503),
        //     http_generate retries up to 3 times with 500ms/1000ms backoff.
        //   - Background playback: play each chunk individually via
        //     double-buffered WAV as soon as its SNAC decode completes.

        std::string wav_buf[2] = { cfg.output_wav, cfg.output_wav + ".buf" };
        int buf_idx = 0;

        float total_server_ms = 0, total_decode_ms = 0;
        int   total_tokens    = 0;
        size_t total_pcm_samples = 0;

        // Sliding window submission — keep at most max_inflight HTTP
        // requests in flight at once.  With -np 3, this means 3 slots
        // actively generating + 2 queued and ready, so the GPU never
        // idles between chunks.  Prevents overwhelming llama-server
        // with dozens of concurrent connections on long responses
        // (which exhausts the retry budget for late chunks).
        //
        // For short responses (≤ max_inflight chunks), this behaves
        // identically to all-ahead submission.
        static constexpr int max_inflight = 5;  // np=3 active + 2 queued

        std::vector<std::future<HttpResult>> http_futures(chunks.size());
        size_t next_submit = 0;

        // Submit initial window with stagger
        for (; next_submit < std::min(chunks.size(), (size_t)max_inflight); next_submit++) {
            if (next_submit > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            http_futures[next_submit] = std::async(std::launch::async,
                http_generate, cfg.api_url, cfg.voice, chunks[next_submit],
                std::cref(cfg), (next_submit == 0) && cfg.verbose);
        }

        std::thread play_thread;
        auto play_end_time = std::chrono::high_resolution_clock::now();

        // Collect results in order; submit next chunk as each completes
        for (size_t i = 0; i < chunks.size(); i++) {
            if (cfg.verbose) {
                std::cerr << "[orpheus-speak]   chunk " << i + 1 << "/" << chunks.size()
                          << ": \"" << chunks[i].substr(0, 60)
                          << (chunks[i].size() > 60 ? "..." : "") << "\"\n";
            }

            // 1. Wait for this chunk's HTTP response (may already be done)
            HttpResult hr = http_futures[i].get();

            // Submit next chunk now that one result has been collected.
            // The slot that just finished (or is about to finish) will be
            // available for this new request.  No stagger needed — the
            // server is no longer at the initial submission burst.
            if (next_submit < chunks.size()) {
                http_futures[next_submit] = std::async(std::launch::async,
                    http_generate, cfg.api_url, cfg.voice, chunks[next_submit],
                    std::cref(cfg), false);
                next_submit++;
            }

            if (!hr.ok) {
                std::cerr << "[orpheus-speak]   chunk " << i + 1
                          << " HTTP failed after retries, skipping\n";
                continue;
            }

            total_server_ms += hr.server_ms;

            // 2. Parse tokens + SNAC decode (main thread — SNAC not thread-safe)
            std::string gen_text = extract_text_from_json(hr.response);
            if (gen_text.empty()) gen_text = hr.response;

            std::vector<int> audio_tokens = extract_audio_tokens(gen_text, false);
            if (audio_tokens.empty()) continue;

            int usable = ((int)audio_tokens.size() / ORPHEUS_FRAME_TOKENS)
                       * ORPHEUS_FRAME_TOKENS;
            audio_tokens.resize(usable);
            total_tokens += usable;

            SnacCodes codes = deinterleave_tokens(audio_tokens, false);

            auto t_s0 = std::chrono::high_resolution_clock::now();
            std::vector<float> pcm = snac.decode(cfg, codes);
            auto t_s1 = std::chrono::high_resolution_clock::now();
            total_decode_ms += std::chrono::duration<float, std::milli>(t_s1 - t_s0).count();

            if (pcm.empty()) continue;
            total_pcm_samples += pcm.size();

            float audio_ms = (float)pcm.size() / cfg.sample_rate * 1000.0f;

            // 3. Wait for PREVIOUS chunk's playback to finish
            if (play_thread.joinable()) play_thread.join();

            // Drain guard: aplay -D pulse may return before PulseAudio
            // finishes playing the audio.  Sleep until expected end time.
            {
                auto now = std::chrono::high_resolution_clock::now();
                if (now < play_end_time)
                    std::this_thread::sleep_until(play_end_time);
            }

            // 4. Write WAV to current buffer and start background playback
            if (!write_wav(wav_buf[buf_idx], pcm, cfg.sample_rate)) {
                std::cerr << "[orpheus-speak] failed to write: " << wav_buf[buf_idx] << "\n";
                continue;
            }

            if (!cfg.play_cmd.empty()) {
                play_end_time = std::chrono::high_resolution_clock::now()
                              + std::chrono::milliseconds((int)audio_ms + 100);

                std::string wav = wav_buf[buf_idx];
                std::string pcmd = cfg.play_cmd;
                play_thread = std::thread([pcmd, wav]() { play_audio(pcmd, wav); });
            }

            buf_idx = 1 - buf_idx;  // toggle double buffer
        }

        // Wait for final playback to finish
        if (play_thread.joinable()) play_thread.join();
        {
            auto now = std::chrono::high_resolution_clock::now();
            if (now < play_end_time)
                std::this_thread::sleep_until(play_end_time);
        }

        // Clean up secondary WAV buffer
        std::remove(wav_buf[1].c_str());

        auto t_end = std::chrono::high_resolution_clock::now();

        if (cfg.verbose) {
            auto total_ms = std::chrono::duration<float, std::milli>(t_end - t0).count();
            float audio_sec = (float)total_pcm_samples / cfg.sample_rate;
            std::cerr << "[orpheus-speak] pipelined timing:\n"
                      << "  llama-server:  " << total_server_ms << " ms (" << total_tokens << " tokens)\n"
                      << "  SNAC decode:   " << total_decode_ms << " ms\n"
                      << "  total wall:    " << total_ms << " ms\n"
                      << "  audio length:  " << audio_sec * 1000.0f << " ms\n"
                      << "  chunks:        " << chunks.size()
                      << " (from " << sentences.size() << " sentences)\n"
                      << "  pipeline:      sliding window (max " << max_inflight << " in-flight) + play-while-decode\n";
        }

    } else {
        // ── Single-shot mode: one Orpheus request for the whole text ─────
        std::string prompt = build_orpheus_prompt(cfg.voice, text);
        std::string body   = build_request_json(cfg, prompt);
        std::string response;

        if (!http_post(cfg.api_url, body, response, cfg.verbose) || response.empty()) {
            std::cerr << "[orpheus-speak] API call failed\n";
            return false;
        }

        auto t1 = std::chrono::high_resolution_clock::now();

        std::string gen_text = extract_text_from_json(response);
        if (gen_text.empty()) gen_text = response;

        std::vector<int> audio_tokens = extract_audio_tokens(gen_text, cfg.verbose);
        if (audio_tokens.empty()) {
            std::cerr << "[orpheus-speak] no audio tokens in response\n";
            return false;
        }

        int usable = ((int)audio_tokens.size() / ORPHEUS_FRAME_TOKENS) * ORPHEUS_FRAME_TOKENS;
        audio_tokens.resize(usable);

        SnacCodes codes = deinterleave_tokens(audio_tokens, cfg.verbose);
        if (codes.codes0.empty()) return false;

        auto t2 = std::chrono::high_resolution_clock::now();

        all_pcm = snac.decode(cfg, codes);
        if (all_pcm.empty()) return false;

        auto t3 = std::chrono::high_resolution_clock::now();

        if (!write_wav(cfg.output_wav, all_pcm, cfg.sample_rate)) {
            std::cerr << "[orpheus-speak] failed to write: " << cfg.output_wav << "\n";
            return false;
        }

        auto t4 = std::chrono::high_resolution_clock::now();

        if (!cfg.play_cmd.empty()) {
            play_audio(cfg.play_cmd, cfg.output_wav);
        }

        auto t5 = std::chrono::high_resolution_clock::now();

        if (cfg.verbose) {
            auto ms = [](auto a, auto b) {
                return std::chrono::duration<float, std::milli>(b - a).count();
            };
            float audio_sec = (float)all_pcm.size() / cfg.sample_rate;
            std::cerr << "[orpheus-speak] timing breakdown:\n"
                      << "  llama-server:  " << ms(t0, t1) << " ms\n"
                      << "  token parse:   " << ms(t1, t2) << " ms\n"
                      << "  SNAC decode:   " << ms(t2, t3) << " ms\n"
                      << "  WAV write:     " << ms(t3, t4) << " ms\n"
                      << "  playback:      " << ms(t4, t5) << " ms\n"
                      << "  total:         " << ms(t0, t5) << " ms\n"
                      << "  audio length:  " << audio_sec * 1000.0f << " ms\n"
                      << "  RTF:           " << ms(t0, t4) / (audio_sec * 1000.0f) << "x\n";
        }
    }

    return true;
}

// Read and trim a text file
static std::string read_text_file(const std::string &path) {
    std::ifstream ifs(path);
    if (!ifs) return "";
    std::ostringstream ss;
    ss << ifs.rdbuf();
    std::string text = ss.str();
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' '))
        text.pop_back();
    return text;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char **argv) {
    Config cfg = parse_args(argc, argv);

    // --compare-wav A B: no model/network needed — score two WAVs and exit.
    if (!cfg.compare_a.empty())
        return compare_wavs(cfg.compare_a, cfg.compare_b);

    // The streaming raw-PCM sink writes to a pipe; if the player dies we want
    // EPIPE from write(), not process death.
    std::signal(SIGPIPE, SIG_IGN);

    curl_global_init(CURL_GLOBAL_DEFAULT);

    // Initialize SNAC decoder once
    SnacDecoder snac;
    snac.init(cfg);
    if (!snac.ok) {
        std::cerr << "[orpheus-speak] SNAC decoder init failed\n";
        return 1;
    }

    // Offline garble discriminator: re-decode a captured .codes file and exit.
    if (!cfg.decode_codes.empty())
        return decode_codes_file(cfg, snac);

    // Capture mode: ensure the dump directory exists (best-effort).
    if (!cfg.capture_dir.empty()) mkdir(cfg.capture_dir.c_str(), 0755);

    if (!cfg.watch_file.empty()) {
        // Pre-warm the streaming window shape so the first session doesn't
        // pay ONNX kernel/arena setup for [1,4]/[1,8]/[1,16] (the one-time
        // ~700 ms first-chunk outlier observed in logs).
        if (cfg.stream_tts) {
            SnacCodes warm;
            warm.codes0.assign(4, 0);
            warm.codes1.assign(8, 0);
            warm.codes2.assign(16, 0);
            auto tw0 = std::chrono::high_resolution_clock::now();
            snac.decode_exact(cfg, warm);
            if (cfg.verbose) {
                auto ms = std::chrono::duration<float, std::milli>(
                    std::chrono::high_resolution_clock::now() - tw0).count();
                std::cerr << "[orpheus-speak] SNAC window warmup: " << (int)ms << " ms\n";
            }
        }
        // ── Watch mode: unified streaming protocol ──────────────────────
        // All speech (generation, goodbye, heard_ok) arrives as lines in
        // the trigger file, terminated by a ---END--- sentinel.
        //
        // Protocol:
        //   talk-llama appends "sentence\n" lines during token generation
        //   talk-llama appends "---END---\n" when done
        //   orpheus-speak reads lines incrementally, batches all available
        //   sentences into speak_text() for sliding-window pipelined playback
        //   orpheus-speak signals .done after END, deletes trigger file,
        //   and drains stale inotify events before returning to idle.
        //
        // Robustness guarantees:
        //   - Drain after session prevents stale events from causing
        //     duplicate processing or cross-session contamination
        //   - File deletion ensures clean state for next session
        //   - Single protocol eliminates batch/stream classification bugs
        std::cerr << "[orpheus-speak] watching " << cfg.watch_file
                  << " (SNAC session warm, Ctrl+C to quit)\n";

        // Derive .done path: <dir>/speak_tts.txt → <dir>/speak_tts.done
        std::string done_file = cfg.watch_file;
        auto dot = done_file.rfind('.');
        if (dot != std::string::npos) done_file = done_file.substr(0, dot);
        done_file += ".done";

        // ── Crash-restart recovery (CHANGES.MD §20) ──────────────────────────
        // If a PREVIOUS daemon instance died mid-session (e.g. the 20260702-134528
        // GPF), the trigger file it never consumed is still on disk — and since a
        // completed trigger receives no further writes, inotify below would NEVER
        // fire for it: a restarted daemon idles forever while the brain waits on a
        // .done that cannot come. Recover at startup:
        //   trigger exists AND contains ---END--- → the turn is fully written and
        //     its audio is unrecoverable (the old process died); consume it and
        //     release the waiting brain with a COMPLETE report.
        //   trigger exists WITHOUT ---END--- → the brain is still mid-generation;
        //     leave it — its next sentence append fires IN_CLOSE_WRITE and the
        //     normal session replays the turn from the top.
        {
            std::ifstream trig(cfg.watch_file);
            if (trig) {
                std::string content((std::istreambuf_iterator<char>(trig)),
                                    std::istreambuf_iterator<char>());
                trig.close();
                if (content.find("---END---") != std::string::npos) {
                    std::remove(cfg.watch_file.c_str());
                    std::ofstream(done_file) << "COMPLETE\n";
                    std::cerr << "[orpheus-speak] recovered stale completed session at startup"
                                 " (prior instance died mid-turn) — trigger consumed, .done released\n";
                } else {
                    std::cerr << "[orpheus-speak] stale in-progress trigger found at startup —"
                                 " leaving for the live session to complete\n";
                }
            }
        }

        // Extract directory and basename for inotify directory watch
        std::string watch_dir, watch_base;
        {
            std::vector<char> dp(cfg.watch_file.begin(), cfg.watch_file.end());
            std::vector<char> bp(cfg.watch_file.begin(), cfg.watch_file.end());
            dp.push_back('\0'); bp.push_back('\0');
            watch_dir  = dirname(dp.data());
            watch_base = basename(bp.data());
        }

        int ifd = inotify_init();
        if (ifd < 0) {
            perror("[orpheus-speak] inotify_init");
            return 1;
        }

        int wd = inotify_add_watch(ifd, watch_dir.c_str(),
                                   IN_MOVED_TO | IN_CLOSE_WRITE);
        if (wd < 0) {
            perror("[orpheus-speak] inotify_add_watch");
            close(ifd);
            return 1;
        }

        char evbuf[4096] __attribute__((aligned(__alignof__(struct inotify_event))));

        while (true) {
            // ── Idle: block until trigger file activity ──────────────────
            int len = read(ifd, evbuf, sizeof(evbuf));
            if (len <= 0) break;

            // Check if the event is for our trigger file, and classify type.
            // IN_MOVED_TO: speak-daemon.sh did atomic mv → batch mode (no END sentinel)
            // IN_CLOSE_WRITE: talk-llama streaming append → streaming mode (expects END)
            bool batch_trigger = false;
            bool stream_trigger = false;
            int pos = 0;
            while (pos < len) {
                auto *ev = reinterpret_cast<struct inotify_event *>(&evbuf[pos]);
                if (ev->len > 0 && watch_base == ev->name) {
                    if (ev->mask & IN_MOVED_TO)    batch_trigger = true;
                    if (ev->mask & IN_CLOSE_WRITE) stream_trigger = true;
                }
                pos += sizeof(struct inotify_event) + ev->len;
            }
            if (!batch_trigger && !stream_trigger) continue;

            // Verify file actually exists (could be stale event after drain)
            {
                struct stat st;
                if (stat(cfg.watch_file.c_str(), &st) != 0) continue;
            }

            // Completion report for the .done file: "COMPLETE", or
            // "INTERRUPTED <line> <char>" after a barge-in abort.
            std::string session_report = "COMPLETE";

            if (batch_trigger) {
                // ── Batch mode: speak-daemon.sh wrote complete file via mv ──
                // No ---END--- sentinel — read entire file, process, done.
                // This is the backward-compatible fallback for when
                // --stream-file is not set on talk-llama.
                if (cfg.verbose) {
                    std::cerr << "[orpheus-speak] batch session (speak-daemon.sh)\n";
                }
                std::string text = read_text_file(cfg.watch_file);
                if (!text.empty()) {
                    speak_text(cfg, snac, text);
                }
            } else if (cfg.stream_tts) {
                // ── Streaming via SSE + incremental window decode ─────────
                // Tokens decode to PCM as they arrive; a persistent raw-PCM
                // player gives gapless ordered playback and exact drain.
                session_report = run_sse_session(cfg, snac);
            } else {

            // ── Streaming: continuous pipeline until ---END--- ──────────
            // Unlike the batch approach (read → speak_text → read → speak_text),
            // this maintains a SINGLE sliding window across the entire session.
            // New sentences are read and chunked into the pipeline between each
            // collect/play cycle, so Orpheus generation for the next chunk
            // overlaps with playback of the current chunk — no gaps.
            if (cfg.verbose) {
                std::cerr << "[orpheus-speak] session started\n";
            }

            static constexpr int max_inflight = 5;
            static constexpr int CHUNK_CHAR_LIMIT = 300;

            // Pipeline state — persists across sentence arrivals
            std::vector<std::string> chunks;
            std::vector<std::future<HttpResult>> http_futures;
            size_t next_submit = 0;
            size_t next_collect = 0;

            // Sentence accumulator for chunking
            std::string pending_chunk;

            // File reading state
            size_t file_pos = 0;
            bool end_received = false;

            // Playback state
            std::string wav_buf[2] = { cfg.output_wav, cfg.output_wav + ".buf" };
            int buf_idx = 0;
            std::thread play_thread;
            auto play_end_time = std::chrono::high_resolution_clock::now();

            // Stats
            float total_server_ms = 0, total_decode_ms = 0;
            int total_tokens = 0;
            size_t total_pcm_samples = 0;
            auto t0 = std::chrono::high_resolution_clock::now();

            // ── Helper: flush pending_chunk into chunks vector ──
            auto flush_pending = [&]() {
                if (!pending_chunk.empty()) {
                    chunks.push_back(pending_chunk);
                    http_futures.resize(chunks.size());
                    pending_chunk.clear();
                }
            };

            // ── Helper: read available lines, chunk sentences ──
            auto read_new_lines = [&]() {
                if (end_received) return;
                std::ifstream ifs(cfg.watch_file);
                if (!ifs) return;
                ifs.seekg(file_pos);
                std::string line;
                while (std::getline(ifs, line)) {
                    file_pos = ifs.tellg();
                    while (!line.empty() && line.back() == '\r') line.pop_back();
                    if (line.empty()) continue;
                    if (line == "---END---") {
                        flush_pending();
                        end_received = true;
                        return;
                    }
                    // Group sentences into chunks ≤ CHUNK_CHAR_LIMIT
                    if (!pending_chunk.empty() &&
                        (int)(pending_chunk.size() + 1 + line.size()) > CHUNK_CHAR_LIMIT) {
                        chunks.push_back(pending_chunk);
                        http_futures.resize(chunks.size());
                        pending_chunk.clear();
                    }
                    if (!pending_chunk.empty()) pending_chunk += " ";
                    pending_chunk += line;
                }
            };

            // ── Helper: submit chunks up to sliding window limit ──
            auto submit_available = [&]() {
                while (next_submit < chunks.size() &&
                       (next_submit - next_collect) < (size_t)max_inflight) {
                    http_futures[next_submit] = std::async(std::launch::async,
                        http_generate, cfg.api_url, cfg.voice,
                        chunks[next_submit], std::cref(cfg),
                        (next_submit == 0) && cfg.verbose);
                    next_submit++;
                }
            };

            // ── Collect-and-decode helper (appends PCM to accumulator) ──
            std::vector<float> pcm_accum;
            float accum_audio_ms = 0;
            int accum_chunks = 0;

            auto collect_one = [&]() -> bool {
                if (next_collect >= next_submit) return false;

                if (cfg.verbose) {
                    std::cerr << "[orpheus-speak]   chunk " << next_collect + 1
                              << ": \"" << chunks[next_collect].substr(0, 60)
                              << (chunks[next_collect].size() > 60 ? "..." : "")
                              << "\"\n";
                }

                HttpResult hr = http_futures[next_collect].get();
                next_collect++;

                // Top up pipeline while we process this result
                read_new_lines();
                if (next_collect >= chunks.size() && !pending_chunk.empty())
                    flush_pending();
                submit_available();

                if (!hr.ok) {
                    std::cerr << "[orpheus-speak]   chunk " << next_collect
                              << " HTTP failed, skipping\n";
                    return false;
                }
                total_server_ms += hr.server_ms;

                std::string gen_text = extract_text_from_json(hr.response);
                if (gen_text.empty()) gen_text = hr.response;
                std::vector<int> audio_tokens = extract_audio_tokens(gen_text, false);
                if (audio_tokens.empty()) return false;
                int usable = ((int)audio_tokens.size() / ORPHEUS_FRAME_TOKENS)
                           * ORPHEUS_FRAME_TOKENS;
                audio_tokens.resize(usable);
                total_tokens += usable;
                SnacCodes codes = deinterleave_tokens(audio_tokens, false);
                auto t_s0 = std::chrono::high_resolution_clock::now();
                std::vector<float> pcm = snac.decode(cfg, codes);
                auto t_s1 = std::chrono::high_resolution_clock::now();
                total_decode_ms += std::chrono::duration<float, std::milli>(t_s1 - t_s0).count();
                if (pcm.empty()) return false;
                total_pcm_samples += pcm.size();

                // Append to accumulator — multiple chunks become one
                // seamless WAV with zero inter-chunk gaps.
                float audio_ms = (float)pcm.size() / cfg.sample_rate * 1000.0f;
                pcm_accum.insert(pcm_accum.end(), pcm.begin(), pcm.end());
                accum_audio_ms += audio_ms;
                accum_chunks++;
                return true;
            };

            // Buffer strategy: wait for MIN_BUFFER_INITIAL chunks before
            // FIRST playback to build a cushion, then switch to playing
            // whatever is available (MIN_BUFFER_SUBSEQUENT=1).  This avoids
            // the 10-30s silence gaps that occur when FILL demands 3 chunks
            // between every batch — after the initial buffer is built, the
            // REFILL phase keeps the pipeline fed during playback.
            static constexpr int MIN_BUFFER_INITIAL    = 1;
            static constexpr int MIN_BUFFER_SUBSEQUENT = 1;
            int min_buffer_now = MIN_BUFFER_INITIAL;

            while (true) {
                // ── FILL: collect + decode until buffer ready ──────────
                while (accum_chunks < min_buffer_now) {
                    read_new_lines();
                    if (next_collect >= chunks.size() && !pending_chunk.empty())
                        flush_pending();
                    submit_available();

                    if (next_collect < next_submit) {
                        collect_one();
                    } else {
                        // Nothing in flight — play what we have or wait
                        if (accum_chunks > 0 || end_received) break;
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    }
                }

                // Nothing decoded and session over → done
                if (pcm_accum.empty()) {
                    if (end_received && next_collect >= chunks.size()) break;
                    continue;
                }

                // ── PLAY: concatenated PCM as one seamless WAV ────────
                if (play_thread.joinable()) play_thread.join();
                {
                    auto now = std::chrono::high_resolution_clock::now();
                    if (now < play_end_time)
                        std::this_thread::sleep_until(play_end_time);
                }

                if (cfg.verbose && accum_chunks > 1) {
                    std::cerr << "[orpheus-speak]   playing " << accum_chunks
                              << " chunks combined (" << (int)accum_audio_ms
                              << " ms audio)\n";
                }

                if (!write_wav(wav_buf[buf_idx], pcm_accum, cfg.sample_rate)) {
                    pcm_accum.clear();
                    accum_audio_ms = 0;
                    accum_chunks = 0;
                    continue;
                }
                if (!cfg.play_cmd.empty()) {
                    play_end_time = std::chrono::high_resolution_clock::now()
                                  + std::chrono::milliseconds((int)accum_audio_ms + 100);
                    std::string wav = wav_buf[buf_idx];
                    std::string pcmd = cfg.play_cmd;
                    play_thread = std::thread([pcmd, wav]() { play_audio(pcmd, wav); });
                }
                buf_idx = 1 - buf_idx;

                pcm_accum.clear();
                accum_audio_ms = 0;
                accum_chunks = 0;

                // After first batch, switch to "play whatever is ready"
                // mode — the pipeline is primed, no need to re-buffer.
                min_buffer_now = MIN_BUFFER_SUBSEQUENT;

                // ── REFILL: collect more while current batch plays ────
                // While the combined WAV plays, decode the next batch.
                // Uses wait_for() instead of blocking .get() so we can
                // exit promptly when playback ends — even if Orpheus is
                // mid-generation.  Whatever chunks are ready at that
                // point get played immediately by FILL (min_buffer=1).
                {
                    auto now = std::chrono::high_resolution_clock::now();
                    while (now < play_end_time) {
                        read_new_lines();
                        if (next_collect >= chunks.size() && !pending_chunk.empty())
                            flush_pending();
                        submit_available();

                        if (next_collect < next_submit) {
                            // Non-blocking check: is the next result ready?
                            auto status = http_futures[next_collect].wait_for(
                                std::chrono::milliseconds(500));
                            if (status == std::future_status::ready) {
                                collect_one();
                            }
                            // If not ready, loop checks clock → exits if
                            // playback ended, avoiding the blocking gap.
                        } else if (end_received) {
                            break;
                        } else {
                            std::this_thread::sleep_for(std::chrono::milliseconds(50));
                        }
                        now = std::chrono::high_resolution_clock::now();
                    }
                }

                // Check if we're completely done
                if (end_received && next_collect >= chunks.size()
                    && pcm_accum.empty()) break;
            }

            // Wait for final playback
            if (play_thread.joinable()) play_thread.join();
            {
                auto now = std::chrono::high_resolution_clock::now();
                if (now < play_end_time)
                    std::this_thread::sleep_until(play_end_time);
            }
            std::remove(wav_buf[1].c_str());

            // Print session timing
            if (cfg.verbose && total_pcm_samples > 0) {
                auto t_end = std::chrono::high_resolution_clock::now();
                auto total_ms = std::chrono::duration<float, std::milli>(t_end - t0).count();
                float audio_sec = (float)total_pcm_samples / cfg.sample_rate;
                std::cerr << "[orpheus-speak] streaming timing:\n"
                          << "  llama-server:  " << total_server_ms
                          << " ms (" << total_tokens << " tokens)\n"
                          << "  SNAC decode:   " << total_decode_ms << " ms\n"
                          << "  total wall:    " << total_ms << " ms\n"
                          << "  audio length:  " << audio_sec * 1000.0f << " ms\n"
                          << "  chunks:        " << chunks.size() << "\n"
                          << "  pipeline:      continuous sliding window (max "
                          << max_inflight << " in-flight)\n";
            }

            } // end streaming mode (else branch)

            // ── Cleanup (shared by batch + streaming) ────────────────────
            std::remove(cfg.watch_file.c_str());

            // Drain ALL pending inotify events.  Each sentence append
            // triggered IN_CLOSE_WRITE; those events are now stale.
            // Without this drain, stale events would trigger duplicate
            // sessions that re-process deleted/recreated files.
            {
                int flags = fcntl(ifd, F_GETFL, 0);
                fcntl(ifd, F_SETFL, flags | O_NONBLOCK);
                while (read(ifd, evbuf, sizeof(evbuf)) > 0) { /* discard */ }
                fcntl(ifd, F_SETFL, flags);  // restore blocking
            }

            // Signal completion LAST.  The moment talk-llama reads the
            // report it may start a new session (barge-in resume writes the
            // unspoken remainder immediately), so the old trigger file and
            // its stale events must already be gone by then.
            std::ofstream(done_file) << session_report << "\n";

            if (cfg.verbose) {
                std::cerr << "[orpheus-speak] session complete\n";
            }
        }

        close(ifd);

    } else {
        // ── Single-shot mode ─────────────────────────────────────────────
        std::string text;
        if (!cfg.input_file.empty()) {
            text = read_text_file(cfg.input_file);
        } else {
            text = cfg.text;
        }

        // Trim
        while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' '))
            text.pop_back();

        if (text.empty()) {
            std::cerr << "[orpheus-speak] no input text\n";
            usage(argv[0]);
            snac.cleanup();
            return 1;
        }

        bool ok = speak_text(cfg, snac, text);
        snac.cleanup();
        curl_global_cleanup();
        return ok ? 0 : 1;
    }
}
