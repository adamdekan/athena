// athena-kmsg-logger.cpp — durable, UNFILTERED kernel-log capture for ATHENA crash
// forensics.
//
// WHY: the 20260701-074343 kernel panic left NO backtrace. Two independent defects
// in the old bash follow (`sudo dmesg -w | grep -iE "NVRM|Xid|nvidia"`) destroyed it:
//   (A) the grep filter discarded the oops body — BUG:/Oops:/RIP:/register dump/
//       "Call Trace:"/stack frames contain none of NVRM/Xid/nvidia, so only the
//       "Modules linked in:" epilogue slipped through (it lists nvidia(OE));
//   (B) `>> file` with no fsync left ~68 s of kernel log in the page cache, lost on
//       the hard power-off.
//
// This program fixes both: it reads EVERY /dev/kmsg record and appends it to
//   <out>/kmsg-full.log   — UNFILTERED (keeps the whole oops/panic report), and
//   <out>/xid.log         — the NVRM/Xid/nvidia/GSP subset, for quick reading,
// calling fdatasync() after every line so at most ONE in-flight record is lost on a
// hard lock instead of tens of seconds.
//
// Pure C++/POSIX, no external deps (no _GNU_SOURCE / strcasestr needed).
//   Build: g++ -O2 -o athena-kmsg-logger athena-kmsg-logger.cpp
//   Run  : sudo -n ./athena-kmsg-logger <out_dir>     (needs dmesg read privilege)
// Stops cleanly on SIGTERM/SIGINT (flushes and exits 0).

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <ctime>
#include <cerrno>
#include <string>
#include <csignal>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>

static volatile sig_atomic_t g_stop = 0;
static void on_sig(int) { g_stop = 1; }

// Dependency-free case-insensitive substring search (avoids the GNU strcasestr).
static bool has_ci(const char* hay, const char* needle) {
    size_t nl = std::strlen(needle);
    if (nl == 0) return true;
    for (; *hay; ++hay) {
        size_t i = 0;
        while (i < nl && hay[i] &&
               std::tolower((unsigned char)hay[i]) == std::tolower((unsigned char)needle[i]))
            ++i;
        if (i == nl) return true;
    }
    return false;
}

static void wall_prefix(char* out, size_t n) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    localtime_r(&ts.tv_sec, &tm);
    char base[32];
    strftime(base, sizeof base, "%Y-%m-%d %H:%M:%S", &tm);
    snprintf(out, n, "[%s.%06ld]", base, (long)(ts.tv_nsec / 1000));
}

// append one line then fdatasync so it survives a hard lock
static void emit(int fd, const char* line, size_t len) {
    if (fd < 0) return;
    ssize_t w = write(fd, line, len);
    (void)w;
    fdatasync(fd);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <out_dir>\n", argv[0]); return 2; }
    std::string out = argv[1];
    std::string full_path = out + "/kmsg-full.log";
    std::string xid_path  = out + "/xid.log";

    signal(SIGTERM, on_sig);
    signal(SIGINT,  on_sig);
    signal(SIGPIPE, SIG_IGN);

    int full_fd = open(full_path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);
    int xid_fd  = open(xid_path.c_str(),  O_WRONLY | O_APPEND | O_CREAT, 0644);
    if (full_fd < 0) {
        fprintf(stderr, "[kmsg-logger] cannot open %s: %s\n", full_path.c_str(), strerror(errno));
        return 1;
    }

    int kfd = open("/dev/kmsg", O_RDONLY | O_NONBLOCK);
    if (kfd < 0) {
        fprintf(stderr, "[kmsg-logger] cannot open /dev/kmsg: %s (need sudo?)\n", strerror(errno));
        return 1;
    }

    {
        char p[64]; wall_prefix(p, sizeof p);
        char hdr[192];
        int hl = snprintf(hdr, sizeof hdr,
                          "%s --- kmsg-full follow start (UNFILTERED, fdatasync per line) ---\n", p);
        emit(full_fd, hdr, (size_t)hl);
        int xl = snprintf(hdr, sizeof hdr, "%s --- kmsg NVRM/Xid/GSP follow start ---\n", p);
        emit(xid_fd, hdr, (size_t)xl);
    }

    char buf[8192];
    while (!g_stop) {
        ssize_t n = read(kfd, buf, sizeof(buf) - 1);
        if (n < 0) {
            if (errno == EAGAIN) {                 // no new record yet
                struct pollfd pfd { kfd, POLLIN, 0 };
                poll(&pfd, 1, 500);                // 500 ms wake to re-check g_stop
                continue;
            }
            if (errno == EPIPE || errno == EINTR)  // ring wrapped / signal → keep going
                continue;
            break;                                 // unrecoverable
        }
        if (n == 0) continue;
        buf[n] = '\0';

        // /dev/kmsg record: "prio,seq,time_us,flags[,...];message\n[ cont dict...]"
        const char* semi = strchr(buf, ';');
        const char* msg  = semi ? semi + 1 : buf;

        long long t_us = 0;                        // 3rd comma-field = monotonic µs
        if (semi) {
            int commas = 0;
            const char* f2 = nullptr;
            for (const char* p = buf; p < semi; ++p)
                if (*p == ',') { if (++commas == 2) { f2 = p + 1; break; } }
            if (f2) t_us = atoll(f2);
        }

        // main message = up to first newline (ignore the key=value continuation dict)
        std::string m;
        for (const char* p = msg; *p && *p != '\n'; ++p) m += *p;

        char pfx[64]; wall_prefix(pfx, sizeof pfx);
        char line[9000];
        int ll = snprintf(line, sizeof line, "%s [%5lld.%06lld] %s\n",
                          pfx, t_us / 1000000, t_us % 1000000, m.c_str());
        if (ll < 0) continue;
        if ((size_t)ll >= sizeof line) ll = (int)sizeof line - 1;

        emit(full_fd, line, (size_t)ll);           // UNFILTERED — keeps oops/panic body
        if (has_ci(m.c_str(), "NVRM") || has_ci(m.c_str(), "Xid") ||
            has_ci(m.c_str(), "nvidia") || has_ci(m.c_str(), "GSP"))
            emit(xid_fd, line, (size_t)ll);         // GPU-fault subset
    }

    {
        char p[64]; wall_prefix(p, sizeof p);
        char hdr[128];
        int hl = snprintf(hdr, sizeof hdr, "%s --- kmsg follow stopped ---\n", p);
        emit(full_fd, hdr, (size_t)hl);
    }
    close(kfd);
    if (xid_fd >= 0) close(xid_fd);
    close(full_fd);
    return 0;
}
