#include "runtime_internal.h"

extern "C" {

extern void *fopen(const char *, const char *);

#ifdef BITS_64
#define WIN32API
#else
#define WIN32API __stdcall
#endif

typedef void *HANDLE;
typedef int BOOL;

HANDLE WIN32API CreateFileA(
    const char *lpFileName,
    unsigned long dwDesiredAccess,
    unsigned long dwShareMode,
    void *lpSecurityAttributes,
    unsigned long dwCreationDisposition,
    unsigned long dwFlagsAndAttributes,
    HANDLE hTemplateFile);

BOOL WIN32API WaitNamedPipeA(const char *lpNamedPipeName, unsigned long nTimeOut);
unsigned long WIN32API GetLastError(void);

int _open_osfhandle(intptr_t osfhandle, int flags);
void *_fdopen(int fd, const char *mode);

#define ERROR_PIPE_BUSY  231UL
#define PIPE_WAIT_MS     30000UL  // 30s should be more than enough

#define GENERIC_WRITE         0x40000000UL
#define OPEN_EXISTING         3UL
#define INVALID_HANDLE_VALUE  ((HANDLE)(intptr_t)-1)
#define HALIDE_O_WRONLY       0x0001
#define HALIDE_O_BINARY       0x8000

namespace {

// Match \\.\pipe\... or \\?\pipe\... (case-insensitive 'pipe')
bool is_named_pipe_path(const char *p) {
    if (!p) return false;
    if (p[0] != '\\' || p[1] != '\\') return false;
    if (p[2] != '.' && p[2] != '?') return false;
    if (p[3] != '\\') return false;
    if (p[4] != 'p' && p[4] != 'P') return false;
    if (p[5] != 'i' && p[5] != 'I') return false;
    if (p[6] != 'p' && p[6] != 'P') return false;
    if (p[7] != 'e' && p[7] != 'E') return false;
    if (p[8] != '\\') return false;
    return true;
}

}  // namespace

WEAK_INLINE void *halide_fopen(const char *filename, const char *type) {
    // Windows CRT fopen() rejects named pipe paths because the "ab" mode
    // (used by HL_TRACE_FILE handling) tries to seek to end-of-file, which
    // pipes don't support. Open the pipe via CreateFile + _open_osfhandle
    // + _fdopen so that live-streaming traces over a named pipe (e.g.
    // HL_TRACE_FILE=\\.\pipe\halide_trace) actually works.
    if (is_named_pipe_path(filename)) {
        // Race window: between the server's CreateNamedPipe and ConnectNamedPipe,
        // the pipe exists but is not yet "available for connection", and
        // CreateFile returns ERROR_PIPE_BUSY. WaitNamedPipe blocks until an
        // instance is connectable (or times out). Retry the open in case we
        // race again between WaitNamedPipe and CreateFile.
        HANDLE h = INVALID_HANDLE_VALUE;
        for (int attempt = 0; attempt < 5; ++attempt) {
            h = CreateFileA(filename, GENERIC_WRITE, 0, nullptr,
                            OPEN_EXISTING, 0, nullptr);
            if (h != INVALID_HANDLE_VALUE) break;
            if (GetLastError() != ERROR_PIPE_BUSY) return nullptr;
            WaitNamedPipeA(filename, PIPE_WAIT_MS);
        }
        if (h == INVALID_HANDLE_VALUE) return nullptr;

        int fd = _open_osfhandle((intptr_t)h, HALIDE_O_WRONLY | HALIDE_O_BINARY);
        if (fd < 0) return nullptr;
        return _fdopen(fd, "wb");
    }
    return fopen(filename, type);
}

}  // extern "C"
