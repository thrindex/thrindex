// crates/thrindex-backends/akida/src/system_linux.cpp
//
// Engine Library runtime hooks required by infra/system.h.
//
// The Akida Engine Library declares four functions in infra/system.h that
// must be implemented by the application.  Without these, the linker will
// report "undefined reference to msleep" (and the others) when linking
// libthrindex_akida_bridge.a against libakida.a.
//
// Reference: AkidaCPPInferenceDeploymentPackage / src/system_linux.cpp
//   msleep       — sleep for N milliseconds (POSIX usleep)
//   time_ms      — monotonic time in milliseconds (clock_gettime CLOCK_MONOTONIC)
//   kick_watchdog — no-op on Linux host (watchdog is an MCU concept)
//   panic        — print formatted error to stderr and abort()
//
// These are declared as extern "C" or plain C in infra/system.h (no namespace).
// The #include below will bring in their exact declarations.

#include "infra/system.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>       // clock_gettime, CLOCK_MONOTONIC
#include <unistd.h>    // usleep

void msleep(uint32_t ms) {
    ::usleep(static_cast<useconds_t>(ms) * 1000u);
}

uint64_t time_ms() {
    struct timespec ts;
    ::clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000u
         + static_cast<uint64_t>(ts.tv_nsec) / 1000000u;
}

void kick_watchdog() {
    // No watchdog on a Linux host; this is a no-op.
}

void panic(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    ::vfprintf(stderr, fmt, args);
    va_end(args);
    ::fputc('\n', stderr);
    ::abort();
}
