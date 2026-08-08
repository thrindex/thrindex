#pragma once
// crates/thrindex-backends/akida/runtime/tests/mock_driver.h
//
// MockAkidaDriver — in-process HardwareDriver implementation for FFI plumbing tests.
//
// Design
// ──────
// The mock driver has no physical hardware.  All register read/writes are no-ops.
// It allocates a 2 MB scratch buffer so the Engine Library can stage inputs.
// akida_visible_memory() returns 0, which forces the Engine Library to use scratch
// for all input/output staging (same as ThrindexPcieDriver on PCIe).
//
// The mock lets us:
//   1.  Prove that HardwareDevice::create() succeeds with an arbitrary driver.
//   2.  Prove that program() parses a real .fbz without segfaulting.
//   3.  Prove that set_batch_size / enqueue / fetch / dequantize bindings
//       compile and do not crash on a zero-tensor input.
//
// What the mock does NOT prove: that the output values are correct.
// Correct values require a physical AKD1500 (item 10).
//
// Confirmed: HardwareDriver is defined in infra/hardware_driver.h.
// Verified directly on the Pi — api/infra/hardware_driver.h exists and contains HardwareDriver.
// (Engine documentation incorrectly says hardware_device.h; the actual file is hardware_driver.h.)
#include "infra/hardware_driver.h"

#include <algorithm>    // std::fill
#include <cstdint>
#include <cstdlib>      // aligned_alloc / free
#include <cstring>      // memset

class MockAkidaDriver final : public HardwareDriver {
public:
    static constexpr size_t   kScratchSize  = 2u * 1024u * 1024u;  // 2 MB
    static constexpr uint32_t kTopLevelReg  = 0xFCC00000u;

    MockAkidaDriver()
    {
        // aligned_alloc is C11/C++17; posix_memalign is POSIX.
        // Use posix_memalign to match ThrindexPcieDriver and keep
        // the API consistent across drivers.
        if (::posix_memalign(&scratch_, 4096, kScratchSize) != 0) {
            scratch_ = nullptr;
        } else {
            std::memset(scratch_, 0, kScratchSize);
        }
    }

    ~MockAkidaDriver() override
    {
        if (scratch_) {
            std::free(scratch_);
            scratch_ = nullptr;
        }
    }

    // Not copyable or movable
    MockAkidaDriver(const MockAkidaDriver&) = delete;
    MockAkidaDriver& operator=(const MockAkidaDriver&) = delete;

    // ── HardwareDriver interface ─────────────────────────────────────────

    void read(uint32_t /*addr*/, void* data, size_t size) override
    {
        // Return all-zeros for every register read.
        std::memset(data, 0, size);
    }

    void write(uint32_t /*addr*/, const void* /*data*/, size_t /*size*/) override
    {
        // Silently discard all register writes.
    }

    const char* desc() const override
    {
        return "thrindex-mock-akida-driver";
    }

    uint32_t scratch_memory() override
    {
        // Same truncation caveat as ThrindexPcieDriver.
        return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(scratch_));
    }

    uint32_t scratch_size() override
    {
        return static_cast<uint32_t>(kScratchSize);
    }

    uint32_t top_level_reg() override
    {
        return kTopLevelReg;
    }

    uint32_t akida_visible_memory() override
    {
        // Force Engine Library to allocate input staging in scratch.
        return 0;
    }

    uint32_t akida_visible_memory_size() override
    {
        return 0;
    }

private:
    void* scratch_ = nullptr;
};
