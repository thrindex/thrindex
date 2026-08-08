#pragma once
// crates/thrindex-backends/akida/runtime/src/pcie_driver.h
//
// ThrindexPcieDriver — HardwareDriver implementation for AKD1500 over PCIe on Linux.
//
// The AKD1500 on the Raspberry Pi 5 M.2 HAT is a PCIe Gen2 endpoint.
// Register access is via BAR0, exposed by the Linux kernel as
//   /sys/bus/pci/devices/{pcie_addr}/resource0
// which is memory-mapped into process address space.
//
// Address model
// ─────────────
// The Engine Library calls driver->read/write with absolute MCU addresses,
// where top_level_reg() == 0xFCC00000 is the AKD1500 register base on the MCU.
// This driver subtracts that base to get the BAR0 byte offset.
//
// Scratch memory
// ──────────────
// Since akida_visible_memory() returns 0, the Engine Library allocates all input
// and output staging in the scratch buffer (set_batch_size allocate_inputs=true).
// The scratch buffer is a 2 MB heap allocation aligned to 4096 bytes.
//
// Thread safety
// ─────────────
// Not thread-safe.  The Engine Library and AkidaDevice wrapper are single-threaded.

// Confirmed: HardwareDriver is defined in infra/hardware_driver.h.
// Verified directly on the Pi — api/infra/hardware_driver.h exists and contains HardwareDriver.
// (Engine documentation incorrectly says hardware_device.h; the actual file is hardware_driver.h.)
#include "infra/hardware_driver.h"

#include <cstdint>
#include <stdexcept>
#include <string>

class ThrindexPcieDriver final : public HardwareDriver {
public:
    // pcie_addr example: "0001:01:00.0"
    // Path: /sys/bus/pci/devices/{pcie_addr}/resource0 (BAR0)
    explicit ThrindexPcieDriver(const char* pcie_addr);
    ~ThrindexPcieDriver() override;

    // Non-copyable, non-movable (holds live fd + mmap)
    ThrindexPcieDriver(const ThrindexPcieDriver&) = delete;
    ThrindexPcieDriver& operator=(const ThrindexPcieDriver&) = delete;

    // HardwareDriver interface ------------------------------------------------
    void     read(uint32_t addr, void* data, size_t size) override;
    void     write(uint32_t addr, const void* data, size_t size) override;
    const char* desc() const override;
    uint32_t scratch_memory() override;
    uint32_t scratch_size() override;
    uint32_t top_level_reg() override;
    uint32_t akida_visible_memory() override;
    uint32_t akida_visible_memory_size() override;

private:
    static constexpr uint32_t kTopLevelReg = 0xFCC00000u;
    static constexpr size_t   kScratchSize = 2u * 1024u * 1024u;  // 2 MB

    std::string pcie_addr_;
    int         bar_fd_   = -1;
    void*       bar_map_  = nullptr;
    size_t      bar_size_ = 0;
    void*       scratch_  = nullptr;

    void open_bar();
    void close_bar() noexcept;
};
