// crates/thrindex-backends/akida/runtime/src/pcie_driver.cpp
//
// ThrindexPcieDriver implementation.
// See pcie_driver.h for design notes.

#include "pcie_driver.h"

#include <cstdlib>    // posix_memalign, free
#include <cstring>    // memcpy, memset
#include <stdexcept>
#include <sstream>

// Linux-specific headers (sysfs mmap BAR access)
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

// ---------------------------------------------------------------------------

ThrindexPcieDriver::ThrindexPcieDriver(const char* pcie_addr)
    : pcie_addr_(pcie_addr)
{
    // Allocate scratch buffer aligned to 4096 bytes.
    // posix_memalign is used per the user's specification.
    // NOTE: on aarch64 Linux, heap allocations are 64-bit; the uint32_t cast in
    // scratch_memory() truncates the upper 32 bits.  This matches the Engine
    // Library's expectation for MCU targets.  If this causes issues, replace
    // with mmap(MAP_32BIT|MAP_ANONYMOUS) to force placement in the first 2 GB.
    if (::posix_memalign(&scratch_, 4096, kScratchSize) != 0) {
        throw std::runtime_error("ThrindexPcieDriver: posix_memalign failed");
    }
    std::memset(scratch_, 0, kScratchSize);

    open_bar();
}

ThrindexPcieDriver::~ThrindexPcieDriver() {
    close_bar();
    if (scratch_) { std::free(scratch_); }
}

void ThrindexPcieDriver::open_bar() {
    // Build resource0 path
    std::string path = "/sys/bus/pci/devices/" + pcie_addr_ + "/resource0";

    bar_fd_ = ::open(path.c_str(), O_RDWR | O_SYNC);
    if (bar_fd_ < 0) {
        std::ostringstream ss;
        ss << "ThrindexPcieDriver: cannot open BAR0 at " << path
           << " (check PCIe address and kernel driver)";
        throw std::runtime_error(ss.str());
    }

    struct stat st;
    if (::fstat(bar_fd_, &st) < 0 || st.st_size == 0) {
        // resource0 st_size may be 0 on some kernels; fall back to 256 KB
        bar_size_ = 256u * 1024u;
    } else {
        bar_size_ = static_cast<size_t>(st.st_size);
    }

    bar_map_ = ::mmap(nullptr, bar_size_, PROT_READ | PROT_WRITE, MAP_SHARED, bar_fd_, 0);
    if (bar_map_ == MAP_FAILED) {
        ::close(bar_fd_);
        bar_fd_ = -1;
        bar_map_ = nullptr;
        throw std::runtime_error("ThrindexPcieDriver: mmap of BAR0 failed");
    }
}

void ThrindexPcieDriver::close_bar() noexcept {
    if (bar_map_ && bar_map_ != MAP_FAILED) {
        ::munmap(bar_map_, bar_size_);
        bar_map_ = nullptr;
    }
    if (bar_fd_ >= 0) {
        ::close(bar_fd_);
        bar_fd_ = -1;
    }
}

// ---------------------------------------------------------------------------
// HardwareDriver implementation
// ---------------------------------------------------------------------------

void ThrindexPcieDriver::read(uint32_t addr, void* data, size_t size) {
    // Translate MCU absolute address → BAR0 byte offset
    uint32_t offset = addr - kTopLevelReg;
    if (static_cast<size_t>(offset) + size > bar_size_) {
        throw std::out_of_range("ThrindexPcieDriver::read: access beyond BAR0 size");
    }
    std::memcpy(data, static_cast<uint8_t*>(bar_map_) + offset, size);
}

void ThrindexPcieDriver::write(uint32_t addr, const void* data, size_t size) {
    uint32_t offset = addr - kTopLevelReg;
    if (static_cast<size_t>(offset) + size > bar_size_) {
        throw std::out_of_range("ThrindexPcieDriver::write: access beyond BAR0 size");
    }
    std::memcpy(static_cast<uint8_t*>(bar_map_) + offset, data, size);
}

const char* ThrindexPcieDriver::desc() const {
    return "thrindex-pcie-akd1500";
}

uint32_t ThrindexPcieDriver::scratch_memory() {
    return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(scratch_));
}

uint32_t ThrindexPcieDriver::scratch_size() {
    return static_cast<uint32_t>(kScratchSize);
}

uint32_t ThrindexPcieDriver::top_level_reg() {
    return kTopLevelReg;
}

uint32_t ThrindexPcieDriver::akida_visible_memory() {
    // No DMA-coherent memory: forces Engine Library to use scratch for all I/O.
    return 0;
}

uint32_t ThrindexPcieDriver::akida_visible_memory_size() {
    return 0;
}
