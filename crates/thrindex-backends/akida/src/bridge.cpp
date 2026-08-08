// crates/thrindex-backends/akida/src/bridge.cpp
//
// C++ shim implementations.  This translation unit is the ONLY one that
// includes Engine Library headers; everything else sees only bridge.h.
//
// Reference implementation: test/akd1500/simple_fully_connected/test.cpp
// from the Engine Library source drop.
//
// API resolutions (all VERIFY comments from the initial draft now resolved):
//
//   1. TensorType / DType
//      Engine docs says "TensorType" in prose; deployment package uses akida::DType.
//      Conclusion: the C++ enum is akida::DType, with enumerator uint8.
//      Fixed: akida::DType::uint8
//
//   2. Dense::Layout::RowMajor
//      Confirmed by user + engine docs.  The Layout is a nested type of Dense.
//      Fixed: akida::Dense::Layout::RowMajor
//
//   3. Shape brace-init
//      Confirmed by user (initializer_list<uint32_t>) + deployment package:
//      akida::Shape{h, w, c}.
//      Fixed: Shape{1u, 1u, n_features}
//
//   4. hardware_device.h include
//      Engine docs: "HardwareDriver class whose definition is in infra/hardware_device.h"
//      Fixed: #include "infra/hardware_device.h"
//
//   5. Dense::buffer()
//      Confirmed by engine docs: "buffer() to obtain a pointer to the underlying
//      Buffer object, that will provide a size and data methods."
//      Retained as-is.
//
//   6. Dequantize guard
//      Deployment package uses program_info.activation_enabled() to decide whether
//      to call dequantize (not always; only when quantized activations are present).
//      Fixed: conditional dequantize on activation_enabled().
//
//   7. Library name
//      Deployment package: libakida.a (not libakida_engine.a).
//      Fixed in build.rs (see build.rs changes).
//
//   REMAINING UNCERTAINTY:
//      to_sparse(const Dense&, const ProgramInfo&): user confirmed 2-arg form;
//      deployment package shows 1-arg form.  Keeping 2-arg per user's header read.
//      If compile fails, remove the second argument.

#include "bridge.h"

// --- Engine Library headers -------------------------------------------------
// Confirmed include paths (with {engine_path}/api/ on the include path):
#include "akida/hardware_device.h"    // akida::HardwareDevice, ProgramInfo, shared_ptr return
#include "akida/dense.h"              // akida::Dense, akida::DType, Dense::Layout, Shape
#include "input_conversion.h"         // akida::conversion::to_sparse, as_dense
// Confirmed: HardwareDriver is in infra/hardware_driver.h (verified on Pi).
// Engine documentation incorrectly says hardware_device.h; the actual file is hardware_driver.h.
#include "infra/hardware_driver.h"    // HardwareDriver (pure virtual interface)

// --- Local headers ----------------------------------------------------------
#include "runtime/src/pcie_driver.h"  // ThrindexPcieDriver
#include "runtime/tests/mock_driver.h" // MockAkidaDriver (test only; harmless in prod)

// --- Standard library -------------------------------------------------------
#include <memory>
#include <optional>
#include <stdexcept>
#include <vector>
#include <cstring>

// ---------------------------------------------------------------------------
// Full definitions of opaque wrapper types
// ---------------------------------------------------------------------------

namespace thrindex {

struct AkidaDevice {
    // The driver MUST outlive the HardwareDevice (shared_ptr holds a raw
    // pointer to the driver via HardwareDevice::create(HardwareDriver*)).
    std::unique_ptr<HardwareDriver>              driver;
    std::shared_ptr<akida::HardwareDevice>       hw;      // create() returns shared_ptr (confirmed)
    std::optional<akida::ProgramInfo>            program_info;

    bool has_program() const noexcept { return program_info.has_value(); }
};

struct AkidaDense {
    // Owns the Tensor returned by HardwareDevice::fetch().
    // Null / not-ready state when device_fetch returns "pending" (no output yet).
    std::unique_ptr<akida::Tensor>  tensor;           // from hw->fetch()
    const akida::Dense*             raw_dense = nullptr;  // from conversion::as_dense()
    bool                            is_ready  = false;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Construct a Dense view of shape [1, 1, n_features] (3D, H=1, W=1, C=n_features)
// from a uint8 host buffer.
//
// Corrected API (from deployment package + docs):
//   - First arg type: const char* per user's confirmed header reading; we cast from uint8_t*.
//   - DType:  akida::DType::uint8  (deployment package; docs confirm DType is the C++ type)
//   - Shape:  brace-init {1u, 1u, n_features}  (confirmed by user + deployment package)
//   - Layout: akida::Dense::Layout::RowMajor    (confirmed by user)
static std::unique_ptr<akida::Dense>
make_input_dense(const uint8_t* data, uint32_t n_features)
{
    akida::Shape shape{1u, 1u, static_cast<uint32_t>(n_features)};

    return akida::Dense::create_view(
        reinterpret_cast<const char*>(data),  // confirmed: const char* in user's header read
        akida::DType::uint8,                  // confirmed: DType enum, enumerator uint8
        shape,
        akida::Dense::Layout::RowMajor        // confirmed by user
    );
}

// ---------------------------------------------------------------------------
// Device lifecycle
// ---------------------------------------------------------------------------

std::unique_ptr<AkidaDevice> create_pcie_device(rust::Str pcie_addr)
{
    std::string addr(pcie_addr.begin(), pcie_addr.end());
    auto dev = std::make_unique<AkidaDevice>();
    dev->driver = std::make_unique<ThrindexPcieDriver>(addr.c_str());
    dev->hw = akida::HardwareDevice::create(dev->driver.get());
    if (!dev->hw) {
        throw std::runtime_error(
            "AkidaDevice: HardwareDevice::create returned null for PCIe addr " + addr);
    }
    return dev;
}

std::unique_ptr<AkidaDevice> create_mock_device()
{
    auto dev = std::make_unique<AkidaDevice>();
    dev->driver = std::make_unique<MockAkidaDriver>();
    dev->hw = akida::HardwareDevice::create(dev->driver.get());
    if (!dev->hw) {
        throw std::runtime_error(
            "AkidaDevice: HardwareDevice::create returned null for mock driver");
    }
    return dev;
}

void device_program(AkidaDevice& dev, rust::Slice<const uint8_t> fbz_bytes)
{
    dev.program_info = dev.hw->program(fbz_bytes.data(), fbz_bytes.size());
}

std::size_t device_set_batch_size(AkidaDevice& dev, std::size_t n)
{
    // allocate_inputs=true because akida_visible_memory()==0 for both ThrindexPcieDriver
    // and MockAkidaDriver; engine docs confirm this is required in that case.
    return dev.hw->set_batch_size(n, /*allocate_inputs=*/true);
}

// ---------------------------------------------------------------------------
// Per-sample enqueue
// ---------------------------------------------------------------------------

bool device_enqueue_u8(AkidaDevice& dev,
                        rust::Slice<const uint8_t> data,
                        uint32_t n_features)
{
    if (!dev.has_program()) {
        throw std::runtime_error("device_enqueue_u8: device not programmed");
    }
    if (data.size() < static_cast<std::size_t>(n_features)) {
        throw std::runtime_error("device_enqueue_u8: data slice shorter than n_features");
    }

    auto view = make_input_dense(data.data(), n_features);

    if (dev.program_info->input_is_dense()) {
        return dev.hw->enqueue(*view);
    } else {
        // 2-arg form confirmed by user's header read.
        // Deployment package shows 1-arg; if compile fails, remove second arg.
        auto sparse = akida::conversion::to_sparse(*view, *dev.program_info);
        return dev.hw->enqueue(*sparse);
    }
}

// ---------------------------------------------------------------------------
// Output fetch
// ---------------------------------------------------------------------------

std::unique_ptr<AkidaDense> device_fetch(AkidaDevice& dev)
{
    auto result = std::make_unique<AkidaDense>();
    auto tensor = dev.hw->fetch();  // TensorUniquePtr = unique_ptr<akida::Tensor>
    if (tensor) {
        result->raw_dense = akida::conversion::as_dense(*tensor);
        result->tensor    = std::move(tensor);
        result->is_ready  = true;
    }
    return result;
}

bool dense_ready(const AkidaDense& dense)
{
    return dense.is_ready;
}

rust::Vec<float> dense_to_float_vec(AkidaDevice& dev, const AkidaDense& dense)
{
    if (!dense.is_ready || !dense.raw_dense) {
        throw std::runtime_error("dense_to_float_vec: Dense is not ready");
    }

    // Conditional dequantize: activation_enabled() is true when quantized activations
    // are present (deployment package step 8; not just "output is float").
    const akida::Dense* out_dense = dense.raw_dense;
    std::unique_ptr<akida::Dense> dequantized_storage;
    if (dev.program_info->activation_enabled()) {
        dequantized_storage = dev.hw->dequantize(*dense.raw_dense);
        out_dense = dequantized_storage.get();
    }

    // Confirmed: Dense::buffer() returns Buffer* with size() (bytes) and data() methods.
    auto* buf  = out_dense->buffer();
    size_t n   = buf->size() / sizeof(float);
    const auto* ptr = reinterpret_cast<const float*>(buf->data());

    rust::Vec<float> out;
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        out.push_back(ptr[i]);
    }
    return out;
}

// ---------------------------------------------------------------------------
// High-level batch inference — exact reference: test.cpp inference loop
// ---------------------------------------------------------------------------

rust::Vec<float> akida_run_batch(AkidaDevice& dev,
                                   rust::Slice<const uint8_t> input_u8,
                                   uint32_t n_samples,
                                   uint32_t n_features)
{
    if (!dev.has_program()) {
        throw std::runtime_error("akida_run_batch: device not programmed");
    }
    if (input_u8.size() < static_cast<size_t>(n_samples) * n_features) {
        throw std::runtime_error("akida_run_batch: input_u8 too short");
    }

    // Step 7: set_batch_size, allocate_inputs=true (akida_visible_memory==0)
    dev.hw->set_batch_size(static_cast<std::size_t>(n_samples), true);

    // Steps 8-10: enqueue loop per sample, dense/sparse branch, retry on full pipeline
    for (uint32_t i = 0; i < n_samples; ++i) {
        const uint8_t* sample = input_u8.data() + static_cast<size_t>(i) * n_features;
        auto view = make_input_dense(sample, n_features);

        bool enqueued = false;
        while (!enqueued) {
            if (dev.program_info->input_is_dense()) {
                enqueued = dev.hw->enqueue(*view);
            } else {
                auto sparse = akida::conversion::to_sparse(*view, *dev.program_info);
                enqueued = dev.hw->enqueue(*sparse);
            }
        }
    }

    // Steps 11-12: fetch loop + conditional dequantize
    rust::Vec<float> result;
    for (uint32_t i = 0; i < n_samples; ++i) {
        // Poll until the hardware produces an output tensor
        std::unique_ptr<akida::Tensor> tensor;
        while (!tensor) {
            tensor = dev.hw->fetch();
        }

        const akida::Dense* raw = akida::conversion::as_dense(*tensor);

        // Conditional dequantize — activation_enabled() (deployment package step 8)
        std::unique_ptr<akida::Dense> dequantized_storage;
        const akida::Dense* out_dense = raw;
        if (dev.program_info->activation_enabled()) {
            dequantized_storage = dev.hw->dequantize(*raw);
            out_dense = dequantized_storage.get();
        }

        // Read float32 values via buffer()
        auto* buf = out_dense->buffer();
        size_t n  = buf->size() / sizeof(float);
        const auto* ptr = reinterpret_cast<const float*>(buf->data());
        for (size_t j = 0; j < n; ++j) {
            result.push_back(ptr[j]);
        }
    }

    return result;
}

} // namespace thrindex
