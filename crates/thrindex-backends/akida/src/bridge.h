#pragma once
// crates/thrindex-backends/akida/src/bridge.h
//
// C++ declarations consumed by src/ffi.rs via include!("bridge.h").
//
// The cxx bridge sees only forward-declared opaque types here.  Their full
// definitions live in bridge.cpp.  This keeps the cxx-generated translation
// unit free of akida/Engine Library includes; only bridge.cpp needs them.
//
// Naming convention: all free functions in namespace thrindex use snake_case to
// match Rust identifiers exactly — cxx maps them 1:1.

#include "rust/cxx.h"   // rust::Str, rust::Slice<T>, rust::Vec<T>

#include <cstdint>
#include <memory>

// VERIFY:  exact include paths against engine/api/.
// The engine source drop may lay these out as:
//   api/akida/hardware_device.h  → HardwareDevice
//   api/infra/hardware_driver.h  → HardwareDriver (pure virtual interface)
//   api/input_conversion.h       → conversion::to_sparse, conversion::as_dense
//   api/akida/dense.h            → Dense, TensorType, Layout
//   api/akida/shape.h            → Shape
//
// We forward-declare the opaque wrapper structs here; the complete definitions
// are in bridge.cpp which has all the akida headers on its include path.

namespace thrindex {

// ---------------------------------------------------------------------------
// Opaque wrapper types
// ---------------------------------------------------------------------------

// AkidaDevice wraps:
//   - ThrindexPcieDriver or MockAkidaDriver (keeps driver alive as long as device)
//   - shared_ptr<akida::HardwareDevice>
//   - optional<akida::ProgramInfo>  (populated after device_program)
struct AkidaDevice;

// AkidaDense wraps the TensorUniquePtr returned by HardwareDevice::fetch().
// Always non-null (cxx UniquePtr safety contract); use dense_ready() to check.
struct AkidaDense;

// ---------------------------------------------------------------------------
// Device lifecycle
// ---------------------------------------------------------------------------

// Create a device backed by the Linux PCIe sysfs BAR0 driver.
// pcie_addr: "XXXX:XX:XX.X" e.g. "0001:01:00.0"
// Throws on BAR open / mmap failure → cxx translates to Rust Err.
std::unique_ptr<AkidaDevice> create_pcie_device(rust::Str pcie_addr);

// Create a device backed by the in-process MockAkidaDriver (no hardware needed).
// Used exclusively in tests/mock_runtime.rs integration test.
std::unique_ptr<AkidaDevice> create_mock_device();

// Load the .fbz program payload onto the device.
// Populates dev.program_info internally.  Must be called before run_batch.
void device_program(AkidaDevice& dev, rust::Slice<const uint8_t> fbz_bytes);

// Allocate input staging for n_samples.  Wraps set_batch_size(n, /*allocate_inputs=*/true).
// allocate_inputs is always true because akida_visible_memory() == 0 for both
// ThrindexPcieDriver and MockAkidaDriver.
// Returns the allocated input buffer size in bytes (from Engine Library).
std::size_t device_set_batch_size(AkidaDevice& dev, std::size_t n);

// ---------------------------------------------------------------------------
// Per-sample enqueue (single timestep T=1, uint8 4-in-u8 encoded features)
// ---------------------------------------------------------------------------

// Enqueue one input sample.  n_features is the length of data (== C when shape
// is [1, 1, n_features]).  Handles dense/sparse conversion internally.
// Returns false if the hardware pipeline is full — caller must retry.
bool device_enqueue_u8(AkidaDevice& dev,
                       rust::Slice<const uint8_t> data,
                       uint32_t n_features);

// ---------------------------------------------------------------------------
// Output fetch and dequantize
// ---------------------------------------------------------------------------

// Poll for one output tensor.  Always returns a non-null UniquePtr<AkidaDense>.
// Check dense_ready() to know whether the hardware produced output this call.
std::unique_ptr<AkidaDense> device_fetch(AkidaDevice& dev);

// Returns true iff the Dense holds a valid tensor (fetch returned non-null).
bool dense_ready(const AkidaDense& dense);

// Dequantize the raw integer output and read as float32.
// Internally calls device.hw->dequantize(dense.raw_dense) and copies the buffer.
rust::Vec<float> dense_to_float_vec(AkidaDevice& dev, const AkidaDense& dense);

// ---------------------------------------------------------------------------
// High-level batch inference (used by akida-runtime binary)
// ---------------------------------------------------------------------------

// Run the full enqueue → fetch → dequantize loop for n_samples in one call.
// input_u8:   flat uint8 buffer, layout [n_samples][n_features]
// Returns:    flat float32 buffer, layout [n_samples][out_features]
// Blocks until all n_samples outputs are received.
// Throws on any Engine Library error → cxx maps to Rust Err.
rust::Vec<float> akida_run_batch(AkidaDevice& dev,
                                  rust::Slice<const uint8_t> input_u8,
                                  uint32_t n_samples,
                                  uint32_t n_features);

} // namespace thrindex
