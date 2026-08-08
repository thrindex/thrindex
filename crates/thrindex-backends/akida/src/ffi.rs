// crates/thrindex-backends/akida/src/ffi.rs
//
// cxx bridge: Rust ↔ C++ bindings for the Engine Library shim.
// Compiled only when the `hardware` feature is enabled (see lib.rs).
//
// The C++ side is in src/bridge.cpp + src/bridge.h.
// Opaque types AkidaDevice and AkidaDense are complete in bridge.cpp;
// bridge.h forward-declares them for cxx's generated code.
//
// Lifetime contract
// ─────────────────
// UniquePtr<AkidaDevice> owns both the driver (PCIe or Mock) AND the
// shared_ptr<HardwareDevice>, in declaration order in the struct.
// Rust must not drop the UniquePtr while any UniquePtr<AkidaDense> from it
// is still live.  The runtime binary and mock test both satisfy this naturally
// (device outlasts all fetched outputs within a single run_batch call).

#[cxx::bridge(namespace = "thrindex")]
pub mod ffi {
    unsafe extern "C++" {
        include!("bridge.h");

        // ── Opaque types ──────────────────────────────────────────────────
        type AkidaDevice;
        type AkidaDense;

        // ── Device lifecycle ──────────────────────────────────────────────

        /// Create a hardware device backed by the Linux PCIe sysfs BAR0 driver.
        /// pcie_addr: "XXXX:XX:XX.X" e.g. "0001:01:00.0"
        fn create_pcie_device(pcie_addr: &str) -> Result<UniquePtr<AkidaDevice>>;

        /// Create a device backed by MockAkidaDriver (no physical chip needed).
        /// Used in tests/mock_runtime.rs integration test.
        fn create_mock_device() -> Result<UniquePtr<AkidaDevice>>;

        /// Load the .fbz program onto the device and cache ProgramInfo internally.
        fn device_program(dev: Pin<&mut AkidaDevice>, fbz_bytes: &[u8]) -> Result<()>;

        /// Allocate input staging for n_samples (always allocate_inputs=true).
        /// Returns the allocated buffer size in bytes.
        fn device_set_batch_size(dev: Pin<&mut AkidaDevice>, n: usize) -> Result<usize>;

        // ── Per-sample enqueue ─────────────────────────────────────────────

        /// Enqueue one uint8-encoded input sample.
        /// Returns false if the pipeline is full (caller must retry).
        fn device_enqueue_u8(
            dev: Pin<&mut AkidaDevice>,
            data: &[u8],
            n_features: u32,
        ) -> Result<bool>;

        // ── Output fetch ───────────────────────────────────────────────────

        /// Poll for one output tensor.  The returned UniquePtr is always non-null;
        /// check dense_ready() to know whether the hardware produced output.
        fn device_fetch(dev: Pin<&mut AkidaDevice>) -> Result<UniquePtr<AkidaDense>>;

        /// True iff the Dense holds a valid tensor (hardware produced output).
        fn dense_ready(dense: &AkidaDense) -> bool;

        /// Dequantize the raw integer output and return as float32 flat vec.
        fn dense_to_float_vec(dev: Pin<&mut AkidaDevice>, dense: &AkidaDense) -> Result<Vec<f32>>;

        // ── High-level batch inference ─────────────────────────────────────

        /// Full enqueue → fetch → dequantize loop for n_samples.
        /// input_u8:  flat uint8 buffer, layout [n_samples * n_features]
        /// Returns:   flat float32 buffer, layout [n_samples * out_features]
        /// Blocks until all n_samples outputs are received.
        fn akida_run_batch(
            dev: Pin<&mut AkidaDevice>,
            input_u8: &[u8],
            n_samples: u32,
            n_features: u32,
        ) -> Result<Vec<f32>>;
    }
}
