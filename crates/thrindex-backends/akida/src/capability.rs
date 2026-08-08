//! Capability descriptor for the BrainChip AKD1500 (ADR-0011 / RFC-004).
//!
//! ## Hardware constants (source notes)
//!
//! The following were confirmed from the official BrainChip product brief (Nov 2025)
//! and shop page (brainchipinc.com):
//! - On-chip SRAM: 1 MB
//! - Clock: 5–400 MHz; typical power: 250 mW at 400 MHz
//! - Process: 22 nm FD-SOI CMOS; package: 7×7 mm MFCTFBGA169
//! - Host interface: PCIe Gen2 endpoint; SPI S/D/Q/O peripheral
//! - Activation: bounded ReLU (Akida 1.0 — no LUT, no TNP)
//! - Weight precision: 4-bit signed values in int8 containers (confirmed: spike test,
//!   aarch64, Python 3.11.9, akida 2.19.2)
//!
//! NP count is UNVERIFIED from primary datasheet and is NOT encoded here.
//! The Engine Library discovers topology at runtime via HardwareDevice::version().
use thrindex_backend_api::{Capability, DelayFallback, Precision};

/// Canonical `.thx` target name for this backend.
pub const AKD1500_TARGET_NAME: &str = "akida-akd1500";

/// Maximum absolute int8 value used for 4-bit weight quantization.
///
/// 4-bit signed symmetric range: [-7, 7].
/// Quantization scale: `scale = max(|W_f32|) / AKD1500_WEIGHT_MAX`.
/// This is ~7× coarser than int8 quantization (which would use /127).
pub const AKD1500_WEIGHT_MAX: i8 = 7;

/// Returns the capability descriptor for the BrainChip AKD1500 (Akida 1.0).
///
/// `native_dt_ms` is declared as `1.0` but is **notional**: all LIF models are rejected
/// via E0401 before dt negotiation is ever consulted. The field is set to avoid
/// triggering the retiming path in the compiler.
pub fn akd1500_capability() -> Capability {
    Capability {
        name: AKD1500_TARGET_NAME.to_owned(),
        // Notional; AKD1500 inference at 400 MHz is well under 1 ms per frame.
        // LIF rejection (E0401) fires before this value is ever used for retiming.
        native_dt_ms: 1.0,
        // No Temporal Neural Processors on Akida 1.0; TNP_B/TNP_R are Akida 2.0 only.
        native_delay_max_steps: 0,
        // With no native delay support, any delayed model is rejected (E0402).
        // Emulation would require multi-frame buffering Akida 1.0 cannot do natively.
        delay_fallback: DelayFallback::Reject,
        // 4-bit weights/activations, per-tensor scale (confirmed: spike test).
        precision: Precision::Int4PerTensor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use thrindex_backend_api::{DelayFallback, Precision};

    #[test]
    fn descriptor_values() {
        let cap = akd1500_capability();
        assert_eq!(cap.name, "akida-akd1500");
        assert_eq!(cap.native_dt_ms, 1.0);
        assert_eq!(cap.native_delay_max_steps, 0);
        assert_eq!(cap.delay_fallback, DelayFallback::Reject);
        assert!(
            matches!(cap.precision, Precision::Int4PerTensor),
            "expected Int4PerTensor, got {:?}",
            cap.precision
        );
    }

    #[test]
    fn target_name_constant() {
        assert_eq!(AKD1500_TARGET_NAME, "akida-akd1500");
    }

    #[test]
    fn weight_max_is_four_bit_ceiling() {
        assert_eq!(AKD1500_WEIGHT_MAX, 7);
    }
}
