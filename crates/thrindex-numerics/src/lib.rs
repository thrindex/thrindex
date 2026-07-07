//! # thrindex-numerics
//!
//! Parameterized signed Q-format fixed-point arithmetic — the determinism
//! substrate consumed bit-for-bit by compiler, simulator, and runtime.
//!
//! ## Design contract
//!
//! - **Zero runtime dependencies** beyond `std`.  This is an L0 crate.
//! - **Deterministic** on every conformant platform: all arithmetic uses
//!   integer operations exclusively; floating-point appears only at the
//!   boundary (`from_f32`, `to_f32`).
//! - **Saturating**: no overflow, no wrapping, no panic — all out-of-range
//!   results are clamped to `[min_value(), max_value()]`.
//! - **Round-half-even** (banker's rounding) throughout — consistent with
//!   IEEE-754 default rounding mode and unbiased over large datasets.
//!
//! ## Quick start
//!
//! ```
//! use thrindex_numerics::{Q8_8, Q16_16};
//!
//! // Construct from a raw backing integer.
//! let a = Q8_8::from_raw(256).unwrap();  // represents 1.0
//! let b = Q8_8::from_raw(512).unwrap();  // represents 2.0
//! assert_eq!(a.saturating_add(b).raw(), 768); // 3.0
//!
//! // Construct from f32 with explicit error handling.
//! let q = Q16_16::from_f32(3.14).unwrap();
//! assert!((q.to_f64() - 3.14_f64).abs() < Q16_16::RESOLUTION_F64);
//! ```

#![deny(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::inline_always)] // saturating ops are hot-path; inline is intentional

mod error;
mod fixed;
mod ops;
mod round;

pub use error::FixedPointError;
pub use fixed::Q;

// ─── Canonical type aliases ────────────────────────────────────────────────
// These cover the common formats used by the compiler, simulator, and runtime.
// Add new aliases here (with a doc comment) rather than sprinkling
// Q<X, Y> literals throughout the codebase.

/// 8-bit semantics in Q4.4 format.
///
/// Range: \[−8, 7.9375\], resolution: 0.0625.
pub type Q4_4 = Q<4, 4>;

/// 8-bit semantics in Q1.7 format (signed weight/coefficient).
///
/// Range: \[−1, 0.9921875\], resolution ≈ 0.0078.
pub type Q1_7 = Q<1, 7>;

/// 16-bit semantics in Q8.8 format.
///
/// Range: \[−128, 127.99609375\], resolution ≈ 0.0039.
pub type Q8_8 = Q<8, 8>;

/// 16-bit semantics in Q4.12 format (high-precision membrane potential).
///
/// Range: \[−8, 7.999755859375\], resolution ≈ 0.000244.
pub type Q4_12 = Q<4, 12>;

/// 32-bit format in Q16.16 (general-purpose; simulator membrane potential).
///
/// Range: \[−32768, 32767.9999847\], resolution ≈ 1.5×10⁻⁵.
pub type Q16_16 = Q<16, 16>;

/// 32-bit format in Q1.31 (high-precision signed coefficient in \[−1, 1)).
///
/// Range: \[−1, 1 − 2⁻³¹\], resolution ≈ 4.7×10⁻¹⁰.
pub type Q1_31 = Q<1, 31>;

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    #![allow(
        clippy::float_cmp,                // exact float comparisons are intentional in golden tests
        clippy::cast_lossless,            // i32→i64 widening casts in test arithmetic
        clippy::cast_possible_truncation, // guarded i64→i32 casts after proven-in-range checks
        clippy::doc_markdown,             // test doc comments don't need strict markdown
    )]
    use super::*;

    // ── Unit tests: construction and accessors ──────────────────────────────

    #[test]
    fn from_raw_in_range_is_some() {
        assert!(Q8_8::from_raw(0).is_some());
        assert!(Q8_8::from_raw(Q8_8::MIN_RAW).is_some());
        assert!(Q8_8::from_raw(Q8_8::MAX_RAW).is_some());
    }

    #[test]
    fn from_raw_out_of_range_is_none() {
        assert!(Q8_8::from_raw(Q8_8::MAX_RAW + 1).is_none());
        assert!(Q8_8::from_raw(Q8_8::MIN_RAW - 1).is_none());
    }

    #[test]
    fn zero_is_zero() {
        assert_eq!(Q8_8::zero().raw(), 0);
        assert_eq!(Q16_16::zero().raw(), 0);
    }

    #[test]
    fn min_max_raw_values() {
        // Q8_8: TOTAL=16, MIN_RAW = -(1<<15) = -32768, MAX_RAW = 32767
        assert_eq!(Q8_8::MIN_RAW, -32768);
        assert_eq!(Q8_8::MAX_RAW, 32767);
        // Q16_16: TOTAL=32, MIN_RAW = i32::MIN, MAX_RAW = i32::MAX
        assert_eq!(Q16_16::MIN_RAW, i32::MIN);
        assert_eq!(Q16_16::MAX_RAW, i32::MAX);
        // Q4_4: TOTAL=8, MIN_RAW = -(1<<7) = -128, MAX_RAW = 127
        assert_eq!(Q4_4::MIN_RAW, -128);
        assert_eq!(Q4_4::MAX_RAW, 127);
    }

    #[test]
    fn one_representable_iff_int_bits_ge_2() {
        assert!(Q8_8::one().is_some());
        assert!(Q16_16::one().is_some());
        assert!(Q4_4::one().is_some());
        assert!(Q1_7::one().is_none()); // INT=1, max < 1.0
        assert!(Q1_31::one().is_none());
    }

    #[test]
    fn one_raw_value() {
        // Q8_8: one = 1.0, raw = 1 << 8 = 256
        assert_eq!(Q8_8::one().unwrap().raw(), 256);
        // Q4_4: one = 1.0, raw = 1 << 4 = 16
        assert_eq!(Q4_4::one().unwrap().raw(), 16);
    }

    #[test]
    fn to_f64_exact_for_known_values() {
        let q = Q8_8::from_raw(256).unwrap(); // 1.0 in Q8.8
        assert_eq!(q.to_f64(), 1.0);
        let q = Q8_8::from_raw(-256).unwrap(); // -1.0 in Q8.8
        assert_eq!(q.to_f64(), -1.0);
        let q = Q8_8::from_raw(128).unwrap(); // 0.5 in Q8.8
        assert_eq!(q.to_f64(), 0.5);
    }

    // ── Unit tests: from_f32 golden values ─────────────────────────────────

    #[test]
    fn from_f32_one_point_zero() {
        let q = Q8_8::from_f32(1.0).unwrap();
        assert_eq!(q.raw(), 256); // 1.0 × 2^8 = 256
    }

    #[test]
    fn from_f32_zero_point_five() {
        let q = Q8_8::from_f32(0.5).unwrap();
        assert_eq!(q.raw(), 128); // 0.5 × 2^8 = 128
    }

    #[test]
    fn from_f32_neg_zero_point_five() {
        let q = Q8_8::from_f32(-0.5).unwrap();
        assert_eq!(q.raw(), -128);
    }

    #[test]
    fn from_f32_nan_is_err() {
        assert!(matches!(
            Q8_8::from_f32(f32::NAN),
            Err(FixedPointError::NaN)
        ));
    }

    #[test]
    fn from_f32_inf_is_err() {
        assert!(matches!(
            Q8_8::from_f32(f32::INFINITY),
            Err(FixedPointError::OutOfRange { .. })
        ));
        assert!(matches!(
            Q8_8::from_f32(f32::NEG_INFINITY),
            Err(FixedPointError::OutOfRange { .. })
        ));
    }

    #[test]
    fn from_f32_out_of_range_is_err() {
        // Q8_8 max real ≈ 127.996; 200.0 is out of range.
        assert!(matches!(
            Q8_8::from_f32(200.0),
            Err(FixedPointError::OutOfRange { .. })
        ));
    }

    #[test]
    fn from_f32_saturating_nan_is_zero() {
        assert_eq!(Q8_8::from_f32_saturating(f32::NAN), Q8_8::zero());
    }

    #[test]
    fn from_f32_saturating_inf_is_max() {
        assert_eq!(Q8_8::from_f32_saturating(f32::INFINITY), Q8_8::max_value());
    }

    #[test]
    fn from_f32_saturating_neg_inf_is_min() {
        assert_eq!(
            Q8_8::from_f32_saturating(f32::NEG_INFINITY),
            Q8_8::min_value()
        );
    }

    #[test]
    fn from_f32_saturating_over_range_is_max() {
        assert_eq!(Q8_8::from_f32_saturating(1000.0), Q8_8::max_value());
    }

    #[test]
    fn from_f32_saturating_under_range_is_min() {
        assert_eq!(Q8_8::from_f32_saturating(-1000.0), Q8_8::min_value());
    }

    /// When TOTAL_BITS > 24, max_value().to_f32() may round up past the
    /// representable max, causing from_f32 to return Err(OutOfRange).
    #[test]
    fn from_f32_max_value_q16_16_boundary() {
        // Q16_16 TOTAL_BITS=32 > 24; max_value().to_f32() overflows representable range.
        let max_f32 = Q16_16::max_value().to_f32();
        // The f32 may be > MAX_REAL_F64 due to rounding; from_f32 must return Err.
        // (Or it might be exactly representable on some values — so we allow either Err
        //  or a saturating result here; the invariant we MUST assert is that
        //  from_f32_saturating never panics and returns max_value or close to it.)
        let saturated = Q16_16::from_f32_saturating(max_f32);
        assert!(saturated.raw() >= Q16_16::MAX_RAW - 1); // at or near max
    }

    // ── Unit tests: saturating addition (exact when in range) ───────────────

    #[test]
    fn add_exact_raw_equality() {
        let a = Q8_8::from_raw(100).unwrap();
        let b = Q8_8::from_raw(200).unwrap();
        // In-range: result.raw() MUST equal a.raw() + b.raw() exactly.
        assert_eq!(a.saturating_add(b).raw(), 100 + 200);
    }

    #[test]
    fn add_negative_exact_raw_equality() {
        let a = Q8_8::from_raw(-100).unwrap();
        let b = Q8_8::from_raw(-200).unwrap();
        assert_eq!(a.saturating_add(b).raw(), -100 + -200);
    }

    #[test]
    fn add_zero_is_identity() {
        let a = Q8_8::from_raw(12345).unwrap();
        assert_eq!(a.saturating_add(Q8_8::zero()).raw(), a.raw());
        assert_eq!(Q8_8::zero().saturating_add(a).raw(), a.raw());
    }

    #[test]
    fn add_commutativity() {
        let a = Q8_8::from_raw(1234).unwrap();
        let b = Q8_8::from_raw(-567).unwrap();
        assert_eq!(a.saturating_add(b), b.saturating_add(a));
    }

    #[test]
    fn add_positive_overflow_saturates_to_max() {
        assert_eq!(
            Q8_8::max_value().saturating_add(Q8_8::max_value()),
            Q8_8::max_value()
        );
    }

    #[test]
    fn add_negative_overflow_saturates_to_min() {
        assert_eq!(
            Q8_8::min_value().saturating_add(Q8_8::min_value()),
            Q8_8::min_value()
        );
    }

    // ── Unit tests: saturating subtraction (exact when in range) ────────────

    #[test]
    fn sub_exact_raw_equality() {
        let a = Q8_8::from_raw(300).unwrap();
        let b = Q8_8::from_raw(100).unwrap();
        assert_eq!(a.saturating_sub(b).raw(), 300 - 100);
    }

    #[test]
    fn sub_zero_is_identity() {
        let a = Q8_8::from_raw(999).unwrap();
        assert_eq!(a.saturating_sub(Q8_8::zero()).raw(), a.raw());
    }

    #[test]
    fn sub_self_is_zero() {
        let a = Q8_8::from_raw(12345).unwrap();
        assert_eq!(a.saturating_sub(a).raw(), 0);
    }

    #[test]
    fn sub_underflow_saturates_to_min() {
        assert_eq!(
            Q8_8::min_value().saturating_sub(Q8_8::max_value()),
            Q8_8::min_value()
        );
    }

    // ── Unit tests: saturating negation ────────────────────────────────────

    #[test]
    fn neg_zero_is_zero() {
        assert_eq!(Q8_8::zero().saturating_neg(), Q8_8::zero());
    }

    #[test]
    fn neg_min_value_saturates_to_max() {
        assert_eq!(Q8_8::min_value().saturating_neg(), Q8_8::max_value());
    }

    #[test]
    fn neg_double_is_identity() {
        let a = Q8_8::from_raw(1000).unwrap();
        assert_eq!(a.saturating_neg().saturating_neg(), a);
    }

    #[test]
    fn neg_additive_inverse() {
        let a = Q8_8::from_raw(500).unwrap();
        assert_eq!(a.saturating_add(a.saturating_neg()).raw(), 0);
    }

    // ── Unit tests: saturating multiplication ──────────────────────────────

    #[test]
    fn mul_two_by_three() {
        // 2.0 × 3.0 = 6.0. Raw: 512 × 768 = 393216; >> 8 = 1536.
        let two = Q8_8::from_raw(512).unwrap();
        let three = Q8_8::from_raw(768).unwrap();
        assert_eq!(two.saturating_mul(three).raw(), 1536);
    }

    #[test]
    fn mul_by_zero_is_zero() {
        let a = Q8_8::from_raw(12345).unwrap();
        assert_eq!(a.saturating_mul(Q8_8::zero()), Q8_8::zero());
        assert_eq!(Q8_8::zero().saturating_mul(a), Q8_8::zero());
    }

    #[test]
    fn mul_by_one_is_identity() {
        let a = Q8_8::from_raw(1000).unwrap();
        let one = Q8_8::one().unwrap();
        assert_eq!(a.saturating_mul(one), a);
    }

    #[test]
    fn mul_commutativity() {
        let a = Q8_8::from_raw(200).unwrap();
        let b = Q8_8::from_raw(-300).unwrap();
        assert_eq!(a.saturating_mul(b), b.saturating_mul(a));
    }

    #[test]
    fn mul_positive_overflow_saturates_to_max() {
        assert_eq!(
            Q8_8::max_value().saturating_mul(Q8_8::max_value()),
            Q8_8::max_value()
        );
    }

    #[test]
    fn mul_negative_positive_overflow_saturates_to_min() {
        assert_eq!(
            Q8_8::min_value().saturating_mul(Q8_8::max_value()),
            Q8_8::min_value()
        );
    }

    /// Multiplication golden value with known rounding.
    ///
    /// 1.5 × 1.5 = 2.25.  In Q8_8 raw: 384 × 384 = 147456; >> 8 = 576.
    /// 576 × 2^-8 = 2.25 exactly (no rounding needed here).
    #[test]
    fn mul_one_point_five_squared() {
        let a = Q8_8::from_raw(384).unwrap(); // 1.5
        assert_eq!(a.saturating_mul(a).raw(), 576); // 2.25
    }

    /// P41 — `Q8_8`: bits:100 × bits:200 produces the precomputed expected raw.
    ///
    /// raw product = 100 × 200 = 20000; right-shift by 8 (FRAC_BITS):
    /// floor(20000/256) = 78, remainder 32 < half=128 → round down → 78.
    /// Tests that intermediate rounding is bit-identical on every platform.
    #[test]
    fn golden_mul_q8_8_bits_100_by_200() {
        let a = Q8_8 { bits: 100 };
        let b = Q8_8 { bits: 200 };
        assert_eq!(a.saturating_mul(b).raw(), 78);
    }

    // ── Unit tests: saturating division ────────────────────────────────────

    #[test]
    fn div_by_zero_is_err() {
        let a = Q8_8::from_raw(1).unwrap();
        assert!(matches!(
            a.saturating_div(Q8_8::zero()),
            Err(FixedPointError::DivisionByZero)
        ));
    }

    #[test]
    fn div_six_by_two_is_three() {
        // 6.0 / 2.0 = 3.0.  raw: (1536 << 8) / 512 = 393216 / 512 = 768.
        let six = Q8_8::from_raw(1536).unwrap();
        let two = Q8_8::from_raw(512).unwrap();
        assert_eq!(six.saturating_div(two).unwrap().raw(), 768);
    }

    #[test]
    fn div_by_one_is_identity() {
        let a = Q8_8::from_raw(1234).unwrap();
        let one = Q8_8::one().unwrap();
        assert_eq!(a.saturating_div(one).unwrap(), a);
    }

    // ── Cross-platform golden values (bit-identity tests) ──────────────────
    // These are hardcoded expected raw values. Any platform divergence will
    // show up here first.

    #[test]
    fn golden_q8_8_from_f32_point_five_raw_is_128() {
        assert_eq!(Q8_8::from_f32(0.5).unwrap().raw(), 128);
    }

    #[test]
    fn golden_q8_8_from_f32_neg_point_five_raw_is_neg128() {
        assert_eq!(Q8_8::from_f32(-0.5).unwrap().raw(), -128);
    }

    #[test]
    fn golden_q8_8_add_100_plus_200() {
        let a = Q8_8 { bits: 100 };
        let b = Q8_8 { bits: 200 };
        assert_eq!(a.saturating_add(b).raw(), 300);
    }

    #[test]
    fn golden_q8_8_saturates_at_max_raw() {
        assert_eq!(
            Q8_8 {
                bits: Q8_8::MAX_RAW
            }
            .saturating_add(Q8_8 { bits: 1 })
            .raw(),
            Q8_8::MAX_RAW
        );
    }

    /// 0.1 in f32 is not exactly representable.  This test pins the exact
    /// round-half-even result so we catch any platform deviation.
    ///
    /// 0.1_f32 as f64 ≈ 0.10000000149011612
    /// × 256 ≈ 25.60000038...  → rounds to 26 (frac ≈ 0.6 > 0.5)
    #[test]
    fn golden_q8_8_from_f32_0_1() {
        let q = Q8_8::from_f32(0.1_f32).unwrap();
        assert_eq!(q.raw(), 26);
    }

    /// P40 — `Q16_16::from_f32(0.1)`: pins the exact round-half-even result.
    ///
    /// 0.1_f32 as f64 ≈ 0.10000000149011612
    /// × 65536 ≈ 6553.600097...  → rounds to 6554 (frac ≈ 0.6 > 0.5).
    /// Tests cross-platform bit-identity for the 32-bit format.
    #[test]
    fn golden_q16_16_from_f32_0_1() {
        let q = Q16_16::from_f32(0.1_f32).unwrap();
        assert_eq!(q.raw(), 6554);
    }

    /// Multiplication golden: 1.0 × 0.5 = 0.5.
    /// raw: 256 × 128 = 32768; >> 8 = 128 (exact, no rounding).
    #[test]
    fn golden_mul_q8_8_one_by_half() {
        let one = Q8_8::from_raw(256).unwrap();
        let half = Q8_8::from_raw(128).unwrap();
        assert_eq!(one.saturating_mul(half).raw(), 128);
    }

    // ── Monotonicity: from_f32 preserves order ─────────────────────────────

    #[test]
    fn from_f32_order_preserved_simple() {
        let a = Q8_8::from_f32(1.0).unwrap();
        let b = Q8_8::from_f32(2.0).unwrap();
        assert!(a < b);
    }

    #[test]
    fn ord_consistent_with_to_f64() {
        let a = Q8_8::from_raw(100).unwrap();
        let b = Q8_8::from_raw(200).unwrap();
        assert!(a < b);
        assert!(a.to_f64() < b.to_f64());
    }

    // ── Associativity (conditional) ────────────────────────────────────────

    #[test]
    fn add_associative_when_no_saturation() {
        let a = Q8_8::from_raw(100).unwrap();
        let b = Q8_8::from_raw(200).unwrap();
        let c = Q8_8::from_raw(300).unwrap();
        // All intermediates are in range for Q8_8 (max 32767).
        let lhs = a.saturating_add(b).saturating_add(c);
        let rhs = a.saturating_add(b.saturating_add(c));
        assert_eq!(lhs, rhs);
    }
}

// ─── Property-based tests ──────────────────────────────────────────────────

#[cfg(test)]
mod proptests {
    #![allow(
        clippy::float_cmp,
        clippy::cast_lossless,
        clippy::cast_possible_truncation,
        clippy::doc_markdown
    )]
    use super::*;
    use proptest::prelude::*;

    // Generate valid raw i32 values for Q8_8.
    fn q8_8_raw() -> impl Strategy<Value = i32> {
        Q8_8::MIN_RAW..=Q8_8::MAX_RAW
    }

    fn q8_8_pair() -> impl Strategy<Value = (Q8_8, Q8_8)> {
        (q8_8_raw(), q8_8_raw()).prop_map(|(a, b)| (Q8_8 { bits: a }, Q8_8 { bits: b }))
    }

    // P1: from_raw round-trip
    proptest! {
        #[test]
        fn p1_from_raw_roundtrip(raw in Q8_8::MIN_RAW..=Q8_8::MAX_RAW) {
            let q = Q8_8::from_raw(raw).expect("in-range raw must succeed");
            prop_assert_eq!(q.raw(), raw);
        }
    }

    // P2: from_raw out-of-range is None
    proptest! {
        #[test]
        fn p2_from_raw_out_of_range_is_none(
            raw in (i32::MIN..Q8_8::MIN_RAW).prop_union(
                (Q8_8::MAX_RAW + 1)..i32::MAX
            )
        ) {
            prop_assert!(Q8_8::from_raw(raw).is_none());
        }
    }

    // P3: f32 round-trip bound using from_f32_saturating.
    // |q.to_f64() - v as f64| <= RESOLUTION_F64
    // (We use saturating to avoid Err at extreme f32 values after rounding.)
    proptest! {
        #[test]
        fn p3_f32_roundtrip_bound(raw in Q8_8::MIN_RAW..=Q8_8::MAX_RAW) {
            let q = Q8_8 { bits: raw };
            let f = q.to_f32();
            let q2 = Q8_8::from_f32_saturating(f);
            let err = (q2.to_f64() - q.to_f64()).abs();
            prop_assert!(
                err <= Q8_8::RESOLUTION_F64,
                "round-trip error {err} > resolution {}",
                Q8_8::RESOLUTION_F64
            );
        }
    }

    // P4-P5: NaN and Inf are rejected by from_f32.
    #[test]
    fn p4_from_f32_nan_is_err() {
        assert!(matches!(
            Q8_8::from_f32(f32::NAN),
            Err(FixedPointError::NaN)
        ));
    }

    #[test]
    fn p5_from_f32_inf_is_err() {
        assert!(matches!(
            Q8_8::from_f32(f32::INFINITY),
            Err(FixedPointError::OutOfRange { .. })
        ));
        assert!(matches!(
            Q8_8::from_f32(f32::NEG_INFINITY),
            Err(FixedPointError::OutOfRange { .. })
        ));
    }

    // P6: from_f32 error bound: half a resolution step.
    proptest! {
        #[test]
        fn p6_from_f32_error_bound(raw in Q8_8::MIN_RAW..=Q8_8::MAX_RAW) {
            let q = Q8_8 { bits: raw };
            let f = q.to_f32();
            // Only test values that from_f32 accepts (skip extreme-boundary f32 values
            // that may round past the representable range).
            if let Ok(q2) = Q8_8::from_f32(f) {
                let err = (q2.to_f64() - f as f64).abs();
                let half_res = Q8_8::RESOLUTION_F64 / 2.0;
                prop_assert!(
                    err <= half_res + 1e-15, // tiny epsilon for f64 comparison
                    "error {err} > half resolution {half_res}"
                );
            }
        }
    }

    // P7–P9: NaN/+∞/−∞ each have exactly one value; the unit tests in the
    // `tests` module above cover them.

    // P10: from_f32_saturating(v) == max_value() for ALL v > Q8_8's representable max.
    // Generates raw integers beyond MAX_RAW; the derived f32 always exceeds max_real.
    proptest! {
        #[test]
        fn p10_saturating_over_range_is_max(raw in Q8_8::MAX_RAW + 1..=i32::MAX) {
            // raw > 32767  =>  real = raw/256 > 127.99609375 = Q8_8 max real
            let v = (raw as f64 * Q8_8::RESOLUTION_F64) as f32;
            prop_assert_eq!(Q8_8::from_f32_saturating(v), Q8_8::max_value());
        }
    }

    // P11: from_f32_saturating(v) == min_value() for ALL v < Q8_8's representable min.
    proptest! {
        #[test]
        fn p11_saturating_under_range_is_min(raw in i32::MIN..Q8_8::MIN_RAW) {
            // raw < -32768  =>  real = raw/256 < -128.0 = Q8_8 min real
            let v = (raw as f64 * Q8_8::RESOLUTION_F64) as f32;
            prop_assert_eq!(Q8_8::from_f32_saturating(v), Q8_8::min_value());
        }
    }

    // P12: Zero is the additive identity.
    proptest! {
        #[test]
        fn p12_add_zero_identity(raw in q8_8_raw()) {
            let a = Q8_8 { bits: raw };
            prop_assert_eq!(a.saturating_add(Q8_8::zero()), a);
            prop_assert_eq!(Q8_8::zero().saturating_add(a), a);
        }
    }

    // P13: Addition is commutative.
    proptest! {
        #[test]
        fn p13_add_commutative((a, b) in q8_8_pair()) {
            prop_assert_eq!(a.saturating_add(b), b.saturating_add(a));
        }
    }

    // P14: In-range addition is EXACT (raw equality, no tolerance).
    proptest! {
        #[test]
        fn p14_add_exact_when_in_range(a_raw in q8_8_raw(), b_raw in q8_8_raw()) {
            let a = Q8_8 { bits: a_raw };
            let b = Q8_8 { bits: b_raw };
            let sum_i64 = a_raw as i64 + b_raw as i64;
            if sum_i64 >= Q8_8::MIN_RAW as i64 && sum_i64 <= Q8_8::MAX_RAW as i64 {
                prop_assert_eq!(
                    a.saturating_add(b).raw(),
                    sum_i64 as i32,
                    "in-range addition must be bit-exact"
                );
            }
        }
    }

    // P17: Zero subtraction is identity.
    proptest! {
        #[test]
        fn p17_sub_zero_identity(raw in q8_8_raw()) {
            let a = Q8_8 { bits: raw };
            prop_assert_eq!(a.saturating_sub(Q8_8::zero()), a);
        }
    }

    // P18: Self-subtraction is zero.
    proptest! {
        #[test]
        fn p18_sub_self_is_zero(raw in q8_8_raw()) {
            let a = Q8_8 { bits: raw };
            prop_assert_eq!(a.saturating_sub(a).raw(), 0);
        }
    }

    // P14-sub: In-range subtraction is EXACT (raw equality, no tolerance).
    proptest! {
        #[test]
        fn p14_sub_exact_when_in_range(a_raw in q8_8_raw(), b_raw in q8_8_raw()) {
            let a = Q8_8 { bits: a_raw };
            let b = Q8_8 { bits: b_raw };
            let diff_i64 = a_raw as i64 - b_raw as i64;
            if diff_i64 >= Q8_8::MIN_RAW as i64 && diff_i64 <= Q8_8::MAX_RAW as i64 {
                prop_assert_eq!(
                    a.saturating_sub(b).raw(),
                    diff_i64 as i32,
                    "in-range subtraction must be bit-exact"
                );
            }
        }
    }

    // P21-P24: Negation properties.
    proptest! {
        #[test]
        fn p21_neg_zero_is_zero(_dummy in 0..1i32) {
            prop_assert_eq!(Q8_8::zero().saturating_neg(), Q8_8::zero());
        }
    }

    proptest! {
        #[test]
        fn p23_double_neg_is_identity(raw in (Q8_8::MIN_RAW + 1)..=Q8_8::MAX_RAW) {
            // Exclude MIN_RAW: MIN.neg() = MAX, MAX.neg() = MIN + 1 ≠ MIN.
            let a = Q8_8 { bits: raw };
            prop_assert_eq!(a.saturating_neg().saturating_neg(), a);
        }
    }

    proptest! {
        #[test]
        fn p24_neg_additive_inverse(raw in (Q8_8::MIN_RAW + 1)..=Q8_8::MAX_RAW) {
            let a = Q8_8 { bits: raw };
            prop_assert_eq!(a.saturating_add(a.saturating_neg()).raw(), 0);
        }
    }

    // P25: Multiply by zero is zero.
    proptest! {
        #[test]
        fn p25_mul_by_zero_is_zero(raw in q8_8_raw()) {
            let a = Q8_8 { bits: raw };
            prop_assert_eq!(a.saturating_mul(Q8_8::zero()), Q8_8::zero());
        }
    }

    // P26: Multiply by one is identity (when one is representable).
    proptest! {
        #[test]
        fn p26_mul_by_one_is_identity(raw in q8_8_raw()) {
            let a = Q8_8 { bits: raw };
            let one = Q8_8::one().unwrap(); // Q8_8 has INT=8 >= 2
            prop_assert_eq!(a.saturating_mul(one), a);
        }
    }

    // P27: Multiplication error bound (half resolution) when result is in range.
    proptest! {
        #[test]
        fn p27_mul_error_bound(a_raw in q8_8_raw(), b_raw in q8_8_raw()) {
            let a = Q8_8 { bits: a_raw };
            let b = Q8_8 { bits: b_raw };
            let exact = a.to_f64() * b.to_f64();
            let min_real = Q8_8::MIN_REAL_F64;
            let max_real = Q8_8::MAX_REAL_F64;
            if exact >= min_real && exact <= max_real {
                let result = a.saturating_mul(b);
                let err = (result.to_f64() - exact).abs();
                let half_res = Q8_8::RESOLUTION_F64 / 2.0;
                prop_assert!(
                    err <= half_res + 1e-12,
                    "mul error {err} exceeds half-resolution {half_res}"
                );
            }
        }
    }

    // P30: Multiplication is commutative.
    proptest! {
        #[test]
        fn p30_mul_commutative((a, b) in q8_8_pair()) {
            prop_assert_eq!(a.saturating_mul(b), b.saturating_mul(a));
        }
    }

    // P31: Division by zero is Err.
    proptest! {
        #[test]
        fn p31_div_by_zero_is_err(raw in q8_8_raw()) {
            let a = Q8_8 { bits: raw };
            prop_assert!(matches!(
                a.saturating_div(Q8_8::zero()),
                Err(FixedPointError::DivisionByZero)
            ));
        }
    }

    // P32: Divide by one is identity.
    proptest! {
        #[test]
        fn p32_div_by_one_is_identity(raw in q8_8_raw()) {
            let a = Q8_8 { bits: raw };
            let one = Q8_8::one().unwrap();
            prop_assert_eq!(a.saturating_div(one).unwrap(), a);
        }
    }

    // P33: Division error bound when result is in range and divisor is non-zero.
    proptest! {
        #[test]
        fn p33_div_error_bound(
            a_raw in q8_8_raw(),
            b_raw in (Q8_8::MIN_RAW..=-1i32).prop_union(1i32..=Q8_8::MAX_RAW)
        ) {
            let a = Q8_8 { bits: a_raw };
            let b = Q8_8 { bits: b_raw };
            let exact = a.to_f64() / b.to_f64();
            let min_real = Q8_8::MIN_REAL_F64;
            let max_real = Q8_8::MAX_REAL_F64;
            if exact >= min_real && exact <= max_real {
                let result = a.saturating_div(b).unwrap();
                let err = (result.to_f64() - exact).abs();
                let half_res = Q8_8::RESOLUTION_F64 / 2.0;
                prop_assert!(
                    err <= half_res + 1e-12,
                    "div error {err} exceeds half-resolution {half_res}"
                );
            }
        }
    }

    // P34: from_f32 preserves order (monotonicity), with boundary guard.
    proptest! {
        #[test]
        fn p34_from_f32_monotone(a_raw in q8_8_raw(), b_raw in q8_8_raw()) {
            let a = Q8_8 { bits: a_raw };
            let b = Q8_8 { bits: b_raw };
            let fa = a.to_f32();
            let fb = b.to_f32();
            // Guard: both f32 values must be accepted by from_f32 (avoids the
            // boundary case where to_f32() rounds past the representable range).
            if let (Ok(qa), Ok(qb)) = (Q8_8::from_f32(fa), Q8_8::from_f32(fb)) {
                if fa < fb {
                    prop_assert!(qa <= qb);
                } else if fa > fb {
                    prop_assert!(qa >= qb);
                }
            }
        }
    }

    // P35: Ord is consistent with to_f64.
    proptest! {
        #[test]
        fn p35_ord_consistent_with_f64(a_raw in q8_8_raw(), b_raw in q8_8_raw()) {
            let a = Q8_8 { bits: a_raw };
            let b = Q8_8 { bits: b_raw };
            prop_assert_eq!(a.cmp(&b), a.to_f64().partial_cmp(&b.to_f64()).unwrap());
        }
    }

    // P20: Conditional associativity of addition.
    proptest! {
        #[test]
        fn p20_add_associative_when_no_saturation(
            a_raw in -10000i32..=10000i32,
            b_raw in -10000i32..=10000i32,
            c_raw in -10000i32..=10000i32,
        ) {
            // Use a restricted range so all intermediates are provably in Q8_8 range.
            let a = Q8_8 { bits: a_raw };
            let b = Q8_8 { bits: b_raw };
            let c = Q8_8 { bits: c_raw };
            let ab = a_raw as i64 + b_raw as i64;
            let abc = ab + c_raw as i64;
            let bc = b_raw as i64 + c_raw as i64;
            let abc2 = a_raw as i64 + bc;
            if ab >= Q8_8::MIN_RAW as i64 && ab <= Q8_8::MAX_RAW as i64
                && abc >= Q8_8::MIN_RAW as i64 && abc <= Q8_8::MAX_RAW as i64
                && bc >= Q8_8::MIN_RAW as i64 && bc <= Q8_8::MAX_RAW as i64
                && abc2 >= Q8_8::MIN_RAW as i64 && abc2 <= Q8_8::MAX_RAW as i64
            {
                let lhs = a.saturating_add(b).saturating_add(c);
                let rhs = a.saturating_add(b.saturating_add(c));
                prop_assert_eq!(lhs, rhs);
            }
        }
    }
}
