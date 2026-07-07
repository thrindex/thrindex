//! The `Q<INT_BITS, FRAC_BITS>` signed fixed-point type.
//!
//! # Format
//!
//! `Q<INT_BITS, FRAC_BITS>` is a signed fixed-point number with:
//!
//! - `INT_BITS` bits for the integer part (the most significant bit is the
//!   sign bit, so `INT_BITS ≥ 1`).
//! - `FRAC_BITS` bits for the fractional part.
//! - A backing store of `i32`; only `INT_BITS + FRAC_BITS` of those 32 bits
//!   carry meaning.  The raw integer value `bits` represents the real value
//!   `bits × 2^(−FRAC_BITS)`.
//!
//! ## Compile-time invariants
//!
//! These are enforced at instantiation time via `const { assert! }` inside
//! every method:
//!
//! - `INT_BITS ≥ 1` (at least a sign bit)
//! - `INT_BITS + FRAC_BITS ≤ 32` (fits in the i32 backing)
//!
//! Attempting to use a type that violates either constraint is a **compile
//! error**.
//!
//! ## Representable range
//!
//! ```text
//! min  = −2^(INT_BITS−1)
//! max  = 2^(INT_BITS−1) − 2^(−FRAC_BITS)
//! res  = 2^(−FRAC_BITS)     (smallest step = 1 raw unit)
//! ```
//!
//! For `Q<16, 16>`: min = −32768, max ≈ 32767.9999847, resolution ≈ 1.5×10⁻⁵.

use crate::error::FixedPointError;

/// Signed fixed-point number in `Q_{INT_BITS}.{FRAC_BITS}` format, backed by
/// `i32`.
///
/// See module-level documentation for the full format specification,
/// representable range, and invariants.
///
/// All arithmetic operations are **saturating** — out-of-range results are
/// clamped to `[min_value(), max_value()]` rather than wrapping or panicking.
/// All rounding uses **round-half-even** (banker's rounding) throughout.
/// All operations use integer arithmetic exclusively; no floating-point is
/// used in the hot path.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[must_use]
pub struct Q<const INT_BITS: u32, const FRAC_BITS: u32> {
    pub(crate) bits: i32,
}

// ─── Compile-time constants ────────────────────────────────────────────────

impl<const INT: u32, const FRAC: u32> Q<INT, FRAC> {
    /// Total bits used: `INT_BITS + FRAC_BITS`.
    pub const TOTAL_BITS: u32 = {
        assert!(INT >= 1, "Q: INT_BITS must be >= 1 (sign bit is mandatory)");
        assert!(
            INT + FRAC <= 32,
            "Q: INT_BITS + FRAC_BITS must be <= 32 (backing is i32)"
        );
        INT + FRAC
    };

    /// The minimum raw backing value for this format.
    ///
    /// Equal to `−2^(TOTAL_BITS−1)`.
    pub const MIN_RAW: i32 = {
        // i64::from(u32) is not const-stable (rust#143874); suppress cast_lossless.
        // u32 always fits in i64, so the cast is exact.
        #[allow(clippy::cast_lossless)]
        let total = (INT + FRAC) as i64;
        let min_i64 = -(1i64 << (total - 1));
        // INT+FRAC ≤ 32 (enforced by TOTAL_BITS assertion), so
        // min_i64 ≥ -(1<<31) = i32::MIN, guaranteed to fit.
        #[allow(clippy::cast_possible_truncation)]
        {
            min_i64 as i32
        }
    };

    /// The maximum raw backing value for this format.
    ///
    /// Equal to `2^(TOTAL_BITS−1) − 1`.
    pub const MAX_RAW: i32 = {
        #[allow(clippy::cast_lossless)]
        let total = (INT + FRAC) as i64;
        let max_i64 = (1i64 << (total - 1)) - 1;
        // INT+FRAC ≤ 32, so max_i64 ≤ (1<<31)-1 = i32::MAX, fits.
        #[allow(clippy::cast_possible_truncation)]
        {
            max_i64 as i32
        }
    };

    /// Resolution (smallest representable step), as `f64`.
    ///
    /// Equal to `2^(−FRAC_BITS)`.  The conversion is exact: `FRAC ≤ 31`
    /// (because `INT ≥ 1` and `INT + FRAC ≤ 32`), so `2^FRAC ≤ 2^31`, which
    /// lies within `f64`'s 52-bit mantissa.
    pub const RESOLUTION_F64: f64 = {
        // 2^FRAC ≤ 2^31 (power of two, exactly representable in f64).
        #[allow(clippy::cast_precision_loss)]
        {
            1.0_f64 / (1u64 << FRAC) as f64
        }
    };

    /// Minimum representable real value, as `f64`.
    pub const MIN_REAL_F64: f64 = {
        #[allow(clippy::cast_lossless)]
        let total = (INT + FRAC) as i64;
        let raw_min = -(1i64 << (total - 1));
        // raw_min ∈ [-2^31, 0] and 2^FRAC ≤ 2^31 — both exactly representable.
        #[allow(clippy::cast_precision_loss)]
        {
            raw_min as f64 / (1u64 << FRAC) as f64
        }
    };

    /// Maximum representable real value, as `f64`.
    pub const MAX_REAL_F64: f64 = {
        #[allow(clippy::cast_lossless)]
        let total = (INT + FRAC) as i64;
        let raw_max = (1i64 << (total - 1)) - 1;
        // raw_max ∈ [0, 2^31-1] and 2^FRAC ≤ 2^31 — both exactly representable.
        #[allow(clippy::cast_precision_loss)]
        {
            raw_max as f64 / (1u64 << FRAC) as f64
        }
    };
}

// ─── Constructors ──────────────────────────────────────────────────────────

impl<const INT: u32, const FRAC: u32> Q<INT, FRAC> {
    /// Construct from a raw backing integer if it falls within the valid range,
    /// otherwise return `None`.
    ///
    /// # Example
    ///
    /// ```
    /// use thrindex_numerics::Q8_8;
    /// assert!(Q8_8::from_raw(128).is_some());
    /// assert!(Q8_8::from_raw(32768).is_none()); // out of range for Q8_8
    /// ```
    #[must_use]
    #[inline]
    pub fn from_raw(bits: i32) -> Option<Self> {
        let _ = Self::TOTAL_BITS; // trigger compile-time assertions
        if bits >= Self::MIN_RAW && bits <= Self::MAX_RAW {
            Some(Self { bits })
        } else {
            None
        }
    }

    /// Zero.
    ///
    /// # Example
    ///
    /// ```
    /// use thrindex_numerics::Q8_8;
    /// assert_eq!(Q8_8::zero().raw(), 0);
    /// ```
    #[inline]
    pub const fn zero() -> Self {
        Self { bits: 0 }
    }

    /// The minimum representable value for this format.
    ///
    /// # Example
    ///
    /// ```
    /// use thrindex_numerics::Q8_8;
    /// // Q8_8: range [-128, 127.996], min raw = -32768
    /// assert_eq!(Q8_8::min_value().raw(), -32768);
    /// ```
    #[inline]
    pub const fn min_value() -> Self {
        Self {
            bits: Self::MIN_RAW,
        }
    }

    /// The maximum representable value for this format.
    ///
    /// # Example
    ///
    /// ```
    /// use thrindex_numerics::Q8_8;
    /// // Q8_8: max raw = 32767
    /// assert_eq!(Q8_8::max_value().raw(), 32767);
    /// ```
    #[inline]
    pub const fn max_value() -> Self {
        Self {
            bits: Self::MAX_RAW,
        }
    }

    /// Returns `Some(1.0)` when `INT_BITS ≥ 2` (1.0 is representable), or
    /// `None` for `Q<1, n>` formats where the maximum value is strictly less
    /// than 1.
    ///
    /// # Example
    ///
    /// ```
    /// use thrindex_numerics::{Q8_8, Q1_7};
    /// assert!(Q8_8::one().is_some());
    /// assert!(Q1_7::one().is_none()); // Q<1,7> max = 0.992
    /// ```
    #[must_use]
    #[inline]
    pub fn one() -> Option<Self> {
        let _ = Self::TOTAL_BITS;
        if INT < 2 {
            return None;
        }
        // 1.0 in raw units = 2^FRAC_BITS
        let raw = 1i32 << FRAC;
        Some(Self { bits: raw })
    }
}

// ─── Accessors ─────────────────────────────────────────────────────────────

impl<const INT: u32, const FRAC: u32> Q<INT, FRAC> {
    /// The raw backing integer.
    ///
    /// The real value equals `self.raw() as f64 * 2^(−FRAC_BITS)`.
    #[must_use]
    #[inline]
    pub const fn raw(self) -> i32 {
        self.bits
    }

    /// Convert to `f64`.
    ///
    /// This conversion is **exact** for all `Q<INT, FRAC>` where
    /// `INT + FRAC ≤ 53` (true for all values since `TOTAL_BITS ≤ 32`): the
    /// backing `i32` is exactly representable as `f64`, and multiplying by
    /// `2^(-FRAC_BITS)` is a pure exponent shift in IEEE-754.
    ///
    /// # Example
    ///
    /// ```
    /// use thrindex_numerics::Q8_8;
    /// let q = Q8_8::from_raw(256).unwrap(); // represents 1.0
    /// assert_eq!(q.to_f64(), 1.0);
    /// ```
    #[must_use]
    #[inline]
    pub fn to_f64(self) -> f64 {
        // f64::from(i32) is lossless (all i32 values are exact in f64's 52-bit mantissa).
        // 2f64.powi(-n) is an exact IEEE-754 exponent shift for n ≤ 31.
        // FRAC ≤ 31 per the TOTAL_BITS invariant; cast to i32 cannot wrap.
        #[allow(clippy::cast_possible_wrap)]
        {
            f64::from(self.bits) * 2f64.powi(-(FRAC as i32))
        }
    }

    /// Convert to `f32`.
    ///
    /// When `TOTAL_BITS ≤ 24`, the conversion is exact (the value fits in the
    /// 24-bit f32 mantissa).  For `TOTAL_BITS > 24`, the f32 result is the
    /// nearest representable f32 value, which may differ from the true value
    /// by up to `2^(INT_BITS − 24) × resolution`.
    ///
    /// Prefer [`to_f64`](Self::to_f64) when precision matters.
    #[must_use]
    #[inline]
    pub fn to_f32(self) -> f32 {
        // Intentional narrowing: documented in the method contract above.
        #[allow(clippy::cast_possible_truncation)]
        {
            self.to_f64() as f32
        }
    }
}

// ─── f32 conversion with explicit error bounds ─────────────────────────────

impl<const INT: u32, const FRAC: u32> Q<INT, FRAC> {
    /// Convert an `f32` to this Q format, returning `Err` on NaN, infinity,
    /// or out-of-range values.
    ///
    /// # Conversion algorithm (deterministic, bit-identical on all platforms)
    ///
    /// 1. Reject NaN and ±∞ immediately.
    /// 2. Promote `v` to `f64` (exact: every f32 value is exactly representable
    ///    as f64).
    /// 3. Multiply by `2^FRAC_BITS` (exact: `powi` is a pure IEEE-754 exponent
    ///    shift for the powers-of-two used here; no mantissa bits are lost).
    /// 4. Round to the nearest integer using `f64::round_ties_even()` (banker's
    ///    rounding; stable in Rust 1.77+).
    /// 5. Range-check: if the rounded integer lies outside `[MIN_RAW, MAX_RAW]`,
    ///    return `Err(OutOfRange)`.
    /// 6. Cast to `i32` and return `Ok(Q { bits })`.
    ///
    /// # Error bound
    ///
    /// For any finite, in-range `f32` value `v`, the returned value `R`
    /// satisfies:
    ///
    /// ```text
    /// |R.to_f64() − v as f64| ≤ 2^(−FRAC_BITS − 1)    (half a resolution step)
    /// ```
    ///
    /// # Errors
    ///
    /// - [`FixedPointError::NaN`] — `v` is NaN.
    /// - [`FixedPointError::OutOfRange`] — `v` is ±∞ or, after rounding, lies
    ///   outside `[min_value(), max_value()]`.
    ///
    /// # Example
    ///
    /// ```
    /// use thrindex_numerics::Q8_8;
    /// let q = Q8_8::from_f32(1.0).unwrap();
    /// assert_eq!(q.raw(), 256); // 1.0 × 2^8 = 256
    /// ```
    pub fn from_f32(v: f32) -> Result<Self, FixedPointError> {
        let _ = Self::TOTAL_BITS;
        if v.is_nan() {
            return Err(FixedPointError::NaN);
        }
        if v.is_infinite() {
            return Err(FixedPointError::OutOfRange {
                value: f64::from(v),
                min: Self::MIN_REAL_F64,
                max: Self::MAX_REAL_F64,
            });
        }
        // Promote to f64 (exact) then scale by 2^FRAC_BITS (exact exponent shift).
        // FRAC ≤ 31 per TOTAL_BITS invariant; cast to i32 cannot wrap.
        #[allow(clippy::cast_possible_wrap)]
        let scaled = f64::from(v) * 2f64.powi(FRAC as i32);
        let rounded = scaled.round_ties_even();
        // Range check in f64 before casting to i32.
        let raw_min = f64::from(Self::MIN_RAW);
        let raw_max = f64::from(Self::MAX_RAW);
        if rounded < raw_min || rounded > raw_max {
            return Err(FixedPointError::OutOfRange {
                value: f64::from(v),
                min: Self::MIN_REAL_F64,
                max: Self::MAX_REAL_F64,
            });
        }
        // rounded ∈ [i32::MIN, i32::MAX] by the range check above; cast is safe.
        #[allow(clippy::cast_possible_truncation)]
        Ok(Self {
            bits: rounded as i32,
        })
    }

    /// Convert an `f32` to this Q format, **saturating** on out-of-range
    /// values.
    ///
    /// - NaN → [`zero()`](Self::zero) (matches Rust's own `NaN as i32 == 0`
    ///   convention for saturating float-to-integer casts; see ADR-0003).
    /// - `+∞` or values above [`max_value()`](Self::max_value) → `max_value()`.
    /// - `−∞` or values below [`min_value()`](Self::min_value) → `min_value()`.
    /// - All other values are rounded with round-half-even and clamped.
    ///
    /// # Example
    ///
    /// ```
    /// use thrindex_numerics::Q8_8;
    /// assert_eq!(Q8_8::from_f32_saturating(f32::NAN), Q8_8::zero());
    /// assert_eq!(Q8_8::from_f32_saturating(f32::INFINITY), Q8_8::max_value());
    /// assert_eq!(Q8_8::from_f32_saturating(f32::NEG_INFINITY), Q8_8::min_value());
    /// ```
    pub fn from_f32_saturating(v: f32) -> Self {
        let _ = Self::TOTAL_BITS;
        if v.is_nan() {
            return Self::zero();
        }
        // Promote to f64 (exact) and scale. For ±∞ this yields ±∞.
        // FRAC ≤ 31 per TOTAL_BITS invariant; cast to i32 cannot wrap.
        #[allow(clippy::cast_possible_wrap)]
        let scaled = f64::from(v) * 2f64.powi(FRAC as i32);
        let rounded = scaled.round_ties_even();
        // Rust's float-to-int cast saturates (not UB) since Rust 1.45.
        // For ±∞ or very large values, `rounded as i64` gives i64::MAX/MIN.
        #[allow(clippy::cast_possible_truncation)]
        let raw_i64 = rounded as i64;
        let raw = raw_i64.clamp(i64::from(Self::MIN_RAW), i64::from(Self::MAX_RAW));
        // raw ∈ [MIN_RAW, MAX_RAW] ⊆ [i32::MIN, i32::MAX]; cast is safe.
        #[allow(clippy::cast_possible_truncation)]
        Self { bits: raw as i32 }
    }
}

// ─── Debug / Display ───────────────────────────────────────────────────────

impl<const INT: u32, const FRAC: u32> core::fmt::Debug for Q<INT, FRAC> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Q<{INT},{FRAC}>({:.prec$} [raw={}])",
            self.to_f64(),
            self.bits,
            prec = (FRAC as usize / 3).max(1),
        )
    }
}

impl<const INT: u32, const FRAC: u32> core::fmt::Display for Q<INT, FRAC> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.to_f64())
    }
}
