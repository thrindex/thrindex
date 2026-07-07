//! Saturating arithmetic for `Q<INT_BITS, FRAC_BITS>`.
//!
//! All operations are:
//! - **Saturating**: results outside `[min_value(), max_value()]` are clamped.
//! - **Deterministic**: only integer arithmetic; no floating-point in the hot path.
//! - **Rounding**: where the result must be narrowed (multiplication, division),
//!   round-half-even is applied before clamping.

use crate::error::FixedPointError;
use crate::fixed::Q;
use crate::round::{round_half_even_div_i128, round_half_even_shr_i64};

impl<const INT: u32, const FRAC: u32> Q<INT, FRAC> {
    // ── Internal helpers ──────────────────────────────────────────────────

    #[inline]
    fn clamp_i64(v: i64) -> Self {
        // The clamped i64 value lies in [MIN_RAW, MAX_RAW] ⊆ [i32::MIN, i32::MAX].
        #[allow(clippy::cast_possible_truncation)]
        Self {
            bits: v.clamp(i64::from(Self::MIN_RAW), i64::from(Self::MAX_RAW)) as i32,
        }
    }

    #[inline]
    fn clamp_i128(v: i128) -> Self {
        #[allow(clippy::cast_possible_truncation)]
        Self {
            bits: v.clamp(i128::from(Self::MIN_RAW), i128::from(Self::MAX_RAW)) as i32,
        }
    }

    // ── Saturating addition ────────────────────────────────────────────────

    /// Add `rhs`, saturating at the representable bounds.
    ///
    /// Addition of two in-range values is **exact** (no rounding): the raw
    /// result is `self.raw() + rhs.raw()`, clamped to `[MIN_RAW, MAX_RAW]`.
    ///
    /// This operation is commutative but **not** unconditionally associative:
    /// saturation at a bound prevents the usual algebraic identity.
    ///
    /// # Example
    ///
    /// ```
    /// use thrindex_numerics::Q8_8;
    /// let a = Q8_8::from_raw(100).unwrap();
    /// let b = Q8_8::from_raw(200).unwrap();
    /// assert_eq!(a.saturating_add(b).raw(), 300);
    /// assert_eq!(Q8_8::max_value().saturating_add(Q8_8::max_value()), Q8_8::max_value());
    /// ```
    #[inline]
    pub fn saturating_add(self, rhs: Self) -> Self {
        Self::clamp_i64(i64::from(self.bits) + i64::from(rhs.bits))
    }

    // ── Saturating subtraction ─────────────────────────────────────────────

    /// Subtract `rhs`, saturating at the representable bounds.
    ///
    /// Subtraction of two in-range values is **exact** (no rounding): the raw
    /// result is `self.raw() - rhs.raw()`, clamped to `[MIN_RAW, MAX_RAW]`.
    ///
    /// # Example
    ///
    /// ```
    /// use thrindex_numerics::Q8_8;
    /// let a = Q8_8::from_raw(300).unwrap();
    /// let b = Q8_8::from_raw(100).unwrap();
    /// assert_eq!(a.saturating_sub(b).raw(), 200);
    /// assert_eq!(Q8_8::min_value().saturating_sub(Q8_8::max_value()), Q8_8::min_value());
    /// ```
    #[inline]
    pub fn saturating_sub(self, rhs: Self) -> Self {
        Self::clamp_i64(i64::from(self.bits) - i64::from(rhs.bits))
    }

    // ── Saturating negation ────────────────────────────────────────────────

    /// Negate, saturating at the representable bounds.
    ///
    /// For two's-complement signed integers, `min_value()` has no positive
    /// counterpart; `min_value().saturating_neg()` returns `max_value()`.
    ///
    /// # Example
    ///
    /// ```
    /// use thrindex_numerics::Q8_8;
    /// let a = Q8_8::from_raw(100).unwrap();
    /// assert_eq!(a.saturating_neg().raw(), -100);
    /// assert_eq!(Q8_8::min_value().saturating_neg(), Q8_8::max_value());
    /// ```
    #[inline]
    pub fn saturating_neg(self) -> Self {
        Self::clamp_i64(-i64::from(self.bits))
    }

    // ── Saturating multiplication ──────────────────────────────────────────

    /// Multiply by `rhs`, saturating at the representable bounds.
    ///
    /// # Algorithm
    ///
    /// 1. Widen both operands to `i64` and compute the product
    ///    `p = self.bits × rhs.bits` (no overflow: each operand ≤ 2^31, so the
    ///    product ≤ 2^62).
    /// 2. The product represents the value in `Q<2·INT, 2·FRAC>` format; it
    ///    must be right-shifted by `FRAC_BITS` to return to `Q<INT, FRAC>`.
    /// 3. Apply round-half-even on the `FRAC_BITS` bits being discarded.
    /// 4. Clamp the result to `[MIN_RAW, MAX_RAW]`.
    ///
    /// # Error bound
    ///
    /// When the mathematical product is in range, the result satisfies:
    /// `|result.to_f64() − a.to_f64() × b.to_f64()| ≤ 2^(−FRAC_BITS − 1)`
    ///
    /// # Example
    ///
    /// ```
    /// use thrindex_numerics::Q8_8;
    /// // 2.0 × 3.0 = 6.0.  In raw: 512 × 768 >> 8 = 1536.
    /// let two   = Q8_8::from_raw(512).unwrap();
    /// let three = Q8_8::from_raw(768).unwrap();
    /// assert_eq!(two.saturating_mul(three).raw(), 1536);
    /// assert_eq!(Q8_8::max_value().saturating_mul(Q8_8::max_value()), Q8_8::max_value());
    /// ```
    #[inline]
    pub fn saturating_mul(self, rhs: Self) -> Self {
        let product = i64::from(self.bits) * i64::from(rhs.bits);
        let rounded = round_half_even_shr_i64(product, FRAC);
        Self::clamp_i64(rounded)
    }

    // ── Saturating division ────────────────────────────────────────────────

    /// Divide by `rhs`, saturating at the representable bounds, with
    /// round-half-even.
    ///
    /// # Algorithm
    ///
    /// 1. Left-shift the numerator by `FRAC_BITS` in `i128` (prevents
    ///    overflow: max shift is 31 bits on a 32-bit-wide numerator, total
    ///    ≤ 63 bits, well within i128).
    /// 2. Apply `round_half_even_div_i128`: compute the truncation-toward-zero
    ///    quotient and same-sign remainder, then use the sign-based direction
    ///    rule to break ties toward even.
    /// 3. Clamp the rounded quotient to `[MIN_RAW, MAX_RAW]`.
    ///
    /// # Error bound
    ///
    /// When the mathematical quotient is in range:
    /// `|result.to_f64() − a.to_f64() / b.to_f64()| ≤ 2^(−FRAC_BITS − 1)`
    ///
    /// # Errors
    ///
    /// Returns [`FixedPointError::DivisionByZero`] when `rhs` is zero.
    ///
    /// # Example
    ///
    /// ```
    /// use thrindex_numerics::Q8_8;
    /// // 6.0 / 2.0 = 3.0.  raw: (1536 << 8) / 512 = 768.
    /// let six = Q8_8::from_raw(1536).unwrap();
    /// let two = Q8_8::from_raw(512).unwrap();
    /// assert_eq!(six.saturating_div(two).unwrap().raw(), 768);
    /// assert!(Q8_8::from_raw(1).unwrap().saturating_div(Q8_8::zero()).is_err());
    /// ```
    #[inline]
    pub fn saturating_div(self, rhs: Self) -> Result<Self, FixedPointError> {
        if rhs.bits == 0 {
            return Err(FixedPointError::DivisionByZero);
        }
        let numerator = i128::from(self.bits) << FRAC;
        let denominator = i128::from(rhs.bits);
        let rounded = round_half_even_div_i128(numerator, denominator);
        Ok(Self::clamp_i128(rounded))
    }
}
