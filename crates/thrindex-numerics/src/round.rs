//! Round-half-even (banker's rounding) for integer arithmetic.
//!
//! # Right-shift rounding semantics
//!
//! Rust's `>>` on signed integers is *arithmetic right shift*, which is
//! **floor division**: `value >> shift == floor(value / 2^shift)`.  The
//! remainder `frac = value & ((1 << shift) - 1)` is therefore always
//! **non-negative** (it is the distance above the floor), regardless of
//! the sign of `value`.
//!
//! Given:
//!   - `trunc = value >> shift`   (floor quotient)
//!   - `frac  = value & mask`     (non-negative remainder, 0 ≤ frac < 2^shift)
//!   - `half  = 1 << (shift - 1)` (the tie point)
//!
//! The round-half-even rule is:
//!   - `frac > half`  →  `trunc + 1`
//!   - `frac < half`  →  `trunc`
//!   - `frac == half` →  `trunc` if `trunc` is even, else `trunc + 1`
//!
//! Note the `+1` is **always** `+1` (we are rounding *up* from the floor, toward
//! the true mathematical value), not `+sign(value)`.  The sign is already
//! accounted for by `trunc` being the floor.
//!
//! # Golden-value verification
//!
//! | value | shift | floor trunc | frac | half | result |
//! |------:|------:|------------:|-----:|-----:|-------:|
//! |    -1 |     1 |          -1 |    1 |    1 |      0 |  ← tie → round to even (+1)
//! |    -3 |     1 |          -2 |    1 |    1 |     -2 |  ← tie → trunc -2 is even → keep
//! |    -5 |     2 |          -2 |    3 |    2 |     -1 |  ← frac > half → +1
//! |     3 |     1 |           1 |    1 |    1 |      2 |  ← tie → trunc 1 is odd → +1
//! |     5 |     2 |           1 |    1 |    2 |      1 |  ← frac < half → keep
//! |     1 |     1 |           0 |    1 |    1 |      0 |  ← tie → trunc 0 is even → keep

/// Round `value >> shift` using round-half-even (banker's rounding).
///
/// `shift` must satisfy `0 < shift < 63` so that `1i64 << shift` does not
/// overflow.  When `shift == 0` no rounding is needed; the function returns
/// `value` unchanged.
///
/// This function uses only integer arithmetic and produces bit-identical
/// results on every platform.
#[inline]
pub(crate) fn round_half_even_shr_i64(value: i64, shift: u32) -> i64 {
    if shift == 0 {
        return value;
    }
    debug_assert!(shift < 63, "shift must be < 63");

    let mask: i64 = (1i64 << shift) - 1;
    let half: i64 = 1i64 << (shift - 1);
    let trunc: i64 = value >> shift; // floor division
    let frac: i64 = value & mask; // always ≥ 0

    match frac.cmp(&half) {
        std::cmp::Ordering::Greater => trunc + 1,
        std::cmp::Ordering::Less => trunc,
        std::cmp::Ordering::Equal => {
            // Tie: round to even.
            if trunc & 1 == 0 {
                trunc
            } else {
                trunc + 1
            }
        }
    }
}

/// Round-half-even for i128, used by `saturating_div` which needs 128-bit
/// to avoid overflow when pre-shifting by `FRAC_BITS`.
///
/// # Algorithm
///
/// Uses **truncation toward zero** (Rust's default `i128::div` / `%`) plus a
/// sign-based direction rule, which works correctly for all sign combinations
/// of numerator and denominator.
///
/// Given:
///   - `q = numerator / denominator`  (truncation toward zero)
///   - `r = numerator % denominator`  (same sign as numerator in Rust)
///   - `abs_r = |r|`, `abs_d = |d|`
///
/// The rule is:
///   - `2·abs_r < abs_d`  → `q`          (closer to truncated side)
///   - `2·abs_r > abs_d`  → far integer   (closer to the opposite side)
///   - `2·abs_r == abs_d` → `q` if even, else far integer  (tie → even)
///
/// The "far" integer is `q + 1` when `r` and `d` have the **same** sign
/// (the fractional part pushes the value upward from `q`), or `q − 1` when
/// they have **opposite** signs (the fractional part pushes downward from `q`).
///
/// # Verification against golden tests
///
/// | num      |  den | `q_trunc` | r   | 2\|r\| vs \|d\| | result |
/// |---------:|-----:|----------:|----:|:-----------:|-------:|
/// |        1 |    2 |         0 |   1 | 2 == 2 → tie|      0 | ← q even
/// |       -1 |    2 |         0 |  -1 | 2 == 2 → tie|      0 | ← q even
/// |       -3 |    2 |        -1 |  -1 | 2 == 2 → tie|     -2 | ← q odd, far=q−1
/// |       -5 |    4 |        -1 |  -1 | 2 < 4       |     -1 | ← keep q
/// |        3 |    2 |         1 |   1 | 2 == 2 → tie|      2 | ← q odd, far=q+1
/// |        5 |    4 |         1 |   1 | 2 < 4       |      1 | ← keep q
/// | 1698304  | -258 |     -6582 | 148 | 296 > 258   |  -6583 | ← p33 regression case
#[inline]
pub(crate) fn round_half_even_div_i128(numerator: i128, denominator: i128) -> i128 {
    debug_assert!(denominator != 0, "denominator must be non-zero");

    let q = numerator / denominator; // truncation toward zero
    let r = numerator % denominator; // same sign as numerator (Rust convention)
    let abs_r = r.unsigned_abs();
    let abs_d = denominator.unsigned_abs();
    let two_abs_r = abs_r.saturating_mul(2);

    // "Far" = the integer on the opposite side of q from zero.
    // r and d having the same sign means the value is above q → far = q+1.
    // r and d having opposite signs means the value is below q → far = q−1.
    let far = if (r > 0) == (denominator > 0) {
        q + 1
    } else {
        q - 1
    };

    match two_abs_r.cmp(&abs_d) {
        std::cmp::Ordering::Less => q,
        std::cmp::Ordering::Greater => far,
        std::cmp::Ordering::Equal => {
            if q & 1 == 0 {
                q
            } else {
                far
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Golden-value table for round_half_even_shr_i64 ─────────────────────
    // Every negative-tie case is mandatory per the design review.

    #[test]
    fn golden_shr_neg1_shift1() {
        // value=-1, shift=1: trunc=floor(-1/2)=-1, frac=1, half=1 → tie, trunc odd → +1 → 0
        assert_eq!(round_half_even_shr_i64(-1, 1), 0);
    }

    #[test]
    fn golden_shr_neg3_shift1() {
        // value=-3, shift=1: trunc=floor(-3/2)=-2, frac=1, half=1 → tie, trunc even → keep → -2
        assert_eq!(round_half_even_shr_i64(-3, 1), -2);
    }

    #[test]
    fn golden_shr_neg5_shift2() {
        // value=-5, shift=2: trunc=floor(-5/4)=-2, frac=3, half=2 → frac>half → +1 → -1
        assert_eq!(round_half_even_shr_i64(-5, 2), -1);
    }

    #[test]
    fn golden_shr_pos3_shift1() {
        // value=3, shift=1: trunc=1, frac=1, half=1 → tie, trunc odd → +1 → 2
        assert_eq!(round_half_even_shr_i64(3, 1), 2);
    }

    #[test]
    fn golden_shr_pos5_shift2() {
        // value=5, shift=2: trunc=1, frac=1, half=2 → frac<half → keep → 1
        assert_eq!(round_half_even_shr_i64(5, 2), 1);
    }

    #[test]
    fn golden_shr_pos1_shift1() {
        // value=1, shift=1: trunc=0, frac=1, half=1 → tie, trunc even → keep → 0
        assert_eq!(round_half_even_shr_i64(1, 1), 0);
    }

    #[test]
    fn shr_shift_zero_is_identity() {
        assert_eq!(round_half_even_shr_i64(42, 0), 42);
        assert_eq!(round_half_even_shr_i64(-42, 0), -42);
    }

    #[test]
    fn shr_rounds_down_below_half() {
        // value=1, shift=2: trunc=0, frac=1, half=2 → frac<half → 0
        assert_eq!(round_half_even_shr_i64(1, 2), 0);
        // value=-1, shift=2: trunc=-1, frac=3, half=2 → frac>half → trunc+1 = 0
        assert_eq!(round_half_even_shr_i64(-1, 2), 0);
    }

    // ── Golden-value table for round_half_even_div_i128 ────────────────────
    // See the inline table in the function's doc comment for derivations.

    #[test]
    fn golden_div_pos_tie() {
        // 1/2: q_trunc=0, r=1, 2|r|=2==|d|=2 → tie, q=0 even → 0
        assert_eq!(round_half_even_div_i128(1, 2), 0);
    }

    #[test]
    fn golden_div_neg_tie() {
        // -1/2: q_trunc=0, r=-1, 2|r|=2==|d|=2 → tie, q=0 even → 0
        assert_eq!(round_half_even_div_i128(-1, 2), 0);
    }

    #[test]
    fn golden_div_neg3_2() {
        // -3/2: q_trunc=-1, r=-1, 2|r|=2==|d|=2 → tie, q=-1 odd, r<0,d>0 → far=q-1=-2
        assert_eq!(round_half_even_div_i128(-3, 2), -2);
    }

    #[test]
    fn golden_div_neg5_4() {
        // -5/4: q_trunc=-1, r=-1, 2|r|=2 < |d|=4 → keep q=-1
        assert_eq!(round_half_even_div_i128(-5, 4), -1);
    }

    #[test]
    fn golden_div_pos3_2() {
        // 3/2: q_trunc=1, r=1, 2|r|=2==|d|=2 → tie, q=1 odd, r>0,d>0 → far=q+1=2
        assert_eq!(round_half_even_div_i128(3, 2), 2);
    }

    #[test]
    fn golden_div_pos5_4() {
        // 5/4: q_trunc=1, r=1, 2|r|=2 < |d|=4 → keep q=1
        assert_eq!(round_half_even_div_i128(5, 4), 1);
    }

    #[test]
    fn golden_div_neg_pos_cross_sign() {
        // 1698304 / -258 (the p33 failing case):
        // q_trunc=-6582, r=148, 2|r|=296 > |d|=258
        // r>0, d<0 → opposite signs → far = q-1 = -6583
        assert_eq!(round_half_even_div_i128(1_698_304, -258), -6583);
    }

    #[test]
    fn golden_div_neg_closer_to_zero() {
        // -3/4: q_trunc=0, r=-3, 2|r|=6 > |d|=4 → r<0, d>0 → far = q-1 = -1
        assert_eq!(round_half_even_div_i128(-3, 4), -1);
    }
}
