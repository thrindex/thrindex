//! Typed error for fixed-point conversions.
//!
//! Hand-written without `thiserror` so `thrindex-numerics` stays a zero-dependency L0 crate.

use core::fmt;

/// All errors that can arise from fixed-point construction or arithmetic.
///
/// Every variant carries enough context for the caller to produce a diagnostic
/// without re-computing anything.
#[derive(Debug, Clone, PartialEq)]
pub enum FixedPointError {
    /// The f32/f64 input was NaN.  NaN has no meaningful fixed-point
    /// representation and must be handled explicitly by the caller.
    NaN,

    /// The input, after rounding, would exceed the representable range
    /// `[min, max]`.
    OutOfRange {
        /// The value that was supplied (as f64 for display precision).
        value: f64,
        /// The minimum representable real value for this Q format.
        min: f64,
        /// The maximum representable real value for this Q format.
        max: f64,
    },

    /// Division by a zero-valued fixed-point number.
    DivisionByZero,
}

impl fmt::Display for FixedPointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FixedPointError::NaN => {
                write!(f, "fixed-point conversion rejected NaN input")
            }
            FixedPointError::OutOfRange { value, min, max } => {
                write!(
                    f,
                    "fixed-point conversion: value {value} is outside the \
                     representable range [{min}, {max}]"
                )
            }
            FixedPointError::DivisionByZero => {
                write!(f, "fixed-point division by zero")
            }
        }
    }
}

impl std::error::Error for FixedPointError {}
