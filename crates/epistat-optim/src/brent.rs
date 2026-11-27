//! Brent's one-dimensional minimiser on an interval.
//!
//! Contract: a local minimum of a fallible objective on `[lower, upper]` by
//! Brent's `localmin` (Brent 1973, ch. 5) with the tolerance convention of
//! Forsythe, Malcolm and Moler (1977).

use crate::error::Error;

/// The result of [`minimize`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Minimum {
    /// The point found.
    pub x: f64,
    /// The function at `x`.
    pub value: f64,
    /// The number of steps after the first point, each one evaluation.
    pub iterations: usize,
    /// The number of function evaluations, `iterations + 1`.
    pub evaluations: usize,
    /// Whether the stopping rule was met before the iteration limit.
    pub converged: bool,
}

/// Settings of [`minimize`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Brent {
    /// The tolerance of the point, R's `tol`.
    pub tolerance: f64,
    /// The most steps after the first point.
    pub max_iterations: usize,
}

/// A local minimum of `f` on `[lower, upper]`.
///
/// # Errors
///
/// Not yet written.
pub fn minimize<E: From<Error>>(
    _f: impl FnMut(f64) -> Result<f64, E>,
    _lower: f64,
    _upper: f64,
    _options: &Brent,
) -> Result<Minimum, E> {
    unimplemented!("written from Brent (1973), chapter 5")
}
