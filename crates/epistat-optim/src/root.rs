//! Roots of a function of one variable in a bracket.
//!
//! Contract: [`bisect`] keeps a bracket whose end values differ in sign and
//! halves it; [`itp`] is the ITP method of Oliveira and Takahashi (2020).

use crate::error::Error;

/// A root found in a bracket.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Root {
    /// The root.
    pub x: f64,
    /// The function at `x`.
    pub value: f64,
    /// The number of steps after the two ends were evaluated.
    pub iterations: usize,
    /// The number of function evaluations, the two ends included.
    pub evaluations: usize,
    /// Whether the stopping rule was met before the iteration limit.
    pub converged: bool,
}

/// Settings of [`bisect`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bisection {
    /// The most steps.
    pub max_iterations: usize,
    /// The width of bracket below which the search stops, added to the
    /// relative part.
    pub absolute_tolerance: f64,
    /// The width of bracket below which the search stops, as a multiple of
    /// the magnitude of the midpoint.
    pub relative_tolerance: f64,
}

impl Default for Bisection {
    fn default() -> Self {
        Self {
            max_iterations: 2_200,
            absolute_tolerance: 0.0,
            relative_tolerance: 2.0 * f64::EPSILON,
        }
    }
}

/// Settings of [`itp`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Itp {
    /// The largest distance allowed between the result and a sign change.
    pub tolerance: f64,
}

/// A root of `f` in `[lower, upper]` by bisection.
///
/// # Errors
///
/// Not yet written.
pub fn bisect(
    _f: impl FnMut(f64) -> f64,
    _lower: f64,
    _upper: f64,
    _options: &Bisection,
) -> Result<Root, Error> {
    unimplemented!("bisection is rewritten from its contract")
}

/// A root of `f` in `[lower, upper]` by the ITP method.
///
/// # Errors
///
/// Not yet written.
pub fn itp(
    _f: impl FnMut(f64) -> f64,
    _lower: f64,
    _upper: f64,
    _options: &Itp,
) -> Result<Root, Error> {
    unimplemented!("ITP is written from Oliveira and Takahashi (2020)")
}
