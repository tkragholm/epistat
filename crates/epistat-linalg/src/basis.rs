//! Basis expansions of one variable, for series and sieve estimators: the
//! polynomial basis and B-splines, with first derivatives.
//!
//! # Polynomial
//!
//! [`Polynomial`] of degree `d` has the `d + 1` functions
//!
//! ```text
//! 1, x, x^2, ..., x^d
//! ```
//!
//! with the derivatives `0, 1, 2x, ..., d x^(d-1)`. Powers are built by
//! repeated multiplication.
//!
//! # B-splines
//!
//! [`BSpline`] of degree `d` is built on the augmented knot vector
//!
//! ```text
//! t = ( lower x (d+1),  internal knots,  upper x (d+1) )
//! ```
//!
//! where the boundary knots are repeated `d + 1` times and the internal knots
//! are sorted and lie strictly inside `(lower, upper)`. Repeated internal
//! knots are allowed. With `m` internal knots there are `n = m + d + 1`
//! functions. The Cox-de Boor recursion \[1, 2, 3\] defines function `i` of
//! degree `k` by
//!
//! ```text
//! B(i, 0)(x) = 1 if t[i] <= x < t[i+1], else 0
//!
//! B(i, k)(x) =   (x - t[i])       / (t[i+k]   - t[i])   * B(i,   k-1)(x)
//!              + (t[i+k+1] - x)   / (t[i+k+1] - t[i+1]) * B(i+1, k-1)(x)
//! ```
//!
//! where a term whose denominator is zero is left out. The first derivative is
//!
//! ```text
//! B'(i, k)(x) =   k / (t[i+k]   - t[i])   * B(i,   k-1)(x)
//!               - k / (t[i+k+1] - t[i+1]) * B(i+1, k-1)(x)
//! ```
//!
//! with the same rule for a zero denominator, and it is 0 for degree 0 \[3\].
//!
//! # Edge cases
//!
//! [`BSpline::new`] returns a [`KnotError`] for internal knots that are
//! unsorted or not strictly inside the boundary, and for a boundary that is
//! not finite and increasing. The degree-0 functions are half open, which would leave the right end of
//! the interval with every function zero. The last function is therefore also
//! 1 when `x` is within 1e-12 of `upper`, so the basis sums to one on the
//! whole closed interval and its derivatives sum to zero. Outside
//! `[lower, upper]` every function is 0. The recursion is evaluated directly
//! without caching, which costs `O(2^d)` calls per function and suits the low
//! degrees of series estimators.
//!
//! This is the basis of R's `splines::bs` with `intercept = TRUE`. The
//! integration test `tests/r_splines.rs` compares values and first
//! derivatives of a cubic with three internal knots against
//! `splines::splineDesign` of R \[4\], and they agree to within four units in
//! the last place, or within an absolute 1e-14 for values and 1e-12 for
//! derivatives, on the test grid.
//!
//! # References
//!
//! 1. de Boor C. A Practical Guide to Splines. Revised ed. New York:
//!    Springer; 2001. (First edition 1978.)
//! 2. Cox MG. The numerical evaluation of B-splines. IMA J Appl Math.
//!    1972;10(2):134-49.
//! 3. de Boor C. On calculating with B-splines. J Approx Theory.
//!    1972;6(1):50-62.
//! 4. R Core Team. R: A Language and Environment for Statistical Computing.
//!    Vienna: R Foundation for Statistical Computing. The `splines` package.

/// A finite set of functions of one variable and their derivatives.
pub trait Basis {
    /// The number of functions.
    fn len(&self) -> usize;

    /// Whether the basis has no functions.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Writes the value of each function at `x` into `out`.
    fn values_into(&self, x: f64, out: &mut [f64]);

    /// Writes the first derivative of each function at `x` into `out`.
    fn derivatives_into(&self, x: f64, out: &mut [f64]);

    /// The value of every function at `x`.
    fn values(&self, x: f64) -> Vec<f64> {
        let mut out = vec![0.0; self.len()];
        self.values_into(x, &mut out);
        out
    }

    /// The first derivative of every function at `x`.
    fn derivatives(&self, x: f64) -> Vec<f64> {
        let mut out = vec![0.0; self.len()];
        self.derivatives_into(x, &mut out);
        out
    }
}

/// `1, x, x^2, ..., x^degree`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Polynomial {
    /// The highest power of `x`.
    pub degree: usize,
}

impl Basis for Polynomial {
    fn len(&self) -> usize {
        self.degree + 1
    }

    fn values_into(&self, x: f64, out: &mut [f64]) {
        let mut power = 1.0;
        for slot in &mut out[..self.len()] {
            *slot = power;
            power *= x;
        }
    }

    fn derivatives_into(&self, x: f64, out: &mut [f64]) {
        out[0] = 0.0;
        let (mut coefficient, mut power) = (1.0, 1.0);
        for slot in &mut out[1..self.len()] {
            *slot = coefficient * power;
            coefficient += 1.0;
            power *= x;
        }
    }
}

/// Invalid knots given to [`BSpline::new`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum KnotError {
    /// The boundary knots are not finite, or the lower is not below the upper.
    #[error("the boundary knots must be finite with lower below upper")]
    Boundary,
    /// An internal knot is not strictly inside the boundary.
    #[error("the internal knots must lie strictly inside the boundary")]
    OutsideBoundary,
    /// The internal knots are not in increasing order.
    #[error("the internal knots must be sorted")]
    Unsorted,
}

/// B-splines of a given degree on `[lower, upper]` with sorted internal
/// knots, by the Cox-de Boor recursion. The same basis as R's
/// `splines::bs(x, knots, degree, intercept = TRUE, Boundary.knots)`.
///
/// The last span is closed on the right, so the basis sums to one on the
/// whole closed interval.
#[derive(Clone, Debug, PartialEq)]
pub struct BSpline {
    degree: usize,
    knots: Vec<f64>,
}

impl BSpline {
    /// # Errors
    ///
    /// [`KnotError`] when the boundary is not an increasing pair of finite
    /// numbers, or the internal knots are not sorted or do not lie strictly
    /// inside `(lower, upper)`. A NaN knot is refused as unsorted.
    pub fn new(
        degree: usize,
        internal_knots: &[f64],
        lower: f64,
        upper: f64,
    ) -> Result<Self, KnotError> {
        if !(lower.is_finite() && upper.is_finite() && lower < upper) {
            return Err(KnotError::Boundary);
        }
        if !internal_knots.iter().all(|&k| lower < k && k < upper) {
            return Err(KnotError::OutsideBoundary);
        }
        if !internal_knots.windows(2).all(|pair| pair[0] <= pair[1]) {
            return Err(KnotError::Unsorted);
        }
        let mut knots = vec![lower; degree + 1];
        knots.extend_from_slice(internal_knots);
        knots.extend(std::iter::repeat_n(upper, degree + 1));
        Ok(Self { degree, knots })
    }

    /// The degree of the splines.
    #[must_use]
    pub const fn degree(&self) -> usize {
        self.degree
    }

    fn spline(&self, i: usize, k: usize, x: f64) -> f64 {
        let t = &self.knots;
        if k == 0 {
            let last = i + 1 == self.len() && (x - t[i + 1]).abs() < 1e-12;
            return if (t[i] <= x && x < t[i + 1]) || last {
                1.0
            } else {
                0.0
            };
        }
        let mut value = 0.0;
        let left = t[i + k] - t[i];
        if left > 0.0 {
            value += (x - t[i]) / left * self.spline(i, k - 1, x);
        }
        let right = t[i + k + 1] - t[i + 1];
        if right > 0.0 {
            value += (t[i + k + 1] - x) / right * self.spline(i + 1, k - 1, x);
        }
        value
    }
}

impl Basis for BSpline {
    fn len(&self) -> usize {
        self.knots.len() - self.degree - 1
    }

    fn values_into(&self, x: f64, out: &mut [f64]) {
        for (i, slot) in out[..self.len()].iter_mut().enumerate() {
            *slot = self.spline(i, self.degree, x);
        }
    }

    fn derivatives_into(&self, x: f64, out: &mut [f64]) {
        let (n, d, t) = (self.len(), self.degree, &self.knots);
        if d == 0 {
            out[..n].fill(0.0);
            return;
        }
        let lower: Vec<f64> = (0..=n).map(|i| self.spline(i, d - 1, x)).collect();
        let p = d as f64;
        for (i, slot) in out[..n].iter_mut().enumerate() {
            let mut value = 0.0;
            let left = t[i + d] - t[i];
            if left > 0.0 {
                value += p / left * lower[i];
            }
            let right = t[i + d + 1] - t[i + 1];
            if right > 0.0 {
                value -= p / right * lower[i + 1];
            }
            *slot = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polynomial_values_and_derivatives() {
        let basis = Polynomial { degree: 3 };
        assert_eq!(basis.values(2.0), [1.0, 2.0, 4.0, 8.0]);
        assert_eq!(basis.derivatives(2.0), [0.0, 1.0, 4.0, 12.0]);
    }

    #[test]
    fn b_splines_sum_to_one_on_the_closed_interval() {
        let basis = BSpline::new(3, &[0.25, 0.5, 0.5, 0.8], -1.0, 2.0).unwrap();
        assert_eq!(basis.len(), 8);
        for x in [-1.0, -0.3, 0.25, 0.5, 0.79, 1.5, 2.0] {
            let total: f64 = basis.values(x).iter().sum();
            assert!((total - 1.0).abs() < 1e-14, "{x}: {total}");
            assert!(basis.derivatives(x).iter().sum::<f64>().abs() < 1e-12);
        }
    }

    #[test]
    fn invalid_knots_are_errors() {
        assert_eq!(
            BSpline::new(2, &[1.5], 0.0, 1.0),
            Err(KnotError::OutsideBoundary)
        );
        assert_eq!(
            BSpline::new(2, &[0.0], 0.0, 1.0),
            Err(KnotError::OutsideBoundary)
        );
        assert_eq!(
            BSpline::new(2, &[0.6, 0.4], 0.0, 1.0),
            Err(KnotError::Unsorted)
        );
        assert_eq!(
            BSpline::new(2, &[f64::NAN], 0.0, 1.0),
            Err(KnotError::OutsideBoundary)
        );
        assert_eq!(BSpline::new(2, &[], 1.0, 0.0), Err(KnotError::Boundary));
        assert_eq!(
            BSpline::new(2, &[], 0.0, f64::INFINITY),
            Err(KnotError::Boundary)
        );
    }
}
