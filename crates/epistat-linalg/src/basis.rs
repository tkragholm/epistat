//! Basis expansions of one variable, for series and sieve estimators and for
//! regression on smooth terms: the polynomial basis, B-splines, restricted
//! cubic splines and natural cubic splines, with first derivatives.
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
//! [`BSpline::new`] returns an [`Error`] for internal knots that are
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
//! # Restricted and natural cubic splines
//!
//! [`RestrictedCubic`] is Harrell's restricted cubic spline basis, as
//! `Hmisc::rcspline.eval` and `rms::rcs` build it, and [`NaturalSpline`] is
//! the basis of `splines::ns`. [`harrell_knots`] and [`natural_spline_knots`]
//! place their knots as R does, and [`basis_matrix`] builds the design matrix of
//! any basis over a vector of values. Each is explained in its own module.
//! [`BSpline::nonzero_values_into`] evaluates only the `d + 1` functions that are
//! not zero at a point, by the triangular scheme of de Boor, in `O(d^2)`.
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

mod knots;
mod natural;
mod rcs;

pub use knots::{harrell_knots, natural_spline_knots};
pub use natural::NaturalSpline;
pub use rcs::{RcsScale, RestrictedCubic};

use std::cmp::Ordering;

use epistat_core::check;

use crate::{Error, Mat};
use rayon::prelude::*;

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

/// The error of the bases and knot rules under its name in 0.3.
#[deprecated(since = "0.4.0", note = "use `epistat_linalg::Error`")]
pub type KnotError = crate::Error;

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
    /// [`Error`] when the boundary is not an increasing pair of finite
    /// numbers, or the internal knots are not sorted or do not lie strictly
    /// inside `(lower, upper)`. A NaN knot is refused as unsorted.
    pub fn new(
        degree: usize,
        internal_knots: &[f64],
        lower: f64,
        upper: f64,
    ) -> Result<Self, Error> {
        if !(lower.is_finite() && upper.is_finite() && lower < upper) {
            return Err(Error::Boundary);
        }
        if !internal_knots.iter().all(|&k| lower < k && k < upper) {
            return Err(Error::OutsideBoundary);
        }
        check::sorted_by("internal_knots", internal_knots, |a, b| {
            a.partial_cmp(b).unwrap_or(Ordering::Greater)
        })?;
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

    /// The span `s` with `t[s] <= x < t[s + 1]`, the last span for `x = upper`;
    /// `None` outside `[lower, upper]` and for NaN.
    fn span(&self, x: f64) -> Option<usize> {
        let t = &self.knots;
        let n = self.len();
        if !(x >= t[0] && x <= t[n + self.degree]) {
            return None;
        }
        let after = t.partition_point(|&k| k <= x);
        Some(after.saturating_sub(1).clamp(self.degree, n - 1))
    }

    /// Writes the values of the `degree + 1` functions that are not zero at `x`
    /// into `out` and returns the index of the first of them, by the triangular
    /// scheme of de Boor in `O(degree^2)`. The other functions are zero. The
    /// values equal [`Basis::values_into`] up to rounding and, like it, the last
    /// function is 1 at `x = upper`. `None` for `x` outside `[lower, upper]`.
    ///
    /// # Panics
    ///
    /// If `out` has fewer than `degree + 1` entries.
    pub fn nonzero_values_into(&self, x: f64, out: &mut [f64]) -> Option<usize> {
        let span = self.span(x)?;
        let d = self.degree;
        let t = &self.knots;
        let mut buffer = [0.0_f64; 32];
        let mut heap;
        let (left, right) = if d < 16 {
            buffer.split_at_mut(16)
        } else {
            heap = vec![0.0; 2 * (d + 1)];
            heap.split_at_mut(d + 1)
        };
        out[0] = 1.0;
        for j in 1..=d {
            left[j] = x - t[span + 1 - j];
            right[j] = t[span + j] - x;
            let mut saved = 0.0;
            for r in 0..j {
                let temp = out[r] / (right[r + 1] + left[j - r]);
                out[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            out[j] = saved;
        }
        Some(span - d)
    }

    /// The values and derivatives up to `order` of the `degree + 1` functions
    /// that are not zero at `x`, as `result[k][j]` for the `k`-th derivative of
    /// function `first + j`, with the index `first`. The recurrence is the
    /// derivative scheme of Piegl and Tiller. `None` outside `[lower, upper]`.
    #[expect(
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "the scheme indexes with offsets that are negative for some terms, as published"
    )]
    pub(crate) fn nonzero_derivatives(
        &self,
        x: f64,
        order: usize,
    ) -> Option<(usize, Vec<Vec<f64>>)> {
        let span = self.span(x)?;
        let p = self.degree;
        let order = order.min(p);
        let t = &self.knots;
        let mut ndu = vec![vec![0.0; p + 1]; p + 1];
        let (mut left, mut right) = (vec![0.0; p + 1], vec![0.0; p + 1]);
        ndu[0][0] = 1.0;
        for j in 1..=p {
            left[j] = x - t[span + 1 - j];
            right[j] = t[span + j] - x;
            let mut saved = 0.0;
            for r in 0..j {
                ndu[j][r] = right[r + 1] + left[j - r];
                let temp = ndu[r][j - 1] / ndu[j][r];
                ndu[r][j] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            ndu[j][j] = saved;
        }
        let mut ders = vec![vec![0.0; p + 1]; order + 1];
        for j in 0..=p {
            ders[0][j] = ndu[j][p];
        }
        let mut a = [vec![0.0; p + 1], vec![0.0; p + 1]];
        for r in 0..=p {
            let (mut s1, mut s2) = (0, 1);
            a[0][0] = 1.0;
            for k in 1..=order {
                let mut d = 0.0;
                let rk = r as isize - k as isize;
                let pk = p as isize - k as isize;
                if r >= k {
                    a[s2][0] = a[s1][0] / ndu[(pk + 1) as usize][rk as usize];
                    d = a[s2][0] * ndu[rk as usize][pk as usize];
                }
                let j1 = if rk >= -1 { 1 } else { (-rk) as usize };
                let j2 = if (r as isize) - 1 <= pk { k - 1 } else { p - r };
                for j in j1..=j2 {
                    let column = (rk + j as isize) as usize;
                    a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[(pk + 1) as usize][column];
                    d += a[s2][j] * ndu[column][pk as usize];
                }
                if (r as isize) <= pk {
                    a[s2][k] = -a[s1][k - 1] / ndu[(pk + 1) as usize][r];
                    d += a[s2][k] * ndu[r][pk as usize];
                }
                ders[k][r] = d;
                std::mem::swap(&mut s1, &mut s2);
            }
        }
        let mut factor = p as f64;
        for (k, row) in ders.iter_mut().enumerate().skip(1) {
            for value in row {
                *value *= factor;
            }
            factor *= (p - k) as f64;
        }
        Some((span - p, ders))
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

/// The design matrix of `basis` at `x`: row `i` holds the functions at
/// `x[i]`. Rows are computed independently, in blocks over the rayon pool,
/// so the result does not depend on the number of threads.
#[must_use]
pub fn basis_matrix<B: Basis + Sync + ?Sized>(basis: &B, x: &[f64]) -> Mat<f64> {
    const BLOCK: usize = 4096;
    let k = basis.len();
    let mut buffer = vec![0.0; x.len() * k];
    if k > 0 {
        buffer
            .par_chunks_mut(BLOCK * k)
            .zip(x.par_chunks(BLOCK))
            .for_each(|(out, values)| {
                for (row, &v) in out.chunks_exact_mut(k).zip(values) {
                    basis.values_into(v, row);
                }
            });
    }
    Mat::from_fn(x.len(), k, |i, j| buffer[i * k + j])
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_core::InputError;

    #[test]
    fn nonzero_functions_equal_the_full_recursion() {
        let basis = BSpline::new(3, &[0.25, 0.5, 0.5, 0.8], -1.0, 2.0).unwrap();
        let full_len = basis.len();
        for x in [-1.0, -0.3, 0.25, 0.5, 0.79, 1.5, 2.0] {
            let full = basis.values(x);
            let mut four = [0.0; 4];
            let first = basis.nonzero_values_into(x, &mut four).unwrap();
            for (i, &f) in full.iter().enumerate() {
                let local = if i >= first && i < first + 4 {
                    four[i - first]
                } else {
                    0.0
                };
                assert!((f - local).abs() < 1e-14, "{x} {i}");
            }
            let (first, ders) = basis.nonzero_derivatives(x, 1).unwrap();
            let slopes = basis.derivatives(x);
            assert_eq!(full_len, slopes.len());
            for (i, &f) in slopes.iter().enumerate() {
                let local = if i >= first && i < first + 4 {
                    ders[1][i - first]
                } else {
                    0.0
                };
                assert!((f - local).abs() < 1e-11, "{x} {i}: {f} {local}");
            }
        }
        assert!(basis.nonzero_values_into(2.1, &mut [0.0; 4]).is_none());
        assert!(basis.nonzero_values_into(f64::NAN, &mut [0.0; 4]).is_none());
    }

    #[test]
    fn the_design_matrix_has_the_values_row_by_row() {
        let basis = Polynomial { degree: 2 };
        let m = basis_matrix(&basis, &[1.0, 2.0, 3.0]);
        assert_eq!((m.nrows(), m.ncols()), (3, 3));
        assert!((m[(2, 2)] - 9.0).abs() < 1e-15);
    }

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
            Err(Error::OutsideBoundary)
        );
        assert_eq!(
            BSpline::new(2, &[0.0], 0.0, 1.0),
            Err(Error::OutsideBoundary)
        );
        assert_eq!(
            BSpline::new(2, &[0.6, 0.4], 0.0, 1.0),
            Err(Error::Input(InputError::Unsorted {
                what: "internal_knots",
                row: 1
            }))
        );
        assert_eq!(
            BSpline::new(2, &[f64::NAN], 0.0, 1.0),
            Err(Error::OutsideBoundary)
        );
        assert_eq!(BSpline::new(2, &[], 1.0, 0.0), Err(Error::Boundary));
        assert_eq!(
            BSpline::new(2, &[], 0.0, f64::INFINITY),
            Err(Error::Boundary)
        );
    }
}
