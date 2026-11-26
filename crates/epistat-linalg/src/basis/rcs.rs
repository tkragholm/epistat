//! The restricted cubic spline basis in Harrell's form.
//!
//! # Algorithm
//!
//! For knots `t_1 < ... < t_k` (`k >= 3`) the nonlinear function `j = 1, ...,
//! k - 2` is, with `(a)_+ = max(a, 0)` \[1, 2\],
//!
//! ```text
//! C_j(x) = [ (x - t_j)_+^3
//!            - (x - t_{k-1})_+^3 (t_k - t_j) / (t_k - t_{k-1})
//!            + (x - t_k)_+^3     (t_{k-1} - t_j) / (t_k - t_{k-1}) ] / D
//! C_j'(x) = 3 [ (x - t_j)_+^2 - (x - t_{k-1})_+^2 a_j + (x - t_k)_+^2 b_j ] / D
//! ```
//!
//! with `a_j = (t_k - t_j) / (t_k - t_{k-1})` and
//! `b_j = (t_{k-1} - t_j) / (t_k - t_{k-1})`. The function is 0 for
//! `x <= t_1`, a cubic with continuous first and second derivatives at every
//! knot, and linear for `x >= t_k`, because the cubic and quadratic terms
//! cancel there. The scale `D` is `(t_k - t_1)^2` ([`RcsScale::Range`], the
//! default of `rms::rcs` and `Hmisc::rcspline.eval(norm = 2)`), `(t_k -
//! t_{k-1})^3` ([`RcsScale::LastGap`], `norm = 1`) or 1 ([`RcsScale::Raw`],
//! `norm = 0`). The optional linear function is `x` itself, unscaled, and comes
//! first, as `inclx = TRUE` puts it.
//!
//! # Conditioning
//!
//! Beyond the last knot the three cubes are each of order `(x - t_1)^3` while
//! their sum is linear, so the direct form loses about two digits for every
//! factor 10 of `(x - t_1) / (t_k - t_{k-1})`. With `u = x - t_k`, `d = t_k -
//! t_{k-1}` and `e_j = t_k - t_j` the sum expands to
//!
//! ```text
//! D C_j(x) = e_j (e_j - d) (3 u + e_j + d),         x >= t_k
//! D C_j'(x) = 3 e_j (e_j - d),                      x >= t_k
//! ```
//!
//! in which the `u^3` and `u^2` terms have cancelled exactly, and this form is
//! used for `x >= t_k`. Inside the knots the direct form has terms no larger
//! than `(t_k - t_1)^3`, so its absolute error is a few times
//! `eps (t_k - t_1)` after scaling by [`RcsScale::Range`]. R evaluates the direct form
//! everywhere, so beyond the last knot R's values carry the larger error and
//! agree with these to that error only.
//!
//! # Prediction
//!
//! The basis stores its knots and evaluates a new value with them, never with
//! quantiles of the new data. A value outside `[t_1, t_k]` is extrapolated
//! linearly, which is a modelling assumption. [`RestrictedCubic::count_outside`]
//! counts such values.
//!
//! # References
//!
//! 1. Harrell FE Jr. Regression Modeling Strategies. 2nd ed. Cham: Springer;
//!    2015. Section 2.4.4.
//! 2. Durrleman S, Simon R. Flexible regression models with cubic splines. Stat
//!    Med. 1989;8(5):551-61.

use epistat_core::check;

use super::Basis;
use crate::Error;

/// How the truncated cubes are scaled, as the `norm` argument of
/// `Hmisc::rcspline.eval`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RcsScale {
    /// Not scaled (`norm = 0`).
    Raw,
    /// Divided by `(t_k - t_{k-1})^3` (`norm = 1`).
    LastGap,
    /// Divided by `(t_k - t_1)^2` (`norm = 2`), which keeps the functions of
    /// the order of the covariate.
    Range,
}

/// Harrell's restricted cubic spline basis: the `k - 2` nonlinear functions
/// of `k` knots, preceded by the linear function when asked.
#[derive(Clone, Debug, PartialEq)]
pub struct RestrictedCubic {
    knots: Vec<f64>,
    linear: bool,
    divisor: f64,
}

impl RestrictedCubic {
    /// The basis on `knots` with [`RcsScale::Range`].
    ///
    /// # Errors
    ///
    /// [`Error::TooFewKnots`] for fewer than three knots,
    /// [`Error::Input`] for a knot that is not finite or when the knots do not
    /// strictly increase.
    pub fn new(knots: &[f64], include_linear: bool) -> Result<Self, Error> {
        Self::with_scale(knots, include_linear, RcsScale::Range)
    }

    /// The basis on `knots` with the given scaling.
    ///
    /// # Errors
    ///
    /// As [`RestrictedCubic::new`].
    pub fn with_scale(knots: &[f64], include_linear: bool, scale: RcsScale) -> Result<Self, Error> {
        if knots.len() < 3 {
            return Err(Error::TooFewKnots);
        }
        check::finite("knots", knots)?;
        check::strictly_sorted_by("knots", knots, f64::total_cmp)?;
        let k = knots.len();
        let range = knots[k - 1] - knots[0];
        let gap = knots[k - 1] - knots[k - 2];
        let divisor = match scale {
            RcsScale::Raw => 1.0,
            RcsScale::LastGap => gap * gap * gap,
            RcsScale::Range => range * range,
        };
        Ok(Self {
            knots: knots.to_vec(),
            linear: include_linear,
            divisor,
        })
    }

    /// The knots.
    #[must_use]
    pub fn knots(&self) -> &[f64] {
        &self.knots
    }

    /// The number of values of `x` below the first knot or above the last,
    /// where the basis is extrapolated.
    #[must_use]
    pub fn count_outside(&self, x: &[f64]) -> usize {
        let (first, last) = (self.knots[0], self.knots[self.knots.len() - 1]);
        x.iter().filter(|&&v| v < first || v > last).count()
    }

    fn offset(&self) -> usize {
        usize::from(self.linear)
    }
}

impl Basis for RestrictedCubic {
    fn len(&self) -> usize {
        self.knots.len() - 2 + self.offset()
    }

    fn values_into(&self, x: f64, out: &mut [f64]) {
        let t = &self.knots;
        let k = t.len();
        let (last, before) = (t[k - 1], t[k - 2]);
        let d = last - before;
        if self.linear {
            out[0] = x;
        }
        for (j, slot) in out[self.offset()..self.len()].iter_mut().enumerate() {
            let e = last - t[j];
            *slot = if x >= last {
                e * (e - d) * (3.0 * (x - last) + e + d) / self.divisor
            } else {
                let cube = |a: f64| {
                    let a = a.max(0.0);
                    a * a * a
                };
                (cube(x - t[j]) - cube(x - before) * e / d + cube(x - last) * (before - t[j]) / d)
                    / self.divisor
            };
        }
    }

    fn derivatives_into(&self, x: f64, out: &mut [f64]) {
        let t = &self.knots;
        let k = t.len();
        let (last, before) = (t[k - 1], t[k - 2]);
        let d = last - before;
        if self.linear {
            out[0] = 1.0;
        }
        for (j, slot) in out[self.offset()..self.len()].iter_mut().enumerate() {
            let e = last - t[j];
            *slot = if x >= last {
                3.0 * e * (e - d) / self.divisor
            } else {
                let square = |a: f64| {
                    let a = a.max(0.0);
                    a * a
                };
                3.0 * (square(x - t[j]) - square(x - before) * e / d
                    + square(x - last) * (before - t[j]) / d)
                    / self.divisor
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_core::InputError;

    fn basis() -> RestrictedCubic {
        RestrictedCubic::new(&[1.0, 2.5, 4.0, 7.0, 9.0], true).unwrap()
    }

    #[test]
    fn the_functions_vanish_left_of_the_first_knot_and_are_linear_right_of_the_last() {
        let b = basis();
        assert_eq!(b.len(), 4);
        for x in [-3.0, 0.5, 1.0] {
            assert_eq!(&b.values(x)[1..], [0.0; 3]);
        }
        let (a, c) = (b.values(10.0), b.values(30.0));
        let mid = b.values(20.0);
        for j in 0..4 {
            assert!((mid[j] - a[j].midpoint(c[j])).abs() < 1e-12 * mid[j].abs().max(1.0));
        }
    }

    #[test]
    fn the_closed_form_meets_the_direct_form_at_the_last_knot() {
        let b = basis();
        let below = b.values(9.0 - 1e-9);
        let at = b.values(9.0);
        for j in 0..4 {
            assert!((below[j] - at[j]).abs() < 1e-8, "{j}");
        }
    }

    #[test]
    fn derivatives_match_central_differences_and_are_c2_at_knots() {
        let b = basis();
        let h = 1e-5;
        for x in [1.7, 3.3, 5.5, 8.0, 12.0] {
            let (up, down) = (b.values(x + h), b.values(x - h));
            let derivative = b.derivatives(x);
            for j in 0..4 {
                let numeric = (up[j] - down[j]) / (2.0 * h);
                assert!((numeric - derivative[j]).abs() < 1e-8, "{x} {j}");
            }
        }
        // Second differences are continuous across each knot.
        for &t in b.knots() {
            let second = |x: f64| {
                let (u, m, d) = (b.values(x + 1e-3), b.values(x), b.values(x - 1e-3));
                (1..4)
                    .map(|j| (u[j] - 2.0 * m[j] + d[j]) / 1e-6)
                    .collect::<Vec<_>>()
            };
            let (left, right) = (second(t - 2e-3), second(t + 2e-3));
            for (l, r) in left.iter().zip(&right) {
                assert!((l - r).abs() < 0.05, "{t}: {l} {r}");
            }
        }
    }

    #[test]
    fn invalid_knots_are_errors() {
        assert_eq!(
            RestrictedCubic::new(&[1.0, 2.0], false),
            Err(Error::TooFewKnots)
        );
        assert_eq!(
            RestrictedCubic::new(&[1.0, 1.0, 2.0], false),
            Err(Error::Input(InputError::Duplicate {
                what: "knots",
                row: 1
            }))
        );
        assert_eq!(
            RestrictedCubic::new(&[1.0, f64::NAN, 2.0], false),
            Err(Error::Input(InputError::NotFinite {
                what: "knots",
                row: 1
            }))
        );
    }

    #[test]
    fn values_outside_are_counted() {
        assert_eq!(basis().count_outside(&[0.0, 1.0, 5.0, 9.0, 9.5]), 2);
    }
}
