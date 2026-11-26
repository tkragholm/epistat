//! Brent's one-dimensional minimiser on an interval, the `fmin` of R's
//! `optimize`.
//!
//! # Algorithm
//!
//! The search keeps an interval `[a, b]` known to hold a local minimum, the
//! best point `x` found so far, the second best `w` and the previous value of
//! `w`, `v` \[1\]. Each step tries the minimum of the parabola through
//! `(v, f(v))`, `(w, f(w))` and `(x, f(x))`:
//!
//! ```text
//! r = (x - w)(f(x) - f(v)),   q = (x - v)(f(x) - f(w)),
//! p = (x - v) q - (x - w) r,  q = 2 (q - r),
//! step d = p / q                (sign arranged so that q > 0)
//! ```
//!
//! and takes it when it falls inside `[a, b]` and is less than half the step
//! before the last. Otherwise it takes a golden-section step of
//! `c = (3 - sqrt 5) / 2` of the larger part of the interval. A point is never
//! evaluated within `tol1 = sqrt(eps) |x| + tol / 3` of `x`, or within `2 tol1`
//! of an end of the interval, with `eps` the machine epsilon. The search stops
//! when `|x - (a + b) / 2| <= 2 tol1 - (b - a) / 2`, that is when the interval
//! around `x` is no wider than about `4 tol1`.
//!
//! The first point is `a + c (b - a)`. The steps, the stopping rule and the
//! treatment of a value that is not finite follow R's C function `Brent_fmin`
//! line by line, so that the points evaluated are R's: a value that is NaN or
//! infinite is replaced by the largest finite double, as `optimize` does.
//!
//! Without derivatives the search converges superlinearly on a smooth function
//! near a minimum with a positive second derivative, and never more slowly
//! than golden-section search, which shrinks the interval by `0.618` each step.
//! On a function with several minima in `[a, b]` it finds one of them.
//!
//! # References
//!
//! 1. Brent RP. Algorithms for Minimization without Derivatives. Englewood
//!    Cliffs: Prentice-Hall; 1973. Chapter 5.
//! 2. R Core Team. `optimize`, src/library/stats/src/optimize.c, R 4.6.2.

use crate::error::{Error, domain};

/// The result of [`minimize`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Minimum {
    /// The point found.
    pub x: f64,
    /// The function at `x`, after a value that is not finite is replaced as
    /// described in the module documentation.
    pub value: f64,
    /// The number of steps after the first point, each one evaluation.
    pub iterations: usize,
    /// The number of function evaluations, `iterations + 1`.
    pub evaluations: usize,
    /// Whether the stopping rule was met. The search has no iteration limit,
    /// as R's has none, so this is always true; the field is there so that the
    /// three minimisers report the same fields.
    pub converged: bool,
}

/// A local minimum of `f` on `[lower, upper]` to within about `tol`, by
/// Brent's method as R's `optimize(f, c(lower, upper), tol = tol)`.
///
/// # Errors
///
/// [`Error::Input`] when an end is not finite, `lower >= upper`, or `tol` is
/// not finite and positive.
#[expect(
    clippy::float_cmp,
    reason = "Brent's algorithm tests whether a trial point is exactly one of the stored points"
)]
#[expect(
    clippy::manual_midpoint,
    reason = "the midpoint is R's (a + b) * .5, kept for the same points"
)]
#[expect(
    clippy::many_single_char_names,
    reason = "the names of Brent's algorithm, kept for comparison with its published form"
)]
pub fn minimize(
    mut f: impl FnMut(f64) -> f64,
    lower: f64,
    upper: f64,
    tol: f64,
) -> Result<Minimum, Error> {
    if !lower.is_finite() {
        return Err(domain("lower", lower, "finite"));
    }
    if !(upper.is_finite() && upper > lower) {
        return Err(domain("upper", upper, "finite and above lower"));
    }
    if !(tol.is_finite() && tol > 0.0) {
        return Err(domain("tol", tol, "(0, inf)"));
    }
    let mut evaluations = 0_usize;
    let mut eval = |x: f64| {
        evaluations += 1;
        let value = f(x);
        if value.is_finite() { value } else { f64::MAX }
    };
    let c = (3.0 - 5.0_f64.sqrt()) * 0.5;
    let eps = f64::EPSILON.sqrt();
    let (mut a, mut b) = (lower, upper);
    let mut v = a + c * (b - a);
    let mut w = v;
    let mut x = v;
    let mut d = 0.0_f64;
    let mut e = 0.0_f64;
    let mut fx = eval(x);
    let mut fv = fx;
    let mut fw = fx;
    let tol3 = tol / 3.0;
    loop {
        let xm = (a + b) * 0.5;
        let tol1 = eps * x.abs() + tol3;
        let t2 = tol1 * 2.0;
        if (x - xm).abs() <= t2 - (b - a) * 0.5 {
            break;
        }
        let mut p = 0.0;
        let mut q = 0.0;
        let mut r = 0.0;
        if e.abs() > tol1 {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = (q - r) * 2.0;
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            r = e;
            e = d;
        }
        if p.abs() >= (q * 0.5 * r).abs() || p <= q * (a - x) || p >= q * (b - x) {
            e = if x < xm { b - x } else { a - x };
            d = c * e;
        } else {
            d = p / q;
            let u = x + d;
            if u - a < t2 || b - u < t2 {
                d = if x >= xm { -tol1 } else { tol1 };
            }
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else if d > 0.0 {
            x + tol1
        } else {
            x - tol1
        };
        let fu = eval(u);
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            v = w;
            w = x;
            x = u;
            fv = fw;
            fw = fx;
            fx = fu;
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                v = w;
                fv = fw;
                w = u;
                fw = fu;
            } else if fu <= fv || v == x || v == w {
                v = u;
                fv = fu;
            }
        }
    }
    Ok(Minimum {
        x,
        value: fx,
        iterations: evaluations - 1,
        evaluations,
        converged: true,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_core::InputError;

    #[test]
    fn refuses_bad_input() {
        let input = |r: Result<Minimum, Error>, name: &str| matches!(r, Err(Error::Input(InputError::Domain { what, .. })) if what == name);
        assert!(input(minimize(|x| x, 1.0, 1.0, 1e-5), "upper"));
        assert!(input(minimize(|x| x, f64::NAN, 1.0, 1e-5), "lower"));
        assert!(input(minimize(|x| x, 0.0, 1.0, 0.0), "tol"));
    }

    #[test]
    fn finds_the_minimum_of_a_parabola() {
        let m = minimize(|x| (x - 0.3) * (x - 0.3), -1.0, 2.0, 1e-10).unwrap();
        assert!((m.x - 0.3).abs() < 1e-8, "{m:?}");
    }
}
