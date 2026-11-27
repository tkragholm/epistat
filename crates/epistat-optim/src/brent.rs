//! Brent's one-dimensional minimiser on an interval.
//!
//! # Algorithm
//!
//! The search \[1\] keeps an interval `[a, b]` known to hold a local minimum,
//! the best point `x` found so far, the second best `w` and the previous
//! value of `w`, `v`. Each step tries the minimum of the parabola through
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
//! `c = (3 - sqrt 5) / 2` of the larger part of the interval. The first point
//! is `a + c (b - a)`.
//!
//! The tolerance follows the convention of Forsythe, Malcolm and Moler \[2\],
//! which R documents for `optimize`: a point is never evaluated within
//! `tol1 = sqrt(ε) |x| + tolerance / 3` of `x`, or within `2 tol1` of an end
//! of the interval, with `ε` the machine epsilon, and the search stops when
//! `|x - (a + b) / 2| <= 2 tol1 - (b - a) / 2`, that is when the interval
//! around `x` is no wider than about `4 tol1`. For finite functions the
//! points evaluated are then those of `optimize` up to the last bits of a
//! parabolic step.
//!
//! Without derivatives the search converges superlinearly, with order about
//! 1.32, on a smooth function near a minimum with a positive second
//! derivative, and never much more slowly than golden-section search, which
//! shrinks the interval by `0.618` each step \[1\]. On a function with several
//! minima in `[a, b]` it finds one of them; on a flat function, some point at
//! which the stopping rule holds.
//!
//! # Values that are not finite
//!
//! The objective returns `Result`, and an error stops the search and is
//! returned, so that each caller states what a failed evaluation means. A
//! value of `+inf` is legal and worse than any finite value; the parabola is
//! not tried while one of the three stored values is infinite, so the step is
//! golden section. NaN, which has no order, and `-inf`, which means the
//! function is unbounded below, are errors ([`Error::NonFinite`]).
//!
//! # References
//!
//! 1. Brent RP. Algorithms for Minimization without Derivatives. Englewood
//!    Cliffs: Prentice-Hall; 1973. Chapter 5.
//! 2. Forsythe GE, Malcolm MA, Moler CB. Computer Methods for Mathematical
//!    Computations. Englewood Cliffs: Prentice-Hall; 1977. Chapter 8.

use crate::error::{Error, domain};

/// The result of [`minimize`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Minimum {
    /// The point found.
    pub x: f64,
    /// The function at `x`, finite or `+inf`.
    pub value: f64,
    /// The number of steps after the first point, each one evaluation.
    pub iterations: usize,
    /// The number of function evaluations, `iterations + 1`.
    pub evaluations: usize,
    /// Whether the stopping rule was met before the iteration limit.
    pub converged: bool,
}

/// Settings of [`minimize`], with R's `optimize` names in the docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Brent {
    /// The tolerance of the point, R's `tol`. The default is R's,
    /// `ε^(1/4)`.
    pub tolerance: f64,
    /// The most steps after the first point. R's `optimize` has no limit.
    /// The default, 10,000, does not bind: golden-section search shrinks any
    /// finite interval to the smallest `tol1` in fewer than 3,100 steps
    /// (`2,100 ln 2 / ln 1.618`), and a parabolic step is taken only when it
    /// is shorter than half the step before the last, so the search needs at
    /// most a small multiple of that.
    pub max_iterations: usize,
}

impl Default for Brent {
    fn default() -> Self {
        Self {
            tolerance: f64::EPSILON.powf(0.25),
            max_iterations: 10_000,
        }
    }
}

/// A local minimum of `f` on `[lower, upper]` to within about the tolerance,
/// by Brent's method, as described in the module documentation.
///
/// `f` returns `Ok` with the value at a point, or an error, which ends the
/// search and is returned. A caller whose objective cannot fail uses
/// [`Error`] as `E`.
///
/// # Errors
///
/// [`Error::Input`] when an end is not finite, `lower >= upper`, or the
/// tolerance is not finite and positive; [`Error::NonFinite`] when `f` returns
/// NaN or `-inf`; and any error `f` returns.
#[expect(
    clippy::float_cmp,
    reason = "points equal to the best one are compared exactly"
)]
pub fn minimize<E: From<Error>>(
    mut f: impl FnMut(f64) -> Result<f64, E>,
    lower: f64,
    upper: f64,
    options: &Brent,
) -> Result<Minimum, E> {
    if !lower.is_finite() {
        return Err(domain("lower", lower, "a finite number").into());
    }
    if !upper.is_finite() {
        return Err(domain("upper", upper, "a finite number").into());
    }
    if lower >= upper {
        return Err(domain("upper", upper, "above lower").into());
    }
    let tolerance = options.tolerance;
    if !(tolerance.is_finite() && tolerance > 0.0) {
        return Err(domain("tolerance", tolerance, "a finite number above 0").into());
    }
    let mut evaluate = |x: f64| -> Result<f64, E> {
        let y = f(x)?;
        if y.is_nan() || y == f64::NEG_INFINITY {
            return Err(Error::NonFinite { x }.into());
        }
        Ok(y)
    };

    let c = 0.5 * (3.0 - 5.0_f64.sqrt());
    let eps = f64::EPSILON.sqrt();
    let t = tolerance / 3.0;
    let (mut a, mut b) = (lower, upper);
    let mut x = a + c * (b - a);
    let (mut w, mut v) = (x, x);
    let mut fx = evaluate(x)?;
    let (mut fw, mut fv) = (fx, fx);
    // d is the last step and e the one before it.
    let mut d: f64 = 0.0;
    let mut e: f64 = 0.0;
    let mut iterations = 0;
    let converged = loop {
        let m = f64::midpoint(a, b);
        let tol1 = eps * x.abs() + t;
        let tol2 = 2.0 * tol1;
        if (x - m).abs() <= tol2 - 0.5 * (b - a) {
            break true;
        }
        if iterations == options.max_iterations {
            break false;
        }
        let (mut p, mut q, mut r) = (0.0, 0.0, 0.0);
        if e.abs() > tol1 && fx.is_finite() && fw.is_finite() && fv.is_finite() {
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            } else {
                q = -q;
            }
            r = e;
            e = d;
        }
        if p.abs() < (0.5 * q * r).abs() && p > q * (a - x) && p < q * (b - x) {
            d = p / q;
            let u = x + d;
            if u - a < tol2 || b - u < tol2 {
                d = if x < m { tol1 } else { -tol1 };
            }
        } else {
            e = (if x < m { b } else { a }) - x;
            d = c * e;
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else if d >= 0.0 {
            x + tol1
        } else {
            x - tol1
        };
        let fu = evaluate(u)?;
        iterations += 1;
        if fu <= fx {
            if u < x {
                b = x;
            } else {
                a = x;
            }
            (v, fv) = (w, fw);
            (w, fw) = (x, fx);
            (x, fx) = (u, fu);
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                (v, fv) = (w, fw);
                (w, fw) = (u, fu);
            } else if fu <= fv || v == x || v == w {
                (v, fv) = (u, fu);
            }
        }
    };
    Ok(Minimum {
        x,
        value: fx,
        iterations,
        evaluations: iterations + 1,
        converged,
    })
}
