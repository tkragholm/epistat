//! Roots of a function of one variable in a bracket: bisection, and Brent's
//! method as R's `uniroot`.
//!
//! Both take an interval `[lower, upper]` at whose ends the function has
//! values of opposite sign, so that a continuous function has a root inside,
//! and shrink it around a point where the sign changes.
//!
//! # Algorithm
//!
//! [`bisect`] keeps a bracket `[a, b]` whose end values differ in sign, a
//! value below zero counting as negative and zero or above as non-negative.
//! Each step evaluates the midpoint `m = a + (b - a) / 2` and replaces the end
//! whose value has the sign of `f(m)`. It stops, converged, when the midpoint
//! is no longer strictly inside the bracket, so `a` and `b` are adjacent
//! doubles, or after a step when
//!
//! ```text
//! b - a <= absolute_tolerance + relative_tolerance * |m|,
//! ```
//!
//! and otherwise after `max_iterations` steps, not converged. The root is the
//! midpoint of the last bracket. A NaN value counts as non-negative. The
//! defaults, no absolute tolerance, a relative tolerance of `2 ε` and 2,200
//! steps, are the rule of survival's Kalbfleisch-Prentice increment and reach
//! adjacent doubles from any finite bracket, since each step halves the
//! bracket and the doubles span fewer than 2,100 halvings.
//!
//! [`zeroin`] is Brent's method \[1\] in the form of R's C function
//! `R_zeroin2`, which `uniroot` calls \[2\]. It keeps the best point `b`, the
//! previous one `a`, and a point `c` with `f(c)` of the sign opposite to
//! `f(b)`. Each step tries inverse quadratic interpolation through the three
//! points, or the secant through two when `a = c`, and takes the step when it
//! falls well inside `[b, c]` and is less than half the step before the last.
//! Otherwise it bisects. A step is never shorter than
//!
//! ```text
//! tol_act = 2 ε |b| + tolerance / 2,
//! ```
//!
//! and the search stops, converged, when `|c - b| / 2 <= tol_act` or
//! `f(b) = 0`. The steps follow `R_zeroin2` line by line, so the points
//! evaluated are R's up to the last bits of an interpolation, which a fused
//! multiply-add in R's build can move and which epistat does not use. A root
//! at an end of the bracket is returned at once. The defaults are `uniroot`'s:
//! `tolerance = ε^(1/4)` and 1,000 iterations.
//!
//! # References
//!
//! 1. Brent RP. Algorithms for Minimization without Derivatives. Englewood
//!    Cliffs: Prentice-Hall; 1973. Chapter 4.
//! 2. R Core Team. `uniroot`, src/library/stats/R/nlm.R, and `R_zeroin2`,
//!    src/library/stats/src/zeroin.c, R 4.6.2.

use crate::error::{Error, domain};

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
    /// 2,200 steps, no absolute tolerance and a relative tolerance of `2 ε`:
    /// the search runs to adjacent doubles or to a bracket of relative width
    /// `2 ε`.
    fn default() -> Self {
        Self {
            max_iterations: 2_200,
            absolute_tolerance: 0.0,
            relative_tolerance: 2.0 * f64::EPSILON,
        }
    }
}

/// Settings of [`zeroin`], with R's `uniroot` names in the docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Zeroin {
    /// The most iterations, R's `maxiter`.
    pub max_iterations: usize,
    /// The tolerance of the root, R's `tol`.
    pub tolerance: f64,
}

impl Default for Zeroin {
    /// `uniroot`'s defaults: 1,000 iterations and `tol = ε^(1/4)`.
    fn default() -> Self {
        Self {
            max_iterations: 1_000,
            tolerance: f64::EPSILON.powf(0.25),
        }
    }
}

fn check_bracket(lower: f64, upper: f64) -> Result<(), Error> {
    if !lower.is_finite() {
        return Err(domain("lower", lower, "finite"));
    }
    if !(upper.is_finite() && upper > lower) {
        return Err(domain("upper", upper, "finite and above lower"));
    }
    Ok(())
}

/// A root of `f` in `[lower, upper]` by bisection, as described in the module
/// documentation.
///
/// # Errors
///
/// [`Error::Input`] when an end is not finite, `lower >= upper`, or a
/// tolerance is negative or not finite; [`Error::NoSignChange`] when `f` does
/// not have values of opposite sign at the ends.
pub fn bisect(
    mut f: impl FnMut(f64) -> f64,
    lower: f64,
    upper: f64,
    options: &Bisection,
) -> Result<Root, Error> {
    check_bracket(lower, upper)?;
    for (what, value) in [
        ("absolute_tolerance", options.absolute_tolerance),
        ("relative_tolerance", options.relative_tolerance),
    ] {
        if !(value.is_finite() && value >= 0.0) {
            return Err(domain(what, value, "[0, inf)"));
        }
    }
    let f_lower = f(lower);
    let f_upper = f(upper);
    let lower_negative = f_lower < 0.0;
    if lower_negative == (f_upper < 0.0) || f_lower.is_nan() || f_upper.is_nan() {
        return Err(Error::NoSignChange);
    }
    let (mut a, mut b) = (lower, upper);
    let mut iterations = 0;
    let mut converged = false;
    while iterations < options.max_iterations {
        let mid = a + (b - a) / 2.0;
        if mid <= a || mid >= b {
            converged = true;
            break;
        }
        iterations += 1;
        if (f(mid) < 0.0) == lower_negative {
            a = mid;
        } else {
            b = mid;
        }
        if b - a <= options.absolute_tolerance + options.relative_tolerance * mid.abs() {
            converged = true;
            break;
        }
    }
    let x = a + (b - a) / 2.0;
    let value = f(x);
    Ok(Root {
        x,
        value,
        iterations,
        evaluations: iterations + 3,
        converged,
    })
}

/// A root of `f` in `[lower, upper]` by Brent's method, as R's
/// `uniroot(f, c(lower, upper), tol = tolerance, maxiter = max_iterations)`.
///
/// `x` and `iterations` are `uniroot`'s `root` and `iter`, and `value` is
/// `f(x)` from the last evaluation, where `uniroot` evaluates `f` once more.
///
/// # Errors
///
/// [`Error::Input`] when an end is not finite, `lower >= upper`, or the
/// tolerance is not finite and positive; [`Error::NoSignChange`] when the
/// values at the ends have the same sign or one is NaN, where `uniroot`
/// stops.
#[expect(
    clippy::float_cmp,
    clippy::many_single_char_names,
    reason = "R_zeroin2's tests and names, kept for comparison with it"
)]
pub fn zeroin(
    mut f: impl FnMut(f64) -> f64,
    lower: f64,
    upper: f64,
    options: &Zeroin,
) -> Result<Root, Error> {
    check_bracket(lower, upper)?;
    let tol = options.tolerance;
    if !(tol.is_finite() && tol > 0.0) {
        return Err(domain("tolerance", tol, "(0, inf)"));
    }
    let mut fa = f(lower);
    let mut fb = f(upper);
    if fa.is_nan() || fb.is_nan() || fa * fb > 0.0 {
        return Err(Error::NoSignChange);
    }
    let found = |x: f64, value: f64, iterations: usize, converged: bool| Root {
        x,
        value,
        iterations,
        evaluations: iterations + 2,
        converged,
    };
    if fa == 0.0 {
        return Ok(found(lower, fa, 0, true));
    }
    if fb == 0.0 {
        return Ok(found(upper, fb, 0, true));
    }
    let (mut a, mut b) = (lower, upper);
    let mut c = a;
    let mut fc = fa;
    let mut iterations = 0;
    // R_zeroin2 runs the loop `maxit + 1` times, so a search that does not
    // converge takes one step more than `max_iterations`.
    while iterations <= options.max_iterations {
        let prev_step = b - a;
        if fc.abs() < fb.abs() {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        let tol_act = 2.0 * f64::EPSILON * b.abs() + tol / 2.0;
        let mut new_step = (c - b) / 2.0;
        if new_step.abs() <= tol_act || fb == 0.0 {
            return Ok(found(b, fb, iterations, true));
        }
        if prev_step.abs() >= tol_act && fa.abs() > fb.abs() {
            let cb = c - b;
            let (mut p, mut q);
            if a == c {
                let t1 = fb / fa;
                p = cb * t1;
                q = 1.0 - t1;
            } else {
                let qa = fa / fc;
                let t1 = fb / fc;
                let t2 = fb / fa;
                p = t2 * (cb * qa * (qa - t1) - (b - a) * (t1 - 1.0));
                q = (qa - 1.0) * (t1 - 1.0) * (t2 - 1.0);
            }
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if p < (0.75 * cb * q - (tol_act * q).abs() / 2.0) && p < (prev_step * q / 2.0).abs() {
                new_step = p / q;
            }
        }
        if new_step.abs() < tol_act {
            new_step = if new_step > 0.0 { tol_act } else { -tol_act };
        }
        a = b;
        fa = fb;
        b += new_step;
        fb = f(b);
        iterations += 1;
        if (fb > 0.0 && fc > 0.0) || (fb < 0.0 && fc < 0.0) {
            c = a;
            fc = fa;
        }
    }
    Ok(found(b, fb, iterations, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bisection_reaches_adjacent_doubles() {
        let root = bisect(|x| x * x - 2.0, 0.0, 2.0, &Bisection::default()).unwrap();
        assert!(root.converged);
        assert!((root.x - std::f64::consts::SQRT_2).abs() <= 4.0 * f64::EPSILON);
        assert_eq!(root.evaluations, root.iterations + 3);
        let falling = bisect(|x| 1.0 - x, 0.0, 3.0, &Bisection::default()).unwrap();
        assert!((falling.x - 1.0).abs() <= 4.0 * f64::EPSILON);
    }

    #[test]
    fn an_iteration_limit_is_reported() {
        let options = Bisection {
            max_iterations: 3,
            ..Bisection::default()
        };
        let root = bisect(|x| x - 0.3, 0.0, 1.0, &options).unwrap();
        assert!(!root.converged);
        assert_eq!(root.iterations, 3);
        assert_eq!(root.x, 0.3125);
    }

    #[test]
    fn brackets_are_checked() {
        let options = Bisection::default();
        assert_eq!(bisect(|x| x, 1.0, 2.0, &options), Err(Error::NoSignChange));
        assert_eq!(
            bisect(|_| f64::NAN, -1.0, 1.0, &options),
            Err(Error::NoSignChange)
        );
        assert!(matches!(
            bisect(|x| x, 1.0, 1.0, &options),
            Err(Error::Input(_))
        ));
        assert!(matches!(
            bisect(|x| x, f64::NEG_INFINITY, 1.0, &options),
            Err(Error::Input(_))
        ));
        let negative = Bisection {
            absolute_tolerance: -1.0,
            ..options
        };
        assert!(matches!(
            bisect(|x| x, -1.0, 1.0, &negative),
            Err(Error::Input(_))
        ));
        assert_eq!(
            zeroin(|x| x * x + 1.0, -1.0, 1.0, &Zeroin::default()),
            Err(Error::NoSignChange)
        );
        let zero = Zeroin {
            tolerance: 0.0,
            ..Zeroin::default()
        };
        assert!(matches!(
            zeroin(|x| x, -1.0, 1.0, &zero),
            Err(Error::Input(_))
        ));
    }

    #[test]
    fn a_root_at_an_end_is_returned_at_once() {
        let root = zeroin(|x| x - 1.0, 0.0, 1.0, &Zeroin::default()).unwrap();
        assert_eq!((root.x, root.iterations, root.evaluations), (1.0, 0, 2));
    }

    #[test]
    fn zeroin_stops_unconverged_at_its_limit() {
        let options = Zeroin {
            max_iterations: 2,
            tolerance: 1e-12,
        };
        let root = zeroin(|x| (x - 1.0 / 3.0).powi(3), 0.0, 1.0, &options).unwrap();
        assert!(!root.converged);
        assert_eq!(root.iterations, 3);
    }
}
