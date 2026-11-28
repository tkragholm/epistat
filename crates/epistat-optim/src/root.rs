//! Roots of a function of one variable in a bracket: bisection, and the ITP
//! method of Oliveira and Takahashi.
//!
//! Both take an interval `[lower, upper]` at whose ends the function has
//! values of opposite sign, so that a continuous function has a root inside,
//! and shrink it around a point where the sign changes.
//!
//! # Bisection
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
//! # ITP
//!
//! [`itp`] is the interpolate, truncate and project method \[1\]. Each step
//! computes the regula falsi point `x_f` of the bracket `[a, b]`, moves it
//! towards the midpoint `m` by `delta = kappa1 (b - a)^kappa2` (or to `m` when
//! it is closer than that), and projects the result into the ball of radius
//!
//! ```text
//! r_j = tolerance * 2^(n_max - j) - (b - a) / 2,   n_max = n_half + n0,
//! n_half = ceil(log2((b0 - a0) / (2 tolerance))),
//! ```
//!
//! around `m`, with `j` the steps taken and `[a0, b0]` the first bracket. The
//! ball keeps the bracket no wider than bisection's after the same number of
//! steps would be, times `2^n0`, so the search ends within `n_max` steps
//! whatever the function (\[1\], Theorem 2.1), and on a twice differentiable
//! function with a simple root it converges with order `kappa2` (Theorem 2.3).
//! It stops, converged, when `b - a <= 2 tolerance`, after `n_max` steps
//! (where the bracket is that narrow but for the rounding of its ends), when
//! the midpoint is no longer strictly inside the bracket, or at a point where
//! the function is 0. It returns the midpoint of the last bracket, or that
//! point, which lies within `tolerance` of a sign change, up to the rounding
//! of the ends. The defaults are those the paper argues
//! for, `kappa1 = 0.2 / (b0 - a0)`, `kappa2 = 2` and `n0 = 1`, with the
//! tolerance of R's `uniroot`, `ε^(1/4)`.
//!
//! The function may return an infinite value, which counts by its sign; the
//! step is then the midpoint. A NaN value is an error.
//!
//! # References
//!
//! 1. Oliveira IFD, Takahashi RHC. An enhancement of the bisection method
//!    average performance preserving minmax optimality. ACM Trans Math Softw.
//!    2020;47(1):5. doi:10.1145/3423597

use crate::Termination;
use crate::error::{Error, domain};

/// A root found in a bracket.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
#[must_use = "a search can stop without converging: check `converged` or call `require_converged()`"]
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
    /// [`Termination::BracketTolerance`], [`Termination::ExactRoot`] or
    /// [`Termination::MaxIterations`].
    pub termination: Termination,
}

impl Root {
    /// The result, or [`Error::NotConverged`] when its convergence test was
    /// not met.
    ///
    /// # Errors
    ///
    /// [`Error::NotConverged`] with the termination and the iterations used.
    pub fn require_converged(self) -> Result<Self, Error> {
        if self.converged {
            Ok(self)
        } else {
            Err(Error::NotConverged {
                termination: self.termination,
                iterations: self.iterations,
            })
        }
    }
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

/// Settings of [`itp`], with the names of Oliveira and Takahashi.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Itp {
    /// The largest distance allowed between the result and a sign change,
    /// the paper's `epsilon`. The default is R's `uniroot` tolerance,
    /// `ε^(1/4)`.
    pub tolerance: f64,
    /// The truncation factor `kappa1`, positive. `None`, the default, takes
    /// `0.2 / (upper - lower)`.
    pub kappa1: Option<f64>,
    /// The truncation exponent `kappa2`, in `[1, 1 + phi)` with `phi` the
    /// golden ratio. The default is 2.
    pub kappa2: f64,
    /// The slack `n0` over bisection's step count. The default is 1.
    pub n0: usize,
    /// The most steps. The default has no effect, since the search ends
    /// within `n_half + n0` steps; a smaller limit ends it early, not
    /// converged.
    pub max_iterations: usize,
}

impl Default for Itp {
    fn default() -> Self {
        Self {
            tolerance: f64::EPSILON.powf(0.25),
            kappa1: None,
            kappa2: 2.0,
            n0: 1,
            max_iterations: usize::MAX,
        }
    }
}

fn check_bracket(lower: f64, upper: f64) -> Result<(), Error> {
    if !lower.is_finite() {
        return Err(domain("lower", lower, "a finite number"));
    }
    if !upper.is_finite() {
        return Err(domain("upper", upper, "a finite number"));
    }
    if lower >= upper {
        return Err(domain("upper", upper, "above lower"));
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
            return Err(domain(what, value, "a finite number at least 0"));
        }
    }
    let lower_negative = f(lower) < 0.0;
    if lower_negative == (f(upper) < 0.0) {
        return Err(Error::NoSignChange);
    }
    let (mut a, mut b) = (lower, upper);
    let mut iterations = 0;
    let mut converged = false;
    while iterations < options.max_iterations {
        let m = a + (b - a) / 2.0;
        if m <= a || m >= b {
            converged = true;
            break;
        }
        iterations += 1;
        if (f(m) < 0.0) == lower_negative {
            a = m;
        } else {
            b = m;
        }
        if b - a <= options.absolute_tolerance + options.relative_tolerance * m.abs() {
            converged = true;
            break;
        }
    }
    let x = a + (b - a) / 2.0;
    Ok(Root {
        x,
        value: f(x),
        iterations,
        evaluations: iterations + 3,
        converged,
        termination: if converged {
            Termination::BracketTolerance
        } else {
            Termination::MaxIterations
        },
    })
}

/// `x * 2^k`, exact while the result is a normal double.
fn times_power_of_two(mut x: f64, mut k: i64) -> f64 {
    // 2^1000 and 2^-1000: biased exponents 2023 and 23.
    let up = f64::from_bits(2_023_u64 << 52);
    let down = f64::from_bits(23_u64 << 52);
    while k > 1_000 {
        x *= up;
        k -= 1_000;
    }
    while k < -1_000 {
        x *= down;
        k += 1_000;
    }
    let biased = u64::try_from(1_023 + k).expect("k lies in [-1000, 1000]");
    x * f64::from_bits(biased << 52)
}

/// The sign of `y` as used by [`itp`]: -1, 0 or 1.
fn sign(y: f64) -> f64 {
    if y > 0.0 {
        1.0
    } else if y < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// A root of `f` in `[lower, upper]` by the ITP method of Oliveira and
/// Takahashi, as described in the module documentation.
///
/// A root at an end is returned at once with no steps. `value` is `f(x)`; when
/// `x` is the midpoint of the last bracket, `f` is evaluated there once more
/// and the evaluation is counted.
///
/// # Errors
///
/// [`Error::Input`] when an end is not finite, `lower >= upper`, the tolerance
/// is not finite and positive, `kappa1` is not finite and positive, or
/// `kappa2` is outside `[1, 1 + phi)`; [`Error::NonFinite`] when `f` is NaN at
/// an evaluated point; [`Error::NoSignChange`] when the values at the ends
/// have the same sign.
pub fn itp(
    mut f: impl FnMut(f64) -> f64,
    lower: f64,
    upper: f64,
    options: &Itp,
) -> Result<Root, Error> {
    check_bracket(lower, upper)?;
    let eps = options.tolerance;
    if !(eps.is_finite() && eps > 0.0) {
        return Err(domain("tolerance", eps, "a finite number above 0"));
    }
    let kappa1 = options.kappa1.unwrap_or(0.2 / (upper - lower));
    if !(kappa1.is_finite() && kappa1 > 0.0) {
        return Err(domain("kappa1", kappa1, "a finite number above 0"));
    }
    let kappa2 = options.kappa2;
    let golden = f64::midpoint(1.0, 5.0_f64.sqrt());
    if !(kappa2 >= 1.0 && kappa2 < 1.0 + golden) {
        return Err(domain("kappa2", kappa2, "[1, 1 + golden ratio)"));
    }
    let mut evaluate = |x: f64| -> Result<f64, Error> {
        let y = f(x);
        if y.is_nan() {
            Err(Error::NonFinite { x })
        } else {
            Ok(y)
        }
    };
    let (mut a, mut b) = (lower, upper);
    let mut ya = evaluate(a)?;
    let mut yb = evaluate(b)?;
    let done = |x: f64, value: f64, iterations: usize, evaluations: usize, termination| Root {
        x,
        value,
        iterations,
        evaluations,
        converged: Termination::is_converged(termination),
        termination,
    };
    if ya == 0.0 {
        return Ok(done(a, ya, 0, 2, Termination::ExactRoot));
    }
    if yb == 0.0 {
        return Ok(done(b, yb, 0, 2, Termination::ExactRoot));
    }
    let s = sign(yb);
    if (ya > 0.0) == (yb > 0.0) {
        return Err(Error::NoSignChange);
    }
    // The bracket's width over 2 eps can overflow; its base-2 logarithm cannot.
    let half_steps = ((b / 2.0 - a / 2.0) / eps).log2().ceil().max(0.0);
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the logarithm of a finite ratio of doubles is below 2,200"
    )]
    let n_max = half_steps as i64 + i64::try_from(options.n0).unwrap_or(i64::MAX / 2);
    let mut j: i64 = 0;
    let mut iterations = 0;
    let mut converged = true;
    // In exact arithmetic the bracket is no wider than 2 tolerance after
    // n_max steps; the rounding of the ends can leave it wider by a few units
    // in their last place, which the step count does not wait for.
    while b - a > 2.0 * eps && j < n_max {
        if iterations == options.max_iterations {
            converged = false;
            break;
        }
        let m = a + (b - a) / 2.0;
        if !(a < m && m < b) {
            break;
        }
        let r = (times_power_of_two(eps, n_max - j) - (b - a) / 2.0).max(0.0);
        let delta = kappa1 * (b - a).powf(kappa2);
        let mut xf = m;
        if ya.is_finite() && yb.is_finite() {
            let interpolated = (yb * a - ya * b) / (yb - ya);
            if interpolated.is_finite() {
                xf = interpolated;
            }
        }
        let sigma = sign(m - xf);
        let xt = if delta <= (m - xf).abs() {
            xf + sigma * delta
        } else {
            m
        };
        let mut x = if (xt - m).abs() <= r {
            xt
        } else {
            m - sigma * r
        };
        if !(a < x && x < b) {
            x = m;
        }
        let y = evaluate(x)?;
        iterations += 1;
        j += 1;
        if s * y > 0.0 {
            b = x;
            yb = y;
        } else if s * y < 0.0 {
            a = x;
            ya = y;
        } else {
            return Ok(done(
                x,
                y,
                iterations,
                iterations + 2,
                Termination::ExactRoot,
            ));
        }
    }
    let x = a + (b - a) / 2.0;
    let value = evaluate(x)?;
    let termination = if converged {
        Termination::BracketTolerance
    } else {
        Termination::MaxIterations
    };
    Ok(done(x, value, iterations, iterations + 3, termination))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powers_of_two_are_exact() {
        for k in [-1_022_i32, -1, 0, 1, 52, 1_023] {
            assert_eq!(
                times_power_of_two(1.0, i64::from(k)),
                2.0_f64.powi(k),
                "{k}"
            );
        }
        let tiny = times_power_of_two(1.0, -1_000);
        assert_eq!(
            times_power_of_two(tiny, 2_000),
            times_power_of_two(1.0, 1_000)
        );
        assert_eq!(times_power_of_two(1.0, -1_074), f64::from_bits(1));
        assert!(times_power_of_two(1.0, 2_100).is_infinite());
    }
}
