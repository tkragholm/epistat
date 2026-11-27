//! The regularised incomplete beta function and the beta distribution, on
//! which the t and F distributions rest.
//!
//! The t tail is a regularised incomplete beta function. With `t` the
//! variable, `nu` the degrees of freedom and `x = nu / (nu + t^2)`,
//!
//! ```text
//! P(T > |t|) = (1/2) I_x(nu/2, 1/2),
//! ```
//!
//! and `I_x(a, b)` is evaluated by its continued fraction with the modified
//! Lentz method \[1\] on the side of `(a + 1) / (a + b + 2)` where it
//! converges, using `I_x(a, b) = 1 - I_{1-x}(b, a)` on the other. The tail is
//! therefore computed directly and keeps its relative precision far out. The
//! prefactor is formed on the log scale from [`super::gamma::ln_gamma`], whose
//! relative error is about `1e-15`; when a shape is 1000 or more, `ln B(a, b)`
//! comes from the Stirling series of the difference `ln Gamma(a + b) -
//! ln Gamma(large)` instead of from three large log-gamma values that cancel.
//! The quantile is found by Newton's method on the tail inside a bracket.
//!
//! The functions agree with R's `pt`, `qt` and `pbeta` to about `1e-13`
//! relative, not bit for bit.
//!
//! # The beta quantile
//!
//! [`quantile`] inverts `I_x(a, b)` for `x` in `(0, 1)`. The unknown is
//! the logit `t = ln(x / y)` with `y = 1 - x`, which keeps both `x` and `y`
//! accurate at either end of the interval and turns a small shape into a
//! smooth problem. With `F(t) = I_x(a, b)` and the density of `t`,
//! `x^a y^b / B(a, b)`, the equation is solved for the smaller tail on the
//! log scale, `ln F = ln p` for `p <= 1/2` and `ln (1 - F) = ln (1 - p)`
//! above, by Newton's method inside a bracket that is halved when a step
//! leaves it. The slope of `ln F` is `x^a y^b / (B F)` and that of `ln (1 - F)`
//! is minus `x^a y^b / (B (1 - F))`. The upper tail is `I_y(b, a)`, formed
//! from `y` directly. A last Newton step on `ln x` (or on `ln y` when `y` is
//! the smaller) removes the rounding that `t` carries, which is of the size
//! `|t| eps` in `x` and matters when `x` is small. The result is `x`; for a
//! `p` near 1 the information about `y` is in `p` and not in `x`, as in R.
//!
//! # References
//!
//! 1. Press WH, Teukolsky SA, Vetterling WT, Flannery BP. Numerical Recipes,
//!    3rd edition. Cambridge University Press 2007, section 6.4.

use super::gamma::ln_gamma;

/// The continued fraction of DLMF 8.17.22 for `I_x(a, b)`, the factor that
/// multiplies `x^a y^b / (a B(a, b))`, or `None` when it has not converged.
pub(super) fn beta_fraction(a: f64, b: f64, x: f64) -> Option<f64> {
    let _ = (a, b, x);
    unimplemented!("the continued fraction of DLMF 8.17.22")
}

/// `P(X <= x)` for `X` beta with shapes `a` and `b`, R's `pbeta(x, a, b)`:
/// [`regularised`] with `y = 1 - x`. NaN when a shape is not positive and
/// finite or `x` is NaN; 0 below 0 and 1 above 1. Near `x = 1`, where `1 - x`
/// loses digits, pass `y` to [`regularised`] directly.
#[must_use]
pub fn cdf(x: f64, a: f64, b: f64) -> f64 {
    if x.is_nan() || !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite()) {
        return f64::NAN;
    }
    regularised(a, b, x, 1.0 - x)
}

/// The regularised incomplete beta function `I_x(a, b)`, R's `pbeta(x, a, b)`,
/// for `x` in `[0, 1]` given with `y = 1 - x` computed without cancellation.
/// NaN when the continued fraction does not converge in 10 000 terms.
///
/// `ln B(a, b)` in the prefactor is that of [`quantile`]: the sum of three
/// log-gamma values while both shapes are below 1000, and from the difference
/// of log-gamma values by its Stirling series above, where the sum would lose
/// `eps ln Gamma(a)` relative, about `1e-8` in a t tail at `1e6` degrees of
/// freedom.
#[must_use]
pub fn regularised(a: f64, b: f64, x: f64, y: f64) -> f64 {
    let _ = (a, b, x, y);
    unimplemented!("I_x(a, b) from the continued fraction and its prefactor")
}

/// [`regularised`] with `ln B(a, b)` given.
fn regularised_with(a: f64, b: f64, x: f64, y: f64, ln_beta: f64) -> f64 {
    let _ = (a, b, x, y, ln_beta);
    unimplemented!("the prefactor x^a y^b / (a B(a, b)) times the continued fraction")
}

/// The tails of the beta distribution at the logit `t`: `(ln x, ln y, F, S)`
/// with `x = 1 / (1 + e^-t)`, `F = I_x(a, b)` and `S = 1 - F = I_y(b, a)`.
/// `F` and `S` are NaN when the continued fraction does not converge.
fn beta_at_logit(a: f64, b: f64, ln_beta: f64, t: f64) -> (f64, f64, f64, f64) {
    let (ln_x, ln_y) = (-(-t).exp().ln_1p(), -t.exp().ln_1p());
    let (x, y) = (ln_x.exp(), ln_y.exp());
    (
        ln_x,
        ln_y,
        regularised_with(a, b, x, y, ln_beta),
        regularised_with(b, a, y, x, ln_beta),
    )
}

/// `ln B(a, b)`. When a shape is 1000 or more the sum of three log-gamma
/// values loses `eps` times its size, so the larger shape enters through the
/// difference of [`ln_gamma_ratio`], `ln B = ln Gamma(small) - (ln Gamma(a + b)
/// - ln Gamma(large))`.
fn ln_beta(a: f64, b: f64) -> f64 {
    let (small, large) = if a < b { (a, b) } else { (b, a) };
    if large < 1000.0 {
        ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
    } else {
        ln_gamma(small) - ln_gamma_ratio(large, small)
    }
}

/// The quantile of the beta distribution with shapes `a` and `b`: the `x` with
/// `I_x(a, b) = p`, R's `qbeta(p, a, b)`. It is 0 for `p = 0` and 1 for
/// `p = 1`. NaN when a shape is not positive and finite, `p` is outside
/// `[0, 1]`, or the incomplete beta function does not converge. A quantile
/// below the smallest positive `f64` is 0.
///
/// See the module documentation for the method.
#[must_use]
pub fn quantile(p: f64, a: f64, b: f64) -> f64 {
    quantile_or_none(p, a, b).unwrap_or(f64::NAN)
}

/// [`quantile`], `None` where it is NaN.
pub(crate) fn quantile_or_none(p: f64, a: f64, b: f64) -> Option<f64> {
    if !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite() && (0.0..=1.0).contains(&p)) {
        return None;
    }
    if p == 0.0 {
        return Some(0.0);
    }
    if p >= 1.0 {
        return Some(1.0);
    }
    let upper = p > 0.5;
    let ln_target = if upper { (1.0 - p).ln() } else { p.ln() };
    let ln_beta = ln_beta(a, b);
    // ln of the tail at t, its excess over the target (positive when t is above
    // the quantile) and the slope of the excess in t.
    let evaluate = |t: f64| -> Option<(f64, f64)> {
        let (ln_x, ln_y, lower, tail) = beta_at_logit(a, b, ln_beta, t);
        let density = (a * ln_x + b * ln_y - ln_beta).exp();
        let (value, sign) = if upper { (tail, -1.0) } else { (lower, 1.0) };
        if value.is_nan() {
            return None;
        }
        Some((sign * (value.ln() - ln_target), density / value))
    };
    // x = 0 and x = 1 are reached at |t| of about 745.
    let (mut lo, mut hi) = (-745.0_f64, 745.0_f64);
    if evaluate(lo)?.0 > 0.0 {
        return Some(0.0);
    }
    if evaluate(hi)?.0 < 0.0 {
        return Some(1.0);
    }
    let mut t = 0.0;
    for _ in 0..300 {
        let (excess, slope) = evaluate(t)?;
        if excess > 0.0 {
            hi = t;
        } else if excess < 0.0 {
            lo = t;
        } else {
            break;
        }
        let newton = t - excess / slope;
        let next = if newton > lo && newton < hi {
            newton
        } else {
            f64::midpoint(lo, hi)
        };
        let done = (next - t).abs() <= 8.0 * f64::EPSILON * next.abs().max(1.0);
        t = next;
        if done {
            break;
        }
    }
    // One step on the log of the smaller of x and y. The excess `E` has the
    // slope density / (y value) in ln x and minus density / (x value) in ln y.
    let (ln_x, ln_y, lower, tail) = beta_at_logit(a, b, ln_beta, t);
    let (value, sign) = if upper { (tail, -1.0) } else { (lower, 1.0) };
    if value.is_nan() {
        return None;
    }
    let density = (a * ln_x + b * ln_y - ln_beta).exp();
    let excess = sign * (value.ln() - ln_target);
    if t < 0.0 {
        let x = ln_x.exp();
        let step = -excess * ln_y.exp() * value / density;
        Some(x + x * step.exp_m1())
    } else {
        let y = ln_y.exp();
        let step = excess * ln_x.exp() * value / density;
        Some(1.0 - (y + y * step.exp_m1()))
    }
}

/// `ln Gamma(a + b) - ln Gamma(a)`, without the cancellation of two large
/// log-gamma values when `a` is large. From `a = 1000` the difference of the
/// Stirling series is formed term by term:
/// `b ln a + (a + b - 1/2) ln(1 + b / a) - b + c(a + b) - c(a)`, with
/// `c(z) = 1/(12 z) - 1/(360 z^3) + 1/(1260 z^5) - 1/(1680 z^7)`.
pub(super) fn ln_gamma_ratio(a: f64, b: f64) -> f64 {
    if a < 1000.0 {
        return ln_gamma(a + b) - ln_gamma(a);
    }
    let c = |z: f64| {
        let z2 = z * z;
        (1.0 / 12.0 - (1.0 / 360.0 - (1.0 / 1260.0 - 1.0 / (1680.0 * z2)) / z2) / z2) / z
    };
    b * a.ln() + (a + b - 0.5) * (b / a).ln_1p() - b + c(a + b) - c(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_distribution_function_is_the_incomplete_beta_function() {
        // pbeta(0.3, 2, 5) in R is 1 - (1 - 0.3)^5 (1 + 5 * 0.3) for these shapes:
        // I_x(2, 5) = 1 - sum_{j<2} C(6, j) x^j (1 - x)^(6 - j).
        let x = 0.3_f64;
        let want = 1.0 - (1.0 - x).powi(6) - 6.0 * x * (1.0 - x).powi(5);
        // The continued fraction and the log-gamma prefactor agree with R's
        // pbeta to about 1e-14 relative (module documentation).
        assert!((cdf(x, 2.0, 5.0) - want).abs() < 1e-14);
        assert_eq!(
            cdf(x, 2.0, 5.0).to_bits(),
            regularised(2.0, 5.0, x, 1.0 - x).to_bits()
        );
        assert_eq!(cdf(-0.5, 2.0, 5.0), 0.0);
        assert_eq!(cdf(1.5, 2.0, 5.0), 1.0);
        assert!(cdf(0.5, -1.0, 5.0).is_nan());
        assert!(cdf(f64::NAN, 1.0, 1.0).is_nan());
    }

    #[test]
    fn the_quantile_is_nan_outside_the_domain() {
        assert!(quantile(1.5, 2.0, 3.0).is_nan());
        assert!(quantile(0.5, 0.0, 3.0).is_nan());
        assert!(quantile(f64::NAN, 2.0, 3.0).is_nan());
        assert_eq!(quantile(0.0, 2.0, 3.0), 0.0);
        assert_eq!(quantile(1.0, 2.0, 3.0), 1.0);
        let x = quantile(0.4, 2.0, 3.0);
        assert!((cdf(x, 2.0, 3.0) - 0.4).abs() < 1e-14);
    }
}
