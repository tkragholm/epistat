//! The Student t distribution and the regularised incomplete beta function.
//!
//! The tail is a regularised incomplete beta function. With `t` the
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
//! prefactor is formed on the log scale from [`crate::gamma::ln_gamma`], whose
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
//! [`beta_quantile`] inverts `I_x(a, b)` for `x` in `(0, 1)`. The unknown is
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

use std::f64::consts::PI;

use crate::gamma::ln_gamma;

const MAX_TERMS: usize = 10_000;
const FLOOR: f64 = 1e-300;

/// `ln f(z)` of the t density with `nu` degrees of freedom.
#[must_use]
pub fn t_ln_density(z: f64, nu: f64) -> f64 {
    let constant = ln_gamma(f64::midpoint(nu, 1.0)) - ln_gamma(0.5 * nu) - 0.5 * (nu * PI).ln();
    constant - f64::midpoint(nu, 1.0) * (z * z / nu).ln_1p()
}

/// The continued fraction of the incomplete beta function, or `None` when it
/// has not converged in `MAX_TERMS` terms.
fn beta_fraction(a: f64, b: f64, x: f64) -> Option<f64> {
    let (qab, qap, qam) = (a + b, a + 1.0, a - 1.0);
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FLOOR {
        d = FLOOR;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_TERMS {
        let m = m as f64;
        let m2 = 2.0 * m;
        let even = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + even * d;
        if d.abs() < FLOOR {
            d = FLOOR;
        }
        c = 1.0 + even / c;
        if c.abs() < FLOOR {
            c = FLOOR;
        }
        d = 1.0 / d;
        h *= d * c;
        let odd = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + odd * d;
        if d.abs() < FLOOR {
            d = FLOOR;
        }
        c = 1.0 + odd / c;
        if c.abs() < FLOOR {
            c = FLOOR;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() <= f64::EPSILON {
            return Some(h);
        }
    }
    None
}

/// The regularised incomplete beta function `I_x(a, b)`, R's `pbeta(x, a, b)`,
/// for `x` in `[0, 1]` given with `y = 1 - x` computed without cancellation.
/// NaN when the continued fraction does not converge in 10 000 terms.
///
/// `ln B(a, b)` in the prefactor is that of [`beta_quantile`]: the sum of three
/// log-gamma values while both shapes are below 1000, and from the difference
/// of log-gamma values by its Stirling series above, where the sum would lose
/// `eps ln Gamma(a)` relative, about `1e-8` in a t tail at `1e6` degrees of
/// freedom.
#[must_use]
pub fn regularised_beta(a: f64, b: f64, x: f64, y: f64) -> f64 {
    regularised_beta_with(a, b, x, y, ln_beta(a, b))
}

/// [`regularised_beta`] with `ln B(a, b)` given.
fn regularised_beta_with(a: f64, b: f64, x: f64, y: f64, ln_beta: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if y <= 0.0 {
        return 1.0;
    }
    let front = (a * x.ln() + b * y.ln() - ln_beta).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        beta_fraction(a, b, x).map_or(f64::NAN, |cf| front * cf / a)
    } else {
        let front = (b * y.ln() + a * x.ln() - ln_beta).exp();
        beta_fraction(b, a, y).map_or(f64::NAN, |cf| 1.0 - front * cf / b)
    }
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
        regularised_beta_with(a, b, x, y, ln_beta),
        regularised_beta_with(b, a, y, x, ln_beta),
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
/// `p = 1`. `None` when a shape is not positive and finite, `p` is outside
/// `[0, 1]`, or the incomplete beta function does not converge. A quantile
/// below the smallest positive `f64` is 0.
///
/// See the module documentation for the method.
#[must_use]
pub fn beta_quantile(p: f64, a: f64, b: f64) -> Option<f64> {
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
fn ln_gamma_ratio(a: f64, b: f64) -> f64 {
    if a < 1000.0 {
        return ln_gamma(a + b) - ln_gamma(a);
    }
    let c = |z: f64| {
        let z2 = z * z;
        (1.0 / 12.0 - (1.0 / 360.0 - (1.0 / 1260.0 - 1.0 / (1680.0 * z2)) / z2) / z2) / z
    };
    b * a.ln() + (a + b - 0.5) * (b / a).ln_1p() - b + c(a + b) - c(a)
}

/// The upper tail `P(F > f)` of the F distribution on `df1` and `df2` degrees
/// of freedom, R's `pf(f, df1, df2, lower.tail = FALSE)`, from the incomplete
/// beta function `I_x(df2 / 2, df1 / 2)` with `x = df2 / (df2 + df1 f)`, and from
/// the chi-square distribution of `df1 f` when `df2` is infinite. It is 1 for
/// `f <= 0`. NaN when a degree of freedom is not positive or the continued
/// fraction does not converge.
///
/// The factor `x^a y^b / B(a, b)` of the incomplete beta function is formed
/// from `y = 1 - x` given exactly, with `a ln x = a ln(1 - y)` by `ln_1p`, and
/// from a log-gamma difference that does not cancel, so that the tail keeps
/// its accuracy for the very large `df2` that multiple imputation gives. For
/// two-sided t tails, `P(|T| > z)` on `nu` degrees of freedom is
/// `f_sf(z^2, 1, nu)`.
///
/// From `df2 = 1e7` the incomplete beta function is evaluated at an `x` within
/// `eps` of 1, whose rounding gives a relative error of `eps / y`, with
/// `y = df1 f / (df2 + df1 f)`. There the tail is the chi-square tail of
/// `c = df1 f / 2` with the first term of its expansion in `1 / df2`, from
/// `W / df2` having mean 1 and variance `2 / df2` for `W` chi-square on `df2`
/// degrees of freedom:
/// `Q(df1 / 2, c) + g(c) c (c - df1 / 2 + 1) / df2`, with `g` the gamma density.
/// The next term is of order `c^4 / df2^2`.
#[must_use]
pub fn f_sf(f: f64, df1: f64, df2: f64) -> f64 {
    if !(df1 > 0.0 && df2 > 0.0) || f.is_nan() {
        return f64::NAN;
    }
    if f <= 0.0 {
        return 1.0;
    }
    if df2.is_infinite() {
        return crate::gamma::chi_square_sf(df1 * f, df1);
    }
    let (a, b) = (0.5 * df2, 0.5 * df1);
    if df2 >= 1e7 {
        let c = 0.5 * df1 * f;
        let density = ((b - 1.0) * c.ln() - c - ln_gamma(b)).exp();
        return crate::gamma::chi_square_sf(df1 * f, df1) + density * c * (c - b + 1.0) / df2;
    }
    let denominator = df2 + df1 * f;
    let (x, y) = (df2 / denominator, df1 * f / denominator);
    let front = (a * (-y).ln_1p() + b * y.ln() + ln_gamma_ratio(a, b) - ln_gamma(b)).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        beta_fraction(a, b, x).map_or(f64::NAN, |cf| front * cf / a)
    } else {
        beta_fraction(b, a, y).map_or(f64::NAN, |cf| 1.0 - front * cf / b)
    }
}

/// `P(T > |z|)`, half the two-sided tail.
fn upper_tail(z: f64, nu: f64) -> f64 {
    if z.is_infinite() {
        return 0.0;
    }
    let q = nu + z * z;
    0.5 * regularised_beta(0.5 * nu, 0.5, nu / q, z * z / q)
}

/// The distribution function and the survival function of the t distribution
/// with `nu` degrees of freedom at `z`, R's `pt(z, nu)` and
/// `pt(z, nu, lower.tail = FALSE)`. The smaller of the two is computed
/// directly, so it keeps its relative precision far out. NaN when the
/// incomplete beta function does not converge.
#[must_use]
pub fn t_cdf_sf(z: f64, nu: f64) -> (f64, f64) {
    let tail = upper_tail(z.abs(), nu);
    if z > 0.0 {
        (1.0 - tail, tail)
    } else {
        (tail, 1.0 - tail)
    }
}

/// The quantile function of the t distribution for `p` in `(0, 1)`, R's
/// `qt(p, nu)`: the root of the tail probability by Newton's method inside a
/// bracket, which is halved whenever a step leaves it. It stops when the step
/// is below `4 eps |t|`. `None` when the search for a bracket passes `1e300`.
#[must_use]
pub fn t_quantile(p: f64, nu: f64) -> Option<f64> {
    if p.to_bits() == 0.5_f64.to_bits() {
        return Some(0.0);
    }
    let lower = p < 0.5;
    let target = if lower { p } else { 1.0 - p };
    // P(T > t) = target, t > 0.
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    while upper_tail(hi, nu) > target {
        lo = hi;
        hi *= 2.0;
        if hi > 1e300 {
            return None;
        }
    }
    let mut t = f64::midpoint(lo, hi);
    for _ in 0..200 {
        let tail = upper_tail(t, nu);
        if tail > target {
            lo = t;
        } else {
            hi = t;
        }
        let density = t_ln_density(t, nu).exp();
        let newton = t + (tail - target) / density;
        let next = if newton > lo && newton < hi {
            newton
        } else {
            f64::midpoint(lo, hi)
        };
        let done = (next - t).abs() <= 4.0 * f64::EPSILON * next.abs();
        t = next;
        if done {
            break;
        }
    }
    Some(if lower { -t } else { t })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One degree of freedom is the Cauchy distribution, `1/2 + atan(z) / pi`.
    /// The error of the evaluation is a few eps in the tail it computes, and
    /// the check allows 64 eps relative to the smaller of the two tails.
    #[test]
    fn one_degree_of_freedom_is_the_cauchy_distribution() {
        for z in [-30.0, -3.0, -0.4, 0.0, 0.7, 2.0, 25.0] {
            let (cdf, sf) = t_cdf_sf(z, 1.0);
            let (want_cdf, want_sf) = (0.5 + z.atan() / PI, 0.5 - z.atan() / PI);
            let (small, want) = if z > 0.0 {
                (sf, want_sf)
            } else {
                (cdf, want_cdf)
            };
            assert!((small - want).abs() <= 64.0 * f64::EPSILON * want, "{z}");
            assert!((cdf + sf - 1.0).abs() <= 4.0 * f64::EPSILON);
        }
    }

    #[test]
    fn the_quantile_inverts_the_distribution_function() {
        for nu in [0.5, 1.0, 2.5, 4.0, 30.0, 400.0] {
            for p in [1e-10, 0.001, 0.1, 0.5, 0.9, 0.999, 1.0 - 1e-10] {
                let z = t_quantile(p, nu).unwrap();
                let (cdf, sf) = t_cdf_sf(z, nu);
                let (got, want) = if p < 0.5 { (cdf, p) } else { (sf, 1.0 - p) };
                assert!((got - want).abs() <= 1e-13 * want, "{nu} {p}: {z}");
            }
        }
    }
}
