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
//! and `I_x(a, b)` is evaluated by the continued fraction of DLMF 8.17.22,
//!
//! ```text
//! I_x(a, b) = x^a y^b / (a B(a, b)) / (1 + d_1 / (1 + d_2 / (1 + ...))),
//! d_2m     =  m (b - m) x / ((a + 2m - 1)(a + 2m)),
//! d_2m+1   = -(a + m)(a + b + m) x / ((a + 2m)(a + 2m + 1)),
//! ```
//!
//! on the side of `(a + 1) / (a + b + 2)` where it converges fast, using
//! `I_x(a, b) = 1 - I_y(b, a)` on the other. The tail below that point is
//! therefore computed directly and keeps its relative precision far out.
//!
//! The fraction is evaluated from the top by the modified Lentz method
//! \[2, 3\], with each step's `1 + c_n` rewritten so that it is not formed by
//! cancellation when `c_n` is close to `-1`, which happens at every odd step
//! when a shape is large and `x` is close to 1: there `1 + d_n` is formed from
//! `y`. It stops when the convergents `n` and `n - 2` agree to `2 eps`. The
//! number of terms grows like the square root of the larger shape, and the
//! limit on it is set from that growth with room to spare, so the fraction
//! converges wherever it is used.
//!
//! The factor `x^a y^b / B(a, b)` is formed from the smaller of `x` and `y`.
//! While a shape is below 10 it is `exp(a ln x + b ln y - ln B(a, b))`, with
//! [`super::gamma::ln_gamma`], whose relative error is about `1e-15`; when a
//! shape is 1000 or more, `ln B(a, b)` comes from the Stirling series of the
//! difference `ln Gamma(a + b) - ln Gamma(large)` instead of from three large
//! log-gamma values that cancel. When both shapes are 10 or more it is formed
//! about the mean from Stirling's formula, so that the terms of first order in
//! `x - a / (a + b)` cancel before they are formed and the error is eps times
//! the exponent itself rather than times its terms, which near the centre are
//! of the size `a + b` (see `power_factor`). The quantile is found by
//! Newton's method on the tail inside a bracket.
//!
//! Against mpmath at 50 digits over shapes from `1e-3` to `1e7`, the error is
//! within a bound of eps times the magnitude of the exponent plus
//! `64 eps + 4 sqrt(a + b) eps` for the fraction. The functions agree with
//! R's `pt`, `qt` and `pbeta` to about `1e-13` relative, not bit for bit.
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
//! 1. NIST Digital Library of Mathematical Functions, sections
//!    8.17 (the incomplete beta function, its continued fraction and its
//!    contiguous relations) and 5.11 (Stirling's series).
//! 2. Lentz WJ. Generating Bessel functions in Mie scattering calculations
//!    using continued fractions. Applied Optics 1976; 15(3): 668-671.
//! 3. Thompson IJ, Barnett AR. Coulomb and Bessel functions of complex
//!    arguments and order. Journal of Computational Physics 1986; 64(2):
//!    490-509.

use super::gamma::ln_gamma;

/// From this smaller shape on, `x^a y^b / B(a, b)` is formed about the mean
/// with Stirling corrections instead of from `ln B(a, b)`.
const CENTRED_FROM: f64 = 10.0;

/// A value with a finite reciprocal, for a Lentz ratio that is exactly zero.
const TINY: f64 = f64::MIN_POSITIVE / f64::EPSILON;

/// The most terms the continued fraction takes for shapes `a` and `b`.
///
/// Near the switching point `x = (a + 1) / (a + b + 2)` the number of terms
/// grows like the square root of the larger shape. Over shapes from `1e-3` to
/// `1e7` the most measured is 121 terms at `a = 100`, `b = 0.003`, which is
/// 0.42 of this limit; elsewhere the share is smaller.
fn term_limit(a: f64, b: f64) -> f64 {
    128.0 + 16.0 * a.max(b).sqrt()
}

/// `d_n` and `1 + d_n` of DLMF 8.17.23, for `n = 2m` or `n = 2m + 1`.
///
/// For odd `n`, `d_n` is close to `-1` when `x` is close to 1 and a shape is
/// large, and `1 + d_n` is then formed from `y` without the cancellation:
/// `(a + 2m)(a + 2m + 1) (1 + d_n) = a (2m + 1 - b) + m (3m + 2 - b)
/// + (a + m)(a + b + m) y`.
fn coefficient(odd: bool, m: f64, a: f64, b: f64, x: f64, y: f64) -> (f64, f64) {
    if odd {
        let denominator = (a + 2.0 * m) * (a + 2.0 * m + 1.0);
        let d = -(a + m) * (a + b + m) * x / denominator;
        let one_plus = if y < x {
            (a * (2.0 * m + 1.0 - b) + m * (3.0 * m + 2.0 - b) + (a + m) * (a + b + m) * y)
                / denominator
        } else {
            1.0 + d
        };
        (d, one_plus)
    } else {
        let d = m * (b - m) * x / ((a + 2.0 * m - 1.0) * (a + 2.0 * m));
        (d, 1.0 + d)
    }
}

/// The reciprocal of the continued fraction `1 + d_1 / (1 + d_2 / (1 + ...))`
/// of DLMF 8.17.22, the factor of `x^a y^b / (a B(a, b))` in `I_x(a, b)`, or
/// `None` when it has not converged within [`term_limit`] terms.
///
/// The fraction is evaluated from the top by the modified Lentz method, with
/// the ratios `C_n = A_n / A_{n-1}` and `D_n = B_{n-1} / B_n` of successive
/// numerators and denominators. With `c_n = d_n / C_{n-1}` and
/// `e_n = d_n D_{n-1}`, the recurrences `C_n = 1 + c_n` and `1 / D_n = 1 + e_n`
/// are written `C_n = (1 + d_n) - c_n c_{n-1}` and
/// `1 / D_n = (1 + d_n) - e_n e_{n-1}`, which are exact rewritings that take
/// `1 + d_n` from [`coefficient`] and so avoid the cancellation of `1 + c_n`
/// when `c_n` is close to `-1`. It stops when the convergents `f_n` and
/// `f_{n-2}` agree to `2 eps`. One step is not enough: where `d_n` is tiny, a
/// single step leaves the value unchanged long before it has converged.
fn beta_fraction(a: f64, b: f64, x: f64, y: f64) -> Option<f64> {
    let limit = term_limit(a, b);
    // n = 1: C_0 = 1 and D_0 = 0.
    let (d, one_plus) = coefficient(true, 0.0, a, b, x, y);
    let (mut big_c, mut c) = (one_plus, d);
    let (mut big_d, mut e) = (1.0, 0.0);
    let mut value = if big_c == 0.0 { TINY } else { big_c };
    big_c = value;
    let (mut before, mut last) = (1.0, value);
    let (mut m, mut odd, mut n) = (1.0, false, 2.0);
    while n <= limit {
        let (d, one_plus) = coefficient(odd, m, a, b, x, y);
        let c_next = d / big_c;
        big_c = one_plus - c_next * c;
        c = c_next;
        if big_c == 0.0 {
            big_c = TINY;
        }
        let e_next = d * big_d;
        let mut inverse = one_plus - e_next * e;
        e = e_next;
        if inverse == 0.0 {
            inverse = TINY;
        }
        big_d = 1.0 / inverse;
        value *= big_c * big_d;
        if (value - before).abs() <= 2.0 * f64::EPSILON * value.abs() {
            return Some(1.0 / value);
        }
        before = last;
        last = value;
        if odd {
            m += 1.0;
        }
        odd = !odd;
        n += 1.0;
    }
    None
}

/// `z - ln(1 + z)` for `z > -1`, with `ratio = 1 + z` formed by the caller
/// without cancellation. For `z` in `[-1/2, 1]`, with `w = z / (2 + z)`,
/// `ln(1 + z) = 2 (w + w^3 / 3 + w^5 / 5 + ...)` and `z - 2 w = z w`, so
/// `z - ln(1 + z) = z w - 2 w^3 (1/3 + w^2 / 5 + ...)`, whose two terms have
/// the same sign or differ by a factor of at least 6 while `|w| <= 1/3`.
/// Outside, `z - ln(ratio)` loses at most a factor of 3, and `ratio` keeps
/// its relative precision as `z` approaches `-1`, where `1 + z` would not.
fn log1p_gap(z: f64, ratio: f64) -> f64 {
    if (-0.5..=1.0).contains(&z) {
        let w = z / (2.0 + z);
        let w2 = w * w;
        let (mut sum, mut power, mut k) = (0.0, 1.0, 3.0);
        loop {
            let term = power / k;
            sum += term;
            if term <= f64::EPSILON * sum {
                break;
            }
            power *= w2;
            k += 2.0;
        }
        z * w - 2.0 * w * w2 * sum
    } else {
        z - ratio.ln()
    }
}

/// The Stirling correction `ln Gamma(z) - [(z - 1/2) ln z - z + ln(2 pi) / 2]`
/// for `z >= 10`, by its asymptotic series
/// `sum_k B_2k / (2k (2k - 1) z^(2k - 1))` (DLMF 5.11.1) to `k = 8`. The first
/// term left out is below `2e-18` from `z = 10`.
fn stirling_correction(z: f64) -> f64 {
    let r = 1.0 / (z * z);
    let series = 1.0 / 12.0
        + r * (-1.0 / 360.0
            + r * (1.0 / 1260.0
                + r * (-1.0 / 1680.0
                    + r * (1.0 / 1188.0
                        + r * (-691.0 / 360_360.0
                            + r * (1.0 / 156.0 + r * (-3617.0 / 122_400.0)))))));
    series / z
}

/// `x^a y^b / B(a, b)`, with `y = 1 - x`; `ln_beta` is `ln B(a, b)` when the
/// caller has it.
///
/// Both logarithms come from the smaller of `x` and `y`, which carries the
/// more information about the point. While a shape is below [`CENTRED_FROM`]
/// the factor is `exp(a ln x + b ln y - ln B(a, b))`, which loses eps times
/// the magnitude of the exponent's terms. From there on, with
/// `x0 = a / (a + b)`, `y0 = b / (a + b)` and `e = x - x0`, Stirling's formula
/// for the three gamma functions gives
///
/// ```text
/// x^a y^b / B(a, b) = sqrt(a b / (2 pi (a + b)))
///     exp(-[a g(e / x0) + b g(-e / y0)] - D(a) - D(b) + D(a + b)),
/// ```
///
/// with `g(z) = z - ln(1 + z)` and `D` the Stirling correction. The terms of
/// first order in `e` cancel exactly before they are formed, since
/// `a e / x0 = b e / y0`, so the exponent is small near the centre and its
/// error is eps times its own size.
pub(super) fn power_factor(a: f64, b: f64, x: f64, y: f64, ln_beta: Option<f64>) -> f64 {
    let (scale, exponent) = power_parts(a, b, x, y, ln_beta);
    scale * exponent.exp()
}

/// `ln(x^a y^b / B(a, b))`, from the terms of [`power_factor`].
pub(super) fn ln_power_factor(a: f64, b: f64, x: f64, y: f64) -> f64 {
    let (scale, exponent) = power_parts(a, b, x, y, None);
    scale.ln() + exponent
}

/// `(scale, exponent)` with `x^a y^b / B(a, b) = scale exp(exponent)`, as
/// [`power_factor`] describes them.
fn power_parts(a: f64, b: f64, x: f64, y: f64, ln_beta: Option<f64>) -> (f64, f64) {
    if a.min(b) >= CENTRED_FROM {
        let s = a + b;
        // (a + b) e = b x - a y, from the smaller of x and y.
        let scaled = if x <= y { s * x - a } else { b - s * y };
        // x / x0 = 1 + (a + b) e / a and y / y0 = 1 - (a + b) e / b.
        let exponent = -(a * log1p_gap(scaled / a, x * s / a)
            + b * log1p_gap(-scaled / b, y * s / b))
            - stirling_correction(a)
            - stirling_correction(b)
            + stirling_correction(s);
        ((a / s * b / std::f64::consts::TAU).sqrt(), exponent)
    } else {
        let (ln_x, ln_y) = if x <= y {
            (x.ln(), (-x).ln_1p())
        } else {
            ((-y).ln_1p(), y.ln())
        };
        let ln_beta = ln_beta.unwrap_or_else(|| self::ln_beta(a, b));
        (1.0, a * ln_x + b * ln_y - ln_beta)
    }
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

/// `P(X > x)`, R's `pbeta(x, a, b, lower.tail = FALSE)`: `I_y(b, a)` with
/// `y = 1 - x`, computed directly so that a small upper tail keeps its
/// relative precision. NaN as for [`cdf`]; 1 below 0 and 0 above 1.
#[must_use]
pub fn sf(x: f64, a: f64, b: f64) -> f64 {
    if x.is_nan() || !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite()) {
        return f64::NAN;
    }
    regularised(b, a, 1.0 - x, x)
}

/// The beta density at `x`, R's `dbeta(x, a, b)`,
/// `x^(a - 1) (1 - x)^(b - 1) / B(a, b)`, from the factor
/// `x^a y^b / B(a, b)` of the incomplete beta function divided by `x y`, so
/// that it has that factor's accuracy at large shapes. At `x = 0` it is
/// infinite for `a < 1`, `b` for `a = 1` and 0 above, and at `x = 1` the same
/// with the shapes exchanged; 0 outside `[0, 1]`. NaN as for [`cdf`].
#[must_use]
pub fn pdf(x: f64, a: f64, b: f64) -> f64 {
    if x.is_nan() || !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite()) {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&x) {
        return 0.0;
    }
    if x == 0.0 {
        return end_density(a, b);
    }
    if x >= 1.0 {
        return end_density(b, a);
    }
    let y = 1.0 - x;
    power_factor(a, b, x, y, None) / (x * y)
}

/// The log of [`pdf`], from the same terms.
#[must_use]
pub fn ln_pdf(x: f64, a: f64, b: f64) -> f64 {
    if x.is_nan() || !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite()) {
        return f64::NAN;
    }
    if !(0.0..=1.0).contains(&x) {
        return f64::NEG_INFINITY;
    }
    if x == 0.0 {
        return end_density(a, b).ln();
    }
    if x >= 1.0 {
        return end_density(b, a).ln();
    }
    let y = 1.0 - x;
    ln_power_factor(a, b, x, y) - x.ln() - y.ln()
}

/// The density at the end of the interval where the shape `near` acts:
/// `x^(near - 1)` at `x = 0` is infinite below 1, 1 at 1 and 0 above, and
/// the constant at `near = 1` is `1 / B(1, far) = far`.
fn end_density(near: f64, far: f64) -> f64 {
    match near.partial_cmp(&1.0) {
        Some(core::cmp::Ordering::Less) => f64::INFINITY,
        Some(core::cmp::Ordering::Equal) => far,
        _ => 0.0,
    }
}

/// The regularised incomplete beta function `I_x(a, b)`, R's `pbeta(x, a, b)`,
/// for `x` in `[0, 1]` given with `y = 1 - x` computed without cancellation.
/// It is 0 for `x <= 0` and 1 for `y <= 0`. NaN when a shape is not positive
/// and finite, `x` or `y` is NaN, or the continued fraction has not converged
/// within its term limit, which over shapes from `1e-3` to `1e7` it does in
/// less than half the limit.
///
/// The continued fraction DLMF 8.17.22 is evaluated on the side of
/// `(a + 1) / (a + b + 2)` where it converges, with
/// `I_x(a, b) = 1 - I_y(b, a)` on the other, so the tail below that point is
/// computed directly. The factor `x^a y^b / B(a, b)` is formed about the mean
/// when both shapes are 10 or more and on the log scale below (module
/// documentation).
#[must_use]
pub fn regularised(a: f64, b: f64, x: f64, y: f64) -> f64 {
    regularised_with(a, b, x, y, None)
}

/// [`regularised`], with `ln B(a, b)` given when the caller has it.
fn regularised_with(a: f64, b: f64, x: f64, y: f64, ln_beta: Option<f64>) -> f64 {
    if x.is_nan() || y.is_nan() || !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite()) {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if y <= 0.0 {
        return 1.0;
    }
    let front = power_factor(a, b, x, y, ln_beta);
    if x < (a + 1.0) / (a + b + 2.0) {
        beta_fraction(a, b, x, y).map_or(f64::NAN, |fraction| front * fraction / a)
    } else {
        beta_fraction(b, a, y, x).map_or(f64::NAN, |fraction| 1.0 - front * fraction / b)
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
        regularised_with(a, b, x, y, Some(ln_beta)),
        regularised_with(b, a, y, x, Some(ln_beta)),
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
    invert(a, b, upper, ln_target)
}

/// The upper-tail quantile, the `x` with `P(X > x) = p`, R's
/// `qbeta(p, a, b, lower.tail = FALSE)`. It is [`quantile`] at `1 - p`
/// without forming `1 - p` where that would round: below `p = 1/2` it solves
/// for the upper tail at `ln p` itself. 1 at `p = 0` and 0 at `p = 1`; NaN as
/// for [`quantile`].
#[must_use]
pub fn isf(p: f64, a: f64, b: f64) -> f64 {
    isf_or_none(p, a, b).unwrap_or(f64::NAN)
}

/// [`isf`], `None` where it is NaN.
pub(crate) fn isf_or_none(p: f64, a: f64, b: f64) -> Option<f64> {
    if !(a > 0.0 && b > 0.0 && a.is_finite() && b.is_finite() && (0.0..=1.0).contains(&p)) {
        return None;
    }
    if p == 0.0 {
        return Some(1.0);
    }
    if p >= 1.0 {
        return Some(0.0);
    }
    // quantile(1 - p) solves the upper tail when 1 - p > 1/2, at ln p.
    let upper = p < 0.5;
    let ln_target = if upper { p.ln() } else { (1.0 - p).ln() };
    invert(a, b, upper, ln_target)
}

/// The `x` whose lower tail (or upper tail, with `upper`) has the log
/// `ln_target`, by the search of the module documentation.
fn invert(a: f64, b: f64, upper: bool, ln_target: f64) -> Option<f64> {
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

    const EPS: f64 = f64::EPSILON;

    /// The relative error allowed in a tail that [`regularised`] computes
    /// directly. The factor `x^a y^b / B(a, b)` is an exponential, so its
    /// relative error is the absolute error of its exponent: below
    /// [`CENTRED_FROM`], eps times the magnitudes of `a ln x`, `b ln y` and the
    /// three log-gamma values, each with a relative error below `5 eps`, so the
    /// coefficient is 8; from there on, eps times `a g(u) + b g(v)` and the
    /// error `eps (a |u| + b |v|)` of the formed `u` and `v` (see
    /// [`power_factor`]), with the same coefficient. The continued fraction
    /// takes of the order of `sqrt(a + b)` terms with a few roundings each,
    /// `64 eps + 4 sqrt(a + b) eps`. The shapes here are below 1000.
    fn direct_bound(a: f64, b: f64, x: f64, y: f64) -> f64 {
        let terms = if a.min(b) >= CENTRED_FROM {
            let s = a + b;
            let scaled = if x <= y { s * x - a } else { b - s * y };
            let (u, v) = (scaled / a, -scaled / b);
            a * (u - u.ln_1p()) + b * (v - v.ln_1p()) + a * u.abs() + b * v.abs() + 1.0
        } else {
            (a * x.ln()).abs()
                + (b * y.ln()).abs()
                + ln_gamma(a).abs()
                + ln_gamma(b).abs()
                + ln_gamma(a + b).abs()
        };
        8.0 * EPS * terms + (64.0 + 4.0 * (a + b).sqrt()) * EPS
    }

    /// Checks `regularised(a, b, x, y)` against `want`, whose complement is
    /// `other`, both from a closed form with an error of `slack` eps. A tail
    /// below the switching point is computed directly and held to
    /// [`direct_bound`] relative; one above is `1 - I_y(b, a)` and held to
    /// that bound times `other`.
    fn assert_tail(a: f64, b: f64, x: f64, y: f64, want: f64, other: f64, slack: f64) {
        let got = regularised(a, b, x, y);
        if want == 0.0 {
            assert_eq!(got.to_bits(), want.to_bits(), "I_{x}({a}, {b}) underflows");
            return;
        }
        let bound = direct_bound(a, b, x, y);
        let (error, allowed) = if x < (a + 1.0) / (a + b + 2.0) {
            (((got - want) / want).abs(), bound + slack * EPS)
        } else {
            ((got - want).abs(), bound * other + slack * EPS * want)
        };
        assert!(
            error <= allowed,
            "I_{x}({a}, {b}) = {got:e}, closed form {want:e}, error {error:e}, bound {allowed:e}"
        );
    }

    const XS: [f64; 7] = [1e-6, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999_999];

    #[test]
    fn closed_forms_with_a_unit_shape() {
        // I_x(a, 1) = x^a and I_x(1, b) = 1 - (1 - x)^b, each to a few eps.
        for shape in [1e-3, 0.3, 1.0, 2.5, 40.0, 1e4] {
            for x in XS {
                let y = 1.0 - x;
                let power = (shape * x.ln()).exp();
                assert_tail(shape, 1.0, x, y, power, -(shape * x.ln()).exp_m1(), 4.0);
                let power = (shape * (-x).ln_1p()).exp();
                assert_tail(
                    1.0,
                    shape,
                    x,
                    y,
                    -(shape * (-x).ln_1p()).exp_m1(),
                    power,
                    4.0,
                );
            }
        }
    }

    #[test]
    fn closed_form_of_the_arcsine_distribution() {
        // I_x(1/2, 1/2) = (2 / pi) arcsin(sqrt x), and the complement is the
        // same at y. Above 1/2 the arcsine of sqrt x is ill-conditioned, so the
        // reference there is one minus that of y.
        let arcsine = |x: f64| std::f64::consts::FRAC_2_PI * x.sqrt().asin();
        for x in XS {
            let y = 1.0 - x;
            let (want, other) = if x <= 0.5 {
                (arcsine(x), 1.0 - arcsine(x))
            } else {
                (1.0 - arcsine(y), arcsine(y))
            };
            assert_tail(0.5, 0.5, x, y, want, other, 4.0);
        }
    }

    /// `I_x(a, b)` and its complement for integer shapes and `x = p / 16`, as
    /// the binomial sum `sum_{j >= a} C(n, j) p^j (16 - p)^(n - j) / 16^n`
    /// with `n = a + b - 1`, exact in integers; each is rounded once.
    fn binomial_sum(a: u32, b: u32, p: u128) -> (f64, f64) {
        let n = a + b - 1;
        let (mut lower, mut upper, mut choose) = (0_u128, 0_u128, 1_u128);
        for j in 0..=n {
            let term = choose * p.pow(j) * (16 - p).pow(n - j);
            if j >= a {
                lower += term;
            } else {
                upper += term;
            }
            choose = choose * u128::from(n - j) / u128::from(j + 1);
        }
        let scale = 16.0_f64.powi(i32::try_from(n).unwrap());
        (lower as f64 / scale, upper as f64 / scale)
    }

    #[test]
    fn integer_shapes_against_the_exact_binomial_sum() {
        for (a, b) in [
            (1, 1),
            (2, 5),
            (3, 3),
            (5, 2),
            (7, 12),
            (10, 20),
            (15, 16),
            (20, 11),
        ] {
            for p in [1, 3, 5, 8, 11, 15] {
                let (lower, upper) = binomial_sum(a, b, p);
                let x = p as f64 / 16.0;
                let (a, b) = (f64::from(a), f64::from(b));
                assert_tail(a, b, x, 1.0 - x, lower, upper, 2.0);
                assert_tail(b, a, 1.0 - x, x, upper, lower, 2.0);
            }
        }
    }

    #[test]
    fn the_t_distribution_with_one_and_two_degrees_of_freedom() {
        use super::super::t;
        // P(T > z) = arctan(1 / z) / pi on one degree of freedom and
        // 1 / (r (r + z)) with r = sqrt(2 + z^2) on two, for z > 0. The tail
        // is I_x(nu / 2, 1/2) / 2 at x = nu / (nu + z^2), whose rounding moves
        // the exponent by (a + b) eps.
        for z in [1e-3_f64, 0.3, 1.0, 2.0, 7.5, 40.0, 1e3] {
            for (nu, one) in [(1.0_f64, true), (2.0, false)] {
                let want = if one {
                    (1.0 / z).atan() * std::f64::consts::FRAC_1_PI
                } else {
                    let r = (2.0 + z * z).sqrt();
                    1.0 / (r * (r + z))
                };
                let q = nu + z * z;
                let (a, b, x, y) = (0.5 * nu, 0.5, nu / q, z * z / q);
                let bound = direct_bound(a, b, x, y) + (a + b + 4.0) * EPS;
                let got = t::sf(z, nu);
                // Above the switching point the tail is (1 - I_y(b, a)) / 2.
                let ok = if x < (a + 1.0) / (a + b + 2.0) {
                    ((got - want) / want).abs() <= bound
                } else {
                    (got - want).abs() <= bound * (1.0 - 2.0 * want) / 2.0 + 2.0 * EPS * want
                };
                assert!(ok, "sf({z}, {nu}) = {got:e}, closed form {want:e}");
                assert!((t::cdf(-z, nu) - got).abs() <= EPS * got);
            }
        }
    }

    #[test]
    fn the_two_tails_sum_to_one() {
        for (a, b) in [
            (1e-3, 2.0),
            (0.3, 0.7),
            (2.5, 8.0),
            (12.0, 40.0),
            (300.0, 0.5),
        ] {
            for x in XS {
                let y = 1.0 - x;
                let (lower, upper) = (regularised(a, b, x, y), regularised(b, a, y, x));
                let direct = if x < (a + 1.0) / (a + b + 2.0) {
                    lower
                } else {
                    upper
                };
                let bound = direct_bound(a, b, x, y) * direct + 2.0 * EPS;
                assert!((lower + upper - 1.0).abs() <= bound, "{a} {b} {x}");
            }
        }
    }

    #[test]
    fn the_contiguous_relations_hold() {
        // DLMF 8.17.20, I_x(a, b) = x I_x(a - 1, b) + y I_x(a, b - 1), and
        // 8.17.21, (a + b) I_x(a, b) = a I_x(a + 1, b) + b I_x(a, b + 1). The
        // terms on the right are positive, so their sum carries the largest of
        // their relative bounds and a rounding; every tail here is below its
        // switching point and computed directly.
        for (a, b) in [
            (1.7, 2.2),
            (3.7, 5.2),
            (6.5, 1.5),
            (8.0, 9.0),
            (25.5, 30.25),
        ] {
            let switch = (a + 1.0) / (a + b + 3.0);
            for x in [1e-4 * switch, 0.1 * switch, 0.5 * switch, 0.9 * switch] {
                let y = 1.0 - x;
                let i = |a, b| regularised(a, b, x, y);
                let bound = |pairs: &[(f64, f64)]| {
                    pairs
                        .iter()
                        .map(|&(a, b)| direct_bound(a, b, x, y))
                        .fold(0.0, f64::max)
                        + 4.0 * EPS
                };
                let lowered = x * i(a - 1.0, b) + y * i(a, b - 1.0);
                let allowed = bound(&[(a, b), (a - 1.0, b), (a, b - 1.0)]);
                assert!(
                    ((lowered - i(a, b)) / i(a, b)).abs() <= allowed,
                    "8.17.20 at {a} {b} {x}"
                );
                let raised = a * i(a + 1.0, b) + b * i(a, b + 1.0);
                let allowed = bound(&[(a, b), (a + 1.0, b), (a, b + 1.0)]);
                let want = (a + b) * i(a, b);
                assert!(
                    ((raised - want) / want).abs() <= allowed,
                    "8.17.21 at {a} {b} {x}"
                );
            }
        }
    }

    #[test]
    fn edge_values_and_the_domain() {
        assert_eq!(regularised(2.0, 3.0, 0.0, 1.0), 0.0);
        assert_eq!(regularised(2.0, 3.0, 1.0, 0.0), 1.0);
        assert!(regularised(0.0, 3.0, 0.5, 0.5).is_nan());
        assert!(regularised(2.0, f64::INFINITY, 0.5, 0.5).is_nan());
        assert!(regularised(2.0, 3.0, f64::NAN, 0.5).is_nan());
        assert!(regularised(2.0, 3.0, 0.5, f64::NAN).is_nan());
    }

    #[test]
    fn the_term_limit_is_not_reached_at_the_switching_point() {
        // The fraction takes the most terms at x = (a + 1) / (a + b + 2); it
        // converges there over the whole range of shapes.
        let shapes = [1e-3, 0.003, 0.1, 1.0, 30.0, 100.0, 1e4, 1e6, 1e7];
        for a in shapes {
            for b in shapes {
                let x = (a + 1.0) / (a + b + 2.0);
                assert!(beta_fraction(a, b, x, 1.0 - x).is_some(), "{a} {b}");
            }
        }
    }
}
