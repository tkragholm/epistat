//! The F distribution.
//!
//! The upper tail is a regularised incomplete beta function,
//!
//! ```text
//! P(F > f) = I_x(df2 / 2, df1 / 2),    x = df2 / (df2 + df1 f),
//! ```
//!
//! with `I_x` from [`super::beta`], and the quantile is formed from the beta
//! quantile as R forms it. Both functions say below how they treat very large
//! degrees of freedom, where the F distribution of `df1 F` tends to the
//! chi-square distribution on `df1` degrees of freedom.
//!
//! The functions agree with R's `pf` and `qf` to about `1e-13` relative, not
//! bit for bit.

use super::beta::{ln_power_factor, power_factor, regularised};
use super::gamma::ln_gamma;

/// The upper tail `P(F > f)` of the F distribution on `df1` and `df2` degrees
/// of freedom, R's `pf(f, df1, df2, lower.tail = FALSE)`, from the incomplete
/// beta function `I_x(df2 / 2, df1 / 2)` with `x = df2 / (df2 + df1 f)`, and from
/// the chi-square distribution of `df1 f` when `df2` is infinite. It is 1 for
/// `f <= 0`. NaN when a degree of freedom is not positive or the continued
/// fraction does not converge.
///
/// Both `x` and `y = 1 - x` are passed to the incomplete beta function, which
/// forms its factor `x^a y^b / B(a, b)` from the smaller of them, so that the
/// tail keeps its accuracy for the very large `df2` that multiple imputation
/// gives. For
/// two-sided t tails, `P(|T| > z)` on `nu` degrees of freedom is
/// `sf(z^2, 1, nu)`.
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
pub fn sf(f: f64, df1: f64, df2: f64) -> f64 {
    if !(df1 > 0.0 && df2 > 0.0) || f.is_nan() {
        return f64::NAN;
    }
    if f <= 0.0 {
        return 1.0;
    }
    if df2.is_infinite() {
        return super::chi_square::sf(df1 * f, df1);
    }
    let (a, b) = (0.5 * df2, 0.5 * df1);
    if df2 >= 1e7 {
        let c = 0.5 * df1 * f;
        let density = ((b - 1.0) * c.ln() - c - ln_gamma(b)).exp();
        return super::chi_square::sf(df1 * f, df1) + density * c * (c - b + 1.0) / df2;
    }
    let denominator = df2 + df1 * f;
    regularised(a, b, df2 / denominator, df1 * f / denominator)
}

/// The lower tail `P(F <= f)`, R's `pf(f, df1, df2)`: the incomplete beta
/// function `I_y(df1 / 2, df2 / 2)` with `y = df1 f / (df2 + df1 f)`, computed
/// directly so that a small lower tail keeps its precision, and the
/// chi-square distribution of `df1 f` when `df2` is infinite. 0 for
/// `f <= 0`; NaN as for [`sf`].
#[must_use]
pub fn cdf(f: f64, df1: f64, df2: f64) -> f64 {
    if !(df1 > 0.0 && df2 > 0.0) || f.is_nan() {
        return f64::NAN;
    }
    if f <= 0.0 {
        return 0.0;
    }
    if df2.is_infinite() {
        return super::chi_square::cdf(df1 * f, df1);
    }
    let denominator = df2 + df1 * f;
    regularised(
        0.5 * df1,
        0.5 * df2,
        df1 * f / denominator,
        df2 / denominator,
    )
}

/// The F density at `f`, R's `df(f, df1, df2)`. With
/// `y = df1 f / (df2 + df1 f)`, which is beta with shapes `df1 / 2` and
/// `df2 / 2`, the density is `y^(df1/2) (1 - y)^(df2/2) / (B(df1/2, df2/2) f)`,
/// from the factor of the incomplete beta function, and `df1` times the
/// chi-square density of `df1 f` when `df2` is infinite. At `f = 0` it is
/// infinite for `df1 < 2`, 1 for `df1 = 2` and 0 above; 0 for `f < 0`. NaN
/// when a degree of freedom is not positive, `df1` is infinite or `f` is
/// NaN.
#[must_use]
pub fn pdf(f: f64, df1: f64, df2: f64) -> f64 {
    if !(df1 > 0.0 && df2 > 0.0 && df1.is_finite()) || f.is_nan() {
        return f64::NAN;
    }
    if f < 0.0 || f.is_infinite() {
        return 0.0;
    }
    if f == 0.0 {
        return at_zero(df1);
    }
    if df2.is_infinite() {
        return df1 * super::chi_square::pdf(df1 * f, df1);
    }
    let denominator = df2 + df1 * f;
    power_factor(
        0.5 * df1,
        0.5 * df2,
        df1 * f / denominator,
        df2 / denominator,
        None,
    ) / f
}

/// The log of [`pdf`], from the same terms.
#[must_use]
pub fn ln_pdf(f: f64, df1: f64, df2: f64) -> f64 {
    if !(df1 > 0.0 && df2 > 0.0 && df1.is_finite()) || f.is_nan() {
        return f64::NAN;
    }
    if f < 0.0 || f.is_infinite() {
        return f64::NEG_INFINITY;
    }
    if f == 0.0 {
        return at_zero(df1).ln();
    }
    if df2.is_infinite() {
        return df1.ln() + super::chi_square::ln_pdf(df1 * f, df1);
    }
    let denominator = df2 + df1 * f;
    ln_power_factor(
        0.5 * df1,
        0.5 * df2,
        df1 * f / denominator,
        df2 / denominator,
    ) - f.ln()
}

/// The density at 0, which `f^(df1/2 - 1)` sets.
fn at_zero(df1: f64) -> f64 {
    match df1.partial_cmp(&2.0) {
        Some(core::cmp::Ordering::Less) => f64::INFINITY,
        Some(core::cmp::Ordering::Equal) => 1.0,
        _ => 0.0,
    }
}

/// The upper-tail quantile, the `f` with `P(F > f) = p`, R's
/// `qf(p, df1, df2, lower.tail = FALSE)`: [`quantile`] at `1 - p` without
/// forming `1 - p`, so a small `p` keeps its precision. With `y` the `p`
/// quantile of the beta distribution with shapes `df2 / 2` and `df1 / 2`, it
/// is `(1 / y - 1) df2 / df1`, and above `4e5` degrees of freedom the
/// chi-square forms of [`quantile`] at the other tail. Infinity at `p = 0`
/// and 0 at `p = 1`; NaN as for [`quantile`].
#[must_use]
pub fn isf(p: f64, df1: f64, df2: f64) -> f64 {
    isf_or_none(p, df1, df2).unwrap_or(f64::NAN)
}

/// [`isf`], `None` where it is NaN.
fn isf_or_none(p: f64, df1: f64, df2: f64) -> Option<f64> {
    if !(df1 > 0.0 && df2 > 0.0 && df1.is_finite() && df2.is_finite() && (0.0..=1.0).contains(&p)) {
        return None;
    }
    if p == 0.0 {
        return Some(f64::INFINITY);
    }
    if p >= 1.0 {
        return Some(0.0);
    }
    if df1 <= df2 && df2 > 4e5 {
        return super::gamma::isf_or_none(p, df1 / 2.0).map(|q| 2.0 * q / df1);
    }
    if df1 > 4e5 {
        return super::chi_square::quantile_or_none(p, df2).map(|q| df2 / q);
    }
    let y = super::beta::quantile_or_none(p, df2 / 2.0, df1 / 2.0)?;
    Some((1.0 / y - 1.0) * (df2 / df1))
}

/// The `p` quantile of the F distribution with `df1` and `df2` degrees of
/// freedom, R's `qf(p, df1, df2)`, formed as R forms it: with `y` the upper
/// `p` quantile of the beta distribution with shapes `df2 / 2` and
/// `df1 / 2` ([`super::beta::isf`], which takes `p` itself rather than
/// `1 - p`), `F = (1 / y - 1) df2 / df1`. Above `4e5` degrees of freedom the
/// F quantile is a chi-square one, `qchisq(p, df1) / df1` when `df2` is the
/// larger and `df2 / qchisq(p, df2, lower.tail = FALSE)` when `df1` is above
/// `4e5`, as in R.
/// NaN when a degree of freedom is not positive and finite, `p` is
/// outside `[0, 1]`, or the beta quantile is NaN. The quantile of `p = 0`
/// is 0 and of `p = 1` is infinity.
#[must_use]
pub fn quantile(p: f64, df1: f64, df2: f64) -> f64 {
    quantile_or_none(p, df1, df2).unwrap_or(f64::NAN)
}

/// [`quantile`], `None` where it is NaN.
pub(crate) fn quantile_or_none(p: f64, df1: f64, df2: f64) -> Option<f64> {
    if !(df1 > 0.0 && df2 > 0.0 && df1.is_finite() && df2.is_finite() && (0.0..=1.0).contains(&p)) {
        return None;
    }
    if p == 0.0 {
        return Some(0.0);
    }
    if p >= 1.0 {
        return Some(f64::INFINITY);
    }
    if df1 <= df2 && df2 > 4e5 {
        return super::chi_square::quantile_or_none(p, df1).map(|q| q / df1);
    }
    if df1 > 4e5 {
        return super::gamma::isf_or_none(p, df2 / 2.0).map(|q| df2 / (2.0 * q));
    }
    let y = super::beta::isf_or_none(p, df2 / 2.0, df1 / 2.0)?;
    Some((1.0 / y - 1.0) * (df2 / df1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_quantile_inverts_the_tail() {
        for (df1, df2) in [(1.0, 5.0), (3.0, 12.0), (10.0, 2.5)] {
            for p in [0.05, 0.5, 0.95] {
                let f = quantile(p, df1, df2);
                assert!(
                    (sf(f, df1, df2) - (1.0 - p)).abs() < 1e-13,
                    "{df1} {df2} {p}"
                );
            }
        }
        assert!(quantile(1.5, 1.0, 1.0).is_nan());
        assert!(quantile(0.5, 0.0, 1.0).is_nan());
        assert_eq!(quantile(0.0, 1.0, 1.0), 0.0);
        assert_eq!(quantile(1.0, 1.0, 1.0), f64::INFINITY);
        assert!(sf(1.0, -1.0, 1.0).is_nan());
        assert_eq!(sf(-1.0, 1.0, 1.0), 1.0);
    }
}
