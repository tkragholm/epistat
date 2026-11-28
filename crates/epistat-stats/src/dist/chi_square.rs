//! The chi-square distribution.
//!
//! With `k` degrees of freedom, `X` is gamma with shape `k/2` and scale 2, so
//!
//! ```text
//! P(X <= x) = P(k/2, x/2),    P(X > x) = Q(k/2, x/2),    q_X(p) = 2 q_G(p, k/2),
//! ```
//!
//! with `P` and `Q` the regularised incomplete gamma functions and `q_G` the
//! quantile of the gamma distribution with scale 1, all from [`super::gamma`],
//! whose documentation gives the methods. The survival function is computed
//! directly, so a small p-value keeps its relative precision. The functions
//! agree with R's `pchisq` and `qchisq` to about `1e-13` relative.

use super::gamma;

/// `P(X <= x)` for `X` chi-square with `df` degrees of freedom, R's
/// `pchisq(x, df)`. 0 for `x <= 0`; NaN for `df` not positive or `x` NaN.
#[must_use]
pub fn cdf(x: f64, df: f64) -> f64 {
    if x <= 0.0 && df > 0.0 {
        return 0.0;
    }
    gamma::lower_regularised(df / 2.0, x / 2.0)
}

/// `P(X > x)` for `X` chi-square with `df` degrees of freedom, R's
/// `pchisq(x, df, lower.tail = FALSE)`, computed directly so that small
/// p-values keep their relative precision. 1 for `x <= 0`; NaN as for [`cdf`].
#[must_use]
pub fn sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 && df > 0.0 {
        return 1.0;
    }
    gamma::upper_regularised(df / 2.0, x / 2.0)
}

/// The chi-square density at `x` with `df` degrees of freedom, R's
/// `dchisq(x, df)`: half the gamma density of shape `df / 2` at `x / 2`. 0
/// for `x < 0`; NaN for `df` not positive or `x` NaN.
#[must_use]
pub fn pdf(x: f64, df: f64) -> f64 {
    0.5 * gamma::pdf(0.5 * x, 0.5 * df)
}

/// The log of [`pdf`].
#[must_use]
pub fn ln_pdf(x: f64, df: f64) -> f64 {
    gamma::ln_pdf(0.5 * x, 0.5 * df) - std::f64::consts::LN_2
}

/// The upper-tail quantile, the `x` with `P(X > x) = p`, R's
/// `qchisq(p, df, lower.tail = FALSE)`: twice [`gamma::isf`] of shape
/// `df / 2`, so a small `p` keeps its precision. Infinity at `p = 0` and 0
/// at `p = 1`; NaN as for [`quantile`].
#[must_use]
pub fn isf(p: f64, df: f64) -> f64 {
    2.0 * gamma::isf(p, df / 2.0)
}

/// The `p` quantile of the chi-square distribution with `df` degrees of
/// freedom, R's `qchisq(p, df)`: twice the gamma quantile of shape `df / 2`.
/// 0 at `p = 0` and infinity at `p = 1`; NaN when `df` is not positive and
/// finite or `p` is outside `[0, 1]`.
#[must_use]
pub fn quantile(p: f64, df: f64) -> f64 {
    quantile_or_none(p, df).unwrap_or(f64::NAN)
}

/// [`quantile`], `None` where it is NaN.
pub(crate) fn quantile_or_none(p: f64, df: f64) -> Option<f64> {
    gamma::quantile_or_none(p, df / 2.0).map(|q| 2.0 * q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn edges() {
        assert_eq!(cdf(0.0, 3.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(sf(0.0, 3.0).to_bits(), 1.0f64.to_bits());
        assert_eq!(sf(f64::INFINITY, 3.0).to_bits(), 0.0f64.to_bits());
        assert!(quantile(0.5, -1.0).is_nan());
        assert_eq!(quantile(1.0, 2.0), f64::INFINITY);
    }

    /// With 2 degrees of freedom the chi-square is exponential with mean 2:
    /// `q(p) = -2 ln(1 - p)`. The gamma quantile is within `1e-12` relative
    /// of that, as its own test shows.
    #[test]
    fn two_degrees_of_freedom_are_exponential() {
        for p in [0.01_f64, 0.3, 0.5, 0.95, 0.999] {
            let want = -2.0 * (-p).ln_1p();
            let got = quantile(p, 2.0);
            assert!((got - want).abs() <= 1e-12 * want, "{p}: {got}");
            assert!((cdf(got, 2.0) - p).abs() < 1e-12);
        }
        // qchisq(0.95, 1) in R.
        assert!((quantile(0.95, 1.0) - 3.841_458_820_694_124).abs() < 1e-12);
    }
}
