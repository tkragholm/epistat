//! The gamma and chi-square functions at their paths before 0.3.0.
//!
//! The functions are in [`crate::dist::gamma`] and [`crate::dist::chi_square`].
//! These forwarders give the same bits; they are removed in 0.4.0.

use crate::dist;

/// `ln |Gamma(x)|`.
#[deprecated(since = "0.3.0", note = "use `epistat_stats::dist::gamma::ln_gamma`")]
#[must_use]
pub fn ln_gamma(x: f64) -> f64 {
    dist::gamma::ln_gamma(x)
}

/// `P(a, x)`, the regularised lower incomplete gamma function.
#[deprecated(since = "0.3.0", note = "use `epistat_stats::dist::gamma::cdf(x, a)`")]
#[must_use]
pub fn lower_regularised(a: f64, x: f64) -> f64 {
    dist::gamma::cdf(x, a)
}

/// `Q(a, x)`, the regularised upper incomplete gamma function.
#[deprecated(since = "0.3.0", note = "use `epistat_stats::dist::gamma::sf(x, a)`")]
#[must_use]
pub fn upper_regularised(a: f64, x: f64) -> f64 {
    dist::gamma::sf(x, a)
}

/// `P(X <= x)` for `X` chi-square with `df` degrees of freedom.
#[deprecated(since = "0.3.0", note = "use `epistat_stats::dist::chi_square::cdf`")]
#[must_use]
pub fn chi_square_cdf(x: f64, df: f64) -> f64 {
    dist::chi_square::cdf(x, df)
}

/// `P(X > x)` for `X` chi-square with `df` degrees of freedom.
#[deprecated(since = "0.3.0", note = "use `epistat_stats::dist::chi_square::sf`")]
#[must_use]
pub fn chi_square_sf(x: f64, df: f64) -> f64 {
    dist::chi_square::sf(x, df)
}
