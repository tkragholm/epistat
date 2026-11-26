//! The normal distribution functions at their paths before 0.3.0.
//!
//! The functions are in [`crate::dist::normal`] and the approximations in
//! [`crate::compat`]. These forwarders give the same bits and the signatures
//! of 0.2; they are removed in 0.4.0.

use crate::{compat, dist};

/// The standard normal density.
#[deprecated(since = "0.3.0", note = "use `epistat_stats::dist::normal::pdf`")]
#[must_use]
pub fn pdf(x: f64) -> f64 {
    dist::normal::pdf(x)
}

/// `P(Z <= x)` for a standard normal `Z`.
#[deprecated(since = "0.3.0", note = "use `epistat_stats::dist::normal::cdf`")]
#[must_use]
pub fn cdf(x: f64) -> f64 {
    dist::normal::cdf(x)
}

/// The `p` quantile of the standard normal.
#[deprecated(since = "0.3.0", note = "use `epistat_stats::dist::normal::quantile`")]
#[must_use]
pub fn quantile(p: f64) -> f64 {
    dist::normal::quantile(p)
}

/// The `p` quantile of a standard normal truncated to `[lower, upper]`, by
/// inverting the distribution function alone; `None` where that underflows.
#[deprecated(
    since = "0.3.0",
    note = "use `epistat_stats::dist::normal::truncated_quantile`, NaN where this is None and defined in the far tails"
)]
#[must_use]
pub fn truncated_quantile(p: f64, lower: f64, upper: f64) -> Option<f64> {
    dist::normal::truncated_quantile_central(p, lower, upper)
}

/// The `1 - (1 - level) / 2` quantile of the standard normal.
#[deprecated(
    since = "0.3.0",
    note = "use `epistat_stats::dist::normal::two_sided_z` with a `Level`"
)]
#[must_use]
pub fn two_sided_z(level: f64) -> f64 {
    dist::normal::quantile(1.0 - (1.0 - level) / 2.0)
}

/// The `p` quantile of `|Z + mu|` for a standard normal `Z`.
#[deprecated(
    since = "0.3.0",
    note = "use `epistat_stats::dist::normal::folded_quantile`, NaN where this is None"
)]
#[must_use]
pub fn folded_quantile(p: f64, mu: f64) -> Option<f64> {
    let q = dist::normal::folded_quantile(p, mu);
    (!q.is_nan()).then_some(q)
}

/// The normal distribution function by Abramowitz and Stegun 7.1.26.
#[deprecated(since = "0.3.0", note = "use `epistat_stats::compat::cdf_as26`")]
#[must_use]
pub fn cdf_as26(x: f64) -> f64 {
    compat::cdf_as26(x)
}

/// The error function by Abramowitz and Stegun 7.1.26.
#[deprecated(since = "0.3.0", note = "use `epistat_stats::compat::erf_as26`")]
#[must_use]
pub fn erf_as26(x: f64) -> f64 {
    compat::erf_as26(x)
}

/// The normal quantile by Acklam's approximation.
#[deprecated(since = "0.3.0", note = "use `epistat_stats::compat::quantile_acklam`")]
#[must_use]
pub fn quantile_acklam(p: f64) -> f64 {
    compat::quantile_acklam(p)
}
