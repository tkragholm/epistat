//! The t distribution and the incomplete beta function at their paths before
//! 0.3.0.
//!
//! The functions are in [`crate::dist::t`] and [`crate::dist::beta`]. These
//! forwarders give the same bits and the signatures of 0.2; they are removed
//! in 0.4.0.

use crate::dist;

/// `ln f(z)` of the t density with `nu` degrees of freedom.
#[deprecated(since = "0.3.0", note = "use `epistat_stats::dist::t::ln_pdf`")]
#[must_use]
pub fn t_ln_density(z: f64, nu: f64) -> f64 {
    dist::t::ln_pdf(z, nu)
}

/// The regularised incomplete beta function `I_x(a, b)` with `y = 1 - x`.
#[deprecated(since = "0.3.0", note = "use `epistat_stats::dist::beta::regularised`")]
#[must_use]
pub fn regularised_beta(a: f64, b: f64, x: f64, y: f64) -> f64 {
    dist::beta::regularised(a, b, x, y)
}

/// The distribution function and the survival function of the t
/// distribution at `z`.
#[deprecated(
    since = "0.3.0",
    note = "use `epistat_stats::dist::t::cdf` and `epistat_stats::dist::t::sf`"
)]
#[must_use]
pub fn t_cdf_sf(z: f64, nu: f64) -> (f64, f64) {
    dist::t::cdf_sf_unchecked(z, nu)
}

/// The quantile function of the t distribution; `None` when the search for a
/// bracket passes `1e300`.
#[deprecated(
    since = "0.3.0",
    note = "use `epistat_stats::dist::t::quantile`, NaN where this is None"
)]
#[must_use]
pub fn t_quantile(p: f64, nu: f64) -> Option<f64> {
    dist::t::search(p, nu)
}
