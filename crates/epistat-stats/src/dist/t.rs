//! The Student t distribution.
//!
//! The tail is a regularised incomplete beta function. With `t` the variable,
//! `nu` the degrees of freedom and `x = nu / (nu + t^2)`,
//!
//! ```text
//! P(T > |t|) = (1/2) I_x(nu/2, 1/2),
//! ```
//!
//! with `I_x` from [`super::beta::regularised`], whose method the module
//! documentation of [`super::beta`] gives. The tail is therefore computed
//! directly and keeps its relative precision far out. The density is
//!
//! ```text
//! ln f(t) = ln Gamma((nu + 1) / 2) - ln Gamma(nu / 2) - ln(nu pi) / 2
//!           - (nu + 1) / 2 ln(1 + t^2 / nu).
//! ```
//!
//! The quantile is found by Newton's method on the tail inside a bracket that
//! is doubled until it holds the root and halved whenever a step leaves it.
//!
//! The functions agree with R's `dt`, `pt` and `qt` to about `1e-13`
//! relative, not bit for bit.
//!
//! # References
//!
//! 1. Johnson NL, Kotz S, Balakrishnan N. Continuous Univariate
//!    Distributions, volume 2, 2nd edition. Wiley 1995, chapter 28.

#![expect(
    clippy::float_cmp,
    reason = "the probabilities 0, 1/2 and 1 are compared exactly"
)]

use std::f64::consts::PI;

use super::beta::regularised;
use super::gamma::ln_gamma;

/// `ln f(z)` of the t density with `nu` degrees of freedom, R's
/// `dt(z, nu, log = TRUE)`.
#[must_use]
pub fn ln_pdf(z: f64, nu: f64) -> f64 {
    let constant = ln_gamma(f64::midpoint(nu, 1.0)) - ln_gamma(0.5 * nu) - 0.5 * (nu * PI).ln();
    constant - f64::midpoint(nu, 1.0) * (z * z / nu).ln_1p()
}

/// `P(T > |z|)`, half the two-sided tail.
fn upper_tail(z: f64, nu: f64) -> f64 {
    if z.is_infinite() {
        return 0.0;
    }
    let q = nu + z * z;
    0.5 * regularised(0.5 * nu, 0.5, nu / q, z * z / q)
}

/// `P(T <= z)` for `T` t-distributed with `nu` degrees of freedom, R's
/// `pt(z, nu)`. Below zero the tail is computed directly, so it keeps its
/// relative precision far out. NaN when `z` is NaN, `nu` is not positive or
/// the incomplete beta function does not converge.
#[must_use]
pub fn cdf(z: f64, nu: f64) -> f64 {
    cdf_sf(z, nu).0
}

/// `P(T > z)`, R's `pt(z, nu, lower.tail = FALSE)`, computed directly above
/// zero. NaN as for [`cdf`].
#[must_use]
pub fn sf(z: f64, nu: f64) -> f64 {
    cdf_sf(z, nu).1
}

/// The distribution function and the survival function at `z`. The smaller
/// of the two is computed directly.
pub(crate) fn cdf_sf(z: f64, nu: f64) -> (f64, f64) {
    if z.is_nan() || nu.is_nan() || nu <= 0.0 {
        return (f64::NAN, f64::NAN);
    }
    cdf_sf_unchecked(z, nu)
}

/// [`cdf_sf`] without the checks of its arguments.
pub(crate) fn cdf_sf_unchecked(z: f64, nu: f64) -> (f64, f64) {
    let tail = upper_tail(z.abs(), nu);
    if z > 0.0 {
        (1.0 - tail, tail)
    } else {
        (tail, 1.0 - tail)
    }
}

/// The quantile function of the t distribution, R's `qt(p, nu)`: the root of
/// the tail probability by Newton's method inside a bracket, which is halved
/// whenever a step leaves it. It stops when the step is below `4 eps |t|`.
/// `-inf` at 0 and `inf` at 1; NaN for `p` outside `[0, 1]`, `nu` not
/// positive, or a search for a bracket that passes `1e300`.
#[must_use]
pub fn quantile(p: f64, nu: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) || nu.is_nan() || nu <= 0.0 {
        return f64::NAN;
    }
    if p == 0.0 {
        return f64::NEG_INFINITY;
    }
    if p == 1.0 {
        return f64::INFINITY;
    }
    search(p, nu).unwrap_or(f64::NAN)
}

/// The Newton search of [`quantile`], `None` when the bracket passes `1e300`.
pub(crate) fn search(p: f64, nu: f64) -> Option<f64> {
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
        let density = ln_pdf(t, nu).exp();
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
            let (cdf, sf) = cdf_sf(z, 1.0);
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
                let z = quantile(p, nu);
                let (cdf, sf) = cdf_sf(z, nu);
                let (got, want) = if p < 0.5 { (cdf, p) } else { (sf, 1.0 - p) };
                assert!((got - want).abs() <= 1e-13 * want, "{nu} {p}: {z}");
            }
        }
    }

    #[test]
    fn the_quantile_is_infinite_at_the_ends_and_nan_outside() {
        assert_eq!(quantile(0.0, 3.0), f64::NEG_INFINITY);
        assert_eq!(quantile(1.0, 3.0), f64::INFINITY);
        assert_eq!(quantile(0.5, 3.0), 0.0);
        assert!(quantile(1.5, 3.0).is_nan());
        assert!(quantile(0.5, -1.0).is_nan());
        assert!(quantile(f64::NAN, 3.0).is_nan());
        assert!(cdf(1.0, 0.0).is_nan());
        assert!(sf(f64::NAN, 3.0).is_nan());
    }
}
