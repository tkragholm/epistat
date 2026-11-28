//! The binomial distribution.
//!
//! [`pmf`], [`cdf`] and [`sf`] are R's `dbinom` and `pbinom`, from
//! the crate's private `discrete` module: the point probability by Loader's saddle-point form,
//!
//! ```text
//! P(X = x) = exp(stirlerr(n) - stirlerr(x) - stirlerr(n - x)
//!                - bd0(x, n p) - bd0(n - x, n q)) sqrt(n / (2 pi x (n - x))),
//! ```
//!
//! and the tails from the regularised incomplete beta function,
//! `P(X <= x) = I_{1 - p}(n - x, x + 1)`. They agree with R's to about `1e-14`
//! relative.
//!
//! # References
//!
//! 1. Loader C. Fast and accurate computation of binomial probabilities.
//!    2000. <https://www.r-project.org/doc/reports/CLoader-dbinom-2002.pdf>

#![expect(
    clippy::float_cmp,
    reason = "the probabilities 0 and 1 and integer counts are compared exactly"
)]

use super::discrete::{
    LOWER_FUZZ, UPPER_FUZZ, cornish_fisher, dbinom, floor_count, ln_dbinom, pbinom_lower,
    pbinom_upper, smallest_count, support_point,
};
use super::normal;

/// Whether `n` and `p` are a binomial distribution: `n` a non-negative
/// integer and `p` in `[0, 1]`.
fn valid(n: f64, p: f64) -> bool {
    n >= 0.0 && n.is_finite() && n.fract() == 0.0 && (0.0..=1.0).contains(&p)
}

/// `P(X = x)` for `X` binomial with `n` trials and probability `p`, R's
/// `dbinom(x, n, p)`.
///
/// 0 for an `x` outside `0..=n` or not an integer. NaN when an argument is
/// NaN, `n` is not a non-negative integer or `p` is outside `[0, 1]`.
#[must_use]
pub fn pmf(x: f64, n: f64, p: f64) -> f64 {
    if x.is_nan() || !valid(n, p) {
        return f64::NAN;
    }
    match support_point(x) {
        Some(k) if k <= n => dbinom(k, n, p),
        _ => 0.0,
    }
}

/// `ln P(X = x)`, R's `dbinom(x, n, p, log = TRUE)`, from the terms of
/// [`pmf`] without the exponential, so it is finite where [`pmf`]
/// underflows. `-inf` where [`pmf`] is 0; NaN as for [`pmf`].
#[must_use]
pub fn ln_pmf(x: f64, n: f64, p: f64) -> f64 {
    if x.is_nan() || !valid(n, p) {
        return f64::NAN;
    }
    match support_point(x) {
        Some(k) if k <= n => ln_dbinom(k, n, p),
        _ => f64::NEG_INFINITY,
    }
}

/// The `q` quantile, R's `qbinom(q, n, p)`: the smallest count `k` with
/// `P(X <= k) >= q`, compared after lowering `q` by a relative 12 eps, so
/// that a `q` computed as `P(X <= k)` up to rounding gives `k`, as in R. The
/// search starts at the Cornish-Fisher approximation and moves by doubling
/// steps and then bisection. 0 at `q = 0` and `n` at `q = 1`; NaN for `q`
/// outside `[0, 1]` or `n` and `p` as for [`pmf`].
#[must_use]
pub fn quantile(q: f64, n: f64, p: f64) -> f64 {
    if !(valid(n, p) && (0.0..=1.0).contains(&q)) {
        return f64::NAN;
    }
    if q == 0.0 || p == 0.0 {
        return 0.0;
    }
    if q == 1.0 || p == 1.0 {
        return n;
    }
    let target = q * (1.0 - LOWER_FUZZ);
    smallest_count(start(n, p, normal::quantile(q)), n, |k| {
        pbinom_lower(k, n, p) >= target
    })
}

/// The upper-tail quantile, R's `qbinom(q, n, p, lower.tail = FALSE)`: the
/// smallest count `k` with `P(X > k) <= q`, compared after raising `q` by a
/// relative 6 eps, from the upper tail computed directly. `n` at `q = 0` and
/// 0 at `q = 1`; NaN as for [`quantile`].
#[must_use]
pub fn isf(q: f64, n: f64, p: f64) -> f64 {
    if !(valid(n, p) && (0.0..=1.0).contains(&q)) {
        return f64::NAN;
    }
    if q == 1.0 || p == 0.0 {
        return 0.0;
    }
    if q == 0.0 || p == 1.0 {
        return n;
    }
    let target = q * (1.0 + UPPER_FUZZ);
    smallest_count(start(n, p, normal::isf(q)), n, |k| {
        pbinom_upper(k, n, p) <= target
    })
}

/// The Cornish-Fisher start of a search at the normal quantile `z`.
fn start(n: f64, p: f64, z: f64) -> f64 {
    let sd = (n * p * (1.0 - p)).sqrt();
    cornish_fisher(n * p, sd, (1.0 - 2.0 * p) / sd, z)
}

/// `P(X <= x)` for `X` binomial with `n` trials and probability `p`, R's
/// `pbinom(x, n, p)`, with `x` taken down to an integer as R takes it,
/// `floor(x + 1e-7)`. NaN as for [`pmf`].
#[must_use]
pub fn cdf(x: f64, n: f64, p: f64) -> f64 {
    if x.is_nan() || !valid(n, p) {
        return f64::NAN;
    }
    let k = floor_count(x);
    if p == 0.0 {
        return f64::from(k >= 0.0);
    }
    if p == 1.0 {
        return f64::from(k >= n);
    }
    pbinom_lower(k, n, p)
}

/// `P(X > x)`, R's `pbinom(x, n, p, lower.tail = FALSE)`, computed directly so
/// that a small tail keeps its relative precision. NaN as for [`pmf`].
#[must_use]
pub fn sf(x: f64, n: f64, p: f64) -> f64 {
    if x.is_nan() || !valid(n, p) {
        return f64::NAN;
    }
    let k = floor_count(x);
    if p == 0.0 {
        return f64::from(k < 0.0);
    }
    if p == 1.0 {
        return f64::from(k < n);
    }
    pbinom_upper(k, n, p)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_cases_are_closed_form() {
        assert!((pmf(2.0, 5.0, 0.3) - 10.0 * 0.09 * 0.343).abs() < 1e-15);
        let total: f64 = (0..=5).map(|i| pmf(f64::from(i), 5.0, 0.3)).sum();
        assert!((total - 1.0).abs() < 1e-14);
        assert!((cdf(2.0, 5.0, 0.3) + sf(2.0, 5.0, 0.3) - 1.0).abs() < 1e-14);
        let partial: f64 = (0..=2).map(|i| pmf(f64::from(i), 5.0, 0.3)).sum();
        // The tail comes from the incomplete beta function, the sum from the
        // point probabilities; each is within about 1e-14 relative of R's.
        assert!((cdf(2.0, 5.0, 0.3) - partial).abs() < 1e-14);
    }

    #[test]
    fn edges() {
        assert_eq!(pmf(6.0, 5.0, 0.3), 0.0);
        assert_eq!(pmf(1.5, 5.0, 0.3), 0.0);
        assert_eq!(cdf(-1.0, 5.0, 0.3), 0.0);
        assert_eq!(cdf(5.0, 5.0, 0.3), 1.0);
        assert_eq!(sf(5.0, 5.0, 0.3), 0.0);
        assert_eq!(cdf(0.0, 5.0, 0.0), 1.0);
        assert_eq!(sf(4.0, 5.0, 1.0), 1.0);
        assert_eq!(pmf(0.0, 0.0, 0.5), 1.0);
        assert!(pmf(1.0, 2.5, 0.3).is_nan());
        assert!(cdf(1.0, 5.0, 1.5).is_nan());
        assert!(sf(f64::NAN, 5.0, 0.3).is_nan());
    }
}
