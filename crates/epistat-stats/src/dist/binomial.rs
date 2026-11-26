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

use super::discrete::{dbinom, floor_count, pbinom_lower, pbinom_upper, support_point};

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
