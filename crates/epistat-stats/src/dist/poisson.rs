//! The Poisson distribution and confidence intervals for a Poisson count or
//! rate.
//!
//! [`pmf`], [`cdf`] and [`sf`] are R's `dpois` and `ppois`, from
//! the crate's private `discrete` module: the point probability by Loader's saddle-point form
//! and the tails from the regularised incomplete gamma function,
//! `P(X <= x) = Q(x + 1, m)`. [`cipoisson`] is `survival::cipoisson`.
//!
//! # Algorithm
//!
//! For an observed count `k` and a level `1 - 2a` the exact interval of
//! Garwood \[1\] has the limits
//!
//! ```text
//! lower = Q_gamma(a, k)       0 when k = 0
//! upper = Q_gamma(1 - a, k + 1)
//! ```
//!
//! where `Q_gamma(p, s)` is the `p` quantile of a gamma distribution with shape
//! `s` and scale 1. These are the limits at which the probability of a count of
//! `k` or more (lower) and `k` or fewer (upper) is `a`. Dividing by the
//! person-time `t` gives an interval for the rate. The interval of Anscombe,
//! given by Breslow and Day \[2\], is
//!
//! ```text
//! lower = (sqrt(k - 1/8) + z_a / 2)^2
//! upper = (sqrt(k + 7/8) - z_a / 2)^2
//! ```
//!
//! with `z_a` the `a` quantile of the standard normal distribution, which is
//! negative for `a < 1/2`. The lower limit is undefined below `k = 1/8`, so
//! it is `None` there. The interval of Byar, also in \[2\], is
//!
//! ```text
//! lower = k (1 - 1/(9k) - z / (3 sqrt(k)))^3        0 for k = 0
//! upper = (k+1) (1 - 1/(9(k+1)) + z / (3 sqrt(k+1)))^3
//! ```
//!
//! with `z = -z_a` the upper `a` quantile. It is the Wilson-Hilferty
//! approximation of the exact limits. The lower limit is `None` when the base of
//! the cube is not positive, which happens for very small counts at levels
//! above 99.9%.
//!
//! # Accuracy
//!
//! The exact limits are as accurate as the gamma quantile of
//! [`super::gamma`]. The tests compare with `survival::cipoisson` and state
//! the deviation found.
//!
//! # References
//!
//! 1. Garwood F. Fiducial limits for the Poisson distribution. Biometrika
//!    1936;28:437-442.
//! 2. Breslow NE, Day NE. Statistical Methods in Cancer Research, volume II.
//!    IARC 1987, section 2.5.

use epistat_core::{InputError, Interval, Level};

use super::discrete::{
    LOWER_FUZZ, UPPER_FUZZ, cornish_fisher, dpois, ln_dpois, ppois_lower, ppois_upper,
    smallest_count,
};
use super::{gamma, normal};
use crate::Error;

/// How the limits of a Poisson interval are found.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// The exact limits of Garwood, from gamma quantiles.
    Exact,
    /// The limits of Anscombe, from the normal quantile.
    Anscombe,
    /// The limits of Byar, the cube-root approximation of the exact limits.
    Byar,
}

/// `P(X = x)` for `X` Poisson with mean `lambda`, R's `dpois(x, lambda)`.
///
/// 0 for an `x` that is negative or not an integer, as in R. NaN when `x` or
/// `lambda` is NaN or `lambda` is negative or infinite.
#[must_use]
pub fn pmf(x: f64, lambda: f64) -> f64 {
    if x.is_nan() || !(lambda >= 0.0 && lambda.is_finite()) {
        return f64::NAN;
    }
    match super::discrete::support_point(x) {
        Some(k) => dpois(k, lambda),
        None => 0.0,
    }
}

/// `P(X <= x)` for `X` Poisson with mean `lambda`, R's `ppois(x, lambda)`.
///
/// `x` is taken down to an integer as R takes it, `floor(x + 1e-7)`. NaN as
/// for [`pmf`].
#[must_use]
pub fn cdf(x: f64, lambda: f64) -> f64 {
    if x.is_nan() || !(lambda >= 0.0 && lambda.is_finite()) {
        return f64::NAN;
    }
    let k = super::discrete::floor_count(x);
    if lambda == 0.0 {
        return f64::from(k >= 0.0);
    }
    ppois_lower(k, lambda)
}

/// `P(X > x)` for `X` Poisson with mean `lambda`, R's
/// `ppois(x, lambda, lower.tail = FALSE)`, computed directly so that a small
/// tail keeps its relative precision. NaN as for [`pmf`].
#[must_use]
pub fn sf(x: f64, lambda: f64) -> f64 {
    if x.is_nan() || !(lambda >= 0.0 && lambda.is_finite()) {
        return f64::NAN;
    }
    let k = super::discrete::floor_count(x);
    if lambda == 0.0 {
        return f64::from(k < 0.0);
    }
    ppois_upper(k, lambda)
}

/// `ln P(X = x)`, R's `dpois(x, lambda, log = TRUE)`, from the terms of
/// [`pmf`] without the exponential, so it is finite where [`pmf`]
/// underflows. `-inf` where [`pmf`] is 0; NaN as for [`pmf`].
#[must_use]
pub fn ln_pmf(x: f64, lambda: f64) -> f64 {
    if x.is_nan() || !(lambda >= 0.0 && lambda.is_finite()) {
        return f64::NAN;
    }
    match super::discrete::support_point(x) {
        Some(k) => ln_dpois(k, lambda),
        None => f64::NEG_INFINITY,
    }
}

/// Whether `p` and `lambda` are a probability and a Poisson mean.
fn valid(p: f64, lambda: f64) -> bool {
    (0.0..=1.0).contains(&p) && lambda >= 0.0 && lambda.is_finite()
}

/// The `p` quantile, R's `qpois(p, lambda)`: the smallest count `k` with
/// `P(X <= k) >= p`, compared after lowering `p` by a relative 12 eps, so
/// that a `p` computed as `P(X <= k)` up to rounding gives `k`, as in R. The
/// search starts at the Cornish-Fisher approximation and moves by doubling
/// steps and then bisection, so it takes a few dozen evaluations of the
/// tail at most. 0 at `p = 0` and infinity at `p = 1`; NaN for `p` outside
/// `[0, 1]` or `lambda` as for [`pmf`].
#[must_use]
pub fn quantile(p: f64, lambda: f64) -> f64 {
    if !valid(p, lambda) {
        return f64::NAN;
    }
    if lambda == 0.0 || p == 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    let target = p * (1.0 - LOWER_FUZZ);
    let sd = lambda.sqrt();
    let start = cornish_fisher(lambda, sd, 1.0 / sd, normal::quantile(p));
    smallest_count(start, f64::INFINITY, |k| ppois_lower(k, lambda) >= target)
}

/// The upper-tail quantile, R's `qpois(p, lambda, lower.tail = FALSE)`: the
/// smallest count `k` with `P(X > k) <= p`, compared after raising `p` by a
/// relative 6 eps, from the upper tail computed directly, so a small `p`
/// keeps its precision. Infinity at `p = 0` and 0 at `p = 1`; NaN as for
/// [`quantile`].
#[must_use]
pub fn isf(p: f64, lambda: f64) -> f64 {
    if !valid(p, lambda) {
        return f64::NAN;
    }
    if lambda == 0.0 || p >= 1.0 {
        return 0.0;
    }
    if p == 0.0 {
        return f64::INFINITY;
    }
    let target = p * (1.0 + UPPER_FUZZ);
    let sd = lambda.sqrt();
    let start = cornish_fisher(lambda, sd, 1.0 / sd, normal::isf(p));
    smallest_count(start, f64::INFINITY, |k| ppois_upper(k, lambda) <= target)
}

/// The confidence interval of a Poisson count `count` divided by `time`, as
/// R's `survival::cipoisson`.
///
/// The count may be fractional, as when it is a sum of weights. With a
/// `time` of 0 or less the rate has no meaning and both limits are `None`.
/// Pass 1 for the interval of the count itself.
///
/// # Errors
///
/// [`Error::Input`] when the count is negative or not finite or the time is
/// not finite.
pub fn cipoisson(count: f64, time: f64, level: Level, method: Method) -> Result<Interval, Error> {
    if !(count.is_finite() && count >= 0.0) {
        return Err(InputError::Domain {
            what: "count",
            value: count,
            domain: "[0, inf)",
            row: None,
        }
        .into());
    }
    if !time.is_finite() {
        return Err(InputError::NotFinite {
            what: "time",
            row: 0,
            column: None,
        }
        .into());
    }
    if time <= 0.0 {
        return Ok(Interval::new(None, None, level));
    }
    let a = level.lower_probability();
    let (lower, upper) = match method {
        Method::Exact => {
            let lower = if count == 0.0 {
                Some(0.0)
            } else {
                gamma::quantile_or_none(a, count)
            };
            (lower, gamma::quantile_or_none(1.0 - a, count + 1.0))
        }
        Method::Anscombe => {
            let half_z = normal::quantile(a) / 2.0;
            let lower = (count >= 0.125).then(|| {
                let root = (count - 0.125).sqrt() + half_z;
                root * root
            });
            let root = (count + 0.875).sqrt() - half_z;
            (lower, Some(root * root))
        }
        Method::Byar => {
            let z = -normal::quantile(a);
            let lower = if count == 0.0 {
                Some(0.0)
            } else {
                let base = 1.0 - 1.0 / (9.0 * count) - z / (3.0 * count.sqrt());
                (base > 0.0).then_some(count * base * base * base)
            };
            let up = count + 1.0;
            let base = 1.0 - 1.0 / (9.0 * up) + z / (3.0 * up.sqrt());
            (lower, Some(up * base * base * base))
        }
    };
    Ok(Interval::new(
        lower.map(|x| x / time),
        upper.map(|x| x / time),
        level,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_count_has_a_lower_limit_of_zero() {
        let ci = cipoisson(0.0, 2.0, Level::default(), Method::Exact).unwrap();
        assert_eq!(ci.lower, Some(0.0));
        let upper = ci.upper.unwrap();
        // -ln(0.025) / 2, from the exponential distribution.
        assert!((upper - (-(0.025_f64).ln()) / 2.0).abs() < 1e-12);
        assert_eq!(
            cipoisson(0.0, 1.0, Level::default(), Method::Anscombe)
                .unwrap()
                .lower,
            None
        );
    }

    #[test]
    fn refuses_bad_input_and_empty_time() {
        let level = Level::default();
        assert!(matches!(
            cipoisson(-1.0, 1.0, level, Method::Exact),
            Err(Error::Input(InputError::Domain { what: "count", .. }))
        ));
        assert!(matches!(
            cipoisson(1.0, f64::NAN, level, Method::Exact),
            Err(Error::Input(InputError::NotFinite { what: "time", .. }))
        ));
        let none = cipoisson(3.0, 0.0, level, Method::Exact).unwrap();
        assert_eq!((none.lower, none.upper), (None, None));
    }

    #[test]
    fn the_distribution_functions_follow_r() {
        // dpois(3, 2) and ppois(2, 2) in closed form.
        assert!((pmf(3.0, 2.0) - 8.0 * (-2.0_f64).exp() / 6.0).abs() < 1e-15);
        assert!((cdf(2.0, 2.0) - 5.0 * (-2.0_f64).exp()).abs() < 1e-14);
        assert_eq!(cdf(2.5, 2.0).to_bits(), cdf(2.0, 2.0).to_bits());
        assert!((cdf(4.0, 3.0) + sf(4.0, 3.0) - 1.0).abs() < 1e-15);
        assert_eq!(pmf(2.5, 2.0), 0.0);
        assert_eq!(pmf(-1.0, 2.0), 0.0);
        assert_eq!(cdf(-1.0, 2.0), 0.0);
        assert_eq!(sf(-1.0, 2.0), 1.0);
        assert_eq!(pmf(0.0, 0.0), 1.0);
        assert_eq!(cdf(0.0, 0.0), 1.0);
        assert_eq!(sf(0.0, 0.0), 0.0);
        assert!(pmf(1.0, -1.0).is_nan());
        assert!(cdf(f64::NAN, 1.0).is_nan());
        assert!(sf(1.0, f64::INFINITY).is_nan());
    }
}
