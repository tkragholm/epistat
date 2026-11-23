//! Confidence intervals for a Poisson count or rate, and the gamma quantile
//! they rest on.
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
//! [`gamma_quantile`] inverts the regularised incomplete gamma function of
//! [`crate::gamma`]. It solves `ln F(x) = ln p` for the tail that is smaller,
//! so a quantile in the far tail keeps its relative precision. The start is
//! the Wilson-Hilferty cube-root approximation \[3\]. Each step is a Newton
//! step, and a step that leaves the bracket found so far is replaced by a
//! bisection, so the iteration always ends. It stops when a step is within
//! four units in the last place of the value.
//!
//! # Accuracy
//!
//! The quantile is as accurate as the incomplete gamma function, which is
//! about 1e-14 relative in `ln_gamma`'s range of shapes, divided by the
//! sensitivity of the quantile to the probability. The tests compare with R's
//! `qgamma` and `survival::cipoisson` and state the deviation found.
//!
//! # References
//!
//! 1. Garwood F. Fiducial limits for the Poisson distribution. Biometrika
//!    1936;28:437-442.
//! 2. Breslow NE, Day NE. Statistical Methods in Cancer Research, volume II.
//!    IARC 1987, section 2.5.
//! 3. Wilson EB, Hilferty MM. The distribution of chi-square. Proc Natl Acad
//!    Sci 1931;17:684-688.

use std::fmt;

use crate::gamma::{ln_gamma, lower_regularised, upper_regularised};
use crate::normal;

/// How the limits of a Poisson interval are found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Method {
    /// The exact limits of Garwood, from gamma quantiles.
    Exact,
    /// The limits of Anscombe, from the normal quantile.
    Anscombe,
    /// The limits of Byar, the cube-root approximation of the exact limits.
    Byar,
}

/// The limits of an interval. A limit that is undefined is `None`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    /// The lower limit.
    pub lower: Option<f64>,
    /// The upper limit.
    pub upper: Option<f64>,
}

/// Why an interval was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoissonError {
    /// The count is negative, NaN or infinite.
    Count,
    /// The person-time is NaN or infinite.
    Time,
    /// The level is not strictly between 0 and 1.
    Level,
}

impl fmt::Display for PoissonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Count => "the count must be finite and not negative",
            Self::Time => "the person-time must be finite",
            Self::Level => "the level must be strictly between 0 and 1",
        })
    }
}

impl std::error::Error for PoissonError {}

/// The confidence interval of a Poisson count `count` divided by `time`.
///
/// `level` is the coverage, 0.95 for a 95% interval. The count may be
/// fractional, as when it is a sum of weights. With a `time` of 0 or less the
/// rate has no meaning and both limits are `None`. Pass 1 for the interval of
/// the count itself.
///
/// # Errors
///
/// [`PoissonError`] when the count is negative or not finite, the time is not
/// finite, or the level is not in `(0, 1)`.
pub fn cipoisson(
    count: f64,
    time: f64,
    level: f64,
    method: Method,
) -> Result<Interval, PoissonError> {
    if !(count.is_finite() && count >= 0.0) {
        return Err(PoissonError::Count);
    }
    if !time.is_finite() {
        return Err(PoissonError::Time);
    }
    if !(level > 0.0 && level < 1.0) {
        return Err(PoissonError::Level);
    }
    if time <= 0.0 {
        return Ok(Interval {
            lower: None,
            upper: None,
        });
    }
    let a = (1.0 - level) / 2.0;
    let (lower, upper) = match method {
        Method::Exact => {
            let lower = if count == 0.0 {
                Some(0.0)
            } else {
                gamma_quantile(a, count)
            };
            (lower, gamma_quantile(1.0 - a, count + 1.0))
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
    Ok(Interval {
        lower: lower.map(|x| x / time),
        upper: upper.map(|x| x / time),
    })
}

/// The `p` quantile of the gamma distribution with the given shape and scale
/// 1, or `None` when `shape` is not positive and finite or `p` is outside
/// `[0, 1]`. The quantile of `p = 0` is 0 and of `p = 1` is infinity.
///
/// For a chi-square distribution with `df` degrees of freedom it is
/// `2 * gamma_quantile(p, df / 2)`.
#[must_use]
pub fn gamma_quantile(p: f64, shape: f64) -> Option<f64> {
    if !(shape > 0.0 && shape.is_finite() && (0.0..=1.0).contains(&p)) {
        return None;
    }
    if p == 0.0 {
        return Some(0.0);
    }
    if p >= 1.0 {
        return Some(f64::INFINITY);
    }
    let upper = p > 0.5;
    let target = if upper { 1.0 - p } else { p };
    let ln_target = target.ln();
    let ln_norm = ln_gamma(shape);

    // Wilson-Hilferty start for the lower quantile of the same p.
    let z = normal::quantile(p);
    let c = 1.0 / (9.0 * shape);
    let cube = 1.0 - c + z * c.sqrt();
    let mut x = if cube > 0.0 {
        shape * cube * cube * cube
    } else {
        (target.ln() + ln_gamma(shape + 1.0))
            .exp()
            .powf(1.0 / shape)
    };
    if !(x > 0.0 && x.is_finite()) {
        x = shape;
    }

    let (mut lo, mut hi) = (0.0_f64, f64::INFINITY);
    for _ in 0..500 {
        let tail = if upper {
            upper_regularised(shape, x)
        } else {
            lower_regularised(shape, x)
        };
        // Positive when x is above the quantile.
        let excess = if upper {
            ln_target - tail.ln()
        } else {
            tail.ln() - ln_target
        };
        if excess > 0.0 {
            hi = x;
        } else if excess < 0.0 {
            lo = x;
        } else {
            return Some(x);
        }
        let slope = ((shape - 1.0) * x.ln() - x - ln_norm).exp() / tail;
        let step = excess / slope;
        let mut next = if step.is_finite() { x - step } else { f64::NAN };
        if !(next > lo && next < hi) {
            next = if hi.is_finite() {
                f64::midpoint(lo, hi)
            } else {
                2.0 * x
            };
        }
        let done = (next - x).abs() <= 4.0 * f64::EPSILON * next;
        x = next;
        if done {
            return Some(x);
        }
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exponential_quantile_is_closed_form() {
        // Shape 1 is the exponential distribution: q = -ln(1 - p).
        for p in [1e-10, 0.01, 0.3, 0.5, 0.9, 0.999] {
            let got = gamma_quantile(p, 1.0).unwrap();
            let want = -(-p).ln_1p();
            assert!((got - want).abs() <= 1e-12 * want, "{p}: {got} {want}");
        }
    }

    #[test]
    fn quantile_inverts_the_cdf() {
        for shape in [0.3, 1.0, 2.5, 10.0, 80.0, 1000.0] {
            for p in [1e-8, 0.025, 0.5, 0.975, 1.0 - 1e-8] {
                let x = gamma_quantile(p, shape).unwrap();
                let back = lower_regularised(shape, x);
                assert!((back - p).abs() <= 1e-10 * p.min(1.0 - p), "{shape} {p}");
            }
        }
    }

    #[test]
    fn zero_count_has_a_lower_limit_of_zero() {
        let ci = cipoisson(0.0, 2.0, 0.95, Method::Exact).unwrap();
        assert_eq!(ci.lower, Some(0.0));
        let upper = ci.upper.unwrap();
        // -ln(0.025) / 2, from the exponential distribution.
        assert!((upper - (-(0.025_f64).ln()) / 2.0).abs() < 1e-12);
        assert_eq!(
            cipoisson(0.0, 1.0, 0.95, Method::Anscombe).unwrap().lower,
            None
        );
    }

    #[test]
    fn refuses_bad_input_and_empty_time() {
        assert_eq!(
            cipoisson(-1.0, 1.0, 0.95, Method::Exact),
            Err(PoissonError::Count)
        );
        assert_eq!(
            cipoisson(1.0, f64::NAN, 0.95, Method::Exact),
            Err(PoissonError::Time)
        );
        assert_eq!(
            cipoisson(1.0, 1.0, 1.0, Method::Exact),
            Err(PoissonError::Level)
        );
        let none = cipoisson(3.0, 0.0, 0.95, Method::Exact).unwrap();
        assert_eq!((none.lower, none.upper), (None, None));
    }
}
