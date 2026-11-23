//! Rates from events and person-time, their ratios and differences, and
//! directly standardised rates, each with an interval.
//!
//! # Algorithm
//!
//! A stratum with `d` events in person-time `y` has the rate `d / y`. Its
//! interval is the exact Poisson interval of [`crate::poisson`], divided by
//! `y` and multiplied by `scale`.
//!
//! A directly standardised rate over strata `s` with weights `w_s`, scaled to
//! sum to 1, is
//!
//! ```text
//! R = sum_s w_s d_s / y_s
//! V = sum_s w_s^2 d_s / y_s^2
//! ```
//!
//! with `V` the variance of `R` when the counts are Poisson. The interval is
//! the gamma interval of Fay and Feuer \[1\] in the form of Tiwari, Clegg and
//! Zou \[2\]. With `w_max = max_s (w_s / y_s)` and `qchisq(p, k)` the `p`
//! quantile of a chi-square distribution on `k` degrees of freedom, which is
//! `2 gamma_quantile(p, k / 2)`,
//!
//! ```text
//! lower = V / (2 R) qchisq(a, 2 R^2 / V)
//! upper = (V + w_max^2) / (2 (R + w_max))
//!         qchisq(1 - a, 2 (R + w_max)^2 / (V + w_max^2)),       a = (1 - level) / 2
//! ```
//!
//! The same two lines are in `epitools::ageadjust.direct`. A standardised
//! rate of 0 has the lower limit 0, as the exact interval of a count of 0
//! does, and the upper limit from the second line with `V = 0`.
//!
//! The rate ratio of two groups has `ln RR = ln((e1 / t1) / (e0 / t0))` with
//! variance `1 / e1 + 1 / e0`, and the Wald interval and p-value follow on
//! that scale. The rate difference `e1 / t1 - e0 / t0` has variance
//! `e1 / t1^2 + e0 / t0^2`, with the normal limits and p-value.
//! [`rate_ratio_exact_test`] is the conditional test: given `e1 + e0`, `e1` is
//! binomial with probability `t1 / (t1 + t0)` under equal rates, and the
//! two-sided p-value is the total probability of the outcomes not more likely
//! than the observed one, within a relative `1e-7`, as `stats::poisson.test`
//! computes it. Its counts must be integers within `1e-7`.
//!
//! # Undefined values
//!
//! A rate over person-time 0 is `None`. A standardised rate is `None` when a
//! stratum with a positive weight has no person-time, since a rate over the
//! other strata would be a different standard. The rate ratio is `None` when
//! the denominator has no events or either time is 0; with no events in the
//! numerator the ratio is 0 and its limits and p-value are `None`.
//!
//! # Accuracy
//!
//! The rate and the variance are sums of a few dozen terms in stratum order,
//! compensated, so they agree with a hand computation in R to about `1e-14`
//! relative. The limits inherit the accuracy of `gamma_quantile`. The exact
//! test's accuracy is that of the Poisson and binomial point probabilities,
//! about `1e-13` relative. The tests state the deviation found against R.
//!
//! # References
//!
//! 1. Fay MP, Feuer EJ. Confidence intervals for directly standardized rates:
//!    a method based on the gamma distribution. Stat Med 1997;16:791-801.
//! 2. Tiwari RC, Clegg LX, Zou Z. Efficient interval estimation for
//!    age-adjusted cancer rates. Stat Methods Med Res 2006;15:547-569.
//! 3. Breslow NE, Day NE. Statistical Methods in Cancer Research, volume II.
//!    IARC 1987, chapter 2.

use std::fmt;

use crate::discrete::binomial_two_sided;
use crate::normal;
use crate::poisson::{self, Method, gamma_quantile};
use crate::sum::compensated;

/// Why a rate was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateError {
    /// The slices differ in length.
    Length,
    /// An event count is negative, NaN or infinite.
    Events,
    /// A person-time is negative, NaN or infinite.
    Time,
    /// A weight is negative, NaN or infinite, or the weights sum to 0.
    Weights,
    /// The level is not strictly between 0 and 1.
    Level,
    /// The scale is not positive and finite.
    Scale,
    /// A count given to the exact test is not an integer.
    NotInteger,
    /// There are no strata.
    Empty,
}

impl fmt::Display for RateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Length => "the inputs differ in length",
            Self::Events => "an event count must be finite and not negative",
            Self::Time => "a person-time must be finite and not negative",
            Self::Weights => "weights must be finite, not negative and sum to more than 0",
            Self::Level => "the level must be strictly between 0 and 1",
            Self::Scale => "the scale must be positive and finite",
            Self::NotInteger => "the exact test needs integer counts",
            Self::Empty => "there are no strata",
        })
    }
}

impl std::error::Error for RateError {}

/// A rate with its exact Poisson interval.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rate {
    /// The events.
    pub events: f64,
    /// The person-time.
    pub time: f64,
    /// The rate times the scale, `None` without person-time.
    pub rate: Option<f64>,
    /// The lower limit of the interval, on the same scale.
    pub lower: Option<f64>,
    /// The upper limit of the interval, on the same scale.
    pub upper: Option<f64>,
}

/// A directly standardised rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Standardised {
    /// The standardised rate, `None` when a stratum with weight has no
    /// person-time.
    pub rate: Option<f64>,
    /// The variance of the rate.
    pub variance: Option<f64>,
    /// The lower limit of the gamma interval.
    pub lower: Option<f64>,
    /// The upper limit of the gamma interval.
    pub upper: Option<f64>,
}

/// A comparison of two rates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Contrast {
    /// The rate ratio or the rate difference.
    pub estimate: Option<f64>,
    /// The lower Wald limit.
    pub lower: Option<f64>,
    /// The upper Wald limit.
    pub upper: Option<f64>,
    /// The two-sided Wald p-value of no difference.
    pub p_value: Option<f64>,
}

fn check_level(level: f64) -> Result<(), RateError> {
    if level > 0.0 && level < 1.0 {
        Ok(())
    } else {
        Err(RateError::Level)
    }
}

fn check_count(x: f64) -> Result<(), RateError> {
    if x.is_finite() && x >= 0.0 {
        Ok(())
    } else {
        Err(RateError::Events)
    }
}

fn check_time(x: f64) -> Result<(), RateError> {
    if x.is_finite() && x >= 0.0 {
        Ok(())
    } else {
        Err(RateError::Time)
    }
}

/// The rate `events / time * scale` with the exact Poisson interval at the
/// coverage `level`.
///
/// # Errors
///
/// [`RateError`] for a negative or non-finite count or time, a level outside
/// `(0, 1)`, or a scale that is not positive and finite.
pub fn crude_rate(events: f64, time: f64, level: f64, scale: f64) -> Result<Rate, RateError> {
    check_count(events)?;
    check_time(time)?;
    check_level(level)?;
    if !(scale > 0.0 && scale.is_finite()) {
        return Err(RateError::Scale);
    }
    if time == 0.0 {
        return Ok(Rate {
            events,
            time,
            rate: None,
            lower: None,
            upper: None,
        });
    }
    let ci =
        poisson::cipoisson(events, time, level, Method::Exact).map_err(|_| RateError::Events)?;
    Ok(Rate {
        events,
        time,
        rate: Some(events / time * scale),
        lower: ci.lower.map(|x| x * scale),
        upper: ci.upper.map(|x| x * scale),
    })
}

/// The rate standardised to `weights`, with its variance and the gamma
/// interval of Fay and Feuer at the coverage `level`. The weights are scaled
/// to sum to 1, and the rate is per unit of person-time.
///
/// # Errors
///
/// [`RateError`] for slices of different length or no strata, a negative or
/// non-finite count, time or weight, weights that sum to 0, or a level
/// outside `(0, 1)`.
pub fn direct_standardised(
    events: &[f64],
    time: &[f64],
    weights: &[f64],
    level: f64,
) -> Result<Standardised, RateError> {
    if events.len() != time.len() || events.len() != weights.len() {
        return Err(RateError::Length);
    }
    if events.is_empty() {
        return Err(RateError::Empty);
    }
    check_level(level)?;
    events.iter().try_for_each(|&x| check_count(x))?;
    time.iter().try_for_each(|&x| check_time(x))?;
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(RateError::Weights);
    }
    let total = compensated(weights.iter().copied());
    if !(total > 0.0 && total.is_finite()) {
        return Err(RateError::Weights);
    }
    let undefined = Standardised {
        rate: None,
        variance: None,
        lower: None,
        upper: None,
    };
    if weights.iter().zip(time).any(|(&w, &y)| w > 0.0 && y == 0.0) {
        return Ok(undefined);
    }
    let share = |w: f64| w / total;
    let rate = compensated(
        weights
            .iter()
            .zip(events)
            .zip(time)
            .filter(|((w, _), _)| **w > 0.0)
            .map(|((&w, &d), &y)| share(w) * (d / y)),
    );
    let variance = compensated(
        weights
            .iter()
            .zip(events)
            .zip(time)
            .filter(|((w, _), _)| **w > 0.0)
            .map(|((&w, &d), &y)| share(w) * share(w) * d / (y * y)),
    );
    let w_max = weights
        .iter()
        .zip(time)
        .filter(|(w, _)| **w > 0.0)
        .fold(0.0_f64, |m, (&w, &y)| m.max(share(w) / y));
    let a = (1.0 - level) / 2.0;
    let lower = if rate == 0.0 {
        Some(0.0)
    } else {
        gamma_quantile(a, rate * rate / variance).map(|q| 2.0 * q * variance / (2.0 * rate))
    };
    let shifted = rate + w_max;
    let v_up = variance + w_max * w_max;
    let upper =
        gamma_quantile(1.0 - a, shifted * shifted / v_up).map(|q| 2.0 * q * v_up / (2.0 * shifted));
    Ok(Standardised {
        rate: Some(rate),
        variance: Some(variance),
        lower,
        upper,
    })
}

fn check_pair(e1: f64, t1: f64, e0: f64, t0: f64, level: f64) -> Result<(), RateError> {
    check_count(e1)?;
    check_count(e0)?;
    check_time(t1)?;
    check_time(t0)?;
    check_level(level)
}

fn wald(estimate: f64, se: f64, level: f64, log_scale: bool) -> Contrast {
    let z = normal::two_sided_z(level);
    let (lower, upper) = (estimate - z * se, estimate + z * se);
    let p = 2.0 * normal::cdf(-(estimate / se).abs());
    if log_scale {
        Contrast {
            estimate: Some(estimate.exp()),
            lower: Some(lower.exp()),
            upper: Some(upper.exp()),
            p_value: Some(p),
        }
    } else {
        Contrast {
            estimate: Some(estimate),
            lower: Some(lower),
            upper: Some(upper),
            p_value: Some(p),
        }
    }
}

/// The rate ratio `(e1 / t1) / (e0 / t0)` with the Wald interval and p-value
/// on the log scale at the coverage `level`.
///
/// # Errors
///
/// [`RateError`] for a negative or non-finite count or time, or a level
/// outside `(0, 1)`.
pub fn rate_ratio(e1: f64, t1: f64, e0: f64, t0: f64, level: f64) -> Result<Contrast, RateError> {
    check_pair(e1, t1, e0, t0, level)?;
    let none = Contrast {
        estimate: None,
        lower: None,
        upper: None,
        p_value: None,
    };
    if t1 == 0.0 || t0 == 0.0 || e0 == 0.0 {
        return Ok(none);
    }
    if e1 == 0.0 {
        return Ok(Contrast {
            estimate: Some(0.0),
            ..none
        });
    }
    let log_ratio = (e1 / t1).ln() - (e0 / t0).ln();
    Ok(wald(log_ratio, (1.0 / e1 + 1.0 / e0).sqrt(), level, true))
}

/// The rate difference `e1 / t1 - e0 / t0` with the normal interval and
/// p-value at the coverage `level`.
///
/// # Errors
///
/// [`RateError`] for a negative or non-finite count or time, or a level
/// outside `(0, 1)`.
pub fn rate_difference(
    e1: f64,
    t1: f64,
    e0: f64,
    t0: f64,
    level: f64,
) -> Result<Contrast, RateError> {
    check_pair(e1, t1, e0, t0, level)?;
    if t1 == 0.0 || t0 == 0.0 {
        return Ok(Contrast {
            estimate: None,
            lower: None,
            upper: None,
            p_value: None,
        });
    }
    let se = (e1 / (t1 * t1) + e0 / (t0 * t0)).sqrt();
    let difference = e1 / t1 - e0 / t0;
    if se == 0.0 {
        return Ok(Contrast {
            estimate: Some(difference),
            lower: None,
            upper: None,
            p_value: None,
        });
    }
    Ok(wald(difference, se, level, false))
}

/// The two-sided p-value of the exact conditional test that the two rates
/// are equal, as `stats::poisson.test(c(e1, e0), c(t1, t0))`. `None` when
/// there are no events or a person-time is 0.
///
/// # Errors
///
/// [`RateError`] for a negative or non-finite count or time, and
/// [`RateError::NotInteger`] for a count more than `1e-7` from an integer.
pub fn rate_ratio_exact_test(e1: f64, t1: f64, e0: f64, t0: f64) -> Result<Option<f64>, RateError> {
    check_pair(e1, t1, e0, t0, 0.5)?;
    let (x1, x0) = (e1.round(), e0.round());
    if (e1 - x1).abs() > 1e-7 || (e0 - x0).abs() > 1e-7 {
        return Err(RateError::NotInteger);
    }
    if t1 == 0.0 || t0 == 0.0 || x1 + x0 == 0.0 {
        return Ok(None);
    }
    Ok(Some(binomial_two_sided(x1, x1 + x0, t1 / (t1 + t0))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standardised_rate_of_equal_strata_is_the_crude_rate() {
        let s = direct_standardised(&[10.0, 20.0], &[100.0, 200.0], &[1.0, 3.0], 0.95).unwrap();
        assert!((s.rate.unwrap() - 0.1).abs() < 1e-15);
        let (lo, hi) = (s.lower.unwrap(), s.upper.unwrap());
        assert!(lo < 0.1 && 0.1 < hi);
    }

    #[test]
    fn empty_stratum_with_weight_has_no_rate() {
        let s = direct_standardised(&[1.0, 0.0], &[10.0, 0.0], &[1.0, 1.0], 0.95).unwrap();
        assert_eq!(s.rate, None);
        let s = direct_standardised(&[1.0, 0.0], &[10.0, 0.0], &[1.0, 0.0], 0.95).unwrap();
        assert_eq!(s.rate, Some(0.1));
    }

    #[test]
    fn refuses_bad_input() {
        assert_eq!(
            direct_standardised(&[1.0], &[1.0, 2.0], &[1.0], 0.95),
            Err(RateError::Length)
        );
        assert_eq!(
            direct_standardised(&[1.0], &[1.0], &[0.0], 0.95),
            Err(RateError::Weights)
        );
        assert_eq!(crude_rate(-1.0, 1.0, 0.95, 1.0), Err(RateError::Events));
        assert_eq!(
            rate_ratio_exact_test(1.5, 1.0, 2.0, 1.0),
            Err(RateError::NotInteger)
        );
    }

    #[test]
    fn zero_events_in_the_denominator_leave_the_ratio_undefined() {
        let c = rate_ratio(3.0, 10.0, 0.0, 10.0, 0.95).unwrap();
        assert_eq!(c.estimate, None);
        let c = rate_ratio(0.0, 10.0, 3.0, 10.0, 0.95).unwrap();
        assert_eq!((c.estimate, c.lower, c.p_value), (Some(0.0), None, None));
    }
}
