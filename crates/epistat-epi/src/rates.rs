//! Rates from events and person-time, their ratios and differences, and
//! directly standardised rates, each with an interval.
//!
//! # Algorithm
//!
//! A stratum with `d` events in person-time `y` has the rate `d / y`. Its
//! interval is the exact Poisson interval of [`epistat_stats::dist::poisson`], divided by
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
//! # Two standardised rates
//!
//! Two groups standardised to the same weights have the rates `R1` and `R0`
//! with variances `V1` and `V0`, and they are independent.
//! [`standardised_ratio`] gives `R1 / R0` with one of three intervals. Each F
//! interval treats each rate as a scaled chi-square variable on `2 R^2 / V`
//! degrees of freedom, so that their ratio is F, and moves the rate in the
//! numerator of the upper limit, and the one in the denominator of the lower
//! limit, up by a weight `s`, with its variance up by `u`, as the upper gamma
//! limit of one rate does. With `qf(p, m, n)` the F quantile
//! ([`epistat_stats::dist::f::quantile`]),
//!
//! ```text
//! lower = R1 / (R0 + s0)  qf(a,     2 R1^2 / V1,                2 (R0 + s0)^2 / (V0 + u0))
//! upper = (R1 + s1) / R0  qf(1 - a, 2 (R1 + s1)^2 / (V1 + u1),  2 R0^2 / V0)
//! ```
//!
//! For the F interval of Fay \[4\], `s = w_max` and `u = w_max^2`, from the
//! largest weight per person-time of the group. For the modified F interval
//! of Tiwari, Clegg and Zou \[2\], `s` is the mean of `w_s / y_s` over the `J`
//! strata with a positive weight and `u` the mean of its square: the
//! variance of `sum w_s (d_s + 1 / J) / y_s` when each `d_s + 1 / J` has its
//! mean as variance, a count of 1 spread over the strata where Fay's interval
//! puts it in the stratum of the largest weight. A ratio of 0 has the lower
//! limit 0. The Wald interval on the log scale has the variance
//! `V1 / R1^2 + V0 / R0^2` of `ln(R1 / R0)`. [`standardised_difference`]
//! gives `R1 - R0` with the normal interval on the variance `V1 + V0`, the
//! interval of Tiwari, Clegg and Zou for two independent rates.
//!
//! # Two crude rates
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
//! [`rate_ratio_exact`] adds the exact interval of the rate ratio: with
//! `p = e1 / (e1 + e0)` bounded by the Clopper-Pearson limits
//! `qbeta(a, e1, e0 + 1)` and `qbeta(1 - a, e1 + 1, e0)`, `a = (1 - level) / 2`,
//! the limits of the ratio are `p / (1 - p) t0 / t1`, as `poisson.test` has
//! them. They are 0 for `e1 = 0` and infinite for `e0 = 0`.
//!
//! # Undefined values
//!
//! A rate over person-time 0 is `None`. A standardised rate is `None` when a
//! stratum with a positive weight has no person-time, since a rate over the
//! other strata would be a different standard, and so is a ratio or
//! difference of two standardised rates when either is; a ratio is also `None`
//! when the reference rate is 0. The rate ratio is `None` when
//! the denominator has no events or either time is 0; with no events in the
//! numerator the ratio is 0 and its limits and p-value are `None`.
//!
//! # Accuracy
//!
//! The rate and the variance are sums of a few dozen terms in stratum order,
//! compensated, so they agree with a hand computation in R to about `1e-14`
//! relative. The limits inherit the accuracy of `gamma_quantile`. The exact
//! test's accuracy is that of the binomial point probabilities, which are
//! exponentials of sums of log-gamma terms. The error of the exponent is the
//! machine epsilon times the size of those terms, so the relative error of a
//! probability grows with the counts: about `1e-13` at a hundred events,
//! `1e-12` at a thousand and `1e-10` at `10^5`. The tests state the deviation
//! found against R.
//!
//! # References
//!
//! 1. Fay MP, Feuer EJ. Confidence intervals for directly standardized rates:
//!    a method based on the gamma distribution. Stat Med 1997;16:791-801.
//! 2. Tiwari RC, Clegg LX, Zou Z. Efficient interval estimation for
//!    age-adjusted cancer rates. Stat Methods Med Res 2006;15:547-569.
//! 3. Breslow NE, Day NE. Statistical Methods in Cancer Research, volume II.
//!    IARC 1987, chapter 2.
//! 4. Fay MP. Approximate confidence intervals for rate ratios from directly
//!    standardized rates with sparse data. Commun Stat Theory Methods
//!    1999;28:2141-2160.

use crate::Error;
use std::fmt;

use crate::exact::binomial_two_sided;
use epistat_core::Level;
use epistat_stats::dist::normal;
use epistat_stats::dist::poisson::{self, Method};
use epistat_stats::dist::{beta, chi_square, f};

/// `x`, or `None` for NaN.
fn finite(x: f64) -> Option<f64> {
    (!x.is_nan()).then_some(x)
}
use epistat_stats::sum::compensated;

/// Why a rate was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RateError {
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

fn check_level(level: f64) -> Result<(), Error> {
    if level > 0.0 && level < 1.0 {
        Ok(())
    } else {
        Err(RateError::Level.into())
    }
}

fn check_count(x: f64) -> Result<(), Error> {
    if x.is_finite() && x >= 0.0 {
        Ok(())
    } else {
        Err(RateError::Events.into())
    }
}

fn check_time(x: f64) -> Result<(), Error> {
    if x.is_finite() && x >= 0.0 {
        Ok(())
    } else {
        Err(RateError::Time.into())
    }
}

/// The rate `events / time * scale` with the exact Poisson interval at the
/// coverage `level`.
///
/// # Errors
///
/// [`crate::Error`] for a negative or non-finite count or time, a level outside
/// `(0, 1)`, or a scale that is not positive and finite.
pub fn crude_rate(events: f64, time: f64, level: f64, scale: f64) -> Result<Rate, Error> {
    check_count(events)?;
    check_time(time)?;
    check_level(level)?;
    if !(scale > 0.0 && scale.is_finite()) {
        return Err(RateError::Scale.into());
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
    let level = Level::confidence(level).map_err(|_| RateError::Level)?;
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

/// The sums of one group standardised to a set of weights, over the strata
/// with a positive weight.
struct Parts {
    /// `R = sum w_s d_s / y_s`.
    rate: f64,
    /// `V = sum w_s^2 d_s / y_s^2`.
    variance: f64,
    /// `max w_s / y_s`.
    w_max: f64,
    /// The mean of `w_s / y_s`.
    w_mean: f64,
    /// The mean of `(w_s / y_s)^2`.
    w_square_mean: f64,
}

/// Checks a group and the weights and forms its sums, `None` when a stratum
/// with a positive weight has no person-time.
fn standardise(events: &[f64], time: &[f64], weights: &[f64]) -> Result<Option<Parts>, Error> {
    if events.len() != time.len() || events.len() != weights.len() {
        return Err(RateError::Length.into());
    }
    if events.is_empty() {
        return Err(RateError::Empty.into());
    }
    events.iter().try_for_each(|&x| check_count(x))?;
    time.iter().try_for_each(|&x| check_time(x))?;
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(RateError::Weights.into());
    }
    let total = compensated(weights.iter().copied());
    if !(total > 0.0 && total.is_finite()) {
        return Err(RateError::Weights.into());
    }
    if weights.iter().zip(time).any(|(&w, &y)| w > 0.0 && y == 0.0) {
        return Ok(None);
    }
    let share = |w: f64| w / total;
    let used = || {
        weights
            .iter()
            .zip(events)
            .zip(time)
            .filter(|((w, _), _)| **w > 0.0)
    };
    let rate = compensated(used().map(|((&w, &d), &y)| share(w) * (d / y)));
    let variance = compensated(used().map(|((&w, &d), &y)| share(w) * share(w) * d / (y * y)));
    let w_max = used().fold(0.0_f64, |m, ((&w, _), &y)| m.max(share(w) / y));
    #[expect(
        clippy::cast_precision_loss,
        reason = "a number of strata is far below 2^53"
    )]
    let j = used().count() as f64;
    let w_mean = compensated(used().map(|((&w, _), &y)| share(w) / y)) / j;
    let w_square_mean = compensated(used().map(|((&w, _), &y)| {
        let x = share(w) / y;
        x * x
    })) / j;
    Ok(Some(Parts {
        rate,
        variance,
        w_max,
        w_mean,
        w_square_mean,
    }))
}

/// The rate standardised to `weights`, with its variance and the gamma
/// interval of Fay and Feuer at the coverage `level`. The weights are scaled
/// to sum to 1, and the rate is per unit of person-time.
///
/// # Errors
///
/// [`crate::Error`] for slices of different length or no strata, a negative or
/// non-finite count, time or weight, weights that sum to 0, or a level
/// outside `(0, 1)`.
pub fn direct_standardised(
    events: &[f64],
    time: &[f64],
    weights: &[f64],
    level: f64,
) -> Result<Standardised, Error> {
    let parts = standardise(events, time, weights)?;
    check_level(level)?;
    let Some(Parts {
        rate,
        variance,
        w_max,
        ..
    }) = parts
    else {
        return Ok(Standardised {
            rate: None,
            variance: None,
            lower: None,
            upper: None,
        });
    };
    let a = (1.0 - level) / 2.0;
    let lower = if rate == 0.0 {
        Some(0.0)
    } else {
        finite(chi_square::quantile(a, 2.0 * (rate * rate / variance)))
            .map(|q| q * variance / (2.0 * rate))
    };
    let shifted = rate + w_max;
    let v_up = variance + w_max * w_max;
    let upper = finite(chi_square::quantile(
        1.0 - a,
        2.0 * (shifted * shifted / v_up),
    ))
    .map(|q| q * v_up / (2.0 * shifted));
    Ok(Standardised {
        rate: Some(rate),
        variance: Some(variance),
        lower,
        upper,
    })
}

/// The events and person-time of one group in each stratum of a standard.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Strata<'a> {
    /// The events in each stratum.
    pub events: &'a [f64],
    /// The person-time in each stratum.
    pub time: &'a [f64],
}

/// The interval of the ratio of two directly standardised rates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RatioInterval {
    /// The F interval of Fay (1999), which shifts each rate by its largest
    /// weight per person-time, as the gamma interval of one rate does.
    F,
    /// The modified F interval of Tiwari, Clegg and Zou (2006), which shifts
    /// each rate by its mean weight per person-time.
    ModifiedF,
    /// The Wald interval on the log scale, with the variance of the log ratio
    /// `V1 / R1^2 + V0 / R0^2`.
    Log,
}

/// The two groups standardised to the same weights, or `None` when either
/// has a stratum with a positive weight and no person-time.
fn standardise_pair(
    group: Strata<'_>,
    reference: Strata<'_>,
    weights: &[f64],
    level: f64,
) -> Result<Option<(Parts, Parts)>, Error> {
    let one = standardise(group.events, group.time, weights)?;
    let zero = standardise(reference.events, reference.time, weights)?;
    check_level(level)?;
    Ok(one.zip(zero))
}

const NO_CONTRAST: Contrast = Contrast {
    estimate: None,
    lower: None,
    upper: None,
    p_value: None,
};

/// The ratio of the rate of `group` to that of `reference`, both directly
/// standardised to `weights`, with the interval `interval` at the coverage
/// `level`.
///
/// The ratio is `None` when the reference rate is 0 or either rate is
/// undefined, as in [`direct_standardised`]. A ratio of 0 has the lower limit
/// 0 and, with [`RatioInterval::Log`], no upper limit. The p-value is that
/// of the Wald test on the log scale for [`RatioInterval::Log`] and `None`
/// for the F intervals.
///
/// # Errors
///
/// [`crate::Error`] as for [`direct_standardised`], for either group.
pub fn standardised_ratio(
    group: Strata<'_>,
    reference: Strata<'_>,
    weights: &[f64],
    level: f64,
    interval: RatioInterval,
) -> Result<Contrast, Error> {
    let Some((one, zero)) = standardise_pair(group, reference, weights, level)? else {
        return Ok(NO_CONTRAST);
    };
    let (r1, v1, r0, v0) = (one.rate, one.variance, zero.rate, zero.variance);
    if r0 == 0.0 {
        return Ok(NO_CONTRAST);
    }
    let estimate = r1 / r0;
    let a = (1.0 - level) / 2.0;
    // A rate with its variance as a gamma variable on 2 r^2 / v degrees of
    // freedom, scaled.
    let df = |r: f64, v: f64| 2.0 * r * r / v;
    let (lower, upper) = match interval {
        RatioInterval::Log => {
            if r1 == 0.0 {
                return Ok(Contrast {
                    estimate: Some(0.0),
                    ..NO_CONTRAST
                });
            }
            let c = wald(
                estimate.ln(),
                (v1 / (r1 * r1) + v0 / (r0 * r0)).sqrt(),
                level,
                true,
            );
            return Ok(Contrast {
                estimate: Some(estimate),
                ..c
            });
        }
        RatioInterval::F => {
            let (s1, u1) = (r1 + one.w_max, v1 + one.w_max * one.w_max);
            let (s0, u0) = (r0 + zero.w_max, v0 + zero.w_max * zero.w_max);
            let lower = if r1 == 0.0 {
                Some(0.0)
            } else {
                finite(f::quantile(a, df(r1, v1), df(s0, u0))).map(|q| r1 / s0 * q)
            };
            let upper = finite(f::quantile(1.0 - a, df(s1, u1), df(r0, v0))).map(|q| s1 / r0 * q);
            (lower, upper)
        }
        RatioInterval::ModifiedF => {
            let (s1, u1) = (r1 + one.w_mean, v1 + one.w_square_mean);
            let (s0, u0) = (r0 + zero.w_mean, v0 + zero.w_square_mean);
            let lower = if r1 == 0.0 {
                Some(0.0)
            } else {
                finite(f::quantile(a, df(r1, v1), df(s0, u0))).map(|q| r1 / s0 * q)
            };
            let upper = finite(f::quantile(1.0 - a, df(s1, u1), df(r0, v0))).map(|q| s1 / r0 * q);
            (lower, upper)
        }
    };
    Ok(Contrast {
        estimate: Some(estimate),
        lower,
        upper,
        p_value: None,
    })
}

/// The difference of the rate of `group` and that of `reference`, both
/// directly standardised to `weights`, with the normal interval and p-value
/// at the coverage `level` from the variance `V1 + V0`.
///
/// The difference is `None` when either rate is undefined, and its limits and
/// p-value are `None` when both variances are 0.
///
/// # Errors
///
/// [`crate::Error`] as for [`direct_standardised`], for either group.
pub fn standardised_difference(
    group: Strata<'_>,
    reference: Strata<'_>,
    weights: &[f64],
    level: f64,
) -> Result<Contrast, Error> {
    let Some((one, zero)) = standardise_pair(group, reference, weights, level)? else {
        return Ok(NO_CONTRAST);
    };
    let difference = one.rate - zero.rate;
    let se = (one.variance + zero.variance).sqrt();
    if se == 0.0 {
        return Ok(Contrast {
            estimate: Some(difference),
            ..NO_CONTRAST
        });
    }
    Ok(wald(difference, se, level, false))
}

fn check_pair(e1: f64, t1: f64, e0: f64, t0: f64, level: f64) -> Result<(), Error> {
    check_count(e1)?;
    check_count(e0)?;
    check_time(t1)?;
    check_time(t0)?;
    check_level(level)
}

fn wald(estimate: f64, se: f64, level: f64, log_scale: bool) -> Contrast {
    let z = normal::quantile(1.0 - (1.0 - level) / 2.0);
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
/// [`crate::Error`] for a negative or non-finite count or time, or a level
/// outside `(0, 1)`.
pub fn rate_ratio(e1: f64, t1: f64, e0: f64, t0: f64, level: f64) -> Result<Contrast, Error> {
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
/// [`crate::Error`] for a negative or non-finite count or time, or a level
/// outside `(0, 1)`.
pub fn rate_difference(e1: f64, t1: f64, e0: f64, t0: f64, level: f64) -> Result<Contrast, Error> {
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
/// [`crate::Error`] for a negative or non-finite count or time, and
/// [`crate::Error`] for a count more than `1e-7` from an integer.
pub fn rate_ratio_exact_test(e1: f64, t1: f64, e0: f64, t0: f64) -> Result<Option<f64>, Error> {
    check_pair(e1, t1, e0, t0, 0.5)?;
    let (x1, x0) = (e1.round(), e0.round());
    if (e1 - x1).abs() > 1e-7 || (e0 - x0).abs() > 1e-7 {
        return Err(RateError::NotInteger.into());
    }
    if t1 == 0.0 || t0 == 0.0 || x1 + x0 == 0.0 {
        return Ok(None);
    }
    Ok(Some(binomial_two_sided(x1, x1 + x0, t1 / (t1 + t0))))
}

/// The rate ratio `(e1 / t1) / (e0 / t0)` with its exact conditional interval
/// at the coverage `level` and the exact two-sided p-value of equal rates, as
/// `stats::poisson.test(c(e1, e0), c(t1, t0))`.
///
/// The estimate is `None` when `e0` is 0, as in [`rate_ratio`], and the upper
/// limit is `None` there since it is infinite. The lower limit of a ratio with
/// `e1 = 0` is 0. Everything is `None` when there are no events or a time is 0.
///
/// # Errors
///
/// [`crate::Error`] for a negative or non-finite count or time, or a level
/// outside `(0, 1)`, and [`crate::Error`] for a count more than `1e-7`
/// from an integer.
pub fn rate_ratio_exact(e1: f64, t1: f64, e0: f64, t0: f64, level: f64) -> Result<Contrast, Error> {
    check_pair(e1, t1, e0, t0, level)?;
    let p_value = rate_ratio_exact_test(e1, t1, e0, t0)?;
    let (x1, x0) = (e1.round(), e0.round());
    if p_value.is_none() {
        return Ok(Contrast {
            estimate: None,
            lower: None,
            upper: None,
            p_value: None,
        });
    }
    let alpha = (1.0 - level) / 2.0;
    let to_ratio = |p: f64| p / (1.0 - p) * (t0 / t1);
    let lower = if x1 == 0.0 {
        Some(0.0)
    } else {
        finite(beta::quantile(alpha, x1, x0 + 1.0)).map(to_ratio)
    };
    let upper = if x0 == 0.0 {
        None
    } else {
        finite(beta::quantile(1.0 - alpha, x1 + 1.0, x0)).map(to_ratio)
    };
    Ok(Contrast {
        estimate: (x0 > 0.0).then(|| (e1 / t1) / (e0 / t0)),
        lower,
        upper,
        p_value,
    })
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
            Err(RateError::Length.into())
        );
        assert_eq!(
            direct_standardised(&[1.0], &[1.0], &[0.0], 0.95),
            Err(RateError::Weights.into())
        );
        assert_eq!(
            crude_rate(-1.0, 1.0, 0.95, 1.0),
            Err(RateError::Events.into())
        );
        assert_eq!(
            rate_ratio_exact_test(1.5, 1.0, 2.0, 1.0),
            Err(RateError::NotInteger.into())
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
