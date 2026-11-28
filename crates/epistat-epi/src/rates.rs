//! Rates from events and person-time, their ratios and differences, and
//! directly standardised rates, each with an interval.
//!
//! # Algorithm
//!
//! A stratum with `d` events in person-time `y` has the rate `d / y`. Its
//! interval is the exact Poisson interval of [`epistat_stats::dist::poisson`], divided by
//! `y`.
//!
//! # Rates per person-time
//!
//! Every function that returns a rate or a difference of rates takes `per`,
//! the person-time the rate is expressed per: 1 for a rate per unit of
//! person-time, 1000 for one per 1000. The estimate, its standard error and
//! its interval are multiplied by it after they are formed, so a p-value does
//! not depend on it. `per` multiplies and is not the `scale` of
//! `survival::pyears`, which divides. A group's events and person-time are
//! named fields of [`PersonTime`] and [`PersonTime2`], so the two cannot be
//! swapped:
//!
//! ```
//! use epistat_core::Level;
//! use epistat_epi::rates::{crude_rate, rate_ratio};
//! use epistat_epi::{PersonTime, PersonTime2};
//!
//! let rate = crude_rate(PersonTime { events: 12.0, time: 5_000.0 }, Level::P95, 1_000.0)?
//!     .expect("person-time");
//! assert!((rate.estimate - 2.4).abs() < 1e-12);
//! let table = PersonTime2 {
//!     exposed_events: 30.0,
//!     exposed_time: 1_200.0,
//!     unexposed_events: 12.0,
//!     unexposed_time: 1_500.0,
//! };
//! let rr = rate_ratio(&table, Level::P95)?.expect("events in both groups");
//! assert!((rr.estimate - 3.125).abs() < 1e-12);
//! # Ok::<(), epistat_epi::Error>(())
//! ```
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
//! them. The lower limit is 0 for `e1 = 0`, and for `e0 = 0` the ratio is
//! infinite and has no upper limit.
//!
//! # Results and levels
//!
//! Every estimate is an [`epistat_core::Estimate`] with its interval at the
//! [`Level`] given. In the formulas `a` is [`Level::lower_probability`] and
//! `1 - a` is [`Level::upper_probability`], so a level given as the
//! confidence `c` uses `(1 - c) / 2` and `1 - (1 - c) / 2` as written.
//!
//! # Undefined values
//!
//! A rate over person-time 0 is `None`. A standardised rate is `None` when a
//! stratum with a positive weight has no person-time, since a rate over the
//! other strata would be a different standard, and so is a ratio or
//! difference of two standardised rates when either is; a ratio is also `None`
//! when the reference rate is 0. The rate ratio is `None` when
//! the denominator has no events or either time is 0; with no events in the
//! numerator the ratio is 0 and has no interval or p-value.
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

use crate::error::{columns, count, counts, finite_number};
use crate::{Error, PersonTime, PersonTime2};
use std::fmt;

use crate::exact::binomial_two_sided;
use crate::limits::interval;
use epistat_core::{Estimate, Interval, Level, Scale, check};
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
    /// The exact interval of an event count could not be formed.
    Events,
    /// The weights sum to 0.
    Weights,
    /// A count given to the exact test is not an integer.
    NotInteger,
}

impl fmt::Display for RateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Events => "the exact interval of the event count is undefined",
            Self::Weights => "the weights sum to 0",
            Self::NotInteger => "the exact test needs integer counts",
        })
    }
}

impl std::error::Error for RateError {}

/// A directly standardised rate: the rate with the gamma interval of Fay and
/// Feuer, and its variance.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Standardised {
    /// The standardised rate. Its standard error is the square root of
    /// [`variance`](Self::variance) and its interval the gamma interval; it
    /// has no p-value.
    pub rate: Estimate,
    /// The variance of the rate, `sum w_s^2 d_s / y_s^2` times `per^2`.
    pub variance: f64,
}

fn check_count(what: &'static str, x: f64) -> Result<(), Error> {
    count(what, x, None)
}

fn check_time(what: &'static str, x: f64) -> Result<(), Error> {
    count(what, x, None)
}

/// A person-time multiplier that is finite and positive.
fn check_per(per: f64) -> Result<(), Error> {
    finite_number("per", per)?;
    Ok(check::positive("per", per)?)
}

/// `estimate` with its point, standard error and limits multiplied by `per`,
/// a rate per unit of person-time made one per `per`.
fn per_time(estimate: Estimate, per: f64) -> Estimate {
    let mut scaled = estimate;
    scaled.estimate = estimate.estimate * per;
    scaled.se = estimate.se.map(|se| se * per);
    scaled.interval = estimate.interval.map(|i| {
        Interval::new(i.lower.map(|x| x * per), i.upper.map(|x| x * per), i.level)
            .with_coverage(i.coverage)
    });
    scaled
}

/// The rate `events / time` per `per` person-time with the exact Poisson
/// interval at `level`, `None` without person-time. The estimate has no
/// standard error or p-value.
///
/// # Errors
///
/// [`crate::Error`] for a negative or non-finite count or time, or a `per`
/// that is not positive and finite.
pub fn crude_rate(group: PersonTime, level: Level, per: f64) -> Result<Option<Estimate>, Error> {
    let PersonTime { events, time } = group;
    check_count("events", events)?;
    check_time("time", time)?;
    check_per(per)?;
    if time == 0.0 {
        return Ok(None);
    }
    let ci =
        poisson::cipoisson(events, time, level, Method::Exact).map_err(|_| RateError::Events)?;
    Ok(Some(per_time(
        Estimate::point(events / time).with_interval(interval(ci.lower, ci.upper, level)),
        per,
    )))
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
    columns(&[
        ("events", events.len()),
        ("time", time.len()),
        ("weights", weights.len()),
    ])?;
    counts("events", events)?;
    counts("time", time)?;
    check::weights("weights", weights)?;
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

/// The rate of `group` standardised to `weights`, per `per` person-time, with
/// its variance and the gamma interval of Fay and Feuer at `level`. The
/// weights are scaled to sum to 1. `None` when a stratum with a positive
/// weight has no person-time.
///
/// # Errors
///
/// [`crate::Error`] for columns of different length or no strata, a negative
/// or non-finite count, time or weight, weights that sum to 0, or a `per`
/// that is not positive and finite.
pub fn direct_standardised(
    group: Strata<'_>,
    weights: &[f64],
    level: Level,
    per: f64,
) -> Result<Option<Standardised>, Error> {
    check_per(per)?;
    let Some(Parts {
        rate,
        variance,
        w_max,
        ..
    }) = standardise(group.events, group.time, weights)?
    else {
        return Ok(None);
    };
    let a = level.lower_probability();
    let lower = if rate == 0.0 {
        Some(0.0)
    } else {
        finite(chi_square::quantile(a, 2.0 * (rate * rate / variance)))
            .map(|q| q * variance / (2.0 * rate))
    };
    let shifted = rate + w_max;
    let v_up = variance + w_max * w_max;
    let upper = finite(chi_square::quantile(
        level.upper_probability(),
        2.0 * (shifted * shifted / v_up),
    ))
    .map(|q| q * v_up / (2.0 * shifted));
    Ok(Some(Standardised {
        rate: per_time(
            Estimate::point(rate)
                .with_se(Some(variance.sqrt()))
                .with_interval(interval(lower, upper, level)),
            per,
        ),
        variance: variance * per * per,
    }))
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
) -> Result<Option<(Parts, Parts)>, Error> {
    let one = standardise(group.events, group.time, weights)?;
    let zero = standardise(reference.events, reference.time, weights)?;
    Ok(one.zip(zero))
}

/// The ratio of the rate of `group` to that of `reference`, both directly
/// standardised to `weights`, with the interval `interval` at `level`.
///
/// The ratio is `None` when the reference rate is 0 or either rate is
/// undefined, as in [`direct_standardised`]. A ratio of 0 has the lower limit
/// 0 and, with [`RatioInterval::Log`], no upper limit. With
/// [`RatioInterval::Log`] the estimate is on [`Scale::Log`], with the
/// standard error of the log ratio and the Wald p-value on that scale; the F
/// intervals have neither.
///
/// # Errors
///
/// [`crate::Error`] as for [`direct_standardised`], for either group.
pub fn standardised_ratio(
    group: Strata<'_>,
    reference: Strata<'_>,
    weights: &[f64],
    level: Level,
    interval: RatioInterval,
) -> Result<Option<Estimate>, Error> {
    let Some((one, zero)) = standardise_pair(group, reference, weights)? else {
        return Ok(None);
    };
    let (r1, v1, r0, v0) = (one.rate, one.variance, zero.rate, zero.variance);
    if r0 == 0.0 {
        return Ok(None);
    }
    let estimate = r1 / r0;
    let a = level.lower_probability();
    let b = level.upper_probability();
    // A rate with its variance as a gamma variable on 2 r^2 / v degrees of
    // freedom, scaled.
    let df = |r: f64, v: f64| 2.0 * r * r / v;
    let (lower, upper) = match interval {
        RatioInterval::Log => {
            if r1 == 0.0 {
                return Ok(Some(Estimate::point(0.0).with_scale(Scale::Log)));
            }
            let w = wald(
                estimate.ln(),
                (v1 / (r1 * r1) + v0 / (r0 * r0)).sqrt(),
                level,
                Scale::Log,
            );
            return Ok(Some(Estimate::new(
                estimate, w.se, w.interval, w.p_value, w.scale,
            )));
        }
        RatioInterval::F => {
            let (s1, u1) = (r1 + one.w_max, v1 + one.w_max * one.w_max);
            let (s0, u0) = (r0 + zero.w_max, v0 + zero.w_max * zero.w_max);
            let lower = if r1 == 0.0 {
                Some(0.0)
            } else {
                finite(f::quantile(a, df(r1, v1), df(s0, u0))).map(|q| r1 / s0 * q)
            };
            let upper = finite(f::quantile(b, df(s1, u1), df(r0, v0))).map(|q| s1 / r0 * q);
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
            let upper = finite(f::quantile(b, df(s1, u1), df(r0, v0))).map(|q| s1 / r0 * q);
            (lower, upper)
        }
    };
    Ok(Some(
        Estimate::point(estimate).with_interval(self::interval(lower, upper, level)),
    ))
}

/// The difference of the rate of `group` and that of `reference`, both
/// directly standardised to `weights`, per `per` person-time, with the
/// normal interval and p-value at `level` from the variance `V1 + V0`.
///
/// The difference is `None` when either rate is undefined, and its interval
/// and p-value are `None` when both variances are 0.
///
/// # Errors
///
/// [`crate::Error`] as for [`direct_standardised`], for either group.
pub fn standardised_difference(
    group: Strata<'_>,
    reference: Strata<'_>,
    weights: &[f64],
    level: Level,
    per: f64,
) -> Result<Option<Estimate>, Error> {
    check_per(per)?;
    let Some((one, zero)) = standardise_pair(group, reference, weights)? else {
        return Ok(None);
    };
    let difference = one.rate - zero.rate;
    let se = (one.variance + zero.variance).sqrt();
    if se == 0.0 {
        return Ok(Some(per_time(
            Estimate::point(difference).with_se(Some(se)),
            per,
        )));
    }
    Ok(Some(per_time(
        wald(difference, se, level, Scale::Identity),
        per,
    )))
}

/// The exposed events and time and the unexposed events and time.
const fn parts(table: &PersonTime2) -> (f64, f64, f64, f64) {
    (
        table.exposed_events,
        table.exposed_time,
        table.unexposed_events,
        table.unexposed_time,
    )
}

/// The cells of `table`, each a count or person-time.
fn check_pair(table: &PersonTime2) -> Result<(f64, f64, f64, f64), Error> {
    let (e1, t1, e0, t0) = parts(table);
    check_count("exposed_events", e1)?;
    check_count("unexposed_events", e0)?;
    check_time("exposed_time", t1)?;
    check_time("unexposed_time", t0)?;
    Ok((e1, t1, e0, t0))
}

/// The Wald estimate of `estimate` with the standard error `se` at `level`.
/// On [`Scale::Log`], `estimate` is the log of the ratio and the limits and
/// the reported estimate are on the ratio's scale.
fn wald(estimate: f64, se: f64, level: Level, scale: Scale) -> Estimate {
    let z = normal::two_sided_z(level);
    let (lower, upper) = (estimate - z * se, estimate + z * se);
    let p = 2.0 * normal::cdf(-(estimate / se).abs());
    let (estimate, lower, upper) = match scale {
        Scale::Log => (estimate.exp(), lower.exp(), upper.exp()),
        _ => (estimate, lower, upper),
    };
    Estimate::new(
        estimate,
        Some(se),
        Some(Interval::new(Some(lower), Some(upper), level)),
        Some(p),
        scale,
    )
}

/// The rate ratio `(e1 / t1) / (e0 / t0)` of the exposed to the unexposed,
/// `e` the events and `t` the person-time of [`PersonTime2`], with the Wald
/// interval and p-value on the log scale at `level`, on [`Scale::Log`].
///
/// `None` when the denominator has no events or either time is 0. With no
/// events in the numerator the ratio is 0 and has no standard error,
/// interval or p-value.
///
/// # Errors
///
/// [`crate::Error`] for a negative or non-finite count or time.
pub fn rate_ratio(table: &PersonTime2, level: Level) -> Result<Option<Estimate>, Error> {
    let (e1, t1, e0, t0) = check_pair(table)?;
    if t1 == 0.0 || t0 == 0.0 || e0 == 0.0 {
        return Ok(None);
    }
    if e1 == 0.0 {
        return Ok(Some(Estimate::point(0.0).with_scale(Scale::Log)));
    }
    let log_ratio = (e1 / t1).ln() - (e0 / t0).ln();
    Ok(Some(wald(
        log_ratio,
        (1.0 / e1 + 1.0 / e0).sqrt(),
        level,
        Scale::Log,
    )))
}

/// The rate difference `e1 / t1 - e0 / t0` of the exposed and the unexposed,
/// per `per` person-time, with the normal interval and p-value at `level`.
///
/// `None` when either time is 0. Without events the difference is 0 and its
/// interval and p-value are `None`.
///
/// # Errors
///
/// [`crate::Error`] for a negative or non-finite count or time, or a `per`
/// that is not positive and finite.
pub fn rate_difference(
    table: &PersonTime2,
    level: Level,
    per: f64,
) -> Result<Option<Estimate>, Error> {
    let (e1, t1, e0, t0) = check_pair(table)?;
    check_per(per)?;
    if t1 == 0.0 || t0 == 0.0 {
        return Ok(None);
    }
    let se = (e1 / (t1 * t1) + e0 / (t0 * t0)).sqrt();
    let difference = e1 / t1 - e0 / t0;
    if se == 0.0 {
        return Ok(Some(per_time(
            Estimate::point(difference).with_se(Some(se)),
            per,
        )));
    }
    Ok(Some(per_time(
        wald(difference, se, level, Scale::Identity),
        per,
    )))
}

/// The two-sided p-value of the exact conditional test that the two rates
/// of `table` are equal, as `stats::poisson.test(c(e1, e0), c(t1, t0))` with
/// the exposed first. `None` when
/// there are no events or a person-time is 0.
///
/// # Errors
///
/// [`crate::Error`] for a negative or non-finite count or time, and
/// [`crate::Error`] for a count more than `1e-7` from an integer.
pub fn rate_ratio_exact_test(table: &PersonTime2) -> Result<Option<f64>, Error> {
    let (e1, t1, e0, t0) = check_pair(table)?;
    let (x1, x0) = (e1.round(), e0.round());
    if (e1 - x1).abs() > 1e-7 || (e0 - x0).abs() > 1e-7 {
        return Err(RateError::NotInteger.into());
    }
    if t1 == 0.0 || t0 == 0.0 || x1 + x0 == 0.0 {
        return Ok(None);
    }
    Ok(Some(binomial_two_sided(x1, x1 + x0, t1 / (t1 + t0))))
}

/// The rate ratio `(e1 / t1) / (e0 / t0)` of the exposed to the unexposed
/// with its exact conditional interval at `level` and the exact two-sided
/// p-value of equal rates, as `stats::poisson.test(c(e1, e0), c(t1, t0))`.
///
/// `None` when there are no events or a time is 0. With `e0 = 0` the
/// estimate is infinite, as `poisson.test` reports it, and the interval has
/// no upper limit. The lower limit of a ratio with `e1 = 0` is 0. The
/// estimate has no standard error.
///
/// # Errors
///
/// [`crate::Error`] for a negative or non-finite count or time, and for a
/// count more than `1e-7` from an integer.
pub fn rate_ratio_exact(table: &PersonTime2, level: Level) -> Result<Option<Estimate>, Error> {
    let Some(p_value) = rate_ratio_exact_test(table)? else {
        return Ok(None);
    };
    let (e1, t1, e0, t0) = parts(table);
    let (x1, x0) = (e1.round(), e0.round());
    let to_ratio = |p: f64| p / (1.0 - p) * (t0 / t1);
    let lower = if x1 == 0.0 {
        Some(0.0)
    } else {
        finite(beta::quantile(level.lower_probability(), x1, x0 + 1.0)).map(to_ratio)
    };
    let upper = if x0 == 0.0 {
        None
    } else {
        finite(beta::quantile(level.upper_probability(), x1 + 1.0, x0)).map(to_ratio)
    };
    let estimate = if x0 > 0.0 {
        (e1 / t1) / (e0 / t0)
    } else {
        f64::INFINITY
    };
    Ok(Some(
        Estimate::point(estimate)
            .with_interval(interval(lower, upper, level))
            .with_p_value(Some(p_value)),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standardised_rate_of_equal_strata_is_the_crude_rate() {
        let level = Level::default();
        let s = direct_standardised(
            Strata {
                events: &[10.0, 20.0],
                time: &[100.0, 200.0],
            },
            &[1.0, 3.0],
            level,
            1.0,
        )
        .unwrap()
        .unwrap();
        assert!((s.rate.estimate - 0.1).abs() < 1e-15);
        let (lo, hi) = (s.rate.lower().unwrap(), s.rate.upper().unwrap());
        assert!(lo < 0.1 && 0.1 < hi);
    }

    #[test]
    fn empty_stratum_with_weight_has_no_rate() {
        let level = Level::default();
        let s = direct_standardised(
            Strata {
                events: &[1.0, 0.0],
                time: &[10.0, 0.0],
            },
            &[1.0, 1.0],
            level,
            1.0,
        )
        .unwrap();
        assert_eq!(s, None);
        let s = direct_standardised(
            Strata {
                events: &[1.0, 0.0],
                time: &[10.0, 0.0],
            },
            &[1.0, 0.0],
            level,
            1.0,
        )
        .unwrap();
        assert_eq!(s.map(|s| s.rate.estimate), Some(0.1));
    }

    #[test]
    fn refuses_bad_input() {
        assert_eq!(
            direct_standardised(
                Strata {
                    events: &[1.0],
                    time: &[1.0, 2.0]
                },
                &[1.0],
                Level::default(),
                1.0
            ),
            Err(epistat_core::InputError::Length {
                what: "time",
                expected: 1,
                found: 2
            }
            .into())
        );
        assert_eq!(
            direct_standardised(
                Strata {
                    events: &[1.0],
                    time: &[1.0]
                },
                &[0.0],
                Level::default(),
                1.0
            ),
            Err(RateError::Weights.into())
        );
        assert_eq!(
            crude_rate(
                PersonTime {
                    events: -1.0,
                    time: 1.0
                },
                Level::default(),
                1.0
            ),
            Err(epistat_core::InputError::Domain {
                what: "events",
                value: -1.0,
                domain: "[0, inf)",
                row: None
            }
            .into())
        );
        assert_eq!(
            rate_ratio_exact_test(&PersonTime2 {
                exposed_events: 1.5,
                exposed_time: 1.0,
                unexposed_events: 2.0,
                unexposed_time: 1.0
            }),
            Err(RateError::NotInteger.into())
        );
    }

    #[test]
    fn per_multiplies_the_rate_and_leaves_the_p_value() {
        let level = Level::default();
        let table = PersonTime2 {
            exposed_events: 30.0,
            exposed_time: 1_200.0,
            unexposed_events: 12.0,
            unexposed_time: 1_500.0,
        };
        let one = rate_difference(&table, level, 1.0).unwrap().unwrap();
        let thousand = rate_difference(&table, level, 1_000.0).unwrap().unwrap();
        assert_eq!(thousand.estimate, one.estimate * 1_000.0);
        assert_eq!(thousand.se, one.se.map(|se| se * 1_000.0));
        assert_eq!(thousand.upper(), one.upper().map(|u| u * 1_000.0));
        assert_eq!(thousand.p_value, one.p_value);
        let strata = Strata {
            events: &[10.0, 20.0],
            time: &[100.0, 200.0],
        };
        let s1 = direct_standardised(strata, &[1.0, 3.0], level, 1.0)
            .unwrap()
            .unwrap();
        let s100 = direct_standardised(strata, &[1.0, 3.0], level, 100.0)
            .unwrap()
            .unwrap();
        assert_eq!(s100.rate.estimate, s1.rate.estimate * 100.0);
        assert_eq!(s100.rate.lower(), s1.rate.lower().map(|l| l * 100.0));
        assert_eq!(s100.variance, s1.variance * 100.0 * 100.0);
        let crude = crude_rate(
            PersonTime {
                events: 30.0,
                time: 300.0,
            },
            level,
            100.0,
        )
        .unwrap()
        .unwrap();
        assert_eq!(crude.estimate, 30.0 / 300.0 * 100.0);
        for per in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(rate_difference(&table, level, per).is_err());
            assert!(direct_standardised(strata, &[1.0, 3.0], level, per).is_err());
            assert!(
                crude_rate(
                    PersonTime {
                        events: 1.0,
                        time: 1.0
                    },
                    level,
                    per
                )
                .is_err()
            );
        }
    }

    #[test]
    fn zero_events_in_the_denominator_leave_the_ratio_undefined() {
        let level = Level::default();
        assert_eq!(
            rate_ratio(
                &PersonTime2 {
                    exposed_events: 3.0,
                    exposed_time: 10.0,
                    unexposed_events: 0.0,
                    unexposed_time: 10.0
                },
                level
            )
            .unwrap(),
            None
        );
        let c = rate_ratio(
            &PersonTime2 {
                exposed_events: 0.0,
                exposed_time: 10.0,
                unexposed_events: 3.0,
                unexposed_time: 10.0,
            },
            level,
        )
        .unwrap()
        .unwrap();
        assert_eq!((c.estimate, c.interval, c.p_value), (0.0, None, None));
    }
}
