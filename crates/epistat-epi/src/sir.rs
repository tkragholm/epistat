//! The standardised incidence or mortality ratio of observed to expected
//! events, with exact, Byar, Wald and profile likelihood intervals, an exact
//! test, the likelihood ratio tests of homogeneity and trend over strata, and
//! the excess absolute risk.
//!
//! The expected count `E = sum_s y_s lambda_s` comes from the person-time of a
//! cohort and the rates of a reference population, for which see
//! `epistat_survival::ratetable`. The reference rates are taken as fixed.
//!
//! # Algorithm
//!
//! With `O` observed and `E` expected events, `SIR = O / E`. Four intervals are
//! offered, all for the Poisson mean `O` and divided by `E`:
//!
//! ```text
//! Exact (Garwood)   lower = Q_gamma(a, O) / E,  upper = Q_gamma(1 - a, O + 1) / E
//! Byar              the cube-root form of epistat_stats::dist::poisson, divided by E
//! Wald              exp(ln SIR -/+ z / sqrt(O)), undefined for O = 0
//! Profile           the t with 2 [ O ln(O / (E t)) - (O - E t) ] = z^2, one on each side of O / E
//! ```
//!
//! Here `a` is the level's [`Level::lower_probability`], `(1 - level) / 2`
//! for a confidence, and `z` the normal quantile at its
//! [`Level::upper_probability`].
//!
//! The profile limits are the likelihood ratio interval of a Poisson mean with
//! offset `ln E`. Each is the root of a smooth function that is monotone on its
//! side of `O`. The root is found by Newton steps inside a bracket, and a step
//! that leaves the bracket is replaced by a bisection, so the iteration always
//! ends. It stops when a step is within four units in the last place of the
//! value. For `O = 0` the lower limit is 0 and the upper is `z^2 / (2 E)`.
//!
//! The p-value is the exact two-sided one of `stats::poisson.test(O, E)`: the
//! total probability of the counts that are not more likely than `O` under a
//! Poisson distribution of mean `E`, within a relative `1e-7`. It needs an
//! integer count and is `None` for a fractional `O`.
//!
//! Over strata `g` with `SIR_all = sum O_g / sum E_g`, the likelihood ratio
//! test of equal SIRs is
//!
//! ```text
//! G = 2 sum_g [ O_g ln(O_g / (E_g SIR_all)) - (O_g - E_g SIR_all) ],      df = strata - 1
//! ```
//!
//! with `O ln(O / x) = 0` at `O = 0`. The test of trend is the likelihood
//! ratio test of `ln SIR_g = a + b s_g` against `b = 0`, with the scores `s_g`
//! and 1 degree of freedom. The model is a Poisson regression with offset
//! `ln E_g`, fitted by Newton steps with step halving from the null model on
//! the centred scores, so the log-likelihood increases at every step. The
//! statistic is accumulated per stratum as
//! `2 sum_g [ O_g (eta_g - a_0) - (mu_g - E_g e^{a_0}) ]`, which does not cancel.
//!
//! The excess absolute risk is `(O - E) / time * per`.
//!
//! # Undefined values
//!
//! With `E = 0` there is no estimate. With `O = 0` and `E > 0` the ratio is
//! 0 and the lower limit is 0, except with the Wald interval, which is then
//! `None`. A stratum with `E = 0` in a test over
//! strata is an error, since its ratio is undefined and dropping it would
//! change the question. The test of trend is `None` when the scores are equal
//! or the maximum likelihood estimate of the slope does not exist.
//!
//! # Accuracy
//!
//! The exact limits and the exact p-value have the accuracy of
//! [`epistat_stats::dist::gamma::quantile`] and of the Poisson point probabilities, about
//! `1e-13` relative. The profile limits are roots of a function whose
//! derivative is `2 (E - O / t)` at the root, so their relative error is a few
//! units of machine epsilon times the condition of the root, which is about
//! `sqrt(O)` for large `O`. The tests state the deviation from R.
//!
//! # References
//!
//! 1. Breslow NE, Day NE. Statistical Methods in Cancer Research, volume II.
//!    IARC 1987, chapters 2 and 3.
//! 2. Garwood F. Fiducial limits for the Poisson distribution. Biometrika
//!    1936;28:437-442.
//! 3. Byar's approximation is given in Breslow and Day \[1\], section 2.5.

use crate::Error;
use crate::error::{count, finite_number};
use std::fmt;

use crate::exact::poisson_two_sided;
use crate::limits::interval;
use epistat_core::{Estimate, Level, Scale, Test, check};
use epistat_stats::dist::chi_square::sf as chi_square_sf;
use epistat_stats::dist::normal;
use epistat_stats::dist::poisson::{self, Method};
use epistat_stats::sum::compensated;

/// Why an SIR was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SirError {
    /// The interval of the expected count could not be formed.
    Expected,
    /// A stratum has an expected count of 0.
    ZeroExpected,
}

impl fmt::Display for SirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Expected => "the interval of the expected count is undefined",
            Self::ZeroExpected => "a stratum has no expected events, so its ratio is undefined",
        })
    }
}

impl std::error::Error for SirError {}

/// How the interval of an SIR is found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SirInterval {
    /// The exact Poisson limits of Garwood.
    Exact,
    /// The limits of Byar.
    Byar,
    /// The Wald limits on the log scale.
    Wald,
    /// The profile likelihood limits.
    Profile,
}

fn check_count(observed: f64, expected: f64, row: Option<usize>) -> Result<(), Error> {
    count("observed", observed, row)?;
    count("expected", expected, row)
}

/// The SIR `observed / expected` with the interval `method` at `level` and
/// the exact two-sided p-value that the ratio is 1, `None` without expected
/// events.
///
/// The observed count may be fractional, as when it is a sum of weights, but
/// then it has no exact p-value. With [`SirInterval::Wald`] the estimate is on
/// [`Scale::Log`] with the standard error `1 / sqrt(O)` of `ln SIR`; the other
/// intervals have no standard error.
///
/// # Errors
///
/// [`crate::Error`] for a negative or non-finite count.
pub fn sir(
    observed: f64,
    expected: f64,
    level: Level,
    method: SirInterval,
) -> Result<Option<Estimate>, Error> {
    check_count(observed, expected, None)?;
    if expected == 0.0 {
        return Ok(None);
    }
    let ratio = observed / expected;
    let (lower, upper) = match method {
        SirInterval::Exact | SirInterval::Byar => {
            let method = if method == SirInterval::Exact {
                Method::Exact
            } else {
                Method::Byar
            };
            let ci = poisson::cipoisson(observed, expected, level, method)
                .map_err(|_| SirError::Expected)?;
            (ci.lower, ci.upper)
        }
        SirInterval::Wald => {
            if observed == 0.0 {
                (None, None)
            } else {
                let half = normal::two_sided_z(level) / observed.sqrt();
                let centre = ratio.ln();
                (Some((centre - half).exp()), Some((centre + half).exp()))
            }
        }
        SirInterval::Profile => {
            let c = normal::two_sided_z(level).powi(2);
            (
                Some(profile_limit(observed, c, false) / expected),
                Some(profile_limit(observed, c, true) / expected),
            )
        }
    };
    let integer = (observed - observed.round()).abs() <= 1e-7;
    let p_value = integer.then(|| poisson_two_sided(observed.round(), expected));
    let estimate = Estimate::point(ratio)
        .with_interval(interval(lower, upper, level))
        .with_p_value(p_value);
    Ok(Some(match method {
        SirInterval::Wald if observed > 0.0 => estimate
            .with_se(Some(1.0 / observed.sqrt()))
            .with_scale(Scale::Log),
        SirInterval::Wald => estimate.with_scale(Scale::Log),
        _ => estimate,
    }))
}

/// The mean `mu` with `2 [ o ln(o / mu) - o + mu ] = c` above or below `o`.
fn profile_limit(o: f64, c: f64, upper: bool) -> f64 {
    if o == 0.0 {
        return if upper { c / 2.0 } else { 0.0 };
    }
    let f = |mu: f64| 2.0 * (o * (o / mu).ln() - o + mu) - c;
    let slope = |mu: f64| 2.0 * (1.0 - o / mu);
    let (mut lo, mut hi) = if upper {
        let mut hi = o + c.sqrt() * o.sqrt() + c;
        while f(hi) < 0.0 {
            hi *= 2.0;
        }
        (o, hi)
    } else {
        (0.0, o)
    };
    // The lower root of a small count lies near 0, so start from a start that
    // keeps the logarithm finite.
    let mut x = if upper {
        f64::midpoint(lo, hi)
    } else {
        let guess = o + 0.5 * c - (c * o + 0.25 * c * c).sqrt();
        if guess > 0.0 { guess } else { o / (1.0 + c) }
    };
    for _ in 0..500 {
        let value = f(x);
        // The function falls on the lower side and rises on the upper.
        let above = if upper { value > 0.0 } else { value < 0.0 };
        if value == 0.0 {
            return x;
        }
        if above {
            hi = x;
        } else {
            lo = x;
        }
        let step = value / slope(x);
        let mut next = if step.is_finite() { x - step } else { f64::NAN };
        if !(next > lo && next < hi) {
            next = f64::midpoint(lo, hi);
        }
        let done = (next - x).abs() <= 4.0 * f64::EPSILON * next;
        x = next;
        if done {
            return x;
        }
    }
    x
}

fn check_strata(observed: &[f64], expected: &[f64]) -> Result<(), Error> {
    check::same_length("expected", observed.len(), expected.len())?;
    for (row, (&o, &e)) in observed.iter().zip(expected).enumerate() {
        check_count(o, e, Some(row))?;
        if e == 0.0 {
            return Err(SirError::ZeroExpected.into());
        }
    }
    Ok(())
}

/// A likelihood ratio statistic on `df` degrees of freedom with its
/// chi-square p-value, `None` for both when the statistic is not finite.
fn chi_square(statistic: f64, df: usize) -> Test {
    let df = df as f64;
    let statistic = statistic.is_finite().then_some(statistic);
    Test::chi_square(statistic, df, statistic.map(|x| chi_square_sf(x, df)))
}

/// The likelihood ratio test that the SIRs of the strata are equal, on one
/// degree of freedom less than the strata. `None` with fewer than two strata.
///
/// # Errors
///
/// [`crate::Error`] for slices of different length, a negative or non-finite
/// count, or a stratum with no expected events.
pub fn sir_homogeneity(observed: &[f64], expected: &[f64]) -> Result<Option<Test>, Error> {
    check_strata(observed, expected)?;
    if observed.len() < 2 {
        return Ok(None);
    }
    let pooled = compensated(observed.iter().copied()) / compensated(expected.iter().copied());
    let statistic = 2.0
        * compensated(observed.iter().zip(expected).map(|(&o, &e)| {
            let fitted = e * pooled;
            let log_part = if o > 0.0 { o * (o / fitted).ln() } else { 0.0 };
            log_part - (o - fitted)
        }));
    Ok(Some(chi_square(statistic.max(0.0), observed.len() - 1)))
}

/// The likelihood ratio test of a trend in the log SIR against the numeric
/// `score` of each stratum, on 1 degree of freedom. `None` with fewer than two
/// strata, equal scores, or a slope that has no finite maximum likelihood
/// estimate.
///
/// # Errors
///
/// [`crate::Error`] for slices of different length, a negative or non-finite
/// count, a stratum with no expected events, or a score that is not finite.
#[expect(
    clippy::float_cmp,
    reason = "equal scores give a model without a slope"
)]
pub fn sir_trend(observed: &[f64], expected: &[f64], score: &[f64]) -> Result<Option<Test>, Error> {
    check_strata(observed, expected)?;
    check::same_length("score", observed.len(), score.len())?;
    check::finite("score", score)?;
    let k = observed.len();
    if k < 2 || score.iter().all(|&s| s == score[0]) {
        return Ok(None);
    }
    let total_o = compensated(observed.iter().copied());
    let total_e = compensated(expected.iter().copied());
    if total_o == 0.0 {
        return Ok(Some(chi_square(0.0, 1)));
    }
    let weight_sum = compensated(expected.iter().copied());
    let centre = compensated(expected.iter().zip(score).map(|(&e, &s)| e * s)) / weight_sum;
    let x: Vec<f64> = score.iter().map(|&s| s - centre).collect();
    let a0 = (total_o / total_e).ln();
    let loglik = |a: f64, b: f64| {
        compensated(
            observed
                .iter()
                .zip(expected)
                .zip(&x)
                .map(|((&o, &e), &xi)| {
                    let eta = a + b * xi;
                    o * eta - e * eta.exp()
                }),
        )
    };
    let (mut a, mut b) = (a0, 0.0);
    let mut current = loglik(a, b);
    let mut converged = false;
    for _ in 0..200 {
        let (mut s0, mut s1, mut s2, mut ua, mut ub) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for ((&o, &e), &xi) in observed.iter().zip(expected).zip(&x) {
            let mu = e * (a + b * xi).exp();
            s0 += mu;
            s1 += mu * xi;
            s2 += mu * xi * xi;
            ua += o - mu;
            ub += xi * (o - mu);
        }
        let det = s0 * s2 - s1 * s1;
        if !(det > 0.0 && det.is_finite()) {
            return Ok(None);
        }
        let da = (s2 * ua - s1 * ub) / det;
        let db = (s0 * ub - s1 * ua) / det;
        let mut scale = 1.0;
        let mut accepted = false;
        for _ in 0..60 {
            let value = loglik(a + scale * da, b + scale * db);
            if value.is_finite() && value >= current {
                a += scale * da;
                b += scale * db;
                current = value;
                accepted = true;
                break;
            }
            scale /= 2.0;
        }
        if !accepted {
            converged = true;
            break;
        }
        let size = (scale * da)
            .abs()
            .max((scale * db).abs() * x.iter().fold(0.0_f64, |m, v| m.max(v.abs())));
        if size <= 1e-10 * (1.0 + a.abs() + b.abs()) {
            converged = true;
            break;
        }
    }
    if !converged || !b.is_finite() || b.abs() > 1e6 {
        return Ok(None);
    }
    let null_fit = total_o / total_e;
    let statistic = 2.0
        * compensated(
            observed
                .iter()
                .zip(expected)
                .zip(&x)
                .map(|((&o, &e), &xi)| {
                    let eta = a + b * xi;
                    o * (eta - a0) - (e * eta.exp() - e * null_fit)
                }),
        );
    Ok(Some(chi_square(statistic.max(0.0), 1)))
}

/// The excess absolute risk `(observed - expected) / time * per`, for example
/// per 1000 person-years. `None` without person-time.
///
/// # Errors
///
/// [`crate::Error`] for a negative or non-finite count or time, or a `per` that is
/// not positive and finite.
pub fn excess_absolute_risk(
    observed: f64,
    expected: f64,
    time: f64,
    per: f64,
) -> Result<Option<f64>, Error> {
    check_count(observed, expected, None)?;
    count("time", time, None)?;
    finite_number("per", per)?;
    check::positive("per", per)?;
    Ok((time > 0.0).then(|| (observed - expected) / time * per))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_limits_satisfy_the_deviance_equation() {
        for (o, e) in [(0.0, 3.0), (1.0, 2.0), (7.0, 4.0), (250.0, 200.0)] {
            let level = Level::default();
            let r = sir(o, e, level, SirInterval::Profile).unwrap().unwrap();
            let c = normal::two_sided_z(level).powi(2);
            for t in [r.lower().unwrap(), r.upper().unwrap()] {
                let mu = e * t;
                if mu > 0.0 {
                    let dev = 2.0 * (o * (o / mu).ln() - o + mu);
                    let dev = if o == 0.0 { 2.0 * mu } else { dev };
                    assert!((dev - c).abs() < 1e-10, "{o} {e} {dev}");
                }
            }
        }
    }

    #[test]
    fn undefined_and_edge_cases() {
        let level = Level::default();
        assert_eq!(sir(3.0, 0.0, level, SirInterval::Exact).unwrap(), None);
        let r = sir(0.0, 2.0, level, SirInterval::Exact).unwrap().unwrap();
        assert_eq!((r.estimate, r.lower()), (0.0, Some(0.0)));
        let r = sir(0.0, 2.0, level, SirInterval::Wald).unwrap().unwrap();
        assert_eq!((r.interval, r.se), (None, None));
        assert_eq!(
            sir(-1.0, 1.0, level, SirInterval::Exact),
            Err(epistat_core::InputError::Domain {
                what: "observed",
                value: -1.0,
                domain: "[0, inf)",
                row: None
            }
            .into())
        );
        assert_eq!(
            sir(1.0, f64::NAN, level, SirInterval::Exact),
            Err(epistat_core::InputError::NotFinite {
                what: "expected",
                row: 0,
                column: None
            }
            .into())
        );
        assert_eq!(
            sir_homogeneity(&[1.0, 2.0], &[1.0, 0.0]),
            Err(SirError::ZeroExpected.into())
        );
    }

    #[test]
    fn equal_ratios_are_homogeneous() {
        let h = sir_homogeneity(&[10.0, 20.0, 30.0], &[5.0, 10.0, 15.0])
            .unwrap()
            .unwrap();
        assert!(h.statistic.unwrap().abs() < 1e-12);
        assert_eq!(h.df, Some(2.0));
        let t = sir_trend(&[10.0, 20.0, 30.0], &[5.0, 10.0, 15.0], &[1.0, 2.0, 3.0])
            .unwrap()
            .unwrap();
        assert!(t.statistic.unwrap().abs() < 1e-12);
    }

    #[test]
    fn trend_finds_a_slope() {
        let o = [5.0, 9.0, 20.0, 41.0];
        let e = [10.0, 10.0, 10.0, 10.0];
        let t = sir_trend(&o, &e, &[0.0, 1.0, 2.0, 3.0]).unwrap().unwrap();
        assert!(t.statistic.unwrap() > 20.0 && t.p_value.unwrap() < 1e-4);
    }
}
