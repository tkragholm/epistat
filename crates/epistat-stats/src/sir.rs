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
//! Exact (Garwood)   lower = Q_gamma(a, O) / E,  upper = Q_gamma(1 - a, O + 1) / E,  a = (1 - level) / 2
//! Byar              the cube-root form of crate::poisson, divided by E
//! Wald              exp(ln SIR -/+ z / sqrt(O)), undefined for O = 0
//! Profile           the t with 2 [ O ln(O / (E t)) - (O - E t) ] = z^2, one on each side of O / E
//! ```
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
//! With `E = 0` the ratio, its interval and its p-value are `None`. With
//! `O = 0` and `E > 0` the ratio is 0 and the lower limit is 0, except for the
//! Wald interval, which is `None`. A stratum with `E = 0` in a test over
//! strata is an error, since its ratio is undefined and dropping it would
//! change the question. The test of trend is `None` when the scores are equal
//! or the maximum likelihood estimate of the slope does not exist.
//!
//! # Accuracy
//!
//! The exact limits and the exact p-value have the accuracy of
//! [`crate::poisson::gamma_quantile`] and of [`crate::discrete`], about
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

use std::fmt;

use crate::discrete::poisson_two_sided;
use crate::gamma::chi_square_sf;
use crate::normal;
use crate::poisson::{self, Method};
use crate::sum::compensated;

/// Why an SIR was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SirError {
    /// An observed count is negative, NaN or infinite.
    Observed,
    /// An expected count is negative, NaN or infinite.
    Expected,
    /// A time or a scale is negative, NaN or infinite, or `per` is not positive.
    Time,
    /// The level is not strictly between 0 and 1.
    Level,
    /// The slices differ in length.
    Length,
    /// A stratum has an expected count of 0.
    ZeroExpected,
    /// A score is NaN or infinite.
    Score,
}

impl fmt::Display for SirError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Observed => "an observed count must be finite and not negative",
            Self::Expected => "an expected count must be finite and not negative",
            Self::Time => "the person-time must be finite and not negative, and per positive",
            Self::Level => "the level must be strictly between 0 and 1",
            Self::Length => "the inputs differ in length",
            Self::ZeroExpected => "a stratum has no expected events, so its ratio is undefined",
            Self::Score => "a score must be finite",
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

/// An SIR with its interval and p-value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sir {
    /// The observed events.
    pub observed: f64,
    /// The expected events.
    pub expected: f64,
    /// `observed / expected`, `None` without expected events.
    pub sir: Option<f64>,
    /// The lower limit.
    pub lower: Option<f64>,
    /// The upper limit.
    pub upper: Option<f64>,
    /// The two-sided exact p-value that the ratio is 1.
    pub p_value: Option<f64>,
}

/// A likelihood ratio statistic with its p-value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Homogeneity {
    /// The statistic.
    pub statistic: f64,
    /// The degrees of freedom.
    pub df: usize,
    /// The upper tail probability of the chi-square distribution.
    pub p_value: f64,
}

fn check_count(observed: f64, expected: f64) -> Result<(), SirError> {
    if !(observed.is_finite() && observed >= 0.0) {
        return Err(SirError::Observed);
    }
    if !(expected.is_finite() && expected >= 0.0) {
        return Err(SirError::Expected);
    }
    Ok(())
}

/// The SIR of `observed` against `expected` events, with the interval at the
/// coverage `level`. The observed count may be fractional, as when it is a sum
/// of weights, but then it has no exact p-value.
///
/// # Errors
///
/// [`SirError`] for a negative or non-finite count or a level outside `(0, 1)`.
pub fn sir(
    observed: f64,
    expected: f64,
    level: f64,
    interval: SirInterval,
) -> Result<Sir, SirError> {
    check_count(observed, expected)?;
    if !(level > 0.0 && level < 1.0) {
        return Err(SirError::Level);
    }
    if expected == 0.0 {
        return Ok(Sir {
            observed,
            expected,
            sir: None,
            lower: None,
            upper: None,
            p_value: None,
        });
    }
    let ratio = observed / expected;
    let (lower, upper) = match interval {
        SirInterval::Exact | SirInterval::Byar => {
            let method = if interval == SirInterval::Exact {
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
                profile_limit(observed, c, false).map(|m| m / expected),
                profile_limit(observed, c, true).map(|m| m / expected),
            )
        }
    };
    let integer = (observed - observed.round()).abs() <= 1e-7;
    let p_value = integer.then(|| poisson_two_sided(observed.round(), expected));
    Ok(Sir {
        observed,
        expected,
        sir: Some(ratio),
        lower,
        upper,
        p_value,
    })
}

/// The mean `mu` with `2 [ o ln(o / mu) - o + mu ] = c` above or below `o`.
fn profile_limit(o: f64, c: f64, upper: bool) -> Option<f64> {
    if o == 0.0 {
        return Some(if upper { c / 2.0 } else { 0.0 });
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
            return Some(x);
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
            return Some(x);
        }
    }
    Some(x)
}

fn check_strata(observed: &[f64], expected: &[f64]) -> Result<(), SirError> {
    if observed.len() != expected.len() {
        return Err(SirError::Length);
    }
    for (&o, &e) in observed.iter().zip(expected) {
        check_count(o, e)?;
        if e == 0.0 {
            return Err(SirError::ZeroExpected);
        }
    }
    Ok(())
}

fn chi_square(statistic: f64, df: usize) -> Homogeneity {
    Homogeneity {
        statistic,
        df,
        p_value: chi_square_sf(statistic, df as f64),
    }
}

/// The likelihood ratio test that the SIRs of the strata are equal, on one
/// degree of freedom less than the strata. `None` with fewer than two strata.
///
/// # Errors
///
/// [`SirError`] for slices of different length, a negative or non-finite
/// count, or a stratum with no expected events.
pub fn sir_homogeneity(
    observed: &[f64],
    expected: &[f64],
) -> Result<Option<Homogeneity>, SirError> {
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
/// [`SirError`] for slices of different length, a negative or non-finite
/// count, a stratum with no expected events, or a score that is not finite.
pub fn sir_trend(
    observed: &[f64],
    expected: &[f64],
    score: &[f64],
) -> Result<Option<Homogeneity>, SirError> {
    check_strata(observed, expected)?;
    if score.len() != observed.len() {
        return Err(SirError::Length);
    }
    if score.iter().any(|s| !s.is_finite()) {
        return Err(SirError::Score);
    }
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
/// [`SirError`] for a negative or non-finite count or time, or a `per` that is
/// not positive and finite.
pub fn excess_absolute_risk(
    observed: f64,
    expected: f64,
    time: f64,
    per: f64,
) -> Result<Option<f64>, SirError> {
    check_count(observed, expected)?;
    if !(time.is_finite() && time >= 0.0 && per.is_finite() && per > 0.0) {
        return Err(SirError::Time);
    }
    Ok((time > 0.0).then(|| (observed - expected) / time * per))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_limits_satisfy_the_deviance_equation() {
        for (o, e) in [(0.0, 3.0), (1.0, 2.0), (7.0, 4.0), (250.0, 200.0)] {
            let r = sir(o, e, 0.95, SirInterval::Profile).unwrap();
            let c = normal::two_sided_z(0.95).powi(2);
            for t in [r.lower.unwrap(), r.upper.unwrap()] {
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
        let r = sir(3.0, 0.0, 0.95, SirInterval::Exact).unwrap();
        assert_eq!((r.sir, r.lower, r.p_value), (None, None, None));
        let r = sir(0.0, 2.0, 0.95, SirInterval::Exact).unwrap();
        assert_eq!((r.sir, r.lower), (Some(0.0), Some(0.0)));
        let r = sir(0.0, 2.0, 0.95, SirInterval::Wald).unwrap();
        assert_eq!(r.lower, None);
        assert_eq!(
            sir(-1.0, 1.0, 0.95, SirInterval::Exact),
            Err(SirError::Observed)
        );
        assert_eq!(
            sir(1.0, f64::NAN, 0.95, SirInterval::Exact),
            Err(SirError::Expected)
        );
        assert_eq!(sir(1.0, 1.0, 0.0, SirInterval::Exact), Err(SirError::Level));
        assert_eq!(
            sir_homogeneity(&[1.0, 2.0], &[1.0, 0.0]),
            Err(SirError::ZeroExpected)
        );
    }

    #[test]
    fn equal_ratios_are_homogeneous() {
        let h = sir_homogeneity(&[10.0, 20.0, 30.0], &[5.0, 10.0, 15.0])
            .unwrap()
            .unwrap();
        assert!(h.statistic.abs() < 1e-12);
        assert_eq!(h.df, 2);
        let t = sir_trend(&[10.0, 20.0, 30.0], &[5.0, 10.0, 15.0], &[1.0, 2.0, 3.0])
            .unwrap()
            .unwrap();
        assert!(t.statistic.abs() < 1e-12);
    }

    #[test]
    fn trend_finds_a_slope() {
        let o = [5.0, 9.0, 20.0, 41.0];
        let e = [10.0, 10.0, 10.0, 10.0];
        let t = sir_trend(&o, &e, &[0.0, 1.0, 2.0, 3.0]).unwrap().unwrap();
        assert!(t.statistic > 20.0 && t.p_value < 1e-4);
    }
}
