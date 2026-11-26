//! Mantel-Haenszel estimators of a common odds ratio, risk ratio, rate ratio,
//! risk difference and rate difference over strata, the Mantel-Haenszel tests
//! of association, the Breslow-Day test of homogeneity with Tarone's
//! correction, and the tests of a common rate ratio:
//! [`rate_ratio_breslow_day`] with the Mantel-Haenszel ratio and
//! [`rate_ratio_homogeneity`], the likelihood ratio and Pearson tests at the
//! conditional maximum likelihood estimate, whose formulas are in their
//! documentation.
//!
//! # Algorithm
//!
//! A stratum is a 2 x 2 table `a, b, c, d` of exposed cases, exposed
//! non-cases, unexposed cases and unexposed non-cases, or the events `x1`,
//! `x0` and person-time `t1`, `t0` of the exposed and the unexposed. With
//! `n = a + b + c + d`, `n1 = a + b`, `n0 = c + d`, `m1 = a + c`,
//! `m0 = b + d` and `T = t1 + t0`, every sum below runs over the strata, in
//! stratum order and compensated.
//!
//! ```text
//! OR   = R / S,   R = sum a d / n,   S = sum b c / n
//! P = (a + d) / n,   Q = (b + c) / n
//! Var(ln OR) = sum P R_i / (2 R^2) + sum (P S_i + Q R_i) / (2 R S) + sum Q S_i / (2 S^2)
//!              with R_i = a d / n, S_i = b c / n          (Robins, Breslow, Greenland)
//!
//! RR   = (sum a n0 / n) / (sum c n1 / n)
//! Var(ln RR) = sum (n1 n0 m1 / n^2 - a c / n) / ((sum a n0 / n) (sum c n1 / n))
//!
//! IRR  = (sum x1 t0 / T) / (sum x0 t1 / T)
//! Var(ln IRR) = sum (t1 t0 (x1 + x0) / T^2) / ((sum x1 t0 / T) (sum x0 t1 / T))
//!
//! RD   = sum (a n0 - c n1) / n  /  sum n1 n0 / n
//! Var(RD) = [ RD sum (c (n1/n)^2 - a (n0/n)^2 + (n1/n)(n0/n)(n0 - n1)/2)
//!             + sum (a (n0 - c) / n + c (n1 - a) / n) / 2 ] / (sum n1 n0 / n)^2
//!
//! IRD  = sum (x1 t0 - x0 t1) / T  /  sum t1 t0 / T
//! Var(IRD) = sum ((t1 t0 / T)^2 (x1 / t1^2 + x0 / t0^2)) / (sum t1 t0 / T)^2
//! ```
//!
//! The Greenland-Robins variance of the risk difference is the one `metafor`
//! writes. The interval is `estimate -/+ z se`, on the log scale for the
//! ratios, with `z` the normal quantile at [`Level::upper_probability`], and
//! the p-value is the two-sided normal one, as [`epistat_stats::wald`] forms
//! them. No continuity
//! constant is added to a zero cell.
//!
//! The Mantel-Haenszel chi-square is
//!
//! ```text
//! (|sum (a - n1 m1 / n)| - c)^2 / sum (n1 n0 m1 m0 / (n^2 (n - 1)))
//! ```
//!
//! and the rate version replaces the variance by `sum xt t1 t0 / T^2`, with
//! `xt = x1 + x0` and the numerator `sum (x1 - xt t1 / T)`. The continuity
//! correction is `c = 1/2` when the numerator is at least 1/2 and 0 otherwise,
//! as in `stats::mantelhaen.test`, so the corrected statistic is never smaller
//! than a half-unit shift allows. A stratum of
//! one person adds nothing to the variance.
//!
//! The Breslow-Day statistic compares each `a` with the value `N11` it would
//! have if the odds ratio were the pooled one. `N11` is the root in `(0, min(n1,
//! m1))` of `(OR - 1) N^2 - A N + OR n1 m1 = 0` with `A = OR (n1 + m1) + n0 - m1`.
//! The root is formed as `2 OR n1 m1 / (A + sqrt(A^2 - 4 OR (OR - 1) n1 m1))`,
//! which does not cancel when `OR` is near 1 and is `n1 m1 / n` at `OR = 1`
//! without a special case. Then
//!
//! ```text
//! BD = sum (a - N11)^2 / V,        V = (1/N11 + 1/N12 + 1/N21 + 1/N22)^-1
//! TA = BD - (sum (a - N11))^2 / sum V
//! ```
//!
//! over the strata with `N11 > 0`, both case and non-case margins positive and
//! both exposure arms non-empty, on one degree of freedom fewer than their
//! number. A stratum with an empty arm says nothing about homogeneity, and
//! `metafor` divides by zero in it.
//!
//! # Undefined values
//!
//! A pooled estimate is `None` when it is undefined, which is when its
//! numerator or denominator sum is 0: all `a d / n` or all `b c / n` for the
//! odds ratio, no exposed or no unexposed cases for the risk and rate ratio, no
//! stratum with both groups for a difference. A test is `None` with no cases or
//! no non-cases, with zero variance, or with fewer than two strata for
//! the Breslow-Day test.
//!
//! # Accuracy
//!
//! Each estimate and variance is a ratio of sums of at most `k` positive or
//! signed terms, so the error is of order `k eps kappa`, with `kappa` the ratio
//! of the sum of the absolute terms to the absolute sum. The tests state the
//! deviation from `metafor::rma.mh` and `stats::mantelhaen.test`.
//!
//! # References
//!
//! 1. Mantel N, Haenszel W. Statistical aspects of the analysis of data from
//!    retrospective studies of disease. J Natl Cancer Inst 1959;22:719-748.
//! 2. Greenland S, Robins JM. Estimation of a common effect parameter from
//!    sparse follow-up data. Biometrics 1985;41:55-68.
//! 3. Robins J, Breslow N, Greenland S. Estimators of the Mantel-Haenszel
//!    variance consistent in both sparse data and large-strata limiting
//!    models. Biometrics 1986;42:311-323.
//! 4. Breslow NE, Day NE. The analysis of case-control studies. IARC 1980,
//!    section 4.4.
//! 5. Tarone RE. On heterogeneity tests based on efficient scores. Biometrika
//!    1985;72:91-95.

use crate::Error;
use std::fmt;

use epistat_core::{Estimate, Level, Scale};
use epistat_stats::dist::chi_square::sf as chi_square_sf;
use epistat_stats::sum::compensated;
use epistat_stats::wald;

use crate::tables::{Counts2x2, PersonTime2};

/// Why a Mantel-Haenszel estimate was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MhError {
    /// There are no strata.
    Empty,
    /// A count or person-time is negative, NaN or infinite, or events have no
    /// person-time.
    Cell,
    /// A stratum has no persons or no person-time.
    EmptyStratum,
    /// The slices given to a table builder differ in length.
    Length,
}

impl fmt::Display for MhError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "there are no strata",
            Self::Cell => "counts and person-time must be finite and not negative, and events need person-time",
            Self::EmptyStratum => "a stratum has no persons or no person-time",
            Self::Length => "the inputs differ in length",
        })
    }
}

impl std::error::Error for MhError {}

/// A pooled estimate.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pooled {
    /// The estimate, a ratio or a difference, with its Wald interval and the
    /// two-sided normal p-value of no effect. A ratio is on [`Scale::Log`]:
    /// its standard error is that of the log ratio. `None` where the pooled
    /// estimate is undefined.
    pub estimate: Option<Estimate>,
    /// The number of strata that compare the two groups, those with persons or
    /// person-time in both.
    pub n_strata_used: usize,
}

/// A test statistic on `df` degrees of freedom with its p-value.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Test {
    /// The statistic.
    pub statistic: f64,
    /// The degrees of freedom of the chi-square reference distribution.
    pub df: usize,
    /// The upper tail probability.
    pub p_value: f64,
}

fn finite_count(x: f64) -> bool {
    x.is_finite() && x >= 0.0
}

fn check_tables(tables: &[Counts2x2]) -> Result<(), Error> {
    if tables.is_empty() {
        return Err(MhError::Empty.into());
    }
    for t in tables {
        if ![
            t.exposed_cases,
            t.exposed_noncases,
            t.unexposed_cases,
            t.unexposed_noncases,
        ]
        .into_iter()
        .all(finite_count)
        {
            return Err(MhError::Cell.into());
        }
        if t.exposed_cases + t.exposed_noncases + t.unexposed_cases + t.unexposed_noncases == 0.0 {
            return Err(MhError::EmptyStratum.into());
        }
    }
    Ok(())
}

fn check_strata(strata: &[PersonTime2]) -> Result<(), Error> {
    if strata.is_empty() {
        return Err(MhError::Empty.into());
    }
    for s in strata {
        if ![
            s.exposed_events,
            s.exposed_time,
            s.unexposed_events,
            s.unexposed_time,
        ]
        .into_iter()
        .all(finite_count)
        {
            return Err(MhError::Cell.into());
        }
        if (s.exposed_time == 0.0 && s.exposed_events > 0.0)
            || (s.unexposed_time == 0.0 && s.unexposed_events > 0.0)
        {
            return Err(MhError::Cell.into());
        }
        if s.exposed_time + s.unexposed_time == 0.0 {
            return Err(MhError::EmptyStratum.into());
        }
    }
    Ok(())
}

fn total(values: impl Iterator<Item = f64>) -> f64 {
    compensated(values)
}

const fn undefined(n_strata_used: usize) -> Pooled {
    Pooled {
        estimate: None,
        n_strata_used,
    }
}

/// The Wald estimate of a ratio from the variance of its log.
fn ratio_result(ratio: f64, variance: f64, level: Level, used: usize) -> Pooled {
    Pooled {
        estimate: Some(wald(ratio, Some(variance.sqrt()), level, Scale::Log)),
        n_strata_used: used,
    }
}

/// The Wald estimate of a difference from its variance.
fn difference_result(difference: f64, variance: f64, level: Level, used: usize) -> Pooled {
    Pooled {
        estimate: Some(wald(difference, Some(variance.sqrt()), level, Scale::Identity)),
        n_strata_used: used,
    }
}

fn used_tables(tables: &[Counts2x2]) -> usize {
    tables
        .iter()
        .filter(|t| {
            t.exposed_cases + t.exposed_noncases > 0.0
                && t.unexposed_cases + t.unexposed_noncases > 0.0
        })
        .count()
}

fn used_strata(strata: &[PersonTime2]) -> usize {
    strata
        .iter()
        .filter(|s| s.exposed_time > 0.0 && s.unexposed_time > 0.0)
        .count()
}

/// The Mantel-Haenszel common odds ratio with the variance of Robins,
/// Breslow and Greenland, at the coverage `level`.
///
/// # Errors
///
/// [`crate::Error`] for no strata, a negative or non-finite cell, or an
/// empty stratum.
pub fn odds_ratio(tables: &[Counts2x2], level: Level) -> Result<Pooled, Error> {
    check_tables(tables)?;
    let used = used_tables(tables);
    let n = |t: &Counts2x2| {
        t.exposed_cases + t.exposed_noncases + t.unexposed_cases + t.unexposed_noncases
    };
    let r_i = |t: &Counts2x2| (t.exposed_cases / n(t)) * t.unexposed_noncases;
    let s_i = |t: &Counts2x2| (t.exposed_noncases / n(t)) * t.unexposed_cases;
    let p_i = |t: &Counts2x2| t.exposed_cases / n(t) + t.unexposed_noncases / n(t);
    let q_i = |t: &Counts2x2| t.exposed_noncases / n(t) + t.unexposed_cases / n(t);
    let r = total(tables.iter().map(r_i));
    let s = total(tables.iter().map(s_i));
    if r == 0.0 || s == 0.0 {
        return Ok(undefined(used));
    }
    let variance = 0.5
        * (total(tables.iter().map(|t| p_i(t) * r_i(t))) / (r * r)
            + total(tables.iter().map(|t| p_i(t) * s_i(t) + q_i(t) * r_i(t))) / (r * s)
            + total(tables.iter().map(|t| q_i(t) * s_i(t))) / (s * s));
    Ok(ratio_result(r / s, variance, level, used))
}

/// The Mantel-Haenszel common risk ratio with the variance of Greenland and
/// Robins, at the coverage `level`.
///
/// # Errors
///
/// As [`odds_ratio`].
pub fn risk_ratio(tables: &[Counts2x2], level: Level) -> Result<Pooled, Error> {
    check_tables(tables)?;
    let used = used_tables(tables);
    let n = |t: &Counts2x2| {
        t.exposed_cases + t.exposed_noncases + t.unexposed_cases + t.unexposed_noncases
    };
    let r = total(
        tables
            .iter()
            .map(|t| t.exposed_cases * ((t.unexposed_cases + t.unexposed_noncases) / n(t))),
    );
    let s = total(
        tables
            .iter()
            .map(|t| t.unexposed_cases * ((t.exposed_cases + t.exposed_noncases) / n(t))),
    );
    if total(tables.iter().map(|t| t.exposed_cases)) == 0.0
        || total(tables.iter().map(|t| t.unexposed_cases)) == 0.0
    {
        return Ok(undefined(used));
    }
    let numerator = total(tables.iter().map(|t| {
        ((t.exposed_cases + t.exposed_noncases) / n(t))
            * ((t.unexposed_cases + t.unexposed_noncases) / n(t))
            * (t.exposed_cases + t.unexposed_cases)
            - (t.exposed_cases / n(t)) * t.unexposed_cases
    }));
    Ok(ratio_result(r / s, numerator / (r * s), level, used))
}

/// The Mantel-Haenszel common rate ratio with its variance, at the coverage
/// `level`.
///
/// # Errors
///
/// As [`odds_ratio`], for strata of events and person-time.
pub fn rate_ratio(strata: &[PersonTime2], level: Level) -> Result<Pooled, Error> {
    check_strata(strata)?;
    let used = used_strata(strata);
    let tt = |s: &PersonTime2| s.exposed_time + s.unexposed_time;
    let r = total(
        strata
            .iter()
            .map(|s| s.exposed_events * (s.unexposed_time / tt(s))),
    );
    let q = total(
        strata
            .iter()
            .map(|s| s.unexposed_events * (s.exposed_time / tt(s))),
    );
    if total(strata.iter().map(|s| s.exposed_events)) == 0.0
        || total(strata.iter().map(|s| s.unexposed_events)) == 0.0
    {
        return Ok(undefined(used));
    }
    let numerator = total(strata.iter().map(|s| {
        (s.exposed_time / tt(s))
            * (s.unexposed_time / tt(s))
            * (s.exposed_events + s.unexposed_events)
    }));
    Ok(ratio_result(r / q, numerator / (r * q), level, used))
}

/// The Mantel-Haenszel common risk difference with the variance of Greenland
/// and Robins, at the coverage `level`.
///
/// # Errors
///
/// As [`odds_ratio`].
pub fn risk_difference(tables: &[Counts2x2], level: Level) -> Result<Pooled, Error> {
    check_tables(tables)?;
    let used = used_tables(tables);
    let n = |t: &Counts2x2| {
        t.exposed_cases + t.exposed_noncases + t.unexposed_cases + t.unexposed_noncases
    };
    let weight = total(tables.iter().map(|t| {
        (t.exposed_cases + t.exposed_noncases) * ((t.unexposed_cases + t.unexposed_noncases) / n(t))
    }));
    if weight == 0.0 {
        return Ok(undefined(used));
    }
    let beta = total(tables.iter().map(|t| {
        t.exposed_cases * ((t.unexposed_cases + t.unexposed_noncases) / n(t))
            - t.unexposed_cases * ((t.exposed_cases + t.exposed_noncases) / n(t))
    })) / weight;
    let n1 = |t: &Counts2x2| t.exposed_cases + t.exposed_noncases;
    let n0 = |t: &Counts2x2| t.unexposed_cases + t.unexposed_noncases;
    let first = total(tables.iter().map(|t| {
        t.unexposed_cases * (n1(t) / n(t)) * (n1(t) / n(t))
            - t.exposed_cases * (n0(t) / n(t)) * (n0(t) / n(t))
            + (n1(t) / n(t)) * (n0(t) / n(t)) * (n0(t) - n1(t)) / 2.0
    }));
    let second = total(tables.iter().map(|t| {
        t.exposed_cases * (n0(t) - t.unexposed_cases) / n(t)
            + t.unexposed_cases * (n1(t) - t.exposed_cases) / n(t)
    })) / 2.0;
    let variance = (beta * first + second) / (weight * weight);
    Ok(difference_result(beta, variance, level, used))
}

/// The Mantel-Haenszel common rate difference with its variance, at the
/// coverage `level`.
///
/// # Errors
///
/// As [`odds_ratio`], for strata of events and person-time.
pub fn rate_difference(strata: &[PersonTime2], level: Level) -> Result<Pooled, Error> {
    check_strata(strata)?;
    let used = used_strata(strata);
    let tt = |s: &PersonTime2| s.exposed_time + s.unexposed_time;
    let weight = total(
        strata
            .iter()
            .map(|s| (s.exposed_time / tt(s)) * s.unexposed_time),
    );
    if weight == 0.0 {
        return Ok(undefined(used));
    }
    let beta = total(strata.iter().map(|s| {
        (s.exposed_events * s.unexposed_time - s.unexposed_events * s.exposed_time) / tt(s)
    })) / weight;
    let spread = total(
        strata
            .iter()
            .filter(|s| s.exposed_time > 0.0 && s.unexposed_time > 0.0)
            .map(|s| {
                let w = (s.exposed_time / tt(s)) * s.unexposed_time;
                w * w
                    * (s.exposed_events / (s.exposed_time * s.exposed_time)
                        + s.unexposed_events / (s.unexposed_time * s.unexposed_time))
            }),
    );
    Ok(difference_result(
        beta,
        spread / (weight * weight),
        level,
        used,
    ))
}

fn chi_square_test(statistic: f64, df: usize) -> Test {
    Test {
        statistic,
        df,
        p_value: chi_square_sf(statistic, df as f64),
    }
}

fn continuity_shift(delta: f64, continuity: bool) -> f64 {
    if continuity && delta.abs() >= 0.5 {
        0.5
    } else {
        0.0
    }
}

/// The Cochran-Mantel-Haenszel test of no association over the 2 x 2 tables,
/// on 1 degree of freedom, with the continuity correction of
/// `stats::mantelhaen.test` when `continuity` is set. `None` without cases or
/// without non-cases, or with zero variance.
///
/// # Errors
///
/// [`crate::Error`] for no strata, a negative or non-finite cell, or an empty
/// stratum.
pub fn cmh_test(tables: &[Counts2x2], continuity: bool) -> Result<Option<Test>, Error> {
    check_tables(tables)?;
    let n = |t: &Counts2x2| {
        t.exposed_cases + t.exposed_noncases + t.unexposed_cases + t.unexposed_noncases
    };
    let cases = total(tables.iter().map(|t| t.exposed_cases + t.unexposed_cases));
    let non_cases = total(
        tables
            .iter()
            .map(|t| t.exposed_noncases + t.unexposed_noncases),
    );
    if cases == 0.0 || non_cases == 0.0 {
        return Ok(None);
    }
    let delta = total(tables.iter().map(|t| {
        t.exposed_cases
            - ((t.exposed_cases + t.exposed_noncases) / n(t))
                * (t.exposed_cases + t.unexposed_cases)
    }));
    let variance = total(tables.iter().filter(|t| n(t) > 1.0).map(|t| {
        let (n1, n0) = (
            t.exposed_cases + t.exposed_noncases,
            t.unexposed_cases + t.unexposed_noncases,
        );
        let (m1, m0) = (
            t.exposed_cases + t.unexposed_cases,
            t.exposed_noncases + t.unexposed_noncases,
        );
        (n1 / n(t)) * (n0 / n(t)) * (m1 * (m0 / (n(t) - 1.0)))
    }));
    if variance == 0.0 {
        return Ok(None);
    }
    let shifted = delta.abs() - continuity_shift(delta, continuity);
    Ok(Some(chi_square_test(shifted * shifted / variance, 1)))
}

/// The Mantel-Haenszel test that the rate ratio is 1 over strata of events
/// and person-time, on 1 degree of freedom. `None` without events or with zero
/// variance.
///
/// # Errors
///
/// [`crate::Error`] for no strata, a negative or non-finite value, or an empty
/// stratum.
pub fn rate_ratio_test(strata: &[PersonTime2], continuity: bool) -> Result<Option<Test>, Error> {
    check_strata(strata)?;
    let tt = |s: &PersonTime2| s.exposed_time + s.unexposed_time;
    if total(strata.iter().map(|s| s.exposed_events + s.unexposed_events)) == 0.0 {
        return Ok(None);
    }
    let delta = total(strata.iter().map(|s| {
        s.exposed_events - (s.exposed_events + s.unexposed_events) * (s.exposed_time / tt(s))
    }));
    let variance = total(strata.iter().map(|s| {
        (s.exposed_events + s.unexposed_events)
            * (s.exposed_time / tt(s))
            * (s.unexposed_time / tt(s))
    }));
    if variance == 0.0 {
        return Ok(None);
    }
    let shifted = delta.abs() - continuity_shift(delta, continuity);
    Ok(Some(chi_square_test(shifted * shifted / variance, 1)))
}

/// The Breslow-Day test that the odds ratio is the same in every stratum, or
/// Tarone's corrected statistic when `tarone` is set. Only the strata with
/// positive expected count, both margins positive and both arms non-empty enter, and the degrees of
/// freedom are their number less one. `None` when the pooled odds ratio is
/// undefined or fewer than two strata enter.
///
/// # Errors
///
/// [`crate::Error`] for no strata, a negative or non-finite cell, or an empty
/// stratum.
pub fn breslow_day(tables: &[Counts2x2], tarone: bool) -> Result<Option<Test>, Error> {
    check_tables(tables)?;
    let n = |t: &Counts2x2| {
        t.exposed_cases + t.exposed_noncases + t.unexposed_cases + t.unexposed_noncases
    };
    let r = total(
        tables
            .iter()
            .map(|t| (t.exposed_cases / n(t)) * t.unexposed_noncases),
    );
    let s = total(
        tables
            .iter()
            .map(|t| (t.exposed_noncases / n(t)) * t.unexposed_cases),
    );
    if r == 0.0 || s == 0.0 {
        return Ok(None);
    }
    let or = r / s;
    let mut terms = Vec::new();
    for t in tables {
        let (n1, n0, m1, m0) = (
            t.exposed_cases + t.exposed_noncases,
            t.unexposed_cases + t.unexposed_noncases,
            t.exposed_cases + t.unexposed_cases,
            t.exposed_noncases + t.unexposed_noncases,
        );
        let nn = n(t);
        let a_coef = or * (n1 + m1) + (n0 - m1);
        let root = (a_coef * a_coef - 4.0 * or * (or - 1.0) * n1 * m1).sqrt();
        let n11 = 2.0 * or * n1 * m1 / (a_coef + root);
        if !(n1 > 0.0 && n0 > 0.0 && n11 > 0.0 && m1 > 0.0 && m0 > 0.0) {
            continue;
        }
        let n12 = n1 - n11;
        let n21 = m1 - n11;
        let n22 = n11 - n1 - m1 + nn;
        let variance = 1.0 / (1.0 / n11 + 1.0 / n12 + 1.0 / n21 + 1.0 / n22);
        terms.push((t.exposed_cases - n11, variance));
    }
    if terms.len() < 2 {
        return Ok(None);
    }
    let mut statistic = total(terms.iter().map(|&(e, v)| e * e / v)).max(0.0);
    if tarone {
        let sum_e = total(terms.iter().map(|&(e, _)| e));
        let sum_v = total(terms.iter().map(|&(_, v)| v));
        statistic = (statistic - sum_e * sum_e / sum_v).max(0.0);
    }
    Ok(Some(chi_square_test(statistic, terms.len() - 1)))
}

/// The strata that say something about a common rate ratio, as
/// `(x1, n, ln(t1 / t0))` with `n = x1 + x0`: both arms have person-time and
/// there is at least one event.
fn informative(strata: &[PersonTime2]) -> Vec<(f64, f64, f64)> {
    strata
        .iter()
        .filter(|s| {
            s.exposed_time > 0.0
                && s.unexposed_time > 0.0
                && s.exposed_events + s.unexposed_events > 0.0
        })
        .map(|s| {
            (
                s.exposed_events,
                s.exposed_events + s.unexposed_events,
                s.exposed_time.ln() - s.unexposed_time.ln(),
            )
        })
        .collect()
}

/// The probability `p = 1 / (1 + exp(-eta))` that an event of a stratum is
/// in the exposed arm, and `1 - p`, each formed without subtracting.
fn arm_probabilities(eta: f64) -> (f64, f64) {
    (1.0 / (1.0 + (-eta).exp()), 1.0 / (1.0 + eta.exp()))
}

/// The analogue of the Breslow-Day test for rate ratios: that the rate ratio
/// is the same in every stratum, with the Mantel-Haenszel rate ratio `psi`
/// in place of the common one, or Tarone's corrected statistic when `tarone`
/// is set.
///
/// Given the `n = x1 + x0` events of a stratum, `x1` is binomial with
/// probability `p = psi t1 / (psi t1 + t0)` under a common ratio, so
///
/// ```text
/// BD = sum (x1 - n p)^2 / (n p (1 - p))
/// TA = BD - (sum (x1 - n p))^2 / sum n p (1 - p)
/// ```
///
/// over the strata with person-time in both arms and at least one event, on
/// one degree of freedom fewer than their number. Tarone's correction makes
/// the statistic the score test at the conditional maximum likelihood
/// estimate to first order, as it does for odds ratios. `None` when the
/// Mantel-Haenszel ratio is 0 or undefined, or fewer than two strata enter.
///
/// # Errors
///
/// [`crate::Error`] for no strata, a negative or non-finite value, events without
/// person-time, or an empty stratum.
pub fn rate_ratio_breslow_day(strata: &[PersonTime2], tarone: bool) -> Result<Option<Test>, Error> {
    check_strata(strata)?;
    let pooled = rate_ratio(strata, Level::default())?;
    let Some(psi) = pooled
        .estimate
        .map(|r| r.estimate)
        .filter(|&r| r > 0.0 && r.is_finite())
    else {
        return Ok(None);
    };
    let ln_psi = psi.ln();
    let terms: Vec<(f64, f64)> = informative(strata)
        .into_iter()
        .map(|(x1, n, offset)| {
            let (p, q) = arm_probabilities(ln_psi + offset);
            (x1 - n * p, n * p * q)
        })
        .collect();
    if terms.len() < 2 {
        return Ok(None);
    }
    let mut statistic = total(terms.iter().map(|&(e, v)| e * e / v)).max(0.0);
    if tarone {
        let sum_e = total(terms.iter().map(|&(e, _)| e));
        let sum_v = total(terms.iter().map(|&(_, v)| v));
        statistic = (statistic - sum_e * sum_e / sum_v).max(0.0);
    }
    Ok(Some(chi_square_test(statistic, terms.len() - 1)))
}

/// The likelihood ratio and Pearson tests that the rate ratio is the same in
/// every stratum, at the conditional maximum likelihood estimate of the
/// common ratio. See [`rate_ratio_homogeneity`].
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateHomogeneity {
    /// The conditional maximum likelihood estimate of the common rate ratio.
    pub ratio: f64,
    /// The likelihood ratio test, the deviance of the model with one common
    /// ratio.
    pub likelihood_ratio: Test,
    /// The Pearson chi-square of the same model, the score test.
    pub pearson: Test,
    /// The Newton iterations taken.
    pub iterations: usize,
}

/// The likelihood ratio and Pearson tests of a common rate ratio over strata.
///
/// Given the `n_s = x1 + x0` events of stratum `s`, `x1` is binomial with
/// `logit p_s = beta + ln(t1 / t0)`, and a common rate ratio is
/// `psi = exp(beta)`. The conditional maximum likelihood estimate solves
/// `sum (x1 - n p) = 0` by Newton's method from the log of the
/// Mantel-Haenszel ratio, with step halving when the log likelihood would
/// fall, and stops when a step is below `4 eps max(1, |beta|)`. Then
///
/// ```text
/// G2 = 2 sum [x1 ln(x1 / (n p)) + x0 ln(x0 / (n (1 - p)))],   a term with a zero count is 0
/// X2 = sum (x1 - n p)^2 / (n p (1 - p))
/// ```
///
/// on one degree of freedom fewer than the strata with person-time in both
/// arms and at least one event. `G2` is the likelihood ratio test of the
/// interaction of exposure and stratum in a Poisson regression of the counts
/// with the log person-time as offset, since that model and the binomial one
/// have the same likelihood ratio, and `X2` is the Pearson statistic of
/// `glm(cbind(x1, x0) ~ 1, offset = log(t1 / t0), family = binomial)`.
///
/// `None` when fewer than two strata enter, or when all their events are in
/// one arm, so that the common ratio is 0 or infinite.
///
/// # Errors
///
/// [`crate::Error`] for no strata, a negative or non-finite value, events without
/// person-time, or an empty stratum.
pub fn rate_ratio_homogeneity(strata: &[PersonTime2]) -> Result<Option<RateHomogeneity>, Error> {
    check_strata(strata)?;
    let cells = informative(strata);
    if cells.len() < 2 {
        return Ok(None);
    }
    let exposed = total(cells.iter().map(|c| c.0));
    let unexposed = total(cells.iter().map(|c| c.1 - c.0));
    if exposed == 0.0 || unexposed == 0.0 {
        return Ok(None);
    }
    let Some(psi) = rate_ratio(strata, Level::default())?
        .estimate
        .map(|r| r.estimate)
        .filter(|&r| r > 0.0) else {
        return Ok(None);
    };
    let log_likelihood = |beta: f64| {
        total(cells.iter().map(|&(x1, n, offset)| {
            let (p, q) = arm_probabilities(beta + offset);
            let x0 = n - x1;
            let mut l = 0.0;
            if x1 > 0.0 {
                l += x1 * p.ln();
            }
            if x0 > 0.0 {
                l += x0 * q.ln();
            }
            l
        }))
    };
    let mut beta = psi.ln();
    let mut current = log_likelihood(beta);
    let mut iterations = 0;
    while iterations < 100 {
        iterations += 1;
        let (score, information) = cells.iter().fold((0.0, 0.0), |(u, i), &(x1, n, offset)| {
            let (p, q) = arm_probabilities(beta + offset);
            (u + (x1 - n * p), i + n * p * q)
        });
        let mut step = score / information;
        let mut next = beta + step;
        let mut value = log_likelihood(next);
        let mut halvings = 0;
        while value < current && halvings < 60 {
            step /= 2.0;
            next = beta + step;
            value = log_likelihood(next);
            halvings += 1;
        }
        beta = next;
        current = value;
        if step.abs() <= 4.0 * f64::EPSILON * beta.abs().max(1.0) {
            break;
        }
    }
    let df = cells.len() - 1;
    let mut deviance = Vec::with_capacity(cells.len());
    let mut pearson = Vec::with_capacity(cells.len());
    for &(x1, n, offset) in &cells {
        let (p, q) = arm_probabilities(beta + offset);
        let x0 = n - x1;
        let mut d = 0.0;
        if x1 > 0.0 {
            d += x1 * (x1 / (n * p)).ln();
        }
        if x0 > 0.0 {
            d += x0 * (x0 / (n * q)).ln();
        }
        deviance.push(2.0 * d);
        let e = x1 - n * p;
        pearson.push(e * e / (n * p * q));
    }
    Ok(Some(RateHomogeneity {
        ratio: beta.exp(),
        likelihood_ratio: chi_square_test(total(deviance.into_iter()).max(0.0), df),
        pearson: chi_square_test(total(pearson.into_iter()), df),
        iterations,
    }))
}

/// Builds the 2 x 2 tables of strata from one record per person: the stratum
/// code, whether the person is exposed, and whether the person is a case.
/// Strata are numbered from 0 and a stratum without a person stays an empty
/// table, which the estimators refuse, so give consecutive codes.
///
/// # Errors
///
/// [`crate::Error`] for slices of different length and [`crate::Error`]
/// for no records.
pub fn tables_from_records(
    stratum: &[usize],
    exposed: &[bool],
    case: &[bool],
) -> Result<Vec<Counts2x2>, Error> {
    if stratum.len() != exposed.len() || stratum.len() != case.len() {
        return Err(MhError::Length.into());
    }
    let Some(&top) = stratum.iter().max() else {
        return Err(MhError::Empty.into());
    };
    let mut tables = vec![
        Counts2x2 {
            exposed_cases: 0.0,
            exposed_noncases: 0.0,
            unexposed_cases: 0.0,
            unexposed_noncases: 0.0,
        };
        top + 1
    ];
    for ((&s, &e), &c) in stratum.iter().zip(exposed).zip(case) {
        let t = &mut tables[s];
        match (e, c) {
            (true, true) => t.exposed_cases += 1.0,
            (true, false) => t.exposed_noncases += 1.0,
            (false, true) => t.unexposed_cases += 1.0,
            (false, false) => t.unexposed_noncases += 1.0,
        }
    }
    Ok(tables)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tab(a: f64, b: f64, c: f64, d: f64) -> Counts2x2 {
        Counts2x2 {
            exposed_cases: a,
            exposed_noncases: b,
            unexposed_cases: c,
            unexposed_noncases: d,
        }
    }

    #[test]
    fn one_stratum_is_the_cross_product_ratio() {
        let p = odds_ratio(&[tab(10.0, 20.0, 5.0, 40.0)], Level::default()).unwrap();
        assert!((p.estimate.unwrap().estimate - 4.0).abs() < 1e-14);
        let p = risk_ratio(&[tab(10.0, 20.0, 5.0, 40.0)], Level::default()).unwrap();
        assert!((p.estimate.unwrap().estimate - (10.0 / 30.0) / (5.0 / 45.0)).abs() < 1e-14);
    }

    #[test]
    fn undefined_cases_are_none() {
        let p = odds_ratio(&[tab(0.0, 5.0, 3.0, 4.0)], Level::default()).unwrap();
        assert_eq!(p.estimate, None);
        let p = rate_ratio(
            &[PersonTime2 {
                exposed_events: 0.0,
                exposed_time: 4.0,
                unexposed_events: 2.0,
                unexposed_time: 3.0,
            }],
            Level::default(),
        )
        .unwrap();
        assert_eq!(p.estimate, None);
        assert_eq!(cmh_test(&[tab(1.0, 0.0, 2.0, 0.0)], false).unwrap(), None);
    }

    #[test]
    fn refuses_bad_input() {
        assert_eq!(odds_ratio(&[], Level::default()), Err(MhError::Empty.into()));
        assert_eq!(
            odds_ratio(&[tab(0.0, 0.0, 0.0, 0.0)], Level::default()),
            Err(MhError::EmptyStratum.into())
        );
        assert_eq!(
            odds_ratio(&[tab(-1.0, 1.0, 1.0, 1.0)], Level::default()),
            Err(MhError::Cell.into())
        );
    }

    #[test]
    fn identical_tables_are_homogeneous() {
        let t = tab(12.0, 30.0, 8.0, 50.0);
        let bd = breslow_day(&[t, t, t], false).unwrap().unwrap();
        assert!(bd.statistic < 1e-20, "{}", bd.statistic);
        assert_eq!(bd.df, 2);
    }

    #[test]
    fn builds_tables_from_records() {
        let t = tables_from_records(
            &[0, 0, 1, 1, 1],
            &[true, false, true, true, false],
            &[true, true, false, true, false],
        )
        .unwrap();
        assert_eq!(t[0], tab(1.0, 0.0, 1.0, 0.0));
        assert_eq!(t[1], tab(1.0, 1.0, 0.0, 1.0));
    }
}
