//! Mantel-Haenszel estimators of a common odds ratio, risk ratio, rate ratio,
//! risk difference and rate difference over strata, the Mantel-Haenszel tests
//! of association, and the Breslow-Day test of homogeneity with Tarone's
//! correction.
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
//! ratios, and the p-value is the two-sided normal one. No continuity
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

use std::fmt;

use crate::gamma::chi_square_sf;
use crate::normal;
use crate::sum::compensated;

/// Why a Mantel-Haenszel estimate was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MhError {
    /// There are no strata.
    Empty,
    /// A count or person-time is negative, NaN or infinite, or events have no
    /// person-time.
    Cell,
    /// A stratum has no persons or no person-time.
    EmptyStratum,
    /// The level is not strictly between 0 and 1.
    Level,
    /// The slices given to a table builder differ in length.
    Length,
}

impl fmt::Display for MhError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Empty => "there are no strata",
            Self::Cell => "counts and person-time must be finite and not negative, and events need person-time",
            Self::EmptyStratum => "a stratum has no persons or no person-time",
            Self::Level => "the level must be strictly between 0 and 1",
            Self::Length => "the inputs differ in length",
        })
    }
}

impl std::error::Error for MhError {}

/// The 2 x 2 table of one stratum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Table {
    /// Exposed cases.
    pub a: f64,
    /// Exposed non-cases.
    pub b: f64,
    /// Unexposed cases.
    pub c: f64,
    /// Unexposed non-cases.
    pub d: f64,
}

/// The events and person-time of one stratum.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RateStratum {
    /// Events among the exposed.
    pub x1: f64,
    /// Person-time of the exposed.
    pub t1: f64,
    /// Events among the unexposed.
    pub x0: f64,
    /// Person-time of the unexposed.
    pub t0: f64,
}

/// A pooled estimate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pooled {
    /// The estimate, a ratio or a difference.
    pub estimate: Option<f64>,
    /// The standard error, of the log ratio for a ratio and of the difference
    /// for a difference.
    pub se: Option<f64>,
    /// The lower limit, on the scale of the estimate.
    pub lower: Option<f64>,
    /// The upper limit, on the scale of the estimate.
    pub upper: Option<f64>,
    /// The two-sided normal p-value of no effect.
    pub p_value: Option<f64>,
    /// The number of strata that compare the two groups, those with persons or
    /// person-time in both.
    pub n_strata_used: usize,
}

/// A test statistic on `df` degrees of freedom with its p-value.
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

fn check_level(level: f64) -> Result<(), MhError> {
    if level > 0.0 && level < 1.0 {
        Ok(())
    } else {
        Err(MhError::Level)
    }
}

fn check_tables(tables: &[Table]) -> Result<(), MhError> {
    if tables.is_empty() {
        return Err(MhError::Empty);
    }
    for t in tables {
        if ![t.a, t.b, t.c, t.d].into_iter().all(finite_count) {
            return Err(MhError::Cell);
        }
        if t.a + t.b + t.c + t.d == 0.0 {
            return Err(MhError::EmptyStratum);
        }
    }
    Ok(())
}

fn check_strata(strata: &[RateStratum]) -> Result<(), MhError> {
    if strata.is_empty() {
        return Err(MhError::Empty);
    }
    for s in strata {
        if ![s.x1, s.t1, s.x0, s.t0].into_iter().all(finite_count) {
            return Err(MhError::Cell);
        }
        if (s.t1 == 0.0 && s.x1 > 0.0) || (s.t0 == 0.0 && s.x0 > 0.0) {
            return Err(MhError::Cell);
        }
        if s.t1 + s.t0 == 0.0 {
            return Err(MhError::EmptyStratum);
        }
    }
    Ok(())
}

fn total(values: impl Iterator<Item = f64>) -> f64 {
    compensated(values)
}

fn undefined(n_strata_used: usize) -> Pooled {
    Pooled {
        estimate: None,
        se: None,
        lower: None,
        upper: None,
        p_value: None,
        n_strata_used,
    }
}

fn ratio_result(ratio: f64, variance: f64, level: f64, used: usize) -> Pooled {
    let se = variance.sqrt();
    let beta = ratio.ln();
    let z = normal::two_sided_z(level);
    Pooled {
        estimate: Some(ratio),
        se: Some(se),
        lower: Some((beta - z * se).exp()),
        upper: Some((beta + z * se).exp()),
        p_value: Some(2.0 * normal::cdf(-(beta / se).abs())),
        n_strata_used: used,
    }
}

fn difference_result(difference: f64, variance: f64, level: f64, used: usize) -> Pooled {
    let se = variance.sqrt();
    let z = normal::two_sided_z(level);
    Pooled {
        estimate: Some(difference),
        se: Some(se),
        lower: Some(difference - z * se),
        upper: Some(difference + z * se),
        p_value: Some(2.0 * normal::cdf(-(difference / se).abs())),
        n_strata_used: used,
    }
}

fn used_tables(tables: &[Table]) -> usize {
    tables
        .iter()
        .filter(|t| t.a + t.b > 0.0 && t.c + t.d > 0.0)
        .count()
}

fn used_strata(strata: &[RateStratum]) -> usize {
    strata.iter().filter(|s| s.t1 > 0.0 && s.t0 > 0.0).count()
}

/// The Mantel-Haenszel common odds ratio with the variance of Robins,
/// Breslow and Greenland, at the coverage `level`.
///
/// # Errors
///
/// [`MhError`] for no strata, a negative or non-finite cell, an empty
/// stratum, or a level outside `(0, 1)`.
pub fn odds_ratio(tables: &[Table], level: f64) -> Result<Pooled, MhError> {
    check_level(level)?;
    check_tables(tables)?;
    let used = used_tables(tables);
    let n = |t: &Table| t.a + t.b + t.c + t.d;
    let r_i = |t: &Table| (t.a / n(t)) * t.d;
    let s_i = |t: &Table| (t.b / n(t)) * t.c;
    let p_i = |t: &Table| t.a / n(t) + t.d / n(t);
    let q_i = |t: &Table| t.b / n(t) + t.c / n(t);
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
pub fn risk_ratio(tables: &[Table], level: f64) -> Result<Pooled, MhError> {
    check_level(level)?;
    check_tables(tables)?;
    let used = used_tables(tables);
    let n = |t: &Table| t.a + t.b + t.c + t.d;
    let r = total(tables.iter().map(|t| t.a * ((t.c + t.d) / n(t))));
    let s = total(tables.iter().map(|t| t.c * ((t.a + t.b) / n(t))));
    if total(tables.iter().map(|t| t.a)) == 0.0 || total(tables.iter().map(|t| t.c)) == 0.0 {
        return Ok(undefined(used));
    }
    let numerator =
        total(tables.iter().map(|t| {
            ((t.a + t.b) / n(t)) * ((t.c + t.d) / n(t)) * (t.a + t.c) - (t.a / n(t)) * t.c
        }));
    Ok(ratio_result(r / s, numerator / (r * s), level, used))
}

/// The Mantel-Haenszel common rate ratio with its variance, at the coverage
/// `level`.
///
/// # Errors
///
/// As [`odds_ratio`], for strata of events and person-time.
pub fn rate_ratio(strata: &[RateStratum], level: f64) -> Result<Pooled, MhError> {
    check_level(level)?;
    check_strata(strata)?;
    let used = used_strata(strata);
    let tt = |s: &RateStratum| s.t1 + s.t0;
    let r = total(strata.iter().map(|s| s.x1 * (s.t0 / tt(s))));
    let q = total(strata.iter().map(|s| s.x0 * (s.t1 / tt(s))));
    if total(strata.iter().map(|s| s.x1)) == 0.0 || total(strata.iter().map(|s| s.x0)) == 0.0 {
        return Ok(undefined(used));
    }
    let numerator = total(
        strata
            .iter()
            .map(|s| (s.t1 / tt(s)) * (s.t0 / tt(s)) * (s.x1 + s.x0)),
    );
    Ok(ratio_result(r / q, numerator / (r * q), level, used))
}

/// The Mantel-Haenszel common risk difference with the variance of Greenland
/// and Robins, at the coverage `level`.
///
/// # Errors
///
/// As [`odds_ratio`].
pub fn risk_difference(tables: &[Table], level: f64) -> Result<Pooled, MhError> {
    check_level(level)?;
    check_tables(tables)?;
    let used = used_tables(tables);
    let n = |t: &Table| t.a + t.b + t.c + t.d;
    let weight = total(tables.iter().map(|t| (t.a + t.b) * ((t.c + t.d) / n(t))));
    if weight == 0.0 {
        return Ok(undefined(used));
    }
    let beta = total(
        tables
            .iter()
            .map(|t| t.a * ((t.c + t.d) / n(t)) - t.c * ((t.a + t.b) / n(t))),
    ) / weight;
    let n1 = |t: &Table| t.a + t.b;
    let n0 = |t: &Table| t.c + t.d;
    let first = total(tables.iter().map(|t| {
        t.c * (n1(t) / n(t)) * (n1(t) / n(t)) - t.a * (n0(t) / n(t)) * (n0(t) / n(t))
            + (n1(t) / n(t)) * (n0(t) / n(t)) * (n0(t) - n1(t)) / 2.0
    }));
    let second = total(
        tables
            .iter()
            .map(|t| t.a * (n0(t) - t.c) / n(t) + t.c * (n1(t) - t.a) / n(t)),
    ) / 2.0;
    let variance = (beta * first + second) / (weight * weight);
    Ok(difference_result(beta, variance, level, used))
}

/// The Mantel-Haenszel common rate difference with its variance, at the
/// coverage `level`.
///
/// # Errors
///
/// As [`odds_ratio`], for strata of events and person-time.
pub fn rate_difference(strata: &[RateStratum], level: f64) -> Result<Pooled, MhError> {
    check_level(level)?;
    check_strata(strata)?;
    let used = used_strata(strata);
    let tt = |s: &RateStratum| s.t1 + s.t0;
    let weight = total(strata.iter().map(|s| (s.t1 / tt(s)) * s.t0));
    if weight == 0.0 {
        return Ok(undefined(used));
    }
    let beta = total(strata.iter().map(|s| (s.x1 * s.t0 - s.x0 * s.t1) / tt(s))) / weight;
    let spread = total(strata.iter().filter(|s| s.t1 > 0.0 && s.t0 > 0.0).map(|s| {
        let w = (s.t1 / tt(s)) * s.t0;
        w * w * (s.x1 / (s.t1 * s.t1) + s.x0 / (s.t0 * s.t0))
    }));
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
/// [`MhError`] for no strata, a negative or non-finite cell, or an empty
/// stratum.
pub fn cmh_test(tables: &[Table], continuity: bool) -> Result<Option<Test>, MhError> {
    check_tables(tables)?;
    let n = |t: &Table| t.a + t.b + t.c + t.d;
    let cases = total(tables.iter().map(|t| t.a + t.c));
    let non_cases = total(tables.iter().map(|t| t.b + t.d));
    if cases == 0.0 || non_cases == 0.0 {
        return Ok(None);
    }
    let delta = total(
        tables
            .iter()
            .map(|t| t.a - ((t.a + t.b) / n(t)) * (t.a + t.c)),
    );
    let variance = total(tables.iter().filter(|t| n(t) > 1.0).map(|t| {
        let (n1, n0) = (t.a + t.b, t.c + t.d);
        let (m1, m0) = (t.a + t.c, t.b + t.d);
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
/// [`MhError`] for no strata, a negative or non-finite value, or an empty
/// stratum.
pub fn rate_ratio_test(strata: &[RateStratum], continuity: bool) -> Result<Option<Test>, MhError> {
    check_strata(strata)?;
    let tt = |s: &RateStratum| s.t1 + s.t0;
    if total(strata.iter().map(|s| s.x1 + s.x0)) == 0.0 {
        return Ok(None);
    }
    let delta = total(strata.iter().map(|s| s.x1 - (s.x1 + s.x0) * (s.t1 / tt(s))));
    let variance = total(
        strata
            .iter()
            .map(|s| (s.x1 + s.x0) * (s.t1 / tt(s)) * (s.t0 / tt(s))),
    );
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
/// [`MhError`] for no strata, a negative or non-finite cell, or an empty
/// stratum.
pub fn breslow_day(tables: &[Table], tarone: bool) -> Result<Option<Test>, MhError> {
    check_tables(tables)?;
    let n = |t: &Table| t.a + t.b + t.c + t.d;
    let r = total(tables.iter().map(|t| (t.a / n(t)) * t.d));
    let s = total(tables.iter().map(|t| (t.b / n(t)) * t.c));
    if r == 0.0 || s == 0.0 {
        return Ok(None);
    }
    let or = r / s;
    let mut terms = Vec::new();
    for t in tables {
        let (n1, n0, m1, m0) = (t.a + t.b, t.c + t.d, t.a + t.c, t.b + t.d);
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
        terms.push((t.a - n11, variance));
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

/// Builds the 2 x 2 tables of strata from one record per person: the stratum
/// code, whether the person is exposed, and whether the person is a case.
/// Strata are numbered from 0 and a stratum without a person stays an empty
/// table, which the estimators refuse, so give consecutive codes.
///
/// # Errors
///
/// [`MhError::Length`] for slices of different length and [`MhError::Empty`]
/// for no records.
pub fn tables_from_records(
    stratum: &[usize],
    exposed: &[bool],
    case: &[bool],
) -> Result<Vec<Table>, MhError> {
    if stratum.len() != exposed.len() || stratum.len() != case.len() {
        return Err(MhError::Length);
    }
    let Some(&top) = stratum.iter().max() else {
        return Err(MhError::Empty);
    };
    let mut tables = vec![
        Table {
            a: 0.0,
            b: 0.0,
            c: 0.0,
            d: 0.0,
        };
        top + 1
    ];
    for ((&s, &e), &c) in stratum.iter().zip(exposed).zip(case) {
        let t = &mut tables[s];
        match (e, c) {
            (true, true) => t.a += 1.0,
            (true, false) => t.b += 1.0,
            (false, true) => t.c += 1.0,
            (false, false) => t.d += 1.0,
        }
    }
    Ok(tables)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tab(a: f64, b: f64, c: f64, d: f64) -> Table {
        Table { a, b, c, d }
    }

    #[test]
    fn one_stratum_is_the_cross_product_ratio() {
        let p = odds_ratio(&[tab(10.0, 20.0, 5.0, 40.0)], 0.95).unwrap();
        assert!((p.estimate.unwrap() - 4.0).abs() < 1e-14);
        let p = risk_ratio(&[tab(10.0, 20.0, 5.0, 40.0)], 0.95).unwrap();
        assert!((p.estimate.unwrap() - (10.0 / 30.0) / (5.0 / 45.0)).abs() < 1e-14);
    }

    #[test]
    fn undefined_cases_are_none() {
        let p = odds_ratio(&[tab(0.0, 5.0, 3.0, 4.0)], 0.95).unwrap();
        assert_eq!(p.estimate, None);
        let p = rate_ratio(
            &[RateStratum {
                x1: 0.0,
                t1: 4.0,
                x0: 2.0,
                t0: 3.0,
            }],
            0.95,
        )
        .unwrap();
        assert_eq!(p.estimate, None);
        assert_eq!(cmh_test(&[tab(1.0, 0.0, 2.0, 0.0)], false).unwrap(), None);
    }

    #[test]
    fn refuses_bad_input() {
        assert_eq!(odds_ratio(&[], 0.95), Err(MhError::Empty));
        assert_eq!(
            odds_ratio(&[tab(0.0, 0.0, 0.0, 0.0)], 0.95),
            Err(MhError::EmptyStratum)
        );
        assert_eq!(
            odds_ratio(&[tab(-1.0, 1.0, 1.0, 1.0)], 0.95),
            Err(MhError::Cell)
        );
        assert_eq!(
            odds_ratio(&[tab(1.0, 1.0, 1.0, 1.0)], 1.0),
            Err(MhError::Level)
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
