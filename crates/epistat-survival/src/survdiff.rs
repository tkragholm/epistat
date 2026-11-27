//! The log-rank test and the G-rho family for two or more groups, and the
//! one-sample test against expected survival, as `survival::survdiff`.
//!
//! # Method
//!
//! Take one stratum. At each distinct event time `t` let `n` be the number at
//! risk, `d` the number of events and `n_j` the number at risk in group `j`.
//! Under the hypothesis that the groups share one hazard, group `j` expects
//! `d n_j / n` of the events, and the counts of the groups, given `n` and `d`,
//! are multivariate hypergeometric with covariance
//!
//! ```text
//! d (n - d) / (n - 1) * (n_j [j = l] / n  -  n_j n_l / n^2).
//! ```
//!
//! The G-rho weight `w(t) = S(t-)^rho` is the left-continuous Kaplan-Meier
//! estimate of the pooled stratum raised to `rho`. Weighting the observed
//! count of each time by `w`, the expected count `w d n_j / n` and the
//! covariance by `w^2` gives, summed over the event times of every stratum,
//! the observed counts `O_j`, the expected counts `E_j` and the variance
//! matrix `V`. `rho = 0` is the log-rank test, `rho = 1` is the Peto and Peto
//! modification of the Gehan-Wilcoxon test that Harrington and Fleming
//! describe. Any finite `rho` is accepted.
//!
//! The statistic is `(O - E)' V^- (O - E)`, chi-square distributed with the
//! rank of `V` degrees of freedom. The groups' counts are tied together, since
//! `O - E` and every row of `V` sum to zero, so `V` has rank at most one less
//! than the number of groups, and a generalised inverse is needed. `V` is
//! symmetric and small, so it is diagonalised by cyclic Jacobi rotations, an
//! eigenvalue below the rank tolerance is treated as zero, and the statistic is
//! the sum over the other eigenvalues of `(u' (O - E))^2 / lambda` with `u` the
//! eigenvector. This is the same number as `survdiff` gets by dropping the
//! first group and solving, because on the subspace where `O - E` lives the
//! generalised inverse is unique. A group that is never at risk when an event
//! occurs has a zero row and column in `V`, which `survdiff` drops by hand
//! and this drops by its zero eigenvalue.
//!
//! # Errors of the computed numbers
//!
//! `O_j`, `E_j` and `V` are sums over at most `m` event times of terms that
//! are bounded by the expected count, so each carries a relative error of the
//! order `m (1 + |rho|) eps`, the second factor from the powers of the product
//! of up to `m` Kaplan-Meier factors. The variance is a sum of non-negative
//! terms with one subtraction per term. The null eigenvalue of `V` is
//! computed as about `m eps` times the largest. Jacobi's eigenvalues have an
//! absolute error of a small multiple of `k eps lambda_max`. The rank
//! tolerance is therefore `4 (m + k) eps lambda_max`, an eigenvalue below it
//! is zero, and the statistic has a relative error of order
//! `(m + k) eps lambda_max / lambda_min` over the retained eigenvalues, the
//! condition number of `V` on its range. The p-value follows through the
//! chi-square tail.
//!
//! # What survdiff supports
//!
//! Right-censored data only. A start column (counting-process data) and
//! several event codes (competing risks, multi-state) are refused, as
//! `survdiff` refuses them. `survdiff` has no case weights, so weights other
//! than 0 and 1 are refused here, and a row of weight 0 is left out.
//! Stratification is the stratum column of the data.
//!
//! # References
//!
//! - Mantel N (1966). Evaluation of survival data and two new rank order
//!   statistics arising in its consideration. Cancer Chemotherapy Reports
//!   50:163-170.
//! - Peto R, Peto J (1972). Asymptotically efficient rank invariant test
//!   procedures. Journal of the Royal Statistical Society A 135:185-207.
//! - Harrington DP, Fleming TR (1982). A class of rank test procedures for
//!   censored survival data. Biometrika 69:553-566.
//! - Therneau TM, Grambsch PM (2000). Modeling Survival Data: Extending the
//!   Cox Model. Springer.

use rayon::prelude::*;

use epistat_core::InputError;
use epistat_linalg::eigen::pinv_quadratic_form;
use epistat_stats::dist::chi_square::sf as chi_square_sf;

use crate::check::check_lengths;
use crate::error::{Error, Result};
use crate::surv::SurvData;
use crate::timefix::TimeFix;

/// Strata handled by one task. The partial sums of the variance matrix are
/// formed per block of this many strata and added in block order, so the
/// result does not depend on the number of threads.
const STRATA_PER_BLOCK: usize = 1024;

/// Settings of [`fn@test`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SurvdiffOptions {
    /// The exponent of the weight `S(t-)^rho`. 0 is the log-rank test and 1
    /// the Peto and Peto test. Any finite value is accepted.
    pub rho: f64,
    /// How times that differ by rounding error are treated before the test.
    /// `survival::aeqSurv` by default.
    pub timefix: TimeFix,
}

/// `SurvdiffOptions`, under its name in 0.3.
#[deprecated(since = "0.4.0", note = "renamed `SurvdiffOptions`")]
pub type Options = SurvdiffOptions;

/// The result of [`fn@test`]. Matrices are row-major.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Survdiff {
    /// The caller's code of each group present among the rows used, ascending.
    pub groups: Vec<u32>,
    /// The number of rows used in each group.
    pub n: Vec<usize>,
    /// The weighted observed events of each group, over all strata.
    pub observed: Vec<f64>,
    /// The weighted expected events of each group, over all strata.
    pub expected: Vec<f64>,
    /// The variance matrix of the expected counts, `groups * groups`, summed
    /// over the strata.
    pub variance: Vec<f64>,
    /// The caller's stratum code of each stratum with rows used, ascending, or
    /// `None` without a stratum column.
    pub strata: Option<Vec<u32>>,
    /// The number of strata with rows used, 1 without a stratum column.
    pub n_strata: usize,
    /// Observed events per stratum and group, `n_strata * groups`.
    pub observed_by_stratum: Vec<f64>,
    /// Expected events per stratum and group, `n_strata * groups`.
    pub expected_by_stratum: Vec<f64>,
    /// The chi-square statistic, or `None` when the variance matrix has rank
    /// 0: one group, or no event.
    pub statistic: Option<f64>,
    /// The degrees of freedom, the rank of the variance matrix.
    pub df: Option<usize>,
    /// The upper tail of the chi-square distribution at the statistic.
    pub p_value: Option<f64>,
    /// The number of distinct event times summed over, counted within strata.
    pub n_event_times: usize,
    /// The number of rows of the data, including rows of weight 0.
    pub n_rows: usize,
    /// The number of rows used, which are the rows of weight 1.
    pub n_used: usize,
}

/// The one-sample test of [`one_sample`].
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct OneSample {
    /// The observed events.
    pub observed: f64,
    /// The expected events, the sum of `-ln` of the expected survival.
    pub expected: f64,
    /// The variance of the observed minus the expected.
    pub variance: f64,
    /// The chi-square statistic on one degree of freedom, or `None` when the
    /// variance is 0.
    pub statistic: Option<f64>,
    /// The upper tail of the chi-square distribution on one degree of freedom.
    pub p_value: Option<f64>,
    /// The number of rows of the data, including rows of weight 0.
    pub n_rows: usize,
    /// The number of rows used.
    pub n_used: usize,
}

/// What the data must be for survdiff, and the rows it uses.
fn prepare<'a>(data: &SurvData<'a>, opts: &SurvdiffOptions) -> Result<SurvData<'a>> {
    let data = data.clone().with_timefix(opts.timefix)?;
    let surv = data.followup()?;
    if surv.start().is_some() {
        return Err(Error::not_applicable(
            "survdiff is defined for right-censored data: leave the start column out",
        ));
    }
    if surv.n_codes() != 1 {
        return Err(Error::not_applicable(
            "survdiff is defined for one event type: n_codes must be 1",
        ));
    }
    if !opts.rho.is_finite() {
        return Err(Error::options("rho must be finite"));
    }
    #[expect(clippy::float_cmp, reason = "a weight of 0 or 1 is exact")]
    if data
        .weight()
        .is_some_and(|w| w.iter().any(|&w| w != 0.0 && w != 1.0))
    {
        return Err(Error::value(
            "survdiff has no case weights: a weight is 0, to leave the row out, or 1",
        ));
    }
    if data.n_used() == 0 {
        return Err(Error::no_information(
            "no row has weight 1, so there is nothing to test",
        ));
    }
    Ok(data)
}

/// The partial result of one block of strata.
struct Block {
    variance: Vec<f64>,
    observed: Vec<f64>,
    expected: Vec<f64>,
    event_times: usize,
}

/// The k-sample test of equal hazards, as `survival::survdiff` with groups
/// and optionally strata.
///
/// `group` holds one code per row of the data, rows of weight 0 included, and
/// any `u32` codes are accepted. The groups tested are the distinct codes among
/// the rows used, reported in ascending order. A separate column is taken, not
/// a column of [`SurvData`], because a group is the factor under test, not part
/// of the data: one dataset is tested against several groupings, and the
/// codes need no dense numbering from the caller. Strata are the stratum
/// column of the data. See the module documentation for the method and what
/// comes back in [`Survdiff`].
///
/// One group, or no event in any stratum, leaves the statistic, its degrees of
/// freedom and the p-value `None`, and the counts are still reported.
///
/// The result is the same on any number of threads.
///
/// # Errors
///
/// An input error when `group` is not as long as the data, when the data have
/// a start column or more than one event code, when a weight is neither 0 nor
/// 1, when no row has weight 1, when `rho` is not finite, or when the tie rule
/// fails.
///
/// # Panics
///
/// Only on a broken internal invariant: a row's group missing from the groups
/// found, or more groups than rows.
pub fn test(data: &SurvData<'_>, group: &[u32], opts: &SurvdiffOptions) -> Result<Survdiff> {
    let data = prepare(data, opts)?;
    if group.is_empty() {
        return Err(Error::from(InputError::Empty { what: "group" }));
    }
    check_lengths(&[("time", data.n_rows()), ("group", group.len())])?;
    let surv = data.followup()?;
    let time = surv.stop();
    let status = surv.status();
    let rho = opts.rho;

    let mut rows: Vec<u32> = data.used_rows().collect();
    let mut codes: Vec<u32> = rows.iter().map(|&r| group[r as usize]).collect();
    codes.sort_unstable();
    codes.dedup();
    let k = codes.len();
    let dense = |row: u32| -> u32 {
        // The code is among the codes by construction, and there are no more
        // codes than rows, which the data builder holds to `u32`.
        let level = codes
            .binary_search(&group[row as usize])
            .expect("every row's group is among the groups");
        u32::try_from(level).expect("fewer groups than rows")
    };

    let stratum = data.stratum();
    let stratum_of = |row: u32| stratum.map_or(0, |s| s[row as usize]);
    rows.par_sort_unstable_by(|&a, &b| {
        stratum_of(a)
            .cmp(&stratum_of(b))
            .then(time[a as usize].total_cmp(&time[b as usize]))
            .then(a.cmp(&b))
    });
    let grp: Vec<u32> = rows.par_iter().map(|&r| dense(r)).collect();

    let mut n = vec![0_usize; k];
    for &g in &grp {
        n[g as usize] += 1;
    }

    // Strata with rows used, as ranges of the sorted rows.
    let mut ranges: Vec<(u32, usize, usize)> = Vec::new();
    let mut begin = 0;
    for i in 1..=rows.len() {
        if i == rows.len() || stratum_of(rows[i]) != stratum_of(rows[begin]) {
            ranges.push((stratum_of(rows[begin]), begin, i));
            begin = i;
        }
    }
    let n_strata = ranges.len();

    let blocks: Vec<Block> = ranges
        .par_chunks(STRATA_PER_BLOCK)
        .map(|chunk| {
            let mut block = Block {
                variance: vec![0.0; k * k],
                observed: vec![0.0; chunk.len() * k],
                expected: vec![0.0; chunk.len() * k],
                event_times: 0,
            };
            let mut risk = vec![0.0; k];
            for (s, &(_, from, to)) in chunk.iter().enumerate() {
                let o = &mut block.observed[s * k..(s + 1) * k];
                let e = &mut block.expected[s * k..(s + 1) * k];
                block.event_times += one_stratum(
                    &rows[from..to],
                    &grp[from..to],
                    time,
                    status,
                    rho,
                    &mut risk,
                    o,
                    e,
                    &mut block.variance,
                );
            }
            block
        })
        .collect();

    let mut variance = vec![0.0; k * k];
    let mut observed_by_stratum = Vec::with_capacity(n_strata * k);
    let mut expected_by_stratum = Vec::with_capacity(n_strata * k);
    let mut n_event_times = 0;
    for block in &blocks {
        for (v, b) in variance.iter_mut().zip(&block.variance) {
            *v += b;
        }
        observed_by_stratum.extend_from_slice(&block.observed);
        expected_by_stratum.extend_from_slice(&block.expected);
        n_event_times += block.event_times;
    }
    let mut observed = vec![0.0; k];
    let mut expected = vec![0.0; k];
    for s in 0..n_strata {
        for g in 0..k {
            observed[g] += observed_by_stratum[s * k + g];
            expected[g] += expected_by_stratum[s * k + g];
        }
    }

    let strata = data.stratum_levels().map(|levels| {
        ranges
            .iter()
            .map(|&(level, _, _)| levels.code(level as usize).unwrap_or(level))
            .collect()
    });

    let difference: Vec<f64> = observed.iter().zip(&expected).map(|(o, e)| o - e).collect();
    let (statistic, rank) = quadratic_form(&variance, &difference, n_event_times);
    let df = (rank > 0).then_some(rank);
    #[expect(clippy::cast_precision_loss, reason = "a rank is a few groups")]
    let p_value = statistic.map(|x| chi_square_sf(x, rank as f64));

    Ok(Survdiff {
        groups: codes,
        n,
        observed,
        expected,
        variance,
        strata,
        n_strata: if data.stratum().is_some() {
            n_strata
        } else {
            1
        },
        observed_by_stratum,
        expected_by_stratum,
        statistic,
        df,
        p_value,
        n_event_times,
        n_rows: data.n_rows(),
        n_used: data.n_used(),
    })
}

/// One stratum, with its rows ordered by time. Adds to `observed`, `expected`
/// and `variance`, and returns the number of event times.
#[expect(clippy::too_many_arguments, reason = "the pieces of one pass")]
#[expect(clippy::float_cmp, reason = "tied times are equal after the tie rule")]
#[expect(
    clippy::cast_precision_loss,
    reason = "counts of rows are exact below 2^53"
)]
fn one_stratum(
    rows: &[u32],
    grp: &[u32],
    time: &[f64],
    status: &[i32],
    rho: f64,
    risk: &mut [f64],
    observed: &mut [f64],
    expected: &mut [f64],
    variance: &mut [f64],
) -> usize {
    let k = risk.len();
    risk.fill(0.0);
    for &g in grp {
        risk[g as usize] += 1.0;
    }
    let mut at_risk = rows.len() as f64;
    // The Kaplan-Meier estimate just before the current time, pooled over
    // the groups of the stratum.
    let mut km = 1.0_f64;
    let mut event_times = 0;
    let mut i = 0;
    while i < rows.len() {
        let t = time[rows[i] as usize];
        let mut j = i;
        let mut deaths = 0.0;
        while j < rows.len() && time[rows[j] as usize] == t {
            deaths += f64::from(status[rows[j] as usize]);
            j += 1;
        }
        if deaths > 0.0 {
            event_times += 1;
            let w = if rho == 0.0 { 1.0 } else { km.powf(rho) };
            for idx in i..j {
                if status[rows[idx] as usize] != 0 {
                    observed[grp[idx] as usize] += w;
                }
            }
            for g in 0..k {
                if risk[g] > 0.0 {
                    expected[g] += w * deaths * risk[g] / at_risk;
                }
            }
            if at_risk > 1.0 {
                let scale = w * w * deaths * (at_risk - deaths) / (at_risk * (at_risk - 1.0));
                for a in 0..k {
                    if risk[a] == 0.0 {
                        continue;
                    }
                    let term = scale * risk[a];
                    variance[a * k + a] += term;
                    for b in 0..k {
                        variance[a * k + b] -= term * risk[b] / at_risk;
                    }
                }
            }
        }
        if rho != 0.0 {
            km *= (at_risk - deaths) / at_risk;
        }
        for &g in &grp[i..j] {
            risk[g as usize] -= 1.0;
        }
        at_risk -= (j - i) as f64;
        i = j;
    }
    event_times
}

/// `d' V^- d` for a symmetric non-negative `V` of order `k` by its
/// eigendecomposition, with the rank of `V`. The statistic is `None` for rank 0.
#[expect(clippy::cast_precision_loss, reason = "counts are far below 2^53")]
fn quadratic_form(v: &[f64], d: &[f64], event_times: usize) -> (Option<f64>, usize) {
    let k = d.len();
    let rank_tolerance = |largest: f64| 4.0 * (event_times + k) as f64 * f64::EPSILON * largest;
    match pinv_quadratic_form(v, d, rank_tolerance) {
        Some((statistic, rank)) => (Some(statistic), rank),
        None => (None, 0),
    }
}

/// The one-sample test of the observed events against the survival that a
/// reference population implies, as `survival::survdiff` with an offset.
///
/// `expected_survival` holds, for each row of the data (rows of weight 0
/// included), the probability that a subject of the reference population
/// survives to the row's time, in `(0, 1]`. The expected events are
/// `sum(-ln S)`. For `rho = 0` the variance is the same sum, and for other
/// `rho` it is `sum(1 - S^(2 rho)) / (2 rho)` with the observed-minus-expected
/// `sum(1/rho - (1/rho + status) S^rho)`. The statistic is the square of that
/// over the variance, on one degree of freedom. Strata and groups play no part.
///
/// # Errors
///
/// An input error as for [`fn@test`] where it applies, when the expected
/// survival is not as long as the data, or when a value of it is outside
/// `(0, 1]`, since a subject cannot have died under an expected survival of 0.
pub fn one_sample(
    data: &SurvData<'_>,
    expected_survival: &[f64],
    opts: &SurvdiffOptions,
) -> Result<OneSample> {
    let data = prepare(data, opts)?;
    if expected_survival.is_empty() {
        return Err(Error::from(InputError::Empty {
            what: "expected_survival",
        }));
    }
    check_lengths(&[
        ("time", data.n_rows()),
        ("expected_survival", expected_survival.len()),
    ])?;
    let status = data.followup()?.status();
    let rho = opts.rho;
    let mut observed = 0.0;
    let mut expected = 0.0;
    let mut num = 0.0;
    let mut variance = 0.0;
    for row in data.used_rows() {
        let s = expected_survival[row as usize];
        if !(s > 0.0 && s <= 1.0) {
            if !s.is_finite() {
                return Err(Error::from(InputError::NotFinite {
                    what: "expected_survival",
                    row: row as usize,
                    column: None,
                }));
            }
            return Err(Error::value(
                "the expected survival must be a probability in (0, 1]",
            ));
        }
        let d = f64::from(status[row as usize]);
        observed += d;
        expected -= s.ln();
        if rho == 0.0 {
            variance -= s.ln();
        } else {
            num += 1.0 / rho - (1.0 / rho + d) * s.powf(rho);
            variance += (1.0 - s.powf(2.0 * rho)) / (2.0 * rho);
        }
    }
    if rho == 0.0 {
        num = observed - variance;
    }
    let statistic = (variance > 0.0).then(|| num * num / variance);
    Ok(OneSample {
        observed,
        expected,
        variance,
        statistic,
        p_value: statistic.map(|x| chi_square_sf(x, 1.0)),
        n_rows: data.n_rows(),
        n_used: data.n_used(),
    })
}
