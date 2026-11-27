//! The expected survival of a cohort: Ederer I, Hakulinen and the conditional
//! (Ederer II) curve.

#![expect(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "times are compared with the values they were copied from, and counts of rows are far below 2^53"
)]

use epistat_core::{InputError, check};

use super::ExpectedHazard;
use super::expected::Pieces;
use crate::check::{check_ascending, refuse_non_finite};
use crate::error::{Error, Result};
use crate::par;
use crate::surv::SurvData;

const BLOCK: usize = 1024;
const WAVE: usize = 64;
/// The most pairs of a person in the set and a time point a sweep visits.
pub(crate) const SWEEP_LIMIT: f64 = 3e8;

/// Which expected survival of a cohort.
#[derive(Clone, Copy, Debug, PartialEq)]
#[non_exhaustive]
pub enum ExpectedMethod<'a> {
    /// Ederer I, the exact method of `survival::survexp(method = "ederer")`:
    /// the mean of the expected survival of every row, each followed to the
    /// time asked for whether or not it was followed that long.
    EdererI,
    /// Hakulinen, `survexp(method = "hakulinen")`: a row is in the cohort
    /// until its potential follow-up, the time it would have been followed to
    /// had it not died, such as the end of the study less the date of
    /// diagnosis. One value per row of the data.
    Hakulinen {
        /// The potential follow-up of each row, finite and not negative.
        potential: &'a [f64],
    },
    /// The conditional curve of Ederer II, `survexp(method =
    /// "conditional")`: the expected hazard is the mean over the rows at risk,
    /// with their observed follow-up and entry times.
    Conditional,
}

/// The expected survival of one stratum at the times asked for.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ExpectedCurve {
    /// The level of the stratum, 0 when the data has no strata.
    pub stratum: usize,
    /// The times asked for.
    pub times: Vec<f64>,
    /// The expected survival at each time, `None` when no row is in the set
    /// that defines it: no one at risk for the conditional curve, no one with
    /// that much potential follow-up for Hakulinen's.
    pub survival: Vec<Option<f64>>,
    /// The number of rows in the set at each time: every row for Ederer I,
    /// the rows with at least that much potential follow-up for Hakulinen,
    /// the rows at risk for the conditional curve.
    pub n: Vec<usize>,
}

/// The rows of `data` that are used, by stratum, with the checks shared by
/// the estimators of the cohort.
pub(crate) fn rows_by_stratum(
    data: &SurvData<'_>,
    expected: &ExpectedHazard,
) -> Result<Vec<Vec<usize>>> {
    if expected.n_rows() != data.n_rows() {
        return Err(Error::mismatch(format!(
            "the expected hazards are for {} rows and the data has {}",
            expected.n_rows(),
            data.n_rows()
        )));
    }
    if let Some(w) = data.weight()
        && w.iter().any(|&v| v != 0.0 && v != 1.0)
    {
        return Err(Error::value(
            "the expected survival takes weights of 0 and 1 only, which select rows",
        ));
    }
    let surv = data.followup()?;
    let (start, stop) = (surv.start(), surv.stop());
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); data.n_strata().max(1)];
    for row in data.used_rows() {
        let i = row as usize;
        if start.is_some_and(|s| s[i] < 0.0) {
            return Err(Error::value(format!("row {i} has a negative entry time")));
        }
        if expected.horizon(i) < stop[i] {
            return Err(Error::value(format!(
                "the expected hazard of row {i} ends at {} before its follow-up at {}",
                expected.horizon(i),
                stop[i]
            )));
        }
        let g = data.stratum().map_or(0, |s| s[i] as usize);
        groups[g].push(i);
    }
    Ok(groups)
}

/// The expected survival of each stratum of a cohort at `times`.
///
/// `data` is the follow-up of the cohort with the status of the rows, in the
/// time scale of `expected`, and its strata give one curve each. The three
/// methods are those of `survival::survexp`, written for hazards that are
/// constant between the cuts of the rate table, so that they need no grid:
///
/// ```text
/// Ederer I      S(t) = (1 / n) sum_i S*_i(t)
/// Hakulinen     S(t) = prod_k  sum_{c_i >= u_k} S*_i(u_k) / sum_{c_i >= u_k} S*_i(u_{k-1})
/// conditional   S(t) = exp(-sum_k (1 / n_k) sum_{i at risk in (u_{k-1}, u_k]} (H*_i(u_k) - H*_i(u_{k-1})))
/// ```
///
/// with `S*_i = exp(-H*_i)` the expected survival of row `i` from time 0,
/// `c_i` its potential follow-up, and `u_0 = 0 < u_1 < ...` the distinct
/// potential follow-up times (Hakulinen) or entry and exit times
/// (conditional) together with the times asked for, up to `t`. Between two
/// such points the set of rows is fixed, and Hakulinen's hazard
/// `sum S*_i lambda*_i / sum S*_i` integrates to the log of the ratio of the
/// sums, which is formed as `ln_1p(sum_i S*_i(a) expm1(-(H*_i(b) - H*_i(a)))
/// / sum_i S*_i(a))` so that a small step keeps its precision; the
/// conditional hazard, the mean of `lambda*_i` over the rows at risk,
/// integrates to the mean of the increments. `survexp` forms the same
/// quantities on the grid of its `times`, weighting each row by the time it
/// spends in an interval, and gives these values when its times include
/// every point `u_k`, since every row in the set then spends the whole
/// interval in it. A row entering at `t` is at risk after `t`, and one
/// leaving at `t` is at risk at it.
///
/// Ederer I and Hakulinen follow rows beyond their observed follow-up, as
/// [`ExpectedHazard::cumulative`] does, so the table must cover that time or
/// have extended ends. They take no delayed entry.
///
/// # Scale
///
/// Ederer I walks each row once, in parallel blocks summed in order, and costs
/// `O(n (p + m))` for `p` pieces a row and `m` times. The other two visit
/// every row in the set at every point `u_k`, serially, and are refused above
/// 3e8 such pairs; rounding the potential follow-up to whole days or months
/// reduces the points.
///
/// # Errors
///
/// An input error for expected hazards of another number of rows than the
/// data, weights other than 0 and 1, times that are not finite, not negative
/// and increasing, an entry before time 0, follow-up beyond what `expected`
/// covers, a potential follow-up of another length than the data or not
/// finite and not negative, delayed entry for Ederer I or Hakulinen, a row
/// whose walk to a time needed leaves the table, and work above the limit.
pub fn expected_survival(
    data: &SurvData<'_>,
    expected: &ExpectedHazard,
    times: &[f64],
    method: ExpectedMethod<'_>,
) -> Result<Vec<ExpectedCurve>> {
    refuse_non_finite(times, "times")?;
    if let Some(row) = times.iter().position(|t| *t < 0.0) {
        return Err(InputError::Domain {
            what: "times",
            value: times[row],
            domain: "[0, inf)",
            row: Some(row),
        }
        .into());
    }
    check_ascending(times, "times", true)?;
    let groups = rows_by_stratum(data, expected)?;
    let surv = data.followup()?;
    let (start, stop) = (surv.start(), surv.stop());
    if !matches!(method, ExpectedMethod::Conditional)
        && let Some(s) = start
        && let Some(i) = groups.iter().flatten().find(|&&i| s[i] > 0.0)
    {
        return Err(Error::not_applicable(format!(
            "row {i} has delayed entry, which Ederer I and Hakulinen do not take"
        )));
    }
    if let ExpectedMethod::Hakulinen { potential } = method {
        check::same_length("potential", data.n_rows(), potential.len())?;
        if let Some(i) = groups
            .iter()
            .flatten()
            .find(|&&i| !(potential[i].is_finite() && potential[i] >= 0.0))
        {
            return Err(Error::value(format!(
                "the potential follow-up of row {i} is not finite and not negative"
            )));
        }
    }
    groups
        .into_iter()
        .enumerate()
        .map(|(stratum, rows)| {
            let (survival, n) = match method {
                ExpectedMethod::EdererI => ederer_one(&rows, expected, times)?,
                ExpectedMethod::Hakulinen { potential } => {
                    let entry = |_: usize| 0.0;
                    let exit = |i: usize| potential[i];
                    sweep(&rows, expected, times, &entry, &exit, Weighting::Survival)?
                }
                ExpectedMethod::Conditional => {
                    let entry = |i: usize| start.map_or(0.0, |s| s[i]);
                    let exit = |i: usize| stop[i];
                    sweep(&rows, expected, times, &entry, &exit, Weighting::Equal)?
                }
            };
            Ok(ExpectedCurve {
                stratum,
                times: times.to_vec(),
                survival,
                n,
            })
        })
        .collect()
}

/// Ederer I: the mean of `exp(-H*_i(t))` over the rows, each walked to the
/// last time.
pub(crate) fn ederer_one(
    rows: &[usize],
    expected: &ExpectedHazard,
    times: &[f64],
) -> Result<(Vec<Option<f64>>, Vec<usize>)> {
    let t_max = times.last().copied().unwrap_or(0.0);
    let sums = par::try_ordered_reduce(
        rows.len(),
        BLOCK,
        WAVE,
        |range| {
            let mut sums = vec![0.0_f64; times.len()];
            let mut reader = expected.reader();
            for &i in &rows[range] {
                let end = t_max.max(expected.horizon(i));
                let pieces = reader.row_until(i, end).ok_or_else(|| {
                    Error::out_of_range(format!(
                        "row {i} leaves the rate table before time {t_max}, which has no expected hazard"
                    ))
                })?;
                for (s, &t) in sums.iter_mut().zip(times) {
                    *s += (-pieces.cumulative_within(t)).exp();
                }
            }
            Ok::<_, Error>(sums)
        },
        |total: &mut Vec<f64>, part| {
            for (t, p) in total.iter_mut().zip(part) {
                *t += p;
            }
        },
    )?;
    let n = rows.len();
    let survival = sums
        .iter()
        .map(|&s| (n > 0).then(|| s / n as f64))
        .collect();
    Ok((survival, vec![n; times.len()]))
}

/// How the rows of the set are averaged in [`sweep`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Weighting {
    /// Each row by its expected survival, Hakulinen's hazard.
    Survival,
    /// Each row equally, the conditional hazard.
    Equal,
}

/// The expected survival of a set of rows that changes at the entry and exit
/// times given, evaluated at `times`: a row is in the set on
/// `(entry(i), exit(i)]`.
pub(crate) fn sweep(
    rows: &[usize],
    expected: &ExpectedHazard,
    times: &[f64],
    entry: &dyn Fn(usize) -> f64,
    exit: &dyn Fn(usize) -> f64,
    weighting: Weighting,
) -> Result<(Vec<Option<f64>>, Vec<usize>)> {
    let mut points: Vec<f64> = rows.iter().map(|&i| exit(i)).collect();
    points.extend(rows.iter().map(|&i| entry(i)).filter(|&s| s > 0.0));
    points.extend(times.iter().copied().filter(|&t| t > 0.0));
    points.sort_by(f64::total_cmp);
    points.dedup();
    let work: f64 = rows
        .iter()
        .map(|&i| {
            let lo = points.partition_point(|&p| p <= entry(i));
            let hi = points.partition_point(|&p| p <= exit(i));
            (hi - lo) as f64
        })
        .sum();
    if work > SWEEP_LIMIT {
        return Err(Error::options(format!(
            "the expected survival would visit {work:.3e} pairs of a person and a time point, \
             above the limit of {SWEEP_LIMIT:.0e}: round the times"
        )));
    }
    let mut by_entry: Vec<usize> = rows.to_vec();
    by_entry.sort_by(|&a, &b| entry(a).total_cmp(&entry(b)).then(a.cmp(&b)));
    let mut next_entrant = 0;
    let mut active: Vec<(usize, f64, (usize, usize))> = Vec::new();
    let mut reader = expected.reader();
    let mut pieces = Pieces::default();
    let mut log_survival = 0.0_f64;
    let mut previous = 0.0_f64;
    let mut survival = Vec::with_capacity(times.len());
    let mut counts = Vec::with_capacity(times.len());
    let mut next_time = 0;
    // A time asked for at 0: everyone has survived, and the set is the rows
    // with an exit after 0 that enter at 0.
    while next_time < times.len() && times[next_time] == 0.0 {
        let n = rows
            .iter()
            .filter(|&&i| entry(i) == 0.0 && exit(i) >= 0.0)
            .count();
        survival.push((n > 0).then_some(1.0));
        counts.push(n);
        next_time += 1;
    }
    for &b in &points {
        while next_entrant < by_entry.len() && entry(by_entry[next_entrant]) < b {
            let i = by_entry[next_entrant];
            next_entrant += 1;
            if exit(i) < b {
                continue;
            }
            let at = reader.append_until(i, exit(i).max(expected.horizon(i)), &mut pieces).ok_or_else(|| {
                Error::out_of_range(format!(
                    "row {i} leaves the rate table before time {}, which has no expected hazard",
                    exit(i)
                ))
            })?;
            let h = pieces
                .slice(at.clone())
                .cumulative_within(previous.max(entry(i)));
            active.push((i, h, (at.start, at.end)));
        }
        let n = active.len();
        if n > 0 {
            let (mut w_a, mut w_step, mut dh) = (0.0_f64, 0.0_f64, 0.0_f64);
            for (_, h_a, at) in &mut active {
                let h_b = pieces.slice(at.0..at.1).cumulative_within(b);
                match weighting {
                    Weighting::Survival => {
                        let s_a = (-*h_a).exp();
                        w_a += s_a;
                        w_step += s_a * (-(h_b - *h_a)).exp_m1();
                    }
                    Weighting::Equal => dh += h_b - *h_a,
                }
                *h_a = h_b;
            }
            log_survival += match weighting {
                Weighting::Survival => (w_step / w_a).ln_1p(),
                Weighting::Equal => -dh / n as f64,
            };
        }
        if next_time < times.len() && times[next_time] == b {
            survival.push((n > 0).then(|| log_survival.exp()));
            counts.push(n);
            next_time += 1;
        }
        active.retain(|&(i, _, _)| exit(i) > b);
        previous = b;
    }
    Ok((survival, counts))
}
