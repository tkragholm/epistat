//! Redistribute-to-the-right weights, as `survival::rttright`.
//!
//! [`estimate`] gives every row a weight at each chosen time such that the
//! weighted mean of any function of the status at that time is the
//! Kaplan-Meier estimate (or, with several event codes, the Aalen-Johansen
//! estimate) of its expectation. A row that is censored before the time has
//! weight 0, and its weight has gone to the rows to its right, in proportion
//! to their own weights. The weights turn a censored sample into a complete
//! one: a regression, a mean or a score computed with them needs no further
//! allowance for censoring.
//!
//! # Method
//!
//! Let `G` be the Kaplan-Meier estimate of the censoring distribution of a
//! stratum, `G(t-)` its value just before `t`, and `w_i` the case weight of
//! row `i`, divided by the total weight of its stratum when
//! [`RttrightOptions::renorm`] is set. For a row with time `T_i` and a
//! reporting time `t` the weight is
//!
//! ```text
//! w_i / G(min(T_i, t)-)   if row i had an event, or T_i >= t
//! 0                       if row i was censored before t.
//! ```
//!
//! Without a reporting time the weights are those of the end of follow-up:
//! `w_i / G(T_i-)` for a row with an event and 0 for a row without. The
//! weights of a stratum sum to the stratum's total at every time, so a
//! weighted mean over the stratum is a mean of a complete sample.
//!
//! A death and a censoring at one time are ordered with the death first: the
//! censoring is in the risk set of nobody who died at that time, and it does
//! not count in `G(t-)` for that time. `survival` makes this so by moving
//! every censoring time right by half the smallest gap between times.
//!
//! # What is supported
//!
//! Right-censored data with one or several event codes, case weights and
//! strata. Rows of weight 0 have weight 0 in the result, and do not enter the
//! censoring distribution.
//!
//! # Counting-process data
//!
//! With a start column the data must have an id column, the rows of a subject
//! must follow each other without gap or overlap, every subject must start at
//! the same time `s`, and a subject's rows must have one case weight. Only a
//! subject's last row may end in an event, and the last row says whether the
//! subject was censored. `rttright` accepts the same data, and `survcheck`
//! finds none of its flags. The subject is the unit: it is what is divided by
//! in `renorm`, and the censoring distribution is that of the subjects' last
//! rows. The weight of a subject at a reporting time `t` goes to one of its
//! rows, the one with start `< t <= stop` when the subject was still under
//! follow-up at `t`, and the last row when it had an event before `t`. The
//! other rows have weight 0. Without a reporting time, only the last row of a
//! subject with an event has a weight. A reporting time must be after `s`.
//!
//! `rttright` in `survival` 3.8-12 is wrong for these data with reporting times:
//! its branch for counting-process data selects the last row of each subject
//! with `Y[, 2] > 0`, which is the stop time, where the status `Y[, 3]` is
//! meant. Every last row then has a weight, and the weights of a column sum to
//! more than 1 whenever a subject is censored before the time, or has a last
//! row that starts at or after the time but stops later. The weights without
//! reporting times use the status and are right. The R script reproduces both
//! defects, and the test asserts them.
//!
//! # Errors of the computed numbers
//!
//! `G` is a product of at most `m` factors, each rounded once, and the
//! reciprocal of a value of `G` is formed once, so a weight differs from the
//! exact value by a relative error of at most `(m + 3) eps`, where `m` is the
//! number of distinct censoring times of its stratum. Dividing by the total
//! weight adds `n eps` for a sum of `n` weights.
//!
//! # References
//!
//! 1. Therneau TM. A Package for Survival Analysis in R. R package `survival`
//!    3.8: `rttright` and its help page.
//! 2. Efron B. The two sample problem with censored data. In: Proceedings of
//!    the Fifth Berkeley Symposium on Mathematical Statistics and
//!    Probability, vol. 4, 1967, p. 831-853. The redistribution to the right.
//! 3. Gerds TA, Kattan MW, Schumacher M, Yu C. Estimating a time-dependent
//!    concordance index for survival prediction models with covariate
//!    dependent censoring. Stat Med 2013;32(13):2173-84.
//!    doi:10.1002/sim.5681

#![expect(
    clippy::float_cmp,
    reason = "a common start and a case weight are equal when they are the same value"
)]

use epistat_linalg::Mat;

use crate::survcheck::survcheck;

use super::censoring::CensoringKm;
use crate::error::{Error, Result};
use crate::surv::SurvData;
use crate::timefix::TimeFix;

/// What [`estimate`] takes besides the data and the times.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RttrightOptions {
    /// The rule that makes times equal when they differ by rounding error,
    /// applied to the data first. The default is that of `survival::aeqSurv`,
    /// which `rttright` applies (`timefix = TRUE`).
    pub timefix: TimeFix,
    /// Divide the case weights of each stratum by their total, so that the
    /// weights of a stratum sum to 1 (`renorm = TRUE`, as in `rttright`).
    pub renorm: bool,
}

impl Default for RttrightOptions {
    fn default() -> Self {
        Self {
            timefix: TimeFix::default(),
            renorm: true,
        }
    }
}

/// The result of [`estimate`].
#[derive(Debug, Clone, PartialEq)]
pub struct Rttright {
    /// The weights, one row per row of the data and one column per reporting
    /// time, or one column when no time was given. Rows of weight 0 have
    /// weight 0.
    pub weights: Mat<f64>,
    /// The number of rows given.
    pub n_rows: usize,
    /// The number of rows of positive weight.
    pub n_used: usize,
}

/// Redistribute-to-the-right weights at `times`, or at the end of follow-up
/// when `times` is `None`, as `rttright(times = )`. See the module
/// documentation for the definition.
///
/// # Errors
///
/// [`Error::Input`] when the data is interval-censored, when a time is not
/// finite, when `timefix` is refused as for [`SurvData::with_timefix`], or when
/// a stratum with positive weights has a total weight that overflows. With a
/// start column also when there is no id column or an `istate` column, when the
/// rows of a subject overlap, have a gap, or have an event before the last row,
/// when subjects do not start at one time, when a subject has several case
/// weights or strata, or when a reporting time is not after the start.
pub fn estimate(
    data: &SurvData<'_>,
    times: Option<&[f64]>,
    opts: &RttrightOptions,
) -> Result<Rttright> {
    if let Some(position) = times.and_then(|t| t.iter().position(|v| !v.is_finite())) {
        return Err(Error::input(format!(
            "times: value {position} is not finite"
        )));
    }
    if times.is_some_and(<[f64]>::is_empty) {
        return Err(Error::input("times: give at least one time, or none"));
    }
    let data = data.clone().with_timefix(opts.timefix)?;
    let surv = data.followup()?;
    if surv.start().is_some() {
        return counting(&data, times, opts);
    }
    let n = data.n_rows();
    let stop = surv.stop();
    let status = surv.status();
    let case_weight = data.weight();
    let n_columns = times.map_or(1, <[f64]>::len);
    let mut weights = Mat::<f64>::zeros(n, n_columns);

    let n_strata = data.n_strata().max(1);
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); n_strata];
    for row in data.used_rows() {
        let s = data.stratum().map_or(0, |s| s[row as usize] as usize);
        members[s].push(row);
    }
    for rows in &mut members {
        rows.sort_unstable_by(|&a, &b| {
            stop[a as usize]
                .total_cmp(&stop[b as usize])
                .then(a.cmp(&b))
        });
    }
    for rows in &members {
        if rows.is_empty() {
            continue;
        }
        let time: Vec<f64> = rows.iter().map(|&r| stop[r as usize]).collect();
        let censored: Vec<bool> = rows.iter().map(|&r| status[r as usize] == 0).collect();
        let mut w: Vec<f64> = rows
            .iter()
            .map(|&r| case_weight.map_or(1.0, |c| c[r as usize]))
            .collect();
        if opts.renorm {
            let total: f64 = w.iter().sum();
            if !total.is_finite() {
                return Err(Error::input(
                    "the case weights of a stratum sum to more than a double can hold",
                ));
            }
            for value in &mut w {
                *value /= total;
            }
        }
        let g = CensoringKm::new(&time, &censored, &w, true);
        for (position, &row) in rows.iter().enumerate() {
            let r = row as usize;
            let event = !censored[position];
            let g_row = g.before(time[position]);
            match times {
                None => {
                    if event {
                        weights[(r, 0)] = w[position] / g_row;
                    }
                }
                Some(times) => {
                    for (j, &t) in times.iter().enumerate() {
                        let g_t = g.before(t);
                        weights[(r, j)] = if event {
                            w[position] / g_row.max(g_t)
                        } else if time[position] >= t {
                            w[position] / g_t
                        } else {
                            0.0
                        };
                    }
                }
            }
        }
    }
    Ok(Rttright {
        weights,
        n_rows: n,
        n_used: data.n_used(),
    })
}

/// The rows of one subject, in time order, and what its last row says.
struct Subject {
    /// The rows, ascending by stop and then start.
    rows: Vec<u32>,
    stratum: usize,
    weight: f64,
}

/// The weights of counting-process data, see the module documentation.
fn counting(
    data: &SurvData<'_>,
    times: Option<&[f64]>,
    opts: &RttrightOptions,
) -> Result<Rttright> {
    let surv = data.followup()?;
    let (start, stop, status) = (
        surv.start().expect("counting data has a start column"),
        surv.stop(),
        surv.status(),
    );
    let ids = data.id().ok_or_else(|| {
        Error::input("data with delayed entry needs an id column, as rttright does")
    })?;
    if data.istate().is_some() {
        return Err(Error::input(
            "rttright is for one starting state: the data has an istate column",
        ));
    }
    let check = survcheck(data)?;
    check.enforce(crate::survcheck::Policy::Strict)?;
    if !check.gap.rows.is_empty() {
        return Err(Error::input(format!(
            "a subject has a gap between rows, first at row {}; rttright refuses data that              survcheck flags",
            check.gap.rows[0]
        )));
    }
    let n = data.n_rows();
    let n_ids = data.n_ids();
    let mut by_id: Vec<Subject> = (0..n_ids)
        .map(|_| Subject {
            rows: Vec::new(),
            stratum: 0,
            weight: 0.0,
        })
        .collect();
    for row in 0..n {
        by_id[ids[row] as usize]
            .rows
            .push(u32::try_from(row).expect("the builder limits rows to u32::MAX"));
    }
    let case_weight = |r: u32| data.weight().map_or(1.0, |c| c[r as usize]);
    let stratum_of = |r: u32| data.stratum().map_or(0, |s| s[r as usize] as usize);
    let mut common_start: Option<f64> = None;
    for subject in &mut by_id {
        subject.rows.sort_unstable_by(|&a, &b| {
            stop[a as usize]
                .total_cmp(&stop[b as usize])
                .then(start[a as usize].total_cmp(&start[b as usize]))
                .then(a.cmp(&b))
        });
        let first = subject.rows[0];
        let s = start[first as usize];
        if *common_start.get_or_insert(s) != s {
            return Err(Error::input(
                "subjects start at different times; rttright needs one common start",
            ));
        }
        if subject.rows[..subject.rows.len() - 1]
            .iter()
            .any(|&r| status[r as usize] != 0)
        {
            return Err(Error::input(format!(
                "a row that is not the last of its subject ends in an event, near row {};                  only the last row of a subject may",
                subject.rows[0]
            )));
        }
        subject.weight = case_weight(first);
        subject.stratum = stratum_of(first);
        if subject
            .rows
            .iter()
            .any(|&r| case_weight(r) != subject.weight || stratum_of(r) != subject.stratum)
        {
            return Err(Error::input(format!(
                "a subject has several case weights or strata, near row {}",
                subject.rows[0]
            )));
        }
    }
    let origin = common_start.expect("there is at least one row");
    if let Some(times) = times {
        if let Some(t) = times.iter().find(|&&t| t <= origin) {
            return Err(Error::input(format!(
                "the reporting time {t} is not after the common start {origin}"
            )));
        }
    }
    let n_columns = times.map_or(1, <[f64]>::len);
    let mut weights = Mat::<f64>::zeros(n, n_columns);
    let n_strata = data.n_strata().max(1);
    let mut members: Vec<Vec<usize>> = vec![Vec::new(); n_strata];
    for (id, subject) in by_id.iter().enumerate() {
        if subject.weight > 0.0 {
            members[subject.stratum].push(id);
        }
    }
    for subjects in &mut members {
        if subjects.is_empty() {
            continue;
        }
        let end_of = |id: usize| {
            *by_id[id]
                .rows
                .last()
                .expect("a subject has at least one row") as usize
        };
        subjects.sort_unstable_by(|&a, &b| {
            stop[end_of(a)]
                .total_cmp(&stop[end_of(b)])
                .then(end_of(a).cmp(&end_of(b)))
        });
        let time: Vec<f64> = subjects.iter().map(|&id| stop[end_of(id)]).collect();
        let censored: Vec<bool> = subjects.iter().map(|&id| status[end_of(id)] == 0).collect();
        let mut w: Vec<f64> = subjects.iter().map(|&id| by_id[id].weight).collect();
        if opts.renorm {
            let total: f64 = w.iter().sum();
            if !total.is_finite() {
                return Err(Error::input(
                    "the case weights of a stratum sum to more than a double can hold",
                ));
            }
            for value in &mut w {
                *value /= total;
            }
        }
        let g = CensoringKm::new(&time, &censored, &w, true);
        for (k, &id) in subjects.iter().enumerate() {
            let rows = &by_id[id].rows;
            let last = *rows.last().expect("a subject has at least one row") as usize;
            let event = !censored[k];
            let g_end = g.before(time[k]);
            match times {
                None => {
                    if event {
                        weights[(last, 0)] = w[k] / g_end;
                    }
                }
                Some(times) => {
                    for (j, &t) in times.iter().enumerate() {
                        let g_t = g.before(t);
                        if time[k] >= t {
                            // Under follow-up at t: the row that covers t.
                            let covering = rows
                                .iter()
                                .copied()
                                .find(|&r| start[r as usize] < t && stop[r as usize] >= t)
                                .expect("chained rows cover every time after the start");
                            weights[(covering as usize, j)] = if event {
                                w[k] / g_end.max(g_t)
                            } else {
                                w[k] / g_t
                            };
                        } else if event {
                            weights[(last, j)] = w[k] / g_end.max(g_t);
                        }
                    }
                }
            }
        }
    }
    Ok(Rttright {
        weights,
        n_rows: n,
        n_used: data.n_used(),
    })
}
