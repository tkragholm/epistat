//! The Brier score and its R-squared at chosen times, as `survival::brier`.
//!
//! [`estimate`] takes the predicted probability of the event by each time, for
//! every row, and returns the censoring-weighted mean squared difference
//! between that probability and the outcome, with the same for a model that
//! predicts the Kaplan-Meier estimate to everyone, and the ratio of the two.
//! The predictions are supplied: for a Cox model they are one minus the
//! survival curves of [`cox::predict::survfit`](crate::cox::predict::survfit)
//! read at the times.
//!
//! # Method
//!
//! Let `T_i` be the time of row `i`, `d_i` its status (1 for the event, 0 for
//! censored), `w_i` its case weight divided by the total weight, `p_i(t)` its
//! predicted probability of the event by `t` and `p_0(t)` one minus the
//! Kaplan-Meier estimate of survival. `G` is the Kaplan-Meier estimate of the
//! censoring distribution, weighted by `w`. At time `t` a row is given the
//! weight and the squared error
//!
//! ```text
//! v_i = 0                          if T_i < t and d_i = 0
//!     = w_i / G(min(T_i, t))       otherwise
//! e_i = p_i(t)^2                   if T_i > t
//!     = (d_i - p_i(t))^2           if T_i <= t
//! ```
//!
//! and the score is `B(t) = sum v_i e_i / sum v_i`. The score of the
//! Kaplan-Meier model, `B_0(t)`, is the same with `p_0` in place of `p_i`,
//! `rsquared = 1 - B(t) / B_0(t)`, and `eff_n = 1 / sum v_i^2` is the number of
//! rows that equal weights would have to carry for the same variance of the
//! weighted mean. `survival` calls this the effective sample size.
//!
//! With `ties`, every censoring time is moved right by half the smallest gap
//! between distinct times before the censoring distribution is estimated, and
//! everything above uses the moved times. A death and a censoring at one time
//! are then ordered with the death first, and a row censored at exactly `t` is
//! counted as still at risk beyond `t`. Without it the times are used as they
//! are, a row censored at `t` counts at `t` with outcome 0, and the censoring
//! distribution is the plain Kaplan-Meier estimate.
//!
//! # What is supported
//!
//! Right-censored data with one event code, case weights and no start column.
//! `survival` supports counting-process data only with an id and one starting
//! time, and that is not implemented here. Strata are not part of the score:
//! the censoring distribution is estimated over all rows, as in `brier`.
//! `brier` can also use the Efron approximation for the baseline of a Cox
//! model fitted with it (`efron = TRUE`), which is not offered here since the
//! predictions are the caller's.
//!
//! # Errors of the computed numbers
//!
//! `G` is a product of at most `m` rounded factors, so a weight has a relative
//! error of at most `(m + 3) eps`. The score is a ratio of two sums of `n`
//! non-negative terms, each with that error, so `B` has a relative error of at
//! most `(2 (m + 3) + 2 n) eps` including a second implementation's, and
//! `rsquared` inherits it multiplied by `B / B_0 / (1 - B / B_0)`.
//!
//! # References
//!
//! 1. Therneau TM. A Package for Survival Analysis in R. R package `survival`
//!    3.8: `brier` and its help page.
//! 2. Graf E, Schmoor C, Sauerbrei W, Schumacher M. Assessment and comparison
//!    of prognostic classification schemes for survival data. Stat Med
//!    1999;18(17-18):2529-45.
//!    doi:10.1002/(SICI)1097-0258(19990915/30)18:17/18<2529::AID-SIM274>3.0.CO;2-5
//! 3. Gerds TA, Schumacher M. Consistent estimation of the expected Brier
//!    score in general survival models with right-censored event times.
//!    Biom J 2006;48(6):1029-40. doi:10.1002/bimj.200610301

use faer::MatRef;
use rayon::prelude::*;

use super::censoring::CensoringKm;
use crate::error::{Error, Result};
use crate::order::sort_by_key;
use crate::surv::SurvData;
use crate::timefix::TimeFix;

/// What [`estimate`] takes besides the data, the predictions and the times.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrierOptions {
    /// The rule that makes times equal when they differ by rounding error,
    /// applied to the data first. The default is that of `survival::aeqSurv`.
    pub timefix: TimeFix,
    /// Move the censoring times right by half the smallest gap, `ties = TRUE`
    /// of `brier`.
    pub ties: bool,
}

impl Default for BrierOptions {
    fn default() -> Self {
        Self {
            timefix: TimeFix::default(),
            ties: true,
        }
    }
}

/// The result of [`estimate`].
#[derive(Debug, Clone, PartialEq)]
pub struct Brier {
    /// The times, as given.
    pub times: Vec<f64>,
    /// The Brier score of the predictions at each time.
    pub brier: Vec<f64>,
    /// The Brier score of the Kaplan-Meier model at each time.
    pub null_brier: Vec<f64>,
    /// `1 - brier / null_brier`, `None` where the Kaplan-Meier model has score 0.
    pub rsquared: Vec<Option<f64>>,
    /// One minus the Kaplan-Meier estimate of survival at each time,
    /// `detail = TRUE` of `brier`.
    pub p0: Vec<f64>,
    /// The effective sample size at each time.
    pub eff_n: Vec<f64>,
    /// The number of rows given.
    pub n_rows: usize,
    /// The number of rows of positive weight.
    pub n_used: usize,
}

/// The distinct event times of the rows of positive weight after the tie rule,
/// ascending: the times `brier` reports at when none are given.
///
/// # Errors
///
/// [`Error::Input`] as for [`SurvData::with_timefix`] and
/// [`SurvData::followup`].
pub fn event_times(data: &SurvData<'_>, timefix: TimeFix) -> Result<Vec<f64>> {
    let data = data.clone().with_timefix(timefix)?;
    let surv = data.followup()?;
    let mut times: Vec<f64> = data
        .used_rows()
        .filter(|&r| surv.status()[r as usize] != 0)
        .map(|r| surv.stop()[r as usize])
        .collect();
    times.sort_by(f64::total_cmp);
    times.dedup();
    Ok(times)
}

/// One minus the weighted Kaplan-Meier estimate at each of `times`, with
/// events before censorings at a tied time. `time` is ascending. The estimate
/// is the product limit of the events, which [`CensoringKm`] computes for the
/// flag it is given, here the event.
fn km_risk(time: &[f64], event: &[bool], weight: &[f64], times: &[f64]) -> Vec<f64> {
    let km = CensoringKm::new(time, event, weight, false);
    times.iter().map(|&t| 1.0 - km.at(t)).collect()
}

/// The Brier score at `times`. `predicted` has one row per row of the data and
/// one column per time, the probability of the event by that time.
///
/// # Errors
///
/// [`Error::Input`] when the data is interval-censored, has a start column or
/// more than one event code, or has fewer than two distinct times; when
/// `times` is empty or has a value that is not finite; when `predicted` has
/// another shape than rows by times or a value outside `[0, 1]`; or when
/// `timefix` is refused as for [`SurvData::with_timefix`].
pub fn estimate(
    data: &SurvData<'_>,
    predicted: MatRef<'_, f64>,
    times: &[f64],
    opts: &BrierOptions,
) -> Result<Brier> {
    if times.is_empty() {
        return Err(Error::input("times: at least one time is needed"));
    }
    if let Some(position) = times.iter().position(|t| !t.is_finite()) {
        return Err(Error::input(format!(
            "times: value {position} is not finite"
        )));
    }
    let data = data.clone().with_timefix(opts.timefix)?;
    let surv = data.followup()?;
    if surv.start().is_some() {
        return Err(Error::input(
            "brier is not implemented for data with delayed entry; give follow-up from time 0",
        ));
    }
    if surv.n_codes() != 1 {
        return Err(Error::input(format!(
            "status has {} event codes, and the Brier score takes one; pass one cause as 0 and 1",
            surv.n_codes()
        )));
    }
    let n = data.n_rows();
    if predicted.nrows() != n || predicted.ncols() != times.len() {
        return Err(Error::input(format!(
            "predicted is {} by {}, but there are {n} rows and {} times",
            predicted.nrows(),
            predicted.ncols(),
            times.len()
        )));
    }
    for j in 0..times.len() {
        for i in 0..n {
            let p = predicted[(i, j)];
            if !(0.0..=1.0).contains(&p) {
                return Err(Error::input(format!(
                    "predicted: row {i}, column {j} is {p}, which is not a probability"
                )));
            }
        }
    }

    let stop = surv.stop();
    let status = surv.status();
    let mut unique = stop.to_vec();
    unique.sort_by(f64::total_cmp);
    unique.dedup();
    if unique.len() < 2 {
        return Err(Error::input(
            "the follow-up times take one value, so no gap between them exists",
        ));
    }
    let min_gap = unique
        .windows(2)
        .map(|w| w[1] - w[0])
        .fold(f64::INFINITY, f64::min);

    let rows: Vec<u32> = data.used_rows().collect();
    let case_weight: Vec<f64> = rows
        .iter()
        .map(|&r| data.weight().map_or(1.0, |w| w[r as usize]))
        .collect();
    let total: f64 = case_weight.iter().sum();
    if !total.is_finite() {
        return Err(Error::input(
            "the case weights sum to more than a double can hold",
        ));
    }
    let w: Vec<f64> = case_weight.iter().map(|&c| c / total).collect();
    // The times the censoring distribution sees.
    let dtime: Vec<f64> = rows
        .iter()
        .map(|&r| {
            let t = stop[r as usize];
            if opts.ties && status[r as usize] == 0 {
                t + min_gap / 2.0
            } else {
                t
            }
        })
        .collect();
    let event: Vec<bool> = rows.iter().map(|&r| status[r as usize] != 0).collect();

    let mut order: Vec<usize> = (0..rows.len()).collect();
    sort_by_key(&mut order, |i| dtime[i]);
    let s_time: Vec<f64> = order.iter().map(|&i| dtime[i]).collect();
    let s_cens: Vec<bool> = order.iter().map(|&i| !event[i]).collect();
    let s_weight: Vec<f64> = order.iter().map(|&i| w[i]).collect();
    let g = CensoringKm::new(&s_time, &s_cens, &s_weight, false);

    // The Kaplan-Meier model is estimated on the times as they are.
    let mut by_stop: Vec<usize> = (0..rows.len()).collect();
    by_stop.sort_unstable_by(|&a, &b| {
        stop[rows[a] as usize]
            .total_cmp(&stop[rows[b] as usize])
            .then(a.cmp(&b))
    });
    let k_time: Vec<f64> = by_stop.iter().map(|&i| stop[rows[i] as usize]).collect();
    let k_event: Vec<bool> = by_stop.iter().map(|&i| event[i]).collect();
    let k_weight: Vec<f64> = by_stop.iter().map(|&i| w[i]).collect();
    let p0 = km_risk(&k_time, &k_event, &k_weight, times);

    let columns: Vec<(f64, f64, f64)> = times
        .par_iter()
        .enumerate()
        .map(|(j, &t)| {
            let (mut sum_v, mut sum_v2, mut sum0, mut sum1) = (0.0, 0.0, 0.0, 0.0);
            for (i, &row) in rows.iter().enumerate() {
                let d = if event[i] { 1.0 } else { 0.0 };
                let v = if dtime[i] < t && !event[i] {
                    0.0
                } else {
                    w[i] / g.at(dtime[i].min(t))
                };
                let p1 = predicted[(row as usize, j)];
                let (e0, e1) = if dtime[i] > t {
                    (p0[j] * p0[j], p1 * p1)
                } else {
                    ((d - p0[j]) * (d - p0[j]), (d - p1) * (d - p1))
                };
                sum_v += v;
                sum_v2 += v * v;
                sum0 += v * e0;
                sum1 += v * e1;
            }
            (sum0 / sum_v, sum1 / sum_v, 1.0 / sum_v2)
        })
        .collect();

    let null_brier: Vec<f64> = columns.iter().map(|c| c.0).collect();
    let brier: Vec<f64> = columns.iter().map(|c| c.1).collect();
    let rsquared = brier
        .iter()
        .zip(&null_brier)
        .map(|(&b, &b0)| (b0 > 0.0).then(|| 1.0 - b / b0))
        .collect();
    Ok(Brier {
        times: times.to_vec(),
        brier,
        null_brier,
        rsquared,
        p0,
        eff_n: columns.iter().map(|c| c.2).collect(),
        n_rows: n,
        n_used: data.n_used(),
    })
}
