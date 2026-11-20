//! The rows of the unweighted metrics, sorted by time, and the checks of
//! their inputs.

#![expect(
    clippy::float_cmp,
    reason = "a case weight is 0 or 1 exactly or it is refused"
)]

use faer::MatRef;

use crate::error::{Error, Result};
use crate::surv::SurvData;
use crate::timefix::TimeFix;

/// What happened to a row at its time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    /// Left follow-up without an event.
    Censored,
    /// The event of interest.
    Cause,
    /// Another event, which ends follow-up for the event of interest.
    Competing,
}

/// The used rows of the data, sorted by time.
#[derive(Debug, Clone)]
pub(super) struct Sorted {
    /// Ascending, ties broken by the row number so the order is fixed.
    pub time: Vec<f64>,
    pub kind: Vec<Kind>,
    /// The row of the input at each position.
    pub row: Vec<u32>,
    /// The number of rows given.
    pub n_rows: usize,
    /// The largest time of the used rows.
    pub max_time: f64,
}

impl Sorted {
    pub(super) fn n(&self) -> usize {
        self.time.len()
    }

    pub(super) fn is_censored(&self) -> Vec<bool> {
        self.kind.iter().map(|&k| k == Kind::Censored).collect()
    }

    /// The number of positions with `time <= t`.
    pub(super) fn count_through(&self, t: f64) -> usize {
        self.time.partition_point(|&u| u <= t)
    }
}

/// Sort the used rows. `cause` is the event code of interest; for data with
/// one event code it must be 1.
pub(super) fn sort_rows(data: &SurvData<'_>, timefix: TimeFix, cause: u32) -> Result<Sorted> {
    let data = data.clone().with_timefix(timefix)?;
    let surv = data.followup()?;
    if surv.start().is_some() {
        return Err(Error::input(
            "data with delayed entry has no censoring distribution to weight by; \
             give follow-up from time 0",
        ));
    }
    if cause == 0 || cause > surv.n_codes() {
        return Err(Error::input(format!(
            "cause {cause} is outside 1..={}; 0 is censoring",
            surv.n_codes()
        )));
    }
    if let Some(weight) = data.weight() {
        if let Some(position) = weight.iter().position(|&w| w != 0.0 && w != 1.0) {
            return Err(Error::input(format!(
                "weights other than 0 and 1 are not supported, as in riskRegression::Score and \
                 timeROC: row {position} has weight {}",
                weight[position]
            )));
        }
    }
    let mut rows: Vec<u32> = data.used_rows().collect();
    if rows.is_empty() {
        return Err(Error::input("no row has positive weight"));
    }
    let stop = surv.stop();
    rows.sort_unstable_by(|&a, &b| {
        stop[a as usize]
            .total_cmp(&stop[b as usize])
            .then(a.cmp(&b))
    });
    let status = surv.status();
    let cause = i32::try_from(cause).expect("the cause is at most n_codes, an i32 range");
    let kind = rows
        .iter()
        .map(|&r| match status[r as usize] {
            0 => Kind::Censored,
            s if s == cause => Kind::Cause,
            _ => Kind::Competing,
        })
        .collect();
    let time: Vec<f64> = rows.iter().map(|&r| stop[r as usize]).collect();
    let max_time = time[time.len() - 1];
    Ok(Sorted {
        time,
        kind,
        row: rows,
        n_rows: data.n_rows(),
        max_time,
    })
}

/// Refuse evaluation times that are not strictly ascending and finite, or that
/// lie after the last observed time.
pub(super) fn check_times(times: &[f64], max_time: f64) -> Result<()> {
    if times.is_empty() {
        return Err(Error::input("times: at least one time is needed"));
    }
    if let Some(position) = times.iter().position(|t| !t.is_finite()) {
        return Err(Error::input(format!(
            "times: value {position} is not finite"
        )));
    }
    if let Some(position) = times.windows(2).position(|w| w[1] <= w[0]) {
        return Err(Error::input(format!(
            "times must be strictly increasing: value {} is not above value {position}",
            position + 1
        )));
    }
    if let Some(&last) = times.last() {
        if last > max_time {
            return Err(Error::input(format!(
                "time {last} is after the last observed time {max_time}; riskRegression::Score \
                 drops such times, here they are refused"
            )));
        }
    }
    Ok(())
}

/// Refuse a matrix of predictions that does not have one row per row of the
/// data and one column per time, or has a value that is not finite.
pub(super) fn check_matrix(
    name: &str,
    m: MatRef<'_, f64>,
    n_rows: usize,
    n_times: usize,
) -> Result<()> {
    if m.nrows() != n_rows || m.ncols() != n_times {
        return Err(Error::input(format!(
            "{name} is {} by {}, but there are {n_rows} rows and {n_times} times",
            m.nrows(),
            m.ncols()
        )));
    }
    for j in 0..n_times {
        for i in 0..n_rows {
            if !m[(i, j)].is_finite() {
                return Err(Error::input(format!(
                    "{name}: row {i}, column {j} is not finite"
                )));
            }
        }
    }
    Ok(())
}
