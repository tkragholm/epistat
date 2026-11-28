//! The rows of the unweighted metrics, sorted by time, and the checks of
//! their inputs.

#![expect(
    clippy::float_cmp,
    reason = "a case weight is 0 or 1 exactly or it is refused"
)]

use epistat_core::InputError;
use epistat_linalg::MatRef;

use crate::check::check_ascending;
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
    let surv = data.followup();
    if surv.start().is_some() {
        return Err(Error::not_applicable(
            "data with delayed entry has no censoring distribution to weight by; \
             give follow-up from time 0",
        ));
    }
    if cause == 0 || cause > surv.n_codes() {
        return Err(Error::out_of_range(format!(
            "cause {cause} is outside 1..={}; 0 is censoring",
            surv.n_codes()
        )));
    }
    if let Some(weight) = data.weight()
        && let Some(position) = weight.iter().position(|&w| w != 0.0 && w != 1.0)
    {
        return Err(Error::unsupported(
            "weight",
            format!(
                "weights other than 0 and 1 are not supported, as in riskRegression::Score and \
                 timeROC: row {position} has weight {}",
                weight[position]
            ),
        ));
    }
    let mut rows: Vec<u32> = data.used_rows().collect();
    if rows.is_empty() {
        return Err(Error::no_information("no row has positive weight"));
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
        return Err(Error::from(InputError::Empty { what: "times" }));
    }
    check_ascending(times, "times", true)?;
    if let Some(&last) = times.last()
        && last > max_time
    {
        return Err(Error::value(format!(
            "time {last} is after the last observed time {max_time}; riskRegression::Score \
                 drops such times, here they are refused"
        )));
    }
    Ok(())
}

/// Refuse a matrix of predictions that does not have one row per row of the
/// data and one column per time, or has a value that is not finite. The first
/// value refused is the first in column order.
pub(super) fn check_matrix(
    name: &'static str,
    m: MatRef<'_, f64>,
    n_rows: usize,
    n_times: usize,
) -> Result<()> {
    check_shape(name, m, n_rows, n_times)?;
    for j in 0..n_times {
        for i in 0..n_rows {
            if !m[(i, j)].is_finite() {
                return Err(Error::from(InputError::NotFinite {
                    what: name,
                    row: i,
                    column: Some(j),
                }));
            }
        }
    }
    Ok(())
}

/// Refuse a matrix of predictions that does not have one row per row of the
/// data and one column per time.
pub(super) fn check_shape(
    name: &'static str,
    m: MatRef<'_, f64>,
    n_rows: usize,
    n_times: usize,
) -> Result<()> {
    if m.nrows() != n_rows {
        if m.nrows() == 0 {
            return Err(Error::from(InputError::Empty { what: name }));
        }
        return Err(Error::from(InputError::Length {
            what: name,
            expected: n_rows,
            found: m.nrows(),
        }));
    }
    if m.ncols() != n_times {
        return Err(Error::mismatch(format!(
            "{name} is {} by {}, but there are {n_rows} rows and {n_times} times",
            m.nrows(),
            m.ncols()
        )));
    }
    Ok(())
}

/// Accept a start column only for the data `survival::brier` computes sensibly
/// with it: an id column, one row per subject and one common start. The start
/// then only says when follow-up began, and the follow-up is the time to the
/// stop.
pub(super) fn check_common_start(data: &SurvData<'_>) -> Result<()> {
    let surv = data.followup();
    let Some(start) = surv.start() else {
        return Ok(());
    };
    let ids = data.id().ok_or_else(|| {
        Error::missing("data with delayed entry needs an id column, as survival::brier does")
    })?;
    if data.istate().is_some() {
        return Err(Error::not_applicable(
            "the data has an istate column: only data with one starting state is supported",
        ));
    }
    let mut seen = vec![false; data.n_ids()];
    for (row, &id) in ids.iter().enumerate() {
        if std::mem::replace(&mut seen[id as usize], true) {
            return Err(Error::inconsistent(format!(
                "a subject has several rows, the second at row {row}: survival::brier counts                  every row as a subject, which makes the score wrong, so one row per subject is                  required with a start column"
            )));
        }
    }
    if let Some(row) = start.iter().position(|&s| s != start[0]) {
        return Err(Error::not_applicable(format!(
            "subjects start at different times (row {row} starts at {}, row 0 at {}); one              common start is required",
            start[row], start[0]
        )));
    }
    Ok(())
}
