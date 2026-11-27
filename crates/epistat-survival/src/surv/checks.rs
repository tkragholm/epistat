//! The build-time rules of [`SurvData`](super::SurvData), one function each.
//!
//! Every function reports the first offending row, counted from 0, so a caller
//! can find the data to repair. The rules the shared checks of `epistat_core`
//! have no error for report the number of offending rows too.

use epistat_core::InputError;

use crate::error::{Error, Result};

/// Count the rows for which `offends` holds and find the first of them.
fn count_and_first(len: usize, mut offends: impl FnMut(usize) -> bool) -> Option<(usize, usize)> {
    let mut count = 0_usize;
    let mut first = 0_usize;
    for row in 0..len {
        if offends(row) {
            if count == 0 {
                first = row;
            }
            count += 1;
        }
    }
    (count > 0).then_some((count, first))
}

/// The error `variant` of the rows of column `name` that have `problem`.
fn refuse(
    variant: fn(String) -> Error,
    name: &str,
    problem: &str,
    (count, first): (usize, usize),
) -> Error {
    variant(format!(
        "{name}: {count} values {problem}, first at row {first}"
    ))
}

/// Refuse `n_codes == 0`, which leaves no code for an event.
pub(super) fn check_n_codes(n_codes: u32) -> Result<()> {
    if n_codes == 0 {
        return Err(Error::options(
            "n_codes is 0; status needs at least one event code",
        ));
    }
    Ok(())
}

/// Refuse NaN and infinities in a time or weight column, as
/// [`InputError::NotFinite`] at the first such row with the number of them.
pub(super) fn check_finite(values: &[f64], name: &'static str) -> Result<()> {
    match count_and_first(values.len(), |row| !values[row].is_finite()) {
        Some((count, row)) => Err(Error::Counted {
            error: InputError::NotFinite {
                what: name,
                row,
                column: None,
            },
            count,
        }),
        None => Ok(()),
    }
}

/// Refuse a weight below zero, as [`InputError::Domain`] at the first such
/// row with the number of them. A weight of exactly 0 is allowed. The weights
/// are finite when this runs.
pub(super) fn check_weights(weight: &[f64]) -> Result<()> {
    match count_and_first(weight.len(), |row| weight[row] < 0.0) {
        Some((count, row)) => Err(Error::Counted {
            error: InputError::Domain {
                what: "weight",
                value: weight[row],
                domain: "[0, inf)",
                row: Some(row),
            },
            count,
        }),
        None => Ok(()),
    }
}

/// Refuse a status below 0 or above `n_codes`.
pub(super) fn check_status(status: &[i32], n_codes: u32) -> Result<()> {
    let limit = i64::from(n_codes);
    match count_and_first(status.len(), |row| {
        let code = i64::from(status[row]);
        code < 0 || code > limit
    }) {
        Some(found) => Err(refuse(
            Error::out_of_range,
            "status",
            &format!("are outside 0..={n_codes}"),
            found,
        )),
        None => Ok(()),
    }
}

/// Refuse an initial state above `n_codes`. 0 means unknown and is allowed.
pub(super) fn check_istate(istate: &[u32], n_codes: u32) -> Result<()> {
    match count_and_first(istate.len(), |row| istate[row] > n_codes) {
        Some(found) => Err(refuse(
            Error::out_of_range,
            "istate",
            &format!("are above n_codes = {n_codes}"),
            found,
        )),
        None => Ok(()),
    }
}

/// Refuse a risk interval that is empty or reversed, `start >= stop`.
///
/// Both columns are finite when this runs.
pub(super) fn check_intervals(start: &[f64], stop: &[f64]) -> Result<()> {
    match count_and_first(stop.len(), |row| start[row] >= stop[row]) {
        Some(found) => Err(refuse(
            Error::value,
            "start",
            "are not below their stop",
            found,
        )),
        None => Ok(()),
    }
}

/// Refuse interval bounds that cannot describe an observation.
///
/// A bound is NaN never. A lower bound may be `-inf` (left-censored) and an
/// upper bound `+inf` (right-censored), and no other infinity is allowed. A
/// row with both infinite says nothing about the time. A lower bound above its
/// upper bound is reversed. The slices have the same length.
pub(super) fn check_interval_bounds(lower: &[f64], upper: &[f64]) -> Result<()> {
    if let Some(found) = count_and_first(lower.len(), |row| {
        lower[row].is_nan() || upper[row].is_nan()
    }) {
        return Err(refuse(
            Error::value,
            "lower and upper",
            "are NaN; a missing bound is written -inf for lower and +inf for upper",
            found,
        ));
    }
    if let Some(found) = count_and_first(lower.len(), |row| {
        lower[row] == f64::INFINITY || upper[row] == f64::NEG_INFINITY
    }) {
        return Err(refuse(
            Error::value,
            "lower and upper",
            "are infinite on the wrong side; only lower may be -inf and only upper +inf",
            found,
        ));
    }
    if let Some(found) = count_and_first(lower.len(), |row| {
        lower[row] == f64::NEG_INFINITY && upper[row] == f64::INFINITY
    }) {
        return Err(refuse(
            Error::value,
            "lower and upper",
            "are both infinite, which says nothing about the time",
            found,
        ));
    }
    match count_and_first(lower.len(), |row| lower[row] > upper[row]) {
        Some(found) => Err(refuse(
            Error::value,
            "lower",
            "are above their upper",
            found,
        )),
        None => Ok(()),
    }
}

/// Refuse a typed interval row whose times cannot be used: a time that is not
/// finite, and for status 3 an upper time that is not finite or not above the
/// lower. The upper time of the other statuses is not read.
pub(super) fn check_interval_times(time: &[f64], time2: &[f64], status: &[i32]) -> Result<()> {
    check_finite(time, "time")?;
    if let Some(found) = count_and_first(time.len(), |row| {
        status[row] == 3 && !time2[row].is_finite()
    }) {
        return Err(refuse(
            Error::value,
            "time2",
            "are not finite on rows of status 3",
            found,
        ));
    }
    match count_and_first(time.len(), |row| {
        status[row] == 3 && time[row] >= time2[row]
    }) {
        Some(found) => Err(refuse(
            Error::value,
            "time",
            "are not below their time2 on rows of status 3",
            found,
        )),
        None => Ok(()),
    }
}
