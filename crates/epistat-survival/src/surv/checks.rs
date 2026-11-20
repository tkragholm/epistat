//! The build-time rules of [`SurvData`](super::SurvData), one function each.
//!
//! Every function reports the number of offending rows and the first one, with
//! row numbers counted from 0, so a caller can find the data to repair.

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

fn refuse(name: &str, problem: &str, (count, first): (usize, usize)) -> Error {
    Error::input(format!(
        "{name}: {count} values {problem}, first at row {first}"
    ))
}

/// Refuse `n_codes == 0`, which leaves no code for an event.
pub(super) fn check_n_codes(n_codes: u32) -> Result<()> {
    if n_codes == 0 {
        return Err(Error::input(
            "n_codes is 0; status needs at least one event code".to_owned(),
        ));
    }
    Ok(())
}

/// Refuse NaN and infinities in a time or weight column.
pub(super) fn check_finite(values: &[f64], name: &str) -> Result<()> {
    match count_and_first(values.len(), |row| !values[row].is_finite()) {
        Some(found) => Err(refuse(
            name,
            "are not finite; drop or resolve them before building",
            found,
        )),
        None => Ok(()),
    }
}

/// Refuse a weight below zero. A weight of exactly 0 is allowed.
pub(super) fn check_weights(weight: &[f64]) -> Result<()> {
    match count_and_first(weight.len(), |row| weight[row] < 0.0) {
        Some(found) => Err(refuse("weight", "are negative", found)),
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
        Some(found) => Err(refuse("start", "are not below their stop", found)),
        None => Ok(()),
    }
}
