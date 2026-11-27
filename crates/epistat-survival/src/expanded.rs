//! What the outputs that expand their input into rows share: carrying a column
//! of the input along to the rows, and the rows as follow-up data.

use epistat_core::InputError;

use crate::error::Result;
use crate::surv::SurvBuf;

/// The value of `column` at the input row of each output row, `rows[k]`.
///
/// # Errors
///
/// [`InputError::Length`] when `column` has no value at some row, naming
/// `what` and the shortest length that would do.
pub(crate) fn gather<T: Copy>(rows: &[u32], column: &[T], what: &'static str) -> Result<Vec<T>> {
    let needed = rows.iter().max().map_or(0, |&r| r as usize + 1);
    if column.len() < needed {
        return Err(InputError::Length {
            what,
            expected: needed,
            found: column.len(),
        }
        .into());
    }
    Ok(rows.iter().map(|&r| column[r as usize]).collect())
}

/// The rows as follow-up data: `start` (if any), `stop`, `status` and
/// `weight` (if any), with the largest status as the number of codes, at
/// least 1.
///
/// # Errors
///
/// The errors of building a [`SurvBuf`] from the columns.
pub(crate) fn surv_buf(
    start: Option<&[f64]>,
    stop: &[f64],
    status: &[i32],
    weight: Option<&[f64]>,
) -> Result<SurvBuf> {
    let codes = status.iter().copied().max().unwrap_or(0).max(1);
    let mut builder =
        SurvBuf::builder(stop.to_vec(), status.to_vec()).n_codes(u32::try_from(codes).unwrap_or(1));
    if let Some(start) = start {
        builder = builder.start(start.to_vec());
    }
    if let Some(weight) = weight {
        builder = builder.weight(weight.to_vec());
    }
    builder.build()
}
