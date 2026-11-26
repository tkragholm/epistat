//! Input checks shared by the estimators, so a refusal reads the same whichever
//! estimator raised it.

use epistat_core::{InputError, check};
use epistat_linalg::MatRef;

use crate::error::Result;

/// Refuse columns of different lengths.
///
/// `columns` pairs each column's name with its length. The first is the
/// reference, and the error names the first column that differs from it.
///
/// # Errors
///
/// [`InputError::Length`] when any length differs from the first.
pub(crate) fn check_lengths(columns: &[(&'static str, usize)]) -> Result<()> {
    let Some(&(_, expected)) = columns.first() else {
        return Ok(());
    };
    for &(name, len) in columns {
        check::same_length(name, expected, len)?;
    }
    Ok(())
}

/// Refuse more rows than a `u32` row index addresses.
///
/// # Errors
///
/// [`InputError::TooLarge`] when `rows` exceeds `u32::MAX`.
pub(crate) fn check_row_count(rows: usize) -> Result<()> {
    if u32::try_from(rows).is_err() {
        return Err(InputError::TooLarge {
            what: "rows",
            limit: u64::from(u32::MAX),
            found: rows as u64,
        }
        .into());
    }
    Ok(())
}

/// Refuse a column holding NaN or an infinity.
///
/// # Errors
///
/// [`InputError::NotFinite`] naming `name` and the first row that is not
/// finite.
pub(crate) fn refuse_non_finite(values: &[f64], name: &'static str) -> Result<()> {
    Ok(check::finite(name, values)?)
}

/// Refuse a matrix holding NaN or an infinity.
///
/// # Errors
///
/// [`InputError::NotFinite`] naming `name` and the first row of `x` that holds
/// an entry that is not finite.
pub(crate) fn refuse_non_finite_matrix(x: MatRef<'_, f64>, name: &'static str) -> Result<()> {
    match (0..x.nrows()).find(|&i| (0..x.ncols()).any(|j| !x[(i, j)].is_finite())) {
        Some(row) => Err(InputError::NotFinite { what: name, row }.into()),
        None => Ok(()),
    }
}

/// Refuse values that are not finite and ascending.
///
/// With `strict` a repeated value is refused too.
///
/// # Errors
///
/// [`InputError::NotFinite`] naming `name` when a value is not finite,
/// [`InputError::Unsorted`] when a value is below the one before it, and
/// [`InputError::Duplicate`] when it equals it and `strict`.
pub(crate) fn check_ascending(values: &[f64], name: &'static str, strict: bool) -> Result<()> {
    check::finite(name, values)?;
    // The values are finite here, so `partial_cmp` orders every pair, and
    // -0 and 0 compare equal.
    let order = |a: &f64, b: &f64| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal);
    if strict {
        check::strictly_sorted_by(name, values, order)?;
    } else {
        check::sorted_by(name, values, order)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{check_ascending, check_lengths, refuse_non_finite, refuse_non_finite_matrix};
    use epistat_linalg::Mat;

    #[test]
    fn a_non_finite_value_is_refused_with_its_column_and_count() {
        assert!(refuse_non_finite(&[0.0, -1.5, 1e308], "time").is_ok());
        let error = refuse_non_finite(&[f64::NAN, 1.0, f64::INFINITY], "time").unwrap_err();
        assert_eq!(error.to_string(), "time is not finite at row 0");
    }

    #[test]
    fn a_column_of_another_length_is_named_against_the_first() {
        assert!(check_lengths(&[]).is_ok());
        assert!(check_lengths(&[("time", 3), ("event", 3)]).is_ok());
        let error = check_lengths(&[("time", 3), ("event", 3), ("weight", 2)]).unwrap_err();
        assert_eq!(error.to_string(), "weight has length 2, expected 3");
    }

    #[test]
    fn a_matrix_with_a_non_finite_entry_is_refused_with_its_count() {
        let clean = Mat::from_fn(2, 2, |i, j| (i + j) as f64);
        assert!(refuse_non_finite_matrix(clean.as_ref(), "x").is_ok());
        let dirty = Mat::from_fn(2, 2, |i, j| if i == j { f64::NAN } else { 1.0 });
        assert_eq!(
            refuse_non_finite_matrix(dirty.as_ref(), "x")
                .unwrap_err()
                .to_string(),
            "x is not finite at row 0"
        );
    }

    #[test]
    fn values_must_be_finite_and_ascending() {
        assert!(check_ascending(&[], "times", true).is_ok());
        assert!(check_ascending(&[1.0, 2.0, 2.0], "times", false).is_ok());
        assert_eq!(
            check_ascending(&[1.0, 2.0, 2.0], "times", true)
                .unwrap_err()
                .to_string(),
            "times repeats at row 2 the row above it"
        );
        assert_eq!(
            check_ascending(&[2.0, 1.0], "times", false)
                .unwrap_err()
                .to_string(),
            "times is not sorted: row 1 sorts before the row above it"
        );
        assert_eq!(
            check_ascending(&[1.0, f64::NAN], "times", false)
                .unwrap_err()
                .to_string(),
            "times is not finite at row 1"
        );
    }
}
