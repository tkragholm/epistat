//! Input checks shared by the estimators, so a refusal reads the same whichever
//! estimator raised it.

use crate::error::{Error, Result};

/// Refuse columns of different lengths.
///
/// `columns` pairs each column's name with its length. The first is the
/// reference, and the message names the first column that differs from it.
///
/// # Errors
///
/// [`Error::Input`] when any length differs from the first.
pub(crate) fn check_lengths(columns: &[(&str, usize)]) -> Result<()> {
    let Some(&(reference, expected)) = columns.first() else {
        return Ok(());
    };
    match columns.iter().find(|&&(_, len)| len != expected) {
        Some(&(name, len)) => Err(Error::input(format!(
            "columns differ in length: {name} has {len} values against {expected} in {reference}"
        ))),
        None => Ok(()),
    }
}

/// Refuse more rows than a `u32` row index addresses.
///
/// # Errors
///
/// [`Error::Input`] when `rows` exceeds `u32::MAX`.
pub(crate) fn check_row_count(rows: usize) -> Result<()> {
    if u32::try_from(rows).is_err() {
        return Err(Error::input(format!(
            "{rows} rows is more than the {} the estimators index",
            u32::MAX
        )));
    }
    Ok(())
}

/// Refuse a column holding NaN or an infinity.
///
/// `name` is in the message, with the count of offending values.
///
/// # Errors
///
/// [`Error::Input`] when any of `values` is not finite.
pub(crate) fn refuse_non_finite(values: &[f64], name: &str) -> Result<()> {
    let non_finite = values.iter().filter(|value| !value.is_finite()).count();
    if non_finite > 0 {
        return Err(Error::input(format!(
            "{name}: {non_finite} non-finite values; drop or resolve them before estimating"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{check_lengths, refuse_non_finite};

    #[test]
    fn a_non_finite_value_is_refused_with_its_column_and_count() {
        assert!(refuse_non_finite(&[0.0, -1.5, 1e308], "time").is_ok());
        let error = refuse_non_finite(&[f64::NAN, 1.0, f64::INFINITY], "time").unwrap_err();
        assert_eq!(
            error.to_string(),
            "time: 2 non-finite values; drop or resolve them before estimating"
        );
    }

    #[test]
    fn a_column_of_another_length_is_named_against_the_first() {
        assert!(check_lengths(&[]).is_ok());
        assert!(check_lengths(&[("time", 3), ("event", 3)]).is_ok());
        let error = check_lengths(&[("time", 3), ("event", 3), ("weight", 2)]).unwrap_err();
        assert_eq!(
            error.to_string(),
            "columns differ in length: weight has 2 values against 3 in time"
        );
    }
}
