//! Input checks shared by the estimators, so a refusal reads the same whichever
//! estimator raised it.

use epistat_linalg::MatRef;

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
    non_finite_result(
        values.iter().filter(|value| !value.is_finite()).count(),
        name,
    )
}

/// Refuse a matrix holding NaN or an infinity.
///
/// `name` is in the message, with the count of offending values.
///
/// # Errors
///
/// [`Error::Input`] when any entry of `x` is not finite.
pub(crate) fn refuse_non_finite_matrix(x: MatRef<'_, f64>, name: &str) -> Result<()> {
    let count = (0..x.nrows())
        .flat_map(|i| (0..x.ncols()).map(move |j| (i, j)))
        .filter(|&(i, j)| !x[(i, j)].is_finite())
        .count();
    non_finite_result(count, name)
}

fn non_finite_result(count: usize, name: &str) -> Result<()> {
    if count > 0 {
        return Err(Error::input(format!(
            "{name}: {count} non-finite values; drop or resolve them before estimating"
        )));
    }
    Ok(())
}

/// Refuse values that are not finite and ascending.
///
/// With `strict` a repeated value is refused too.
///
/// # Errors
///
/// [`Error::Input`] naming `name` when a value is not finite or a value is
/// below the one before it, or equal to it when `strict`.
pub(crate) fn check_ascending(values: &[f64], name: &str, strict: bool) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(Error::input(format!("{name} must be finite")));
    }
    if values
        .windows(2)
        .any(|w| if strict { w[0] >= w[1] } else { w[0] > w[1] })
    {
        return Err(Error::input(format!(
            "{name} must be {}ascending",
            if strict { "strictly " } else { "" }
        )));
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

    #[test]
    fn a_matrix_with_a_non_finite_entry_is_refused_with_its_count() {
        let clean = Mat::from_fn(2, 2, |i, j| (i + j) as f64);
        assert!(refuse_non_finite_matrix(clean.as_ref(), "x").is_ok());
        let dirty = Mat::from_fn(2, 2, |i, j| if i == j { f64::NAN } else { 1.0 });
        assert_eq!(
            refuse_non_finite_matrix(dirty.as_ref(), "x")
                .unwrap_err()
                .to_string(),
            "x: 2 non-finite values; drop or resolve them before estimating"
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
            "times must be strictly ascending"
        );
        assert_eq!(
            check_ascending(&[2.0, 1.0], "times", false)
                .unwrap_err()
                .to_string(),
            "times must be ascending"
        );
        assert_eq!(
            check_ascending(&[1.0, f64::NAN], "times", false)
                .unwrap_err()
                .to_string(),
            "times must be finite"
        );
    }
}
