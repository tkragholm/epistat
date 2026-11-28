//! The first treated period of each row from R's coding of a cohort column.
//!
//! R's `did` codes a never-treated unit's `gname` as 0, and `fixest::sunab`
//! and `etwfe` use 0, a period beyond the panel, `Inf` or `NA`. Here a
//! never-treated unit's first treated period is `None`, and a cohort of 0
//! among positive periods is refused by [`StaggeredPanel`](crate::StaggeredPanel),
//! since it would read as a cohort treated before the panel.

use epistat_core::InputError;

use crate::error::Error;

/// 2^63: the whole numbers below it in magnitude convert to `i64` exactly.
const LIMIT: f64 = 9_223_372_036_854_775_808.0;

/// The first treated periods of a cohort column in R's coding: 0, an
/// infinity and NaN (R's `NA`) are never treated, `None`, and every other
/// value is the period itself, which must be a whole number.
///
/// ```
/// use epistat_did::cohort;
///
/// let gname = [2004.0, 0.0, f64::INFINITY, 2006.0];
/// assert_eq!(cohort::from_r(&gname)?, vec![Some(2004), None, None, Some(2006)]);
/// # Ok::<(), epistat_did::Error>(())
/// ```
///
/// # Errors
///
/// [`Error::Input`] with [`InputError::Domain`] at the first value that is
/// neither a never-treated code nor a whole number within `i64`.
pub fn from_r(cohort: &[f64]) -> Result<Vec<Option<i64>>, Error> {
    cohort
        .iter()
        .enumerate()
        .map(|(row, &value)| {
            if value == 0.0 || value.is_infinite() || value.is_nan() {
                return Ok(None);
            }
            if value.fract() != 0.0 || value.abs() >= LIMIT {
                return Err(InputError::Domain {
                    what: "cohort",
                    value,
                    domain: "a whole-number period, or 0, Inf or NA for never treated",
                    row: Some(row),
                }
                .into());
            }
            #[expect(
                clippy::cast_possible_truncation,
                reason = "a whole number below 2^63 in magnitude, so exact"
            )]
            Ok(Some(value as i64))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_core::Classify;

    #[test]
    fn r_codes_of_never_treated_become_none() {
        let cohorts = from_r(&[
            0.0,
            -0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            2004.0,
            -3.0,
        ])
        .unwrap();
        assert_eq!(
            cohorts,
            [None, None, None, None, None, Some(2004), Some(-3)]
        );
        assert_eq!(from_r(&[2004.5]).unwrap_err().code(), "input.domain");
        assert_eq!(from_r(&[1e19]).unwrap_err().code(), "input.domain");
        assert_eq!(from_r(&[]).unwrap(), Vec::new());
    }
}
