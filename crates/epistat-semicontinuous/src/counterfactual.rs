//! A design and outcome built from the caller's rows, with the exposure set to 1
//! and to 0 for counterfactual contrasts.

use epistat_core::InputError;
use epistat_linalg::Mat;

use crate::Error;
use crate::data::CostData;
use crate::preprocess::{column_has_variation, drop_constant_columns};

/// Default threshold where two-part models are treated as not applicable.
pub const TWO_PART_NOT_APPLICABLE_POSITIVE_SHARE: f64 = 0.995;

/// The design and outcome of a counterfactual contrast, with the designs of
/// every row exposed and every row unexposed.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct CounterfactualDesign {
    /// The design, without its constant columns other than the intercept.
    pub x: Mat<f64>,
    /// The outcome, negative values clamped to 0.
    pub y: Vec<f64>,
    /// The design with the exposure set to 1.
    pub x_treated: Mat<f64>,
    /// The design with the exposure set to 0.
    pub x_control: Mat<f64>,
    /// The column of the exposure in `x`.
    pub exposure_column: usize,
    /// The number of rows.
    pub n_rows: usize,
    /// The number of positive outcomes.
    pub n_positive: usize,
}

impl CounterfactualDesign {
    /// The design and outcome as data.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the design holds a value that is not finite.
    pub fn data(&self) -> Result<CostData<'_>, Error> {
        CostData::builder(self.x.as_ref(), &self.y).build()
    }
}

/// Clamp finite outcomes to non-negative values for semi-continuous modeling.
#[must_use]
pub const fn sanitize_model_outcome(raw: f64) -> f64 {
    if raw.is_finite() { raw.max(0.0) } else { raw }
}

/// Return whether two-part models are applicable for a positive outcome share.
#[must_use]
pub fn two_part_applicability(positive_share: f64) -> bool {
    positive_share < TWO_PART_NOT_APPLICABLE_POSITIVE_SHARE
}

/// Build a design and outcome from the caller's rows, with the exposure set
/// to 1 and to 0 in two copies of the design.
///
/// `design_value` must return the observed design matrix element for `column`.
/// The `exposure_column` values are toggled to 1/0 in the returned counterfactual
/// matrices after constant-column dropping.
///
/// # Errors
///
/// [`Error::Input`] for no rows, an exposure column outside the design or an
/// outcome that is not finite; [`Error::Invalid`] when the exposure does not
/// vary or was dropped as constant.
pub fn build_counterfactual_input<'a, R: 'a, FRow, FOutcome, FDesign>(
    n_rows: usize,
    mut row_at: FRow,
    outcome: FOutcome,
    design_columns: usize,
    exposure_column: usize,
    design_value: FDesign,
    constant_column_tolerance: f64,
) -> Result<CounterfactualDesign, Error>
where
    FRow: FnMut(usize) -> &'a R,
    FOutcome: Fn(&R) -> f64,
    FDesign: Fn(&R, usize) -> f64,
{
    epistat_core::check::non_empty("rows", n_rows)?;
    if exposure_column >= design_columns {
        return Err(crate::error::domain(
            "exposure column",
            exposure_column as f64,
            "a column of the design",
        ));
    }

    let mut y_values = Vec::with_capacity(n_rows);
    let mut n_positive = 0_usize;
    for row_idx in 0..n_rows {
        let row = row_at(row_idx);
        let y = sanitize_model_outcome(outcome(row));
        if !y.is_finite() {
            return Err(InputError::NotFinite {
                what: "outcome",
                row: row_idx,
            }
            .into());
        }
        if y > 0.0 {
            n_positive += 1;
        }
        y_values.push(y);
    }

    let x_full = Mat::from_fn(n_rows, design_columns, |row_idx, col_idx| {
        let row = row_at(row_idx);
        design_value(row, col_idx)
    });

    if !column_has_variation(x_full.as_ref(), exposure_column, constant_column_tolerance) {
        return Err(Error::Invalid {
            what: "the exposure does not vary",
        });
    }

    let (x, kept_columns) = drop_constant_columns(
        x_full.as_ref(),
        constant_column_tolerance,
        &[0, exposure_column],
    );
    let Some(exposure_col_reindexed) = kept_columns
        .iter()
        .position(|column| *column == exposure_column)
    else {
        return Err(Error::Invalid {
            what: "the exposure column was dropped as constant",
        });
    };

    let mut x_treated = x.clone();
    let mut x_control = x.clone();
    for row_idx in 0..n_rows {
        x_treated[(row_idx, exposure_col_reindexed)] = 1.0;
        x_control[(row_idx, exposure_col_reindexed)] = 0.0;
    }

    Ok(CounterfactualDesign {
        x,
        y: y_values,
        x_treated,
        x_control,
        exposure_column: exposure_col_reindexed,
        n_rows,
        n_positive,
    })
}

#[cfg(test)]
mod tests {
    use super::{build_counterfactual_input, two_part_applicability};
    use crate::Error;
    use epistat_core::InputError;

    #[derive(Debug, Clone, Copy)]
    struct Row {
        outcome: f64,
        exposed: bool,
        followup: i32,
        male: bool,
    }

    fn design_value(row: &Row, column: usize) -> f64 {
        match column {
            0 => 1.0,
            1 if row.exposed => 1.0,
            2 => f64::from(row.followup),
            3 if row.male => 1.0,
            _ => 0.0,
        }
    }

    #[test]
    fn build_counterfactual_input_rejects_empty_rows() {
        let rows: [Row; 0] = [];
        let result = build_counterfactual_input(
            rows.len(),
            |idx| &rows[idx],
            |row| row.outcome,
            4,
            1,
            design_value,
            1e-12,
        );
        assert!(matches!(
            result,
            Err(Error::Input(InputError::Empty { .. }))
        ));
    }

    #[test]
    fn build_counterfactual_input_rejects_non_finite_outcomes() {
        let rows = [Row {
            outcome: f64::NAN,
            exposed: true,
            followup: 0,
            male: false,
        }];
        let result = build_counterfactual_input(
            rows.len(),
            |idx| &rows[idx],
            |row| row.outcome,
            4,
            1,
            design_value,
            1e-12,
        );
        assert!(matches!(
            result,
            Err(Error::Input(InputError::NotFinite { row: 0, .. }))
        ));
    }

    #[test]
    fn build_counterfactual_input_constructs_counterfactual_matrices() {
        let rows = [
            Row {
                outcome: 0.0,
                exposed: false,
                followup: 0,
                male: false,
            },
            Row {
                outcome: 10.0,
                exposed: true,
                followup: 1,
                male: true,
            },
        ];
        let prepared = build_counterfactual_input(
            rows.len(),
            |idx| &rows[idx],
            |row| row.outcome,
            4,
            1,
            design_value,
            1e-12,
        )
        .expect("valid prepared input");
        assert_eq!(prepared.n_rows, 2);
        assert_eq!(prepared.n_positive, 1);
        assert!((prepared.x_treated[(0, 1)] - 1.0).abs() < 1e-12);
        assert!((prepared.x_control[(0, 1)] - 0.0).abs() < 1e-12);
        assert_eq!(prepared.exposure_column, 1);
        assert!(prepared.data().is_ok());
    }

    #[test]
    fn two_part_applicability_respects_default_threshold() {
        assert!(two_part_applicability(0.5));
        assert!(!two_part_applicability(0.995));
    }
}
