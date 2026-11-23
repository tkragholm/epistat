//! Longitudinal model input containers.
//!
//! This module defines a model-agnostic person-period input used by
//! longitudinal semi-continuous models such as MTP.

use epistat_linalg::Mat;
use thiserror::Error;

use crate::utils::matrix_is_finite;

/// Errors returned when validating longitudinal model inputs.
#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum LongitudinalInputError {
    /// Binary design matrix must have at least one column.
    #[error("binary design matrix must have at least one column")]
    EmptyBinaryDesign,
    /// Mean design matrix must have at least one column.
    #[error("mean design matrix must have at least one column")]
    EmptyMeanDesign,
    /// Outcome must be a single-column matrix.
    #[error("outcome must be a single-column matrix")]
    InvalidOutcomeShape,
    /// The designs and the outcome have different numbers of rows.
    #[error(
        "binary design rows ({binary_rows}), mean design rows ({mean_rows}), and outcome rows ({outcome_rows}) must match"
    )]
    DimensionMismatch {
        /// The number of rows of the binary-part design.
        binary_rows: usize,
        /// The number of rows of the mean-part design.
        mean_rows: usize,
        /// The number of rows of the outcome.
        outcome_rows: usize,
    },
    /// Subject id length must equal number of rows.
    #[error("subject id length ({len}) must equal number of rows ({rows})")]
    InvalidSubjectLength {
        /// The number of subject ids.
        len: usize,
        /// The number of rows.
        rows: usize,
    },
    /// Time length must equal number of rows.
    #[error("time length ({len}) must equal number of rows ({rows})")]
    InvalidTimeLength {
        /// The number of times.
        len: usize,
        /// The number of rows.
        rows: usize,
    },
    /// Family id length must equal number of rows.
    #[error("family id length ({len}) must equal number of rows ({rows})")]
    InvalidFamilyLength {
        /// The number of family ids.
        len: usize,
        /// The number of rows.
        rows: usize,
    },
    /// Input matrices contain non-finite values.
    #[error("input matrices contain non-finite values")]
    NonFiniteMatrix,
    /// Time contains non-finite values.
    #[error("time contains non-finite values")]
    NonFiniteTime,
    /// Outcome contains negative values.
    #[error("outcome contains negative values")]
    NegativeOutcome,
}

/// Generic longitudinal input for person-period semi-continuous models.
#[derive(Debug, Clone)]
pub struct LongitudinalModelInput {
    /// The outcome, one row per person-period, `n x 1`.
    pub outcome: Mat<f64>,
    /// The design of the binary part, `Pr(y > 0)`.
    pub x_binary: Mat<f64>,
    /// The design of the marginal mean.
    pub x_mean: Mat<f64>,
    /// The subject of each row.
    pub subject_ids: Vec<u64>,
    /// The time of each row.
    pub time: Vec<f64>,
    /// The family of each row, for a family-level random effect.
    pub family_ids: Option<Vec<u64>>,
}

impl LongitudinalModelInput {
    /// An input without family ids.
    #[must_use]
    pub const fn new(
        outcome: Mat<f64>,
        x_binary: Mat<f64>,
        x_mean: Mat<f64>,
        subject_ids: Vec<u64>,
        time: Vec<f64>,
    ) -> Self {
        Self {
            outcome,
            x_binary,
            x_mean,
            subject_ids,
            time,
            family_ids: None,
        }
    }

    /// The input with the family of each row.
    #[must_use]
    pub fn with_family_ids(mut self, family_ids: Vec<u64>) -> Self {
        self.family_ids = Some(family_ids);
        self
    }

    /// # Errors
    ///
    /// Returns `LongitudinalInputError` if shapes or values are malformed.
    pub fn validate(&self) -> Result<(), LongitudinalInputError> {
        if self.x_binary.ncols() == 0 {
            return Err(LongitudinalInputError::EmptyBinaryDesign);
        }
        if self.x_mean.ncols() == 0 {
            return Err(LongitudinalInputError::EmptyMeanDesign);
        }
        if self.outcome.ncols() != 1 {
            return Err(LongitudinalInputError::InvalidOutcomeShape);
        }

        let rows = self.outcome.nrows();
        if self.x_binary.nrows() != rows || self.x_mean.nrows() != rows {
            return Err(LongitudinalInputError::DimensionMismatch {
                binary_rows: self.x_binary.nrows(),
                mean_rows: self.x_mean.nrows(),
                outcome_rows: rows,
            });
        }
        if self.subject_ids.len() != rows {
            return Err(LongitudinalInputError::InvalidSubjectLength {
                len: self.subject_ids.len(),
                rows,
            });
        }
        if self.time.len() != rows {
            return Err(LongitudinalInputError::InvalidTimeLength {
                len: self.time.len(),
                rows,
            });
        }
        if let Some(family_ids) = &self.family_ids
            && family_ids.len() != rows
        {
            return Err(LongitudinalInputError::InvalidFamilyLength {
                len: family_ids.len(),
                rows,
            });
        }

        if !matrix_is_finite(&self.outcome)
            || !matrix_is_finite(&self.x_binary)
            || !matrix_is_finite(&self.x_mean)
        {
            return Err(LongitudinalInputError::NonFiniteMatrix);
        }
        if self.time.iter().any(|value| !value.is_finite()) {
            return Err(LongitudinalInputError::NonFiniteTime);
        }
        if (0..rows).any(|i| self.outcome[(i, 0)] < 0.0) {
            return Err(LongitudinalInputError::NegativeOutcome);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_accepts_well_formed_input() {
        let outcome = Mat::from_fn(3, 1, |_i, _| 1.0);
        let x_binary = Mat::from_fn(3, 2, |_i, j| if j == 0 { 1.0 } else { 0.5 });
        let x_mean = Mat::from_fn(3, 2, |_i, j| if j == 0 { 1.0 } else { 0.25 });
        let subject_ids = vec![10, 10, 11];
        let time = vec![0.0, 1.0, 0.0];
        let input = LongitudinalModelInput::new(outcome, x_binary, x_mean, subject_ids, time);
        assert!(input.validate().is_ok());
    }

    #[test]
    fn validate_rejects_subject_length_mismatch() {
        let outcome = Mat::from_fn(2, 1, |_i, _| 1.0);
        let x_binary = Mat::from_fn(2, 1, |_i, _| 1.0);
        let x_mean = Mat::from_fn(2, 1, |_i, _| 1.0);
        let input = LongitudinalModelInput::new(outcome, x_binary, x_mean, vec![1], vec![0.0, 1.0]);
        let error = input
            .validate()
            .expect_err("subject id mismatch should fail");
        assert!(matches!(
            error,
            LongitudinalInputError::InvalidSubjectLength { .. }
        ));
    }
}
