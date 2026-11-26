//! The coefficients of an event study, the input of HonestDiD and of the
//! pre-trend power analysis.
//!
//! An event study estimates one coefficient per period relative to
//! treatment, each against a reference period that is normalised to zero and
//! so has no coefficient. [`EventStudy`] holds the coefficients in increasing
//! order of their relative periods, their [`Covariance`], the relative
//! periods and the reference. The periods before the reference are the
//! pre-treatment coefficients and those after it the post-treatment ones, the
//! split Rambachan and Roth's sensitivity analysis and Roth's pre-trend power
//! use \[1\], \[2\]. With anticipation, the reference sits earlier and the
//! periods between it and treatment count as post-treatment.
//!
//! The fields are private, so that a value that passed the checks of
//! [`EventStudy::new`] cannot be edited into one that fails them.
//!
//! # References
//!
//! 1. Rambachan A, Roth J. A more credible approach to parallel trends. Rev
//!    Econ Stud. 2023;90(5):2555-91. doi:10.1093/restud/rdad018
//! 2. Roth J. Pretest with caution: event-study estimates after testing for
//!    parallel trends. Am Econ Rev Insights. 2022;4(3):305-22.
//!    doi:10.1257/aeri.20210236

use crate::check::{self, InputError};
use crate::covariance::Covariance;

/// Event-study coefficients by relative period, their covariance and the
/// reference period.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(try_from = "EventStudyParts", into = "EventStudyParts")
)]
pub struct EventStudy {
    coefficients: Vec<f64>,
    covariance: Covariance,
    relative_periods: Vec<i64>,
    reference: i64,
}

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct EventStudyParts {
    coefficients: Vec<f64>,
    covariance: Covariance,
    relative_periods: Vec<i64>,
    reference: i64,
}

impl TryFrom<EventStudyParts> for EventStudy {
    type Error = InputError;

    fn try_from(parts: EventStudyParts) -> Result<Self, InputError> {
        Self::new(
            parts.coefficients,
            parts.covariance,
            parts.relative_periods,
            parts.reference,
        )
    }
}

impl From<EventStudy> for EventStudyParts {
    fn from(study: EventStudy) -> Self {
        Self {
            coefficients: study.coefficients,
            covariance: study.covariance,
            relative_periods: study.relative_periods,
            reference: study.reference,
        }
    }
}

impl EventStudy {
    /// An event study from its coefficients, their covariance, the relative
    /// period of each coefficient and the reference period.
    ///
    /// # Errors
    ///
    /// [`InputError::Empty`] for no coefficients, [`InputError::Length`] when
    /// the covariance's dimension or the number of periods differs from the
    /// number of coefficients, [`InputError::NotFinite`] for a coefficient that
    /// is not finite, [`InputError::Unsorted`] or [`InputError::Duplicate`] for
    /// periods that do not increase, and [`InputError::Duplicate`] for a
    /// reference that is among the periods, reported at its position.
    pub fn new(
        coefficients: Vec<f64>,
        covariance: Covariance,
        relative_periods: Vec<i64>,
        reference: i64,
    ) -> Result<Self, InputError> {
        check::non_empty("coefficients", coefficients.len())?;
        check::same_length("covariance", coefficients.len(), covariance.dim())?;
        check::same_length(
            "relative_periods",
            coefficients.len(),
            relative_periods.len(),
        )?;
        check::finite("coefficients", &coefficients)?;
        check::strictly_sorted_by("relative_periods", &relative_periods, i64::cmp)?;
        if let Ok(row) = relative_periods.binary_search(&reference) {
            return Err(InputError::Duplicate {
                what: "reference",
                row,
            });
        }
        Ok(Self {
            coefficients,
            covariance,
            relative_periods,
            reference,
        })
    }

    /// The coefficients, in increasing order of their relative periods.
    #[must_use]
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// The covariance of the coefficients.
    #[must_use]
    pub const fn covariance(&self) -> &Covariance {
        &self.covariance
    }

    /// The relative period of each coefficient, increasing.
    #[must_use]
    pub fn relative_periods(&self) -> &[i64] {
        &self.relative_periods
    }

    /// The reference period, normalised to zero and without a coefficient.
    #[must_use]
    pub const fn reference(&self) -> i64 {
        self.reference
    }

    /// The number of coefficients before the reference period.
    #[must_use]
    pub fn n_pre(&self) -> usize {
        self.relative_periods
            .partition_point(|&p| p < self.reference)
    }

    /// The number of coefficients after the reference period.
    #[must_use]
    pub fn n_post(&self) -> usize {
        self.relative_periods.len() - self.n_pre()
    }

    /// The relative periods before the reference.
    #[must_use]
    pub fn pre_periods(&self) -> &[i64] {
        &self.relative_periods[..self.n_pre()]
    }

    /// The relative periods after the reference.
    #[must_use]
    pub fn post_periods(&self) -> &[i64] {
        &self.relative_periods[self.n_pre()..]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity(dim: usize) -> Covariance {
        let values: Vec<f64> = (0..dim * dim)
            .map(|k| if k % (dim + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        Covariance::from_values(dim, &values).unwrap()
    }

    #[test]
    fn periods_split_at_the_reference() {
        let study = EventStudy::new(
            vec![0.1, -0.2, 0.3, 0.5],
            identity(4),
            vec![-3, -2, 0, 1],
            -1,
        )
        .unwrap();
        assert_eq!(study.n_pre(), 2);
        assert_eq!(study.n_post(), 2);
        assert_eq!(study.pre_periods(), &[-3, -2]);
        assert_eq!(study.post_periods(), &[0, 1]);
        assert_eq!(study.reference(), -1);
        assert_eq!(study.coefficients().len(), 4);
        assert_eq!(study.covariance().dim(), 4);
        assert_eq!(study.relative_periods(), &[-3, -2, 0, 1]);
    }

    #[test]
    fn anticipation_moves_periods_after_the_reference_into_the_post_set() {
        let study =
            EventStudy::new(vec![0.0; 5], identity(5), vec![-5, -4, -2, -1, 0], -3).unwrap();
        assert_eq!(study.pre_periods(), &[-5, -4]);
        assert_eq!(study.post_periods(), &[-2, -1, 0]);
    }

    #[test]
    fn inconsistent_inputs_are_refused() {
        assert_eq!(
            EventStudy::new(Vec::new(), identity(0), Vec::new(), -1),
            Err(InputError::Empty {
                what: "coefficients"
            })
        );
        assert_eq!(
            EventStudy::new(vec![0.0; 2], identity(3), vec![0, 1], -1)
                .unwrap_err()
                .to_string(),
            "covariance has length 3, expected 2"
        );
        assert!(EventStudy::new(vec![0.0; 2], identity(2), vec![0], -1).is_err());
        assert!(EventStudy::new(vec![0.0, f64::NAN], identity(2), vec![0, 1], -1).is_err());
        assert_eq!(
            EventStudy::new(vec![0.0; 2], identity(2), vec![1, 0], -1),
            Err(InputError::Unsorted {
                what: "relative_periods",
                row: 1
            })
        );
        assert_eq!(
            EventStudy::new(vec![0.0; 2], identity(2), vec![-1, 0], -1),
            Err(InputError::Duplicate {
                what: "reference",
                row: 0
            })
        );
    }
}
