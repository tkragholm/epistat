//! Bias parameters named by the stratum they belong to.
//!
//! A misclassification has a sensitivity and a specificity in each of two
//! strata, and which strata depends on what is misclassified: cases and
//! non-cases for the exposure of a 2 x 2 table, the exposed and the unexposed
//! for its outcome, events and person-time for the exposure of a rate table.
//! Each analysis takes its parameters in a type whose fields name those
//! strata, so parameters built for one cannot be read as the other's.
//!
//! ```
//! use epistat_epi::bias::{ExposureMisclass, OutcomeMisclass, SeSp, correct_exposure};
//! use epistat_epi::Counts2x2;
//!
//! let exposure = ExposureMisclass {
//!     cases: SeSp::new(0.9, 0.95)?,
//!     noncases: SeSp::new(0.8, 0.97)?,
//! };
//! let outcome = OutcomeMisclass::nondifferential(SeSp::new(0.85, 0.99)?);
//! assert_eq!(outcome.exposed, outcome.unexposed);
//! let t = Counts2x2 {
//!     exposed_cases: 45.0,
//!     exposed_noncases: 94.0,
//!     unexposed_cases: 257.0,
//!     unexposed_noncases: 945.0,
//! };
//! assert!(correct_exposure(&t, &exposure)?.risk_ratio.is_some());
//! assert!(SeSp::new(0.5, 0.4).is_err());
//! # Ok::<(), epistat_epi::Error>(())
//! ```

use super::BiasError;
use crate::Error;

/// The sensitivity and specificity of a classification in one stratum, each
/// in `[0, 1]`, with `sensitivity + specificity > 1`, a classification better
/// than chance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeSp {
    sensitivity: f64,
    specificity: f64,
}

impl SeSp {
    /// The pair, checked.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a value outside `[0, 1]`, NaN included, or a
    /// sum of at most 1.
    pub fn new(sensitivity: f64, specificity: f64) -> Result<Self, Error> {
        if !(0.0..=1.0).contains(&sensitivity) || !(0.0..=1.0).contains(&specificity) {
            return Err(
                BiasError::Parameter("sensitivity and specificity must be in [0, 1]").into(),
            );
        }
        if sensitivity - (1.0 - specificity) <= 0.0 {
            return Err(BiasError::Parameter("sensitivity plus specificity must exceed 1").into());
        }
        Ok(Self {
            sensitivity,
            specificity,
        })
    }

    /// The sensitivity.
    #[must_use]
    pub const fn sensitivity(self) -> f64 {
        self.sensitivity
    }

    /// The specificity.
    #[must_use]
    pub const fn specificity(self) -> f64 {
        self.specificity
    }
}

/// The positive and negative predictive values of a classification in one
/// stratum, each in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Predictive {
    ppv: f64,
    npv: f64,
}

impl Predictive {
    /// The pair, checked.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a value outside `[0, 1]`, NaN included.
    pub fn new(ppv: f64, npv: f64) -> Result<Self, Error> {
        if !(0.0..=1.0).contains(&ppv) || !(0.0..=1.0).contains(&npv) {
            return Err(BiasError::Parameter("predictive values must be in [0, 1]").into());
        }
        Ok(Self { ppv, npv })
    }

    /// The positive predictive value.
    #[must_use]
    pub const fn ppv(self) -> f64 {
        self.ppv
    }

    /// The negative predictive value.
    #[must_use]
    pub const fn npv(self) -> f64 {
        self.npv
    }
}

/// The positive predictive value of a classification and its sensitivity in
/// one stratum, as a validation study of recorded cases reports them: the
/// predictive value in `[0, 1]` and the sensitivity in `(0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PpvSe {
    ppv: f64,
    sensitivity: f64,
}

impl PpvSe {
    /// The pair, checked.
    ///
    /// # Errors
    ///
    /// [`Error::Invalid`] for a predictive value outside `[0, 1]` or a
    /// sensitivity outside `(0, 1]`, NaN included.
    pub fn new(ppv: f64, sensitivity: f64) -> Result<Self, Error> {
        if !(0.0..=1.0).contains(&ppv) {
            return Err(BiasError::Parameter("predictive values must be in [0, 1]").into());
        }
        if !(sensitivity > 0.0 && sensitivity <= 1.0) {
            return Err(BiasError::Parameter("sensitivities must be in (0, 1]").into());
        }
        Ok(Self { ppv, sensitivity })
    }

    /// The positive predictive value.
    #[must_use]
    pub const fn ppv(self) -> f64 {
        self.ppv
    }

    /// The sensitivity.
    #[must_use]
    pub const fn sensitivity(self) -> f64 {
        self.sensitivity
    }
}

/// A parameter among the cases and among the non-cases, the strata of a
/// misclassified exposure.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ByOutcome<T> {
    /// Among the cases.
    pub cases: T,
    /// Among the non-cases.
    pub noncases: T,
}

/// A parameter among the exposed and among the unexposed, the strata of a
/// misclassified outcome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ByExposure<T> {
    /// Among the exposed.
    pub exposed: T,
    /// Among the unexposed.
    pub unexposed: T,
}

/// A parameter for the events and for the person-time of a rate table, the
/// strata of a misclassified exposure there.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ByRateCell<T> {
    /// For the classification of the events.
    pub events: T,
    /// For the classification of the person-time.
    pub person_time: T,
}

impl<T: Copy> ByOutcome<T> {
    /// The same parameter among cases and non-cases, a non-differential
    /// misclassification.
    #[must_use]
    pub const fn nondifferential(both: T) -> Self {
        Self {
            cases: both,
            noncases: both,
        }
    }
}

impl<T: Copy> ByExposure<T> {
    /// The same parameter among the exposed and the unexposed, a
    /// non-differential misclassification.
    #[must_use]
    pub const fn nondifferential(both: T) -> Self {
        Self {
            exposed: both,
            unexposed: both,
        }
    }
}

impl<T: Copy> ByRateCell<T> {
    /// The same parameter for events and person-time.
    #[must_use]
    pub const fn nondifferential(both: T) -> Self {
        Self {
            events: both,
            person_time: both,
        }
    }
}

/// The sensitivity and specificity of the exposure classification among
/// cases and among non-cases.
pub type ExposureMisclass = ByOutcome<SeSp>;

/// The sensitivity and specificity of the outcome classification among the
/// exposed and among the unexposed.
pub type OutcomeMisclass = ByExposure<SeSp>;

/// The sensitivity and specificity of the exposure classification of the
/// events and of the person-time of a rate table.
pub type RateMisclass = ByRateCell<SeSp>;

/// A value for each cell of a 2 x 2 table, named as the cells of
/// [`Counts2x2`](crate::Counts2x2).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cells<T> {
    /// Exposed cases.
    pub exposed_cases: T,
    /// Exposed non-cases.
    pub exposed_noncases: T,
    /// Unexposed cases.
    pub unexposed_cases: T,
    /// Unexposed non-cases.
    pub unexposed_noncases: T,
}

impl<T: Copy> Cells<T> {
    /// The same value in every cell.
    #[must_use]
    pub const fn every(value: T) -> Self {
        Self {
            exposed_cases: value,
            exposed_noncases: value,
            unexposed_cases: value,
            unexposed_noncases: value,
        }
    }
}

/// The two sensitivities and specificities as the arrays the corrections
/// compute with, stratum 0 first.
pub(crate) fn arrays(first: SeSp, second: SeSp) -> ([f64; 2], [f64; 2]) {
    (
        [first.sensitivity, second.sensitivity],
        [first.specificity, second.specificity],
    )
}

/// Sensitivities and specificities as two arrays, stratum 0 first, for the
/// tests that build one set of parameters and read it as either analysis.
#[cfg(test)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Pairs {
    pub se: [f64; 2],
    pub sp: [f64; 2],
}

#[cfg(test)]
impl Pairs {
    fn sesp(self, k: usize) -> SeSp {
        SeSp::new(self.se[k], self.sp[k]).expect("valid test parameters")
    }

    /// Cases first.
    pub(crate) fn exposure(self) -> ExposureMisclass {
        ByOutcome {
            cases: self.sesp(0),
            noncases: self.sesp(1),
        }
    }

    /// The exposed first.
    pub(crate) fn outcome(self) -> OutcomeMisclass {
        ByExposure {
            exposed: self.sesp(0),
            unexposed: self.sesp(1),
        }
    }

    /// The events first.
    pub(crate) fn rates(self) -> RateMisclass {
        ByRateCell {
            events: self.sesp(0),
            person_time: self.sesp(1),
        }
    }
}
