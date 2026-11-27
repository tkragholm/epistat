//! The one error of the crate.
//!
//! Every fit, comparison and summary returns [`Error`]. Invalid input is
//! [`Error::Input`] with the shared [`InputError`], checked once where it
//! enters: [`CostData`](crate::CostData) and
//! [`LongitudinalData`](crate::LongitudinalData) check the columns, and each
//! entry point checks its options. The other variants are failures of the
//! computation on accepted input. [`Classify`] gives a binding the kind and a
//! stable code.
//!
//! Under [`FitStrategy::Relaxed`](crate::FitStrategy::Relaxed) a fit that ends
//! in [`Error::NotConverged`], [`Error::Saturated`], [`Error::Singular`] or
//! [`Error::NotFinite`] is
//! retried with a stronger ridge; every other error ends the fit at once.

use epistat_core::{Classify, ErrorKind, InputError};

/// Why a model could not be fitted or a summary could not be formed.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// The input cannot be used: a column of the wrong length, a value that
    /// is not finite, a negative outcome or weight, an option outside its
    /// range.
    #[error(transparent)]
    Input(#[from] InputError),
    /// An option, or a combination of options, is not valid as a whole, such
    /// as a prior with a variance of zero.
    #[error("invalid {what}")]
    Invalid {
        /// What is invalid.
        what: &'static str,
    },
    /// No outcome is positive, so the positive part has nothing to fit.
    #[error("no outcome is positive")]
    NoPositiveOutcome,
    /// The family random effects were asked for and the data have no family
    /// codes.
    #[error("the family random effects need the family of each row")]
    MissingFamilies,
    /// The options ask for something the model does not do, such as weights
    /// in the log-normal model or an elastic net on a compact design.
    #[error("not available: {what}")]
    Unsupported {
        /// What was asked for.
        what: &'static str,
    },
    /// The iteration did not meet its convergence rule.
    #[error("no convergence in {iterations} iterations")]
    NotConverged {
        /// The iterations run.
        iterations: usize,
    },
    /// The converged linear predictor reached the bound of the log link, so
    /// the predictions are no longer multiplicative in the covariates.
    #[error("the linear predictor reached the bound of the log link")]
    Saturated,
    /// A linear system of the fit could not be solved.
    #[error("a linear system could not be solved")]
    Singular,
    /// The information matrix of a compact fit is too close to singular for
    /// the compact products to be trusted; the dense fit can still be run.
    #[error("the information matrix is too close to singular for a compact fit")]
    IllConditioned,
    /// A solve or an iterate gave values that are not finite.
    #[error("the fit gave values that are not finite")]
    NotFinite,
    /// More replicates of a bootstrap failed than were allowed.
    #[error("{failures} bootstrap replicates failed, more than the {allowed} allowed")]
    TooManyFailures {
        /// The replicates that failed.
        failures: usize,
        /// The failures allowed.
        allowed: usize,
    },
    /// A chain kept fewer draws than a convergence summary needs.
    #[error("a chain kept {found} draws, fewer than the {minimum} needed")]
    TooFewDraws {
        /// The fewest draws needed.
        minimum: usize,
        /// The fewest draws any chain kept.
        found: usize,
    },
    /// A weighted product of the design was refused.
    #[error(transparent)]
    Linalg(#[from] epistat_linalg::Error),
}

impl Error {
    /// Whether a stronger ridge can be expected to help: the failures that
    /// [`FitStrategy::Relaxed`](crate::FitStrategy::Relaxed) retries.
    pub(crate) const fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::NotConverged { .. } | Self::Saturated | Self::Singular | Self::NotFinite
        )
    }
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(error) => error.kind(),
            Self::Linalg(error) => error.kind(),
            Self::Invalid { .. } | Self::NoPositiveOutcome | Self::MissingFamilies => {
                ErrorKind::Input
            }
            Self::Unsupported { .. } => ErrorKind::Unsupported,
            Self::NotConverged { .. }
            | Self::Saturated
            | Self::Singular
            | Self::IllConditioned
            | Self::NotFinite
            | Self::TooManyFailures { .. }
            | Self::TooFewDraws { .. } => ErrorKind::Numerical,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::Linalg(error) => error.code(),
            Self::Invalid { .. } => "semicontinuous.invalid",
            Self::NoPositiveOutcome => "semicontinuous.no_positive_outcome",
            Self::MissingFamilies => "semicontinuous.missing_families",
            Self::Unsupported { .. } => "semicontinuous.unsupported",
            Self::NotConverged { .. } => "semicontinuous.not_converged",
            Self::Saturated => "semicontinuous.saturated",
            Self::Singular => "semicontinuous.singular",
            Self::IllConditioned => "semicontinuous.ill_conditioned",
            Self::NotFinite => "semicontinuous.not_finite",
            Self::TooManyFailures { .. } => "semicontinuous.too_many_failures",
            Self::TooFewDraws { .. } => "semicontinuous.too_few_draws",
        }
    }
}

/// A value outside `domain`, as [`InputError::Domain`].
pub(crate) fn domain(what: &'static str, value: f64, domain: &'static str) -> Error {
    InputError::Domain {
        what,
        value,
        domain,
        row: None,
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_distinct_and_kinds_follow_the_variant() {
        let own = [
            Error::Invalid { what: "priors" },
            Error::NoPositiveOutcome,
            Error::MissingFamilies,
            Error::Unsupported { what: "weights" },
            Error::NotConverged { iterations: 50 },
            Error::Saturated,
            Error::Singular,
            Error::IllConditioned,
            Error::NotFinite,
            Error::TooManyFailures {
                failures: 51,
                allowed: 50,
            },
            Error::TooFewDraws {
                minimum: 4,
                found: 2,
            },
        ];
        epistat_testkit::assert_codes_unique("semicontinuous", &own);
        for error in &own[..3] {
            assert_eq!(error.kind(), ErrorKind::Input, "{error:?}");
        }
        assert_eq!(own[3].kind(), ErrorKind::Unsupported);
        for error in &own[4..] {
            assert_eq!(error.kind(), ErrorKind::Numerical, "{error:?}");
        }
        assert_eq!(
            Error::from(InputError::Empty { what: "y" }).code(),
            "input.empty"
        );
    }

    #[test]
    fn only_numerical_failures_of_the_iteration_are_retried() {
        assert!(Error::NotConverged { iterations: 1 }.is_retryable());
        assert!(Error::Saturated.is_retryable());
        assert!(Error::Singular.is_retryable());
        assert!(Error::NotFinite.is_retryable());
        assert!(!Error::NoPositiveOutcome.is_retryable());
        assert!(!Error::Linalg(epistat_linalg::Error::Singular).is_retryable());
    }
}
