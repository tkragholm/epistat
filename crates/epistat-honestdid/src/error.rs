//! The error of the crate.
//!
//! [`Error`] says why an event study, a target or the options could not be
//! used ([`Error::Input`], with the shared [`InputError`] that every epistat
//! crate reports), which combination of restriction and method the analysis
//! does not do ([`Error::Unsupported`]), or why the computation failed on
//! accepted input. [`Classify`] gives a binding the kind and a stable code.

use epistat_core::{Classify, ErrorKind, InputError};

/// Why a sensitivity analysis or a pre-trend power calculation failed.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// An event study, a target or an option that cannot be used: a target
    /// of the wrong length or with no non-zero weight, a period that is not
    /// in the event study, a sensitivity value that is negative or not
    /// finite, a covariance entry that is undefined, too few pre-treatment
    /// coefficients.
    #[error(transparent)]
    Input(#[from] InputError),
    /// The restriction, its variant and the method do not go together, such
    /// as the fixed-length interval under a relative-magnitude restriction.
    #[error("not available: {what}")]
    Unsupported {
        /// What was asked for.
        what: &'static str,
    },
    /// A covariance block that must be positive definite is not.
    #[error("the covariance of the {what} is not positive definite")]
    NotPositiveDefinite {
        /// The block.
        what: &'static str,
    },
    /// No slope in the search interval gives the pre-test the requested
    /// power.
    #[error("no slope in [0, {upper}] has power {target}; the power there is {power_at_upper}")]
    NoSlope {
        /// The requested power.
        target: f64,
        /// The upper end of the search interval.
        upper: f64,
        /// The power at the upper end.
        power_at_upper: f64,
    },
    /// A linear or conic program of the analysis did not solve, or a
    /// critical value could not be found; the message is the solver's.
    #[error("the sensitivity computation failed: {0}")]
    Solver(String),
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(_) => ErrorKind::Input,
            Self::Unsupported { .. } => ErrorKind::Unsupported,
            Self::NotPositiveDefinite { .. } | Self::NoSlope { .. } | Self::Solver(_) => {
                ErrorKind::Numerical
            }
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::Unsupported { .. } => "honestdid.unsupported",
            Self::NotPositiveDefinite { .. } => "honestdid.not_positive_definite",
            Self::NoSlope { .. } => "honestdid.no_slope",
            Self::Solver(_) => "honestdid.solver",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_distinct_and_kinds_follow_the_variant() {
        let errors = [
            Error::Unsupported { what: "x" },
            Error::NotPositiveDefinite { what: "x" },
            Error::NoSlope {
                target: 0.5,
                upper: 1.0,
                power_at_upper: 0.1,
            },
            Error::Solver("x".into()),
        ];
        let mut codes: Vec<_> = errors.iter().map(Classify::code).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), errors.len());
        assert_eq!(errors[0].kind(), ErrorKind::Unsupported);
        assert_eq!(errors[3].kind(), ErrorKind::Numerical);
        let input = Error::from(InputError::Empty { what: "x" });
        assert_eq!(input.kind(), ErrorKind::Input);
        assert_eq!(input.code(), "input.empty");
    }
}
