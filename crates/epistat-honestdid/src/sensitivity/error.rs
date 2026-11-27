//! The error of the sensitivity engine, which the facade maps into
//! [`crate::Error`].

use epistat_core::InputError;

use crate::error::{Error, SolverStatus};

/// Why a sensitivity computation failed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum EngineError {
    /// An event study, a target or a setting the engine cannot use.
    Input(InputError),
    /// A combination of restriction, variant and method the engine does not
    /// do.
    Unsupported(&'static str),
    /// A linear or conic program did not solve.
    Solver {
        /// The program.
        program: &'static str,
        /// How it ended.
        status: SolverStatus,
    },
    /// A critical value or quantile could not be found.
    CriticalValue(&'static str),
    /// A variance, direction or basis that must be non-degenerate is not.
    Degenerate(&'static str),
    /// A covariance that must be positive (semi)definite is not.
    NotPositiveDefinite(&'static str),
    /// The conditional test accepted no value of the grid.
    NoAcceptedValues,
    /// Internal objects disagree, such as prepared branches of another count
    /// than the functionals.
    Internal(&'static str),
    /// A simulation of a critical value failed.
    Inference(epistat_inference::Error),
    /// A program of epistat-optim failed.
    Optim(epistat_optim::Error),
}

impl EngineError {
    /// A value outside its domain.
    pub(crate) const fn domain(what: &'static str, value: f64, domain: &'static str) -> Self {
        Self::Input(InputError::Domain {
            what,
            value,
            domain,
            row: None,
        })
    }

    /// An argument of the wrong length.
    pub(crate) const fn length(what: &'static str, expected: usize, found: usize) -> Self {
        Self::Input(InputError::Length {
            what,
            expected,
            found,
        })
    }

    /// An argument without entries.
    pub(crate) const fn empty(what: &'static str) -> Self {
        Self::Input(InputError::Empty { what })
    }

    /// An argument with a value that is not finite at `row`.
    pub(crate) const fn not_finite(what: &'static str, row: usize) -> Self {
        Self::Input(InputError::NotFinite {
            what,
            row,
            column: None,
        })
    }

    /// The first value of `values` that is not finite.
    pub(crate) fn check_finite(what: &'static str, values: &[f64]) -> Result<(), Self> {
        epistat_core::check::finite(what, values).map_err(Self::Input)
    }

    /// Weights with no entry away from zero.
    pub(crate) const fn all_zero(what: &'static str) -> Self {
        Self::domain(what, 0.0, "at least one weight non-zero")
    }
}

impl From<InputError> for EngineError {
    fn from(error: InputError) -> Self {
        Self::Input(error)
    }
}

impl From<EngineError> for Error {
    fn from(error: EngineError) -> Self {
        match error {
            EngineError::Input(error) => Self::Input(error),
            EngineError::Unsupported(what) => Self::Unsupported { what },
            EngineError::Solver { program, status } => Self::Solver { program, status },
            EngineError::CriticalValue(what) => Self::CriticalValue { what },
            EngineError::Degenerate(what) => Self::Degenerate { what },
            EngineError::NotPositiveDefinite(what) => Self::NotPositiveDefinite { what },
            EngineError::NoAcceptedValues => Self::NoAcceptedValues,
            EngineError::Internal(what) => Self::Internal { what },
            EngineError::Inference(error) => Self::Inference(error),
            EngineError::Optim(error) => Self::Optim(error),
        }
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Error::from(self.clone()).fmt(f)
    }
}
