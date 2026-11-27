//! The errors of the crate.
//!
//! [`Error`] is the error of the data views: a [`StaggeredPanel`] that does
//! not build returns [`Error::Input`] with the shared [`InputError`], as every
//! epistat crate reports invalid input. [`Classify`] gives a binding the kind
//! and a stable code.
//!
//! [`StaggeredPanel`]: crate::StaggeredPanel

use std::fmt;

use epistat_core::{Classify, ErrorKind, InputError};

/// Why the input of an estimator could not be used.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// A column of the wrong length, a value that is not finite, a negative
    /// weight, rows out of order, or a unit-level column that changes within
    /// a unit.
    #[error(transparent)]
    Input(#[from] InputError),
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(error) => error.kind(),
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
        }
    }
}
/// Errors from internal estimation steps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum InternalDidError {
    /// An estimation step failed, with a description.
    Estimation(String),
    /// An iterative procedure did not converge, with a description.
    Convergence(String),
}

impl fmt::Display for InternalDidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Estimation(message) => write!(f, "Estimation failed: {message}"),
            Self::Convergence(message) => write!(f, "Convergence error: {message}"),
        }
    }
}

impl std::error::Error for InternalDidError {}
