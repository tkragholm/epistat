//! The error of the draws that take parameters.

use std::fmt;

use epistat_core::{Classify, ErrorKind, InputError};

/// Why a draw was refused.
///
/// Every refusal is a parameter outside the range of its distribution, so the
/// one variant wraps the [`InputError`] that says which parameter and why. No
/// draw is taken before the check, so a refused call leaves the stream where
/// it was.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A parameter outside the range of its distribution.
    Input(InputError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Input(error) => error.fmt(f),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Input(error) => Some(error),
        }
    }
}

impl From<InputError> for Error {
    fn from(error: InputError) -> Self {
        Self::Input(error)
    }
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
