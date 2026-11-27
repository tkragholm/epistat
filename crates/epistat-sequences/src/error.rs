//! The one error of the crate.
//!
//! Every check runs before any computation, so an error means nothing was
//! computed. A slice of the wrong length, a symbol or label outside its range
//! or a negative weight is [`Error::Input`] with the shared [`InputError`], as
//! in every other epistat crate. Input that is well formed but breaks a
//! function's own contract, such as a number of types the sequences cannot
//! fill, is [`Error::Invalid`]. [`Classify`] gives a binding the kind and a
//! stable code.

use epistat_core::{Classify, ErrorKind, InputError, check};

/// Why a function refused its input.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Input the function cannot take: slices of different lengths, a symbol
    /// outside the alphabet, a cost table that is not dense, a negative weight
    /// or a cluster label outside the clusters declared.
    #[error(transparent)]
    Input(#[from] InputError),
    /// Input that is well formed but breaks the function's contract: no
    /// restart, no type, more types than sequences, an unknown distance.
    #[error("{0}")]
    Invalid(String),
}

impl Error {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        ErrorKind::Input
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::Invalid(_) => "sequences.invalid",
        }
    }
}

/// `found` equals `expected`, or [`InputError::Length`].
pub(crate) fn length(what: &'static str, expected: usize, found: usize) -> Result<(), Error> {
    Ok(check::same_length(what, expected, found)?)
}

/// A rank `value` below `limit`, or [`InputError::Domain`].
pub(crate) fn below(
    what: &'static str,
    value: usize,
    limit: usize,
    domain: &'static str,
) -> Result<(), Error> {
    if value < limit {
        Ok(())
    } else {
        #[expect(clippy::cast_precision_loss, reason = "the value is reported only")]
        Err(InputError::Domain {
            what,
            value: value as f64,
            domain,
            row: None,
        }
        .into())
    }
}

/// Every symbol rank is below `alphabet`, or [`InputError::Domain`] with the
/// first that is not.
pub(crate) fn check_symbols(
    what: &'static str,
    symbols: &[u8],
    alphabet: usize,
) -> Result<(), Error> {
    match symbols.iter().find(|&&rank| usize::from(rank) >= alphabet) {
        None => Ok(()),
        Some(&rank) => Err(InputError::Domain {
            what,
            value: f64::from(rank),
            domain: "below the alphabet size",
            row: None,
        }
        .into()),
    }
}

/// Every weight is finite and not negative, or the error of the first that is
/// not.
pub(crate) fn weights(what: &'static str, values: &[f64]) -> Result<(), Error> {
    for (row, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(InputError::NotFinite {
                what,
                row,
                column: None,
            }
            .into());
        }
        if value < 0.0 {
            return Err(InputError::Domain {
                what,
                value,
                domain: "[0, inf)",
                row: None,
            }
            .into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_unique_and_prefixed() {
        epistat_testkit::assert_codes_unique("sequences", &[Error::Invalid(String::new())]);
        assert_eq!(
            Error::from(InputError::Empty { what: "x" }).code(),
            "input.empty"
        );
    }
}
