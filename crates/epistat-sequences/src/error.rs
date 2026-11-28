//! The one error of the crate.
//!
//! Every check runs before any computation, so an error means nothing was
//! computed. A slice of the wrong length, a symbol or label outside its range
//! or a negative weight is [`Error::Input`] with the shared [`InputError`], as
//! in every other epistat crate. Input that is well formed but breaks a
//! function's own contract, such as a number of types the sequences cannot
//! fill, has a variant of its own. [`Classify`] gives a binding the kind and a
//! stable code.

use epistat_core::{Classify, ErrorKind, InputError, check};

/// Why a function refused its input.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Input the function cannot take: slices of different lengths, a symbol
    /// outside the alphabet, a cost table that is not dense, a negative weight,
    /// a cluster label outside the clusters declared, a panel without length
    /// or alphabet, or more sequences than 32-bit positions can number.
    #[error(transparent)]
    Input(#[from] InputError),
    /// `found` values do not divide into rows of `width`: encoded symbols into
    /// sequences of a common length, or replicate weights into replicates of
    /// one weight per sequence.
    #[error("{what}: {found} values do not divide into rows of {width}")]
    Ragged {
        /// The argument.
        what: &'static str,
        /// The number of values given.
        found: usize,
        /// The length of a row.
        width: usize,
    },
    /// A fit asked for no restart.
    #[error("a fit needs at least one restart")]
    NoRestarts,
    /// A fit asked for `k` types from `sequences` distinct sequences: none, or
    /// more types than sequences.
    #[error("asked for {k} types from {sequences} distinct sequences")]
    Types {
        /// The types asked for.
        k: usize,
        /// The distinct sequences there are.
        sequences: usize,
    },
    /// No sequence has a positive weight, so there is nothing to draw from.
    #[error("no sequence has a positive weight; there is nothing to draw from")]
    NoPositiveWeight,
    /// A distance name other than `"dhd"`, `"hamming"` and `"om"`.
    #[error("unknown distance {name:?}; choose from \"dhd\", \"hamming\" and \"om\"")]
    UnknownDistance {
        /// The name given.
        name: String,
    },
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(error) => error.kind(),
            Self::Ragged { .. }
            | Self::NoRestarts
            | Self::Types { .. }
            | Self::NoPositiveWeight
            | Self::UnknownDistance { .. } => ErrorKind::Input,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::Ragged { .. } => "sequences.ragged",
            Self::NoRestarts => "sequences.no_restarts",
            Self::Types { .. } => "sequences.types",
            Self::NoPositiveWeight => "sequences.no_positive_weight",
            Self::UnknownDistance { .. } => "sequences.unknown_distance",
        }
    }
}

/// The error of a count `what` of 0 where at least 1 is needed.
pub(crate) const fn positive(what: &'static str) -> InputError {
    InputError::Domain {
        what,
        value: 0.0,
        domain: "at least 1",
        row: None,
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
        epistat_testkit::assert_codes_unique(
            "sequences",
            &[
                Error::Ragged {
                    what: "x",
                    found: 3,
                    width: 2,
                },
                Error::NoRestarts,
                Error::Types { k: 0, sequences: 1 },
                Error::NoPositiveWeight,
                Error::UnknownDistance {
                    name: String::new(),
                },
            ],
        );
        assert_eq!(
            Error::from(InputError::Empty { what: "x" }).code(),
            "input.empty"
        );
    }
}
