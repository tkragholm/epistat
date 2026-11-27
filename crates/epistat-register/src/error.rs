//! The one error of the crate.
//!
//! Every function that can refuse its input returns [`Error`]. A column of the
//! wrong length, offsets that are not a layout over the rows or rows out of
//! order are [`Error::Input`] with the shared [`InputError`], so they read as
//! in every other epistat crate. Input that is well formed but breaks a
//! function's own contract, such as a key code outside its range or an interval
//! that ends before it starts, is [`Error::Invalid`]. [`Classify`] gives a
//! binding the kind and a stable code.

use epistat_core::{Classify, ErrorKind, InputError, check};

/// Why a function refused its input or a price index lacked a year.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Input the function cannot take: columns of different lengths, offsets
    /// that are not a layout over the rows, rows out of order or more rows
    /// than a `u32` row number holds.
    #[error(transparent)]
    Input(#[from] InputError),
    /// Input that is well formed but breaks the function's contract: a key
    /// code outside its range, an interval that ends before it starts, a
    /// negative number of days.
    #[error("{0}")]
    Invalid(String),
    /// The reference year has no price index.
    #[error("Reference price year {0} is missing from the CPI table.")]
    MissingReferenceYear(i32),
    /// One or more years have no price index, sorted and without duplicates.
    #[error("Missing CPI adjustment factors for years: {}", join_years(.0))]
    MissingYears(Vec<i32>),
    /// A fit that could not be computed from valid input.
    #[error("{0}")]
    Numerical(String),
}

impl Error {
    pub(crate) fn invalid(message: impl Into<String>) -> Self {
        Self::Invalid(message.into())
    }

    pub(crate) fn numerical(message: impl Into<String>) -> Self {
        Self::Numerical(message.into())
    }
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(_)
            | Self::Invalid(_)
            | Self::MissingReferenceYear(_)
            | Self::MissingYears(_) => ErrorKind::Input,
            Self::Numerical(_) => ErrorKind::Numerical,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::Invalid(_) => "register.invalid",
            Self::MissingReferenceYear(_) => "register.missing_reference_year",
            Self::MissingYears(_) => "register.missing_years",
            Self::Numerical(_) => "register.numerical",
        }
    }
}

/// `found` equals `expected`, or [`InputError::Length`].
pub(crate) fn length(what: &'static str, expected: usize, found: usize) -> Result<(), Error> {
    Ok(check::same_length(what, expected, found)?)
}

/// `offsets` lay out `rows` rows, or the error of the first fault.
pub(crate) fn csr(what: &'static str, offsets: &[usize], rows: usize) -> Result<(), Error> {
    Ok(check::csr_offsets(what, offsets, rows)?)
}

/// `n` fits a `u32` row number, or [`InputError::TooLarge`].
pub(crate) fn row_count(what: &'static str, n: usize) -> Result<u32, Error> {
    u32::try_from(n).map_err(|_| {
        InputError::TooLarge {
            what,
            limit: u64::from(u32::MAX),
            found: n as u64,
        }
        .into()
    })
}

fn join_years(years: &[i32]) -> String {
    years
        .iter()
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_unique_and_prefixed() {
        let errors = [
            Error::Invalid(String::new()),
            Error::MissingReferenceYear(1900),
            Error::MissingYears(vec![1900]),
            Error::Numerical(String::new()),
        ];
        epistat_testkit::assert_codes_unique("register", &errors);
        assert_eq!(
            Error::from(InputError::Empty { what: "x" }).code(),
            "input.empty"
        );
    }
}
