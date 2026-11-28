//! The one error of the crate.

use epistat_core::{Classify, ErrorKind, InputError};

/// Why a solve, a product over a design, or a basis refused its input or
/// failed.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// The input cannot be used: a matrix that is not square, a right-hand
    /// side, weight or group column of the wrong length, a weight that is
    /// negative or not finite, a knot that is not finite or out of order.
    #[error(transparent)]
    Input(#[from] InputError),
    /// No attempted factorisation gave a finite solution: the matrix is
    /// singular to working precision, is not positive definite for a
    /// Cholesky-only sequence, or holds a value that is not finite.
    #[error("no factorisation gave a finite solution")]
    Singular,
    /// The boundary knots are not finite, or the lower is not below the upper.
    #[error("the boundary knots must be finite with lower below upper")]
    Boundary,
    /// An internal knot is not strictly inside the boundary.
    #[error("the internal knots must lie strictly inside the boundary")]
    OutsideBoundary,
    /// Fewer than three knots were given for a restricted cubic spline, or
    /// fewer were asked for.
    #[error("a restricted cubic spline needs at least three knots")]
    TooFewKnots,
    /// The sample has too few values to place knots.
    #[error("too few observations to place the knots")]
    TooFewObservations,
    /// Fewer than three distinct knots can be placed.
    #[error("fewer than three distinct knots can be placed")]
    TooFewDistinct,
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(error) => error.kind(),
            Self::Singular => ErrorKind::Numerical,
            Self::Boundary
            | Self::OutsideBoundary
            | Self::TooFewKnots
            | Self::TooFewObservations
            | Self::TooFewDistinct => ErrorKind::Input,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::Singular => "linalg.singular",
            Self::Boundary => "linalg.boundary",
            Self::OutsideBoundary => "linalg.outside_boundary",
            Self::TooFewKnots => "linalg.too_few_knots",
            Self::TooFewObservations => "linalg.too_few_observations",
            Self::TooFewDistinct => "linalg.too_few_distinct",
        }
    }
}

/// Checks that there is one weight per row and that each is finite and not
/// negative. Zero is a valid weight.
pub(crate) fn check_weights(weights: &[f64], rows: usize) -> Result<(), Error> {
    check_length("weights", weights.len(), rows)?;
    weight_error(weights).map_or(Ok(()), Err)
}

/// The error of the first weight that is negative or not finite, if any:
/// [`InputError::NotFinite`] for NaN or an infinity, [`InputError::Domain`]
/// for a negative weight. The products over a design refuse their weights
/// with it, and a caller that checks weights itself can return the same
/// error.
#[must_use]
pub fn weight_error(weights: &[f64]) -> Option<Error> {
    let row = weights.iter().position(|&w| !(w.is_finite() && w >= 0.0))?;
    let value = weights[row];
    Some(Error::Input(if value.is_finite() {
        InputError::Domain {
            what: "weights",
            value,
            domain: "[0, inf)",
            row: None,
        }
    } else {
        InputError::NotFinite {
            what: "weights",
            row,
            column: None,
        }
    }))
}

pub(crate) const fn check_length(
    what: &'static str,
    got: usize,
    expected: usize,
) -> Result<(), Error> {
    if got == expected {
        Ok(())
    } else {
        Err(Error::Input(InputError::Length {
            what,
            expected,
            found: got,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_distinct_and_kinds_follow_the_cause() {
        let all = [
            Error::Singular,
            Error::Boundary,
            Error::OutsideBoundary,
            Error::TooFewKnots,
            Error::TooFewObservations,
            Error::TooFewDistinct,
        ];
        epistat_testkit::assert_codes_unique("linalg", &all);
        assert_eq!(Error::Singular.kind(), ErrorKind::Numerical);
        assert_eq!(Error::TooFewKnots.kind(), ErrorKind::Input);
        let weight = weight_error(&[1.0, -2.0]).unwrap();
        assert_eq!(
            (weight.kind(), weight.code()),
            (ErrorKind::Input, "input.domain")
        );
        assert_eq!(
            weight_error(&[1.0, f64::NAN]),
            Some(Error::Input(InputError::NotFinite {
                what: "weights",
                row: 1,
                column: None
            }))
        );
        assert_eq!(weight_error(&[0.0, 1.0]), None);
    }
}
