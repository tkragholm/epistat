//! The error of the functions that take a design, weights or groups.

/// Invalid input to [`weighted_gram`](crate::weighted_gram),
/// [`weighted_xtz`](crate::weighted_xtz) and
/// [`independent_columns`](crate::design::independent_columns).
///
/// The functions check their arguments in the same way, so a design that is
/// accepted by one is accepted by the others.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DesignError {
    /// An argument that should have one entry per row has another length.
    #[error("{what} has {got} entries, expected {expected}")]
    Length {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    /// A weight is negative, infinite or NaN.
    #[error("the weight of row {row} is negative or not finite")]
    Weight { row: usize },
    /// The tolerance is negative, infinite or NaN.
    #[error("the tolerance is negative or not finite")]
    Tolerance,
}

/// Checks that there is one weight per row and that each is finite and not
/// negative. Zero is a valid weight.
pub(crate) fn check_weights(weights: &[f64], rows: usize) -> Result<(), DesignError> {
    check_length("weights", weights.len(), rows)?;
    match weights.iter().position(|&w| !(w.is_finite() && w >= 0.0)) {
        Some(row) => Err(DesignError::Weight { row }),
        None => Ok(()),
    }
}

pub(crate) const fn check_length(
    what: &'static str,
    got: usize,
    expected: usize,
) -> Result<(), DesignError> {
    if got == expected {
        Ok(())
    } else {
        Err(DesignError::Length {
            what,
            expected,
            got,
        })
    }
}
