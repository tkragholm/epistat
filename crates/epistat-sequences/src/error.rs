//! The error type of the crate.

/// Why a function refused its input.
///
/// Every check runs before any computation, so an error means nothing was
/// computed. The message names the lengths, labels or settings at fault.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum SequenceError {
    /// Input the function cannot take: slices of different lengths, a symbol
    /// outside the alphabet, a cost table that is not dense, a negative weight
    /// or a setting outside its range.
    #[error("{0}")]
    Input(String),
}

impl SequenceError {
    pub(crate) fn input(message: impl Into<String>) -> Self {
        Self::Input(message.into())
    }
}
