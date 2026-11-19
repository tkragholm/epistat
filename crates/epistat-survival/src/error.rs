//! The error type of the crate.

/// Why an estimator refused its input.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The input cannot be used: slices of different lengths, a code outside
    /// its range, a setting without a meaning, or data that contradicts
    /// itself. Nothing was estimated.
    #[error("{0}")]
    Input(String),
}

impl Error {
    pub(crate) fn input(message: impl Into<String>) -> Self {
        Self::Input(message.into())
    }
}

/// The result of an estimator.
pub type Result<T> = std::result::Result<T, Error>;
