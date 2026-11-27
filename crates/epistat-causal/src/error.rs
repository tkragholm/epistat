//! The error type of the crate.

/// Why an estimator refused its input.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// The input cannot be used: slices of different lengths, a value outside
    /// its range, an estimand the method does not estimate, a model with an
    /// aliased column, or data that contradicts itself. Nothing was estimated.
    #[error("{0}")]
    Input(String),
    /// The conditions the method asks of the weights cannot be met by any
    /// weights, for example target moments outside the range of the group.
    /// Nothing was estimated.
    #[error("{0}")]
    Infeasible(String),
    /// An iteration did not reach its tolerance: a fit of a nuisance model, a
    /// fluctuation or a solver. Nothing was estimated.
    #[error("{0}")]
    NoConvergence(String),
}

impl Error {
    pub(crate) fn input(message: impl Into<String>) -> Self {
        Self::Input(message.into())
    }
}

/// The result of an estimator.
pub type Result<T> = std::result::Result<T, Error>;
