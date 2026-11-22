//! Standard errors, covariances, confidence intervals and tests for
//! estimators.
//!
//! The crate has eight parts. [`influence`] gives variances from an
//! influence function, with or without clusters. [`sandwich`] gives the
//! M-estimator sandwich covariance. [`bootstrap`] draws multiplier and
//! cluster bootstrap replicates. [`interval`] turns an estimate and a standard
//! error, or a set of replicates, into a confidence interval. [`wald`] tests
//! that several estimates are jointly zero. [`Clusters`]
//! numbers the clusters the others sum over. [`simultaneous`] gives critical
//! values for bands that cover several estimates at once. [`pooling`] pools an
//! estimate over multiply imputed data sets by Rubin's rules.
//!
//! Clusters are numbered in order of first appearance and every sum runs in a
//! fixed order, so results do not depend on hashing or on the number of
//! threads. Each module documents its formulas, edge cases and references.
#![warn(missing_docs)]

pub mod bootstrap;
mod clusters;
pub mod influence;
pub mod interval;
pub mod pooling;
pub mod sandwich;
pub mod simultaneous;
pub mod wald;

pub use clusters::Clusters;

/// The ways an inference routine can refuse its input or fail.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum InferenceError {
    /// The input has no observations.
    #[error("no observations")]
    Empty,
    /// An input has the wrong number of entries.
    #[error("{what} has {found} entries where {expected} were expected")]
    Length {
        /// The name of the input.
        what: &'static str,
        /// The length that was required.
        expected: usize,
        /// The length that was given.
        found: usize,
    },
    /// A bootstrap replicate is NaN.
    #[error("a replicate is NaN")]
    NanReplicate,
    /// The result is NaN or infinite.
    #[error("the result is not finite")]
    NotFinite,
    /// The significance level is outside the open interval from 0 to 1.
    #[error("alpha must lie in (0, 1), got {0}")]
    Alpha(f64),
    /// A linear system could not be solved.
    #[error(transparent)]
    Solve(#[from] epistat_linalg::SolveError),
}

fn check_length(what: &'static str, expected: usize, found: usize) -> Result<(), InferenceError> {
    if expected == found {
        Ok(())
    } else {
        Err(InferenceError::Length {
            what,
            expected,
            found,
        })
    }
}
