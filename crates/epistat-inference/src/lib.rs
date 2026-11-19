//! Standard errors, covariances, confidence intervals and tests for
//! estimators.
//!
//! The crate has six parts. [`influence`] gives variances from an
//! influence function, with or without clusters. [`sandwich`] gives the
//! M-estimator sandwich covariance. [`bootstrap`] draws multiplier and
//! cluster bootstrap replicates. [`interval`] turns an estimate and a standard
//! error, or a set of replicates, into a confidence interval. [`wald`] tests
//! that several estimates are jointly zero. [`Clusters`]
//! numbers the clusters the others sum over. [`simultaneous`] gives critical
//! values for bands that cover several estimates at once.
//!
//! Clusters are numbered in order of first appearance and every sum runs in a
//! fixed order, so results do not depend on hashing or on the number of
//! threads. Each module documents its formulas, edge cases and references.

pub mod bootstrap;
mod clusters;
pub mod influence;
pub mod interval;
pub mod sandwich;
pub mod simultaneous;
pub mod wald;

pub use clusters::Clusters;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum InferenceError {
    #[error("no observations")]
    Empty,
    #[error("{what} has {found} entries where {expected} were expected")]
    Length {
        what: &'static str,
        expected: usize,
        found: usize,
    },
    #[error("a replicate is NaN")]
    NanReplicate,
    #[error("the result is not finite")]
    NotFinite,
    #[error("alpha must lie in (0, 1), got {0}")]
    Alpha(f64),
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
