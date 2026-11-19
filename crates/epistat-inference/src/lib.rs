//! Variances and intervals for estimators: from influence functions, from
//! sandwich estimators with or without clusters, and from the bootstrap.
//!
//! Clusters are numbered in order of first appearance and every sum runs in a
//! fixed order, so results do not depend on hashing or on the number of
//! threads.

pub mod bootstrap;
mod clusters;
pub mod influence;
pub mod interval;
pub mod sandwich;

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
