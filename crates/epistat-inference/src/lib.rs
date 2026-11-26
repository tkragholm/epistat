//! Standard errors, covariances, confidence intervals and tests for
//! estimators.
//!
//! The crate has ten parts. [`influence`] gives variances from an
//! influence function, with or without clusters. [`sandwich`] gives the
//! M-estimator sandwich covariance. [`bootstrap`] draws multiplier and
//! cluster bootstrap replicates. [`interval`] turns an estimate and a standard
//! error, or a set of replicates, into a confidence interval. [`wald`] tests
//! that several estimates are jointly zero. [`Clusters`]
//! numbers the clusters the others sum over. [`simultaneous`] gives critical
//! values for bands that cover several estimates at once. [`pooling`] pools an
//! estimate over multiply imputed data sets by Rubin's rules. [`multiplicity`]
//! adjusts p-values for several tests. [`curve`] gives the contrast of a spline
//! term against a reference value, with pointwise or simultaneous bands.
//!
//! Clusters are numbered in order of first appearance and every sum runs in a
//! fixed order, so results do not depend on hashing or on the number of
//! threads. Each module documents its formulas, edge cases and references.
#![warn(missing_docs)]

pub mod bootstrap;
mod clusters;
pub mod curve;
pub mod influence;
pub mod interval;
pub mod multiplicity;
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
    /// A p-value is NaN or outside `[0, 1]`.
    #[error("a p-value must lie in [0, 1], got {0}")]
    PValue(f64),
    /// The number of comparisons is below the number of p-values.
    #[error("{comparisons} comparisons for {p_values} p-values")]
    Comparisons {
        /// The number of comparisons that was given.
        comparisons: usize,
        /// The number of p-values.
        p_values: usize,
    },
    /// A covariance matrix is not symmetric.
    #[error("a covariance matrix is not symmetric")]
    NotSymmetric,
    /// A value is outside the domain of a transformation.
    #[error("{value} is outside the domain of the {transform} transformation")]
    Domain {
        /// The name of the transformation.
        transform: &'static str,
        /// The value.
        value: f64,
    },
    /// Too few imputed data sets for a test of several parameters.
    #[error("{what} needs k (m - 1) > 4, with k = {k} parameters and m = {m} data sets")]
    Imputations {
        /// The method that needs more data sets.
        what: &'static str,
        /// The number of parameters.
        k: usize,
        /// The number of imputed data sets.
        m: usize,
    },
    /// A linear system could not be solved.
    #[error(transparent)]
    Solve(#[from] epistat_linalg::Error),
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
