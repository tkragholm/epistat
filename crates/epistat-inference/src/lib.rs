//! Standard errors, covariances, confidence intervals and tests for
//! estimators.
//!
//! The crate has ten parts. [`influence`] gives covariances from an
//! influence function, with or without clusters, through [`Influence`].
//! [`sandwich`] gives the M-estimator sandwich covariance and its
//! small-sample factors. [`bootstrap`] draws multiplier and cluster bootstrap
//! replicates and unit resampling counts. [`interval`] turns an estimate and
//! a standard error, or a set of replicates, into a confidence interval.
//! [`wald`] tests that several estimates are jointly zero. [`Clusters`]
//! numbers the clusters the others sum over. [`simultaneous`] gives critical
//! values for bands that cover several estimates at once. [`pooling`] pools
//! an estimate over multiply imputed data sets by Rubin's rules.
//! [`multiplicity`] adjusts p-values for several tests. [`curve`] gives the
//! contrast of a spline term against a reference value, with pointwise or
//! simultaneous bands.
//!
//! # Conventions
//!
//! - Influence functions are on one scale, per unit, with
//!   `theta_hat - theta ~ sum(psi_i) / n` and variance `sum(psi_i^2) / n^2`.
//!   [`Influence`] records the [`ClusterRule`] its estimator used.
//! - A confidence level is an `epistat_core::Level` and an interval an
//!   `epistat_core::Interval`, which carries its level.
//! - A randomised call takes an `epistat_core::Bootstrap`, the number of
//!   replicates and a root seed; replicate `r` draws from
//!   `replicate_seed(seed, r)`.
//! - Invalid input is [`Error::Input`] with the shared
//!   `epistat_core::InputError`; [`Error`] implements `epistat_core::Classify`.
//!
//! Clusters are numbered in a fixed order (sorted code order for
//! [`Clusters::from_codes`] and [`Clusters::per_unit`]) and every sum runs in
//! a fixed order, so results do not depend on hashing or on the number of
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
pub use influence::{ClusterRule, Influence};

use epistat_core::check;
use epistat_core::{Classify, ErrorKind, InputError};

/// The ways an inference routine can refuse its input or fail.
///
/// Invalid input is [`Error::Input`] with the shared [`InputError`], so a
/// column of the wrong length or a NaN replicate is reported as it is in every
/// other epistat crate. The other variants are failures of the computation on
/// accepted input. [`Classify`] gives a binding the kind and a stable code.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// The input cannot be used.
    #[error(transparent)]
    Input(#[from] InputError),
    /// The result is NaN or infinite.
    #[error("the result is not finite")]
    NotFinite,
    /// A covariance matrix is not symmetric.
    #[error("a covariance matrix is not symmetric")]
    NotSymmetric,
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

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(error) => error.kind(),
            Self::NotSymmetric | Self::Imputations { .. } => ErrorKind::Input,
            Self::NotFinite | Self::Solve(_) => ErrorKind::Numerical,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::NotFinite => "inference.not_finite",
            Self::NotSymmetric => "inference.not_symmetric",
            Self::Imputations { .. } => "inference.imputations",
            Self::Solve(_) => "inference.solve",
        }
    }
}

/// The error type of epistat-inference 0.3.
#[deprecated(since = "0.4.0", note = "use epistat_inference::Error")]
pub type InferenceError = Error;

fn check_length(what: &'static str, expected: usize, found: usize) -> Result<(), Error> {
    Ok(check::same_length(what, expected, found)?)
}

fn empty(what: &'static str) -> Error {
    Error::Input(InputError::Empty { what })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_are_distinct_and_kinds_follow_the_variant() {
        let errors = [
            Error::NotFinite,
            Error::NotSymmetric,
            Error::Imputations {
                what: "D1",
                k: 1,
                m: 2,
            },
            Error::Solve(epistat_linalg::Error::Singular),
        ];
        let mut codes: Vec<&str> = errors.iter().map(Classify::code).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), errors.len());
        assert!(codes.iter().all(|c| c.starts_with("inference.")));
        assert_eq!(Error::NotFinite.kind(), ErrorKind::Numerical);
        assert_eq!(Error::NotSymmetric.kind(), ErrorKind::Input);
        let input = Error::from(InputError::Empty { what: "x" });
        assert_eq!(
            (input.kind(), input.code()),
            (ErrorKind::Input, "input.empty")
        );
    }
}
