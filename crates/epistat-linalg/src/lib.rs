//! Dense linear algebra on [faer] whose results do not depend on the number
//! of threads: solves that fall back through a chosen sequence of
//! factorisations, and products over the rows of a design matrix computed in
//! fixed blocks, and the B-spline and polynomial bases of series estimators.
//!
//! [`Mat`] and [`MatRef`] are faer's matrix types, re-exported so that a crate
//! that takes or returns them need not depend on faer itself. [`Side`] is
//! faer's triangle selector for symmetric routines, re-exported for the same
//! reason.
//!
//! The modules, each with its own method notes:
//!
//! - [`solve`](crate::solve()) solves with a fallback sequence of
//!   factorisations, each on one thread.
//! - `blocks` holds the products over the rows of a design matrix, computed
//!   in blocks of [`ROW_BLOCK`] rows and added in block order.
//! - [`design`] finds linearly independent columns and selects rows and
//!   columns.
//! - [`basis`] holds the polynomial and B-spline bases.
//! - [`eigen`] holds a Jacobi eigendecomposition of a small symmetric matrix and
//!   the quadratic form of its generalised inverse.

pub mod basis;
mod blocks;
pub mod design;
pub mod eigen;
mod error;
mod solve;

pub use blocks::{ROW_BLOCK, matvec_into, row_scaled_gram, weighted_gram, weighted_xtz, xtv};
pub use error::DesignError;
pub use faer::{Mat, MatRef, Side, mat};
pub use solve::{Attempt, Factor, Method, Ridge, Solution, SolveError, inverse, solve};

/// Whether every entry of `matrix` is finite.
#[must_use]
pub fn is_finite(matrix: faer::MatRef<'_, f64>) -> bool {
    (0..matrix.ncols()).all(|j| (0..matrix.nrows()).all(|i| matrix[(i, j)].is_finite()))
}
