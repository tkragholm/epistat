//! Dense linear algebra on [faer] whose results do not depend on the number
//! of threads: solves that fall back through a chosen sequence of
//! factorisations, and products over the rows of a design matrix computed in
//! fixed blocks, and the polynomial, B-spline and spline bases of series and regression estimators.
//!
//! [`Mat`] and [`MatRef`] are faer's matrix types, re-exported so that a crate
//! that takes or returns them need not depend on faer itself. [`Side`] is
//! faer's triangle selector for symmetric routines, re-exported for the same
//! reason. [`matmul`], [`Accum`], [`MatMut`] and [`get_global_parallelism`] are
//! faer's general matrix product and the types it takes, for callers that
//! need the product of faer's kernel into their own buffers.
//!
//! The modules, each with its own method notes:
//!
//! - [`solve`](crate::solve()) solves with a fallback sequence of
//!   factorisations, each on one thread.
//! - `blocks` holds the products over the rows of a design matrix, computed
//!   in blocks of [`ROW_BLOCK`] rows and added in block order.
//! - [`cholesky`] holds the lower Cholesky factor with a caller-chosen pivot
//!   rule.
//! - [`design`] finds linearly independent columns and selects rows and
//!   columns.
//! - [`basis`] holds the polynomial, B-spline, restricted cubic spline and
//!   natural cubic spline bases, Harrell's knot rules and design matrices.
//! - [`eigen`] holds a Jacobi eigendecomposition of a small symmetric matrix and
//!   the generalised inverse and its quadratic form.
#![warn(missing_docs)]

pub mod basis;
mod blocks;
pub mod cholesky;
pub mod design;
pub mod eigen;
mod error;
mod solve;

pub use blocks::{ROW_BLOCK, matvec_into, row_scaled_gram, weighted_gram, weighted_xtz, xtv};
pub use error::DesignError;
pub use faer::linalg::matmul::matmul;
pub use faer::{Accum, Mat, MatMut, MatRef, Side, get_global_parallelism, mat};
pub use solve::{Attempt, Factor, Method, Ridge, Solution, SolveError, inverse, solve};

/// Whether every entry of `matrix` is finite.
#[must_use]
pub fn is_finite(matrix: faer::MatRef<'_, f64>) -> bool {
    (0..matrix.ncols()).all(|j| (0..matrix.nrows()).all(|i| matrix[(i, j)].is_finite()))
}
