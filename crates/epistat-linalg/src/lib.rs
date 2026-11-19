//! Dense linear algebra on [faer] whose results do not depend on the number
//! of threads: solves that fall back through a chosen sequence of
//! factorisations, and products over the rows of a design matrix computed in
//! fixed blocks, and the B-spline and polynomial bases of series estimators.
//!
//! Factorisations run on one thread; the systems they solve are the small
//! `p x p` ones of regression. The row products run in parallel, one block of
//! 4096 rows per task, and add the blocks in order.

pub mod basis;
mod blocks;
mod solve;

pub use blocks::{ROW_BLOCK, matvec_into, row_scaled_gram, weighted_gram, weighted_xtz, xtv};
pub use solve::{Attempt, Factor, Method, Ridge, Solution, SolveError, solve};

/// Whether every entry of `matrix` is finite.
#[must_use]
pub fn is_finite(matrix: faer::MatRef<'_, f64>) -> bool {
    (0..matrix.ncols()).all(|j| (0..matrix.nrows()).all(|i| matrix[(i, j)].is_finite()))
}
