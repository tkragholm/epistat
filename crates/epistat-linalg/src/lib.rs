//! Dense linear algebra on [faer] whose results do not depend on the number
//! of threads: solves that fall back through a chosen sequence of
//! factorisations, and products over the rows of a design matrix computed in
//! fixed blocks, and the polynomial, B-spline and spline bases of series and regression estimators.
//!
//! [`Mat`], [`MatRef`] and [`Col`] are faer's matrix and column types,
//! re-exported so that a crate that takes or returns them need not depend on
//! faer itself. [`Side`] is faer's triangle selector for symmetric routines,
//! re-exported for the same reason. [`matmul`], [`Accum`], [`MatMut`], [`Par`]
//! and [`get_global_parallelism`] are faer's general matrix product and the
//! types it takes, for callers that need the product of faer's kernel into
//! their own buffers. faer's factorisation traits are not re-exported: a
//! caller solves through [`solve`](crate::solve()), [`Factor`] or the
//! wrappers of [`solve`](mod@solve), each of which runs on one thread.
//!
//! The modules, each with its own method notes:
//!
//! - [`solve`](crate::solve()) solves with a fallback sequence of
//!   factorisations, each on one thread.
//!   [`cholesky_factor`] is its Cholesky step alone, with faer's dynamic
//!   regularisation as an option, and [`CholeskySolver`] repeats it for many
//!   systems of one size without allocating.
//! - `blocks` holds the products over the rows of a design matrix, computed
//!   in blocks of [`ROW_BLOCK`] rows and added in block order.
//! - [`cholesky`] holds the lower Cholesky factor with a caller-chosen pivot
//!   rule.
//! - [`design`] builds a design from columns, finds linearly independent
//!   columns and selects rows and columns.
//! - [`small`] holds the triangular solves, inverse, log determinant and
//!   quadratic form of a small positive definite matrix on a row-major slice,
//!   in a fixed order of operations.
//! - [`solve`](mod@solve) holds faer's LU, Cholesky and column-pivoted QR
//!   solves, each one faer call on one thread, for code that applies its own
//!   rule to the solution.
//! - [`basis`] holds the polynomial, B-spline, restricted cubic spline and
//!   natural cubic spline bases, Harrell's knot rules and design matrices.
//! - [`eigen`] holds a Jacobi eigendecomposition of a small symmetric matrix and
//!   the generalised inverse and its quadratic form.
//!
//! # Which solve to use
//!
//! | the system | use | why |
//! |---|---|---|
//! | symmetric positive definite, one or a few right-hand sides | [`solve`](crate::solve())`(a, b, &[Attempt::plain(Method::Cholesky)])` | the one route: refuses a matrix that is not positive definite with [`Error::Singular`] |
//! | the same, falling back when it is not positive definite | [`solve`](crate::solve())`(a, b, &Attempt::SYMMETRIC)` | Cholesky, then Bunch-Kaufman, then LU |
//! | the same, near singular and to be ridged | an [`Attempt::ridged`] sequence | records the ridge used in the [`Solution`] |
//! | its inverse | [`inverse`] with the same attempts | `solve` against the identity |
//! | many right-hand sides that arrive one by one | [`Factor::new`], then [`Factor::solve`] | factorises once |
//! | many systems of one size in a hot loop | [`CholeskySolver::solve_into`] | the arithmetic of the Cholesky route without allocating |
//! | a factor that must exist, with pivots regularised | [`cholesky_factor`] with a [`Regularisation`] | faer's dynamic regularisation |
//! | a small matrix whose bits must not depend on faer's kernels | [`cholesky::cholesky_lower`] and [`small`] | fixed loops on row-major slices |
//! | a square system that needs faer's LU or Cholesky itself | [`solve::lu_partial`], [`solve::llt`] | the faer call, on one thread |
//! | least squares | [`solve::least_squares`] | column-pivoted QR |
//!
//! # Threads
//!
//! Every factorisation runs on one thread, so no result depends on the
//! number of threads of the calling program. faer's routines without a
//! parallelism argument run inside a rayon pool of one thread that the crate
//! builds on first use; a binding sees it as an extra thread named
//! `epistat-linalg`. The products over the rows of a design run in parallel in
//! fixed blocks, added in block order.
//!
//! # Errors
//!
//! Every fallible function returns [`Error`], whose `Input` variant carries
//! epistat-core's `InputError`.
#![warn(missing_docs)]

pub mod basis;
mod blocks;
pub mod cholesky;
pub mod design;
pub mod eigen;
mod error;
mod factor;
pub mod small;
pub mod solve;

pub use blocks::{
    BlockSum, ROW_BLOCK, block_scaled_gram, block_xtv, congruence, gram, matvec_into,
    row_scaled_gram, weighted_gram, weighted_xtz, xtv,
};
pub use error::{Error, weight_error};
pub use factor::{
    Attempt, CholeskySolver, Factor, Method, Regularisation, Ridge, Solution, cholesky_factor,
    inverse, solve,
};
pub use faer::linalg::matmul::matmul;
pub use faer::{Accum, Col, Mat, MatMut, MatRef, Par, Side, get_global_parallelism, mat};

/// The error of the products over a design under its name in 0.3.
#[deprecated(since = "0.4.0", note = "use `epistat_linalg::Error`")]
pub type DesignError = Error;

/// The error of the solves under its name in 0.3.
#[deprecated(since = "0.4.0", note = "use `epistat_linalg::Error`")]
pub type SolveError = Error;

/// Whether every entry of `matrix` is finite.
#[must_use]
pub fn is_finite(matrix: faer::MatRef<'_, f64>) -> bool {
    (0..matrix.ncols()).all(|j| (0..matrix.nrows()).all(|i| matrix[(i, j)].is_finite()))
}
