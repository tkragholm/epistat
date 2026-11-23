//! Small linear-algebra helpers shared by estimator implementations.

use epistat_linalg::{Attempt, CholeskySolver, Mat, MatRef, Method, Solve, solve};

/// Solve `(A + λI) X = B` for `X`, where `A` is expected to be square.
///
/// This helper adds a ridge term `lambda` to the diagonal in-place and solves
/// with LU decomposition.
///
/// # Notes
/// - `lambda` is applied uniformly to every diagonal entry.
///
/// # Errors
/// Returns an error if the solution contains a non-finite value.
pub fn ridge_solve_lu(
    mut a: Mat<f64>,
    b: &Mat<f64>,
    lambda: f64,
) -> Result<Mat<f64>, &'static str> {
    for i in 0..a.ncols() {
        *a.get_mut(i, i) += lambda;
    }
    let solution = a.partial_piv_lu().solve(b);
    if (0..solution.nrows()).all(|r| (0..solution.ncols()).all(|c| solution.get(r, c).is_finite()))
    {
        Ok(solution)
    } else {
        Err(NON_FINITE)
    }
}

/// Solve a dense linear system `A x = b` using Cholesky when possible.
///
/// Fallback chain, each attempt running on one thread:
/// 1. Cholesky (`LLT`) on SPD matrices.
/// 2. Bunch-Kaufman (`LBLT`) for symmetric-indefinite systems.
/// 3. Partial-pivot LU if previous attempts are non-finite.
/// 4. Full-pivot LU when partial-pivot output is non-finite.
///
/// # Errors
/// Returns an error if:
/// - `a.len() != b.len() * b.len()` (shape mismatch), or
/// - all solve attempts produce non-finite values.
pub fn solve_spd_system(a: &[f64], b: &[f64]) -> Result<Vec<f64>, &'static str> {
    solve_flat(a, b, &Attempt::SYMMETRIC)
}

/// Solve a dense linear system `A X = B` where `A` is square and `B` may have
/// multiple right-hand sides.
///
/// The same fallback chain as [`solve_spd_system`] is applied.
///
/// # Errors
/// Returns an error if:
/// - `a` does not represent a square matrix matching `rhs.nrows()`, or
/// - all solve attempts produce non-finite values.
pub fn solve_spd_system_multi_rhs(a: &[f64], rhs: &Mat<f64>) -> Result<Mat<f64>, &'static str> {
    let n = rhs.nrows();
    validate_square_system_shape(a, n)?;
    let system = MatRef::from_row_major_slice(a, n, n);
    solve(system, rhs.as_ref(), &Attempt::SYMMETRIC)
        .map(|solution| solution.x)
        .map_err(|_| NON_FINITE)
}

/// Solve a dense linear system `A x = b` with robust full-pivot LU.
///
/// This explicit path is intended for rank-deficient or near-singular systems.
///
/// # Errors
/// Returns an error if:
/// - `a.len() != b.len() * b.len()` (shape mismatch), or
/// - the resulting solution contains non-finite values.
pub fn solve_general_system_robust(a: &[f64], b: &[f64]) -> Result<Vec<f64>, &'static str> {
    solve_flat(a, b, &[Attempt::plain(Method::FullPivLu)])
}

/// Solve a dense linear system `A x = b` from flat row-major inputs.
///
/// # Arguments
/// - `a`: Flattened `n x n` coefficient matrix in row-major order.
/// - `b`: Right-hand side vector with length `n`.
///
/// # Errors
/// Returns an error if:
/// - `a.len() != b.len() * b.len()` (shape mismatch), or
/// - the resulting solution contains non-finite values.
pub fn solve_dense_system(a: &[f64], b: &[f64]) -> Result<Vec<f64>, &'static str> {
    solve_general_system_robust(a, b)
}

const NON_FINITE: &str = "dense solve produced a non-finite value";

fn solve_flat(a: &[f64], b: &[f64], attempts: &[Attempt]) -> Result<Vec<f64>, &'static str> {
    let n = b.len();
    validate_square_system_shape(a, n)?;
    let system = MatRef::from_row_major_slice(a, n, n);
    let rhs = MatRef::from_column_major_slice(b, n, 1);
    let solution = solve(system, rhs, attempts).map_err(|_| NON_FINITE)?;
    Ok((0..n).map(|row| solution.x[(row, 0)]).collect())
}

/// Repeated SPD solves of one size, reusing the factor and work memory.
pub struct SpdCholeskyScratch {
    solver: CholeskySolver,
}

impl SpdCholeskyScratch {
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            solver: CholeskySolver::new(dim),
        }
    }

    /// Solve `A x = b` by Cholesky, `A` row-major.
    ///
    /// # Errors
    /// Returns an error if dimensions mismatch, the factorisation fails, or the
    /// solution is non-finite.
    pub fn solve_single_rhs_into(
        &mut self,
        a: &[f64],
        b: &[f64],
        out: &mut [f64],
    ) -> Result<(), &'static str> {
        self.solver
            .solve_into(a, b, out)
            .map_err(|error| match error {
                epistat_linalg::SolveError::Shape { .. } => {
                    "coefficient matrix shape does not match rhs length"
                }
                epistat_linalg::SolveError::NotFinite => NON_FINITE,
            })
    }
}

const fn validate_square_system_shape(a: &[f64], n: usize) -> Result<(), &'static str> {
    if a.len() != n * n {
        return Err("coefficient matrix shape does not match rhs length");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solve_spd_system_handles_positive_definite_system() {
        let a = vec![4.0, 1.0, 1.0, 3.0];
        let b = vec![1.0, 2.0];
        let solution = solve_spd_system(&a, &b).expect("solve");
        assert!((solution[0] - 0.090_909_090_909_090_91).abs() < 1e-12);
        assert!((solution[1] - 0.636_363_636_363_636_4).abs() < 1e-12);
    }

    #[test]
    fn solve_spd_system_falls_back_to_lu_for_indefinite_matrix() {
        let a = vec![0.0, 1.0, 1.0, 0.0];
        let b = vec![1.0, 2.0];
        let solution = solve_spd_system(&a, &b).expect("solve");
        assert!((solution[0] - 2.0).abs() < 1e-12);
        assert!((solution[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn solve_spd_system_rejects_singular_matrix() {
        let a = vec![1.0, 2.0, 2.0, 4.0];
        let b = vec![1.0, 2.0];
        let err = solve_spd_system(&a, &b).expect_err("singular matrix must fail");
        assert!(err.contains("non-finite"));
    }

    #[test]
    fn solve_dense_system_preserves_robust_path_for_singular_case() {
        let singular = vec![1.0, 2.0, 2.0, 4.0];
        let b = vec![1.0, 2.0];
        let err = solve_dense_system(&singular, &b).expect_err("must fail");
        assert!(err.contains("non-finite"));
    }
}
