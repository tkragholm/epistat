//! Small linear-algebra helpers shared by estimator implementations.

use epistat_linalg::{Attempt, CholeskySolver, Mat, MatRef, Method, solve};

/// Solve `(A + lambda I) X = B` for `X` by LU with partial pivoting, the
/// normal equations of the outcome regressions with a ridge `lambda` on every
/// diagonal entry, the intercept's included.
///
/// The ridge is absolute, not scaled to the diagonal of `A`, and no other
/// factorisation is tried: a singular or nearly singular `A` plus a small
/// ridge gives a finite solution that can be far from a least-squares one,
/// and nothing in the result says so. R's `lm`, which the outcome
/// regressions of `DRDID` use, has no ridge and reports aliased columns as
/// `NA`; the parity tests set `lambda` to 0.
///
/// # Errors
/// Returns an error if `lambda` is not finite and non-negative, `A` is not
/// square or `B` has another number of rows, or the solution contains a
/// non-finite value.
pub fn ridge_solve_lu(
    mut a: Mat<f64>,
    b: &Mat<f64>,
    lambda: f64,
) -> Result<Mat<f64>, &'static str> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err("ridge must be finite and non-negative");
    }
    if a.nrows() != a.ncols() || b.nrows() != a.nrows() {
        return Err(SHAPE);
    }
    for i in 0..a.ncols() {
        *a.get_mut(i, i) += lambda;
    }
    solve(
        a.as_ref(),
        b.as_ref(),
        &[Attempt::plain(Method::PartialPivLu)],
    )
    .map(|solution| solution.x)
    .map_err(|_| NON_FINITE)
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
const SHAPE: &str = "coefficient matrix shape does not match rhs length";

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
                epistat_linalg::Error::Input(_) => {
                    "coefficient matrix shape does not match rhs length"
                }
                _ => NON_FINITE,
            })
    }
}

const fn validate_square_system_shape(a: &[f64], n: usize) -> Result<(), &'static str> {
    if a.len() != n * n {
        return Err(SHAPE);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_linalg::Solve;

    /// `ridge_solve_lu` as it called faer directly, kept to show that the
    /// route through `epistat_linalg::solve` gives the same bits.
    fn ridge_solve_lu_direct(mut a: Mat<f64>, b: &Mat<f64>, lambda: f64) -> Mat<f64> {
        for i in 0..a.ncols() {
            *a.get_mut(i, i) += lambda;
        }
        a.partial_piv_lu().solve(b)
    }

    #[test]
    fn ridge_solve_lu_gives_the_bits_of_the_direct_faer_call() {
        let mut stream = epistat_rng::Stream::seeded(41);
        for (dim, rhs) in [(1, 1), (3, 1), (7, 2), (22, 1), (40, 3)] {
            let x = Mat::from_fn(4 * dim, dim, |_, _| stream.uniform() - 0.5);
            let a = x.transpose() * &x;
            let b = Mat::from_fn(dim, rhs, |_, _| stream.uniform());
            for lambda in [0.0, 1e-8, 0.5] {
                let ours = ridge_solve_lu(a.clone(), &b, lambda).expect("finite");
                let direct = ridge_solve_lu_direct(a.clone(), &b, lambda);
                for r in 0..dim {
                    for c in 0..rhs {
                        assert_eq!(ours[(r, c)].to_bits(), direct[(r, c)].to_bits());
                    }
                }
            }
        }
    }

    #[test]
    fn ridge_solve_lu_refuses_an_invalid_ridge_or_shape() {
        let a = Mat::<f64>::identity(2, 2);
        let b = Mat::from_fn(2, 1, |_, _| 1.0);
        assert!(ridge_solve_lu(a.clone(), &b, -1.0).is_err());
        assert!(ridge_solve_lu(a.clone(), &b, f64::NAN).is_err());
        assert!(ridge_solve_lu(a.clone(), &b, f64::INFINITY).is_err());
        assert!(ridge_solve_lu(a, &Mat::from_fn(3, 1, |_, _| 1.0), 0.0).is_err());
        assert!(ridge_solve_lu(Mat::from_fn(2, 3, |_, _| 1.0), &b, 0.0).is_err());
    }

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
