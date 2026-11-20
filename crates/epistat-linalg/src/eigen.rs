//! The eigendecomposition of a small symmetric matrix and the quadratic form
//! of its generalised inverse.
//!
//! The decomposition is by cyclic Jacobi rotations \[1\], on a row-major slice,
//! in a fixed order of operations with no fused multiply-add, so the result
//! does not depend on the number of threads or the platform's kernels. Each
//! sweep rotates every pair `(p, q)` with `p < q` and a non-zero off-diagonal
//! entry `a_pq` by the angle that zeroes it: with `theta = (a_qq - a_pp) /
//! (2 a_pq)`, `t = sign(theta) / (|theta| + sqrt(theta^2 + 1))` (1 for
//! `theta = 0`), `c = 1 / sqrt(t^2 + 1)` and `s = t c`. The rotation is
//! applied to the columns and then the rows of the matrix and to the columns
//! of the accumulated eigenvectors. Sweeps stop when the squared off-diagonal
//! mass is at most `1e-4 eps^2` of the total, or after 64 sweeps. It is meant
//! for matrices of order a few tens, where its cost `O(k^3)` per sweep is
//! small and its accuracy, high relative accuracy in the small eigenvalues of
//! a positive semi-definite matrix, is good.
//!
//! # References
//!
//! 1. Golub GH, Van Loan CF. Matrix Computations, 4th edition. Johns Hopkins
//!    University Press 2013, section 8.5.

/// Eigenvalues and eigenvectors of the symmetric matrix `a` of order `k`,
/// given as `k * k` values in row-major order. The eigenvalues are in the
/// order of the diagonal after the rotations, not sorted. Eigenvector `i` is
/// column `i` of the returned row-major matrix, that is `vectors[g * k + i]`
/// for its entry `g`.
///
/// The matrix is read in full and its symmetry is not checked.
///
/// # Panics
///
/// When `a` does not have `k * k` entries.
#[must_use]
pub fn symmetric_eigen_jacobi(a: &[f64], k: usize) -> (Vec<f64>, Vec<f64>) {
    assert_eq!(
        a.len(),
        k * k,
        "a matrix of order {k} has {} entries",
        k * k
    );
    let mut a = a.to_vec();
    let mut u = vec![0.0; k * k];
    for i in 0..k {
        u[i * k + i] = 1.0;
    }
    for _ in 0..64 {
        let mut off = 0.0;
        let mut all = 0.0;
        for i in 0..k {
            for j in 0..k {
                let x = a[i * k + j] * a[i * k + j];
                all += x;
                if i != j {
                    off += x;
                }
            }
        }
        if off <= all * f64::EPSILON * f64::EPSILON * 1e-4 {
            break;
        }
        for p in 0..k {
            for q in p + 1..k {
                let apq = a[p * k + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * k + q] - a[p * k + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + theta.hypot(1.0));
                let t = if theta == 0.0 { 1.0 } else { t };
                let c = 1.0 / t.hypot(1.0);
                let s = t * c;
                for r in 0..k {
                    let (x, y) = (a[r * k + p], a[r * k + q]);
                    a[r * k + p] = c * x - s * y;
                    a[r * k + q] = s * x + c * y;
                }
                for r in 0..k {
                    let (x, y) = (a[p * k + r], a[q * k + r]);
                    a[p * k + r] = c * x - s * y;
                    a[q * k + r] = s * x + c * y;
                }
                for r in 0..k {
                    let (x, y) = (u[r * k + p], u[r * k + q]);
                    u[r * k + p] = c * x - s * y;
                    u[r * k + q] = s * x + c * y;
                }
            }
        }
    }
    ((0..k).map(|i| a[i * k + i]).collect(), u)
}

/// `b' M^- b` for a symmetric non-negative `M` of order `k = b.len()` given
/// row-major, with `M^-` the generalised inverse that drops every eigenvalue
/// not above `tolerance(largest)`, and the number of eigenvalues kept.
///
/// `tolerance` takes the largest eigenvalue and returns the threshold, so a
/// caller states its own rank rule. `None` when no eigenvalue is positive, so
/// that the form is of a zero matrix and has no rank.
///
/// # Panics
///
/// When `m` does not have `b.len()` squared entries.
#[must_use]
pub fn pinv_quadratic_form(
    m: &[f64],
    b: &[f64],
    tolerance: impl Fn(f64) -> f64,
) -> Option<(f64, usize)> {
    let k = b.len();
    let (values, vectors) = symmetric_eigen_jacobi(m, k);
    let largest = values.iter().copied().fold(0.0_f64, f64::max);
    if largest <= 0.0 {
        return None;
    }
    let threshold = tolerance(largest);
    let mut rank = 0;
    let mut statistic = 0.0;
    for (i, &lambda) in values.iter().enumerate() {
        if lambda > threshold {
            rank += 1;
            let mut projection = 0.0;
            for g in 0..k {
                projection += vectors[g * k + i] * b[g];
            }
            statistic += projection * projection / lambda;
        }
    }
    Some((statistic, rank))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `[[2, 1], [1, 2]]` has eigenvalues 1 and 3 with eigenvectors
    /// `(1, -1)` and `(1, 1)` over `sqrt 2`. One rotation by 45 degrees
    /// diagonalises it, so the error is a few eps.
    #[test]
    fn a_two_by_two_matrix_has_its_closed_form_eigenpairs() {
        let (values, vectors) = symmetric_eigen_jacobi(&[2.0, 1.0, 1.0, 2.0], 2);
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        assert!((sorted[0] - 1.0).abs() < 8.0 * f64::EPSILON);
        assert!((sorted[1] - 3.0).abs() < 8.0 * f64::EPSILON);
        for i in 0..2 {
            let (x, y) = (vectors[i], vectors[2 + i]);
            assert!((x * x + y * y - 1.0).abs() < 8.0 * f64::EPSILON);
            let (ax, ay) = (2.0 * x + y, x + 2.0 * y);
            assert!((ax - values[i] * x).abs() < 16.0 * f64::EPSILON);
            assert!((ay - values[i] * y).abs() < 16.0 * f64::EPSILON);
        }
    }

    /// For a matrix of rank one, `[[1, 1], [1, 1]]`, `b' M^- b` with `b` in
    /// its range is `(b_1 + b_2)^2 / 4`: here `b = (1, 1)` gives 1, and one
    /// eigenvalue is kept.
    #[test]
    fn a_singular_matrix_drops_the_null_direction() {
        let got = pinv_quadratic_form(&[1.0, 1.0, 1.0, 1.0], &[1.0, 1.0], |top| 1e-12 * top);
        let (statistic, rank) = got.unwrap();
        assert_eq!(rank, 1);
        assert!((statistic - 1.0).abs() < 16.0 * f64::EPSILON);
    }

    #[test]
    fn a_zero_matrix_has_no_form() {
        assert_eq!(pinv_quadratic_form(&[0.0; 4], &[1.0, 2.0], |top| top), None);
    }

    /// The reconstruction `V diag(values) V'` equals the input to a few eps
    /// times the norm, for a matrix of order 5.
    #[test]
    fn the_eigenpairs_reconstruct_the_matrix() {
        let k = 5;
        let a: Vec<f64> = (0..k * k)
            .map(|idx| {
                let (i, j) = (idx / k, idx % k);
                1.0 / (1.0 + (i + j) as f64) + if i == j { 1.0 } else { 0.0 }
            })
            .collect();
        let (values, v) = symmetric_eigen_jacobi(&a, k);
        for i in 0..k {
            for j in 0..k {
                let mut total = 0.0;
                for g in 0..k {
                    total += v[i * k + g] * values[g] * v[j * k + g];
                }
                assert!(
                    (total - a[i * k + j]).abs() < 64.0 * f64::EPSILON,
                    "{i} {j}"
                );
            }
        }
    }
}
