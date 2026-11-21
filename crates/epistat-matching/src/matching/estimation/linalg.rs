//! Dense inversion, with a generalised inverse when the matrix is singular.

use super::EstimationError;
use epistat_linalg::eigen::symmetric_eigen_jacobi;
use epistat_linalg::{Attempt, Mat, MatRef, Method, inverse};

/// A pivot at most this times the largest absolute entry makes a factorisation count as failed.
const MIN_PIVOT: f64 = 1e-14;

/// Inverts a row-major symmetric square `matrix`. Returns the inverse and its
/// rank.
///
/// The ordinary inverse is used when the factorisation has no pivot at most
/// `MIN_PIVOT` times the largest absolute entry. Otherwise the matrix is
/// singular or nearly so and the generalised inverse is used, which drops the
/// eigenvalues at most `relative_tolerance` times the largest. MatchIt's
/// Mahalanobis distance does the same for a singular covariance. No ridge is
/// added.
pub(super) fn invert_or_generalise(
    matrix: &[f64],
    relative_tolerance: f64,
) -> Result<(Vec<f64>, usize), EstimationError> {
    let dimension = to_dimension(matrix.len()).ok_or(EstimationError::InvalidCovarianceMatrix)?;
    if dimension == 0 || matrix.iter().any(|value| !value.is_finite()) {
        return Err(EstimationError::InvalidCovarianceMatrix);
    }
    let largest = matrix.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
    let a = MatRef::from_row_major_slice(matrix, dimension, dimension);
    let plain = [Attempt::plain(Method::PartialPivLu).with_min_pivot(MIN_PIVOT * largest)];
    match inverse(a, &plain) {
        Ok(solution) => Ok((row_major(&solution.x, dimension), dimension)),
        Err(_) => pseudo_inverse(matrix, relative_tolerance),
    }
}

/// The generalised inverse of a symmetric non-negative row-major `matrix`: the
/// eigenvalues at most `relative_tolerance` times the largest are dropped and
/// the rest inverted. Returns the inverse and the number of eigenvalues kept.
/// The matrix must have a positive largest eigenvalue.
pub(super) fn pseudo_inverse(
    matrix: &[f64],
    relative_tolerance: f64,
) -> Result<(Vec<f64>, usize), EstimationError> {
    let dimension = to_dimension(matrix.len()).ok_or(EstimationError::InvalidCovarianceMatrix)?;
    if dimension == 0 || matrix.iter().any(|value| !value.is_finite()) {
        return Err(EstimationError::InvalidCovarianceMatrix);
    }
    let (values, vectors) = symmetric_eigen_jacobi(matrix, dimension);
    let largest = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if largest <= 0.0 {
        return Err(EstimationError::InvalidCovarianceMatrix);
    }
    let threshold = relative_tolerance * largest;
    let mut inverse = vec![0.0_f64; dimension * dimension];
    let mut kept = 0usize;
    for (index, value) in values.iter().enumerate() {
        if *value <= threshold {
            continue;
        }
        kept += 1;
        for row in 0..dimension {
            for col in 0..dimension {
                inverse[row * dimension + col] +=
                    vectors[row * dimension + index] * vectors[col * dimension + index] / value;
            }
        }
    }
    Ok((inverse, kept))
}

fn row_major(matrix: &Mat<f64>, dimension: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(dimension * dimension);
    for row in 0..dimension {
        for col in 0..dimension {
            out.push(matrix[(row, col)]);
        }
    }
    out
}

fn to_dimension(length: usize) -> Option<usize> {
    // The side of the square this flattened matrix is, or None if it is not
    // square.
    let root = length.isqrt();
    (root * root == length).then_some(root)
}
