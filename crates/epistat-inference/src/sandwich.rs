//! Sandwich covariances `A^-1 M A^-T` for M-estimators.

use epistat_linalg::{Attempt, Factor, row_scaled_gram};
use faer::{Mat, MatRef};

use crate::{Clusters, InferenceError, check_length};

/// The meat `M` from per-row score residuals `r_i = residual(i)`.
///
/// Without clusters it is `sum_i r_i^2 x_i x_i'`. With clusters it is
/// `sum_g s_g s_g'` with `s_g = sum_{i in g} r_i x_i`, the score sums formed in
/// row order.
///
/// # Errors
///
/// [`InferenceError::Length`] when `clusters` covers another number of rows.
pub fn meat(
    x: MatRef<'_, f64>,
    residual: impl Fn(usize) -> f64 + Sync,
    clusters: Option<&Clusters>,
) -> Result<Mat<f64>, InferenceError> {
    let Some(clusters) = clusters else {
        return Ok(row_scaled_gram(x, residual));
    };
    check_length("clusters", x.nrows(), clusters.len())?;
    let p = x.ncols();
    let mut sums = Mat::<f64>::zeros(clusters.count(), p);
    for (row, &cluster) in clusters.of().iter().enumerate() {
        let r = residual(row);
        for j in 0..p {
            sums[(cluster, j)] += r * x[(row, j)];
        }
    }
    Ok(row_scaled_gram(sums.as_ref(), |_| 1.0))
}

/// `A^-1 M A^-T`, by two solves with `A` factorised once through `attempts`.
///
/// # Errors
///
/// [`InferenceError::Solve`] when `A` cannot be factorised or a solve is not
/// finite.
pub fn sandwich(
    information: MatRef<'_, f64>,
    meat: MatRef<'_, f64>,
    attempts: &[Attempt],
) -> Result<Mat<f64>, InferenceError> {
    let left = Factor::new(information, attempts)?.solve(meat)?;
    let transposed = Factor::new(information.transpose(), attempts)?.solve(left.transpose())?;
    Ok(transposed.transpose().to_owned())
}

/// The square roots of the diagonal, with negative variances read as zero.
#[must_use]
pub fn standard_errors(covariance: MatRef<'_, f64>) -> Vec<f64> {
    (0..covariance.nrows())
        .map(|i| covariance[(i, i)].max(0.0).sqrt())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::mat;

    #[test]
    fn the_least_squares_sandwich_is_the_white_covariance() {
        // Regress y on an intercept; the sandwich variance of the mean is
        // sum(r^2) / n^2.
        let x = Mat::from_fn(4, 1, |_, _| 1.0);
        let residuals = [1.0, -1.0, 2.0, -2.0];
        let information = mat![[4.0]];
        let m = meat(x.as_ref(), |i| residuals[i], None).unwrap();
        let covariance = sandwich(information.as_ref(), m.as_ref(), &Attempt::SYMMETRIC).unwrap();
        assert!((covariance[(0, 0)] - 10.0 / 16.0).abs() < 1e-15);
        assert_eq!(
            standard_errors(covariance.as_ref()),
            [covariance[(0, 0)].sqrt()]
        );
    }

    #[test]
    fn clustered_meat_sums_scores_within_clusters() {
        let x = Mat::from_fn(4, 1, |_, _| 1.0);
        let residuals = [1.0, -1.0, 2.0, 1.0];
        let clusters = Clusters::from_labels(&["a", "a", "b", "b"]);
        let m = meat(x.as_ref(), |i| residuals[i], Some(&clusters)).unwrap();
        assert!((m[(0, 0)] - 9.0).abs() < 1e-15);
    }
}
