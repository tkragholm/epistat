//! Sandwich covariances for M-estimators.
//!
//! An M-estimator solves `sum_i psi(z_i; theta) = 0`. Under regularity
//! conditions its covariance is `A^-1 M A^-T`, the form of Huber \[1\] and
//! White \[2\], where `A` is the derivative of the summed score (the
//! information) and `M` is the variance of the summed score (the meat).
//! The functions here build `M`, form the sandwich and read standard errors
//! from it. The caller supplies `A`, the rows `x_i` and the score residuals.
//!
//! # Method
//!
//! For a score `r_i x_i`, where `r_i` is the residual of row `i` and `x_i` its
//! row of the design matrix, [`meat`] without clusters is the
//! heteroskedasticity-robust form \[2\]:
//!
//! ```text
//! M = sum_i r_i^2 x_i x_i'
//! ```
//!
//! With clusters it sums the scores within each cluster `g` first, as in \[3\]
//! and \[4\]:
//!
//! ```text
//! s_g = sum_{i in g} r_i x_i
//! M   = sum_g s_g s_g'
//! ```
//!
//! No small-sample factor is applied. [`sandwich`] returns
//!
//! ```text
//! V = A^-1 M A^-T
//! ```
//!
//! by two solves. It factorises `A` and solves `A L = M`, then factorises
//! `A'` and solves `A' R = L'`, so that `R'` is `L A^-T`. `A` is never
//! inverted explicitly. The factorisation is tried with the given `attempts`,
//! in order. The two solves round differently, so `R'` is symmetric only up to
//! rounding, and the function returns the midpoint of `R'` and `R`. That matrix
//! is exactly symmetric, its diagonal equals that of `R'` to the bit, and its
//! off-diagonal entries differ from `R'` in the last bits.
//!
//! [`standard_errors`] is `sqrt(max(V_ii, 0))`. A negative diagonal entry,
//! which rounding can produce when `V` is near singular, is read as zero.
//!
//! # Determinism
//!
//! Without clusters the meat is the Gram matrix of the scaled rows from
//! `epistat-linalg`. With clusters the score sums are formed in row order
//! into one row per cluster, in order of first appearance, and the meat is the
//! Gram matrix of those rows.
//!
//! # Edge cases
//!
//! - A `clusters` argument that covers another number of rows is
//!   [`InferenceError::Length`]. Without clusters the length of the
//!   residuals is not checked, since they are a function of the row.
//! - A singular or non-finite solve is [`InferenceError::Solve`].
//!
//! # References
//!
//! 1. Huber PJ. The behavior of maximum likelihood estimates under
//!    nonstandard conditions. In: Proceedings of the Fifth Berkeley Symposium
//!    on Mathematical Statistics and Probability, vol 1. Berkeley: University
//!    of California Press; 1967. p. 221-33.
//! 2. White H. A heteroskedasticity-consistent covariance matrix estimator and
//!    a direct test for heteroskedasticity. Econometrica. 1980;48(4):817-38.
//! 3. Liang KY, Zeger SL. Longitudinal data analysis using generalized linear
//!    models. Biometrika. 1986;73(1):13-22.
//! 4. Cameron AC, Gelbach JB, Miller DL. Bootstrap-based improvements for
//!    inference with clustered errors. Rev Econ Stat. 2008;90(3):414-27.

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

/// `A^-1 M A^-T`, by two solves through `attempts`, averaged with its
/// transpose so that the result is exactly symmetric.
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
    let n = transposed.nrows();
    Ok(Mat::from_fn(n, n, |i, j| {
        transposed[(j, i)].midpoint(transposed[(i, j)])
    }))
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

    #[test]
    fn the_sandwich_is_exactly_symmetric() {
        let information = mat![[2.0, 0.3, 0.1], [0.7, 1.5, 0.2], [0.1, 0.9, 3.0]];
        let x = Mat::from_fn(7, 3, |i, j| ((i * 3 + j) % 5) as f64 - 1.7);
        let m = meat(x.as_ref(), |i| 0.3 * i as f64 - 1.0, None).unwrap();
        let v = sandwich(information.as_ref(), m.as_ref(), &Attempt::SYMMETRIC).unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert_eq!(v[(i, j)].to_bits(), v[(j, i)].to_bits());
            }
        }
    }
}
