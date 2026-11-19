//! Variances of an estimator from its influence function.
//!
//! An estimator `theta_hat` is asymptotically linear when
//! `theta_hat - theta = sum(psi_i) / n` plus a negligible remainder, where
//! `psi_i` is the influence function of observation `i` \[1\]. Its variance is
//! then estimated from the sample values of `psi_i`. The functions take those
//! values as given.
//!
//! # Method
//!
//! [`standard_error`] centres the values and treats the observations as
//! independent:
//!
//! ```text
//! se = sqrt( mean((psi - mean(psi))^2) ) / sqrt(n)
//! ```
//!
//! [`clustered_variance`] sums the values within each of `G` clusters and
//! does not centre them. It follows the cluster-robust variance of \[2\] and
//! \[3\] with the finite-cluster factor `G / (G - 1)`, as R's `did` package
//! computes it:
//!
//! ```text
//! s_g = sum_{i in g} psi_i
//! V   = G / (G - 1) * sum_g s_g^2 / n^2
//! ```
//!
//! With fewer than two clusters the factor is undefined, and the variance is
//! `sum(psi_i^2) / n^2`, the value for singleton clusters without the factor.
//! [`clustered_standard_error`] is its square root.
//!
//! [`clustered_covariance`] takes `K` influence functions and returns the
//! `K x K` matrix with entry `(a, b)`
//!
//! ```text
//! G / (G - 1) * sum_g s_ga s_gb / n^2
//! ```
//!
//! with the same fallback below two clusters.
//!
//! # Properties
//!
//! - [`standard_error`] subtracts the mean, the clustered forms do not. With
//!   singleton clusters and values of mean zero, as an influence function
//!   has at the estimate, the clustered variance is the square of
//!   [`standard_error`] times `G / (G - 1)`.
//! - The matrix is symmetric, since the entry for `(b, a)` is a copy of the
//!   entry for `(a, b)`.
//!
//! # Determinism
//!
//! Cluster sums are formed in observation order and added in cluster order.
//! The covariance computes the variance of column `a` with the same sums and
//! the same products in the same order, so its diagonal entry equals
//! [`clustered_variance`] of that column to the bit.
//!
//! # Edge cases
//!
//! - No observations, or no columns, is [`InferenceError::Empty`].
//! - A `clusters` argument that covers another number of observations, or a
//!   column of another length, is [`InferenceError::Length`].
//! - A non-finite variance, or covariance entry, is
//!   [`InferenceError::NotFinite`]. [`standard_error`] does not check, and
//!   returns NaN or infinity for non-finite input.
//! - One observation gives a standard error of 0.
//!
//! # References
//!
//! 1. Hampel FR. The influence curve and its role in robust estimation. J Am
//!    Stat Assoc. 1974;69(346):383-93.
//! 2. Liang KY, Zeger SL. Longitudinal data analysis using generalized linear
//!    models. Biometrika. 1986;73(1):13-22.
//! 3. Cameron AC, Gelbach JB, Miller DL. Bootstrap-based improvements for
//!    inference with clustered errors. Rev Econ Stat. 2008;90(3):414-27.

use crate::{Clusters, InferenceError, check_length};

/// `sqrt(mean((psi - mean(psi))^2)) / sqrt(n)`.
///
/// # Errors
///
/// [`InferenceError::Empty`] for no values.
pub fn standard_error(psi: &[f64]) -> Result<f64, InferenceError> {
    if psi.is_empty() {
        return Err(InferenceError::Empty);
    }
    let n = psi.len() as f64;
    let mean = psi.iter().sum::<f64>() / n;
    let mean_square = psi.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    Ok(mean_square.sqrt() / n.sqrt())
}

/// The cluster-robust variance `G / (G - 1) * sum_g (sum_{i in g} psi_i)^2 / n^2`,
/// as R's `did` computes it. With fewer than two clusters it is
/// `sum(psi^2) / n^2`.
///
/// Cluster sums are formed in observation order and added in cluster order.
///
/// # Errors
///
/// [`InferenceError::Empty`] for no values, [`InferenceError::Length`] when
/// `clusters` covers another number of observations, and
/// [`InferenceError::NotFinite`].
pub fn clustered_variance(psi: &[f64], clusters: &Clusters) -> Result<f64, InferenceError> {
    if psi.is_empty() {
        return Err(InferenceError::Empty);
    }
    check_length("clusters", psi.len(), clusters.len())?;
    let n = psi.len() as f64;
    let variance = if clusters.count() >= 2 {
        let g = clusters.count() as f64;
        let squares: f64 = clusters
            .sums(psi.iter().copied())
            .iter()
            .map(|s| s * s)
            .sum();
        g / (g - 1.0) * squares / (n * n)
    } else {
        psi.iter().map(|v| v * v).sum::<f64>() / (n * n)
    };
    if variance.is_finite() {
        Ok(variance.max(0.0))
    } else {
        Err(InferenceError::NotFinite)
    }
}

/// The square root of [`clustered_variance`].
///
/// # Errors
///
/// As [`clustered_variance`].
pub fn clustered_standard_error(psi: &[f64], clusters: &Clusters) -> Result<f64, InferenceError> {
    clustered_variance(psi, clusters).map(f64::sqrt)
}

/// The covariance matrix of several estimators from their influence
/// functions, `columns[k][i]` for estimator `k` and observation `i`, clustered
/// as in [`clustered_variance`]. Returned row-major, `k x k`.
///
/// # Errors
///
/// As [`clustered_variance`], and [`InferenceError::Length`] when the columns
/// differ in length.
pub fn clustered_covariance(
    columns: &[&[f64]],
    clusters: &Clusters,
) -> Result<Vec<f64>, InferenceError> {
    let n = clusters.len();
    if n == 0 || columns.is_empty() {
        return Err(InferenceError::Empty);
    }
    for column in columns {
        check_length("influence column", n, column.len())?;
    }
    let k = columns.len();
    let (nf, g) = (n as f64, clusters.count() as f64);
    let sums: Vec<Vec<f64>> = if clusters.count() >= 2 {
        columns
            .iter()
            .map(|c| clusters.sums(c.iter().copied()))
            .collect()
    } else {
        columns.iter().map(|c| c.to_vec()).collect()
    };
    let scale = if clusters.count() >= 2 {
        g / (g - 1.0)
    } else {
        1.0
    };
    let mut covariance = vec![0.0; k * k];
    for a in 0..k {
        for b in a..k {
            let cross: f64 = sums[a].iter().zip(&sums[b]).map(|(x, y)| x * y).sum();
            let value = scale * cross / (nf * nf);
            if !value.is_finite() {
                return Err(InferenceError::NotFinite);
            }
            covariance[a * k + b] = value;
            covariance[b * k + a] = value;
        }
    }
    Ok(covariance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_clusters_give_the_uncentred_variance_with_a_correction() {
        let psi = [1.0, -2.0, 0.5, 0.5];
        let variance = clustered_variance(&psi, &Clusters::singletons(4)).unwrap();
        assert!((variance - 4.0 / 3.0 * 5.5 / 16.0).abs() < 1e-15);
    }

    #[test]
    fn one_cluster_falls_back_to_the_sum_of_squares() {
        let psi = [1.0, -1.0];
        let one = Clusters::from_labels(&[7, 7]);
        assert!((clustered_variance(&psi, &one).unwrap() - 0.5).abs() < 1e-15);
    }

    #[test]
    fn the_covariance_diagonal_is_the_variance() {
        let a = [1.0, -2.0, 0.5, 0.5, 3.0];
        let b = [0.0, 1.0, -1.0, 2.0, 0.5];
        let clusters = Clusters::from_labels(&[1, 1, 2, 3, 3]);
        let covariance = clustered_covariance(&[&a, &b], &clusters).unwrap();
        assert_eq!(
            covariance[0].to_bits(),
            clustered_variance(&a, &clusters).unwrap().to_bits()
        );
        assert_eq!(covariance[1].to_bits(), covariance[2].to_bits());
    }

    #[test]
    fn errors() {
        assert_eq!(standard_error(&[]), Err(InferenceError::Empty));
        assert!(matches!(
            clustered_variance(&[1.0], &Clusters::singletons(2)),
            Err(InferenceError::Length { .. })
        ));
    }
}
