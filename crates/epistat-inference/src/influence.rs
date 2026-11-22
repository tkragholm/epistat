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
//! \[3\] with the finite-cluster factor `G / (G - 1)`:
//!
//! ```text
//! s_g = sum_{i in g} psi_i
//! V   = G / (G - 1) * sum_g s_g^2 / n^2
//! ```
//!
//! With fewer than two clusters the factor is undefined and so is the
//! variance, and the function returns `None`. The cross-products of the
//! unclustered values are not a stand-in, since they would report a variance
//! for a design that has none. [`clustered_standard_error`] is its square
//! root.
//!
//! [`clustered_covariance`] takes `K` influence functions and returns the
//! `K x K` matrix with entry `(a, b)`
//!
//! ```text
//! G / (G - 1) * sum_g s_ga s_gb / n^2
//! ```
//!
//! and is `None` below two clusters.
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
//! - Fewer than two clusters gives `None`, for the variance, the standard error
//!   and the covariance alike, after the checks above.
//! - One observation gives a [`standard_error`] of 0.
//!
//! # References
//!
//! 1. Hampel FR. The influence curve and its role in robust estimation. J Am
//!    Stat Assoc. 1974;69(346):383-93.
//! 2. Liang KY, Zeger SL. Longitudinal data analysis using generalized linear
//!    models. Biometrika. 1986;73(1):13-22.
//! 3. Cameron AC, Gelbach JB, Miller DL. Bootstrap-based improvements for
//!    inference with clustered errors. Rev Econ Stat. 2008;90(3):414-27.

use epistat_stats::lanes;

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
    let mean = lanes::sum(psi) / n;
    let mean_square = lanes::centred_squares(psi, mean) / n;
    Ok(mean_square.sqrt() / n.sqrt())
}

/// The cluster-robust variance `G / (G - 1) * sum_g (sum_{i in g} psi_i)^2 / n^2`.
/// It is `None` with fewer than two clusters.
///
/// Cluster sums are formed in observation order and added in cluster order.
///
/// # Errors
///
/// [`InferenceError::Empty`] for no values, [`InferenceError::Length`] when
/// `clusters` covers another number of observations, and
/// [`InferenceError::NotFinite`].
pub fn clustered_variance(psi: &[f64], clusters: &Clusters) -> Result<Option<f64>, InferenceError> {
    if psi.is_empty() {
        return Err(InferenceError::Empty);
    }
    check_length("clusters", psi.len(), clusters.len())?;
    if clusters.count() < 2 {
        return Ok(None);
    }
    let n = psi.len() as f64;
    let g = clusters.count() as f64;
    let squares = lanes::sum_squares(&clusters.sums(psi.iter().copied()));
    let variance = g / (g - 1.0) * squares / (n * n);
    if variance.is_finite() {
        Ok(Some(variance.max(0.0)))
    } else {
        Err(InferenceError::NotFinite)
    }
}

/// The square root of [`clustered_variance`], `None` where that is.
///
/// # Errors
///
/// As [`clustered_variance`].
pub fn clustered_standard_error(
    psi: &[f64],
    clusters: &Clusters,
) -> Result<Option<f64>, InferenceError> {
    clustered_variance(psi, clusters).map(|variance| variance.map(f64::sqrt))
}

/// The covariance matrix of several estimators from their influence
/// functions, `columns[k][i]` for estimator `k` and observation `i`, clustered
/// as in [`clustered_variance`]. Returned row-major, `k x k`, or `None` with
/// fewer than two clusters.
///
/// # Errors
///
/// As [`clustered_variance`], and [`InferenceError::Length`] when the columns
/// differ in length.
pub fn clustered_covariance(
    columns: &[&[f64]],
    clusters: &Clusters,
) -> Result<Option<Vec<f64>>, InferenceError> {
    let n = clusters.len();
    if n == 0 || columns.is_empty() {
        return Err(InferenceError::Empty);
    }
    for column in columns {
        check_length("influence column", n, column.len())?;
    }
    if clusters.count() < 2 {
        return Ok(None);
    }
    let k = columns.len();
    let (nf, g) = (n as f64, clusters.count() as f64);
    let sums: Vec<Vec<f64>> = columns
        .iter()
        .map(|c| clusters.sums(c.iter().copied()))
        .collect();
    let scale = g / (g - 1.0);
    let mut covariance = vec![0.0; k * k];
    for a in 0..k {
        for b in a..k {
            let cross = lanes::dot(&sums[a], &sums[b]);
            let value = scale * cross / (nf * nf);
            if !value.is_finite() {
                return Err(InferenceError::NotFinite);
            }
            covariance[a * k + b] = value;
            covariance[b * k + a] = value;
        }
    }
    Ok(Some(covariance))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_clusters_give_the_uncentred_variance_with_a_correction() {
        let psi = [1.0, -2.0, 0.5, 0.5];
        let variance = clustered_variance(&psi, &Clusters::singletons(4))
            .unwrap()
            .unwrap();
        assert!((variance - 4.0 / 3.0 * 5.5 / 16.0).abs() < 1e-15);
    }

    #[test]
    fn fewer_than_two_clusters_have_no_variance() {
        let psi = [1.0, -1.0];
        let one = Clusters::from_labels(&[7, 7]);
        assert_eq!(clustered_variance(&psi, &one), Ok(None));
        assert_eq!(clustered_standard_error(&psi, &one), Ok(None));
        assert_eq!(clustered_covariance(&[&psi, &psi], &one), Ok(None));
    }

    #[test]
    fn the_covariance_diagonal_is_the_variance() {
        let a = [1.0, -2.0, 0.5, 0.5, 3.0];
        let b = [0.0, 1.0, -1.0, 2.0, 0.5];
        let clusters = Clusters::from_labels(&[1, 1, 2, 3, 3]);
        let covariance = clustered_covariance(&[&a, &b], &clusters).unwrap().unwrap();
        assert_eq!(
            covariance[0].to_bits(),
            clustered_variance(&a, &clusters)
                .unwrap()
                .unwrap()
                .to_bits()
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
