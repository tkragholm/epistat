//! Variances of an estimator from its influence function: one value `psi_i`
//! per observation, with the estimator's error about `sum(psi) / n`.

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
