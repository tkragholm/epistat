//! Cluster-robust variance in the convention of R's `did`.
//!
//! # Method
//!
//! For `n` units with influence function values `psi_i` and `G` clusters, the
//! cluster sum is the sum of the centred values in the cluster, divided by `n`:
//!
//! ```text
//! s_g = (1/n) sum_{i in g} (psi_i - mean(psi))
//! ```
//!
//! The standard error is `sqrt(sum_g s_g^2)` and the covariance of two
//! estimates `a` and `b` is `sum_g s_ga s_gb`. There is no finite-cluster
//! factor `G / (G - 1)`, which is what `did::aggte` and `did::att_gt` do when
//! `clustervars` is set and `bstrap = FALSE`. The cluster-robust variance
//! with that factor is [`epistat_inference::Influence::variance`] under
//! `ClusterRule::SmallSample`; the two are not interchangeable.
//!
//! R forms the same sums without centring. The two differ by
//! `mean(psi)^2 / n` per unit, which is zero to rounding for an influence
//! function evaluated at its own estimate.
//!
//! With every unit its own cluster, the standard error is the unclustered
//! one, `sqrt(mean((psi_i - mean(psi))^2) / n)`, up to the order of the
//! floating-point sums.
//!
//! # Determinism
//!
//! Observations are added in input order, clusters in the order of
//! [`Clusters`]. Nothing is parallel, so the results do not depend on the
//! number of threads. The diagonal of [`did_covariance`] is the square of
//! [`did_standard_error`] of the same column up to the rounding of the square
//! root.
//!
//! # References
//!
//! Callaway B, Sant'Anna PHC. Difference-in-differences with multiple time
//! periods. J Econom 2021;225(2):200-30. R package `did` 2.5.1, functions
//! `getSE` and `mboot`.

use epistat_inference::Clusters;

use crate::util::usize_to_f64;

/// Errors from the cluster sums.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum DidClusterError {
    /// No units, or no estimates.
    #[error("cluster sums need at least one unit and one estimate")]
    Empty,
    /// An influence vector and the clusters differ in length.
    #[error("influence has {actual} values but the clusters cover {expected} units")]
    Length {
        /// Units covered by the clusters.
        expected: usize,
        /// Length of the influence vector.
        actual: usize,
    },
    /// An influence value is NaN or infinite.
    #[error("influence values must be finite")]
    NotFinite,
}

fn check(influence: &[f64], clusters: &Clusters) -> Result<(), DidClusterError> {
    if influence.is_empty() {
        return Err(DidClusterError::Empty);
    }
    if influence.len() != clusters.len() {
        return Err(DidClusterError::Length {
            expected: clusters.len(),
            actual: influence.len(),
        });
    }
    if influence.iter().any(|value| !value.is_finite()) {
        return Err(DidClusterError::NotFinite);
    }
    Ok(())
}

/// The centred influence function summed within each cluster and divided by
/// the unit count, one value per cluster in cluster order.
///
/// The Euclidean norm of the result is the standard error of the estimate.
///
/// # Errors
/// [`DidClusterError`] for an empty vector, a length that differs from the
/// clusters' or a value that is not finite.
pub fn did_cluster_sums(
    influence: &[f64],
    clusters: &Clusters,
) -> Result<Vec<f64>, DidClusterError> {
    check(influence, clusters)?;
    Ok(cluster_sums(influence, clusters))
}

/// [`did_cluster_sums`] for checked input.
fn cluster_sums(influence: &[f64], clusters: &Clusters) -> Vec<f64> {
    let n_f = usize_to_f64(influence.len());
    let mean = influence.iter().sum::<f64>() / n_f;
    let mut sums = vec![0.0; clusters.count()];
    for (value, &group) in influence.iter().zip(clusters.of()) {
        sums[group] += value - mean;
    }
    for sum in &mut sums {
        *sum /= n_f;
    }
    sums
}

/// The standard error of an estimate from its influence function, clustered as
/// `did` does. See the module documentation for the convention.
///
/// # Errors
/// As [`did_cluster_sums`].
pub fn did_standard_error(influence: &[f64], clusters: &Clusters) -> Result<f64, DidClusterError> {
    check(influence, clusters)?;
    Ok(standard_error(influence, clusters))
}

/// [`did_standard_error`] for checked input.
pub(crate) fn standard_error(influence: &[f64], clusters: &Clusters) -> f64 {
    cluster_sums(influence, clusters)
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

/// The covariance matrix of several estimates from their influence functions,
/// clustered as `did` does, row-major `k x k`.
///
/// # Errors
/// As [`did_cluster_sums`], for every vector, and
/// [`DidClusterError::Empty`] for no vectors.
pub fn did_covariance(
    influence: &[Vec<f64>],
    clusters: &Clusters,
) -> Result<Vec<f64>, DidClusterError> {
    if influence.is_empty() {
        return Err(DidClusterError::Empty);
    }
    for vector in influence {
        check(vector, clusters)?;
    }
    Ok(covariance(influence, clusters))
}

/// [`did_covariance`] for checked input.
pub(crate) fn covariance<V: AsRef<[f64]>>(influence: &[V], clusters: &Clusters) -> Vec<f64> {
    let dim = influence.len();
    let sums: Vec<Vec<f64>> = influence
        .iter()
        .map(|vector| cluster_sums(vector.as_ref(), clusters))
        .collect();
    let mut out = vec![0.0; dim * dim];
    for row in 0..dim {
        for col in 0..dim {
            out[row * dim + col] = sums[row]
                .iter()
                .zip(&sums[col])
                .map(|(left, right)| left * right)
                .sum();
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn singleton_clusters_give_the_unclustered_standard_error() {
        let psi = [0.5, -1.25, 2.0, 0.75, -2.0, 0.0];
        let got = did_standard_error(&psi, &Clusters::singletons(psi.len())).unwrap();
        let want = crate::inference::standard_error_from_influence(&psi);
        assert!((got - want).abs() < 1e-15, "{got} vs {want}");
    }

    #[test]
    fn the_covariance_diagonal_is_the_squared_standard_error_to_the_bit() {
        let a = vec![0.5, -1.25, 2.0, 0.75, -2.0, 0.0, 1.0, -1.0];
        let b = vec![1.5, 0.25, -0.5, 0.75, 1.0, -3.0, 0.0, 0.5];
        let clusters = Clusters::from_labels_first_seen(&[1_i64, 1, 2, 2, 3, 3, 3, 4]);
        let cov = did_covariance(&[a.clone(), b.clone()], &clusters).unwrap();
        let se_a = did_standard_error(&a, &clusters).unwrap();
        let se_b = did_standard_error(&b, &clusters).unwrap();
        assert!((cov[0].sqrt() - se_a).abs() <= f64::EPSILON * se_a);
        assert!((cov[3].sqrt() - se_b).abs() <= f64::EPSILON * se_b);
        assert_eq!(cov[1].to_bits(), cov[2].to_bits());
    }

    #[test]
    fn there_is_no_finite_cluster_factor() {
        // Two clusters of equal size with sums +s and -s: the variance is
        // 2 (s / n)^2, not 2 times G / (G - 1) = 4 times that.
        let psi = [1.0, 1.0, -1.0, -1.0];
        let clusters = Clusters::from_labels_first_seen(&[0_i64, 0, 1, 1]);
        let se = did_standard_error(&psi, &clusters).unwrap();
        assert!((se - (2.0_f64 * 0.25).sqrt()).abs() < 1e-15);
    }

    /// PHD's `did_cluster_sums` and `event_time_covariance`, transcribed. The
    /// public functions must give the same bits, so that the kernel can call
    /// them in place of its copy.
    #[test]
    fn the_sums_and_the_covariance_follow_the_summation_order_of_the_kernel() {
        let mut stream = epistat_rng::Stream::seeded(17);
        let n: u32 = 301;
        let labels: Vec<u32> = (0..n).map(|i| (i * 7) % 41).collect();
        let psi: Vec<Vec<f64>> = (0..3)
            .map(|_| (0..n).map(|_| stream.uniform() - 0.4).collect())
            .collect();
        let clusters = Clusters::from_labels_first_seen(&labels);

        let transcribed_sums = |values: &[f64]| {
            let n_f = values.len() as f64;
            let mean = values.iter().sum::<f64>() / n_f;
            let mut sums = vec![0.0; clusters.count()];
            for (value, &group) in values.iter().zip(clusters.of()) {
                sums[group] += value - mean;
            }
            for sum in &mut sums {
                *sum /= n_f;
            }
            sums
        };
        for values in &psi {
            let want = transcribed_sums(values);
            assert_eq!(did_cluster_sums(values, &clusters).unwrap(), want);
            let se = want.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert_eq!(
                did_standard_error(values, &clusters).unwrap().to_bits(),
                se.to_bits()
            );
        }
        let covariance = did_covariance(&psi, &clusters).unwrap();
        for row in 0..3 {
            for col in 0..3 {
                let want: f64 = transcribed_sums(&psi[row])
                    .iter()
                    .zip(transcribed_sums(&psi[col]))
                    .map(|(l, r)| l * r)
                    .sum();
                assert_eq!(covariance[row * 3 + col].to_bits(), want.to_bits());
            }
        }
    }

    #[test]
    fn bad_input_is_an_error() {
        let clusters = Clusters::singletons(3);
        assert_eq!(
            did_standard_error(&[1.0, 2.0], &clusters),
            Err(DidClusterError::Length {
                expected: 3,
                actual: 2
            })
        );
        assert_eq!(
            did_standard_error(&[1.0, f64::NAN, 2.0], &clusters),
            Err(DidClusterError::NotFinite)
        );
        assert_eq!(
            did_standard_error(&[], &Clusters::singletons(0)),
            Err(DidClusterError::Empty)
        );
    }
}
