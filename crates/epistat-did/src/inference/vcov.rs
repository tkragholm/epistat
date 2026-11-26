use epistat_inference::Clusters;
use epistat_linalg::Mat;
use epistat_stats::sum::lanes;
use std::hash::Hash;

use crate::util::usize_to_f64;

/// Compute the covariance matrix from a set of influence function vectors.
///
/// Returns the centred covariance estimate
/// `Sigma = (1/n^2) * sum_i (psi_i - mean(psi)) (psi_i - mean(psi))'`, the
/// convention of [`standard_error_from_influence`](crate::inference::standard_error_from_influence):
/// every sum runs over the fixed lanes of `epistat_stats::sum::lanes`, as that
/// standard error's does, so the diagonal is the square of the standard error
/// before its square root is taken, and the entries do not depend on the
/// platform or the number of threads. R's `did` and the `HonestDiD` vignette
/// form the uncentred `t(psi) %*% psi / n^2` instead; the two differ by
/// `mean(psi) mean(psi)' / n`, which is zero to rounding for influence
/// functions that average to zero.
///
/// # Errors
/// Returns an error if:
/// - Influence vectors are empty or zero-length.
/// - Influence vectors have unequal lengths.
/// - Influence vectors contain non-finite values.
pub fn covariance_matrix_from_influence_matrix(
    influence_functions: &[Vec<f64>],
) -> Result<Vec<Vec<f64>>, String> {
    let (k, _, n_f) = validate_influence_matrix(influence_functions, "covariance matrix")?;
    let means = influence_functions
        .iter()
        .map(|influence| lanes::sum(influence) / n_f)
        .collect::<Vec<_>>();
    let mut covariance = vec![vec![0.0; k]; k];
    for row in 0..k {
        for col in row..k {
            let (row_mean, col_mean) = (means[row], means[col]);
            let cross = lanes::sum_map2(
                &influence_functions[row],
                &influence_functions[col],
                |a, b| (a - row_mean) * (b - col_mean),
            );
            let value = cross / n_f / n_f;
            covariance[row][col] = value;
            covariance[col][row] = value;
        }
    }
    Ok(covariance)
}

/// Compute the cluster-robust covariance matrix from a set of influence function vectors.
///
/// With `G` clusters and `n` rows, entry `(a, b)` is
/// `G / (G - 1) * sum_g S_ga S_gb / n^2`, where `S_ga` is the uncentred sum of
/// estimator `a`'s influence values in cluster `g`, as
/// `epistat_inference::influence::clustered_covariance` forms it.
///
/// # Errors
/// Returns an error if:
/// - Influence vectors are empty or zero-length.
/// - Influence vectors have unequal lengths or contain non-finite values.
/// - Cluster labels length does not match influence vector length.
/// - There are fewer than two clusters.
pub fn clustered_covariance_matrix_from_influence_matrix<L: Eq + Hash>(
    influence_functions: &[Vec<f64>],
    cluster_labels: &[L],
) -> Result<Vec<Vec<f64>>, String> {
    if influence_functions.is_empty() {
        return Err("clustered covariance matrix requires non-empty influence vectors".to_string());
    }
    let n = influence_functions[0].len();
    if n == 0 {
        return Err("clustered covariance matrix requires non-empty influence vectors".to_string());
    }
    if cluster_labels.len() != n {
        return Err(format!(
            "clustered covariance matrix requires aligned influence/cluster lengths ({} vs {})",
            n,
            cluster_labels.len()
        ));
    }

    for influence in influence_functions {
        if influence.len() != n {
            return Err(
                "clustered covariance matrix requires equal-length influence vectors".to_string(),
            );
        }
        if influence.iter().any(|value| !value.is_finite()) {
            return Err("clustered covariance matrix requires finite influence values".to_string());
        }
    }

    clustered_covariance_from_clusters(influence_functions, &Clusters::from_labels(cluster_labels))
}

/// Compute the cluster-robust covariance matrix from a set of influence function vectors using a pre-computed index.
///
/// With `G` clusters and `n` rows, entry `(a, b)` is
/// `G / (G - 1) * sum_g S_ga S_gb / n^2`, where `S_ga` is the uncentred sum of
/// estimator `a`'s influence values in cluster `g`, as
/// `epistat_inference::influence::clustered_covariance` forms it.
///
/// # Errors
/// Returns an error if:
/// - Influence vectors are empty or zero-length.
/// - Influence vectors have unequal lengths or contain non-finite values.
/// - Cluster index length does not match influence vector length.
/// - There are fewer than two clusters.
pub fn clustered_covariance_matrix_from_influence_matrix_index(
    influence_functions: &[Vec<f64>],
    cluster_index: &crate::inference::ClusterIndex,
) -> Result<Vec<Vec<f64>>, String> {
    let (_, n, _) = validate_influence_matrix(influence_functions, "clustered covariance matrix")?;
    if cluster_index.group_assignments.len() != n {
        return Err(format!(
            "clustered covariance matrix requires aligned influence/cluster lengths ({} vs {})",
            n,
            cluster_index.group_assignments.len()
        ));
    }
    clustered_covariance_from_clusters(
        influence_functions,
        &Clusters::from_labels(&cluster_index.group_assignments),
    )
}

fn clustered_covariance_from_clusters(
    influence_functions: &[Vec<f64>],
    clusters: &Clusters,
) -> Result<Vec<Vec<f64>>, String> {
    let columns = influence_functions
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let k = columns.len();
    let flat = epistat_inference::influence::clustered_covariance(&columns, clusters)
        .map_err(|error| error.to_string())?
        .ok_or_else(|| "clustered covariance needs at least two clusters".to_string())?;
    Ok(flat.chunks(k).map(<[f64]>::to_vec).collect())
}

fn validate_influence_matrix(
    influence_functions: &[Vec<f64>],
    context: &str,
) -> Result<(usize, usize, f64), String> {
    if influence_functions.is_empty() {
        return Err(format!("{context} requires non-empty influence vectors"));
    }
    let n = influence_functions[0].len();
    if n == 0 {
        return Err(format!("{context} requires non-empty influence vectors"));
    }

    for influence in influence_functions {
        if influence.len() != n {
            return Err(format!("{context} requires equal-length influence vectors"));
        }
        if influence.iter().any(|value| !value.is_finite()) {
            return Err(format!("{context} requires finite influence values"));
        }
    }

    Ok((influence_functions.len(), n, usize_to_f64(n)))
}

/// Joint Wald test of `beta = 0`, with statistic `W = beta' Sigma^-1 beta`
/// referred to a chi-square distribution with `k` degrees of freedom.
/// Returns the statistic, the p-value and `k`.
///
/// # Errors
/// Returns an error if:
/// - Estimates are empty.
/// - The covariance matrix is not `k x k`.
/// - The covariance matrix is singular.
pub fn joint_wald_test(
    estimates: &[f64],
    covariance_matrix: &[Vec<f64>],
) -> Result<(f64, f64, usize), String> {
    let k = estimates.len();
    if k == 0 {
        return Err("Wald test requires at least one estimate".to_string());
    }
    if covariance_matrix.len() != k || covariance_matrix.iter().any(|row| row.len() != k) {
        return Err("covariance matrix shape does not match the estimates".to_string());
    }
    let covariance = Mat::from_fn(k, k, |i, j| covariance_matrix[i][j]);
    let test = epistat_inference::wald::wald(estimates, covariance.as_ref())
        .map_err(|err| err.to_string())?;
    Ok((test.statistic, test.p_value, test.df))
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_linalg::{Accum, MatRef, Par, matmul};

    const VCOV_BLOCK_SIZE: usize = 32;

    fn manual_covariance(influence_functions: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let k = influence_functions.len();
        let n = influence_functions[0].len();
        let n_f = usize_to_f64(n);
        let means = influence_functions
            .iter()
            .map(|influence| influence.iter().sum::<f64>() / n_f)
            .collect::<Vec<_>>();
        let mut covariance = vec![vec![0.0; k]; k];
        for row in 0..k {
            for col in 0..k {
                let mut sum = 0.0;
                for (&row_value, &col_value) in influence_functions[row]
                    .iter()
                    .zip(influence_functions[col].iter())
                    .take(n)
                {
                    sum = (row_value - means[row]).mul_add(col_value - means[col], sum);
                }
                covariance[row][col] = sum / (n_f * n_f);
            }
        }
        covariance
    }

    fn manual_clustered_covariance(
        influence_functions: &[Vec<f64>],
        group_assignments: &[usize],
        n_groups: usize,
    ) -> Vec<Vec<f64>> {
        let k = influence_functions.len();
        let n = influence_functions[0].len();
        let n_f = usize_to_f64(n);
        let correction = if n_groups >= 2 {
            let g_f = usize_to_f64(n_groups);
            g_f / (g_f - 1.0)
        } else {
            1.0
        };
        let scale = correction / (n_f * n_f);
        let mut covariance = vec![vec![0.0; k]; k];
        for row in 0..k {
            for col in 0..k {
                let mut value = 0.0;
                for group in 0..n_groups {
                    let mut row_sum = 0.0;
                    let mut col_sum = 0.0;
                    for sample in 0..n {
                        if group_assignments[sample] == group {
                            row_sum += influence_functions[row][sample];
                            col_sum += influence_functions[col][sample];
                        }
                    }
                    value = row_sum.mul_add(col_sum, value);
                }
                covariance[row][col] = value * scale;
            }
        }
        covariance
    }

    /// The covariance as it was formed with blocked faer products, kept to
    /// show that the lane sums agree with it to rounding.
    fn covariance_matrix_blocked_matmul(
        influence_functions: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, String> {
        let (k, n, n_f) = validate_influence_matrix(influence_functions, "covariance matrix")?;
        let means = influence_functions
            .iter()
            .map(|influence| influence.iter().sum::<f64>() / n_f)
            .collect::<Vec<_>>();
        let mut covariance = vec![vec![0.0; k]; k];

        for row_start in (0..k).step_by(VCOV_BLOCK_SIZE) {
            let row_end = (row_start + VCOV_BLOCK_SIZE).min(k);
            for col_start in (row_start..k).step_by(VCOV_BLOCK_SIZE) {
                let col_end = (col_start + VCOV_BLOCK_SIZE).min(k);
                let row_block_len = row_end - row_start;
                let col_block_len = col_end - col_start;
                let mut row_centered = vec![0.0; row_block_len * n];
                let mut col_centered = vec![0.0; col_block_len * n];

                for sample_idx in 0..n {
                    for (local_row, estimate_idx) in (row_start..row_end).enumerate() {
                        row_centered[local_row * n + sample_idx] =
                            influence_functions[estimate_idx][sample_idx] - means[estimate_idx];
                    }
                    for (local_col, estimate_idx) in (col_start..col_end).enumerate() {
                        col_centered[local_col * n + sample_idx] =
                            influence_functions[estimate_idx][sample_idx] - means[estimate_idx];
                    }
                }

                let left = MatRef::from_row_major_slice(&row_centered, row_block_len, n);
                let right = MatRef::from_row_major_slice(&col_centered, col_block_len, n);
                let mut block = Mat::<f64>::zeros(row_block_len, col_block_len);
                let scale = 1.0 / (n_f * n_f);
                matmul(
                    block.as_mut(),
                    Accum::Replace,
                    left,
                    right.transpose(),
                    scale,
                    Par::Seq,
                );
                for local_row in 0..row_block_len {
                    let global_row = row_start + local_row;
                    for local_col in 0..col_block_len {
                        let global_col = col_start + local_col;
                        let value = block[(local_row, local_col)];
                        covariance[global_row][global_col] = value;
                        covariance[global_col][global_row] = value;
                    }
                }
            }
        }

        Ok(covariance)
    }

    #[test]
    fn covariance_matrix_agrees_with_the_blocked_products_and_the_standard_error() {
        let mut stream = epistat_rng::Stream::seeded(5);
        for (k, n) in [(1, 7), (3, 100), (40, 257), (70, 1_000)] {
            let influence: Vec<Vec<f64>> = (0..k)
                .map(|_| (0..n).map(|_| stream.uniform() - 0.4).collect())
                .collect();
            let ours = covariance_matrix_from_influence_matrix(&influence).unwrap();
            let blocked = covariance_matrix_blocked_matmul(&influence).unwrap();
            for row in 0..k {
                for col in 0..k {
                    let scale = (ours[row][row] * ours[col][col]).sqrt();
                    assert!((ours[row][col] - blocked[row][col]).abs() <= 1e-13 * scale);
                }
                let se = crate::inference::standard_error_from_influence(&influence[row]);
                let n_f = crate::util::usize_to_f64(n);
                let mean = lanes::sum(&influence[row]) / n_f;
                let mean_square = lanes::centred_squares(&influence[row], mean) / n_f;
                assert_eq!(ours[row][row].to_bits(), (mean_square / n_f).to_bits());
                assert_eq!(se.to_bits(), (mean_square.sqrt() / n_f.sqrt()).to_bits());
            }
        }
    }

    #[test]
    fn covariance_matrix_matches_manual_reference() {
        let influence = vec![
            vec![2.0, -1.0, 0.0, 1.0, 3.0],
            vec![1.0, 1.0, -1.0, -1.0, 0.5],
            vec![0.5, -0.25, 0.75, -1.25, 2.0],
        ];
        let actual = covariance_matrix_from_influence_matrix(&influence).unwrap();
        let expected = manual_covariance(&influence);
        for row in 0..expected.len() {
            for col in 0..expected.len() {
                assert!((actual[row][col] - expected[row][col]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn clustered_covariance_matrix_matches_manual_reference() {
        let influence = vec![
            vec![1.0, 2.0, -1.0, -2.0, 0.5, 1.5],
            vec![0.5, 0.5, -0.5, -0.5, 1.0, 1.0],
            vec![2.0, -1.0, 3.0, -2.0, 0.0, 0.5],
        ];
        let cluster_index = crate::inference::ClusterIndex {
            group_assignments: vec![0, 0, 1, 1, 2, 2],
            n_groups: 3,
        };
        let actual =
            clustered_covariance_matrix_from_influence_matrix_index(&influence, &cluster_index)
                .unwrap();
        let expected = manual_clustered_covariance(
            &influence,
            &cluster_index.group_assignments,
            cluster_index.n_groups,
        );
        for row in 0..expected.len() {
            for col in 0..expected.len() {
                assert!((actual[row][col] - expected[row][col]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn wald_test_simple_case() {
        let estimates = [2.0];
        let vcov = vec![vec![1.0]];
        let (stat, p, df) = joint_wald_test(&estimates, &vcov).unwrap();
        assert_eq!(df, 1);
        assert!((stat - 4.0).abs() < 1e-10);
        // For Chi2(1), stat=4 is roughly 2 standard deviations, so p should be small (~0.0455)
        assert!(p < 0.05 && p > 0.045);
    }
}
