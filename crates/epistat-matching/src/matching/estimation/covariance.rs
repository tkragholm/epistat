//! Covariance matrices and the rank transform for Mahalanobis preparation.

use super::EstimationError;
use super::config::MahalanobisCovarianceStrategy;
use super::encoding::CovariateMatrix;
use crate::matching::to_f64;
use itertools::Itertools;

fn pooled_covariance_matrix(
    matrix: &CovariateMatrix,
    n_anchor: usize,
    n_candidate: usize,
) -> Result<Vec<f64>, EstimationError> {
    if matrix.n_cols == 0 || matrix.n_rows != n_anchor.saturating_add(n_candidate) {
        return Err(EstimationError::InvalidCovarianceMatrix);
    }
    if n_anchor == 0 || n_candidate == 0 {
        return Err(EstimationError::EmptyInput);
    }

    let anchor_cov = covariance_for_slice(matrix, 0, n_anchor);
    let candidate_cov = covariance_for_slice(matrix, n_anchor, n_candidate);
    let mut pooled = vec![0.0_f64; matrix.n_cols * matrix.n_cols];
    for idx in 0..pooled.len() {
        pooled[idx] = f64::midpoint(anchor_cov[idx], candidate_cov[idx]);
    }
    Ok(pooled)
}

pub(super) fn covariance_matrix(
    matrix: &CovariateMatrix,
    n_anchor: usize,
    n_candidate: usize,
    strategy: MahalanobisCovarianceStrategy,
) -> Result<Vec<f64>, EstimationError> {
    match strategy {
        MahalanobisCovarianceStrategy::PooledWithinGroups => {
            pooled_covariance_matrix(matrix, n_anchor, n_candidate)
        }
        MahalanobisCovarianceStrategy::PooledByDegreesOfFreedom => {
            pooled_by_degrees_of_freedom(matrix, n_anchor, n_candidate)
        }
        MahalanobisCovarianceStrategy::FullSample => {
            if matrix.n_cols == 0 || matrix.n_rows != n_anchor.saturating_add(n_candidate) {
                return Err(EstimationError::InvalidCovarianceMatrix);
            }
            if matrix.n_rows <= 1 {
                return Err(EstimationError::EmptyInput);
            }
            Ok(covariance_for_slice(matrix, 0, matrix.n_rows))
        }
    }
}

/// Sums of squares and cross-products about the group mean, lower triangle
/// mirrored, rows in order and without fused multiply-add.
fn scatter_for_slice(matrix: &CovariateMatrix, start_row: usize, len: usize) -> Vec<f64> {
    let cols = matrix.n_cols;
    let mut means = vec![0.0_f64; cols];
    for row_idx in start_row..(start_row + len) {
        let row = matrix.row(row_idx);
        for col_idx in 0..cols {
            means[col_idx] += row[col_idx];
        }
    }
    for mean in &mut means {
        *mean /= to_f64(len);
    }
    let mut scatter = vec![0.0_f64; cols * cols];
    for row_idx in start_row..(start_row + len) {
        let row = matrix.row(row_idx);
        for left in 0..cols {
            let left_centered = row[left] - means[left];
            for right in 0..=left {
                scatter[left * cols + right] += left_centered * (row[right] - means[right]);
            }
        }
    }
    for left in 0..cols {
        for right in 0..left {
            scatter[right * cols + left] = scatter[left * cols + right];
        }
    }
    scatter
}

/// The within-group scatter of both groups over `n_anchor + n_candidate - 2`.
fn pooled_by_degrees_of_freedom(
    matrix: &CovariateMatrix,
    n_anchor: usize,
    n_candidate: usize,
) -> Result<Vec<f64>, EstimationError> {
    if matrix.n_cols == 0 || matrix.n_rows != n_anchor.saturating_add(n_candidate) {
        return Err(EstimationError::InvalidCovarianceMatrix);
    }
    if n_anchor == 0 || n_candidate == 0 || matrix.n_rows <= 2 {
        return Err(EstimationError::EmptyInput);
    }
    let anchor = scatter_for_slice(matrix, 0, n_anchor);
    let candidate = scatter_for_slice(matrix, n_anchor, n_candidate);
    let denom = to_f64(matrix.n_rows - 2);
    Ok(anchor
        .iter()
        .zip(&candidate)
        .map(|(left, right)| (left + right) / denom)
        .collect())
}

/// The pooled within-group variance of one column, the groups being the first
/// `n_anchor` rows and the rest. A column of only 0 and 1 has
/// `sum_g n_g p_g (1 - p_g) / n`, any other the within-group sum of squares
/// over `n - 2`.
pub(super) fn pooled_within_group_variance(
    matrix: &CovariateMatrix,
    col_idx: usize,
    n_anchor: usize,
    n_candidate: usize,
) -> f64 {
    let value = |row_idx: usize| matrix.values[row_idx * matrix.n_cols + col_idx];
    let n = matrix.n_rows;
    let binary = (0..n).all(|row_idx| value(row_idx) == 0.0 || value(row_idx) == 1.0);
    let groups = [(0, n_anchor), (n_anchor, n_candidate)];
    let mut total = 0.0_f64;
    for (start, len) in groups {
        let mut sum = 0.0_f64;
        for row_idx in start..(start + len) {
            sum += value(row_idx);
        }
        let size = to_f64(len);
        if binary {
            total += sum * (1.0 - sum / size);
        } else {
            let mean = sum / size;
            for row_idx in start..(start + len) {
                let centered = value(row_idx) - mean;
                total += centered * centered;
            }
        }
    }
    if binary {
        total / to_f64(n)
    } else {
        total / to_f64(n - 2)
    }
}

/// Rescales a covariance matrix of the ranks of `n_rows` units so that every
/// diagonal entry is `n (n + 1) / 12`, the variance of `1, ..., n`: entry
/// `(i, j)` is multiplied by `m_i m_j` with `m_i = sd(1..n) / sqrt(cov_ii)`.
/// A column whose rank variance is not positive is an error.
pub(super) fn rescale_rank_covariance(
    covariance: &mut [f64],
    n_cols: usize,
    n_rows: usize,
) -> Result<(), EstimationError> {
    let untied = (to_f64(n_rows) * (to_f64(n_rows) + 1.0) / 12.0).sqrt();
    let mut multiplier = Vec::with_capacity(n_cols);
    for col in 0..n_cols {
        let variance = covariance[col * n_cols + col];
        if !(variance > 0.0) || !variance.is_finite() {
            return Err(EstimationError::InvalidCovarianceMatrix);
        }
        multiplier.push(untied / variance.sqrt());
    }
    for row in 0..n_cols {
        for col in 0..n_cols {
            covariance[row * n_cols + col] *= multiplier[row] * multiplier[col];
        }
    }
    Ok(())
}

fn covariance_for_slice(matrix: &CovariateMatrix, start_row: usize, len: usize) -> Vec<f64> {
    let mut means = vec![0.0_f64; matrix.n_cols];
    for row_idx in start_row..(start_row + len) {
        let row = matrix.row(row_idx);
        for col_idx in 0..matrix.n_cols {
            means[col_idx] += row[col_idx];
        }
    }
    for mean in &mut means {
        *mean /= to_f64(len);
    }

    let mut cov = vec![0.0_f64; matrix.n_cols * matrix.n_cols];
    if len <= 1 {
        return cov;
    }
    for row_idx in start_row..(start_row + len) {
        let row = matrix.row(row_idx);
        for left in 0..matrix.n_cols {
            let left_centered = row[left] - means[left];
            for right in 0..=left {
                cov[left * matrix.n_cols + right] = left_centered
                    .mul_add(row[right] - means[right], cov[left * matrix.n_cols + right]);
            }
        }
    }
    let denom = to_f64(len - 1);
    for left in 0..matrix.n_cols {
        for right in 0..=left {
            let value = cov[left * matrix.n_cols + right] / denom;
            cov[left * matrix.n_cols + right] = value;
            cov[right * matrix.n_cols + left] = value;
        }
    }
    cov
}

pub(super) fn rank_transform_columns(matrix: &mut CovariateMatrix) {
    if matrix.n_rows == 0 || matrix.n_cols == 0 {
        return;
    }
    let mut transformed = matrix.values.clone();

    for col_idx in 0..matrix.n_cols {
        let mut entries = (0..matrix.n_rows)
            .map(|row_idx| (row_idx, matrix.values[row_idx * matrix.n_cols + col_idx]))
            .collect_vec();
        entries.sort_by(|left, right| left.1.total_cmp(&right.1));

        let mut cursor = 0usize;
        while cursor < entries.len() {
            let mut end = cursor + 1;
            while end < entries.len() && entries[end].1.total_cmp(&entries[cursor].1).is_eq() {
                end += 1;
            }

            let low_rank = cursor + 1;
            let high_rank = end;
            let avg_rank = f64::midpoint(to_f64(low_rank), to_f64(high_rank));
            for (row_idx, _) in entries.iter().take(end).skip(cursor) {
                transformed[*row_idx * matrix.n_cols + col_idx] = avg_rank;
            }
            cursor = end;
        }
    }

    matrix.values = transformed;
}
