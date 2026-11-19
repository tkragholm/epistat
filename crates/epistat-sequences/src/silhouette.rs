//! The weighted average silhouette width, from the cluster distance sums.
//!
//! Both distance families produce the same intermediate: for every sequence
//! `i` and cluster `c`, `S(i, c) = sum_{j in c} w_j d(i, j)`. The position-wise
//! functions derive it from symbol-weight tables and the optimal-matching
//! triangle from a quadratic pass. What turns those sums into the silhouette
//! is the same for both and lives here.
//!
//! # Algorithm
//!
//! The silhouette of Rousseeuw \[1\] compares the mean distance of a sequence to
//! its own cluster with the mean distance to the nearest other cluster. With
//! cluster weights `W_c = sum_{j in c} w_j`, and `c(i)` the cluster of `i`:
//!
//! ```text
//! a(i) = S(i, c(i)) / W_{c(i)}
//! b(i) = min over c != c(i) with W_c > 0 of  S(i, c) / W_c
//! s(i) = (b(i) - a(i)) / max(a(i), b(i))
//! ASWw = sum_i w_i s(i) / sum_i w_i          over the sequences scored
//! ```
//!
//! This is the weighting of the ASWw statistic of WeightedCluster \[2\]. The
//! denominator of `a(i)` is the full cluster weight, including `w_i` with its
//! zero distance to itself, and the average over sequences is weighted by
//! `w_i`. The cluster weights and the two running totals accumulate in the
//! caller's order. The nearest other cluster is the first strict minimum in
//! rank order.
//!
//! # Edge cases
//!
//! - Fewer than two clusters give 0.
//! - A cluster of zero total weight is left out of `b(i)`.
//! - A sequence is not scored when its own cluster has zero weight, when no
//!   other cluster has positive weight, or when `a(i) = b(i) = 0`.
//! - When no sequence is scored the result is 0.
//!
//! # References
//!
//! 1. Rousseeuw PJ. Silhouettes: a graphical aid to the interpretation and
//!    validation of cluster analysis. J Comput Appl Math. 1987;20:53-65.
//!    doi:10.1016/0377-0427(87)90125-7
//! 2. Studer M. WeightedCluster library manual: a practical guide to creating
//!    typologies of trajectories in the social sciences with R. LIVES Working
//!    Papers 24. Lausanne: NCCR LIVES; 2013.

#![expect(
    clippy::cast_possible_truncation,
    reason = "sequence indices are bounded by the u32 limit checked when a panel or triangle is built, symbol ranks by the alphabet, and cluster ranks by the number of labels"
)]

/// Labels ranked densely in ascending order of value, and the number of ranks.
///
/// The caller may number its clusters any way it likes, and the numbers need
/// not be dense. Ascending order keeps
/// the columns of the sums in the order of the labels, which is the order the
/// nearest other cluster is searched in.
// `usize as u32` on a rank, bounded by the number of labels, which the callers
// bound by `u32::MAX`.
#[must_use]
pub fn rank_labels(labels: &[i64]) -> (Vec<u32>, usize) {
    let mut distinct = labels.to_vec();
    distinct.sort_unstable();
    distinct.dedup();
    let ranks = labels
        .iter()
        .map(|label| distinct.binary_search(label).unwrap_or_else(|at| at) as u32)
        .collect();
    (ranks, distinct.len())
}

/// The weighted average silhouette width over `k` clusters.
///
/// `sums` is row-major with `k` columns, and `row(i)` is the row that holds
/// sequence `i`'s sums, which is `i` itself for the position-wise kernel and
/// the sequence's offset into the triangle for optimal matching. `weights` and
/// `ranks` are in the caller's order.
///
/// The cluster weights and the two running totals accumulate in the caller's
/// order, and the nearest other cluster is the first strict minimum in rank
/// order. Returns 0.0 when fewer than two clusters are given or no sequence
/// can be scored.
#[must_use]
pub fn weighted_average(
    sums: &[f64],
    k: usize,
    row: impl Fn(usize) -> usize,
    weights: &[f64],
    ranks: &[u32],
) -> f64 {
    if k < 2 {
        return 0.0;
    }
    let mut cluster_weight = vec![0.0f64; k];
    for (&weight, &rank) in weights.iter().zip(ranks) {
        cluster_weight[rank as usize] += weight;
    }

    let mut total = 0.0f64;
    let mut total_weight = 0.0f64;
    for (index, (&weight, &rank)) in weights.iter().zip(ranks).enumerate() {
        let own_cluster = rank as usize;
        if cluster_weight[own_cluster] <= 0.0 {
            continue;
        }
        let sums = &sums[row(index) * k..(row(index) + 1) * k];
        let own = sums[own_cluster] / cluster_weight[own_cluster];

        let mut nearest: Option<f64> = None;
        for (cluster, (&sum, &weight_of_cluster)) in sums.iter().zip(&cluster_weight).enumerate() {
            if cluster == own_cluster || weight_of_cluster <= 0.0 {
                continue;
            }
            let mean = sum / weight_of_cluster;
            if nearest.is_none_or(|best| mean < best) {
                nearest = Some(mean);
            }
        }
        let Some(nearest) = nearest else {
            continue;
        };

        let scale = if nearest > own { nearest } else { own };
        if scale <= 0.0 {
            continue;
        }
        total += (nearest - own) / scale * weight;
        total_weight += weight;
    }
    if total_weight == 0.0 {
        0.0
    } else {
        total / total_weight
    }
}

#[cfg(test)]
mod tests {
    use super::{rank_labels, weighted_average};

    #[test]
    fn labels_are_ranked_densely_in_ascending_order() {
        let (ranks, k) = rank_labels(&[7, -2, 7, 3, -2]);
        assert_eq!(k, 3);
        assert_eq!(ranks, vec![2, 0, 2, 1, 0]);
        assert_eq!(rank_labels(&[]), (vec![], 0));
    }

    #[test]
    fn one_cluster_has_no_silhouette() {
        let sums = [0.0, 1.0, 1.0];
        assert!(
            weighted_average(&sums, 1, |i| i, &[1.0, 1.0, 1.0], &[0, 0, 0]).abs() < f64::EPSILON
        );
    }

    #[test]
    fn two_points_per_cluster_by_hand() {
        // Four points on a line at 0, 1, 10, 11 with unit weights and clusters
        // {0, 1} and {10, 11}. With the full cluster weight as denominator,
        // a = 1/2 for every point, b = 21/2, 19/2, 19/2, 21/2, and the average
        // of (b - a) / b is (20/21 + 18/19) / 2.
        let points = [0.0f64, 1.0, 10.0, 11.0];
        let ranks = [0u32, 0, 1, 1];
        let mut sums = vec![0.0f64; 8];
        for (i, &x) in points.iter().enumerate() {
            for (j, &y) in points.iter().enumerate() {
                sums[i * 2 + ranks[j] as usize] += (x - y).abs();
            }
        }
        let value = weighted_average(&sums, 2, |i| i, &[1.0; 4], &ranks);
        let expected = f64::midpoint(20.0 / 21.0, 18.0 / 19.0);
        assert!(
            (value - expected).abs() < 1e-15,
            "{value} against {expected}"
        );
    }

    #[test]
    fn a_cluster_without_weight_is_neither_scored_nor_a_neighbour() {
        // The third cluster carries no weight, so it can be no one's nearest
        // other cluster, and its member is not scored.
        let sums = [0.0, 5.0, 1.0, 5.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let with_empty = weighted_average(&sums, 3, |i| i, &[1.0, 1.0, 0.0], &[0, 1, 2]);
        let two = [0.0, 5.0, 5.0, 0.0];
        let without = weighted_average(&two, 2, |i| i, &[1.0, 1.0], &[0, 1]);
        assert!((with_empty - without).abs() < 1e-15);
        assert!((without - 1.0).abs() < 1e-15);
    }

    #[test]
    fn rows_can_be_read_through_an_offset_map() {
        let points = [0.0f64, 1.0, 10.0, 11.0];
        let ranks = [0u32, 0, 1, 1];
        let mut sums = vec![0.0f64; 8];
        for (i, &x) in points.iter().enumerate() {
            for (j, &y) in points.iter().enumerate() {
                sums[i * 2 + ranks[j] as usize] += (x - y).abs();
            }
        }
        // The same sums, with the caller's order reversed against the rows.
        let direct = weighted_average(&sums, 2, |i| i, &[1.0, 2.0, 3.0, 4.0], &ranks);
        let reversed = weighted_average(&sums, 2, |i| 3 - i, &[4.0, 3.0, 2.0, 1.0], &[1, 1, 0, 0]);
        assert!((direct - reversed).abs() < 1e-15);
    }
}
