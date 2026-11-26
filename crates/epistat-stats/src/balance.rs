//! Deprecated: this module moved to `epistat_epi::balance` in 0.3.0, where its
//! documentation, explainer and tests are. The copy here keeps the items and
//! the results of 0.2 for one release and is removed in 0.4.0.

use crate::quantile;
use crate::sum::lanes;

/// Standardised mean difference of a continuous variable, the difference in
/// means over the square root of the mean of the two variances (Cohen 1988).
#[must_use]
#[deprecated(since = "0.3.0", note = "moved to `epistat_epi::balance::smd`")]
pub fn smd(mean_a: f64, mean_b: f64, var_a: f64, var_b: f64) -> Option<f64> {
    let pooled = f64::midpoint(var_a, var_b);
    (pooled > 0.0).then(|| (mean_a - mean_b) / pooled.sqrt())
}

/// Standardised mean difference of a binary variable from its two
/// proportions (Austin 2009).
#[must_use]
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::balance::smd_proportion`"
)]
pub fn smd_proportion(p_a: f64, p_b: f64) -> Option<f64> {
    let pooled = f64::midpoint(p_a * (1.0 - p_a), p_b * (1.0 - p_b));
    (pooled > 0.0).then(|| (p_a - p_b) / pooled.sqrt())
}

/// `var_a / var_b`, for finite variances with `var_b` positive.
#[must_use]
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::balance::variance_ratio`"
)]
pub fn variance_ratio(var_a: f64, var_b: f64) -> Option<f64> {
    (var_a.is_finite() && var_b.is_finite() && var_a >= 0.0 && var_b > 0.0).then(|| var_a / var_b)
}

/// The mean and the maximum absolute difference between the two empirical
/// distribution functions, taken over the pooled distinct values. The
/// maximum is the Kolmogorov-Smirnov statistic.
///
/// Non-finite values are ignored.
#[must_use]
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::balance::ecdf_distance`"
)]
pub fn ecdf_distance(a: &[f64], b: &[f64]) -> Option<(f64, f64)> {
    let a = sorted_finite(a)?;
    let b = sorted_finite(b)?;
    let (mut sum, mut max, mut points) = (0.0_f64, 0.0_f64, 0usize);
    for gap in ecdf_gaps_sorted(&a, &b) {
        sum += gap;
        max = max.max(gap);
        points += 1;
    }
    Some((sum / points as f64, max))
}

/// The absolute difference between the two empirical distribution functions
/// at each distinct value of the pooled sample, in ascending order of value.
///
/// Both samples are given in ascending order, without NaN, and neither is
/// empty. Negative and positive zero are one value.
fn ecdf_gaps_sorted<'a>(a: &'a [f64], b: &'a [f64]) -> impl Iterator<Item = f64> + 'a {
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let (mut i, mut j) = (0, 0);
    std::iter::from_fn(move || {
        let point = match (a.get(i), b.get(j)) {
            (Some(&x), Some(&y)) => x.min(y),
            (Some(&x), None) => x,
            (None, Some(&y)) => y,
            (None, None) => return None,
        };
        while i < a.len() && a[i] <= point {
            i += 1;
        }
        while j < b.len() && b[j] <= point {
            j += 1;
        }
        Some((i as f64 / na - j as f64 / nb).abs())
    })
}

/// The mean and the maximum absolute difference between the two empirical
/// quantile functions (Ho, Imai, King and Stuart 2011).
///
/// Groups of equal size compare their order statistics. Otherwise both are
/// read at `max(n_a, n_b)` evenly spaced probabilities from 0 to 1 by R's
/// type 7 quantile. Non-finite values are ignored.
#[must_use]
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::balance::eqq_distance`"
)]
pub fn eqq_distance(a: &[f64], b: &[f64]) -> Option<(f64, f64)> {
    let a = sorted_finite(a)?;
    let b = sorted_finite(b)?;
    let gaps: Vec<f64> = if a.len() == b.len() {
        a.iter().zip(&b).map(|(x, y)| (x - y).abs()).collect()
    } else {
        let points = a.len().max(b.len());
        (0..points)
            .map(|k| {
                let qa = quantile::type7_ratio(&a, k, points - 1).unwrap_or(a[0]);
                let qb = quantile::type7_ratio(&b, k, points - 1).unwrap_or(b[0]);
                (qa - qb).abs()
            })
            .collect()
    };
    let max = gaps.iter().fold(0.0_f64, |m, &g| m.max(g));
    Some((lanes::sum(&gaps) / gaps.len() as f64, max))
}

/// The mean and the maximum absolute difference between the two weighted
/// empirical quantile functions, as `summary()` of MatchIt reports them.
///
/// Each group is given as its values and a non-negative weight per value.
/// Units of weight zero and non-finite values carry no mass. The weights of a
/// group are scaled to sum to one.
///
/// Both groups are compared at the cumulative weights of the group with fewer
/// units of positive weight. The other group is read there as a step function:
/// at a cumulative weight `p` it returns the value of the last unit whose
/// cumulative weight is at most `p`, or its smallest value when there is none.
/// If both groups have the same number of units of positive weight their
/// order statistics are compared in pairs. When the two groups together take at
/// most five distinct values, counting those of units without weight, the
/// larger group is read at the cumulative weights of its distinct values
/// instead. A value counts for the cumulative weight at which it is
/// reached, so a `p` equal to the weight of the smaller values returns the next
/// value.
///
/// At `p = 1` the step function returns the largest value of the larger group
/// with positive weight. MatchIt's few-valued branch returns the largest value
/// of the pooled sample there, which differs when the larger group does not
/// reach it. That is a defect in `MatchIt` 4.8.2, and this function does not
/// reproduce it.
///
/// `None` when a group has no unit of positive weight, or the lengths of a
/// group's values and weights differ, or a weight is negative or not finite.
#[must_use]
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::balance::eqq_distance_weighted`"
)]
pub fn eqq_distance_weighted(
    a: &[f64],
    weights_a: &[f64],
    b: &[f64],
    weights_b: &[f64],
) -> Option<(f64, f64)> {
    let group_a = QuantileGroup::new(a, weights_a)?;
    let group_b = QuantileGroup::new(b, weights_b)?;
    eqq_distance_of_groups(&group_a, &group_b)
}

/// [`eqq_distance_weighted`] for groups already in ascending order of value.
///
/// Each group is given as finite values in ascending order, with negative and
/// positive zero counted as equal and units that tie in their original order,
/// the weight of each unit, and the total weight of the group summed in the
/// original order before sorting. The result is the same as that of
/// [`eqq_distance_weighted`] on the unsorted group, bit for bit. This lets a
/// caller that needs the distance under several weightings sort once.
///
/// `None` when a group has no unit of positive weight, or the lengths of a
/// group's values and weights differ, or a weight is negative or not finite.
#[must_use]
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::balance::eqq_distance_weighted_sorted`"
)]
pub fn eqq_distance_weighted_sorted(
    a: &[f64],
    weights_a: &[f64],
    total_a: f64,
    b: &[f64],
    weights_b: &[f64],
    total_b: f64,
) -> Option<(f64, f64)> {
    let group_a = QuantileGroup::from_sorted(a, weights_a, total_a)?;
    let group_b = QuantileGroup::from_sorted(b, weights_b, total_b)?;
    eqq_distance_of_groups(&group_a, &group_b)
}

#[expect(
    clippy::float_cmp,
    reason = "values are compared for being the same sample value"
)]
fn eqq_distance_of_groups(group_a: &QuantileGroup, group_b: &QuantileGroup) -> Option<(f64, f64)> {
    if group_a.values.is_empty() || group_b.values.is_empty() {
        return None;
    }
    let mut pooled: Vec<f64> = group_a
        .all_values
        .iter()
        .chain(&group_b.all_values)
        .copied()
        .collect();
    pooled.sort_by(|p, q| p.partial_cmp(q).unwrap_or(std::cmp::Ordering::Equal));
    pooled.dedup_by(|x, y| *x == *y);
    let few_values = pooled.len() <= 5;

    let (small, large) = if group_a.values.len() <= group_b.values.len() {
        (group_a, group_b)
    } else {
        (group_b, group_a)
    };
    let gaps: Vec<f64> = if small.values.len() == large.values.len() {
        small
            .values
            .iter()
            .zip(&large.values)
            .map(|(x, y)| (x - y).abs())
            .collect()
    } else {
        let probabilities = cumulative_probabilities(&small.weights);
        let read = if few_values {
            large.read_at_distinct_values(&pooled, &probabilities)
        } else {
            large.read_as_steps(&probabilities)
        };
        small
            .values
            .iter()
            .zip(&read)
            .map(|(x, y)| (x - y).abs())
            .collect()
    };
    let max = gaps.iter().fold(0.0_f64, |m, &g| m.max(g));
    Some((lanes::sum(&gaps) / gaps.len() as f64, max))
}

/// Cumulative sums of the weights scaled to end at one.
fn cumulative_probabilities(weights: &[f64]) -> Vec<f64> {
    let mut running = 0.0_f64;
    let sums: Vec<f64> = weights
        .iter()
        .map(|w| {
            running += w;
            running
        })
        .collect();
    let last = running;
    sums.into_iter().map(|s| s / last).collect()
}

/// One group's units of positive weight in ascending order of value, with
/// weights scaled to sum to one over the whole group.
struct QuantileGroup {
    all_values: Vec<f64>,
    values: Vec<f64>,
    weights: Vec<f64>,
}

impl QuantileGroup {
    fn new(values: &[f64], weights: &[f64]) -> Option<Self> {
        if values.len() != weights.len() || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return None;
        }
        let mut units: Vec<(f64, f64)> = values
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .filter(|(x, _)| x.is_finite())
            .collect();
        // The total is summed in the order given, before sorting.
        let given: Vec<f64> = units.iter().map(|u| u.1).collect();
        let total = lanes::sum(&given);
        // Negative and positive zero are one value, so units that tie keep
        // their given order.
        units.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
        let sorted_values: Vec<f64> = units.iter().map(|u| u.0).collect();
        let sorted_weights: Vec<f64> = units.iter().map(|u| u.1).collect();
        Some(Self::scaled(sorted_values, &sorted_weights, total))
    }

    /// The group from values that are finite and already ascending, with the
    /// total weight summed in the original order.
    fn from_sorted(values: &[f64], weights: &[f64], total: f64) -> Option<Self> {
        if values.len() != weights.len() || weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
            return None;
        }
        Some(Self::scaled(values.to_vec(), weights, total))
    }

    fn scaled(all_values: Vec<f64>, weights: &[f64], total: f64) -> Self {
        let positive: Vec<(f64, f64)> = all_values
            .iter()
            .copied()
            .zip(weights.iter().copied())
            .filter(|u| u.1 > 0.0)
            .collect();
        Self {
            all_values,
            values: positive.iter().map(|u| u.0).collect(),
            weights: positive.iter().map(|u| u.1 / total).collect(),
        }
    }

    /// The value of the last unit whose cumulative weight is at most each
    /// probability, or the smallest value when there is none.
    fn read_as_steps(&self, probabilities: &[f64]) -> Vec<f64> {
        let cumulative = cumulative_probabilities(&self.weights);
        probabilities
            .iter()
            .map(|&p| {
                let reached = cumulative.partition_point(|&c| c <= p);
                self.values[reached.saturating_sub(1)]
            })
            .collect()
    }

    #[expect(
        clippy::float_cmp,
        reason = "values are compared for being the same sample value"
    )]
    /// The distinct value at each probability, where the distinct values of
    /// `pooled` in ascending order take the share of the group's weight that
    /// their units have.
    fn read_at_distinct_values(&self, pooled: &[f64], probabilities: &[f64]) -> Vec<f64> {
        let total = lanes::sum(&self.weights);
        let mass: Vec<f64> = pooled
            .iter()
            .map(|&value| {
                let at_value: f64 = self
                    .values
                    .iter()
                    .zip(&self.weights)
                    .filter(|(x, _)| **x == value)
                    .map(|(_, w)| *w)
                    .sum();
                at_value / total
            })
            .collect();
        let mut edges = vec![0.0_f64];
        edges.extend(cumulative_probabilities(&mass));
        let top = self.values[self.values.len() - 1];
        probabilities
            .iter()
            .map(|&p| {
                let reached = edges.partition_point(|&e| e <= p);
                if p >= edges[edges.len() - 1] {
                    top
                } else {
                    pooled[reached.max(1) - 1]
                }
            })
            .collect()
    }
}

/// Cramér's V of a categorical variable across two groups from the weight of
/// each category in each group.
///
/// With weights of one it is [`cramers_v`] on counts, and it does not depend
/// on the scale of the weights. Categories without weight in both groups are
/// skipped. `None` when the total weight is not positive, or a weight is
/// negative or not finite, or the slices differ in length.
///
/// MatchIt and cobalt have no Cramér's V, so there is no R reference. The
/// weighted form treats the weights as counts.
#[must_use]
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::balance::cramers_v_weighted`"
)]
pub fn cramers_v_weighted(weights_a: &[f64], weights_b: &[f64]) -> Option<f64> {
    if weights_a.len() != weights_b.len()
        || weights_a
            .iter()
            .chain(weights_b)
            .any(|w| !w.is_finite() || *w < 0.0)
    {
        return None;
    }
    let total_a = lanes::sum(weights_a);
    let total_b = lanes::sum(weights_b);
    let n = total_a + total_b;
    if n <= 0.0 {
        return None;
    }
    let mut chi2 = 0.0;
    for (&ca, &cb) in weights_a.iter().zip(weights_b) {
        let row = ca + cb;
        if row == 0.0 {
            continue;
        }
        let expected_a = row * total_a / n;
        let expected_b = row * total_b / n;
        if expected_a > 0.0 {
            chi2 += (ca - expected_a).powi(2) / expected_a;
        }
        if expected_b > 0.0 {
            chi2 += (cb - expected_b).powi(2) / expected_b;
        }
    }
    Some((chi2 / n).sqrt())
}

/// Cramér's V of a categorical variable across two groups, from the count of
/// each category in each group.
///
/// Categories absent from both groups are skipped. Returns `None` when both
/// groups are empty.
#[must_use]
#[deprecated(since = "0.3.0", note = "moved to `epistat_epi::balance::cramers_v`")]
pub fn cramers_v(counts_a: &[usize], counts_b: &[usize]) -> Option<f64> {
    let total_a: usize = counts_a.iter().sum();
    let total_b: usize = counts_b.iter().sum();
    let total = total_a + total_b;
    if total == 0 {
        return None;
    }
    let (na, nb, n) = (total_a as f64, total_b as f64, total as f64);
    let mut chi2 = 0.0;
    for (&ca, &cb) in counts_a.iter().zip(counts_b) {
        let row = (ca + cb) as f64;
        if row == 0.0 {
            continue;
        }
        let expected_a = row * na / n;
        let expected_b = row * nb / n;
        if expected_a > 0.0 {
            chi2 += (ca as f64 - expected_a).powi(2) / expected_a;
        }
        if expected_b > 0.0 {
            chi2 += (cb as f64 - expected_b).powi(2) / expected_b;
        }
    }
    // Two groups, so min(rows - 1, columns - 1) is 1 for two or more
    // categories.
    Some((chi2 / n).sqrt())
}

fn sorted_finite(values: &[f64]) -> Option<Vec<f64>> {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    if sorted.is_empty() {
        return None;
    }
    sorted.sort_by(f64::total_cmp);
    Some(sorted)
}
