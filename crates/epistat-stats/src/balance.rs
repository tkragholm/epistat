//! Covariate balance between two groups, such as treated and control
//! samples after matching or weighting.
//!
//! Each function returns `None` when the statistic is undefined: an empty
//! group, or a pooled variance that is not positive.
//!
//! # Formulas
//!
//! [`smd`] is the standardised mean difference of a continuous variable \[2\], and
//! [`smd_proportion`] the same for a binary variable from its proportions \[1\]:
//!
//! ```text
//! d = (mean_a - mean_b) / sqrt((var_a + var_b) / 2)
//! d = (p_a - p_b) / sqrt((p_a (1 - p_a) + p_b (1 - p_b)) / 2)
//! ```
//!
//! The denominator is not weighted by group size. [`variance_ratio`] is
//! `var_a / var_b`.
//!
//! [`ecdf_distance`] compares the empirical distribution functions `F_a` and
//! `F_b` at every distinct value `v_1, ..., v_K` in the pooled sample, and returns
//!
//! ```text
//! mean = (1/K) sum_k |F_a(v_k) - F_b(v_k)|,    max = max_k |F_a(v_k) - F_b(v_k)|
//! ```
//!
//! The maximum is the Kolmogorov-Smirnov statistic. [`eqq_distance`] does the same
//! for the empirical quantile functions \[3\]. For groups of equal size it
//! compares the order statistics pair by pair. Otherwise it reads both groups at
//! `N = max(n_a, n_b)` probabilities `k / (N - 1)`, `k = 0, ..., N - 1`, with the
//! type 7 quantile. Both functions ignore non-finite values and return `None` if a
//! group has none left.
//!
//! [`cramers_v`] is `sqrt(chi² / n)` for the two-way table of group by category,
//! with expected counts `row * n_group / n`. With two groups the usual divisor
//! `min(rows - 1, columns - 1)` is 1. Categories absent from both groups are
//! skipped, and an expected count is floored at `1e-12` in the denominator.
//!
//! # References
//!
//! 1. Austin PC. Balance diagnostics for comparing the distribution of baseline
//!    covariates between treatment groups in propensity-score matched samples.
//!    Statistics in Medicine 2009;28:3083-3107.
//! 2. Cohen J. Statistical Power Analysis for the Behavioral Sciences. 2nd ed.
//!    Hillsdale: Lawrence Erlbaum Associates; 1988.
//! 3. Ho DE, Imai K, King G, Stuart EA. MatchIt: nonparametric preprocessing for
//!    parametric causal inference. Journal of Statistical Software
//!    2011;42(8):1-28.

use crate::quantile;

/// Standardised mean difference of a continuous variable, the difference in
/// means over the square root of the mean of the two variances (Cohen 1988).
#[must_use]
pub fn smd(mean_a: f64, mean_b: f64, var_a: f64, var_b: f64) -> Option<f64> {
    let pooled = f64::midpoint(var_a, var_b);
    (pooled > 0.0).then(|| (mean_a - mean_b) / pooled.sqrt())
}

/// Standardised mean difference of a binary variable from its two
/// proportions (Austin 2009).
#[must_use]
pub fn smd_proportion(p_a: f64, p_b: f64) -> Option<f64> {
    let pooled = f64::midpoint(p_a * (1.0 - p_a), p_b * (1.0 - p_b));
    (pooled > 0.0).then(|| (p_a - p_b) / pooled.sqrt())
}

/// `var_a / var_b`, for finite variances with `var_b` positive.
#[must_use]
pub fn variance_ratio(var_a: f64, var_b: f64) -> Option<f64> {
    (var_a.is_finite() && var_b.is_finite() && var_a >= 0.0 && var_b > 0.0).then(|| var_a / var_b)
}

/// The mean and the maximum absolute difference between the two empirical
/// distribution functions, taken over the pooled distinct values. The
/// maximum is the Kolmogorov-Smirnov statistic.
///
/// Non-finite values are ignored.
#[must_use]
pub fn ecdf_distance(a: &[f64], b: &[f64]) -> Option<(f64, f64)> {
    let a = sorted_finite(a)?;
    let b = sorted_finite(b)?;
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let (mut i, mut j, mut points) = (0, 0, 0usize);
    let (mut sum, mut max) = (0.0_f64, 0.0_f64);
    while i < a.len() || j < b.len() {
        let point = match (a.get(i), b.get(j)) {
            (Some(&x), Some(&y)) => x.min(y),
            (Some(&x), None) => x,
            (None, Some(&y)) => y,
            (None, None) => break,
        };
        while i < a.len() && a[i] <= point {
            i += 1;
        }
        while j < b.len() && b[j] <= point {
            j += 1;
        }
        let gap = (i as f64 / na - j as f64 / nb).abs();
        sum += gap;
        max = max.max(gap);
        points += 1;
    }
    Some((sum / points as f64, max))
}

/// The mean and the maximum absolute difference between the two empirical
/// quantile functions (Ho, Imai, King and Stuart 2011).
///
/// Groups of equal size compare their order statistics. Otherwise both are
/// read at `max(n_a, n_b)` evenly spaced probabilities from 0 to 1 by R's
/// type 7 quantile. Non-finite values are ignored.
#[must_use]
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
    Some((gaps.iter().sum::<f64>() / gaps.len() as f64, max))
}

/// Cramér's V of a categorical variable across two groups, from the count of
/// each category in each group.
///
/// Categories absent from both groups are skipped. Returns `None` when both
/// groups are empty.
#[must_use]
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
        chi2 += (ca as f64 - expected_a).powi(2) / expected_a.max(1e-12);
        chi2 += (cb as f64 - expected_b).powi(2) / expected_b.max(1e-12);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn undefined_statistics_are_none() {
        assert_eq!(smd(1.0, 0.0, 0.0, 0.0), None);
        assert_eq!(smd_proportion(1.0, 1.0), None);
        assert_eq!(variance_ratio(1.0, 0.0), None);
        assert_eq!(ecdf_distance(&[], &[1.0]), None);
        assert_eq!(eqq_distance(&[f64::NAN], &[1.0]), None);
        assert_eq!(cramers_v(&[0, 0], &[0, 0]), None);
    }

    #[test]
    fn identical_groups_are_balanced() {
        let x = [1.0, 2.0, 2.0, 5.0];
        assert_eq!(ecdf_distance(&x, &x), Some((0.0, 0.0)));
        assert_eq!(eqq_distance(&x, &x), Some((0.0, 0.0)));
        assert_eq!(cramers_v(&[3, 5], &[6, 10]), Some(0.0));
    }

    #[test]
    fn known_values() {
        assert!((smd(1.0, 0.0, 1.0, 1.0).unwrap() - 1.0).abs() < 1e-15);
        assert!((smd_proportion(0.5, 0.5).unwrap()).abs() < 1e-15);
        // Disjoint supports: the empirical distribution functions differ by 1
        // at the top of the lower group.
        let (_, ks) = ecdf_distance(&[0.0, 1.0], &[2.0, 3.0]).unwrap();
        assert!((ks - 1.0).abs() < 1e-15);
        // A shift by 2 moves every quantile by 2.
        let (mean, max) = eqq_distance(&[0.0, 1.0, 2.0], &[2.0, 3.0, 4.0, 2.5]).unwrap();
        assert!(mean > 0.0 && max <= 2.0 + 1e-12);
        // Perfect association between group and category.
        assert!((cramers_v(&[10, 0], &[0, 10]).unwrap() - 1.0).abs() < 1e-12);
    }
}
