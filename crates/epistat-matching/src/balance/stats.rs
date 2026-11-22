//! The standardised difference and the standard deviation behind it, as the
//! balance report uses them.
//!
//! The standardised difference is `difference / sqrt(variance)`, and both
//! functions give `None` for a variance that is not positive and finite, so a
//! zero or undefined spread never produces a number.

/// Standardised difference `difference / sqrt(variance)`, `None` when the
/// variance is not positive and finite or the difference is not finite.
///
/// The variance is the one the caller chose as the standardisation reference.
/// MatchIt documents the choice: the treated variance for the ATT, the control
/// variance for the ATC and the mean of the two for the ATE, always from the
/// sample before matching.
pub(super) fn standardised_difference(difference: f64, variance: Option<f64>) -> Option<f64> {
    let variance = variance.filter(|value| value.is_finite() && *value > 0.0)?;
    Some(difference / variance.sqrt()).filter(|value| value.is_finite())
}

/// The standard deviation for a variance, `None` when the variance is not
/// positive and finite.
pub(super) fn standard_deviation(variance: f64) -> Option<f64> {
    (variance.is_finite() && variance > 0.0).then(|| variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::to_f64;
    use epistat_stats::balance::{
        ecdf_distance as ecdf_distance_stats, eqq_distance as eqq_distance_stats,
    };
    use itertools::Itertools;

    fn quantile_from_sorted_reference(sorted: &[f64], numerator: usize, denominator: usize) -> f64 {
        if sorted.len() == 1 || denominator == 0 {
            return sorted[0];
        }
        let points = sorted.len() - 1;
        let position = numerator * points;
        let lower_idx = position / denominator;
        let upper_idx = (lower_idx + 1).min(points);
        let remainder = position % denominator;
        if remainder == 0 {
            sorted[lower_idx]
        } else {
            let weight = to_f64(remainder) / to_f64(denominator);
            sorted[lower_idx] + weight * (sorted[upper_idx] - sorted[lower_idx])
        }
    }

    fn ecdf_distance_stats_sorted_reference(
        case_values: &[f64],
        control_values: &[f64],
    ) -> Option<(f64, f64)> {
        if case_values.is_empty() || control_values.is_empty() {
            return None;
        }
        let mut case_sorted = case_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect_vec();
        let mut control_sorted = control_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect_vec();
        if case_sorted.is_empty() || control_sorted.is_empty() {
            return None;
        }
        case_sorted.sort_by(f64::total_cmp);
        control_sorted.sort_by(f64::total_cmp);
        let support = case_sorted
            .iter()
            .chain(control_sorted.iter())
            .copied()
            .sorted_by(f64::total_cmp)
            .dedup_by(|left, right| left.total_cmp(right).is_eq())
            .collect_vec();
        let support_len = support.len();

        let case_n = to_f64(case_sorted.len());
        let control_n = to_f64(control_sorted.len());
        let mut diff_sum = 0.0_f64;
        let mut diff_max = 0.0_f64;

        for point in support {
            let case_cdf = to_f64(case_sorted.partition_point(|value| *value <= point)) / case_n;
            let control_cdf =
                to_f64(control_sorted.partition_point(|value| *value <= point)) / control_n;
            let diff = (case_cdf - control_cdf).abs();
            diff_sum += diff;
            diff_max = diff_max.max(diff);
        }

        Some((diff_sum / to_f64(support_len), diff_max))
    }

    fn eqq_distance_stats_reference(
        case_values: &[f64],
        control_values: &[f64],
    ) -> Option<(f64, f64)> {
        let mut case_sorted = case_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect_vec();
        let mut control_sorted = control_values
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .collect_vec();
        if case_sorted.is_empty() || control_sorted.is_empty() {
            return None;
        }
        case_sorted.sort_by(f64::total_cmp);
        control_sorted.sort_by(f64::total_cmp);

        let quantile_points = case_sorted.len().max(control_sorted.len()).max(2);
        let denominator = quantile_points - 1;
        let mut diff_sum = 0.0_f64;
        let mut diff_max = 0.0_f64;
        for idx in 0..quantile_points {
            let case_q = quantile_from_sorted_reference(&case_sorted, idx, denominator);
            let control_q = quantile_from_sorted_reference(&control_sorted, idx, denominator);
            let diff = (case_q - control_q).abs();
            diff_sum += diff;
            diff_max = diff_max.max(diff);
        }
        Some((diff_sum / to_f64(quantile_points), diff_max))
    }

    use proptest::prelude::*;
    use rstest::rstest;

    proptest! {
        #[test]
        fn ecdf_distance_is_invariant_to_permutation(
            mut case in prop::collection::vec(-100.0..100.0f64, 1..50),
            mut control in prop::collection::vec(-100.0..100.0f64, 1..50)
        ) {
            let d1 = ecdf_distance_stats(&case, &control).unwrap();
            case.reverse();
            control.reverse();
            let d2 = ecdf_distance_stats(&case, &control).unwrap();
            prop_assert_eq!(d1.0.to_bits(), d2.0.to_bits());
            prop_assert_eq!(d1.1.to_bits(), d2.1.to_bits());
        }

        #[test]
        fn eqq_distance_is_invariant_to_permutation(
            mut case in prop::collection::vec(-100.0..100.0f64, 1..50),
            mut control in prop::collection::vec(-100.0..100.0f64, 1..50)
        ) {
            let d1 = eqq_distance_stats(&case, &control).unwrap();
            case.reverse();
            control.reverse();
            let d2 = eqq_distance_stats(&case, &control).unwrap();
            prop_assert_eq!(d1.0.to_bits(), d2.0.to_bits());
            prop_assert_eq!(d1.1.to_bits(), d2.1.to_bits());
        }
    }

    #[rstest]
    #[case(1.0, Some(4.0), Some(0.5))]
    #[case(1.0, Some(0.0), None)]
    #[case(1.0, None, None)]
    #[case(f64::NAN, Some(1.0), None)]
    fn test_standardised_difference(
        #[case] difference: f64,
        #[case] variance: Option<f64>,
        #[case] expected: Option<f64>,
    ) {
        assert_eq!(standardised_difference(difference, variance), expected);
    }

    #[test]
    fn ecdf_distance_stats_detects_distribution_difference() {
        let identical = ecdf_distance_stats(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap();
        assert!((identical.0 - 0.0).abs() < 1e-12);
        assert!((identical.1 - 0.0).abs() < 1e-12);

        let shifted = ecdf_distance_stats(&[1.0, 1.0, 1.0], &[2.0, 2.0, 2.0]).unwrap();
        assert!(shifted.0 > 0.0);
        assert!((shifted.1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn eqq_distance_stats_detects_quantile_shift() {
        let identical = eqq_distance_stats(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).unwrap();
        assert!((identical.0 - 0.0).abs() < 1e-12);
        assert!((identical.1 - 0.0).abs() < 1e-12);

        let shifted = eqq_distance_stats(&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0]).unwrap();
        assert!(shifted.0 > 0.0);
        assert!((shifted.1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn eqq_distance_stats_handles_missing_inputs() {
        assert_eq!(eqq_distance_stats(&[], &[1.0, 2.0]), None);
        assert_eq!(eqq_distance_stats(&[1.0, 2.0], &[]), None);
        assert_eq!(ecdf_distance_stats(&[], &[1.0, 2.0]), None);
    }

    #[rstest]
    fn ecdf_distance_stats_matches_sorted_reference(
        #[values(1usize, 3, 17, 257)] n_case: usize,
        #[values(1usize, 4, 31, 251)] n_control: usize,
    ) {
        let case_values = (0..n_case)
            .map(|idx| {
                let x = to_f64(idx);
                0.1f64.mul_add(to_f64(idx % 5), (0.31 * x).sin())
            })
            .collect_vec();
        let control_values = (0..n_control)
            .map(|idx| {
                let x = to_f64(idx);
                0.07f64.mul_add(to_f64(idx % 7), (0.23 * x).cos())
            })
            .collect_vec();

        let expected = ecdf_distance_stats_sorted_reference(&case_values, &control_values).unwrap();
        let actual = ecdf_distance_stats(&case_values, &control_values).unwrap();
        assert_eq!(actual.0.to_bits(), expected.0.to_bits());
        assert_eq!(actual.1.to_bits(), expected.1.to_bits());
    }

    #[rstest]
    fn eqq_distance_stats_matches_reference_for_equal_and_unequal_lengths(
        #[values((1usize, 1usize), (4, 3), (31, 31), (251, 257))] lengths: (usize, usize),
    ) {
        let (n_case, n_control) = lengths;
        let case_values = (0..n_case)
            .map(|idx| {
                let x = to_f64(idx);
                0.11f64.mul_add(to_f64(idx % 5), (0.19 * x).sin())
            })
            .collect_vec();
        let control_values = (0..n_control)
            .map(|idx| {
                let x = to_f64(idx);
                0.09f64.mul_add(to_f64(idx % 7), (0.13 * x).cos())
            })
            .collect_vec();
        let expected = eqq_distance_stats_reference(&case_values, &control_values).unwrap();
        let actual = eqq_distance_stats(&case_values, &control_values).unwrap();
        assert!((actual.0 - expected.0).abs() <= 1e-12 * expected.0.abs().max(1.0));
        assert!((actual.1 - expected.1).abs() <= 1e-12 * expected.1.abs().max(1.0));
    }
}
