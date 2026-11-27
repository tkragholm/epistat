//! Shared inference utilities: influence-function standard errors, the
//! cluster sums of R's `did`, a multiplier bootstrap interval, quantiles and
//! normal critical values.

pub mod did_clusters;

use epistat_core::{Estimate, Interval, Level, Scale};

use crate::types::BootstrapConfig;

/// An estimate with its standard error, interval at `level` and normal
/// p-value; a part that is not finite is `None`.
pub(crate) fn estimate_from(
    att: f64,
    se: f64,
    (lower, upper): (f64, f64),
    level: Level,
) -> Estimate {
    estimate_from_parts(att, finite(se), (finite(lower), finite(upper)), level)
}

/// [`estimate_from`] with the parts already `None` where undefined; a part
/// that is not finite is `None` as well.
pub(crate) fn estimate_from_parts(
    att: f64,
    se: Option<f64>,
    (lower, upper): (Option<f64>, Option<f64>),
    level: Level,
) -> Estimate {
    let se = se.and_then(finite);
    let p_value = epistat_stats::wald(att, se, level, Scale::Identity).p_value;
    Estimate::new(
        att,
        se,
        Some(Interval::new(
            lower.and_then(finite),
            upper.and_then(finite),
            level,
        )),
        p_value,
        Scale::Identity,
    )
}

/// `Some(value)` when it is finite.
pub(crate) fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

/// Whether the confidence level is finite and strictly between 0 and 1.
#[must_use]
pub(crate) fn validate_confidence_level(confidence_level: f64) -> bool {
    confidence_level.is_finite() && confidence_level > 0.0 && confidence_level < 1.0
}

/// The Wald interval `estimate -/+ z se` at the given confidence level, `NaN`
/// at both ends when the level is outside (0, 1).
pub(crate) fn normal_interval(estimate: f64, se: f64, confidence_level: f64) -> (f64, f64) {
    let Ok(level) = epistat_core::Level::alpha(1.0 - confidence_level) else {
        return (f64::NAN, f64::NAN);
    };
    let interval = epistat_inference::interval::normal(estimate, se, level);
    (
        interval.lower.unwrap_or(f64::NAN),
        interval.upper.unwrap_or(f64::NAN),
    )
}

/// Two-sided normal critical value for a confidence level in (0, 1).
#[must_use]
pub(crate) fn z_score_for_confidence(confidence_level: f64) -> f64 {
    epistat_stats::dist::normal::quantile(0.5 * confidence_level + 0.5)
}

/// Standard error of an estimate from its influence function: the standard
/// deviation of the values, with divisor `n`, divided by `sqrt(n)`. NaN for an
/// empty influence function.
#[must_use]
#[expect(
    deprecated,
    reason = "Study I's standard errors stay on the 0.3 influence functions"
)]
pub(crate) fn standard_error_from_influence(influence_function: &[f64]) -> f64 {
    epistat_inference::influence::standard_error(influence_function).unwrap_or(f64::NAN)
}

/// Percentile interval of the Rademacher multiplier bootstrap of an estimate
/// with the given influence function: the `alpha / 2` and `1 - alpha / 2`
/// quantiles of the replicates, with `alpha = 1 - confidence_level`.
/// Replicate `r` is seeded with `replicate_seed(seed, r)`, so the interval does
/// not depend on the number of threads. An empty influence function gives
/// `(NaN, NaN)`.
#[must_use]
pub(crate) fn multiplier_bootstrap_ci(
    estimate: f64,
    influence_function: &[f64],
    confidence_level: f64,
    bootstrap: BootstrapConfig,
) -> (f64, f64) {
    let Ok(mut draws) = epistat_inference::bootstrap::multiplier(
        estimate,
        influence_function,
        epistat_core::Bootstrap::new(bootstrap.reps, bootstrap.seed),
    ) else {
        return (f64::NAN, f64::NAN);
    };

    let alpha = (1.0 - confidence_level).clamp(0.0, 1.0);
    let lower = quantile_unsorted(&mut draws, alpha / 2.0);
    let upper = quantile_unsorted(&mut draws, 1.0 - alpha / 2.0);
    (lower, upper)
}

/// Type 7 quantile of an unsorted sample, which it partially reorders, with
/// `q` clamped to `[0, 1]`. NaN for an empty sample.
#[must_use]
pub(crate) fn quantile_unsorted(values: &mut [f64], q: f64) -> f64 {
    epistat_stats::quantile::type7_unsorted(values, q.clamp(0.0, 1.0)).unwrap_or(f64::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::usize_to_f64;
    use epistat_rng::Stream;

    fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
        epistat_stats::quantile::type7(sorted, q.clamp(0.0, 1.0)).unwrap_or(f64::NAN)
    }

    fn multiplier_bootstrap_ci_sorted_reference(
        estimate: f64,
        influence_function: &[f64],
        confidence_level: f64,
        bootstrap: BootstrapConfig,
    ) -> (f64, f64) {
        let n_f = usize_to_f64(influence_function.len());
        let mut draws = Vec::with_capacity(bootstrap.reps);

        for replicate in 0..bootstrap.reps as u64 {
            let mut stream =
                epistat_rng::Stream::seeded(epistat_rng::replicate_seed(bootstrap.seed, replicate));
            let perturbation = epistat_rng::rademacher_sum(influence_function, &mut stream);
            draws.push(estimate + perturbation / n_f);
        }
        draws.sort_by(f64::total_cmp);

        let alpha = (1.0 - confidence_level).clamp(0.0, 1.0);
        let lower = quantile_sorted(&draws, alpha / 2.0);
        let upper = quantile_sorted(&draws, 1.0 - alpha / 2.0);
        (lower, upper)
    }

    fn rademacher_perturbation_scalar(influence_function: &[f64], rng: &mut Stream) -> f64 {
        use rand::RngExt;

        influence_function.iter().fold(0.0, |acc, value| {
            let sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            acc + sign * value
        })
    }

    use proptest::prelude::*;
    use rstest::rstest;

    proptest! {
        #[test]
        fn standard_error_is_invariant_to_permutation(mut values in prop::collection::vec(-100.0..100.0f64, 1..100)) {
            let se1 = standard_error_from_influence(&values);
            values.reverse();
            let se2 = standard_error_from_influence(&values);
            prop_assert!((se1 - se2).abs() < 1e-12);
        }

        #[test]
        fn quantile_unsorted_matches_reference_proptest(
            mut values in prop::collection::vec(-1000.0..1000.0f64, 1..200),
            q in 0.0..1.0f64
        ) {
            let mut sorted = values.clone();
            sorted.sort_by(f64::total_cmp);
            let expected = quantile_sorted(&sorted, q);
            let actual = quantile_unsorted(&mut values, q);
            prop_assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }

    #[test]
    fn z_score_matches_known_95_percent_value() {
        let z = z_score_for_confidence(0.95);
        assert!((z - 1.959_963_984_540_05).abs() < 1e-8);
    }

    #[test]
    fn confidence_validation_rejects_bounds() {
        assert!(!validate_confidence_level(0.0));
        assert!(!validate_confidence_level(1.0));
        assert!(validate_confidence_level(0.95));
    }

    #[test]
    fn bootstrap_is_reproducible() {
        let influence = [1.0, -1.0, 2.0, -2.0];
        let bootstrap = BootstrapConfig {
            reps: 200,
            seed: 42,
        };
        let ci_a = multiplier_bootstrap_ci(1.0, &influence, 0.95, bootstrap);
        let ci_b = multiplier_bootstrap_ci(1.0, &influence, 0.95, bootstrap);
        assert_eq!(ci_a, ci_b);
    }

    #[rstest]
    fn quantile_unsorted_matches_sorted_interpolation_contract(
        #[values(10usize, 100, 999, 1_000, 1_001)] n: usize,
        #[values(0.025, 0.05, 0.5, 0.95, 0.975)] q: f64,
    ) {
        let values = (0..n)
            .map(|index| {
                let index_f = usize_to_f64(index);
                ((0.37 * index_f).sin() + (0.11 * index_f).cos())
                    .mul_add(0.5, usize_to_f64(index % 7) * 0.01)
            })
            .collect::<Vec<_>>();
        let mut sorted = values.clone();
        sorted.sort_by(f64::total_cmp);
        let expected = quantile_sorted(&sorted, q);
        let mut unsorted = values;
        let actual = quantile_unsorted(&mut unsorted, q);
        assert_eq!(actual.to_bits(), expected.to_bits(), "n={n}, q={q}");
    }

    #[test]
    fn multiplier_bootstrap_ci_matches_sorted_reference() {
        let influence = (0..257usize)
            .map(|index| {
                let index_f = usize_to_f64(index);
                0.4f64.mul_add((0.13 * index_f).cos(), (0.07 * index_f).sin())
            })
            .collect::<Vec<_>>();
        for bootstrap in [
            BootstrapConfig { reps: 19, seed: 11 },
            BootstrapConfig {
                reps: 200,
                seed: 42,
            },
            BootstrapConfig {
                reps: 999,
                seed: 17_431,
            },
        ] {
            let expected =
                multiplier_bootstrap_ci_sorted_reference(0.7, &influence, 0.95, bootstrap);
            let actual = multiplier_bootstrap_ci(0.7, &influence, 0.95, bootstrap);
            assert_eq!(actual.0.to_bits(), expected.0.to_bits());
            assert_eq!(actual.1.to_bits(), expected.1.to_bits());
        }
    }

    #[test]
    fn multiplier_bootstrap_ci_does_not_depend_on_threads() {
        let influence = (0..5_000usize)
            .map(|index| usize_to_f64(index % 13) - 6.0)
            .collect::<Vec<_>>();
        let bootstrap = BootstrapConfig { reps: 300, seed: 5 };
        let on = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| multiplier_bootstrap_ci(0.2, &influence, 0.95, bootstrap))
        };
        let one = on(1);
        let four = on(4);
        assert_eq!(one.0.to_bits(), four.0.to_bits());
        assert_eq!(one.1.to_bits(), four.1.to_bits());
    }

    #[test]
    fn influence_standard_error_matches_reference() {
        let influence = [2.0, -1.0, 0.0, 1.0];
        let se = standard_error_from_influence(&influence);
        assert!((se - 0.559_016_994_374_947_5).abs() < 1e-12);
    }

    #[test]
    fn bitpacked_signs_are_statistically_equivalent_to_scalar_signs() {
        let influence = (0..251usize)
            .map(|index| {
                let index_f = usize_to_f64(index);
                0.3f64.mul_add((0.07 * index_f).cos(), (0.11 * index_f).sin())
            })
            .collect::<Vec<_>>();
        let reps = 20_000usize;
        let seed_scalar = 1_337u64;
        let seed_bitpacked = 7_331u64;

        let mut scalar_rng = Stream::seeded(seed_scalar);
        let scalar_draws = (0..reps)
            .map(|_| rademacher_perturbation_scalar(&influence, &mut scalar_rng))
            .collect::<Vec<_>>();
        let mut bitpacked_stream = epistat_rng::Stream::seeded(seed_bitpacked);
        let bitpacked_draws = (0..reps)
            .map(|_| epistat_rng::rademacher_sum(&influence, &mut bitpacked_stream))
            .collect::<Vec<_>>();

        let mean = |draws: &[f64]| draws.iter().sum::<f64>() / usize_to_f64(draws.len());
        let variance = |draws: &[f64], mean_val: f64| {
            draws
                .iter()
                .map(|value| {
                    let centered = *value - mean_val;
                    centered * centered
                })
                .sum::<f64>()
                / usize_to_f64(draws.len())
        };

        let scalar_mean = mean(&scalar_draws);
        let bitpacked_mean = mean(&bitpacked_draws);
        let scalar_var = variance(&scalar_draws, scalar_mean);
        let bitpacked_var = variance(&bitpacked_draws, bitpacked_mean);

        let mut scalar_sorted = scalar_draws;
        scalar_sorted.sort_by(f64::total_cmp);
        let mut bitpacked_sorted = bitpacked_draws;
        bitpacked_sorted.sort_by(f64::total_cmp);
        let scalar_q05 = quantile_sorted(&scalar_sorted, 0.05);
        let scalar_q50 = quantile_sorted(&scalar_sorted, 0.50);
        let scalar_q95 = quantile_sorted(&scalar_sorted, 0.95);
        let bitpacked_q05 = quantile_sorted(&bitpacked_sorted, 0.05);
        let bitpacked_q50 = quantile_sorted(&bitpacked_sorted, 0.50);
        let bitpacked_q95 = quantile_sorted(&bitpacked_sorted, 0.95);
        let quantile_tolerance = 0.1 * scalar_var.sqrt().max(1.0);

        let mean_diff = (scalar_mean - bitpacked_mean).abs();
        let mean_se = (scalar_var / usize_to_f64(reps) + bitpacked_var / usize_to_f64(reps)).sqrt();
        assert!(
            mean_diff <= 5.0f64.mul_add(mean_se, 0.05),
            "mean mismatch: scalar={scalar_mean}, bitpacked={bitpacked_mean}, se={mean_se}"
        );
        assert!(
            (scalar_var - bitpacked_var).abs() / scalar_var.max(1e-12) <= 0.08,
            "variance mismatch: scalar={scalar_var}, bitpacked={bitpacked_var}"
        );
        assert!(
            (scalar_q05 - bitpacked_q05).abs() <= quantile_tolerance,
            "q05 mismatch: scalar={scalar_q05}, bitpacked={bitpacked_q05}"
        );
        assert!(
            (scalar_q50 - bitpacked_q50).abs() <= quantile_tolerance,
            "q50 mismatch: scalar={scalar_q50}, bitpacked={bitpacked_q50}"
        );
        assert!(
            (scalar_q95 - bitpacked_q95).abs() <= quantile_tolerance,
            "q95 mismatch: scalar={scalar_q95}, bitpacked={bitpacked_q95}"
        );
    }
}
