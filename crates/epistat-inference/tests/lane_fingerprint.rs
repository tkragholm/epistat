//! The results of the eight-lane reductions in epistat-stats and
//! epistat-inference, hashed to the bit. The hash is the same on NEON, SSE4.2
//! and AVX2 builds and on any thread count, because the lane width and the
//! order of the additions are fixed in the source. The digest also covers
//! normal tail probabilities and normal draws, which go through the platform's
//! libm, so its recorded value is compared only on aarch64 macOS. Equality
//! across thread counts is checked everywhere.

#![expect(deprecated, reason = "the digest pins the 0.3 influence functions")]

use epistat_core::{Bootstrap, Level};
use epistat_epi::{balance, rosenbaum, signed_rank};
use epistat_inference::Clusters;
use epistat_inference::bootstrap;
use epistat_inference::influence::{clustered_covariance, clustered_variance, standard_error};
use epistat_inference::simultaneous::multiplier_bands;
use epistat_stats::{mcmc, quantile};
use epistat_testkit::{Fingerprint, on_threads};

fn data(n: usize, salt: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (((i + salt) * 2_654_435_761) % 1_000_003) as f64 / 997.0 - 500.0)
        .collect()
}

fn fingerprint() -> u64 {
    let mut h = Fingerprint::new();
    for n in [3, 8, 13, 100, 1_003] {
        let x = data(n, 1);
        let w: Vec<f64> = data(n, 2).iter().map(|v| v.abs() + 0.5).collect();
        let y = data(n + 5, 3);
        h = h.option_f64(mcmc::autocorrelation(&x, 1));
        h = h.option_f64(mcmc::effective_sample_size(&x));
        h = h.option_f64(mcmc::split_rhat(&[
            &x,
            &x[..n].iter().rev().copied().collect::<Vec<_>>(),
        ]));
        let mut sorted = x.clone();
        sorted.sort_by(f64::total_cmp);
        h = h.option_f64(quantile::weighted_type1(&sorted, &w, 0.37));
        h = h.option_f64(signed_rank::normal_upper_tail(
            &w,
            0.5 * w.iter().sum::<f64>(),
            0.5,
        ));
        let (m, mx) = balance::eqq_distance(&x, &y).unwrap_or((f64::NAN, f64::NAN));
        h = h.f64(m).f64(mx);
        let wy = data(n + 5, 4).iter().map(|v| v.abs()).collect::<Vec<_>>();
        if let Some((m, mx)) = balance::eqq_distance_weighted(&x, &w, &y, &wy) {
            h = h.f64(m).f64(mx);
        }
        h = h.option_f64(balance::cramers_v_weighted(
            &w,
            &w.iter().rev().copied().collect::<Vec<_>>(),
        ));
        if let Ok(b) = rosenbaum::wilcoxon_bounds(&x, 1.5, rosenbaum::WilcoxonDistribution::Normal)
        {
            h = h.f64(b.statistic);
        }
        h = h.f64(standard_error(&x).unwrap());
        let clusters =
            Clusters::from_labels_first_seen(&(0..n).map(|i| i % 11).collect::<Vec<_>>());
        h = h.option_f64(clustered_variance(&x, &clusters).unwrap());
        if let Some(cov) = clustered_covariance(&[&x, &w], &clusters).unwrap() {
            h = h.f64s(&cov);
        }
        h = h.f64s(&bootstrap::cluster_totals(&[&x, &w], Bootstrap::new(40, 9)).unwrap());
        h = h.f64s(&bootstrap::multiplier(1.0, &x, Bootstrap::new(40, 5)).unwrap());
        let bands = multiplier_bands(
            &[&x, &w],
            None,
            Level::alpha(0.05).unwrap(),
            Bootstrap::new(60, 3),
        )
        .unwrap();
        h = h.f64(bands.critical);
        for s in &bands.se {
            h = h.option_f64(*s);
        }
    }
    h.finish()
}

#[test]
fn the_lane_reductions_have_fixed_bits() {
    let expected = fingerprint();
    let counts = [1, 4, 12];
    for (run, threads) in on_threads(&counts, fingerprint).into_iter().zip(counts) {
        assert_eq!(run, expected, "{threads} threads");
    }
    // The digest covers the normal distribution function and normal draws,
    // which go through `exp`, `ln` and `cos`, whose last bits depend on the
    // maths library and, on Windows, on the processor the library dispatches
    // for, so the constant is checked only on aarch64 macOS, where it was
    // recorded. The thread counts above must agree everywhere. Elsewhere the
    // digest is written to `EPISTAT_HASH_OUT` when that is set.
    epistat_testkit::hash_out("lane_reductions", expected);
    if epistat_testkit::RECORDED_PLATFORM {
        assert_eq!(expected, 0xf4a0_3bd1_4db8_f150);
    }
}
