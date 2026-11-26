//! The calls Studies I and II make into this crate, hashed to the bit.
//!
//! Each case makes the calls of PHD's `crates/kernels` (`interval.rs`,
//! `stats.rs`, `numeric.rs`, `equity.rs`) and `crates/native` with the
//! arguments they pass: the compatibility approximations of the normal
//! distribution with the kernels' clamping, the accurate normal quantile, the
//! chi-square tail, the rounded-rank quantile, weighted moments, direct
//! standardisation, the Kitagawa decomposition and the two exact sums. A change
//! that moves a hash moves a protected study, so a port of these calls to new
//! paths must reproduce every hash.
//!
//! The hashes were recorded on aarch64 macOS. Print them with
//! `STUDY_PATH_PRINT=1 cargo test --test study_path -- --nocapture`.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use epistat_stats::{decomposition, gamma, normal, quantile, standardisation, sum, weighted};
use epistat_testkit::{Fingerprint, SplitMix64};

/// A fingerprint that hashes NaN and `None` alike, so a value that becomes
/// `Option` without changing leaves the hash alone.
struct Hash(Fingerprint);

impl Hash {
    fn new() -> Self {
        Self(Fingerprint::new())
    }

    fn o(&mut self, value: Option<f64>) {
        let value = value.filter(|v| !v.is_nan());
        self.0 = std::mem::take(&mut self.0).option_f64(value);
    }

    fn f(&mut self, value: f64) {
        self.o(Some(value));
    }

    fn finish(&self) -> u64 {
        self.0.finish()
    }
}

/// Probabilities across the range, the extremes the kernels clamp to and a
/// dense grid inside.
fn probabilities() -> Vec<f64> {
    let mut p = vec![0.0, f64::MIN_POSITIVE, 1e-300, 1e-12, 1e-6, 0.025, 0.5];
    p.extend((1..1_000).map(|i| f64::from(i) / 1_000.0));
    p.extend([0.975, 1.0 - 1e-6, 1.0 - 1e-12, 1.0 - f64::EPSILON, 1.0]);
    p
}

/// Positive costs with about a third zeros, weights, strata and an exposure,
/// shaped like a Study II slice.
struct Slice {
    cost: Vec<f64>,
    weight: Vec<f64>,
    stratum: Vec<u32>,
    exposed: Vec<bool>,
}

const N_STRATA: usize = 12;

fn slice() -> Slice {
    let mut stream = SplitMix64::new(20_261_126);
    let n = 5_000;
    let mut slice = Slice {
        cost: Vec::with_capacity(n),
        weight: Vec::with_capacity(n),
        stratum: Vec::with_capacity(n),
        exposed: Vec::with_capacity(n),
    };
    for _ in 0..n {
        let zero = stream.unit() < 0.35;
        let cost = if zero {
            0.0
        } else {
            (8.0 + stream.uniform(-1.5, 2.5)).exp()
        };
        slice.cost.push(cost);
        slice
            .weight
            .push(0.25 * f64::from(1 + u32::try_from(stream.below(4)).expect("small")));
        slice
            .stratum
            .push(u32::try_from(stream.below(N_STRATA as u64)).expect("small"));
        slice.exposed.push(stream.unit() < 0.2);
    }
    slice
}

fn cases() -> Vec<(&'static str, u64)> {
    let mut out = Vec::new();
    let p = probabilities();

    // kernels::interval::{normal_cdf, erf, inverse_normal_cdf}.
    let mut h = Hash::new();
    for i in -4_000..=4_000 {
        let x = f64::from(i) / 500.0;
        h.f(normal::cdf_as26(x));
        h.f(normal::erf_as26(x));
    }
    for &p in &p {
        h.f(normal::quantile_acklam(
            p.clamp(f64::MIN_POSITIVE, 1.0 - f64::EPSILON),
        ));
    }
    out.push(("normal_compat", h.finish()));

    // kernels::stats::{normal_quantile, chi_square_sf}.
    let mut h = Hash::new();
    for &p in &p {
        h.f(normal::quantile(p));
    }
    for df in 1..=12 {
        for i in 0..=400 {
            h.f(gamma::chi_square_sf(f64::from(i) / 10.0, f64::from(df)));
        }
    }
    out.push(("normal_quantile_chi_square_sf", h.finish()));

    // kernels::interval::quantile_sorted.
    let data = slice();
    let mut sorted = data.cost.clone();
    sorted.sort_by(f64::total_cmp);
    let mut h = Hash::new();
    for &p in &p {
        h.o(quantile::rounded_rank(&sorted, p.clamp(0.0, 1.0)));
    }
    out.push(("rounded_rank", h.finish()));

    // kernels::stats::{weighted_mean, weighted_variance}.
    let mut h = Hash::new();
    for weights in [None, Some(data.weight.as_slice())] {
        h.o(weighted::mean(&data.cost, weights));
        h.o(weighted::variance(&data.cost, weights));
        h.o(weighted::mean(&data.cost[..1], weights.map(|w| &w[..1])));
        h.o(weighted::variance(
            &data.cost[..1],
            weights.map(|w| &w[..1]),
        ));
    }
    h.f(f64::from(weighted::WEIGHTED_DECIMAL_PLACES));
    out.push(("weighted_moments", h.finish()));

    // kernels::stats::standardisation_columns, unit and bootstrap weights.
    let mut h = Hash::new();
    let ones = vec![1.0; data.cost.len()];
    for weights in [ones.as_slice(), data.weight.as_slice()] {
        let counts =
            standardisation::stratum_counts(&data.stratum, &data.exposed, weights, N_STRATA)
                .expect("counts");
        let standardised = standardisation::standardisation_weights(
            &data.stratum,
            &data.exposed,
            weights,
            N_STRATA,
        )
        .expect("weights");
        for [comparators, exposed] in counts {
            h.f(comparators);
            h.f(exposed);
        }
        for &w in &standardised {
            h.f(w);
        }
    }
    out.push(("standardisation", h.finish()));

    // kernels::equity: Kitagawa parts of the excess-cost gap.
    let mut h = Hash::new();
    let mut stream = SplitMix64::new(7);
    for _ in 0..200 {
        let mut draw =
            |k: usize, scale: f64| -> Vec<f64> { (0..k).map(|_| scale * stream.unit()).collect() };
        let (shares, excess, ref_shares, ref_excess) =
            (draw(6, 1.0), draw(6, 5e4), draw(6, 1.0), draw(6, 5e4));
        let total: f64 = shares.iter().sum();
        let shares: Vec<f64> = shares.iter().map(|s| s / total).collect();
        let total: f64 = ref_shares.iter().sum();
        let ref_shares: Vec<f64> = ref_shares.iter().map(|s| s / total).collect();
        let parts = decomposition::kitagawa_parts(&shares, &excess, &ref_shares, &ref_excess);
        h.o(parts.gap);
        h.o(parts.composition);
        h.o(parts.rate);
    }
    out.push(("kitagawa_parts", h.finish()));

    // kernels::numeric::{compensated_sum, exact_float_sum}.
    let mut h = Hash::new();
    for chunk in data.cost.chunks(97) {
        h.f(sum::compensated(chunk.iter().copied()));
        h.f(sum::exact(chunk.iter().copied()));
    }
    let wide = [1e16, 1.0, -1e16, 3.0, 1e-3, -2.5e15, 2.5e15];
    h.f(sum::compensated(wide));
    h.f(sum::exact(wide));
    out.push(("sums", h.finish()));
    out
}

const GOLDEN: &[(&str, u64)] = &[
    ("normal_compat", 0xe56f_cc5e_b5a8_cf1f),
    ("normal_quantile_chi_square_sf", 0x3122_08bc_1d2b_b7fb),
    ("rounded_rank", 0xa822_9567_b4df_a6f7),
    ("weighted_moments", 0x43b3_eae4_9d1c_c391),
    ("standardisation", 0xe413_b9e2_bf88_91fa),
    ("kitagawa_parts", 0x380b_80c5_92d3_9da9),
    ("sums", 0xd236_98d2_f484_6ab8),
];

#[test]
fn study_paths_are_bit_identical_to_the_recorded_hashes() {
    let current = cases();
    if std::env::var_os("STUDY_PATH_PRINT").is_some() {
        for (name, hash) in &current {
            println!(
                "    (\"{name}\", 0x{:04x}_{:04x}_{:04x}_{:04x}),",
                hash >> 48,
                (hash >> 32) & 0xffff,
                (hash >> 16) & 0xffff,
                hash & 0xffff
            );
        }
    }
    assert_eq!(current.as_slice(), GOLDEN);
}
