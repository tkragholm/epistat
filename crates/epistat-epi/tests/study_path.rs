//! The calls Study II makes into this crate, hashed to the bit.
//!
//! Each case makes the calls of PHD's `crates/kernels` (`stats.rs`,
//! `equity.rs`) with the arguments they pass: direct standardisation and the
//! Kitagawa decomposition. The hashes are those epistat-stats recorded for
//! the same calls before the modules moved here; a change that moves a hash
//! moves a protected study.
//!
//! The hashes were recorded on aarch64 macOS. Print them with
//! `STUDY_PATH_PRINT=1 cargo test --test study_path -- --nocapture`.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use epistat_epi::{decomposition, standardisation};
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
    let data = slice();

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

    out
}

const GOLDEN: &[(&str, u64)] = &[
    ("standardisation", 0xe413_b9e2_bf88_91fa),
    ("kitagawa_parts", 0x380b_80c5_92d3_9da9),
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
