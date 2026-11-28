//! The calls Studies I and II make into this crate, hashed to the bit.
//!
//! Study II's bootstrap intervals are `kernels::interval::{percentile_ci,
//! trimmed_percentile_ci, bias_corrected_ci}` in PHD: type 7 percentiles at
//! `alpha = 0.05`, the trimmed interval at the cost models' default trim of
//! 0.025, and the bias-corrected interval. Study I's simultaneous window bands
//! are `kernels::did::curve::simultaneous_bands`: `multiplier_bands` on
//! cluster-level influence rows, no clusters, `Level::alpha(1 - 0.95)`,
//! 1,000 replicates and seed 17,431. Study II passes its `alpha` as
//! `Level::alpha(alpha)`. A change that moves a hash moves a protected
//! study, except where a decision says otherwise (decision 104 moves Study I's
//! bands at the next bump in PHD), so a port of these calls to new signatures
//! must reproduce every hash.
//!
//! The hashes were recorded on aarch64 macOS and are asserted only there.
//! Elsewhere the test computes them and, when `EPISTAT_HASH_OUT` is set,
//! appends them to that file. Print them with
//! `STUDY_PATH_PRINT=1 cargo test --test study_path -- --nocapture`.

use epistat_core::{Bootstrap, Interval, Level};
use epistat_inference::interval::{self, Percentile};
use epistat_inference::simultaneous::multiplier_bands;
use epistat_linalg::MatRef;
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

    fn pair(&mut self, interval: Option<Interval>) {
        self.o(interval.and_then(|i| i.lower));
        self.o(interval.and_then(|i| i.upper));
    }

    fn finish(&self) -> u64 {
        self.0.finish()
    }
}

const ALPHA: f64 = 0.05;
const TRIM: f64 = 0.025;
/// `1.0 - confidence_level` as Study I computes it, which is not 0.05.
const BAND_ALPHA: f64 = 1.0 - 0.95;

/// Bootstrap replicates of a marginal cost difference: a skewed bulk, a few
/// far draws, and the sizes Study II runs (1,000) and refuses below (100).
fn replicates(n: usize, seed: u64) -> Vec<f64> {
    let mut stream = SplitMix64::new(seed);
    (0..n)
        .map(|i| {
            let bulk = 1_500.0 + 400.0 * (stream.unit() - 0.5) + 300.0 * stream.unit().powi(3);
            if i % 211 == 0 { bulk * 6.0 } else { bulk }
        })
        .collect()
}

fn cases() -> Vec<(&'static str, u64)> {
    let mut out = Vec::new();

    let alpha = Level::alpha(ALPHA).expect("alpha");
    let mut h = Hash::new();
    for (n, seed) in [(1_000, 1), (100, 2), (999, 3), (1_000, 4)] {
        let draws = replicates(n, seed);
        let point = draws[0] * 0.98;
        h.pair(interval::percentile(&mut draws.clone(), alpha, Percentile::Type7).ok());
        h.pair(
            interval::trimmed_percentile(&mut draws.clone(), alpha, TRIM, Percentile::Type7).ok(),
        );
        h.pair(
            interval::trimmed_percentile(&mut draws.clone(), alpha, 0.0, Percentile::Type7).ok(),
        );
        h.pair(interval::bias_corrected(&mut draws.clone(), point, alpha).ok());
    }
    out.push(("bootstrap_intervals", h.finish()));

    // One row per window aggregate, one column per family cluster, as
    // `simultaneous_bands` slices them out of the flat row-major influence.
    let mut stream = SplitMix64::new(17);
    let (n_aggregates, n_clusters) = (6, 2_500);
    let flat: Vec<f64> = (0..n_aggregates * n_clusters)
        .map(|i| {
            let aggregate = i / n_clusters;
            let scale = 1.0 + 0.25 * f64::from(u32::try_from(aggregate).expect("small"));
            scale * (stream.unit() - 0.5) * 1e-3
        })
        .collect();
    // Row-major aggregates by clusters is column-major clusters by aggregates.
    let influence = MatRef::from_column_major_slice(&flat, n_clusters, n_aggregates);
    let level = Level::alpha(BAND_ALPHA).expect("alpha");
    let bands =
        multiplier_bands(influence, None, level, Bootstrap::new(1_000, 17_431)).expect("bands");
    let mut h = Hash::new();
    h.f(bands.critical);
    for &se in &bands.se {
        h.o(se);
    }
    h.f(bands.at_least_pointwise());
    out.push(("multiplier_bands", h.finish()));
    out
}

const GOLDEN: &[(&str, u64)] = &[
    ("bootstrap_intervals", 0x1b01_c95d_6dfa_981f),
    ("multiplier_bands", 0xdbac_2117_e886_d9d0),
];

#[test]
fn study_paths_are_bit_identical_to_the_recorded_hashes() {
    let current = cases();
    for (name, hash) in &current {
        epistat_testkit::hash_out(name, *hash);
    }
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
    if epistat_testkit::RECORDED_PLATFORM {
        assert_eq!(current.as_slice(), GOLDEN);
    }
}
