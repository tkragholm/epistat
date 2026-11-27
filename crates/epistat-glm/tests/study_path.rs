//! The call Study II makes into this crate, hashed to the bit.
//!
//! Study II's one-part gamma model reports the modified Park test of its fit
//! through `models::cost_model::park::modified_park_test` in PHD: the fitted
//! means `exp(x' beta)` of a gamma cost model, the outcome and the analysis
//! weights or none, into `park::modified_park_test`. A change that moves a
//! hash moves a protected study, so a port of the call to new signatures must
//! reproduce every hash.
//!
//! The hashes were recorded on aarch64 macOS and are asserted only there.
//! Elsewhere the test computes them and, when `EPISTAT_HASH_OUT` is set,
//! appends them to that file. Print them with
//! `STUDY_PATH_PRINT=1 cargo test --test study_path -- --nocapture`.

use epistat_glm::park::modified_park_test;
use epistat_testkit::{Fingerprint, SplitMix64};

/// A fingerprint that hashes NaN and `None` alike, so a value that becomes
/// `Option` without changing leaves the hash alone.
struct Hash(Fingerprint);

impl Hash {
    fn new() -> Self {
        Self(Fingerprint::new())
    }

    fn word(&mut self, value: u64) {
        self.0 = std::mem::take(&mut self.0).u64(value);
    }

    fn f(&mut self, value: f64) {
        let value = (!value.is_nan()).then_some(value);
        self.0 = std::mem::take(&mut self.0).option_f64(value);
    }

    fn finish(&self) -> u64 {
        self.0.finish()
    }
}

/// Positive costs with a variance about proportional to the squared mean, the
/// fitted means of a log-linear model on three covariates, and weights of a
/// quarter to one, as Study II's match weights are.
fn slice(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut stream = SplitMix64::new(seed);
    let beta = [7.9, 0.35, -0.2, 0.15];
    let mut fitted = Vec::with_capacity(n);
    let mut outcome = Vec::with_capacity(n);
    let mut weight = Vec::with_capacity(n);
    for _ in 0..n {
        let x = [
            1.0,
            f64::from(stream.unit() < 0.3),
            stream.uniform(-1.0, 1.0),
            stream.uniform(0.0, 2.0),
        ];
        let eta: f64 = (0..4).map(|j| x[j] * beta[j]).sum();
        let mean = eta.exp();
        // A gamma draw of shape 2 as the sum of two unit exponentials.
        let draw = -(1.0 - stream.unit()).ln() - (1.0 - stream.unit()).ln();
        fitted.push(mean);
        outcome.push(mean * draw / 2.0);
        weight.push(0.25 * f64::from(1 + u32::try_from(stream.below(4)).expect("small")));
    }
    (fitted, outcome, weight)
}

fn cases() -> Vec<(&'static str, u64)> {
    let mut out = Vec::new();
    for (name, n, seed) in [("park/2000", 2_000, 1), ("park/40", 40, 2)] {
        let (fitted, outcome, weight) = slice(n, seed);
        let mut h = Hash::new();
        for weights in [None, Some(weight.as_slice())] {
            let test = modified_park_test(&fitted, &outcome, weights)
                .expect("a fit")
                .expect("a test");
            h.f(test.intercept);
            h.f(test.power);
            h.f(test.standard_error);
            h.word(test.rows as u64);
            h.word(u64::from(test.converged));
            h.word(test.iterations as u64);
        }
        out.push((name, h.finish()));
    }
    out
}

const GOLDEN: &[(&str, u64)] = &[
    ("park/2000", 0xb880_bc78_2197_f7c7),
    ("park/40", 0x6491_483e_d2bc_dbb8),
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
