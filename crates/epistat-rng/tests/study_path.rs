//! The draws Studies I and II take from this crate, hashed to the bit.
//!
//! Each case makes the calls of PHD's `crates/kernels` (`rng.rs`,
//! `bootstrap.rs`, `weighted.rs`, `stats.rs`, `equity.rs`) with the arguments
//! they pass: the stream primitives, the replicate seeds, the seeded id hash,
//! the cluster resampling counts and the Dirichlet weights of the Bayesian
//! bootstrap. A change that moves a hash moves a protected study, so a port of
//! these calls to new names must reproduce every hash.
//!
//! The hashes were recorded on aarch64 macOS. Print them with
//! `STUDY_PATH_PRINT=1 cargo test --test study_path -- --nocapture`.
#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use epistat_rng::{
    Stream, dirichlet_scale, dirichlet_weights, replicate_seed, resample_counts, seeded_id_hash,
    splitmix64,
};
use epistat_testkit::Fingerprint;

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

const SEED: u64 = 20_260_819;

fn cases() -> Vec<(&'static str, u64)> {
    let mut out = Vec::new();

    // kernels::rng::probe: three streams on one seed.
    let mut h = Hash::new();
    let (mut raw, mut uniform, mut below) = (
        Stream::seeded(SEED),
        Stream::seeded(SEED),
        Stream::seeded(SEED),
    );
    for _ in 0..1_000 {
        h.word(raw.next_u64());
        h.f(uniform.uniform());
        h.word(below.below(997) as u64);
    }
    out.push(("stream_primitives", h.finish()));

    // kernels::stats::corrected_draws: a choice and a normal per draw.
    let mut h = Hash::new();
    let mut stream = Stream::seeded(SEED);
    let sensitivities = [0.61, 0.72, 0.8, 0.93];
    for _ in 0..1_000 {
        h.f(*stream.choice(&sensitivities).expect("non-empty"));
        h.f(stream.normal());
    }
    let mut items: Vec<u64> = (0..257).collect();
    stream.shuffle(&mut items);
    for &item in &items {
        h.word(item);
    }
    out.push(("stream_choice_normal_shuffle", h.finish()));

    let mut h = Hash::new();
    for replicate in 0..2_000 {
        h.word(replicate_seed(SEED, replicate));
        h.word(splitmix64(replicate));
    }
    for id in ["0000000001", "1234567890", "9999999967", ""] {
        h.word(seeded_id_hash(id.as_bytes(), SEED));
    }
    out.push(("seeds", h.finish()));

    // kernels::rng::cluster_draw_counts, one replicate seed per draw.
    let mut h = Hash::new();
    for replicate in 0..50 {
        for count in resample_counts(1_303, replicate_seed(SEED, replicate)) {
            h.word(u64::from(count));
        }
    }
    out.push(("resample_counts", h.finish()));

    // kernels::bootstrap::cluster_weights and kernels::weighted's scales.
    let mut h = Hash::new();
    let mut weights = Vec::new();
    for replicate in 0..50 {
        let seed = replicate_seed(SEED, replicate);
        dirichlet_weights(1_303, seed, &mut weights);
        for &w in &weights {
            h.f(w);
        }
        h.f(dirichlet_scale(1_303, seed));
    }
    out.push(("dirichlet", h.finish()));
    out
}

const GOLDEN: &[(&str, u64)] = &[
    ("stream_primitives", 0x3e04_56b2_0edb_cbf9),
    ("stream_choice_normal_shuffle", 0xdd44_cec9_c6ab_4033),
    ("seeds", 0xafe7_2e00_9d82_1cb3),
    ("resample_counts", 0x8137_3b54_f3f9_2be5),
    ("dirichlet", 0xfc82_4f00_beff_b0f0),
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
