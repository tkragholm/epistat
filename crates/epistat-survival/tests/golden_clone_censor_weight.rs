//! The bits of the clone-censor-weight analysis and of one bootstrap replicate on
//! the simulated data of `data/clone_censor_weight.R`.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output, with
//! its length, and they must not change. To record again, run with
//! `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The bits depend on the platform's `ln` and `exp`, so the test asserts the
//! recorded values only on aarch64 macOS, where it was recorded. Thread-count
//! independence is checked everywhere by `threads_clone_censor_weight`.
//! Elsewhere it computes the hashes and, when `EPISTAT_HASH_OUT` is set,
//! appends them to that file.

mod ccw_run;

use epistat_survival::Bootstrap;
use epistat_survival::weighting::trials::PersonCounts;
use epistat_testkit::Fingerprint;

/// The root seed whose replicate 0 draws from seed 99,
/// `replicate_seed(ROOT, 0) == 99`, the seed the recorded replicate was drawn
/// from. `replicate_seed` is `splitmix64(base ^ r * GAMMA)` and `splitmix64`
/// is a bijection, so the root is its inverse at 99.
const ROOT: u64 = 0xc7da_422f_dc3c_0db5;

/// `(name, length, digest)` of each recorded output.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("analysis", 31826, 0xe02c_ae66_bbf1_bb5d),
    ("replicate", 31826, 0xaf4d_a8da_b393_9028),
];

#[test]
fn the_root_seed_draws_the_recorded_replicate() {
    assert_eq!(epistat_rng::replicate_seed(ROOT, 0), 99);
}

#[test]
fn bits_are_unchanged() {
    let r = ccw_run::load();
    let d = ccw_run::data(&r);
    let c = ccw_run::cloned(&r, &d);
    let counts = c
        .case_weights(&PersonCounts::draw(&r.input.id, Bootstrap::new(1, ROOT), 0))
        .unwrap();
    let got: Vec<(String, usize, u64)> = [
        ("analysis", ccw_run::analyse(&r, &c, None).flatten()),
        (
            "replicate",
            ccw_run::analyse(&r, &c, Some(&counts)).flatten(),
        ),
    ]
    .into_iter()
    .map(|(name, v)| {
        let digest = Fingerprint::new().u64(v.len() as u64).f64s(&v).finish();
        (name.to_string(), v.len(), digest)
    })
    .collect();
    for (name, _, digest) in &got {
        epistat_testkit::hash_out(name, *digest);
    }
    if std::env::var_os("GOLDEN_RECORD").is_some() {
        for (name, len, digest) in &got {
            println!("    (\"{name}\", {len}, {digest:#018x}),");
        }
        return;
    }
    if !epistat_testkit::RECORDED_PLATFORM {
        return;
    }
    let expected: Vec<(String, usize, u64)> = EXPECTED
        .iter()
        .map(|&(n, l, d)| (n.to_string(), l, d))
        .collect();
    assert_eq!(got, expected);
}
