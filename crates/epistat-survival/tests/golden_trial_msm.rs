//! The bits of the weights and the outcome model of the sequential trials on the
//! simulated data of `data/trial_msm.R`.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output, with
//! its length, and they must not change. To record again, run with
//! `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The bits depend on the platform's `ln` and `exp`, so the test asserts the
//! recorded values only on aarch64 macOS, where it was recorded. Thread-count
//! independence is checked everywhere by `threads_trial_msm`. Elsewhere it
//! computes the hashes and, when `EPISTAT_HASH_OUT` is set, appends them to
//! that file.

mod trial_msm_run;

use epistat_testkit::Fingerprint;

/// `(name, length, digest)` of each recorded output.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("switching", 11047, 0x2216_255e_8d39_8958),
    ("with censoring", 15743, 0x1351_4c75_c27b_8997),
];

#[test]
fn bits_are_unchanged() {
    let got: Vec<(String, usize, u64)> = trial_msm_run::load()
        .iter()
        .map(|a| {
            let v = trial_msm_run::run(a).flatten();
            let digest = Fingerprint::new().u64(v.len() as u64).f64s(&v).finish();
            (
                if a.use_censor {
                    "with censoring".to_string()
                } else {
                    "switching".to_string()
                },
                v.len(),
                digest,
            )
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
