//! The bits of the weights and the outcome model of the sequential trials on the
//! simulated data of `data/trial_msm.R`.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output, with
//! its length, and they must not change. To record again, run with
//! `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The bits depend on the platform's `ln` and `exp`, so the test runs only on
//! aarch64 macOS, where it was recorded. Thread-count independence is checked
//! everywhere by `threads_trial_msm`.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod trial_msm_run;

use epistat_testkit::Fingerprint;

/// `(name, length, digest)` of each recorded output.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("switching", 11047, 0xf373_58ff_cda2_d9be),
    ("with censoring", 15743, 0xe7fb_c2d2_9fa7_2ba5),
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
    if std::env::var_os("GOLDEN_RECORD").is_some() {
        for (name, len, digest) in &got {
            println!("    (\"{name}\", {len}, {digest:#018x}),");
        }
        return;
    }
    let expected: Vec<(String, usize, u64)> = EXPECTED
        .iter()
        .map(|&(n, l, d)| (n.to_string(), l, d))
        .collect();
    assert_eq!(got, expected);
}
