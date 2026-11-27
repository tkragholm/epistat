//! The bits of the augmented risk with censoring on a simulated cohort.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output, with
//! its length, and they must not change. To record again, run with
//! `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The bits depend on the platform's `ln` and `exp`, so the test runs only on
//! aarch64 macOS, where it was recorded. Thread-count independence is checked
//! everywhere by `threads_aipw_surv`.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod aipw_surv_sim;

use epistat_survival::aipw::Estimator;
use epistat_testkit::Fingerprint;

/// `(name, length, digest)` of each recorded output.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("aiptw km", 7224, 0x1e95_cb32_9231_3cb5),
    ("aiptw cox", 7224, 0xdfbb_70e3_cfa7_18ec),
    ("iptw km", 7224, 0x412f_1fcf_3b65_533b),
    ("iptw cox", 7224, 0x23a1_d66b_3867_31c9),
];

fn outputs() -> Vec<(String, Vec<f64>)> {
    let s = aipw_surv_sim::simulate(1200, 7);
    let mut out = Vec::new();
    for (name, estimator) in [("aiptw", Estimator::Aiptw), ("iptw", Estimator::Iptw)] {
        for (label, km) in [("km", true), ("cox", false)] {
            let r = aipw_surv_sim::run(&s, estimator, km, &[2.0, 5.0, 8.0]);
            out.push((format!("{name} {label}"), aipw_surv_sim::flatten(&r)));
        }
    }
    out
}

#[test]
fn bits_are_unchanged() {
    let got: Vec<(String, usize, u64)> = outputs()
        .into_iter()
        .map(|(name, v)| {
            let digest = Fingerprint::new().u64(v.len() as u64).f64s(&v).finish();
            (name, v.len(), digest)
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
