//! The bits of the targeted estimators on a simulated data set.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output, with
//! its length, and they must not change. To record again, run with
//! `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The bits depend on the platform's `ln` and `exp`, so the test runs only on
//! aarch64 macOS, where it was recorded. Thread-count independence is checked
//! everywhere by `threads_tmle`.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

#[expect(
    dead_code,
    reason = "the augmented estimators' flatten is not used here"
)]
mod sim;
mod tmle_sim;

use epistat_testkit::Fingerprint;

/// `(name, length, digest)` of each recorded output.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("glm ate pooled", 9029, 0xdae9_a385_b984_d875),
    ("glm att pooled", 9028, 0x89af_51d8_1d6c_a707),
    ("glm atc by arm", 9028, 0x3e1b_6330_252d_af3a),
    ("core ate", 9029, 0x7696_aa07_a432_de04),
    ("core att", 9028, 0x7231_f2fe_0ded_fdd2),
];

#[test]
fn bits_are_unchanged() {
    let s = sim::simulate();
    let got: Vec<(String, usize, u64)> = tmle_sim::CASES
        .iter()
        .map(|&case| {
            let v = tmle_sim::run(&s, case);
            let digest = Fingerprint::new().u64(v.len() as u64).f64s(&v).finish();
            (case.0.to_string(), v.len(), digest)
        })
        .collect();
    if std::env::var_os("GOLDEN_RECORD").is_some() {
        for (name, len, digest) in &got {
            println!("    (\"{name}\", {len}, {digest:#018x}),");
        }
        return;
    }
    assert_eq!(got.len(), EXPECTED.len());
    for ((name, len, digest), (en, el, ed)) in got.iter().zip(EXPECTED) {
        assert_eq!(name, en);
        assert_eq!(len, el, "{name}: length");
        assert_eq!(digest, ed, "{name}: digest {digest:#018x}");
    }
}
