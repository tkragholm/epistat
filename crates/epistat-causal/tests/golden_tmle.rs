//! The bits of the targeted estimators on a simulated data set.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output, with
//! its length, and they must not change. To record again, run with
//! `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The bits depend on the platform's `ln` and `exp`, so the test asserts the
//! recorded values only on aarch64 macOS, where it was recorded. Thread-count
//! independence is checked everywhere by `threads_tmle`. Elsewhere it computes
//! the hashes and, when `EPISTAT_HASH_OUT` is set, appends them to that file.

#[expect(
    dead_code,
    reason = "the augmented estimators' flatten is not used here"
)]
mod sim;
mod tmle_sim;

use epistat_testkit::Fingerprint;

/// `(name, length, digest)` of each recorded output.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("glm ate pooled", 15032, 0x99a6_4aef_cbf2_ef34),
    ("glm att pooled", 15019, 0xa26a_9002_f983_b7c7),
    ("glm atc by arm", 15013, 0xfede_d7af_c4dc_bd2a),
    ("core ate", 15032, 0x534e_6945_fc8b_a687),
    ("core att", 15019, 0xa652_62ab_acb3_9f92),
    ("survival horizon 3", 9033, 0x072b_73be_9352_a0f3),
    ("survival horizon 5", 9032, 0x1e90_f3cc_ea3b_6ad4),
];

#[test]
fn bits_are_unchanged() {
    let s = sim::simulate();
    let mut outputs: Vec<(String, Vec<f64>)> = tmle_sim::CASES
        .iter()
        .map(|&case| (case.0.to_string(), tmle_sim::run(&s, case)))
        .collect();
    for horizon in [3, 5] {
        outputs.push((
            format!("survival horizon {horizon}"),
            tmle_sim::run_survival(&s, horizon),
        ));
    }
    let got: Vec<(String, usize, u64)> = outputs
        .into_iter()
        .map(|(name, v)| {
            let digest = Fingerprint::new().u64(v.len() as u64).f64s(&v).finish();
            (name, v.len(), digest)
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
    assert_eq!(got.len(), EXPECTED.len());
    for ((name, len, digest), (en, el, ed)) in got.iter().zip(EXPECTED) {
        assert_eq!(name, en);
        assert_eq!(len, el, "{name}: length");
        assert_eq!(digest, ed, "{name}: digest {digest:#018x}");
    }
}
