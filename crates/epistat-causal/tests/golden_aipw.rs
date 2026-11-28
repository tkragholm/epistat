//! The bits of the augmented estimators on a simulated data set.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output, with
//! its length, and they must not change. To record again, run with
//! `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The bits depend on the platform's `ln` and `exp`, so the test asserts the
//! recorded values only on aarch64 macOS, where it was recorded. Thread-count
//! independence is checked everywhere by `threads_aipw`. Elsewhere it computes
//! the hashes and, when `EPISTAT_HASH_OUT` is set, appends them to that file.

mod sim;

use epistat_causal::CaseWeights;
use epistat_causal::aipw::{AipwInput, AipwOptions, aipw, aipw_glm};
use epistat_causal::{Estimand, ProbabilityBounds};
use epistat_glm::Family;
use epistat_testkit::Fingerprint;

/// `(name, length, digest)` of each recorded output.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("glm ate", 9023, 0xe767_230d_7417_9751),
    ("glm att", 9023, 0xc823_15e5_4e7e_6512),
    ("glm atc", 9023, 0xec7a_6a9c_28d4_c69b),
    ("core truncated", 9023, 0xe14d_1353_3315_35de),
];

fn outputs() -> Vec<(String, Vec<f64>)> {
    let s = sim::simulate();
    let mut out = Vec::new();
    for (name, target) in [
        ("ate", Estimand::Ate),
        ("att", Estimand::Att),
        ("atc", Estimand::Atc),
    ] {
        let r = aipw_glm(
            &s.input(),
            Family::Binomial,
            &AipwOptions {
                estimand: target,
                ..AipwOptions::default()
            },
        )
        .unwrap();
        out.push((format!("glm {name}"), sim::flatten(&r)));
    }
    let m = s.models();
    let r = aipw(
        &AipwInput {
            outcome: &s.y,
            treatment: &s.a,
            mu1: &m.0,
            mu0: &m.1,
            propensity: &m.2,
            weight: Some(CaseWeights::new(&s.weight).expect("weights")),
            cluster: Some(&s.cluster),
        },
        &AipwOptions {
            estimand: Estimand::Ate,
            ..AipwOptions {
                truncation: Some(ProbabilityBounds::new(0.05, 0.95).unwrap()),
                ..AipwOptions::default()
            }
        },
    )
    .unwrap();
    out.push(("core truncated".to_string(), sim::flatten(&r)));
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
