//! The bits of entropy balancing and CBPS on a simulated data set.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output, with
//! its length, and they must not change. To record again, run with
//! `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The bits depend on the platform's `ln` and `exp`, so the test runs only on
//! aarch64 macOS, where it was recorded. Thread-count independence is checked
//! everywhere by `threads_balance`.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod balance_sim;

use epistat_causal::aipw::Target;
use epistat_causal::balance::{
    CbpsInput, CbpsOptions, EntropyInput, EntropyOptions, Identification, balance_columns, cbps,
    entropy_balance,
};
use epistat_testkit::Fingerprint;

/// `(name, length, digest)` of each recorded output.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("entropy att", 2558, 0x31cebc0dd1199adc),
    ("entropy ate", 2566, 0x1067f5e4fc15e8ba),
    ("entropy atc", 2558, 0xcf0a69c1e418c89a),
    ("cbps ate exact", 5056, 0x95eb202f6d1a4647),
    ("cbps att over", 5056, 0xb9a5a8ea1adc3feb),
    ("cbps atc over updated", 5056, 0x2001a1fd3c1b2bfc),
];

fn outputs() -> Vec<(String, Vec<f64>)> {
    let s = balance_sim::simulate(2500);
    let cols = balance_columns(s.c.as_ref(), 2, true).unwrap();
    let mut out = Vec::new();
    for (name, target) in [
        ("att", Target::Att),
        ("ate", Target::Ate),
        ("atc", Target::Atc),
    ] {
        let r = entropy_balance(
            &EntropyInput {
                treatment: &s.a,
                covariates: cols.as_ref(),
                sample_weight: Some(&s.s),
                base_weight: Some(&s.q),
            },
            &EntropyOptions {
                target,
                ..EntropyOptions::default()
            },
        )
        .unwrap();
        let mut v = balance_sim::flatten(&r.weights, &s);
        v.extend(r.multipliers.iter().flatten());
        out.push((format!("entropy {name}"), v));
    }
    for (name, target, identification, two_step) in [
        ("ate exact", Target::Ate, Identification::Exact, true),
        ("att over", Target::Att, Identification::Over, true),
        ("atc over updated", Target::Atc, Identification::Over, false),
    ] {
        let r = cbps(
            &CbpsInput {
                treatment: &s.a,
                design: s.x.as_ref(),
                sample_weight: Some(&s.s),
            },
            &CbpsOptions {
                target,
                identification,
                two_step,
                ..CbpsOptions::default()
            },
        )
        .unwrap();
        let mut v = balance_sim::flatten(&r.weights, &s);
        v.extend(&r.coefficients);
        v.extend(&r.propensity);
        v.extend([r.objective.unwrap_or(f64::NAN), r.max_imbalance]);
        out.push((format!("cbps {name}"), v));
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
    assert_eq!(got.len(), EXPECTED.len());
    for ((name, len, digest), (en, el, ed)) in got.iter().zip(EXPECTED) {
        assert_eq!(name, en);
        assert_eq!(len, el, "{name}: length");
        assert_eq!(digest, ed, "{name}: digest {digest:#018x}");
    }
}
