//! The bits of entropy balancing and CBPS on a simulated data set.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output, with
//! its length, and they must not change. To record again, run with
//! `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The bits depend on the platform's `ln` and `exp`, so the test asserts the
//! recorded values only on aarch64 macOS, where it was recorded. Thread-count
//! independence is checked everywhere by `threads_balance`. Elsewhere it
//! computes the hashes and, when `EPISTAT_HASH_OUT` is set, appends them to
//! that file.

mod balance_sim;

use epistat_causal::Estimand;
use epistat_causal::balance::{
    CbpsInput, CbpsOptions, EntropyInput, EntropyOptions, Identification, balance_columns, cbps,
    entropy_balance,
};
use epistat_testkit::Fingerprint;

/// `(name, length, digest)` of each recorded output.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("entropy att", 2558, 0x31ce_bc0d_d119_9adc),
    ("entropy ate", 2566, 0x1067_f5e4_fc15_e8ba),
    ("entropy atc", 2558, 0xcf0a_69c1_e418_c89a),
    ("cbps ate exact", 5056, 0x95eb_202f_6d1a_4647),
    ("cbps att over", 5056, 0xb9a5_a8ea_1adc_3feb),
    ("cbps atc over updated", 5056, 0x24ec_7d84_0ff9_8a9b),
];

fn outputs() -> Vec<(String, Vec<f64>)> {
    let s = balance_sim::simulate(2500);
    let cols = balance_columns(s.c.as_ref(), 2, true).unwrap();
    let mut out = Vec::new();
    for (name, target) in [
        ("att", Estimand::Att),
        ("ate", Estimand::Ate),
        ("atc", Estimand::Atc),
    ] {
        let r = entropy_balance(
            &EntropyInput {
                treatment: &s.a,
                covariates: cols.as_ref(),
                sample_weight: Some(&s.s),
                base_weight: Some(&s.q),
            },
            &EntropyOptions {
                estimand: target,
                ..EntropyOptions::default()
            },
        )
        .unwrap();
        let mut v = balance_sim::flatten(&r.weights, &s);
        v.extend(r.multipliers.iter().flatten());
        out.push((format!("entropy {name}"), v));
    }
    for (name, target, identification, two_step) in [
        ("ate exact", Estimand::Ate, Identification::Exact, true),
        ("att over", Estimand::Att, Identification::Over, true),
        (
            "atc over updated",
            Estimand::Atc,
            Identification::Over,
            false,
        ),
    ] {
        let r = cbps(
            &CbpsInput {
                treatment: &s.a,
                design: s.x.as_ref(),
                sample_weight: Some(&s.s),
            },
            &CbpsOptions {
                estimand: target,
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
