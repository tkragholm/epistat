//! The bits of the g-formula on simulated data.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output, with
//! its length, and they must not change. To record again, run with
//! `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The bits depend on the platform's `ln`, `exp` and `cos`, so the test runs only
//! on aarch64 macOS, where it was recorded. Thread-count independence is checked
//! everywhere by `threads_gformula`.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod gformula_sim;

use epistat_causal::gformula::{GFormulaOptions, IntervalMethod, gformula};
use epistat_testkit::Fingerprint;
use gformula_sim::{complete, covariates, flatten, interventions, simulate};

/// `(name, length, digest)` of each recorded output.
const EXPECTED: &[(&str, usize, u64)] = &[
    (
        "survival, each person once, bootstrap",
        342,
        0xc0bb_74e6_703d_3e94,
    ),
    (
        "survival, 2500 simulated, normal limits",
        342,
        0x5db0_9793_3473_51c2,
    ),
    (
        "survival, baseline lags, no truncation",
        266,
        0xcd0a_115f_675a_e634,
    ),
    ("binary end, bootstrap", 108, 0x0afb_b3ef_45cd_1aac),
];

fn outputs() -> Vec<(String, Vec<f64>)> {
    let sim = simulate(900, 4, 5);
    let baseline = sim.baseline();
    let tv = sim.time_varying();
    let data = sim.data(&baseline, &tv);
    let outcome = sim.survival();
    let mut out = Vec::new();
    for (name, options) in [
        (
            "survival, each person once, bootstrap",
            GFormulaOptions {
                bootstrap: 8,
                ..GFormulaOptions::default()
            },
        ),
        (
            "survival, 2500 simulated, normal limits",
            GFormulaOptions {
                simulated: Some(2_500),
                bootstrap: 5,
                interval: IntervalMethod::Normal,
                reference: 2,
                ..GFormulaOptions::default()
            },
        ),
        (
            "survival, baseline lags, no truncation",
            GFormulaOptions {
                baseline_lags: true,
                truncate: false,
                periods: Some(3),
                ..GFormulaOptions::default()
            },
        ),
    ] {
        let g = gformula(&data, &covariates(), &outcome, &interventions(), &options).unwrap();
        out.push((name.to_string(), flatten(&g)));
    }
    let end = complete(&sim, 4);
    let baseline = end.baseline();
    let tv = end.time_varying();
    let data = end.data(&baseline, &tv);
    let outcome = end.binary_end();
    let options = GFormulaOptions {
        bootstrap: 4,
        ..GFormulaOptions::default()
    };
    let g = gformula(&data, &covariates(), &outcome, &interventions(), &options).unwrap();
    out.push(("binary end, bootstrap".to_string(), flatten(&g)));
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
