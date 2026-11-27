//! The bits of the overlap, matching and optimal-subset weights and of the
//! pooled logistic treatment weights with the Cox fit weighted by them.
//!
//! The recorded values are an FNV-1a digest of the `to_bits` of each output
//! field, with its length, and they must not change. To record again, run with
//! `GOLDEN_RECORD=1 -- --nocapture` and paste the lines.
//!
//! The bits depend on the platform's `ln` and `exp`, so the test runs only on
//! aarch64 macOS, where it was recorded. Thread-count independence is checked
//! everywhere by `threads_weighting_treatment`.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

mod treatment_sim;

use epistat_linalg::Mat;
use epistat_survival::SurvData;
use epistat_survival::cox::CoxOptions;
use epistat_survival::weighting::cox as wcox;
use epistat_survival::weighting::iptw::{self, Estimand, IptwOptions};
use epistat_survival::weighting::pooled::PooledOptions;
use epistat_survival::weighting::pooled::{PooledEvent, TreatmentHistory};
use epistat_survival::weighting::{Censoring, WeightAt, Weights};
use epistat_testkit::Fingerprint;

/// `(name, length, digest)` of each recorded field.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("ato/weights", 1500, 0x4733_5960_6936_36b6),
    ("ato/propensity", 1500, 0xb644_e40b_50d0_cf50),
    ("ato/alpha", 0, 0xa8c7_f832_281a_39c5),
    ("atm/weights", 1500, 0x70b1_4f4a_8161_67c4),
    ("atm/propensity", 1500, 0xb644_e40b_50d0_cf50),
    ("atm/alpha", 0, 0xa8c7_f832_281a_39c5),
    ("atos/weights", 1500, 0x4446_8e42_6d94_63f8),
    ("atos/propensity", 1500, 0xb644_e40b_50d0_cf50),
    ("atos/alpha", 1, 0x6670_3bf1_9e8d_8bf4),
    ("all/weight", 7718, 0xc601_2474_cf42_bbc8),
    ("all/weight_before", 7718, 0xc195_1c1e_af0e_2477),
    ("all/coefficients", 2, 0x6e3e_2e6a_7e18_7c3f),
    ("all/covariance", 4, 0x3021_91ef_fb37_c402),
    ("all/robust", 4, 0xf0a9_3ae4_0285_730c),
    ("all stab trunc/weight", 7718, 0x309e_1f60_5009_b6f5),
    ("all stab trunc/weight_before", 7718, 0x03ff_77a7_b61c_6035),
    ("all stab trunc/coefficients", 2, 0xedb6_56d2_9a94_db23),
    ("all stab trunc/covariance", 4, 0xf22c_5db9_ef52_f7a6),
    ("all stab trunc/robust", 4, 0x51d7_2832_70c0_1243),
    ("first stab/weight", 7718, 0x7476_7ee5_a6dd_182e),
    ("first stab/weight_before", 7718, 0xe2ec_3ff1_ab5c_be91),
    ("first stab/coefficients", 2, 0x98f5_c62d_f20d_7ea6),
    ("first stab/covariance", 4, 0xf03b_148d_8118_f69e),
    ("first stab/robust", 4, 0xb5fd_6dd1_bb14_dbb7),
];

fn fields() -> Vec<(String, Vec<f64>)> {
    let s = treatment_sim::simulate();
    let mut out = Vec::new();
    for (name, estimand) in [
        ("ato", Estimand::Ato),
        ("atm", Estimand::Atm),
        ("atos", Estimand::Atos),
    ] {
        let w = iptw::fit(
            &s.treat,
            s.z.as_ref(),
            &IptwOptions {
                estimand,
                stabilise: false,
                alpha: None,
            },
        )
        .unwrap();
        out.push((format!("{name}/weights"), w.weights.clone()));
        out.push((format!("{name}/propensity"), w.propensity.clone()));
        out.push((format!("{name}/alpha"), w.alpha.into_iter().collect()));
    }
    let rows = s.stop.len();
    let x = Mat::from_fn(rows, 2, |i, j| {
        if j == 0 {
            f64::from(u8::from(s.row_treated[i]))
        } else {
            s.design[(i, 1)]
        }
    });
    let data = SurvData::builder(&s.stop, &s.status)
        .start(&s.start)
        .id(&s.id)
        .build()
        .unwrap();
    let numerator = Mat::from_fn(rows, 2, |i, j| s.design[(i, 3 * j)]);
    for (name, history, stabilise, trunc) in [
        ("all", TreatmentHistory::All, false, None),
        (
            "all stab trunc",
            TreatmentHistory::All,
            true,
            Some((0.02, 0.98)),
        ),
        ("first stab", TreatmentHistory::First, true, None),
    ] {
        let w = epistat_survival::weighting::pooled::fit(
            &data,
            PooledEvent::Treatment {
                treated: &s.row_treated,
                history,
            },
            s.design.as_ref(),
            stabilise.then(|| numerator.as_ref()),
            &PooledOptions { truncation: trunc },
        )
        .unwrap();
        out.push((format!("{name}/weight"), w.weight.clone()));
        out.push((format!("{name}/weight_before"), w.weight_before.clone()));
        let estimated = Weights {
            iptw: None,
            subject: None,
            censoring: Some(Censoring::Pooled {
                fit: &w,
                at: WeightAt::Stop,
            }),
        };
        let cox = wcox::fit(&data, x.as_ref(), &CoxOptions::default(), &estimated).unwrap();
        out.push((format!("{name}/coefficients"), cox.coefficients.clone()));
        out.push((format!("{name}/covariance"), cox.covariance.clone()));
        out.push((format!("{name}/robust"), cox.robust_covariance.clone()));
    }
    out
}

#[test]
fn bits_are_unchanged() {
    let got: Vec<(String, usize, u64)> = fields()
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
