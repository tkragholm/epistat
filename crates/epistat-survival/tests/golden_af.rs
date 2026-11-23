//! The bits of the attributable fraction function of a Cox model on seeded
//! data, as an FNV-1a digest of the `to_bits` of each output with its length.
//! To record again, run with `GOLDEN_RECORD=1 -- --nocapture` and paste the
//! lines.
//!
//! The bits depend on the platform's `ln` and `exp`, so the test runs only on
//! the target it was recorded on, aarch64 macOS. Thread-count independence is
//! checked everywhere by `threads_af`.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use epistat_glm::attributable::IntervalScale;
use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, Ties, fit};
use epistat_survival::standardise::attributable::af_from_standardised;
use epistat_survival::standardise::{Counterfactual, Level, StandardiseOptions, standardise_cox};
use epistat_testkit::{Fingerprint, absent_all};

/// `(name, length, digest)` of each recorded field.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("identity/af", 4, 0xffb1_abe1_b241_f6e5),
    ("identity/se", 4, 0xfd05_f330_6279_2adf),
    ("identity/lower", 4, 0xc64b_80a3_52e5_a4a3),
    ("identity/upper", 4, 0x96a6_eb09_2184_ef1f),
    ("identity/ratio", 4, 0xd2df_1c21_a4a2_cbba),
    ("log/af", 4, 0xffb1_abe1_b241_f6e5),
    ("log/se", 4, 0xfd05_f330_6279_2adf),
    ("log/lower", 4, 0xcb22_5086_7e06_d4d3),
    ("log/upper", 4, 0x73ed_e32c_96a3_56be),
    ("log/ratio", 4, 0xd2df_1c21_a4a2_cbba),
];

#[track_caller]
fn check(name: &str, values: &[f64]) {
    let got = (values.len(), Fingerprint::new().f64s(values).finish());
    if std::env::var_os("GOLDEN_RECORD").is_some() {
        let hex = format!("{:016x}", got.1);
        let hex: Vec<&str> = (0..4).map(|i| &hex[4 * i..4 * i + 4]).collect();
        println!("    (\"{name}\", {}, 0x{}),", got.0, hex.join("_"));
        return;
    }
    let want = EXPECTED
        .iter()
        .find(|entry| entry.0 == name)
        .unwrap_or_else(|| panic!("{name} has no recorded value"));
    assert_eq!(
        got,
        (want.1, want.2),
        "{name}: the bits of the output changed"
    );
}

#[test]
fn attributable_fraction_bits_are_recorded() {
    let mut rng = Stream::seeded(67);
    let n = 500_usize;
    let z: Vec<f64> = (0..n)
        .map(|_| {
            let (a, b) = (rng.uniform(), rng.uniform());
            (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
        })
        .collect();
    let e: Vec<f64> = (0..n)
        .map(|i| f64::from(u8::from(rng.uniform() < 1.0 / (1.0 + (-0.4 * z[i]).exp()))))
        .collect();
    let time: Vec<f64> = (0..n)
        .map(|i| {
            (rng.exp1() / (0.1 * (0.6 * e[i] + 0.4 * z[i]).exp())).min(30.0 * rng.uniform() + 3.0)
        })
        .collect();
    let status: Vec<i32> = (0..n).map(|_| i32::from(rng.uniform() < 0.7)).collect();
    let x = Mat::from_fn(n, 3, |i, j| [e[i], z[i], e[i] * z[i]][j]);
    let x0 = Mat::from_fn(n, 3, |i, j| if j == 1 { x[(i, j)] } else { 0.0 });
    let data = SurvData::builder(&time, &status).build().unwrap();
    let options = CoxOptions {
        ties: Ties::Breslow,
        ..CoxOptions::default()
    };
    let model = fit(&data, x.as_ref(), &options).unwrap();
    let times = [1.0, 2.0, 4.0, 8.0];
    let s = standardise_cox(
        &data,
        x.as_ref(),
        &model,
        &options,
        &[
            Level::single(Counterfactual::Design(x0.as_ref())),
            Level::single(Counterfactual::Columns(&[])),
        ],
        &times,
        &StandardiseOptions {
            std_err: true,
            ..StandardiseOptions::default()
        },
    )
    .unwrap();
    for (name, scale) in [
        ("identity", IntervalScale::Identity),
        ("log", IntervalScale::LogComplement),
    ] {
        let c = af_from_standardised(&s, &times, 0, 1, 0.95, scale).unwrap();
        for (field, v) in [
            ("af", &c.af),
            ("se", &c.se),
            ("lower", &c.lower),
            ("upper", &c.upper),
            ("ratio", &c.ratio),
        ] {
            check(&format!("{name}/{field}"), &absent_all(v));
        }
    }
}
