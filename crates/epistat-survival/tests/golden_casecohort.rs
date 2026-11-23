//! The coefficients and covariance matrices of the five case-cohort methods on
//! seeded data, as an FNV-1a digest of the `to_bits` of each output with its
//! length. To record again, run with `GOLDEN_RECORD=1 -- --nocapture` and paste
//! the lines.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::casecohort::{CaseCohortMethod, CaseCohortOptions, fit};
use epistat_testkit::{Fingerprint, absent};

/// `(name, length, digest)` of each recorded field.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("prentice/coefficients", 2, 0x8f84_324a_64f6_2273),
    ("prentice/covariance", 4, 0xd094_10bd_f5e5_fbe7),
    ("prentice/phase2", 4, 0x68fb_536b_37ae_5331),
    ("self_prentice/coefficients", 2, 0x0025_5285_ecde_35ec),
    ("self_prentice/covariance", 4, 0x3ead_b187_ab13_8991),
    ("self_prentice/phase2", 4, 0xd84d_59fc_424c_b36a),
    ("lin_ying/coefficients", 2, 0xe91d_bda6_0a81_5f42),
    ("lin_ying/covariance", 4, 0xbb9c_e983_1339_8606),
    ("lin_ying/phase2", 4, 0x7b9c_e443_1458_4caf),
    ("borgan_i/coefficients", 2, 0x6c86_904f_e747_452c),
    ("borgan_i/covariance", 4, 0x8263_f391_e250_f167),
    ("borgan_i/phase2", 4, 0xa595_b0f6_6543_152c),
    ("borgan_i/allocation", 6, 0xece7_0111_2120_8d6f),
    ("borgan_ii/coefficients", 2, 0x96da_fd65_d802_a060),
    ("borgan_ii/covariance", 4, 0xe5bd_45e6_3db0_32d1),
    ("borgan_ii/phase2", 4, 0x340f_7cec_d2a9_b155),
    ("borgan_ii/allocation", 6, 0xbb5e_2d55_8abb_545a),
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
    assert_eq!(got, (want.1, want.2), "{name}: the output changed");
}

#[test]
fn case_cohort_outputs_are_recorded() {
    let cohort = 1800_usize;
    let mut rng = Stream::seeded(78);
    let (mut stop, mut start, mut status, mut sub, mut stratum) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut a, mut b) = (Vec::new(), Vec::new());
    let mut sizes = [0.0_f64; 3];
    for _ in 0..cohort {
        let x1 = rng.normal();
        let x2 = f64::from(u8::from(rng.below(3) == 0));
        let t = (rng.exp1() * (-0.5 * x1 + 0.3 * x2).exp() * 8.0).ceil() / 10.0 + 0.1;
        let c = (rng.exp1() * 12.0).ceil() / 10.0 + 0.1;
        let s = rng.below(3);
        sizes[s] += 1.0;
        let in_sub = rng.uniform() < 0.25;
        let event = t <= c;
        if event || in_sub {
            stop.push(t.min(c));
            start.push((rng.uniform() * t.min(c) * 8.0).floor() / 10.0 * 0.9);
            status.push(i32::from(event));
            sub.push(in_sub);
            stratum.push(s as u32);
            a.push(x1);
            b.push(x2);
        }
    }
    let rows = stop.len();
    let x = Mat::from_fn(rows, 2, |i, j| if j == 0 { a[i] } else { b[i] });
    let data = SurvData::builder(&stop, &status)
        .start(&start)
        .build()
        .unwrap();
    for (name, method) in [
        ("prentice", CaseCohortMethod::Prentice),
        ("self_prentice", CaseCohortMethod::SelfPrentice),
        ("lin_ying", CaseCohortMethod::LinYing),
        ("borgan_i", CaseCohortMethod::BorganI),
        ("borgan_ii", CaseCohortMethod::BorganII),
    ] {
        let total = [cohort as f64];
        let stratified = matches!(
            method,
            CaseCohortMethod::BorganI | CaseCohortMethod::BorganII
        );
        let cohort_size: &[f64] = if stratified { &sizes } else { &total };
        let strata = stratified.then_some(stratum.as_slice());
        let model = fit(
            &data,
            x.as_ref(),
            &sub,
            strata,
            &CaseCohortOptions::new(method, cohort_size),
        )
        .unwrap();
        let flat = |v: &[Option<f64>]| v.iter().copied().map(absent).collect::<Vec<f64>>();
        check(&format!("{name}/coefficients"), &flat(&model.coefficients));
        check(&format!("{name}/covariance"), &flat(&model.covariance));
        check(
            &format!("{name}/phase2"),
            &flat(&model.phase_two_covariance),
        );
        if let Some(opt) = &model.optimal_allocation {
            check(&format!("{name}/allocation"), &flat(opt));
        }
    }
}
