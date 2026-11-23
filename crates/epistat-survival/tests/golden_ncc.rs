//! Samuelsen's inclusion probabilities, the weighted fit and the absolute risk
//! on seeded data, as an FNV-1a digest of the `to_bits` of each output with its
//! length. To record again, run with `GOLDEN_RECORD=1 -- --nocapture` and paste
//! the lines.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few thousand rows, far below the limits of the casts"
)]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::CoxOptions;
use epistat_survival::ncc::{
    NccDesign, absolute_risk, baseline, fit_weighted, inclusion_probabilities,
};
use epistat_testkit::{Fingerprint, absent};

/// `(name, length, digest)` of each recorded field.
const EXPECTED: &[(&str, usize, u64)] = &[
    ("plain/probability", 2500, 0x5ddf_b06c_d268_6456),
    ("plain/risk_set_size", 946, 0xcae6_9a90_05ea_e1af),
    ("plain/coefficients", 2, 0x4de7_f86c_1229_8ba1),
    ("plain/robust_covariance", 4, 0xce2c_9a55_ba60_c22d),
    ("plain/baseline", 946, 0x988e_7963_6902_1228),
    ("plain/risk", 6, 0xcb72_1eb8_2feb_aa3a),
    ("entry/probability", 2500, 0xb6f0_4834_0225_aaa9),
    ("entry/risk_set_size", 946, 0xf9c7_0362_8e45_5d6e),
    ("entry/coefficients", 2, 0x00bb_5b0f_40a9_4255),
    ("entry/robust_covariance", 4, 0xe7e9_7f8c_c1db_2c83),
    ("entry/baseline", 946, 0x350d_3d78_cceb_b1f9),
    ("entry/risk", 6, 0xbc91_f2b0_3c87_3e4e),
    ("matched/probability", 2500, 0x4c23_dae1_6d3c_23f0),
    ("matched/risk_set_size", 946, 0xa715_71df_f8ae_eb3d),
    ("matched/coefficients", 2, 0x3f6d_3a65_400f_277f),
    ("matched/robust_covariance", 4, 0x6e65_bc8b_6aa2_a905),
    ("matched/baseline", 946, 0x9659_eda6_64fb_4dec),
    ("matched/risk", 6, 0x3562_85bd_6836_6974),
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
fn nested_case_control_outputs_are_recorded() {
    let n = 2500;
    let mut rng = Stream::seeded(92);
    let (mut entry, mut exit, mut case, mut stratum) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for _ in 0..n {
        let x1 = rng.normal();
        let x2 = f64::from(u8::from(rng.below(3) == 0));
        let t = rng.exp1() / (0.1 * (0.5 * x1 - 0.4 * x2).exp());
        let c = rng.exp1() / 0.04;
        let e = t.min(c).min(6.0);
        entry.push(if rng.below(3) == 0 {
            rng.uniform() * 0.5 * e
        } else {
            0.0
        });
        exit.push(e);
        case.push(t <= c.min(6.0));
        stratum.push(rng.below(3) as u32);
        a.push(x1);
        b.push(x2);
    }
    let sampled: Vec<bool> = (0..n).map(|i| case[i] || rng.below(4) == 0).collect();
    for (name, with_entry, with_strata) in [
        ("plain", false, false),
        ("entry", true, false),
        ("matched", true, true),
    ] {
        let design = NccDesign {
            entry: with_entry.then_some(entry.as_slice()),
            exit: &exit,
            case: &case,
            controls_per_case: 2,
            stratum: with_strata.then_some(stratum.as_slice()),
        };
        let inc = inclusion_probabilities(&design).unwrap();
        check(&format!("{name}/probability"), &inc.probability);
        check(
            &format!("{name}/risk_set_size"),
            &inc.risk_set_size
                .iter()
                .map(|&s| s as f64)
                .collect::<Vec<_>>(),
        );
        let rows: Vec<usize> = (0..n)
            .filter(|&i| sampled[i] && (case[i] || inc.probability[i] > 0.0))
            .collect();
        let w: Vec<f64> = rows
            .iter()
            .map(|&i| {
                if case[i] {
                    1.0
                } else {
                    1.0 / inc.probability[i]
                }
            })
            .collect();
        let stop: Vec<f64> = rows.iter().map(|&i| exit[i]).collect();
        let start: Vec<f64> = rows.iter().map(|&i| entry[i]).collect();
        let status: Vec<i32> = rows.iter().map(|&i| i32::from(case[i])).collect();
        let x = Mat::from_fn(
            rows.len(),
            2,
            |i, j| if j == 0 { a[rows[i]] } else { b[rows[i]] },
        );
        let data = SurvData::builder(&stop, &status)
            .start(&start)
            .build()
            .unwrap();
        let opts = CoxOptions::default();
        let model = fit_weighted(&data, x.as_ref(), &w, &opts).unwrap();
        let flat = |v: &[Option<f64>]| v.iter().copied().map(absent).collect::<Vec<f64>>();
        check(&format!("{name}/coefficients"), &flat(&model.coefficients));
        check(
            &format!("{name}/robust_covariance"),
            &flat(model.robust_covariance.as_ref().unwrap()),
        );
        let base = baseline(&model, &data, x.as_ref(), &w, &opts).unwrap();
        check(&format!("{name}/baseline"), &base.cumulative_hazard);
        let profile = Mat::from_fn(2, 2, |i, j| [[0.0, 0.0], [1.0, 1.0]][i][j]);
        let risk = absolute_risk(
            &base,
            &model.coefficients,
            profile.as_ref(),
            &[1.0, 3.0, 5.0],
        )
        .unwrap();
        let flat_risk: Vec<f64> = risk.iter().flatten().copied().map(absent).collect();
        check(&format!("{name}/risk"), &flat_risk);
    }
}
