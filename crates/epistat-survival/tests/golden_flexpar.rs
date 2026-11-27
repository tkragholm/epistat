//! The bits of `flexpar::fit`, `predict` and `contrast` on varied inputs, pinned
//! by a hash of every output field, so that a change to any last bit is a failure
//! here. The bits depend on the platform's `ln` and `exp`, which come from its
//! maths library, so the test runs only on the target it was recorded on,
//! aarch64 macOS. Thread-count independence is checked everywhere by
//! `threads_flexpar`.

#![cfg(all(target_os = "macos", target_arch = "aarch64"))]

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvBuf;
use epistat_survival::flexpar::{
    Contrast, FlexParFit, FlexParOptions, Predict, PredictOptions, Scale, TimeVarying, contrast,
    fit, predict,
};
use epistat_testkit::Fingerprint;

fn hash_fit(f: &FlexParFit) -> Fingerprint {
    let mut h = Fingerprint::new().f64s(&f.gamma);
    for b in &f.beta {
        h = h.option_f64(*b);
    }
    for c in &f.tvc_coefficients {
        h = h.f64s(c);
    }
    for matrix in [&f.covariance, &f.robust_covariance] {
        h = match matrix {
            Some(v) => v.as_slice().iter().fold(h.u64(1), |h, &x| h.option_f64(x)),
            None => h.u64(0),
        };
    }
    h.f64(f.log_likelihood)
        .option_f64(f.null_log_likelihood)
        .f64s(&f.knots)
        .u64(u64::try_from(f.iterations).unwrap())
        .u64(u64::from(f.converged))
        .u64(u64::from(f.diverging))
        .u64(u64::from(f.monotone_on_data))
        .u64(f.n_rows as u64)
        .f64(f.n_events)
}

#[expect(
    clippy::struct_excessive_bools,
    reason = "a table of which features each pinned case switches on"
)]
struct Case {
    name: &'static str,
    entry: bool,
    weights: bool,
    clusters: bool,
    tvc: bool,
    columns: usize,
}

fn run(scale: Scale, case: &Case) -> (u64, u64) {
    let n = 500;
    let mut rng = Stream::seeded(77);
    let (mut time, mut start, mut status, mut weight, mut cluster) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut x = Mat::<f64>::zeros(n, 3);
    for i in 0..n {
        let a = rng.normal();
        let b = f64::from(u8::from(rng.below(5) < 2));
        let t = (rng.exp1().powf(0.75) * (-(0.5 * a - 0.4 * b)).exp() * 200.0).ceil() / 100.0;
        let c = (rng.exp1() * 500.0).ceil() / 100.0;
        time.push(t.min(c));
        status.push(i32::from(t <= c));
        let entry = (0.3 * t.min(c) * 100.0).floor() / 100.0;
        start.push(if rng.below(3) == 0 { entry } else { 0.0 });
        weight.push(match rng.below(4) {
            0 => 0.5,
            1 => 2.0,
            _ => 1.0,
        });
        cluster.push(u32::try_from(rng.below(40)).unwrap());
        x[(i, 0)] = a;
        x[(i, 1)] = b;
        x[(i, 2)] = a - 2.0 * b;
    }
    let mut builder = SurvBuf::builder(time, status);
    if case.entry {
        builder = builder.start(start);
    }
    if case.weights {
        builder = builder.weight(weight);
    }
    if case.clusters {
        builder = builder.cluster(cluster);
    }
    let buf = builder.build().unwrap();
    let data = buf.as_data();
    let xs = Mat::from_fn(n, case.columns, |i, j| x[(i, j)]);
    let tvc = [TimeVarying { column: 1, df: 2 }];
    let options = FlexParOptions {
        tvc: if case.tvc { &tvc } else { &[] },
        ..FlexParOptions::new(scale, 3)
    };
    let f = fit(&data, xs.as_ref(), &options).unwrap();
    assert!(f.converged, "{}", case.name);
    let new = Mat::from_fn(2, case.columns, |i, j| {
        [[0.4, 1.0, 0.0], [-0.3, 0.0, 0.0]][i][j]
    });
    let reference = Mat::from_fn(1, case.columns, |_, j| [-0.3, 0.0, 0.0][j]);
    let o = PredictOptions {
        robust: case.clusters,
        ..PredictOptions::default()
    };
    let times = [0.2, 0.7, 1.5, 3.0];
    let mut h = Fingerprint::new();
    for what in [
        Predict::Survival,
        Predict::CumulativeHazard,
        Predict::Hazard,
    ] {
        for curve in predict(&f, new.as_ref(), &times, what, &o).unwrap() {
            for p in curve {
                h = h
                    .f64(p.estimate)
                    .f64(p.se)
                    .option_f64(p.lower)
                    .option_f64(p.upper);
            }
        }
    }
    for what in [
        Contrast::HazardRatio,
        Contrast::HazardDifference,
        Contrast::SurvivalDifference,
    ] {
        for curve in contrast(&f, new.as_ref(), reference.as_ref(), &times, what, &o).unwrap() {
            for p in curve {
                h = h
                    .f64(p.estimate)
                    .f64(p.se)
                    .option_f64(p.lower)
                    .option_f64(p.upper);
            }
        }
    }
    (hash_fit(&f).finish(), h.finish())
}

type Line = (String, u64, u64);

fn check(lines: &[Line], expected: &[(&str, u64, u64)]) {
    let want: Vec<Line> = expected
        .iter()
        .map(|&(name, a, b)| (name.to_string(), a, b))
        .collect();
    let recorded: Vec<String> = lines
        .iter()
        .map(|(name, a, b)| format!("(\"{name}\", {a:#018x}, {b:#018x}),"))
        .collect();
    assert_eq!(lines, want, "recorded:\n{}", recorded.join("\n"));
}

#[test]
fn bits_are_pinned() {
    let cases = [
        Case {
            name: "plain",
            entry: false,
            weights: false,
            clusters: false,
            tvc: false,
            columns: 2,
        },
        Case {
            name: "entry weights clusters",
            entry: true,
            weights: true,
            clusters: true,
            tvc: false,
            columns: 2,
        },
        Case {
            name: "time-varying effect",
            entry: true,
            weights: true,
            clusters: true,
            tvc: true,
            columns: 2,
        },
        Case {
            name: "aliased",
            entry: false,
            weights: true,
            clusters: true,
            tvc: true,
            columns: 3,
        },
    ];
    let mut lines: Vec<Line> = Vec::new();
    for (label, scale) in [
        ("hazard", Scale::Hazard),
        ("odds", Scale::Odds),
        ("probit", Scale::Probit),
    ] {
        for case in &cases {
            let (fit, predictions) = run(scale, case);
            lines.push((format!("{} {label}", case.name), fit, predictions));
        }
    }
    check(&lines, EXPECTED);
}

const EXPECTED: &[(&str, u64, u64)] = &[
    ("plain hazard", 0xa4aa_c405_2980_8894, 0xb82f_3f66_1e78_85ad),
    (
        "entry weights clusters hazard",
        0xd1b8_4101_af7f_ecc7,
        0x8177_ef5c_1307_2a6b,
    ),
    (
        "time-varying effect hazard",
        0x6b05_8f1e_edd2_0f5c,
        0x16aa_e11b_555e_60c5,
    ),
    (
        "aliased hazard",
        0xe0d0_c6d1_7297_7572,
        0x0ad8_ab11_bcfd_a402,
    ),
    ("plain odds", 0xf404_b057_0d50_230c, 0x1b6f_2f6c_bac9_26ae),
    (
        "entry weights clusters odds",
        0x1edc_5ec1_23f1_d5d4,
        0x8ed9_b8b4_ecac_4d65,
    ),
    (
        "time-varying effect odds",
        0x501b_8ebf_70fe_979d,
        0x43df_8e2f_e8b4_757c,
    ),
    ("aliased odds", 0x1267_12e8_4111_c357, 0x15bf_7512_2714_981f),
    ("plain probit", 0xaa95_3bfc_4b83_65ab, 0xa68c_299c_67df_f895),
    (
        "entry weights clusters probit",
        0xf6e9_be36_9e92_20c0,
        0x3413_002c_01ec_1eae,
    ),
    (
        "time-varying effect probit",
        0xb45e_372a_d726_2d83,
        0x75e3_bbb0_0e9b_fbdb,
    ),
    (
        "aliased probit",
        0xeb99_d384_ed0a_39fb,
        0xc815_8e6f_3ffe_33ec,
    ),
];
