//! The bits of `flexpar::fit`, `predict` and `contrast` on varied inputs,
//! pinned by a hash of every output field, so that a change to any last bit is
//! a failure here. The bits depend on the platform's `ln` and `exp`, which come
//! from its maths library, so the test asserts the recorded values only on the
//! target it was recorded on, aarch64 macOS. Thread-count independence is
//! checked everywhere by `threads_flexpar`. Elsewhere it computes the hashes
//! and, when `EPISTAT_HASH_OUT` is set, appends them to that file.

use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvBuf;
use epistat_survival::flexpar::{
    FlexParFit, FlexParOptions, FlexparContrast, FlexparPredictOptions, FlexparScale, Predict,
    TimeVarying, contrast, fit, predict,
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

fn run(scale: FlexparScale, case: &Case) -> (u64, u64) {
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
        tvc: if case.tvc { tvc.to_vec() } else { Vec::new() },
        ..FlexParOptions::new(scale, 3)
    };
    let f = fit(&data, xs.as_ref(), None, &options).unwrap();
    assert!(f.converged, "{}", case.name);
    let new = Mat::from_fn(2, case.columns, |i, j| {
        [[0.4, 1.0, 0.0], [-0.3, 0.0, 0.0]][i][j]
    });
    let reference = Mat::from_fn(1, case.columns, |_, j| [-0.3, 0.0, 0.0][j]);
    let o = FlexparPredictOptions {
        robust: case.clusters,
        ..FlexparPredictOptions::default()
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
        FlexparContrast::HazardRatio,
        FlexparContrast::HazardDifference,
        FlexparContrast::SurvivalDifference,
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
    for (name, a, b) in lines {
        epistat_testkit::hash_out(&format!("{name}/0"), *a);
        epistat_testkit::hash_out(&format!("{name}/1"), *b);
    }
    if !epistat_testkit::RECORDED_PLATFORM {
        return;
    }
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
        ("hazard", FlexparScale::Hazard),
        ("odds", FlexparScale::Odds),
        ("probit", FlexparScale::Probit),
    ] {
        for case in &cases {
            let (fit, predictions) = run(scale, case);
            lines.push((format!("{} {label}", case.name), fit, predictions));
        }
    }
    check(&lines, EXPECTED);
}

const EXPECTED: &[(&str, u64, u64)] = &[
    ("plain hazard", 0x38e3_35c6_7b0c_915c, 0xfb8e_b25f_0447_d9a1),
    (
        "entry weights clusters hazard",
        0xf12e_ed0f_aea0_57b0,
        0x0733_fc6d_3f07_76b4,
    ),
    (
        "time-varying effect hazard",
        0x2e86_ba6b_61d2_d4a6,
        0xde02_ccda_1e4d_3519,
    ),
    (
        "aliased hazard",
        0xa331_200b_986a_582a,
        0x79c9_4cc3_355a_8dc5,
    ),
    ("plain odds", 0x0828_6a3d_baf6_72f6, 0xe7c7_d384_6c28_50b1),
    (
        "entry weights clusters odds",
        0xe5d0_8236_4cd8_af83,
        0xb884_d622_6170_190a,
    ),
    (
        "time-varying effect odds",
        0x9c14_c9ce_d247_de1c,
        0x4249_75e3_82b7_5eb7,
    ),
    ("aliased odds", 0x18a6_4065_1780_6111, 0xc602_dea3_6fa2_51cf),
    ("plain probit", 0x968a_d1cb_7a21_4be1, 0x88e2_6591_f31c_a25d),
    (
        "entry weights clusters probit",
        0xf0ce_6cec_eed4_c24b,
        0xb664_6971_d8e1_475d,
    ),
    (
        "time-varying effect probit",
        0x3fcf_3bfb_1782_7584,
        0xf977_98fb_4c51_aeb4,
    ),
    (
        "aliased probit",
        0x8db9_5bc7_46e1_52b2,
        0xbfa8_d3fd_0904_cfc5,
    ),
];
