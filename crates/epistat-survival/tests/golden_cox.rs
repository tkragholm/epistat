//! The bits of `fit_cox` and `cox_score_residuals` on varied inputs, pinned by
//! a hash of every output field. The numbers of downstream studies depend on
//! them, so a change to any last bit is a failure here. The hashes were
//! recorded once from the implementation before the shared input type.
//!
//! Hashes depend on the platform's libm, so they are pinned on one target.

#![expect(deprecated, reason = "pins the deprecated entry points")]
#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::cox::{CoxDesign, CoxModelFit, Ties, cox_score_residuals, fit_cox};
use epistat_testkit::Fingerprint;
use faer::Mat;

const ABSENT: u64 = 0x7ff8_0000_dead_beef;

fn opt(hash: Fingerprint, value: Option<f64>) -> Fingerprint {
    match value {
        Some(v) => hash.f64(v),
        None => hash.u64(ABSENT),
    }
}

fn opts(hash: Fingerprint, values: &[Option<f64>]) -> Fingerprint {
    values
        .iter()
        .fold(hash.u64(values.len() as u64), |h, &v| opt(h, v))
}

fn hash_fit(fit: &CoxModelFit) -> Fingerprint {
    let mut h = Fingerprint::new();
    h = opts(h, &fit.coefficients);
    h = opts(h, &fit.standard_errors);
    h = opts(h, &fit.covariance);
    h = match &fit.robust_standard_errors {
        Some(v) => opts(h.u64(1), v),
        None => h.u64(0),
    };
    h = match &fit.robust_covariance {
        Some(v) => opts(h.u64(1), v),
        None => h.u64(0),
    };
    h = opt(h, fit.log_partial_likelihood);
    h = opt(h, fit.null_log_partial_likelihood);
    h = h
        .f64(fit.n_events)
        .u64(fit.n_rows as u64)
        .u64(fit.rank as u64)
        .u64(u64::from(fit.iterations))
        .u64(u64::from(fit.converged));
    for &c in &fit.monotone_columns {
        h = h.u64(c as u64);
    }
    h
}

fn hash_residuals(residuals: &[Option<Vec<f64>>]) -> Fingerprint {
    let mut h = Fingerprint::new().u64(residuals.len() as u64);
    for column in residuals {
        h = match column {
            Some(values) => h.u64(1).f64s(values),
            None => h.u64(0),
        };
    }
    h
}

fn hash_text(text: &str) -> Fingerprint {
    text.bytes()
        .fold(Fingerprint::new(), |h, b| h.u64(u64::from(b)))
}

struct Sample {
    x: Mat<f64>,
    is_event: Vec<bool>,
    start: Vec<f64>,
    stop: Vec<f64>,
    weight: Vec<f64>,
    stratum: Vec<u32>,
    cluster: Vec<u32>,
}

/// Times on a grid of tenths so that they tie, three columns of which the
/// third is the first plus the second (aliased), entry times before the exit.
fn sample(n: usize, seed: u64) -> Sample {
    let mut rng = Stream::seeded(seed);
    let mut s = Sample {
        x: Mat::zeros(n, 3),
        is_event: Vec::new(),
        start: Vec::new(),
        stop: Vec::new(),
        weight: Vec::new(),
        stratum: Vec::new(),
        cluster: Vec::new(),
    };
    for i in 0..n {
        let a = rng.normal();
        let b = f64::from(rng.below(4) as u32);
        s.x[(i, 0)] = a;
        s.x[(i, 1)] = b;
        s.x[(i, 2)] = a + b;
        let t = (rng.exp1() * (-0.4 * a).exp() * 8.0).ceil() / 10.0 + 0.1;
        s.stop.push(t);
        s.start.push((rng.uniform() * t * 8.0).floor() / 10.0 * 0.9);
        s.is_event.push(rng.uniform() < 0.65);
        s.weight.push(match rng.below(6) {
            0 => 0.0,
            1 => 2.0,
            2 => 0.5,
            _ => 1.0,
        });
        s.stratum.push(rng.below(3) as u32 * 7);
        s.cluster.push(rng.below(25) as u32 * 3 + 1);
    }
    s
}

struct Case {
    name: &'static str,
    use_start: bool,
    use_strata: bool,
    use_weights: bool,
    clusters: Option<usize>,
    columns: usize,
}

fn run(s: &Sample, case: &Case, ties: Ties) -> (u64, u64) {
    let n = s.stop.len();
    let x = Mat::from_fn(n, case.columns, |i, j| s.x[(i, j)]);
    let start = if case.use_start {
        s.start.clone()
    } else {
        vec![f64::NEG_INFINITY; n]
    };
    let stratum = if case.use_strata {
        s.stratum.clone()
    } else {
        vec![0; n]
    };
    let weight = if case.use_weights {
        s.weight.clone()
    } else {
        vec![1.0; n]
    };
    let cluster: Option<Vec<u32>> = case
        .clusters
        .map(|k| s.cluster.iter().map(|&c| (c as usize % k) as u32).collect());
    let design = CoxDesign {
        x: x.as_ref(),
        is_event: &s.is_event,
        start: &start,
        stop: &s.stop,
        weight: &weight,
        stratum: &stratum,
        ties,
        cluster: cluster.as_deref(),
    };
    let fit = fit_cox(&design).unwrap();
    assert!(fit.converged, "{}", case.name);
    assert_eq!(fit.rank, 2, "{}", case.name);
    let residuals = cox_score_residuals(&design, &fit.coefficients).unwrap();
    // Residuals away from the estimate, with the last column left out.
    let mut other: Vec<Option<f64>> = (0..case.columns).map(|j| Some(0.1 * j as f64)).collect();
    *other.last_mut().unwrap() = None;
    let away = cox_score_residuals(&design, &other).unwrap();
    (
        hash_fit(&fit).finish(),
        hash_residuals(&residuals).finish() ^ hash_residuals(&away).finish().rotate_left(1),
    )
}

#[test]
fn bits_are_pinned() {
    let s = sample(400, 20);
    let cases = [
        Case {
            name: "plain",
            use_start: false,
            use_strata: false,
            use_weights: false,
            clusters: None,
            columns: 2,
        },
        Case {
            name: "entry",
            use_start: true,
            use_strata: false,
            use_weights: false,
            clusters: None,
            columns: 2,
        },
        Case {
            name: "strata weights",
            use_start: false,
            use_strata: true,
            use_weights: true,
            clusters: None,
            columns: 2,
        },
        Case {
            name: "everything",
            use_start: true,
            use_strata: true,
            use_weights: true,
            clusters: Some(11),
            columns: 2,
        },
        Case {
            name: "clusters and weights",
            use_start: false,
            use_strata: false,
            use_weights: true,
            clusters: Some(25),
            columns: 2,
        },
        Case {
            name: "one cluster",
            use_start: true,
            use_strata: true,
            use_weights: true,
            clusters: Some(1),
            columns: 2,
        },
        Case {
            name: "aliased",
            use_start: true,
            use_strata: true,
            use_weights: true,
            clusters: Some(7),
            columns: 3,
        },
    ];
    let mut lines: Vec<Line> = Vec::new();
    for case in &cases {
        for (label, ties) in [("breslow", Ties::Breslow), ("efron", Ties::Efron)] {
            let (fit, residuals) = run(&s, case, ties);
            lines.push((format!("{} {label}", case.name), fit, residuals));
        }
    }
    check(&lines, EXPECTED);
}

/// A column that is one in rows that are all censored makes the coefficient
/// diverge.
#[test]
fn diverging_coefficient() {
    let s = sample(300, 21);
    let n = s.stop.len();
    let x = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            f64::from(u8::from(!s.is_event[i] && i % 3 == 0))
        } else {
            s.x[(i, 0)]
        }
    });
    let mut lines: Vec<Line> = Vec::new();
    for (label, ties) in [("breslow", Ties::Breslow), ("efron", Ties::Efron)] {
        let design = CoxDesign {
            x: x.as_ref(),
            is_event: &s.is_event,
            start: &s.start,
            stop: &s.stop,
            weight: &s.weight,
            stratum: &s.stratum,
            ties,
            cluster: Some(&s.cluster),
        };
        let fit = fit_cox(&design).unwrap();
        assert_ne!(fit.monotone_columns.len(), 0);
        let residuals = cox_score_residuals(&design, &[Some(0.3), Some(-0.2)]).unwrap();
        lines.push((
            format!("diverging {label}"),
            hash_fit(&fit).finish(),
            hash_residuals(&residuals).finish(),
        ));
    }
    check(&lines, EXPECTED_DIVERGING);
}

/// No events, no weight, a cluster whose rows all have weight zero, and the
/// errors.
#[test]
fn edge_cases() {
    let s = sample(60, 22);
    let n = s.stop.len();
    let x = Mat::from_fn(n, 1, |i, _| s.x[(i, 0)]);
    let base = CoxDesign {
        x: x.as_ref(),
        is_event: &s.is_event,
        start: &s.start,
        stop: &s.stop,
        weight: &s.weight,
        stratum: &s.stratum,
        ties: Ties::Efron,
        cluster: Some(&s.cluster),
    };
    let mut lines: Vec<Line> = Vec::new();

    let none = vec![false; n];
    let fit = fit_cox(&CoxDesign {
        is_event: &none,
        ..base
    })
    .unwrap();
    lines.push(("no events".to_string(), hash_fit(&fit).finish(), 0));

    let zeros = vec![0.0; n];
    let design = CoxDesign {
        weight: &zeros,
        ..base
    };
    let fit = fit_cox(&design).unwrap();
    let residuals = cox_score_residuals(&design, &fit.coefficients).unwrap();
    lines.push((
        "no weight".to_string(),
        hash_fit(&fit).finish(),
        hash_residuals(&residuals).finish(),
    ));

    // Every row of cluster 1 has weight zero.
    let weight: Vec<f64> = (0..n)
        .map(|i| {
            if s.cluster[i] == 1 {
                0.0
            } else {
                s.weight[i].max(1.0)
            }
        })
        .collect();
    let design = CoxDesign {
        weight: &weight,
        ..base
    };
    let fit = fit_cox(&design).unwrap();
    let residuals = cox_score_residuals(&design, &fit.coefficients).unwrap();
    lines.push((
        "empty cluster".to_string(),
        hash_fit(&fit).finish(),
        hash_residuals(&residuals).finish(),
    ));

    // A bad entry time on a row of weight zero is not an error.
    let mut start = s.start.clone();
    start[0] = f64::NAN;
    let mut weight = s.weight.clone();
    weight[0] = 0.0;
    let design = CoxDesign {
        start: &start,
        weight: &weight,
        ..base
    };
    let fit = fit_cox(&design).unwrap();
    lines.push((
        "bad entry, no weight".to_string(),
        hash_fit(&fit).finish(),
        0,
    ));

    let mut errors: Vec<(&str, String)> = Vec::new();
    let mut message = |name: &'static str, r: Result<CoxModelFit, epistat_survival::Error>| {
        errors.push((name, r.unwrap_err().to_string()));
    };
    let short = vec![1.0; n - 1];
    message(
        "short weight",
        fit_cox(&CoxDesign {
            weight: &short,
            ..base
        }),
    );
    let mut negative = s.weight.clone();
    negative[3] = -1.0;
    message(
        "negative weight",
        fit_cox(&CoxDesign {
            weight: &negative,
            ..base
        }),
    );
    let mut infinite = s.stop.clone();
    infinite[4] = f64::INFINITY;
    message(
        "infinite stop",
        fit_cox(&CoxDesign {
            stop: &infinite,
            ..base
        }),
    );
    let mut late = s.start.clone();
    late[5] = s.stop[5];
    let mut positive = s.weight.clone();
    positive[5] = 1.0;
    message(
        "late entry",
        fit_cox(&CoxDesign {
            start: &late,
            weight: &positive,
            ..base
        }),
    );
    let nan = Mat::from_fn(n, 1, |i, _| if i == 2 { f64::NAN } else { 1.0 });
    message(
        "nan x",
        fit_cox(&CoxDesign {
            x: nan.as_ref(),
            ..base
        }),
    );
    let empty = Mat::<f64>::zeros(n, 0);
    message(
        "no columns",
        fit_cox(&CoxDesign {
            x: empty.as_ref(),
            ..base
        }),
    );
    for (name, text) in &errors {
        lines.push((name.to_string(), hash_text(text).finish(), 0));
    }
    let wrong = cox_score_residuals(&base, &[Some(0.0), Some(0.0)]).unwrap_err();
    lines.push((
        "wrong coefficients".to_string(),
        hash_text(&wrong.to_string()).finish(),
        0,
    ));
    let nonfinite = cox_score_residuals(&base, &[Some(f64::NAN)]).unwrap_err();
    lines.push((
        "nonfinite coefficient".to_string(),
        hash_text(&nonfinite.to_string()).finish(),
        0,
    ));
    let overflow = cox_score_residuals(&base, &[Some(1e6)]).unwrap_err();
    lines.push((
        "overflow".to_string(),
        hash_text(&overflow.to_string()).finish(),
        0,
    ));
    check(&lines, EXPECTED_EDGE);
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

const EXPECTED: &[(&str, u64, u64)] = &[
    (
        "plain breslow",
        0x3a48_7f5e_0e1f_2ee9,
        0xc751_d7ce_8386_cc8a,
    ),
    ("plain efron", 0x0ece_2920_a7b2_25f9, 0x7928_3901_3788_7686),
    (
        "entry breslow",
        0x6251_7338_b2c2_056a,
        0x0c9e_b248_8c94_56bc,
    ),
    ("entry efron", 0xb341_fd80_9906_f426, 0x5361_4df1_9586_d4cd),
    (
        "strata weights breslow",
        0xe52c_c06a_eaf8_cbc8,
        0xae0a_ae71_a8bd_3873,
    ),
    (
        "strata weights efron",
        0x2af5_a72e_9c0b_0827,
        0xab76_e24c_2d6b_3022,
    ),
    (
        "everything breslow",
        0x2983_9a02_0b95_dd75,
        0xf5e7_733d_895a_803f,
    ),
    (
        "everything efron",
        0x01cf_652e_18a3_7190,
        0xa9b5_7746_19f2_77d4,
    ),
    (
        "clusters and weights breslow",
        0x2b13_bc57_58a3_f9b8,
        0x62ea_e40c_375d_1c2a,
    ),
    (
        "clusters and weights efron",
        0x7b02_e0c5_fae5_bffc,
        0xae44_aca3_ca18_06ef,
    ),
    (
        "one cluster breslow",
        0x1a54_2550_7feb_cc3c,
        0xf5e7_733d_895a_803f,
    ),
    (
        "one cluster efron",
        0xa902_d6fd_8949_e726,
        0xa9b5_7746_19f2_77d4,
    ),
    (
        "aliased breslow",
        0x7e40_4fd8_687d_dc2c,
        0xadd9_5b4b_3171_c82b,
    ),
    (
        "aliased efron",
        0x78a8_74f4_e438_b4b4,
        0x27d2_d9c7_44d8_b6cf,
    ),
];
const EXPECTED_DIVERGING: &[(&str, u64, u64)] = &[
    (
        "diverging breslow",
        0xc835_b689_7e70_0968,
        0x1165_b1c9_9819_ab95,
    ),
    (
        "diverging efron",
        0xd6d4_0f2f_3958_3466,
        0x8fdd_ffb2_5871_27bf,
    ),
];
const EXPECTED_EDGE: &[(&str, u64, u64)] = &[
    ("no events", 0x10d0_e035_6912_f630, 0),
    ("no weight", 0xb9ad_8b70_5eeb_141f, 0x3922_09f1_4dea_4c24),
    (
        "empty cluster",
        0xc7ac_3043_2ac1_d019,
        0x1a46_505a_c9ac_c763,
    ),
    ("bad entry, no weight", 0x1645_3870_1cd5_0765, 0),
    ("short weight", 0xd8e2_3414_a04d_a4cb, 0),
    ("negative weight", 0x0b62_d251_a63a_cddf, 0),
    ("infinite stop", 0x107f_5522_2c89_2f50, 0),
    ("late entry", 0xff79_f512_fa54_de0c, 0),
    ("nan x", 0x6249_3f14_1384_1f90, 0),
    ("no columns", 0x9b6b_0fb0_4013_c103, 0),
    ("wrong coefficients", 0xcbae_9c82_4a20_4b9c, 0),
    ("nonfinite coefficient", 0x1637_3757_1948_17c5, 0),
    ("overflow", 0x7b41_5d57_a467_fe58, 0),
];
