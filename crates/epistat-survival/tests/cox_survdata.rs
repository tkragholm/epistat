//! `cox::fit` and `cox::score_residuals` on a `SurvData` give the bits of the
//! deprecated entry points on the same rows, and read the event code, the
//! missing start and the tie rule as documented.

#![expect(deprecated, reason = "compares with the deprecated entry points")]
#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use epistat_rng::Stream;
use epistat_survival::cox::{
    CoxDesign, CoxOptions, Ties, cox_score_residuals, fit, fit_cox, score_residuals,
};
use epistat_survival::timefix::TimeFix;
use epistat_survival::{Error, SurvData};
use faer::Mat;

struct Sample {
    x: Mat<f64>,
    status: Vec<i32>,
    start: Vec<f64>,
    stop: Vec<f64>,
    weight: Vec<f64>,
    stratum: Vec<u32>,
    cluster: Vec<u32>,
}

fn sample(n: usize, seed: u64) -> Sample {
    let mut rng = Stream::seeded(seed);
    let mut s = Sample {
        x: Mat::zeros(n, 3),
        status: Vec::new(),
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
        s.status.push(match rng.below(8) {
            0..=2 => 1,
            3 | 4 => 2,
            5 => 3,
            _ => 0,
        });
        s.weight.push(match rng.below(6) {
            0 => 0.0,
            1 => 2.0,
            2 => 0.5,
            _ => 1.0,
        });
        s.stratum.push(rng.below(3) as u32 * 7 + 5);
        s.cluster.push(rng.below(25) as u32);
    }
    s
}

const NO_FIX: CoxOptions = CoxOptions {
    ties: Ties::Efron,
    timefix: TimeFix::Exact,
    cause: None,
};

#[test]
fn matches_the_deprecated_entry_points() {
    let s = sample(400, 31);
    let n = s.stop.len();
    let x = Mat::from_fn(n, 3, |i, j| s.x[(i, j)]);
    let event: Vec<bool> = s.status.iter().map(|&c| c == 1).collect();
    let status: Vec<i32> = event.iter().map(|&e| i32::from(e)).collect();
    for ties in [Ties::Breslow, Ties::Efron] {
        for entry in [false, true] {
            for weighted in [false, true] {
                for grouped in [false, true] {
                    let mut builder = SurvData::builder(&s.stop, &status);
                    if entry {
                        builder = builder.start(&s.start);
                    }
                    if weighted {
                        builder = builder.weight(&s.weight);
                    }
                    if grouped {
                        builder = builder.stratum(&s.stratum).cluster(&s.cluster);
                    }
                    let data = builder.build().unwrap();
                    let opts = CoxOptions { ties, ..NO_FIX };
                    let new = fit(&data, x.as_ref(), &opts).unwrap();

                    let start = if entry {
                        s.start.clone()
                    } else {
                        vec![f64::NEG_INFINITY; n]
                    };
                    let weight = if weighted {
                        s.weight.clone()
                    } else {
                        vec![1.0; n]
                    };
                    let (stratum, cluster) = if grouped {
                        (s.stratum.clone(), Some(s.cluster.clone()))
                    } else {
                        (vec![0; n], None)
                    };
                    let design = CoxDesign {
                        x: x.as_ref(),
                        is_event: &event,
                        start: &start,
                        stop: &s.stop,
                        weight: &weight,
                        stratum: &stratum,
                        ties,
                        cluster: cluster.as_deref(),
                    };
                    let old = fit_cox(&design).unwrap();
                    assert!(new.converged && new.coefficients[2].is_none());
                    assert_eq!(new.n_rows, n);
                    assert_eq!(new.n_used, old.n_rows);
                    assert_eq!(old.n_used, old.n_rows);
                    let mut same = new.clone();
                    same.n_rows = old.n_rows;
                    assert_eq!(format!("{same:?}"), format!("{old:?}"));

                    let got = score_residuals(&data, x.as_ref(), &new.coefficients, &opts).unwrap();
                    let want = cox_score_residuals(&design, &old.coefficients).unwrap();
                    assert_eq!(format!("{got:?}"), format!("{want:?}"));
                }
            }
        }
    }
}

#[test]
fn rows_of_weight_zero_are_counted_and_not_fitted() {
    let s = sample(200, 32);
    let n = s.stop.len();
    let x = Mat::from_fn(n, 1, |i, _| s.x[(i, 0)]);
    let status: Vec<i32> = s.status.iter().map(|&c| i32::from(c == 1)).collect();
    let data = SurvData::builder(&s.stop, &status)
        .weight(&s.weight)
        .build()
        .unwrap();
    let model = fit(&data, x.as_ref(), &NO_FIX).unwrap();
    let positive = s.weight.iter().filter(|&&w| w > 0.0).count();
    assert!(positive < n);
    assert_eq!((model.n_rows, model.n_used), (n, positive));
}

#[test]
fn the_event_code_picks_the_cause() {
    let s = sample(300, 33);
    let n = s.stop.len();
    let x = Mat::from_fn(n, 2, |i, j| s.x[(i, j)]);
    let data = SurvData::builder(&s.stop, &s.status)
        .n_codes(3)
        .weight(&s.weight)
        .build()
        .unwrap();
    for code in 1..=3_i32 {
        let opts = CoxOptions {
            cause: Some(code.unsigned_abs()),
            ..NO_FIX
        };
        let got = fit(&data, x.as_ref(), &opts).unwrap();
        let event: Vec<bool> = s.status.iter().map(|&c| c == code).collect();
        let design = CoxDesign {
            x: x.as_ref(),
            is_event: &event,
            start: &vec![f64::NEG_INFINITY; n],
            stop: &s.stop,
            weight: &s.weight,
            stratum: &vec![0; n],
            ties: Ties::Efron,
            cluster: None,
        };
        let want = fit_cox(&design).unwrap();
        let mut same = got.clone();
        same.n_rows = want.n_rows;
        assert_eq!(format!("{same:?}"), format!("{want:?}"));
    }
    for cause in [None, Some(0), Some(4)] {
        let opts = CoxOptions { cause, ..NO_FIX };
        assert!(matches!(
            fit(&data, x.as_ref(), &opts),
            Err(Error::Input(_))
        ));
        assert!(matches!(
            score_residuals(&data, x.as_ref(), &[Some(0.0), Some(0.0)], &opts),
            Err(Error::Input(_))
        ));
    }
    let binary: Vec<i32> = s.status.iter().map(|&c| i32::from(c == 1)).collect();
    let data = SurvData::builder(&s.stop, &binary).build().unwrap();
    let one = fit(&data, x.as_ref(), &NO_FIX).unwrap();
    let explicit = CoxOptions {
        cause: Some(1),
        ..NO_FIX
    };
    assert_eq!(
        format!("{one:?}"),
        format!("{:?}", fit(&data, x.as_ref(), &explicit).unwrap())
    );
    let two = CoxOptions {
        cause: Some(2),
        ..NO_FIX
    };
    assert!(fit(&data, x.as_ref(), &two).is_err());
}

#[test]
fn times_that_differ_by_rounding_tie_by_default() {
    let s = sample(300, 34);
    let n = s.stop.len();
    let x = Mat::from_fn(n, 2, |i, j| s.x[(i, j)]);
    let status: Vec<i32> = s.status.iter().map(|&c| i32::from(c == 1)).collect();
    let blurred: Vec<f64> = s
        .stop
        .iter()
        .enumerate()
        .map(|(i, &t)| if i % 2 == 0 { t * (1.0 + 1e-12) } else { t })
        .collect();
    let exact = SurvData::builder(&s.stop, &status).build().unwrap();
    let rough = SurvData::builder(&blurred, &status).build().unwrap();
    let want = fit(&exact, x.as_ref(), &NO_FIX).unwrap();
    let opts = CoxOptions::default();
    assert_eq!(opts.timefix, TimeFix::default());
    let got = fit(&rough, x.as_ref(), &opts).unwrap();
    // Tied times take the smallest value of their chain, which is a time of
    // the unblurred data, so the fit has the same bits.
    assert_eq!(format!("{got:?}"), format!("{want:?}"));
    let untied = fit(&rough, x.as_ref(), &NO_FIX).unwrap();
    assert_ne!(format!("{untied:?}"), format!("{want:?}"));
}

#[test]
fn the_design_must_fit_the_data() {
    let s = sample(50, 35);
    let n = s.stop.len();
    let status: Vec<i32> = s.status.iter().map(|&c| i32::from(c == 1)).collect();
    let data = SurvData::builder(&s.stop, &status).build().unwrap();
    let short = Mat::from_fn(n - 1, 1, |i, _| s.x[(i, 0)]);
    let empty = Mat::<f64>::zeros(n, 0);
    let nan = Mat::from_fn(n, 1, |i, _| if i == 3 { f64::NAN } else { 1.0 });
    for x in [short.as_ref(), empty.as_ref(), nan.as_ref()] {
        assert!(matches!(fit(&data, x, &NO_FIX), Err(Error::Input(_))));
        assert!(matches!(
            score_residuals(&data, x, &[Some(0.0)], &NO_FIX),
            Err(Error::Input(_))
        ));
    }
    let x = Mat::from_fn(n, 1, |i, _| s.x[(i, 0)]);
    assert!(score_residuals(&data, x.as_ref(), &[Some(0.0), Some(0.0)], &NO_FIX).is_err());
}

/// With cluster codes that skip values the old entry point sums over the empty
/// clusters as rows of zeros, and the dense codes of `SurvData` do not. The
/// sums are the same up to the order of the additions.
#[test]
fn sparse_cluster_codes_change_only_the_order_of_a_sum() {
    let s = sample(400, 36);
    let n = s.stop.len();
    let x = Mat::from_fn(n, 2, |i, j| s.x[(i, j)]);
    let status: Vec<i32> = s.status.iter().map(|&c| i32::from(c == 1)).collect();
    let sparse: Vec<u32> = s.cluster.iter().map(|&c| c * 3 + 1).collect();
    let dense = SurvData::builder(&s.stop, &status)
        .cluster(&s.cluster)
        .build()
        .unwrap();
    let spread = SurvData::builder(&s.stop, &status)
        .cluster(&sparse)
        .build()
        .unwrap();
    let a = fit(&dense, x.as_ref(), &NO_FIX).unwrap();
    let b = fit(&spread, x.as_ref(), &NO_FIX).unwrap();
    assert_eq!(
        format!("{:?}", a.coefficients),
        format!("{:?}", b.coefficients)
    );
    for (u, v) in a
        .robust_covariance
        .unwrap()
        .iter()
        .zip(b.robust_covariance.unwrap())
    {
        let (u, v) = (u.unwrap(), v.unwrap());
        assert!((u - v).abs() <= 1e-12 * u.abs().max(v.abs()));
    }
}
