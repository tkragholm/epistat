//! `cox::fit` and `cox::score_residuals` on a `SurvData` read the event code,
//! the missing start, the weights, the strata, the clusters and the tie rule as
//! documented.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the inputs are a few hundred rows, far below the limits of the casts"
)]

use epistat_core::{Classify, ErrorKind};
use epistat_linalg::Mat;
use epistat_rng::Stream;
use epistat_survival::SurvData;
use epistat_survival::cox::{CoxOptions, Ties, fit, score_residuals};
use epistat_survival::timefix::TimeFix;

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

fn no_fix() -> CoxOptions {
    CoxOptions {
        ties: Ties::Efron,
        timefix: TimeFix::Exact,
        cause: None,
        start: None,
        max_iterations: 20,
        tolerance: 1e-9,
        ..Default::default()
    }
}

#[test]
fn every_combination_of_columns_fits() {
    let s = sample(400, 31);
    let n = s.stop.len();
    let x = Mat::from_fn(n, 3, |i, j| s.x[(i, j)]);
    let status: Vec<i32> = s.status.iter().map(|&c| i32::from(c == 1)).collect();
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
                    let opts = CoxOptions { ties, ..no_fix() };
                    let model = fit(&data, x.as_ref(), &opts).unwrap();
                    assert!(model.converged && model.coefficients[2].is_none());
                    assert_eq!(model.n_rows, n);
                    assert_eq!(model.n_used, data.n_used());
                    assert_eq!(model.robust_covariance.is_some(), grouped);
                    let residuals =
                        score_residuals(&data, x.as_ref(), &model.coefficients, &opts).unwrap();
                    assert_eq!(residuals.len(), 3);
                    assert!(residuals[2].is_none());
                    for column in residuals.iter().flatten() {
                        assert_eq!(column.len(), n);
                    }
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
    let model = fit(&data, x.as_ref(), &no_fix()).unwrap();
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
            ..no_fix()
        };
        let got = fit(&data, x.as_ref(), &opts).unwrap();
        // The same rows with the status 1 for this cause and 0 for the rest.
        let binary: Vec<i32> = s.status.iter().map(|&c| i32::from(c == code)).collect();
        let reference = SurvData::builder(&s.stop, &binary)
            .weight(&s.weight)
            .build()
            .unwrap();
        let want = fit(&reference, x.as_ref(), &no_fix()).unwrap();
        assert_eq!(format!("{got:?}"), format!("{want:?}"));
    }
    for cause in [None, Some(0), Some(4)] {
        let opts = CoxOptions { cause, ..no_fix() };
        assert!(matches!(
            fit(&data, x.as_ref(), &opts),
            Err(ref e) if e.kind() == ErrorKind::Input
        ));
        assert!(matches!(
            score_residuals(&data, x.as_ref(), &[Some(0.0), Some(0.0)], &opts),
            Err(ref e) if e.kind() == ErrorKind::Input
        ));
    }
    let binary: Vec<i32> = s.status.iter().map(|&c| i32::from(c == 1)).collect();
    let data = SurvData::builder(&s.stop, &binary).build().unwrap();
    let one = fit(&data, x.as_ref(), &no_fix()).unwrap();
    let explicit = CoxOptions {
        cause: Some(1),
        ..no_fix()
    };
    assert_eq!(
        format!("{one:?}"),
        format!("{:?}", fit(&data, x.as_ref(), &explicit).unwrap())
    );
    let two = CoxOptions {
        cause: Some(2),
        ..no_fix()
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
    let want = fit(&exact, x.as_ref(), &no_fix()).unwrap();
    let opts = CoxOptions::default();
    assert_eq!(opts.timefix, TimeFix::default());
    let got = fit(&rough, x.as_ref(), &opts).unwrap();
    // Tied times take the smallest value of their chain, which is a time of
    // the unblurred data, so the fit has the same bits.
    assert_eq!(format!("{got:?}"), format!("{want:?}"));
    let untied = fit(&rough, x.as_ref(), &no_fix()).unwrap();
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
    for x in [short.as_ref(), nan.as_ref()] {
        assert!(matches!(
            fit(&data, x, &no_fix()),
            Err(ref e) if e.kind() == ErrorKind::Input
        ));
        assert!(matches!(
            score_residuals(&data, x, &[Some(0.0)], &no_fix()),
            Err(ref e) if e.kind() == ErrorKind::Input
        ));
    }
    // A design without columns is the null model, as coxph(~ 1); a coefficient
    // given for it is still the wrong length.
    let null = fit(&data, empty.as_ref(), &no_fix()).unwrap();
    assert!(null.coefficients.is_empty() && null.rank == 0);
    assert_eq!(
        null.log_partial_likelihood,
        null.null_log_partial_likelihood
    );
    assert!(matches!(
        score_residuals(&data, empty.as_ref(), &[Some(0.0)], &no_fix()),
        Err(ref e) if e.kind() == ErrorKind::Input
    ));
    let x = Mat::from_fn(n, 1, |i, _| s.x[(i, 0)]);
    assert!(score_residuals(&data, x.as_ref(), &[Some(0.0), Some(0.0)], &no_fix()).is_err());
}

/// Cluster codes that skip values are recoded to dense ones. The sums are the
/// same up to the order of the additions.
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
    let a = fit(&dense, x.as_ref(), &no_fix()).unwrap();
    let b = fit(&spread, x.as_ref(), &no_fix()).unwrap();
    assert_eq!(
        format!("{:?}", a.coefficients),
        format!("{:?}", b.coefficients)
    );
    for (u, v) in a
        .robust_covariance
        .unwrap()
        .as_slice()
        .iter()
        .zip(b.robust_covariance.unwrap().as_slice())
    {
        let (u, v) = (u.unwrap(), v.unwrap());
        assert!((u - v).abs() <= 1e-12 * u.abs().max(v.abs()));
    }
}
