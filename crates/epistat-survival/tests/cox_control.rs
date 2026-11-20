//! Offsets, starting values, iteration limits and tests of `cox::fit`: input
//! checks, undefined cases and identities that need no reference.

use epistat_survival::cox::{CoxOptions, fit};
use epistat_survival::{Error, SurvData};
use faer::Mat;

const STOP: [f64; 10] = [1., 2., 2., 3., 4., 5., 6., 7., 8., 9.];
const STATUS: [i32; 10] = [1, 1, 1, 0, 1, 1, 0, 1, 1, 0];
const X1: [f64; 10] = [0.3, -1.2, 0.5, 0.9, -0.4, 1.1, -0.7, 0.2, 0.8, -0.1];

fn design() -> Mat<f64> {
    Mat::from_fn(
        10,
        2,
        |i, j| {
            if j == 0 { X1[i] } else { f64::from(i % 2 == 1) }
        },
    )
}

fn data() -> SurvData<'static> {
    SurvData::builder(&STOP, &STATUS).build().unwrap()
}

fn is_input_error<T>(result: &Result<T, Error>) -> bool {
    matches!(result, Err(Error::Input(_)))
}

#[test]
fn invalid_control_is_an_error() {
    let (x, d) = (design(), data());
    let run = |opts: CoxOptions<'_>| fit(&d, x.as_ref(), &opts);
    let short = [0.0; 9];
    assert!(is_input_error(&run(CoxOptions {
        offset: Some(&short),
        ..CoxOptions::default()
    })));
    let mut bad = [0.0; 10];
    bad[3] = f64::NAN;
    assert!(is_input_error(&run(CoxOptions {
        offset: Some(&bad),
        ..CoxOptions::default()
    })));
    bad[3] = 800.0;
    assert!(is_input_error(&run(CoxOptions {
        offset: Some(&bad),
        ..CoxOptions::default()
    })));
    for eps in [0.0, -1e-9, f64::NAN, f64::INFINITY] {
        assert!(is_input_error(&run(CoxOptions {
            eps,
            ..CoxOptions::default()
        })));
    }
    assert!(is_input_error(&run(CoxOptions {
        init: Some(&[0.0]),
        ..CoxOptions::default()
    })));
    assert!(is_input_error(&run(CoxOptions {
        init: Some(&[0.0, f64::NAN]),
        ..CoxOptions::default()
    })));
    // A starting value whose risk score overflows.
    assert!(is_input_error(&run(CoxOptions {
        init: Some(&[1e4, 0.0]),
        ..CoxOptions::default()
    })));
}

#[test]
fn a_zero_offset_changes_nothing_and_a_shift_changes_nothing_in_value() {
    let (x, d) = (design(), data());
    let plain = fit(&d, x.as_ref(), &CoxOptions::default()).unwrap();
    let zeros = [0.0; 10];
    let with_zero = fit(
        &d,
        x.as_ref(),
        &CoxOptions {
            offset: Some(&zeros),
            ..CoxOptions::default()
        },
    )
    .unwrap();
    assert_eq!(format!("{plain:?}"), format!("{with_zero:?}"));
    let shifted: Vec<f64> = (0..10).map(|_| 3.0).collect();
    let with_shift = fit(
        &d,
        x.as_ref(),
        &CoxOptions {
            offset: Some(&shifted),
            ..CoxOptions::default()
        },
    )
    .unwrap();
    // A constant offset is absorbed by the baseline hazard.
    for (a, b) in plain.coefficients.iter().zip(&with_shift.coefficients) {
        assert!((a.unwrap() - b.unwrap()).abs() < 1e-12);
    }
    assert!(
        (plain.log_partial_likelihood.unwrap() - with_shift.log_partial_likelihood.unwrap()).abs()
            < 1e-10
    );
}

#[test]
fn iteration_limit_zero_is_the_model_at_the_starting_values() {
    let (x, d) = (design(), data());
    let init = [0.25, -0.5];
    let model = fit(
        &d,
        x.as_ref(),
        &CoxOptions {
            init: Some(&init),
            iter_max: 0,
            ..CoxOptions::default()
        },
    )
    .unwrap();
    assert_eq!(model.iterations, 0);
    assert!(!model.converged);
    assert_eq!(model.coefficients, vec![Some(0.25), Some(-0.5)]);
    // At the starting values the Wald and likelihood-ratio statistics are zero.
    let (wald, lr) = (
        model.tests.wald.unwrap(),
        model.tests.likelihood_ratio.unwrap(),
    );
    assert!(wald.statistic.abs() < 1e-12 && lr.statistic == 0.0);
    assert_eq!((wald.p_value, wald.df), (1.0, 2));
    // The score test is that of the fit from zero when the start is zero.
    let from_zero = fit(&d, x.as_ref(), &CoxOptions::default()).unwrap();
    let at_zero = fit(
        &d,
        x.as_ref(),
        &CoxOptions {
            iter_max: 0,
            ..CoxOptions::default()
        },
    )
    .unwrap();
    assert_eq!(from_zero.tests.score, at_zero.tests.score);
    assert_eq!(
        from_zero.null_log_partial_likelihood,
        at_zero.log_partial_likelihood
    );
}

#[test]
fn the_null_likelihood_stays_at_zero_when_the_start_is_not() {
    let (x, d) = (design(), data());
    let plain = fit(&d, x.as_ref(), &CoxOptions::default()).unwrap();
    let started = fit(
        &d,
        x.as_ref(),
        &CoxOptions {
            init: Some(&[1.0, 1.0]),
            ..CoxOptions::default()
        },
    )
    .unwrap();
    assert_eq!(
        started.null_log_partial_likelihood,
        plain.null_log_partial_likelihood
    );
}

#[test]
fn tests_are_undefined_without_events_and_rank() {
    let x = design();
    let none = SurvData::builder(&STOP, &[0; 10]).build().unwrap();
    let model = fit(&none, x.as_ref(), &CoxOptions::default()).unwrap();
    assert_eq!(model.tests, epistat_survival::cox::CoxTests::default());
    // Two columns without variation have rank zero.
    let constant = Mat::from_fn(10, 2, |_, j| f64::from(j == 1));
    let model = fit(&data(), constant.as_ref(), &CoxOptions::default()).unwrap();
    assert_eq!(model.rank, 0);
    assert_eq!(model.tests, epistat_survival::cox::CoxTests::default());
}

#[test]
fn a_diverging_coefficient_keeps_the_score_test_only() {
    // The second column is 1 only for censored rows.
    let x = Mat::from_fn(10, 2, |i, j| {
        if j == 0 {
            X1[i]
        } else {
            f64::from(STATUS[i] == 0 && i > 5)
        }
    });
    let model = fit(&data(), x.as_ref(), &CoxOptions::default()).unwrap();
    assert_ne!(model.monotone_columns, Vec::<usize>::new());
    let t = model.tests;
    assert!(t.score.is_some());
    assert!(t.wald.is_none() && t.likelihood_ratio.is_none() && t.robust_wald.is_none());
}
