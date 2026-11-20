//! Interval-censored input, the refusals of `survreg::fit` and the handling of
//! rows of weight zero.

use epistat_linalg::Mat;
use epistat_survival::survreg::{Distribution, SurvregOptions, fit};
use epistat_survival::{IntervalKind, Outcome, SurvData};

const INF: f64 = f64::INFINITY;

fn message<T>(result: epistat_survival::Result<T>) -> String {
    match result {
        Ok(_) => panic!("expected an error"),
        Err(error) => error.to_string(),
    }
}

#[test]
fn bounds_give_the_kinds_and_status_columns_give_the_same_bounds() {
    let lower = [2.0, 3.0, -INF, 1.0];
    let upper = [INF, 3.0, 4.0, 2.5];
    let data = SurvData::interval_builder(&lower, &upper).build().unwrap();
    let surv = data.interval().unwrap();
    assert_eq!(
        surv.kind(),
        [
            IntervalKind::Right,
            IntervalKind::Exact,
            IntervalKind::Left,
            IntervalKind::Interval
        ]
    );
    assert_eq!(
        surv.kind().iter().map(|k| k.code()).collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    assert!(matches!(data.outcome(), Outcome::Interval(_)));
    let time = [2.0, 3.0, 4.0, 1.0];
    let time2 = [f64::NAN, f64::NAN, f64::NAN, 2.5];
    let status = [0, 1, 2, 3];
    let typed = SurvData::interval_status_builder(&time, &time2, &status)
        .build()
        .unwrap();
    assert_eq!(typed.interval().unwrap().lower(), &lower);
    assert_eq!(typed.interval().unwrap().upper(), &upper);
    assert_eq!((data.n_rows(), data.n_used()), (4, 4));
}

#[test]
fn follow_up_estimators_refuse_interval_data() {
    let data = SurvData::interval_builder(&[1.0], &[2.0]).build().unwrap();
    assert!(message(data.followup()).contains("interval-censored"));
    assert!(message(data.cause(1)).contains("interval-censored"));
    assert!(
        message(
            data.clone()
                .with_timefix(epistat_survival::timefix::TimeFix::default())
        )
        .contains("tie policy")
    );
    assert!(
        data.clone()
            .with_timefix(epistat_survival::timefix::TimeFix::Exact)
            .is_ok()
    );
    let follow = SurvData::builder(&[1.0], &[1]).build().unwrap();
    assert!(message(follow.interval()).contains("follow-up"));
}

#[test]
fn impossible_bounds_are_refused() {
    let build = |l: &[f64], u: &[f64]| message(SurvData::interval_builder(l, u).build());
    assert!(build(&[f64::NAN], &[1.0]).contains("NaN"));
    assert!(build(&[INF], &[INF]).contains("wrong side"));
    assert!(build(&[0.0], &[-INF]).contains("wrong side"));
    assert!(build(&[-INF], &[INF]).contains("both infinite"));
    assert!(build(&[2.0], &[1.0]).contains("above their upper"));
    assert!(build(&[1.0, 2.0], &[1.0]).contains("upper has 1"));
    let statuses = |t: &[f64], t2: &[f64], s: &[i32]| {
        message(SurvData::interval_status_builder(t, t2, s).build())
    };
    assert!(statuses(&[1.0], &[2.0], &[4]).contains("status"));
    assert!(statuses(&[1.0], &[1.0], &[3]).contains("not below"));
    assert!(statuses(&[1.0], &[f64::NAN], &[3]).contains("not finite"));
    assert!(statuses(&[f64::NAN], &[2.0], &[1]).contains("time"));
}

fn sample() -> (Vec<f64>, Vec<i32>, Mat<f64>) {
    let time: Vec<f64> = (1..=20)
        .map(|i| f64::from(i) * 0.7 + (f64::from(i) * 1.3).sin())
        .collect();
    let status: Vec<i32> = (0..20).map(|i| i32::from(i % 4 != 0)).collect();
    let x = Mat::from_fn(
        20,
        2,
        |i, j| if j == 0 { 1.0 } else { (i as f64 * 0.37).cos() },
    );
    (time, status, x)
}

#[test]
fn rows_of_weight_zero_are_left_out_and_counted() {
    let (time, status, x) = sample();
    let plain = SurvData::builder(&time, &status).build().unwrap();
    let reference = fit(&plain, x.as_ref(), &SurvregOptions::default()).unwrap();
    // Four extra rows, whose times would be refused if they were used.
    let mut time2 = time.clone();
    let mut status2 = status.clone();
    let mut weight = vec![1.0; 20];
    for k in 0..4 {
        time2.push(-1.0 - f64::from(k));
        status2.push(1);
        weight.push(0.0);
    }
    let x2 = Mat::from_fn(
        24,
        2,
        |i, j| if i < 20 { x[(i, j)] } else { 100.0 * j as f64 },
    );
    let data = SurvData::builder(&time2, &status2)
        .weight(&weight)
        .build()
        .unwrap();
    let got = fit(&data, x2.as_ref(), &SurvregOptions::default()).unwrap();
    assert_eq!((got.n_rows, got.n_used), (24, 20));
    assert_eq!(got.coefficients, reference.coefficients);
    assert_eq!(got.variance, reference.variance);
    assert_eq!(got.loglik.to_bits(), reference.loglik.to_bits());
}

#[test]
fn bad_input_is_refused() {
    let (time, status, x) = sample();
    let data = SurvData::builder(&time, &status).build().unwrap();
    let go = |opts: &SurvregOptions, x: &Mat<f64>| message(fit(&data, x.as_ref(), opts));
    let base = SurvregOptions::default();
    assert!(go(&base, &Mat::zeros(19, 2)).contains("19 rows"));
    assert!(go(&base, &Mat::zeros(20, 0)).contains("no columns"));
    let mut bad = x.clone();
    bad[(3, 1)] = f64::NAN;
    assert!(go(&base, &bad).contains("not finite"));
    let fixed = |scale| SurvregOptions {
        scale: Some(scale),
        ..base
    };
    assert!(go(&fixed(-1.0), &x).contains("positive"));
    let exponential = SurvregOptions {
        distribution: Distribution::Exponential,
        scale: Some(2.0),
        ..base
    };
    assert!(go(&exponential, &x).contains("fixed scale"));
    let none = SurvregOptions {
        max_iterations: 0,
        ..base
    };
    assert!(go(&none, &x).contains("max_iterations"));
    // Strata with a fixed scale.
    let stratum: Vec<u32> = (0..20).map(|i| i % 2).collect();
    let strata = SurvData::builder(&time, &status)
        .stratum(&stratum)
        .build()
        .unwrap();
    assert!(message(fit(&strata, x.as_ref(), &fixed(1.0))).contains("strata"));
    // A time that is not positive under a log distribution.
    let mut t = time.clone();
    t[2] = 0.0;
    let zero = SurvData::builder(&t, &status).build().unwrap();
    assert!(message(fit(&zero, x.as_ref(), &base)).contains("first at row 2"));
    // Delayed entry and several codes.
    let start = vec![0.0; 20];
    let entry = SurvData::builder(&time, &status)
        .start(&start)
        .build()
        .unwrap();
    assert!(message(fit(&entry, x.as_ref(), &base)).contains("delayed entry"));
    let multi = SurvData::builder(&time, &status)
        .n_codes(2)
        .build()
        .unwrap();
    assert!(message(fit(&multi, x.as_ref(), &base)).contains("event codes"));
    // No row used.
    let w = vec![0.0; 20];
    let none_used = SurvData::builder(&time, &status)
        .weight(&w)
        .build()
        .unwrap();
    assert!(message(fit(&none_used, x.as_ref(), &base)).contains("no row"));
}

#[test]
fn an_interval_from_zero_is_left_censored_for_every_log_distribution() {
    let lower = [0.0, 1.0, 2.0, 0.5, 3.0, 1.5, 0.0, 2.5];
    let upper = [1.0, 2.0, 3.0, 1.5, INF, 2.5, 0.8, 3.5];
    let as_left_lower = [-INF, 1.0, 2.0, 0.5, 3.0, 1.5, -INF, 2.5];
    let x = Mat::from_fn(8, 1, |_, _| 1.0);
    let zero = SurvData::interval_builder(&lower, &upper).build().unwrap();
    let left = SurvData::interval_builder(&as_left_lower, &upper)
        .build()
        .unwrap();
    let options = SurvregOptions {
        distribution: Distribution::LogLogistic,
        ..SurvregOptions::default()
    };
    let a = fit(&zero, x.as_ref(), &options).unwrap();
    let b = fit(&left, x.as_ref(), &options).unwrap();
    assert_eq!(a.loglik.to_bits(), b.loglik.to_bits());
}
