//! Edge cases of `metrics::royston` that do not need R.

use epistat_survival::SurvData;
use epistat_survival::metrics::royston::{Adjustment, RoystonOptions, estimate};

fn sample() -> (Vec<f64>, Vec<i32>, Vec<f64>) {
    let stop: Vec<f64> = (0..40).map(|i| f64::from((i * 7) % 19) + 1.0).collect();
    let status: Vec<i32> = (0..40).map(|i| i32::from(i % 3 != 0)).collect();
    let eta: Vec<f64> = (0..40).map(|i| f64::from((i * 5) % 7) / 3.0).collect();
    (stop, status, eta)
}

#[test]
fn rows_of_weight_zero_are_left_out() {
    let (stop, status, eta) = sample();
    let mut w = vec![1.0; 40];
    // Weights other than 0 and 1 do not enter: only the rows of positive weight.
    for i in [3, 11, 25] {
        w[i] = 0.0;
    }
    w[0] = 2.5;
    let all = SurvData::builder(&stop, &status)
        .weight(&w)
        .build()
        .unwrap();
    let keep: Vec<usize> = (0..40).filter(|i| w[*i] > 0.0).collect();
    let (s2, t2, e2): (Vec<f64>, Vec<i32>, Vec<f64>) = (
        keep.iter().map(|&i| stop[i]).collect(),
        keep.iter().map(|&i| status[i]).collect(),
        keep.iter().map(|&i| eta[i]).collect(),
    );
    let sub = SurvData::builder(&s2, &t2).build().unwrap();
    let opts = RoystonOptions::default();
    let a = estimate(&all, &eta, &opts).unwrap();
    let b = estimate(&sub, &e2, &opts).unwrap();
    assert_eq!((a.n_rows, a.n_used), (40, 37));
    assert_eq!(a.d, b.d);
    assert_eq!(a.se_d, b.se_d);
    assert_eq!(a.r_ko, b.r_ko);
    assert_eq!(a.c_gh, b.c_gh);
}

#[test]
fn ties_options_differ_only_with_ties_in_the_predictor() {
    let (stop, status, eta) = sample();
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let on = RoystonOptions::default();
    let off = RoystonOptions { ties: false, ..on };
    assert_ne!(
        estimate(&data, &eta, &on).unwrap().d,
        estimate(&data, &eta, &off).unwrap().d
    );
    let distinct: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.1).collect();
    assert_eq!(
        estimate(&data, &distinct, &on).unwrap(),
        estimate(&data, &distinct, &off).unwrap()
    );
}

#[test]
fn concordance_can_be_skipped_and_the_rest_is_unchanged() {
    let (stop, status, eta) = sample();
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let full = estimate(&data, &eta, &RoystonOptions::default()).unwrap();
    let skipped = estimate(
        &data,
        &eta,
        &RoystonOptions {
            gonen_heller: false,
            ..RoystonOptions::default()
        },
    )
    .unwrap();
    assert!(full.c_gh.is_some());
    assert_eq!(skipped.c_gh, None);
    assert_eq!(skipped.d, full.d);
}

#[test]
fn undefined_values_are_none() {
    let (stop, status, _) = sample();
    let data = SurvData::builder(&stop, &status).build().unwrap();
    // A predictor that does not vary: no slope to estimate, no variance.
    let r = estimate(&data, &[0.5; 40], &RoystonOptions::default()).unwrap();
    assert_eq!((r.d, r.se_d, r.r_d, r.beta), (None, None, None, None));
    assert_eq!(r.r_ko, Some(0.0));
    assert_eq!(r.c_gh, Some(0.5));
    // No events.
    let none = vec![0; 40];
    let data = SurvData::builder(&stop, &none).build().unwrap();
    let eta: Vec<f64> = (0..40).map(f64::from).collect();
    let r = estimate(&data, &eta, &RoystonOptions::default()).unwrap();
    assert_eq!(r.d, None);
    assert!(r.r_ko.is_some() && r.c_gh.is_some());
    // One row.
    let one = SurvData::builder(&stop[..1], &status[..1]).build().unwrap();
    let r = estimate(&one, &[1.0], &RoystonOptions::default()).unwrap();
    assert_eq!((r.d, r.r_ko, r.c_gh), (None, None, None));
    assert_eq!((r.n_rows, r.n_used), (1, 1));
}

#[test]
fn adjusted_standard_error_is_none_when_the_correction_cancels() {
    let (stop, status, eta) = sample();
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let plain = estimate(&data, &eta, &RoystonOptions::default()).unwrap();
    let b = plain.beta.unwrap();
    // Choose r so that 1 + b^2 - r = 0, that is e / (e - p) = 1 + b^2, with p = 1.
    let e = (1.0 + b * b) / (b * b);
    let r = estimate(
        &data,
        &eta,
        &RoystonOptions {
            adjust: Some(Adjustment {
                n_events: e,
                n_coef: 1,
            }),
            ..RoystonOptions::default()
        },
    )
    .unwrap();
    // t is zero up to rounding, so D is tiny and the standard error is either
    // None (t exactly 0) or very large.
    assert!(r.d.unwrap().abs() < 1e-7);
    assert!(r.se_d.is_none_or(|s| s > 1e6));
}

#[test]
fn invalid_input_is_an_error() {
    let (stop, status, eta) = sample();
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let opts = RoystonOptions::default();
    assert!(estimate(&data, &eta[..39], &opts).is_err());
    let mut bad = eta.clone();
    bad[4] = f64::NAN;
    assert!(estimate(&data, &bad, &opts).is_err());
    for (n_events, n_coef) in [(3.0, 3), (2.0, 3), (f64::INFINITY, 1), (f64::NAN, 1)] {
        let o = RoystonOptions {
            adjust: Some(Adjustment { n_events, n_coef }),
            ..opts
        };
        assert!(estimate(&data, &eta, &o).is_err(), "{n_events} {n_coef}");
    }
    let two = [1, 2, 0, 1, 2, 1, 0, 2, 1, 1];
    let multi = SurvData::builder(&stop[..10], &two)
        .n_codes(2)
        .build()
        .unwrap();
    assert!(estimate(&multi, &eta[..10], &opts).is_err());
    let o = RoystonOptions {
        cause: Some(2),
        ..opts
    };
    assert!(estimate(&multi, &eta[..10], &o).is_ok());
}
