//! `attributable_logistic`: what it refuses and the results that are not
//! numbers, on small fits.

use epistat_glm::attributable::{AfOptions, Design, IntervalScale, attributable_logistic};
use epistat_core::Level;
use epistat_glm::{Error, Family, Glm, Link};
use epistat_inference::Clusters;
use epistat_testkit::{BadInput, assert_rejects_bad_input};
use epistat_linalg::Mat;

mod common;
use common::*;

/// Exposure `x` and confounder `z` over 400 rows with a fixed pattern.
fn data(protective: bool) -> (Mat<f64>, Mat<f64>, Vec<f64>) {
    let n = 400;
    let x = Mat::from_fn(n, 3, |i, j| match j {
        0 => 1.0,
        1 => f64::from(u8::from(i % 3 == 0)),
        _ => ((i * 7919) % 101) as f64 / 50.0 - 1.0,
    });
    let x0 = Mat::from_fn(n, 3, |i, j| if j == 1 { 0.0 } else { x[(i, j)] });
    let y = (0..n)
        .map(|i| {
            let e = x[(i, 1)];
            let z = x[(i, 2)];
            let eta = -0.5 + if protective { -0.9 } else { 0.9 } * e + 0.4 * z;
            // A fixed quasi-random draw.
            let u = ((i * 2_654_435_761) % 1000) as f64 / 1000.0;
            f64::from(u8::from(u < 1.0 / (1.0 + (-eta).exp())))
        })
        .collect();
    (x, x0, y)
}

fn fitted(x: &Mat<f64>, y: &[f64]) -> epistat_glm::Fit {
    Glm::new(Family::Binomial, Link::Logit)
        .fit5(x.as_ref(), y, None, None, None)
        .unwrap()
}

fn run(
    fit: &epistat_glm::Fit,
    x: &Mat<f64>,
    x0: &Mat<f64>,
    y: &[f64],
    weights: Option<&[f64]>,
    clusters: Option<&Clusters>,
    options: &AfOptions,
) -> Result<epistat_glm::attributable::Attributable, Error> {
    let data = glm_data(x.as_ref(), y, weights, None, None)?;
    attributable_logistic(fit, &data, x0.as_ref(), clusters, options)
}

#[test]
fn a_harmful_exposure_gives_a_positive_fraction_and_a_protective_one_a_negative() {
    for (protective, sign) in [(false, 1.0), (true, -1.0)] {
        let (x, x0, y) = data(protective);
        let fit = fitted(&x, &y);
        let a = run(&fit, &x, &x0, &y, None, None, &AfOptions::default())
            .unwrap()
            .af
            .unwrap();
        assert!(a.estimate * sign > 0.0);
        assert!(a.se.unwrap() > 0.0);
        assert!(a.lower().unwrap() < a.estimate && a.estimate < a.upper().unwrap());
        assert_eq!(a.interval.unwrap().level, Level::default());
    }
}

#[test]
fn nobody_exposed_gives_zero_and_no_events_gives_nothing() {
    let (x, _, y) = data(false);
    let fit = fitted(&x, &y);
    let same = run(&fit, &x, &x, &y, None, None, &AfOptions::default()).unwrap();
    assert!(same.af.unwrap().estimate.abs() < 1e-12);
    let zeros = vec![0.0; y.len()];
    let none_fit = Glm::new(Family::Binomial, Link::Logit)
        .fit5(x.as_ref(), &y, Some(&zeros), None, None)
        .unwrap();
    let a = run(&none_fit, &x, &x, &y, Some(&zeros), None, &AfOptions::default()).unwrap();
    assert!(a.af.is_none());
    assert_eq!(a.n_cases, 0);
}

#[test]
fn an_exposure_effect_of_one_sided_size_keeps_the_log_limits_below_one() {
    let (x, x0, y) = data(false);
    let fit = fitted(&x, &y);
    let options = AfOptions {
        design: Design::CaseControl,
        scale: IntervalScale::LogComplement,
        level: Level::confidence(0.999_999).unwrap(),
    };
    let a = run(&fit, &x, &x0, &y, None, None, &options).unwrap();
    assert!(a.af.unwrap().upper().unwrap() < 1.0);
}

#[test]
fn unusable_input_is_refused() {
    let (x, x0, y) = data(false);
    let fit = fitted(&x, &y);
    let d = AfOptions::default();
    assert!(run(&fit, &x, &x0, &y[..399], None, None, &d).is_err());
    let narrow = Mat::from_fn(400, 2, |i, j| x0[(i, j)]);
    assert!(run(&fit, &x, &narrow, &y, None, None, &d).is_err());
    let mut bad_y = y.clone();
    bad_y[3] = 0.5;
    assert!(run(&fit, &x, &x0, &bad_y, None, None, &d).is_err());
    let negative = vec![-1.0; 400];
    assert!(run(&fit, &x, &x0, &y, Some(&negative), None, &d).is_err());
    assert!(run(&fit, &x, &x0, &y, Some(&negative[..10]), None, &d).is_err());
    let two = Clusters::from_labels(&[1, 2]);
    assert!(run(&fit, &x, &x0, &y, None, Some(&two), &d).is_err());
    // A fit with another link is not logistic.
    let probit = Glm::new(Family::Binomial, Link::Probit)
        .fit5(x.as_ref(), &y, None, None, None)
        .unwrap();
    assert!(run(&probit, &x, &x0, &y, None, None, &d).is_err());
}

#[test]
fn bad_columns_are_input_errors() {
    let (x, x0, y) = data(false);
    let fit = fitted(&x, &y);
    assert_rejects_bad_input("attributable_logistic", |bad| {
        Some(match bad {
            BadInput::Unsorted => return None,
            _ => run(&fit, &x, &x0, &bad.column(&y), None, None, &AfOptions::default()),
        })
    });
}

#[test]
fn one_cluster_has_no_standard_error() {
    let (x, x0, y) = data(false);
    let fit = fitted(&x, &y);
    let one = Clusters::from_labels(&vec![1_u32; 400]);
    let a = run(&fit, &x, &x0, &y, None, Some(&one), &AfOptions::default())
        .unwrap()
        .af
        .unwrap();
    assert!(a.se.is_none() && a.lower().is_none());
}
