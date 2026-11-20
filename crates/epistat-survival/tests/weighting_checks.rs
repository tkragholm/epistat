//! The invalid input of the weighting entry points is an error, and the edge
//! cases give the values they document.

#![expect(
    clippy::float_cmp,
    reason = "a weight of exactly 1 and a curve of exactly 0 or 1 are the documented values"
)]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the inputs have a few dozen rows, far below the limits of the casts"
)]

use epistat_survival::SurvData;
use epistat_survival::cox::CoxOptions;
use epistat_survival::weighting::cox as wcox;
use epistat_survival::weighting::iptw::{self, Estimand, IptwOptions};
use epistat_survival::weighting::km::{self, KmOptions};
use faer::Mat;

fn design(n: usize) -> (Mat<f64>, Vec<bool>) {
    let z = Mat::from_fn(n, 2, |i, j| {
        if j == 0 {
            1.0
        } else {
            f64::from((i * 7 % 11) as u32) / 5.0
        }
    });
    let treat = (0..n).map(|i| (i * 3 + i / 4) % 5 < 2).collect();
    (z, treat)
}

#[test]
fn propensity_model_input_is_checked() {
    let (z, treat) = design(40);
    let ok = IptwOptions::default();
    assert!(iptw::fit(&treat, z.as_ref(), &ok).is_ok());
    let short = Mat::from_fn(39, 2, |_, _| 1.0);
    assert!(iptw::fit(&treat, short.as_ref(), &ok).is_err());
    let all = vec![true; 40];
    assert!(iptw::fit(&all, z.as_ref(), &ok).is_err());
    let att = IptwOptions {
        estimand: Estimand::Att,
        stabilise: true,
    };
    assert!(iptw::fit(&treat, z.as_ref(), &att).is_err());
    let aliased = Mat::from_fn(
        40,
        3,
        |i, j| if j == 2 { z[(i, 1)] * 2.0 } else { z[(i, j)] },
    );
    assert!(iptw::fit(&treat, aliased.as_ref(), &ok).is_err());
    let nan = Mat::from_fn(40, 2, |i, j| if i == 3 { f64::NAN } else { z[(i, j)] });
    assert!(iptw::fit(&treat, nan.as_ref(), &ok).is_err());
}

#[test]
fn weights_of_each_estimand() {
    let (z, treat) = design(60);
    let ate = iptw::fit(&treat, z.as_ref(), &IptwOptions::default()).unwrap();
    for ((&w, &p), &a) in ate.weights.iter().zip(&ate.propensity).zip(&treat) {
        let expected = if a { 1.0 / p } else { 1.0 / (1.0 - p) };
        assert!((w - expected).abs() <= 8.0 * f64::EPSILON * expected);
    }
    let att = iptw::fit(
        &treat,
        z.as_ref(),
        &IptwOptions {
            estimand: Estimand::Att,
            stabilise: false,
        },
    )
    .unwrap();
    assert!(
        att.weights
            .iter()
            .zip(&treat)
            .all(|(&w, &a)| !a || w == 1.0)
    );
    let atc = iptw::fit(
        &treat,
        z.as_ref(),
        &IptwOptions {
            estimand: Estimand::Atc,
            stabilise: false,
        },
    )
    .unwrap();
    assert!(atc.weights.iter().zip(&treat).all(|(&w, &a)| a || w == 1.0));
    // The weights of the treated and of the controls each sum to the size of the sample in the
    // average treatment effect, as the sums of the inverse of a propensity score do.
    let summary = ate.summary;
    assert!(summary.treated.unwrap().effective_sample_size < summary.treated.unwrap().n as f64);
}

#[test]
fn outcome_input_is_checked() {
    let (z, treat) = design(30);
    let w = iptw::fit(&treat, z.as_ref(), &IptwOptions::default()).unwrap();
    let stop: Vec<f64> = (0..30)
        .map(|i| 1.0 + f64::from((i * 5 % 13) as u32))
        .collect();
    let status: Vec<i32> = (0..30).map(|i| i32::from(i % 3 != 0)).collect();
    let x = Mat::from_fn(30, 1, |i, _| f64::from(u8::from(treat[i])));
    let data = SurvData::builder(&stop, &status).build().unwrap();
    assert!(wcox::fit(&data, x.as_ref(), &CoxOptions::default(), &w, None).is_ok());
    // Weights already on the data.
    let weights = vec![1.0; 30];
    let weighted = SurvData::builder(&stop, &status)
        .weight(&weights)
        .build()
        .unwrap();
    assert!(wcox::fit(&weighted, x.as_ref(), &CoxOptions::default(), &w, None).is_err());
    // A subject map of the wrong length, or naming a missing subject.
    assert!(wcox::fit(&data, x.as_ref(), &CoxOptions::default(), &w, Some(&[0, 1])).is_err());
    let beyond: Vec<u32> = (0..30).map(|_| 30).collect();
    assert!(wcox::fit(&data, x.as_ref(), &CoxOptions::default(), &w, Some(&beyond)).is_err());
    // Rows are not subjects.
    let few = SurvData::builder(&stop[..20], &status[..20])
        .build()
        .unwrap();
    let x20 = Mat::from_fn(20, 1, |i, _| x[(i, 0)]);
    assert!(wcox::fit(&few, x20.as_ref(), &CoxOptions::default(), &w, None).is_err());
    // A status with several codes needs a cause.
    let multi: Vec<i32> = (0..30).map(|i| i % 3).collect();
    let competing = SurvData::builder(&stop, &multi).n_codes(2).build().unwrap();
    assert!(km::estimate(&competing, &[3.0], &KmOptions::default(), &w, None).is_err());
    assert!(
        km::estimate(
            &competing,
            &[3.0],
            &KmOptions {
                cause: Some(2),
                ..KmOptions::default()
            },
            &w,
            None
        )
        .is_ok()
    );
    assert!(
        km::estimate(
            &competing,
            &[3.0],
            &KmOptions {
                cause: Some(3),
                ..KmOptions::default()
            },
            &w,
            None
        )
        .is_err()
    );
    assert!(km::estimate(&data, &[f64::NAN], &KmOptions::default(), &w, None).is_err());
}

#[test]
fn a_curve_at_zero_has_no_variance() {
    let (z, treat) = design(60);
    let w = iptw::fit(&treat, z.as_ref(), &IptwOptions::default()).unwrap();
    // Every row has the event, so the curve reaches 0 at the last time.
    let stop: Vec<f64> = (1..=60).map(f64::from).collect();
    let status = vec![1; 60];
    let data = SurvData::builder(&stop, &status).build().unwrap();
    let curves = km::estimate(&data, &[0.5, 6.0, 60.0], &KmOptions::default(), &w, None).unwrap();
    assert_eq!(curves.surv[0], 1.0);
    assert_eq!(curves.standard_error(0, 0), Some(0.0));
    assert!(curves.standard_error(0, 1).is_some());
    assert_eq!(curves.surv[2], 0.0);
    assert!(curves.standard_error(0, 2).is_none());
    assert!(curves.standard_error_known(0, 2).is_none());
}
