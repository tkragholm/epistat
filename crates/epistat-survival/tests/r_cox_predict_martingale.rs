//! `predict(fit, type = "expected")` of the fitted data from survival 3.8-12,
//! the event indicator less the martingale residual, against
//! [`predict_expected_martingale`], on the fits of `cox_predict.json`: lung and
//! simulated counting-process data with strata, weights and an offset, with
//! Breslow and Efron ties.
//!
//! The coefficients are R's, so what is compared is the prediction.
//!
//! # Tolerance
//!
//! The value is `status - m` for a martingale residual `m = status - theta H`,
//! where `H` is a sum of at most `n` increments each of a risk-set sum, and
//! `theta` a risk score with the relative error `(p + 2 + xb) eps` of
//! `r_cox_predict.rs`. The bound is `8 eps (n + p + 4 + xb)` in units of
//! `max(|value|, 1)`, where `xb` is the largest `sum_a 2 |x_a beta_a|` plus the
//! largest offset. It is not changed after a run.

use epistat_survival::SurvData;
use epistat_survival::cox::predict::{
    FittedModel, PredictOptions, PredictType, predict, predict_expected_martingale,
};
use epistat_survival::cox::{CoxOptions, Ties, fit};
use epistat_testkit::load_json;
use faer::Mat;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    fits: Vec<RFit>,
}

#[derive(Deserialize)]
#[expect(
    clippy::struct_excessive_bools,
    reason = "one flag per option of the R call"
)]
struct RFit {
    name: String,
    ties: String,
    entry: bool,
    strata: bool,
    weights: bool,
    offset: bool,
    x: Vec<Vec<f64>>,
    coefficients: Vec<f64>,
    naive_var: Vec<Vec<f64>>,
    start: Vec<f64>,
    stop: Vec<f64>,
    event: Vec<i32>,
    stratum: Vec<u32>,
    weight: Vec<f64>,
    offset_values: Vec<f64>,
    expected: RExpected,
}

#[derive(Deserialize)]
struct RExpected {
    fit: Vec<f64>,
}

#[test]
fn martingale_form_matches_r() {
    let reference: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/cox_predict.json"
    ));
    let mut worst = 0.0f64;
    let mut differing = 0;
    for r in &reference.fits {
        let n = r.stop.len();
        let p = r.coefficients.len();
        let x = Mat::from_fn(n, p, |i, j| r.x[i][j]);
        let mut builder = SurvData::builder(&r.stop, &r.event);
        if r.entry {
            builder = builder.start(&r.start);
        }
        if r.strata {
            builder = builder.stratum(&r.stratum);
        }
        if r.weights {
            builder = builder.weight(&r.weight);
        }
        let data = builder.build().unwrap();
        let opts = CoxOptions {
            ties: if r.ties == "efron" {
                Ties::Efron
            } else {
                Ties::Breslow
            },
            offset: r.offset.then_some(r.offset_values.as_slice()),
            ..CoxOptions::default()
        };
        let mut model = fit(&data, x.as_ref(), &opts).unwrap();
        model.coefficients = r.coefficients.iter().copied().map(Some).collect();
        model.covariance = r.naive_var.iter().flatten().copied().map(Some).collect();
        let fitted = FittedModel {
            data: &data,
            x: x.as_ref(),
            fit: &model,
            options: &opts,
        };
        let got = predict_expected_martingale(&fitted).unwrap();
        let mut xb = 0.0f64;
        for a in 0..p {
            let big = r.x.iter().map(|row| row[a].abs()).fold(0.0, f64::max);
            xb += 2.0 * big * r.coefficients[a].abs();
        }
        if r.offset {
            xb += r.offset_values.iter().map(|o| o.abs()).fold(0.0, f64::max);
        }
        let bound = 8.0 * f64::EPSILON * (n as f64 + p as f64 + 4.0 + xb);
        for (i, (g, w)) in got.iter().zip(&r.expected.fit).enumerate() {
            let dev = (g - w).abs() / (bound * w.abs().max(1.0));
            assert!(dev <= 1.0, "{} row {i}: {g} vs {w}", r.name);
            worst = worst.max(dev);
        }
        // The default form differs from R's exactly at Efron-tied events.
        let default = predict(
            &fitted,
            PredictType::Expected,
            None,
            &PredictOptions::default(),
        )
        .unwrap();
        let differs = default
            .fit
            .iter()
            .zip(&r.expected.fit)
            .any(|(d, w)| (d - w).abs() > 1e-6);
        if r.ties == "breslow" {
            assert!(!differs, "{}: the forms agree with Breslow ties", r.name);
        }
        differing += usize::from(differs);
    }
    assert!(differing > 0, "some fit has Efron-tied events");
    eprintln!("worst deviation: {worst:.2e} of its bound");
}
