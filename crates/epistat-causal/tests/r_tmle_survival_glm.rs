//! `tmle_survival_glm`, which fits the event, censoring and propensity models
//! itself, against the glm fits of `data/tmle_survival.R` and the estimator
//! written out there.
//!
//! # The tolerances
//!
//! The models are fitted by IRLS from the same start with the same stopping
//! rule as `glm`, so a coefficient differs by `8 kappa eps` relative to
//! `max(|c|, 1)`, with `kappa` below `10^3` for the designs here (an intercept,
//! interval indicators, two covariates and the treatment), and a linear
//! predictor of at most nine terms of size at most 3 by `9 * 3 * 8e3 eps =
//! 5e-11`. A predicted probability then differs by a quarter of that, and is
//! checked to `1e-10`. The estimator is a smooth map of the predictions, whose
//! derivatives are bounded by the inverse of the smallest `g_a G(k - 1)`
//! (above 0.05 here) times the horizon, at most 120, so the reported numbers
//! agree with those of R to `120 * 1.3e-11 = 1.6e-9` on top of the tolerance of
//! the estimator itself (`r_tmle_survival.rs`, below `1e-12` here); they are
//! checked to `1e-8`.

use epistat_causal::tmle::PropensityBound;
use epistat_causal::tmle::survival::{
    IntervalDesign, SurvivalGlmInput, SurvivalInput, SurvivalOptions, tmle_survival,
    tmle_survival_glm,
};
use epistat_linalg::Mat;
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Scenario {
    name: String,
    horizon: usize,
    bound: f64,
    weighted: bool,
    weights: Vec<f64>,
    cluster: Option<Vec<u32>>,
    hazard1: Vec<Vec<f64>>,
    hazard0: Vec<Vec<f64>>,
    censoring1: Vec<Vec<f64>>,
    censoring0: Vec<Vec<f64>>,
    propensity: Vec<f64>,
    numbers: Vec<f64>,
    iterations: Vec<usize>,
}

#[derive(Deserialize)]
struct Reference {
    time: Vec<usize>,
    event: Vec<bool>,
    treatment: Vec<u8>,
    x1: Vec<f64>,
    x2: Vec<f64>,
    scenarios: Vec<Scenario>,
}

/// The number of intervals of the data.
const K: usize = 6;

#[test]
fn the_fitted_models_and_the_estimator_agree_with_r() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/tmle_survival.json"
    ));
    let treatment: Vec<bool> = r.treatment.iter().map(|&a| a == 1).collect();
    let n = treatment.len();
    // glm's factor(k, levels = 1:K) + x1 + x2 + a, and 1:(K - 1) for censoring.
    let design = |levels: usize| {
        let (x1, x2) = (&r.x1, &r.x2);
        move |i: usize, k: usize, a: bool, out: &mut [f64]| {
            out.fill(0.0);
            out[0] = 1.0;
            if k >= 2 {
                out[k - 1] = 1.0;
            }
            out[levels] = x1[i];
            out[levels + 1] = x2[i];
            out[levels + 2] = f64::from(u8::from(a));
        }
    };
    let (event_fill, censoring_fill) = (design(K), design(K - 1));
    let z = Mat::from_fn(n, 3, |i, j| [1.0, r.x1[i], r.x2[i]][j]);
    for s in &r.scenarios {
        let opts = SurvivalOptions {
            propensity_bound: if s.bound > 0.0 {
                PropensityBound::At(s.bound)
            } else {
                PropensityBound::None
            },
            ..SurvivalOptions::new(s.horizon)
        };
        let input = SurvivalGlmInput {
            time: &r.time,
            event: &r.event,
            treatment: &treatment,
            hazard: IntervalDesign {
                columns: K + 3,
                fill: &event_fill,
            },
            censoring: Some(IntervalDesign {
                columns: K + 2,
                fill: &censoring_fill,
            }),
            propensity: z.as_ref(),
            weight: s.weighted.then_some(s.weights.as_slice()),
            cluster: s.cluster.as_deref(),
        };
        let got = tmle_survival_glm(&input, &opts).expect("fits");
        let t = s.horizon;
        for i in 0..n {
            for k in 0..t {
                for (m, want) in [(&got.hazard1, &s.hazard1), (&got.hazard0, &s.hazard0)] {
                    assert!((m[(i, k)] - want[i][k]).abs() < 1e-10, "{} hazard", s.name);
                }
            }
            for k in 0..t - 1 {
                for (m, want) in [
                    (&got.censoring1, &s.censoring1),
                    (&got.censoring0, &s.censoring0),
                ] {
                    assert!(
                        (m[(i, k)] - want[i][k]).abs() < 1e-10,
                        "{} censoring",
                        s.name
                    );
                }
            }
            assert!((got.propensity[i] - s.propensity[i]).abs() < 1e-10);
        }
        // The estimator is tmle_survival on the fitted predictions.
        let alone = tmle_survival(
            &SurvivalInput {
                time: &r.time,
                event: &r.event,
                treatment: &treatment,
                hazard1: got.hazard1.as_ref(),
                hazard0: got.hazard0.as_ref(),
                censoring1: got.censoring1.as_ref(),
                censoring0: got.censoring0.as_ref(),
                propensity: &got.propensity,
                weight: input.weight,
                cluster: input.cluster,
            },
            &opts,
        )
        .expect("estimator");
        assert_eq!(format!("{alone:?}"), format!("{:?}", got.tmle));
        assert_eq!(got.tmle.iterations.to_vec(), s.iterations, "{}", s.name);
        let ratio = got.tmle.ratio.expect("a ratio");
        let values = [
            got.tmle.risk1.estimate,
            got.tmle.risk0.estimate,
            got.tmle.difference.estimate,
            ratio.estimate,
            got.tmle.risk1.se,
            got.tmle.risk0.se,
            got.tmle.difference.se,
            ratio.se,
        ];
        for (v, want) in values.iter().zip(&s.numbers) {
            assert!(
                (v - want).abs() < 1e-8,
                "{}: {v} against {want}",
                s.name
            );
        }
    }
}

#[test]
fn censoring_without_a_model_is_an_error() {
    let time = [1, 2, 3];
    let event = [false, true, false];
    let treatment = [true, false, true];
    let fill = |_: usize, _: usize, _: bool, out: &mut [f64]| out[0] = 1.0;
    let z = Mat::from_fn(3, 1, |_, _| 1.0);
    let input = SurvivalGlmInput {
        time: &time,
        event: &event,
        treatment: &treatment,
        hazard: IntervalDesign {
            columns: 1,
            fill: &fill,
        },
        censoring: None,
        propensity: z.as_ref(),
        weight: None,
        cluster: None,
    };
    assert!(tmle_survival_glm(&input, &SurvivalOptions::new(2)).is_err());
}
