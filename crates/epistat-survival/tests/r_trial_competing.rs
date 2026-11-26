//! `fit_msm_competing` and `cumulative_incidence_competing` against the models
//! and the stacked sandwich written out in R, `data/trial_competing.R`.
//!
//! # The tolerances
//!
//! The two logistic models are fitted by IRLS from the same start with the same
//! stopping rule as `glm`, so a coefficient differs by `8 kappa eps` relative to
//! `max(|c|, 1)`, with `kappa` below `10^3` for the design here (the time
//! squared runs to 25): `2e-12`, checked to `1e-10`. The diagonal blocks of the
//! covariance are those of `vcovCL`, which the module's own tests check to the
//! same order, and the cross block is formed from the coefficients, so a
//! relative error of the coefficients of `2e-12` carries to it at most with a
//! factor of the condition number, `2e-9`, checked to `1e-8` relative to the
//! largest entry. The risks are smooth functions of the coefficients with
//! derivatives below 1, so they agree to `1e-10`. The standard errors of R come
//! from a gradient by central differences with step `h = 1e-6`: its truncation
//! error is `h^2 / 6` times a third derivative of order 1, `2e-13`, and its
//! rounding error `eps / h = 2e-10` against gradients of order `0.1`, so the
//! standard errors agree to `1e-8` relative; they are checked to `1e-7`.

use epistat_linalg::Mat;
use epistat_survival::weighting::trials::{
    MsmInput, cumulative_incidence_competing, fit_msm, fit_msm_competing,
};
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Rows {
    id: Vec<u64>,
    a: Vec<f64>,
    t: Vec<f64>,
    x: Vec<f64>,
    w: Vec<f64>,
    y: Vec<u8>,
    d: Vec<u8>,
}

#[derive(Deserialize)]
struct Arm {
    risk: Vec<f64>,
    se: Vec<f64>,
    competing_risk: Vec<f64>,
    competing_se: Vec<f64>,
}

#[derive(Deserialize)]
struct Reference {
    rows: Rows,
    event_coef: Vec<f64>,
    competing_coef: Vec<f64>,
    event_vcov: Vec<f64>,
    competing_vcov: Vec<f64>,
    cross: Vec<f64>,
    arm0: Arm,
    arm1: Arm,
}

fn worst(a: &[f64], b: &[f64], scale: f64) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs() / scale)
        .fold(0.0, f64::max)
}

fn largest(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0, f64::max)
}

#[test]
fn the_two_models_and_the_risks_agree_with_r() {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/trial_competing.json"
    ));
    let d = &r.rows;
    let n = d.id.len();
    let design = Mat::from_fn(n, 5, |i, j| [1.0, d.a[i], d.t[i], d.t[i] * d.t[i], d.x[i]][j]);
    let y: Vec<bool> = d.y.iter().map(|&v| v == 1).collect();
    let competing: Vec<bool> = d.d.iter().map(|&v| v == 1).collect();
    let input = MsmInput {
        design: design.as_ref(),
        outcome: &y,
        weight: &d.w,
        person: &d.id,
    };
    let fit = fit_msm_competing(&input, &competing).expect("fits");
    assert!(worst(&fit.event.coefficients, &r.event_coef, 1.0) < 1e-10);
    assert!(worst(&fit.competing.coefficients, &r.competing_coef, 1.0) < 1e-10);
    assert!(worst(&fit.event.covariance, &r.event_vcov, largest(&r.event_vcov)) < 1e-8);
    assert!(
        worst(
            &fit.competing.covariance,
            &r.competing_vcov,
            largest(&r.competing_vcov)
        ) < 1e-8
    );
    assert!(worst(&fit.cross_covariance, &r.cross, largest(&r.cross)) < 1e-8);
    // The competing model is fit_msm on every row with the competing event as
    // the outcome.
    let alone = fit_msm(&MsmInput {
        outcome: &competing,
        ..input
    })
    .expect("fit");
    assert_eq!(alone, fit.competing);

    let base: Vec<usize> = (0..n).filter(|&i| d.t[i] == 0.0).collect();
    for (arm, want) in [(0.0, &r.arm0), (1.0, &r.arm1)] {
        let got = cumulative_incidence_competing(&fit, base.len(), 6, &|k, time, out| {
            let t = f64::from(u32::try_from(time).expect("small"));
            out.copy_from_slice(&[1.0, arm, t, t * t, d.x[base[k]]]);
        })
        .expect("risks");
        assert!(worst(&got.risk, &want.risk, 1.0) < 1e-10, "arm {arm}");
        assert!(worst(&got.competing_risk, &want.competing_risk, 1.0) < 1e-10);
        for (g, w) in got.std_err.iter().zip(&want.se) {
            assert!((g - w).abs() / w < 1e-7, "arm {arm}: {g} against {w}");
        }
        for (g, w) in got.competing_std_err.iter().zip(&want.competing_se) {
            assert!((g - w).abs() / w < 1e-7, "arm {arm}: {g} against {w}");
        }
    }
}

#[test]
fn a_row_with_both_events_is_an_error() {
    let design = Mat::from_fn(4, 1, |_, _| 1.0);
    let y = [true, false, false, false];
    let d = [true, false, true, false];
    let w = [1.0; 4];
    let person = [1, 1, 2, 2];
    assert!(
        fit_msm_competing(
            &MsmInput {
                design: design.as_ref(),
                outcome: &y,
                weight: &w,
                person: &person,
            },
            &d
        )
        .is_err()
    );
}
