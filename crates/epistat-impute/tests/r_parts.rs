//! The deterministic parts of the methods, with the random inputs injected,
//! against mice 3.19.0 and nnet 7.3-21, as written by `tests/data/impute_parts.R`.
//!
//! Tolerances. The linear model is solved here by IRLS on the normal equations
//! and in R by QR, so `b` and `(X'X)^-1` differ by the rounding of the normal
//! equations, a relative `cond(X'X) * 2.2e-16`. The design has a condition
//! number of `X'X` below 10^3, so 1e-10 relative leaves a factor of about 10^3
//! of room. The draws are then a fixed linear map of these and share the
//! bound. The logistic fit is `glm.fit`'s own IRLS with the same stopping rule,
//! so the iterates agree and 1e-8 relative is the rule's `1e-8` on the
//! deviance. The augmentation rows are arithmetic on the data and agree to
//! 1e-12 relative. The multinomial model is compared to `multinom` run to
//! `reltol = 1e-14`, which fixes its coefficients to about 1e-7, so 1e-6.
//! Matching sets and the Nelson-Aalen hazard are exact up to the sum order
//! of the hazard, a few units in the last place: 1e-13 relative.

use epistat_impute::parts::{
    SortedDonors, augmentation, draw_coefficients, draw_linear, linear_fit, logistic_fit,
    multinomial_fit,
};
use epistat_impute::{Ties, nelson_aalen};
use epistat_linalg::Mat;
use epistat_testkit::{Tolerance, assert_close, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Versions {
    r: String,
    mice: String,
    nnet: String,
    survival: String,
}

#[derive(Deserialize)]
struct Norm {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    coef: Vec<f64>,
    v: Vec<Vec<f64>>,
    rss: f64,
    df: f64,
    chi: f64,
    z: Vec<f64>,
    sigma: f64,
    beta: Vec<f64>,
}

#[derive(Deserialize)]
struct AugCase {
    x_rows: Vec<Vec<f64>>,
    y_aug: Vec<f64>,
    weight: f64,
}

#[derive(Deserialize)]
struct AugBinary {
    aug: AugCase,
}

#[derive(Deserialize)]
struct Augment {
    x: Vec<Vec<f64>>,
    binary: AugBinary,
    nominal: AugBinary,
}

#[derive(Deserialize)]
struct Logreg {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    missing: Vec<usize>,
    coef: Vec<f64>,
    v: Vec<Vec<f64>>,
    z: Vec<f64>,
    beta: Vec<f64>,
}

#[derive(Deserialize)]
struct Polyreg {
    x: Vec<Vec<f64>>,
    y: Vec<usize>,
    missing: Vec<usize>,
    coef: Vec<Vec<f64>>,
    vcov: Vec<Vec<f64>>,
    classes: usize,
}

#[derive(Deserialize)]
struct PmmCase {
    donors: usize,
    eligible: Vec<Vec<usize>>,
}

#[derive(Deserialize)]
struct Pmm {
    donor_yhat: Vec<f64>,
    targets: Vec<f64>,
    cases: Vec<PmmCase>,
}

#[derive(Deserialize)]
struct Nelson {
    time: Vec<f64>,
    status: Vec<i32>,
    hazard: Vec<f64>,
    breslow: Vec<f64>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    norm: Norm,
    augment: Augment,
    logreg: Logreg,
    polyreg: Polyreg,
    pmm: Pmm,
    nelson_aalen: Nelson,
}

fn reference() -> Reference {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/impute_parts.json"
    ));
    assert!(r.versions.r.starts_with("R version"), "{}", r.versions.r);
    assert_eq!(r.versions.mice, "3.19.0");
    assert_eq!(r.versions.nnet, "7.3.21");
    assert!(!r.versions.survival.is_empty());
    r
}

fn flatten(rows: &[Vec<f64>]) -> Vec<f64> {
    rows.iter().flatten().copied().collect()
}

fn columns(rows: &[Vec<f64>]) -> Vec<Vec<f64>> {
    (0..rows[0].len())
        .map(|j| rows.iter().map(|r| r[j]).collect())
        .collect()
}

#[test]
fn norm_draw_matches_mice() {
    let r = reference().norm;
    let cols = columns(&r.x);
    let n = r.y.len();
    let x = Mat::from_fn(
        n,
        cols.len() + 1,
        |i, k| if k == 0 { 1.0 } else { cols[k - 1][i] },
    );
    let fit = linear_fit(x.as_ref(), &r.y, 0).unwrap();
    let tol = Tolerance::exact().rel(1e-10).abs(1e-12);
    assert_close(&fit.coefficients, &r.coef, tol);
    assert_close(&fit.covariance, &flatten(&r.v), tol);
    assert_close(&[fit.rss], &[r.rss], tol);
    assert!((fit.df - r.df).abs() == 0.0);
    let draw = draw_linear(&fit, r.chi, &r.z, 0).unwrap();
    assert_close(&[draw.sigma], &[r.sigma], tol);
    assert_close(&draw.beta, &r.beta, tol);
}

#[test]
fn augmentation_rows_match_mice() {
    let r = reference().augment;
    let cols = columns(&r.x);
    for (name, case, outcomes) in [
        ("binary", &r.binary.aug, vec![0.0, 1.0]),
        ("nominal", &r.nominal.aug, vec![0.0, 1.0, 2.0]),
    ] {
        let aug = augmentation(&cols, &outcomes).unwrap();
        let tol = Tolerance::exact().rel(1e-12).abs(1e-12);
        assert_close(&aug.x, &flatten(&case.x_rows), tol);
        let expected_y: Vec<f64> = if name == "nominal" {
            case.y_aug.iter().map(|v| v - 1.0).collect()
        } else {
            case.y_aug.clone()
        };
        assert_close(&aug.y, &expected_y, Tolerance::exact());
        assert_close(&[aug.weight], &[case.weight], tol);
    }
}

#[test]
fn logistic_draw_matches_mice() {
    let r = reference().logreg;
    let cols = columns(&r.x);
    let n = r.y.len();
    let obs: Vec<usize> = (0..n).filter(|i| !r.missing.contains(i)).collect();
    let y_obs: Vec<f64> = obs.iter().map(|&i| r.y[i]).collect();
    let aug = augmentation(&cols, &[0.0, 1.0]).unwrap();
    let p = cols.len();
    let x = Mat::from_fn(obs.len() + aug.rows, p + 1, |i, k| {
        if k == 0 {
            1.0
        } else if i < obs.len() {
            cols[k - 1][obs[i]]
        } else {
            aug.x[(i - obs.len()) * p + k - 1]
        }
    });
    let mut y = y_obs;
    y.extend(&aug.y);
    let mut w = vec![1.0; obs.len()];
    w.extend(vec![aug.weight; aug.rows]);
    let fit = logistic_fit(x.as_ref(), &y, &w, 0).unwrap();
    let tol = Tolerance::exact().rel(1e-8).abs(1e-10);
    assert_close(&fit.coefficients, &r.coef, tol);
    assert_close(&fit.covariance, &flatten(&r.v), tol);
    let beta = draw_coefficients(
        &fit.coefficients,
        &fit.active,
        &fit.covariance,
        &r.z,
        1.0,
        0,
    )
    .unwrap();
    assert_close(&beta, &r.beta, tol);
}

#[test]
fn multinomial_fit_matches_nnet() {
    let r = reference().polyreg;
    let cols = columns(&r.x);
    let n = r.y.len();
    let obs: Vec<usize> = (0..n).filter(|i| !r.missing.contains(i)).collect();
    let outcomes: Vec<f64> = (0..r.classes).map(|c| c as f64).collect();
    let aug = augmentation(&cols, &outcomes).unwrap();
    let p = cols.len();
    let x = Mat::from_fn(obs.len() + aug.rows, p + 1, |i, k| {
        if k == 0 {
            1.0
        } else if i < obs.len() {
            cols[k - 1][obs[i]]
        } else {
            aug.x[(i - obs.len()) * p + k - 1]
        }
    });
    let mut y: Vec<usize> = obs.iter().map(|&i| r.y[i]).collect();
    y.extend(aug.y.iter().map(|&v| v as usize));
    let mut w = vec![1.0; obs.len()];
    w.extend(vec![aug.weight; aug.rows]);
    let fit = multinomial_fit(x.as_ref(), &y, r.classes, &w).unwrap();
    let tol = Tolerance::exact().rel(1e-6).abs(1e-6);
    assert_close(&fit.coefficients, &flatten(&r.coef), tol);
    assert_close(
        &fit.covariance,
        &flatten(&r.vcov),
        Tolerance::exact().rel(1e-5).abs(1e-6),
    );
    assert!(fit.active.iter().all(|&a| a));
}

#[test]
fn matching_sets_match_mice() {
    let r = reference().pmm;
    let donors = SortedDonors::new(&r.donor_yhat, 0).unwrap();
    for case in &r.cases {
        for (target, expected) in r.targets.iter().zip(&case.eligible) {
            let range = donors.eligible(*target, case.donors);
            let mut got: Vec<usize> = range.clone().map(|pos| donors.donor_at(pos)).collect();
            got.sort_unstable();
            assert_eq!(&got, expected, "donors {} target {target}", case.donors);
            // A uniform near 0 and near 1 pick the first and last eligible donor.
            assert_eq!(
                donors.pick(*target, case.donors, 0.0),
                donors.donor_at(range.start)
            );
            assert_eq!(
                donors.pick(*target, case.donors, 1.0 - 1e-16),
                donors.donor_at(range.end - 1)
            );
        }
    }
}

#[test]
fn nelson_aalen_matches_mice() {
    let r = reference().nelson_aalen;
    let h = nelson_aalen(&r.time, &r.status, Ties::Efron).unwrap();
    assert_close(&h, &r.hazard, Tolerance::exact().rel(1e-13));
    let b = nelson_aalen(&r.time, &r.status, Ties::Breslow).unwrap();
    assert_close(&b, &r.breslow, Tolerance::exact().rel(1e-13));
    assert!(h.iter().zip(&b).any(|(e, b)| e != b), "the data have ties");
}
