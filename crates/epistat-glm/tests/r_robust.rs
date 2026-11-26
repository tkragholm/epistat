//! Sandwich covariances and risk regression checked against R's `glm` with
//! `sandwich::vcovHC` and `vcovCL` (sandwich 3.1.3).

use epistat_core::Covariance;
use epistat_glm::{Family, Glm, Link, RiskScale, SandwichType, risk_regression, robust_covariance};
use epistat_inference::Clusters;
use epistat_linalg::Mat;
use epistat_testkit::{assert_same_on_threads, load_json};
use serde::Deserialize;

mod common;
use common::*;

#[derive(Deserialize)]
struct Reference {
    cases: Vec<Case>,
    difference: Difference,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    cluster: Option<Vec<usize>>,
    coef: Vec<Option<f64>>,
    hc0: Vec<Option<f64>>,
    hc1: Vec<Option<f64>>,
    hc3: Vec<Option<f64>>,
    cl_hc1: Option<Vec<Option<f64>>>,
    cl_hc0: Option<Vec<Option<f64>>>,
}

#[derive(Deserialize)]
struct Difference {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    cluster: Vec<usize>,
    coef: Vec<f64>,
    hc0: Vec<Option<f64>>,
    hc1: Vec<Option<f64>>,
    hc3: Vec<Option<f64>>,
    cl_hc1: Vec<Option<f64>>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/robust.json"
    ))
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

/// The largest deviation over the largest entry of the reference.
fn worst(got: &Covariance, want: &[Option<f64>], label: &str) -> f64 {
    let got = got.as_slice();
    assert_eq!(got.len(), want.len(), "{label}");
    let scale = want.iter().flatten().fold(0.0_f64, |a, &b| a.max(b.abs()));
    let mut worst = 0.0_f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        match (g, w) {
            (None, None) => {}
            (Some(g), Some(w)) => worst = worst.max((g - w).abs() / scale),
            _ => panic!("{label} entry {i}: {g:?} against {w:?}"),
        }
    }
    worst
}

#[test]
fn modified_poisson_matches_sandwich() {
    let mut overall = 0.0_f64;
    for case in &reference().cases {
        let x = matrix(&case.x);
        let model = Glm::new(Family::Poisson, Link::Log);
        let fit = model
            .fit5(
                x.as_ref(),
                &case.y,
                case.weights.as_deref(),
                case.offset.as_deref(),
                None,
            )
            .unwrap();
        for (j, want) in case.coef.iter().enumerate() {
            assert_eq!(
                fit.coefficients[j].is_some(),
                want.is_some(),
                "{}",
                case.name
            );
        }
        let clusters = case.cluster.as_ref().map(|c| Clusters::from_labels(c));
        // The bread and the meat are sums of at most 1589 terms over the same
        // rows as R's, in a fixed order, and both fits stop at the same
        // iteration, so the covariances agree to the rounding of the sums,
        // about 1e-14 of the largest entry; 1e-12 allows for the conditioning of
        // the bread.
        let bound = 1e-12;
        let none = |kind| robust_covariance(&model, &fit, x.as_ref(), &case.y, None, kind).unwrap();
        for (kind, want, label) in [
            (SandwichType::Hc0, &case.hc0, "hc0"),
            (SandwichType::Hc1, &case.hc1, "hc1"),
            (SandwichType::Hc3, &case.hc3, "hc3"),
        ] {
            let d = worst(&none(kind), want, &format!("{} {label}", case.name));
            assert!(d <= bound, "{} {label}: {d:e}", case.name);
            overall = overall.max(d);
        }
        if let Some(clusters) = &clusters {
            for (kind, want, label) in [
                (
                    SandwichType::Hc1,
                    case.cl_hc1.as_ref().unwrap(),
                    "cluster hc1",
                ),
                (
                    SandwichType::Hc0,
                    case.cl_hc0.as_ref().unwrap(),
                    "cluster hc0",
                ),
            ] {
                let got =
                    robust_covariance(&model, &fit, x.as_ref(), &case.y, Some(clusters), kind)
                        .unwrap();
                let d = worst(&got, want, &format!("{} {label}", case.name));
                assert!(d <= bound, "{} {label}: {d:e}", case.name);
                overall = overall.max(d);
            }
        }
    }
    eprintln!("worst deviation over the largest entry: {overall:e}");
}

#[test]
fn risk_regression_is_modified_poisson_with_its_diagnostics() {
    let reference = reference();
    let case = &reference.cases[1];
    let x = matrix(&case.x);
    let clusters = Clusters::from_labels(case.cluster.as_ref().unwrap());
    let fit = risk_regression(
        RiskScale::Ratio,
        &glm_data(x.as_ref(), &case.y, None, None, None).unwrap(),
        Some(&clusters),
        SandwichType::Hc1,
    )
    .unwrap();
    let d = worst(
        &fit.robust_covariance,
        case.cl_hc1.as_ref().unwrap(),
        "ratio",
    );
    assert!(d <= 1e-9, "{d:e}");
    assert_eq!(
        fit.n_fitted_above_one,
        fit.fit.fitted.iter().filter(|&&m| m > 1.0).count()
    );
    assert!(fit.max_fitted > 0.0);
    let bad = risk_regression(
        RiskScale::Ratio,
        &glm_data(x.as_ref(), &[2.0; 1589], None, None, None).unwrap(),
        None,
        SandwichType::Hc0,
    );
    assert!(bad.is_err());
    // HC3 is not defined with clusters.
    let model = Glm::new(Family::Poisson, Link::Log);
    let plain = model.fit5(x.as_ref(), &case.y, None, None, None).unwrap();
    assert!(
        robust_covariance(
            &model,
            &plain,
            x.as_ref(),
            &case.y,
            Some(&clusters),
            SandwichType::Hc3
        )
        .is_err()
    );
}

#[test]
fn risk_difference_matches_binomial_identity() {
    let d = &reference().difference;
    let x = matrix(&d.x);
    let clusters = Clusters::from_labels(&d.cluster);
    for (kind, want, with_clusters) in [
        (SandwichType::Hc0, &d.hc0, false),
        (SandwichType::Hc1, &d.hc1, false),
        (SandwichType::Hc3, &d.hc3, false),
        (SandwichType::Hc1, &d.cl_hc1, true),
    ] {
        let fit = risk_regression(
        RiskScale::Difference,
        &glm_data(x.as_ref(), &d.y, None, None, None).unwrap(),
        with_clusters.then_some(&clusters),
        kind,
    )
        .unwrap();
        for (j, &c) in d.coef.iter().enumerate() {
            assert!(
                (fit.fit.coefficients[j].unwrap() - c).abs() <= 1e-8,
                "coefficient {j}"
            );
        }
        let dev = worst(
            &fit.robust_covariance,
            want,
            &format!("{kind:?} {with_clusters}"),
        );
        // Both fits stop at a deviance change below 1e-8 and end at the same
        // iterate, so the covariances agree to the rounding of the sums.
        assert!(dev <= 1e-12, "{kind:?} {with_clusters}: {dev:e}");
        assert_eq!(fit.n_fitted_above_one, 0);
    }
}

fn covariance_bits() -> Vec<f64> {
    let reference = reference();
    let case = &reference.cases[2];
    let x = matrix(&case.x);
    let clusters = Clusters::from_labels(case.cluster.as_ref().unwrap());
    let fit = risk_regression(
        RiskScale::Ratio,
        &glm_data(x.as_ref(), &case.y, case.weights.as_deref(), case.offset.as_deref(), None).unwrap(),
        Some(&clusters),
        SandwichType::Hc1,
    )
    .unwrap();
    fit.robust_covariance
        .as_slice()
        .iter()
        .map(|v| v.unwrap())
        .collect()
}

#[test]
fn the_covariance_does_not_depend_on_the_number_of_threads() {
    assert_same_on_threads(covariance_bits);
}

#[test]
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
fn the_covariance_has_its_recorded_bits() {
    let hash = epistat_testkit::Fingerprint::new()
        .f64s(&covariance_bits())
        .finish();
    assert_eq!(hash, 5_813_520_643_598_786_199, "recorded hash");
}
