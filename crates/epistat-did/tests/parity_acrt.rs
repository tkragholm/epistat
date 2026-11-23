//! The continuous-treatment ACRT checked against R: the estimate against
//! `contdid::cont_did(target_parameter = "slope", dose_est_method =
//! "parametric", degree = 3)`, and the influence function and standard error
//! against the infinitesimal jackknife of the whole procedure, central
//! differences of refits in the case weight of each unit. See
//! `data/generate_acrt_reference.R`.
//!
//! # The tolerances
//!
//! A refit in R solves the weighted least squares problem by QR, so its ACRT
//! has a rounding error of about `rho = 8 kappa eps max(|ACRT|, 1)`, with
//! `kappa` the condition number of the weighted design that R records. The
//! Richardson combination of central differences at steps `h` and `h / 2`
//! carries `3 rho / h` of it into a derivative, times `n` into an influence
//! value, and its truncation is bounded by twice the distance to the plain
//! central difference. This crate solves the normal equations, whose condition
//! number is `kappa^2`, so its influence values differ by a relative
//! `8 kappa^2 eps` of the largest of them.

use epistat_did::{
    BSplineBasis, Basis, ContinuousObservation, PolynomialBasis, estimate_acrt_sieve,
};
use epistat_linalg::Col;
use epistat_testkit::load_json;
use serde::Deserialize;

const EPS: f64 = f64::EPSILON;

#[derive(Deserialize)]
struct Versions {
    r: String,
    contdid: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    dose: Vec<f64>,
    delta_y: Vec<f64>,
    weight: Vec<f64>,
    acrt: f64,
    influence: Vec<f64>,
    influence_plain: Vec<f64>,
    h: f64,
    se_ij: f64,
    se_delete_one: f64,
    kappa: f64,
    contdid_acrt: Option<f64>,
    contdid_se: Option<f64>,
    boundary: Option<Vec<f64>>,
}

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    cases: Vec<Case>,
}

/// The basis `(d, d^2)`, which does not span the constants.
struct Quadratic;

impl Basis for Quadratic {
    fn eval(&self, x: f64) -> Col<f64> {
        Col::from_fn(2, |j| if j == 0 { x } else { x * x })
    }

    fn eval_deriv(&self, x: f64) -> Col<f64> {
        Col::from_fn(2, |j| if j == 0 { 1.0 } else { 2.0 * x })
    }

    fn num_basis(&self) -> usize {
        2
    }
}

fn load() -> Reference {
    let r: Reference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/acrt_ref.json"
    ));
    assert!(r.versions.r.starts_with("R version 4.6"));
    assert_eq!(r.versions.contdid, "0.1.2");
    r
}

fn observations(case: &Case) -> Vec<ContinuousObservation> {
    (0..case.dose.len())
        .map(|i| ContinuousObservation {
            dose: case.dose[i],
            delta_outcome: case.delta_y[i],
            weight: case.weight[i],
        })
        .collect()
}

fn check<B: Basis>(case: &Case, basis: &B, label: &str) {
    let result = estimate_acrt_sieve(&observations(case), basis).expect("sieve");
    let n = case.dose.len() as f64;
    let largest = case.influence.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let rho = 8.0 * case.kappa * EPS * case.acrt.abs().max(1.0);
    let own = 8.0 * case.kappa * case.kappa * EPS * largest;
    let estimate_tol = rho + own / n;
    assert!(
        (result.acrt_glob - case.acrt).abs() <= estimate_tol,
        "{label}: ACRT {} against R {}",
        result.acrt_glob,
        case.acrt
    );
    let mut worst = 0.0_f64;
    let mut squares = 0.0;
    for i in 0..case.influence.len() {
        let tol = 2.0 * (case.influence_plain[i] - case.influence[i]).abs()
            + 3.0 * n * rho / case.h
            + own;
        let deviation = (result.influence_function[i] - case.influence[i]).abs();
        assert!(
            deviation <= tol,
            "{label} unit {i}: influence {} against the jackknife {} (tolerance {tol:e})",
            result.influence_function[i],
            case.influence[i]
        );
        worst = worst.max(deviation / tol);
        squares += tol * tol;
    }
    // The standard error is a 2-norm of the influence values over n, so it moves
    // by at most the 2-norm of their tolerances over n.
    let se_tol = squares.sqrt() / n;
    assert!(
        (result.se - case.se_ij).abs() <= se_tol,
        "{label}: standard error {} against the jackknife {} (tolerance {se_tol:e})",
        result.se,
        case.se_ij
    );
    println!(
        "{label}: worst influence deviation {worst:.3e} of its tolerance; se {:.6} (jackknife {:.6}, delete-one {:.6})",
        result.se, case.se_ij, case.se_delete_one
    );
}

#[test]
fn the_influence_function_is_the_jackknife_of_the_procedure() {
    let reference = load();
    let contdid = &reference.cases[0];
    assert_eq!(contdid.name, "contdid");
    check(
        contdid,
        &PolynomialBasis::new(3),
        "contdid, cubic polynomial",
    );
    let boundary = contdid.boundary.as_ref().expect("boundary");
    let spline = BSplineBasis::new(3, Vec::new(), (boundary[0], boundary[1])).expect("knots");
    check(contdid, &spline, "contdid, cubic B-spline");
    let weighted = &reference.cases[1];
    assert_eq!(weighted.name, "weighted");
    check(weighted, &Quadratic, "weighted, (d, d^2)");
}

/// The cubic B-spline basis without interior knots spans the cubic polynomials,
/// so its ACRT is contdid's. The two fits differ by the rounding of the normal
/// equations, `8 kappa^2 eps` relative.
#[test]
fn the_estimate_is_contdid_s() {
    let reference = load();
    let case = &reference.cases[0];
    let boundary = case.boundary.as_ref().expect("boundary");
    let spline = BSplineBasis::new(3, Vec::new(), (boundary[0], boundary[1])).expect("knots");
    let want = case.contdid_acrt.expect("contdid");
    let tol = 8.0 * case.kappa * case.kappa * EPS * want.abs().max(1.0);
    for (label, got) in [
        (
            "polynomial",
            estimate_acrt_sieve(&observations(case), &PolynomialBasis::new(3))
                .unwrap()
                .acrt_glob,
        ),
        (
            "B-spline",
            estimate_acrt_sieve(&observations(case), &spline)
                .unwrap()
                .acrt_glob,
        ),
    ] {
        assert!(
            (got - want).abs() <= tol,
            "{label}: {got} against contdid {want}"
        );
    }
    // contdid's own standard error divides an influence function scaled by the
    // treated units by the number of all units, so it is about n_T / n of the
    // jackknife's. Recorded, not matched.
    let se = case.contdid_se.expect("contdid se");
    assert!(
        se < 0.7 * case.se_ij,
        "contdid se {se}, jackknife {}",
        case.se_ij
    );
}

/// Units with a negative dose are in neither group and have no influence;
/// invalid observations are errors.
#[test]
fn negative_doses_have_no_influence_and_invalid_input_is_an_error() {
    let reference = load();
    let case = &reference.cases[1];
    let result = estimate_acrt_sieve(&observations(case), &Quadratic).unwrap();
    for (i, &dose) in case.dose.iter().enumerate() {
        if dose < 0.0 {
            assert_eq!(result.influence_function[i].to_bits(), 0.0_f64.to_bits());
        }
    }
    let mut bad = observations(case);
    bad[0].weight = -1.0;
    assert!(estimate_acrt_sieve(&bad, &Quadratic).is_err());
    let mut bad = observations(case);
    bad[1].delta_outcome = f64::NAN;
    assert!(estimate_acrt_sieve(&bad, &Quadratic).is_err());
    let treated_only: Vec<_> = observations(case)
        .into_iter()
        .filter(|o| o.dose > 0.0)
        .collect();
    assert_eq!(
        estimate_acrt_sieve(&treated_only, &Quadratic),
        Err(epistat_did::ContinuousDidError::NoControlUnits)
    );
}
