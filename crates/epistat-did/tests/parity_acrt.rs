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

use epistat_did::Error;
use epistat_did::continuous::{self, Acrt, AcrtOptions, BasisSpec, ContinuousData};
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
/// `(d, d^2)`, the quadratic without its constant.
const QUADRATIC: BasisSpec = BasisSpec::Polynomial {
    degree: 2,
    constant: false,
};

const CUBIC: BasisSpec = BasisSpec::Polynomial {
    degree: 3,
    constant: true,
};

fn spline(lower: f64, upper: f64) -> BasisSpec {
    BasisSpec::BSpline {
        degree: 3,
        internal_knots: Vec::new(),
        lower,
        upper,
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

fn fit(dose: &[f64], change: &[f64], weight: &[f64], basis: &BasisSpec) -> Result<Acrt, Error> {
    let data = ContinuousData::builder(dose, change)
        .weights(epistat_core::CaseWeights::new(weight)?)
        .build()?;
    continuous::estimate(
        &data,
        &AcrtOptions {
            basis: basis.clone(),
            ..AcrtOptions::default()
        },
    )
}

fn estimate(case: &Case, basis: &BasisSpec) -> Result<Acrt, Error> {
    fit(&case.dose, &case.delta_y, &case.weight, basis)
}

fn check(case: &Case, basis: &BasisSpec, label: &str) {
    let result = estimate(case, basis).expect("sieve");
    let acrt = result.estimate.estimate;
    let se = result.estimate.se.expect("se");
    let influence = result.influence.column(0);
    let n = case.dose.len() as f64;
    let largest = case.influence.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let rho = 8.0 * case.kappa * EPS * case.acrt.abs().max(1.0);
    let own = 8.0 * case.kappa * case.kappa * EPS * largest;
    let estimate_tol = rho + own / n;
    assert!(
        (acrt - case.acrt).abs() <= estimate_tol,
        "{label}: ACRT {acrt} against R {}",
        case.acrt
    );
    let mut worst = 0.0_f64;
    let mut squares = 0.0;
    for i in 0..case.influence.len() {
        let tol = 2.0 * (case.influence_plain[i] - case.influence[i]).abs()
            + 3.0 * n * rho / case.h
            + own;
        let deviation = (influence[i] - case.influence[i]).abs();
        assert!(
            deviation <= tol,
            "{label} unit {i}: influence {} against the jackknife {} (tolerance {tol:e})",
            influence[i],
            case.influence[i]
        );
        worst = worst.max(deviation / tol);
        squares += tol * tol;
    }
    // The standard error is a 2-norm of the influence values over n, so it moves
    // by at most the 2-norm of their tolerances over n.
    let se_tol = squares.sqrt() / n;
    assert!(
        (se - case.se_ij).abs() <= se_tol,
        "{label}: standard error {se} against the jackknife {} (tolerance {se_tol:e})",
        case.se_ij
    );
    println!(
        "{label}: worst influence deviation {worst:.3e} of its tolerance; se {se:.6} (jackknife {:.6}, delete-one {:.6})",
        case.se_ij, case.se_delete_one
    );
}

#[test]
fn the_influence_function_is_the_jackknife_of_the_procedure() {
    let reference = load();
    let contdid = &reference.cases[0];
    assert_eq!(contdid.name, "contdid");
    check(contdid, &CUBIC, "contdid, cubic polynomial");
    let boundary = contdid.boundary.as_ref().expect("boundary");
    check(
        contdid,
        &spline(boundary[0], boundary[1]),
        "contdid, cubic B-spline",
    );
    let weighted = &reference.cases[1];
    assert_eq!(weighted.name, "weighted");
    check(weighted, &QUADRATIC, "weighted, (d, d^2)");
}

/// The cubic B-spline basis without interior knots spans the cubic polynomials,
/// so its ACRT is contdid's. The two fits differ by the rounding of the normal
/// equations, `8 kappa^2 eps` relative.
#[test]
fn the_estimate_is_contdid_s() {
    let reference = load();
    let case = &reference.cases[0];
    let boundary = case.boundary.as_ref().expect("boundary");
    let spline = spline(boundary[0], boundary[1]);
    let want = case.contdid_acrt.expect("contdid");
    let tol = 8.0 * case.kappa * case.kappa * EPS * want.abs().max(1.0);
    for (label, got) in [
        (
            "polynomial",
            estimate(case, &CUBIC).unwrap().estimate.estimate,
        ),
        (
            "B-spline",
            estimate(case, &spline).unwrap().estimate.estimate,
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
    let result = estimate(case, &QUADRATIC).unwrap();
    for (i, &dose) in case.dose.iter().enumerate() {
        if dose < 0.0 {
            assert_eq!(result.influence.column(0)[i].to_bits(), 0.0_f64.to_bits());
        }
    }
    let mut weight = case.weight.clone();
    weight[0] = -1.0;
    assert!(fit(&case.dose, &case.delta_y, &weight, &QUADRATIC).is_err());
    let mut change = case.delta_y.clone();
    change[1] = f64::NAN;
    assert!(fit(&case.dose, &change, &case.weight, &QUADRATIC).is_err());
    let treated: Vec<usize> = (0..case.dose.len())
        .filter(|&i| case.dose[i] > 0.0)
        .collect();
    let pick = |v: &[f64]| treated.iter().map(|&i| v[i]).collect::<Vec<_>>();
    assert_eq!(
        fit(
            &pick(&case.dose),
            &pick(&case.delta_y),
            &pick(&case.weight),
            &QUADRATIC
        )
        .unwrap_err(),
        Error::NoComparison
    );
}
