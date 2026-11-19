//! Fits checked against R 4.6.2's `glm()`.

use epistat_glm::{Family, Glm, Link};
use epistat_testkit::{Tolerance, assert_close, load_json};
use faer::Mat;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    x: Vec<Vec<f64>>,
    weights: Vec<f64>,
    offset: Vec<f64>,
    fits: Vec<RFit>,
}

#[derive(Deserialize)]
struct RFit {
    name: String,
    y: Vec<f64>,
    coefficients: Vec<f64>,
    se: Vec<f64>,
    deviance: f64,
    iterations: usize,
    weighted: bool,
    offset: bool,
}

/// The values of a vector without aliased entries.
fn all(values: &[Option<f64>]) -> Vec<f64> {
    values
        .iter()
        .map(|v| v.expect("no aliased column"))
        .collect()
}

#[test]
fn fits_match_r() {
    let reference: Reference =
        load_json(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/glm.json"));
    let x = Mat::from_fn(reference.x.len(), reference.x[0].len(), |i, j| {
        reference.x[i][j]
    });
    let close = Tolerance::exact().rel(1e-11).abs(1e-13);
    for r in &reference.fits {
        let (family, link) = match r.name.as_str() {
            "binomial_logit" => (Family::Binomial, Link::Logit),
            "poisson_log" => (Family::Poisson, Link::Log),
            "gamma_log" => (Family::Gamma, Link::Log),
            "gaussian_identity" => (Family::Gaussian, Link::Identity),
            other => panic!("unknown fit {other}"),
        };
        let weights = r.weighted.then_some(reference.weights.as_slice());
        let offset = r.offset.then_some(reference.offset.as_slice());
        let fit = Glm::new(family, link)
            .fit(x.as_ref(), &r.y, weights, offset, None)
            .unwrap();
        assert!(fit.converged, "{}", r.name);
        assert_eq!(fit.iterations, r.iterations, "{}", r.name);
        assert_close(&all(&fit.coefficients), &r.coefficients, close);
        assert_close(
            &all(&fit.standard_errors()),
            &r.se,
            Tolerance::exact().rel(1e-10),
        );
        assert_close(&[fit.deviance], &[r.deviance], close);
    }
}

#[derive(Deserialize)]
struct TweedieReference {
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
    fits: Vec<TweedieFit>,
}

#[derive(Deserialize)]
struct TweedieFit {
    power: f64,
    coefficients: Vec<f64>,
    se: Vec<f64>,
    deviance: f64,
    iterations: usize,
    dispersion: f64,
}

/// The largest relative deviation of `got` from `expected`.
fn worst(got: &[f64], expected: &[f64]) -> f64 {
    got.iter()
        .zip(expected)
        .map(|(g, e)| (g - e).abs() / e.abs().max(1e-300))
        .fold(0.0, f64::max)
}

/// glm() with statmod::tweedie(var.power = p, link.power = 0), with zero
/// responses and prior weights.
#[test]
fn tweedie_fits_match_statmod() {
    let reference: TweedieReference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/glm_tweedie.json"
    ));
    assert!(reference.y.contains(&0.0));
    let x = Mat::from_fn(reference.x.len(), reference.x[0].len(), |i, j| {
        reference.x[i][j]
    });
    for r in &reference.fits {
        let fit = Glm::new(Family::Tweedie { power: r.power }, Link::Log)
            .fit(
                x.as_ref(),
                &reference.y,
                Some(&reference.weights),
                None,
                None,
            )
            .unwrap();
        assert!(fit.converged, "power {}", r.power);
        assert_eq!(fit.iterations, r.iterations, "power {}", r.power);
        let (beta, se) = (all(&fit.coefficients), all(&fit.standard_errors()));
        eprintln!(
            "tweedie {}: worst relative deviation, coefficients {:e}, se {:e}, deviance {:e}, dispersion {:e}",
            r.power,
            worst(&beta, &r.coefficients),
            worst(&se, &r.se),
            worst(&[fit.deviance], &[r.deviance]),
            worst(&[fit.dispersion.unwrap()], &[r.dispersion]),
        );
        assert_close(
            &beta,
            &r.coefficients,
            Tolerance::exact().rel(1e-11).abs(1e-13),
        );
        assert_close(&se, &r.se, Tolerance::exact().rel(1e-10));
        assert_close(
            &[fit.deviance, fit.dispersion.unwrap()],
            &[r.deviance, r.dispersion],
            Tolerance::exact().rel(1e-11),
        );
    }
}

#[derive(Deserialize)]
struct AliasedReference {
    fits: Vec<AliasedFit>,
}

#[derive(Deserialize)]
struct AliasedFit {
    name: String,
    family: String,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    coefficients: Vec<Option<f64>>,
    se: Vec<Option<f64>>,
    rank: usize,
    deviance: f64,
    iterations: usize,
    gram_condition: f64,
}

/// glm() on designs with an exactly aliased column, where R reports NA, and
/// with nearly aliased columns, which R keeps.
#[test]
fn aliased_columns_are_none_as_in_r() {
    let reference: AliasedReference = load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/glm_aliased.json"
    ));
    for r in &reference.fits {
        let x = Mat::from_fn(r.x.len(), r.x[0].len(), |i, j| r.x[i][j]);
        let (family, link) = if r.family == "gaussian" {
            (Family::Gaussian, Link::Identity)
        } else {
            (Family::Poisson, Link::Log)
        };
        let fit = Glm::new(family, link)
            .fit(x.as_ref(), &r.y, None, None, None)
            .unwrap();
        assert_eq!(fit.rank, r.rank, "{}", r.name);
        assert_eq!(fit.iterations, r.iterations, "{}", r.name);
        let missing =
            |values: &[Option<f64>]| values.iter().map(Option::is_none).collect::<Vec<_>>();
        assert_eq!(
            missing(&fit.coefficients),
            missing(&r.coefficients),
            "{}",
            r.name
        );
        assert_eq!(
            missing(&fit.standard_errors()),
            missing(&r.se),
            "{}",
            r.name
        );
        let present = |values: &[Option<f64>]| values.iter().flatten().copied().collect::<Vec<_>>();
        let (beta, se) = (present(&fit.coefficients), present(&fit.standard_errors()));
        let (beta_r, se_r) = (present(&r.coefficients), present(&r.se));
        // The fit solves the normal equations, R a QR of sqrt(W) X, so the two
        // can differ by about the condition number of X'WX times the unit
        // roundoff, which the reference records. The bound grows from about
        // 4e-15 for the exactly aliased designs to 2e-3 at a perturbation of
        // 1e-6, and the CPU's choice of kernels moves results within it.
        let rel = (8.0 * r.gram_condition * f64::EPSILON).max(1e-13);
        eprintln!(
            "{}: worst relative deviation, coefficients {:e}, se {:e}, deviance {:e}",
            r.name,
            worst(&beta, &beta_r),
            worst(&se, &se_r),
            worst(&[fit.deviance], &[r.deviance]),
        );
        assert_close(&beta, &beta_r, Tolerance::exact().rel(rel));
        assert_close(&se, &se_r, Tolerance::exact().rel(rel));
        assert_close(&[fit.deviance], &[r.deviance], Tolerance::exact().rel(1e-7));
    }
}
