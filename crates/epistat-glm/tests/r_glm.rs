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
        assert_close(&fit.coefficients, &r.coefficients, close);
        assert_close(&fit.standard_errors(), &r.se, Tolerance::exact().rel(1e-10));
        assert_close(&[fit.deviance], &[r.deviance], close);
    }
}
