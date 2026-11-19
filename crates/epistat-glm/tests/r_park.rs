//! The Park tests checked against R 4.6.2 (`park.R`).

use epistat_glm::park::{classic_park_test, modified_park_test, modified_park_test_with};
use epistat_glm::{Family, Glm, Link};
use epistat_testkit::{Tolerance, assert_close, load_json};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    weights: Vec<f64>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    weighted: bool,
    y: Vec<f64>,
    fitted: Vec<f64>,
    coefficients: Vec<f64>,
    se: Vec<f64>,
    iterations: usize,
    tight_coefficients: Vec<f64>,
    tight_se: Vec<f64>,
    classic_coefficients: Vec<f64>,
    classic_se: Vec<f64>,
}

fn reference() -> Reference {
    load_json(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/park.json"))
}

/// R's `glm(r2 ~ log(mu), family = Gamma("log"))`, with R's stopping rule and
/// with a tight one.
#[test]
fn modified_test_matches_r_glm() {
    let reference = reference();
    for c in &reference.cases {
        let weights = c.weighted.then_some(reference.weights.as_slice());
        let label = format!("{} weighted {}", c.name, c.weighted);
        let got = modified_park_test(&c.fitted, &c.y, weights)
            .unwrap()
            .unwrap();
        assert!(got.converged, "{label}");
        assert_eq!(got.iterations, c.iterations, "{label}");
        assert_eq!(got.rows, c.y.len(), "{label}");
        let close = Tolerance::exact().rel(1e-10);
        assert_close(&[got.intercept, got.power], &c.coefficients, close);
        eprintln!(
            "{label}: power {:e}, se {:e}",
            (got.power - c.coefficients[1]).abs() / c.coefficients[1].abs(),
            (got.standard_error - c.se[1]).abs() / c.se[1]
        );
        assert_close(&[got.standard_error], &c.se[1..], close);

        let controls = Glm {
            tolerance: 1e-14,
            max_iterations: 200,
            ..Glm::new(Family::Gamma, Link::Log)
        };
        let tight = modified_park_test_with(controls, &c.fitted, &c.y, weights)
            .unwrap()
            .unwrap();
        assert_close(
            &[tight.intercept, tight.power],
            &c.tight_coefficients,
            close,
        );
        assert_close(&[tight.standard_error], &c.tight_se[1..], close);
    }
}

#[test]
fn classic_test_matches_r_lm() {
    for c in &reference().cases {
        let got = classic_park_test(&c.fitted, &c.y).unwrap().unwrap();
        let close = Tolerance::exact().rel(1e-11);
        assert_close(&[got.intercept, got.power], &c.classic_coefficients, close);
        assert_close(&[got.standard_error], &c.classic_se[1..], close);
    }
}

#[test]
fn undefined_cases_are_none_and_bad_input_is_an_error() {
    let mu = [2.0; 6];
    let y = [1.0, 2.5, 3.0, 1.5, 2.2, 0.7];
    assert_eq!(modified_park_test(&mu, &y, None).unwrap(), None);
    assert_eq!(classic_park_test(&mu, &y).unwrap(), None);
    assert_eq!(
        modified_park_test(&[1.0, 2.0], &[0.0, 5.0], None).unwrap(),
        None
    );
    assert!(modified_park_test(&mu, &y[..3], None).is_err());
    assert!(modified_park_test(&mu, &y, Some(&[1.0, 1.0, -1.0, 1.0, 1.0, 1.0])).is_err());
}
