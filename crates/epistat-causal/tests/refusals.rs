//! The refusals and coefficient tables of the estimators: predictions of a
//! binary outcome that are not probabilities, designs without an intercept,
//! asymmetric propensity bounds where only symmetric ones apply, and the
//! tables of the two means.

use epistat_causal::aipw::{AipwInput, AipwOptions, GlmInput, aipw, aipw_glm};
use epistat_causal::tmle::{PropensityBound, TmleOptions, tmle};
use epistat_causal::{Error, Estimand, ProbabilityBounds};
use epistat_core::{InputError, Level, Tidy, VarianceKind};
use epistat_glm::Family;
use epistat_linalg::Mat;

const N: usize = 60;

fn x(i: usize) -> f64 {
    ((i * 7 % 13) as f64 - 6.0) / 6.5
}

fn data() -> (Vec<f64>, Vec<bool>, Vec<f64>, Vec<f64>) {
    let a: Vec<bool> = (0..N).map(|i| (i * 5 + 3) % 7 < 3).collect();
    let y: Vec<f64> = (0..N)
        .map(|i| f64::from(u8::from((i * 3 + 1) % 5 < 2)))
        .collect();
    let e: Vec<f64> = (0..N).map(|i| 0.4 + 0.2 * x(i)).collect();
    let mu: Vec<f64> = (0..N).map(|i| 0.4 + 0.1 * x(i)).collect();
    (y, a, e, mu)
}

#[test]
fn predictions_of_a_binary_outcome_outside_zero_one_are_refused() {
    let (y, a, e, mu) = data();
    // Log odds passed where probabilities belong.
    let log_odds: Vec<f64> = mu.iter().map(|&p| (p / (1.0 - p)).ln()).collect();
    let input = AipwInput {
        outcome: &y,
        treatment: &a,
        mu1: &log_odds,
        mu0: &mu,
        propensity: &e,
        weight: None,
        cluster: None,
    };
    for result in [
        aipw(&input, &AipwOptions::default()).map(|_| ()),
        tmle(&input, &TmleOptions::default()).map(|_| ()),
    ] {
        assert!(
            matches!(
                result,
                Err(Error::Input(InputError::Domain { what: "mu1", .. }))
            ),
            "{result:?}"
        );
    }
    // A continuous outcome takes predictions on its own scale.
    let continuous: Vec<f64> = (0..N).map(|i| 2.0 + x(i)).collect();
    let input = AipwInput {
        outcome: &continuous,
        mu1: &log_odds,
        ..input
    };
    assert!(aipw(&input, &AipwOptions::default()).is_ok());
}

#[test]
fn a_nuisance_design_without_an_intercept_is_refused() {
    let (y, a, ..) = data();
    let with = Mat::from_fn(N, 2, |i, j| if j == 0 { 1.0 } else { x(i) });
    let without = Mat::from_fn(N, 1, |i, _| x(i));
    let input = GlmInput {
        outcome: &y,
        treatment: &a,
        outcome_design: with.as_ref(),
        propensity_design: without.as_ref(),
        weight: None,
        cluster: None,
    };
    assert!(matches!(
        aipw_glm(&input, Family::Binomial, &AipwOptions::default()),
        Err(Error::Invalid {
            what: "propensity_design",
            ..
        })
    ));
    let input = GlmInput {
        propensity_design: with.as_ref(),
        ..input
    };
    assert!(aipw_glm(&input, Family::Binomial, &AipwOptions::default()).is_ok());
}

#[test]
fn asymmetric_bounds_are_taken_by_the_population_effect_only() {
    let (y, a, e, mu) = data();
    let input = AipwInput {
        outcome: &y,
        treatment: &a,
        mu1: &mu,
        mu0: &mu,
        propensity: &e,
        weight: None,
        cluster: None,
    };
    let bounds = ProbabilityBounds::new(0.3, 0.6).expect("bounds");
    let options = TmleOptions {
        propensity_bound: PropensityBound::At(bounds),
        ..TmleOptions::default()
    };
    let fit = tmle(&input, &options).expect("ate");
    assert_eq!(fit.propensity_bound, Some(bounds));
    let att = TmleOptions {
        estimand: Estimand::Att,
        ..options
    };
    assert!(matches!(
        tmle(&input, &att),
        Err(Error::Invalid {
            what: "propensity_bound",
            ..
        })
    ));
}

#[test]
fn the_table_of_the_two_means_reproduces_the_contrast() {
    let (y, a, e, mu) = data();
    let input = AipwInput {
        outcome: &y,
        treatment: &a,
        mu1: &mu,
        mu0: &mu,
        propensity: &e,
        weight: None,
        cluster: None,
    };
    let fit = aipw(&input, &AipwOptions::default()).expect("aipw");
    let table = fit.tidy().expect("table");
    assert_eq!(table.variance(), VarianceKind::Robust);
    let mean1 = table.get("mean1").and_then(|t| t.estimate).expect("mean1");
    let mean0 = table.get("mean0").and_then(|t| t.estimate).expect("mean0");
    // Both are formed by the same Wald routine from the same variance.
    assert_eq!(mean1, fit.contrast.mean1);
    assert_eq!(mean0, fit.contrast.mean0);
    assert!(
        fit.coefficients(Level::default(), VarianceKind::ModelBased)
            .is_err()
    );
    let targeted = tmle(&input, &TmleOptions::default()).expect("tmle");
    let table = targeted.tidy().expect("table");
    assert_eq!(
        table.get("mean1").and_then(|t| t.estimate),
        Some(targeted.contrast.mean1)
    );
}
