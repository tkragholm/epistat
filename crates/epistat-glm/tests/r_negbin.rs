//! Negative binomial and quasi families checked against R's `MASS::glm.nb`
//! (MASS 7.3.66) and `glm(family = quasipoisson / quasibinomial)`.

use epistat_glm::{Family, Glm, Link, NegBin, ThetaRule};
use epistat_linalg::Mat;
use epistat_testkit::{assert_same_on_threads, load_json};
use serde::Deserialize;

mod common;
use common::*;

#[derive(Deserialize)]
struct Reference {
    scenarios: Vec<Scenario>,
    quasi: Vec<Quasi>,
}

#[derive(Deserialize)]
struct Scenario {
    name: String,
    link: String,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Option<Vec<f64>>,
    offset: Option<Vec<f64>>,
    start: Option<Vec<f64>>,
    runs: Vec<Run>,
}

#[derive(Deserialize)]
struct Run {
    tight: bool,
    error: Option<String>,
    coef: Option<Vec<Option<f64>>>,
    se: Option<Vec<Option<f64>>>,
    theta: Option<f64>,
    se_theta: Option<f64>,
    twologlik: Option<f64>,
    deviance: Option<f64>,
    df_residual: Option<usize>,
    alternations: Option<usize>,
}

#[derive(Deserialize)]
struct Quasi {
    family: String,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    weights: Vec<f64>,
    coef: Vec<f64>,
    se: Vec<f64>,
    dispersion: f64,
    df_residual: usize,
    deviance: f64,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/negbin.json"
    ))
}

fn matrix(rows: &[Vec<f64>]) -> Mat<f64> {
    Mat::from_fn(rows.len(), rows[0].len(), |i, j| rows[i][j])
}

fn relative(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(1e-300)
}

#[test]
fn negative_binomial_matches_glm_nb() {
    let mut worst = [0.0_f64; 2];
    for scenario in &reference().scenarios {
        let x = matrix(&scenario.x);
        for run in &scenario.runs {
            if run.error.is_some() {
                continue;
            }
            let rule = if run.tight {
                ThetaRule::Tight
            } else {
                ThetaRule::RParity
            };
            let link = match scenario.link.as_str() {
                "identity" => Link::Identity,
                _ => Link::Log,
            };
            let model = if run.tight {
                NegBin {
                    link,
                    tolerance: 1e-14,
                    max_iterations: 200,
                    theta_rule: rule,
                    ..NegBin::new()
                }
            } else {
                NegBin {
                    link,
                    theta_rule: rule,
                    ..NegBin::new()
                }
            };
            let fit = model
                .fit_from(
                    x.as_ref(),
                    &scenario.y,
                    scenario.weights.as_deref(),
                    scenario.offset.as_deref(),
                    scenario.start.as_deref(),
                )
                .unwrap();
            let label = format!("{} tight {}", scenario.name, run.tight);
            if scenario.name == "poisson_limit" {
                assert!(fit.poisson_limit && fit.theta.is_none(), "{label}");
                continue;
            }
            assert!(!fit.poisson_limit, "{label}");
            // The default run reproduces R's loops step by step, so the same
            // arithmetic gives theta to rounding, 1e-11 relative here. R's tight
            // run is limited by the flat likelihood in theta, which it locates
            // to about 1e-9 (the worst case is printed below), this crate to 1e-10.
            let (theta_bound, beta_bound, ll_bound) = if run.tight {
                (1e-8, 1e-7, 1e-10)
            } else {
                (1e-11, 1e-9, 1e-11)
            };
            let theta = fit.theta.unwrap();
            let d = relative(theta, run.theta.unwrap());
            assert!(
                d <= theta_bound,
                "{label}: theta {theta} against {:?}",
                run.theta
            );
            worst[usize::from(run.tight)] = worst[usize::from(run.tight)].max(d);
            // R takes the information of theta from the last Newton step of
            // theta.ml, evaluated before the step, and stops at a step of
            // eps^(1/4) = 1.2e-4 relative, so its standard error is that of a
            // theta within about 1e-4 of the estimate. The default run
            // reproduces that point, the tight one does not.
            let d = relative(fit.theta_se.unwrap(), run.se_theta.unwrap());
            assert!(
                d <= if run.tight { 2e-4 } else { 1e-8 },
                "{label}: se theta {d:e}"
            );
            let d = relative(2.0 * fit.log_likelihood, run.twologlik.unwrap());
            assert!(d <= ll_bound, "{label}: twologlik {d:e}");
            assert_eq!(fit.glm.residual_df, run.df_residual.unwrap(), "{label}");
            let deviance_bound = if run.tight { 1e-7 } else { 1e-8 };
            if !run.tight {
                assert_eq!(fit.alternations, run.alternations.unwrap(), "{label}");
            }
            // The deviance and the fit are those of the last alternation, at
            // the theta before the final update, as R reports them.
            let d = relative(fit.glm.deviance, run.deviance.unwrap());
            assert!(d <= deviance_bound, "{label}: deviance {d:e}");
            let se = fit.glm.standard_errors();
            for (j, want) in run.coef.as_ref().unwrap().iter().enumerate() {
                match want {
                    None => assert!(fit.glm.coefficients[j].is_none(), "{label} column {j}"),
                    Some(want) => {
                        let got = fit.glm.coefficients[j].unwrap();
                        assert!(
                            (got - want).abs() <= beta_bound * want.abs().max(1e-3),
                            "{label} column {j}: {got} against {want}"
                        );
                        let want_se = run.se.as_ref().unwrap()[j].unwrap();
                        let d = relative(se[j].unwrap(), want_se);
                        assert!(
                            d <= if run.tight { 1e-6 } else { 1e-5 },
                            "{label} se {j}: {d:e}"
                        );
                    }
                }
            }
        }
    }
    eprintln!(
        "worst relative deviation of theta: default {:e}, tight {:e}",
        worst[0], worst[1]
    );
}

/// With an offset the identity link has no valid first fit from the default
/// start, in R and here, and starting coefficients make it fit.
#[test]
fn identity_link_with_an_offset_needs_starting_coefficients() {
    let r = reference();
    let scenario = r
        .scenarios
        .iter()
        .find(|s| s.name == "identity_link_offset")
        .unwrap();
    let x = matrix(&scenario.x);
    let model = NegBin {
        link: Link::Identity,
        ..NegBin::new()
    };
    let without = model.fit4(x.as_ref(), &scenario.y, None, scenario.offset.as_deref());
    assert!(matches!(without, Err(epistat_glm::Error::NoValidStart)));
    let short = model.fit_from(
        x.as_ref(),
        &scenario.y,
        None,
        scenario.offset.as_deref(),
        Some(&[1.0]),
    );
    assert!(matches!(
        short,
        Err(epistat_glm::Error::Input(
            epistat_core::InputError::Length { .. }
        ))
    ));
}

#[test]
fn quasi_families_match_glm() {
    for case in &reference().quasi {
        let x = matrix(&case.x);
        let (family, link) = match case.family.as_str() {
            "quasipoisson" => (Family::QuasiPoisson, Link::Log),
            _ => (Family::QuasiBinomial, Link::Logit),
        };
        let fit = Glm::new(family, link)
            .fit5(x.as_ref(), &case.y, Some(&case.weights), None, None)
            .unwrap();
        assert_eq!(fit.residual_df, case.df_residual, "{}", case.family);
        let dispersion = fit.dispersion.unwrap();
        assert!(
            relative(dispersion, case.dispersion) <= 1e-10,
            "{} {dispersion}",
            case.family
        );
        assert!(
            relative(fit.deviance, case.deviance) <= 1e-10,
            "{}",
            case.family
        );
        let se = fit.standard_errors();
        for (j, &want) in case.coef.iter().enumerate() {
            let got = fit.coefficients[j].unwrap();
            assert!(
                (got - want).abs() <= 1e-10 * want.abs().max(1e-3),
                "{} {j}",
                case.family
            );
            assert!(
                relative(se[j].unwrap(), case.se[j]) <= 1e-9,
                "{} se {j}",
                case.family
            );
        }
    }
}

fn fit_bits() -> Vec<f64> {
    let reference = reference();
    let scenario = &reference.scenarios[2];
    let x = matrix(&scenario.x);
    let fit = NegBin::new()
        .fit4(x.as_ref(), &scenario.y, None, None)
        .unwrap();
    let mut out: Vec<f64> = fit.glm.coefficients.iter().map(|c| c.unwrap()).collect();
    out.extend(fit.glm.covariance.as_slice().iter().map(|c| c.unwrap()));
    out.extend([
        fit.theta.unwrap(),
        fit.theta_se.unwrap(),
        fit.log_likelihood,
    ]);
    out
}

#[test]
fn the_fit_does_not_depend_on_the_number_of_threads() {
    assert_same_on_threads(fit_bits);
}

#[test]
fn the_fit_has_its_recorded_bits() {
    let hash = epistat_testkit::Fingerprint::new()
        .f64s(&fit_bits())
        .finish();
    epistat_testkit::hash_out("the_fit_has_its_recorded_bits", hash);
    if epistat_testkit::RECORDED_PLATFORM {
        assert_eq!(hash, 13_519_557_135_473_899_766, "recorded hash");
    }
}
