//! The propensity score checked against MatchIt 4.8.2's
//! `matchit(distance = "glm")`.
//!
//! The reference is `tests/data/matchit_propensity.json`, written by
//! `tests/data/matchit_propensity.R`. It holds, for MatchIt's `lalonde` data
//! and three simulated data sets, the distance on the probability scale
//! (`link = "logit"`, `"probit"`, `"cloglog"`), on the linear predictor scale
//! (`link = "linear.logit"` and its siblings), with and without sampling
//! weights (`s.weights`), and after common-support trimming with the model
//! refitted on the kept units (`discard` with `reestimate = TRUE`). The fits
//! use MatchIt's own `glm` control, an epsilon of 1e-8 and 25 iterations.
//!
//! The Rust fit is `epistat-glm`'s IRLS, which takes the same steps as R's
//! `glm.fit` and stops by the same rule, so the iteration counts are equal.
//! R solves each step by a pivoted QR decomposition and the Rust fit by the
//! normal equations, which square the condition number. The two therefore agree
//! to the rounding of a solve with condition number kappa^2, where kappa is the
//! condition number of `sqrt(W) X` at the solution, with its columns scaled to
//! unit length, which the R script writes beside each fit. The scaling matters
//! because the Cholesky factorisation of the normal equations is not affected
//! by the scale of a column.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_matching::prelude::*;

use r_reference::{Versions, reference as load_reference, unit_id as id};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    fits: Vec<RFit>,
}

#[derive(Deserialize)]
struct RFit {
    name: String,
    treat: Vec<f64>,
    covariates: Vec<Vec<f64>>,
    covariate_names: Vec<String>,
    weights: Vec<f64>,
    links: Vec<RLink>,
    trimmed: Vec<RTrimmed>,
}

#[derive(Deserialize)]
struct RLink {
    link: String,
    probability: Vec<f64>,
    linear: Vec<f64>,
    coefficient_names: Vec<String>,
    coefficients: Vec<f64>,
    iterations: usize,
    converged: bool,
    condition: f64,
    column_norms: Vec<f64>,
    weighted_probability: Vec<f64>,
    weighted_linear: Vec<f64>,
    weighted_coefficients: Vec<f64>,
    weighted_iterations: usize,
    weighted_condition: f64,
    weighted_column_norms: Vec<f64>,
}

#[derive(Deserialize)]
struct RTrimmed {
    policy: String,
    probability: RReestimated,
    linear_probit: RReestimated,
    weighted: RReestimated,
}

#[derive(Deserialize)]
struct RReestimated {
    distance: Vec<f64>,
    condition: f64,
    discarded: Vec<bool>,
    same_discarded: bool,
}

fn reference() -> Reference {
    load_reference("matchit_propensity.json")
}

fn link_of(name: &str) -> PropensityLink {
    match name {
        "logit" => PropensityLink::Logit,
        "probit" => PropensityLink::Probit,
        "cloglog" => PropensityLink::Cloglog,
        other => panic!("unknown link {other}"),
    }
}

fn policy_of(name: &str) -> CommonSupport {
    match name {
        "treated" => CommonSupport::Treated,
        "control" => CommonSupport::Control,
        "both" => CommonSupport::Both,
        other => panic!("unknown policy {other}"),
    }
}

struct Options<'a> {
    link: PropensityLink,
    scale: PropensityScoreOutputScale,
    weights: Option<&'a [f64]>,
    discard: Option<CommonSupport>,
}

fn fit_ours(fit: &RFit, options: &Options<'_>) -> PropensityDistancePreparation {
    let birth = NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date");
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for (idx, label) in fit.treat.iter().enumerate() {
        let mut record = BalanceRecord::new(id(idx), birth);
        for (name, column) in fit.covariate_names.iter().zip(&fit.covariates) {
            record
                .covariates
                .insert(name.clone(), CovariateValue::Numeric(column[idx]));
        }
        if *label > 0.5 {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    let mut config = PropensityScoreConfig::builder()
        .output_scale(options.scale)
        .logistic(
            LogisticRegressionConfig::builder()
                .link(options.link)
                .build(),
        )
        .reestimate(options.discard.is_some())
        .build();
    config.sampling_weights = options.weights.map(|weights| {
        weights
            .iter()
            .enumerate()
            .map(|(idx, w)| (id(idx), *w))
            .collect()
    });
    prepare_propensity_distance_config_with_discard(&anchors, &candidates, &config, options.discard)
        .expect("the fit succeeds")
}

/// Allowance for the rounding of a linear solve with the squared condition
/// number of `sqrt(W) X`. A solve in double precision has a relative error of
/// about `kappa^2 u` in the coefficients, with `u` the unit roundoff, and the
/// linear predictor is a sum of at most ten terms each that size times its
/// largest value, so the allowance is `8 kappa^2 u` times `1 + max |eta|`,
/// where the 1 covers a predictor near 0.
fn allowance(condition: f64, eta_max: f64) -> f64 {
    8.0 * condition * condition * f64::EPSILON * (1.0 + eta_max)
}

/// The largest slope of the inverse link: the logistic function has 1/4, the
/// normal density 0.399 and the cloglog inverse exp(-1).
const MAX_SLOPE: f64 = 0.4;

fn max_abs(values: &[f64]) -> f64 {
    values.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

/// The coefficients of `ours` against `expected`. With `D` the lengths of the
/// columns of `sqrt(W) X` and `b = D beta` the coefficients of the scaled
/// system, the error of coefficient `j` is `8 kappa^2 u max|b| / D_j`, where
/// kappa is the condition number of the scaled system and `u` the unit
/// roundoff.
fn compare_coefficients(
    name: &str,
    ours: &PropensityDistancePreparation,
    r: &RLink,
    expected: &[f64],
    condition: f64,
    norms: &[f64],
) {
    let scaled_max = expected
        .iter()
        .zip(norms)
        .fold(0.0_f64, |acc, (b, d)| acc.max((b * d).abs()));
    for ((column, want), norm) in r.coefficient_names.iter().zip(expected).zip(norms) {
        let ours_name = if column == "(Intercept)" {
            "intercept"
        } else {
            column
        };
        let position = ours
            .estimate
            .column_names
            .iter()
            .position(|n| n == ours_name)
            .expect("column exists");
        let got = ours.estimate.coefficients[position].expect("identified");
        let allowed = 8.0 * condition * condition * f64::EPSILON * scaled_max / norm;
        assert!(
            (got - want).abs() <= allowed,
            "{name} {column}: {got} against {want}, allowed {allowed:e}"
        );
    }
}

fn compare(
    what: &str,
    ours: &PropensityDistancePreparation,
    expected: &[f64],
    skip: &[bool],
    allowed: f64,
) -> f64 {
    let mut worst = 0.0_f64;
    for (idx, want) in expected.iter().enumerate() {
        if skip[idx] {
            continue;
        }
        let got = ours.estimate.distance_score(&id(idx)).unwrap();
        let deviation = (got - want).abs();
        worst = worst.max(deviation);
        assert!(
            deviation <= allowed,
            "{what} unit {idx}: {got} against {want}, allowed {allowed:e}"
        );
    }
    worst
}

/// The linear predictor and the probability agree with MatchIt's distance
/// under `link = "linear.*"` and `link = "*"`, for the logit, probit and cloglog
/// links, and the coefficients with `glm`'s. The iteration counts are equal.
#[test]
fn links_match_matchit() {
    let reference = reference();
    println!(
        "{} / MatchIt {}",
        reference.versions["r"], reference.versions["matchit"]
    );
    for fit in &reference.fits {
        let none = vec![false; fit.treat.len()];
        for r in &fit.links {
            let name = format!("{} {}", fit.name, r.link);
            assert!(r.converged, "{name}");
            let link = link_of(&r.link);
            let linear = fit_ours(
                fit,
                &Options {
                    link,
                    scale: PropensityScoreOutputScale::LinearPredictor,
                    weights: None,
                    discard: None,
                },
            );
            assert!(linear.estimate.converged, "{name}");
            assert_eq!(linear.estimate.iterations, r.iterations, "{name}");
            assert!(linear.estimate.aliased_columns.is_empty(), "{name}");
            let allowed = allowance(r.condition, max_abs(&r.linear));
            let worst = compare(&name, &linear, &r.linear, &none, allowed);
            println!("{name}: linear predictor deviation {worst:e}, allowed {allowed:e}");

            compare_coefficients(
                &name,
                &linear,
                r,
                &r.coefficients,
                r.condition,
                &r.column_norms,
            );

            let probability = fit_ours(
                fit,
                &Options {
                    link,
                    scale: PropensityScoreOutputScale::Probability,
                    weights: None,
                    discard: None,
                },
            );
            let allowed = MAX_SLOPE * allowance(r.condition, max_abs(&r.linear));
            let worst = compare(&name, &probability, &r.probability, &none, allowed);
            println!("{name}: probability deviation {worst:e}, allowed {allowed:e}");
        }
    }
}

/// Sampling weights enter the fit as prior weights, as MatchIt passes
/// `s.weights` to `glm()`: the distances and coefficients agree with
/// `matchit(s.weights = ...)`.
#[test]
fn sampling_weights_match_matchit() {
    let reference = reference();
    for fit in &reference.fits {
        let none = vec![false; fit.treat.len()];
        for r in &fit.links {
            let name = format!("{} {} weighted", fit.name, r.link);
            let link = link_of(&r.link);
            let linear = fit_ours(
                fit,
                &Options {
                    link,
                    scale: PropensityScoreOutputScale::LinearPredictor,
                    weights: Some(&fit.weights),
                    discard: None,
                },
            );
            assert_eq!(linear.estimate.iterations, r.weighted_iterations, "{name}");
            let allowed = allowance(r.weighted_condition, max_abs(&r.weighted_linear));
            let worst = compare(&name, &linear, &r.weighted_linear, &none, allowed);
            println!("{name}: linear predictor deviation {worst:e}, allowed {allowed:e}");
            compare_coefficients(
                &name,
                &linear,
                r,
                &r.weighted_coefficients,
                r.weighted_condition,
                &r.weighted_column_norms,
            );
            let probability = fit_ours(
                fit,
                &Options {
                    link,
                    scale: PropensityScoreOutputScale::Probability,
                    weights: Some(&fit.weights),
                    discard: None,
                },
            );
            let allowed = MAX_SLOPE * allowance(r.weighted_condition, max_abs(&r.weighted_linear));
            compare(&name, &probability, &r.weighted_probability, &none, allowed);
        }
    }
}

/// A missing, negative or non-finite sampling weight is an error.
#[test]
fn invalid_sampling_weights_are_errors() {
    let reference = reference();
    let fit = &reference.fits[1];
    for bad in [f64::NAN, -1.0, f64::INFINITY] {
        let mut weights = fit.weights.clone();
        weights[7] = bad;
        let birth = NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date");
        let record = |idx: usize| {
            let mut record = BalanceRecord::new(id(idx), birth);
            for (name, column) in fit.covariate_names.iter().zip(&fit.covariates) {
                record
                    .covariates
                    .insert(name.clone(), CovariateValue::Numeric(column[idx]));
            }
            record
        };
        let (anchors, candidates): (Vec<_>, Vec<_>) =
            (0..fit.treat.len()).partition(|idx| fit.treat[*idx] > 0.5);
        let anchors: Vec<_> = anchors.into_iter().map(record).collect();
        let candidates: Vec<_> = candidates.into_iter().map(record).collect();
        let config = PropensityScoreConfig {
            sampling_weights: Some(
                weights
                    .iter()
                    .enumerate()
                    .map(|(idx, w)| (id(idx), *w))
                    .collect(),
            ),
            ..PropensityScoreConfig::default()
        };
        let err = prepare_propensity_distance_config(&anchors, &candidates, &config)
            .expect_err("an invalid weight is refused");
        assert!(
            matches!(err, EstimationError::InvalidSamplingWeight { ref record_id } if record_id == "u7"),
            "{err:?}"
        );
    }
}

/// With `reestimate`, the model is refitted on the units that trimming keeps.
/// The discarded set and the refitted distances of the kept units agree with
/// `matchit(discard = ..., reestimate = TRUE)`, on the probability scale, on
/// the linear predictor scale of a probit fit, and with sampling weights. The
/// discarded units keep the distance of the first fit.
#[test]
fn reestimated_scores_match_matchit() {
    let reference = reference();
    let mut compared = 0;
    for fit in &reference.fits {
        for trimmed in &fit.trimmed {
            let policy = policy_of(&trimmed.policy);
            let cases = [
                (
                    "logit",
                    &trimmed.probability,
                    PropensityLink::Logit,
                    PropensityScoreOutputScale::Probability,
                    None,
                ),
                (
                    "linear probit",
                    &trimmed.linear_probit,
                    PropensityLink::Probit,
                    PropensityScoreOutputScale::LinearPredictor,
                    None,
                ),
                (
                    "logit weighted",
                    &trimmed.weighted,
                    PropensityLink::Logit,
                    PropensityScoreOutputScale::Probability,
                    Some(fit.weights.as_slice()),
                ),
            ];
            for (label, r, link, scale, weights) in cases {
                let name = format!("{} {} {label}", fit.name, trimmed.policy);
                assert!(r.same_discarded, "{name}");
                let ours = fit_ours(
                    fit,
                    &Options {
                        link,
                        scale,
                        weights,
                        discard: Some(policy),
                    },
                );
                let mut ours_discarded: Vec<String> = ours
                    .estimate
                    .discarded_anchors
                    .iter()
                    .chain(&ours.estimate.discarded_candidates)
                    .cloned()
                    .collect();
                ours_discarded.sort();
                let mut expected_discarded: Vec<String> = r
                    .discarded
                    .iter()
                    .enumerate()
                    .filter(|(_, dropped)| **dropped)
                    .map(|(idx, _)| id(idx))
                    .collect();
                expected_discarded.sort();
                assert_eq!(ours_discarded, expected_discarded, "{name}");

                let kept_max = max_abs(&r.distance);
                let eta_max = if scale == PropensityScoreOutputScale::Probability {
                    MAX_SLOPE * (1.0 + kept_max) * 20.0
                } else {
                    kept_max
                };
                let allowed = allowance(r.condition, eta_max);
                let worst = compare(&name, &ours, &r.distance, &r.discarded, allowed);
                println!("{name}: refitted distance deviation {worst:e}, allowed {allowed:e}");
                compared += 1;
            }
        }
    }
    assert!(compared > 0);
}

/// The two output scales of one fit are the logit of each other, and the
/// distance configuration carries the selected scale.
#[test]
fn output_scales_are_one_fit() {
    let reference = reference();
    let fit = &reference.fits[1];
    let ours = fit_ours(
        fit,
        &Options {
            link: PropensityLink::Logit,
            scale: PropensityScoreOutputScale::LinearPredictor,
            weights: None,
            discard: None,
        },
    );
    for (position, eta) in ours.estimate.linear_predictor.iter().enumerate() {
        let p = ours.estimate.probabilities[position];
        assert!((p - 1.0 / (1.0 + (-eta).exp())).abs() < 1e-12);
        assert_eq!(
            ours.estimate.distance_scores[position].to_bits(),
            eta.to_bits()
        );
    }
}
