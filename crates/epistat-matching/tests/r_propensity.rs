//! The logistic propensity score checked against MatchIt 4.8.2's
//! `matchit(distance = "glm")`.
//!
//! The reference is `tests/data/matchit_propensity.json`, written by
//! `tests/data/matchit_propensity.R`. It holds, for MatchIt's `lalonde` data
//! and two simulated data sets, the distance on the probability scale
//! (`link = "logit"`), on the linear predictor scale (`link = "linear.logit"`)
//! and the fitted coefficients, all from a `glm` fit run to an epsilon of
//! 1e-14.
//!
//! The Rust fit adds an L2 penalty of 1e-6 to the non-intercept coefficients,
//! so it does not reproduce the maximum likelihood fit exactly. The R script
//! computes the first-order shift of every linear predictor that the penalty
//! causes. The tests allow twice that shift plus a floor for rounding in the
//! normal equations.

use chrono::NaiveDate;
use epistat_matching::prelude::*;
use epistat_testkit::load_json;
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    fits: Vec<RFit>,
}

#[derive(Deserialize)]
struct Versions {
    r: String,
    matchit: String,
}

#[derive(Deserialize)]
struct RFit {
    name: String,
    treat: Vec<f64>,
    covariates: Vec<Vec<f64>>,
    covariate_names: Vec<String>,
    probability: Vec<f64>,
    linear: Vec<f64>,
    coefficient_names: Vec<String>,
    coefficients: Vec<f64>,
    linear_link_coefficients: Vec<f64>,
    penalty_shift_linear: Vec<f64>,
    penalty_shift_coefficients: Vec<f64>,
}

fn reference() -> Reference {
    load_json(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/matchit_propensity.json"
    ))
}

fn id(idx: usize) -> String {
    format!("u{idx}")
}

fn fit_ours(fit: &RFit, scale: PropensityScoreOutputScale) -> PropensityDistancePreparation {
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
    prepare_propensity_distance_config(
        &anchors,
        &candidates,
        &PropensityScoreConfig::builder().output_scale(scale).build(),
    )
    .expect("logistic fit succeeds")
}

/// Allowance for rounding in the normal equations, which are solved in double
/// precision: 1e-8 relative to the linear predictor, and 1e-9 absolute for
/// predictors near zero. The observed deviations are the penalty shift alone,
/// well inside this floor.
fn rounding_floor(eta: f64) -> f64 {
    1e-9 + 1e-8 * eta.abs()
}

/// The fitted linear predictor agrees with R's `link = "linear.logit"` distance
/// within the penalty's first-order shift, and the coefficients with R's `glm`
/// coefficients likewise.
#[test]
fn linear_predictor_scale_matches_matchit() {
    let reference = reference();
    println!(
        "{} / MatchIt {}",
        reference.versions.r, reference.versions.matchit
    );
    for fit in &reference.fits {
        let ours = fit_ours(fit, PropensityScoreOutputScale::LinearPredictor);
        assert!(ours.estimate.converged, "{}", fit.name);
        let mut worst = 0.0_f64;
        for (idx, expected) in fit.linear.iter().enumerate() {
            let got = ours.estimate.distance_scores[&id(idx)];
            let allowed = 2.0 * fit.penalty_shift_linear[idx] + rounding_floor(*expected);
            let ratio = (got - expected).abs() / allowed;
            worst = worst.max(ratio);
            assert!(
                ratio <= 1.0,
                "{} unit {idx}: {got} against {expected}, allowed {allowed}",
                fit.name
            );
        }
        println!(
            "{}: linear predictor worst ratio to allowance {worst:.3}",
            fit.name
        );
        // MatchIt's linear link and its logit link fit the same model.
        for (a, b) in fit.coefficients.iter().zip(&fit.linear_link_coefficients) {
            assert!((a - b).abs() <= 1e-12 * (1.0 + a.abs()), "{}", fit.name);
        }
        for (name, (expected, shift)) in fit
            .coefficient_names
            .iter()
            .zip(fit.coefficients.iter().zip(&fit.penalty_shift_coefficients))
        {
            let ours_name = if name == "(Intercept)" {
                "intercept"
            } else {
                name
            };
            let position = ours
                .estimate
                .column_names
                .iter()
                .position(|n| n == ours_name)
                .expect("column exists");
            let got = ours.estimate.coefficients[position];
            // Relative floor on the coefficient: columns of large scale have
            // tiny coefficients, and rounding acts on them relatively.
            let allowed = 2.0 * shift + 1e-6 * (1.0 + expected.abs());
            assert!(
                (got - expected).abs() <= allowed,
                "{} {name}: {got} against {expected}, allowed {allowed}",
                fit.name
            );
        }
    }
}

/// The probability-scale score agrees with R's `link = "logit"` distance. The
/// logistic function has slope at most 1/4, so the allowance is a quarter of
/// the linear predictor's.
#[test]
fn probability_scale_matches_matchit() {
    let reference = reference();
    for fit in &reference.fits {
        let ours = fit_ours(fit, PropensityScoreOutputScale::Probability);
        let mut worst = 0.0_f64;
        for (idx, expected) in fit.probability.iter().enumerate() {
            let got = ours.estimate.distance_scores[&id(idx)];
            let eta = fit.linear[idx];
            let allowed = 0.25 * (2.0 * fit.penalty_shift_linear[idx] + rounding_floor(eta));
            let ratio = (got - expected).abs() / allowed;
            worst = worst.max(ratio);
            assert!(
                ratio <= 1.0,
                "{} unit {idx}: {got} against {expected}, allowed {allowed}",
                fit.name
            );
        }
        println!(
            "{}: probability worst ratio to allowance {worst:.3}",
            fit.name
        );
    }
}

/// The two output scales of one fit are the logit of each other, and the
/// distance configuration carries the selected scale.
#[test]
fn output_scales_are_one_fit() {
    let reference = reference();
    let fit = &reference.fits[1];
    let ours = fit_ours(fit, PropensityScoreOutputScale::LinearPredictor);
    for (id, eta) in &ours.estimate.linear_predictor {
        let p = ours.estimate.probabilities[id];
        assert!((p - 1.0 / (1.0 + (-eta).exp())).abs() < 1e-12);
        assert_eq!(ours.estimate.distance_scores[id].to_bits(), eta.to_bits());
    }
}
