//! The covariate balancing propensity score checked against CBPS 0.24 and
//! MatchIt 4.8.2's `matchit(distance = "cbps")`.
//!
//! The reference is `tests/data/cbps.json`, written by `tests/data/cbps.R`. It
//! holds MatchIt's `lalonde` data and a simulated data set, each fitted by
//! `CBPS()` for the ATT and the ATE with `method = "over"` and `"exact"` at the
//! package's defaults, and MatchIt's distance for the same fits. MatchIt's
//! distance is the fitted value of `CBPS()` to the last bit, which the R script
//! asserts, so the comparison with MatchIt is the comparison with CBPS.
//!
//! CBPS stops at the tolerance of `optim()`, a relative change of the loss of
//! 1.5e-8, and this is far from the minimum when the loss is flat. The Rust fit
//! uses `epistat-optim`'s BFGS and stops on the gradient. The two therefore
//! differ by what the loss allows: if the loss at a point exceeds its minimum
//! by `d` and the smallest eigenvalue of the Hessian is `lambda`, the point
//! is within `sqrt(2 d / lambda)` of the minimum, in the orthogonalised
//! coordinates in which the design has orthonormal columns. A point whose
//! gradient has norm `g` is within `g / lambda`. For the exact fit, whose
//! balance moments are zero at the solution, a point whose moments have norm `m`
//! is within `m / sigma`, with `sigma` the smallest singular value of the
//! Jacobian of the moments. The script finds the minimum or the root with
//! numerical derivatives and writes `lambda`, `sigma` and the norms, so every
//! tolerance below is computed from them.
//!
//! The linear predictor of a unit is the inner product of a row of the
//! orthogonal design, which has norm at most 1, with the coefficients, so it
//! deviates by at most the distance in coordinates. A coefficient on the scale
//! of the covariates deviates by at most `t_norm` times that distance. A
//! probability deviates by at most a quarter of the linear predictor.
//!
//! For the ATT with the over-identified estimator CBPS stops far from the
//! minimum on both data sets: the loss is 19 per cent and 130 per cent above
//! its minimum and its gradient there is not zero. The fits of this crate are
//! compared with the minimum the script found, and the test asserts the size of
//! CBPS's shortfall.

#[path = "common/r_reference.rs"]
mod r_reference;

use chrono::NaiveDate;
use epistat_core::Estimand;
use epistat_matching::Error;
use epistat_matching::{
    BalanceRecord, CovariateEncodingConfig, CovariateValue, PropensityDistancePreparation,
    PropensityEstimator, PropensityScoreConfig, PropensityScoreOutputScale,
    prepare_propensity_distance_config,
};
use epistat_matching::{CbpsConfig, CbpsMethod};

use r_reference::{Versions, assert_r_version, reference as load_reference, unit_id as id};
use serde::Deserialize;

#[derive(Deserialize)]
struct Reference {
    versions: Versions,
    data: Vec<Data>,
}

#[derive(Deserialize)]
struct Data {
    name: String,
    treat: Vec<f64>,
    covariate_names: Vec<String>,
    covariates: Vec<Vec<f64>>,
    fits: Vec<RFit>,
}

#[derive(Deserialize)]
struct RFit {
    estimand: String,
    method: String,
    coefficient_names: Vec<String>,
    coefficients: Vec<f64>,
    fitted: Vec<f64>,
    linear: Vec<f64>,
    #[serde(rename = "J")]
    j: f64,
    #[serde(rename = "mle_J")]
    logistic_j: f64,
    matchit_distance: Vec<f64>,
    matchit_linear: Vec<f64>,
    start_scale: f64,
    v_cond: f64,
    j_scale: f64,
    loss_at_cbps: f64,
    star_coefficients: Vec<f64>,
    star_linear: Vec<f64>,
    #[serde(rename = "star_J")]
    star_j: f64,
    grad_cbps: f64,
    grad_star: f64,
    lambda_min: Option<f64>,
    sigma_min: Option<f64>,
    moment_cbps: Option<f64>,
    jhess_norm: f64,
    t_norm: f64,
    balance_cbps: Vec<f64>,
    balance_star: Vec<f64>,
}

fn reference() -> Reference {
    load_reference("cbps.json")
}

fn fit_ours(
    data: &Data,
    fit: &RFit,
    scale: PropensityScoreOutputScale,
) -> PropensityDistancePreparation {
    let birth = NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date");
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for (idx, label) in data.treat.iter().enumerate() {
        let mut record = BalanceRecord::new(id(idx), birth);
        for (name, column) in data.covariate_names.iter().zip(&data.covariates) {
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
    let cbps = CbpsConfig {
        estimand: match fit.estimand.as_str() {
            "att" => Estimand::Att,
            "ate" => Estimand::Ate,
            other => panic!("unknown estimand {other}"),
        },
        method: match fit.method.as_str() {
            "over" => CbpsMethod::Over,
            "exact" => CbpsMethod::Exact,
            other => panic!("unknown method {other}"),
        },
        ..CbpsConfig::default()
    };
    let config = PropensityScoreConfig {
        estimator: PropensityEstimator::Cbps(cbps),
        output_scale: scale,
        ..PropensityScoreConfig::default()
    };
    prepare_propensity_distance_config(&anchors, &candidates, &config, None)
        .expect("the fit succeeds")
}

/// The values of `column`, which is aligned with the ids of `estimate`, in the
/// order of the data rows.
fn in_row_order(
    estimate: &epistat_matching::PropensityScoreEstimate,
    column: &[f64],
    n: usize,
) -> Vec<f64> {
    (0..n)
        .map(|idx| column[estimate.position(&id(idx)).expect("a unit")])
        .collect()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .fold(0.0_f64, |acc, (x, y)| acc.max((x - y).abs()))
}

/// The distance in the orthogonal coordinates of a point from the minimum or
/// root, from the gradient or moment norm of that point.
///
/// Over-identified: `g / lambda`. Exact: `m / sigma`.
fn distance_from_gradient(fit: &RFit, gradient: f64, moments: f64) -> f64 {
    match fit.method.as_str() {
        "over" => gradient / fit.lambda_min.expect("over fit has lambda"),
        _ => moments / fit.sigma_min.expect("exact fit has sigma"),
    }
}

/// The balance moment `(1/n) sum w_i x_i` on the original design, the
/// intercept first, for fitted probabilities `p`.
fn balance_of(data: &Data, fit: &RFit, p: &[f64]) -> Vec<f64> {
    let n = p.len() as f64;
    let n_t: f64 = data.treat.iter().sum();
    let weight = |i: usize| -> f64 {
        let t = data.treat[i];
        if fit.estimand == "att" {
            n / n_t * (t - p[i]) / (1.0 - p[i])
        } else {
            1.0 / (p[i] - 1.0 + t)
        }
    };
    let mut out = vec![0.0; 1 + data.covariates.len()];
    for i in 0..p.len() {
        out[0] += weight(i);
        for (j, column) in data.covariates.iter().enumerate() {
            out[1 + j] += weight(i) * column[i];
        }
    }
    out.iter().map(|x| x / n).collect()
}

/// A first-order bound on the change of the balance moments, column by column,
/// when every linear predictor moves by at most `delta`: `(delta / n) sum
/// |x_ij| |dw_i / d eta_i|`.
fn balance_bound(data: &Data, fit: &RFit, p: &[f64], delta: f64) -> Vec<f64> {
    let n = p.len() as f64;
    let n_t: f64 = data.treat.iter().sum();
    let slope = |i: usize| -> f64 {
        let t = data.treat[i];
        if fit.estimand == "att" {
            if t > 0.5 {
                0.0
            } else {
                n / n_t * p[i] / (1.0 - p[i])
            }
        } else if t > 0.5 {
            (1.0 - p[i]) / p[i]
        } else {
            p[i] / (1.0 - p[i])
        }
    };
    let mut out = vec![0.0; 1 + data.covariates.len()];
    for i in 0..p.len() {
        out[0] += slope(i);
        for (j, column) in data.covariates.iter().enumerate() {
            out[1 + j] += slope(i) * column[i].abs();
        }
    }
    out.iter().map(|x| x * delta / n).collect()
}

#[test]
fn versions_are_recorded() {
    let reference = reference();
    assert_r_version(&reference.versions["r"]);
    assert_eq!(reference.versions["cbps"], "0.24");
    assert_eq!(reference.versions["matchit"], "4.8.2");
}

#[test]
fn cbps_matches_the_minimum_and_cbps() {
    let reference = reference();
    for data in &reference.data {
        let n = data.treat.len();
        for fit in &data.fits {
            let label = format!("{} {} {}", data.name, fit.estimand, fit.method);
            let ours = fit_ours(data, fit, PropensityScoreOutputScale::Probability);
            let estimate = &ours.estimate;
            let diag = estimate.cbps.expect("a CBPS fit has diagnostics");
            assert!(estimate.converged, "{label}: not converged");
            assert!(!estimate.fitted_at_boundary, "{label}");
            assert!(estimate.aliased_columns.is_empty(), "{label}");
            assert_eq!(
                estimate.estimator,
                format!("cbps_{}_{}", fit.method, fit.estimand),
                "{label}"
            );

            // The distance of our fit from the minimum or the root found in R.
            let ours_distance = distance_from_gradient(
                fit,
                diag.gradient_norm + fit.grad_star,
                diag.balance_residual_norm,
            );
            let theta = in_row_order(estimate, &estimate.linear_predictor, n);
            let to_star = max_abs_diff(&theta, &fit.star_linear);
            assert!(
                to_star <= ours_distance,
                "{label}: linear predictor {to_star:e} from the minimum, bound {ours_distance:e}"
            );
            // The coefficients, by name.
            for (name, want) in fit.coefficient_names.iter().zip(&fit.star_coefficients) {
                let ours_name = if name == "(Intercept)" {
                    "intercept"
                } else {
                    name
                };
                let position = estimate
                    .column_names
                    .iter()
                    .position(|c| c == ours_name)
                    .unwrap_or_else(|| panic!("{label}: column {ours_name}"));
                let got = estimate.coefficients[position].expect("identified");
                assert!(
                    (got - want).abs() <= fit.t_norm * ours_distance,
                    "{label}: coefficient {name} {got} against {want}, bound {:e}",
                    fit.t_norm * ours_distance
                );
            }
            // The weight matrix is the inverse of a matrix of condition number
            // `v_cond` on its kept eigenvalues, so its inverse has an error of
            // about `eps v_cond` relative to its norm, and two evaluations of J
            // differ by about `eps v_cond` times `j_scale`, the norm of the
            // inverse times the squared norm of the moments. The allowance is
            // eight times that.
            let rounding = 8.0 * f64::EPSILON * fit.v_cond * fit.j_scale;
            assert!(
                (diag.start_scale - fit.start_scale).abs() <= 1e-9,
                "{label}: scale {} against {}",
                diag.start_scale,
                fit.start_scale
            );
            // J at the minimum: for a point at distance `d`, J exceeds its
            // value at the minimum by at most `grad d + hess d^2 / 2`.
            let j_bound = fit.grad_star * ours_distance
                + 0.5 * fit.jhess_norm * ours_distance * ours_distance
                + rounding;
            assert!(
                (diag.j_statistic - fit.star_j).abs() <= j_bound,
                "{label}: J {} against {}, bound {j_bound:e}",
                diag.j_statistic,
                fit.star_j
            );
            assert!(
                diag.logistic_j_statistic >= diag.j_statistic * (1.0 - 1e-12)
                    || fit.method == "exact"
            );
            assert!(
                (diag.logistic_j_statistic - fit.logistic_j).abs()
                    <= 1e-9 * fit.logistic_j.abs().max(1.0),
                "{label}: J at the logistic fit {} against {}",
                diag.logistic_j_statistic,
                fit.logistic_j
            );

            // The balance moments, against the moments at the minimum.
            let p = in_row_order(estimate, &estimate.probabilities, n);
            let balance = balance_of(data, fit, &p);
            let bound = balance_bound(data, fit, &p, ours_distance);
            for ((got, want), tol) in balance.iter().zip(&fit.balance_star).zip(&bound) {
                assert!(
                    (got - want).abs() <= *tol + 1e-12 * want.abs().max(1.0),
                    "{label}: balance {got} against {want}, bound {tol:e}"
                );
            }
            if fit.method == "exact" {
                // The balance conditions hold: the moments are within their
                // first-order bound of zero.
                for (got, tol) in balance.iter().zip(&bound) {
                    assert!(got.abs() <= *tol + 1e-12, "{label}: balance {got}");
                }
            }

            // CBPS and MatchIt. CBPS's own distance from the minimum is
            // measured by the loss it reached (over) or the moments (exact).
            let cbps_gap = fit.j - fit.star_j;
            let stationary = fit.method == "exact" || cbps_gap <= 1e-3 * fit.star_j;
            assert_eq!(fit.matchit_distance, fit.fitted, "{label}");
            if stationary {
                let cbps_distance = match fit.method.as_str() {
                    "over" => (2.0 * cbps_gap.max(0.0) / fit.lambda_min.expect("lambda")).sqrt(),
                    _ => fit.moment_cbps.expect("moments") / fit.sigma_min.expect("sigma"),
                };
                let allowed = cbps_distance + ours_distance;
                for (what, got, want) in [
                    ("fitted", &p, &fit.fitted),
                    ("matchit", &p, &fit.matchit_distance),
                ] {
                    let diff = max_abs_diff(got, want);
                    assert!(
                        diff <= allowed / 4.0 + 1e-9,
                        "{label}: {what} probabilities differ by {diff:e}, allowed {:e}",
                        allowed / 4.0
                    );
                }
                for (what, want) in [
                    ("linear", &fit.linear),
                    ("matchit linear", &fit.matchit_linear),
                ] {
                    let diff = max_abs_diff(&theta, want);
                    assert!(
                        diff <= allowed,
                        "{label}: {what} {diff:e} allowed {allowed:e}"
                    );
                }
                for (name, want) in fit.coefficient_names.iter().zip(&fit.coefficients) {
                    let ours_name = if name == "(Intercept)" {
                        "intercept"
                    } else {
                        name
                    };
                    let position = estimate
                        .column_names
                        .iter()
                        .position(|c| c == ours_name)
                        .expect("column");
                    let got = estimate.coefficients[position].expect("identified");
                    assert!(
                        (got - want).abs() <= fit.t_norm * allowed,
                        "{label}: coefficient {name} {got} against CBPS {want}"
                    );
                }
                // J: CBPS's J at its own point against ours. The loss is
                // smooth, so J differs by at most the gradient at the
                // minimum times the distance plus the curvature term.
                let j_tol =
                    fit.grad_star * allowed + 0.5 * fit.jhess_norm * allowed * allowed + rounding;
                assert!(
                    (diag.j_statistic - fit.j).abs() <= j_tol + 1e-12,
                    "{label}: J {} against CBPS {}, allowed {j_tol:e}",
                    diag.j_statistic,
                    fit.j
                );
                let balance_r = &fit.balance_cbps;
                let bound_r = balance_bound(data, fit, &p, allowed);
                for ((got, want), tol) in balance.iter().zip(balance_r).zip(&bound_r) {
                    assert!(
                        (got - want).abs() <= *tol + 1e-12 * want.abs().max(1.0),
                        "{label}: balance {got} against CBPS {want}, bound {tol:e}"
                    );
                }
            } else {
                // CBPS stopped above the minimum of its own loss. Its gradient
                // there is not zero, and ours reaches a lower loss.
                assert_eq!(fit.method, "over", "{label}");
                assert!(cbps_gap > 0.1 * fit.star_j, "{label}: gap {cbps_gap}");
                assert!(
                    fit.grad_cbps > 1e6 * fit.grad_star.max(diag.gradient_norm),
                    "{label}: CBPS gradient {}",
                    fit.grad_cbps
                );
                assert!(diag.j_statistic < fit.j, "{label}");
                assert!(
                    (fit.loss_at_cbps - fit.j).abs() <= 1e-9 * fit.j,
                    "{label}: the loss written in the script reproduces CBPS's J"
                );
            }
        }
    }
}

#[test]
fn the_linear_scale_distance_is_the_linear_predictor() {
    let reference = reference();
    for data in &reference.data {
        let n = data.treat.len();
        for fit in &data.fits {
            let label = format!("{} {} {}", data.name, fit.estimand, fit.method);
            let ours = fit_ours(data, fit, PropensityScoreOutputScale::LinearPredictor);
            let scores = in_row_order(&ours.estimate, &ours.estimate.distance_scores, n);
            let theta = in_row_order(&ours.estimate, &ours.estimate.linear_predictor, n);
            assert_eq!(scores, theta, "{label}");
        }
    }
}

/// Anchors and candidates of the first data set with the covariates of
/// `names`.
fn records_of(data: &Data, names: &[&str]) -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
    let birth = NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date");
    let mut anchors = Vec::new();
    let mut candidates = Vec::new();
    for (idx, label) in data.treat.iter().enumerate() {
        let mut record = BalanceRecord::new(id(idx), birth);
        for (name, column) in data.covariate_names.iter().zip(&data.covariates) {
            if names.contains(&name.as_str()) {
                record
                    .covariates
                    .insert(name.clone(), CovariateValue::Numeric(column[idx]));
            }
        }
        if *label > 0.5 {
            anchors.push(record);
        } else {
            candidates.push(record);
        }
    }
    (anchors, candidates)
}

#[test]
fn equal_sampling_weights_change_nothing_and_bad_settings_are_errors() {
    let reference = reference();
    let data = &reference.data[1];
    let (anchors, candidates) = records_of(data, &["x1", "x2", "x3"]);
    let fit = |config: PropensityScoreConfig| {
        prepare_propensity_distance_config(&anchors, &candidates, &config, None)
    };
    let cbps = |config: CbpsConfig| PropensityScoreConfig {
        estimator: PropensityEstimator::Cbps(config),
        ..PropensityScoreConfig::default()
    };
    let plain = fit(cbps(CbpsConfig::default())).expect("fits");
    // Weights are scaled to mean 1, so a constant weight is no weight. Both
    // fits are within `g / lambda` of the same minimum, which is at most 1.2e-7
    // for the gradient tolerance of 3.7e-11 and the smallest eigenvalue of
    // 3e-4 of the reference fits.
    let mut weighted = cbps(CbpsConfig::default());
    weighted.sampling_weights = Some((0..data.treat.len()).map(|idx| (id(idx), 3.0)).collect());
    let weighted = fit(weighted).expect("fits");
    for (key, want) in plain
        .estimate
        .ids
        .iter()
        .zip(&plain.estimate.linear_predictor)
    {
        let got = weighted.estimate.linear_predictor_of(key).expect("a unit");
        assert!((got - want).abs() <= 3e-7, "{key}: {got} against {want}");
    }

    // A fit stopped by the iteration limit is reported, not repaired.
    let stopped = fit(cbps(CbpsConfig {
        max_iter: 2,
        ..CbpsConfig::default()
    }))
    .expect("fits");
    assert!(!stopped.estimate.converged);

    // No intercept, and a collinear column, are errors.
    let no_intercept = PropensityScoreConfig {
        encoding: CovariateEncodingConfig {
            include_intercept: false,
            ..CovariateEncodingConfig::default()
        },
        estimator: PropensityEstimator::Cbps(CbpsConfig::default()),
        ..PropensityScoreConfig::default()
    };
    assert!(matches!(fit(no_intercept), Err(Error::CbpsNeedsIntercept)));
    let (mut anchors, mut candidates) = records_of(data, &["x1", "x2", "x3"]);
    for record in anchors.iter_mut().chain(candidates.iter_mut()) {
        let CovariateValue::Numeric(x1) = record.covariates["x1"] else {
            panic!("numeric")
        };
        record
            .covariates
            .insert("twice".into(), CovariateValue::Numeric(2.0 * x1));
    }
    assert!(matches!(
        prepare_propensity_distance_config(
            &anchors,
            &candidates,
            &cbps(CbpsConfig::default()),
            None
        ),
        Err(Error::NotFullRank)
    ));
}
