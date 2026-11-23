//! # Model comparison utilities
//!
//! Provides in-sample metrics, quasi information criteria, and cross-validation
//! summaries for two-part, Tweedie, and log-normal models.

use thiserror::Error;

use super::lognormal::{
    LogNormalError, LogNormalModel, LogNormalOptions, fit_lognormal_smearing_input,
    log_likelihood as lognormal_log_likelihood,
};
use super::selection::{
    CrossValidationError, CrossValidationOptions, CrossValidationResult, ModelFitMetrics,
    TweedieCandidate, compute_information_criteria, compute_model_fit_metrics,
    cross_validate_models_input, fit_tweedie_candidates_input,
};
use super::tweedie::{
    TweedieError, TweedieModel, TweedieOptions, fit_tweedie_input,
    quasi_log_likelihood as tweedie_quasi_log_likelihood,
};
use super::two_part::{
    FitOptions, TwoPartError, fit_two_part_input, log_likelihood as two_part_log_likelihood,
};
use crate::input::{InputError, ModelInput};
use crate::models::Model;

type TwoPartFit = (
    super::two_part::TwoPartModel,
    super::two_part::TwoPartReport,
);

/// The models a comparison fitted on the full sample.
#[derive(Debug, Clone, Default)]
pub struct FittedComparisonModels {
    /// The two-part model and its report.
    pub two_part: Option<TwoPartFit>,
    /// The two-part model with the elastic-net options, when they were given.
    pub two_part_elastic_net: Option<TwoPartFit>,
    /// The Tweedie model at the comparison's power.
    pub tweedie: Option<TweedieModel>,
    /// The log-normal model.
    pub lognormal: Option<LogNormalModel>,
}

/// Configuration for model comparisons.
#[derive(bon::Builder, Debug, Clone)]
pub struct ModelComparisonOptions {
    /// The Tweedie powers to fit. The comparison's Tweedie model takes 1.5 when listed, else the first.
    #[builder(default = vec![1.2, 1.5, 1.8])]
    pub tweedie_powers: Vec<f64>,
    /// Options of the two-part fit.
    #[builder(default)]
    pub two_part_options: FitOptions,
    /// Options of a second two-part fit with an elastic-net penalty.
    pub two_part_elastic_net_options: Option<FitOptions>,
    /// Options of the Tweedie fits.
    #[builder(default)]
    pub tweedie_options: TweedieOptions,
    /// Options of the log-normal fit.
    #[builder(default)]
    pub lognormal_options: LogNormalOptions,
    /// The folds and seed of the cross-validation.
    #[builder(default)]
    pub cv_options: CrossValidationOptions,
    /// Whether to fit the two-part model.
    #[builder(default = true)]
    pub include_two_part: bool,
    /// Whether to fit the Tweedie model.
    #[builder(default = true)]
    pub include_tweedie: bool,
    /// Whether to fit the log-normal model.
    #[builder(default = true)]
    pub include_lognormal: bool,
}

impl Default for ModelComparisonOptions {
    fn default() -> Self {
        Self {
            tweedie_powers: vec![1.2, 1.5, 1.8],
            two_part_options: FitOptions::default(),
            two_part_elastic_net_options: None,
            tweedie_options: TweedieOptions::default(),
            lognormal_options: LogNormalOptions::default(),
            cv_options: CrossValidationOptions::default(),
            include_two_part: true,
            include_tweedie: true,
            include_lognormal: true,
        }
    }
}

/// Errors returned by the comparison workflow.
#[derive(Debug, Error)]
pub enum ModelComparisonError {
    /// Invalid model input.
    #[error("invalid model input: {0}")]
    Input(#[from] InputError),
    /// Two-part fit failed.
    #[error("two-part fit failed: {0}")]
    TwoPart(#[from] TwoPartError),
    /// Tweedie fit failed.
    #[error("tweedie fit failed: {0}")]
    Tweedie(#[from] TweedieError),
    /// Log-normal fit failed.
    #[error("log-normal fit failed: {0}")]
    LogNormal(#[from] LogNormalError),
    /// Cross-validation failed.
    #[error("cross-validation failed: {0}")]
    CrossValidation(#[from] CrossValidationError),
}

/// Per-model metrics with a descriptive label.
#[derive(Debug, Clone)]
pub struct ModelScore {
    /// The model's label.
    pub name: String,
    /// The fit metrics.
    pub metrics: ModelFitMetrics,
}

/// Per-model information criteria with a descriptive label.
#[derive(Debug, Clone)]
pub struct ModelInformationCriteria {
    /// The model's label.
    pub name: String,
    /// The log-likelihood, a quasi-likelihood for Tweedie.
    pub loglik: f64,
    /// Akaike's information criterion.
    pub aic: f64,
    /// The Bayesian information criterion.
    pub bic: f64,
}

/// Tweedie CV ranking row (metrics + in-sample IC).
#[derive(Debug, Clone)]
pub struct TweedieRankingRow {
    /// The Tweedie power.
    pub power: f64,
    /// The cross-validated metrics.
    pub metrics: ModelFitMetrics,
    /// Akaike's information criterion in the full sample.
    pub aic: f64,
    /// The Bayesian information criterion in the full sample.
    pub bic: f64,
}

/// Comparison output for model evaluation.
#[derive(Debug, Clone)]
pub struct ModelComparison {
    /// Metrics of each model in the full sample.
    pub in_sample: Vec<ModelScore>,
    /// Information criteria of each model in the full sample.
    pub information_criteria: Vec<ModelInformationCriteria>,
    /// Cross-validated metrics of each model.
    pub cv_summary: Vec<ModelScore>,
    /// The cross-validated metrics ordered from best to worst.
    pub cv_ranking: Vec<ModelScore>,
    /// Information criteria of each Tweedie power.
    pub tweedie_ic: Vec<ModelInformationCriteria>,
    /// The Tweedie powers ordered by their cross-validated metrics.
    pub tweedie_cv_ranking: Vec<TweedieRankingRow>,
}

#[derive(Debug, Clone, Default)]
struct InSamplePredictions {
    two_part: Option<super::two_part::TwoPartPrediction>,
    two_part_elastic_net: Option<super::two_part::TwoPartPrediction>,
    tweedie: Option<super::tweedie::TweediePrediction>,
    lognormal: Option<super::lognormal::LogNormalPrediction>,
}

/// Run model comparisons for a `ModelInput` dataset.
///
/// Notes:
/// - Tweedie AIC/BIC use quasi-likelihood (-0.5 * deviance).
/// - Log-normal log-likelihood is evaluated on positive outcomes only.
///
/// # Errors
///
/// Returns `ModelComparisonError` if any model fit or cross-validation step fails.
pub fn compare_models_input(
    input: &ModelInput,
    options: &ModelComparisonOptions,
) -> Result<ModelComparison, ModelComparisonError> {
    compare_models_input_with_fits(input, options).map(|(comparison, _)| comparison)
}

/// [`compare_models_input`], returning the fitted models with the comparison.
///
/// # Errors
///
/// As [`compare_models_input`].
pub fn compare_models_input_with_fits(
    input: &ModelInput,
    options: &ModelComparisonOptions,
) -> Result<(ModelComparison, FittedComparisonModels), ModelComparisonError> {
    input.validate()?;

    let two_part_default = options
        .include_two_part
        .then(|| fit_two_part_input(input, options.two_part_options))
        .transpose()?;
    let two_part_elastic_net_options = options.two_part_elastic_net_options;
    let two_part_elastic_net = options
        .include_two_part
        .then_some(two_part_elastic_net_options)
        .flatten()
        .map(|elastic_net_options| fit_two_part_input(input, elastic_net_options))
        .transpose()?;

    let tweedie_power = options
        .tweedie_powers
        .iter()
        .copied()
        .find(|power| (*power - 1.5).abs() < 1.0e-12)
        .or_else(|| options.tweedie_powers.first().copied())
        .unwrap_or(1.5);
    let tweedie_model = options
        .include_tweedie
        .then(|| fit_tweedie_input(input, tweedie_power, options.tweedie_options))
        .transpose()?
        .map(|(model, _report)| model);
    let lognormal_model = if options.include_lognormal {
        let (model, _report) = fit_lognormal_smearing_input(input, options.lognormal_options)?;
        Some(model)
    } else {
        None
    };

    let tweedie_candidates = fit_tweedie_candidates_input(input, &options.tweedie_powers);

    let predictions = build_in_sample_predictions(
        input,
        two_part_default.as_ref(),
        two_part_elastic_net.as_ref(),
        tweedie_model.as_ref(),
        lognormal_model.as_ref(),
    );
    let in_sample = build_in_sample(input, &predictions, tweedie_model.as_ref());
    let information_criteria = build_information_criteria(
        input,
        &predictions,
        tweedie_model.as_ref(),
        lognormal_model.as_ref(),
    );
    let tweedie_ic = build_tweedie_candidate_ic(&tweedie_candidates);

    let cv_default = cross_validate_models_input(
        input,
        &options.tweedie_powers,
        CrossValidationOptions {
            two_part_options: options.two_part_options,
            tweedie_options: options.tweedie_options,
            lognormal_options: options.lognormal_options,
            include_two_part: options.include_two_part,
            include_lognormal: options.include_lognormal,
            ..options.cv_options
        },
    )?;
    let cv_elastic_net = if let Some(elastic_net_options) = two_part_elastic_net_options {
        Some(cross_validate_models_input(
            input,
            &[],
            CrossValidationOptions {
                two_part_options: elastic_net_options,
                tweedie_options: options.tweedie_options,
                lognormal_options: options.lognormal_options,
                include_two_part: true,
                include_lognormal: options.include_lognormal,
                ..options.cv_options
            },
        )?)
    } else {
        None
    };
    let cv_summary = build_cv_summary(&cv_default, cv_elastic_net.as_ref());
    let cv_ranking = build_cv_ranking(&cv_summary);
    let tweedie_cv_ranking = build_tweedie_cv_ranking(&cv_default, &tweedie_candidates);

    Ok((
        ModelComparison {
            in_sample,
            information_criteria,
            cv_summary,
            cv_ranking,
            tweedie_ic,
            tweedie_cv_ranking,
        },
        FittedComparisonModels {
            two_part: two_part_default,
            two_part_elastic_net,
            tweedie: tweedie_model,
            lognormal: lognormal_model,
        },
    ))
}

fn build_in_sample_predictions(
    input: &ModelInput,
    two_part: Option<&TwoPartFit>,
    two_part_elastic_net: Option<&TwoPartFit>,
    tweedie: Option<&TweedieModel>,
    lognormal: Option<&LogNormalModel>,
) -> InSamplePredictions {
    InSamplePredictions {
        two_part: two_part.map(|fit| fit.0.predict(&input.design_matrix)),
        two_part_elastic_net: two_part_elastic_net.map(|fit| fit.0.predict(&input.design_matrix)),
        tweedie: tweedie.map(|model| model.predict(&input.design_matrix)),
        lognormal: lognormal.map(|model| model.predict(&input.design_matrix)),
    }
}

fn build_in_sample(
    input: &ModelInput,
    predictions: &InSamplePredictions,
    tweedie: Option<&TweedieModel>,
) -> Vec<ModelScore> {
    let mut scores = Vec::new();
    if let Some(two_part_pred) = predictions.two_part.as_ref() {
        scores.push(ModelScore {
            name: "two_part".to_string(),
            metrics: compute_model_fit_metrics(
                &input.outcome,
                &two_part_pred.expected_outcome,
                None,
            ),
        });
    }
    if let Some(pred) = predictions.two_part_elastic_net.as_ref() {
        scores.push(ModelScore {
            name: "two_part_elastic_net".to_string(),
            metrics: compute_model_fit_metrics(&input.outcome, &pred.expected_outcome, None),
        });
    }
    if let (Some(tweedie), Some(tweedie_pred)) = (tweedie, predictions.tweedie.as_ref()) {
        scores.push(ModelScore {
            name: format!("tweedie p={:.1}", tweedie.power),
            metrics: compute_model_fit_metrics(
                &input.outcome,
                &tweedie_pred.mean,
                Some(tweedie.power),
            ),
        });
    }
    if let Some(pred) = predictions.lognormal.as_ref() {
        scores.push(ModelScore {
            name: "lognormal".to_string(),
            metrics: compute_model_fit_metrics(&input.outcome, &pred.mean, None),
        });
    }
    scores
}

fn build_information_criteria(
    input: &ModelInput,
    predictions: &InSamplePredictions,
    tweedie: Option<&TweedieModel>,
    lognormal: Option<&LogNormalModel>,
) -> Vec<ModelInformationCriteria> {
    let mut rows = Vec::new();
    if let Some(two_part_pred) = predictions.two_part.as_ref() {
        let ll_two_part = two_part_log_likelihood(
            &input.outcome,
            &two_part_pred.prob_positive,
            &two_part_pred.mean_positive,
        );
        let ic_two_part = compute_information_criteria(
            ll_two_part,
            2 * input.design_matrix.ncols(),
            input.outcome.nrows(),
        );
        rows.push(ModelInformationCriteria {
            name: "two_part".to_string(),
            loglik: ic_two_part.loglik,
            aic: ic_two_part.aic,
            bic: ic_two_part.bic,
        });
    }
    if let Some(pred) = predictions.two_part_elastic_net.as_ref() {
        let ll = two_part_log_likelihood(&input.outcome, &pred.prob_positive, &pred.mean_positive);
        let ic = compute_information_criteria(
            ll,
            2 * input.design_matrix.ncols(),
            input.outcome.nrows(),
        );
        rows.push(ModelInformationCriteria {
            name: "two_part_elastic_net".to_string(),
            loglik: ic.loglik,
            aic: ic.aic,
            bic: ic.bic,
        });
    }
    if let (Some(tweedie), Some(tweedie_pred)) = (tweedie, predictions.tweedie.as_ref()) {
        let ll_tweedie =
            tweedie_quasi_log_likelihood(&input.outcome, &tweedie_pred.mean, tweedie.power);
        let ic_tweedie = compute_information_criteria(
            ll_tweedie,
            input.design_matrix.ncols(),
            input.outcome.nrows(),
        );
        rows.push(ModelInformationCriteria {
            name: format!("tweedie p={:.1}", tweedie.power),
            loglik: ic_tweedie.loglik,
            aic: ic_tweedie.aic,
            bic: ic_tweedie.bic,
        });
    }
    if let Some(lognormal) = lognormal {
        let ll = lognormal_log_likelihood(&input.design_matrix, &input.outcome, lognormal);
        if ll.is_finite() {
            let ic = compute_information_criteria(
                ll,
                input.design_matrix.ncols(),
                input.outcome.nrows(),
            );
            rows.push(ModelInformationCriteria {
                name: "lognormal".to_string(),
                loglik: ic.loglik,
                aic: ic.aic,
                bic: ic.bic,
            });
        }
    }
    rows
}

fn build_tweedie_candidate_ic(candidates: &[TweedieCandidate]) -> Vec<ModelInformationCriteria> {
    candidates
        .iter()
        .map(|candidate| ModelInformationCriteria {
            name: format!("{:.2}", candidate.power),
            loglik: candidate.information_criteria.loglik,
            aic: candidate.information_criteria.aic,
            bic: candidate.information_criteria.bic,
        })
        .collect()
}

fn build_cv_summary(
    cv_default: &CrossValidationResult,
    cv_elastic_net: Option<&CrossValidationResult>,
) -> Vec<ModelScore> {
    let mut rows = Vec::new();
    if cv_default.include_two_part {
        rows.push(ModelScore {
            name: "two_part".to_string(),
            metrics: cv_default.two_part_metrics,
        });
    }
    if let Some(elastic_net) = cv_elastic_net {
        rows.push(ModelScore {
            name: "two_part_elastic_net".to_string(),
            metrics: elastic_net.two_part_metrics,
        });
    }
    for candidate in &cv_default.tweedie_candidates {
        rows.push(ModelScore {
            name: format!("tweedie p={:.1}", candidate.power),
            metrics: candidate.metrics,
        });
    }
    if cv_default.include_lognormal
        && let Some(lognormal) = &cv_default.lognormal_metrics
    {
        rows.push(ModelScore {
            name: "lognormal".to_string(),
            metrics: *lognormal,
        });
    }
    rows
}

/// How far past the mean predictor a CV score may fall and still be a score.
///
/// Matches the runner's `DEFAULT_MAGNITUDE_GUARD_K`: something is implausible
/// once it exceeds its reference scale by 50x. Here the reference is the
/// trivial mean predictor, whose RMSE is the outcome's own spread and whose r2
/// is 0 by definition, so `rmse > 50 * rmse_mean` is exactly `r2 < -2499`.
const CV_MAGNITUDE_GUARD_K: f64 = 50.0;

/// Whether a cross-validated score can be compared with the others at all.
///
/// A model that generalises badly still has a score: `two_part` at r2 = -0.06
/// and a Tweedie at r2 = -3.7 are poor fits, and ranking them is meaningful.
/// A model whose held-out RMSE is ten orders of magnitude above the outcome
/// scale has not produced a score, it has failed, and putting it in a ranked
/// list says it came last when what happened is that it did not run.
fn cv_score_is_comparable(metrics: &ModelFitMetrics) -> bool {
    metrics.rmse.is_finite()
        && metrics.r2.is_finite()
        && metrics.r2 >= 1.0 - CV_MAGNITUDE_GUARD_K * CV_MAGNITUDE_GUARD_K
}

/// Comparable scores first, ordered by RMSE; failures after them, and within
/// each group ties broken by name so the order does not depend on the order the
/// candidates happened to be built in.
fn cv_order(a: &ModelScore, b: &ModelScore) -> std::cmp::Ordering {
    cv_score_is_comparable(&b.metrics)
        .cmp(&cv_score_is_comparable(&a.metrics))
        .then_with(|| a.metrics.rmse.total_cmp(&b.metrics.rmse))
        .then_with(|| a.name.cmp(&b.name))
}

fn build_cv_ranking(cv_summary: &[ModelScore]) -> Vec<ModelScore> {
    let mut rows = cv_summary.to_vec();
    rows.sort_by(cv_order);
    rows
}

fn build_tweedie_cv_ranking(
    cv_default: &CrossValidationResult,
    tweedie_candidates: &[TweedieCandidate],
) -> Vec<TweedieRankingRow> {
    let mut candidates = cv_default.tweedie_candidates.clone();
    candidates.sort_by(|a, b| {
        cv_score_is_comparable(&b.metrics)
            .cmp(&cv_score_is_comparable(&a.metrics))
            .then_with(|| a.metrics.rmse.total_cmp(&b.metrics.rmse))
            .then_with(|| a.power.total_cmp(&b.power))
    });
    candidates
        .into_iter()
        .map(|candidate| {
            let in_sample = tweedie_candidates
                .iter()
                .find(|entry| (entry.power - candidate.power).abs() < 1e-12);
            let (aic, bic) = in_sample.map_or((f64::NAN, f64::NAN), |entry| {
                (
                    entry.information_criteria.aic,
                    entry.information_criteria.bic,
                )
            });
            TweedieRankingRow {
                power: candidate.power,
                metrics: candidate.metrics,
                aic,
                bic,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::two_part::Regularization;
    use crate::utils::usize_to_f64;
    use epistat_linalg::Mat;

    fn sample_input(n: usize) -> ModelInput {
        let design_matrix = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 25.0 },
        );
        let outcome = Mat::from_fn(n, 1, |i, _| {
            if i % 5 == 0 {
                0.0
            } else {
                0.15f64.mul_add(usize_to_f64(i), 1.2)
            }
        });
        ModelInput::new(design_matrix, outcome)
    }

    fn score(name: &str, rmse: f64, r2: f64) -> ModelScore {
        ModelScore {
            name: name.to_string(),
            metrics: ModelFitMetrics {
                rmse,
                mae: 0.0,
                rmsle: 0.0,
                r2,
                deviance: 0.0,
            },
        }
    }

    #[test]
    fn a_failed_cv_fit_ranks_below_every_real_one() {
        // A model that generalises badly keeps its place; one whose held-out
        // error is orders of magnitude past the outcome scale did not produce a
        // score and must not be ranked as though it merely came last.
        let ranked = build_cv_ranking(&[
            score("diverged", 2.7e11, -1.1e13),
            score("poor", 276_867.0, -3.71),
            score("mediocre", 149_339.0, -0.06),
            score("best", 144_962.0, 0.009),
        ]);

        let names: Vec<&str> = ranked.iter().map(|row| row.name.as_str()).collect();
        assert_eq!(names, ["best", "mediocre", "poor", "diverged"]);
    }

    #[test]
    fn cv_ties_break_on_name_not_construction_order() {
        // Equal RMSE must not leave the order to whatever built the list.
        let one = build_cv_ranking(&[score("b", 1.0, 0.5), score("a", 1.0, 0.5)]);
        let other = build_cv_ranking(&[score("a", 1.0, 0.5), score("b", 1.0, 0.5)]);
        assert_eq!(one[0].name, "a");
        assert_eq!(one[0].name, other[0].name);
    }

    #[test]
    fn compare_models_input_runs() {
        let input = sample_input(80);
        let report =
            compare_models_input(&input, &ModelComparisonOptions::default()).expect("comparison");
        assert!(!report.in_sample.is_empty());
        assert!(!report.information_criteria.is_empty());
        assert!(!report.cv_summary.is_empty());
    }

    #[test]
    fn compare_models_input_excludes_lognormal_when_disabled() {
        let input = sample_input(70);
        let options = ModelComparisonOptions {
            include_lognormal: false,
            ..ModelComparisonOptions::default()
        };
        let report = compare_models_input(&input, &options).expect("comparison");
        assert!(report.in_sample.iter().all(|row| row.name != "lognormal"));
        assert!(
            report
                .information_criteria
                .iter()
                .all(|row| row.name != "lognormal")
        );
        assert!(report.cv_summary.iter().all(|row| row.name != "lognormal"));
        assert!(report.cv_ranking.iter().all(|row| row.name != "lognormal"));
    }

    #[test]
    fn compare_models_input_excludes_two_part_when_disabled() {
        let input = sample_input(70);
        let options = ModelComparisonOptions {
            include_two_part: false,
            ..ModelComparisonOptions::default()
        };
        let report = compare_models_input(&input, &options).expect("comparison");
        assert!(report.in_sample.iter().all(|row| row.name != "two_part"));
        assert!(
            report
                .information_criteria
                .iter()
                .all(|row| row.name != "two_part")
        );
        assert!(report.cv_summary.iter().all(|row| row.name != "two_part"));
        assert!(report.cv_ranking.iter().all(|row| row.name != "two_part"));
    }

    #[test]
    fn compare_models_input_handles_empty_tweedie_power_grid() {
        let input = sample_input(60);
        let options = ModelComparisonOptions {
            include_tweedie: false,
            tweedie_powers: Vec::new(),
            ..ModelComparisonOptions::default()
        };
        let report = compare_models_input(&input, &options).expect("comparison");
        assert!(report.tweedie_ic.is_empty());
        assert!(report.tweedie_cv_ranking.is_empty());
        assert!(
            report
                .in_sample
                .iter()
                .all(|row| !row.name.starts_with("tweedie"))
        );
    }

    #[test]
    fn compare_models_input_adds_elastic_net_two_part_results() {
        let input = sample_input(80);
        let options = ModelComparisonOptions {
            two_part_elastic_net_options: Some(FitOptions {
                regularization: Regularization::ElasticNet {
                    lambda: 0.05,
                    alpha: 0.5,
                    exclude_intercept: true,
                },
                ..FitOptions::default()
            }),
            ..ModelComparisonOptions::default()
        };
        let report = compare_models_input(&input, &options).expect("comparison");
        assert!(
            report
                .in_sample
                .iter()
                .any(|row| row.name == "two_part_elastic_net")
        );
        assert!(
            report
                .information_criteria
                .iter()
                .any(|row| row.name == "two_part_elastic_net")
        );
        assert!(
            report
                .cv_summary
                .iter()
                .any(|row| row.name == "two_part_elastic_net")
        );
    }
}
