//! Propensity scores and Mahalanobis configurations estimated from covariate records.

mod config;
mod covariance;
mod cross_validation;
mod encoding;
mod linalg;
mod logistic;

pub use config::*;
pub use cross_validation::{CrossValidation, cross_validation_folds};
pub use encoding::CovariateMatrix;

use super::constraints::ConstraintGroup;
use super::distance::DistanceConfig;
use super::engine::{StandardMatchRequest, match_standard};
use super::records::{CovariateRecord, MatchingRecord};
use super::selection::SelectionStrategy;
use crate::types::{CommonSupport, DistanceCaliper, MatchOutcome, MatchRatio, MatchingCriteria};
use covariance::{covariance_matrix, rank_transform_columns, rescale_rank_covariance};
use cross_validation::fit_elastic_net_cv;
use encoding::{build_covariate_matrix, build_covariate_matrix_marked};
use itertools::Itertools;
use linalg::{invert_matrix_with_ridge, pseudo_inverse};
use logistic::{any_fitted_at_boundary, fit_elastic_net_logistic, fit_glm};
use rapidhash::RapidHashMap;

/// Whether a linear solve needed a ridge, and how large it was.
///
/// The inversion adds a ridge to the diagonal when a pivot of the matrix is at
/// most 1e-14, which happens when the covariates are collinear. The Mahalanobis
/// covariance also adds its configured starting ridge
/// ([`MahalanobisPreparationConfig::ridge`]) from the start, so `ridge` alone
/// does not show collinearity, and `singular_without_ridge` is the flag to read.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RidgeReport {
    /// The ridge added to the covariance diagonal before the inversion, the one
    /// used, counting the configured starting ridge. 0 when none was added.
    pub ridge: f64,
    /// Whether the covariance matrix, without any ridge, had a pivot at most
    /// 1e-14. `true` means the covariates are collinear or near collinear and
    /// the inverse depends on the ridge.
    pub singular_without_ridge: bool,
}

/// Propensity-score estimation output.
#[derive(Debug, Clone)]
pub struct PropensityScoreEstimate {
    /// Probability-scale propensity scores by id.
    pub probabilities: RapidHashMap<String, f64>,
    /// Linear predictor values by id.
    pub linear_predictor: RapidHashMap<String, f64>,
    /// Distance-scale scores (selected by [`PropensityScoreOutputScale`]).
    pub distance_scores: RapidHashMap<String, f64>,
    /// Fitted coefficients aligned with `column_names`. `None` for a column
    /// the generalised linear model could not identify, which R reports as
    /// `NA`; the names are in `aliased_columns`.
    pub coefficients: Vec<Option<f64>>,
    /// Design-matrix column names.
    pub column_names: Vec<String>,
    /// Estimator used for this run.
    pub estimator: &'static str,
    /// Whether the fit met its stopping rule within the iteration limit.
    pub converged: bool,
    /// Iterations executed.
    pub iterations: usize,
    /// The columns the generalised linear model dropped because they are a
    /// linear combination of the columns before them, as R's `glm()` does.
    /// Always empty for the elastic net.
    pub aliased_columns: Vec<String>,
    /// Whether a fitted probability lies within ten machine epsilons of 0 or
    /// 1, the condition R reports as "fitted probabilities numerically 0 or 1
    /// occurred". It marks complete or quasi-complete separation, where the
    /// likelihood has no finite maximum and the scores are not reliable.
    pub fitted_at_boundary: bool,
    /// The penalty of an elastic net fit, given or chosen by cross-validation.
    /// `None` for the generalised linear model.
    pub lambda: Option<f64>,
    /// The cross-validation curve and the folds, for the cross-validated
    /// elastic net.
    pub cross_validation: Option<CrossValidation>,
    /// Anchor ids discarded by common-support trimming before the refit, when
    /// the model was re-estimated. Their scores are those of the first fit.
    pub discarded_anchors: Vec<String>,
    /// Candidate ids discarded by common-support trimming before the refit.
    pub discarded_candidates: Vec<String>,
}

/// Combined propensity output and distance config.
#[derive(Debug, Clone)]
pub struct PropensityDistancePreparation {
    /// Prepared distance config for matching APIs.
    pub distance_config: DistanceConfig,
    /// Estimation details and diagnostics.
    pub estimate: PropensityScoreEstimate,
}

/// In-crate Mahalanobis preprocessing result.
#[derive(Debug, Clone)]
pub struct MahalanobisDistancePreparation {
    /// Prepared distance config for matching APIs.
    pub distance_config: DistanceConfig,
    /// Encoded vectors by id.
    pub vectors: RapidHashMap<String, Vec<f64>>,
    /// Inverse covariance in row-major format.
    pub inverse_covariance: Vec<f64>,
    /// Vector dimension.
    pub dimension: usize,
    /// Encoded vector column names.
    pub column_names: Vec<String>,
    /// The ridge added to the covariance matrix before inversion.
    pub ridge: RidgeReport,
    /// The rank of the inverse: `dimension` for an ordinary inverse, fewer
    /// when [`MahalanobisTransform::RankTieAdjusted`] gives a singular
    /// covariance and the generalised inverse drops directions.
    pub covariance_rank: usize,
}

/// Euclidean distance preprocessing result.
#[derive(Debug, Clone)]
pub struct EuclideanDistancePreparation {
    /// Prepared distance config for matching APIs.
    pub distance_config: DistanceConfig,
    /// Encoded, and if asked scaled, vectors by id.
    pub vectors: RapidHashMap<String, Vec<f64>>,
    /// Vector dimension.
    pub dimension: usize,
    /// Encoded vector column names.
    pub column_names: Vec<String>,
    /// The divisor of each column, all 1 without scaling.
    pub divisors: Vec<f64>,
}

/// Matched output combined with in-crate propensity estimation details.
#[derive(Debug, Clone)]
pub struct PropensityMatchedOutcome {
    /// Matching outcome.
    pub outcome: MatchOutcome,
    /// Propensity estimation output used to build matching distance.
    pub propensity: PropensityScoreEstimate,
}

/// Errors for in-crate preprocessing and estimation.
#[derive(Debug, Clone)]
pub enum EstimationError {
    EmptyInput,
    MissingValuesNotAllowed {
        covariate: String,
    },
    NonFiniteNumericCovariate {
        record_id: String,
        covariate: String,
    },
    MixedCovariateType {
        covariate: String,
    },
    NoObservedValuesForCovariate {
        covariate: String,
    },
    NoCovariatesAvailable,
    InvalidElasticNetConfig,
    /// The iteration limit is 0 or the tolerance is not positive and finite.
    InvalidGlmConfig,
    /// The cross-validation settings are out of range, a training set lacks one
    /// of the two groups, or the penalty path is undefined.
    InvalidCrossValidation,
    /// A sampling weight is missing for a record, negative or not finite.
    InvalidSamplingWeight {
        record_id: String,
    },
    /// `reestimate` was set but no common-support policy was given.
    ReestimateWithoutDiscard,
    /// The anchors' and the candidates' scores do not overlap, so common-support
    /// trimming leaves nothing to refit on.
    NoCommonSupport,
    /// The generalised linear model could not be fitted.
    GlmFitFailed(epistat_glm::GlmError),
    LinearSystemSolveFailed,
    InvalidCovarianceMatrix,
    MahalanobisConfigBuildFailed,
    /// A column varies, but not within either group, so it has no pooled
    /// within-group standard deviation to scale by.
    ZeroWithinGroupVariance {
        column: String,
    },
}

/// Build propensity-score distance config directly from covariate records.
///
/// With `config.reestimate` set, use
/// [`prepare_propensity_distance_config_with_discard`], which takes the
/// trimming policy.
///
/// # Errors
///
/// Returns [`EstimationError`] when preprocessing or model fitting fails, or
/// when `config.reestimate` is set, since no trimming policy is given here.
pub fn prepare_propensity_distance_config<R: CovariateRecord>(
    anchors: &[R],
    candidates: &[R],
    config: &PropensityScoreConfig,
) -> Result<PropensityDistancePreparation, EstimationError> {
    prepare_propensity_distance_config_with_discard(anchors, candidates, config, None)
}

/// The estimator's fit on a design, with prior weights.
fn fit_estimator(
    matrix: &CovariateMatrix,
    labels: &[f64],
    weights: Option<&[f64]>,
    config: &PropensityScoreConfig,
) -> Result<(logistic::LogisticFit, &'static str), EstimationError> {
    Ok(match &config.estimator {
        PropensityEstimator::GlmLogit => (
            fit_glm(matrix, labels, weights, &config.logistic)?,
            match config.logistic.link {
                PropensityLink::Logit => "glm_logit",
                PropensityLink::Probit => "glm_probit",
                PropensityLink::Cloglog => "glm_cloglog",
            },
        ),
        PropensityEstimator::ElasticNetLogit(elastic_net) => (
            fit_elastic_net_logistic(matrix, labels, weights, elastic_net)?,
            "elastic_net_logit",
        ),
        PropensityEstimator::ElasticNetCv(cv) => (
            fit_elastic_net_cv(matrix, labels, weights, cv)?,
            "elastic_net_cv",
        ),
    })
}

/// The rows of `matrix` at `rows`, in that order.
fn select_rows(matrix: &CovariateMatrix, rows: &[usize]) -> CovariateMatrix {
    let mut values = Vec::with_capacity(rows.len() * matrix.n_cols);
    for &row in rows {
        values.extend_from_slice(matrix.row(row));
    }
    CovariateMatrix {
        row_ids: rows
            .iter()
            .map(|&row| matrix.row_ids[row].clone())
            .collect(),
        column_names: matrix.column_names.clone(),
        values,
        n_rows: rows.len(),
        n_cols: matrix.n_cols,
    }
}

/// The rows that common-support trimming discards: those whose score is not
/// inside the interval where the two groups' score ranges overlap, with its
/// ends inside, for the groups the policy names. `n_anchors` rows come first.
/// `None` when the groups' ranges do not overlap.
fn rows_outside_common_support(
    scores: &[f64],
    n_anchors: usize,
    policy: CommonSupport,
) -> Option<Vec<usize>> {
    let bounds = |rows: std::ops::Range<usize>| -> Option<(f64, f64)> {
        rows.filter(|&row| scores[row].is_finite())
            .fold(None, |acc, row| match acc {
                None => Some((scores[row], scores[row])),
                Some((low, high)) => {
                    Some((f64::min(low, scores[row]), f64::max(high, scores[row])))
                }
            })
    };
    let anchor_bounds = bounds(0..n_anchors)?;
    let candidate_bounds = bounds(n_anchors..scores.len())?;
    let lower = anchor_bounds.0.max(candidate_bounds.0);
    let upper = anchor_bounds.1.min(candidate_bounds.1);
    if lower > upper {
        return None;
    }
    let outside =
        |row: usize| !(scores[row].is_finite() && scores[row] >= lower && scores[row] <= upper);
    let mut rows = Vec::new();
    if matches!(policy, CommonSupport::Treated | CommonSupport::Both) {
        rows.extend((0..n_anchors).filter(|&row| outside(row)));
    }
    if matches!(policy, CommonSupport::Control | CommonSupport::Both) {
        rows.extend((n_anchors..scores.len()).filter(|&row| outside(row)));
    }
    Some(rows)
}

/// Build propensity-score distance config from covariate records, with
/// common-support trimming for `config.reestimate`.
///
/// The model is fitted on all units. When `config.reestimate` is set, the units
/// that `discard` removes are found from those scores, on the scale of the
/// distance used, as MatchIt's `discard` does, and the model is fitted again on
/// the others, as `matchit(reestimate = TRUE)` does. The scores of the kept
/// units are those of the second fit. The discarded units keep the scores of the
/// first fit and are listed in the estimate. Without `config.reestimate`,
/// `discard` is not used.
///
/// # Errors
///
/// Returns [`EstimationError`] when preprocessing or model fitting fails, when
/// a sampling weight is missing or invalid, when `config.reestimate` is set
/// without a `discard` policy, or when the two groups' scores do not overlap.
pub fn prepare_propensity_distance_config_with_discard<R: CovariateRecord>(
    anchors: &[R],
    candidates: &[R],
    config: &PropensityScoreConfig,
    discard: Option<CommonSupport>,
) -> Result<PropensityDistancePreparation, EstimationError> {
    if anchors.is_empty() || candidates.is_empty() {
        return Err(EstimationError::EmptyInput);
    }
    if config.reestimate && discard.is_none() {
        return Err(EstimationError::ReestimateWithoutDiscard);
    }

    let records = anchors.iter().chain(candidates.iter()).collect_vec();
    let matrix = build_covariate_matrix(&records, &config.encoding)?;

    let labels = std::iter::repeat_n(1.0_f64, anchors.len())
        .chain(std::iter::repeat_n(0.0_f64, candidates.len()))
        .collect_vec();
    let weights = config
        .sampling_weights
        .as_ref()
        .map(|by_id| {
            matrix
                .row_ids
                .iter()
                .map(|id| match by_id.get(id) {
                    Some(weight) if weight.is_finite() && *weight >= 0.0 => Ok(*weight),
                    _ => Err(EstimationError::InvalidSamplingWeight {
                        record_id: id.clone(),
                    }),
                })
                .collect::<Result<Vec<_>, _>>()
        })
        .transpose()?;
    let (mut fit, estimator_name) = fit_estimator(&matrix, &labels, weights.as_deref(), config)?;

    let score_of = |fit: &logistic::LogisticFit| match config.output_scale {
        PropensityScoreOutputScale::Probability => fit.probabilities.clone(),
        PropensityScoreOutputScale::LinearPredictor => fit.linear_predictor.clone(),
    };
    let mut probabilities = fit.probabilities.clone();
    let mut linear_predictor = fit.linear_predictor.clone();
    let mut distance = score_of(&fit);
    let mut discarded_anchors = Vec::new();
    let mut discarded_candidates = Vec::new();

    if let (true, Some(policy)) = (config.reestimate, discard) {
        let outside = rows_outside_common_support(&distance, anchors.len(), policy)
            .ok_or(EstimationError::NoCommonSupport)?;
        let dropped: std::collections::HashSet<usize> = outside.iter().copied().collect();
        let kept = (0..matrix.n_rows)
            .filter(|row| !dropped.contains(row))
            .collect_vec();
        let kept_labels = kept.iter().map(|&row| labels[row]).collect_vec();
        let kept_weights = weights
            .as_ref()
            .map(|w| kept.iter().map(|&row| w[row]).collect_vec());
        let (refit, _) = fit_estimator(
            &select_rows(&matrix, &kept),
            &kept_labels,
            kept_weights.as_deref(),
            config,
        )?;
        let refit_distance = score_of(&refit);
        for (position, &row) in kept.iter().enumerate() {
            probabilities[row] = refit.probabilities[position];
            linear_predictor[row] = refit.linear_predictor[position];
            distance[row] = refit_distance[position];
        }
        for &row in &outside {
            let id = matrix.row_ids[row].clone();
            if row < anchors.len() {
                discarded_anchors.push(id);
            } else {
                discarded_candidates.push(id);
            }
        }
        fit = refit;
    }

    let fitted_at_boundary = any_fitted_at_boundary(&fit.probabilities);
    let to_map = |values: &[f64]| {
        matrix
            .row_ids
            .iter()
            .cloned()
            .zip(values.iter().copied())
            .collect::<RapidHashMap<_, _>>()
    };
    let distance_scores = to_map(&distance);
    let estimate = PropensityScoreEstimate {
        probabilities: to_map(&probabilities),
        linear_predictor: to_map(&linear_predictor),
        distance_scores: distance_scores.clone(),
        coefficients: fit.coefficients,
        column_names: matrix.column_names,
        estimator: estimator_name,
        converged: fit.converged,
        iterations: fit.iterations,
        aliased_columns: fit.aliased_columns,
        fitted_at_boundary,
        lambda: fit.lambda,
        cross_validation: fit.cross_validation,
        discarded_anchors,
        discarded_candidates,
    };
    let distance_config = DistanceConfig::propensity_score_map(distance_scores, config.caliper);

    Ok(PropensityDistancePreparation {
        distance_config,
        estimate,
    })
}

/// Prepare Mahalanobis vectors and inverse covariance directly from covariates.
///
/// # Errors
///
/// Returns [`EstimationError`] when preprocessing or covariance inversion fails.
pub fn prepare_mahalanobis_distance_config<R: CovariateRecord>(
    anchors: &[R],
    candidates: &[R],
    caliper: Option<DistanceCaliper>,
    config: &MahalanobisPreparationConfig,
) -> Result<MahalanobisDistancePreparation, EstimationError> {
    if anchors.is_empty() || candidates.is_empty() {
        return Err(EstimationError::EmptyInput);
    }

    let tie_adjusted = matches!(config.transform, MahalanobisTransform::RankTieAdjusted);
    let mut encoding = config.encoding.clone();
    encoding.include_intercept = false;
    if tie_adjusted {
        encoding.drop_first_categorical_level = false;
    }
    let records = anchors.iter().chain(candidates.iter()).collect_vec();
    let mut matrix = build_covariate_matrix(&records, &encoding)?;
    if matches!(
        config.transform,
        MahalanobisTransform::Rank | MahalanobisTransform::RankTieAdjusted
    ) {
        rank_transform_columns(&mut matrix);
    }
    let mut covariance = covariance_matrix(
        &matrix,
        anchors.len(),
        candidates.len(),
        config.covariance_strategy,
    )?;
    let (inverse_covariance, ridge, covariance_rank) = if tie_adjusted {
        rescale_rank_covariance(&mut covariance, matrix.n_cols, matrix.n_rows)?;
        let (inverse, rank) = pseudo_inverse(&covariance, GENERALISED_INVERSE_TOLERANCE)?;
        let report = RidgeReport {
            ridge: 0.0,
            singular_without_ridge: rank < matrix.n_cols,
        };
        (inverse, report, rank)
    } else {
        let (inverse, report) =
            invert_matrix_with_ridge(&covariance, config.ridge, config.max_ridge_attempts)?;
        (inverse, report, matrix.n_cols)
    };

    let vectors = matrix
        .row_ids
        .iter()
        .enumerate()
        .map(|(row_idx, id)| (id.clone(), matrix.row(row_idx).to_vec()))
        .collect::<RapidHashMap<_, _>>();
    let dimension = matrix.n_cols;
    let distance_config = DistanceConfig::mahalanobis_map(
        vectors.clone(),
        inverse_covariance.clone(),
        dimension,
        caliper,
    )
    .map_err(|_| EstimationError::MahalanobisConfigBuildFailed)?;

    Ok(MahalanobisDistancePreparation {
        distance_config,
        vectors,
        inverse_covariance,
        dimension,
        column_names: matrix.column_names,
        ridge,
        covariance_rank,
    })
}

/// Eigenvalues of the covariance of the ranks at most this fraction of the
/// largest count as zero when the generalised inverse is formed.
const GENERALISED_INVERSE_TOLERANCE: f64 = 1e-8;

/// Prepare Euclidean vectors directly from covariates.
///
/// Anchors are the first group and candidates the second, which matters only
/// for [`EuclideanScale::PooledWithinGroupSd`]. The covariates of units left
/// out of the matching, for instance trimmed to common support, should not be
/// passed: they would enter the standard deviations.
///
/// # Errors
///
/// Returns [`EstimationError`] when encoding fails, or when a scaled column
/// has no variance within either group.
pub fn prepare_euclidean_distance_config<R: CovariateRecord>(
    anchors: &[R],
    candidates: &[R],
    caliper: Option<DistanceCaliper>,
    config: &EuclideanPreparationConfig,
) -> Result<EuclideanDistancePreparation, EstimationError> {
    if anchors.is_empty() || candidates.is_empty() {
        return Err(EstimationError::EmptyInput);
    }

    let mut encoding = config.encoding.clone();
    encoding.include_intercept = false;
    encoding.drop_first_categorical_level = false;
    let records = anchors.iter().chain(candidates.iter()).collect_vec();
    let (mut matrix, indicator) = build_covariate_matrix_marked(&records, &encoding)?;
    for row_idx in 0..matrix.n_rows {
        for (col_idx, is_indicator) in indicator.iter().enumerate() {
            if *is_indicator {
                matrix.values[row_idx * matrix.n_cols + col_idx] *= std::f64::consts::FRAC_1_SQRT_2;
            }
        }
    }

    let mut divisors = vec![1.0_f64; matrix.n_cols];
    if config.scale == EuclideanScale::PooledWithinGroupSd {
        if matrix.n_rows <= 2 {
            return Err(EstimationError::EmptyInput);
        }
        for (col_idx, divisor) in divisors.iter_mut().enumerate() {
            let variance = covariance::pooled_within_group_variance(
                &matrix,
                col_idx,
                anchors.len(),
                candidates.len(),
            );
            let valid = variance.is_finite() && variance > 0.0;
            if !valid {
                return Err(EstimationError::ZeroWithinGroupVariance {
                    column: matrix.column_names[col_idx].clone(),
                });
            }
            *divisor = variance.sqrt();
        }
        for row_idx in 0..matrix.n_rows {
            for (col_idx, divisor) in divisors.iter().enumerate() {
                matrix.values[row_idx * matrix.n_cols + col_idx] /= divisor;
            }
        }
    }

    let vectors = matrix
        .row_ids
        .iter()
        .enumerate()
        .map(|(row_idx, id)| (id.clone(), matrix.row(row_idx).to_vec()))
        .collect::<RapidHashMap<_, _>>();
    let dimension = matrix.n_cols;
    let distance_config = DistanceConfig::euclidean_map(vectors.clone(), dimension, caliper)
        .map_err(|_| EstimationError::MahalanobisConfigBuildFailed)?;

    Ok(EuclideanDistancePreparation {
        distance_config,
        vectors,
        dimension,
        column_names: matrix.column_names,
        divisors,
    })
}

/// Run matching with propensity scores estimated internally from covariates.
///
/// With `propensity_config.reestimate`, the model is refitted on the units that
/// `criteria.common_support` keeps, and the matching then trims to common
/// support again on the refitted scores, which can remove further units. The
/// scores themselves are MatchIt's.
///
/// # Errors
///
/// Returns [`EstimationError`] when preprocessing or model fitting fails.
pub fn estimate_propensity_and_match<
    R: MatchingRecord + CovariateRecord,
    S: SelectionStrategy<R> + Clone + Send + Sync,
    G: ConstraintGroup<R> + ?Sized,
>(
    anchors: &[R],
    candidates: &[R],
    criteria: &MatchingCriteria,
    strategy: S,
    ratio_fallback: &[MatchRatio],
    extra_constraints: &G,
    propensity_config: &PropensityScoreConfig,
) -> Result<PropensityMatchedOutcome, EstimationError> {
    let prepared = prepare_propensity_distance_config_with_discard(
        anchors,
        candidates,
        propensity_config,
        criteria.common_support,
    )?;
    let outcome = match_standard(
        anchors,
        candidates,
        StandardMatchRequest {
            criteria,
            strategy,
            constraints: extra_constraints,
            ratio_fallback,
            distance_config: Some(&prepared.distance_config),
        },
    );

    Ok(PropensityMatchedOutcome {
        outcome,
        propensity: prepared.estimate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::date;
    use crate::matching::DeterministicSelection;
    use crate::types::{BalanceRecord, CovariateValue, MatchDiagnostics, MatchedPair};
    use chrono::NaiveDate;

    fn record(id: &str, birth_date: NaiveDate, age: f64, region: &str) -> BalanceRecord {
        let mut row = BalanceRecord::new(id, birth_date);
        row.covariates
            .insert("age".to_string(), CovariateValue::Numeric(age));
        row.covariates.insert(
            "region".to_string(),
            CovariateValue::Categorical(region.to_string()),
        );
        row
    }

    #[test]
    fn propensity_distance_preparation_returns_finite_scores() {
        let anchors = vec![
            record("a1", date(2010, 1, 1), 10.0, "north"),
            record("a2", date(2010, 1, 2), 11.0, "north"),
        ];
        let candidates = vec![
            record("c1", date(2010, 1, 3), 1.0, "south"),
            record("c2", date(2010, 1, 4), 2.0, "south"),
        ];

        let prepared = prepare_propensity_distance_config(
            &anchors,
            &candidates,
            &PropensityScoreConfig::default(),
        )
        .expect("propensity estimation should succeed");
        assert_eq!(prepared.estimate.estimator, "glm_logit");
        assert_eq!(prepared.estimate.distance_scores.len(), 4);
        assert!(
            prepared
                .estimate
                .distance_scores
                .values()
                .all(|value| value.is_finite())
        );

        let anchor_mean = f64::midpoint(
            prepared.estimate.probabilities["a1"],
            prepared.estimate.probabilities["a2"],
        );
        let candidate_mean = f64::midpoint(
            prepared.estimate.probabilities["c1"],
            prepared.estimate.probabilities["c2"],
        );
        assert!(anchor_mean > candidate_mean);
    }

    #[test]
    fn elastic_net_propensity_estimator_runs_and_shrinks_coefficients() {
        let anchors = vec![
            record("a1", date(2010, 1, 1), 10.0, "north"),
            record("a2", date(2010, 1, 2), 11.0, "north"),
            record("a3", date(2010, 1, 3), 12.0, "north"),
        ];
        let candidates = vec![
            record("c1", date(2010, 1, 4), 1.0, "south"),
            record("c2", date(2010, 1, 5), 2.0, "south"),
            record("c3", date(2010, 1, 6), 3.0, "south"),
        ];

        let glm_prepared = prepare_propensity_distance_config(
            &anchors,
            &candidates,
            &PropensityScoreConfig::default(),
        )
        .expect("glm propensity estimation should succeed");
        let elastic_net_prepared = prepare_propensity_distance_config(
            &anchors,
            &candidates,
            &PropensityScoreConfig::builder()
                .estimator(PropensityEstimator::ElasticNetLogit(
                    ElasticNetLogisticConfig::builder()
                        .lambda(0.5)
                        .alpha(1.0)
                        .build(),
                ))
                .build(),
        )
        .expect("elastic-net propensity estimation should succeed");

        assert_eq!(elastic_net_prepared.estimate.estimator, "elastic_net_logit");
        let glm_abs_sum = glm_prepared
            .estimate
            .coefficients
            .iter()
            .enumerate()
            .filter(|(idx, _)| glm_prepared.estimate.column_names[*idx] != "intercept")
            .map(|(_, value)| value.expect("glm identifies every column").abs())
            .sum::<f64>();
        let elastic_abs_sum = elastic_net_prepared
            .estimate
            .coefficients
            .iter()
            .enumerate()
            .filter(|(idx, _)| elastic_net_prepared.estimate.column_names[*idx] != "intercept")
            .map(|(_, value)| value.expect("the elastic net fits every column").abs())
            .sum::<f64>();
        assert!(elastic_abs_sum <= glm_abs_sum + 1e-8);
    }

    #[test]
    fn elastic_net_respects_linear_predictor_output_scale() {
        let anchors = vec![
            record("a1", date(2010, 1, 1), 10.0, "north"),
            record("a2", date(2010, 1, 2), 11.0, "north"),
        ];
        let candidates = vec![
            record("c1", date(2010, 1, 3), 1.0, "south"),
            record("c2", date(2010, 1, 4), 2.0, "south"),
        ];

        let prepared = prepare_propensity_distance_config(
            &anchors,
            &candidates,
            &PropensityScoreConfig::builder()
                .estimator(PropensityEstimator::ElasticNetLogit(
                    ElasticNetLogisticConfig::default(),
                ))
                .output_scale(PropensityScoreOutputScale::LinearPredictor)
                .build(),
        )
        .expect("elastic-net propensity estimation should succeed");
        assert_eq!(prepared.estimate.estimator, "elastic_net_logit");
        for (id, score) in &prepared.estimate.distance_scores {
            assert!(
                (*score - prepared.estimate.linear_predictor[id]).abs() < 1e-12,
                "distance score should equal linear predictor for {id}"
            );
        }
    }

    #[test]
    fn elastic_net_rejects_invalid_config() {
        let anchors = vec![record("a1", date(2010, 1, 1), 10.0, "north")];
        let candidates = vec![record("c1", date(2010, 1, 2), 1.0, "south")];

        let err = prepare_propensity_distance_config(
            &anchors,
            &candidates,
            &PropensityScoreConfig::builder()
                .estimator(PropensityEstimator::ElasticNetLogit(
                    ElasticNetLogisticConfig::builder()
                        .probability_clip(0.75)
                        .build(),
                ))
                .build(),
        )
        .expect_err("invalid elastic-net configuration should fail");
        assert!(matches!(err, EstimationError::InvalidElasticNetConfig));
    }

    #[test]
    fn estimated_propensity_matching_wrapper_runs() {
        let anchors = vec![record("a1", date(2010, 1, 1), 10.0, "north")];
        let candidates = vec![
            record("c1", date(2010, 1, 1), 9.5, "north"),
            record("c2", date(2010, 1, 2), 1.5, "south"),
        ];
        let criteria = MatchingCriteria::default();
        let matched = estimate_propensity_and_match(
            &anchors,
            &candidates,
            &criteria,
            DeterministicSelection,
            &[],
            &(),
            &PropensityScoreConfig::builder()
                .caliper(DistanceCaliper::new(10.0).expect("valid positive caliper"))
                .build(),
        )
        .expect("matching wrapper should succeed");

        assert_eq!(matched.propensity.distance_scores.len(), 3);
        assert_eq!(matched.outcome.matched_cases, 1);
        assert_eq!(matched.outcome.pairs.len(), 1);
    }

    #[test]
    fn mahalanobis_preparation_produces_config() {
        let anchors = vec![
            record("a1", date(2010, 1, 1), 10.0, "north"),
            record("a2", date(2010, 1, 2), 11.0, "north"),
        ];
        let candidates = vec![
            record("c1", date(2010, 1, 3), 1.0, "south"),
            record("c2", date(2010, 1, 4), 2.0, "south"),
        ];

        let prepared = prepare_mahalanobis_distance_config(
            &anchors,
            &candidates,
            Some(DistanceCaliper::new(5.0).expect("valid positive caliper")),
            &MahalanobisPreparationConfig::default(),
        )
        .expect("mahalanobis preparation should succeed");
        assert_eq!(prepared.dimension, prepared.column_names.len());
        assert_eq!(prepared.vectors.len(), 4);
        assert_eq!(
            prepared.inverse_covariance.len(),
            prepared.dimension * prepared.dimension
        );
    }

    #[test]
    fn mahalanobis_preparation_supports_rank_transform_and_full_sample_covariance() {
        let anchors = vec![
            record("a1", date(2010, 1, 1), 10.0, "north"),
            record("a2", date(2010, 1, 2), 10.0, "south"),
        ];
        let candidates = vec![
            record("c1", date(2010, 1, 3), 1.0, "south"),
            record("c2", date(2010, 1, 4), 2.0, "north"),
        ];

        let prepared = prepare_mahalanobis_distance_config(
            &anchors,
            &candidates,
            Some(DistanceCaliper::new(5.0).expect("valid positive caliper")),
            &MahalanobisPreparationConfig::builder()
                .covariance_strategy(MahalanobisCovarianceStrategy::FullSample)
                .transform(MahalanobisTransform::Rank)
                .build(),
        )
        .expect("rank + full-sample config should succeed");
        assert_eq!(prepared.vectors.len(), 4);
        assert!(
            prepared
                .inverse_covariance
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn euclidean_config_restricts_candidates_as_a_caliper() {
        let anchors = [record("a1", date(2010, 1, 1), 10.0, "x")];
        let candidates = [
            record("c1", date(2010, 1, 1), 11.0, "x"),
            record("c2", date(2010, 1, 1), 14.0, "x"),
            record("c3", date(2010, 1, 1), 40.0, "x"),
        ];
        let matched = |caliper: f64| {
            let prepared = prepare_euclidean_distance_config(
                &anchors,
                &candidates,
                Some(DistanceCaliper::new(caliper).expect("valid positive caliper")),
                &EuclideanPreparationConfig::default(),
            )
            .expect("euclidean preparation should succeed");
            // The region column is constant, so the distance is the age gap.
            assert_eq!(prepared.column_names, ["age"]);
            let outcome = crate::MatchJob::new_standard(&anchors, &candidates, 1)
                .with_ratio(MatchRatio::new(3).expect("ratio"))
                .with_distance_config(prepared.distance_config)
                .run();
            let mut ids: Vec<_> = outcome.pairs.into_iter().map(|p| p.control_id).collect();
            ids.sort();
            ids
        };
        assert_eq!(matched(5.0), ["c1", "c2"]);
        assert_eq!(matched(2.0), ["c1"]);
    }

    #[test]
    fn pooled_covariance_by_degrees_of_freedom_weights_groups_by_size() {
        let matrix = CovariateMatrix {
            row_ids: ["a1", "a2", "c1", "c2", "c3", "c4"]
                .map(str::to_string)
                .to_vec(),
            column_names: vec!["x".to_string()],
            values: vec![0.0, 2.0, 10.0, 12.0, 14.0, 20.0],
            n_rows: 6,
            n_cols: 1,
        };
        // Group means 1 and 14, sums of squares 2 and 56, over 6 - 2 units.
        let pooled = covariance_matrix(
            &matrix,
            2,
            4,
            MahalanobisCovarianceStrategy::PooledByDegreesOfFreedom,
        )
        .expect("pooled covariance should compute");
        assert!((pooled[0] - 14.5).abs() < 1e-12);
        // The mean of the two group variances, 2 and 56 / 3, differs.
        let mean_of_groups = covariance_matrix(
            &matrix,
            2,
            4,
            MahalanobisCovarianceStrategy::PooledWithinGroups,
        )
        .expect("pooled covariance should compute");
        assert!((mean_of_groups[0] - f64::midpoint(2.0, 56.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn covariance_strategy_changes_covariance_matrix() {
        let matrix = CovariateMatrix {
            row_ids: vec![
                "a1".to_string(),
                "a2".to_string(),
                "c1".to_string(),
                "c2".to_string(),
            ],
            column_names: vec!["x".to_string()],
            values: vec![0.0, 2.0, 10.0, 12.0],
            n_rows: 4,
            n_cols: 1,
        };

        let pooled = covariance_matrix(
            &matrix,
            2,
            2,
            MahalanobisCovarianceStrategy::PooledWithinGroups,
        )
        .expect("pooled covariance should compute");
        let full = covariance_matrix(&matrix, 2, 2, MahalanobisCovarianceStrategy::FullSample)
            .expect("full covariance should compute");

        assert!((pooled[0] - full[0]).abs() > 1e-12);
        assert!((pooled[0] - 2.0).abs() < 1e-12);
        assert!((full[0] - 34.666_666_666_666_664).abs() < 1e-12);
    }

    #[test]
    fn rank_transform_columns_uses_average_ranks_for_ties() {
        let mut matrix = CovariateMatrix {
            row_ids: vec!["r1".to_string(), "r2".to_string(), "r3".to_string()],
            column_names: vec!["x".to_string(), "y".to_string()],
            values: vec![
                5.0, 10.0, //
                5.0, 30.0, //
                9.0, 20.0, //
            ],
            n_rows: 3,
            n_cols: 2,
        };

        rank_transform_columns(&mut matrix);

        assert!((matrix.values[0] - 1.5).abs() < 1e-12);
        assert!((matrix.values[2] - 1.5).abs() < 1e-12);
        assert!((matrix.values[4] - 3.0).abs() < 1e-12);
        assert!((matrix.values[1] - 1.0).abs() < 1e-12);
        assert!((matrix.values[3] - 3.0).abs() < 1e-12);
        assert!((matrix.values[5] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn missing_value_policy_error_rejects_missing_covariate() {
        let mut anchor = BalanceRecord::new("a1", date(2010, 1, 1));
        anchor
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(5.0));
        let candidate = BalanceRecord::new("c1", date(2010, 1, 1));

        let config = PropensityScoreConfig::builder()
            .encoding(
                CovariateEncodingConfig::builder()
                    .covariate_keys(vec!["age".to_string()])
                    .include_intercept(true)
                    .build(),
            )
            .build();

        let err = prepare_propensity_distance_config(&[anchor], &[candidate], &config)
            .expect_err("missing values should fail");
        assert!(matches!(
            err,
            EstimationError::MissingValuesNotAllowed { .. }
        ));
    }

    #[test]
    fn impute_policy_handles_missing_values() {
        let mut anchor = BalanceRecord::new("a1", date(2010, 1, 1));
        anchor
            .covariates
            .insert("age".to_string(), CovariateValue::Numeric(5.0));
        let candidate = BalanceRecord::new("c1", date(2010, 1, 1));

        let config = PropensityScoreConfig::builder()
            .encoding(
                CovariateEncodingConfig::builder()
                    .covariate_keys(vec!["age".to_string()])
                    .include_intercept(true)
                    .missing_value_policy(MissingValuePolicy::Impute)
                    .build(),
            )
            .build();

        let prepared = prepare_propensity_distance_config(&[anchor], &[candidate], &config)
            .expect("impute mode should succeed");
        assert_eq!(prepared.estimate.distance_scores.len(), 2);
    }

    fn record_with_duplicate_age(id: &str, day: u32, age: f64) -> BalanceRecord {
        let mut row = BalanceRecord::new(id, date(2010, 1, day));
        row.covariates
            .insert("age".to_string(), CovariateValue::Numeric(age));
        row.covariates
            .insert("age_twice".to_string(), CovariateValue::Numeric(2.0 * age));
        row
    }

    fn collinear_groups() -> (Vec<BalanceRecord>, Vec<BalanceRecord>) {
        let ages_anchor = [3.0, 5.0, 4.0, 9.0, 6.0];
        let ages_candidate = [1.0, 2.0, 8.0, 3.5, 7.0];
        let anchors = ages_anchor
            .iter()
            .enumerate()
            .map(|(idx, age)| {
                record_with_duplicate_age(&format!("a{idx}"), 1 + u32::try_from(idx).unwrap(), *age)
            })
            .collect();
        let candidates = ages_candidate
            .iter()
            .enumerate()
            .map(|(idx, age)| {
                record_with_duplicate_age(
                    &format!("c{idx}"),
                    10 + u32::try_from(idx).unwrap(),
                    *age,
                )
            })
            .collect();
        (anchors, candidates)
    }

    #[test]
    fn collinear_covariates_are_reported_by_the_propensity_fit() {
        let (anchors, candidates) = collinear_groups();
        let prepared = prepare_propensity_distance_config(
            &anchors,
            &candidates,
            &PropensityScoreConfig::default(),
        )
        .expect("an aliased column is dropped, as in R");
        assert_eq!(prepared.estimate.aliased_columns, ["age_twice"]);
        let position = prepared
            .estimate
            .column_names
            .iter()
            .position(|name| name == "age_twice")
            .expect("column exists");
        assert_eq!(prepared.estimate.coefficients[position], None);
        assert!(
            prepared
                .estimate
                .distance_scores
                .values()
                .all(|score| score.is_finite())
        );
    }

    #[test]
    fn independent_covariates_report_no_aliased_column_in_the_propensity_fit() {
        let anchors = vec![
            record("a1", date(2010, 1, 1), 10.0, "north"),
            record("a2", date(2010, 1, 2), 12.0, "south"),
            record("a3", date(2010, 1, 3), 7.0, "north"),
        ];
        let candidates = vec![
            record("c1", date(2010, 1, 3), 1.0, "south"),
            record("c2", date(2010, 1, 4), 9.0, "north"),
            record("c3", date(2010, 1, 5), 5.0, "south"),
        ];
        let prepared = prepare_propensity_distance_config(
            &anchors,
            &candidates,
            &PropensityScoreConfig::default(),
        )
        .expect("propensity estimation should succeed");
        assert_eq!(prepared.estimate.aliased_columns, Vec::<String>::new());
        assert!(prepared.estimate.coefficients.iter().all(Option::is_some));
    }

    #[test]
    fn separated_data_is_flagged_and_a_short_fit_reports_non_convergence() {
        let anchors = (0..6)
            .map(|i| {
                record(
                    &format!("a{i}"),
                    date(2010, 1, 1),
                    10.0 + f64::from(i),
                    "north",
                )
            })
            .collect::<Vec<_>>();
        let candidates = (0..6)
            .map(|i| record(&format!("c{i}"), date(2010, 1, 2), f64::from(i), "north"))
            .collect::<Vec<_>>();
        let prepared = prepare_propensity_distance_config(
            &anchors,
            &candidates,
            &PropensityScoreConfig::default(),
        )
        .expect("a separated fit is returned and flagged");
        assert!(prepared.estimate.fitted_at_boundary);

        let (anchors, candidates) = collinear_groups();
        let short = PropensityScoreConfig::builder()
            .logistic(LogisticRegressionConfig::builder().max_iter(1).build())
            .build();
        let prepared = prepare_propensity_distance_config(&anchors, &candidates, &short)
            .expect("a short fit is returned");
        assert!(!prepared.estimate.converged);
        assert!(!prepared.estimate.fitted_at_boundary);
    }

    #[test]
    fn invalid_glm_settings_and_reestimate_without_a_policy_are_errors() {
        let (anchors, candidates) = collinear_groups();
        for logistic in [
            LogisticRegressionConfig::builder().max_iter(0).build(),
            LogisticRegressionConfig::builder().tolerance(0.0).build(),
        ] {
            let config = PropensityScoreConfig::builder().logistic(logistic).build();
            let err = prepare_propensity_distance_config(&anchors, &candidates, &config)
                .expect_err("invalid settings are refused");
            assert!(matches!(err, EstimationError::InvalidGlmConfig));
        }
        let config = PropensityScoreConfig::builder().reestimate(true).build();
        let err = prepare_propensity_distance_config(&anchors, &candidates, &config)
            .expect_err("reestimate needs a policy");
        assert!(matches!(err, EstimationError::ReestimateWithoutDiscard));
    }

    #[test]
    fn collinear_covariates_are_reported_by_the_mahalanobis_inverse() {
        let (anchors, candidates) = collinear_groups();
        let prepared = prepare_mahalanobis_distance_config(
            &anchors,
            &candidates,
            None,
            &MahalanobisPreparationConfig::default(),
        )
        .expect("a ridged inverse should succeed");
        assert!(prepared.ridge.singular_without_ridge);
        assert!(prepared.ridge.ridge >= 1e-8);
    }

    #[test]
    fn independent_covariates_report_only_the_configured_ridge_in_the_mahalanobis_inverse() {
        let anchors = vec![
            record("a1", date(2010, 1, 1), 10.0, "north"),
            record("a2", date(2010, 1, 2), 12.0, "south"),
            record("a3", date(2010, 1, 3), 7.0, "north"),
        ];
        let candidates = vec![
            record("c1", date(2010, 1, 3), 1.0, "south"),
            record("c2", date(2010, 1, 4), 9.0, "north"),
            record("c3", date(2010, 1, 5), 5.0, "south"),
        ];
        let prepared = prepare_mahalanobis_distance_config(
            &anchors,
            &candidates,
            None,
            &MahalanobisPreparationConfig::default(),
        )
        .expect("mahalanobis preparation should succeed");
        assert!(!prepared.ridge.singular_without_ridge);
        assert!((prepared.ridge.ridge - 1e-8).abs() < 1e-20);
    }

    #[test]
    fn elastic_net_fit_reports_no_aliased_column() {
        let (anchors, candidates) = collinear_groups();
        let config = PropensityScoreConfig::builder()
            .estimator(PropensityEstimator::ElasticNetLogit(
                ElasticNetLogisticConfig::default(),
            ))
            .build();
        let prepared = prepare_propensity_distance_config(&anchors, &candidates, &config)
            .expect("elastic net fit should succeed");
        assert_eq!(prepared.estimate.aliased_columns, Vec::<String>::new());
    }

    #[test]
    fn helper_type_is_constructible_for_external_use() {
        let _unused = PropensityMatchedOutcome {
            outcome: MatchOutcome {
                pairs: vec![MatchedPair::new("a", "c")],
                unmatched_cases: 0,
                used_controls: 1,
                matched_cases: 1,
                avg_controls_per_case: 1.0,
                diagnostics: MatchDiagnostics::default(),
            },
            propensity: PropensityScoreEstimate {
                probabilities: RapidHashMap::default(),
                linear_predictor: RapidHashMap::default(),
                distance_scores: RapidHashMap::default(),
                coefficients: Vec::new(),
                column_names: Vec::new(),
                estimator: "glm_logit",
                converged: true,
                iterations: 0,
                aliased_columns: Vec::new(),
                fitted_at_boundary: false,
                lambda: None,
                cross_validation: None,
                discarded_anchors: Vec::new(),
                discarded_candidates: Vec::new(),
            },
        };
    }
}
