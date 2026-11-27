//! Difference-in-differences with compositional changes in repeated
//! cross-sections, after Sant'Anna and Xu \[1\].
//!
//! [`estimate`] takes a repeated cross-section [`TwoPeriodData`] and
//! [`DidCcOptions`] and returns a [`DidCc`]; [`test()`] compares the two
//! estimators with a Hausman test.
//!
//! # Algorithm
//!
//! This is a working-model version of the estimator for the canonical
//! two-period design. The composition-robust estimator
//! ([`Assumption::Robust`]) combines four outcome regressions, one per arm and
//! period, fitted by local linear regression with the ridge of
//! [`NuisanceOptions::ridge`], with a generalised propensity score over the
//! four (arm, period) cells, a multinomial logit of the cell on the design.
//! The estimator under stationarity ([`Assumption::Stationary`]) uses the
//! Sant'Anna-Zhao weighting with a pooled propensity score of the arm. When
//! the multinomial or the pooled logit cannot be fitted on the covariates,
//! it is fitted on the intercept alone, and failing that the cell shares are
//! used. With [`DidCcOptions::cross_fitting`] each row's nuisance values come
//! from the models fitted on the other folds, the folds drawn within each
//! cell.
//!
//! The influence function is the centred score, the standard error
//! `sqrt(mean(psi^2) / n)`, and the interval normal or, with
//! [`DidCcOptions::bootstrap`], the multiplier bootstrap percentile interval.
//! The Hausman statistic is `(theta_s - theta_r)^2 / se^2`, with `se` from
//! the difference of the two influence functions, against a chi-square
//! distribution with one degree of freedom.
//!
//! # References
//!
//! 1. Sant'Anna PHC, Xu Q. Difference-in-differences with compositional
//!    changes. Working paper, 2023.

use epistat_linalg::{Mat, MatRef};
use epistat_rng::Stream;
use epistat_stats::dist::chi_square::sf as chi_square_sf;

use crate::estimators::outcome::nonparametric::{LocalLinearOutcome, LocalLinearScratch};
use crate::estimators::propensity::logistic::LogisticPS;
use crate::estimators::propensity::multinomial::MultinomialSoftmaxPS;
use crate::estimators::propensity::types::{
    Config as PropensityConfig, MultinomialPropensityEstimator, PropensityEstimator,
};
use crate::inference::{
    multiplier_bootstrap_ci, normal_interval, standard_error_from_influence,
    validate_confidence_level,
};
use crate::methods::drdid::moments::normalize_weights_to_n;
use crate::types::{
    DidCcConfig, DidCcError, DidCcEstimate, DidCcHausmanTest, DidCell, DrDidRepeatedObservation,
    TimePeriod, TreatmentGroup,
};

use epistat_core::{Bootstrap, Estimate, InputError, Level};
use epistat_inference::Influence;

use crate::error::Error;
use crate::methods::two_period::{
    Design, NuisanceOptions, TwoPeriodData, domain, repeated_rows, row_influence,
};

/// The identifying assumption of the estimate.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum Assumption {
    /// Robust to changes in the composition of the sample between the
    /// periods.
    #[default]
    Robust,
    /// The covariates' distribution is stationary between the periods.
    Stationary,
}

/// The folds of cross-fitting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrossFitting {
    /// The number of folds, at least 2.
    pub folds: usize,
    /// The seed of the assignment of rows to folds, drawn within each cell.
    pub seed: u64,
}

/// The options of [`estimate`] and [`test()`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DidCcOptions {
    /// The identifying assumption of [`estimate`]; [`test()`] forms both.
    pub assumption: Assumption,
    /// The nuisance models: the clip of the propensity scores, the ridge of
    /// the outcome regressions and the iterations of the logits. The legacy
    /// switch is not read.
    pub nuisance: NuisanceOptions,
    /// The confidence level of the intervals.
    pub level: Level,
    /// A multiplier bootstrap for the intervals: the percentile interval of
    /// Rademacher draws of the influence function. `None`, the default, gives
    /// normal intervals.
    pub bootstrap: Option<Bootstrap>,
    /// Cross-fitting of the nuisance models; `None`, the default, fits them
    /// on every row.
    pub cross_fitting: Option<CrossFitting>,
}

/// An ATT under compositional changes.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DidCc {
    /// The assumption it was formed under.
    pub assumption: Assumption,
    /// The ATT with its standard error, interval and normal p-value.
    pub estimate: Estimate,
    /// Treated rows in the post period.
    pub n_treated_post: usize,
    /// Treated rows in the pre period.
    pub n_treated_pre: usize,
    /// Comparison rows in the post period.
    pub n_control_post: usize,
    /// Comparison rows in the pre period.
    pub n_control_pre: usize,
    /// The sum of the prior weights.
    pub total_weight: f64,
    /// The influence function, one value per row, under
    /// [`epistat_inference::ClusterRule::NoSmallSample`].
    pub influence: Influence,
}

/// The Hausman comparison of the robust and the stationary estimates.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DidCcTest {
    /// The composition-robust estimate.
    pub robust: DidCc,
    /// The estimate under stationarity.
    pub stationary: DidCc,
    /// The stationary estimate minus the robust one.
    pub difference: f64,
    /// The standard error of the difference, from the difference of the two
    /// influence functions.
    pub se: f64,
    /// `(difference / se)^2`.
    pub statistic: f64,
    /// The p-value of the statistic against a chi-square distribution with
    /// one degree of freedom; a small value rejects stationarity.
    pub p_value: f64,
}

/// The ATT of the repeated cross-section `data` under `options.assumption`.
///
/// # Errors
///
/// [`Error::Input`] for a nuisance option outside its domain, a bootstrap
/// without replicates or fewer than two folds; [`Error::Unsupported`] for a
/// panel or clusters; [`Error::NoTreated`] or [`Error::NoComparison`] for an
/// arm without rows in a period; [`Error::CrossFitCell`] for a cell with
/// fewer rows than folds; and [`Error::Singular`] for an outcome regression
/// that cannot be solved.
pub fn estimate(data: &TwoPeriodData<'_>, options: &DidCcOptions) -> Result<DidCc, Error> {
    let (rows, config) = prepare(data, options)?;
    let fit = match options.assumption {
        Assumption::Robust => robust(&rows, config),
        Assumption::Stationary => stationary(&rows, config),
    }
    .map_err(Error::from)?;
    result(options.assumption, fit, options.level)
}

/// The Hausman test of stationarity: both estimates of [`estimate`] and the
/// test of their difference. `options.assumption` is not read.
///
/// # Errors
///
/// As [`estimate`], and [`Error::DegenerateVariance`] when the difference
/// has no positive variance.
pub fn test(data: &TwoPeriodData<'_>, options: &DidCcOptions) -> Result<DidCcTest, Error> {
    let (rows, config) = prepare(data, options)?;
    let fit = hausman(&rows, config).map_err(Error::from)?;
    Ok(DidCcTest {
        robust: result(Assumption::Robust, fit.robust, options.level)?,
        stationary: result(Assumption::Stationary, fit.stationary, options.level)?,
        difference: fit.difference,
        se: fit.difference_se,
        statistic: fit.statistic,
        p_value: fit.p_value,
    })
}

fn prepare(
    data: &TwoPeriodData<'_>,
    options: &DidCcOptions,
) -> Result<(Vec<DrDidRepeatedObservation>, DidCcConfig), Error> {
    options.nuisance.validate()?;
    let Design::RepeatedCrossSection { post, outcome } = data.design() else {
        return Err(Error::Unsupported {
            what: "the compositional-change estimator on a panel",
        });
    };
    if data.clusters().is_some() {
        return Err(Error::Unsupported {
            what: "clusters in the compositional-change estimator",
        });
    }
    if let Some(bootstrap) = options.bootstrap
        && bootstrap.replicates == 0
    {
        return Err(domain("bootstrap replicates", 0.0, "[1, inf)"));
    }
    if let Some(cross) = options.cross_fitting
        && cross.folds < 2
    {
        return Err(InputError::Domain {
            what: "cross-fitting folds",
            value: crate::util::usize_to_f64(cross.folds),
            domain: "[2, inf)",
        }
        .into());
    }
    let config = DidCcConfig {
        drdid: options.nuisance.config(options.level, options.bootstrap),
        bootstrap_confidence_intervals: options.bootstrap.is_some(),
        cross_fit_folds: options.cross_fitting.map_or(1, |cross| cross.folds),
        cross_fit_seed: options.cross_fitting.map_or(0, |cross| cross.seed),
    };
    Ok((repeated_rows(data, post, outcome), config))
}

fn result(assumption: Assumption, fit: DidCcEstimate, level: Level) -> Result<DidCc, Error> {
    Ok(DidCc {
        assumption,
        estimate: crate::inference::estimate_from(
            fit.att,
            fit.se,
            (fit.ci_low, fit.ci_high),
            level,
        ),
        n_treated_post: fit.treated_post_n,
        n_treated_pre: fit.treated_pre_n,
        n_control_post: fit.control_post_n,
        n_control_pre: fit.control_pre_n,
        total_weight: fit.total_weight,
        influence: row_influence(fit.influence_function)?,
    })
}

impl From<DidCcError> for Error {
    fn from(error: DidCcError) -> Self {
        match error {
            DidCcError::EmptyInput => InputError::Empty { what: "rows" }.into(),
            DidCcError::NoTreated => Self::NoTreated,
            DidCcError::NoControl => Self::NoComparison,
            DidCcError::MissingCell { cell } => {
                let (treated, post) = cell.flags();
                Self::EmptyCell { treated, post }
            }
            DidCcError::InvalidWeight { value } => domain("weights", value, "(0, inf)"),
            DidCcError::InvalidOutcome { .. } => InputError::NotFinite {
                what: "outcome",
                row: 0,
            }
            .into(),
            DidCcError::InvalidCovariate { .. } => InputError::NotFinite {
                what: "covariates",
                row: 0,
            }
            .into(),
            DidCcError::InconsistentCovariateCount { expected, actual } => InputError::Length {
                what: "covariates",
                expected,
                found: actual,
            }
            .into(),
            DidCcError::InvalidConfidenceLevel | DidCcError::InvalidCrossFitFolds => {
                Self::Unsupported {
                    what: "the compositional-change options",
                }
            }
            DidCcError::InvalidConfig(_) => Self::Singular,
            DidCcError::InsufficientCellForCrossFit { folds, cell } => {
                let (treated, post) = cell.flags();
                Self::CrossFitCell {
                    folds,
                    treated,
                    post,
                }
            }
            DidCcError::DegenerateHausmanVariance => Self::DegenerateVariance,
        }
    }
}

#[derive(Debug, Clone)]
struct DidCcPreparedData {
    feature_count: usize,
    treated_post_n: usize,
    treated_pre_n: usize,
    control_post_n: usize,
    control_pre_n: usize,
    total_weight: f64,
    treated: Vec<f64>,
    post_period: Vec<f64>,
    outcomes: Vec<f64>,
    sampling_weights: Vec<f64>,
    class_labels: Vec<usize>,
    design_matrix_flat: Vec<f64>,
}

#[derive(Debug, Clone)]
struct OutcomePredictions {
    m11: Vec<f64>,
    m10: Vec<f64>,
    m01: Vec<f64>,
    m00: Vec<f64>,
}

#[derive(Debug, Clone)]
struct DidCcNuisance {
    normalized_weights: Vec<f64>,
    outcome_predictions: OutcomePredictions,
    generalized_propensity: Vec<[f64; 4]>,
    pooled_treated_propensity: Vec<f64>,
}

type CrossFitNuisance = (OutcomePredictions, Vec<[f64; 4]>, Vec<f64>);

struct OutcomeCellScratch {
    subset_design: Mat<f64>,
    subset_outcomes: Vec<f64>,
    subset_weights: Vec<f64>,
    local_linear: LocalLinearScratch,
}

impl OutcomeCellScratch {
    fn new(max_rows: usize, feature_count: usize) -> Self {
        Self {
            subset_design: Mat::zeros(max_rows, feature_count),
            subset_outcomes: Vec::with_capacity(max_rows),
            subset_weights: Vec::with_capacity(max_rows),
            local_linear: LocalLinearScratch::new(feature_count.saturating_sub(1)),
        }
    }
}

struct OutcomeTrainData<'a> {
    outcomes: &'a [f64],
    weights: &'a [f64],
    design: MatRef<'a, f64>,
}

fn robust(
    observations: &[DrDidRepeatedObservation],
    config: DidCcConfig,
) -> Result<DidCcEstimate, DidCcError> {
    validate_did_cc_config(config)?;
    let prepared = prepare_did_cc_data(observations)?;
    let nuisance = fit_did_cc_nuisance(&prepared, config)?;
    Ok(build_robust_estimate(&prepared, &nuisance, config))
}

fn stationary(
    observations: &[DrDidRepeatedObservation],
    config: DidCcConfig,
) -> Result<DidCcEstimate, DidCcError> {
    validate_did_cc_config(config)?;
    let prepared = prepare_did_cc_data(observations)?;
    let nuisance = fit_did_cc_nuisance(&prepared, config)?;
    Ok(build_stationary_estimate(&prepared, &nuisance, config))
}

fn hausman(
    observations: &[DrDidRepeatedObservation],
    config: DidCcConfig,
) -> Result<DidCcHausmanTest, DidCcError> {
    validate_did_cc_config(config)?;
    let prepared = prepare_did_cc_data(observations)?;
    let nuisance = fit_did_cc_nuisance(&prepared, config)?;
    let robust = build_robust_estimate(&prepared, &nuisance, config);
    let stationary = build_stationary_estimate(&prepared, &nuisance, config);

    let difference = stationary.att - robust.att;
    let difference_se =
        standard_error_from_difference(&stationary.influence_function, &robust.influence_function);
    if !difference_se.is_finite() || difference_se <= 0.0 {
        return Err(DidCcError::DegenerateHausmanVariance);
    }
    let statistic = (difference / difference_se).powi(2);
    let p_value = chi_square_sf(statistic, 1.0);

    Ok(DidCcHausmanTest {
        robust,
        stationary,
        difference,
        difference_se,
        statistic,
        p_value,
    })
}

fn validate_did_cc_config(config: DidCcConfig) -> Result<(), DidCcError> {
    if !validate_confidence_level(config.drdid.confidence_level) {
        return Err(DidCcError::InvalidConfidenceLevel);
    }
    if config.cross_fit_folds == 0 {
        return Err(DidCcError::InvalidCrossFitFolds);
    }
    if config.bootstrap_confidence_intervals && config.drdid.bootstrap_reps == 0 {
        return Err(DidCcError::InvalidConfig(
            "bootstrap confidence intervals need drdid.bootstrap_reps > 0".to_string(),
        ));
    }
    Ok(())
}

fn prepare_did_cc_data(
    observations: &[DrDidRepeatedObservation],
) -> Result<DidCcPreparedData, DidCcError> {
    if observations.is_empty() {
        return Err(DidCcError::EmptyInput);
    }
    let covariate_count = observations.first().map_or(0, |row| row.covariates.len());
    let feature_count = covariate_count + 1;
    let observation_count = observations.len();

    let mut treated_post_n = 0usize;
    let mut treated_pre_n = 0usize;
    let mut control_post_n = 0usize;
    let mut control_pre_n = 0usize;
    let mut total_weight = 0.0;

    let mut treated = Vec::with_capacity(observation_count);
    let mut post_period = Vec::with_capacity(observation_count);
    let mut outcomes = Vec::with_capacity(observation_count);
    let mut sampling_weights = Vec::with_capacity(observation_count);
    let mut class_labels = Vec::with_capacity(observation_count);
    let mut design_matrix_flat = Vec::with_capacity(observation_count * feature_count);

    for row in observations {
        let covariates = &row.covariates;
        if covariates.len() != covariate_count {
            return Err(DidCcError::InconsistentCovariateCount {
                expected: covariate_count,
                actual: covariates.len(),
            });
        }
        let outcome = row.outcome;
        if !outcome.is_finite() {
            return Err(DidCcError::InvalidOutcome { value: outcome });
        }
        let weight = row.weight;
        if !weight.is_finite() || weight <= 0.0 {
            return Err(DidCcError::InvalidWeight { value: weight });
        }
        for covariate in covariates {
            if !covariate.is_finite() {
                return Err(DidCcError::InvalidCovariate { value: *covariate });
            }
        }

        match (row.treated, row.post_period) {
            (true, true) => treated_post_n += 1,
            (true, false) => treated_pre_n += 1,
            (false, true) => control_post_n += 1,
            (false, false) => control_pre_n += 1,
        }

        let treated_value = if row.treated { 1.0 } else { 0.0 };
        let post_value = if row.post_period { 1.0 } else { 0.0 };
        treated.push(treated_value);
        post_period.push(post_value);
        class_labels.push(class_index_raw(treated_value, post_value));
        outcomes.push(outcome);
        sampling_weights.push(weight);
        total_weight += weight;
        design_matrix_flat.push(1.0);
        design_matrix_flat.extend_from_slice(covariates);
    }

    if treated_post_n == 0 || treated_pre_n == 0 {
        return Err(DidCcError::NoTreated);
    }
    if control_post_n == 0 || control_pre_n == 0 {
        return Err(DidCcError::NoControl);
    }

    Ok(DidCcPreparedData {
        feature_count,
        treated_post_n,
        treated_pre_n,
        control_post_n,
        control_pre_n,
        total_weight,
        treated,
        post_period,
        outcomes,
        sampling_weights,
        class_labels,
        design_matrix_flat,
    })
}

fn fit_did_cc_nuisance(
    prepared: &DidCcPreparedData,
    config: DidCcConfig,
) -> Result<DidCcNuisance, DidCcError> {
    let normalized_weights = normalize_weights_to_n(&prepared.sampling_weights)
        .map_err(|_| DidCcError::InvalidWeight { value: 0.0 })?;
    let design_matrix = design_matrix_view(prepared);
    let (outcome_predictions, generalized_propensity, pooled_treated_propensity) =
        if config.cross_fit_folds <= 1 {
            let cell_indices = collect_cell_indices(&prepared.class_labels);
            (
                fit_outcome_predictions(
                    &prepared.outcomes,
                    &prepared.sampling_weights,
                    &cell_indices,
                    design_matrix,
                    design_matrix,
                    config.drdid.ridge,
                )?,
                fit_generalized_propensity(
                    &prepared.class_labels,
                    &prepared.sampling_weights,
                    design_matrix,
                    design_matrix,
                    config.drdid,
                )?,
                fit_pooled_treated_propensity(
                    &prepared.treated,
                    design_matrix,
                    design_matrix,
                    config.drdid,
                )?,
            )
        } else {
            let nuisance = fit_cross_fitted_nuisance(
                prepared,
                design_matrix,
                config.cross_fit_folds,
                config.cross_fit_seed,
                config.drdid,
            )?;
            (nuisance.0, nuisance.1, nuisance.2)
        };

    Ok(DidCcNuisance {
        normalized_weights,
        outcome_predictions,
        generalized_propensity,
        pooled_treated_propensity,
    })
}

fn fit_cross_fitted_nuisance(
    prepared: &DidCcPreparedData,
    _full_design_matrix: MatRef<'_, f64>,
    folds: usize,
    seed: u64,
    drdid_config: crate::types::DrDidConfig,
) -> Result<CrossFitNuisance, DidCcError> {
    let assignment = assign_cross_fit_folds(prepared, folds, seed)?;
    let fold_indices = collect_fold_indices(&assignment, folds);
    let row_count = prepared.outcomes.len();
    let mut m11 = vec![0.0; row_count];
    let mut m10 = vec![0.0; row_count];
    let mut m01 = vec![0.0; row_count];
    let mut m00 = vec![0.0; row_count];
    let mut generalized_propensity = vec![[0.0; 4]; row_count];
    let mut pooled_treated_propensity = vec![0.0; row_count];
    let mut train_outcomes = Vec::with_capacity(row_count);
    let mut train_weights = Vec::with_capacity(row_count);
    let mut train_class_labels = Vec::with_capacity(row_count);
    let mut train_treated = Vec::with_capacity(row_count);
    let mut train_indices = Vec::with_capacity(row_count);
    let mut train_cell_indices: [Vec<usize>; 4] = std::array::from_fn(|_| Vec::new());
    let mut train_scratch = Mat::<f64>::zeros(row_count, prepared.feature_count);
    let mut test_scratch = Mat::<f64>::zeros(row_count, prepared.feature_count);

    for fold in 0..folds {
        let test_indices = &fold_indices[fold];
        if test_indices.is_empty() {
            continue;
        }
        train_indices.clear();
        for (other_fold, indices) in fold_indices.iter().enumerate() {
            if other_fold != fold {
                train_indices.extend(indices.iter().copied());
            }
        }
        let train_size = train_indices.len();
        let test_size = test_indices.len();

        fill_design_from_flat(
            &prepared.design_matrix_flat,
            prepared.feature_count,
            &train_indices,
            &mut train_scratch,
        );
        fill_design_from_flat(
            &prepared.design_matrix_flat,
            prepared.feature_count,
            test_indices,
            &mut test_scratch,
        );
        let train_design = train_scratch.as_ref().subrows(0, train_size);
        let test_design = test_scratch.as_ref().subrows(0, test_size);
        train_outcomes.clear();
        train_weights.clear();
        train_class_labels.clear();
        train_treated.clear();
        for cell_indices in &mut train_cell_indices {
            cell_indices.clear();
        }
        for (local_idx, &idx) in train_indices.iter().enumerate() {
            train_outcomes.push(prepared.outcomes[idx]);
            train_weights.push(prepared.sampling_weights[idx]);
            train_class_labels.push(prepared.class_labels[idx]);
            train_treated.push(prepared.treated[idx]);
            train_cell_indices[prepared.class_labels[idx]].push(local_idx);
        }

        let fold_outcomes = fit_outcome_predictions(
            &train_outcomes,
            &train_weights,
            &train_cell_indices,
            train_design.as_ref(),
            test_design.as_ref(),
            drdid_config.ridge,
        )?;
        let fold_generalized = fit_generalized_propensity(
            &train_class_labels,
            &train_weights,
            train_design.as_ref(),
            test_design.as_ref(),
            drdid_config,
        )?;
        let fold_pooled = fit_pooled_treated_propensity(
            &train_treated,
            train_design.as_ref(),
            test_design.as_ref(),
            drdid_config,
        )?;

        for (local_idx, global_idx) in test_indices.iter().enumerate() {
            m11[*global_idx] = fold_outcomes.m11[local_idx];
            m10[*global_idx] = fold_outcomes.m10[local_idx];
            m01[*global_idx] = fold_outcomes.m01[local_idx];
            m00[*global_idx] = fold_outcomes.m00[local_idx];
            generalized_propensity[*global_idx] = fold_generalized[local_idx];
            pooled_treated_propensity[*global_idx] = fold_pooled[local_idx];
        }
    }

    Ok((
        OutcomePredictions { m11, m10, m01, m00 },
        generalized_propensity,
        pooled_treated_propensity,
    ))
}

fn assign_cross_fit_folds(
    prepared: &DidCcPreparedData,
    folds: usize,
    seed: u64,
) -> Result<Vec<usize>, DidCcError> {
    let mut assignment = vec![0; prepared.outcomes.len()];
    let mut rng = Stream::seeded(seed);
    for (treated_cell, post_cell) in [(true, true), (true, false), (false, true), (false, false)] {
        let mut indices = prepared
            .treated
            .iter()
            .zip(prepared.post_period.iter())
            .enumerate()
            .filter_map(|(idx, (treated, post))| {
                matches_cell(*treated, *post, treated_cell, post_cell).then_some(idx)
            })
            .collect::<Vec<_>>();
        if indices.len() < folds {
            return Err(DidCcError::InsufficientCellForCrossFit {
                folds,
                cell: DidCell::from_parts(
                    TreatmentGroup::from_bool(treated_cell),
                    TimePeriod::from_bool(post_cell),
                ),
            });
        }
        rng.shuffle(&mut indices);
        for (position, idx) in indices.into_iter().enumerate() {
            assignment[idx] = position % folds;
        }
    }
    Ok(assignment)
}

fn collect_fold_indices(assignment: &[usize], folds: usize) -> Vec<Vec<usize>> {
    let mut fold_indices = vec![Vec::new(); folds];
    for (idx, fold) in assignment.iter().copied().enumerate() {
        fold_indices[fold].push(idx);
    }
    fold_indices
}

fn fill_design_from_flat(
    flat: &[f64],
    feature_count: usize,
    indices: &[usize],
    out: &mut Mat<f64>,
) {
    debug_assert!(out.nrows() >= indices.len());
    debug_assert_eq!(out.ncols(), feature_count);
    for (out_row, &src_row) in indices.iter().enumerate() {
        let src = &flat[src_row * feature_count..(src_row + 1) * feature_count];
        for (col, &value) in src.iter().enumerate() {
            out[(out_row, col)] = value;
        }
    }
}

fn design_matrix_view(prepared: &DidCcPreparedData) -> MatRef<'_, f64> {
    MatRef::from_row_major_slice(
        &prepared.design_matrix_flat,
        prepared.outcomes.len(),
        prepared.feature_count,
    )
}

fn fit_outcome_predictions(
    train_outcomes: &[f64],
    train_weights: &[f64],
    cell_indices: &[Vec<usize>; 4],
    train_design: epistat_linalg::MatRef<'_, f64>,
    query_design: epistat_linalg::MatRef<'_, f64>,
    ridge: f64,
) -> Result<OutcomePredictions, DidCcError> {
    let train_data = OutcomeTrainData {
        outcomes: train_outcomes,
        weights: train_weights,
        design: train_design,
    };
    let mut scratch = OutcomeCellScratch::new(train_design.nrows(), train_design.ncols());
    Ok(OutcomePredictions {
        m11: fit_cell_outcome(
            &train_data,
            query_design,
            &cell_indices[0],
            0,
            ridge,
            &mut scratch,
        )?,
        m10: fit_cell_outcome(
            &train_data,
            query_design,
            &cell_indices[1],
            1,
            ridge,
            &mut scratch,
        )?,
        m01: fit_cell_outcome(
            &train_data,
            query_design,
            &cell_indices[2],
            2,
            ridge,
            &mut scratch,
        )?,
        m00: fit_cell_outcome(
            &train_data,
            query_design,
            &cell_indices[3],
            3,
            ridge,
            &mut scratch,
        )?,
    })
}

fn collect_cell_indices(class_labels: &[usize]) -> [Vec<usize>; 4] {
    let mut cells = std::array::from_fn::<_, 4, _>(|_| Vec::<usize>::new());
    for (idx, class_idx) in class_labels.iter().copied().enumerate() {
        cells[class_idx].push(idx);
    }
    cells
}

fn copy_outcome_subset_into_scratch(
    train_data: &OutcomeTrainData<'_>,
    indices: &[usize],
    scratch: &mut OutcomeCellScratch,
) {
    scratch.subset_outcomes.clear();
    scratch.subset_weights.clear();
    for (out_row, &src_row) in indices.iter().enumerate() {
        for col in 0..train_data.design.ncols() {
            scratch.subset_design[(out_row, col)] = *train_data.design.get(src_row, col);
        }
        scratch.subset_outcomes.push(train_data.outcomes[src_row]);
        scratch.subset_weights.push(train_data.weights[src_row]);
    }
}

fn fit_cell_outcome(
    train_data: &OutcomeTrainData<'_>,
    query_design: epistat_linalg::MatRef<'_, f64>,
    indices: &[usize],
    class_idx: usize,
    ridge: f64,
    scratch: &mut OutcomeCellScratch,
) -> Result<Vec<f64>, DidCcError> {
    if indices.is_empty() {
        let (treated_cell, post_cell) = class_cell_flags(class_idx);
        return Err(DidCcError::MissingCell {
            cell: DidCell::from_parts(
                TreatmentGroup::from_bool(treated_cell),
                TimePeriod::from_bool(post_cell),
            ),
        });
    }

    copy_outcome_subset_into_scratch(train_data, indices, scratch);
    let subset_design = scratch.subset_design.as_ref().subrows(0, indices.len());

    let model = LocalLinearOutcome {
        ridge,
        ..LocalLinearOutcome::default()
    };
    model
        .predict_from_training(
            subset_design,
            &scratch.subset_outcomes,
            &scratch.subset_weights,
            query_design,
            &mut scratch.local_linear,
        )
        .map_err(|err| DidCcError::InvalidConfig(err.to_string()))
}

fn fit_generalized_propensity(
    class_labels: &[usize],
    sampling_weights: &[f64],
    train_design: epistat_linalg::MatRef<'_, f64>,
    query_design: epistat_linalg::MatRef<'_, f64>,
    config: crate::types::DrDidConfig,
) -> Result<Vec<[f64; 4]>, DidCcError> {
    let estimator = MultinomialSoftmaxPS::new(propensity_cfg(config)?, config.ridge);
    if let Ok(params) = estimator.fit_multinomial(train_design, class_labels, sampling_weights, 4) {
        return Ok(
            crate::estimators::propensity::common::softmax_scores_fixed4(
                query_design,
                params.beta.as_ref(),
                config.propensity_clip,
            ),
        );
    }

    if train_design.ncols() > 1 {
        let intercept_only = Mat::<f64>::ones(train_design.nrows(), 1);
        if let Ok(params) =
            estimator.fit_multinomial(intercept_only.as_ref(), class_labels, sampling_weights, 4)
        {
            let query_intercept_only = Mat::<f64>::ones(query_design.nrows(), 1);
            return Ok(
                crate::estimators::propensity::common::softmax_scores_fixed4(
                    query_intercept_only.as_ref(),
                    params.beta.as_ref(),
                    config.propensity_clip,
                ),
            );
        }
    }

    Ok(empirical_generalized_propensity(
        class_labels,
        sampling_weights,
        query_design.nrows(),
        config.propensity_clip,
    ))
}

fn fit_pooled_treated_propensity(
    treated: &[f64],
    train_design: epistat_linalg::MatRef<'_, f64>,
    query_design: epistat_linalg::MatRef<'_, f64>,
    config: crate::types::DrDidConfig,
) -> Result<Vec<f64>, DidCcError> {
    let target = MatRef::from_column_major_slice(treated, treated.len(), 1);
    let estimator = LogisticPS::new(propensity_cfg(config)?);
    if let Ok(params) = estimator.fit(train_design, target) {
        return Ok(crate::estimators::propensity::common::logistic_scores(
            query_design,
            params.beta.as_ref(),
        )
        .into_iter()
        .map(|score| score.clamp(config.propensity_clip, 1.0 - config.propensity_clip))
        .collect());
    }

    if train_design.ncols() > 1 {
        let intercept_only = Mat::<f64>::ones(train_design.nrows(), 1);
        if let Ok(params) = estimator.fit(intercept_only.as_ref(), target) {
            let query_intercept_only = Mat::<f64>::ones(query_design.nrows(), 1);
            return Ok(crate::estimators::propensity::common::logistic_scores(
                query_intercept_only.as_ref(),
                params.beta.as_ref(),
            )
            .into_iter()
            .map(|score| score.clamp(config.propensity_clip, 1.0 - config.propensity_clip))
            .collect());
        }
    }

    Ok(empirical_pooled_treated_propensity(
        treated,
        query_design.nrows(),
        config.propensity_clip,
    ))
}

fn class_index_raw(treated: f64, post_period: f64) -> usize {
    match (treated > 0.5, post_period > 0.5) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (false, false) => 3,
    }
}

const fn class_cell_flags(class_idx: usize) -> (bool, bool) {
    match class_idx {
        0 => (true, true),
        1 => (true, false),
        2 => (false, true),
        _ => (false, false),
    }
}

fn empirical_generalized_propensity(
    class_labels: &[usize],
    sampling_weights: &[f64],
    query_rows: usize,
    propensity_clip: f64,
) -> Vec<[f64; 4]> {
    let mut weighted_counts = [0.0; 4];
    for (class_idx, weight) in class_labels.iter().zip(sampling_weights.iter()) {
        weighted_counts[*class_idx] += *weight;
    }
    let total = weighted_counts.iter().sum::<f64>().max(f64::EPSILON);
    let mut shares = weighted_counts.map(|count| (count / total).clamp(propensity_clip, 1.0));
    let share_sum = shares.iter().sum::<f64>().max(f64::EPSILON);
    for share in &mut shares {
        *share /= share_sum;
    }
    vec![shares; query_rows]
}

fn empirical_pooled_treated_propensity(
    treated: &[f64],
    query_rows: usize,
    propensity_clip: f64,
) -> Vec<f64> {
    let treated_share = (treated.iter().sum::<f64>() / crate::util::usize_to_f64(treated.len()))
        .clamp(propensity_clip, 1.0 - propensity_clip);
    vec![treated_share; query_rows]
}

fn propensity_cfg(config: crate::types::DrDidConfig) -> Result<PropensityConfig, DidCcError> {
    Ok(PropensityConfig {
        max_iter: u64::try_from(config.max_iter)
            .map_err(|_| DidCcError::InvalidConfig("max_iter exceeds u64".to_string()))?,
        tol: config.tol,
        min_weight: config.propensity_clip,
    })
}

fn build_robust_estimate(
    prepared: &DidCcPreparedData,
    nuisance: &DidCcNuisance,
    config: DidCcConfig,
) -> DidCcEstimate {
    let (w11, w10, w01, w00) = precompute_robust_weights(prepared, nuisance);
    let scores = (0..prepared.outcomes.len())
        .map(|idx| {
            robust_score(
                prepared, nuisance, idx, w11[idx], w10[idx], w01[idx], w00[idx],
            )
        })
        .collect::<Vec<_>>();
    build_did_cc_estimate(prepared, config, &scores)
}

fn build_stationary_estimate(
    prepared: &DidCcPreparedData,
    nuisance: &DidCcNuisance,
    config: DidCcConfig,
) -> DidCcEstimate {
    let weights = precompute_stationary_weights(prepared, nuisance);
    let scores = (0..prepared.outcomes.len())
        .map(|idx| stationary_score(prepared, nuisance, idx, &weights))
        .collect::<Vec<_>>();
    build_did_cc_estimate(prepared, config, &scores)
}

fn build_did_cc_estimate(
    prepared: &DidCcPreparedData,
    config: DidCcConfig,
    scores: &[f64],
) -> DidCcEstimate {
    let n = crate::util::usize_to_f64(scores.len());
    let att = scores.iter().sum::<f64>() / n;
    let influence_function = scores.iter().map(|score| score - att).collect::<Vec<_>>();
    let se = standard_error_from_influence(&influence_function);
    let (ci_low, ci_high) = if config.bootstrap_confidence_intervals {
        multiplier_bootstrap_ci(
            att,
            &influence_function,
            config.drdid.confidence_level,
            config.drdid.bootstrap(),
        )
    } else {
        normal_interval(att, se, config.drdid.confidence_level)
    };

    DidCcEstimate {
        att,
        se,
        ci_low,
        ci_high,
        treated_post_n: prepared.treated_post_n,
        treated_pre_n: prepared.treated_pre_n,
        control_post_n: prepared.control_post_n,
        control_pre_n: prepared.control_pre_n,
        total_weight: prepared.total_weight,
        influence_function,
    }
}

fn robust_score(
    prepared: &DidCcPreparedData,
    nuisance: &DidCcNuisance,
    idx: usize,
    w11: f64,
    w10: f64,
    w01: f64,
    w00: f64,
) -> f64 {
    let tau_dr_signal = prepared.outcomes[idx]
        - (nuisance.outcome_predictions.m10[idx] + nuisance.outcome_predictions.m01[idx]
            - nuisance.outcome_predictions.m00[idx]);

    let treated_pre_residual = prepared.outcomes[idx] - nuisance.outcome_predictions.m10[idx];
    let control_post_residual = prepared.outcomes[idx] - nuisance.outcome_predictions.m01[idx];
    let control_pre_residual = prepared.outcomes[idx] - nuisance.outcome_predictions.m00[idx];

    let treated_component = w11.mul_add(tau_dr_signal, -(w10 * treated_pre_residual));
    let control_component = w00.mul_add(control_pre_residual, -(w01 * control_post_residual));
    treated_component + control_component
}

fn stationary_score(
    prepared: &DidCcPreparedData,
    nuisance: &DidCcNuisance,
    idx: usize,
    weights: &DidCcStationaryWeights,
) -> f64 {
    let tau_x = nuisance.outcome_predictions.m11[idx]
        - nuisance.outcome_predictions.m10[idx]
        - nuisance.outcome_predictions.m01[idx]
        + nuisance.outcome_predictions.m00[idx];

    let treated_post_residual = prepared.outcomes[idx] - nuisance.outcome_predictions.m11[idx];
    let treated_pre_residual = prepared.outcomes[idx] - nuisance.outcome_predictions.m10[idx];
    let control_post_residual = prepared.outcomes[idx] - nuisance.outcome_predictions.m01[idx];
    let control_pre_residual = prepared.outcomes[idx] - nuisance.outcome_predictions.m00[idx];

    let treated_component = weights.treated_weight[idx].mul_add(
        tau_x,
        weights.w11[idx].mul_add(
            treated_post_residual,
            -(weights.w10[idx] * treated_pre_residual),
        ),
    );
    let control_component = weights.w00[idx].mul_add(
        control_pre_residual,
        -(weights.w01[idx] * control_post_residual),
    );
    treated_component + control_component
}

fn precompute_robust_weights(
    prepared: &DidCcPreparedData,
    nuisance: &DidCcNuisance,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut w11 = Vec::with_capacity(prepared.outcomes.len());
    let mut w10 = Vec::with_capacity(prepared.outcomes.len());
    let mut w01 = Vec::with_capacity(prepared.outcomes.len());
    let mut w00 = Vec::with_capacity(prepared.outcomes.len());
    for idx in 0..prepared.outcomes.len() {
        let normalized = nuisance.normalized_weights[idx];
        let gp = nuisance.generalized_propensity[idx];
        w11.push(normalized * treated_post_indicator(prepared, idx));
        w10.push(normalized * treated_pre_indicator(prepared, idx) * gp[0] / gp[1].max(1e-12));
        w01.push(normalized * control_post_indicator(prepared, idx) * gp[0] / gp[2].max(1e-12));
        w00.push(normalized * control_pre_indicator(prepared, idx) * gp[0] / gp[3].max(1e-12));
    }
    normalize_cell_weight_vector_in_place(&mut w11);
    normalize_cell_weight_vector_in_place(&mut w10);
    normalize_cell_weight_vector_in_place(&mut w01);
    normalize_cell_weight_vector_in_place(&mut w00);
    (w11, w10, w01, w00)
}

struct DidCcStationaryWeights {
    treated_weight: Vec<f64>,
    w11: Vec<f64>,
    w10: Vec<f64>,
    w01: Vec<f64>,
    w00: Vec<f64>,
}

fn precompute_stationary_weights(
    prepared: &DidCcPreparedData,
    nuisance: &DidCcNuisance,
) -> DidCcStationaryWeights {
    let mut treated_weight = Vec::with_capacity(prepared.outcomes.len());
    let mut w11 = Vec::with_capacity(prepared.outcomes.len());
    let mut w10 = Vec::with_capacity(prepared.outcomes.len());
    let mut w01 = Vec::with_capacity(prepared.outcomes.len());
    let mut w00 = Vec::with_capacity(prepared.outcomes.len());
    for idx in 0..prepared.outcomes.len() {
        let normalized = nuisance.normalized_weights[idx];
        let pooled = nuisance.pooled_treated_propensity[idx];
        let odds = pooled / (1.0 - pooled).max(1e-12);
        treated_weight.push(normalized * prepared.treated[idx]);
        w11.push(normalized * treated_post_indicator(prepared, idx));
        w10.push(normalized * treated_pre_indicator(prepared, idx));
        w01.push(normalized * control_post_indicator(prepared, idx) * odds);
        w00.push(normalized * control_pre_indicator(prepared, idx) * odds);
    }
    normalize_cell_weight_vector_in_place(&mut treated_weight);
    normalize_cell_weight_vector_in_place(&mut w11);
    normalize_cell_weight_vector_in_place(&mut w10);
    normalize_cell_weight_vector_in_place(&mut w01);
    normalize_cell_weight_vector_in_place(&mut w00);
    DidCcStationaryWeights {
        treated_weight,
        w11,
        w10,
        w01,
        w00,
    }
}

fn normalize_cell_weight_vector_in_place(weights: &mut [f64]) {
    let denominator = weights.iter().sum::<f64>() / crate::util::usize_to_f64(weights.len());
    if denominator <= 0.0 {
        weights.fill(0.0);
    } else {
        for weight in weights {
            *weight /= denominator;
        }
    }
}

fn standard_error_from_difference(left: &[f64], right: &[f64]) -> f64 {
    debug_assert_eq!(left.len(), right.len());
    let n_f = crate::util::usize_to_f64(left.len());
    let mut mean = 0.0;
    let mut m2 = 0.0;
    for (idx, (l, r)) in left.iter().zip(right.iter()).enumerate() {
        let x = l - r;
        let count = crate::util::usize_to_f64(idx + 1);
        let delta = x - mean;
        mean += delta / count;
        let delta2 = x - mean;
        m2 += delta * delta2;
    }
    let centered_mean_square = m2 / n_f;
    centered_mean_square.sqrt() / n_f.sqrt()
}

fn treated_post_indicator(prepared: &DidCcPreparedData, idx: usize) -> f64 {
    prepared.treated[idx] * prepared.post_period[idx]
}

fn treated_pre_indicator(prepared: &DidCcPreparedData, idx: usize) -> f64 {
    prepared.treated[idx] * (1.0 - prepared.post_period[idx])
}

fn control_post_indicator(prepared: &DidCcPreparedData, idx: usize) -> f64 {
    (1.0 - prepared.treated[idx]) * prepared.post_period[idx]
}

fn control_pre_indicator(prepared: &DidCcPreparedData, idx: usize) -> f64 {
    (1.0 - prepared.treated[idx]) * (1.0 - prepared.post_period[idx])
}

fn matches_cell(treated: f64, post_period: f64, treated_cell: bool, post_cell: bool) -> bool {
    (treated > 0.5) == treated_cell && (post_period > 0.5) == post_cell
}

#[cfg(test)]
mod tests {
    use super::*;
    use epistat_rng::Stream;
    use rand_distr::{Bernoulli, Distribution, Normal};

    fn row(treated: bool, post_period: bool, outcome: f64, x: f64) -> DrDidRepeatedObservation {
        DrDidRepeatedObservation {
            treated,
            post_period,
            outcome,
            weight: 1.0,
            covariates: vec![x],
        }
    }

    #[test]
    fn did_cc_robust_and_stationary_align_without_compositional_change() {
        let mut rows = Vec::new();
        for post in [false, true] {
            for replicate in 0_i32..200 {
                let x = if replicate % 2 == 0 { 0.0 } else { 1.0 };
                let base = x + if post { 1.0 } else { 0.0 };
                rows.push(row(false, post, base, x));
                rows.push(row(true, post, base + if post { 2.0 } else { 0.0 }, x));
            }
        }

        let robust = robust(&rows, DidCcConfig::default()).unwrap();
        let stationary = stationary(&rows, DidCcConfig::default()).unwrap();
        let test = hausman(&rows, DidCcConfig::default()).unwrap();

        assert!((robust.att - 2.0).abs() < 1e-6);
        assert!((stationary.att - 2.0).abs() < 1e-6);
        assert!(test.difference.abs() < 1e-6);
        assert!(test.p_value > 0.05);
    }

    #[test]
    fn did_cc_hausman_detects_composition_shift_when_catt_is_heterogeneous() {
        let mut rows = Vec::new();
        for replicate in 0_i32..300 {
            let control_pre_x = match replicate % 3 {
                0 => 0.0,
                1 => 1.0,
                _ => 2.0,
            };
            let control_post_x = control_pre_x;
            rows.push(row(false, false, control_pre_x, control_pre_x));
            rows.push(row(false, true, control_post_x + 1.0, control_post_x));

            let treated_pre_x = if replicate % 4 == 0 { 0.0 } else { 1.0 };
            let treated_post_x = if replicate % 4 == 0 { 1.0 } else { 2.0 };
            rows.push(row(true, false, treated_pre_x, treated_pre_x));
            rows.push(row(
                true,
                true,
                treated_post_x + 1.0 + (1.0 + treated_post_x),
                treated_post_x,
            ));
        }

        let robust = robust(&rows, DidCcConfig::default()).unwrap();
        let stationary = stationary(&rows, DidCcConfig::default()).unwrap();
        let test = hausman(&rows, DidCcConfig::default()).unwrap();

        assert!(robust.att > stationary.att + 0.5);
        assert!(test.difference.abs() > 0.5);
        assert!(test.p_value < 0.05);
    }

    #[test]
    fn multinomial_generalized_propensity_is_normalized() {
        let rows = vec![
            row(true, true, 3.0, 0.0),
            row(true, false, 1.0, 0.5),
            row(false, true, 2.0, 1.0),
            row(false, false, 0.0, 1.5),
            row(true, true, 4.0, 2.0),
            row(false, false, 1.0, 2.5),
        ];

        let prepared = prepare_did_cc_data(&rows).unwrap();
        let design_matrix = design_matrix_view(&prepared);
        let probabilities = fit_generalized_propensity(
            &prepared.class_labels,
            &prepared.sampling_weights,
            design_matrix,
            design_matrix,
            crate::types::DrDidConfig::default(),
        )
        .unwrap();

        assert_eq!(probabilities.len(), rows.len());
        for probability in probabilities {
            let sum = probability.iter().sum::<f64>();
            assert!((sum - 1.0).abs() < 1e-10);
            assert!(
                probability
                    .iter()
                    .all(|value| value.is_finite() && *value > 0.0)
            );
        }
    }

    #[test]
    fn did_cc_cross_fit_matches_stable_design() {
        let mut rows = Vec::new();
        for post in [false, true] {
            for replicate in 0_i32..240 {
                let x = f64::from(replicate % 4) / 3.0;
                let base = 2.0f64.mul_add(x, if post { 1.0 } else { 0.0 });
                rows.push(row(false, post, base, x));
                rows.push(row(true, post, base + if post { 1.5 } else { 0.0 }, x));
            }
        }

        let no_split = robust(&rows, DidCcConfig::default()).unwrap();
        let cross_fit = robust(
            &rows,
            DidCcConfig {
                drdid: crate::types::DrDidConfig {
                    max_iter: 1_000,
                    tol: 1e-6,
                    ..crate::types::DrDidConfig::default()
                },
                cross_fit_folds: 4,
                cross_fit_seed: 7,
                ..DidCcConfig::default()
            },
        )
        .unwrap();

        assert!((no_split.att - 1.5).abs() < 1e-3);
        assert!((cross_fit.att - 1.5).abs() < 0.1);
        assert!((cross_fit.att - no_split.att).abs() < 0.1);
    }

    #[test]
    fn did_cc_cross_fit_rejects_too_many_folds_for_a_cell() {
        let rows = vec![
            row(true, true, 3.0, 0.0),
            row(true, false, 1.0, 0.0),
            row(false, true, 2.0, 1.0),
            row(false, false, 0.0, 1.0),
        ];

        let err = robust(
            &rows,
            DidCcConfig {
                cross_fit_folds: 2,
                cross_fit_seed: 11,
                ..DidCcConfig::default()
            },
        )
        .expect_err("cross-fit should reject tiny cells");

        assert_eq!(
            err,
            DidCcError::InsufficientCellForCrossFit {
                folds: 2,
                cell: DidCell::TreatedPost,
            }
        );
    }

    fn simulate_did_cc_sample(
        rng: &mut Stream,
        observations_per_cell: usize,
        composition_shift: bool,
    ) -> Vec<DrDidRepeatedObservation> {
        let noise = Normal::new(0.0, 0.35).expect("normal");
        let treated_pre_mix =
            Bernoulli::new(if composition_shift { 0.25 } else { 0.5 }).expect("mix");
        let treated_post_mix =
            Bernoulli::new(if composition_shift { 0.75 } else { 0.5 }).expect("mix");
        let control_mix = Bernoulli::new(0.5).expect("mix");
        let mut rows = Vec::with_capacity(observations_per_cell * 4);

        for _ in 0..observations_per_cell {
            let x_control_pre = if control_mix.sample(rng) { 0.0 } else { 1.0 };
            let x_control_post = if control_mix.sample(rng) { 0.0 } else { 1.0 };
            let x_treated_pre = if treated_pre_mix.sample(rng) {
                1.0
            } else {
                0.0
            };
            let x_treated_post = if treated_post_mix.sample(rng) {
                1.0
            } else {
                0.0
            };

            let y_control_pre = x_control_pre + noise.sample(rng);
            let y_control_post = x_control_post + 1.0 + noise.sample(rng);
            let y_treated_pre = x_treated_pre + noise.sample(rng);
            let treatment_effect = if composition_shift {
                1.0 + x_treated_post
            } else {
                1.5
            };
            let y_treated_post = x_treated_post + 1.0 + treatment_effect + noise.sample(rng);

            rows.push(row(false, false, y_control_pre, x_control_pre));
            rows.push(row(false, true, y_control_post, x_control_post));
            rows.push(row(true, false, y_treated_pre, x_treated_pre));
            rows.push(row(true, true, y_treated_post, x_treated_post));
        }

        rows
    }

    #[test]
    fn did_cc_cross_fit_simulation_recovers_constant_effect() {
        let true_att = 1.5;
        let draws = 8;
        let mut rng = Stream::seeded(24_0310);
        let mut estimates = Vec::with_capacity(draws);

        for draw in 0..draws {
            let rows = simulate_did_cc_sample(&mut rng, 80, false);
            let estimate = robust(
                &rows,
                DidCcConfig {
                    drdid: crate::types::DrDidConfig {
                        bootstrap_reps: 19,
                        bootstrap_seed: 400 + u64::try_from(draw).expect("draw fits u64"),
                        max_iter: 1_000,
                        tol: 1e-6,
                        ..crate::types::DrDidConfig::default()
                    },
                    bootstrap_confidence_intervals: true,
                    cross_fit_folds: 4,
                    cross_fit_seed: 91,
                },
            )
            .expect("cross-fit estimate");
            estimates.push(estimate);
        }

        let mean_att = estimates.iter().map(|estimate| estimate.att).sum::<f64>()
            / crate::util::usize_to_f64(draws);
        let rmse = (estimates
            .iter()
            .map(|estimate| (estimate.att - true_att).powi(2))
            .sum::<f64>()
            / crate::util::usize_to_f64(draws))
        .sqrt();
        let coverage = estimates
            .iter()
            .filter(|estimate| estimate.ci_low <= true_att && true_att <= estimate.ci_high)
            .count();
        let coverage_rate = crate::util::usize_to_f64(coverage) / crate::util::usize_to_f64(draws);

        assert!((mean_att - true_att).abs() < 0.18, "mean_att={mean_att}");
        assert!(rmse < 0.35, "rmse={rmse}");
        assert!(coverage_rate > 0.50, "coverage_rate={coverage_rate}");
    }

    #[test]
    fn did_cc_robust_outperforms_stationary_under_composition_shift_simulation() {
        let target_att = 1.75;
        let draws = 8;
        let mut rng = Stream::seeded(24_0311);
        let mut robust_abs_error = 0.0;
        let mut stationary_abs_error = 0.0;
        let mut hausman_rejections = 0usize;

        for draw in 0..draws {
            let rows = simulate_did_cc_sample(&mut rng, 100, true);
            let config = DidCcConfig {
                drdid: crate::types::DrDidConfig {
                    bootstrap_reps: 19,
                    bootstrap_seed: 700 + u64::try_from(draw).expect("draw fits u64"),
                    max_iter: 1_000,
                    tol: 1e-6,
                    ..crate::types::DrDidConfig::default()
                },
                bootstrap_confidence_intervals: true,
                cross_fit_folds: 4,
                cross_fit_seed: 17,
            };
            let robust = robust(&rows, config).expect("robust estimate");
            let stationary = stationary(&rows, config).expect("stationary estimate");
            let hausman = hausman(&rows, config).expect("hausman");

            robust_abs_error += (robust.att - target_att).abs();
            stationary_abs_error += (stationary.att - target_att).abs();
            hausman_rejections += usize::from(hausman.p_value < 0.05);
        }

        let mean_robust_abs_error = robust_abs_error / crate::util::usize_to_f64(draws);
        let mean_stationary_abs_error = stationary_abs_error / crate::util::usize_to_f64(draws);
        let rejection_rate =
            crate::util::usize_to_f64(hausman_rejections) / crate::util::usize_to_f64(draws);

        assert!(
            mean_robust_abs_error + 0.05 < mean_stationary_abs_error,
            "robust={mean_robust_abs_error} stationary={mean_stationary_abs_error}"
        );
        assert!(rejection_rate > 0.35, "rejection_rate={rejection_rate}");
    }

    #[test]
    fn did_cc_can_skip_bootstrap_confidence_intervals() {
        let rows = simulate_did_cc_sample(&mut Stream::seeded(24_0312), 40, false);
        let estimate = robust(
            &rows,
            DidCcConfig {
                drdid: crate::types::DrDidConfig {
                    bootstrap_reps: 0,
                    max_iter: 1_000,
                    tol: 1e-6,
                    ..crate::types::DrDidConfig::default()
                },
                bootstrap_confidence_intervals: false,
                cross_fit_folds: 4,
                cross_fit_seed: 33,
            },
        )
        .expect("estimate without bootstrap ci");

        assert!(estimate.se.is_finite());
        assert!(estimate.ci_low.is_finite());
        assert!(estimate.ci_high.is_finite());
        assert!(estimate.ci_low <= estimate.att && estimate.att <= estimate.ci_high);
    }
}
