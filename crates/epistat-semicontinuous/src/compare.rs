//! Comparison, cross-validation and selection of the cost models.
//!
//! [`compare`] fits the two-part model, the Tweedie model and the log-normal
//! model to one sample and reports, for each, in-sample metrics, information
//! criteria and K-fold cross-validated metrics, with the Tweedie powers ranked
//! on their own. [`cross_validate`] runs the cross-validation alone and
//! [`select`] fits a grid of Tweedie powers and the log-normal model, with the
//! classic Park test. Each names the models by [`Candidate`].
//!
//! # Metrics
//!
//! For outcomes `y_i` and predicted means `m_i` on `n` rows: the RMSE and MAE,
//! the RMSLE `sqrt(mean((ln(1 + y) - ln(1 + m))^2))`, `r2 = 1 - SSE / SST`,
//! and a deviance, the Tweedie deviance for a Tweedie model and the mean
//! squared error otherwise. `r2` is `None` when the outcome does not vary,
//! where `SST` is 0 and the ratio is undefined.
//!
//! The cross-validated metrics are the means of the fold metrics, except
//! `r2`, which is the mean over the folds where it is defined and `None` when
//! no fold has one, as caret's `na.rm = TRUE`. A fold whose held-out outcome
//! is constant, as a fold of zero costs can be, has no `r2`; counting it as 0
//! would pull the mean toward 0 and could let a fit whose other folds failed
//! pass the ranking guard. [`CrossValidation::constant_folds`] and
//! [`ModelComparison::constant_folds`] count those folds.
//!
//! # Information criteria
//!
//! `AIC = 2k - 2 loglik` and `BIC = k ln n - 2 loglik`, with `k` the number of
//! coefficients (twice the columns for the two-part model). The Tweedie
//! log-likelihood is the quasi-likelihood `-D / 2`, the log-normal one is that
//! of the positive outcomes, and the two-part one is the binomial plus the
//! gamma log-likelihood at the moment estimate of the dispersion.
//!
//! # Weights
//!
//! The two-part and Tweedie fits use the weights and clusters of the data.
//! The log-normal model has no prior weights, so [`compare`] and
//! [`cross_validate`] fit it to the data without their weights. [`select`]
//! fits unweighted models and refuses weighted data.

use epistat_core::check;
use epistat_linalg::{Mat, MatRef};
use epistat_rng::Stream;
use epistat_stats::likelihood;
use rayon::prelude::*;

use crate::data::{CostData, column};
use crate::fit::Model;
use crate::lognormal::{
    LogNormalModel, LogNormalOptions, fit_lognormal, log_likelihood as lognormal_log_likelihood,
};
use crate::matrix_ops::{select_rows_ref, select_values_ref};
use crate::tweedie::{
    TweedieModel, TweedieOptions, deviance as tweedie_deviance, fit_tweedie,
    quasi_log_likelihood as tweedie_quasi_log_likelihood,
};
use crate::two_part::{TwoPartModel, TwoPartOptions, log_likelihood as two_part_log_likelihood};
use crate::{Error, lognormal, tweedie, two_part};

/// A model the comparison fits.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Candidate {
    /// The two-part model.
    TwoPart,
    /// The two-part model with the elastic-net options.
    TwoPartElasticNet,
    /// The Tweedie model at a power.
    Tweedie {
        /// The power.
        power: f64,
    },
    /// The log-normal model with smearing.
    LogNormal,
}

impl Candidate {
    /// A short label: `"two_part"`, `"two_part_elastic_net"`, `"tweedie p=1.5"`
    /// with the power to one decimal, or `"lognormal"`.
    #[must_use]
    pub fn label(&self) -> String {
        match self {
            Self::TwoPart => "two_part".to_string(),
            Self::TwoPartElasticNet => "two_part_elastic_net".to_string(),
            Self::Tweedie { power } => format!("tweedie p={power:.1}"),
            Self::LogNormal => "lognormal".to_string(),
        }
    }
}

/// Goodness-of-fit metrics, as the [module](self) defines them.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelFitMetrics {
    /// Root mean squared error.
    pub rmse: f64,
    /// Mean absolute error.
    pub mae: f64,
    /// Root mean squared log error (log1p scale).
    pub rmsle: f64,
    /// Coefficient of determination, `None` when the outcome does not vary,
    /// where it is undefined. Cross-validated, the mean over the folds where
    /// it is defined.
    pub r2: Option<f64>,
    /// The Tweedie deviance, or the mean squared error.
    pub deviance: f64,
}

/// The information criteria of a fitted model.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InformationCriteria {
    /// The log-likelihood.
    pub loglik: f64,
    /// Akaike's information criterion, `2k - 2 loglik`.
    pub aic: f64,
    /// The Bayesian information criterion, `k ln n - 2 loglik`.
    pub bic: f64,
}

/// The metrics of one model.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ModelScore {
    /// The model.
    pub candidate: Candidate,
    /// The fit metrics.
    pub metrics: ModelFitMetrics,
}

/// The information criteria of one model.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct ModelInformationCriteria {
    /// The model.
    pub candidate: Candidate,
    /// The log-likelihood, a quasi-likelihood for Tweedie.
    pub loglik: f64,
    /// Akaike's information criterion.
    pub aic: f64,
    /// The Bayesian information criterion.
    pub bic: f64,
}

/// A Tweedie power with its cross-validated metrics and in-sample criteria.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct TweedieRankingRow {
    /// The Tweedie power.
    pub power: f64,
    /// The cross-validated metrics.
    pub metrics: ModelFitMetrics,
    /// Akaike's information criterion in the full sample, `None` when the
    /// full-sample fit at this power failed.
    pub aic: Option<f64>,
    /// The Bayesian information criterion in the full sample, as `aic`.
    pub bic: Option<f64>,
}

/// A Tweedie fit at one power with its in-sample metrics.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TweedieCandidate {
    /// The Tweedie power.
    pub power: f64,
    /// The fitted model.
    pub model: TweedieModel,
    /// In-sample metrics.
    pub metrics: ModelFitMetrics,
    /// Information criteria from the quasi-likelihood.
    pub information_criteria: InformationCriteria,
}

/// The log-normal fit with its in-sample metrics.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct LogNormalCandidate {
    /// The fitted model.
    pub model: LogNormalModel,
    /// In-sample metrics.
    pub metrics: ModelFitMetrics,
    /// Information criteria on the positive outcomes.
    pub information_criteria: InformationCriteria,
}

/// The cross-validated metrics of one Tweedie power.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct TweedieCvCandidate {
    /// The Tweedie power.
    pub power: f64,
    /// Cross-validated metrics.
    pub metrics: ModelFitMetrics,
}

/// The models [`compare`] fitted to the full sample.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct ComparisonModels {
    /// The two-part model.
    pub two_part: Option<TwoPartModel>,
    /// The two-part model with the elastic-net options, when they were given.
    pub two_part_elastic_net: Option<TwoPartModel>,
    /// The Tweedie model at the comparison's power.
    pub tweedie: Option<TweedieModel>,
    /// The log-normal model.
    pub lognormal: Option<LogNormalModel>,
}

/// The options of [`compare`].
///
/// No default: the seed of the fold assignment is the caller's.
#[derive(Debug, Clone, PartialEq)]
pub struct CompareOptions {
    /// The Tweedie powers to fit. The comparison's Tweedie model takes 1.5
    /// when listed, else the first.
    pub tweedie_powers: Vec<f64>,
    /// Options of the two-part fit.
    pub two_part: TwoPartOptions,
    /// Options of a second two-part fit with an elastic-net penalty.
    pub two_part_elastic_net: Option<TwoPartOptions>,
    /// Options of the Tweedie fits at the comparison's power and in the
    /// cross-validation.
    pub tweedie: TweedieOptions,
    /// Options of the log-normal fit.
    pub lognormal: LogNormalOptions,
    /// The number of folds of the cross-validation.
    pub folds: usize,
    /// The seed of the fold assignment.
    pub seed: u64,
    /// Whether to fit the two-part model.
    pub include_two_part: bool,
    /// Whether to fit the Tweedie model.
    pub include_tweedie: bool,
    /// Whether to fit the log-normal model.
    pub include_lognormal: bool,
}

impl CompareOptions {
    /// Powers 1.2, 1.5 and 1.8, default fits, five folds assigned from `seed`,
    /// every model included.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            tweedie_powers: vec![1.2, 1.5, 1.8],
            two_part: TwoPartOptions::default(),
            two_part_elastic_net: None,
            tweedie: TweedieOptions::default(),
            lognormal: LogNormalOptions::default(),
            folds: 5,
            seed,
            include_two_part: true,
            include_tweedie: true,
            include_lognormal: true,
        }
    }
}

/// The result of [`compare`].
#[non_exhaustive]
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
    /// The models fitted to the full sample.
    pub models: ComparisonModels,
    /// The number of cross-validation folds whose held-out outcome is
    /// constant, which have no `r2` and are left out of its mean.
    pub constant_folds: usize,
}

/// The options of [`cross_validate`].
///
/// No default: the seed of the fold assignment is the caller's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossValidationOptions {
    /// The number of folds, at least 2 and at most the number of rows.
    pub folds: usize,
    /// The seed of the fold assignment.
    pub seed: u64,
    /// Options of the two-part fits.
    pub two_part: TwoPartOptions,
    /// Options of the Tweedie fits.
    pub tweedie: TweedieOptions,
    /// Options of the log-normal fits.
    pub lognormal: LogNormalOptions,
    /// Whether to cross-validate the two-part model.
    pub include_two_part: bool,
    /// Whether to cross-validate the log-normal model.
    pub include_lognormal: bool,
}

impl CrossValidationOptions {
    /// Five folds assigned from `seed`, default fits, every model included.
    #[must_use]
    pub fn new(seed: u64) -> Self {
        Self {
            folds: 5,
            seed,
            two_part: TwoPartOptions::default(),
            tweedie: TweedieOptions::default(),
            lognormal: LogNormalOptions::default(),
            include_two_part: true,
            include_lognormal: true,
        }
    }
}

/// The result of [`cross_validate`].
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct CrossValidation {
    /// The number of folds.
    pub folds_used: usize,
    /// The number of folds whose held-out outcome is constant, which have no
    /// `r2` and are left out of its mean.
    pub constant_folds: usize,
    /// Cross-validated metrics of the two-part model, when included.
    pub two_part_metrics: Option<ModelFitMetrics>,
    /// Cross-validated metrics of each Tweedie power.
    pub tweedie_candidates: Vec<TweedieCvCandidate>,
    /// Cross-validated metrics of the log-normal model, when included.
    pub lognormal_metrics: Option<ModelFitMetrics>,
}

impl CrossValidation {
    /// The model with the smallest cross-validated RMSE, the first in the
    /// order two-part, Tweedie powers, log-normal on a tie; `None` when no
    /// RMSE is below infinity.
    #[must_use]
    pub fn best(&self) -> Option<Candidate> {
        let mut best: Option<Candidate> = None;
        let mut best_rmse = f64::INFINITY;
        if let Some(metrics) = &self.two_part_metrics
            && metrics.rmse < best_rmse
        {
            best_rmse = metrics.rmse;
            best = Some(Candidate::TwoPart);
        }
        for candidate in &self.tweedie_candidates {
            if candidate.metrics.rmse < best_rmse {
                best_rmse = candidate.metrics.rmse;
                best = Some(Candidate::Tweedie {
                    power: candidate.power,
                });
            }
        }
        if let Some(metrics) = &self.lognormal_metrics
            && metrics.rmse < best_rmse
        {
            best = Some(Candidate::LogNormal);
        }
        best
    }
}

/// The classic Park test: the slope of `ln (y - mu)^2` on `ln mu` from a
/// gamma fit of the positive outcomes, an estimate of the power of the mean in
/// the variance.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParkTest {
    /// The intercept of the regression.
    pub intercept: f64,
    /// The slope, the estimated power.
    pub slope: f64,
}

/// The result of [`select`].
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct Selection {
    /// The classic Park test, `None` when its gamma fit or regression fails.
    pub park_test: Option<ParkTest>,
    /// The Tweedie fits that succeeded, one per power.
    pub tweedie_candidates: Vec<TweedieCandidate>,
    /// The log-normal fit, if it succeeded.
    pub lognormal_candidate: Option<LogNormalCandidate>,
    /// The candidate with the smallest AIC, the first on a tie.
    pub recommended_by_aic: Option<Candidate>,
    /// The candidate with the smallest BIC, the first on a tie.
    pub recommended_by_bic: Option<Candidate>,
}

/// Compare the two-part, Tweedie and log-normal models on `data`.
///
/// # Errors
///
/// [`Error::Input`] for a Tweedie power that is not finite; the error of a
/// full-sample fit or of a fold.
pub fn compare(data: &CostData<'_>, options: &CompareOptions) -> Result<ModelComparison, Error> {
    let x = data.x();
    let y = data.y_column();
    let two_part_default = options
        .include_two_part
        .then(|| two_part::fit(data, &options.two_part, None))
        .transpose()?;
    let two_part_elastic_net = options
        .include_two_part
        .then_some(options.two_part_elastic_net)
        .flatten()
        .map(|elastic_net_options| two_part::fit(data, &elastic_net_options, None))
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
        .then(|| tweedie::fit(data, tweedie_power, &options.tweedie, None, None))
        .transpose()?;
    let lognormal_model = options
        .include_lognormal
        .then(|| lognormal::fit(&data.without_weights(), &options.lognormal))
        .transpose()?;

    let tweedie_candidates = tweedie_candidates(data, &options.tweedie_powers)?;

    let predictions = InSamplePredictions {
        two_part: two_part_default
            .as_ref()
            .map(|m| m.predict(x))
            .transpose()?,
        two_part_elastic_net: two_part_elastic_net
            .as_ref()
            .map(|m| m.predict(x))
            .transpose()?,
        tweedie: tweedie_model.as_ref().map(|m| m.predict(x)).transpose()?,
        lognormal: lognormal_model.as_ref().map(|m| m.predict(x)).transpose()?,
    };
    let in_sample = build_in_sample(y, &predictions, tweedie_model.as_ref())?;
    let information_criteria = build_information_criteria(
        x,
        y,
        &predictions,
        tweedie_model.as_ref(),
        lognormal_model.as_ref(),
    )?;
    let tweedie_ic = tweedie_candidates
        .iter()
        .map(|candidate| ModelInformationCriteria {
            candidate: Candidate::Tweedie {
                power: candidate.power,
            },
            loglik: candidate.information_criteria.loglik,
            aic: candidate.information_criteria.aic,
            bic: candidate.information_criteria.bic,
        })
        .collect();

    let cv_default = cross_validate(
        data,
        &options.tweedie_powers,
        &CrossValidationOptions {
            folds: options.folds,
            seed: options.seed,
            two_part: options.two_part,
            tweedie: options.tweedie,
            lognormal: options.lognormal,
            include_two_part: options.include_two_part,
            include_lognormal: options.include_lognormal,
        },
    )?;
    let cv_elastic_net = options
        .two_part_elastic_net
        .map(|elastic_net_options| {
            cross_validate(
                data,
                &[],
                &CrossValidationOptions {
                    folds: options.folds,
                    seed: options.seed,
                    two_part: elastic_net_options,
                    tweedie: options.tweedie,
                    lognormal: options.lognormal,
                    include_two_part: true,
                    include_lognormal: options.include_lognormal,
                },
            )
        })
        .transpose()?;
    let cv_summary = build_cv_summary(&cv_default, cv_elastic_net.as_ref());
    let cv_ranking = build_cv_ranking(&cv_summary);
    let tweedie_cv_ranking = build_tweedie_cv_ranking(&cv_default, &tweedie_candidates);

    Ok(ModelComparison {
        in_sample,
        information_criteria,
        cv_summary,
        cv_ranking,
        tweedie_ic,
        tweedie_cv_ranking,
        models: ComparisonModels {
            two_part: two_part_default,
            two_part_elastic_net,
            tweedie: tweedie_model,
            lognormal: lognormal_model,
        },
        constant_folds: cv_default.constant_folds,
    })
}

/// K-fold cross-validation of the two-part model, the Tweedie model at each
/// of `powers` and the log-normal model.
///
/// The rows are shuffled by an [`epistat_rng::Stream`] seeded with
/// `options.seed` and cut into `folds` consecutive folds of
/// `ceil(n / folds)` rows, the last shorter. Each model is fitted to the
/// other folds and scored on the fold; the folds run in parallel and are
/// averaged in fold order. A fold whose held-out outcome is constant has no
/// `r2`; the cross-validated `r2` is the mean over the other folds.
///
/// # Errors
///
/// [`Error::Input`] for fewer than 2 folds or more folds than rows, or a power
/// that is not finite; the error of a fold's fit.
pub fn cross_validate(
    data: &CostData<'_>,
    powers: &[f64],
    options: &CrossValidationOptions,
) -> Result<CrossValidation, Error> {
    check::finite("powers", powers)?;
    let n = data.n_rows();
    if options.folds < 2 || options.folds > n {
        return Err(crate::error::domain(
            "folds",
            options.folds as f64,
            "from 2 to the number of rows",
        ));
    }
    let options = *options;

    let mut indices = (0..n).collect::<Vec<_>>();
    let mut rng = Stream::seeded(options.seed);
    rng.shuffle(&mut indices);

    let fold_size = n.div_ceil(options.folds);
    let fold_bounds = (0..options.folds)
        .filter_map(|fold| {
            let start = fold * fold_size;
            let end = (start + fold_size).min(n);
            (start < end).then_some((start, end))
        })
        .collect::<Vec<_>>();

    let fold_metrics = fold_bounds
        .par_iter()
        .map(|&(start, end)| -> Result<FoldMetrics, Error> {
            let test_idx = &indices[start..end];
            let train_idx = complement_indices(n, test_idx);
            let train = Subset::new(data, &train_idx);
            let test = Subset::new(data, test_idx);
            let train_data = train.data();

            let mut two_part = zero_metrics();
            let mut lognormal = zero_metrics();
            let mut tweedie = powers.iter().map(|_| zero_metrics()).collect::<Vec<_>>();
            let constant = is_constant(test.y());

            if options.include_two_part {
                let fitted = two_part::fit(&train_data, &options.two_part, None)?;
                let prediction = fitted.predict(test.x.as_ref())?;
                two_part = compute_model_fit_metrics(test.y(), &prediction.expected_outcome, None)?;
            }

            for (pos, power) in powers.iter().copied().enumerate() {
                let model = tweedie::fit(&train_data, power, &options.tweedie, None, None)?;
                let prediction = model.predict(test.x.as_ref())?;
                tweedie[pos] = compute_model_fit_metrics(test.y(), &prediction.mean, Some(power))?;
            }

            if options.include_lognormal {
                let model = lognormal::fit(&train_data.without_weights(), &options.lognormal)?;
                let prediction = model.predict(test.x.as_ref())?;
                lognormal = compute_model_fit_metrics(test.y(), &prediction.mean, None)?;
            }

            Ok(FoldMetrics {
                two_part,
                lognormal,
                tweedie,
                constant,
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<Result<Vec<_>, Error>>()?;

    let folds_used = fold_metrics.len();
    let constant_folds = fold_metrics.iter().filter(|fold| fold.constant).count();
    let two_part_metrics = options
        .include_two_part
        .then(|| average_folds(fold_metrics.iter().map(|fold| &fold.two_part)));
    let lognormal_metrics = options
        .include_lognormal
        .then(|| average_folds(fold_metrics.iter().map(|fold| &fold.lognormal)));
    let tweedie_candidates = powers
        .iter()
        .copied()
        .enumerate()
        .map(|(pos, power)| TweedieCvCandidate {
            power,
            metrics: average_folds(fold_metrics.iter().map(|fold| &fold.tweedie[pos])),
        })
        .collect();

    Ok(CrossValidation {
        folds_used,
        constant_folds,
        two_part_metrics,
        tweedie_candidates,
        lognormal_metrics,
    })
}

/// Fit the Tweedie model at each of `powers` and the log-normal model, all
/// unweighted with default options, and recommend one by AIC and one by BIC.
/// A power whose fit fails is left out.
///
/// ```
/// use epistat_linalg::Mat;
/// use epistat_semicontinuous::{CostData, compare};
///
/// let n = 50;
/// let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { i as f64 / 15.0 });
/// let y: Vec<f64> = (0..n).map(|i| if i % 5 == 0 { 0.0 } else { 1.5 + 0.2 * i as f64 }).collect();
/// let data = CostData::builder(x.as_ref(), &y).build()?;
/// let selection = compare::select(&data, &[1.2, 1.5])?;
/// assert_ne!(selection.tweedie_candidates.len(), 0);
/// # Ok::<(), epistat_semicontinuous::Error>(())
/// ```
///
/// # Errors
///
/// [`Error::Unsupported`] for data with weights, and [`Error::Input`] for a
/// power that is not finite.
pub fn select(data: &CostData<'_>, powers: &[f64]) -> Result<Selection, Error> {
    check::finite("powers", powers)?;
    if data.weights().is_some() {
        return Err(Error::Unsupported {
            what: "weights in the selection, whose fits are unweighted",
        });
    }
    let x = data.x();
    let y = data.y_column();
    let park_test = classic_park_test(x, y);

    let mut tweedie_candidates = Vec::new();
    for &power in powers {
        if let Ok(model) = fit_tweedie(x, y, None, power, TweedieOptions::default()) {
            let pred = model.predict(x)?;
            let metrics = compute_model_fit_metrics(y, &pred.mean, Some(power))?;
            let ll = tweedie_quasi_log_likelihood(y, &pred.mean, power);
            let ic = compute_information_criteria(ll, x.ncols(), y.nrows())?;
            tweedie_candidates.push(TweedieCandidate {
                power,
                model,
                metrics,
                information_criteria: ic,
            });
        }
    }

    let lognormal_candidate = match fit_lognormal(x, y, None, LogNormalOptions::default()) {
        Ok(model) => {
            let pred = model.predict(x)?;
            let metrics = compute_model_fit_metrics(y, &pred.mean, None)?;
            let ll = lognormal_log_likelihood(x, y, &model);
            let ic = compute_information_criteria(ll, x.ncols(), y.nrows())?;
            Some(LogNormalCandidate {
                model,
                metrics,
                information_criteria: ic,
            })
        }
        Err(_) => None,
    };

    let recommended = recommend_by_ic(&tweedie_candidates, lognormal_candidate.as_ref());

    Ok(Selection {
        park_test,
        tweedie_candidates,
        lognormal_candidate,
        recommended_by_aic: recommended.0,
        recommended_by_bic: recommended.1,
    })
}

/// Park's (1966) test of the variance function: a gamma fit of the positive outcomes
/// on `x` with a log link, then the least-squares regression of `ln (y - mu)^2` on
/// `ln mu`. The regression is `epistat_glm::classic_park_test`. It is biased where
/// Manning and Mullahy's (2001) modified test, `epistat_glm::modified_park_test`, is
/// not, and is kept for comparison.
fn classic_park_test(x: MatRef<'_, f64>, y: MatRef<'_, f64>) -> Option<ParkTest> {
    let positive_indices: Vec<usize> = (0..y.nrows()).filter(|&idx| y[(idx, 0)] > 0.0).collect();
    if positive_indices.is_empty() {
        return None;
    }
    let x_pos = select_rows_ref(x, &positive_indices);
    let y_pos = select_values_ref(y, &positive_indices);

    let gamma = fit_tweedie(
        x_pos.as_ref(),
        y_pos.as_ref(),
        None,
        2.0,
        TweedieOptions::default(),
    )
    .ok()?;
    let mu = gamma.predict(x_pos.as_ref()).ok()?.mean;
    let fitted: Vec<f64> = (0..mu.nrows()).map(|i| mu[(i, 0)]).collect();
    let outcome: Vec<f64> = (0..y_pos.nrows()).map(|i| y_pos[(i, 0)]).collect();
    let test = epistat_glm::park::classic_park_test(&fitted, &outcome).ok()??;

    Some(ParkTest {
        intercept: test.intercept,
        slope: test.power,
    })
}

/// Tweedie fits of `data` at each power with default options, with in-sample
/// metrics and quasi-likelihood criteria; a power whose fit fails is left out,
/// and a power that is not finite is an input error.
fn tweedie_candidates(data: &CostData<'_>, powers: &[f64]) -> Result<Vec<TweedieCandidate>, Error> {
    check::finite("powers", powers)?;
    let (x, y) = (data.x(), data.y_column());
    let mut candidates = Vec::new();
    for &power in powers {
        let Ok(model) = tweedie::fit(data, power, &TweedieOptions::default(), None, None) else {
            continue;
        };
        let pred = model.predict(x)?;
        let metrics = compute_model_fit_metrics(y, &pred.mean, Some(power))?;
        let ll = tweedie_quasi_log_likelihood(y, &pred.mean, power);
        let information_criteria = compute_information_criteria(ll, x.ncols(), y.nrows())?;
        candidates.push(TweedieCandidate {
            power,
            model,
            metrics,
            information_criteria,
        });
    }
    Ok(candidates)
}

/// The metrics of the [module](self) for `y` and the predictions `mu`.
///
/// # Errors
///
/// [`InputError::Empty`] for no rows and [`InputError::Length`] for
/// predictions of another length, where the metrics are undefined.
fn compute_model_fit_metrics(
    y: MatRef<'_, f64>,
    mu: &Mat<f64>,
    tweedie_power: Option<f64>,
) -> Result<ModelFitMetrics, Error> {
    check::same_length("predictions", y.nrows(), mu.nrows())?;
    check::non_empty("outcome", y.nrows())?;
    let mut mse = 0.0;
    let mut mae = 0.0;
    let mut log_error_sq_sum = 0.0;
    let mut sst = 0.0;
    let mut mean_y = 0.0;
    let denom = f64::from(u32::try_from(y.nrows()).unwrap_or(u32::MAX));
    for i in 0..y.nrows() {
        mean_y += y[(i, 0)];
    }
    mean_y /= denom.max(1.0);

    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        let mui = mu[(i, 0)];
        let diff = yi - mui;
        mse = diff.mul_add(diff, mse);
        mae += diff.abs();
        let log_diff = yi.max(0.0).ln_1p() - mui.max(0.0).ln_1p();
        log_error_sq_sum += log_diff * log_diff;
        let dev = yi - mean_y;
        sst += dev * dev;
    }
    let rmse = (mse / denom).sqrt();
    let mae = mae / denom;
    let rms_log_error = (log_error_sq_sum / denom).sqrt();
    let r2 = (!is_constant(y) && sst > 0.0).then(|| 1.0 - (mse / sst));
    let deviance = tweedie_power.map_or(rmse * rmse, |power| tweedie_deviance(y, mu, power));

    Ok(ModelFitMetrics {
        rmse,
        mae,
        rmsle: rms_log_error,
        r2,
        deviance,
    })
}

/// AIC and BIC for a log-likelihood, parameter count and sample size.
///
/// # Errors
///
/// [`InputError::Empty`] for no rows, where the BIC is undefined.
fn compute_information_criteria(
    loglik: f64,
    k: usize,
    n: usize,
) -> Result<InformationCriteria, Error> {
    check::non_empty("outcome", n)?;
    Ok(InformationCriteria {
        loglik,
        aic: likelihood::aic(loglik, k),
        bic: likelihood::bic(loglik, k, n),
    })
}

#[derive(Debug, Clone, Default)]
struct InSamplePredictions {
    two_part: Option<two_part::TwoPartPrediction>,
    two_part_elastic_net: Option<two_part::TwoPartPrediction>,
    tweedie: Option<tweedie::TweediePrediction>,
    lognormal: Option<lognormal::LogNormalPrediction>,
}

fn build_in_sample(
    y: MatRef<'_, f64>,
    predictions: &InSamplePredictions,
    tweedie: Option<&TweedieModel>,
) -> Result<Vec<ModelScore>, Error> {
    let mut scores = Vec::new();
    if let Some(pred) = predictions.two_part.as_ref() {
        scores.push(ModelScore {
            candidate: Candidate::TwoPart,
            metrics: compute_model_fit_metrics(y, &pred.expected_outcome, None)?,
        });
    }
    if let Some(pred) = predictions.two_part_elastic_net.as_ref() {
        scores.push(ModelScore {
            candidate: Candidate::TwoPartElasticNet,
            metrics: compute_model_fit_metrics(y, &pred.expected_outcome, None)?,
        });
    }
    if let (Some(tweedie), Some(pred)) = (tweedie, predictions.tweedie.as_ref()) {
        scores.push(ModelScore {
            candidate: Candidate::Tweedie {
                power: tweedie.power,
            },
            metrics: compute_model_fit_metrics(y, &pred.mean, Some(tweedie.power))?,
        });
    }
    if let Some(pred) = predictions.lognormal.as_ref() {
        scores.push(ModelScore {
            candidate: Candidate::LogNormal,
            metrics: compute_model_fit_metrics(y, &pred.mean, None)?,
        });
    }
    Ok(scores)
}

fn build_information_criteria(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    predictions: &InSamplePredictions,
    tweedie: Option<&TweedieModel>,
    lognormal: Option<&LogNormalModel>,
) -> Result<Vec<ModelInformationCriteria>, Error> {
    let row = |candidate, ic: InformationCriteria| ModelInformationCriteria {
        candidate,
        loglik: ic.loglik,
        aic: ic.aic,
        bic: ic.bic,
    };
    let mut rows = Vec::new();
    for (candidate, pred) in [
        (Candidate::TwoPart, predictions.two_part.as_ref()),
        (
            Candidate::TwoPartElasticNet,
            predictions.two_part_elastic_net.as_ref(),
        ),
    ] {
        if let Some(pred) = pred {
            let ll = two_part_log_likelihood(y, &pred.prob_positive, &pred.mean_positive);
            let ic = compute_information_criteria(ll, 2 * x.ncols(), y.nrows())?;
            rows.push(row(candidate, ic));
        }
    }
    if let (Some(tweedie), Some(pred)) = (tweedie, predictions.tweedie.as_ref()) {
        let ll = tweedie_quasi_log_likelihood(y, &pred.mean, tweedie.power);
        let ic = compute_information_criteria(ll, x.ncols(), y.nrows())?;
        rows.push(row(
            Candidate::Tweedie {
                power: tweedie.power,
            },
            ic,
        ));
    }
    if let Some(lognormal) = lognormal {
        let ll = lognormal_log_likelihood(x, y, lognormal);
        if ll.is_finite() {
            let ic = compute_information_criteria(ll, x.ncols(), y.nrows())?;
            rows.push(row(Candidate::LogNormal, ic));
        }
    }
    Ok(rows)
}

fn build_cv_summary(
    cv_default: &CrossValidation,
    cv_elastic_net: Option<&CrossValidation>,
) -> Vec<ModelScore> {
    let mut rows = Vec::new();
    if let Some(metrics) = cv_default.two_part_metrics {
        rows.push(ModelScore {
            candidate: Candidate::TwoPart,
            metrics,
        });
    }
    if let Some(metrics) = cv_elastic_net.and_then(|cv| cv.two_part_metrics) {
        rows.push(ModelScore {
            candidate: Candidate::TwoPartElasticNet,
            metrics,
        });
    }
    for candidate in &cv_default.tweedie_candidates {
        rows.push(ModelScore {
            candidate: Candidate::Tweedie {
                power: candidate.power,
            },
            metrics: candidate.metrics,
        });
    }
    if let Some(metrics) = cv_default.lognormal_metrics {
        rows.push(ModelScore {
            candidate: Candidate::LogNormal,
            metrics,
        });
    }
    rows
}

/// How far past the mean predictor a CV score may fall and still be a score.
///
/// Something is implausible once it exceeds its reference scale by 50x. Here the
/// reference is the trivial mean predictor, whose RMSE is the outcome's own spread and
/// whose r2 is 0 by definition, so `rmse > 50 * rmse_mean` is exactly `r2 < -2499`.
/// Without an r2, where every fold's outcome was constant, there is no spread to
/// compare with and only the RMSE being finite is judged.
const CV_MAGNITUDE_GUARD_K: f64 = 50.0;

/// Whether a cross-validated score can be compared with the others at all.
///
/// A model that generalises badly still has a score: `two_part` at r2 = -0.06
/// and a Tweedie at r2 = -3.7 are poor fits, and ranking them is meaningful.
/// A model whose held-out RMSE is ten orders of magnitude above the outcome
/// scale has not produced a score, it has failed, and putting it in a ranked
/// list says it came last when what happened is that it did not run.
///
/// The r2 judged is the mean over the folds where it is defined. A constant
/// fold counted as r2 = 0 would pull that mean toward 0 and let a failed fit
/// pass, which is why such a fold has no r2.
fn cv_score_is_comparable(metrics: &ModelFitMetrics) -> bool {
    metrics.rmse.is_finite()
        && metrics.r2.is_none_or(|r2| {
            r2.is_finite() && r2 >= 1.0 - CV_MAGNITUDE_GUARD_K * CV_MAGNITUDE_GUARD_K
        })
}

/// Comparable scores first, ordered by RMSE; failures after them, and within
/// each group ties broken by label so the order does not depend on the order the
/// candidates happened to be built in.
fn cv_order(a: &ModelScore, b: &ModelScore) -> std::cmp::Ordering {
    cv_score_is_comparable(&b.metrics)
        .cmp(&cv_score_is_comparable(&a.metrics))
        .then_with(|| a.metrics.rmse.total_cmp(&b.metrics.rmse))
        .then_with(|| a.candidate.label().cmp(&b.candidate.label()))
}

fn build_cv_ranking(cv_summary: &[ModelScore]) -> Vec<ModelScore> {
    let mut rows = cv_summary.to_vec();
    rows.sort_by(cv_order);
    rows
}

fn build_tweedie_cv_ranking(
    cv_default: &CrossValidation,
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
            TweedieRankingRow {
                power: candidate.power,
                metrics: candidate.metrics,
                aic: in_sample.map(|entry| entry.information_criteria.aic),
                bic: in_sample.map(|entry| entry.information_criteria.bic),
            }
        })
        .collect()
}

#[derive(Clone)]
struct FoldMetrics {
    two_part: ModelFitMetrics,
    lognormal: ModelFitMetrics,
    tweedie: Vec<ModelFitMetrics>,
    /// Whether the held-out outcome is constant.
    constant: bool,
}

/// The rows `indices` of the data, copied.
struct Subset {
    x: Mat<f64>,
    y: Vec<f64>,
    weights: Option<Vec<f64>>,
    clusters: Option<Vec<u32>>,
}

impl Subset {
    fn new(data: &CostData<'_>, indices: &[usize]) -> Self {
        let y = data.y();
        Self {
            x: select_rows_ref(data.x(), indices),
            y: indices.iter().map(|&i| y[i]).collect(),
            weights: data
                .weights()
                .map(|w| indices.iter().map(|&i| w[i]).collect()),
            clusters: data
                .clusters()
                .map(|c| indices.iter().map(|&i| c[i]).collect()),
        }
    }

    fn y(&self) -> MatRef<'_, f64> {
        column(&self.y)
    }

    /// The rows as data. Rows of checked data are checked, so no check runs.
    fn data(&self) -> CostData<'_> {
        CostData::from_checked(
            self.x.as_ref(),
            &self.y,
            self.weights.as_deref(),
            self.clusters.as_deref(),
        )
    }
}

const fn zero_metrics() -> ModelFitMetrics {
    ModelFitMetrics {
        rmse: 0.0,
        mae: 0.0,
        rmsle: 0.0,
        r2: None,
        deviance: 0.0,
    }
}

/// Whether every value of the outcome equals the first, so it does not vary.
#[expect(
    clippy::float_cmp,
    reason = "an outcome is constant when its values are exactly equal"
)]
fn is_constant(y: MatRef<'_, f64>) -> bool {
    (1..y.nrows()).all(|i| y[(i, 0)] == y[(0, 0)])
}

/// The means of the fold metrics, summed in fold order; `r2` is the mean over
/// the folds that have one, `None` when none has.
fn average_folds<'a>(folds: impl Iterator<Item = &'a ModelFitMetrics>) -> ModelFitMetrics {
    let mut sum = zero_metrics();
    let mut count = 0_u32;
    let mut r2_sum = 0.0;
    let mut r2_count = 0_u32;
    for fold in folds {
        sum.rmse += fold.rmse;
        sum.mae += fold.mae;
        sum.rmsle += fold.rmsle;
        sum.deviance += fold.deviance;
        count += 1;
        if let Some(r2) = fold.r2 {
            r2_sum += r2;
            r2_count += 1;
        }
    }
    if count > 0 {
        let folds = f64::from(count);
        sum.rmse /= folds;
        sum.mae /= folds;
        sum.rmsle /= folds;
        sum.deviance /= folds;
    }
    sum.r2 = (r2_count > 0).then(|| r2_sum / f64::from(r2_count));
    sum
}

fn complement_indices(n: usize, test_idx: &[usize]) -> Vec<usize> {
    let mut mask = vec![false; n];
    for &idx in test_idx {
        mask[idx] = true;
    }
    (0..n).filter(|&idx| !mask[idx]).collect()
}

fn recommend_by_ic(
    tweedie_candidates: &[TweedieCandidate],
    lognormal: Option<&LogNormalCandidate>,
) -> (Option<Candidate>, Option<Candidate>) {
    let mut by_akaike: Option<(f64, Candidate)> = None;
    let mut by_bayes: Option<(f64, Candidate)> = None;
    let mut consider = |aic: f64, bic: f64, candidate: Candidate| {
        by_akaike = pick_best(by_akaike, aic, candidate);
        by_bayes = pick_best(by_bayes, bic, candidate);
    };
    for candidate in tweedie_candidates {
        consider(
            candidate.information_criteria.aic,
            candidate.information_criteria.bic,
            Candidate::Tweedie {
                power: candidate.power,
            },
        );
    }
    if let Some(lognormal) = lognormal {
        consider(
            lognormal.information_criteria.aic,
            lognormal.information_criteria.bic,
            Candidate::LogNormal,
        );
    }
    (
        by_akaike.map(|(_, candidate)| candidate),
        by_bayes.map(|(_, candidate)| candidate),
    )
}

fn pick_best(
    current: Option<(f64, Candidate)>,
    value: f64,
    candidate: Candidate,
) -> Option<(f64, Candidate)> {
    match current {
        Some((best, _)) if value >= best => current,
        _ => Some((value, candidate)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::two_part::Regularization;
    use crate::utils::usize_to_f64;
    use epistat_core::InputError;

    const SEED: u64 = 42;

    struct Sample {
        x: Mat<f64>,
        y: Vec<f64>,
    }

    impl Sample {
        fn data(&self) -> CostData<'_> {
            CostData::builder(self.x.as_ref(), &self.y)
                .build()
                .expect("data")
        }
    }

    fn sample(n: usize, slope: f64, every: usize, base: f64, step: f64) -> Sample {
        Sample {
            x: Mat::from_fn(
                n,
                2,
                |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / slope },
            ),
            y: (0..n)
                .map(|i| {
                    if i % every == 0 {
                        0.0
                    } else {
                        step.mul_add(usize_to_f64(i), base)
                    }
                })
                .collect(),
        }
    }

    fn sample_input(n: usize) -> Sample {
        sample(n, 25.0, 5, 1.2, 0.15)
    }

    fn score(candidate: Candidate, rmse: f64, r2: f64) -> ModelScore {
        score_with(candidate, rmse, Some(r2))
    }

    fn score_with(candidate: Candidate, rmse: f64, r2: Option<f64>) -> ModelScore {
        ModelScore {
            candidate,
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
            score(Candidate::LogNormal, 2.7e11, -1.1e13),
            score(Candidate::Tweedie { power: 1.5 }, 276_867.0, -3.71),
            score(Candidate::TwoPartElasticNet, 149_339.0, -0.06),
            score(Candidate::TwoPart, 144_962.0, 0.009),
        ]);
        let candidates: Vec<Candidate> = ranked.iter().map(|row| row.candidate).collect();
        assert_eq!(
            candidates,
            [
                Candidate::TwoPart,
                Candidate::TwoPartElasticNet,
                Candidate::Tweedie { power: 1.5 },
                Candidate::LogNormal
            ]
        );
    }

    #[test]
    fn cv_ties_break_on_label_not_construction_order() {
        let one = build_cv_ranking(&[
            score(Candidate::TwoPart, 1.0, 0.5),
            score(Candidate::LogNormal, 1.0, 0.5),
        ]);
        let other = build_cv_ranking(&[
            score(Candidate::LogNormal, 1.0, 0.5),
            score(Candidate::TwoPart, 1.0, 0.5),
        ]);
        assert_eq!(one[0].candidate, Candidate::LogNormal);
        assert_eq!(one[0].candidate, other[0].candidate);
    }

    #[test]
    fn labels_are_the_names_of_the_models() {
        assert_eq!(Candidate::TwoPart.label(), "two_part");
        assert_eq!(Candidate::Tweedie { power: 1.5 }.label(), "tweedie p=1.5");
        assert_eq!(Candidate::LogNormal.label(), "lognormal");
    }

    #[test]
    fn compare_runs() {
        let s = sample_input(80);
        let report = compare(&s.data(), &CompareOptions::new(SEED)).expect("comparison");
        assert_ne!(report.in_sample.len(), 0);
        assert_ne!(report.information_criteria.len(), 0);
        assert_ne!(report.cv_summary.len(), 0);
        assert!(report.models.two_part.is_some());
    }

    #[test]
    fn compare_excludes_lognormal_when_disabled() {
        let s = sample_input(70);
        let options = CompareOptions {
            include_lognormal: false,
            ..CompareOptions::new(SEED)
        };
        let report = compare(&s.data(), &options).expect("comparison");
        let lognormal = |c: &Candidate| *c == Candidate::LogNormal;
        assert!(!report.in_sample.iter().any(|row| lognormal(&row.candidate)));
        assert!(
            !report
                .information_criteria
                .iter()
                .any(|row| lognormal(&row.candidate))
        );
        assert!(
            !report
                .cv_summary
                .iter()
                .any(|row| lognormal(&row.candidate))
        );
        assert!(
            !report
                .cv_ranking
                .iter()
                .any(|row| lognormal(&row.candidate))
        );
    }

    #[test]
    fn compare_excludes_two_part_when_disabled() {
        let s = sample_input(70);
        let options = CompareOptions {
            include_two_part: false,
            ..CompareOptions::new(SEED)
        };
        let report = compare(&s.data(), &options).expect("comparison");
        let two_part = |c: &Candidate| *c == Candidate::TwoPart;
        assert!(!report.in_sample.iter().any(|row| two_part(&row.candidate)));
        assert!(!report.cv_summary.iter().any(|row| two_part(&row.candidate)));
    }

    #[test]
    fn compare_handles_empty_tweedie_power_grid() {
        let s = sample_input(60);
        let options = CompareOptions {
            include_tweedie: false,
            tweedie_powers: Vec::new(),
            ..CompareOptions::new(SEED)
        };
        let report = compare(&s.data(), &options).expect("comparison");
        assert_eq!(report.tweedie_ic.len(), 0);
        assert_eq!(report.tweedie_cv_ranking.len(), 0);
        assert!(
            !report
                .in_sample
                .iter()
                .any(|row| matches!(row.candidate, Candidate::Tweedie { .. }))
        );
    }

    #[test]
    fn compare_adds_elastic_net_two_part_results() {
        let s = sample_input(80);
        let options = CompareOptions {
            two_part_elastic_net: Some(TwoPartOptions {
                regularization: Regularization::ElasticNet {
                    lambda: 0.05,
                    alpha: 0.5,
                    exclude_intercept: true,
                },
                ..TwoPartOptions::default()
            }),
            ..CompareOptions::new(SEED)
        };
        let report = compare(&s.data(), &options).expect("comparison");
        let elastic = |c: &Candidate| *c == Candidate::TwoPartElasticNet;
        assert!(report.in_sample.iter().any(|row| elastic(&row.candidate)));
        assert!(
            report
                .information_criteria
                .iter()
                .any(|row| elastic(&row.candidate))
        );
        assert!(report.cv_summary.iter().any(|row| elastic(&row.candidate)));
    }

    #[test]
    fn selection_runs() {
        let s = sample(60, 10.0, 6, 1.0, 0.2);
        let result = select(&s.data(), &[1.2, 1.5, 1.8]).expect("selection");
        assert_ne!(result.tweedie_candidates.len(), 0);
        assert!(result.park_test.expect("park").slope.is_finite());
        assert!(result.recommended_by_aic.is_some());
    }

    #[test]
    fn selection_refuses_weights() {
        let s = sample(20, 10.0, 6, 1.0, 0.2);
        let w = vec![1.0; 20];
        let data = s.data().with_weights(&w).expect("weights");
        assert!(matches!(
            select(&data, &[1.5]),
            Err(Error::Unsupported { .. })
        ));
    }

    #[test]
    fn cross_validate_runs() {
        let s = sample(50, 10.0, 7, 1.0, 0.2);
        let result =
            cross_validate(&s.data(), &[1.2, 1.5], &CrossValidationOptions::new(SEED)).expect("cv");
        assert_eq!(result.tweedie_candidates.len(), 2);
        assert!(result.folds_used > 0);
        assert!(result.best().is_some());
    }

    #[test]
    fn compute_metrics_refuses_a_dimension_mismatch() {
        let y = [0.0, 1.0];
        let mu = Mat::from_fn(3, 1, |_, _| 1.0);
        assert!(matches!(
            compute_model_fit_metrics(column(&y), &mu, None),
            Err(Error::Input(InputError::Length { .. }))
        ));
    }

    #[test]
    fn bic_is_defined_from_one_observation() {
        let ic = compute_information_criteria(-10.0, 3, 1).unwrap();
        assert!((ic.bic - 20.0).abs() < 1e-12);
        assert!(compute_information_criteria(-10.0, 3, 0).is_err());
    }

    #[test]
    fn cross_validate_rejects_invalid_fold_counts() {
        let n = 8;
        let s = sample(n, 10.0, 3, 1.0, 0.0);
        for folds in [1, n + 1] {
            let options = CrossValidationOptions {
                folds,
                ..CrossValidationOptions::new(SEED)
            };
            assert!(matches!(
                cross_validate(&s.data(), &[1.5], &options),
                Err(Error::Input(_))
            ));
        }
    }

    #[test]
    fn cross_validate_allows_empty_power_grid() {
        let s = sample(30, 10.0, 5, 1.0, 0.0);
        let result = cross_validate(&s.data(), &[], &CrossValidationOptions::new(SEED))
            .expect("cv should run without Tweedie powers");
        assert_eq!(result.tweedie_candidates.len(), 0);
    }

    #[test]
    fn cross_validate_can_skip_two_part_and_lognormal() {
        let s = sample(30, 10.0, 5, 1.0, 0.0);
        let options = CrossValidationOptions {
            include_two_part: false,
            include_lognormal: false,
            ..CrossValidationOptions::new(SEED)
        };
        let result = cross_validate(&s.data(), &[1.2], &options)
            .expect("cv should run with only Tweedie enabled");
        assert!(result.two_part_metrics.is_none());
        assert!(result.lognormal_metrics.is_none());
        assert_eq!(result.tweedie_candidates.len(), 1);
    }

    #[test]
    fn cross_validate_is_deterministic_for_fixed_seed() {
        let s = sample(40, 8.0, 6, 1.0, 0.0);
        let options = CrossValidationOptions {
            folds: 4,
            ..CrossValidationOptions::new(12345)
        };
        let first = cross_validate(&s.data(), &[1.2, 1.5], &options).expect("first run");
        let second = cross_validate(&s.data(), &[1.2, 1.5], &options).expect("second run");
        assert_eq!(first, second);
    }

    #[test]
    fn cross_validate_propagates_two_part_model_errors() {
        let n = 20;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 5.0 },
        );
        let y = vec![0.0; n];
        let data = CostData::builder(x.as_ref(), &y).build().expect("data");
        let err = cross_validate(&data, &[], &CrossValidationOptions::new(SEED))
            .expect_err("all-zero outcomes should fail two-part fitting");
        assert_eq!(err, Error::NoPositiveOutcome);
    }

    fn fold(rmse: f64, r2: Option<f64>) -> ModelFitMetrics {
        score_with(Candidate::TwoPart, rmse, r2).metrics
    }

    #[test]
    fn a_constant_outcome_has_no_r2() {
        let y = [3.0, 3.0, 3.0];
        let mu = Mat::from_fn(3, 1, |i, _| usize_to_f64(i));
        let metrics = compute_model_fit_metrics(column(&y), &mu, None).unwrap();
        assert_eq!(metrics.r2, None);
        assert!(metrics.rmse > 0.0);
    }

    #[test]
    fn the_cv_r2_is_the_mean_over_the_folds_that_have_one() {
        let mean =
            average_folds([fold(1.0, Some(0.2)), fold(2.0, None), fold(3.0, Some(0.4))].iter());
        // The RMSE averages over all three folds, the r2 over the two that have one.
        assert_eq!(mean.rmse, (1.0 + 2.0 + 3.0) / 3.0);
        assert_eq!(mean.r2, Some(f64::midpoint(0.2, 0.4)));
        let none = average_folds([fold(1.0, None), fold(2.0, None)].iter());
        assert_eq!(none.r2, None);
        assert_eq!(none.rmse, 1.5);
    }

    #[test]
    fn a_constant_fold_does_not_carry_a_failed_fit_past_the_guard() {
        // Four folds where the fit failed, with r2 = -2800 (RMSE about 53 times the
        // outcome's spread), and one constant fold. Counted as 0, the constant fold
        // would give a mean r2 of -2240, inside the guard's -2499.
        let folds = [
            fold(5.3e7, Some(-2800.0)),
            fold(5.3e7, Some(-2800.0)),
            fold(0.0, None),
            fold(5.3e7, Some(-2800.0)),
            fold(5.3e7, Some(-2800.0)),
        ];
        let mean = average_folds(folds.iter());
        assert_eq!(mean.r2, Some(-2800.0));
        assert!(!cv_score_is_comparable(&mean));
        let ranked = build_cv_ranking(&[
            score_with(Candidate::LogNormal, mean.rmse, mean.r2),
            score(Candidate::TwoPart, 1.0e8, 0.1),
        ]);
        assert_eq!(ranked[0].candidate, Candidate::TwoPart);
    }

    #[test]
    fn a_score_without_r2_is_judged_by_its_rmse() {
        assert!(cv_score_is_comparable(&fold(1.0e9, None)));
        assert!(!cv_score_is_comparable(&fold(f64::INFINITY, None)));
        assert!(!cv_score_is_comparable(&fold(f64::NAN, None)));
    }

    /// Sixty rows, the even ones zero, and the first shuffle seed from 0 whose
    /// first fold holds only zero rows and whose other folds all vary.
    #[expect(clippy::float_cmp, reason = "the zero costs are exactly 0")]
    fn sample_with_a_zero_fold(folds: usize) -> (Sample, u64) {
        let s = sample(60, 10.0, 2, 1.0, 0.2);
        let n = s.y.len();
        let size = n.div_ceil(folds);
        let seed = (0..100_000_u64)
            .find(|&seed| {
                let mut indices = (0..n).collect::<Vec<_>>();
                Stream::seeded(seed).shuffle(&mut indices);
                indices[..size].iter().all(|&i| s.y[i] == 0.0)
                    && (size..n).step_by(size).all(|start| {
                        let fold = &indices[start..(start + size).min(n)];
                        fold.iter().any(|&i| s.y[i] != s.y[fold[0]])
                    })
            })
            .expect("a seed with one zero fold");
        (s, seed)
    }

    #[test]
    fn cross_validation_counts_a_constant_fold_and_leaves_it_out_of_r2() {
        let folds = 10;
        let (s, seed) = sample_with_a_zero_fold(folds);
        let options = CrossValidationOptions {
            folds,
            include_lognormal: false,
            ..CrossValidationOptions::new(seed)
        };
        let cv = cross_validate(&s.data(), &[1.5], &options).expect("cv");
        assert_eq!(cv.constant_folds, 1);
        assert_eq!(cv.folds_used, folds);
        assert!(cv.two_part_metrics.expect("two-part").r2.is_some());
        assert!(cv.tweedie_candidates[0].metrics.r2.is_some());
        let report = compare(
            &s.data(),
            &CompareOptions {
                folds,
                include_lognormal: false,
                ..CompareOptions::new(seed)
            },
        )
        .expect("comparison");
        assert_eq!(report.constant_folds, 1);
    }

    #[test]
    fn cross_validation_without_variation_in_any_fold_has_no_r2() {
        // Positive costs of 2 everywhere: each fold, and each training set, is
        // constant, and only the Tweedie fit, which needs no zero, runs.
        let n = 12;
        let x = Mat::from_fn(n, 1, |_, _| 1.0);
        let y = vec![2.0; n];
        let data = CostData::builder(x.as_ref(), &y).build().expect("data");
        let options = CrossValidationOptions {
            folds: 3,
            include_two_part: false,
            include_lognormal: false,
            ..CrossValidationOptions::new(SEED)
        };
        let cv = cross_validate(&data, &[1.5], &options).expect("cv");
        assert_eq!(cv.constant_folds, 3);
        let metrics = cv.tweedie_candidates[0].metrics;
        assert_eq!(metrics.r2, None);
        assert!(metrics.rmse.is_finite());
        assert!(cv_score_is_comparable(&metrics));
    }
}
