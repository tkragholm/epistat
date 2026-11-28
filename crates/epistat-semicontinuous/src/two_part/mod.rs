//! The two-part model of a semicontinuous outcome.
//!
//! # The model
//!
//! The mean of an outcome `y >= 0` is split into the probability that it is
//! positive and the mean of the positive values,
//!
//! ```text
//! E[y | x] = Pr(y > 0 | x) E[y | y > 0, x]
//!          = expit(x' a) exp(x' b),
//! ```
//!
//! a logit fitted to the indicator `y > 0` on every row and a gamma regression
//! with log link fitted to the positive rows \[1, 2\]. The two likelihoods
//! separate, so each part is fitted on its own.
//!
//! # Algorithm
//!
//! Each part is IRLS with prior weights `w_i`. The logit has working weights
//! `w_i p_i (1 - p_i)`, with `p_i (1 - p_i)` floored at `min_weight`, and
//! working response `eta_i + (y_i - p_i) / (p_i (1 - p_i))`. A logit step that
//! makes the deviance worse or non-finite is halved, up to `max_step_halvings`
//! times, and the logit stops when the relative change of the deviance,
//! `|D_new - D| / (|D_new| + 0.1)`, falls below `deviance_tolerance`, as R's
//! `glm.fit` does \[3\], or when no coefficient moves by more than `tolerance`.
//! A logit whose smaller class has no more rows than the design has columns is
//! separable, and so is a constant indicator; the fit refuses both unless
//! [`Separation::InterceptOnly`] asks for the intercept-only prevalence model.
//!
//! The gamma part has working weights `w_i`, so that without an elastic net
//! `X' W X` is factorised once, and working response `eta_i + (y_i - mu_i) / mu_i`. It is fitted on a
//! design whose non-intercept columns are centred, stops when no coefficient
//! moves by more than `tolerance`, and maps its coefficients back to the design
//! as given. The linear predictors are clamped, at 40 for the logit and at
//! 30 for the gamma part, and a gamma fit that ends at
//! the clamp is reported as not converged. A ridge, an elastic-net or a normal
//! prior penalty can be added ([`Regularization`]), and
//! [`FitStrategy::Relaxed`] retries a fit
//! that fails with a growing ridge.
//!
//! The intercept-only fallback and every retry are reported in the fit's
//! [`TwoPartReport`].
//!
//! # Standard errors
//!
//! The model-based covariance of each part is the inverse of its information
//! `X' W X` at the final coefficients, scaled for the gamma part by the
//! Pearson dispersion `sum w_i ((y_i - mu_i) / mu_i)^2 / (n - p)`, as R's
//! `summary.glm` does. The sandwich is `B M B`, `B` the inverse information at
//! unit dispersion and `M` the sum of the outer products of the score
//! contributions `x_i w_i (y_i - p_i)` and `x_i w_i (y_i - mu_i) / mu_i`,
//! summed within clusters first when cluster labels are given (HC0 without
//! and CR0 with clusters) \[4\]. [`Variance`] chooses between them; by default
//! clustered data get the sandwich.
//!
//! # References
//!
//! 1. Duan N, Manning WG, Morris CN, Newhouse JP. A comparison of alternative
//!    models for the demand for medical care. J Bus Econ Stat. 1983;1:115-26.
//! 2. Mihaylova B, Briggs A, O'Hagan A, Thompson SG. Review of statistical
//!    methods for analysing healthcare resources and costs. Health Econ.
//!    2011;20:897-916.
//! 3. R Core Team. Fitting generalized linear models: `glm.fit`. Help pages of
//!    the stats package of R.
//! 4. Zeileis A. Object-oriented computation of sandwich estimators. J Stat
//!    Softw. 2006;16(9):1-16.

mod bootstrap;
mod covariance;
mod elastic_net;
pub(crate) mod irls;

pub use bootstrap::{BootstrapDraws, BootstrapSummary, bootstrap, percentile_intervals, summary};

use covariance::{covariance_gamma, covariance_logit};
use irls::{GAMMA_LOG_LINK_ETA_CLAMP, fit_gamma_log_link_weighted, fit_logit_weighted};

use crate::data::CostData;
use crate::fit::{
    AttemptDiagnostics, FitMetadata, FitStrategy, Model, SolverKind, run_with_retries,
};
use crate::matrix_ops::{select_rows_ref, select_values_ref};
use crate::tidy::{TableParts, table};
#[cfg(feature = "bench-internals")]
use crate::utils::weighted_xtz;
use crate::utils::{column_mat, column_values, linear_predictor};
use crate::variance::{
    Inference, Variance, pearson_dispersion, residual_df, standard_errors, to_covariance,
};
use crate::{Error, check_columns};
use epistat_core::{
    Coefficients, Covariance, EffectScale, Glance, Level, Tidy, VarianceKind, check,
};
use epistat_linalg::{Mat, MatRef};
use epistat_stats::dist::gamma::ln_gamma;
use std::collections::BTreeSet;

/// The penalty of the two parts.
///
/// [`fit`] refuses a penalty that is negative or not finite, an elastic-net
/// mixing share outside `[0, 1]` and a prior scale that is not positive.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Regularization {
    /// No penalty.
    None,
    /// L2 (ridge) penalty.
    Ridge {
        /// The penalty on the sum scale of the weights.
        lambda: f64,
        /// Whether the first column is left unpenalised.
        exclude_intercept: bool,
    },
    /// Elastic net penalty (L1 + L2).
    ElasticNet {
        /// The overall penalty on the sum scale of the weights.
        lambda: f64,
        /// The share of the penalty on the L1 norm, in `[0, 1]`: glmnet's
        /// `alpha`, a parameter of the penalty, not a confidence level.
        alpha: f64,
        /// Whether the first column is left unpenalised.
        exclude_intercept: bool,
    },
    /// Bayesian ridge (Normal prior), equivalent to an L2 penalty.
    BayesianRidge {
        /// The standard deviation of the normal prior, giving an L2 penalty of `1 / prior_scale^2`.
        prior_scale: f64,
        /// Whether the first column is left unpenalised.
        exclude_intercept: bool,
    },
}

impl Regularization {
    /// The penalty's parameters checked.
    fn check(self) -> Result<(), Error> {
        let non_negative = |lambda: f64| {
            if lambda.is_finite() && lambda >= 0.0 {
                Ok(())
            } else {
                Err(crate::error::domain("lambda", lambda, "[0, inf)"))
            }
        };
        match self {
            Self::None => Ok(()),
            Self::Ridge { lambda, .. } => non_negative(lambda),
            Self::ElasticNet { lambda, alpha, .. } => {
                non_negative(lambda)?;
                if !(0.0..=1.0).contains(&alpha) {
                    return Err(crate::error::domain("alpha", alpha, "[0, 1]"));
                }
                Ok(())
            }
            Self::BayesianRidge { prior_scale, .. } => {
                check::positive("prior_scale", prior_scale).map_err(Into::into)
            }
        }
    }
}

/// What [`fit`] does with a logit part that has no finite estimate.
///
/// A constant indicator has no logit to fit, and neither has one whose
/// smaller class has no more rows than the design has columns: a separating
/// hyperplane then always exists and the maximum-likelihood estimate is at
/// infinity ([`InterceptOnly`]).
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Separation {
    /// Refuse the fit with [`Error::Separation`].
    #[default]
    Refuse,
    /// Replace the logit by the intercept-only prevalence model: the
    /// intercept `logit(p)` of the weighted share `p` of positive outcomes,
    /// clamped to `[min_weight, 1 - min_weight]`, and zero slopes. The fit
    /// reports the replacement in
    /// [`TwoPartReport::logit_intercept_only`].
    InterceptOnly,
}

/// The options of [`fit`].
///
/// The defaults are those of R's `glm.control` where it has the control
/// (`epsilon` is `deviance_tolerance`, `maxit` is `max_iterations` at 50
/// rather than R's 25, since a near-separated logit takes more steps).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TwoPartOptions {
    /// The most IRLS iterations of each part.
    pub max_iterations: usize,
    /// Convergence tolerance on the largest change of a coefficient, the
    /// secondary rule.
    ///
    /// It is absolute and so scale-dependent: a logit whose coefficients sit
    /// at 1e3, as separation produces, would need ten significant digits of
    /// agreement between iterates, so on those fits the loop would exhaust
    /// `max_iterations`. `deviance_tolerance` is the primary rule.
    pub tolerance: f64,
    /// The relative change of the deviance below which the logit has
    /// converged, `|D - D_prev| / (|D| + 0.1)`, R's `epsilon` and its default.
    pub deviance_tolerance: f64,
    /// How many times a logit step may be halved when it does not lower the
    /// deviance, as `glm.fit` does.
    pub max_step_halvings: usize,
    /// Lower bound on the variance function `p (1 - p)` in the logit's working
    /// weights.
    pub min_weight: f64,
    /// The penalty.
    pub regularization: Regularization,
    /// The covariance of each part; by default the sandwich for clustered
    /// data or sampling weights and the model-based covariance otherwise.
    pub variance: Variance,
    /// What a logit without a finite estimate does; refused by default, as
    /// R's `glm` warns that fitted probabilities of 0 or 1 occurred.
    pub separation: Separation,
    /// What a fit that fails does.
    pub strategy: FitStrategy,
    #[doc(hidden)]
    pub _extensible: epistat_core::Extensible,
}

impl Default for TwoPartOptions {
    fn default() -> Self {
        Self {
            _extensible: epistat_core::Extensible::NEW,
            max_iterations: 50,
            tolerance: 1e-6,
            deviance_tolerance: 1e-8,
            max_step_halvings: 25,
            min_weight: 1e-6,
            regularization: Regularization::None,
            variance: Variance::Auto,
            separation: Separation::Refuse,
            strategy: FitStrategy::Strict,
        }
    }
}

impl TwoPartOptions {
    /// A ridge of 1e-4 on both parts, with the relaxed strategy from a ridge
    /// of 1e-3 and up to three retries.
    #[must_use]
    pub fn stable_defaults() -> Self {
        Self {
            regularization: Regularization::Ridge {
                lambda: 1e-4,
                exclude_intercept: true,
            },
            strategy: FitStrategy::Relaxed {
                fallback_lambda: 1e-3,
                max_retries: 3,
            },
            ..Self::default()
        }
    }
}

/// Why the logit part has no finite estimate.
///
/// A constant indicator has no logit to fit, and neither has one whose
/// smaller class has no more rows than the design has columns: a separating
/// hyperplane then always exists and the maximum-likelihood estimate is at
/// infinity. Rows with a weight of 0 are not counted.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterceptOnly {
    /// Every outcome is positive.
    AllPositive,
    /// No outcome is positive.
    AllZero,
    /// The smaller class has `minority` rows, no more than the `columns` of
    /// the design.
    Separable {
        /// The rows of the smaller class.
        minority: usize,
        /// The columns of the design.
        columns: usize,
    },
}

/// Which part of a two-part model a [`PartFit`] is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Part {
    Logit,
    Gamma,
}

/// One part of a two-part model: its coefficients and covariance.
///
/// [`Tidy`] gives its coefficient table: log odds ratios for the logit part,
/// log mean ratios for the gamma part. The model-based table of the gamma
/// part refers its Wald statistics to Student's t on the residual degrees of
/// freedom, as R's `summary.glm` does for a family with an estimated
/// dispersion; the sandwich tables and the logit part use the normal.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct PartFit {
    /// The coefficients, one per column of the design.
    pub coefficients: Vec<f64>,
    /// The covariance of the coefficients, `None` when the options asked for
    /// none ([`Variance::None`]).
    pub covariance: Option<Covariance>,
    /// The standard errors, the roots of the covariance's diagonal; `None`
    /// where the variance is undefined or negative, and for every coefficient
    /// without a covariance.
    pub se: Vec<Option<f64>>,
    /// The dispersion: 1 for the logit part, and for the gamma part R's
    /// Pearson estimate `sum w_i ((y_i - mu_i) / mu_i)^2 / (n - p)` on the
    /// positive rows, `None` without residual degrees of freedom. The
    /// model-based covariance of the gamma part is scaled by it.
    pub dispersion: Option<f64>,
    /// The residual degrees of freedom, the rows that entered the part (or
    /// the sum of frequency weights) less the coefficients; `None` when not
    /// positive.
    pub df_residual: Option<f64>,
    /// The rows the part was fitted to: every row for the logit part, the
    /// positive outcomes for the gamma part.
    pub n: usize,
    /// The rows among them with a positive weight, which entered the fit.
    pub n_used: usize,
    /// Fit metadata of the part.
    pub meta: FitMetadata,
    part: Part,
    variance: Option<VarianceKind>,
}

impl Tidy for PartFit {
    type Error = Error;

    fn coefficients(&self, level: Level, variance: VarianceKind) -> Result<Coefficients, Error> {
        let (effect, t_df) = match self.part {
            Part::Logit => (EffectScale::LogOddsRatio, None),
            Part::Gamma => (
                EffectScale::LogMeanRatio,
                if variance == VarianceKind::ModelBased {
                    self.df_residual
                } else {
                    None
                },
            ),
        };
        table(
            &TableParts {
                coefficients: &self.coefficients,
                covariance: self.covariance.as_ref(),
                computed: self.variance,
                effect,
                t_df,
            },
            level,
            variance,
        )
    }

    /// The covariance the fit computed, model-based when it computed none.
    fn default_variance(&self) -> VarianceKind {
        self.variance.unwrap_or(VarianceKind::ModelBased)
    }

    fn glance(&self) -> Glance {
        Glance::new(self.n, self.n_used, self.meta.converged)
            .with_iterations(Some(self.meta.iterations))
            .with_deviance(None, self.df_residual)
    }
}

/// A fitted two-part model.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TwoPartModel {
    /// The logit part, `Pr(y > 0)`.
    pub logit: PartFit,
    /// The gamma part with log link, `E[y | y > 0]`.
    pub gamma: PartFit,
    /// Fit diagnostics.
    pub report: TwoPartReport,
}

/// Two-part model predictions, one entry per row of the design.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TwoPartPrediction {
    /// The probability of a positive outcome.
    pub prob_positive: Vec<f64>,
    /// The mean of a positive outcome, `E[y | y > 0, x]`.
    pub mean_positive: Vec<f64>,
    /// The mean of the outcome, `E[y | x]`, the product of the two.
    pub mean: Vec<f64>,
}

/// The diagnostics of a two-part fit.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TwoPartReport {
    /// Fit metadata of the two parts together.
    pub meta: FitMetadata,
    /// Why the logit part is the intercept-only prevalence model, `None` when
    /// it was fitted. Only [`Separation::InterceptOnly`] gives one.
    pub logit_intercept_only: Option<InterceptOnly>,
    /// The covariance of both parts, `None` when none was computed.
    pub variance: Option<VarianceKind>,
    /// The number of clusters of a clustered sandwich, `None` otherwise.
    pub cluster_count: Option<usize>,
    /// The failed attempts before the one that succeeded, under
    /// [`FitStrategy::Relaxed`].
    pub attempts: Vec<AttemptDiagnostics>,
}

impl TwoPartModel {
    /// The mean of the predicted outcome over the rows of `x`, without forming the
    /// predictions.
    ///
    /// With `weights` this is `Σ w_i m_i / Σ w_i` over the rows with `w_i != 0`, with
    /// `None` it is the plain mean. It is bit-identical to averaging the column that
    /// [`Model::predict`] returns in row order. Only blocks of the linear predictors
    /// are held in memory.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when `x` does not have one column per coefficient or
    /// `weights` is not one value per row.
    pub fn predict_mean(&self, x: MatRef<'_, f64>, weights: Option<&[f64]>) -> Result<f64, Error> {
        check_columns(x, self.logit.coefficients.len(), weights)?;
        let (logit, gamma) = (
            column_mat(&self.logit.coefficients),
            column_mat(&self.gamma.coefficients),
        );
        Ok(crate::utils::mean_of_row_values(
            x,
            [&logit, &gamma],
            weights,
            |[eta_logit, eta_gamma]| {
                let prob = 1.0 / (1.0 + (-eta_logit).exp());
                let mean = eta_gamma
                    .clamp(-GAMMA_LOG_LINK_ETA_CLAMP, GAMMA_LOG_LINK_ETA_CLAMP)
                    .exp();
                prob * mean
            },
        ))
    }

    /// The log-likelihood of `data` under the model: the binomial
    /// log-likelihood of the logit part and the gamma log-likelihood of the
    /// positive outcomes, with the gamma dispersion the mean squared Pearson
    /// residual of the positive outcomes. The weights of `data` are not used.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the design of `data` does not have one column
    /// per coefficient.
    pub fn log_likelihood(&self, data: &CostData<'_>) -> Result<f64, Error> {
        let prediction = self.predict(data.x())?;
        Ok(log_likelihood(
            data.y(),
            &prediction.prob_positive,
            &prediction.mean_positive,
        ))
    }
}

impl Model for TwoPartModel {
    type Prediction = TwoPartPrediction;
    type Report = TwoPartReport;

    fn predict(&self, x: MatRef<'_, f64>) -> Result<Self::Prediction, Error> {
        check_columns(x, self.logit.coefficients.len(), None)?;
        let eta_logit = linear_predictor(x, &column_mat(&self.logit.coefficients));
        let eta_gamma = linear_predictor(x, &column_mat(&self.gamma.coefficients));
        let n = x.nrows();
        let mut out = TwoPartPrediction {
            prob_positive: Vec::with_capacity(n),
            mean_positive: Vec::with_capacity(n),
            mean: Vec::with_capacity(n),
        };
        for i in 0..n {
            let prob = 1.0 / (1.0 + (-eta_logit[(i, 0)]).exp());
            let mean = eta_gamma[(i, 0)]
                .clamp(-GAMMA_LOG_LINK_ETA_CLAMP, GAMMA_LOG_LINK_ETA_CLAMP)
                .exp();
            out.prob_positive.push(prob);
            out.mean_positive.push(mean);
            out.mean.push(prob * mean);
        }
        Ok(out)
    }

    fn report(&self) -> &Self::Report {
        &self.report
    }
}

/// Starting coefficients of the two IRLS loops, such as the full-sample fit
/// for a bootstrap replicate: a replicate converges from them to the same
/// optimum in fewer iterations. Coefficients of the wrong length are not
/// used, and that part starts cold.
#[derive(Clone, Debug)]
pub struct TwoPartWarmStart {
    /// Logit coefficients.
    pub logit: Vec<f64>,
    /// Gamma coefficients.
    pub gamma: Vec<f64>,
}

impl From<&TwoPartModel> for TwoPartWarmStart {
    fn from(model: &TwoPartModel) -> Self {
        Self {
            logit: model.logit.coefficients.clone(),
            gamma: model.gamma.coefficients.clone(),
        }
    }
}

/// Fit the two-part model: a logit for `y > 0` on every row and a gamma
/// regression with log link on the positive rows, from `warm` when given.
///
/// The weights of `data` are case weights of both parts; a row with weight 0
/// leaves the fit. The covariances are those `options.variance` chooses,
/// clustered sandwiches when `data` has clusters and the sandwich is chosen.
///
/// # Errors
///
/// [`Error::Input`] for a penalty outside its domain ([`Regularization`]);
/// [`Error::Unsupported`] for a model-based covariance with sampling weights;
/// [`Error::NoPositiveOutcome`] when no outcome with a positive weight is
/// positive; [`Error::Separation`] for a logit without a finite estimate
/// under [`Separation::Refuse`]; [`Error::NotConverged`], [`Error::Saturated`],
/// [`Error::NotFinite`] or [`Error::Singular`] when a part fails and the
/// strategy does not recover it.
pub fn fit(
    data: &CostData<'_>,
    options: &TwoPartOptions,
    warm: Option<&TwoPartWarmStart>,
) -> Result<TwoPartModel, Error> {
    let options = *options;
    options.regularization.check()?;
    let inference = options.variance.resolve(data)?;
    let x = data.x();
    let y = data.y_column();
    let ones = Mat::from_fn(y.nrows(), 1, |_, _| 1.0);
    let weights = data.weights_column().unwrap_or(ones.as_ref());
    let retried = run_with_retries(
        options,
        options.strategy,
        |base, fallback_lambda, scale| TwoPartOptions {
            regularization: Regularization::Ridge {
                lambda: fallback_lambda * scale,
                exclude_intercept: true,
            },
            ..base
        },
        |current| match current.regularization {
            Regularization::Ridge { lambda, .. } | Regularization::ElasticNet { lambda, .. } => {
                lambda
            }
            Regularization::BayesianRidge { prior_scale, .. } => 1.0 / (prior_scale * prior_scale),
            Regularization::None => 0.0,
        },
        |current_options| {
            fit_two_part_weighted(
                x,
                y,
                weights,
                data.clusters(),
                current_options,
                inference,
                warm,
            )
        },
    )?;

    let mut model = retried.fit;
    model.report.attempts = retried.attempts;
    model.report.meta.fallback_attempts = retried.attempt_idx;
    Ok(model)
}

/// The weighted fit of checked columns, with the covariance `inference`
/// asks for, clustered when `clusters` is given and the sandwich is asked for.
pub(crate) fn fit_two_part_weighted(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    weights: MatRef<'_, f64>,
    clusters: Option<&[u32]>,
    options: TwoPartOptions,
    inference: Inference,
    warm: Option<&TwoPartWarmStart>,
) -> Result<TwoPartModel, Error> {
    let is_positive = Mat::from_fn(y.nrows(), 1, |i, _| if y[(i, 0)] > 0.0 { 1.0 } else { 0.0 });
    let warm_logit = warm.map(|w| column_mat(&w.logit));
    let (beta_logit, iterations_logit, logit_intercept_only) = fit_logit_weighted(
        x,
        is_positive.as_ref(),
        weights,
        options,
        warm_logit.as_ref(),
    )?;
    match logit_intercept_only {
        Some(InterceptOnly::AllZero) => return Err(Error::NoPositiveOutcome),
        Some(reason) if options.separation == Separation::Refuse => {
            return Err(Error::Separation { reason });
        }
        _ => {}
    }

    // A positive row with weight 0 leaves the gamma part.
    let positive_indices: Vec<usize> = (0..y.nrows())
        .filter(|&idx| y[(idx, 0)] > 0.0 && weights[(idx, 0)] > 0.0)
        .collect();
    if positive_indices.is_empty() {
        return Err(Error::NoPositiveOutcome);
    }

    let x_pos = select_rows_ref(x, &positive_indices);
    let y_pos = select_values_ref(y, &positive_indices);
    let w_pos = select_values_ref(weights, &positive_indices);

    let warm_gamma = warm.map(|w| column_mat(&w.gamma));
    let (beta_gamma, iterations_gamma) = fit_gamma_log_link_weighted(
        x_pos.as_ref(),
        y_pos.as_ref(),
        w_pos.as_ref(),
        options,
        warm_gamma.as_ref(),
    )?;

    let p = x.ncols();
    let (n_used_logit, df_logit) = residual_df(weights, inference.frequency, p);
    let (n_used_gamma, df_gamma) = residual_df(w_pos.as_ref(), inference.frequency, p);
    let eta_gamma = linear_predictor(x_pos.as_ref(), &beta_gamma);
    let dispersion_gamma = pearson_dispersion(
        w_pos.as_ref(),
        |i| {
            let mu = eta_gamma[(i, 0)].exp();
            (y_pos[(i, 0)] - mu) / mu
        },
        df_gamma,
    );

    let robust = inference.robust();
    let clusters = clusters.filter(|_| robust);
    let clusters_pos: Option<Vec<u32>> =
        clusters.map(|labels| positive_indices.iter().map(|&i| labels[i]).collect());
    let (cov_logit, cov_gamma) = match inference.variance {
        Some(_) => {
            let logit = covariance_logit(
                x,
                is_positive.as_ref(),
                weights,
                &beta_logit,
                clusters,
                options,
                robust,
            )?;
            let gamma = covariance_gamma(
                x_pos.as_ref(),
                y_pos.as_ref(),
                w_pos.as_ref(),
                &beta_gamma,
                clusters_pos.as_deref(),
                options,
                robust,
            )?;
            // The model-based gamma covariance is the inverse information at unit
            // dispersion scaled by the Pearson dispersion; the sandwich does not
            // involve the dispersion.
            let gamma = if robust {
                Some(to_covariance(&gamma, 1.0))
            } else {
                dispersion_gamma.map(|phi| to_covariance(&gamma, phi))
            };
            (Some(to_covariance(&logit, 1.0)), gamma)
        }
        None => (None, None),
    };

    let meta = |iterations| FitMetadata {
        iterations,
        converged: true,
        solver: SolverKind::Irls,
        ..FitMetadata::default()
    };
    let part = |part, beta: &Mat<f64>, covariance: Option<Covariance>, rows| {
        let (n, n_used, df, dispersion, iterations) = match part {
            Part::Logit => (rows, n_used_logit, df_logit, Some(1.0), iterations_logit),
            Part::Gamma => (
                rows,
                n_used_gamma,
                df_gamma,
                dispersion_gamma,
                iterations_gamma,
            ),
        };
        PartFit {
            coefficients: column_values(beta),
            se: standard_errors(covariance.as_ref(), p),
            covariance,
            dispersion,
            df_residual: df,
            n,
            n_used,
            meta: meta(iterations),
            part,
            variance: inference.variance,
        }
    };
    let cluster_count =
        clusters.map(|labels| labels.iter().copied().collect::<BTreeSet<_>>().len());
    let report = TwoPartReport {
        meta: meta(iterations_logit + iterations_gamma),
        logit_intercept_only,
        variance: inference.variance,
        cluster_count,
        attempts: Vec::new(),
    };

    Ok(TwoPartModel {
        logit: part(Part::Logit, &beta_logit, cov_logit, y.nrows()),
        gamma: part(Part::Gamma, &beta_gamma, cov_gamma, positive_indices.len()),
        report,
    })
}

/// The two-part log-likelihood of `y` at the predicted probabilities and
/// positive means, with the gamma dispersion from the Pearson residuals of
/// the positive outcomes.
pub(crate) fn log_likelihood(y: &[f64], prob: &[f64], mean_pos: &[f64]) -> f64 {
    let mut loglik = 0.0;
    let mut phi_sum = 0.0;
    let mut n_pos = 0.0;
    for i in 0..y.len() {
        let yi = y[i];
        let pi = prob[i].clamp(1e-12, 1.0 - 1e-12);
        if yi > 0.0 {
            loglik += pi.ln();
            let mu = mean_pos[i].max(1e-12);
            phi_sum += (yi - mu).powi(2) / (mu * mu);
            n_pos += 1.0;
        } else {
            loglik += (1.0 - pi).ln();
        }
    }

    if n_pos == 0.0 {
        return loglik;
    }

    let phi = (phi_sum / n_pos).max(1e-8);
    let shape = 1.0 / phi;
    for i in 0..y.len() {
        let yi = y[i];
        if yi > 0.0 {
            let mu = mean_pos[i].max(1e-12);
            let scale = mu * phi;
            loglik += (shape - 1.0).mul_add(yi.ln(), -(yi / scale))
                - ln_gamma(shape)
                - shape * scale.ln();
        }
    }
    loglik
}

#[cfg(feature = "bench-internals")]
/// # Errors
///
/// Returns `epistat_linalg::Error` for a negative or non-finite weight.
pub fn benchmark_two_part_weighted_xtz(
    x: &Mat<f64>,
    weights: &Mat<f64>,
    z: &Mat<f64>,
) -> Result<Mat<f64>, epistat_linalg::Error> {
    weighted_xtz(x, weights, z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::usize_to_f64;
    use approx::assert_relative_eq;
    use epistat_core::{Bootstrap, CaseWeights, Matrix, Seed};
    use rstest::rstest;

    fn fitted(x: &Mat<f64>, y: &[f64], options: TwoPartOptions) -> TwoPartModel {
        let data = CostData::builder(x.as_ref(), y).build().expect("data");
        fit(&data, &options, None).expect("fit")
    }

    fn intercept_only() -> TwoPartOptions {
        TwoPartOptions {
            separation: Separation::InterceptOnly,
            ..TwoPartOptions::default()
        }
    }

    /// A design of an intercept and `i / scale`, with every third outcome
    /// zero and the others rising with `i`.
    fn linear_sample(n: usize, scale: f64) -> (Mat<f64>, Vec<f64>) {
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / scale },
        );
        let y = (0..n)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    1.0 + usize_to_f64(i)
                }
            })
            .collect();
        (x, y)
    }

    #[test]
    fn two_part_model_runs_on_synthetic_data() {
        let n = 200;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 50.0 },
        );
        let mut y = vec![0.0; n];
        let mut rng = epistat_rng::Stream::seeded(42);
        for i in 0..n {
            let p = 1.0 / (1.0 + (-0.5f64.mul_add(x[(i, 1)], -1.0)).exp());
            if rng.uniform() < p {
                let mean = 0.3f64.mul_add(x[(i, 1)], 1.0).exp();
                y[i] = mean * (0.5 + rng.uniform());
            }
        }

        let model = fitted(&x, &y, TwoPartOptions::default());
        assert!(model.logit.meta.iterations > 0);
        assert!(model.gamma.meta.iterations > 0);
        assert!(model.report.logit_intercept_only.is_none());
        assert_eq!(model.report.variance, Some(VarianceKind::ModelBased));
        assert!(model.logit.se.iter().all(Option::is_some));
        assert!(model.gamma.se.iter().all(Option::is_some));

        let prediction = model.predict(x.as_ref()).expect("predict");
        assert_eq!(prediction.mean.len(), n);
        assert_relative_eq!(
            prediction.mean[0],
            prediction.prob_positive[0] * prediction.mean_positive[0]
        );
        assert!(
            model
                .log_likelihood(&CostData::builder(x.as_ref(), &y).build().expect("data"))
                .expect("ll")
                .is_finite()
        );
    }

    #[rstest]
    #[case::ridge(Regularization::Ridge { lambda: 0.1, exclude_intercept: true }, Variance::Sandwich)]
    #[case::elastic_net(Regularization::ElasticNet { lambda: 0.1, alpha: 0.5, exclude_intercept: true }, Variance::Auto)]
    fn test_two_part_regularization_options(
        #[case] reg: Regularization,
        #[case] variance: Variance,
    ) {
        let n = 50;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 10.0 },
        );
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    0.1f64.mul_add(usize_to_f64(i), 1.0)
                }
            })
            .collect();
        let options = TwoPartOptions {
            regularization: reg,
            variance,
            ..TwoPartOptions::default()
        };
        let model = fitted(&x, &y, options);
        assert!(model.logit.meta.iterations > 0 || model.gamma.meta.iterations > 0);
        if variance == Variance::Sandwich {
            assert_eq!(model.report.variance, Some(VarianceKind::Robust));
            assert!(model.logit.se.iter().all(Option::is_some));
            assert!(model.gamma.se.iter().all(Option::is_some));
        }
    }

    #[rstest]
    #[case::alpha_above_one(Regularization::ElasticNet { lambda: 0.1, alpha: 1.5, exclude_intercept: true })]
    #[case::alpha_below_zero(Regularization::ElasticNet { lambda: 0.1, alpha: -0.1, exclude_intercept: true })]
    #[case::elastic_net_negative_lambda(Regularization::ElasticNet { lambda: -0.1, alpha: 0.5, exclude_intercept: true })]
    #[case::ridge_negative_lambda(Regularization::Ridge { lambda: -1.0, exclude_intercept: true })]
    #[case::ridge_infinite_lambda(Regularization::Ridge { lambda: f64::INFINITY, exclude_intercept: true })]
    fn a_penalty_outside_its_domain_is_refused(#[case] reg: Regularization) {
        let (x, y) = linear_sample(30, 10.0);
        let data = CostData::builder(x.as_ref(), &y).build().expect("data");
        let options = TwoPartOptions {
            regularization: reg,
            ..TwoPartOptions::default()
        };
        assert!(matches!(fit(&data, &options, None), Err(Error::Input(_))));
    }

    #[test]
    fn all_positive_outcomes_report_the_intercept_only_logit() {
        let n = 40;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 10.0 },
        );
        let y: Vec<f64> = (0..n).map(|i| 1.0 + usize_to_f64(i) / 100.0).collect();
        let model = fitted(&x, &y, intercept_only());
        assert_eq!(model.logit.meta.iterations, 0);
        assert_eq!(
            model.report.logit_intercept_only,
            Some(InterceptOnly::AllPositive)
        );
        assert!(model.gamma.meta.iterations > 0);
    }

    #[test]
    fn all_positive_outcomes_are_refused_by_default() {
        let n = 40;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 10.0 },
        );
        let y: Vec<f64> = (0..n).map(|i| 1.0 + usize_to_f64(i) / 100.0).collect();
        let data = CostData::builder(x.as_ref(), &y).build().expect("data");
        assert_eq!(
            fit(&data, &TwoPartOptions::default(), None).expect_err("constant indicator"),
            Error::Separation {
                reason: InterceptOnly::AllPositive
            }
        );
    }

    fn separable_sample() -> (Mat<f64>, Vec<f64>) {
        let n = 30;
        let x = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => usize_to_f64(i) / 10.0,
            _ => usize_to_f64(i % 4),
        });
        let y: Vec<f64> = (0..n)
            .map(|i| if i < 2 { 0.0 } else { 5.0 + usize_to_f64(i) })
            .collect();
        (x, y)
    }

    #[test]
    fn a_separable_logit_is_reported() {
        let (x, y) = separable_sample();
        let report = fitted(&x, &y, intercept_only()).report;
        assert_eq!(
            report.logit_intercept_only,
            Some(InterceptOnly::Separable {
                minority: 2,
                columns: 3
            })
        );
    }

    #[test]
    fn a_separable_logit_is_refused_by_default() {
        let (x, y) = separable_sample();
        let data = CostData::builder(x.as_ref(), &y).build().expect("data");
        assert_eq!(
            fit(&data, &TwoPartOptions::default(), None).expect_err("separable"),
            Error::Separation {
                reason: InterceptOnly::Separable {
                    minority: 2,
                    columns: 3
                }
            }
        );
    }

    #[test]
    fn gamma_fit_survives_heavy_tailed_outcome() {
        // A mass of small positive values with a few extreme outliers, on which an
        // unclamped log-link predictor drives mu = exp(eta) to +inf. The fit must give
        // finite coefficients.
        let n = 80;
        let x = Mat::from_fn(n, 2, |i, j| {
            if j == 0 {
                1.0
            } else {
                usize_to_f64(i) / usize_to_f64(n)
            }
        });
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 16 == 0 {
                5.0e7
            } else {
                40.0 + usize_to_f64(i)
            }
        });
        let w = Mat::from_fn(n, 1, |_, _| 1.0);

        let (beta, iterations) = fit_gamma_log_link_weighted(
            x.as_ref(),
            y.as_ref(),
            w.as_ref(),
            TwoPartOptions::stable_defaults(),
            None,
        )
        .expect("heavy-tailed gamma fit should converge");
        assert!(iterations >= 1);
        assert!(
            crate::utils::matrix_is_finite(&beta),
            "estimated coefficients must be finite"
        );
    }

    #[test]
    fn gamma_prediction_clamps_extreme_linear_predictor() {
        let n = 30;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| {
                if j == 0 { 1.0 } else { usize_to_f64(i) / 10.0 }
            },
        );
        let y: Vec<f64> = (0..n).map(|i| 10.0 + usize_to_f64(i)).collect();
        let options = TwoPartOptions {
            separation: Separation::InterceptOnly,
            ..TwoPartOptions::stable_defaults()
        };
        let model = fitted(&x, &y, options);

        let x_extreme = Mat::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { 1.0e6 });
        let prediction = model.predict(x_extreme.as_ref()).expect("predict");
        assert!(
            prediction.mean_positive[0].is_finite(),
            "clamp must keep exp(eta) finite at extreme covariates"
        );
        assert!(prediction.mean[0].is_finite());
    }

    #[test]
    fn predictions_refuse_a_design_of_the_wrong_width() {
        let x = Mat::from_fn(20, 2, |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) });
        let y: Vec<f64> = (0..20)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    2.0 + usize_to_f64(i)
                }
            })
            .collect();
        let model = fitted(&x, &y, TwoPartOptions::default());
        let narrow = Mat::from_fn(3, 1, |_, _| 1.0);
        assert!(matches!(
            model.predict(narrow.as_ref()),
            Err(Error::Input(_))
        ));
        assert!(matches!(
            model.predict_mean(x.as_ref(), Some(&[1.0])),
            Err(Error::Input(_))
        ));
    }

    #[test]
    fn the_tables_of_the_parts_carry_their_coefficients_and_standard_errors() {
        let (x, y) = linear_sample(60, 20.0);
        let model = fitted(&x, &y, TwoPartOptions::default());
        let level = Level::alpha(0.05).expect("level");
        for (part, effect) in [
            (&model.logit, EffectScale::LogOddsRatio),
            (&model.gamma, EffectScale::LogMeanRatio),
        ] {
            assert_eq!(part.default_variance(), VarianceKind::ModelBased);
            let table = part
                .coefficients(level, VarianceKind::ModelBased)
                .expect("model-based table");
            assert_eq!(table.effect(), effect);
            assert_eq!(table.len(), part.coefficients.len());
            for (j, term) in table.terms().iter().enumerate() {
                let estimate = term.estimate.expect("estimated");
                assert_eq!(estimate.estimate, part.coefficients[j]);
                assert_eq!(estimate.se, part.se[j]);
                let interval = estimate.interval.expect("interval");
                assert!(interval.upper > interval.lower);
            }
            assert_eq!(
                part.tidy().expect("tidy"),
                part.coefficients(Level::default(), VarianceKind::ModelBased)
                    .expect("table at the default level")
            );
            assert_eq!(
                part.coefficients(level, VarianceKind::Robust)
                    .expect_err("no sandwich"),
                Error::NoCovariance {
                    asked: VarianceKind::Robust
                }
            );
        }

        let without = fitted(
            &x,
            &y,
            TwoPartOptions {
                variance: Variance::None,
                ..TwoPartOptions::default()
            },
        );
        for part in [&without.logit, &without.gamma] {
            assert_eq!(
                part.coefficients(level, VarianceKind::ModelBased)
                    .expect_err("no covariance"),
                Error::NoCovariance {
                    asked: VarianceKind::ModelBased
                }
            );
        }
    }

    /// The inverse of a symmetric positive-definite 2 x 2 matrix in closed
    /// form, with its condition number in the 2-norm.
    fn inverse_2x2(a: [[f64; 2]; 2]) -> ([[f64; 2]; 2], f64) {
        let det = a[0][0].mul_add(a[1][1], -(a[0][1] * a[1][0]));
        let inverse = [
            [a[1][1] / det, -a[0][1] / det],
            [-a[1][0] / det, a[0][0] / det],
        ];
        let half_trace = f64::midpoint(a[0][0], a[1][1]);
        let root = half_trace.mul_add(half_trace, -det).sqrt();
        let condition = (half_trace + root) / (half_trace - root);
        (inverse, condition)
    }

    #[test]
    fn the_model_based_gamma_covariance_is_the_dispersion_times_the_inverse_information() {
        let (x, y) = linear_sample(60, 20.0);
        let model = fitted(&x, &y, TwoPartOptions::default());
        let positive: Vec<usize> = (0..y.len()).filter(|&i| y[i] > 0.0).collect();

        // X' W X on the positive rows: the gamma part with log link has
        // working weight 1 at unit dispersion, so W holds the prior weights,
        // here all 1.
        let mut information = [[0.0; 2]; 2];
        for &i in &positive {
            for (a, row) in information.iter_mut().enumerate() {
                for (b, entry) in row.iter_mut().enumerate() {
                    *entry += x[(i, a)] * x[(i, b)];
                }
            }
        }
        let (inverse, condition) = inverse_2x2(information);

        // The Pearson dispersion of the positive rows, R's summary.glm.
        let beta = &model.gamma.coefficients;
        let pearson: f64 = positive
            .iter()
            .map(|&i| {
                let mu = beta[0].mul_add(x[(i, 0)], beta[1] * x[(i, 1)]).exp();
                ((y[i] - mu) / mu).powi(2)
            })
            .sum::<f64>()
            / (usize_to_f64(positive.len()) - 2.0);
        let phi = model.gamma.dispersion.expect("dispersion");
        // The library and the test sum the same squared residuals in
        // possibly different orders from linear predictors that can differ in
        // the last bit; a relative bound of 1e3 eps covers the 40 terms with
        // room for the exp amplification of an eta rounding.
        assert_relative_eq!(phi, pearson, max_relative = 1e3 * f64::EPSILON);

        // A backward-stable solve for the inverse has relative error of
        // order condition * eps on its entries, and the closed form has the
        // same order through its determinant; a factor of 16 covers the
        // dimension and the products of the factorisation, relative to the
        // largest entry since the off-diagonal one can be small.
        let covariance = model.gamma.covariance.as_ref().expect("covariance");
        let scale = inverse
            .iter()
            .flatten()
            .fold(0.0_f64, |m, v| m.max(v.abs()))
            * phi;
        let tolerance = 16.0 * condition * f64::EPSILON * scale;
        for (a, row) in inverse.iter().enumerate() {
            for (b, &entry) in row.iter().enumerate() {
                let expected = phi * entry;
                let actual = covariance.get(a, b).expect("defined");
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "entry ({a}, {b}): {actual} against {expected}, tolerance {tolerance}"
                );
            }
        }
    }

    #[test]
    fn clustered_data_get_the_sandwich_unless_the_model_based_covariance_is_asked_for() {
        let (x, y) = linear_sample(60, 20.0);
        let clusters: Vec<u32> = (0..60)
            .map(|i| u32::try_from(i / 6).expect("small"))
            .collect();
        let data = CostData::builder(x.as_ref(), &y)
            .clusters(&clusters)
            .build()
            .expect("data");
        let auto = fit(&data, &TwoPartOptions::default(), None).expect("auto");
        assert_eq!(auto.report.variance, Some(VarianceKind::Robust));
        assert_eq!(auto.report.cluster_count, Some(10));
        let model_based = fit(
            &data,
            &TwoPartOptions {
                variance: Variance::ModelBased,
                ..TwoPartOptions::default()
            },
            None,
        )
        .expect("model-based");
        assert_eq!(model_based.report.variance, Some(VarianceKind::ModelBased));
        assert_eq!(model_based.report.cluster_count, None);
        // The clusters change the covariance and not the coefficients.
        assert_eq!(auto.gamma.coefficients, model_based.gamma.coefficients);
        let plain = fitted(&x, &y, TwoPartOptions::default());
        assert_eq!(
            plain.gamma.covariance.as_ref().map(Covariance::as_slice),
            model_based
                .gamma
                .covariance
                .as_ref()
                .map(Covariance::as_slice)
        );
    }

    #[test]
    fn a_model_based_covariance_with_sampling_weights_is_refused() {
        let (x, y) = linear_sample(30, 10.0);
        let w: Vec<f64> = (0..30).map(|i| 1.0 + usize_to_f64(i % 3)).collect();
        let data = CostData::builder(x.as_ref(), &y)
            .weights(CaseWeights::sampling(&w).expect("weights"))
            .build()
            .expect("data");
        let options = TwoPartOptions {
            variance: Variance::ModelBased,
            ..TwoPartOptions::default()
        };
        assert!(matches!(
            fit(&data, &options, None),
            Err(Error::Unsupported { .. })
        ));
        let auto = fit(&data, &TwoPartOptions::default(), None).expect("auto");
        assert_eq!(auto.report.variance, Some(VarianceKind::Robust));
    }

    #[test]
    fn a_row_with_weight_zero_leaves_the_fit() {
        let (x, y) = linear_sample(60, 20.0);
        // Rows 5, 6, 17 and 40 leave: one zero outcome (6) and three positive.
        let dropped = [5, 6, 17, 40];
        let w: Vec<f64> = (0..60)
            .map(|i| if dropped.contains(&i) { 0.0 } else { 1.0 })
            .collect();
        let data = CostData::builder(x.as_ref(), &y)
            .weights(CaseWeights::new(&w).expect("weights"))
            .build()
            .expect("data");
        let weighted = fit(&data, &TwoPartOptions::default(), None).expect("weighted");

        let kept: Vec<usize> = (0..60).filter(|i| !dropped.contains(i)).collect();
        let x_kept = Mat::from_fn(kept.len(), 2, |i, j| x[(kept[i], j)]);
        let y_kept: Vec<f64> = kept.iter().map(|&i| y[i]).collect();
        let removed = fitted(&x_kept, &y_kept, TwoPartOptions::default());

        // A zero weight multiplies its row's terms to exact zeros, and the
        // positive rows of weight 0 are not passed to the gamma part, so the
        // two fits sum the same terms, though in blocks of other bounds and
        // so in another order. Each sum of the m = 56 rows that enter then
        // differs by at most gamma_m = m eps / (1 - m eps) relative to the
        // sum of its absolute terms, and the solve of a system of condition
        // kappa moves its solution by at most 2 kappa gamma_m relative to its
        // norm, at most sqrt(2) times the largest coefficient. The final
        // IRLS step contracts the differences of the earlier iterates, so
        // that bound holds for the coefficients. The covariances invert the
        // information at those coefficients: a change of b by d moves the
        // information by at most |x|_1 d relative, |x|_1 <= 1 + 59 / 20,
        // and the inverse by kappa times that, plus its own 2 kappa gamma_m.
        let condition = |beta: &[f64], positive_only: bool| {
            let mut information = [[0.0; 2]; 2];
            for &i in &kept {
                if positive_only && y[i] <= 0.0 {
                    continue;
                }
                let weight = if positive_only {
                    1.0
                } else {
                    let prob =
                        1.0 / (1.0 + (-beta[0].mul_add(x[(i, 0)], beta[1] * x[(i, 1)])).exp());
                    prob * (1.0 - prob)
                };
                for (a, row) in information.iter_mut().enumerate() {
                    for (b, entry) in row.iter_mut().enumerate() {
                        *entry += weight * x[(i, a)] * x[(i, b)];
                    }
                }
            }
            inverse_2x2(information).1
        };
        let m = usize_to_f64(kept.len());
        let gamma_m = m * f64::EPSILON / (1.0 - m * f64::EPSILON);
        for (name, a, b, positive_only) in [
            ("logit", &weighted.logit, &removed.logit, false),
            ("gamma", &weighted.gamma, &removed.gamma, true),
        ] {
            let kappa = condition(&b.coefficients, positive_only);
            let largest = b
                .coefficients
                .iter()
                .fold(0.0_f64, |acc, v| acc.max(v.abs()));
            let coefficient_tolerance = 2.0 * kappa * gamma_m * 2.0_f64.sqrt() * largest;
            for (u, v) in a.coefficients.iter().zip(&b.coefficients) {
                assert!(
                    (u - v).abs() <= coefficient_tolerance,
                    "{name} coefficient {u} against {v}, tolerance {coefficient_tolerance}"
                );
            }
            let relative = kappa * (3.95 * coefficient_tolerance + 2.0 * gamma_m);
            for (u, v) in a.se.iter().zip(&b.se) {
                let (u, v) = (u.expect("se"), v.expect("se"));
                assert!(
                    (u - v).abs() <= relative * v,
                    "{name} se {u} against {v}, relative tolerance {relative}"
                );
            }
            assert_eq!(a.df_residual, b.df_residual);
            assert_eq!(a.n_used, b.n_used);
        }
        assert_eq!(weighted.logit.n, 60);
        assert_eq!(weighted.logit.n_used, 56);
        assert_eq!(weighted.gamma.n, removed.gamma.n);
        // The dispersion is sum r_i^2 / df over the Pearson residuals
        // r_i = y_i / mu_i - 1 of the positive rows. A change d of the
        // coefficients moves eta_i by at most |x_i|_1 d <= 3.95 d and r_i by
        // (y_i / mu_i) times that to first order, so the sum moves by at most
        // sum 2 |r_i| (y_i / mu_i) 3.95 d, plus gamma_m of its own rounding.
        let beta = &removed.gamma.coefficients;
        let kappa = condition(beta, true);
        let largest = beta.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
        let d = 2.0 * kappa * gamma_m * 2.0_f64.sqrt() * largest;
        let (mut squares, mut moved) = (0.0, 0.0);
        for &i in kept.iter().filter(|&&i| y[i] > 0.0) {
            let ratio = y[i] / beta[0].mul_add(x[(i, 0)], beta[1] * x[(i, 1)]).exp();
            let r = ratio - 1.0;
            squares += r * r;
            moved += 2.0 * r.abs() * ratio * 3.95 * d;
        }
        let df = removed.gamma.df_residual.expect("df");
        let tolerance = gamma_m.mul_add(squares, moved) / df;
        let (phi_a, phi_b) = (
            weighted.gamma.dispersion.expect("dispersion"),
            removed.gamma.dispersion.expect("dispersion"),
        );
        assert!(
            (phi_a - phi_b).abs() <= tolerance,
            "dispersion {phi_a} against {phi_b}, tolerance {tolerance}"
        );
    }

    #[test]
    fn bootstrap_produces_parameter_samples() {
        let n = 40;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 10.0 },
        );
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if i % 4 == 0 {
                    0.0
                } else {
                    0.1f64.mul_add(usize_to_f64(i), 1.0)
                }
            })
            .collect();
        let data = CostData::builder(x.as_ref(), &y).build().expect("data");
        let result = bootstrap(
            &data,
            &TwoPartOptions::default(),
            Bootstrap::new(10, Seed(7)),
            Some(50),
        )
        .expect("bootstrap");
        assert_eq!(result.logit.shape(), (10, 2));
        assert_eq!(result.gamma.shape(), (10, 2));

        let ci = percentile_intervals(&result.logit, Level::alpha(0.1).expect("level"));
        assert_eq!(ci.len(), x.ncols());
    }

    #[test]
    fn bootstrap_refuses_clusters_and_no_replicates() {
        let x = Mat::from_fn(4, 1, |_, _| 1.0);
        let y = [0.0, 1.0, 2.0, 3.0];
        let data = CostData::builder(x.as_ref(), &y)
            .clusters(&[1, 1, 2, 2])
            .build()
            .expect("data");
        let options = TwoPartOptions::default();
        assert!(matches!(
            bootstrap(&data, &options, Bootstrap::new(5, Seed(1)), None),
            Err(Error::Unsupported { .. })
        ));
        assert!(matches!(
            bootstrap(
                &data.without_clusters(),
                &options,
                Bootstrap::new(0, Seed(1)),
                None
            ),
            Err(Error::Input(_))
        ));
    }

    /// Draws with one row per replicate.
    fn draws(rows: &[Vec<f64>]) -> Matrix {
        Matrix::from_rows(rows).expect("draws")
    }

    #[test]
    fn bootstrap_summary_produces_mean_and_ci() {
        let betas = draws(&[vec![1.0, 0.5], vec![1.2, 0.4], vec![0.8, 0.6]]);
        let summary = summary(&betas, Level::alpha(0.1).expect("level"));
        assert_eq!(summary.mean.len(), 2);
        assert_eq!(summary.se.len(), 2);
        assert_eq!(summary.intervals.len(), 2);
        assert_relative_eq!(summary.mean[0], 1.0, max_relative = 4.0 * f64::EPSILON);
        // sd(c(1, 1.2, 0.8)) = 0.2
        assert_relative_eq!(
            summary.se[0].expect("two draws or more"),
            0.2,
            max_relative = 16.0 * f64::EPSILON
        );
    }

    fn bootstrap_percentile_ci_sorted_reference(betas: &Matrix, alpha: f64) -> Vec<(f64, f64)> {
        betas
            .columns()
            .map(|column| {
                let mut values = column.collect::<Vec<_>>();
                values.sort_by(f64::total_cmp);
                let lower = epistat_stats::quantile::type7(&values, alpha / 2.0).unwrap();
                let upper = epistat_stats::quantile::type7(&values, 1.0 - alpha / 2.0).unwrap();
                (lower, upper)
            })
            .collect()
    }

    #[test]
    fn bootstrap_percentile_ci_equals_r_type_7() {
        // R: x <- c(2.5, -1.0, 4.0, 0.5, 3.0, 1.5, -0.5, 6.0, 2.0, 1.0, 3.5)
        //    quantile(x, c(0.025, 0.975), type = 7)  # -0.875  5.5
        //    quantile(x, c(0.1, 0.9), type = 7)      # -0.5    4.0
        let x = [2.5, -1.0, 4.0, 0.5, 3.0, 1.5, -0.5, 6.0, 2.0, 1.0, 3.5];
        let betas = Matrix::new(x.len(), 1, x.to_vec()).expect("draws");
        for (alpha, lower, upper) in [(0.05, -0.875, 5.5), (0.2, -0.5, 4.0)] {
            let ci = percentile_intervals(&betas, Level::alpha(alpha).expect("level"));
            assert!(
                (ci[0].lower.unwrap() - lower).abs() < 1e-12,
                "alpha={alpha}"
            );
            assert!(
                (ci[0].upper.unwrap() - upper).abs() < 1e-12,
                "alpha={alpha}"
            );
        }
    }

    #[test]
    fn bootstrap_percentile_ci_matches_sorted_reference() {
        for reps in [5usize, 19, 200, 999] {
            let betas = Matrix::from_fn(reps, 4, |draw, coef| {
                let d = usize_to_f64(draw);
                let c = usize_to_f64(coef);
                0.001f64.mul_add(-d, 0.05f64.mul_add(c, (0.17f64.mul_add(d, 0.31 * c)).sin()))
            })
            .expect("draws");
            for alpha in [0.01, 0.05, 0.10, 0.20] {
                let expected = bootstrap_percentile_ci_sorted_reference(&betas, alpha);
                let actual = percentile_intervals(&betas, Level::alpha(alpha).expect("level"));
                assert_eq!(actual.len(), expected.len());
                for (a, e) in actual.iter().zip(expected.iter()) {
                    assert_eq!(
                        a.lower.unwrap().to_bits(),
                        e.0.to_bits(),
                        "reps={reps}, alpha={alpha}"
                    );
                    assert_eq!(
                        a.upper.unwrap().to_bits(),
                        e.1.to_bits(),
                        "reps={reps}, alpha={alpha}"
                    );
                }
            }
        }
    }

    #[test]
    fn bootstrap_summary_handles_empty_input() {
        let empty = Matrix::new(0, 0, Vec::new()).expect("empty");
        let summary = summary(&empty, Level::default());
        assert_eq!(summary.mean, Vec::<f64>::new());
        assert_eq!(summary.se, Vec::<Option<f64>>::new());
        assert_eq!(summary.intervals, Vec::new());
    }

    #[test]
    fn fit_with_cluster_robust_se() {
        let n = 30;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 5.0 },
        );
        let clusters: Vec<u32> = (0..n)
            .map(|i| u32::try_from(i / 5).expect("small"))
            .collect();
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if i % 4 == 0 {
                    0.0
                } else {
                    0.1f64.mul_add(usize_to_f64(i), 1.0)
                }
            })
            .collect();
        let options = TwoPartOptions {
            variance: Variance::Sandwich,
            ..TwoPartOptions::default()
        };
        let data = CostData::builder(x.as_ref(), &y)
            .clusters(&clusters)
            .build()
            .expect("data");
        let model = fit(&data, &options, None).expect("fit");
        assert_eq!(model.report.variance, Some(VarianceKind::Robust));
        assert_eq!(model.report.cluster_count, Some(6));
        assert!(model.logit.se.iter().all(Option::is_some));
        assert!(model.gamma.se.iter().all(Option::is_some));
    }

    #[test]
    fn variance_none_keeps_coefficients_and_drops_ses() {
        // A bootstrap replicate turns the covariance off; the coefficients must not
        // change.
        let (x, y) = linear_sample(60, 20.0);
        let without_cov = TwoPartOptions {
            variance: Variance::None,
            ..TwoPartOptions::default()
        };
        let model_a = fitted(&x, &y, TwoPartOptions::default());
        let model_b = fitted(&x, &y, without_cov);
        assert_eq!(model_a.logit.coefficients, model_b.logit.coefficients);
        assert_eq!(model_a.gamma.coefficients, model_b.gamma.coefficients);
        assert!(model_a.logit.se.iter().all(Option::is_some));
        assert!(model_a.gamma.se.iter().all(Option::is_some));
        assert!(model_b.logit.se.iter().all(Option::is_none));
        assert!(model_b.gamma.se.iter().all(Option::is_none));
        assert!(model_b.logit.covariance.is_none() && model_b.gamma.covariance.is_none());
        assert_eq!(model_b.report.variance, None);
    }

    #[test]
    fn two_part_warm_start_matches_cold_start() {
        let (x, y) = linear_sample(80, 20.0);
        let w = vec![1.0; 80];
        let data = CostData::builder(x.as_ref(), &y)
            .weights(CaseWeights::new(&w).expect("weights"))
            .build()
            .expect("data");
        let options = TwoPartOptions::default();

        let cold = fit(&data, &options, None).expect("cold fit");
        let warm_start = TwoPartWarmStart::from(&cold);
        let warm = fit(&data, &options, Some(&warm_start)).expect("warm fit");

        // Both loops converge to the same optimum, to the step tolerance of 1e-6
        // times the few iterations a cold start takes.
        for (c, w) in cold.logit.coefficients.iter().zip(&warm.logit.coefficients) {
            assert!((c - w).abs() < 1e-3);
        }
        for (c, w) in cold.gamma.coefficients.iter().zip(&warm.gamma.coefficients) {
            assert!((c - w).abs() < 1e-3);
        }
        assert!(warm.logit.meta.iterations <= cold.logit.meta.iterations);
        assert!(warm.gamma.meta.iterations <= cold.gamma.meta.iterations);
    }

    #[test]
    fn no_positive_outcome_is_an_error() {
        let x = Mat::from_fn(6, 1, |_, _| 1.0);
        let data = CostData::builder(x.as_ref(), &[0.0; 6])
            .build()
            .expect("data");
        for options in [TwoPartOptions::default(), intercept_only()] {
            assert_eq!(
                fit(&data, &options, None).expect_err("no positives"),
                Error::NoPositiveOutcome
            );
        }
    }

    #[test]
    fn fit_strategy_relaxed_handles_convergence_failure_with_retry() {
        // The minority class is larger than the parameter count, so the logit is
        // identified and the separation guard does not apply, and
        // `max_iterations: 1` is what forces the retry.
        let n = 10;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) });
        let y: Vec<f64> = (0..n).map(|i| if i % 3 == 0 { 0.0 } else { 1.0 }).collect();

        let options = TwoPartOptions {
            max_iterations: 1,
            strategy: FitStrategy::Relaxed {
                fallback_lambda: 1.0,
                max_retries: 2,
            },
            ..TwoPartOptions::default()
        };
        let data = CostData::builder(x.as_ref(), &y).build().expect("data");
        match fit(&data, &options, None) {
            Ok(model) => {
                assert!(model.report.meta.fallback_attempts > 0);
                assert_ne!(model.report.attempts.len(), 0);
            }
            Err(Error::NotConverged { .. } | Error::Singular) => {
                // On tiny pathological samples, relaxed retries may still fail.
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }
}
