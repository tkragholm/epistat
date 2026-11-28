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
//! separable, and it is replaced by the intercept-only prevalence model.
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
//! `X' W X` at the final coefficients, at unit dispersion for the gamma part. With
//! `robust_se` it is the sandwich `B M B`, `B` that inverse and `M` the sum of
//! the outer products of the score contributions `x_i w_i (y_i - p_i)` and
//! `x_i w_i (y_i - mu_i) / mu_i`, summed within clusters first when cluster
//! labels are given (HC0 without and CR0 with clusters) \[4\].
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

use crate::covariance::diag_sqrt;
use crate::data::CostData;
use crate::fit::{
    AttemptDiagnostics, FitMetadata, FitStrategy, Model, SolverKind, run_with_retries,
};
use crate::matrix_ops::{select_rows_ref, select_values_ref};
use crate::utils::linear_predictor;
#[cfg(feature = "bench-internals")]
use crate::utils::weighted_xtz;
use crate::{Error, check_columns};
use epistat_core::{Interval, Level};
use epistat_linalg::{Mat, MatRef};
use epistat_stats::dist::gamma::ln_gamma;
use std::collections::HashSet;

/// The penalty of the two parts.
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
    /// Whether the covariance is the sandwich, cluster-robust when the data
    /// have clusters, rather than the model-based one.
    pub robust_se: bool,
    /// Whether to compute the model-based covariance. A bootstrap that reads
    /// only the coefficients of its replicates turns it off; the coefficients
    /// do not change.
    pub compute_covariance: bool,
    /// What a fit that fails does.
    pub strategy: FitStrategy,
}

impl Default for TwoPartOptions {
    fn default() -> Self {
        Self {
            max_iterations: 50,
            tolerance: 1e-6,
            deviance_tolerance: 1e-8,
            max_step_halvings: 25,
            min_weight: 1e-6,
            regularization: Regularization::None,
            robust_se: false,
            compute_covariance: true,
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

/// Why the logit part is the intercept-only prevalence model.
///
/// A constant indicator has no logit to fit, and neither has one whose
/// smaller class has no more rows than the design has columns: a separating
/// hyperplane then always exists and the maximum-likelihood estimate is at
/// infinity. The fit then reports the intercept `logit(p)` of the weighted
/// share `p` of positive outcomes, clamped to `[min_weight, 1 - min_weight]`,
/// and zero slopes.
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

/// A fitted two-part model.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TwoPartModel {
    /// Logistic regression coefficients for Pr(y > 0).
    pub beta_logit: Mat<f64>,
    /// Gamma-log regression coefficients for E[y | y > 0].
    pub beta_gamma: Mat<f64>,
    /// Fit diagnostics.
    pub report: TwoPartReport,
}

/// Two-part model predictions, one row per row of the design.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TwoPartPrediction {
    /// Predicted probability of any positive outcome.
    pub prob_positive: Mat<f64>,
    /// Predicted mean of positive outcomes.
    pub mean_positive: Mat<f64>,
    /// Predicted expected outcome (probability * `mean_positive`).
    pub expected_outcome: Mat<f64>,
}

/// The diagnostics and covariances of a two-part fit.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TwoPartReport {
    /// Fit metadata of the two parts together.
    pub meta: FitMetadata,
    /// Fit metadata of the logit part.
    pub logit_meta: FitMetadata,
    /// Fit metadata of the gamma part.
    pub gamma_meta: FitMetadata,
    /// Iterations of the logit part, 0 when it is intercept-only.
    pub iterations_logit: usize,
    /// Iterations of the gamma part.
    pub iterations_gamma: usize,
    /// Why the logit part is the intercept-only prevalence model, `None` when
    /// it was fitted.
    pub logit_intercept_only: Option<InterceptOnly>,
    /// Standard errors for the logistic coefficients.
    pub se_logit: Option<Mat<f64>>,
    /// Standard errors for the gamma-log coefficients.
    pub se_gamma: Option<Mat<f64>>,
    /// Covariance matrix for the logistic coefficients.
    pub cov_logit: Option<Mat<f64>>,
    /// Covariance matrix for the gamma-log coefficients.
    pub cov_gamma: Option<Mat<f64>>,
    /// True if robust (sandwich) covariance was used.
    pub robust: bool,
    /// True if cluster-robust covariance was used.
    pub clustered: bool,
    /// Number of clusters used for cluster-robust covariance.
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
        check_columns(x, self.beta_logit.nrows(), weights)?;
        Ok(crate::utils::mean_of_row_values(
            x,
            [&self.beta_logit, &self.beta_gamma],
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
            data.y_column(),
            &prediction.prob_positive,
            &prediction.mean_positive,
        ))
    }
}

impl Model for TwoPartModel {
    type Prediction = TwoPartPrediction;
    type Report = TwoPartReport;

    fn predict(&self, x: MatRef<'_, f64>) -> Result<Self::Prediction, Error> {
        check_columns(x, self.beta_logit.nrows(), None)?;
        let eta_logit = linear_predictor(x, &self.beta_logit);
        let eta_gamma = linear_predictor(x, &self.beta_gamma);
        let n = x.nrows();
        let mut out = TwoPartPrediction {
            prob_positive: Mat::zeros(n, 1),
            mean_positive: Mat::zeros(n, 1),
            expected_outcome: Mat::zeros(n, 1),
        };
        for i in 0..n {
            let prob = 1.0 / (1.0 + (-eta_logit[(i, 0)]).exp());
            let mean = eta_gamma[(i, 0)]
                .clamp(-GAMMA_LOG_LINK_ETA_CLAMP, GAMMA_LOG_LINK_ETA_CLAMP)
                .exp();
            out.prob_positive[(i, 0)] = prob;
            out.mean_positive[(i, 0)] = mean;
            out.expected_outcome[(i, 0)] = prob * mean;
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
    pub beta_logit: Mat<f64>,
    /// Gamma coefficients.
    pub beta_gamma: Mat<f64>,
}

/// Fit the two-part model: a logit for `y > 0` on every row and a gamma
/// regression with log link on the positive rows, from `warm` when given.
///
/// The weights of `data` are prior weights of both parts. With
/// `robust_se` the covariances are sandwiches, cluster-robust when `data`
/// has clusters.
///
/// # Errors
///
/// [`Error::NoPositiveOutcome`] when no outcome is positive;
/// [`Error::NotConverged`], [`Error::Saturated`], [`Error::NotFinite`] or
/// [`Error::Singular`] when a part fails and the strategy does not recover it.
pub fn fit(
    data: &CostData<'_>,
    options: &TwoPartOptions,
    warm: Option<&TwoPartWarmStart>,
) -> Result<TwoPartModel, Error> {
    let options = *options;
    let x = data.x();
    let y = data.y_column();
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
        |current_options| match (data.weights_column(), data.clusters()) {
            (Some(weights), Some(clusters)) => {
                fit_two_part_clustered_weighted(x, y, weights, clusters, current_options, warm)
            }
            (Some(weights), None) => fit_two_part_weighted(x, y, weights, current_options, warm),
            (None, Some(clusters)) => fit_two_part_clustered(x, y, clusters, current_options),
            (None, None) => fit_two_part_unweighted(x, y, current_options),
        },
    )?;

    let mut model = retried.fit;
    model.report.attempts = retried.attempts;
    model.report.meta.fallback_attempts = retried.attempt_idx;
    Ok(model)
}

/// The unweighted fit, with every prior weight 1.
pub(crate) fn fit_two_part_unweighted(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    options: TwoPartOptions,
) -> Result<TwoPartModel, Error> {
    let weights = Mat::from_fn(y.nrows(), 1, |_, _| 1.0);
    fit_two_part_weighted(x, y, weights.as_ref(), options, None)
}

/// The weighted fit with model-based covariances.
pub(crate) fn fit_two_part_weighted(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    weights: MatRef<'_, f64>,
    options: TwoPartOptions,
    warm: Option<&TwoPartWarmStart>,
) -> Result<TwoPartModel, Error> {
    let is_positive = Mat::from_fn(y.nrows(), 1, |i, _| if y[(i, 0)] > 0.0 { 1.0 } else { 0.0 });
    let (beta_logit, iterations_logit, logit_intercept_only) = fit_logit_weighted(
        x,
        is_positive.as_ref(),
        weights,
        options,
        warm.map(|w| &w.beta_logit),
    )?;

    let positive_indices: Vec<usize> = (0..y.nrows()).filter(|&idx| y[(idx, 0)] > 0.0).collect();
    if positive_indices.is_empty() {
        return Err(Error::NoPositiveOutcome);
    }

    let x_pos = select_rows_ref(x, &positive_indices);
    let y_pos = select_values_ref(y, &positive_indices);
    let w_pos = select_values_ref(weights, &positive_indices);

    let (beta_gamma, iterations_gamma) = fit_gamma_log_link_weighted(
        x_pos.as_ref(),
        y_pos.as_ref(),
        w_pos.as_ref(),
        options,
        warm.map(|w| &w.beta_gamma),
    )?;

    // The cluster-robust path replaces these with sandwiches, and a bootstrap
    // replicate reads only the coefficients, so both turn them off.
    let (cov_logit, cov_gamma) = if options.compute_covariance {
        (
            Some(covariance_logit(
                x,
                is_positive.as_ref(),
                weights,
                &beta_logit,
                None,
                options,
            )?),
            Some(covariance_gamma(
                x_pos.as_ref(),
                y_pos.as_ref(),
                w_pos.as_ref(),
                &beta_gamma,
                None,
                options,
            )?),
        )
    } else {
        (None, None)
    };
    let se_logit = cov_logit.as_ref().map(diag_sqrt);
    let se_gamma = cov_gamma.as_ref().map(diag_sqrt);

    let meta = |iterations| FitMetadata {
        iterations,
        converged: true,
        solver: SolverKind::Irls,
        ..FitMetadata::default()
    };
    let report = TwoPartReport {
        meta: meta(iterations_logit + iterations_gamma),
        logit_meta: meta(iterations_logit),
        gamma_meta: meta(iterations_gamma),
        iterations_logit,
        iterations_gamma,
        logit_intercept_only,
        se_logit,
        se_gamma,
        cov_logit,
        cov_gamma,
        robust: options.robust_se,
        clustered: false,
        cluster_count: None,
        attempts: Vec::new(),
    };

    Ok(TwoPartModel {
        beta_logit,
        beta_gamma,
        report,
    })
}

fn fit_two_part_clustered(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    clusters: &[u32],
    options: TwoPartOptions,
) -> Result<TwoPartModel, Error> {
    let weights = Mat::from_fn(y.nrows(), 1, |_, _| 1.0);
    fit_two_part_clustered_weighted(x, y, weights.as_ref(), clusters, options, None)
}

fn fit_two_part_clustered_weighted(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    weights: MatRef<'_, f64>,
    clusters: &[u32],
    options: TwoPartOptions,
    warm: Option<&TwoPartWarmStart>,
) -> Result<TwoPartModel, Error> {
    let mut model = fit_two_part_weighted(x, y, weights, options, warm)?;
    if !options.robust_se {
        return Ok(model);
    }

    let is_positive = Mat::from_fn(y.nrows(), 1, |i, _| if y[(i, 0)] > 0.0 { 1.0 } else { 0.0 });
    let positive_indices: Vec<usize> = (0..y.nrows()).filter(|&idx| y[(idx, 0)] > 0.0).collect();
    let x_pos = select_rows_ref(x, &positive_indices);
    let y_pos = select_values_ref(y, &positive_indices);
    let w_pos = select_values_ref(weights, &positive_indices);
    let clusters_pos: Vec<u32> = positive_indices.iter().map(|&i| clusters[i]).collect();

    let cov_logit = covariance_logit(
        x,
        is_positive.as_ref(),
        weights,
        &model.beta_logit,
        Some(clusters),
        options,
    )?;
    let cov_gamma = covariance_gamma(
        x_pos.as_ref(),
        y_pos.as_ref(),
        w_pos.as_ref(),
        &model.beta_gamma,
        Some(&clusters_pos),
        options,
    )?;

    let report = &mut model.report;
    report.se_logit = Some(diag_sqrt(&cov_logit));
    report.se_gamma = Some(diag_sqrt(&cov_gamma));
    report.cov_logit = Some(cov_logit);
    report.cov_gamma = Some(cov_gamma);
    report.robust = true;
    report.clustered = true;
    report.cluster_count = Some(clusters.iter().collect::<HashSet<_>>().len());
    Ok(model)
}

/// The two-part log-likelihood of `y` at the predicted probabilities and
/// positive means, with the gamma dispersion from the Pearson residuals of
/// the positive outcomes.
pub(crate) fn log_likelihood(y: MatRef<'_, f64>, prob: &Mat<f64>, mean_pos: &Mat<f64>) -> f64 {
    let mut loglik = 0.0;
    let mut phi_sum = 0.0;
    let mut n_pos = 0.0;
    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        let pi = prob[(i, 0)].clamp(1e-12, 1.0 - 1e-12);
        if yi > 0.0 {
            loglik += pi.ln();
            let mu = mean_pos[(i, 0)].max(1e-12);
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
    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        if yi > 0.0 {
            let mu = mean_pos[(i, 0)].max(1e-12);
            let scale = mu * phi;
            loglik += (shape - 1.0).mul_add(yi.ln(), -(yi / scale))
                - ln_gamma(shape)
                - shape * scale.ln();
        }
    }
    loglik
}

/// Wald intervals `beta_j ± z se_j` from a covariance, with `z` the normal
/// quantile at the level's upper probability and `se_j` the root of the
/// diagonal, a negative variance read as 0.
///
/// # Errors
///
/// [`Error::Input`] when `cov` is not square with one row per coefficient.
pub fn wald_intervals(
    beta: &Mat<f64>,
    cov: &Mat<f64>,
    level: Level,
) -> Result<Vec<Interval>, Error> {
    let p = beta.nrows();
    epistat_core::check::same_length("rows of cov", p, cov.nrows())?;
    epistat_core::check::same_length("columns of cov", p, cov.ncols())?;
    let z = epistat_stats::dist::normal::quantile(level.upper_probability());
    Ok((0..p)
        .map(|i| {
            let se = cov[(i, i)].max(0.0).sqrt();
            Interval::new(
                Some(beta[(i, 0)] - z * se),
                Some(beta[(i, 0)] + z * se),
                level,
            )
        })
        .collect())
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
    use epistat_core::Bootstrap;
    use rstest::rstest;

    fn fitted(x: &Mat<f64>, y: &[f64], options: TwoPartOptions) -> TwoPartModel {
        let data = CostData::builder(x.as_ref(), y).build().expect("data");
        fit(&data, &options, None).expect("fit")
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
        let report = &model.report;
        assert!(report.iterations_logit > 0);
        assert!(report.iterations_gamma > 0);
        assert!(report.logit_intercept_only.is_none());
        assert!(report.se_logit.is_some());
        assert!(report.se_gamma.is_some());

        let prediction = model.predict(x.as_ref()).expect("predict");
        assert_eq!(prediction.expected_outcome.nrows(), n);
        assert_relative_eq!(
            prediction.expected_outcome[(0, 0)],
            prediction.prob_positive[(0, 0)] * prediction.mean_positive[(0, 0)]
        );
        assert!(
            model
                .log_likelihood(&CostData::builder(x.as_ref(), &y).build().expect("data"))
                .expect("ll")
                .is_finite()
        );
    }

    #[rstest]
    #[case::ridge(Regularization::Ridge { lambda: 0.1, exclude_intercept: true }, true)]
    #[case::elastic_net(Regularization::ElasticNet { lambda: 0.1, alpha: 0.5, exclude_intercept: true }, false)]
    fn test_two_part_regularization_options(#[case] reg: Regularization, #[case] robust_se: bool) {
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
            robust_se,
            ..TwoPartOptions::default()
        };
        let report = fitted(&x, &y, options).report;
        assert!(report.iterations_logit > 0 || report.iterations_gamma > 0);
        if robust_se {
            assert!(report.se_logit.is_some());
            assert!(report.se_gamma.is_some());
        }
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
        let report = fitted(&x, &y, TwoPartOptions::default()).report;
        assert_eq!(report.iterations_logit, 0);
        assert_eq!(
            report.logit_intercept_only,
            Some(InterceptOnly::AllPositive)
        );
        assert!(report.iterations_gamma > 0);
    }

    #[test]
    fn a_separable_logit_is_reported() {
        let n = 30;
        let x = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => usize_to_f64(i) / 10.0,
            _ => usize_to_f64(i % 4),
        });
        let y: Vec<f64> = (0..n)
            .map(|i| if i < 2 { 0.0 } else { 5.0 + usize_to_f64(i) })
            .collect();
        let report = fitted(&x, &y, TwoPartOptions::default()).report;
        assert_eq!(
            report.logit_intercept_only,
            Some(InterceptOnly::Separable {
                minority: 2,
                columns: 3
            })
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
        let model = fitted(&x, &y, TwoPartOptions::stable_defaults());

        let x_extreme = Mat::from_fn(1, 2, |_, j| if j == 0 { 1.0 } else { 1.0e6 });
        let prediction = model.predict(x_extreme.as_ref()).expect("predict");
        assert!(
            prediction.mean_positive[(0, 0)].is_finite(),
            "clamp must keep exp(eta) finite at extreme covariates"
        );
        assert!(prediction.expected_outcome[(0, 0)].is_finite());
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
    fn wald_intervals_from_covariance() {
        let beta = Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { -0.5 });
        let cov = Mat::from_fn(2, 2, |i, j| {
            if i == j {
                if i == 0 { 0.04 } else { 0.01 }
            } else {
                0.0
            }
        });
        let level = Level::alpha(0.05).expect("level");
        let ci = wald_intervals(&beta, &cov, level).expect("intervals");
        assert_eq!(ci.len(), 2);
        assert!(ci[0].upper > ci[0].lower);
        assert_eq!(ci[0].level, level);
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
            Bootstrap::new(10, 7),
            Some(50),
        )
        .expect("bootstrap");
        assert_eq!(result.betas_logit.len(), 10);
        assert_eq!(result.betas_gamma.len(), 10);

        let ci = percentile_intervals(&result.betas_logit, Level::alpha(0.1).expect("level"));
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
            bootstrap(&data, &options, Bootstrap::new(5, 1), None),
            Err(Error::Unsupported { .. })
        ));
        assert!(matches!(
            bootstrap(
                &data.without_clusters(),
                &options,
                Bootstrap::new(0, 1),
                None
            ),
            Err(Error::Input(_))
        ));
    }

    #[test]
    fn bootstrap_summary_produces_mean_and_ci() {
        let betas = vec![
            Mat::from_fn(2, 1, |i, _| if i == 0 { 1.0 } else { 0.5 }),
            Mat::from_fn(2, 1, |i, _| if i == 0 { 1.2 } else { 0.4 }),
            Mat::from_fn(2, 1, |i, _| if i == 0 { 0.8 } else { 0.6 }),
        ];
        let summary = summary(&betas, Level::alpha(0.1).expect("level"));
        assert_eq!(summary.mean.nrows(), 2);
        assert_eq!(summary.se.nrows(), 2);
        assert_eq!(summary.intervals.len(), 2);
    }

    fn bootstrap_percentile_ci_sorted_reference(betas: &[Mat<f64>], alpha: f64) -> Vec<(f64, f64)> {
        let n = betas[0].nrows();
        let mut intervals = Vec::with_capacity(n);
        for col in 0..n {
            let mut values = betas.iter().map(|b| b[(col, 0)]).collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            let lower = epistat_stats::quantile::type7(&values, alpha / 2.0).unwrap();
            let upper = epistat_stats::quantile::type7(&values, 1.0 - alpha / 2.0).unwrap();
            intervals.push((lower, upper));
        }
        intervals
    }

    #[test]
    fn bootstrap_percentile_ci_equals_r_type_7() {
        // R: x <- c(2.5, -1.0, 4.0, 0.5, 3.0, 1.5, -0.5, 6.0, 2.0, 1.0, 3.5)
        //    quantile(x, c(0.025, 0.975), type = 7)  # -0.875  5.5
        //    quantile(x, c(0.1, 0.9), type = 7)      # -0.5    4.0
        let x = [2.5, -1.0, 4.0, 0.5, 3.0, 1.5, -0.5, 6.0, 2.0, 1.0, 3.5];
        let betas = x
            .iter()
            .map(|&v| Mat::from_fn(1, 1, |_, _| v))
            .collect::<Vec<_>>();
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
            let betas = (0..reps)
                .map(|draw| {
                    Mat::from_fn(4, 1, |coef, _| {
                        let d = usize_to_f64(draw);
                        let c = usize_to_f64(coef);
                        0.001f64
                            .mul_add(-d, 0.05f64.mul_add(c, (0.17f64.mul_add(d, 0.31 * c)).sin()))
                    })
                })
                .collect::<Vec<_>>();
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
        let summary = summary(&[], Level::default());
        assert_eq!(summary.mean.nrows(), 0);
        assert_eq!(summary.se.nrows(), 0);
        assert_eq!(summary.intervals, []);
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
            robust_se: true,
            ..TwoPartOptions::default()
        };
        let data = CostData::builder(x.as_ref(), &y)
            .clusters(&clusters)
            .build()
            .expect("data");
        let report = fit(&data, &options, None).expect("fit").report;
        assert!(report.clustered);
        assert_eq!(report.cluster_count, Some(6));
        assert!(report.se_logit.is_some());
        assert!(report.se_gamma.is_some());
    }

    #[test]
    fn compute_covariance_false_keeps_coefficients_and_drops_ses() {
        // A bootstrap replicate turns the covariance off; the coefficients must not
        // change.
        let n = 60;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 20.0 },
        );
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    1.0 + usize_to_f64(i)
                }
            })
            .collect();
        let without_cov = TwoPartOptions {
            compute_covariance: false,
            ..TwoPartOptions::default()
        };
        let model_a = fitted(&x, &y, TwoPartOptions::default());
        let model_b = fitted(&x, &y, without_cov);
        for j in 0..model_a.beta_logit.nrows() {
            assert_eq!(
                model_a.beta_logit[(j, 0)].to_bits(),
                model_b.beta_logit[(j, 0)].to_bits()
            );
            assert_eq!(
                model_a.beta_gamma[(j, 0)].to_bits(),
                model_b.beta_gamma[(j, 0)].to_bits()
            );
        }
        assert!(model_a.report.se_logit.is_some() && model_a.report.se_gamma.is_some());
        assert!(model_b.report.se_logit.is_none() && model_b.report.se_gamma.is_none());
        assert!(model_b.report.cov_logit.is_none() && model_b.report.cov_gamma.is_none());
    }

    #[test]
    fn two_part_warm_start_matches_cold_start() {
        let n = 80;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 20.0 },
        );
        let y: Vec<f64> = (0..n)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    1.0 + usize_to_f64(i)
                }
            })
            .collect();
        let w = vec![1.0; n];
        let data = CostData::builder(x.as_ref(), &y)
            .weights(&w)
            .build()
            .expect("data");
        let options = TwoPartOptions::default();

        let cold = fit(&data, &options, None).expect("cold fit");
        let warm_start = TwoPartWarmStart {
            beta_logit: cold.beta_logit.clone(),
            beta_gamma: cold.beta_gamma.clone(),
        };
        let warm = fit(&data, &options, Some(&warm_start)).expect("warm fit");

        // Both loops converge to the same optimum, to the step tolerance of 1e-6
        // times the few iterations a cold start takes.
        for i in 0..cold.beta_logit.nrows() {
            assert!((cold.beta_logit[(i, 0)] - warm.beta_logit[(i, 0)]).abs() < 1e-3);
        }
        for i in 0..cold.beta_gamma.nrows() {
            assert!((cold.beta_gamma[(i, 0)] - warm.beta_gamma[(i, 0)]).abs() < 1e-3);
        }
        assert!(warm.report.iterations_logit <= cold.report.iterations_logit);
        assert!(warm.report.iterations_gamma <= cold.report.iterations_gamma);
    }

    #[test]
    fn no_positive_outcome_is_an_error() {
        let x = Mat::from_fn(6, 1, |_, _| 1.0);
        let data = CostData::builder(x.as_ref(), &[0.0; 6])
            .build()
            .expect("data");
        assert_eq!(
            fit(&data, &TwoPartOptions::default(), None).expect_err("no positives"),
            Error::NoPositiveOutcome
        );
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
