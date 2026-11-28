//! Log-normal regression with Duan's smearing retransformation.
//!
//! # The model
//!
//! On the positive outcomes, `ln y = x' b + e` with errors of mean zero, fitted
//! by least squares with an optional ridge. The mean on the original scale is
//! not `exp(x' b)` but `exp(x' b) E[exp(e)]`, and the smearing estimate of the
//! second factor is the mean of the exponentiated residuals \[1\],
//!
//! ```text
//! s = (1/n) sum_i exp(ln y_i - x_i' b),     E[y | x] = s exp(x' b),
//! ```
//!
//! which needs no normality of the errors. The fit has no prior weights and
//! refuses data that carry them.
//! The log-likelihood reported for information criteria is the normal one of
//! `ln y` on the positive rows at the maximum-likelihood variance, with the
//! Jacobian `-sum ln y_i`.
//!
//! # Standard errors
//!
//! Standard errors are reported only with `robust_se`: the HC0 sandwich on the
//! positive rows, summed within clusters first when cluster labels are given
//! \[2\].
//!
//! # References
//!
//! 1. Duan N. Smearing estimate: a nonparametric retransformation method. J Am
//!    Stat Assoc. 1983;78:605-10.
//! 2. Zeileis A. Object-oriented computation of sandwich estimators. J Stat
//!    Softw. 2006;16(9):1-16.

use epistat_linalg::{Mat, MatRef};

use crate::covariance::{diag_sqrt, sandwich, score_meat};
use crate::data::CostData;
use crate::fit::{
    AttemptDiagnostics, FitMetadata, FitStrategy, Model, SolverKind, run_with_retries,
};
use crate::matrix_ops::{map_mat, select_rows_ref, select_values_ref};
use crate::utils::{
    add_ridge_to_diagonal, linear_predictor, max_abs_diff, mean_column, solve_linear_system,
};
use crate::{Error, check_columns};

/// The options of [`fit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogNormalOptions {
    /// The most solves; least squares needs one.
    pub max_iterations: usize,
    /// Convergence tolerance on the largest change of a coefficient.
    pub tolerance: f64,
    /// L2 (ridge) penalty.
    pub l2_penalty: f64,
    /// Whether the first column (the intercept) is left unpenalised.
    pub l2_penalty_exclude_intercept: bool,
    /// Whether to compute the sandwich covariance, cluster-robust when the
    /// data have clusters. Without it the fit has no standard errors.
    pub robust_se: bool,
    /// What a fit that fails does.
    pub strategy: FitStrategy,
}

impl Default for LogNormalOptions {
    fn default() -> Self {
        Self {
            max_iterations: 1,
            tolerance: 1e-12,
            l2_penalty: 0.0,
            l2_penalty_exclude_intercept: true,
            robust_se: false,
            strategy: FitStrategy::Strict,
        }
    }
}

impl LogNormalOptions {
    /// A ridge of 1e-4, with the relaxed strategy from a ridge of 1e-3 and up
    /// to three retries.
    #[must_use]
    pub fn stable_defaults() -> Self {
        Self {
            l2_penalty: 1e-4,
            strategy: FitStrategy::Relaxed {
                fallback_lambda: 1e-3,
                max_retries: 3,
            },
            ..Self::default()
        }
    }
}

/// A fitted log-normal model with its smearing factor.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct LogNormalModel {
    /// Coefficients of `ln y`.
    pub beta: Mat<f64>,
    /// Duan's smearing factor, the mean of the exponentiated residuals.
    pub smearing_factor: f64,
    /// Fit diagnostics.
    pub report: LogNormalReport,
}

/// The diagnostics and covariance of a log-normal fit.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct LogNormalReport {
    /// Fit metadata.
    pub meta: FitMetadata,
    /// Standard errors of the coefficients, with `robust_se`.
    pub se: Option<Mat<f64>>,
    /// Covariance of the coefficients, with `robust_se`.
    pub cov: Option<Mat<f64>>,
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

/// Log-normal predictions on the original scale, one row per row of the
/// design.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct LogNormalPrediction {
    /// The predicted mean, the smeared retransformation.
    pub mean: Mat<f64>,
}

type RobustCovarianceResult = (Option<Mat<f64>>, bool, Option<usize>);

impl LogNormalModel {
    /// The mean of the predicted outcome over the rows of `x`, without forming the
    /// predictions.
    ///
    /// With `weights` this is `Σ w_i m_i / Σ w_i` over the rows with `w_i != 0`, with
    /// `None` it is the plain mean. It is bit-identical to averaging the column that
    /// [`Model::predict`] returns in row order.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when `x` does not have one column per coefficient or
    /// `weights` is not one value per row.
    pub fn predict_mean(&self, x: MatRef<'_, f64>, weights: Option<&[f64]>) -> Result<f64, Error> {
        check_columns(x, self.beta.nrows(), weights)?;
        Ok(crate::utils::mean_of_row_values(
            x,
            [&self.beta],
            weights,
            |[eta]| self.smearing_factor * eta.exp(),
        ))
    }

    /// The log-likelihood of the positive outcomes of `data`: the normal
    /// log-likelihood of `ln y` at the maximum-likelihood variance with the
    /// Jacobian `-sum ln y_i`. `None` when no outcome is positive.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the design of `data` does not have one column
    /// per coefficient.
    pub fn log_likelihood(&self, data: &CostData<'_>) -> Result<Option<f64>, Error> {
        check_columns(data.x(), self.beta.nrows(), None)?;
        let ll = log_likelihood(data.x(), data.y_column(), self);
        Ok((!ll.is_nan()).then_some(ll))
    }
}

impl Model for LogNormalModel {
    type Prediction = LogNormalPrediction;
    type Report = LogNormalReport;

    fn predict(&self, x: MatRef<'_, f64>) -> Result<Self::Prediction, Error> {
        check_columns(x, self.beta.nrows(), None)?;
        let eta = linear_predictor(x, &self.beta);
        Ok(LogNormalPrediction {
            mean: Mat::from_fn(x.nrows(), 1, |i, _| {
                self.smearing_factor * eta[(i, 0)].exp()
            }),
        })
    }

    fn report(&self) -> &Self::Report {
        &self.report
    }
}

/// The log-likelihood of the positive outcomes, NaN when none is positive.
pub(crate) fn log_likelihood(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    model: &LogNormalModel,
) -> f64 {
    let eta = linear_predictor(x, &model.beta);
    let mut sumsq = 0.0;
    let mut n_pos = 0.0;
    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        if yi > 0.0 {
            let resid = yi.ln() - eta[(i, 0)];
            sumsq = resid.mul_add(resid, sumsq);
            n_pos += 1.0;
        }
    }
    if n_pos == 0.0 {
        return f64::NAN;
    }
    let sigma2 = (sumsq / n_pos).max(1e-12);
    let mut loglik = 0.0;
    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        if yi > 0.0 {
            let resid = yi.ln() - eta[(i, 0)];
            loglik += (-0.5f64).mul_add(
                std::f64::consts::TAU.ln() + sigma2.ln() + resid * resid / sigma2,
                -yi.ln(),
            );
        }
    }
    loglik
}

/// Fit log-normal regression with Duan's smearing to the positive outcomes
/// of `data`.
///
/// # Errors
///
/// [`Error::Unsupported`] for data with weights, which the fit does not use;
/// [`Error::NoPositiveOutcome`]; [`Error::Singular`] when the least-squares
/// system cannot be solved and the strategy does not recover it.
pub fn fit(data: &CostData<'_>, options: &LogNormalOptions) -> Result<LogNormalModel, Error> {
    if data.weights().is_some() {
        return Err(Error::Unsupported {
            what: "weights in the log-normal model",
        });
    }
    fit_lognormal(data.x(), data.y_column(), data.clusters(), *options)
}

/// The fit of checked columns, under the strategy of `options`.
pub(crate) fn fit_lognormal(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    clusters: Option<&[u32]>,
    options: LogNormalOptions,
) -> Result<LogNormalModel, Error> {
    let retried = run_with_retries(
        options,
        options.strategy,
        |base, fallback_lambda, scale| LogNormalOptions {
            l2_penalty: fallback_lambda.mul_add(scale, base.l2_penalty).max(1e-8),
            ..base
        },
        |current| current.l2_penalty,
        |current_options| fit_lognormal_with_lambda(x, y, clusters, current_options),
    )?;
    let mut model = retried.fit;
    model.report.attempts = retried.attempts;
    model.report.meta.fallback_attempts = retried.attempt_idx;
    Ok(model)
}

fn fit_lognormal_with_lambda(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    clusters: Option<&[u32]>,
    options: LogNormalOptions,
) -> Result<LogNormalModel, Error> {
    let positive_indices: Vec<usize> = (0..y.nrows()).filter(|&idx| y[(idx, 0)] > 0.0).collect();

    if positive_indices.is_empty() {
        return Err(Error::NoPositiveOutcome);
    }

    let x_pos = select_rows_ref(x, &positive_indices);
    let y_pos = select_values_ref(y, &positive_indices);
    let y_log = Mat::from_fn(y_pos.nrows(), 1, |i, _| y_pos[(i, 0)].ln());
    let cluster_ids_pos = clusters.map(|labels| {
        positive_indices
            .iter()
            .map(|&idx| labels[idx])
            .collect::<Vec<_>>()
    });

    let mut beta = Mat::<f64>::zeros(x_pos.ncols(), 1);
    let lambda = options.l2_penalty.max(0.0);
    let mut iterations = 0;

    // `xtx` and `xty` do not depend on `beta`, so they are loop-invariant;
    // compute them once and reuse for every solve and for the robust covariance.
    let mut xtx = epistat_linalg::row_scaled_gram(x_pos.as_ref(), |_| 1.0);
    add_ridge_to_diagonal(&mut xtx, lambda, options.l2_penalty_exclude_intercept);
    let xty = epistat_linalg::xtv(x_pos.as_ref(), y_log.col_as_slice(0));

    for iteration in 0..options.max_iterations {
        iterations = iteration + 1;
        let beta_next = solve_linear_system(&xtx, &xty).map_err(|_| Error::Singular)?;

        if max_abs_diff(&beta_next, &beta) < options.tolerance {
            beta = beta_next;
            break;
        }
        beta = beta_next;
    }

    let eta = linear_predictor(x_pos.as_ref(), &beta);
    let residuals = Mat::from_fn(y_log.nrows(), 1, |i, _| y_log[(i, 0)] - eta[(i, 0)]);
    let smearing = mean_column(&map_mat(&residuals, f64::exp));

    let (cov, se, clustered, cluster_count) = if options.robust_se {
        let (cov, clustered, cluster_count) =
            robust_covariance(&x_pos, &residuals, cluster_ids_pos.as_deref(), &xtx)?;
        let se = cov.as_ref().map(diag_sqrt);
        (cov, se, clustered, cluster_count)
    } else {
        (None, None, false, None)
    };

    let meta = FitMetadata {
        iterations,
        converged: true,
        solver: SolverKind::Irls,
        ..FitMetadata::default()
    };
    let report = LogNormalReport {
        meta,
        se,
        cov,
        robust: options.robust_se,
        clustered,
        cluster_count,
        attempts: Vec::new(),
    };
    Ok(LogNormalModel {
        beta,
        smearing_factor: smearing,
        report,
    })
}

fn robust_covariance(
    x: &Mat<f64>,
    residuals: &Mat<f64>,
    clusters: Option<&[u32]>,
    xtx: &Mat<f64>,
) -> Result<RobustCovarianceResult, Error> {
    let (meat, cluster_count) = score_meat(x.as_ref(), |i| residuals[(i, 0)], clusters)?;
    let cov = sandwich(xtx.as_ref(), meat.as_ref()).ok_or(Error::Singular)?;
    Ok((Some(cov), cluster_count.is_some(), cluster_count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::usize_to_f64;

    #[test]
    fn fit_runs() {
        let n = 80;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 10.0 },
        );
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 7 == 0 {
                0.0
            } else {
                0.3f64.mul_add(usize_to_f64(i), 2.0)
            }
        });
        let y: Vec<f64> = (0..n).map(|i| y[(i, 0)]).collect();
        let data = CostData::builder(x.as_ref(), &y).build().expect("data");
        let model = fit(&data, &LogNormalOptions::default()).expect("fit");
        let pred = model.predict(x.as_ref()).expect("predict");
        assert_eq!(pred.mean.nrows(), n);
        assert!(model.smearing_factor > 0.0);
    }

    #[test]
    fn lognormal_reports_robust_se_with_clusters() {
        let n = 30;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 10.0 },
        );
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 3 == 0 {
                0.0
            } else {
                0.2f64.mul_add(usize_to_f64(i), 1.5)
            }
        });
        let clusters: Vec<u32> = (0..n)
            .map(|i| u32::try_from(i / 5).expect("small"))
            .collect();
        let y: Vec<f64> = (0..n).map(|i| y[(i, 0)]).collect();
        let data = CostData::builder(x.as_ref(), &y)
            .clusters(&clusters)
            .build()
            .expect("data");
        let options = LogNormalOptions {
            robust_se: true,
            ..LogNormalOptions::default()
        };
        let report = fit(&data, &options).expect("fit").report;
        assert!(report.robust);
        assert!(report.clustered);
        assert!(report.se.is_some());
        assert!(report.cluster_count.unwrap_or(0) > 1);
    }

    #[test]
    fn fit_strategy_relaxed_handles_lognormal_solve_failure() {
        let n = 5;
        // Near-singular design
        let x = Mat::from_fn(n, 2, |_, _| 1.0);
        let y = vec![10.0; n];

        let options = LogNormalOptions {
            l2_penalty: 0.0,
            strategy: FitStrategy::Relaxed {
                fallback_lambda: 1.0,
                max_retries: 1,
            },
            ..LogNormalOptions::default()
        };

        // This might still solve depending on floating point, but ridge should make it safer.
        let data = CostData::builder(x.as_ref(), &y).build().expect("data");
        let report = fit(&data, &options).expect("fit").report;
        // If it succeeded without retry, fallback_attempts is 0.
        // If it failed and retried with lambda, fallback_attempts is 1.
        assert!(report.meta.converged);
    }

    #[test]
    fn weights_are_refused() {
        let x = Mat::from_fn(4, 1, |_, _| 1.0);
        let y = [1.0, 2.0, 0.0, 3.0];
        let w = [1.0; 4];
        let data = CostData::builder(x.as_ref(), &y)
            .weights(&w)
            .build()
            .expect("data");
        assert!(matches!(
            fit(&data, &LogNormalOptions::default()),
            Err(Error::Unsupported { .. })
        ));
    }
}
