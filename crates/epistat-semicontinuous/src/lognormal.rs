//! Log-normal regression with Duan's smearing retransformation.
//!
//! # The model
//!
//! On the positive outcomes, `ln y = x' b + e` with errors of mean zero, fitted
//! by weighted least squares with an optional ridge. The mean of a positive
//! outcome on the original scale is not `exp(x' b)` but `exp(x' b) E[exp(e)]`,
//! and the smearing estimate of the second factor is the weighted mean of the
//! exponentiated residuals \[1\],
//!
//! ```text
//! s = sum_i w_i exp(ln y_i - x_i' b) / sum_i w_i,     E[y | y > 0, x] = s exp(x' b),
//! ```
//!
//! which needs no normality of the errors. The model is fitted to the positive
//! rows only, so its prediction is the mean of a positive outcome,
//! [`LogNormalPrediction::mean_positive`], not the mean of the outcome over
//! every row; [`compare`](crate::compare) scores it on the positive rows and
//! ranks it apart from the models of `E[y | x]`.
//! The log-likelihood reported for information criteria is the weighted
//! normal one of `ln y` on the positive rows at the maximum-likelihood
//! variance, with the Jacobian `-sum w_i ln y_i`.
//!
//! # Standard errors
//!
//! The model-based covariance is `s^2 (X' W X)^-1` with the residual variance
//! `s^2 = sum w_i e_i^2 / (n - k)` on the positive rows, as R's `summary.lm`;
//! the sandwich is HC0 on the positive rows, summed within clusters first
//! when cluster labels are given \[2\]. [`Variance`] chooses, the sandwich by
//! default for clustered data. Both tables refer the Wald statistics to
//! Student's t on `n - k` degrees of freedom, as `lmtest::coeftest` does for
//! a linear model.
//!
//! # References
//!
//! 1. Duan N. Smearing estimate: a nonparametric retransformation method. J Am
//!    Stat Assoc. 1983;78:605-10.
//! 2. Zeileis A. Object-oriented computation of sandwich estimators. J Stat
//!    Softw. 2006;16(9):1-16.

use epistat_core::{Coefficients, Covariance, EffectScale, Glance, Level, Tidy, VarianceKind};
use epistat_linalg::{Mat, MatRef};

use crate::covariance::{sandwich, score_meat};
use crate::data::CostData;
use crate::fit::{
    AttemptDiagnostics, FitMetadata, FitStrategy, Model, SolverKind, run_with_retries,
};
use crate::matrix_ops::{map_mat, select_rows_ref, select_values_ref};
use crate::tidy::{TableParts, table};
use crate::utils::{
    add_ridge_to_diagonal, column_mat, column_values, linear_predictor, max_abs_diff, mean_column,
    solve_linear_system,
};
use crate::variance::{Inference, Variance, residual_df, standard_errors, to_covariance};
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
    /// The covariance; by default the sandwich for clustered data or
    /// sampling weights and the model-based covariance otherwise.
    pub variance: Variance,
    /// What a fit that fails does.
    pub strategy: FitStrategy,
    #[doc(hidden)]
    pub _extensible: epistat_core::Extensible,
}

impl Default for LogNormalOptions {
    fn default() -> Self {
        Self {
            _extensible: epistat_core::Extensible::NEW,
            max_iterations: 1,
            tolerance: 1e-12,
            l2_penalty: 0.0,
            l2_penalty_exclude_intercept: true,
            variance: Variance::Auto,
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
///
/// [`Tidy`] gives its coefficient table on the log scale of the outcome:
/// `exp` of a coefficient is a ratio of geometric means of the positive
/// outcomes, which with a common smearing factor is also the ratio of their
/// means.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct LogNormalModel {
    /// The coefficients of `ln y`, one per column of the design.
    pub coefficients: Vec<f64>,
    /// Duan's smearing factor, the weighted mean of the exponentiated residuals.
    pub smearing_factor: f64,
    /// The covariance of the coefficients, `None` when the options asked for
    /// none ([`Variance::None`]).
    pub covariance: Option<Covariance>,
    /// The standard errors, the roots of the covariance's diagonal; `None`
    /// where the variance is undefined or negative, and for every coefficient
    /// without a covariance.
    pub se: Vec<Option<f64>>,
    /// The residual variance `sum w_i e_i^2 / (n - k)` of `ln y`, `None`
    /// without residual degrees of freedom.
    pub residual_variance: Option<f64>,
    /// The residual degrees of freedom, the positive rows with a positive
    /// weight (or the sum of their frequency weights) less the coefficients;
    /// `None` when not positive.
    pub df_residual: Option<f64>,
    /// The positive rows, which the model is fitted to.
    pub n: usize,
    /// The positive rows with a positive weight, which entered the fit.
    pub n_used: usize,
    /// Fit diagnostics.
    pub report: LogNormalReport,
}

/// The diagnostics of a log-normal fit.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct LogNormalReport {
    /// Fit metadata.
    pub meta: FitMetadata,
    /// The covariance computed, `None` for none.
    pub variance: Option<VarianceKind>,
    /// The number of clusters of a clustered sandwich, `None` otherwise.
    pub cluster_count: Option<usize>,
    /// The failed attempts before the one that succeeded, under
    /// [`FitStrategy::Relaxed`].
    pub attempts: Vec<AttemptDiagnostics>,
}

/// Log-normal predictions on the original scale, one entry per row of the
/// design.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct LogNormalPrediction {
    /// The smeared retransformation, the mean of a positive outcome
    /// `E[y | y > 0, x]`.
    pub mean_positive: Vec<f64>,
}

impl Tidy for LogNormalModel {
    type Error = Error;

    fn coefficients(&self, level: Level, variance: VarianceKind) -> Result<Coefficients, Error> {
        table(
            &TableParts {
                coefficients: &self.coefficients,
                covariance: self.covariance.as_ref(),
                computed: self.report.variance,
                effect: EffectScale::LogMeanRatio,
                t_df: self.df_residual,
            },
            level,
            variance,
        )
    }

    /// The covariance the fit computed, model-based when it computed none.
    fn default_variance(&self) -> VarianceKind {
        self.report.variance.unwrap_or(VarianceKind::ModelBased)
    }

    fn glance(&self) -> Glance {
        Glance::new(self.n, self.n_used, self.report.meta.converged)
            .with_iterations(Some(self.report.meta.iterations))
            .with_deviance(None, self.df_residual)
    }
}

impl LogNormalModel {
    /// The mean of the predicted positive outcome over the rows of `x`,
    /// without forming the predictions.
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
        check_columns(x, self.coefficients.len(), weights)?;
        Ok(crate::utils::mean_of_row_values(
            x,
            [&column_mat(&self.coefficients)],
            weights,
            |[eta]| self.smearing_factor * eta.exp(),
        ))
    }

    /// The log-likelihood of the positive outcomes of `data`, weighted by its
    /// weights: the normal log-likelihood of `ln y` at the maximum-likelihood
    /// variance with the Jacobian `-sum w_i ln y_i`. `None` when no outcome
    /// with a positive weight is positive.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the design of `data` does not have one column
    /// per coefficient.
    pub fn log_likelihood(&self, data: &CostData<'_>) -> Result<Option<f64>, Error> {
        check_columns(data.x(), self.coefficients.len(), None)?;
        let ll = log_likelihood(data.x(), data.y_column(), data.weights_column(), self);
        Ok((!ll.is_nan()).then_some(ll))
    }
}

impl Model for LogNormalModel {
    type Prediction = LogNormalPrediction;
    type Report = LogNormalReport;

    fn predict(&self, x: MatRef<'_, f64>) -> Result<Self::Prediction, Error> {
        check_columns(x, self.coefficients.len(), None)?;
        let eta = linear_predictor(x, &column_mat(&self.coefficients));
        Ok(LogNormalPrediction {
            mean_positive: (0..x.nrows())
                .map(|i| self.smearing_factor * eta[(i, 0)].exp())
                .collect(),
        })
    }

    fn report(&self) -> &Self::Report {
        &self.report
    }
}

/// The weight of row `i`, 1 without weights.
fn weight_at(weights: Option<MatRef<'_, f64>>, i: usize) -> f64 {
    weights.map_or(1.0, |w| w[(i, 0)])
}

/// The log-likelihood of the positive outcomes, NaN when none with a
/// positive weight is positive.
pub(crate) fn log_likelihood(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    weights: Option<MatRef<'_, f64>>,
    model: &LogNormalModel,
) -> f64 {
    let eta = linear_predictor(x, &column_mat(&model.coefficients));
    let mut sumsq = 0.0;
    let mut n_pos = 0.0;
    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        let wi = weight_at(weights, i);
        if yi > 0.0 && wi > 0.0 {
            let resid = yi.ln() - eta[(i, 0)];
            if weights.is_some() {
                sumsq = (wi * resid).mul_add(resid, sumsq);
                n_pos += wi;
            } else {
                sumsq = resid.mul_add(resid, sumsq);
                n_pos += 1.0;
            }
        }
    }
    if n_pos == 0.0 {
        return f64::NAN;
    }
    let sigma2 = (sumsq / n_pos).max(1e-12);
    let mut loglik = 0.0;
    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        let wi = weight_at(weights, i);
        if yi > 0.0 && wi > 0.0 {
            let resid = yi.ln() - eta[(i, 0)];
            let term = (-0.5f64).mul_add(
                std::f64::consts::TAU.ln() + sigma2.ln() + resid * resid / sigma2,
                -yi.ln(),
            );
            if weights.is_some() {
                loglik += wi * term;
            } else {
                loglik += term;
            }
        }
    }
    loglik
}

/// Fit log-normal regression with Duan's smearing to the positive outcomes
/// of `data`, by weighted least squares under its case weights.
///
/// # Errors
///
/// [`Error::Unsupported`] for a model-based covariance with sampling weights;
/// [`Error::NoPositiveOutcome`] when no outcome with a positive weight is
/// positive; [`Error::Singular`] when the least-squares system cannot be
/// solved and the strategy does not recover it.
pub fn fit(data: &CostData<'_>, options: &LogNormalOptions) -> Result<LogNormalModel, Error> {
    let inference = options.variance.resolve(data)?;
    fit_lognormal(
        data.x(),
        data.y_column(),
        data.weights_column(),
        data.clusters(),
        *options,
        inference,
    )
}

/// The fit of checked columns, under the strategy of `options`.
pub(crate) fn fit_lognormal(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    weights: Option<MatRef<'_, f64>>,
    clusters: Option<&[u32]>,
    options: LogNormalOptions,
    inference: Inference,
) -> Result<LogNormalModel, Error> {
    let retried = run_with_retries(
        options,
        options.strategy,
        |base, fallback_lambda, scale| LogNormalOptions {
            l2_penalty: fallback_lambda.mul_add(scale, base.l2_penalty).max(1e-8),
            ..base
        },
        |current| current.l2_penalty,
        |current_options| {
            fit_lognormal_with_lambda(x, y, weights, clusters, current_options, inference)
        },
    )?;
    let mut model = retried.fit;
    model.report.attempts = retried.attempts;
    model.report.meta.fallback_attempts = retried.attempt_idx;
    Ok(model)
}

fn fit_lognormal_with_lambda(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    weights: Option<MatRef<'_, f64>>,
    clusters: Option<&[u32]>,
    options: LogNormalOptions,
    inference: Inference,
) -> Result<LogNormalModel, Error> {
    // A positive row with weight 0 leaves the fit.
    let positive_indices: Vec<usize> = (0..y.nrows())
        .filter(|&idx| y[(idx, 0)] > 0.0 && weight_at(weights, idx) > 0.0)
        .collect();

    if positive_indices.is_empty() {
        return Err(Error::NoPositiveOutcome);
    }

    let x_pos = select_rows_ref(x, &positive_indices);
    let y_pos = select_values_ref(y, &positive_indices);
    let w_pos = weights.map(|w| select_values_ref(w, &positive_indices));
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
    // compute them once and reuse for every solve and for the covariance.
    let (mut xtx, xty) = match &w_pos {
        None => (
            epistat_linalg::row_scaled_gram(x_pos.as_ref(), |_| 1.0),
            epistat_linalg::xtv(x_pos.as_ref(), y_log.col_as_slice(0)),
        ),
        Some(w) => {
            let weighted: Vec<f64> = (0..w.nrows()).map(|i| w[(i, 0)] * y_log[(i, 0)]).collect();
            (
                epistat_linalg::row_scaled_gram(x_pos.as_ref(), |i| w[(i, 0)].sqrt()),
                epistat_linalg::xtv(x_pos.as_ref(), &weighted),
            )
        }
    };
    add_ridge_to_diagonal(&mut xtx, lambda, options.l2_penalty_exclude_intercept);

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
    let smearing = match &w_pos {
        None => mean_column(&map_mat(&residuals, f64::exp)),
        Some(w) => {
            let (mut sum, mut total) = (0.0, 0.0);
            for i in 0..w.nrows() {
                sum += w[(i, 0)] * residuals[(i, 0)].exp();
                total += w[(i, 0)];
            }
            sum / total
        }
    };

    let p = x_pos.ncols();
    let ones = Mat::from_fn(x_pos.nrows(), 1, |_, _| 1.0);
    let prior = w_pos.as_ref().map_or(ones.as_ref(), Mat::as_ref);
    let (n_used, df_residual) = residual_df(prior, inference.frequency, p);
    let residual_variance = df_residual.and_then(|df| {
        let mut sum = 0.0;
        for i in 0..prior.nrows() {
            sum += prior[(i, 0)] * residuals[(i, 0)] * residuals[(i, 0)];
        }
        let variance = sum / df;
        variance.is_finite().then_some(variance)
    });

    let (covariance, cluster_count) = match inference.variance {
        Some(VarianceKind::Robust) => {
            // The score of row i is `x_i w_i e_i`.
            let scores = Mat::from_fn(prior.nrows(), 1, |i, _| {
                weight_at(w_pos.as_ref().map(Mat::as_ref), i) * residuals[(i, 0)]
            });
            let (meat, cluster_count) = score_meat(
                x_pos.as_ref(),
                |i| scores[(i, 0)],
                cluster_ids_pos.as_deref(),
            )?;
            let cov = sandwich(xtx.as_ref(), meat.as_ref()).ok_or(Error::Singular)?;
            (Some(to_covariance(&cov, 1.0)), cluster_count)
        }
        Some(_) => {
            let identity = Mat::<f64>::identity(p, p);
            let inverse = solve_linear_system(&xtx, &identity).map_err(|_| Error::Singular)?;
            (
                residual_variance.map(|s2| to_covariance(&inverse, s2)),
                None,
            )
        }
        None => (None, None),
    };

    let meta = FitMetadata {
        iterations,
        converged: true,
        solver: SolverKind::Irls,
        ..FitMetadata::default()
    };
    let report = LogNormalReport {
        meta,
        variance: inference.variance,
        cluster_count,
        attempts: Vec::new(),
    };
    Ok(LogNormalModel {
        coefficients: column_values(&beta),
        smearing_factor: smearing,
        se: standard_errors(covariance.as_ref(), p),
        covariance,
        residual_variance,
        df_residual,
        n: (0..y.nrows()).filter(|&i| y[(i, 0)] > 0.0).count(),
        n_used,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::usize_to_f64;
    use epistat_core::CaseWeights;

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
        assert_eq!(pred.mean_positive.len(), n);
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
            variance: Variance::Sandwich,
            ..LogNormalOptions::default()
        };
        let model = fit(&data, &options).expect("fit");
        assert_eq!(model.report.variance, Some(VarianceKind::Robust));
        assert!(model.se.iter().all(Option::is_some));
        assert_eq!(model.report.cluster_count, Some(6));
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

    /// A design of an intercept and `i / 10` with positive outcomes that
    /// scatter around a log-linear trend, every seventh outcome zero.
    fn sample(n: usize) -> (Mat<f64>, Vec<f64>) {
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 10.0 },
        );
        let y = (0..n)
            .map(|i| {
                if i % 7 == 0 {
                    0.0
                } else {
                    0.1f64.mul_add(usize_to_f64(i), 1.0).exp() * [0.6, 1.0, 1.5][i % 3]
                }
            })
            .collect();
        (x, y)
    }

    #[test]
    fn frequency_weights_equal_duplicated_rows() {
        let n = 42;
        let (x, y) = sample(n);
        let counts: Vec<f64> = (0..n).map(|i| usize_to_f64(1 + i % 3)).collect();
        let data = CostData::builder(x.as_ref(), &y)
            .weights(CaseWeights::frequency(&counts).expect("weights"))
            .build()
            .expect("data");
        let weighted = fit(&data, &LogNormalOptions::default()).expect("weighted");

        let rows: Vec<usize> = (0..n)
            .flat_map(|i| std::iter::repeat_n(i, 1 + i % 3))
            .collect();
        let x_long = Mat::from_fn(rows.len(), 2, |i, j| x[(rows[i], j)]);
        let y_long: Vec<f64> = rows.iter().map(|&i| y[i]).collect();
        let long = CostData::builder(x_long.as_ref(), &y_long)
            .build()
            .expect("data");
        let duplicated = fit(&long, &LogNormalOptions::default()).expect("duplicated");

        // The two fits sum the same terms of X'WX and X'Wz, a weight of k
        // standing for k equal terms, so each sum differs by at most
        // gamma_m = m eps / (1 - m eps) relative to the sum of its absolute
        // terms, m the rows of the long data. A perturbation of relative size
        // e of a system of condition kappa moves its solution by at most
        // 2 kappa e relative to its norm, here bounded by the largest
        // coefficient times the 2 coefficients.
        let mut information = [[0.0_f64; 2]; 2];
        for &i in rows.iter().filter(|&&i| y[i] > 0.0) {
            for (a, row) in information.iter_mut().enumerate() {
                for (b, entry) in row.iter_mut().enumerate() {
                    *entry += x[(i, a)] * x[(i, b)];
                }
            }
        }
        let half_trace = f64::midpoint(information[0][0], information[1][1]);
        let det = information[0][0].mul_add(information[1][1], -information[0][1].powi(2));
        let root = half_trace.mul_add(half_trace, -det).sqrt();
        let kappa = (half_trace + root) / (half_trace - root);
        let m = usize_to_f64(rows.len());
        let gamma_m = m * f64::EPSILON / (1.0 - m * f64::EPSILON);
        let scale = weighted
            .coefficients
            .iter()
            .fold(0.0_f64, |acc, b| acc.max(b.abs()));
        let tolerance = 2.0 * kappa * gamma_m * 2.0 * scale;
        for (a, b) in weighted.coefficients.iter().zip(&duplicated.coefficients) {
            assert!((a - b).abs() <= tolerance, "{a} against {b}");
        }
        // Each residual moves by at most |x_i|_1 times the largest change of
        // a coefficient, |x_i|_1 <= 1 + 4.1, and exp turns that into the same
        // relative change of its term; the weighted mean adds gamma_m.
        let smearing_tolerance = (5.1 * tolerance + gamma_m) * duplicated.smearing_factor;
        assert!(
            (weighted.smearing_factor - duplicated.smearing_factor).abs() <= smearing_tolerance,
            "{} against {}",
            weighted.smearing_factor,
            duplicated.smearing_factor
        );
        // Frequency weights count sum(w) rows in the residual degrees of
        // freedom, as the duplicated data do.
        assert_eq!(weighted.df_residual, duplicated.df_residual);

        // Weights that are not all equal move the fit off the unweighted one:
        // they up-weight the rows whose noise factor is 1.5.
        let unweighted = fit(&data.without_weights(), &LogNormalOptions::default()).expect("fit");
        assert!(
            (unweighted.coefficients[0] - weighted.coefficients[0]).abs() > 1e3 * tolerance,
            "the weights did not enter the fit"
        );
    }

    #[test]
    fn the_table_carries_the_coefficients_and_standard_errors() {
        let (x, y) = sample(60);
        let data = CostData::builder(x.as_ref(), &y).build().expect("data");
        let model = fit(&data, &LogNormalOptions::default()).expect("fit");
        assert_eq!(model.report.variance, Some(VarianceKind::ModelBased));
        let level = Level::alpha(0.05).expect("level");
        let table = model
            .coefficients(level, VarianceKind::ModelBased)
            .expect("table");
        assert_eq!(table.effect(), EffectScale::LogMeanRatio);
        assert_eq!(table.len(), model.coefficients.len());
        for (j, term) in table.terms().iter().enumerate() {
            let estimate = term.estimate.expect("estimated");
            assert_eq!(estimate.estimate, model.coefficients[j]);
            assert_eq!(estimate.se, model.se[j]);
            assert!(estimate.se.is_some());
        }
        assert_eq!(
            model
                .coefficients(level, VarianceKind::Robust)
                .expect_err("no sandwich"),
            Error::NoCovariance {
                asked: VarianceKind::Robust
            }
        );
        let without = fit(
            &data,
            &LogNormalOptions {
                variance: Variance::None,
                ..LogNormalOptions::default()
            },
        )
        .expect("fit");
        assert!(without.se.iter().all(Option::is_none));
        assert_eq!(
            without
                .coefficients(level, VarianceKind::ModelBased)
                .expect_err("no covariance"),
            Error::NoCovariance {
                asked: VarianceKind::ModelBased
            }
        );
    }

    #[test]
    fn a_model_based_covariance_with_sampling_weights_is_refused() {
        let (x, y) = sample(40);
        let w: Vec<f64> = (0..40).map(|i| 1.0 + usize_to_f64(i % 4)).collect();
        let data = CostData::builder(x.as_ref(), &y)
            .weights(CaseWeights::sampling(&w).expect("weights"))
            .build()
            .expect("data");
        let model_based = LogNormalOptions {
            variance: Variance::ModelBased,
            ..LogNormalOptions::default()
        };
        assert!(matches!(
            fit(&data, &model_based),
            Err(Error::Unsupported { .. })
        ));
        let auto = fit(&data, &LogNormalOptions::default()).expect("auto");
        assert_eq!(auto.report.variance, Some(VarianceKind::Robust));
    }
}
