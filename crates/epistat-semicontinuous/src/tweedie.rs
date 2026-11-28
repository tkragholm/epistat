//! The Tweedie GLM with log link, fitted by quasi-likelihood.
//!
//! # The model
//!
//! The mean is `mu = exp(x' b)` and the variance `phi mu^p`, with the power
//! `p` in `[1, 2]`: at `p = 1` the quasi-Poisson, at `p = 2` the gamma, and in
//! between the compound Poisson-gamma, which has a mass at zero and a
//! continuous positive part \[1\]. Only the mean and variance are used, so the
//! fit is the solution of the quasi-score equations
//!
//! ```text
//! sum_i w_i x_i (y_i - mu_i) mu_i^(1 - p) = 0.
//! ```
//!
//! # Algorithm
//!
//! IRLS on a design whose non-intercept columns are centred, with working
//! weights `w_i mu_i^(2 - p)` and working response `eta_i + (y_i - mu_i) / mu_i`.
//! At `p = 2` the working weights are the prior weights and the factorisation
//! of `X' W X` is computed once. A step that does not lower the deviance
//!
//! ```text
//! D = 2 sum_i w_i [ y_i^(2-p) / ((1-p)(2-p)) - y_i mu_i^(1-p) / (1-p) + mu_i^(2-p) / (2-p) ],
//! ```
//!
//! with its limits at `p = 1` and `p = 2`, is halved, up to 12 times. The fit
//! stops when no coefficient moves by more than `tolerance` or the deviance
//! changes by less than `tolerance` relative to `max(|D|, 1)`. The
//! quasi-likelihood used for information criteria is `-D / 2` \[2\].
//!
//! # Standard errors
//!
//! The model-based covariance is `phi (X' diag(w mu^(2-p)) X)^-1` with the
//! Pearson dispersion `phi = sum w_i (y_i - mu_i)^2 / mu_i^p / (n - k)`, as R's
//! `summary.glm` reports it. The sandwich has that inverse at `phi = 1` as
//! its bread and meat from the quasi-score residual
//! `w_i (y_i - mu_i) mu_i^(1-p)`, summed within clusters first when cluster
//! labels are given \[3\]; [`Variance`] chooses, the sandwich by default for
//! clustered data. The fit agrees with R's `glm` with
//! `statmod::tweedie(var.power = p, link.power = 0)` and with that sandwich
//! computed in R (`tests/tweedie_reference.rs`).
//!
//! # References
//!
//! 1. Jorgensen B. The Theory of Dispersion Models. London: Chapman & Hall;
//!    1997.
//! 2. McCullagh P, Nelder JA. Generalized Linear Models. 2nd ed. London:
//!    Chapman & Hall; 1989. Chapter 9.
//! 3. Zeileis A. Object-oriented computation of sandwich estimators. J Stat
//!    Softw. 2006;16(9):1-16.

use epistat_linalg::{Mat, MatRef};

use crate::covariance::{sandwich, score_meat};
use crate::data::CostData;
use crate::fit::{
    AttemptDiagnostics, FitMetadata, FitStrategy, Model, SolverKind, run_with_retries,
};
use crate::matrix_ops::{
    center_beta, center_columns_into, map_mat, max_abs_linear_predictor, uncenter_beta,
    weighted_column_means,
};
use crate::tidy::{TableParts, table};
use crate::utils::{
    add_ridge_to_diagonal, cached_factor, col_slice, fill_constant_irls_weights, linear_predictor,
    matvec_into, max_abs_diff, solve_linear_system_ref, weighted_xtx_ref,
    weighted_xtz_ref_with_buffer,
};
use crate::utils::{column_mat, column_values, solve_linear_system};
use crate::variance::{
    Inference, Variance, pearson_dispersion, residual_df, standard_errors, to_covariance,
};
use crate::{Error, check_columns};
use epistat_core::{Coefficients, Covariance, EffectScale, Glance, Level, Tidy, VarianceKind};

pub(crate) const LINEAR_PREDICTOR_CLIP: f64 = 30.0;
pub(crate) const BACKTRACKING_STEPS: usize = 12;

/// The options of [`fit`].
///
/// `max_iterations` and `tolerance` are R's `maxit` and `epsilon` of
/// `glm.control`, at 50 and 1e-6 rather than R's 25 and 1e-8: the fit also
/// stops on the change of the coefficients, which reaches 1e-6 when the
/// deviance has converged far beyond it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TweedieOptions {
    /// The most IRLS iterations.
    pub max_iterations: usize,
    /// Convergence tolerance on the largest change of a coefficient and on
    /// the relative change of the deviance.
    pub tolerance: f64,
    /// Lower bound on the working weight `mu^(2 - p)`.
    pub min_weight: f64,
    /// L2 (ridge) penalty on the sum scale of the weights.
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

impl Default for TweedieOptions {
    fn default() -> Self {
        Self {
            _extensible: epistat_core::Extensible::NEW,
            max_iterations: 50,
            tolerance: 1e-6,
            min_weight: 1e-6,
            l2_penalty: 0.0,
            l2_penalty_exclude_intercept: true,
            variance: Variance::Auto,
            strategy: FitStrategy::Strict,
        }
    }
}

/// Column buffers of a Tweedie fit, reusable from one fit to the next.
///
/// A caller that runs many fits on one thread keeps one scratch and passes it to
/// [`fit`], so the fits share one centred
/// design and one set of working columns. The buffers carry nothing from one fit to the
/// next: a result does not depend on what the scratch held before. A fit of another
/// shape reallocates them.
#[derive(Debug)]
pub struct TweedieScratch {
    centered: Mat<f64>,
    weights: Mat<f64>,
    eta: Mat<f64>,
    mu: Mat<f64>,
    z: Mat<f64>,
    search_eta: Mat<f64>,
    search_mu: Mat<f64>,
    weight_buffer: Mat<f64>,
    xtz_buffer: Vec<f64>,
}

impl Default for TweedieScratch {
    fn default() -> Self {
        Self {
            centered: Mat::zeros(0, 0),
            weights: Mat::zeros(0, 0),
            eta: Mat::zeros(0, 0),
            mu: Mat::zeros(0, 0),
            z: Mat::zeros(0, 0),
            search_eta: Mat::zeros(0, 0),
            search_mu: Mat::zeros(0, 0),
            weight_buffer: Mat::zeros(0, 0),
            xtz_buffer: Vec::new(),
        }
    }
}

impl TweedieScratch {
    fn fit_shape(&mut self, n: usize) {
        for column in [
            &mut self.weights,
            &mut self.eta,
            &mut self.mu,
            &mut self.z,
            &mut self.search_eta,
            &mut self.search_mu,
            &mut self.weight_buffer,
        ] {
            if column.nrows() != n || column.ncols() != 1 {
                *column = Mat::zeros(n, 1);
            }
        }
    }

    fn centred_shape(&mut self, n: usize, p: usize) {
        if self.centered.nrows() != n || self.centered.ncols() != p {
            self.centered = Mat::zeros(n, p);
        }
    }
}

/// A fitted Tweedie model.
///
/// [`Tidy`] gives its coefficient table of log mean ratios; the model-based
/// table refers the Wald statistics to Student's t on the residual degrees
/// of freedom, as R's `summary.glm` does, the sandwich table to the normal.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TweedieModel {
    /// The coefficients, one per column of the design.
    pub coefficients: Vec<f64>,
    /// The Tweedie power `p`.
    pub power: f64,
    /// The covariance of the coefficients, `None` when the options asked for
    /// none ([`Variance::None`]).
    pub covariance: Option<Covariance>,
    /// The standard errors, the roots of the covariance's diagonal; `None`
    /// where the variance is undefined or negative, and for every coefficient
    /// without a covariance.
    pub se: Vec<Option<f64>>,
    /// The Pearson dispersion `sum w_i (y_i - mu_i)^2 / mu_i^p / (n - k)`,
    /// `None` without residual degrees of freedom or when no covariance was
    /// computed.
    pub dispersion: Option<f64>,
    /// The residual degrees of freedom, the rows with a positive weight (or
    /// the sum of frequency weights) less the coefficients; `None` when not
    /// positive.
    pub df_residual: Option<f64>,
    /// The rows of the data.
    pub n: usize,
    /// The rows with a positive weight, which entered the fit.
    pub n_used: usize,
    /// Fit diagnostics.
    pub report: TweedieReport,
}

/// The diagnostics of a Tweedie fit.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TweedieReport {
    /// Standardized fit metadata.
    pub meta: FitMetadata,
    /// The covariance computed, `None` for none.
    pub variance: Option<VarianceKind>,
    /// The number of clusters of a clustered sandwich, `None` otherwise.
    pub cluster_count: Option<usize>,
    /// The failed attempts before the one that succeeded, under
    /// [`FitStrategy::Relaxed`].
    pub attempts: Vec<AttemptDiagnostics>,
}

/// Tweedie predictions, one entry per row of the design.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TweediePrediction {
    /// The predicted mean, `E[y | x]`.
    pub mean: Vec<f64>,
}

impl Tidy for TweedieModel {
    type Error = Error;

    fn coefficients(&self, level: Level, variance: VarianceKind) -> Result<Coefficients, Error> {
        table(
            &TableParts {
                coefficients: &self.coefficients,
                covariance: self.covariance.as_ref(),
                computed: self.report.variance,
                effect: EffectScale::LogMeanRatio,
                t_df: (variance == VarianceKind::ModelBased)
                    .then_some(self.df_residual)
                    .flatten(),
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

impl TweedieModel {
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
        check_columns(x, self.coefficients.len(), weights)?;
        Ok(crate::utils::mean_of_row_values(
            x,
            [&column_mat(&self.coefficients)],
            weights,
            |[eta]| exp_clamped(eta),
        ))
    }

    /// The deviance of `data` at the fitted means, weighted by the weights of
    /// `data`: R's `deviance` of the fit when `data` is the data it was
    /// fitted to.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the design of `data` does not have one column
    /// per coefficient.
    pub fn deviance(&self, data: &CostData<'_>) -> Result<f64, Error> {
        let mu = column_mat(&self.predict(data.x())?.mean);
        Ok(weighted_deviance(
            data.y_column(),
            &mu,
            self.power,
            data.weights_column(),
        ))
    }

    /// The quasi-log-likelihood `-D / 2` of `data`, with `D` the unweighted
    /// deviance, as the information criteria of [`compare`](crate::compare)
    /// use it.
    ///
    /// # Errors
    ///
    /// As [`deviance`](Self::deviance).
    pub fn quasi_log_likelihood(&self, data: &CostData<'_>) -> Result<f64, Error> {
        let mu = column_mat(&self.predict(data.x())?.mean);
        Ok(quasi_log_likelihood(data.y_column(), &mu, self.power))
    }
}

impl Model for TweedieModel {
    type Prediction = TweediePrediction;
    type Report = TweedieReport;

    fn predict(&self, x: MatRef<'_, f64>) -> Result<Self::Prediction, Error> {
        check_columns(x, self.coefficients.len(), None)?;
        let eta = linear_predictor(x, &column_mat(&self.coefficients));
        Ok(TweediePrediction {
            mean: (0..x.nrows()).map(|i| exp_clamped(eta[(i, 0)])).collect(),
        })
    }

    fn report(&self) -> &Self::Report {
        &self.report
    }
}

/// The Tweedie fit of `x` and `y` without weights or clusters, with the
/// covariance `inference` asks for.
pub(crate) fn fit_tweedie(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    power: f64,
    options: TweedieOptions,
    inference: Inference,
) -> Result<TweedieModel, Error> {
    let mut scratch = TweedieScratch::default();
    fit_tweedie_weighted(
        x,
        y,
        None,
        None,
        power,
        options,
        inference,
        None,
        &mut scratch,
    )
}

/// The Tweedie fit of checked columns, under the strategy of `options`.
#[expect(
    clippy::too_many_arguments,
    reason = "the columns of the data, the power, the options, the warm start and the buffers"
)]
fn fit_tweedie_weighted(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    sample_weights: Option<MatRef<'_, f64>>,
    clusters: Option<&[u32]>,
    power: f64,
    options: TweedieOptions,
    inference: Inference,
    // Warm-start coefficients for the IRLS. A bootstrap replicate is a perturbation of
    // the full sample, so seeding from the full-sample fit converges in far fewer
    // iterations to the same optimum. `None` cold-starts from the intercept-only model.
    initial_beta: Option<&Mat<f64>>,
    scratch: &mut TweedieScratch,
) -> Result<TweedieModel, Error> {
    if !(1.0..=2.0).contains(&power) {
        return Err(crate::error::domain("power", power, "[1, 2]"));
    }
    let retried = run_with_retries(
        options,
        options.strategy,
        |base, fallback_lambda, scale| TweedieOptions {
            l2_penalty: fallback_lambda.mul_add(scale, base.l2_penalty).max(1e-8),
            ..base
        },
        |current| current.l2_penalty,
        |current_options| {
            fit_tweedie_with_lambda(
                TweedieFitRequest {
                    x,
                    y,
                    sample_weights,
                    clusters,
                    power,
                    options: current_options,
                    inference,
                    lambda: current_options.l2_penalty,
                    initial_beta_override: initial_beta,
                },
                scratch,
            )
        },
    )?;

    let mut model = retried.fit;
    model.report.attempts = retried.attempts;
    model.report.meta.fallback_attempts = retried.attempt_idx;
    Ok(model)
}

#[derive(Clone, Copy)]
struct TweedieFitRequest<'a> {
    x: MatRef<'a, f64>,
    y: MatRef<'a, f64>,
    sample_weights: Option<MatRef<'a, f64>>,
    clusters: Option<&'a [u32]>,
    power: f64,
    options: TweedieOptions,
    inference: Inference,
    lambda: f64,
    initial_beta_override: Option<&'a Mat<f64>>,
}

fn fit_tweedie_with_lambda(
    request: TweedieFitRequest<'_>,
    scratch: &mut TweedieScratch,
) -> Result<TweedieModel, Error> {
    let TweedieFitRequest {
        x,
        y,
        sample_weights,
        clusters,
        power,
        options,
        inference,
        lambda,
        initial_beta_override,
    } = request;

    // Center the covariate columns for numerical conditioning (see the module note in
    // `matrix_ops`). The IRLS below runs entirely on the centered design `cx`, so the
    // intercept stays near `ln(mean y)` and η stays in the data-supported range instead of
    // drifting into the `exp_clamped` saturation ceiling. `col_means` is `None` when the
    // design has no usable intercept in column 0, in which case `cx == x` (no centering).
    // The converged coefficients are un-centered back to the raw-x scale before
    // `finalize_fit`, which computes the standard errors and covariance, and everything
    // downstream, on the original design.
    let col_means = weighted_column_means(x, sample_weights);
    scratch.fit_shape(x.nrows());
    if let Some(means) = &col_means {
        scratch.centred_shape(x.nrows(), x.ncols());
        center_columns_into(&mut scratch.centered, x, means);
    }
    // Column buffers reused across iterations and fits, so a bootstrap's thousands of
    // parallel fits do not churn the allocator. `eta`/`mu`/`z` are the linear predictor,
    // mean and working response; `search_*` are line-search scratch; `weight_buffer`
    // holds the per-iteration weight on the variable-weight path; `constant_weights` the
    // weight of the constant-weight path.
    let TweedieScratch {
        centered,
        weights: constant_weights,
        eta,
        mu,
        z,
        search_eta,
        search_mu,
        weight_buffer,
        xtz_buffer,
    } = scratch;
    let cx: MatRef<'_, f64> = match &col_means {
        Some(_) => centered.as_ref(),
        None => x,
    };

    let mut beta = match (initial_beta_override, &col_means) {
        // Warm-start coefficients arrive on the raw-x scale; map them onto the centered
        // scale so IRLS resumes from the same fitted model.
        (Some(init), Some(means)) => center_beta(init, means),
        (Some(init), None) => init.clone(),
        (None, _) => initial_beta(cx, y, sample_weights, options.min_weight),
    };

    // For power == 2 (gamma) the Fisher-scoring weight `μ^(2-power) == 1`, so the
    // information matrix `X'WX` (W = the fixed sample weights) is invariant across
    // iterations. Build and factor it once here; each iteration then only recomputes
    // the working-response RHS and back-substitutes — removing the O(n·p²) Gram build
    // and O(p³) factorization from every pass but the first. Exact, not an
    // approximation: the cached factor is the true information matrix at every iterate,
    // so the β-path is identical to rebuilding it each iteration.
    let constant_weight = if (2.0 - power).abs() < 1e-12 {
        fill_constant_irls_weights(constant_weights, sample_weights);
        let mut xtwx = weighted_xtx_ref(cx, constant_weights.as_ref())?;
        if lambda > 0.0 {
            add_ridge_to_diagonal(&mut xtwx, lambda, options.l2_penalty_exclude_intercept);
        }
        Some(cached_factor(&xtwx).map_err(|_| Error::Singular)?)
    } else {
        None
    };

    let n = cx.nrows();

    // Deviance of `beta` when the line search of the previous iteration already left
    // that beta's linear predictor and mean in `search_eta`/`search_mu`.
    let mut carried_deviance: Option<f64> = None;

    for iteration in 0..options.max_iterations {
        let current_deviance = if let Some(deviance) = carried_deviance.take() {
            std::mem::swap(eta, search_eta);
            std::mem::swap(mu, search_mu);
            deviance
        } else {
            matvec_into(eta, cx, &beta);
            for (m, &e) in mu.col_as_slice_mut(0).iter_mut().zip(eta.col_as_slice(0)) {
                *m = exp_clamped(e);
            }
            weighted_deviance(y, mu, power, sample_weights)
        };

        // Working response `z = η + (y - μ)/μ`; independent of the IRLS weights.
        for (((slot, &e), &m), &yi) in z
            .col_as_slice_mut(0)
            .iter_mut()
            .zip(eta.col_as_slice(0))
            .zip(mu.col_as_slice(0))
            .zip(col_slice(y, 0))
        {
            *slot = e + (yi - m) / m;
        }

        let beta_candidate = if let Some(factor) = &constant_weight {
            // Constant-weight path: reuse the cached factor; only the RHS changes.
            let xtw_rhs =
                weighted_xtz_ref_with_buffer(cx, constant_weights.as_ref(), z.as_ref(), xtz_buffer);
            if let Ok(candidate) = factor.solve(xtw_rhs.as_ref()) {
                candidate
            } else {
                // Pathological RHS: stabilized solve on the (constant) matrix.
                let mut xtwx = weighted_xtx_ref(cx, constant_weights.as_ref())?;
                if lambda > 0.0 {
                    add_ridge_to_diagonal(&mut xtwx, lambda, options.l2_penalty_exclude_intercept);
                }
                solve_with_stabilization(xtwx.as_ref(), xtw_rhs.as_ref())?
            }
        } else {
            // Variable-weight path (1 < power < 2): the weight depends on μ, so the
            // information matrix must be rebuilt and factored each iteration.
            for i in 0..n {
                let base_weight = sample_weight_at(sample_weights, i);
                let w = mu[(i, 0)].powf(2.0 - power).max(options.min_weight);
                weight_buffer[(i, 0)] = base_weight * w;
            }
            let mut xtwx = weighted_xtx_ref(cx, weight_buffer.as_ref())?;
            if lambda > 0.0 {
                add_ridge_to_diagonal(&mut xtwx, lambda, options.l2_penalty_exclude_intercept);
            }
            let xtw_rhs =
                weighted_xtz_ref_with_buffer(cx, weight_buffer.as_ref(), z.as_ref(), xtz_buffer);
            solve_with_stabilization(xtwx.as_ref(), xtw_rhs.as_ref())?
        };

        let (beta_next, dev_next, scratch_holds_beta_next) = backtracking_update(
            cx,
            y,
            sample_weights,
            &beta,
            &beta_candidate,
            current_deviance,
            power,
            search_eta,
            search_mu,
        );

        let beta_converged = max_abs_diff(&beta_next, &beta) < options.tolerance;
        let dev_converged = relative_change(current_deviance, dev_next) < options.tolerance;
        if beta_converged || dev_converged {
            // Un-center back to the raw-x scale so SE/cov, predict and the marginal
            // standardisation all operate on the original design.
            let beta_raw = match &col_means {
                Some(means) => uncenter_beta(&beta_next, means),
                None => beta_next,
            };
            // The line search left this beta's linear predictor in `search_eta`, so the
            // saturation check needs no product with the raw design.
            let known_max_eta = scratch_holds_beta_next.then(|| {
                search_eta
                    .col_as_slice(0)
                    .iter()
                    .fold(0.0_f64, |acc, value| acc.max(value.abs()))
            });
            return finalize_fit(
                FinalFit {
                    x,
                    y,
                    sample_weights,
                    clusters,
                    power,
                    options,
                    inference,
                    lambda,
                },
                &beta_raw,
                iteration + 1,
                known_max_eta,
            );
        }
        if scratch_holds_beta_next {
            carried_deviance = Some(dev_next);
        }
        beta = beta_next;
    }

    Err(Error::NotConverged {
        iterations: options.max_iterations,
    })
}

fn initial_beta(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    sample_weights: Option<MatRef<'_, f64>>,
    min_weight: f64,
) -> Mat<f64> {
    let mut beta = Mat::<f64>::zeros(x.ncols(), 1);
    if x.ncols() == 0 || y.nrows() == 0 {
        return beta;
    }
    let mut mean_y = 0.0;
    let mut weight_sum = 0.0;
    for i in 0..y.nrows() {
        let weight = sample_weight_at(sample_weights, i);
        mean_y = weight.mul_add(y[(i, 0)], mean_y);
        weight_sum += weight;
    }
    mean_y /= weight_sum.max(1.0);
    beta[(0, 0)] = mean_y.max(min_weight).ln();
    beta
}

fn sample_weight_at(sample_weights: Option<MatRef<'_, f64>>, row: usize) -> f64 {
    sample_weights.map_or(1.0, |weights| weights[(row, 0)])
}

/// The data and choices a converged fit is finished with.
#[derive(Clone, Copy)]
struct FinalFit<'a> {
    x: MatRef<'a, f64>,
    y: MatRef<'a, f64>,
    sample_weights: Option<MatRef<'a, f64>>,
    clusters: Option<&'a [u32]>,
    power: f64,
    options: TweedieOptions,
    inference: Inference,
    lambda: f64,
}

fn finalize_fit(
    request: FinalFit<'_>,
    beta_final: &Mat<f64>,
    iterations: usize,
    known_max_eta: Option<f64>,
) -> Result<TweedieModel, Error> {
    let FinalFit {
        x,
        y,
        sample_weights,
        clusters,
        power,
        options,
        inference,
        lambda,
    } = request;
    // Saturation guard: a converged fit whose linear predictor still reaches the
    // `exp_clamped` ceiling has distorted, non-multiplicative predictions (the marginal
    // ratio collapses toward 1 and the counterfactual means blow up). Report it as
    // non-convergence instead of a degenerate "ok" fit — under the Relaxed strategy this
    // retries with stronger ridge (which shrinks η back into range); if it still saturates
    // the fit fails loudly rather than exporting a meaningless estimate. Centering keeps η
    // well inside the range for well-posed data, so this only fires on genuine divergence.
    let max_eta = known_max_eta.unwrap_or_else(|| max_abs_linear_predictor(x, beta_final));
    if max_eta >= LINEAR_PREDICTOR_CLIP {
        return Err(Error::Saturated);
    }
    let p = x.ncols();
    let ones = Mat::from_fn(x.nrows(), 1, |_, _| 1.0);
    let prior = sample_weights.unwrap_or(ones.as_ref());
    let (n_used, df_residual) = residual_df(prior, inference.frequency, p);
    // The information matrix and fitted means feed only the covariance, so a
    // fit that does not ask for one skips both.
    let (covariance, dispersion, cluster_count) = match inference.variance {
        Some(kind) => {
            let eta = linear_predictor(x, beta_final);
            let mu = map_mat(&eta, exp_clamped);
            let weights = Mat::from_fn(mu.nrows(), 1, |i, _| {
                let base_weight = sample_weight_at(sample_weights, i);
                let w = mu[(i, 0)].powf(2.0 - power).max(options.min_weight);
                base_weight * w
            });
            let mut xtwx = weighted_xtx_ref(x, weights.as_ref())?;
            if lambda > 0.0 {
                add_ridge_to_diagonal(&mut xtwx, lambda, options.l2_penalty_exclude_intercept);
            }
            let dispersion = pearson_dispersion(
                prior,
                |i| (y[(i, 0)] - mu[(i, 0)]) / mu[(i, 0)].powf(0.5 * power),
                df_residual,
            );
            if kind == VarianceKind::Robust {
                let (cov, cluster_count) =
                    robust_covariance(x, y, &mu, sample_weights, power, clusters, &xtwx)?;
                (Some(to_covariance(&cov, 1.0)), dispersion, cluster_count)
            } else {
                let identity = Mat::<f64>::identity(p, p);
                let inverse = solve_linear_system(&xtwx, &identity).map_err(|_| Error::Singular)?;
                let covariance = dispersion.map(|phi| to_covariance(&inverse, phi));
                (covariance, dispersion, None)
            }
        }
        None => (None, None, None),
    };
    let meta = FitMetadata {
        iterations,
        converged: true,
        solver: SolverKind::Irls,
        ..FitMetadata::default()
    };
    let report = TweedieReport {
        meta,
        variance: inference.variance,
        cluster_count,
        attempts: Vec::new(),
    };
    Ok(TweedieModel {
        coefficients: column_values(beta_final),
        power,
        se: standard_errors(covariance.as_ref(), p),
        covariance,
        dispersion,
        df_residual,
        n: x.nrows(),
        n_used,
        report,
    })
}

/// Fit the Tweedie GLM with log link and power `power` by quasi-likelihood,
/// from the coefficients `warm` when given.
///
/// The weights of `data` are case weights, a row with weight 0 leaving the
/// fit; the covariance is the one `options.variance` chooses, clustered when
/// `data` has clusters and the sandwich is chosen. `scratch` lends the column
/// buffers of the fit, so that a thread fitting many replicates of one sample
/// allocates them once; the result does not depend on it.
///
/// # Errors
///
/// [`Error::Input`] for a power outside `[1, 2]`; [`Error::Unsupported`] for a
/// model-based covariance with sampling weights; [`Error::NotConverged`],
/// [`Error::Saturated`] or [`Error::Singular`] when the fit fails and the
/// strategy does not recover it.
pub fn fit(
    data: &CostData<'_>,
    power: f64,
    options: &TweedieOptions,
    warm: Option<&[f64]>,
    scratch: Option<&mut TweedieScratch>,
) -> Result<TweedieModel, Error> {
    let inference = options.variance.resolve(data)?;
    let mut own = None;
    let scratch = match scratch {
        Some(scratch) => scratch,
        None => own.insert(TweedieScratch::default()),
    };
    let warm = warm.map(column_mat);
    fit_tweedie_weighted(
        data.x(),
        data.y_column(),
        data.weights_column(),
        data.clusters(),
        power,
        *options,
        inference,
        warm.as_ref(),
        scratch,
    )
}

/// The unweighted Tweedie deviance of `y` at the means `mu`, NaN for a power
/// outside `[1, 2]`.
pub(crate) fn deviance(y: MatRef<'_, f64>, mu: &Mat<f64>, power: f64) -> f64 {
    if !power.is_finite() || !(1.0..=2.0).contains(&power) {
        return f64::NAN;
    }
    weighted_deviance(y, mu, power, None)
}

fn weighted_deviance(
    y: MatRef<'_, f64>,
    mu: &Mat<f64>,
    power: f64,
    sample_weights: Option<MatRef<'_, f64>>,
) -> f64 {
    // Closed-form endpoint for p = 1 (Poisson deviance limit).
    if (power - 1.0).abs() < 1e-12 {
        let mut result = 0.0;
        for i in 0..y.nrows() {
            let yi = y[(i, 0)].max(0.0);
            let mui = mu[(i, 0)].max(1e-12);
            let weight = sample_weight_at(sample_weights, i);
            if yi == 0.0 {
                result = (weight * 2.0).mul_add(mui, result);
            } else {
                result = (weight * 2.0).mul_add(yi * (yi / mui).ln() - (yi - mui), result);
            }
        }
        return result;
    }

    // Closed-form endpoint for p = 2 (Gamma deviance limit).
    if (power - 2.0).abs() < 1e-12 {
        let weight_column = sample_weights.map(|weights| col_slice(weights, 0));
        let mut result = 0.0;
        for (i, (&yi, &mu_i)) in col_slice(y, 0).iter().zip(mu.col_as_slice(0)).enumerate() {
            let mui = mu_i.max(1e-12);
            let weight = weight_column.map_or(1.0, |weights| weights[i]);
            if yi <= 0.0 {
                return f64::INFINITY;
            }
            result = (weight * 2.0).mul_add(((yi - mui) / mui) - (yi / mui).ln(), result);
        }
        return result;
    }

    let mut deviance = 0.0;
    for i in 0..y.nrows() {
        let yi = y[(i, 0)];
        let mui = mu[(i, 0)].max(1e-12);
        if yi == 0.0 {
            deviance +=
                sample_weight_at(sample_weights, i) * 2.0 * mui.powf(2.0 - power) / (2.0 - power);
        } else {
            let term1 = yi.powf(2.0 - power) / ((1.0 - power) * (2.0 - power));
            let term2 = yi * mui.powf(1.0 - power) / (1.0 - power);
            let term3 = mui.powf(2.0 - power) / (2.0 - power);
            deviance = (sample_weight_at(sample_weights, i) * 2.0)
                .mul_add(term1 - term2 + term3, deviance);
        }
    }
    deviance
}

/// The quasi-log-likelihood `-D / 2` of the unweighted deviance.
pub(crate) fn quasi_log_likelihood(y: MatRef<'_, f64>, mu: &Mat<f64>, power: f64) -> f64 {
    let dev = deviance(y, mu, power);
    if dev.is_finite() { -0.5 * dev } else { dev }
}

/// Sandwich covariance from the Tweedie log-link quasi-score.
///
/// Row `i` contributes `u_i = x_i * w_i * (y_i - mu_i) * (dmu/deta) / V(mu_i)` with
/// prior weight `w_i`, `V(mu) = mu^power` and `dmu/deta = mu`, so the score residual
/// is `w_i * (y_i - mu_i) * mu_i^(1 - power)`. The clustered and unclustered meat
/// are built from this one residual.
fn robust_covariance(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    mu: &Mat<f64>,
    sample_weights: Option<MatRef<'_, f64>>,
    power: f64,
    clusters: Option<&[u32]>,
    xtwx: &Mat<f64>,
) -> Result<(Mat<f64>, Option<usize>), Error> {
    let score_residual = |i: usize| {
        sample_weight_at(sample_weights, i)
            * (y[(i, 0)] - mu[(i, 0)])
            * mu[(i, 0)].powf(1.0 - power)
    };
    let (meat, cluster_count) = score_meat(x, score_residual, clusters)?;
    let cov = stabilized_sandwich(xtwx, &meat)?;
    Ok((cov, cluster_count))
}

/// Returns the accepted step's `(beta, deviance, scratch_holds_beta)`. The last is true
/// when `scratch_eta` and `scratch_mu` hold that beta's linear predictor and mean, so the
/// caller can carry them into its next iteration. The deviance is the one already
/// computed for the returned beta inside the line search, so the caller reuses it
/// instead of recomputing `x*beta`, `exp`, and the deviance a second time per IRLS
/// iteration (the value is identical — same beta, same deviance function).
fn backtracking_update(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    sample_weights: Option<MatRef<'_, f64>>,
    beta_current: &Mat<f64>,
    beta_candidate: &Mat<f64>,
    current_deviance: f64,
    power: f64,
    scratch_eta: &mut Mat<f64>,
    scratch_mu: &mut Mat<f64>,
) -> (Mat<f64>, f64, bool) {
    let mut step = 1.0_f64;
    let mut proposal = beta_candidate.clone();
    let mut best_beta = beta_candidate.clone();
    let mut best_deviance = f64::INFINITY;
    for _ in 0..BACKTRACKING_STEPS {
        if (step - 1.0).abs() < f64::EPSILON {
            proposal.clone_from(beta_candidate);
        } else {
            blend_betas_into(&mut proposal, beta_current, beta_candidate, step);
        }
        let dev = trial_deviance(
            x,
            y,
            sample_weights,
            &proposal,
            power,
            scratch_eta,
            scratch_mu,
        );
        if dev.is_finite() && dev <= current_deviance {
            return (proposal, dev, true);
        }
        if dev.is_finite() && dev < best_deviance {
            best_deviance = dev;
            best_beta.clone_from(&proposal);
        }
        step *= 0.5;
    }
    // No trial reduced the deviance. Report best_beta's deviance — recomputed when the
    // search never recorded a finite one (all steps non-finite), so the returned value
    // matches what the caller would have computed for best_beta.
    let dev = if best_deviance.is_finite() {
        best_deviance
    } else {
        trial_deviance(
            x,
            y,
            sample_weights,
            &best_beta,
            power,
            scratch_eta,
            scratch_mu,
        )
    };
    (best_beta, dev, false)
}

/// Deviance at `beta`, reusing the caller's `scratch_eta`/`scratch_mu` columns so the
/// line search allocates nothing per trial step.
fn trial_deviance(
    x: MatRef<'_, f64>,
    y: MatRef<'_, f64>,
    sample_weights: Option<MatRef<'_, f64>>,
    beta: &Mat<f64>,
    power: f64,
    scratch_eta: &mut Mat<f64>,
    scratch_mu: &mut Mat<f64>,
) -> f64 {
    matvec_into(scratch_eta, x, beta);
    for (m, &e) in scratch_mu
        .col_as_slice_mut(0)
        .iter_mut()
        .zip(scratch_eta.col_as_slice(0))
    {
        *m = exp_clamped(e);
    }
    weighted_deviance(y, scratch_mu, power, sample_weights)
}

fn blend_betas_into(
    output: &mut Mat<f64>,
    beta_current: &Mat<f64>,
    beta_candidate: &Mat<f64>,
    step: f64,
) {
    for row in 0..output.nrows() {
        let current = beta_current[(row, 0)];
        output[(row, 0)] = (beta_candidate[(row, 0)] - current).mul_add(step, current);
    }
}

pub(crate) fn relative_change(previous: f64, next: f64) -> f64 {
    if !previous.is_finite() || !next.is_finite() {
        return f64::INFINITY;
    }
    (next - previous).abs() / previous.abs().max(1.0)
}

/// The gamma deviance of [`weighted_deviance`] on slices of positive outcomes.
pub(crate) fn gamma_deviance(y: &[f64], mu: &[f64], weights: &[f64]) -> f64 {
    let mut result = 0.0;
    for ((&yi, &mu_i), &weight) in y.iter().zip(mu).zip(weights) {
        let mui = mu_i.max(1e-12);
        result = (weight * 2.0).mul_add(((yi - mui) / mui) - (yi / mui).ln(), result);
    }
    result
}

pub(crate) fn exp_clamped(value: f64) -> f64 {
    value
        .clamp(-LINEAR_PREDICTOR_CLIP, LINEAR_PREDICTOR_CLIP)
        .exp()
}

/// Solve `lhs * x = rhs`, retrying with growing diagonal jitter from 1e-10 when the
/// unregularised solve fails.
fn solve_with_stabilization(lhs: MatRef<'_, f64>, rhs: MatRef<'_, f64>) -> Result<Mat<f64>, Error> {
    if let Ok(solution) = solve_linear_system_ref(lhs, rhs) {
        return Ok(solution);
    }

    let mut stabilized = Mat::from_fn(lhs.nrows(), lhs.ncols(), |row, col| lhs[(row, col)]);
    let dim = lhs.nrows().min(lhs.ncols());
    let mut jitter = 1e-10;
    for _ in 0..8 {
        for idx in 0..dim {
            stabilized[(idx, idx)] = lhs[(idx, idx)] + jitter;
        }
        if let Ok(solution) = solve_linear_system_ref(stabilized.as_ref(), rhs) {
            return Ok(solution);
        }
        jitter *= 10.0;
    }
    Err(Error::Singular)
}

/// [`sandwich`] with the same growing diagonal jitter from 1e-10 as
/// [`solve_with_stabilization`], added to the information matrix when the plain one fails.
fn stabilized_sandwich(information: &Mat<f64>, meat: &Mat<f64>) -> Result<Mat<f64>, Error> {
    if let Some(cov) = sandwich(information.as_ref(), meat.as_ref()) {
        return Ok(cov);
    }
    let mut stabilized = information.clone();
    let dim = information.nrows().min(information.ncols());
    let mut jitter = 1e-10;
    for _ in 0..8 {
        for idx in 0..dim {
            stabilized[(idx, idx)] = information[(idx, idx)] + jitter;
        }
        if let Some(cov) = sandwich(stabilized.as_ref(), meat.as_ref()) {
            return Ok(cov);
        }
        jitter *= 10.0;
    }
    Err(Error::Singular)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::usize_to_f64;
    use epistat_core::CaseWeights;

    /// The fit of `x` and the outcome column `y` with the public entry
    /// point, clustered when `clusters` is given.
    fn fit_tweedie(
        x: MatRef<'_, f64>,
        y: MatRef<'_, f64>,
        clusters: Option<&[u32]>,
        power: f64,
        options: TweedieOptions,
    ) -> Result<TweedieModel, Error> {
        let y: Vec<f64> = (0..y.nrows()).map(|i| y[(i, 0)]).collect();
        let builder = CostData::builder(x, &y);
        let data = match clusters {
            Some(clusters) => builder.clusters(clusters),
            None => builder,
        }
        .build()?;
        fit(&data, power, &options, None, None)
    }

    #[test]
    fn fit_tweedie_runs() {
        let n = 100;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 20.0 },
        );
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 5 == 0 {
                0.0
            } else {
                0.2f64.mul_add(usize_to_f64(i), 2.0)
            }
        });
        let model =
            fit_tweedie(x.as_ref(), y.as_ref(), None, 1.5, TweedieOptions::default()).expect("fit");
        let pred = model.predict(x.as_ref()).expect("predict");
        assert_eq!(pred.mean.len(), n);
        let dev = deviance(y.as_ref(), &column_mat(&pred.mean), 1.5);
        assert!(dev.is_finite());
    }

    // Mean of the predictions (the g-computation counterfactual mean).
    fn mean_of(pred: &[f64]) -> f64 {
        pred.iter().sum::<f64>() / pred.len() as f64
    }

    // Set the exposure column (index 1) to a constant `value` for the whole design — the
    // counterfactual design the runner feeds the marginal standardisation.
    fn set_exposure(x: &Mat<f64>, value: f64) -> Mat<f64> {
        Mat::from_fn(
            x.nrows(),
            x.ncols(),
            |i, j| if j == 1 { value } else { x[(i, j)] },
        )
    }

    #[test]
    fn gamma_marginal_ratio_equals_exposure_coefficient() {
        // Regression guard for the counterfactual-mean explosion (run_010726 fully_adjusted:
        // exposed/unexposed means ~35M DKK, marginal_ratio 1.007 vs exp(β_exposed)=4.85).
        // For a log-link gamma the g-computation ratio is EXACTLY exp(β_exposed) — unless the
        // fit saturates the η-clamp, which collapses the ratio toward 1. Design: intercept,
        // binary exposure (col 1), and a large-offset covariate (col 2 ~1000) that makes the
        // UNCENTERED fit ill-conditioned. With centering the identity must hold tightly and
        // the means must stay on the data scale.
        let n = 400;
        let x = Mat::from_fn(n, 3, |i, j| match j {
            0 => 1.0,
            1 => f64::from(u32::try_from(i % 2).unwrap()),
            _ => 1000.0 + usize_to_f64(i % 50),
        });
        // log μ = 9 + 0.02·(cov-1025) + 1.3·exposed, with deterministic within-cell scatter.
        let y = Mat::from_fn(n, 1, |i, _| {
            let cov = 1000.0 + usize_to_f64(i % 50);
            let exposed = f64::from(u32::try_from(i % 2).unwrap());
            let lin = 1.3f64.mul_add(exposed, 0.02f64.mul_add(cov - 1025.0, 9.0));
            let perturb = [0.7, 1.0, 1.4][i % 3];
            lin.exp() * perturb
        });

        let model =
            fit_tweedie(x.as_ref(), y.as_ref(), None, 2.0, TweedieOptions::default()).expect("fit");

        let mean_exposed = mean_of(
            &model
                .predict(set_exposure(&x, 1.0).as_ref())
                .expect("predict")
                .mean,
        );
        let mean_unexposed = mean_of(
            &model
                .predict(set_exposure(&x, 0.0).as_ref())
                .expect("predict")
                .mean,
        );
        let ratio = mean_exposed / mean_unexposed;
        let expected = model.coefficients[1].exp();

        assert!(
            (ratio - expected).abs() / expected < 1e-6,
            "marginal ratio {ratio} != exp(beta_exposed) {expected} (clamp saturation?)"
        );
        // ...and the counterfactual means are on the data scale, not blown up past the clamp.
        let max_y = (0..n).map(|i| y[(i, 0)]).fold(0.0_f64, f64::max);
        assert!(
            mean_exposed.is_finite() && mean_exposed < 100.0 * max_y,
            "exposed counterfactual mean exploded: {mean_exposed} (max y {max_y})"
        );
    }

    #[test]
    fn gamma_fit_conditions_large_offset_covariate() {
        // Centering must keep η in the data-supported range for a covariate whose raw scale
        // (~1000) would otherwise ill-condition the fit and let η drift toward the ±30 clamp.
        // The fitted mean tracks the data mean and no row's |η| approaches the ceiling.
        let n = 300;
        let x = Mat::from_fn(n, 2, |i, j| {
            if j == 0 {
                1.0
            } else {
                1000.0 + usize_to_f64(i % 40)
            }
        });
        let y = Mat::from_fn(n, 1, |i, _| {
            let cov = 1000.0 + usize_to_f64(i % 40);
            0.01f64.mul_add(cov - 1020.0, 8.5).exp() * [0.8, 1.0, 1.2][i % 3]
        });

        let model =
            fit_tweedie(x.as_ref(), y.as_ref(), None, 2.0, TweedieOptions::default()).expect("fit");

        let max_abs_eta = max_abs_linear_predictor(x.as_ref(), &column_mat(&model.coefficients));
        assert!(
            max_abs_eta < LINEAR_PREDICTOR_CLIP,
            "linear predictor reached the clamp: max|eta| = {max_abs_eta}"
        );
        let mean_pred = mean_of(&model.predict(x.as_ref()).expect("predict").mean);
        let mean_y = (0..n).map(|i| y[(i, 0)]).sum::<f64>() / n as f64;
        assert!(
            (mean_pred - mean_y).abs() / mean_y < 0.05,
            "fitted mean {mean_pred} does not track data mean {mean_y}"
        );
    }

    #[test]
    fn gamma_fit_fails_loudly_when_outcome_forces_clamp() {
        // Saturation guard: an outcome so large that its log-mean exceeds the ±30 clamp
        // (exp(34) ≈ 5.8e14) cannot be represented below the ceiling. The fit must report
        // an error rather than silently returning a degenerate model whose predictions are
        // pinned at exp(30) — the failure mode that produced the meaningless fully_adjusted
        // headline. Strict strategy (the default) surfaces it directly with no ridge retry.
        let n = 50;
        let x = Mat::from_fn(n, 1, |_, _| 1.0); // intercept-only
        let y = Mat::from_fn(n, 1, |_, _| 34.0_f64.exp());
        let result = fit_tweedie(x.as_ref(), y.as_ref(), None, 2.0, TweedieOptions::default());
        assert!(
            result.is_err(),
            "a fit that can only converge at a saturated linear predictor must fail loudly"
        );
    }

    #[test]
    fn tweedie_reports_robust_se_with_clusters() {
        let n = 40;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 20.0 },
        );
        let y = Mat::from_fn(n, 1, |i, _| {
            if i % 5 == 0 {
                0.0
            } else {
                0.2f64.mul_add(usize_to_f64(i), 2.0)
            }
        });
        let clusters: Vec<u32> = (0..n)
            .map(|i| u32::try_from(i / 4).expect("small"))
            .collect();
        let options = TweedieOptions {
            variance: Variance::Sandwich,
            ..TweedieOptions::default()
        };
        let model =
            fit_tweedie(x.as_ref(), y.as_ref(), Some(&clusters), 1.5, options).expect("fit");
        let pred = model.predict(x.as_ref()).expect("predict");
        assert_eq!(pred.mean.len(), n);
        assert_eq!(model.report.variance, Some(VarianceKind::Robust));
        assert_eq!(model.report.cluster_count, Some(10));
        assert!(model.se.iter().all(Option::is_some));
    }

    #[test]
    fn robust_se_is_scale_invariant_for_log_link_gamma() {
        // Regression guard: the sandwich score must use the WORKING residual (y-μ)/μ,
        // not the raw response residual (y-μ). Under a log link, scaling y by a constant
        // shifts only the intercept and leaves every coefficient's robust SE unchanged.
        // The old bug used the raw residual, so SE(c*y) ≈ c*SE(y) — which on cost-scale
        // outcomes (μ ~ 1e4–1e5 DKK) inflated the gamma SEs by ~1e4×.
        let n = 60;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { usize_to_f64(i % 6) });
        // Strictly positive, associated with the covariate, with within-level scatter so
        // the residuals (and hence the robust SE) are non-degenerate.
        let make_y = |scale: f64| {
            Mat::from_fn(n, 1, |i, _| {
                let lin = 0.4f64.mul_add(usize_to_f64(i % 6), 1.0);
                let perturb = 0.4f64.mul_add(usize_to_f64(i % 3) - 1.0, 1.0); // 0.6, 1.0, 1.4
                lin.exp() * perturb * scale
            })
        };
        let clusters: Vec<u32> = (0..n)
            .map(|i| u32::try_from(i / 4).expect("small"))
            .collect();
        let slope_se = |scale: f64| {
            let options = TweedieOptions {
                variance: Variance::Sandwich,
                ..TweedieOptions::default()
            };
            fit_tweedie(
                x.as_ref(),
                make_y(scale).as_ref(),
                Some(&clusters),
                2.0,
                options,
            )
            .expect("fit")
            .se[1]
                .expect("se")
        };
        let se_base = slope_se(1.0);
        let se_scaled = slope_se(1000.0);
        // Scale-invariant to <1% (the raw-residual bug gives ~1000x here).
        assert!(
            (se_scaled - se_base).abs() / se_base < 0.01,
            "robust SE not scale-invariant: base={se_base} scaled={se_scaled}"
        );
        // ...and a sane log-scale magnitude, not cost-scale.
        assert!(se_base < 1.0, "robust SE implausibly large: {se_base}");
    }

    #[test]
    fn fit_strategy_relaxed_handles_tweedie_non_convergence() {
        let n = 10;
        let x = Mat::from_fn(n, 2, |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) });
        let y: Vec<f64> = (0..n).map(|i| if i == 0 { 0.0 } else { 1.0 }).collect();

        let options = TweedieOptions {
            max_iterations: 1, // Force early failure
            strategy: FitStrategy::Relaxed {
                fallback_lambda: 1.0,
                max_retries: 2,
            },
            ..TweedieOptions::default()
        };

        let data = CostData::builder(x.as_ref(), &y).build().expect("data");
        match fit(&data, 1.5, &options, None, None) {
            Ok(TweedieModel { report, .. }) => {
                assert!(report.meta.fallback_attempts > 0);
                assert_ne!(report.attempts.len(), 0);
            }
            Err(Error::NotConverged { .. } | Error::Singular) => {
                // On tiny pathological samples, relaxed retries may still fail.
            }
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn power_two_factor_once_recovers_group_log_means() {
        // Gamma (power == 2) log-link with intercept + one binary covariate: the MLE is
        // exactly the per-group log-means. This exercises the cached-factor
        // (constant-weight) IRLS path end to end and pins it to the correct optimum.
        let n = 200;
        let x = Mat::from_fn(n, 2, |i, j| {
            if j == 0 {
                1.0
            } else {
                f64::from(u32::from(i % 2 == 1))
            }
        });
        // Group 0 (even rows) mean 10, group 1 (odd rows) mean 30.
        let y = Mat::from_fn(n, 1, |i, _| if i % 2 == 0 { 10.0 } else { 30.0 });

        let model = fit_tweedie(x.as_ref(), y.as_ref(), None, 2.0, TweedieOptions::default())
            .expect("power-2 fit");

        // Intercept -> ln(10); slope -> ln(30) - ln(10) = ln(3).
        assert!(
            (model.coefficients[0] - 10f64.ln()).abs() < 1e-4,
            "intercept = {}",
            model.coefficients[0]
        );
        assert!(
            (model.coefficients[1] - 3f64.ln()).abs() < 1e-4,
            "slope = {}",
            model.coefficients[1]
        );
        assert!(model.report.meta.converged);
    }

    #[test]
    fn power_two_factor_once_respects_sample_weights() {
        // With constant weights the cached factor is X'diag(w)X; up-weighting one group
        // must still recover that group's log-mean (the weights cancel within a group),
        // confirming the cached factor carries the prior weights correctly.
        let n = 120;
        let x = Mat::from_fn(n, 2, |i, j| {
            if j == 0 {
                1.0
            } else {
                f64::from(u32::from(i % 2 == 1))
            }
        });
        let y: Vec<f64> = (0..n)
            .map(|i| if i % 2 == 0 { 5.0 } else { 50.0 })
            .collect();
        let w: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 3.0 } else { 1.0 }).collect();
        let data = CostData::builder(x.as_ref(), &y)
            .weights(CaseWeights::new(&w).expect("weights"))
            .build()
            .expect("data");

        let model =
            fit(&data, 2.0, &TweedieOptions::default(), None, None).expect("weighted power-2 fit");
        assert_eq!(model.deviance(&data).map(f64::is_finite), Ok(true));
        assert!((model.coefficients[0] - 5f64.ln()).abs() < 1e-4);
        assert!((model.coefficients[1] - (50f64.ln() - 5f64.ln())).abs() < 1e-4);
    }

    #[test]
    fn warm_start_matches_cold_start_and_converges_faster() {
        let n = 60;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 20.0 },
        );
        let y: Vec<f64> = (0..n).map(|i| 1.0 + usize_to_f64(i)).collect();
        let data = CostData::builder(x.as_ref(), &y).build().expect("data");
        let options = TweedieOptions::default();

        let cold = fit(&data, 2.0, &options, None, None).expect("cold fit");
        let mut scratch = TweedieScratch::default();
        let warm = fit(
            &data,
            2.0,
            &options,
            Some(&cold.coefficients),
            Some(&mut scratch),
        )
        .expect("warm fit");
        let (cold_report, warm_report) = (&cold.report, &warm.report);

        // Same optimum to the IRLS convergence tolerance. (IRLS stops on a small STEP,
        // so cold and warm both land within tolerance of the unique MLE but not at the
        // identical point — they agree to ~the step tolerance, far below any reported
        // precision, which is exactly the warm-start guarantee.)
        for (i, (c, w)) in cold.coefficients.iter().zip(&warm.coefficients).enumerate() {
            assert!(
                (c - w).abs() < 1e-3,
                "coefficient {i} diverged: cold={c} warm={w}"
            );
        }
        // Starting at the converged solution converges essentially immediately, and
        // never in more iterations than the cold start.
        assert!(warm_report.meta.iterations <= cold_report.meta.iterations);
        assert!(warm_report.meta.iterations <= 2);
    }

    fn tidy_sample() -> (Mat<f64>, Vec<f64>) {
        let n = 60;
        let x = Mat::from_fn(
            n,
            2,
            |i, j| if j == 0 { 1.0 } else { usize_to_f64(i) / 20.0 },
        );
        let y = (0..n)
            .map(|i| {
                if i % 5 == 0 {
                    0.0
                } else {
                    0.2f64.mul_add(usize_to_f64(i), 2.0) * [0.7, 1.0, 1.3][i % 3]
                }
            })
            .collect();
        (x, y)
    }

    #[test]
    fn the_table_carries_the_coefficients_and_standard_errors() {
        let (x, y) = tidy_sample();
        let data = CostData::builder(x.as_ref(), &y).build().expect("data");
        let model = fit(&data, 1.5, &TweedieOptions::default(), None, None).expect("fit");
        assert_eq!(model.report.variance, Some(VarianceKind::ModelBased));
        assert_eq!(model.default_variance(), VarianceKind::ModelBased);
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
            1.5,
            &TweedieOptions {
                variance: Variance::None,
                ..TweedieOptions::default()
            },
            None,
            None,
        )
        .expect("fit");
        assert!(without.covariance.is_none());
        assert!(without.se.iter().all(Option::is_none));
        assert_eq!(without.coefficients, model.coefficients);
        assert_eq!(
            without.tidy().expect_err("no covariance"),
            Error::NoCovariance {
                asked: VarianceKind::ModelBased
            }
        );
    }

    #[test]
    fn a_model_based_covariance_with_sampling_weights_is_refused() {
        let (x, y) = tidy_sample();
        let w: Vec<f64> = (0..y.len()).map(|i| 1.0 + usize_to_f64(i % 4)).collect();
        let data = CostData::builder(x.as_ref(), &y)
            .weights(CaseWeights::sampling(&w).expect("weights"))
            .build()
            .expect("data");
        let model_based = TweedieOptions {
            variance: Variance::ModelBased,
            ..TweedieOptions::default()
        };
        assert!(matches!(
            fit(&data, 1.5, &model_based, None, None),
            Err(Error::Unsupported { .. })
        ));
        let auto = fit(&data, 1.5, &TweedieOptions::default(), None, None).expect("auto");
        assert_eq!(auto.report.variance, Some(VarianceKind::Robust));
    }
}
