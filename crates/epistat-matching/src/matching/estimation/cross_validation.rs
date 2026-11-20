//! Choosing the elastic-net penalty by cross-validation, as `cv.glmnet` does.
//!
//! # Method
//!
//! Let `w` be the prior weights and `w~ = w / sum(w)`. The elastic net at
//! penalty `lambda` has the objective of [`super::logistic`]. The path of
//! `lambda` starts at the smallest value that sets every penalised coefficient
//! to zero. With the penalised columns scaled as the fit scales them, `x~_j`,
//! the null model has the weighted mean `y_bar` as its probability and the
//! gradient of the log-likelihood at the null model is
//! `g_j = sum_i w~_i x~_ij (y_i - y_bar)`, so
//!
//! ```text
//! lambda_max = max_j |g_j| / (f_j * max(alpha, 1e-3))
//! ```
//!
//! over the columns with penalty factor `f_j > 0`. The `max(alpha, 1e-3)` keeps
//! the ridge path finite. The path has `nlambda` values
//! `lambda_k = lambda_max * r^(k / (nlambda - 1))`, `k = 0, ..., nlambda - 1`,
//! with `r` the smallest-to-largest ratio, 1e-4 when there are more rows than
//! penalised columns and 1e-2 otherwise.
//!
//! The rows are dealt to `nfolds` folds in turn and the assignment is shuffled
//! with a stream from `epistat-rng`. For each fold the model is fitted on the
//! other folds along the whole path, each fit warm-started from the previous
//! one, with the columns scaled by the training rows. The held-out rows get the
//! binomial deviance `-2 [y ln p + (1 - y) ln(1 - p)]`, with `p` clipped to
//! `[1e-5, 1 - 1e-5]`. For fold `k` and `lambda`, `d_k` is the weighted mean of
//! those deviances and `W_k` the sum of the weights. The curve and its standard
//! error are
//!
//! ```text
//! cvm    = sum_k W_k d_k / sum_k W_k
//! cvsd   = sqrt( sum_k W_k (d_k - cvm)^2 / sum_k W_k / (nfolds - 1) )
//! ```
//!
//! `lambda.min` is the largest `lambda` at which `cvm` equals its minimum.
//! `lambda.1se` is the largest `lambda` with `cvm` at most
//! `cvm + cvsd` taken at `lambda.min`. The final model is the fit on all rows at
//! the chosen `lambda`.
//!
//! The path is not cut short. `glmnet` stops a path early when the deviance
//! explained passes 0.999 or changes by less than 1e-5 of the null deviance
//! between two values, so on such data its path is shorter than `nlambda`.
//!
//! # References
//!
//! 1. Friedman J, Hastie T, Tibshirani R. Regularization paths for generalized
//!    linear models via coordinate descent. J Stat Softw. 2010;33(1):1-22.
//!    doi:10.18637/jss.v033.i01
//! 2. Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning.
//!    2nd ed. Springer; 2009. Section 7.10 on cross-validation and the one
//!    standard error rule.

use super::EstimationError;
use super::config::{ElasticNetCvConfig, LambdaRule};
use super::encoding::CovariateMatrix;
use super::logistic::{LogisticFit, NetProblem, NetSettings, penalty_factors};
use epistat_rng::Stream;

/// The probability clip of the cross-validated deviance.
const DEVIANCE_CLIP: f64 = 1e-5;

/// The cross-validation curve and the choice made from it.
#[derive(Debug, Clone)]
pub struct CrossValidation {
    /// The fold of each row, from 0, in the order of the design matrix: the
    /// anchors, then the candidates.
    pub fold_ids: Vec<usize>,
    /// The penalties of the path, largest first.
    pub lambda: Vec<f64>,
    /// The mean cross-validated deviance at each penalty (`cvm`).
    pub deviance: Vec<f64>,
    /// Its standard error over the folds (`cvsd`).
    pub deviance_se: Vec<f64>,
    /// Position in `lambda` of `lambda.min`.
    pub index_min: usize,
    /// Position in `lambda` of `lambda.1se`.
    pub index_one_se: usize,
    /// The position that was used.
    pub index_chosen: usize,
    /// The number of fits, over the folds and the full data, that did not meet
    /// their stopping rule.
    pub unconverged_fits: usize,
}

impl CrossValidation {
    /// `lambda.min`.
    #[must_use]
    pub fn lambda_min(&self) -> f64 {
        self.lambda[self.index_min]
    }

    /// `lambda.1se`.
    #[must_use]
    pub fn lambda_one_se(&self) -> f64 {
        self.lambda[self.index_one_se]
    }

    /// The penalty that was used.
    #[must_use]
    pub fn lambda_chosen(&self) -> f64 {
        self.lambda[self.index_chosen]
    }
}

/// Assigns `n` rows to `nfolds` folds, dealt in turn and then shuffled.
fn assign_folds(n: usize, nfolds: usize, seed: u64) -> Vec<usize> {
    let mut folds: Vec<usize> = (0..n).map(|row| row % nfolds).collect();
    Stream::seeded(seed).shuffle(&mut folds);
    folds
}

fn binomial_deviance(y: f64, eta: f64) -> f64 {
    let p = (1.0 / (1.0 + (-eta).exp())).clamp(DEVIANCE_CLIP, 1.0 - DEVIANCE_CLIP);
    -2.0 * (y * p.ln() + (1.0 - y) * (1.0 - p).ln())
}

/// Elastic-net logistic regression with the penalty chosen by cross-validation.
pub(super) fn fit_elastic_net_cv(
    matrix: &CovariateMatrix,
    labels: &[f64],
    weights: Option<&[f64]>,
    config: &ElasticNetCvConfig,
) -> Result<LogisticFit, EstimationError> {
    let n = matrix.n_rows;
    if labels.len() != n || weights.is_some_and(|w| w.len() != n) {
        return Err(EstimationError::LinearSystemSolveFailed);
    }
    let settings = NetSettings {
        alpha: config.alpha,
        tolerance: config.tolerance,
        max_iter: config.max_iter,
        probability_clip: config.probability_clip,
        standardize: config.standardize,
    };
    settings.validate()?;
    let penalty = penalty_factors(matrix);
    let penalised = penalty.iter().filter(|f| **f > 0.0).count();
    if !(0.0..=1.0).contains(&config.alpha)
        || config.nfolds < 3
        || config.nfolds > n
        || config.nlambda < 2
        || config
            .lambda_min_ratio
            .is_some_and(|r| !(r > 0.0 && r < 1.0))
    {
        return Err(EstimationError::InvalidCrossValidation);
    }
    let weight = |row: usize| weights.map_or(1.0, |w| w[row]);

    let all_rows: Vec<usize> = (0..n).collect();
    let full = NetProblem::new(
        matrix,
        &all_rows,
        labels,
        weights,
        &penalty,
        config.standardize,
    );

    // The path starts where the null model's gradient first allows a nonzero
    // penalised coefficient.
    let total: f64 = (0..n).map(weight).sum();
    if !(total.is_finite() && total > 0.0) {
        return Err(EstimationError::InvalidCrossValidation);
    }
    let y_bar = (0..n).map(|row| weight(row) * labels[row]).sum::<f64>() / total;
    let mut lambda_max = 0.0_f64;
    for col in 0..matrix.n_cols {
        if penalty[col] <= 0.0 {
            continue;
        }
        let gradient = (0..n)
            .map(|row| weight(row) * full.values[row * matrix.n_cols + col] * (labels[row] - y_bar))
            .sum::<f64>()
            / total;
        lambda_max = lambda_max.max(gradient.abs() / penalty[col]);
    }
    lambda_max /= config.alpha.max(1e-3);
    if !(lambda_max.is_finite() && lambda_max > 0.0) {
        return Err(EstimationError::InvalidCrossValidation);
    }
    let ratio = config
        .lambda_min_ratio
        .unwrap_or(if n > penalised { 1e-4 } else { 1e-2 });
    let steps = config.nlambda - 1;
    let lambda: Vec<f64> = (0..config.nlambda)
        .map(|k| lambda_max * ratio.powf(k as f64 / steps as f64))
        .collect();

    let fold_ids = assign_folds(n, config.nfolds, config.seed);
    let mut unconverged = 0usize;

    // Path on all rows, kept as scaled coefficients to warm-start the final fit.
    let mut full_path: Vec<Vec<f64>> = Vec::with_capacity(lambda.len());
    let mut start: Option<Vec<f64>> = None;
    for &value in &lambda {
        let fit = full.fit(&settings, value, &penalty, start.as_deref())?;
        unconverged += usize::from(!fit.converged);
        start = Some(fit.coefficients.clone());
        full_path.push(fit.coefficients);
    }

    // Held-out deviance, by fold and penalty.
    let mut fold_deviance = vec![vec![0.0_f64; lambda.len()]; config.nfolds];
    let mut fold_weight = vec![0.0_f64; config.nfolds];
    for fold in 0..config.nfolds {
        let train: Vec<usize> = (0..n).filter(|&row| fold_ids[row] != fold).collect();
        let held: Vec<usize> = (0..n).filter(|&row| fold_ids[row] == fold).collect();
        let has_both = [false, true].iter().all(|&positive| {
            train
                .iter()
                .any(|&row| (labels[row] > 0.5) == positive && weight(row) > 0.0)
        });
        let held_weight: f64 = held.iter().map(|&row| weight(row)).sum();
        if !has_both || held_weight <= 0.0 {
            return Err(EstimationError::InvalidCrossValidation);
        }
        fold_weight[fold] = held_weight;
        let problem = NetProblem::new(
            matrix,
            &train,
            labels,
            weights,
            &penalty,
            config.standardize,
        );
        let mut start: Option<Vec<f64>> = None;
        for (k, &value) in lambda.iter().enumerate() {
            let fit = problem.fit(&settings, value, &penalty, start.as_deref())?;
            unconverged += usize::from(!fit.converged);
            let beta = problem.unscale(&fit.coefficients);
            start = Some(fit.coefficients);
            let sum: f64 = held
                .iter()
                .map(|&row| {
                    let eta: f64 = matrix.row(row).iter().zip(&beta).map(|(x, b)| x * b).sum();
                    weight(row) * binomial_deviance(labels[row], eta)
                })
                .sum();
            fold_deviance[fold][k] = sum / held_weight;
        }
    }

    let fold_total: f64 = fold_weight.iter().sum();
    let mut deviance = Vec::with_capacity(lambda.len());
    let mut deviance_se = Vec::with_capacity(lambda.len());
    for k in 0..lambda.len() {
        let mean = (0..config.nfolds)
            .map(|f| fold_weight[f] * fold_deviance[f][k])
            .sum::<f64>()
            / fold_total;
        let spread = (0..config.nfolds)
            .map(|f| {
                let d = fold_deviance[f][k] - mean;
                fold_weight[f] * d * d
            })
            .sum::<f64>()
            / fold_total;
        deviance.push(mean);
        deviance_se.push((spread / (config.nfolds as f64 - 1.0)).sqrt());
    }
    if deviance.iter().any(|d| !d.is_finite()) {
        return Err(EstimationError::InvalidCrossValidation);
    }

    // The path runs from the largest penalty down, so the first position that
    // meets a condition is the largest penalty that does.
    let minimum = deviance.iter().copied().fold(f64::INFINITY, f64::min);
    let index_min = deviance
        .iter()
        .position(|d| *d <= minimum)
        .unwrap_or_default();
    let bound = deviance[index_min] + deviance_se[index_min];
    let index_one_se = deviance
        .iter()
        .position(|d| *d <= bound)
        .unwrap_or(index_min);
    let index_chosen = match config.rule {
        LambdaRule::Min => index_min,
        LambdaRule::OneStandardError => index_one_se,
    };

    let chosen = lambda[index_chosen];
    let fit = full.fit(&settings, chosen, &penalty, Some(&full_path[index_chosen]))?;
    Ok(LogisticFit {
        coefficients: full
            .unscale(&fit.coefficients)
            .into_iter()
            .map(Some)
            .collect(),
        linear_predictor: fit.linear_predictor,
        probabilities: fit.fitted,
        converged: fit.converged,
        iterations: fit.iterations,
        aliased_columns: Vec::new(),
        lambda: Some(chosen),
        cross_validation: Some(CrossValidation {
            fold_ids,
            lambda,
            deviance,
            deviance_se,
            index_min,
            index_one_se,
            index_chosen,
            unconverged_fits: unconverged,
        }),
    })
}
