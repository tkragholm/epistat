//! Choosing the elastic-net penalty by cross-validation, as `cv.glmnet` does.
//!
//! # Method
//!
//! Let `w` be the prior weights and `w~ = w / sum(w)`. The elastic net at
//! penalty `lambda` has the objective of [`super::logistic`].
//!
//! The path. It starts at the smallest `lambda` that sets every penalised
//! coefficient to zero. With the penalised columns scaled as the fit scales
//! them, `x~_j`, the null model has the weighted mean `y_bar` as its
//! probability and the gradient of the log-likelihood at the null model is
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
//! penalised columns and 1e-2 otherwise. The first fit is the null model, with
//! every penalised coefficient zero, whatever `alpha` is: `glmnet` fits it at a
//! penalty of 9.9e35, and reports `lambda_max` for it. For `alpha = 0` this
//! differs from the ridge fit at `lambda_max`. Each later fit is warm-started
//! from the previous one. The path is cut short as `glmnet`'s is: with `D` the fraction
//! of the null deviance that the fit explains, it stops after the `m`-th value,
//! `m` at least 5, when `D_m > 0.999` or when `D_m - D_(m-1) < 1e-5`. These are
//! the defaults of `glmnet.control()`: `devmax`, `fdev` and `mnlam`.
//!
//! The folds. The rows are dealt to `nfolds` folds in turn and the assignment
//! is shuffled with a stream from `epistat-rng`. For each fold the model is
//! fitted on the other folds along a path of the fold's own, built as above from
//! the training rows with the columns scaled by them. `cv.glmnet` does the same
//! when it is not given a `lambda`, which is what MatchIt does. The fold's
//! coefficients at a penalty of the master path, the path of all rows, are the
//! linear interpolation in `lambda` between the two neighbouring values of the
//! fold's path, and those of the nearest end where the penalty lies outside it.
//! This is `cv.glmnet`'s `alignment = "lambda"`.
//!
//! The curve. The held-out rows get the binomial deviance
//! `-2 [y ln p + (1 - y) ln(1 - p)]`, with `p` clipped to `[1e-5, 1 - 1e-5]`.
//! For fold `k` and master penalty `lambda`, `d_k` is the weighted mean of those
//! deviances and `W_k` the sum of the weights. The curve and its standard error
//! are
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
//! # References
//!
//! 1. Friedman J, Hastie T, Tibshirani R. Regularization paths for generalized
//!    linear models via coordinate descent. J Stat Softw. 2010;33(1):1-22.
//!    doi:10.18637/jss.v033.i01
//! 2. Hastie T, Tibshirani R, Friedman J. The Elements of Statistical Learning.
//!    2nd ed. Springer; 2009. Section 7.10 on cross-validation and the one
//!    standard error rule.
//! 3. The help pages of `cv.glmnet` and `glmnet.control` in the R package
//!    glmnet, for the folds, the alignment of the folds' paths and the
//!    stopping rule of a path.

use super::EstimationError;
use super::config::{ElasticNetCvConfig, LambdaRule};
use super::encoding::CovariateMatrix;
use super::logistic::{LogisticFit, NetProblem, NetSettings, penalty_factors};
use epistat_rng::Stream;

/// `glmnet.control()`'s `mnlam`: the path has at least this many values.
const MIN_PATH: usize = 5;
/// `glmnet.control()`'s `fdev`: the path stops when the fraction of the null
/// deviance explained grows by less than this.
const FRACTIONAL_CHANGE: f64 = 1e-5;
/// `glmnet.control()`'s `devmax`: the path stops when this fraction of the null
/// deviance is explained.
const MAX_EXPLAINED: f64 = 0.999;

/// The penalty at which a path's first fit is made, `glmnet.control()`'s `big`.
const NULL_PENALTY: f64 = 9.9e35;

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

/// The fold of each of `n` rows, from 0 to `nfolds - 1`: the rows are dealt to
/// the folds in turn and the assignment is shuffled with a stream of
/// `epistat-rng` seeded with `seed`. The rows are those of the design matrix,
/// the anchors first and then the candidates.
///
/// # Errors
///
/// Returns [`EstimationError::InvalidCrossValidation`] when `nfolds` is 0.
pub fn cross_validation_folds(
    n: usize,
    nfolds: usize,
    seed: u64,
) -> Result<Vec<usize>, EstimationError> {
    if nfolds == 0 {
        return Err(EstimationError::InvalidCrossValidation);
    }
    let mut folds: Vec<usize> = (0..n).map(|row| row % nfolds).collect();
    Stream::seeded(seed).shuffle(&mut folds);
    Ok(folds)
}

fn binomial_deviance(y: f64, eta: f64) -> f64 {
    let p = (1.0 / (1.0 + (-eta).exp())).clamp(DEVIANCE_CLIP, 1.0 - DEVIANCE_CLIP);
    -2.0 * (y * p.ln() + (1.0 - y) * (1.0 - p).ln())
}

/// A path of fits, from the largest penalty down.
struct Path {
    lambda: Vec<f64>,
    /// The penalty each fit was made at: `lambda`, except the first, which is
    /// the null model.
    fitted_at: Vec<f64>,
    /// The coefficients of each fit on the problem's scaled design.
    scaled: Vec<Vec<f64>>,
    /// The number of fits that did not meet their stopping rule.
    unconverged: usize,
}

/// `glmnet`'s path for `problem`: `nlambda` penalties from the smallest one that
/// zeroes every penalised coefficient, cut short by the deviance rule.
fn glmnet_path(
    problem: &NetProblem,
    penalty: &[f64],
    settings: &NetSettings,
    config: &ElasticNetCvConfig,
) -> Result<Path, EstimationError> {
    let (n, p) = (problem.n_rows, problem.n_cols);
    let weight = |row: usize| problem.weights.as_ref().map_or(1.0, |w| w[row]);
    let total: f64 = (0..n).map(weight).sum();
    if !(total.is_finite() && total > 0.0) {
        return Err(EstimationError::InvalidCrossValidation);
    }
    let y_bar = (0..n).map(|row| weight(row) * problem.y[row]).sum::<f64>() / total;
    if !(y_bar > 0.0 && y_bar < 1.0) {
        return Err(EstimationError::InvalidCrossValidation);
    }
    // The path starts where the null model's gradient first allows a nonzero
    // penalised coefficient.
    let mut lambda_max = 0.0_f64;
    for (col, factor) in penalty.iter().enumerate().filter(|(_, f)| **f > 0.0) {
        let gradient = (0..n)
            .map(|row| weight(row) * problem.values[row * p + col] * (problem.y[row] - y_bar))
            .sum::<f64>()
            / total;
        lambda_max = lambda_max.max(gradient.abs() / factor);
    }
    lambda_max /= config.alpha.max(1e-3);
    if !(lambda_max.is_finite() && lambda_max > 0.0) {
        return Err(EstimationError::InvalidCrossValidation);
    }
    let penalised = penalty.iter().filter(|f| **f > 0.0).count();
    let ratio = config
        .lambda_min_ratio
        .unwrap_or(if n > penalised { 1e-4 } else { 1e-2 });
    let steps = config.nlambda - 1;

    let null_deviance = -2.0
        * (0..n)
            .map(|row| {
                let y = problem.y[row];
                weight(row) * (y * y_bar.ln() + (1.0 - y) * (1.0 - y_bar).ln())
            })
            .sum::<f64>()
        / total;
    let mut path = Path {
        lambda: Vec::with_capacity(config.nlambda),
        fitted_at: Vec::with_capacity(config.nlambda),
        scaled: Vec::with_capacity(config.nlambda),
        unconverged: 0,
    };
    let mut start: Option<Vec<f64>> = None;
    let mut explained_before = 0.0_f64;
    for k in 0..config.nlambda {
        let value = lambda_max * ratio.powf(k as f64 / steps as f64);
        let fitted_at = if k == 0 { NULL_PENALTY } else { value };
        let fit = problem.fit(settings, fitted_at, penalty, start.as_deref())?;
        path.unconverged += usize::from(!fit.converged);
        let deviance = -2.0
            * (0..n)
                .map(|row| {
                    let eta = fit.linear_predictor[row];
                    // y eta - ln(1 + exp(eta)), without overflow.
                    let log_one_plus = eta.max(0.0) + (-eta.abs()).exp().ln_1p();
                    weight(row) * (problem.y[row] * eta - log_one_plus)
                })
                .sum::<f64>()
            / total;
        let explained = 1.0 - deviance / null_deviance;
        start = Some(fit.coefficients.clone());
        path.scaled.push(fit.coefficients);
        path.lambda.push(value);
        path.fitted_at.push(fitted_at);
        let stop = path.lambda.len() >= MIN_PATH
            && (explained > MAX_EXPLAINED || explained - explained_before < FRACTIONAL_CHANGE);
        explained_before = explained;
        if stop {
            break;
        }
    }
    Ok(path)
}

/// The coefficients of a fold's path at `lambda`: linear in `lambda` between the
/// two neighbouring values of the path, the nearest end outside it.
fn interpolate(lambda: &[f64], coefficients: &[Vec<f64>], at: f64) -> Vec<f64> {
    let last = lambda.len() - 1;
    if at >= lambda[0] {
        return coefficients[0].clone();
    }
    if at <= lambda[last] {
        return coefficients[last].clone();
    }
    // The path is decreasing: find j with lambda[j] > at >= lambda[j + 1].
    let j = lambda.partition_point(|value| *value > at) - 1;
    let fraction = (lambda[j] - at) / (lambda[j] - lambda[j + 1]);
    coefficients[j]
        .iter()
        .zip(&coefficients[j + 1])
        .map(|(left, right)| (1.0 - fraction) * left + fraction * right)
        .collect()
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
    let master = glmnet_path(&full, &penalty, &settings, config)?;
    let lambda = master.lambda.clone();
    let mut unconverged = master.unconverged;

    let fold_ids = cross_validation_folds(n, config.nfolds, config.seed)?;
    // Held-out deviance, by fold and master penalty.
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
        let path = glmnet_path(&problem, &penalty, &settings, config)?;
        unconverged += path.unconverged;
        let unscaled: Vec<Vec<f64>> = path.scaled.iter().map(|b| problem.unscale(b)).collect();
        for (k, &value) in lambda.iter().enumerate() {
            let beta = interpolate(&path.lambda, &unscaled, value);
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
        let by_fold: Vec<f64> = fold_deviance.iter().map(|curve| curve[k]).collect();
        let mean = by_fold
            .iter()
            .zip(&fold_weight)
            .map(|(d, w)| w * d)
            .sum::<f64>()
            / fold_total;
        let spread = by_fold
            .iter()
            .zip(&fold_weight)
            .map(|(d, w)| w * (d - mean) * (d - mean))
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
    let fit = full.fit(
        &settings,
        chosen,
        &penalty,
        Some(&master.scaled[index_chosen]),
    )?;
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
        cbps: None,
        at_boundary: false,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolation_is_linear_in_lambda_and_clamped_at_the_ends() {
        let lambda = [4.0, 2.0, 1.0];
        let coefficients = vec![vec![0.0, 0.0], vec![2.0, 4.0], vec![3.0, 8.0]];
        assert_eq!(interpolate(&lambda, &coefficients, 9.0), vec![0.0, 0.0]);
        assert_eq!(interpolate(&lambda, &coefficients, 4.0), vec![0.0, 0.0]);
        assert_eq!(interpolate(&lambda, &coefficients, 3.0), vec![1.0, 2.0]);
        assert_eq!(interpolate(&lambda, &coefficients, 2.0), vec![2.0, 4.0]);
        assert_eq!(interpolate(&lambda, &coefficients, 1.5), vec![2.5, 6.0]);
        assert_eq!(interpolate(&lambda, &coefficients, 0.5), vec![3.0, 8.0]);
    }

    #[test]
    fn folds_are_balanced_and_depend_on_the_seed() {
        let folds = cross_validation_folds(23, 5, 3).unwrap();
        let mut sizes = [0usize; 5];
        for fold in &folds {
            sizes[*fold] += 1;
        }
        assert!(sizes.iter().all(|size| (4..=5).contains(size)));
        assert_eq!(folds, cross_validation_folds(23, 5, 3).unwrap());
        assert_ne!(folds, cross_validation_folds(23, 5, 4).unwrap());
    }
}
