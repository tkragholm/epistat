//! Fixed-length confidence intervals (FLCIs) for the smoothness restriction
//! `Delta^SD(M)`, as `HonestDiD`'s `findOptimalFLCI`.
//!
//! # Formulas
//!
//! The interval is built on an affine estimator `a' beta_hat` of
//! `theta = l' tau_post`, with `a = (l_pre, l)` and the pre-period weights
//! written through `w` as `l_pre,1 = w_1` and `l_pre,k = w_k - w_{k-1}`. Over
//! `Delta^SD(M)` the estimator's worst-case bias is finite exactly when
//! `sum_k w_k = sum_t t l_t`, and is then `M b(w)` with
//!
//! ```text
//! b(w) = c(l) + sum_{k=1..K} |w_1 + ... + w_k|
//! c(l) = sum_{s=1..T} |sum_{j=1..s} j l_{T-s+j}| - sum_{t=1..T} t l_t
//! ```
//!
//! for `K` pre and `T` post periods. With `Sigma = L L'` its standard deviation
//! is `sd(w) = ||L' a||`, linear in `w` inside the norm. An interval
//! `a' beta_hat +/- chi(M b / h) h` with `sd(w) <= h` covers at level
//! `1 - alpha`, where `chi(mu)` is the `1 - alpha` quantile of `|N(mu, 1)|`.
//!
//! # Algorithm
//!
//! 1. `h_min`: the smallest `sd(w)` under the sum constraint, a second-order
//!    cone program.
//! 2. `h_0`: `sd(w)` of the weights of smallest bias, all of the sum on the
//!    last pre period.
//! 3. On `num_grid_points` equally spaced `h` from `h_min` to `h_0`, the
//!    smallest `b(w)` with `sd(w) <= h`, a second-order cone program with
//!    `u_k >= |w_1 + ... + w_k|` carried as linear constraints, and its
//!    half-length `chi(M b / h) h`.
//! 4. The interval of the `h` with the shortest half-length.
//!
//! `HonestDiD` searches `h` by bisection on a numerical derivative and takes
//! `chi` from 10^6 normal draws; here the grid is fixed and `chi` is
//! [`epistat_stats::dist::normal::folded_quantile`]. The two agree to the Monte
//! Carlo error of `HonestDiD`'s quantile once the grid is fine.
//!
//! # References
//!
//! - Rambachan, A. and Roth, J. (2023). "A More Credible Approach to Parallel
//!   Trends". *Review of Economic Studies* 90(5), 2555-2591.
//! - Armstrong, T. B. and Kolesar, M. (2018). "Optimal Inference in a Class
//!   of Regression Models". *Econometrica* 86(2), 655-683.
//! - `HonestDiD` 0.2.8, `flci.R`: `.createObjectiveObjectForBias`,
//!   `.findLowestH`, `.findHForMinimumBias`, `.findWorstCaseBiasGivenH`.

use crate::sensitivity::EngineError;
use clarabel::algebra::CscMatrix;
use clarabel::solver::SupportedConeT;
use rayon::prelude::*;

use crate::util::InferenceConfig;
use crate::util::validate_confidence_level;

use super::super::conic;
use super::super::linear_algebra::{
    cholesky_lower, critical_value_from_pointwise_confidence, dot, functional_covariance_matrix,
    pointwise_confidence_level_from_critical, post_covariance_block, sup_t_critical_value,
};
use super::super::{
    HonestEventStudyInput, HonestIdentifiedSet, HonestJointPathConfig, HonestJointPathMethod,
    HonestOriginalConfidenceSet, SmoothnessMultiFlciResult,
};
use super::super::{SmoothnessMultiFlciPoint, compute_original_confidence_set};
use crate::util::usize_to_f64;

/// Numerical settings of the smoothness fixed-length interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SmoothnessFlciConfig {
    /// Number of grid points used to search for the optimal interval.
    pub num_grid_points: usize,
}

impl SmoothnessFlciConfig {
    /// The settings used in production runs.
    #[must_use]
    pub const fn default_for_production() -> Self {
        Self {
            num_grid_points: 48,
        }
    }
}

/// Inputs of a fixed-length confidence interval under a smoothness restriction.
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothnessFlciProblem {
    /// Event-study coefficients and covariance.
    pub input: HonestEventStudyInput,
    /// Weights of the post-period functional.
    pub post_weights: Vec<f64>,
    /// Smoothness bound on the second differences of the trend.
    pub m: f64,
    /// Confidence level settings.
    pub inference: InferenceConfig,
    /// Original confidence set of the functional.
    pub original: HonestOriginalConfidenceSet,
    /// Identified set of the functional under the smoothness bound.
    pub identified_set: HonestIdentifiedSet,
}

/// Fixed-length confidence interval under a smoothness restriction.
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothnessFlciResult {
    /// Fixed-length confidence interval.
    pub flci: (f64, f64),
    /// Original confidence interval.
    pub original_ci: (f64, f64),
    /// Identified set, NaN at both ends when it is empty. The interval does not
    /// depend on it.
    pub identified_set: (f64, f64),
    /// Weights of the optimal affine estimator.
    pub optimal_vec: Vec<f64>,
    /// Half-length of the interval.
    pub optimal_half_length: f64,
}

/// Inputs of simultaneous fixed-length intervals under a smoothness restriction.
#[derive(Debug, Clone, PartialEq)]
pub struct SmoothnessMultiFlciProblem {
    /// Event-study coefficients and covariance.
    pub input: HonestEventStudyInput,
    /// Weights of each post-period functional.
    pub post_weight_sets: Vec<Vec<f64>>,
    /// Smoothness bound on the second differences of the trend.
    pub m: f64,
    /// Confidence level settings.
    pub inference: InferenceConfig,
    /// Original confidence set of each functional.
    pub originals: Vec<HonestOriginalConfidenceSet>,
    /// Identified set of each functional.
    pub identified_sets: Vec<HonestIdentifiedSet>,
}

#[derive(Debug, Clone)]
struct SmoothnessBiasAtH {
    value_at_m_one: f64,
    optimal_l_pre: Vec<f64>,
}

#[derive(Debug)]
struct SmoothnessSocpSolution {
    solution: Vec<f64>,
}

fn weighted_post_sum(post_weights: &[f64]) -> f64 {
    post_weights
        .iter()
        .enumerate()
        .map(|(idx, weight)| usize_to_f64(idx + 1) * weight)
        .sum()
}

fn w_to_l_pre(w: &[f64]) -> Vec<f64> {
    if w.is_empty() {
        return Vec::new();
    }
    let mut out = vec![0.0; w.len()];
    out[0] = w[0];
    for idx in 1..w.len() {
        out[idx] = w[idx] - w[idx - 1];
    }
    out
}

fn full_estimator_vec(num_pre: usize, post_weights: &[f64], w: &[f64]) -> Vec<f64> {
    let mut out = w_to_l_pre(w);
    debug_assert_eq!(out.len(), num_pre);
    out.extend_from_slice(post_weights);
    out
}

/// The constant `c(l)` of the worst-case bias at `M = 1`,
/// `sum_{s=1..T} |sum_{j=1..s} j l_{T-s+j}| - sum_t t l_t`, as in
/// `HonestDiD`'s `.createObjectiveObjectForBias`.
fn bias_objective_constant(post_weights: &[f64]) -> f64 {
    let num_post = post_weights.len();
    let mut sum_abs = 0.0;
    for s in 1..=num_post {
        let mut inner = 0.0;
        for j in 1..=s {
            inner += usize_to_f64(j) * post_weights[num_post - s + j - 1];
        }
        sum_abs += f64::abs(inner);
    }
    sum_abs - weighted_post_sum(post_weights)
}

fn lower_triangular_ones(num_pre: usize) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; num_pre]; num_pre];
    for (row_idx, row) in out.iter_mut().enumerate() {
        for value in &mut row[..=row_idx] {
            *value = 1.0;
        }
    }
    out
}

/// The estimator's standard deviation as a norm, `sd(w) = ||d + T w||`, with
/// `d = L_post' l` and `T w = L_pre' l_pre(w)`. The rows returned are `-T`, so
/// that a cone row `b - A x` with right-hand side `d` holds `d + T w`.
fn build_variance_soc_terms(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
) -> Result<(Vec<Vec<f64>>, Vec<f64>), EngineError> {
    let num_pre = input.num_pre_periods();
    let total_periods = input.betahat.len();
    let lower = cholesky_lower(&input.covariance)?;
    let mut d = vec![0.0; total_periods];
    for row in 0..total_periods {
        let mut acc = 0.0;
        for (post_idx, post_weight) in post_weights.iter().copied().enumerate() {
            let idx = num_pre + post_idx;
            acc = lower[(idx, row)].mul_add(post_weight, acc);
        }
        d[row] = acc;
    }

    let mut transform = vec![vec![0.0; num_pre]; total_periods];
    for row in 0..total_periods {
        for col in 0..num_pre {
            let mut acc = 0.0;
            acc += lower[(col, row)];
            if col + 1 < num_pre {
                acc -= lower[(col + 1, row)];
            }
            transform[row][col] = -acc;
        }
    }
    Ok((transform, d))
}

/// Split the variance rows into the ones `w` can move and the norm it cannot.
///
/// `transform` has one row per period but only `num_pre` columns, and its
/// entries come from a lower-triangular Cholesky factor, so every row at or
/// after the first post period is **identically zero**: that part of the
/// estimator's variance belongs to the post-period coefficients and no choice of
/// pre-period weights touches it. Its `d` entries are still real numbers, so in
/// the cone they appear as all-zero rows with non-zero right-hand sides.
///
/// Those rows are what made Clarabel stall with `InsufficientProgress`. They fix
/// a floor under the norm -- on the case that failed, 0.2828 of an available
/// 0.2898 -- so the cone is a sliver 0.06 wide whose analytic centre sits almost
/// on the boundary, and an interior-point method has nowhere to move. The
/// problem is feasible and well-posed; it is the FORM that is degenerate.
///
/// Returned separately, the caller can put the constant where it belongs: as one
/// scalar rather than as a block of structurally rank-deficient rows.
fn split_constant_variance(
    transform: Vec<Vec<f64>>,
    d: Vec<f64>,
) -> (Vec<Vec<f64>>, Vec<f64>, f64) {
    let mut active_rows = Vec::with_capacity(transform.len());
    let mut active_d = Vec::with_capacity(d.len());
    let mut constant_sq = 0.0;
    for (row, value) in transform.into_iter().zip(d) {
        if row.iter().any(|coefficient| *coefficient != 0.0) {
            active_rows.push(row);
            active_d.push(value);
        } else {
            constant_sq += value * value;
        }
    }
    (active_rows, active_d, constant_sq.max(0.0).sqrt())
}

fn variance_for_w(input: &HonestEventStudyInput, post_weights: &[f64], w: &[f64]) -> f64 {
    let estimator = full_estimator_vec(input.num_pre_periods(), post_weights, w);
    estimator
        .iter()
        .enumerate()
        .map(|(i, left)| {
            estimator
                .iter()
                .enumerate()
                .map(|(j, right)| left * right * input.covariance[i][j])
                .sum::<f64>()
        })
        .sum::<f64>()
        .max(0.0)
}

fn solve_socp(
    objective: &[f64],
    inequalities: &[Vec<f64>],
    equalities: &[Vec<f64>],
    soc_rows: &[Vec<f64>],
    rhs: &[f64],
    counter: &crate::sensitivity::SolverCounter,
) -> Result<Option<SmoothnessSocpSolution>, EngineError> {
    let mut equality_and_soc = Vec::with_capacity(equalities.len() + soc_rows.len());
    equality_and_soc.extend_from_slice(equalities);
    equality_and_soc.extend_from_slice(soc_rows);
    let constraint_matrix =
        super::super::linear_algebra::build_clarabel_matrix(inequalities, &equality_and_soc);
    let cones = vec![
        SupportedConeT::NonnegativeConeT(inequalities.len()),
        SupportedConeT::ZeroConeT(equalities.len()),
        SupportedConeT::SecondOrderConeT(soc_rows.len()),
    ];
    let quadratic = CscMatrix::<f64>::zeros((objective.len(), objective.len()));
    Ok(conic::solve(
        "smoothness SOCP",
        &quadratic,
        objective,
        &constraint_matrix,
        rhs,
        &cones,
        counter,
    )?
    .map(|solution| SmoothnessSocpSolution { solution }))
}

fn find_lowest_h(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    counter: &crate::sensitivity::SolverCounter,
) -> Result<Option<(f64, Vec<f64>)>, EngineError> {
    let num_pre = input.num_pre_periods();
    let weighted_sum = weighted_post_sum(post_weights);
    let (transform, d) = build_variance_soc_terms(input, post_weights)?;
    // `h` is a variable here, so the immovable part cannot be folded into the
    // bound the way it is in `find_worst_case_bias_given_h`. It can still be
    // carried as ONE row holding its norm instead of a block of zero rows
    // holding its components: same cone, same optimum, one rank-deficient row
    // rather than several.
    let (transform, d, constant) = split_constant_variance(transform, d);
    let num_vars = num_pre + 1;
    let objective = {
        let mut out = vec![0.0; num_vars];
        out[num_pre] = 1.0;
        out
    };
    let equalities = vec![{
        let mut row = vec![0.0; num_vars];
        for value in &mut row[..num_pre] {
            *value = 1.0;
        }
        row
    }];
    let soc_rows = {
        let mut rows = Vec::with_capacity(d.len() + 2);
        let mut first = vec![0.0; num_vars];
        first[num_pre] = -1.0;
        rows.push(first);
        rows.extend(transform.into_iter().map(|coeffs| {
            let mut row = vec![0.0; num_vars];
            row[..num_pre].copy_from_slice(&coeffs);
            row
        }));
        rows.push(vec![0.0; num_vars]);
        rows
    };
    let mut rhs = vec![weighted_sum];
    rhs.push(0.0);
    rhs.extend(d);
    rhs.push(constant);
    let Some(solution) = solve_socp(&objective, &[], &equalities, &soc_rows, &rhs, counter)? else {
        return Ok(None);
    };
    let w = solution.solution[..num_pre].to_vec();
    Ok(Some((solution.solution[num_pre], w)))
}

fn find_h_for_minimum_bias(input: &HonestEventStudyInput, post_weights: &[f64]) -> (f64, Vec<f64>) {
    let num_pre = input.num_pre_periods();
    let mut w = vec![0.0; num_pre];
    if let Some(last) = w.last_mut() {
        *last = weighted_post_sum(post_weights);
    }
    (variance_for_w(input, post_weights, &w).sqrt(), w)
}

fn find_worst_case_bias_given_h(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    h: f64,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<Option<SmoothnessBiasAtH>, EngineError> {
    let num_pre = input.num_pre_periods();
    let weighted_sum = weighted_post_sum(post_weights);
    let constant = bias_objective_constant(post_weights);
    let lower_tri = lower_triangular_ones(num_pre);
    let (transform, d) = build_variance_soc_terms(input, post_weights)?;
    // `h` is FIXED here, so the immovable variance folds into the bound:
    // ||d - Tw|| <= h with a constant block c is exactly
    // ||d_active - T_active w|| <= sqrt(h^2 - c^2), and if h < c there is no w
    // at all. Solving it in that form is what stops Clarabel stalling: the cone
    // that failed was 0.2898 wide with 0.2828 of it already spent on rows no
    // variable appears in, and the same problem restated is 0.0637 wide with
    // nothing fixed inside it.
    // NOT `constant`: that name is already taken in this function by
    // `bias_objective_constant`, and shadowing it here would have silently
    // changed the returned bias rather than the cone.
    let (transform, d, immovable) = split_constant_variance(transform, d);
    let radius_sq = h.mul_add(h, -(immovable * immovable));
    if radius_sq < 0.0 {
        if std::env::var("SOCP_DEBUG").is_ok() {
            eprintln!("INFEASIBLE h={h} immovable={immovable}");
        }
        // Infeasible rather than unsolvable: no weighting achieves a standard
        // deviation below the part of it that does not depend on the weights.
        // The caller's `None` arm already means exactly this.
        return Ok(None);
    }
    let radius = radius_sq.sqrt();
    let num_vars = num_pre * 2;
    let objective = {
        let mut out = vec![0.0; num_vars];
        for value in &mut out[..num_pre] {
            *value = 1.0;
        }
        out
    };
    let mut inequalities = Vec::with_capacity(num_pre * 2);
    for row_idx in 0..num_pre {
        let mut row = vec![0.0; num_vars];
        row[row_idx] = -1.0;
        row[num_pre..].copy_from_slice(&lower_tri[row_idx]);
        inequalities.push(row);
    }
    for row_idx in 0..num_pre {
        let mut row = vec![0.0; num_vars];
        row[row_idx] = -1.0;
        for (dst, src) in row[num_pre..].iter_mut().zip(lower_tri[row_idx].iter()) {
            *dst = -*src;
        }
        inequalities.push(row);
    }
    let equalities = vec![{
        let mut row = vec![0.0; num_vars];
        for value in &mut row[num_pre..] {
            *value = 1.0;
        }
        row
    }];
    let soc_rows = {
        let mut rows = Vec::with_capacity(d.len() + 1);
        rows.push(vec![0.0; num_vars]);
        rows.extend(transform.into_iter().map(|coeffs| {
            let mut row = vec![0.0; num_vars];
            row[num_pre..].copy_from_slice(&coeffs);
            row
        }));
        rows
    };
    let mut rhs = vec![0.0; inequalities.len()];
    rhs.push(weighted_sum);
    rhs.push(radius);
    rhs.extend(d);
    let Some(solution) = solve_socp(
        &objective,
        &inequalities,
        &equalities,
        &soc_rows,
        &rhs,
        counter,
    )?
    else {
        return Ok(None);
    };
    let w = solution.solution[num_pre..].to_vec();
    Ok(Some(SmoothnessBiasAtH {
        value_at_m_one: constant + solution.solution[..num_pre].iter().sum::<f64>(),
        optimal_l_pre: w_to_l_pre(&w),
    }))
}

fn folded_normal_quantile(p: f64, mu: f64) -> Result<f64, EngineError> {
    Some(epistat_stats::dist::normal::folded_quantile(p, mu))
        .filter(|q| !q.is_nan())
        .ok_or(EngineError::CriticalValue("folded-normal quantile"))
}

fn half_length_for_bias(alpha: f64, max_bias: f64, h: f64) -> Result<f64, EngineError> {
    if h <= 1e-12 {
        return Ok(max_bias.abs());
    }
    Ok(folded_normal_quantile(1.0 - alpha, max_bias / h)? * h)
}

fn find_optimal_flci(
    problem: &SmoothnessFlciProblem,
    config: SmoothnessFlciConfig,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<Option<(Vec<f64>, f64)>, EngineError> {
    let Some((h_min, _)) = find_lowest_h(&problem.input, &problem.post_weights, counter)? else {
        return Ok(None);
    };
    let (h_zero_bias, fallback_w) = find_h_for_minimum_bias(&problem.input, &problem.post_weights);
    let alpha = 1.0 - problem.inference.confidence_level;
    let lower = h_min.max(0.0);
    let upper = h_zero_bias.max(lower);
    let grid_points = config.num_grid_points.max(2);
    let step = if grid_points == 1 {
        0.0
    } else {
        (upper - lower) / usize_to_f64(grid_points - 1)
    };
    let mut best: Option<(Vec<f64>, f64, f64)> = None;
    for idx in 0..grid_points {
        let h = if idx + 1 == grid_points {
            upper
        } else {
            step.mul_add(usize_to_f64(idx), lower)
        };
        let Some(bias) =
            find_worst_case_bias_given_h(&problem.input, &problem.post_weights, h, counter)?
        else {
            continue;
        };
        let max_bias = problem.m * bias.value_at_m_one;
        let half_length = half_length_for_bias(alpha, max_bias, h)?;
        let candidate = (
            {
                let mut vec = bias.optimal_l_pre.clone();
                vec.extend_from_slice(&problem.post_weights);
                vec
            },
            half_length,
            h,
        );
        match &best {
            Some((_, best_half_length, _)) if *best_half_length <= half_length => {}
            _ => best = Some(candidate),
        }
    }
    if let Some((optimal_vec, half_length, _)) = best {
        return Ok(Some((optimal_vec, half_length)));
    }

    let estimator = full_estimator_vec(
        problem.input.num_pre_periods(),
        &problem.post_weights,
        &fallback_w,
    );
    // The minimum-bias weights put the whole sum on the last pre period, so
    // the only non-zero partial sum of `w` is that sum itself.
    let max_bias = problem.m
        * (bias_objective_constant(&problem.post_weights)
            + f64::abs(weighted_post_sum(&problem.post_weights)));
    let half_length = half_length_for_bias(alpha, max_bias, h_zero_bias)?;
    Ok(Some((estimator, half_length)))
}

/// Build the typed scalar smoothness-FLCI problem for one post-treatment
/// functional.
///
/// # Errors
/// Returns an error if the input, post weights, or smoothness bound are
/// invalid, or if the original / identified-set builders fail.
#[cfg(test)]
pub fn build_smoothness_flci_problem(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    m: f64,
    inference: InferenceConfig,
) -> Result<SmoothnessFlciProblem, EngineError> {
    build_smoothness_flci_problem_counted(
        input,
        post_weights,
        m,
        inference,
        &crate::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::sensitivity) fn build_smoothness_flci_problem_counted(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    m: f64,
    inference: InferenceConfig,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<SmoothnessFlciProblem, EngineError> {
    let original = compute_original_confidence_set(input, post_weights, inference)?;
    let identified_set = crate::sensitivity::smoothness::compute_smoothness_identified_set_counted(
        input,
        post_weights,
        m,
        counter,
    )?;
    Ok(SmoothnessFlciProblem {
        input: input.clone(),
        post_weights: post_weights.to_vec(),
        m,
        inference,
        original,
        identified_set,
    })
}

/// Compute the scalar smoothness FLCI using production defaults.
///
/// # Errors
/// Returns an error if the typed problem is invalid or the smoothness
/// least-favorable optimization fails.
#[cfg(test)]
pub fn compute_smoothness_flci(
    problem: &SmoothnessFlciProblem,
) -> Result<SmoothnessFlciResult, EngineError> {
    compute_smoothness_flci_counted(problem, &crate::sensitivity::SolverCounter::new())
}

#[cfg(test)]
pub(in crate::sensitivity) fn compute_smoothness_flci_counted(
    problem: &SmoothnessFlciProblem,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<SmoothnessFlciResult, EngineError> {
    compute_smoothness_flci_with_config_counted(
        problem,
        SmoothnessFlciConfig::default_for_production(),
        counter,
    )
}

/// Compute the scalar smoothness FLCI with explicit grid settings.
///
/// # Errors
/// Returns an error if the typed problem is invalid or the smoothness
/// least-favorable optimization fails.
#[cfg(test)]
pub(crate) fn compute_smoothness_flci_with_config(
    problem: &SmoothnessFlciProblem,
    config: SmoothnessFlciConfig,
) -> Result<SmoothnessFlciResult, EngineError> {
    compute_smoothness_flci_with_config_counted(
        problem,
        config,
        &crate::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::sensitivity) fn compute_smoothness_flci_with_config_counted(
    problem: &SmoothnessFlciProblem,
    config: SmoothnessFlciConfig,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<SmoothnessFlciResult, EngineError> {
    if !validate_confidence_level(problem.inference.confidence_level) {
        return Err(EngineError::domain(
            "confidence level",
            problem.inference.confidence_level,
            "(0, 1)",
        ));
    }
    let Some((optimal_vec, optimal_half_length)) = find_optimal_flci(problem, config, counter)?
    else {
        return Err(EngineError::CriticalValue(
            "fixed-length confidence interval",
        ));
    };
    let center = dot(&optimal_vec, &problem.input.betahat);
    Ok(SmoothnessFlciResult {
        flci: (center - optimal_half_length, center + optimal_half_length),
        original_ci: problem.original.ci,
        identified_set: (problem.identified_set.lb, problem.identified_set.ub),
        optimal_vec,
        optimal_half_length,
    })
}

/// Build the typed simultaneous smoothness-FLCI problem for a set of
/// post-treatment functionals.
///
/// # Errors
/// Returns an error if the functional list is empty or if any original /
/// identified-set calculation fails.
pub fn build_smoothness_multi_flci_problem(
    input: &HonestEventStudyInput,
    post_weight_sets: &[Vec<f64>],
    m: f64,
    inference: InferenceConfig,
) -> Result<SmoothnessMultiFlciProblem, EngineError> {
    build_smoothness_multi_flci_problem_counted(
        input,
        post_weight_sets,
        m,
        inference,
        &crate::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::sensitivity) fn build_smoothness_multi_flci_problem_counted(
    input: &HonestEventStudyInput,
    post_weight_sets: &[Vec<f64>],
    m: f64,
    inference: InferenceConfig,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<SmoothnessMultiFlciProblem, EngineError> {
    if post_weight_sets.is_empty() {
        return Err(EngineError::empty("targets"));
    }
    let mut originals = Vec::with_capacity(post_weight_sets.len());
    let mut identified_sets = Vec::with_capacity(post_weight_sets.len());
    for post_weights in post_weight_sets {
        originals.push(compute_original_confidence_set(
            input,
            post_weights,
            inference,
        )?);
        identified_sets.push(
            crate::sensitivity::smoothness::compute_smoothness_identified_set_counted(
                input,
                post_weights,
                m,
                counter,
            )?,
        );
    }
    Ok(SmoothnessMultiFlciProblem {
        input: input.clone(),
        post_weight_sets: post_weight_sets.to_vec(),
        m,
        inference,
        originals,
        identified_sets,
    })
}

fn simulated_joint_pointwise_confidence_level(
    input: &HonestEventStudyInput,
    confidence_level: f64,
    post_weight_sets: &[Vec<f64>],
    simulation_draws: usize,
    simulation_seed: u64,
) -> Result<f64, EngineError> {
    let sigma_post = post_covariance_block(
        &input.covariance,
        input.num_pre_periods(),
        input.num_post_periods(),
    );
    let covariance = functional_covariance_matrix(
        &sigma_post,
        post_weight_sets,
        "one or more functionals have near-zero variance under post covariance",
    )?;
    sup_t_critical_value(
        &covariance,
        confidence_level,
        simulation_draws,
        simulation_seed,
    )
}

fn joint_pointwise_confidence_level(
    problem: &SmoothnessMultiFlciProblem,
    joint_config: HonestJointPathConfig,
) -> Result<(f64, f64), EngineError> {
    let n = problem.post_weight_sets.len();
    if n == 1 {
        return Ok((
            problem.inference.confidence_level,
            critical_value_from_pointwise_confidence(problem.inference.confidence_level),
        ));
    }
    let alpha = 1.0 - problem.inference.confidence_level;
    match joint_config.method {
        HonestJointPathMethod::Bonferroni => {
            let pointwise = 1.0 - alpha / usize_to_f64(n);
            Ok((
                pointwise,
                critical_value_from_pointwise_confidence(pointwise),
            ))
        }
        HonestJointPathMethod::GaussianSimulated => {
            let critical = simulated_joint_pointwise_confidence_level(
                &problem.input,
                problem.inference.confidence_level,
                &problem.post_weight_sets,
                joint_config.simulation_draws,
                joint_config.simulation_seed,
            )?;
            Ok((pointwise_confidence_level_from_critical(critical), critical))
        }
    }
}

/// Compute simultaneous smoothness FLCIs using production defaults.
///
/// # Errors
/// Returns an error if the typed problem is invalid or the joint calibration /
/// scalar smoothness solves fail.
#[cfg(test)]
pub fn compute_smoothness_multi_flci(
    problem: &SmoothnessMultiFlciProblem,
) -> Result<SmoothnessMultiFlciResult, EngineError> {
    compute_smoothness_multi_flci_counted(problem, &crate::sensitivity::SolverCounter::new())
}

#[cfg(test)]
pub(in crate::sensitivity) fn compute_smoothness_multi_flci_counted(
    problem: &SmoothnessMultiFlciProblem,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<SmoothnessMultiFlciResult, EngineError> {
    compute_smoothness_multi_flci_with_config_counted(
        problem,
        SmoothnessFlciConfig::default_for_production(),
        HonestJointPathConfig::default_for_production(),
        counter,
    )
}

/// Compute simultaneous smoothness FLCIs with explicit scalar and joint
/// calibration settings.
///
/// # Errors
/// Returns an error if the typed problem is invalid or the joint calibration /
/// scalar smoothness solves fail.
pub(crate) fn compute_smoothness_multi_flci_with_config(
    problem: &SmoothnessMultiFlciProblem,
    config: SmoothnessFlciConfig,
    joint_config: HonestJointPathConfig,
) -> Result<SmoothnessMultiFlciResult, EngineError> {
    compute_smoothness_multi_flci_with_config_counted(
        problem,
        config,
        joint_config,
        &crate::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::sensitivity) fn compute_smoothness_multi_flci_with_config_counted(
    problem: &SmoothnessMultiFlciProblem,
    config: SmoothnessFlciConfig,
    joint_config: HonestJointPathConfig,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<SmoothnessMultiFlciResult, EngineError> {
    let (pointwise_confidence_level, calibrated_max_t_critical_value) =
        joint_pointwise_confidence_level(problem, joint_config)?;
    let pointwise_inference = InferenceConfig::new(pointwise_confidence_level);
    let points = problem
        .post_weight_sets
        .par_iter()
        .zip(problem.originals.par_iter())
        .zip(problem.identified_sets.par_iter())
        .map(|((post_weights, original), identified_set)| {
            let scalar_problem = SmoothnessFlciProblem {
                input: problem.input.clone(),
                post_weights: post_weights.clone(),
                m: problem.m,
                inference: pointwise_inference,
                original: original.clone(),
                identified_set: identified_set.clone(),
            };
            let flci =
                compute_smoothness_flci_with_config_counted(&scalar_problem, config, counter)?;
            Ok(SmoothnessMultiFlciPoint {
                post_weights: post_weights.clone(),
                flci: flci.flci,
                original_ci: original.ci,
                identified_set: (identified_set.lb, identified_set.ub),
                null_value: 0.0,
                robustly_significant: flci.flci.0 > 0.0 || flci.flci.1 < 0.0,
            })
        })
        .collect::<Result<Vec<_>, EngineError>>()?;

    Ok(SmoothnessMultiFlciResult {
        confidence_level: problem.inference.confidence_level,
        pointwise_confidence_level,
        calibrated_max_t_critical_value,
        method: joint_config.method,
        points,
    })
}

#[cfg(test)]
mod tests {
    use clarabel::algebra::CscMatrix;
    use clarabel::solver::SupportedConeT;

    use super::super::super::conic;
    use super::super::super::linear_algebra::{build_clarabel_matrix, dot};
    use super::super::build_smoothness_constraint_matrix;
    use super::{bias_objective_constant, build_smoothness_flci_problem, compute_smoothness_flci};
    use crate::sensitivity::{HonestEventStudyInput, SolverCounter};
    use crate::util::InferenceConfig;

    #[derive(serde::Deserialize)]
    struct EventStudy {
        betahat: Vec<f64>,
        sigma: Vec<Vec<f64>>,
    }

    /// `HonestDiD`'s `BCdata_EventStudy`, four pre and four post periods.
    fn bc_input() -> HonestEventStudyInput {
        let data: EventStudy = epistat_testkit::load_json("tests/data/honest_did_ref.json");
        HonestEventStudyInput {
            betahat: data.betahat,
            covariance: data.sigma,
            pre_periods: vec![-5, -4, -3, -2],
            post_periods: vec![0, 1, 2, 3],
        }
    }

    /// `sup a' delta` over `|second differences of (delta_pre, 0, delta_post)| <= 1`,
    /// by a linear program that does not use the closed form of the bias.
    fn worst_case_bias_at_m_one(num_pre: usize, num_post: usize, a: &[f64]) -> f64 {
        let rows = build_smoothness_constraint_matrix(num_pre, num_post, false);
        let rhs = vec![1.0; rows.len()];
        let objective: Vec<f64> = a.iter().map(|value| -value).collect();
        let solution = conic::solve(
            "worst-case bias",
            &CscMatrix::<f64>::zeros((a.len(), a.len())),
            &objective,
            &build_clarabel_matrix(&rows, &[]),
            &rhs,
            &[SupportedConeT::NonnegativeConeT(rows.len())],
            &SolverCounter::new(),
        )
        .expect("bias program")
        .expect("bounded bias");
        dot(a, &solution)
    }

    #[test]
    fn the_bias_constant_weights_each_tail_by_position() {
        // c(l) by hand for four post periods.
        assert_eq!(
            bias_objective_constant(&[1.0, 0.0, 0.0, 0.0]).to_bits(),
            f64::to_bits(0.0)
        );
        assert_eq!(
            bias_objective_constant(&[0.0, 1.0, 0.0, 0.0]).to_bits(),
            f64::to_bits(1.0)
        );
        assert_eq!(
            bias_objective_constant(&[0.25, 0.25, 0.25, 0.25]).to_bits(),
            f64::to_bits(2.5)
        );
        assert_eq!(
            bias_objective_constant(&[1.0, -1.0, 0.0, 0.0]).to_bits(),
            f64::to_bits(3.0)
        );
    }

    #[test]
    fn each_interval_covers_with_its_own_weights() {
        // The half-length must be at least chi(M b / sd) sd for the returned
        // weights, with b their worst-case bias from a separate linear program
        // and sd their standard deviation from the covariance, and equal to it
        // when the weights of the functional are not negative. For weights of
        // both signs `HonestDiD`'s bias constant exceeds the exact worst case,
        // for (1, -1, 0, 0) by 2, and the interval is conservative; at these M
        // the bias dominates the half-length, which then exceeds the exact one
        // by 2 M. Clarabel solves the cone programs to
        // gap and feasibility tolerances of 1e-8; 1e-6 of the half-length is a
        // hundred times that.
        let input = bc_input();
        let inference = InferenceConfig::new(0.95);
        let targets: [&[f64]; 5] = [
            &[1.0, 0.0, 0.0, 0.0],
            &[0.0, 0.0, 1.0, 0.0],
            &[0.25, 0.25, 0.25, 0.25],
            &[0.4, 0.3, 0.2, 0.1],
            &[1.0, -1.0, 0.0, 0.0],
        ];
        for post_weights in targets {
            for m in [0.0, 0.025, 0.1, 0.3] {
                let problem = build_smoothness_flci_problem(&input, post_weights, m, inference)
                    .expect("problem");
                let result = compute_smoothness_flci(&problem).expect("flci");
                let a = &result.optimal_vec;
                let variance: f64 = (0..a.len())
                    .map(|i| {
                        (0..a.len())
                            .map(|j| a[i] * input.covariance[i][j] * a[j])
                            .sum::<f64>()
                    })
                    .sum();
                let sd = variance.sqrt();
                let bias = m * worst_case_bias_at_m_one(4, 4, a);
                let expected = epistat_stats::dist::normal::folded_quantile(0.95, bias / sd) * sd;
                let tolerance = 1e-6 * expected;
                assert!(
                    result.optimal_half_length >= expected - tolerance,
                    "{post_weights:?} M = {m}: half-length {} below {expected}",
                    result.optimal_half_length
                );
                if post_weights.iter().all(|weight| *weight >= 0.0) {
                    assert!(
                        result.optimal_half_length <= expected + tolerance,
                        "{post_weights:?} M = {m}: half-length {} above {expected}",
                        result.optimal_half_length
                    );
                } else {
                    let excess = result.optimal_half_length - expected;
                    assert!((excess - 2.0 * m).abs() <= tolerance + 1e-6 * m);
                }
                let centre = dot(a, &input.betahat);
                assert_eq!(
                    result.flci.0.to_bits(),
                    (centre - result.optimal_half_length).to_bits()
                );
                assert_eq!(
                    result.flci.1.to_bits(),
                    (centre + result.optimal_half_length).to_bits()
                );
            }
        }
    }
}
