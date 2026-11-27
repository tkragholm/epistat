//! Relative-magnitude (`DeltaRM`) sensitivity analysis for event-study `ATT(g,t)`.
//!
//! This module implements the `\Delta^{RM}(\bar M)` restrictions from
//! Rambachan & Roth (2023), where changes of the violation of parallel trends
//! between adjacent post-periods are bounded by `Mbar` times the largest such
//! change between adjacent pre-periods. The linear-trend variants bound second
//! differences instead.
//!
//! The implementation is split into:
//! - public identified-set and conditional-CS entrypoints in this file
//! - branch geometry / constraint preparation in [`geometry`]
//! - Clarabel-backed confidence-set solver machinery in [`conditional_confidence_set`]
//!
//! References:
//! - Rambachan, A. and Roth, J. (2023), "A More Credible Approach to Parallel Trends"
//! - `HonestDiD` R package, `DeltaRM` path

pub(in crate::sensitivity) mod conditional_confidence_set;
pub(in crate::sensitivity) mod geometry;
pub mod least_favorable_intervals;

use rayon::prelude::*;

use crate::util::InferenceConfig;
use crate::util::normal_interval;

use super::adaptive_grid::{AcceptedGridRange, nearest_grid_index};
use super::{
    HonestBiasDirection, HonestConditionalConfidenceSet, HonestEventStudyInput,
    HonestIdentifiedSet, HonestMonotonicityDirection, HonestOriginalConfidenceSet,
    RelativeMagnitudeConfidenceSetConfig,
};
use crate::sensitivity::linear_algebra::linear_grid;
use crate::sensitivity::smoothness;
use conditional_confidence_set::{
    RelativeMagnitudeConditionalBranch,
    compute_relative_magnitude_branch_accepted_range_for_matrix,
    prepare_relative_magnitude_branch_draws, solve_relative_magnitude_branch_with_matrix,
};
use geometry::prepare_relative_magnitude_functional_transform;

pub(in crate::sensitivity) fn grid_bounds_around_identified_set(
    original: &HonestOriginalConfidenceSet,
    identified: &HonestIdentifiedSet,
) -> (f64, f64) {
    let padding = (20.0 * original.se).max(1e-8);
    let lower = identified.lb.min(original.estimate) - padding;
    let upper = identified.ub.max(original.estimate) + padding;
    if upper > lower {
        (lower, upper)
    } else {
        (lower - padding, upper + padding)
    }
}

#[derive(Debug, Clone, Copy)]
pub(in crate::sensitivity) enum RelativeMagnitudeFamily {
    Base,
    SignedBase(HonestBiasDirection),
    MonotoneBase(HonestMonotonicityDirection),
    Linear,
    SignedLinear(HonestBiasDirection),
    MonotoneLinear(HonestMonotonicityDirection),
}

impl RelativeMagnitudeFamily {
    const fn description(self) -> &'static str {
        match self {
            Self::Base => "relative-magnitude sensitivity",
            Self::SignedBase(_) => "sign-restricted relative-magnitude sensitivity",
            Self::MonotoneBase(_) => "monotone relative-magnitude sensitivity",
            Self::Linear => "linear-trend relative-magnitude sensitivity",
            Self::SignedLinear(_) => "sign-restricted linear-trend relative-magnitude sensitivity",
            Self::MonotoneLinear(_) => "monotone linear-trend relative-magnitude sensitivity",
        }
    }

    const fn requires_two_pre_periods(self) -> bool {
        matches!(
            self,
            Self::Linear | Self::SignedLinear(_) | Self::MonotoneLinear(_)
        )
    }

    fn min_s(self, num_pre: usize) -> Result<isize, String> {
        let num_pre = isize::try_from(num_pre).map_err(|_| "too many pre-periods".to_string())?;
        Ok(match self {
            Self::Base | Self::SignedBase(_) | Self::MonotoneBase(_) => -(num_pre - 1),
            Self::Linear | Self::SignedLinear(_) | Self::MonotoneLinear(_) => -(num_pre - 2),
        })
    }

    const fn empty_message(self) -> &'static str {
        match self {
            Self::Base => "relative-magnitude conditional confidence set accepted no grid points",
            Self::SignedBase(_) => {
                "sign-restricted relative-magnitude conditional confidence set accepted no grid points"
            }
            Self::MonotoneBase(_) => {
                "monotone relative-magnitude conditional confidence set accepted no grid points"
            }
            Self::Linear => {
                "linear-trend relative-magnitude conditional confidence set accepted no grid points"
            }
            Self::SignedLinear(_) => {
                "sign-restricted linear-trend relative-magnitude conditional confidence set accepted no grid points"
            }
            Self::MonotoneLinear(_) => {
                "monotone linear-trend relative-magnitude conditional confidence set accepted no grid points"
            }
        }
    }

    fn build_constraint_matrix(
        self,
        num_pre: usize,
        num_post: usize,
        mbar: f64,
        s: isize,
        max_positive: bool,
    ) -> Result<Vec<Vec<f64>>, String> {
        match self {
            Self::Base => geometry::build_relative_magnitude_constraint_matrix(
                num_pre,
                num_post,
                mbar,
                s,
                max_positive,
            ),
            Self::SignedBase(direction) => build_signed_parallel_trend_constraint_matrix(
                num_pre,
                num_post,
                mbar,
                s,
                max_positive,
                direction,
            ),
            Self::MonotoneBase(direction) => build_monotone_parallel_trend_constraint_matrix(
                num_pre,
                num_post,
                mbar,
                s,
                max_positive,
                direction,
            ),
            Self::Linear => build_linear_trend_relative_magnitude_constraint_matrix(
                num_pre,
                num_post,
                mbar,
                s,
                max_positive,
            ),
            Self::SignedLinear(direction) => build_signed_linear_trend_constraint_matrix(
                num_pre,
                num_post,
                mbar,
                s,
                max_positive,
                direction,
            ),
            Self::MonotoneLinear(direction) => build_monotone_linear_trend_constraint_matrix(
                num_pre,
                num_post,
                mbar,
                s,
                max_positive,
                direction,
            ),
        }
    }
}

fn validate_post_weights(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    require_nonzero: bool,
) -> Result<(), String> {
    if post_weights.len() != input.num_post_periods() {
        return Err(format!(
            "post_weights length {} does not match number of post periods {}",
            post_weights.len(),
            input.num_post_periods()
        ));
    }
    if post_weights.iter().any(|weight| !weight.is_finite()) {
        return Err("post_weights weights must be finite".to_string());
    }
    if require_nonzero && post_weights.iter().all(|weight| weight.abs() <= 1e-12) {
        return Err("post_weights must contain at least one non-zero weight".to_string());
    }
    Ok(())
}

fn validate_relative_magnitude_family_inputs(
    family: RelativeMagnitudeFamily,
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
) -> Result<(), String> {
    input.validate()?;
    if !mbar.is_finite() || mbar < 0.0 {
        return Err(format!(
            "{} requires finite non-negative Mbar, got {mbar}",
            family.description()
        ));
    }
    if family.requires_two_pre_periods() && input.num_pre_periods() <= 1 {
        return Err(format!(
            "{} requires at least two pre-periods",
            family.description()
        ));
    }
    validate_post_weights(input, post_weights, true)
}

fn build_signed_parallel_trend_constraint_matrix(
    num_pre: usize,
    num_post: usize,
    mbar: f64,
    s: isize,
    max_positive: bool,
    bias_direction: HonestBiasDirection,
) -> Result<Vec<Vec<f64>>, String> {
    let mut a_matrix = geometry::build_relative_magnitude_constraint_matrix(
        num_pre,
        num_post,
        mbar,
        s,
        max_positive,
    )?;
    a_matrix.extend(smoothness::build_bias_sign_restriction_matrix(
        num_pre,
        num_post,
        bias_direction,
    ));
    Ok(a_matrix)
}

fn build_monotone_parallel_trend_constraint_matrix(
    num_pre: usize,
    num_post: usize,
    mbar: f64,
    s: isize,
    max_positive: bool,
    monotonicity_direction: HonestMonotonicityDirection,
) -> Result<Vec<Vec<f64>>, String> {
    let mut a_matrix = geometry::build_relative_magnitude_constraint_matrix(
        num_pre,
        num_post,
        mbar,
        s,
        max_positive,
    )?;
    a_matrix.extend(smoothness::build_monotonicity_restriction_matrix(
        num_pre,
        num_post,
        monotonicity_direction,
        false,
    ));
    Ok(a_matrix)
}

fn build_linear_trend_relative_magnitude_constraint_matrix(
    num_pre: usize,
    num_post: usize,
    mbar: f64,
    s: isize,
    max_positive: bool,
) -> Result<Vec<Vec<f64>>, String> {
    let total_diffs = num_pre + num_post - 1;
    let full_cols = num_pre + num_post + 1;
    let mut a_tilde = vec![vec![0.0; full_cols]; total_diffs];
    for (row_idx, row) in a_tilde.iter_mut().enumerate() {
        row[row_idx] = 1.0;
        row[row_idx + 1] = -2.0;
        row[row_idx + 2] = 1.0;
    }

    let start = isize::try_from(num_pre).map_err(|_| "too many pre-periods".to_string())? + s - 2;
    if start < 0
        || start + 2 >= isize::try_from(full_cols).map_err(|_| "too many periods".to_string())?
    {
        return Err(format!(
            "invalid linear-trend relative-magnitude branch index s={s} for num_pre={num_pre}"
        ));
    }
    let start = usize::try_from(start)
        .map_err(|_| "invalid linear-trend relative-magnitude branch index".to_string())?;
    let mut v_max = vec![0.0; full_cols];
    v_max[start] = 1.0;
    v_max[start + 1] = -2.0;
    v_max[start + 2] = 1.0;
    if !max_positive {
        for value in &mut v_max {
            *value = -*value;
        }
    }

    let mut a_ub = Vec::with_capacity(total_diffs);
    for _ in 0..num_pre.saturating_sub(1) {
        a_ub.push(v_max.clone());
    }
    for _ in 0..num_post {
        a_ub.push(v_max.iter().map(|value| mbar * value).collect());
    }

    let mut constraints = Vec::with_capacity(total_diffs * 2);
    for (tilde_row, ub_row) in a_tilde.iter().zip(&a_ub) {
        constraints.push(
            tilde_row
                .iter()
                .zip(ub_row)
                .map(|(left, right)| left - right)
                .collect::<Vec<_>>(),
        );
        constraints.push(
            tilde_row
                .iter()
                .zip(ub_row)
                .map(|(left, right)| -left - right)
                .collect::<Vec<_>>(),
        );
    }

    let zero_col = num_pre;
    let mut dropped = Vec::new();
    for row in constraints {
        let mut compact = Vec::with_capacity(num_pre + num_post);
        for (col_idx, value) in row.into_iter().enumerate() {
            if col_idx != zero_col {
                compact.push(value);
            }
        }
        if compact.iter().any(|value| value.abs() > 1e-10) {
            dropped.push(compact);
        }
    }
    Ok(dropped)
}

fn build_signed_linear_trend_constraint_matrix(
    num_pre: usize,
    num_post: usize,
    mbar: f64,
    s: isize,
    max_positive: bool,
    bias_direction: HonestBiasDirection,
) -> Result<Vec<Vec<f64>>, String> {
    let mut a_matrix = build_linear_trend_relative_magnitude_constraint_matrix(
        num_pre,
        num_post,
        mbar,
        s,
        max_positive,
    )?;
    a_matrix.extend(smoothness::build_bias_sign_restriction_matrix(
        num_pre,
        num_post,
        bias_direction,
    ));
    Ok(a_matrix)
}

fn build_monotone_linear_trend_constraint_matrix(
    num_pre: usize,
    num_post: usize,
    mbar: f64,
    s: isize,
    max_positive: bool,
    monotonicity_direction: HonestMonotonicityDirection,
) -> Result<Vec<Vec<f64>>, String> {
    let mut a_matrix = build_linear_trend_relative_magnitude_constraint_matrix(
        num_pre,
        num_post,
        mbar,
        s,
        max_positive,
    )?;
    a_matrix.extend(smoothness::build_monotonicity_restriction_matrix(
        num_pre,
        num_post,
        monotonicity_direction,
        false,
    ));
    Ok(a_matrix)
}

fn compute_relative_magnitude_identified_set_with_builder<F>(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    min_s: isize,
    build_matrix: F,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestIdentifiedSet, String>
where
    F: Fn(isize, bool) -> Result<Vec<Vec<f64>>, String> + Sync,
{
    let num_pre = input.num_pre_periods();
    let num_post = input.num_post_periods();
    if num_pre == 0 {
        return Err("relative-magnitude sensitivity requires at least one pre-period".to_string());
    }
    let branches: Vec<(isize, bool)> = (min_s..=0)
        .flat_map(|s| {
            [true, false]
                .into_iter()
                .map(move |max_positive| (s, max_positive))
        })
        .collect();
    let branch_results: Result<Vec<Option<HonestIdentifiedSet>>, String> = branches
        .into_par_iter()
        .map(|(s, max_positive)| {
            let inequality_matrix = build_matrix(s, max_positive)?;
            solve_relative_magnitude_branch_with_matrix(
                num_pre,
                num_post,
                &input.betahat,
                post_weights,
                &inequality_matrix,
                counter,
            )
        })
        .collect();
    let branch_results = branch_results?;

    let mut global_lower = f64::INFINITY;
    let mut global_upper = f64::NEG_INFINITY;
    let mut solved_any_branch = false;
    for branch in branch_results.into_iter().flatten() {
        solved_any_branch = true;
        global_lower = global_lower.min(branch.lb);
        global_upper = global_upper.max(branch.ub);
    }
    // No branch admits a violation path through the pre-period coefficients:
    // the set is empty, which is undefined, as for the smoothness classes.
    if solved_any_branch {
        Ok(HonestIdentifiedSet {
            lb: global_lower,
            ub: global_upper,
        })
    } else {
        Ok(HonestIdentifiedSet {
            lb: f64::NAN,
            ub: f64::NAN,
        })
    }
}

fn compute_relative_magnitude_confidence_set_with_builder<F>(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
    original: &HonestOriginalConfidenceSet,
    identified: &HonestIdentifiedSet,
    min_s: isize,
    build_matrix: F,
    empty_message: &'static str,
    shared_draws: Option<&[Vec<f64>]>,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestConditionalConfidenceSet, String>
where
    F: Fn(isize, bool) -> Result<Vec<Vec<f64>>, String> + Sync,
{
    let num_pre = input.num_pre_periods();
    let (grid_lower, grid_upper) = grid_bounds_around_identified_set(original, identified);
    let grid_points = 1_000usize;
    let alpha = 1.0 - inference.confidence_level;
    let prepared_transform = prepare_relative_magnitude_functional_transform(post_weights)?;
    let grid = linear_grid(grid_lower, grid_upper, grid_points);
    let anchor_idx = nearest_grid_index(&grid, original.estimate);

    let branches: Vec<(isize, bool)> = (min_s..=0)
        .flat_map(|s| {
            [true, false]
                .into_iter()
                .map(move |max_positive| (s, max_positive))
        })
        .collect();
    if let Some(shared_draws) = shared_draws
        && shared_draws.len() != branches.len()
    {
        return Err(format!(
            "shared least-favorable draws cover {} branches but the family has {}",
            shared_draws.len(),
            branches.len()
        ));
    }
    let branch_acceptance: Result<Vec<Option<AcceptedGridRange>>, String> = branches
        .into_par_iter()
        .enumerate()
        .map(|(branch_idx, (s, max_positive))| {
            let inequality_matrix = build_matrix(s, max_positive)?;
            let branch_problem = RelativeMagnitudeConditionalBranch {
                input,
                num_pre,
                prepared_transform: &prepared_transform,
                alpha,
                hybrid: config.hybrid,
                hybrid_kappa: config.hybrid_kappa,
                least_favorable_draws: config.least_favorable_draws,
                least_favorable_seed: config.least_favorable_seed,
                grid: &grid,
                anchor_idx,
            };
            compute_relative_magnitude_branch_accepted_range_for_matrix(
                &branch_problem,
                &inequality_matrix,
                shared_draws.map(|draws| draws[branch_idx].as_slice()),
                counter,
            )
        })
        .collect();
    let branch_acceptance = branch_acceptance?;
    let accepted_range = branch_acceptance.into_iter().flatten().fold(
        None,
        |acc: Option<AcceptedGridRange>, range| {
            Some(acc.map_or(range, |existing| AcceptedGridRange {
                lower_idx: existing.lower_idx.min(range.lower_idx),
                upper_idx: existing.upper_idx.max(range.upper_idx),
            }))
        },
    );
    let Some(accepted_range) = accepted_range else {
        return Err(empty_message.to_string());
    };
    Ok(HonestConditionalConfidenceSet {
        lb: grid[accepted_range.lower_idx],
        ub: grid[accepted_range.upper_idx],
    })
}

/// Two-sided Wald confidence set of `post_weights' τ_post` under exact parallel
/// trends, at `inference.confidence_level`.
///
/// This matches `HonestDiD::constructOriginalCS`. The standard error comes from
/// the post-treatment block of the covariance matrix.
///
/// # Errors
/// Returns an error if `input` is invalid, or if `post_weights` does not have
/// one finite weight per post-treatment period.
pub fn compute_original_confidence_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    inference: InferenceConfig,
) -> Result<HonestOriginalConfidenceSet, String> {
    input.validate()?;
    validate_post_weights(input, post_weights, false)?;
    let post_offset = input.num_pre_periods();
    let post_betahat = &input.betahat[post_offset..];
    let estimate = post_weights
        .iter()
        .zip(post_betahat)
        .map(|(weight, beta)| weight * beta)
        .sum::<f64>();
    let mut variance = 0.0;
    for (i, left_weight) in post_weights.iter().enumerate() {
        for (j, right_weight) in post_weights.iter().enumerate() {
            variance = (left_weight * right_weight)
                .mul_add(input.covariance[post_offset + i][post_offset + j], variance);
        }
    }
    let se = variance.max(0.0).sqrt();
    Ok(HonestOriginalConfidenceSet {
        estimate,
        se,
        ci: normal_interval(estimate, se, inference.confidence_level),
    })
}

pub(in crate::sensitivity) fn compute_relative_magnitude_family_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    family: RelativeMagnitudeFamily,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestIdentifiedSet, String> {
    validate_relative_magnitude_family_inputs(family, input, post_weights, mbar)?;
    let num_pre = input.num_pre_periods();
    let num_post = input.num_post_periods();
    let min_s = family.min_s(num_pre)?;
    compute_relative_magnitude_identified_set_with_builder(
        input,
        post_weights,
        min_s,
        move |s, max_positive| {
            family.build_constraint_matrix(num_pre, num_post, mbar, s, max_positive)
        },
        counter,
    )
}

fn compute_relative_magnitude_family_conditional_cs_with_config(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
    family: RelativeMagnitudeFamily,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestConditionalConfidenceSet, String> {
    validate_relative_magnitude_family_inputs(family, input, post_weights, mbar)?;
    config.validate(inference)?;
    let original = compute_original_confidence_set(input, post_weights, inference)?;
    let identified = compute_relative_magnitude_family_identified_set(
        input,
        post_weights,
        mbar,
        family,
        counter,
    )?;
    compute_relative_magnitude_family_conditional_cs_with_precomputed_sets(
        input,
        post_weights,
        mbar,
        inference,
        config,
        &original,
        &identified,
        family,
        None,
        counter,
    )
}

pub(in crate::sensitivity) fn compute_relative_magnitude_family_conditional_cs_with_precomputed_sets(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
    original: &HonestOriginalConfidenceSet,
    identified: &HonestIdentifiedSet,
    family: RelativeMagnitudeFamily,
    shared_draws: Option<&[Vec<f64>]>,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestConditionalConfidenceSet, String> {
    let num_pre = input.num_pre_periods();
    let num_post = input.num_post_periods();
    compute_relative_magnitude_confidence_set_with_builder(
        input,
        post_weights,
        inference,
        config,
        original,
        identified,
        family.min_s(num_pre)?,
        move |s, max_positive| {
            family.build_constraint_matrix(num_pre, num_post, mbar, s, max_positive)
        },
        family.empty_message(),
        shared_draws,
        counter,
    )
}

/// The least-favorable simulation draws of every branch of `family` at `mbar`,
/// in the branch order the confidence set walks.
///
/// The draws depend on a branch's ARP covariance and nothing else, so every
/// functional bounded at this `mbar` uses the same ones. Made once here and
/// handed to each functional's confidence set, they cost a twentieth of what
/// twenty functionals making their own did, for the same critical values.
///
/// # Errors
/// Returns an error if a branch's constraint matrix cannot be built or its ARP
/// covariance is not positive definite.
pub(in crate::sensitivity) fn prepare_relative_magnitude_family_draws(
    input: &HonestEventStudyInput,
    mbar: f64,
    family: RelativeMagnitudeFamily,
    draws: usize,
    seed: u64,
) -> Result<Vec<Vec<f64>>, String> {
    let num_pre = input.num_pre_periods();
    let num_post = input.num_post_periods();
    let min_s = family.min_s(num_pre)?;
    let branches: Vec<(isize, bool)> = (min_s..=0)
        .flat_map(|s| {
            [true, false]
                .into_iter()
                .map(move |max_positive| (s, max_positive))
        })
        .collect();
    branches
        .into_par_iter()
        .map(|(s, max_positive)| {
            let inequality_matrix =
                family.build_constraint_matrix(num_pre, num_post, mbar, s, max_positive)?;
            prepare_relative_magnitude_branch_draws(input, num_pre, &inequality_matrix, draws, seed)
        })
        .collect()
}

/// Identified set of the functional `post_weights' τ_post` under `ΔRM(Mbar)`.
///
/// `post_weights` has one weight per post period, finite and not all zero.
/// `mbar` is finite and non-negative. The set is the union over the
/// restriction's branches of the range of the functional that satisfies the
/// constraints and the pre-period coefficients. If no branch is feasible the
/// set is empty and both bounds are NaN.
///
/// # Errors
/// Returns an error if the input is invalid, `post_weights` or `mbar` is
/// invalid, there are too few pre-periods, or a branch program fails.
#[cfg(test)]
pub fn compute_relative_magnitude_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
) -> Result<HonestIdentifiedSet, String> {
    compute_relative_magnitude_identified_set_counted(
        input,
        post_weights,
        mbar,
        &crate::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::sensitivity) fn compute_relative_magnitude_identified_set_counted(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestIdentifiedSet, String> {
    compute_relative_magnitude_family_identified_set(
        input,
        post_weights,
        mbar,
        RelativeMagnitudeFamily::Base,
        counter,
    )
}

/// Conditional confidence set of the functional `post_weights' τ_post` under
/// `ΔRM(Mbar)`, with the settings of
/// [`RelativeMagnitudeConfidenceSetConfig::from_inference`].
///
/// The set is the range of accepted values on a grid of 1,000 equally spaced
/// points, so its ends are grid points. The grid spans the identified set and
/// the original estimate, widened by 20 standard errors on each side.
///
/// # Errors
/// Returns an error if the input, `post_weights` or `mbar` is invalid, or if no
/// grid point is accepted.
#[cfg(test)]
pub fn compute_relative_magnitude_confidence_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_relative_magnitude_confidence_set_counted(
        input,
        post_weights,
        mbar,
        inference,
        &crate::sensitivity::SolverCounter::new(),
    )
}

#[cfg(test)]
pub(in crate::sensitivity) fn compute_relative_magnitude_confidence_set_counted(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_relative_magnitude_confidence_set_with_config_counted(
        input,
        post_weights,
        mbar,
        inference,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        counter,
    )
}

/// [`compute_relative_magnitude_confidence_set`] with the hybrid settings given
/// in `config`.
///
/// `config.hybrid_kappa` must lie in `[0, alpha]` and
/// `config.least_favorable_draws` must be positive.
///
/// # Errors
/// Returns an error if `config` fails validation, if the input, `post_weights`
/// or `mbar` is invalid, or if no grid point is accepted.
#[cfg(test)]
pub fn compute_relative_magnitude_confidence_set_with_config(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_relative_magnitude_confidence_set_with_config_counted(
        input,
        post_weights,
        mbar,
        inference,
        config,
        &crate::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::sensitivity) fn compute_relative_magnitude_confidence_set_with_config_counted(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_relative_magnitude_family_conditional_cs_with_config(
        input,
        post_weights,
        mbar,
        inference,
        config,
        RelativeMagnitudeFamily::Base,
        counter,
    )
}

pub(in crate::sensitivity) fn compute_relative_magnitude_confidence_set_with_precomputed_sets(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    config: RelativeMagnitudeConfidenceSetConfig,
    original: &HonestOriginalConfidenceSet,
    identified: &HonestIdentifiedSet,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestConditionalConfidenceSet, String> {
    compute_relative_magnitude_family_conditional_cs_with_precomputed_sets(
        input,
        post_weights,
        mbar,
        inference,
        config,
        original,
        identified,
        RelativeMagnitudeFamily::Base,
        None,
        counter,
    )
}

/// Identified set of the functional `post_weights' τ_post` under `ΔRM(Mbar)`
/// with a monotonicity restriction given by `monotonicity_direction`.
///
/// `post_weights` has one weight per post period, finite and not all zero.
/// `mbar` is finite and non-negative. The set is the union over the
/// restriction's branches of the range of the functional that satisfies the
/// constraints and the pre-period coefficients. If no branch is feasible the
/// set is empty and both bounds are NaN.
///
/// # Errors
/// Returns an error if the input is invalid, `post_weights` or `mbar` is
/// invalid, there are too few pre-periods, or a branch program fails.
#[cfg(test)]
pub fn compute_monotone_relative_magnitude_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    monotonicity_direction: HonestMonotonicityDirection,
) -> Result<HonestIdentifiedSet, String> {
    compute_monotone_relative_magnitude_identified_set_counted(
        input,
        post_weights,
        mbar,
        monotonicity_direction,
        &crate::sensitivity::SolverCounter::new(),
    )
}

#[cfg(test)]
pub(in crate::sensitivity) fn compute_monotone_relative_magnitude_identified_set_counted(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    monotonicity_direction: HonestMonotonicityDirection,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestIdentifiedSet, String> {
    compute_relative_magnitude_family_identified_set(
        input,
        post_weights,
        mbar,
        RelativeMagnitudeFamily::MonotoneBase(monotonicity_direction),
        counter,
    )
}

/// Identified set of the functional `post_weights' τ_post` under `ΔSDRM(Mbar)`,
/// where the bound is on deviations from a linear trend.
///
/// `post_weights` has one weight per post period, finite and not all zero.
/// `mbar` is finite and non-negative. Needs at least two pre-periods. The set
/// is the union over the restriction's branches of the range of the functional
/// that satisfies the constraints and the pre-period coefficients. If no branch
/// is feasible the set is empty and both bounds are NaN.
///
/// # Errors
/// Returns an error if the input is invalid, `post_weights` or `mbar` is
/// invalid, there are too few pre-periods, or a branch program fails.
#[cfg(test)]
pub fn compute_linear_trend_relative_magnitude_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
) -> Result<HonestIdentifiedSet, String> {
    compute_linear_trend_relative_magnitude_identified_set_counted(
        input,
        post_weights,
        mbar,
        &crate::sensitivity::SolverCounter::new(),
    )
}

#[cfg(test)]
pub(in crate::sensitivity) fn compute_linear_trend_relative_magnitude_identified_set_counted(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestIdentifiedSet, String> {
    compute_relative_magnitude_family_identified_set(
        input,
        post_weights,
        mbar,
        RelativeMagnitudeFamily::Linear,
        counter,
    )
}

/// Identified set of the functional `post_weights' τ_post` under `ΔSDRM(Mbar)`
/// with a sign restriction on the bias given by `bias_direction`.
///
/// `post_weights` has one weight per post period, finite and not all zero.
/// `mbar` is finite and non-negative. Needs at least two pre-periods. The set
/// is the union over the restriction's branches of the range of the functional
/// that satisfies the constraints and the pre-period coefficients. If no branch
/// is feasible the set is empty and both bounds are NaN.
///
/// # Errors
/// Returns an error if the input is invalid, `post_weights` or `mbar` is
/// invalid, there are too few pre-periods, or a branch program fails.
#[cfg(test)]
pub fn compute_signed_linear_trend_relative_magnitude_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    bias_direction: HonestBiasDirection,
) -> Result<HonestIdentifiedSet, String> {
    compute_signed_linear_trend_relative_magnitude_identified_set_counted(
        input,
        post_weights,
        mbar,
        bias_direction,
        &crate::sensitivity::SolverCounter::new(),
    )
}

#[cfg(test)]
pub(in crate::sensitivity) fn compute_signed_linear_trend_relative_magnitude_identified_set_counted(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    bias_direction: HonestBiasDirection,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestIdentifiedSet, String> {
    compute_relative_magnitude_family_identified_set(
        input,
        post_weights,
        mbar,
        RelativeMagnitudeFamily::SignedLinear(bias_direction),
        counter,
    )
}

/// Identified set of the functional `post_weights' τ_post` under `ΔSDRM(Mbar)`
/// with a monotonicity restriction given by `monotonicity_direction`.
///
/// `post_weights` has one weight per post period, finite and not all zero.
/// `mbar` is finite and non-negative. Needs at least two pre-periods. The set
/// is the union over the restriction's branches of the range of the functional
/// that satisfies the constraints and the pre-period coefficients. If no branch
/// is feasible the set is empty and both bounds are NaN.
///
/// # Errors
/// Returns an error if the input is invalid, `post_weights` or `mbar` is
/// invalid, there are too few pre-periods, or a branch program fails.
#[cfg(test)]
pub fn compute_monotone_linear_trend_relative_magnitude_identified_set(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    monotonicity_direction: HonestMonotonicityDirection,
) -> Result<HonestIdentifiedSet, String> {
    compute_monotone_linear_trend_relative_magnitude_identified_set_counted(
        input,
        post_weights,
        mbar,
        monotonicity_direction,
        &crate::sensitivity::SolverCounter::new(),
    )
}

#[cfg(test)]
pub(in crate::sensitivity) fn compute_monotone_linear_trend_relative_magnitude_identified_set_counted(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    monotonicity_direction: HonestMonotonicityDirection,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestIdentifiedSet, String> {
    compute_relative_magnitude_family_identified_set(
        input,
        post_weights,
        mbar,
        RelativeMagnitudeFamily::MonotoneLinear(monotonicity_direction),
        counter,
    )
}
