#[path = "event_study_regions.rs"]
mod regions;
#[path = "event_study_workflow.rs"]
mod workflow;

use crate::types::{AttGtEstimate, InferenceConfig};
use rayon::prelude::*;

use super::relative_magnitude::conditional_confidence_set::{
    RelativeMagnitudePreparedFunctionalBranch, RelativeMagnitudePreparedInputBranch,
    compute_relative_magnitude_branch_identified_sets_with_prepared_branches,
    compute_relative_magnitude_confidence_set_for_basis_period_with_prepared_input,
    compute_relative_magnitude_confidence_set_with_prepared_functional_branches_and_identified_sets,
    compute_relative_magnitude_identified_set_for_basis_period,
    global_identified_set_from_branch_identified_sets,
    prepare_relative_magnitude_functional_branches, prepare_relative_magnitude_input_branches,
};
use super::relative_magnitude::geometry::{
    RelativeMagnitudePreparedBranch, basis_post_weights, prepare_relative_magnitude_branches,
};
use super::{
    HonestAssessment, HonestEventStudyInput, HonestEventStudyPoint, HonestJointPathConfig,
    HonestMultiFlciResult, HonestPostFunctional, HonestResult, HonestSensitivity,
    LeastFavorableSimulation, RelativeMagnitudeConfidenceSetConfig, SmoothnessConfidenceSetConfig,
    relative_magnitude::{
        compute_original_confidence_set,
        compute_relative_magnitude_confidence_set_with_precomputed_sets,
    },
};
use crate::inference::normal_interval;
use crate::util::usize_to_f64;

pub use regions::{
    assess_honest_event_study_directional_region,
    assess_honest_event_study_directional_region_with_config,
    assess_honest_event_study_joint_path_region,
    assess_honest_event_study_joint_path_region_with_config,
    assess_honest_event_study_optimization_surface_region,
    assess_honest_event_study_optimization_surface_region_adaptive,
    assess_honest_event_study_optimization_surface_region_adaptive_with_config,
    assess_honest_event_study_optimization_surface_region_with_config,
};
pub use workflow::{
    assess_honest_event_study_workflow, assess_honest_event_study_workflow_with_config,
};

/// Simple sensitivity interval for one post-treatment estimate, from the
/// estimates alone.
///
/// This is a closed-form approximation, not the conditional confidence set of
/// the event-study functions. The identified set is built around the target
/// estimate `tau`:
///
/// - `Smoothness(M)`: `[tau + b - M, tau + b + M]`, where `b` is the estimate of
///   the last pre-period.
/// - `RelativeMagnitude(Mbar)`: `[tau - d, tau + d]`, where
///   `d = Mbar * max |pre-period estimate|`.
///
/// The robust interval runs from the lower end of the two-sided Wald interval
/// around the lower end of the identified set to the upper end of the Wald
/// interval around the upper end, using `target_estimate.se`.
///
/// # Arguments
/// * `target_estimate` - The post-treatment estimate and its standard error.
/// * `pre_period_estimates` - Estimates for the pre-treatment periods, the last
///   one adjacent to treatment.
/// * `vcov` - Matrix of side `pre_period_estimates.len() + 1`. Only its
///   dimensions are checked.
/// * `sensitivity` - The restriction and its parameter.
/// * `inference` - The confidence level.
///
/// # Errors
/// Returns an error if covariance dimensions are inconsistent or the
/// sensitivity problem is numerically invalid for the supplied target and
/// pre-period estimates.
pub fn compute_honest_ci_scalar(
    target_estimate: &AttGtEstimate,
    pre_period_estimates: &[AttGtEstimate],
    vcov: &[Vec<f64>],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
) -> Result<HonestResult, String> {
    let k = pre_period_estimates.len();
    if vcov.len() != k + 1 || vcov[0].len() != k + 1 {
        return Err("VCOV matrix dimensions must match pre_periods + 1".to_string());
    }

    let (identified_lower, identified_upper) = match sensitivity {
        HonestSensitivity::Smoothness(m) => {
            let last_pre = pre_period_estimates
                .last()
                .ok_or("requires at least one pre-period")?;
            let beta_pre = last_pre.att;
            let delta_min = -beta_pre - m;
            let delta_max = -beta_pre + m;
            (
                target_estimate.att - delta_max,
                target_estimate.att - delta_min,
            )
        }
        HonestSensitivity::RelativeMagnitude(m) => {
            let max_pre_bias = pre_period_estimates
                .iter()
                .map(|estimate| estimate.att.abs())
                .fold(0.0, f64::max);

            let delta_bound = m * max_pre_bias;
            (
                target_estimate.att - delta_bound,
                target_estimate.att + delta_bound,
            )
        }
    };

    let se = target_estimate.se;
    let robust_lower = normal_interval(identified_lower, se, inference.confidence_level).0;
    let robust_upper = normal_interval(identified_upper, se, inference.confidence_level).1;

    Ok(HonestResult {
        original_estimate: target_estimate.att,
        original_se: target_estimate.se,
        robust_ci: (robust_lower, robust_upper),
        identified_set: (identified_lower, identified_upper),
        m: match sensitivity {
            HonestSensitivity::Smoothness(m) | HonestSensitivity::RelativeMagnitude(m) => m,
        },
    })
}

/// Compute a scalar `HonestDiD` result for each post-treatment event-study
/// coefficient, each period on its own.
///
/// `betahat` holds the pre-period estimates followed by the post-period
/// estimates, and `covariance` is its covariance matrix. The labels in
/// `pre_periods` and `post_periods` give the split, and the post labels are
/// returned in the results. The robust interval is the conditional confidence
/// set with the default configuration for the restriction.
///
/// # Errors
/// Returns an error if dimensions are inconsistent or covariance entries are
/// invalid.
pub fn compute_honest_event_study_scalar(
    betahat: &[f64],
    covariance: &[Vec<f64>],
    pre_periods: &[i32],
    post_periods: &[i32],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
) -> Result<Vec<HonestEventStudyPoint>, String> {
    compute_honest_event_study_scalar_counted(
        betahat,
        covariance,
        pre_periods,
        post_periods,
        sensitivity,
        inference,
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::inference::sensitivity) fn compute_honest_event_study_scalar_counted(
    betahat: &[f64],
    covariance: &[Vec<f64>],
    pre_periods: &[i32],
    post_periods: &[i32],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<Vec<HonestEventStudyPoint>, String> {
    let input = HonestEventStudyInput {
        betahat: betahat.to_vec(),
        covariance: covariance.to_vec(),
        pre_periods: pre_periods.to_vec(),
        post_periods: post_periods.to_vec(),
    };
    compute_honest_event_study_scalar_for_input_counted(&input, sensitivity, inference, counter)
}

/// Compute a scalar `HonestDiD` result for each post-treatment coefficient of
/// `input`, in the order of its post periods. See
/// [`compute_honest_event_study_scalar`].
///
/// # Errors
/// Returns an error if `input` is inconsistent.
pub fn compute_honest_event_study_scalar_for_input(
    input: &HonestEventStudyInput,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
) -> Result<Vec<HonestEventStudyPoint>, String> {
    compute_honest_event_study_scalar_for_input_counted(
        input,
        sensitivity,
        inference,
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::inference::sensitivity) fn compute_honest_event_study_scalar_for_input_counted(
    input: &HonestEventStudyInput,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<Vec<HonestEventStudyPoint>, String> {
    input.validate()?;

    let prepared_branches: Option<Vec<RelativeMagnitudePreparedBranch>> = match sensitivity {
        HonestSensitivity::RelativeMagnitude(mbar) => Some(prepare_relative_magnitude_branches(
            input.num_pre_periods(),
            input.num_post_periods(),
            mbar,
        )?),
        HonestSensitivity::Smoothness(_) => None,
    };
    let prepared_input_branches = prepared_branches
        .as_ref()
        .map(|prepared| prepare_relative_magnitude_input_branches(input, prepared));

    input
        .post_periods
        .par_iter()
        .enumerate()
        .map(|(post_idx, period)| {
            let result = match sensitivity {
                HonestSensitivity::RelativeMagnitude(mbar) => {
                    let prepared = prepared_branches
                        .as_ref()
                        .ok_or("missing prepared relative-magnitude branch geometry")?;
                    let original = compute_original_confidence_set(
                        input,
                        &basis_post_weights(input.num_post_periods(), post_idx),
                        inference,
                    )?;
                    let identified = compute_relative_magnitude_identified_set_for_basis_period(
                        input, post_idx, prepared, counter,
                    )?;
                    let prepared_input = prepared_input_branches
                        .as_ref()
                        .ok_or("missing prepared relative-magnitude input branch geometry")?;
                    let conditional =
                        compute_relative_magnitude_confidence_set_for_basis_period_with_prepared_input(
                            input,
                            post_idx,
                            inference,
                            RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
                            prepared,
                            prepared_input, counter,
                        )?;
                    HonestResult {
                        original_estimate: original.estimate,
                        original_se: original.se,
                        robust_ci: (conditional.lb, conditional.ub),
                        identified_set: (identified.lb, identified.ub),
                        m: mbar,
                    }
                }
                HonestSensitivity::Smoothness(m) => {
                    let post_weights = basis_post_weights(input.num_post_periods(), post_idx);
                    let original =
                        compute_original_confidence_set(input, &post_weights, inference)?;
                    let identified = crate::inference::sensitivity::smoothness::compute_smoothness_identified_set_counted(input, &post_weights, m, counter)?;
                    let conditional = crate::inference::sensitivity::smoothness::compute_smoothness_confidence_set_with_config_counted(
                        input,
                        &post_weights,
                        m,
                        inference,
                        SmoothnessConfidenceSetConfig::from_inference(inference), LeastFavorableSimulation::default(), counter,
                    )?;
                    HonestResult {
                        original_estimate: original.estimate,
                        original_se: original.se,
                        robust_ci: (conditional.lb, conditional.ub),
                        identified_set: (identified.lb, identified.ub),
                        m,
                    }
                }
            };
            Ok(HonestEventStudyPoint {
                post_period: *period,
                result,
            })
        })
        .collect()
}

/// Assess whether a linear functional of the post-treatment coefficients
/// remains distinguishable from `null_value` under the `HonestDiD`
/// sensitivity restriction.
///
/// `post_weights` has one weight per post period, finite and not all zero. For
/// one period it is the unit vector `e_t`. For an average over five periods it
/// is `1/5` on those periods and 0 elsewhere.
///
/// Under `RelativeMagnitude(Mbar)` the identified set and the conditional
/// confidence set are exact for any `post_weights`, with the settings of
/// [`RelativeMagnitudeConfidenceSetConfig::from_inference`]. Under
/// `Smoothness(M)` the conditional confidence set uses the settings of
/// [`SmoothnessConfidenceSetConfig::from_inference`]. Its identified set is
/// NaN where it is empty, as in
/// [`compute_smoothness_identified_set`](super::compute_smoothness_identified_set).
///
/// # Errors
/// Returns an error if the event-study input is inconsistent or if the chosen
/// sensitivity routine fails.
pub fn assess_honest_event_study_functional(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
) -> Result<HonestAssessment, String> {
    assess_honest_event_study_functional_counted(
        input,
        post_weights,
        sensitivity,
        inference,
        null_value,
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::inference::sensitivity) fn assess_honest_event_study_functional_counted(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<HonestAssessment, String> {
    assess_honest_event_study_functional_with_config_counted(
        input,
        post_weights,
        sensitivity,
        inference,
        null_value,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        counter,
    )
}

/// [`assess_honest_event_study_functional`] with the `DeltaRM` conditional
/// confidence-set settings given explicitly.
///
/// `relative_magnitude_config` applies only to
/// [`HonestSensitivity::RelativeMagnitude`]. A smoothness restriction ignores
/// it. The default of [`assess_honest_event_study_functional`] is
/// [`RelativeMagnitudeConfidenceSetConfig::from_inference`], the least-favorable
/// (`LF`) hybrid of the reference `HonestDiD` package.
///
/// # Errors
/// Returns an error if the event-study input is inconsistent or if the chosen
/// sensitivity routine fails.
pub fn assess_honest_event_study_functional_with_config(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<HonestAssessment, String> {
    assess_honest_event_study_functional_with_config_counted(
        input,
        post_weights,
        sensitivity,
        inference,
        null_value,
        relative_magnitude_config,
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::inference::sensitivity) fn assess_honest_event_study_functional_with_config_counted(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<HonestAssessment, String> {
    let original = compute_original_confidence_set(input, post_weights, inference)?;
    let (robust_ci, identified_set, sensitivity_value) = match sensitivity {
        HonestSensitivity::RelativeMagnitude(mbar) => {
            let identified =
                super::relative_magnitude::compute_relative_magnitude_identified_set_counted(
                    input,
                    post_weights,
                    mbar,
                    counter,
                )?;
            let conditional = compute_relative_magnitude_confidence_set_with_precomputed_sets(
                input,
                post_weights,
                mbar,
                inference,
                relative_magnitude_config,
                &original,
                &identified,
                counter,
            )?;
            (
                (conditional.lb, conditional.ub),
                (identified.lb, identified.ub),
                mbar,
            )
        }
        HonestSensitivity::Smoothness(m) => {
            if post_weights.len() != input.num_post_periods() {
                return Err(format!(
                    "post_weights length {} does not match number of post periods {}",
                    post_weights.len(),
                    input.num_post_periods()
                ));
            }
            if post_weights.iter().all(|weight| weight.abs() <= 1e-12) {
                return Err(
                    "Smoothness sensitivity requires at least one non-zero post_weights weight"
                        .to_string(),
                );
            }
            let identified = crate::inference::sensitivity::smoothness::compute_smoothness_identified_set_counted(input, post_weights, m, counter)?;
            let conditional = crate::inference::sensitivity::smoothness::compute_smoothness_confidence_set_with_config_counted(
                input,
                post_weights,
                m,
                inference,
                SmoothnessConfidenceSetConfig::from_inference(inference), LeastFavorableSimulation::default(), counter,
            )?;
            (
                (conditional.lb, conditional.ub),
                (identified.lb, identified.ub),
                m,
            )
        }
    };
    Ok(HonestAssessment {
        original_ci: original.ci,
        robust_ci,
        identified_set,
        estimate: original.estimate,
        se: original.se,
        null_value,
        sensitivity_value,
        robustly_significant: robust_ci.0 > null_value || robust_ci.1 < null_value,
    })
}

/// [`assess_honest_event_study_functional`] for the coefficient of the post
/// period with the label `post_period`.
///
/// # Errors
/// Returns an error if `post_period` is not present in the event-study input or
/// if the underlying `HonestDiD` calculation fails.
pub fn assess_honest_event_study_period(
    input: &HonestEventStudyInput,
    post_period: i32,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
) -> Result<HonestAssessment, String> {
    assess_honest_event_study_period_counted(
        input,
        post_period,
        sensitivity,
        inference,
        null_value,
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::inference::sensitivity) fn assess_honest_event_study_period_counted(
    input: &HonestEventStudyInput,
    post_period: i32,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<HonestAssessment, String> {
    let Some(post_idx) = input
        .post_periods
        .iter()
        .position(|period| *period == post_period)
    else {
        return Err(format!(
            "post period {post_period} is not present in HonestEventStudyInput"
        ));
    };
    let mut post_weights = vec![0.0; input.num_post_periods()];
    post_weights[post_idx] = 1.0;
    assess_honest_event_study_functional_counted(
        input,
        &post_weights,
        sensitivity,
        inference,
        null_value,
        counter,
    )
}

/// [`assess_honest_event_study_period`] with the `DeltaRM` conditional
/// confidence-set settings given explicitly. A smoothness restriction ignores
/// them.
///
/// # Errors
/// Returns an error if `post_period` is not present in the event-study input or
/// if the underlying `HonestDiD` calculation fails.
pub fn assess_honest_event_study_period_with_config(
    input: &HonestEventStudyInput,
    post_period: i32,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<HonestAssessment, String> {
    assess_honest_event_study_period_with_config_counted(
        input,
        post_period,
        sensitivity,
        inference,
        null_value,
        relative_magnitude_config,
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::inference::sensitivity) fn assess_honest_event_study_period_with_config_counted(
    input: &HonestEventStudyInput,
    post_period: i32,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<HonestAssessment, String> {
    let Some(post_idx) = input
        .post_periods
        .iter()
        .position(|period| *period == post_period)
    else {
        return Err(format!(
            "post period {post_period} is not present in HonestEventStudyInput"
        ));
    };
    let mut post_weights = vec![0.0; input.num_post_periods()];
    post_weights[post_idx] = 1.0;
    assess_honest_event_study_functional_with_config_counted(
        input,
        &post_weights,
        sensitivity,
        inference,
        null_value,
        relative_magnitude_config,
        counter,
    )
}

fn post_weights_for_functional(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
) -> Result<Vec<f64>, String> {
    let mut post_weights = vec![0.0; input.num_post_periods()];
    match functional {
        HonestPostFunctional::Period(post_period) => {
            let Some(post_idx) = input
                .post_periods
                .iter()
                .position(|period| period == post_period)
            else {
                return Err(format!(
                    "post period {post_period} is not present in HonestEventStudyInput"
                ));
            };
            post_weights[post_idx] = 1.0;
        }
        HonestPostFunctional::AverageWindow {
            start_period,
            end_period,
        } => {
            if start_period > end_period {
                return Err(format!(
                    "average window start_period {start_period} exceeds end_period {end_period}"
                ));
            }
            let covered: Vec<usize> = input
                .post_periods
                .iter()
                .enumerate()
                .filter_map(|(idx, period)| {
                    (*period >= *start_period && *period <= *end_period).then_some(idx)
                })
                .collect();
            if covered.is_empty() {
                return Err(format!(
                    "average window [{start_period}, {end_period}] covers no post periods"
                ));
            }
            let weight = 1.0 / usize_to_f64(covered.len());
            for idx in covered {
                post_weights[idx] = weight;
            }
        }
        HonestPostFunctional::Weighted(weights) => {
            if weights.is_empty() {
                return Err(
                    "weighted functional requires at least one post-period weight".to_string(),
                );
            }
            for (post_period, weight) in weights {
                let Some(post_idx) = input
                    .post_periods
                    .iter()
                    .position(|period| period == post_period)
                else {
                    return Err(format!(
                        "post period {post_period} is not present in HonestEventStudyInput"
                    ));
                };
                post_weights[post_idx] += *weight;
            }
        }
    }
    Ok(post_weights)
}

/// [`assess_honest_event_study_functional`] for a functional given by period
/// labels instead of weights, such as one period, an average over a window of
/// periods or a weighted sum. See [`HonestPostFunctional`] for how the labels
/// become weights.
///
/// # Errors
/// Returns an error if the requested functional cannot be represented on the
/// supplied post-period support or if the underlying `HonestDiD` calculation
/// fails.
pub fn assess_honest_event_study_post_functional(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
) -> Result<HonestAssessment, String> {
    assess_honest_event_study_post_functional_counted(
        input,
        functional,
        sensitivity,
        inference,
        null_value,
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::inference::sensitivity) fn assess_honest_event_study_post_functional_counted(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<HonestAssessment, String> {
    let post_weights = post_weights_for_functional(input, functional)?;
    assess_honest_event_study_functional_counted(
        input,
        &post_weights,
        sensitivity,
        inference,
        null_value,
        counter,
    )
}

/// [`assess_honest_event_study_post_functional`] with the `DeltaRM` conditional
/// confidence-set settings given explicitly. A smoothness restriction ignores
/// them.
///
/// # Errors
/// Returns an error if the requested functional cannot be represented on the
/// supplied post-period support or if the underlying `HonestDiD` calculation
/// fails.
pub fn assess_honest_event_study_post_functional_with_config(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<HonestAssessment, String> {
    assess_honest_event_study_post_functional_with_config_counted(
        input,
        functional,
        sensitivity,
        inference,
        null_value,
        relative_magnitude_config,
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::inference::sensitivity) fn assess_honest_event_study_post_functional_with_config_counted(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<HonestAssessment, String> {
    let post_weights = post_weights_for_functional(input, functional)?;
    assess_honest_event_study_functional_with_config_counted(
        input,
        &post_weights,
        sensitivity,
        inference,
        null_value,
        relative_magnitude_config,
        counter,
    )
}

fn assess_relative_magnitude_functional_with_prepared_functional_branches(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    prepared_branches: &[RelativeMagnitudePreparedBranch],
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
    prepared_functional_branches: &[RelativeMagnitudePreparedFunctionalBranch],
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<HonestAssessment, String> {
    let original = compute_original_confidence_set(input, post_weights, inference)?;
    let branch_identified_sets =
        compute_relative_magnitude_branch_identified_sets_with_prepared_branches(
            input,
            post_weights,
            prepared_branches,
            counter,
        )?;
    let identified = global_identified_set_from_branch_identified_sets(&branch_identified_sets);
    let conditional =
        compute_relative_magnitude_confidence_set_with_prepared_functional_branches_and_identified_sets(
        input,
        post_weights,
        inference,
        relative_magnitude_config,
        &original,
        &identified,
        prepared_branches,
        prepared_input_branches,
        prepared_functional_branches,
        &branch_identified_sets, counter,
    )?;
    let robust_ci = (conditional.lb, conditional.ub);
    Ok(HonestAssessment {
        original_ci: original.ci,
        robust_ci,
        identified_set: (identified.lb, identified.ub),
        estimate: original.estimate,
        se: original.se,
        null_value,
        sensitivity_value: mbar,
        robustly_significant: robust_ci.0 > null_value || robust_ci.1 < null_value,
    })
}

fn assess_relative_magnitude_basis_period_with_prepared_input(
    input: &HonestEventStudyInput,
    post_idx: usize,
    mbar: f64,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    prepared_branches: &[RelativeMagnitudePreparedBranch],
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<HonestAssessment, String> {
    let post_weights = basis_post_weights(input.num_post_periods(), post_idx);
    let prepared_functional_branches = prepare_relative_magnitude_functional_branches(
        &post_weights,
        prepared_branches,
        prepared_input_branches,
    )?;
    assess_relative_magnitude_basis_period_with_prepared_functional_branches(
        input,
        &post_weights,
        mbar,
        inference,
        null_value,
        relative_magnitude_config,
        prepared_branches,
        prepared_input_branches,
        &prepared_functional_branches,
        counter,
    )
}

pub(in crate::inference::sensitivity) fn assess_relative_magnitude_basis_period_with_prepared_functional_branches(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    prepared_branches: &[RelativeMagnitudePreparedBranch],
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
    prepared_functional_branches: &[RelativeMagnitudePreparedFunctionalBranch],
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<HonestAssessment, String> {
    let original = compute_original_confidence_set(input, post_weights, inference)?;
    let branch_identified_sets =
        compute_relative_magnitude_branch_identified_sets_with_prepared_branches(
            input,
            post_weights,
            prepared_branches,
            counter,
        )?;
    let identified = global_identified_set_from_branch_identified_sets(&branch_identified_sets);
    let conditional =
        compute_relative_magnitude_confidence_set_with_prepared_functional_branches_and_identified_sets(
        input,
        post_weights,
        inference,
        relative_magnitude_config,
        &original,
        &identified,
        prepared_branches,
        prepared_input_branches,
        prepared_functional_branches,
        &branch_identified_sets, counter,
    )?;
    let robust_ci = (conditional.lb, conditional.ub);
    Ok(HonestAssessment {
        original_ci: original.ci,
        robust_ci,
        identified_set: (identified.lb, identified.ub),
        estimate: original.estimate,
        se: original.se,
        null_value,
        sensitivity_value: mbar,
        robustly_significant: robust_ci.0 > null_value || robust_ci.1 < null_value,
    })
}

struct PreparedRelativeMagnitudeBaseContext {
    mbar: f64,
    branches: Vec<RelativeMagnitudePreparedBranch>,
    input_branches: Vec<RelativeMagnitudePreparedInputBranch>,
}

fn prepare_relative_magnitude_base_context(
    input: &HonestEventStudyInput,
    mbar: f64,
) -> Result<PreparedRelativeMagnitudeBaseContext, String> {
    let branches = prepare_relative_magnitude_branches(
        input.num_pre_periods(),
        input.num_post_periods(),
        mbar,
    )?;
    let input_branches = prepare_relative_magnitude_input_branches(input, &branches);
    Ok(PreparedRelativeMagnitudeBaseContext {
        mbar,
        branches,
        input_branches,
    })
}

/// Least-favorable interval of a post-treatment functional under
/// `ΔRM(Mbar)`, returned as an [`HonestAssessment`] whose `robust_ci` is that
/// interval.
///
/// The hybrid settings are those of
/// [`RelativeMagnitudeConfidenceSetConfig::from_inference`].
///
/// # Errors
/// Returns an error if the functional is invalid for the supplied event-study
/// input or if the underlying `DeltaRM` least-favorable interval path fails.
pub fn assess_honest_event_study_post_functional_flci(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
    mbar: f64,
    inference: InferenceConfig,
    null_value: f64,
) -> Result<HonestAssessment, String> {
    assess_honest_event_study_post_functional_flci_counted(
        input,
        functional,
        mbar,
        inference,
        null_value,
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::inference::sensitivity) fn assess_honest_event_study_post_functional_flci_counted(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
    mbar: f64,
    inference: InferenceConfig,
    null_value: f64,
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<HonestAssessment, String> {
    assess_honest_event_study_post_functional_flci_with_config_counted(
        input,
        functional,
        mbar,
        inference,
        null_value,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        counter,
    )
}

/// [`assess_honest_event_study_post_functional_flci`] with the hybrid settings
/// given explicitly.
///
/// # Errors
/// Returns an error if the functional is invalid for the supplied event-study
/// input or if the underlying `DeltaRM` least-favorable interval path fails.
pub fn assess_honest_event_study_post_functional_flci_with_config(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
    mbar: f64,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<HonestAssessment, String> {
    assess_honest_event_study_post_functional_flci_with_config_counted(
        input,
        functional,
        mbar,
        inference,
        null_value,
        relative_magnitude_config,
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::inference::sensitivity) fn assess_honest_event_study_post_functional_flci_with_config_counted(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
    mbar: f64,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<HonestAssessment, String> {
    let post_weights = post_weights_for_functional(input, functional)?;
    let problem = crate::inference::sensitivity::relative_magnitude::least_favorable_intervals::build_relative_magnitude_flci_problem_counted(input, &post_weights, mbar, inference, counter)?;
    let flci = crate::inference::sensitivity::relative_magnitude::least_favorable_intervals::compute_relative_magnitude_flci_with_config_counted(&problem, relative_magnitude_config, counter)?;
    Ok(HonestAssessment {
        original_ci: flci.original_ci,
        robust_ci: flci.flci,
        identified_set: flci.identified_set,
        estimate: problem.original.estimate,
        se: problem.original.se,
        null_value,
        sensitivity_value: mbar,
        robustly_significant: flci.flci.0 > null_value || flci.flci.1 < null_value,
    })
}

/// Least-favorable intervals under `ΔRM(Mbar)` for several post-treatment
/// functionals at once, each run at a pointwise level calibrated so that the
/// intervals cover jointly at `inference.confidence_level`.
///
/// The hybrid settings are those of
/// [`RelativeMagnitudeConfidenceSetConfig::from_inference`] and the joint
/// calibration is [`HonestJointPathConfig::default_for_production`]. There must
/// be at least one functional.
///
/// # Errors
/// Returns an error if any functional is invalid for the supplied input or if
/// the underlying joint multi-functional FLCI solve fails.
pub fn assess_honest_event_study_post_functional_multi_flci(
    input: &HonestEventStudyInput,
    functionals: &[HonestPostFunctional],
    mbar: f64,
    inference: InferenceConfig,
) -> Result<HonestMultiFlciResult, String> {
    assess_honest_event_study_post_functional_multi_flci_counted(
        input,
        functionals,
        mbar,
        inference,
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::inference::sensitivity) fn assess_honest_event_study_post_functional_multi_flci_counted(
    input: &HonestEventStudyInput,
    functionals: &[HonestPostFunctional],
    mbar: f64,
    inference: InferenceConfig,
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<HonestMultiFlciResult, String> {
    assess_honest_event_study_post_functional_multi_flci_with_config_counted(
        input,
        functionals,
        mbar,
        inference,
        RelativeMagnitudeConfidenceSetConfig::from_inference(inference),
        HonestJointPathConfig::default_for_production(),
        counter,
    )
}

/// [`assess_honest_event_study_post_functional_multi_flci`] with the hybrid
/// settings and the joint calibration given explicitly.
///
/// # Errors
/// Returns an error if any functional is invalid for the supplied input or if
/// the underlying joint multi-functional FLCI solve fails.
pub fn assess_honest_event_study_post_functional_multi_flci_with_config(
    input: &HonestEventStudyInput,
    functionals: &[HonestPostFunctional],
    mbar: f64,
    inference: InferenceConfig,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
) -> Result<HonestMultiFlciResult, String> {
    assess_honest_event_study_post_functional_multi_flci_with_config_counted(
        input,
        functionals,
        mbar,
        inference,
        relative_magnitude_config,
        joint_config,
        &crate::inference::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::inference::sensitivity) fn assess_honest_event_study_post_functional_multi_flci_with_config_counted(
    input: &HonestEventStudyInput,
    functionals: &[HonestPostFunctional],
    mbar: f64,
    inference: InferenceConfig,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
    counter: &crate::inference::sensitivity::SolverCounter,
) -> Result<HonestMultiFlciResult, String> {
    if functionals.is_empty() {
        return Err("multi-functional FLCI requires at least one functional".to_string());
    }
    let post_weight_sets = functionals
        .iter()
        .map(|functional| post_weights_for_functional(input, functional))
        .collect::<Result<Vec<_>, String>>()?;
    let problem =
        crate::inference::sensitivity::relative_magnitude::least_favorable_intervals::build_relative_magnitude_multi_flci_problem_counted(input, &post_weight_sets, mbar, inference, counter)?;
    crate::inference::sensitivity::relative_magnitude::least_favorable_intervals::compute_relative_magnitude_multi_flci_with_config_counted(
        &problem,
        relative_magnitude_config,
        joint_config, counter,
    )
}
