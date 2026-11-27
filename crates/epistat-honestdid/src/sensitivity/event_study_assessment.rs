#[path = "event_study_regions.rs"]
mod regions;
#[cfg(test)]
#[path = "event_study_workflow.rs"]
mod workflow;

use crate::util::InferenceConfig;
#[cfg(test)]
use rayon::prelude::*;

use super::relative_magnitude::conditional_confidence_set::{
    RelativeMagnitudePreparedFunctionalBranch, RelativeMagnitudePreparedInputBranch,
    compute_relative_magnitude_branch_identified_sets_with_prepared_branches,
    compute_relative_magnitude_confidence_set_with_prepared_functional_branches_and_identified_sets,
    global_identified_set_from_branch_identified_sets,
    prepare_relative_magnitude_functional_branches, prepare_relative_magnitude_input_branches,
};
#[cfg(test)]
use super::relative_magnitude::conditional_confidence_set::{
    compute_relative_magnitude_confidence_set_for_basis_period_with_prepared_input,
    compute_relative_magnitude_identified_set_for_basis_period,
};
use super::relative_magnitude::geometry::{
    RelativeMagnitudePreparedBranch, basis_post_weights, prepare_relative_magnitude_branches,
};
use super::{
    HonestAssessment, HonestEventStudyInput, HonestPostFunctional, HonestSensitivity,
    LeastFavorableSimulation, RelativeMagnitudeConfidenceSetConfig, SmoothnessConfidenceSetConfig,
    relative_magnitude::{
        compute_original_confidence_set,
        compute_relative_magnitude_confidence_set_with_precomputed_sets,
    },
};
#[cfg(test)]
use super::{HonestEventStudyPoint, HonestJointPathConfig, HonestMultiFlciResult, HonestResult};
use crate::util::usize_to_f64;

#[cfg(test)]
pub use regions::assess_honest_event_study_joint_path_region;
pub use regions::{
    assess_honest_event_study_directional_region_with_config,
    assess_honest_event_study_joint_path_region_with_config,
    assess_honest_event_study_optimization_surface_region_adaptive_with_config,
    assess_honest_event_study_optimization_surface_region_with_config,
};
#[cfg(test)]
pub use workflow::{
    assess_honest_event_study_workflow, assess_honest_event_study_workflow_with_config,
};

/// Compute a scalar `HonestDiD` result for each post-treatment coefficient of
/// `input`, each period on its own, in the order of its post periods.
///
/// The robust interval is the conditional confidence set with the default
/// configuration for the restriction, and the post labels are returned in the
/// results.
///
/// # Errors
/// Returns an error if `input` is inconsistent or its covariance entries are
/// invalid.
#[cfg(test)]
pub fn compute_honest_event_study_scalar_for_input(
    input: &HonestEventStudyInput,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
) -> Result<Vec<HonestEventStudyPoint>, String> {
    compute_honest_event_study_scalar_for_input_counted(
        input,
        sensitivity,
        inference,
        &crate::sensitivity::SolverCounter::new(),
    )
}

#[cfg(test)]
pub(in crate::sensitivity) fn compute_honest_event_study_scalar_for_input_counted(
    input: &HonestEventStudyInput,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    counter: &crate::sensitivity::SolverCounter,
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
                    let identified = crate::sensitivity::smoothness::compute_smoothness_identified_set_counted(input, &post_weights, m, counter)?;
                    let conditional = crate::sensitivity::smoothness::compute_smoothness_confidence_set_with_config_counted(
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
#[cfg(test)]
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
        &crate::sensitivity::SolverCounter::new(),
    )
}

#[cfg(test)]
pub(in crate::sensitivity) fn assess_honest_event_study_functional_counted(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    counter: &crate::sensitivity::SolverCounter,
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
#[cfg(test)]
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
        &crate::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::sensitivity) fn assess_honest_event_study_functional_with_config_counted(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    counter: &crate::sensitivity::SolverCounter,
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
            let identified =
                crate::sensitivity::smoothness::compute_smoothness_identified_set_counted(
                    input,
                    post_weights,
                    m,
                    counter,
                )?;
            let conditional = crate::sensitivity::smoothness::compute_smoothness_confidence_set_with_config_counted(
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
#[cfg(test)]
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
        &crate::sensitivity::SolverCounter::new(),
    )
}

#[cfg(test)]
pub(in crate::sensitivity) fn assess_honest_event_study_period_counted(
    input: &HonestEventStudyInput,
    post_period: i32,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    counter: &crate::sensitivity::SolverCounter,
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
#[cfg(test)]
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
        &crate::sensitivity::SolverCounter::new(),
    )
}

#[cfg(test)]
pub(in crate::sensitivity) fn assess_honest_event_study_post_functional_counted(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    counter: &crate::sensitivity::SolverCounter,
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
#[cfg(test)]
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
        &crate::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::sensitivity) fn assess_honest_event_study_post_functional_with_config_counted(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    counter: &crate::sensitivity::SolverCounter,
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
    counter: &crate::sensitivity::SolverCounter,
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
    counter: &crate::sensitivity::SolverCounter,
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

pub(in crate::sensitivity) fn assess_relative_magnitude_basis_period_with_prepared_functional_branches(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    prepared_branches: &[RelativeMagnitudePreparedBranch],
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
    prepared_functional_branches: &[RelativeMagnitudePreparedFunctionalBranch],
    counter: &crate::sensitivity::SolverCounter,
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
#[cfg(test)]
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
        &crate::sensitivity::SolverCounter::new(),
    )
}

#[cfg(test)]
pub(in crate::sensitivity) fn assess_honest_event_study_post_functional_flci_counted(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
    mbar: f64,
    inference: InferenceConfig,
    null_value: f64,
    counter: &crate::sensitivity::SolverCounter,
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
#[cfg(test)]
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
        &crate::sensitivity::SolverCounter::new(),
    )
}

#[cfg(test)]
pub(in crate::sensitivity) fn assess_honest_event_study_post_functional_flci_with_config_counted(
    input: &HonestEventStudyInput,
    functional: &HonestPostFunctional,
    mbar: f64,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestAssessment, String> {
    let post_weights = post_weights_for_functional(input, functional)?;
    let problem = crate::sensitivity::relative_magnitude::least_favorable_intervals::build_relative_magnitude_flci_problem_counted(input, &post_weights, mbar, inference, counter)?;
    let flci = crate::sensitivity::relative_magnitude::least_favorable_intervals::compute_relative_magnitude_flci_with_config_counted(&problem, relative_magnitude_config, counter)?;
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

/// [`assess_honest_event_study_post_functional_multi_flci`] with the hybrid
/// settings and the joint calibration given explicitly.
///
/// # Errors
/// Returns an error if any functional is invalid for the supplied input or if
/// the underlying joint multi-functional FLCI solve fails.
#[cfg(test)]
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
        &crate::sensitivity::SolverCounter::new(),
    )
}

#[cfg(test)]
pub(in crate::sensitivity) fn assess_honest_event_study_post_functional_multi_flci_with_config_counted(
    input: &HonestEventStudyInput,
    functionals: &[HonestPostFunctional],
    mbar: f64,
    inference: InferenceConfig,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestMultiFlciResult, String> {
    if functionals.is_empty() {
        return Err("multi-functional FLCI requires at least one functional".to_string());
    }
    let post_weight_sets = functionals
        .iter()
        .map(|functional| post_weights_for_functional(input, functional))
        .collect::<Result<Vec<_>, String>>()?;
    let problem =
        crate::sensitivity::relative_magnitude::least_favorable_intervals::build_relative_magnitude_multi_flci_problem_counted(input, &post_weight_sets, mbar, inference, counter)?;
    crate::sensitivity::relative_magnitude::least_favorable_intervals::compute_relative_magnitude_multi_flci_with_config_counted(
        &problem,
        relative_magnitude_config,
        joint_config, counter,
    )
}
