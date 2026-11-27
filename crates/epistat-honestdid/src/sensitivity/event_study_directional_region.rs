use super::super::super::linear_algebra::{
    critical_value_from_pointwise_confidence, functional_covariance_matrix,
    pointwise_confidence_level_from_critical, post_covariance_block, sup_t_critical_value,
};
use super::super::super::relative_magnitude::geometry::basis_vector_index;
use super::super::super::{
    HonestDirectionalFunctionalPoint, HonestDirectionalRegion, HonestDirectionalRegionDiagnostics,
    HonestEventStudyInput, HonestJointPathConfig, HonestJointPathMethod, HonestSensitivity,
    RelativeMagnitudeConfidenceSetConfig,
};
use crate::sensitivity::EngineError;
use crate::util::InferenceConfig;
use crate::util::usize_to_f64;
use crate::util::validate_confidence_level;
use rayon::prelude::*;

fn validate_direction(direction: &[f64], num_post: usize) -> Result<(), EngineError> {
    if direction.len() != num_post {
        return Err(EngineError::length("direction", num_post, direction.len()));
    }
    EngineError::check_finite("direction", direction)?;
    let l1: f64 = direction.iter().map(|value| value.abs()).sum();
    if l1 <= 1e-12 {
        return Err(EngineError::all_zero("direction"));
    }
    Ok(())
}

fn simulated_directional_pointwise_confidence_level(
    input: &HonestEventStudyInput,
    confidence_level: f64,
    simulation_draws: usize,
    simulation_seed: u64,
    directions: &[Vec<f64>],
) -> Result<f64, EngineError> {
    if simulation_draws == 0 {
        return Err(EngineError::domain("calibration draws", 0.0, "[1, inf)"));
    }
    let sigma_post = post_covariance_block(
        &input.covariance,
        input.num_pre_periods(),
        input.num_post_periods(),
    );
    let covariance = functional_covariance_matrix(
        &sigma_post,
        directions,
        "one or more direction vectors have near-zero variance under the post covariance",
    )?;
    sup_t_critical_value(
        &covariance,
        confidence_level,
        simulation_draws,
        simulation_seed,
    )
}

fn validate_direction_set(
    input: &HonestEventStudyInput,
    directions: &[Vec<f64>],
) -> Result<(), EngineError> {
    input.validate()?;
    if directions.is_empty() {
        return Err(EngineError::empty("directions"));
    }
    let num_post = input.num_post_periods();
    for direction in directions {
        validate_direction(direction, num_post)?;
    }
    Ok(())
}

pub(super) fn calibrate_directional_region(
    input: &HonestEventStudyInput,
    directions: &[Vec<f64>],
    inference: InferenceConfig,
    joint_config: HonestJointPathConfig,
) -> Result<(f64, f64), EngineError> {
    validate_direction_set(input, directions)?;
    let alpha = 1.0 - inference.confidence_level;
    match joint_config.method {
        HonestJointPathMethod::Bonferroni => {
            let pointwise = 1.0 - alpha / usize_to_f64(directions.len());
            if !validate_confidence_level(pointwise) {
                return Err(EngineError::domain(
                    "pointwise confidence level",
                    pointwise,
                    "(0, 1)",
                ));
            }
            Ok((
                pointwise,
                critical_value_from_pointwise_confidence(pointwise),
            ))
        }
        HonestJointPathMethod::GaussianSimulated => {
            let critical = simulated_directional_pointwise_confidence_level(
                input,
                inference.confidence_level,
                joint_config.simulation_draws,
                joint_config.simulation_seed,
                directions,
            )?;
            Ok((pointwise_confidence_level_from_critical(critical), critical))
        }
    }
}

/// Simultaneous `HonestDiD` region over a finite set of post-treatment linear
/// functionals.
///
/// Each direction in `directions` is a vector of weights with one entry per
/// post period, finite and not all zero. Each functional's interval is run at a
/// pointwise level that `joint_config` calibrates so that the intervals cover
/// jointly at `inference.confidence_level`. The relative-magnitude settings are
/// ignored under a smoothness restriction. `null_value` is the value that each
/// point's `robustly_significant` is tested against.
///
/// # Errors
/// Returns an error if any direction is malformed, if calibration fails, or if
/// any directional scalar assessment fails.
pub(crate) fn assess_honest_event_study_directional_region_with_config(
    input: &HonestEventStudyInput,
    directions: &[Vec<f64>],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
) -> Result<HonestDirectionalRegion, EngineError> {
    assess_honest_event_study_directional_region_with_config_counted(
        input,
        directions,
        sensitivity,
        inference,
        null_value,
        relative_magnitude_config,
        joint_config,
        &crate::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::sensitivity) fn assess_honest_event_study_directional_region_with_config_counted(
    input: &HonestEventStudyInput,
    directions: &[Vec<f64>],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestDirectionalRegion, EngineError> {
    assess_honest_event_study_directional_region_with_optional_prepared_relative_magnitude(
        input,
        directions,
        sensitivity,
        inference,
        null_value,
        relative_magnitude_config,
        joint_config,
        None,
        counter,
    )
}

pub(in crate::sensitivity) fn assess_honest_event_study_directional_region_with_optional_prepared_relative_magnitude(
    input: &HonestEventStudyInput,
    directions: &[Vec<f64>],
    sensitivity: HonestSensitivity,
    inference: InferenceConfig,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
    prepared_relative_magnitude: Option<&super::super::PreparedRelativeMagnitudeBaseContext>,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestDirectionalRegion, EngineError> {
    let (pointwise_confidence_level, calibrated_max_t_critical_value) =
        calibrate_directional_region(input, directions, inference, joint_config)?;
    assess_honest_event_study_directional_region_with_precalibrated_pointwise(
        input,
        directions,
        sensitivity,
        null_value,
        relative_magnitude_config,
        prepared_relative_magnitude,
        joint_config.method,
        inference.confidence_level,
        pointwise_confidence_level,
        calibrated_max_t_critical_value,
        counter,
    )
}

pub(in crate::sensitivity) fn assess_honest_event_study_directional_region_with_precalibrated_pointwise(
    input: &HonestEventStudyInput,
    directions: &[Vec<f64>],
    sensitivity: HonestSensitivity,
    null_value: f64,
    relative_magnitude_config: RelativeMagnitudeConfidenceSetConfig,
    prepared_relative_magnitude: Option<&super::super::PreparedRelativeMagnitudeBaseContext>,
    method: HonestJointPathMethod,
    confidence_level: f64,
    pointwise_confidence_level: f64,
    calibrated_max_t_critical_value: f64,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<HonestDirectionalRegion, EngineError> {
    let pointwise_inference = InferenceConfig::new(pointwise_confidence_level);
    let pointwise_relative_magnitude_config = RelativeMagnitudeConfidenceSetConfig::from_inference(
        pointwise_inference,
        relative_magnitude_config.simulation(),
    );
    let owned_relative_magnitude = match (sensitivity, prepared_relative_magnitude) {
        (HonestSensitivity::RelativeMagnitude(mbar), None) => Some(
            super::super::prepare_relative_magnitude_base_context(input, mbar)?,
        ),
        _ => None,
    };
    let prepared_relative_magnitude =
        prepared_relative_magnitude.or(owned_relative_magnitude.as_ref());
    let basis_direction_indices = directions
        .iter()
        .map(|direction| basis_vector_index(direction))
        .collect::<Option<Vec<_>>>();
    let prepared_functional_branch_sets =
        match (prepared_relative_magnitude, &basis_direction_indices) {
            (Some(prepared), None) => Some(
                directions
                    .par_iter()
                    .map(|direction| {
                        {
                            super::super::prepare_relative_magnitude_functional_branches(
                                direction,
                                &prepared.branches,
                                &prepared.input_branches,
                            )
                        }
                    })
                    .collect::<Result<Vec<_>, EngineError>>()?,
            ),
            _ => None,
        };

    let points = directions
        .par_iter()
        .enumerate()
        .map(|(direction_index, direction)| {
            let assessment = match prepared_relative_magnitude {
                Some(prepared) => match &basis_direction_indices {
                    Some(basis_indices) => {
                        super::super::assess_relative_magnitude_basis_period_with_prepared_input(
                            input,
                            basis_indices[direction_index],
                            prepared.mbar,
                            pointwise_inference,
                            null_value,
                            RelativeMagnitudeConfidenceSetConfig {
                                hybrid: relative_magnitude_config.hybrid,
                                hybrid_kappa: pointwise_relative_magnitude_config.hybrid_kappa,
 least_favorable_draws: relative_magnitude_config.least_favorable_draws,
                            least_favorable_seed: relative_magnitude_config.least_favorable_seed,
                            },
                            &prepared.branches,
                            &prepared.input_branches, counter,
                        )?
                    }
                    None => super::super::assess_relative_magnitude_functional_with_prepared_functional_branches(
                        input,
                        direction,
                        prepared.mbar,
                        pointwise_inference,
                        null_value,
                        RelativeMagnitudeConfidenceSetConfig {
                            hybrid: relative_magnitude_config.hybrid,
                            hybrid_kappa: pointwise_relative_magnitude_config.hybrid_kappa,
 least_favorable_draws: relative_magnitude_config.least_favorable_draws,
                            least_favorable_seed: relative_magnitude_config.least_favorable_seed,
                        },
                        &prepared.branches,
                        &prepared.input_branches,
                        &prepared_functional_branch_sets
                            .as_ref()
                            .ok_or(EngineError::Internal("missing prepared functional branch sets"))?[direction_index], counter,
                    )?,
                },
                None => super::super::assess_honest_event_study_functional_with_config_counted(
                    input,
                    direction,
                    sensitivity,
                    pointwise_inference,
                    null_value,
                    relative_magnitude_config, counter,
                )?,
            };
            Ok(HonestDirectionalFunctionalPoint {
                post_weights: direction.clone(),
                assessment,
            })
        })
        .collect::<Result<Vec<_>, EngineError>>()?;

    Ok(HonestDirectionalRegion {
        confidence_level,
        pointwise_confidence_level,
        calibrated_max_t_critical_value,
        method,
        points,
        diagnostics: HonestDirectionalRegionDiagnostics::fixed(directions.len(), 0)
            .with_solver_counts(counter.counts()),
    })
}
