//! Least-favorable intervals under `DeltaRM` for post-treatment linear
//! functionals, as typed problems and results.
//!
//! A problem bundles the event-study input with, for each functional `l' τ_post`:
//!
//! - the original parallel-trends confidence set
//! - the exact `DeltaRM(Mbar)` identified set
//!
//! and the interval is the `DeltaRM` conditional confidence set of the same
//! functional with a least-favorable (or ARP-only) first stage. It is computed
//! functional by functional, and does not use the optimization over the
//! multi-period surface that the R `HonestDiD` package uses for its FLCI.
//!
//! References:
//! - Rambachan, A. and Roth, J. (2023), "A More Credible Approach to Parallel
//!   Trends", especially the least-favorable / FLCI discussion
//! - `HonestDiD` R package `flci.R`

use crate::util::InferenceConfig;
use rayon::prelude::*;

#[cfg(test)]
use super::super::HonestConditionalConfidenceSet;
use super::super::linear_algebra::{
    critical_value_from_pointwise_confidence, functional_covariance_matrix,
    pointwise_confidence_level_from_critical, post_covariance_block, sup_t_critical_value,
};
use super::super::{
    HonestEventStudyInput, HonestIdentifiedSet, HonestJointPathConfig, HonestJointPathMethod,
    HonestOriginalConfidenceSet, RelativeMagnitudeConfidenceSetConfig,
    RelativeMagnitudeMultiFlciPoint, RelativeMagnitudeMultiFlciResult,
};
use super::compute_original_confidence_set;
use super::conditional_confidence_set::{
    RelativeMagnitudePreparedFunctionalBranch, RelativeMagnitudePreparedInputBranch,
    compute_relative_magnitude_confidence_set_with_prepared_functional_branches,
    prepare_relative_magnitude_functional_branches, prepare_relative_magnitude_input_branches,
};
use super::geometry::{RelativeMagnitudePreparedBranch, prepare_relative_magnitude_branches};
use crate::util::usize_to_f64;
use crate::util::validate_confidence_level;

/// Typed `DeltaRM` FLCI problem for a post-treatment linear functional.
///
/// The target is the scalar functional
///
/// ```text
/// θ = l' τ_post
/// ```
///
/// under the relative-magnitude restriction `ΔRM(Mbar)`.
#[derive(Debug, Clone, PartialEq)]
#[cfg(test)]
pub struct RelativeMagnitudeFlciProblem {
    /// Event-study coefficients and covariance matrix.
    pub input: HonestEventStudyInput,
    /// Weights of the functional on the post-treatment coefficients, one per
    /// post period.
    pub post_weights: Vec<f64>,
    /// Relative-magnitude bound `Mbar`, finite and non-negative.
    pub mbar: f64,
    /// Confidence level of the original confidence set and of the interval.
    pub inference: InferenceConfig,
    /// Original parallel-trends confidence set for `l' τ_post`.
    pub original: HonestOriginalConfidenceSet,
    /// Exact `DeltaRM(Mbar)` identified set for `l' τ_post`.
    pub identified_set: HonestIdentifiedSet,
}

/// Least-favorable `DeltaRM` interval result for a post-treatment linear
/// functional.
///
/// The `flci` field is the `DeltaRM` conditional confidence interval for
/// `l' τ_post`, given with the original interval and the identified set.
#[derive(Debug, Clone, PartialEq)]
#[cfg(test)]
pub struct RelativeMagnitudeFlciResult {
    /// Conditional confidence interval `(lower, upper)` under `DeltaRM(Mbar)`.
    pub flci: (f64, f64),
    /// Original parallel-trends confidence interval `(lower, upper)`.
    pub original_ci: (f64, f64),
    /// Exact `DeltaRM(Mbar)` identified set `(lower, upper)`.
    pub identified_set: (f64, f64),
}

/// Typed `DeltaRM` simultaneous FLCI problem over multiple post-treatment
/// linear functionals.
#[derive(Debug, Clone, PartialEq)]
pub struct RelativeMagnitudeMultiFlciProblem {
    /// Event-study coefficients and covariance matrix.
    pub input: HonestEventStudyInput,
    /// One weight vector per functional, each with one weight per post period.
    pub post_weight_sets: Vec<Vec<f64>>,
    /// Relative-magnitude bound `Mbar`, finite and non-negative.
    pub mbar: f64,
    /// Requested joint confidence level of the intervals.
    pub inference: InferenceConfig,
    /// Original parallel-trends confidence set of each functional, in the order
    /// of `post_weight_sets`.
    pub originals: Vec<HonestOriginalConfidenceSet>,
    /// Exact identified set of each functional, in the order of
    /// `post_weight_sets`.
    pub identified_sets: Vec<HonestIdentifiedSet>,
}

/// Build a typed `DeltaRM` FLCI problem for a post-treatment linear
/// functional.
///
/// The problem holds the original confidence set and the identified set of the
/// functional.
///
/// # Errors
/// Returns an error if the event-study input is inconsistent, `post_weights` is
/// not one finite weight per post period with at least one non-zero, or the
/// identified set cannot be computed.
#[cfg(test)]
pub fn build_relative_magnitude_flci_problem(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
) -> Result<RelativeMagnitudeFlciProblem, String> {
    build_relative_magnitude_flci_problem_counted(
        input,
        post_weights,
        mbar,
        inference,
        &crate::sensitivity::SolverCounter::new(),
    )
}

#[cfg(test)]
pub(in crate::sensitivity) fn build_relative_magnitude_flci_problem_counted(
    input: &HonestEventStudyInput,
    post_weights: &[f64],
    mbar: f64,
    inference: InferenceConfig,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<RelativeMagnitudeFlciProblem, String> {
    let original = compute_original_confidence_set(input, post_weights, inference)?;
    let identified_set =
        crate::sensitivity::relative_magnitude::compute_relative_magnitude_identified_set_counted(
            input,
            post_weights,
            mbar,
            counter,
        )?;
    Ok(RelativeMagnitudeFlciProblem {
        input: input.clone(),
        post_weights: post_weights.to_vec(),
        mbar,
        inference,
        original,
        identified_set,
    })
}

/// Conditional confidence interval of the problem's functional, with the
/// settings of [`RelativeMagnitudeConfidenceSetConfig::from_inference`] at
/// `problem.inference`.
///
/// # Errors
/// Returns an error if the conditional confidence set fails, for example when no
/// grid point is accepted.
#[cfg(test)]
pub fn compute_relative_magnitude_flci(
    problem: &RelativeMagnitudeFlciProblem,
) -> Result<RelativeMagnitudeFlciResult, String> {
    compute_relative_magnitude_flci_counted(problem, &crate::sensitivity::SolverCounter::new())
}

#[cfg(test)]
pub(in crate::sensitivity) fn compute_relative_magnitude_flci_counted(
    problem: &RelativeMagnitudeFlciProblem,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<RelativeMagnitudeFlciResult, String> {
    compute_relative_magnitude_flci_with_config_counted(
        problem,
        RelativeMagnitudeConfidenceSetConfig::from_inference(problem.inference),
        counter,
    )
}

/// [`compute_relative_magnitude_flci`] with the hybrid settings given in
/// `config`. The interval is that of
/// [`super::compute_relative_magnitude_confidence_set_with_config`], returned
/// with the
/// original interval and the identified set.
///
/// # Errors
/// Returns an error if `config` fails validation or the conditional confidence
/// set fails.
#[cfg(test)]
pub fn compute_relative_magnitude_flci_with_config(
    problem: &RelativeMagnitudeFlciProblem,
    config: RelativeMagnitudeConfidenceSetConfig,
) -> Result<RelativeMagnitudeFlciResult, String> {
    compute_relative_magnitude_flci_with_config_counted(
        problem,
        config,
        &crate::sensitivity::SolverCounter::new(),
    )
}

#[cfg(test)]
pub(in crate::sensitivity) fn compute_relative_magnitude_flci_with_config_counted(
    problem: &RelativeMagnitudeFlciProblem,
    config: RelativeMagnitudeConfidenceSetConfig,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<RelativeMagnitudeFlciResult, String> {
    let conditional: HonestConditionalConfidenceSet =
        crate::sensitivity::relative_magnitude::compute_relative_magnitude_confidence_set_with_config_counted(
            &problem.input,
            &problem.post_weights,
            problem.mbar,
            problem.inference,
            config, counter,
        )?;
    Ok(RelativeMagnitudeFlciResult {
        flci: (conditional.lb, conditional.ub),
        original_ci: problem.original.ci,
        identified_set: (problem.identified_set.lb, problem.identified_set.ub),
    })
}

/// Build a simultaneous `DeltaRM` problem over several functionals, computing
/// the original confidence set and the identified set of each.
///
/// # Errors
/// Returns an error if `post_weight_sets` is empty, any functional is invalid,
/// or an identified set cannot be computed.
pub fn build_relative_magnitude_multi_flci_problem(
    input: &HonestEventStudyInput,
    post_weight_sets: &[Vec<f64>],
    mbar: f64,
    inference: InferenceConfig,
) -> Result<RelativeMagnitudeMultiFlciProblem, String> {
    build_relative_magnitude_multi_flci_problem_counted(
        input,
        post_weight_sets,
        mbar,
        inference,
        &crate::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::sensitivity) fn build_relative_magnitude_multi_flci_problem_counted(
    input: &HonestEventStudyInput,
    post_weight_sets: &[Vec<f64>],
    mbar: f64,
    inference: InferenceConfig,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<RelativeMagnitudeMultiFlciProblem, String> {
    if post_weight_sets.is_empty() {
        return Err("multi FLCI requires at least one functional".to_string());
    }
    let mut originals = Vec::with_capacity(post_weight_sets.len());
    let mut identified_sets = Vec::with_capacity(post_weight_sets.len());
    for post_weights in post_weight_sets {
        originals.push(compute_original_confidence_set(
            input,
            post_weights,
            inference,
        )?);
        identified_sets.push(crate::sensitivity::relative_magnitude::compute_relative_magnitude_identified_set_counted(
            input,
            post_weights,
            mbar, counter,
        )?);
    }
    Ok(RelativeMagnitudeMultiFlciProblem {
        input: input.clone(),
        post_weight_sets: post_weight_sets.to_vec(),
        mbar,
        inference,
        originals,
        identified_sets,
    })
}

/// Build a simultaneous `DeltaRM` problem from original confidence sets and
/// identified sets computed elsewhere. Neither the weights nor the sets are
/// checked against the input.
///
/// # Errors
/// Returns an error if `post_weight_sets` is empty or if `originals` or
/// `identified_sets` does not have one entry per set of weights.
#[cfg(test)]
pub fn build_relative_magnitude_multi_flci_problem_with_precomputed_sets(
    input: &HonestEventStudyInput,
    post_weight_sets: &[Vec<f64>],
    mbar: f64,
    inference: InferenceConfig,
    originals: Vec<HonestOriginalConfidenceSet>,
    identified_sets: Vec<HonestIdentifiedSet>,
) -> Result<RelativeMagnitudeMultiFlciProblem, String> {
    if post_weight_sets.is_empty() {
        return Err("multi FLCI requires at least one functional".to_string());
    }
    if originals.len() != post_weight_sets.len() {
        return Err(format!(
            "original confidence-set count {} does not match functional count {}",
            originals.len(),
            post_weight_sets.len()
        ));
    }
    if identified_sets.len() != post_weight_sets.len() {
        return Err(format!(
            "identified-set count {} does not match functional count {}",
            identified_sets.len(),
            post_weight_sets.len()
        ));
    }
    Ok(RelativeMagnitudeMultiFlciProblem {
        input: input.clone(),
        post_weight_sets: post_weight_sets.to_vec(),
        mbar,
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
) -> Result<f64, String> {
    if simulation_draws == 0 {
        return Err("multi FLCI simulation_draws must be positive".to_string());
    }
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
    problem: &RelativeMagnitudeMultiFlciProblem,
    joint_config: HonestJointPathConfig,
) -> Result<(f64, f64), String> {
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
            if !validate_confidence_level(pointwise) {
                return Err(format!(
                    "invalid Bonferroni-adjusted confidence level {pointwise}"
                ));
            }
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

/// Simultaneous `DeltaRM` intervals for the problem's functionals, with the
/// settings of [`RelativeMagnitudeConfidenceSetConfig::from_inference`] and
/// [`HonestJointPathConfig::default_for_production`]. Each interval is run at a
/// pointwise level calibrated so that they cover jointly at
/// `problem.inference.confidence_level`.
///
/// # Errors
/// Returns an error if the joint calibration fails or any interval cannot be
/// computed.
#[cfg(test)]
pub fn compute_relative_magnitude_multi_flci(
    problem: &RelativeMagnitudeMultiFlciProblem,
) -> Result<RelativeMagnitudeMultiFlciResult, String> {
    compute_relative_magnitude_multi_flci_counted(
        problem,
        &crate::sensitivity::SolverCounter::new(),
    )
}

#[cfg(test)]
pub(in crate::sensitivity) fn compute_relative_magnitude_multi_flci_counted(
    problem: &RelativeMagnitudeMultiFlciProblem,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<RelativeMagnitudeMultiFlciResult, String> {
    compute_relative_magnitude_multi_flci_with_config_counted(
        problem,
        RelativeMagnitudeConfidenceSetConfig::from_inference(problem.inference),
        HonestJointPathConfig::default_for_production(),
        counter,
    )
}

/// [`compute_relative_magnitude_multi_flci`] with the hybrid settings and the
/// joint calibration given explicitly.
///
/// # Errors
/// Returns an error if the joint calibration fails or any interval cannot be
/// computed.
pub fn compute_relative_magnitude_multi_flci_with_config(
    problem: &RelativeMagnitudeMultiFlciProblem,
    config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
) -> Result<RelativeMagnitudeMultiFlciResult, String> {
    compute_relative_magnitude_multi_flci_with_config_counted(
        problem,
        config,
        joint_config,
        &crate::sensitivity::SolverCounter::new(),
    )
}

pub(in crate::sensitivity) fn compute_relative_magnitude_multi_flci_with_config_counted(
    problem: &RelativeMagnitudeMultiFlciProblem,
    config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
    counter: &crate::sensitivity::SolverCounter,
) -> Result<RelativeMagnitudeMultiFlciResult, String> {
    let prepared_branches = prepare_relative_magnitude_branches(
        problem.input.num_pre_periods(),
        problem.input.num_post_periods(),
        problem.mbar,
    )?;
    let prepared_input_branches =
        prepare_relative_magnitude_input_branches(&problem.input, &prepared_branches);
    compute_relative_magnitude_multi_flci_with_prepared_branches(
        problem,
        config,
        joint_config,
        &prepared_branches,
        &prepared_input_branches,
        counter,
    )
}

pub(in crate::sensitivity) fn compute_relative_magnitude_multi_flci_with_prepared_branches(
    problem: &RelativeMagnitudeMultiFlciProblem,
    config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
    prepared_branches: &[RelativeMagnitudePreparedBranch],
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
    counter: &crate::sensitivity::SolverCounter,
) -> Result<RelativeMagnitudeMultiFlciResult, String> {
    let prepared_functional_branch_sets = problem
        .post_weight_sets
        .iter()
        .map(|post_weights| {
            prepare_relative_magnitude_functional_branches(
                post_weights,
                prepared_branches,
                prepared_input_branches,
            )
        })
        .collect::<Result<Vec<_>, String>>()?;
    compute_relative_magnitude_multi_flci_with_prepared_functional_branches(
        problem,
        config,
        joint_config,
        prepared_branches,
        prepared_input_branches,
        &prepared_functional_branch_sets,
        counter,
    )
}

pub(in crate::sensitivity) fn compute_relative_magnitude_multi_flci_with_prepared_functional_branches(
    problem: &RelativeMagnitudeMultiFlciProblem,
    config: RelativeMagnitudeConfidenceSetConfig,
    joint_config: HonestJointPathConfig,
    prepared_branches: &[RelativeMagnitudePreparedBranch],
    prepared_input_branches: &[RelativeMagnitudePreparedInputBranch],
    prepared_functional_branch_sets: &[Vec<RelativeMagnitudePreparedFunctionalBranch>],
    counter: &crate::sensitivity::SolverCounter,
) -> Result<RelativeMagnitudeMultiFlciResult, String> {
    let (pointwise_confidence_level, calibrated_max_t_critical_value) =
        joint_pointwise_confidence_level(problem, joint_config)?;
    let pointwise_inference = InferenceConfig::new(pointwise_confidence_level);
    let pointwise_default =
        RelativeMagnitudeConfidenceSetConfig::from_inference(pointwise_inference);
    // Intentional override: `hybrid_kappa` is re-derived from the
    // pointwise-adjusted confidence level so first-stage size tracks the
    // simultaneous calibration.
    let pointwise_config = RelativeMagnitudeConfidenceSetConfig {
        hybrid: config.hybrid,
        hybrid_kappa: pointwise_default.hybrid_kappa,
        least_favorable_draws: config.least_favorable_draws,
        least_favorable_seed: config.least_favorable_seed,
    };
    if prepared_functional_branch_sets.len() != problem.post_weight_sets.len() {
        return Err(format!(
            "prepared functional set count {} does not match functional count {}",
            prepared_functional_branch_sets.len(),
            problem.post_weight_sets.len()
        ));
    }

    // A single functional has nothing to be simultaneous about, so this must
    // return exactly what the scalar FLCI returns -- and before this it did not.
    //
    // Both paths end up choosing a grid point from a 1000-point grid spanning the
    // identified set padded by 20 standard errors, so the answer is quantised to
    // roughly 1e-3 on this data. The prepared-branch path and the direct path
    // reached that grid slightly differently and landed about two steps apart:
    // lower bounds agreed to 12 decimals while upper bounds differed by 1.8e-3.
    // Quantised disagreement cannot be closed by tightening anything, only by
    // having one path stop re-deriving what the other already computes.
    //
    // `joint_pointwise_confidence_level` already special-cases `n == 1` and
    // returns the nominal level with no calibration, so delegating here changes
    // no inference: the returned `pointwise_confidence_level` and critical value
    // are the ones this function would have reported anyway.
    if problem.post_weight_sets.len() == 1 {
        let post_weights = &problem.post_weight_sets[0];
        let original = &problem.originals[0];
        let identified_set = &problem.identified_sets[0];
        let conditional = crate::sensitivity::relative_magnitude::compute_relative_magnitude_confidence_set_with_config_counted(
            &problem.input,
            post_weights,
            problem.mbar,
            pointwise_inference,
            pointwise_config, counter,
        )?;
        return Ok(RelativeMagnitudeMultiFlciResult {
            confidence_level: problem.inference.confidence_level,
            pointwise_confidence_level,
            calibrated_max_t_critical_value,
            method: joint_config.method,
            points: vec![RelativeMagnitudeMultiFlciPoint {
                post_weights: post_weights.clone(),
                flci: (conditional.lb, conditional.ub),
                original_ci: original.ci,
                identified_set: (identified_set.lb, identified_set.ub),
                null_value: 0.0,
                robustly_significant: conditional.lb > 0.0 || conditional.ub < 0.0,
            }],
        });
    }

    let points = problem
        .post_weight_sets
        .par_iter()
        .zip(problem.originals.par_iter())
        .zip(problem.identified_sets.par_iter())
        .zip(prepared_functional_branch_sets.par_iter())
        .map(
            |(((post_weights, original), identified_set), prepared_functional_branches)| {
                let conditional =
                    compute_relative_magnitude_confidence_set_with_prepared_functional_branches(
                        &problem.input,
                        post_weights,
                        pointwise_inference,
                        pointwise_config,
                        original,
                        identified_set,
                        prepared_branches,
                        prepared_input_branches,
                        prepared_functional_branches,
                        counter,
                    )?;
                Ok(RelativeMagnitudeMultiFlciPoint {
                    post_weights: post_weights.clone(),
                    flci: (conditional.lb, conditional.ub),
                    original_ci: original.ci,
                    identified_set: (identified_set.lb, identified_set.ub),
                    null_value: 0.0,
                    robustly_significant: conditional.lb > 0.0 || conditional.ub < 0.0,
                })
            },
        )
        .collect::<Result<Vec<_>, String>>()?;

    Ok(RelativeMagnitudeMultiFlciResult {
        confidence_level: problem.inference.confidence_level,
        pointwise_confidence_level,
        calibrated_max_t_critical_value,
        method: joint_config.method,
        points,
    })
}
