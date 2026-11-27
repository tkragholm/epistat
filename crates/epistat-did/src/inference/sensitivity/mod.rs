//! `HonestDiD` sensitivity analysis helpers.
//!
//! This module implements the parts of the Rambachan and Roth sensitivity
//! framework that an event-study robustness analysis needs.
//!
//! Implemented scope:
//! - Original confidence sets for linear functionals of post-treatment
//!   event-study coefficients.
//! - Smoothness-based robustness (`ΔSD(M)`), with sign and monotonicity
//!   variants.
//! - Exact relative-magnitude identified sets (`ΔRM(Mbar)`).
//! - Relative-magnitude conditional confidence sets for arbitrary
//!   post-treatment linear functionals `l' τ_post`.
//! - Event-study-facing API helpers that lift the scalar routines onto a typed
//!   event-study coefficient and covariance surface.
//! - Simultaneous joint post-treatment path regions constructed from scalar
//!   `HonestDiD` intervals, either with a conservative Bonferroni correction or
//!   a tighter covariance-aware simulated max-`t` critical value.
//!
//! Limitation:
//! - Joint regions over several post-treatment functionals use a finite set of
//!   directions, optionally grown adaptively, with Bonferroni or simulated
//!   Gaussian calibration. They approximate, and do not reproduce, the full R
//!   `HonestDiD` optimization surface.
//!
//! Mathematical setup:
//! Let `β̂ = (β̂_pre, β̂_post)` denote the event-study coefficient vector and
//! `Σ` its covariance matrix. For a post-treatment linear functional
//! `θ = l' τ_post`, the original confidence set is formed from:
//!
//! ```text
//! θ̂ = l' β̂_post
//! Var(θ̂) = l' Σ_post l
//! ```
//!
//! Under `ΔRM(Mbar)`, each change of the violation `δ` between adjacent
//! post-treatment periods is bounded by `Mbar` times the largest such change
//! between adjacent pre-treatment periods:
//!
//! ```text
//! |δ_{t+1} − δ_t| ≤ Mbar · max_{s<0} |δ_{s+1} − δ_s|    for t ≥ 0
//! ```
//!
//! The exact identified set implemented here follows the LP construction used
//! by the `HonestDiD` package. The conditional confidence-set path follows the
//! same branch-wise least-favorable hybrid test used there for scalar targets.
//!
//! References:
//! - Rambachan, A. and Roth, J. (2023). "A More Credible Approach to Parallel
//!   Trends". *Review of Economic Studies*.
//! - `HonestDiD` R package, the reference for the implemented `ΔRM` paths.

mod adaptive_grid;
mod api_types;
mod conditional_confidence_sets;
mod conic;
mod event_study_assessment;
mod linear_algebra;
mod relative_magnitude;
mod smoothness;
mod solver_tally;
mod summary_api;

pub use api_types::*;
#[doc(hidden)]
pub use event_study_assessment::{
    assess_honest_event_study_directional_region,
    assess_honest_event_study_directional_region_with_config, assess_honest_event_study_functional,
    assess_honest_event_study_functional_with_config, assess_honest_event_study_joint_path_region,
    assess_honest_event_study_joint_path_region_with_config,
    assess_honest_event_study_optimization_surface_region,
    assess_honest_event_study_optimization_surface_region_adaptive,
    assess_honest_event_study_optimization_surface_region_adaptive_with_config,
    assess_honest_event_study_optimization_surface_region_with_config,
    assess_honest_event_study_period, assess_honest_event_study_period_with_config,
    assess_honest_event_study_post_functional, assess_honest_event_study_post_functional_flci,
    assess_honest_event_study_post_functional_flci_with_config,
    assess_honest_event_study_post_functional_multi_flci,
    assess_honest_event_study_post_functional_multi_flci_with_config,
    assess_honest_event_study_post_functional_with_config, assess_honest_event_study_workflow,
    assess_honest_event_study_workflow_with_config, compute_honest_event_study_scalar_for_input,
};
pub use relative_magnitude::least_favorable_intervals::{
    RelativeMagnitudeFlciProblem, RelativeMagnitudeFlciResult, RelativeMagnitudeMultiFlciProblem,
    build_relative_magnitude_flci_problem, build_relative_magnitude_multi_flci_problem,
    build_relative_magnitude_multi_flci_problem_with_precomputed_sets,
    compute_relative_magnitude_flci, compute_relative_magnitude_flci_with_config,
    compute_relative_magnitude_multi_flci, compute_relative_magnitude_multi_flci_with_config,
};
pub use relative_magnitude::{
    compute_linear_trend_relative_magnitude_confidence_set,
    compute_linear_trend_relative_magnitude_confidence_set_with_config,
    compute_linear_trend_relative_magnitude_identified_set,
    compute_monotone_linear_trend_relative_magnitude_confidence_set,
    compute_monotone_linear_trend_relative_magnitude_confidence_set_with_config,
    compute_monotone_linear_trend_relative_magnitude_identified_set,
    compute_monotone_relative_magnitude_confidence_set,
    compute_monotone_relative_magnitude_confidence_set_with_config,
    compute_monotone_relative_magnitude_identified_set, compute_original_confidence_set,
    compute_relative_magnitude_confidence_set,
    compute_relative_magnitude_confidence_set_with_config,
    compute_relative_magnitude_identified_set,
    compute_signed_linear_trend_relative_magnitude_confidence_set,
    compute_signed_linear_trend_relative_magnitude_confidence_set_with_config,
    compute_signed_linear_trend_relative_magnitude_identified_set,
    compute_signed_relative_magnitude_confidence_set,
    compute_signed_relative_magnitude_confidence_set_with_config,
    compute_signed_relative_magnitude_identified_set,
};
pub use smoothness::least_favorable_intervals::{
    SmoothnessFlciConfig, SmoothnessFlciProblem, SmoothnessFlciResult, SmoothnessMultiFlciProblem,
    build_smoothness_flci_problem, build_smoothness_multi_flci_problem, compute_smoothness_flci,
    compute_smoothness_flci_with_config, compute_smoothness_multi_flci,
    compute_smoothness_multi_flci_with_config,
};
pub use smoothness::{
    compute_monotone_smoothness_confidence_set,
    compute_monotone_smoothness_confidence_set_with_config,
    compute_monotone_smoothness_confidence_set_with_simulation,
    compute_monotone_smoothness_identified_set, compute_signed_smoothness_confidence_set,
    compute_signed_smoothness_confidence_set_with_config,
    compute_signed_smoothness_confidence_set_with_simulation,
    compute_signed_smoothness_identified_set, compute_smoothness_confidence_set,
    compute_smoothness_confidence_set_with_config,
    compute_smoothness_confidence_set_with_simulation, compute_smoothness_identified_set,
};
pub(crate) use solver_tally::SolverCounter;
pub use solver_tally::SolverCounts;
pub use summary_api::{
    SensitivityRestrictionKind, SensitivitySummary, SensitivitySummaryMethod,
    SensitivitySummaryRow, estimate_smoothness_upper_bound_from_pretrends,
    summarize_relative_magnitude_sensitivity, summarize_relative_magnitude_sensitivity_many,
    summarize_relative_magnitude_sensitivity_many_with_simulation,
    summarize_relative_magnitude_sensitivity_with_simulation, summarize_smoothness_sensitivity,
    summarize_smoothness_sensitivity_with_simulation,
};

#[cfg(test)]
mod tests;
