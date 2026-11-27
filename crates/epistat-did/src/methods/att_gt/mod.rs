use crate::inference::validate_confidence_level;
use crate::types::{
    AttGtConfig, AttGtDrConfig, AttGtDrObservation, AttGtError, AttGtEstimate,
    AttGtInfluenceOutput, AttGtObservation, ComparisonGroup,
};

pub(crate) mod aggregation;
pub mod aggte;
mod bands;
mod core;
mod overlap;
pub mod pair_estimators;
pub mod panel_pairs;
mod preprocess;
mod pretest;

pub use aggregation::{
    aggregate_att_gt_by_calendar_time, aggregate_att_gt_by_cohort,
    aggregate_att_gt_event_time_with_influence, aggregate_att_gt_overall,
};
pub use aggte::{
    AggteBootstrap, AggteConfig, AggteError, AggteEstimate, AggteResult, AggteType, UnitPanel,
    aggregate_att_gt, att_gt_clustered_standard_errors, row_panel,
};
pub use bands::att_gt_mboot_bands;
pub use pair_estimators::estimate_att_gt_dr_efficient_with_influence;
pub use panel_pairs::{
    estimate_att_gt_dr_panel_with_influence, estimate_att_gt_ipw_panel_with_influence,
    estimate_att_gt_or_panel_with_influence, unit_panel,
};
pub use pretest::{PretestError, PretrendWald, pretrend_wald_test};

/// Estimate staggered-adoption `ATT(g,t)` effects without covariates, from
/// weighted cell means.
///
/// Where rows carry `unit_id`, a cell is estimated as a panel, from each unit's
/// change in outcome. Otherwise it is estimated from repeated cross-sections.
/// The standard error is analytic and the interval a two-sided normal one.
/// `ComparisonGroup::MatchedNeverTreated` is refused.
///
/// # Errors
///
/// Returns [`AttGtError`] when configuration/inputs are invalid or no estimable
/// group-time pairs can be formed.
pub fn estimate_att_gt(
    observations: &[AttGtObservation],
    config: AttGtConfig,
) -> Result<Vec<AttGtEstimate>, AttGtError> {
    core::estimate_att_gt(observations, config)
}

/// Estimate staggered-adoption `ATT(g,t)` via DR repeated cross-section `DiD`,
/// and return aligned influence vectors for joint-process inference.
///
/// # Errors
///
/// Returns [`AttGtError`] for invalid inputs/configuration, missing required
/// cells, failed nuisance estimation, or incompatible influence-vector shapes.
pub fn estimate_att_gt_dr_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    pair_estimators::estimate_att_gt_dr_with_influence(observations, config)
}

/// Estimate staggered-adoption ATT(g,t) using an outcome-regression (OR) path,
/// and return aligned influence vectors for joint-process inference.
///
/// # Errors
///
/// Returns [`AttGtError`] for invalid inputs/configuration, missing required
/// cells, failed pair estimation, or incompatible influence-vector shapes.
pub fn estimate_att_gt_or_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    pair_estimators::estimate_att_gt_or_with_influence(observations, config)
}

/// Estimate staggered-adoption ATT(g,t) using an inverse-probability-weighted
/// (IPW) path, and return aligned influence vectors for joint-process inference.
///
/// # Errors
///
/// Returns [`AttGtError`] for invalid inputs/configuration, missing required
/// cells, failed pair estimation, or incompatible influence-vector shapes.
pub fn estimate_att_gt_ipw_with_influence(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<AttGtInfluenceOutput, AttGtError> {
    pair_estimators::estimate_att_gt_ipw_with_influence(observations, config)
}

/// Whether a row is a control for the cell of cohort `group` whose later
/// period is `pair_time`.
///
/// `comparison_cohort` is the cohort a never-treated row was matched to, and
/// is read only under [`ComparisonGroup::MatchedNeverTreated`]; the other two
/// modes ignore it, so a caller without the tag passes `None`.
fn is_control_for_pair(
    first_treated_time: Option<i32>,
    comparison_cohort: Option<i32>,
    group: i32,
    pair_time: i32,
    mode: ComparisonGroup,
    anticipation_periods: i32,
) -> bool {
    match mode {
        ComparisonGroup::NeverTreated => first_treated_time.is_none(),
        ComparisonGroup::NotYetTreated => {
            first_treated_time.is_none_or(|g| g > pair_time + anticipation_periods)
        }
        ComparisonGroup::MatchedNeverTreated => {
            first_treated_time.is_none() && comparison_cohort == Some(group)
        }
    }
}

fn validate_config(config: AttGtConfig) -> Result<(), AttGtError> {
    if !validate_confidence_level(config.confidence_level.confidence_level) {
        return Err(AttGtError::InvalidConfidenceLevel);
    }
    if config.anticipation_periods < 0 {
        return Err(AttGtError::InvalidAnticipationPeriods);
    }
    Ok(())
}

#[cfg(test)]
mod tests;
