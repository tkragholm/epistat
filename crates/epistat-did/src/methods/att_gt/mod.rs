#[cfg(feature = "honest")]
use crate::inference::sensitivity::{
    HonestAssessment, HonestEventStudyInput, HonestSensitivity, HonestWorkflowConfig,
    assess_honest_event_study_functional_with_config,
};
use crate::inference::{
    ClusterIndex, clustered_standard_error_from_index, normal_interval, validate_confidence_level,
};
use crate::types::{
    AttGtAggregationError, AttGtConfig, AttGtDrConfig, AttGtDrObservation, AttGtError,
    AttGtEstimate, AttGtEventTimeEstimate, AttGtInfluenceOutput, AttGtObservation, ComparisonGroup,
    EventStudyConfig, ReportingSummary,
};

pub(crate) mod aggregation;
pub mod aggte;
mod bands;
mod core;
pub mod pair_estimators;
pub mod panel_pairs;
mod pretest;

#[expect(
    deprecated,
    reason = "re-exports a deprecated function until it is removed"
)]
pub use aggregation::aggregate_att_gt_event_time;
pub use aggregation::{
    aggregate_att_gt_by_calendar_time, aggregate_att_gt_by_cohort,
    aggregate_att_gt_event_time_with_influence, aggregate_att_gt_overall,
};
pub use aggte::{
    AggteBootstrap, AggteConfig, AggteError, AggteEstimate, AggteResult, AggteType, UnitPanel,
    aggregate_att_gt, att_gt_clustered_standard_errors, row_panel,
};
pub use bands::att_gt_mboot_bands;
#[expect(
    deprecated,
    reason = "re-exports deprecated functions until they are removed"
)]
pub use bands::{att_gt_simultaneous_bands, att_gt_simultaneous_bands_with_influence};
pub use pair_estimators::estimate_att_gt_dr_efficient_with_influence;
pub use panel_pairs::{
    estimate_att_gt_dr_panel, estimate_att_gt_dr_panel_with_influence, unit_panel,
};
pub use pretest::{PretestError, PretrendWald, pretrend_wald_test};

/// Consolidated result of an event-study estimation.
#[deprecated(
    note = "part of the `InferenceDataset` workflow, which clusters with `G / (G - 1)` and labels \
            every estimator DR-DiD; use `estimate_att_gt_dr_panel_with_influence` and \
            `aggregate_att_gt`, with `did_clusters` for clustered variances"
)]
#[derive(Debug, Clone, PartialEq)]
pub struct EventStudyResult {
    /// ATT(g,t) estimates.
    pub estimates: Vec<AttGtEstimate>,
    /// Influence function of each estimate, one vector per estimate and one
    /// value per input observation.
    pub influence_functions: Vec<Vec<f64>>,
    /// Cluster membership of the units, if clustered inference was requested.
    /// Aggregated results use it for their standard errors.
    pub cluster_index: Option<ClusterIndex>,
    /// Configuration the estimates were produced with.
    pub config: EventStudyConfig,
    /// Name of the outcome variable, if known.
    pub outcome_name: Option<String>,
}

#[expect(
    deprecated,
    reason = "the deprecated workflow is built from its own deprecated types"
)]
impl EventStudyResult {
    /// Aggregate estimates by event-time with
    /// [`aggregate_att_gt_event_time_with_influence`]. With a cluster index, the
    /// standard errors and intervals are recomputed from the clustered
    /// influence functions.
    ///
    /// # Errors
    ///
    /// Returns [`AttGtAggregationError`] when event-time aggregation fails.
    pub fn aggregate_event_time(self) -> Result<EventTimeResult, AttGtAggregationError> {
        let aggregated = aggregate_att_gt_event_time_with_influence(
            &self.estimates,
            &self.influence_functions,
            self.config.aggregation,
        )?;

        let mut res = EventTimeResult {
            estimates: aggregated.estimates,
            influence_functions: aggregated.influence_functions,
            cluster_index: self.cluster_index,
            config: self.config,
            outcome_name: self.outcome_name,
        };

        res.update_clustered_inference();
        Ok(res)
    }
}

/// Robust assessment result for a post-treatment window.
#[cfg(feature = "honest")]
#[derive(Debug, Clone, PartialEq)]
pub struct WindowAssessment {
    /// Robust assessment of the window average.
    pub assessment: HonestAssessment,
    /// Post-treatment event times within the requested window that have an
    /// estimate. The assessment averages these, with equal weights.
    pub matched_periods: Vec<i32>,
    /// Inclusive range `(start, end)` of the requested event times.
    pub requested_range: (i32, i32),
}

#[cfg(feature = "honest")]
impl WindowAssessment {
    /// The assessment, if every integer event time in the requested window has
    /// an estimate.
    ///
    /// # Errors
    /// Returns an error if any integer in `[start, end]` is not in
    /// `matched_periods`.
    pub fn require_full_window(self) -> Result<HonestAssessment, String> {
        let (start, end) = self.requested_range;
        for p in start..=end {
            if !self.matched_periods.contains(&p) {
                return Err(format!(
                    "Partial coverage: window [{start}, {end}] is missing period {p}"
                ));
            }
        }
        Ok(self.assessment)
    }
}

/// Consolidated result of an event-time aggregated event-study.
#[deprecated(
    note = "part of the `InferenceDataset` workflow; use `aggregate_att_gt` and `did_clusters`"
)]
#[derive(Debug, Clone, PartialEq)]
pub struct EventTimeResult {
    /// Event-time aggregated estimates, in increasing event time.
    pub estimates: Vec<AttGtEventTimeEstimate>,
    /// Influence function of each aggregated estimate, one vector per estimate
    /// and one value per input observation.
    pub influence_functions: Vec<Vec<f64>>,
    /// Cluster membership of the units, if clustered inference was requested.
    pub cluster_index: Option<ClusterIndex>,
    /// Configuration the estimates were produced with.
    pub config: EventStudyConfig,
    /// Name of the outcome variable, if known.
    pub outcome_name: Option<String>,
}

#[expect(
    deprecated,
    reason = "the deprecated workflow is built from its own deprecated types"
)]
impl EventTimeResult {
    fn update_clustered_inference(&mut self) {
        if let Some(ref ci) = self.cluster_index {
            for (idx, est) in self.estimates.iter_mut().enumerate() {
                if let Ok(se) =
                    clustered_standard_error_from_index(&self.influence_functions[idx], ci)
                {
                    est.summary.se = se;
                    (est.summary.ci_low, est.summary.ci_high) =
                        normal_interval(est.summary.estimate, se, self.config.confidence_level);
                }
            }
        }
    }

    /// Standard error of the estimate at position `index` in `estimates`.
    ///
    /// # Panics
    /// Panics if `index` is out of range.
    #[must_use]
    pub fn se(&self, index: usize) -> f64 {
        self.estimates[index].summary.se
    }

    /// Standard error of the estimate at the given event time, `None` if there
    /// is no estimate at it.
    #[must_use]
    pub fn se_at(&self, event_time: i32) -> Option<f64> {
        self.estimates
            .iter()
            .find(|est| est.event_time == event_time)
            .map(|est| est.summary.se)
    }

    /// Confidence interval `(lower, upper)` of the estimate at position `index`
    /// in `estimates`.
    ///
    /// # Panics
    /// Panics if `index` is out of range.
    #[must_use]
    pub fn ci(&self, index: usize) -> (f64, f64) {
        let est = &self.estimates[index].summary;
        (est.ci_low, est.ci_high)
    }

    /// Confidence interval `(lower, upper)` of the estimate at the given event
    /// time, `None` if there is no estimate at it.
    #[must_use]
    pub fn ci_at(&self, event_time: i32) -> Option<(f64, f64)> {
        self.estimates
            .iter()
            .find(|est| est.event_time == event_time)
            .map(|est| (est.summary.ci_low, est.summary.ci_high))
    }

    /// Convert to `HonestEventStudyInput` for sensitivity analysis.
    ///
    /// The estimates are ordered by event time. Negative event times are
    /// pre-periods and the others post-periods. The covariance matrix is
    /// computed from the influence functions, clustered when a cluster index is
    /// present.
    ///
    /// # Errors
    ///
    /// Returns an error when there are no pre-treatment periods or covariance
    /// construction fails.
    #[cfg(feature = "honest")]
    pub fn to_honest_input(&self) -> Result<HonestEventStudyInput, String> {
        let mut pre_periods = Vec::new();
        let mut post_periods = Vec::new();
        let mut betahat = Vec::new();
        let mut if_matrix = Vec::new();

        // Sort estimates by event time to ensure consistent ordering
        let mut sorted_estimates = self.estimates.iter().enumerate().collect::<Vec<_>>();
        sorted_estimates.sort_by_key(|(_, est)| est.event_time);

        for (idx, est) in sorted_estimates {
            if est.event_time < 0 {
                pre_periods.push(est.event_time);
            } else {
                post_periods.push(est.event_time);
            }
            betahat.push(est.summary.estimate);
            if_matrix.push(self.influence_functions[idx].clone());
        }

        if pre_periods.is_empty() {
            return Err("Honest DiD analysis requires at least one pre-treatment event-time point to bound bias (none found in estimates)".to_string());
        }

        // Compute covariance matrix from influence functions
        // If clustering is present, use clustered covariance
        let covariance = if let Some(ref ci) = self.cluster_index {
            crate::inference::vcov::clustered_covariance_matrix_from_influence_matrix_index(
                &if_matrix, ci,
            )?
        } else {
            crate::inference::vcov::covariance_matrix_from_influence_matrix(&if_matrix)?
        };

        Ok(HonestEventStudyInput {
            betahat,
            covariance,
            pre_periods,
            post_periods,
        })
    }

    /// Sensitivity assessment of the equal-weight average of the post-treatment
    /// estimates with event times in `start..=end`.
    ///
    /// The null value is 0, the confidence level is that of the result's
    /// configuration, and only `config.relative_magnitude` is read from
    /// `config`.
    ///
    /// # Errors
    /// Returns an error if the honest input construction or assessment fails, or
    /// if no post-treatment estimate has an event time in the window.
    #[cfg(feature = "honest")]
    #[expect(
        clippy::needless_pass_by_value,
        reason = "forward-compatible API accepts owned workflow config"
    )]
    pub fn assess_window(
        &self,
        start: i32,
        end: i32,
        sensitivity: HonestSensitivity,
        config: HonestWorkflowConfig,
    ) -> Result<WindowAssessment, String> {
        let input = self.to_honest_input()?;
        let mut post_weights = vec![0.0; input.post_periods.len()];
        let mut matched_periods = Vec::new();

        for (idx, &p) in input.post_periods.iter().enumerate() {
            if p >= start && p <= end {
                post_weights[idx] = 1.0;
                matched_periods.push(p);
            }
        }

        if matched_periods.is_empty() {
            return Err(format!(
                "Window [{start}, {end}] covers no post-treatment periods"
            ));
        }

        let total_weight: f64 = post_weights.iter().sum();
        for val in &mut post_weights {
            *val /= total_weight;
        }

        let assessment = assess_honest_event_study_functional_with_config(
            &input,
            &post_weights,
            sensitivity,
            self.config.inference(),
            0.0,
            config.relative_magnitude,
        )?;

        Ok(WindowAssessment {
            assessment,
            matched_periods,
            requested_range: (start, end),
        })
    }

    /// Reporting summary of the estimate at the given event time, `None` if
    /// there is none. `n_obs` is the number of ATT(g,t) estimates aggregated at
    /// that event time and the estimator label is `DR-DiD`.
    #[must_use]
    pub fn to_summary(
        &self,
        outcome: &str,
        subgroup: &str,
        event_time: i32,
    ) -> Option<ReportingSummary> {
        self.estimates
            .iter()
            .find(|est| est.event_time == event_time)
            .map(|est| ReportingSummary {
                schema_version: 1,
                crate_version: env!("CARGO_PKG_VERSION").to_string(),
                config_hash: None,
                estimate: est.summary.estimate,
                std_error: est.summary.se,
                ci_lower: est.summary.ci_low,
                ci_upper: est.summary.ci_high,
                p_value: None, // Could be computed if needed
                outcome: outcome.to_string(),
                subgroup: subgroup.to_string(),
                estimator: "DR-DiD".to_string(),
                n_obs: est.summary.components, // This is component count, maybe we want unit count?
                n_clusters: self.cluster_index.as_ref().map(|ci| ci.n_groups),
            })
    }

    /// Reporting summaries of all estimates, in the order of `estimates`, with
    /// the outcome name `unnamed_outcome` if none is set. `n_obs` is the number
    /// of ATT(g,t) estimates aggregated at the event time.
    #[must_use]
    pub fn to_summaries(&self, subgroup: &str) -> Vec<ReportingSummary> {
        let outcome = self
            .outcome_name
            .as_deref()
            .unwrap_or("unnamed_outcome")
            .to_string();
        self.estimates
            .iter()
            .map(|est| ReportingSummary {
                schema_version: 1,
                crate_version: env!("CARGO_PKG_VERSION").to_string(),
                config_hash: None,
                estimate: est.summary.estimate,
                std_error: est.summary.se,
                ci_lower: est.summary.ci_low,
                ci_upper: est.summary.ci_high,
                p_value: None,
                outcome: outcome.clone(),
                subgroup: subgroup.to_string(),
                estimator: format!("DR-DiD (t={})", est.event_time),
                n_obs: est.summary.components,
                n_clusters: self.cluster_index.as_ref().map(|ci| ci.n_groups),
            })
            .collect()
    }
}

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

/// Estimate staggered-adoption `ATT(g,t)` effects with the doubly robust
/// repeated cross-section `DiD` estimator for each `(g, t)` pair, adjusting for
/// the covariates of the rows.
///
/// # Errors
///
/// Returns [`AttGtError`] when configuration/inputs are invalid, when required
/// pair cells are missing, or when DR nuisance estimation fails.
pub fn estimate_att_gt_dr(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<Vec<AttGtEstimate>, AttGtError> {
    pair_estimators::estimate_att_gt_dr(observations, config)
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

/// Estimate staggered-adoption ATT(g,t) using an outcome-regression (OR) path.
///
/// # Errors
///
/// Returns [`AttGtError`] for invalid inputs/configuration, missing required
/// cells, or failed pair estimation.
pub fn estimate_att_gt_or(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<Vec<AttGtEstimate>, AttGtError> {
    estimate_att_gt_or_with_influence(observations, config).map(|out| out.estimates)
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

/// Estimate staggered-adoption ATT(g,t) using an inverse-probability-weighted (IPW) path.
///
/// # Errors
///
/// Returns [`AttGtError`] for invalid inputs/configuration, missing required
/// cells, or failed pair estimation.
pub fn estimate_att_gt_ipw(
    observations: &[AttGtDrObservation],
    config: AttGtDrConfig,
) -> Result<Vec<AttGtEstimate>, AttGtError> {
    estimate_att_gt_ipw_with_influence(observations, config).map(|out| out.estimates)
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
