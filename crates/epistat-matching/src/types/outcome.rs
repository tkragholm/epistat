//! Match outcomes, diagnostics and the typed reasons they count.

use super::criteria::{CommonSupport, Estimand, default_estimand};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};

/// Canonical exclusion-reason key for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExclusionReason {
    /// A pair removed by a filter other than the named constraints.
    AdditionalFilter,
    /// A pair refused by a constraint.
    Constraint(ConstraintReason),
    /// A pair refused by a distance caliper.
    DistanceCaliper(DistanceCaliperReason),
    /// The job was refused because its criteria are invalid.
    InvalidCriteria(InvalidCriteriaReason),
    /// The job was refused because common support could not be set.
    CommonSupportFailure(CommonSupportFailureReason),
    /// A case whose event date is before its birth date, by id. Risk-set
    /// matching under the default policy refuses such a cohort.
    EventBeforeBirth(String),
    /// An id that two units of the cohort share. Risk-set matching under the
    /// default policy refuses such a cohort.
    DuplicateId(String),
}

impl ExclusionReason {
    /// Whether the reason says the whole job was refused, so that the outcome
    /// has no pairs and one count of this reason.
    #[must_use]
    pub const fn refuses_job(&self) -> bool {
        matches!(
            self,
            Self::InvalidCriteria(_)
                | Self::CommonSupportFailure(_)
                | Self::EventBeforeBirth(_)
                | Self::DuplicateId(_)
        )
    }
}

impl ExclusionReason {
    /// The error for a reason that refuses the whole job, `None` for a reason
    /// that excludes pairs or units.
    pub(crate) fn refusal(&self) -> Option<crate::Error> {
        use crate::Error;
        Some(match self {
            Self::InvalidCriteria(InvalidCriteriaReason::NegativeBirthDateWindow) => {
                Error::NegativeBirthDateWindow
            }
            Self::InvalidCriteria(InvalidCriteriaReason::ZeroMatchRatio) => Error::ZeroMatchRatio,
            Self::InvalidCriteria(reason) => Error::InvalidCriteria(*reason),
            Self::CommonSupportFailure(reason) => Error::CommonSupport(*reason),
            Self::EventBeforeBirth(id) => Error::EventBeforeBirth { id: id.clone() },
            Self::DuplicateId(id) => Error::DuplicateId { id: id.clone() },
            Self::AdditionalFilter | Self::Constraint(_) | Self::DistanceCaliper(_) => return None,
        })
    }
}

/// The outcome, or the error when the matcher refused the job: the outcome
/// then has no pairs and counts the reason once.
///
/// # Errors
///
/// The error of the first reason in the diagnostics that refuses the job.
pub(crate) fn reject_invalid(outcome: MatchOutcome) -> Result<MatchOutcome, crate::Error> {
    match outcome
        .diagnostics
        .exclusion_counts
        .keys()
        .find_map(ExclusionReason::refusal)
    {
        Some(error) => Err(error),
        None => Ok(outcome),
    }
}

impl From<InvalidCriteriaReason> for ExclusionReason {
    fn from(reason: InvalidCriteriaReason) -> Self {
        Self::InvalidCriteria(reason)
    }
}

impl From<CommonSupportFailureReason> for ExclusionReason {
    fn from(reason: CommonSupportFailureReason) -> Self {
        Self::CommonSupportFailure(reason)
    }
}

/// Typed estimand drift reason recorded by matching diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EstimandDriftReason {
    /// Cases were dropped as outside common support.
    CommonSupportTrimming,
    /// Some cases found no control.
    UnmatchedCases,
    /// Some cases found fewer controls than the ratio.
    RatioShortfall,
    /// A distance caliper removed pairs.
    DistanceCaliperExclusion,
    /// Common-support trimming and a caliper both removed units.
    CommonSupportCaliperInteraction,
}

impl Display for EstimandDriftReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let key = match self {
            Self::CommonSupportTrimming => "common_support_trimming",
            Self::UnmatchedCases => "unmatched_cases",
            Self::RatioShortfall => "ratio_shortfall",
            Self::DistanceCaliperExclusion => "distance_caliper_exclusion",
            Self::CommonSupportCaliperInteraction => "common_support_caliper_interaction",
        };
        f.write_str(key)
    }
}

/// Typed constraint exclusion reason.
///
/// Each variant is named by the text the matching constraint of the same name
/// reports, so [`Self::from_reason_str`] and `Display` are inverses.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ConstraintReason {
    /// The control was already used and replacement is off.
    ReplacementDisallowed,
    /// The control is the case itself.
    NoSelfMatch,
    /// The control differs from the case on a required stratum.
    StrataMismatch,
    /// The control's unique value was already used.
    UniqueKeyReused,
    /// The control's sex differs from the case's.
    GenderMismatch,
    /// A numeric caliper on a field is exceeded.
    CaliperExceeded,
    /// The birth-date window is exceeded.
    DateWindowExceeded,
    /// An anti-exact field is equal.
    AntiExact,
    /// The control is dead at the index date.
    NotAliveAtIndex,
    /// The control is not resident at the index date.
    NotResidentAtIndex,
    /// A constraint of the caller's, by its own text.
    Custom(String),
}

impl ConstraintReason {
    /// The typed reason for the text a constraint reports.
    #[must_use]
    pub fn from_reason_str(reason: &str) -> Self {
        match reason {
            "used_control_replacement_disabled" => Self::ReplacementDisallowed,
            "self_match" => Self::NoSelfMatch,
            "strata_mismatch" => Self::StrataMismatch,
            "unique_key_reused" => Self::UniqueKeyReused,
            "gender_mismatch" => Self::GenderMismatch,
            "caliper_mismatch" => Self::CaliperExceeded,
            "date_window_mismatch" => Self::DateWindowExceeded,
            "antiexact" => Self::AntiExact,
            "control_not_alive_at_index_date" => Self::NotAliveAtIndex,
            "control_non_resident_at_index_date" => Self::NotResidentAtIndex,
            _ => Self::Custom(reason.to_string()),
        }
    }
}

impl Display for ConstraintReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let key = match self {
            Self::ReplacementDisallowed => "used_control_replacement_disabled",
            Self::NoSelfMatch => "self_match",
            Self::StrataMismatch => "strata_mismatch",
            Self::UniqueKeyReused => "unique_key_reused",
            Self::GenderMismatch => "gender_mismatch",
            Self::CaliperExceeded => "caliper_mismatch",
            Self::DateWindowExceeded => "date_window_mismatch",
            Self::AntiExact => "antiexact",
            Self::NotAliveAtIndex => "control_not_alive_at_index_date",
            Self::NotResidentAtIndex => "control_non_resident_at_index_date",
            Self::Custom(reason) => reason.as_str(),
        };
        f.write_str(key)
    }
}

/// Typed distance-caliper exclusion reason.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DistanceCaliperReason {
    /// A caliper on the birth-date distance.
    Date,
    /// A caliper on the propensity score distance.
    PropensityScore,
    /// A caliper on the Mahalanobis distance.
    Mahalanobis,
    /// A caliper on the Euclidean distance.
    Euclidean,
    /// A caliper on a distance matrix.
    DistanceMatrix,
    /// A caliper with a name of the caller's.
    Custom(String),
}

impl DistanceCaliperReason {
    /// The typed reason for the text a distance channel reports.
    #[must_use]
    pub fn from_reason_str(reason: &str) -> Self {
        match reason {
            "date_caliper" => Self::Date,
            "propensity_score_caliper" => Self::PropensityScore,
            "mahalanobis_caliper" => Self::Mahalanobis,
            "euclidean_caliper" => Self::Euclidean,
            "distance_matrix_caliper" => Self::DistanceMatrix,
            _ => Self::Custom(reason.to_string()),
        }
    }
}

impl Display for DistanceCaliperReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Date => "date_caliper",
            Self::PropensityScore => "propensity_score_caliper",
            Self::Mahalanobis => "mahalanobis_caliper",
            Self::Euclidean => "euclidean_caliper",
            Self::DistanceMatrix => "distance_matrix_caliper",
            Self::Custom(reason) => reason,
        })
    }
}

/// Typed invalid-criteria diagnostics code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum InvalidCriteriaReason {
    /// The birth-date window is negative.
    NegativeBirthDateWindow,
    /// The match ratio is 0.
    ZeroMatchRatio,
    /// The study options are invalid.
    StudyOptions,
    /// The nearest-neighbour options contradict each other or the distance.
    NearestOptions,
    /// A fallback ratio is above the primary ratio.
    FallbackRatioAbovePrimary,
}

impl Display for InvalidCriteriaReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let key = match self {
            Self::NegativeBirthDateWindow => "negative_birth_date_window",
            Self::ZeroMatchRatio => "zero_match_ratio",
            Self::StudyOptions => "study_options",
            Self::NearestOptions => "nearest_options",
            Self::FallbackRatioAbovePrimary => "fallback_ratio_above_primary",
        };
        f.write_str(key)
    }
}

impl InvalidCriteriaReason {
    /// The reason for an error of the criteria's validation.
    pub(crate) const fn of(error: &crate::Error) -> Self {
        match error {
            crate::Error::NegativeBirthDateWindow => Self::NegativeBirthDateWindow,
            crate::Error::ZeroMatchRatio => Self::ZeroMatchRatio,
            _ => Self::StudyOptions,
        }
    }
}

/// Typed common-support failure diagnostics code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CommonSupportFailureReason {
    /// Common support was asked for without propensity scores.
    RequiresPropensityScoreMap,
    /// The cases' scores and the controls' scores do not overlap.
    NoOverlap,
}

impl Display for CommonSupportFailureReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let key = match self {
            Self::RequiresPropensityScoreMap => "requires_propensity_score_map",
            Self::NoOverlap => "no_overlap",
        };
        f.write_str(key)
    }
}

impl Display for ExclusionReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AdditionalFilter => f.write_str("additional_filter"),
            Self::Constraint(reason) => Display::fmt(reason, f),
            Self::DistanceCaliper(reason) => Display::fmt(reason, f),
            Self::InvalidCriteria(reason) => write!(f, "invalid_criteria:{reason}"),
            Self::CommonSupportFailure(reason) => write!(f, "common_support_failure:{reason}"),
            Self::EventBeforeBirth(id) => write!(f, "event_before_birth:{id}"),
            Self::DuplicateId(id) => write!(f, "duplicate_id:{id}"),
        }
    }
}

/// The ids of the units that common-support trimming left out, cases and
/// controls apart, each in the order of the units given.
///
/// Both lists are empty when nothing was trimmed.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrimmedUnits {
    /// Ids of the trimmed cases.
    pub cases: Vec<String>,
    /// Ids of the trimmed controls.
    pub controls: Vec<String>,
}

impl TrimmedUnits {
    /// Whether no unit was trimmed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.cases.is_empty() && self.controls.is_empty()
    }
}

/// A case and one of its controls.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchedPair {
    /// The id of the case.
    pub case_id: String,
    /// The id of the control.
    pub control_id: String,
}

impl MatchedPair {
    /// The pair of the case `case_id` and the control `control_id`.
    #[must_use]
    pub fn new(case_id: impl Into<String>, control_id: impl Into<String>) -> Self {
        Self {
            case_id: case_id.into(),
            control_id: control_id.into(),
        }
    }
}

/// Matching summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchOutcome {
    /// Selected case/control pairs.
    pub pairs: Vec<MatchedPair>,
    /// Number of unmatched eligible cases.
    pub unmatched_cases: usize,
    /// Number of unique controls used (0 when matching with replacement).
    pub used_controls: usize,
    /// Number of cases with at least one match.
    pub matched_cases: usize,
    /// Average number of selected controls among matched cases, `None` when no case
    /// is matched.
    pub avg_controls_per_case: Option<f64>,
    /// Structured run diagnostics.
    #[serde(default)]
    pub diagnostics: MatchDiagnostics,
}

impl MatchDiagnostics {
    /// Count `count` more units or controls excluded for `reason`.
    pub fn add_exclusion(&mut self, reason: ExclusionReason, count: usize) {
        *self.exclusion_counts.entry(reason).or_insert(0) += count;
    }

    /// Merge another diagnostics object into this one.
    pub fn merge(&mut self, other: Self) {
        self.total_cases_evaluated += other.total_cases_evaluated;
        self.cases_with_no_controls += other.cases_with_no_controls;
        self.cases_below_required_ratio += other.cases_below_required_ratio;
        self.cases_on_lower_tier += other.cases_on_lower_tier;
        self.matched_cases += other.matched_cases;
        self.pairs_selected += other.pairs_selected;
        self.common_support_trimmed_cases += other.common_support_trimmed_cases;
        self.common_support_trimmed_controls += other.common_support_trimmed_controls;
        self.common_support_trimmed_ids
            .cases
            .extend(other.common_support_trimmed_ids.cases);
        self.common_support_trimmed_ids
            .controls
            .extend(other.common_support_trimmed_ids.controls);

        for (reason, count) in other.exclusion_counts {
            self.add_exclusion(reason, count);
        }
        for reason in other.estimand_drift_reasons {
            if !self.estimand_drift_reasons.contains(&reason) {
                self.estimand_drift_reasons.push(reason);
            }
        }
    }
}

/// Run diagnostics for matching.
#[non_exhaustive]
#[derive(Clone, Serialize, Deserialize)]
pub struct MatchDiagnostics {
    /// Total cases evaluated by the engine.
    pub total_cases_evaluated: usize,
    /// Cases with no eligible controls after filtering.
    pub cases_with_no_controls: usize,
    /// Cases that did not get the ratio asked for and have no declared
    /// fallback tier to account for it: without fallback tiers, the cases
    /// with some controls but fewer than the ratio, and with them, the
    /// cases with fewer controls than the lowest tier, which get none. An
    /// case matched on a declared lower tier is not counted here; see
    /// [`Self::cases_on_lower_tier`].
    pub cases_below_required_ratio: usize,
    /// With declared fallback tiers, the cases matched on a tier below the
    /// primary ratio.
    #[serde(default)]
    pub cases_on_lower_tier: usize,
    /// Matched case count.
    pub matched_cases: usize,
    /// Number of selected pairs.
    pub pairs_selected: usize,
    /// Count of exclusions by reason.
    pub exclusion_counts: BTreeMap<ExclusionReason, usize>,
    /// Requested estimand from criteria.
    #[serde(default = "default_estimand")]
    pub requested_estimand: Estimand,
    /// Realized estimand after trimming/matching exclusions.
    #[serde(default = "default_estimand")]
    pub realized_estimand: Estimand,
    /// Number of cases discarded by common-support trimming.
    #[serde(default)]
    pub common_support_trimmed_cases: usize,
    /// Number of controls discarded by common-support trimming.
    #[serde(default)]
    pub common_support_trimmed_controls: usize,
    /// Ids of the units that common-support trimming discarded. The debug
    /// rendering leaves them out, since the counts above say how many there
    /// are.
    #[serde(default)]
    pub common_support_trimmed_ids: TrimmedUnits,
    /// Applied overlap interval from common-support trimming when available.
    #[serde(default)]
    pub common_support_overlap: Option<(f64, f64)>,
    /// Applied common-support policy when configured.
    #[serde(default)]
    pub common_support_policy: Option<CommonSupport>,
    /// Case score bounds used to compute common-support overlap when available.
    #[serde(default)]
    pub common_support_case_score_bounds: Option<(f64, f64)>,
    /// Control score bounds used to compute common-support overlap when available.
    #[serde(default)]
    pub common_support_control_score_bounds: Option<(f64, f64)>,
    /// Structured machine-readable drift reasons.
    #[serde(default)]
    pub estimand_drift_reasons: Vec<EstimandDriftReason>,
}

/// Empty diagnostics, with the ATT as the requested and the realised
/// estimand.
impl Default for MatchDiagnostics {
    fn default() -> Self {
        Self {
            total_cases_evaluated: 0,
            cases_with_no_controls: 0,
            cases_below_required_ratio: 0,
            cases_on_lower_tier: 0,
            matched_cases: 0,
            pairs_selected: 0,
            exclusion_counts: BTreeMap::new(),
            requested_estimand: default_estimand(),
            realized_estimand: default_estimand(),
            common_support_trimmed_cases: 0,
            common_support_trimmed_controls: 0,
            common_support_trimmed_ids: TrimmedUnits::default(),
            common_support_overlap: None,
            common_support_policy: None,
            common_support_case_score_bounds: None,
            common_support_control_score_bounds: None,
            estimand_drift_reasons: Vec::new(),
        }
    }
}

/// The debug rendering lists every field but the ids of the trimmed units, which
/// the trimmed counts summarise. `cases_on_lower_tier` is left out
/// when it is 0, so that the rendering of a job that matches on no lower tier
/// does not mention it.
#[expect(clippy::missing_fields_in_debug)]
impl std::fmt::Debug for MatchDiagnostics {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut out = f.debug_struct("MatchDiagnostics");
        out.field("total_cases_evaluated", &self.total_cases_evaluated)
            .field("cases_with_no_controls", &self.cases_with_no_controls)
            .field(
                "cases_below_required_ratio",
                &self.cases_below_required_ratio,
            );
        if self.cases_on_lower_tier != 0 {
            out.field("cases_on_lower_tier", &self.cases_on_lower_tier);
        }
        out.field("matched_cases", &self.matched_cases)
            .field("pairs_selected", &self.pairs_selected)
            .field("exclusion_counts", &self.exclusion_counts)
            .field("requested_estimand", &self.requested_estimand)
            .field("realized_estimand", &self.realized_estimand)
            .field(
                "common_support_trimmed_cases",
                &self.common_support_trimmed_cases,
            )
            .field(
                "common_support_trimmed_controls",
                &self.common_support_trimmed_controls,
            )
            .field("common_support_overlap", &self.common_support_overlap)
            .field("common_support_policy", &self.common_support_policy)
            .field(
                "common_support_case_score_bounds",
                &self.common_support_case_score_bounds,
            )
            .field(
                "common_support_control_score_bounds",
                &self.common_support_control_score_bounds,
            )
            .field("estimand_drift_reasons", &self.estimand_drift_reasons)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_pair_and_outcome_accessors_are_consistent() {
        let pair = MatchedPair::new("anchor", "candidate");
        assert_eq!(pair.case_id.as_str(), "anchor");
        assert_eq!(pair.control_id.as_str(), "candidate");

        let outcome = MatchOutcome {
            pairs: vec![pair],
            unmatched_cases: 1,
            used_controls: 1,
            matched_cases: 1,
            avg_controls_per_case: Some(1.0),
            diagnostics: MatchDiagnostics::default(),
        };
        assert_eq!(outcome.matched_cases, 1);
        assert_eq!(outcome.unmatched_cases, 1);
        assert_eq!(outcome.used_controls, 1);
        assert_eq!(outcome.avg_controls_per_case, Some(1.0));
    }
}
