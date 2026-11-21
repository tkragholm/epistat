//! Match outcomes, diagnostics and the typed reasons they count.

use super::criteria::{CommonSupport, CriteriaValidationError, Estimand};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::{Display, Formatter};

/// Canonical exclusion-reason key for diagnostics.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ExclusionReason {
    AdditionalFilter,
    Constraint(ConstraintReason),
    DistanceCaliper(DistanceCaliperReason),
    InvalidCriteria(InvalidCriteriaReason),
    CommonSupportFailure(CommonSupportFailureReason),
    /// A case whose event date is before its birth date, by id. Risk-set
    /// matching under the default policy refuses such a cohort.
    EventBeforeBirth(String),
    /// An id that two units of the cohort share. Risk-set matching under the
    /// default policy refuses such a cohort.
    DuplicateId(String),
}

/// Typed estimand drift reason recorded by matching diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EstimandDriftReason {
    CommonSupportTrimming,
    UnmatchedAnchors,
    RatioShortfall,
    DistanceCaliperExclusion,
    CommonSupportCaliperInteraction,
}

impl Display for EstimandDriftReason {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let key = match self {
            Self::CommonSupportTrimming => "common_support_trimming",
            Self::UnmatchedAnchors => "unmatched_anchors",
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
    /// The control is the anchor itself.
    NoSelfMatch,
    /// The control differs from the anchor on a required stratum.
    StrataMismatch,
    /// The control's unique value was already used.
    UniqueKeyReused,
    /// The control's sex differs from the anchor's.
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
    NegativeBirthDateWindow,
    ZeroMatchRatio,
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

impl From<CriteriaValidationError> for InvalidCriteriaReason {
    fn from(err: CriteriaValidationError) -> Self {
        match err {
            CriteriaValidationError::NegativeBirthDateWindow => Self::NegativeBirthDateWindow,
            CriteriaValidationError::ZeroMatchRatio => Self::ZeroMatchRatio,
        }
    }
}

/// Typed common-support failure diagnostics code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CommonSupportFailureReason {
    RequiresPropensityScoreMap,
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

/// Matched anchor/candidate pair.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MatchedPair {
    /// Anchor identifier.
    pub case_id: String,
    /// Candidate identifier.
    pub control_id: String,
}

impl MatchedPair {
    /// Construct a pair using neutral anchor/comparator naming.
    #[must_use]
    pub fn new(anchor_id: impl Into<String>, comparator_id: impl Into<String>) -> Self {
        Self {
            case_id: anchor_id.into(),
            control_id: comparator_id.into(),
        }
    }

    /// Neutral accessor for the index/anchor identifier.
    #[must_use]
    pub fn anchor_id(&self) -> &str {
        self.case_id.as_str()
    }

    /// Neutral accessor for the comparator identifier.
    #[must_use]
    pub fn comparator_id(&self) -> &str {
        self.control_id.as_str()
    }
}

/// Matching summary statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MatchOutcome {
    /// Selected anchor/candidate pairs.
    pub pairs: Vec<MatchedPair>,
    /// Number of unmatched eligible anchors.
    pub unmatched_cases: usize,
    /// Number of unique candidates used (0 when matching with replacement).
    pub used_controls: usize,
    /// Number of anchors with at least one match.
    pub matched_cases: usize,
    /// Average number of selected candidates among matched anchors.
    pub avg_controls_per_case: f64,
    /// Structured run diagnostics.
    #[serde(default)]
    pub diagnostics: MatchDiagnostics,
}

impl MatchDiagnostics {
    /// Merge another diagnostics object into this one.
    pub fn merge(&mut self, other: Self) {
        self.total_anchors_evaluated += other.total_anchors_evaluated;
        self.anchors_with_no_candidates += other.anchors_with_no_candidates;
        self.anchors_below_required_ratio += other.anchors_below_required_ratio;
        self.anchors_on_lower_tier += other.anchors_on_lower_tier;
        self.matched_anchors += other.matched_anchors;
        self.pairs_selected += other.pairs_selected;
        self.common_support_trimmed_anchors += other.common_support_trimmed_anchors;
        self.common_support_trimmed_candidates += other.common_support_trimmed_candidates;

        for (reason, count) in other.exclusion_counts {
            *self.exclusion_counts.entry(reason).or_insert(0) += count;
        }
        for reason in other.estimand_drift_reasons {
            if !self.estimand_drift_reasons.contains(&reason) {
                self.estimand_drift_reasons.push(reason);
            }
        }
    }
}

/// Run diagnostics for matching.
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct MatchDiagnostics {
    /// Total anchors evaluated by the engine.
    pub total_anchors_evaluated: usize,
    /// Anchors with no eligible candidates after filtering.
    pub anchors_with_no_candidates: usize,
    /// Anchors that did not get the ratio asked for and have no declared
    /// fallback tier to account for it: without fallback tiers, the anchors
    /// with some candidates but fewer than the ratio, and with them, the
    /// anchors with fewer candidates than the lowest tier, which get none. An
    /// anchor matched on a declared lower tier is not counted here; see
    /// [`Self::anchors_on_lower_tier`].
    pub anchors_below_required_ratio: usize,
    /// With declared fallback tiers, the anchors matched on a tier below the
    /// primary ratio.
    #[serde(default)]
    pub anchors_on_lower_tier: usize,
    /// Matched anchor count.
    pub matched_anchors: usize,
    /// Number of selected pairs.
    pub pairs_selected: usize,
    /// Count of exclusions by reason.
    pub exclusion_counts: BTreeMap<ExclusionReason, usize>,
    /// Requested estimand from criteria.
    #[serde(default)]
    pub requested_estimand: Estimand,
    /// Realized estimand after trimming/matching exclusions.
    #[serde(default)]
    pub realized_estimand: Estimand,
    /// Number of anchors discarded by common-support trimming.
    #[serde(default)]
    pub common_support_trimmed_anchors: usize,
    /// Number of candidates discarded by common-support trimming.
    #[serde(default)]
    pub common_support_trimmed_candidates: usize,
    /// Applied overlap interval from common-support trimming when available.
    #[serde(default)]
    pub common_support_overlap: Option<(f64, f64)>,
    /// Applied common-support policy when configured.
    #[serde(default)]
    pub common_support_policy: Option<CommonSupport>,
    /// Anchor score bounds used to compute common-support overlap when available.
    #[serde(default)]
    pub common_support_anchor_score_bounds: Option<(f64, f64)>,
    /// Candidate score bounds used to compute common-support overlap when available.
    #[serde(default)]
    pub common_support_candidate_score_bounds: Option<(f64, f64)>,
    /// Structured machine-readable drift reasons.
    #[serde(default)]
    pub estimand_drift_reasons: Vec<EstimandDriftReason>,
}

/// The debug rendering lists every field. `anchors_on_lower_tier` is left out
/// when it is 0, so that the rendering of a job that matches on no lower tier
/// is the same as before the field existed and the hashes of renderings that
/// tests pin stay valid.
impl std::fmt::Debug for MatchDiagnostics {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut out = f.debug_struct("MatchDiagnostics");
        out.field("total_anchors_evaluated", &self.total_anchors_evaluated)
            .field(
                "anchors_with_no_candidates",
                &self.anchors_with_no_candidates,
            )
            .field(
                "anchors_below_required_ratio",
                &self.anchors_below_required_ratio,
            );
        if self.anchors_on_lower_tier != 0 {
            out.field("anchors_on_lower_tier", &self.anchors_on_lower_tier);
        }
        out.field("matched_anchors", &self.matched_anchors)
            .field("pairs_selected", &self.pairs_selected)
            .field("exclusion_counts", &self.exclusion_counts)
            .field("requested_estimand", &self.requested_estimand)
            .field("realized_estimand", &self.realized_estimand)
            .field(
                "common_support_trimmed_anchors",
                &self.common_support_trimmed_anchors,
            )
            .field(
                "common_support_trimmed_candidates",
                &self.common_support_trimmed_candidates,
            )
            .field("common_support_overlap", &self.common_support_overlap)
            .field("common_support_policy", &self.common_support_policy)
            .field(
                "common_support_anchor_score_bounds",
                &self.common_support_anchor_score_bounds,
            )
            .field(
                "common_support_candidate_score_bounds",
                &self.common_support_candidate_score_bounds,
            )
            .field("estimand_drift_reasons", &self.estimand_drift_reasons)
            .finish()
    }
}

impl MatchOutcome {
    /// Neutral accessor for matched anchor count.
    #[must_use]
    pub const fn matched_anchors(&self) -> usize {
        self.matched_cases
    }

    /// Neutral accessor for unmatched anchor count.
    #[must_use]
    pub const fn unmatched_anchors(&self) -> usize {
        self.unmatched_cases
    }

    /// Neutral accessor for average comparators per matched anchor.
    #[must_use]
    pub const fn avg_comparators_per_anchor(&self) -> f64 {
        self.avg_controls_per_case
    }

    /// Neutral accessor for unique comparator usage count.
    #[must_use]
    pub const fn used_comparators(&self) -> usize {
        self.used_controls
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matched_pair_and_outcome_accessors_are_consistent() {
        let pair = MatchedPair::new("anchor", "candidate");
        assert_eq!(pair.anchor_id(), "anchor");
        assert_eq!(pair.comparator_id(), "candidate");

        let outcome = MatchOutcome {
            pairs: vec![pair],
            unmatched_cases: 1,
            used_controls: 1,
            matched_cases: 1,
            avg_controls_per_case: 1.0,
            diagnostics: MatchDiagnostics::default(),
        };
        assert_eq!(outcome.matched_anchors(), 1);
        assert_eq!(outcome.unmatched_anchors(), 1);
        assert_eq!(outcome.used_comparators(), 1);
        assert!((outcome.avg_comparators_per_anchor() - 1.0).abs() < 1e-12);
    }
}
