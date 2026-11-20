//! Balance report, threshold and diagnostic types.

use super::records::CovariateValue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cohort-level balance diagnostics from match results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceDiagnostics {
    /// Proportion of anchors matched.
    pub match_rate: f64,
    /// Number of matched anchors.
    pub matched_cases: usize,
    /// Number of unmatched anchors.
    pub unmatched_cases: usize,
    /// Average candidates per matched anchor.
    pub avg_controls_per_case: f64,
    /// Counts by strata key (`anchor_count`, `candidate_count`).
    pub strata_counts: HashMap<String, (usize, usize)>,
}

impl BalanceDiagnostics {
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
}

/// Numeric covariate balance summary.
///
/// A statistic that does not exist for the data is `None`: a mean or proportion of an
/// empty group, a standardized mean difference or variance ratio whose variances are
/// not usable, a distance with an empty group, Cramér's V with no observations.
/// Serialized as `null`; the CSV reports write `NA`.
///
/// The standardized mean differences divide by a standard deviation from the
/// sample before matching, chosen by the realised estimand as in
/// [`balance_report`](crate::balance_report). A binary variable has no variance
/// ratio, and its eCDF and eQQ distances are the absolute difference in
/// proportions. The eQQ distances read the larger group at the smaller group's
/// cumulative weights as a step function, as `summary()` of MatchIt does with
/// `standardize = FALSE`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NumericBalance {
    /// Covariate name.
    pub name: String,
    /// Pre-match anchor mean.
    pub mean_case_pre: Option<f64>,
    /// Pre-match candidate mean.
    pub mean_control_pre: Option<f64>,
    /// Pre-match standardized mean difference.
    pub smd_pre: Option<f64>,
    /// Pre-match variance ratio (`anchor / candidate`).
    #[serde(default)]
    pub var_ratio_pre: Option<f64>,
    /// Pre-match mean absolute eCDF distance.
    #[serde(default)]
    pub ecdf_mean_diff_pre: Option<f64>,
    /// Pre-match max absolute eCDF distance.
    #[serde(default)]
    pub ecdf_max_diff_pre: Option<f64>,
    /// Pre-match mean absolute eQQ distance.
    #[serde(default)]
    pub eqq_mean_diff_pre: Option<f64>,
    /// Pre-match max absolute eQQ distance.
    #[serde(default)]
    pub eqq_max_diff_pre: Option<f64>,
    /// Post-match anchor mean.
    pub mean_case_post: Option<f64>,
    /// Post-match candidate mean.
    pub mean_control_post: Option<f64>,
    /// Post-match standardized mean difference.
    pub smd_post: Option<f64>,
    /// Post-match variance ratio (`anchor / candidate`).
    #[serde(default)]
    pub var_ratio_post: Option<f64>,
    /// Post-match mean absolute eCDF distance.
    #[serde(default)]
    pub ecdf_mean_diff_post: Option<f64>,
    /// Post-match max absolute eCDF distance.
    #[serde(default)]
    pub ecdf_max_diff_post: Option<f64>,
    /// Post-match mean absolute eQQ distance.
    #[serde(default)]
    pub eqq_mean_diff_post: Option<f64>,
    /// Post-match max absolute eQQ distance.
    #[serde(default)]
    pub eqq_max_diff_post: Option<f64>,
}

/// Categorical level balance summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalLevelBalance {
    /// Level name.
    pub level: String,
    /// Pre-match anchor proportion.
    pub p_case_pre: Option<f64>,
    /// Pre-match candidate proportion.
    pub p_control_pre: Option<f64>,
    /// Pre-match standardized mean difference.
    pub smd_pre: Option<f64>,
    /// Post-match anchor proportion.
    pub p_case_post: Option<f64>,
    /// Post-match candidate proportion.
    pub p_control_post: Option<f64>,
    /// Post-match standardized mean difference.
    pub smd_post: Option<f64>,
}

/// Categorical covariate balance summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CategoricalBalance {
    /// Covariate name.
    pub name: String,
    /// Per-level balance statistics.
    pub levels: Vec<CategoricalLevelBalance>,
    /// Pre-match Cramer's V.
    pub cramers_v_pre: Option<f64>,
    /// Post-match Cramer's V.
    pub cramers_v_post: Option<f64>,
}

/// Full balance report across numeric and categorical covariates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceReport {
    /// Numeric covariate summaries.
    pub numeric: Vec<NumericBalance>,
    /// Categorical covariate summaries.
    pub categorical: Vec<CategoricalBalance>,
}

/// Sample sizes of one group, as in the sample size table of MatchIt's
/// `summary()`.
///
/// Weights are the matching weights multiplied by the sampling weights, so a
/// unit is matched when that product is positive. The effective sample size of
/// a set of weights is `sum(w)^2 / sum(w^2)`, `None` when no weight is positive.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct GroupSampleSizes {
    /// Units in the group before matching.
    pub all: usize,
    /// Effective sample size of the group before matching. It differs from
    /// `all` only with sampling weights.
    pub all_ess: Option<f64>,
    /// Units with a positive weight.
    pub matched: usize,
    /// Effective sample size of the weights after matching.
    pub matched_ess: Option<f64>,
    /// Units that were not discarded and have no weight.
    pub unmatched: usize,
    /// Units discarded before matching, for example by common support.
    pub discarded: usize,
}

/// The sample size table of a balance summary: one column per group.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SampleSizeTable {
    /// The anchors (treated units).
    pub cases: GroupSampleSizes,
    /// The candidates (control units).
    pub controls: GroupSampleSizes,
}

/// Mean absolute difference between the members of matched pairs, for one
/// covariate or one level of a categorical covariate.
///
/// A unit that is in several pairs contributes one difference per pair. Pairs in
/// which either unit lacks the covariate are left out. Sampling weights do not
/// enter.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PairDistance {
    /// Covariate name; a level of a categorical covariate is the covariate
    /// name followed by the level.
    pub name: String,
    /// Mean absolute difference on the scale of the covariate. `None` without
    /// a pair that has the covariate on both sides.
    pub raw: Option<f64>,
    /// The mean absolute difference divided by the standard deviation that
    /// standardises the mean difference. `None` when either is undefined.
    pub standardised: Option<f64>,
}

/// A balance report with the sample size table and pair distances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceSummary {
    /// Covariate balance before and after matching.
    pub report: BalanceReport,
    /// Sample sizes before and after matching.
    pub sample_sizes: SampleSizeTable,
    /// Pair distances of the numeric covariates and then of each level of the
    /// categorical covariates, in the order of the report. Empty when no pairs
    /// were given.
    pub pair_distances: Vec<PairDistance>,
}

/// Percent balance improvement of one numeric covariate.
///
/// Each value is `100 (|before| - |after|) / |before|`, and for the variance
/// ratio the same on the absolute logarithm of the ratio. It is `None` when
/// the statistic before matching is 0 or either value is undefined.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NumericImprovement {
    /// Covariate name.
    pub name: String,
    /// Improvement in the standardised mean difference.
    pub smd: Option<f64>,
    /// Improvement in the variance ratio.
    pub var_ratio: Option<f64>,
    /// Improvement in the mean eCDF distance.
    pub ecdf_mean: Option<f64>,
    /// Improvement in the maximum eCDF distance.
    pub ecdf_max: Option<f64>,
    /// Improvement in the mean eQQ distance.
    pub eqq_mean: Option<f64>,
    /// Improvement in the maximum eQQ distance.
    pub eqq_max: Option<f64>,
}

/// Percent balance improvement of one level of a categorical covariate.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct LevelImprovement {
    /// Covariate name.
    pub name: String,
    /// Level name.
    pub level: String,
    /// Improvement in the standardised mean difference.
    pub smd: Option<f64>,
}

/// Percent balance improvement for a balance report.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct BalanceImprovement {
    /// Numeric covariates in the order of the report.
    pub numeric: Vec<NumericImprovement>,
    /// Levels of categorical covariates in the order of the report.
    pub categorical: Vec<LevelImprovement>,
}

/// Optional transformed numeric covariates included in balance summaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum NumericBalanceTransform {
    /// Include only observed covariates.
    #[default]
    None,
    /// Include squared numeric terms (for example `age^2`).
    Squares,
    /// Include squared terms and pairwise interactions (for example `age * income`).
    ///
    /// As with `interactions = TRUE` in MatchIt, the columns are the numeric
    /// covariates and one indicator per level of each categorical covariate,
    /// in the order of the covariate names, and a level is named by the
    /// covariate name followed by the level. Each column's square and its
    /// products with the later columns follow the covariates in that order. A
    /// term that is zero everywhere or equal to its first factor, such as the
    /// square of an indicator or the product of two levels of one covariate, is
    /// left out.
    SquaresAndPairwiseInteractions,
}

/// Configuration for balance reporting behavior.
#[derive(bon::Builder, Debug, Clone, Serialize, Deserialize)]
pub struct BalanceReportOptions {
    /// Which transformed numeric terms to include.
    #[builder(default)]
    #[serde(default)]
    pub numeric_transforms: NumericBalanceTransform,
    /// Optional supplemental covariates for balance-only diagnostics.
    #[builder(default)]
    #[serde(default)]
    pub supplemental_covariates: SupplementalBalanceCovariates,
}

impl Default for BalanceReportOptions {
    fn default() -> Self {
        Self {
            numeric_transforms: NumericBalanceTransform::None,
            supplemental_covariates: SupplementalBalanceCovariates::default(),
        }
    }
}

/// Supplemental covariates used only for balance diagnostics.
///
/// This enables `addlvariables`-style workflows where diagnostics include
/// variables not stored in the matching records.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SupplementalBalanceCovariates {
    /// Supplemental covariates for anchor/case records keyed by record id.
    #[serde(default)]
    pub cases: HashMap<String, HashMap<String, CovariateValue>>,
    /// Supplemental covariates for candidate/control records keyed by record id.
    #[serde(default)]
    pub controls: HashMap<String, HashMap<String, CovariateValue>>,
}

/// Threshold configuration for post-match balance checks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceThresholds {
    /// Maximum allowed absolute post-match SMD.
    pub smd_abs_max: Option<f64>,
    /// Minimum allowed post-match variance ratio.
    pub var_ratio_min: Option<f64>,
    /// Maximum allowed post-match variance ratio.
    pub var_ratio_max: Option<f64>,
    /// Maximum allowed post-match eCDF max distance.
    pub ecdf_max_diff_max: Option<f64>,
    /// Maximum allowed post-match eQQ max distance.
    pub eqq_max_diff_max: Option<f64>,
}

impl Default for BalanceThresholds {
    fn default() -> Self {
        Self {
            smd_abs_max: Some(0.1),
            var_ratio_min: Some(0.8),
            var_ratio_max: Some(1.25),
            ecdf_max_diff_max: Some(0.1),
            eqq_max_diff_max: None,
        }
    }
}

impl BalanceThresholds {
    /// Strict threshold profile for diagnostics-sensitive applications.
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            smd_abs_max: Some(0.05),
            var_ratio_min: Some(0.9),
            var_ratio_max: Some(1.11),
            ecdf_max_diff_max: Some(0.05),
            eqq_max_diff_max: None,
        }
    }

    /// Moderate threshold profile aligned with crate defaults.
    #[must_use]
    pub fn moderate() -> Self {
        Self::default()
    }

    /// Lenient threshold profile for exploratory checks.
    #[must_use]
    pub const fn lenient() -> Self {
        Self {
            smd_abs_max: Some(0.15),
            var_ratio_min: Some(0.67),
            var_ratio_max: Some(1.5),
            ecdf_max_diff_max: Some(0.15),
            eqq_max_diff_max: None,
        }
    }
}

/// Threshold-check result for one numeric covariate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumericBalanceThresholdCheck {
    /// Covariate name.
    pub name: String,
    /// Whether post-match SMD is within threshold (or `None` when disabled).
    pub smd_post_ok: Option<bool>,
    /// Whether post-match variance ratio is within threshold (or `None` when disabled).
    pub var_ratio_post_ok: Option<bool>,
    /// Whether post-match eCDF max distance is within threshold (or `None` when disabled).
    pub ecdf_max_diff_post_ok: Option<bool>,
    /// Whether post-match eQQ max distance is within threshold (or `None` when disabled).
    pub eqq_max_diff_post_ok: Option<bool>,
    /// Aggregate pass/fail over enabled checks.
    pub all_enabled_checks_ok: bool,
}

/// Summary of threshold checks across numeric covariates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BalanceThresholdSummary {
    /// Per-covariate threshold checks.
    pub numeric: Vec<NumericBalanceThresholdCheck>,
    /// Aggregate pass/fail over all numeric covariates and enabled checks.
    pub all_enabled_checks_ok: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balance_thresholds_default_is_reasonable() {
        let thresholds = BalanceThresholds::default();
        assert_eq!(thresholds.smd_abs_max, Some(0.1));
        assert_eq!(thresholds.var_ratio_min, Some(0.8));
        assert_eq!(thresholds.var_ratio_max, Some(1.25));
        assert_eq!(thresholds.ecdf_max_diff_max, Some(0.1));
        assert_eq!(thresholds.eqq_max_diff_max, None);
    }

    #[test]
    fn balance_report_options_defaults_to_no_transforms() {
        let options = BalanceReportOptions::default();
        assert_eq!(options.numeric_transforms, NumericBalanceTransform::None);
        assert!(options.supplemental_covariates.cases.is_empty());
        assert!(options.supplemental_covariates.controls.is_empty());
    }

    #[test]
    fn balance_threshold_presets_are_ordered_by_strictness() {
        let strict = BalanceThresholds::strict();
        let moderate = BalanceThresholds::moderate();
        let lenient = BalanceThresholds::lenient();

        assert!(strict.smd_abs_max < moderate.smd_abs_max);
        assert!(moderate.smd_abs_max < lenient.smd_abs_max);
        assert!(strict.ecdf_max_diff_max < moderate.ecdf_max_diff_max);
        assert!(moderate.ecdf_max_diff_max < lenient.ecdf_max_diff_max);
    }
}
