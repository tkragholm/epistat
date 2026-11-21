//! # `epistat_matching`
//!
//! Caliper-based cohort matching utilities for observational cohort studies.
//!
//! The crate provides a high-level, research-ready API for creating matched cohorts using
//! both standard anchor-to-candidate matching and longitudinal role-transition (risk-set)
//! sampling.
//!
//! ## Primary API: `MatchJob`
//!
//! The [`MatchJob`] builder is the recommended entry point for configuring and executing
//! matching. It supports:
//!
//! - **Standard Matching**: Fixed sets of anchors and candidates.
//! - **Role-Transition Matching**: Risk-set sampling within a longitudinal cohort.
//! - **Flexible Metrics**: Birth date calipers, exact match strata, Propensity Scores,
//!   and Mahalanobis distance.
//! - **Declarative Constraints**: Built-in constraints for gender, residency, and more.
//!
//! ```rust,ignore
//! use epistat_matching::prelude::*;
//!
//! // Standard matching example
//! let outcome = MatchJob::new_standard(&anchors, &candidates, 42)
//!     .with_ratio(MatchRatio::new(4).expect("non-zero ratio"))
//!     .with_birth_window(BirthDateWindowDays::new(30).expect("non-negative birth window"))
//!     .with_gender_match()
//!     .run();
//! ```
//!
//! For more details, see the [GUIDE.md](https://github.com/tkragholm/epistat_matching/blob/main/GUIDE.md).

// ---------------------------------------------------------------------------
// Shared builder-setter macros — eliminates identical 3-line setter bodies
// that arise from the builder pattern on structs with a `reason` or
// `distance_config` field.
// ---------------------------------------------------------------------------

/// Generates a `with_reason` builder setter for any struct that has a
/// `reason: &'static str` field.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_with_reason {
    () => {
        /// Override the default reason string used in diagnostics or rejection messages.
        #[must_use]
        pub const fn with_reason(mut self, reason: &'static str) -> Self {
            self.reason = reason;
            self
        }
    };
}

/// Generates a `with_distance_config` builder setter for structs that carry
/// `distance_config: Option<&'a DistanceConfig>`.  The invoking `impl` block
/// must have a lifetime `'a` in scope and `DistanceConfig` must be imported.
#[doc(hidden)]
#[macro_export]
macro_rules! impl_with_distance_config {
    () => {
        #[must_use]
        pub const fn with_distance_config(mut self, distance_config: &'a DistanceConfig) -> Self {
            self.distance_config = Some(distance_config);
            self
        }
    };
}

/// Generates a full `MatchingRecord` impl that delegates every method to an
/// inner field. A method added to `MatchingRecord` must be added here too:
/// `matching::records::tests::delegate_macro_forwards_every_trait_method`
/// fails until it is.  Used in test fixtures that wrap a `BaseRecord` or
/// `RoleTransitionRecord<BaseRecord>` to avoid repeating identical 3-line
/// getter bodies.
///
/// ```ignore
/// impl crate::MatchingRecord for MyWrapper {
///     crate::delegate_matching_record!(inner);
/// }
/// ```
#[doc(hidden)]
#[macro_export]
macro_rules! delegate_matching_record {
    ($field:ident) => {
        fn id(&self) -> &str {
            self.$field.id()
        }
        fn birth_date(&self) -> chrono::NaiveDate {
            self.$field.birth_date()
        }
        fn strata(&self) -> &std::collections::HashMap<String, String> {
            self.$field.strata()
        }
        fn numeric(&self, name: &str) -> Option<f64> {
            self.$field.numeric(name)
        }
        fn unique_key(&self) -> Option<&str> {
            self.$field.unique_key()
        }
        fn death_date(&self) -> Option<chrono::NaiveDate> {
            self.$field.death_date()
        }
        fn emigration_date(&self) -> Option<chrono::NaiveDate> {
            self.$field.emigration_date()
        }
        fn entry_date(&self) -> Option<chrono::NaiveDate> {
            self.$field.entry_date()
        }
    };
}

mod balance;
pub mod constraints;
mod errors;
mod matching;
pub mod output;
#[cfg(feature = "reporting")]
pub mod reporting;
mod role_transition;
pub mod spec;
mod types;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;
#[cfg(any(test, feature = "test-utils"))]
pub use test_utils::{date, record, test_outcome};

pub use rapidhash::RapidHashMap;

pub use balance::{
    BalanceInputError, BalanceWeights, balance_improvement, balance_report_on_weights,
    balance_summary,
};
pub use balance::{
    CategoricalCovariateSpec, NumericCovariateSpec, ProjectedBalanceRow, balance_diagnostics,
    balance_report, balance_report_with_options, balance_threshold_summary,
    build_projected_balance_rows, categorical_max_abs_diff, smd_numeric_from_values, variance,
};
pub use matching::{
    Assignment, CaliperConstraint, Chained, Constraint, ConstraintContext, ConstraintGroup,
    CovariateEncodingConfig, CovariateMatrix, CovariateRecord, CrossValidation, DateDistance,
    DeterministicSelection, DiscardedUnits, DistanceChannel, DistanceConfig, DistanceMetric,
    ElasticNetCvConfig, ElasticNetLogisticConfig, EstimationError, EuclideanDistancePreparation,
    EuclideanPreparationConfig, EuclideanScale, ExactMatchConstraint, IdMapEuclideanDistance,
    IdMapMahalanobisDistance, IdMapPropensityScoreDistance, LambdaRule, LogisticRegressionConfig,
    MahalanobisCovarianceStrategy, MahalanobisDistance, MahalanobisDistancePreparation,
    MahalanobisError, MahalanobisPreparationConfig, MahalanobisTransform, MatchJob, MatchOrder,
    MatchWeightMethod, MatchingRecord, MissingValuePolicy, NearestBirthDateSelection,
    NearestSelection, PairWeight, PairWeightRow, PairWeightSet, PairWeightTable,
    PropensityDistancePreparation, PropensityEstimator, PropensityLink, PropensityMatchedOutcome,
    PropensityScoreConfig, PropensityScoreDistance, PropensityScoreEstimate,
    PropensityScoreOutputScale, RandomSelection, ReconstructedEpisode, ReconstructedMatchedPair,
    ReconstructedMatchingOptions, ReconstructedMatchingOutput, ReconstructedMatchingTier,
    ResidentAtIndexRecord, RidgeReport, RoleIndexedRecord, Rounds, SelectionStrategy,
    StandardMatchRequest, StrataExactConstraint, SubclassReferenceGroup, SubclassSummary,
    SubclassificationConfig, SubclassificationOutcome, UnitRole, UnitWeightRow, UnitWeightSet,
    UnitWeightTable, UsedControlsVec, VariableRatio, VariableRatioError, cross_validation_folds,
    effective_sample_size, estimate_propensity_and_match, match_standard,
    match_standard_with_discard, match_weights_from_pairs, pair_weights_from_pairs,
    prepare_euclidean_distance_config, prepare_mahalanobis_distance_config,
    prepare_propensity_distance_config, prepare_propensity_distance_config_with_discard,
    reconstruct_case_control_pairs, subclassify_by_propensity_score_map,
};
pub use matching::{ExactMatchError, ExactMatchOutcome, exact_match_subclasses};
pub use matching::{
    QuickMatchError, QuickMatchOptions, QuickMatchOutcome, SeedMethod, quick_clusters,
    quick_match_subclasses,
};
pub use role_transition::{
    DefaultRiskSetPolicy, RiskSetPolicy, TransitionMatchRequest, match_transition,
};
pub use types::{
    AgeLimitYears, AnchorRecord, BalanceDiagnostics, BalanceRecord, BalanceRecordBuilder,
    BalanceReport, BalanceReportOptions, BalanceThresholdSummary, BalanceThresholds, BaseRecord,
    BirthDateWindowDays, CandidateRecord, CategoricalBalance, CategoricalLevelBalance,
    CommonSupport, CommonSupportFailureReason, ConstraintReason, ControlIdx, CovariateValue,
    CriteriaValidationError, DistanceCaliper, DistanceCaliperReason, Estimand, EstimandDriftReason,
    ExclusionReason, InvalidCriteriaReason, MatchDiagnostics, MatchOutcome, MatchRatio,
    MatchedPair, MatchingCriteria, NumericBalance, NumericBalanceThresholdCheck,
    NumericBalanceTransform, RoleTransitionOptions, RoleTransitionRecord,
    SupplementalBalanceCovariates, UniqueValueId, ValidatedMatchingCriteria,
};
pub use types::{
    BalanceImprovement, BalanceSummary, GroupSampleSizes, LevelImprovement, NumericImprovement,
    PairDistance, SampleSizeTable,
};

pub use output::{
    LongMatches, MatchData, MatchDataOptions, MatchMatrix, OutputError, Units, get_matches,
    match_data, match_matrix,
};
pub use spec::{
    AssignmentSpec, CovarianceSpec, CovariatesSpec, DistanceSpec, EstimatorSpec, LambdaSpec,
    LinkSpec, MatchMethod, MatchMode, MatchSpec, MissingSpec, NearestSpec, OrderSpec, QuickSpec,
    ReferenceSpec, ScaleSpec, SpecError, SpecInput, SpecOutput, SubclassSpec, TransformSpec,
    run_spec,
};

#[cfg(feature = "bench-internals")]
#[must_use]
pub fn benchmark_ecdf_distance_stats(
    case_values: &[f64],
    control_values: &[f64],
) -> Option<(f64, f64)> {
    balance::benchmark_ecdf_distance_stats(case_values, control_values)
}

#[cfg(feature = "bench-internals")]
#[must_use]
pub fn benchmark_eqq_distance_stats(
    case_values: &[f64],
    control_values: &[f64],
) -> Option<(f64, f64)> {
    balance::benchmark_eqq_distance_stats(case_values, control_values)
}

/// Common imports for consumers of this crate.
pub mod prelude {
    pub use crate::constraints::{AntiExact, GenderMatch};
    pub use crate::matching::{
        Assignment, DeterministicSelection, ExactMatchError, ExactMatchOutcome, MatchOrder,
        NearestBirthDateSelection, NearestSelection, RandomSelection, Rounds, SelectionStrategy,
        exact_match_subclasses,
    };
    pub use crate::{
        AgeLimitYears, AnchorRecord, AssignmentSpec, BalanceDiagnostics, BalanceRecord,
        BalanceRecordBuilder, BalanceReport, BalanceReportOptions, BalanceThresholdSummary,
        BalanceThresholds, BaseRecord, BirthDateWindowDays, CaliperConstraint, CandidateRecord,
        CategoricalBalance, CategoricalCovariateSpec, CategoricalLevelBalance, Chained,
        CommonSupport, CommonSupportFailureReason, Constraint, ConstraintContext, ConstraintGroup,
        ConstraintReason, ControlIdx, CovarianceSpec, CovariateEncodingConfig, CovariateMatrix,
        CovariateRecord, CovariateValue, CovariatesSpec, CriteriaValidationError, CrossValidation,
        DateDistance, DefaultRiskSetPolicy, DistanceCaliper, DistanceCaliperReason,
        DistanceChannel, DistanceConfig, DistanceMetric, DistanceSpec, ElasticNetCvConfig,
        ElasticNetLogisticConfig, Estimand, EstimandDriftReason, EstimationError, EstimatorSpec,
        EuclideanDistancePreparation, EuclideanPreparationConfig, EuclideanScale,
        ExactMatchConstraint, ExclusionReason, IdMapEuclideanDistance, IdMapMahalanobisDistance,
        IdMapPropensityScoreDistance, InvalidCriteriaReason, LambdaRule, LambdaSpec, LinkSpec,
        LogisticRegressionConfig, LongMatches, MahalanobisCovarianceStrategy, MahalanobisDistance,
        MahalanobisDistancePreparation, MahalanobisError, MahalanobisPreparationConfig,
        MahalanobisTransform, MatchData, MatchDataOptions, MatchDiagnostics, MatchJob, MatchMatrix,
        MatchMethod, MatchMode, MatchOutcome, MatchRatio, MatchSpec, MatchWeightMethod,
        MatchedPair, MatchingCriteria, MatchingRecord, MissingSpec, MissingValuePolicy,
        NearestSpec, NumericBalance, NumericBalanceThresholdCheck, NumericBalanceTransform,
        NumericCovariateSpec, OrderSpec, OutputError, PairWeight, PairWeightRow, PairWeightSet,
        PairWeightTable, ProjectedBalanceRow, PropensityDistancePreparation, PropensityEstimator,
        PropensityLink, PropensityMatchedOutcome, PropensityScoreConfig, PropensityScoreDistance,
        PropensityScoreEstimate, PropensityScoreOutputScale, ReconstructedEpisode,
        ReconstructedMatchedPair, ReconstructedMatchingOptions, ReconstructedMatchingOutput,
        ReconstructedMatchingTier, ReferenceSpec, ResidentAtIndexRecord, RidgeReport,
        RiskSetPolicy, RoleIndexedRecord, RoleTransitionOptions, RoleTransitionRecord, ScaleSpec,
        SpecError, SpecInput, SpecOutput, StandardMatchRequest, StrataExactConstraint,
        SubclassReferenceGroup, SubclassSpec, SubclassSummary, SubclassificationConfig,
        SubclassificationOutcome, SupplementalBalanceCovariates, TransformSpec,
        TransitionMatchRequest, UniqueValueId, UnitRole, UnitWeightRow, UnitWeightSet,
        UnitWeightTable, Units, ValidatedMatchingCriteria, balance_diagnostics, balance_report,
        balance_report_with_options, balance_threshold_summary, build_projected_balance_rows,
        categorical_max_abs_diff, cross_validation_folds, effective_sample_size,
        estimate_propensity_and_match, get_matches, match_data, match_matrix, match_standard,
        match_standard_with_discard, match_transition, match_weights_from_pairs,
        pair_weights_from_pairs, prepare_euclidean_distance_config,
        prepare_mahalanobis_distance_config, prepare_propensity_distance_config,
        prepare_propensity_distance_config_with_discard, reconstruct_case_control_pairs, run_spec,
        smd_numeric_from_values, subclassify_by_propensity_score_map,
    };
    pub use crate::{
        BalanceImprovement, BalanceInputError, BalanceSummary, BalanceWeights, GroupSampleSizes,
        LevelImprovement, NumericImprovement, PairDistance, SampleSizeTable, balance_improvement,
        balance_report_on_weights, balance_summary,
    };

    pub mod constraints {
        pub use crate::constraints::{
            AntiExact, Caliper, DateWindow, GenderMatch, MissingPolicy, MustBeAlive,
            MustBeResident, caliper_on_field, numeric_field_sd,
        };
    }
}
