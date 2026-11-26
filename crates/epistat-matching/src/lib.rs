//! # `epistat_matching`
//!
//! Matching for cohort studies, with results that do not depend on the number
//! of threads.
//!
//! The methods follow `MatchIt` where it has them: nearest-neighbour,
//! optimal, full, subclass, exact, coarsened exact, quick, genetic and
//! cardinality matching, with propensity scores, Mahalanobis and Euclidean
//! distances, calipers, common support and `mahvars`. Longitudinal cohorts can
//! be matched in risk sets at the time a person becomes a case, by exposure
//! density, or across sequential trials. Balance statistics, matching
//! weights, effect estimates and sensitivity analyses work on the output of
//! any of them.
//!
//! ## Two ways in
//!
//! [`MatchJob`] is a typed builder for pair matching. Each strategy or
//! constraint changes its type, which suits Rust code.
//!
//! ```rust,ignore
//! use epistat_matching::prelude::*;
//!
//! let outcome = MatchJob::new_standard(&anchors, &candidates, 42)
//!     .with_ratio(MatchRatio::new(4).expect("non-zero ratio"))
//!     .with_birth_window(BirthDateWindowDays::new(30).expect("non-negative birth window"))
//!     .with_gender_match()
//!     .run();
//! ```
//!
//! [`MatchSpec`] describes any of the methods as plain values with serde, and
//! [`run_spec`] checks it and runs it. It suits a caller that reads its
//! options at run time, such as a binding to R or Python. An option that a
//! method does not read, a score that is missing or not finite, and any other
//! invalid option are errors that name the option or the unit.
//!
//! ## Large cohorts
//!
//! The matchers take any type that implements [`MatchingRecord`], and the
//! covariate methods [`CovariateRecord`]. [`BalanceRecord`] owns its values
//! in hash maps, about 2 KiB per unit. A [`Cohort`] holds the same values as
//! columns, about 120 bytes per unit with five strata and five numeric
//! covariates, and its rows are records that give the same results, so ten
//! million units fit in memory.
//!
//! ## Arrow input
//!
//! With the cargo feature `arrow`, the module `arrow` builds the records or a
//! [`Cohort`] from Apache Arrow arrays, so a cohort held in columns, in Arrow
//! or Polars, needs no per-row structs of its own.
#![warn(missing_docs)]

// ---------------------------------------------------------------------------
// Shared builder-setter macros, which remove identical 3-line setter bodies
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
        /// Sets the distance configuration the matcher reads its distances from.
        #[must_use]
        pub const fn with_distance_config(mut self, distance_config: &'a DistanceConfig) -> Self {
            self.distance_config = Some(distance_config);
            self
        }
    };
}

/// Generates a full `MatchingRecord` impl that delegates every method to an
/// inner field, or to the record behind a reference with `*`. A method added
/// to `MatchingRecord` must be added to `delegate_matching_methods` too:
/// `matching::records::tests::delegate_macro_forwards_every_trait_method`
/// fails until it is.  Used for the wrappers of a record, in the crate and in
/// test fixtures.
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
        $crate::delegate_matching_methods!(self, self.$field);
    };
    (*) => {
        $crate::delegate_matching_methods!(self, (**self));
    };
}

/// The methods of `MatchingRecord`, each forwarded to `$target`, which names
/// the record through `$this`.
#[doc(hidden)]
#[macro_export]
macro_rules! delegate_matching_methods {
    ($this:ident, $target:expr) => {
        fn id(&$this) -> &str {
            $target.id()
        }
        fn birth_date(&$this) -> chrono::NaiveDate {
            $target.birth_date()
        }
        fn strata(&$this) -> &std::collections::HashMap<String, String> {
            $target.strata()
        }
        fn stratum(&$this, key: &str) -> Option<&str> {
            $target.stratum(key)
        }
        fn numeric(&$this, name: &str) -> Option<f64> {
            $target.numeric(name)
        }
        fn numeric_table(&$this) -> Option<usize> {
            $target.numeric_table()
        }
        fn numeric_slot(&$this, name: &str) -> Option<usize> {
            $target.numeric_slot(name)
        }
        fn numeric_in_slot(&$this, slot: usize) -> Option<f64> {
            $target.numeric_in_slot(slot)
        }
        fn unique_key(&$this) -> Option<&str> {
            $target.unique_key()
        }
        fn death_date(&$this) -> Option<chrono::NaiveDate> {
            $target.death_date()
        }
        fn emigration_date(&$this) -> Option<chrono::NaiveDate> {
            $target.emigration_date()
        }
        fn entry_date(&$this) -> Option<chrono::NaiveDate> {
            $target.entry_date()
        }
    };
}

/// The same for the methods of `CovariateRecord`. A method added to the trait
/// must be added here:
/// `matching::records::tests::delegate_macro_forwards_every_covariate_method`
/// fails until it is.
macro_rules! delegate_covariate_record {
    ($field:ident) => {
        delegate_covariate_record!(@methods self, self.$field);
    };
    (*) => {
        delegate_covariate_record!(@methods self, (**self));
    };
    (@methods $this:ident, $target:expr) => {
        fn covariates(&$this) -> &HashMap<String, CovariateValue> {
            $target.covariates()
        }
        fn covariate(&$this, name: &str) -> Option<CovariateRef<'_>> {
            $target.covariate(name)
        }
        fn covariate_names(&$this) -> impl Iterator<Item = &str> {
            $target.covariate_names()
        }
        fn covariate_table(&$this) -> Option<usize> {
            $target.covariate_table()
        }
        fn covariate_slot(&$this, name: &str) -> Option<usize> {
            $target.covariate_slot(name)
        }
        fn covariate_in_slot(&$this, slot: usize) -> CovariateRef<'_> {
            $target.covariate_in_slot(slot)
        }
    };
}

pub mod abadie_imbens;
#[cfg(feature = "arrow")]
pub mod arrow;
mod balance;
mod cohort;
pub mod constraints;
pub mod effects;
mod errors;
pub mod imputation;
mod kdtree;
mod matching;
pub mod multigroup;
pub mod output;
mod overlap;
#[cfg(feature = "reporting")]
pub mod reporting;
mod role_transition;
pub mod sensitivity;
mod sorting;
pub mod spec;
mod trials;
mod types;

#[cfg(any(test, feature = "test-utils"))]
pub mod test_utils;
#[cfg(any(test, feature = "test-utils"))]
pub use test_utils::{date, record, test_outcome};

pub use rapidhash::RapidHashMap;

pub use cohort::{Cohort, CohortBuilder, CohortError, CohortRow};

pub use balance::{
    BalanceInputError, BalanceWeights, balance_improvement, balance_report_on_weights,
    balance_summary,
};
pub use balance::{
    CategoricalCovariateSpec, NumericCovariateSpec, ProjectedBalanceRow, balance_diagnostics,
    balance_report, balance_report_with_options, balance_threshold_summary,
    build_projected_balance_rows, categorical_max_abs_diff, smd_numeric_from_values,
};
pub use matching::{
    Assignment, CaliperConstraint, CbpsConfig, CbpsEstimand, CbpsFit, CbpsMethod, Chained,
    Constraint, ConstraintContext, ConstraintGroup, CovariateEncodingConfig, CovariateMatrix,
    CovariateRecord, CrossValidation, DateDistance, DeterministicSelection, DiscardedUnits,
    DistanceChannel, DistanceConfig, DistanceMetric, ElasticNetCvConfig, ElasticNetLogisticConfig,
    EstimationError, EuclideanDistancePreparation, EuclideanPreparationConfig, EuclideanScale,
    ExactMatchConstraint, IdMapEuclideanDistance, IdMapMahalanobisDistance,
    IdMapPropensityScoreDistance, InvalidWeightError, LambdaRule, LogisticRegressionConfig,
    MahalanobisCovarianceStrategy, MahalanobisDistance, MahalanobisDistancePreparation,
    MahalanobisError, MahalanobisPreparationConfig, MahalanobisTransform, MatchJob, MatchOrder,
    MatchWeightMethod, MatchingRecord, MatrixDistance, MissingValuePolicy,
    NearestBirthDateSelection, NearestSelection, PairDistanceMatrix, PairDistanceMatrixError,
    PairWeight, PairWeightRow, PairWeightSet, PairWeightTable, PropensityDistancePreparation,
    PropensityEstimator, PropensityLink, PropensityMatchedOutcome, PropensityScoreConfig,
    PropensityScoreDistance, PropensityScoreEstimate, PropensityScoreOutputScale, RandomSelection,
    ReconstructedEpisode, ReconstructedMatchedPair, ReconstructedMatchingError,
    ReconstructedMatchingOptions, ReconstructedMatchingOutput, ReconstructedMatchingTier,
    ResidentAtIndexRecord, RoleIndexedRecord, Rounds, ScoreTable, SelectionStrategy,
    StandardMatchRequest, StrataExactConstraint, SubclassReferenceGroup, SubclassSummary,
    SubclassedUnits, SubclassificationConfig, SubclassificationError, SubclassificationOutcome,
    UnitRole, UnitWeightRow, UnitWeightSet, UnitWeightTable, UsedControlsVec, VariableRatio,
    VariableRatioError, VectorTable, cross_validation_folds, effective_sample_size,
    estimate_propensity_and_match, match_standard, match_standard_with_discard,
    match_weights_from_pairs, pair_weights_from_pairs, prepare_euclidean_distance_config,
    prepare_mahalanobis_distance_config, prepare_propensity_distance_config,
    prepare_propensity_distance_config_with_discard, reconstruct_case_control_pairs,
    subclassify_by_propensity_score_map, subclassify_with_set_aside_scores,
};
pub use matching::{
    BalanceCategory, BalanceLevel, BalanceMode, FineBalance, FineBalanceOutcome, match_fine_balance,
};
pub use matching::{
    BalanceMeasure, CardinalityError, CardinalityLimits, CardinalityOutcome, CardinalityRequest,
    GeneticError, GeneticFit, GeneticLoss, GeneticMatchOptions, GeneticMatches, GeneticOutcome,
    GeneticProblem, GeneticSearch, KsTest, LossRule, StratumReport, ks_limit_upper, ks_two_sample,
    match_cardinality, pair_units,
};
pub use matching::{
    CemBinning, CemError, CemKind, CemRule, CemStrata, CemVariable, cem_k2k_pairs, cem_strata,
    cem_subclasses,
};
pub use matching::{ExactMatchError, ExactMatchOutcome, exact_match_subclasses};
pub use matching::{
    FullMatchConfig, FullMatchOutcome, OptimalError, OptimalRequest, QuickMatchError,
    QuickMatchOptions, QuickMatchOutcome, Retain, SeedMethod, match_full, match_optimal,
    quick_clusters, quick_match_subclasses,
};
pub use role_transition::{
    AtRiskSpan, DefaultAtRiskRule, DefaultRiskSetPolicy, RiskSetPolicy, TransitionMatchRequest,
    match_transition,
};
pub use role_transition::{
    ExposedPerson, ExposureDensityControl, ExposureDensityError, ExposureDensityOutcome,
    exposure_density_sets, match_exposure_density,
};
pub use trials::{
    ControlReuse, ExpandedRow, PanelColumns, PeriodDates, PeriodStrata, TrialColumns,
    TrialInputError, TrialMatching, TrialMember, TrialOptions, TrialOutcome, TrialPeriod,
    TrialPerson, TrialPool, TrialRecord, TrialResult, TrialView, TrialsError, expand_trial_columns,
    expand_trials, match_trials, match_trials_with, matched_trial_columns, trial_seed,
};
pub use types::{
    AgeLimitYears, AnchorRecord, BalanceDiagnostics, BalanceRecord, BalanceRecordBuilder,
    BalanceReport, BalanceReportOptions, BalanceThresholdSummary, BalanceThresholds, BaseRecord,
    BirthDateWindowDays, CandidateRecord, CategoricalBalance, CategoricalLevelBalance,
    CommonSupport, CommonSupportFailureReason, ConstraintReason, ControlIdx, CovariateRef,
    CovariateValue, CriteriaValidationError, DistanceCaliper, DistanceCaliperReason, Estimand,
    EstimandDriftReason, ExclusionReason, InvalidCriteriaReason, MatchDiagnostics, MatchOutcome,
    MatchRatio, MatchedPair, MatchingCriteria, NumericBalance, NumericBalanceThresholdCheck,
    NumericBalanceTransform, RoleTransitionOptions, RoleTransitionRecord,
    SupplementalBalanceCovariates, TrimmedUnits, UniqueValueId, ValidatedMatchingCriteria,
};
pub use types::{
    BalanceImprovement, BalanceSummary, GroupSampleSizes, LevelImprovement, NumericImprovement,
    PairDistance, SampleSizeTable,
};

pub use output::{
    LongMatches, MatchData, MatchDataOptions, MatchMatrix, OutputError, Units, get_matches,
    get_matches_from_pairs, match_data, match_data_from_weights, match_matrix,
};
pub use spec::{
    AssignmentSpec, BinRuleSpec, BinningSpec, CardinalityMatch, CardinalitySpec, CemSpec,
    CemVariableSpec, CovarianceSpec, CovariatesSpec, DiscardSpec, DistanceSpec, EstimatorSpec,
    FitFuncSpec, FullSpec, GeneticSpec, K2kSpec, LambdaSpec, LinkSpec, LossSpec, MatchMethod,
    MatchMode, MatchSpec, MissingSpec, NearestSpec, OrderSpec, QuickSpec, ReferenceSpec, ScaleSpec,
    SpecError, SpecInput, SpecOutput, StdCaliperEach, StdCaliperSpec, SubclassSpec, TransformSpec,
    VariableRatioSpec, run_spec,
};

/// The two eCDF distance statistics of `case_values` against `control_values`, for benchmarks. `None` when either slice is empty.
#[cfg(feature = "bench-internals")]
#[must_use]
pub fn benchmark_ecdf_distance_stats(
    case_values: &[f64],
    control_values: &[f64],
) -> Option<(f64, f64)> {
    balance::benchmark_ecdf_distance_stats(case_values, control_values)
}

/// The two empirical QQ distance statistics of `case_values` against `control_values`, for benchmarks. `None` when either slice is empty.
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
        Assignment, DeterministicSelection, ExactMatchError, ExactMatchOutcome, FullMatchConfig,
        FullMatchOutcome, MatchOrder, NearestBirthDateSelection, NearestSelection, OptimalError,
        OptimalRequest, RandomSelection, Retain, Rounds, SelectionStrategy, exact_match_subclasses,
        match_full, match_optimal,
    };
    pub use crate::{
        AgeLimitYears, AnchorRecord, AssignmentSpec, AtRiskSpan, BalanceDiagnostics, BalanceRecord,
        BalanceRecordBuilder, BalanceReport, BalanceReportOptions, BalanceThresholdSummary,
        BalanceThresholds, BaseRecord, BirthDateWindowDays, CaliperConstraint, CandidateRecord,
        CardinalityMatch, CardinalitySpec, CategoricalBalance, CategoricalCovariateSpec,
        CategoricalLevelBalance, Chained, CommonSupport, CommonSupportFailureReason, Constraint,
        ConstraintContext, ConstraintGroup, ConstraintReason, ControlIdx, CovarianceSpec,
        CovariateEncodingConfig, CovariateMatrix, CovariateRecord, CovariateRef, CovariateValue,
        CovariatesSpec, CriteriaValidationError, CrossValidation, DateDistance,
        DefaultRiskSetPolicy, DistanceCaliper, DistanceCaliperReason, DistanceChannel,
        DistanceConfig, DistanceMetric, DistanceSpec, ElasticNetCvConfig, ElasticNetLogisticConfig,
        Estimand, EstimandDriftReason, EstimationError, EstimatorSpec,
        EuclideanDistancePreparation, EuclideanPreparationConfig, EuclideanScale,
        ExactMatchConstraint, ExclusionReason, FitFuncSpec, FullSpec, GeneticSpec,
        IdMapEuclideanDistance, IdMapMahalanobisDistance, IdMapPropensityScoreDistance,
        InvalidCriteriaReason, LambdaRule, LambdaSpec, LinkSpec, LogisticRegressionConfig,
        LongMatches, LossSpec, MahalanobisCovarianceStrategy, MahalanobisDistance,
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
        ReconstructedMatchingTier, ReferenceSpec, ResidentAtIndexRecord, RiskSetPolicy,
        RoleIndexedRecord, RoleTransitionOptions, RoleTransitionRecord, ScaleSpec, ScoreTable,
        SpecError, SpecInput, SpecOutput, StandardMatchRequest, StrataExactConstraint,
        SubclassReferenceGroup, SubclassSpec, SubclassSummary, SubclassificationConfig,
        SubclassificationError, SubclassificationOutcome, SupplementalBalanceCovariates,
        TransformSpec, TransitionMatchRequest, TrimmedUnits, UniqueValueId, UnitRole,
        UnitWeightRow, UnitWeightSet, UnitWeightTable, Units, ValidatedMatchingCriteria,
        VectorTable, balance_diagnostics, balance_report, balance_report_with_options,
        balance_threshold_summary, build_projected_balance_rows, categorical_max_abs_diff,
        cross_validation_folds, effective_sample_size, estimate_propensity_and_match, get_matches,
        get_matches_from_pairs, match_data, match_data_from_weights, match_matrix, match_standard,
        match_standard_with_discard, match_transition, match_weights_from_pairs,
        pair_weights_from_pairs, prepare_euclidean_distance_config,
        prepare_mahalanobis_distance_config, prepare_propensity_distance_config,
        prepare_propensity_distance_config_with_discard, reconstruct_case_control_pairs, run_spec,
        smd_numeric_from_values, subclassify_by_propensity_score_map,
        subclassify_with_set_aside_scores,
    };
    pub use crate::{
        BalanceImprovement, BalanceInputError, BalanceSummary, BalanceWeights, GroupSampleSizes,
        LevelImprovement, NumericImprovement, PairDistance, SampleSizeTable, balance_improvement,
        balance_report_on_weights, balance_summary,
    };
    pub use crate::{
        BinRuleSpec, BinningSpec, CemBinning, CemError, CemKind, CemRule, CemSpec, CemStrata,
        CemVariable, CemVariableSpec, K2kSpec, MatrixDistance, PairDistanceMatrix,
        PairDistanceMatrixError, cem_k2k_pairs, cem_strata, cem_subclasses,
    };
    pub use crate::{
        DiscardSpec, FineBalance, QuickSpec, RapidHashMap, StdCaliperEach, StdCaliperSpec,
        VariableRatio, VariableRatioError, VariableRatioSpec,
    };

    pub use crate::{
        ExposedPerson, ExposureDensityControl, ExposureDensityError, ExposureDensityOutcome,
        exposure_density_sets, match_exposure_density,
    };

    pub use crate::{
        ControlReuse, ExpandedRow, PeriodDates, PeriodStrata, TrialInputError, TrialMatching,
        TrialMember, TrialOptions, TrialOutcome, TrialPeriod, TrialPerson, TrialPool, TrialRecord,
        TrialResult, TrialView, TrialsError, expand_trials, match_trials, match_trials_with,
        trial_seed,
    };

    /// The constraints that can be added to a [`MatchJob`].
    pub mod constraints {
        pub use crate::constraints::{
            AntiExact, Caliper, DateWindow, GenderMatch, MissingPolicy, MustBeAlive,
            MustBeResident, MustBeResidentInSpells, ResidenceSpell, caliper_on_field,
            numeric_field_sd,
        };
    }
}
