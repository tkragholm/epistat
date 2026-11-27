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
//! let outcome = MatchJob::new_standard(&cases, &controls, 42)
//!     .with_ratio(MatchRatio::new(4).expect("non-zero ratio"))
//!     .with_birth_window(BirthDateWindowDays::new(30).expect("non-negative birth window"))
//!     .with_gender_match()
//!     .run()?;
//! ```
//!
//! A job returns [`Error`] for input it cannot use: two units with one id, a
//! case whose event is before its birth, a caliper window that is negative or
//! not finite, criteria that contradict each other. A case that finds no
//! control is not an error; the outcome counts it.
//!
//! [`MatchSpec`] describes any of the methods as plain values with serde, and
//! [`run_spec`] checks it and runs it. It suits a caller that reads its
//! options at run time, such as a binding to R or Python. An option that a
//! method does not read, a score that is missing or not finite, and any other
//! invalid option are errors that name the option or the unit.
//!
//! ## Large cohorts
//!
//! The units are cases and controls throughout: the pairs, the weights, the
//! diagnostics and the outputs use the two words, whatever the design calls
//! them (treated and untreated, exposed and unexposed).
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

/// Implements every method of [`MatchingRecord`] by forwarding it to an inner
/// record: to the field `$field` with `delegate_matching_record!(field)`, or
/// to the record behind a reference or a box with `delegate_matching_record!(*)`.
///
/// A wrapper that adds data of its own to a record, such as the flags a
/// risk-set policy reads, implements the trait with this macro, so that a
/// method the wrapper does not write by hand reads the inner record rather
/// than the trait's default. Forwarding by hand misses a method silently: a
/// forgotten [`MatchingRecord::numeric`] refuses every pair on a caliper, and
/// a forgotten [`MatchingRecord::entry_date`] ignores the entry date. The
/// crate's tests check that the macro forwards every method of the trait.
///
/// ```
/// use chrono::NaiveDate;
/// use epistat_matching::{BaseRecord, MatchingRecord, delegate_matching_record};
///
/// struct Person {
///     base: BaseRecord,
///     exposed: bool,
/// }
///
/// impl MatchingRecord for Person {
///     delegate_matching_record!(base);
/// }
///
/// let birth = NaiveDate::from_ymd_opt(2010, 1, 1).unwrap();
/// let mut base = BaseRecord::new("p1", birth);
/// base.strata.insert("sex".into(), "F".into());
/// let person = Person { base, exposed: true };
/// assert_eq!(person.stratum("sex"), Some("F"));
/// assert!(person.exposed);
/// ```
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
mod error;
pub use error::Error;
pub mod imputation;
mod kdtree;
mod matching;
pub mod multigroup;
pub mod output;
mod overlap;
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

pub use cohort::{Cohort, CohortBuilder, CohortRow};

pub use balance::{
    BalanceWeights, balance_diagnostics, balance_improvement, balance_report, balance_summary,
    balance_threshold_summary,
};
#[expect(
    deprecated,
    reason = "the released path of the effective sample size, kept for one release"
)]
pub use matching::effective_sample_size;
pub use matching::exact_match_subclasses;
pub use matching::{
    Assignment, CbpsConfig, CbpsFit, CbpsMethod, CovariateEncodingConfig, CovariateMatrix,
    CovariateRecord, CrossValidation, DateDistance, DeterministicSelection, DiscardedUnits,
    DistanceChannel, DistanceConfig, DistanceMetric, ElasticNetCvConfig, ElasticNetLogisticConfig,
    EuclideanDistancePreparation, EuclideanPreparationConfig, EuclideanScale,
    IdMapEuclideanDistance, IdMapMahalanobisDistance, IdMapPropensityScoreDistance, LambdaRule,
    LogisticRegressionConfig, MahalanobisCovarianceStrategy, MahalanobisDistance,
    MahalanobisDistancePreparation, MahalanobisPreparationConfig, MahalanobisTransform, MatchJob,
    MatchOrder, MatchWeightMethod, MatchingRecord, MatrixDistance, MissingValuePolicy,
    NearestBirthDateSelection, NearestSelection, PairDistanceMatrix, PairWeight, PairWeightRow,
    PairWeightSet, PairWeightTable, PropensityDistancePreparation, PropensityEstimator,
    PropensityLink, PropensityMatchedOutcome, PropensityScoreConfig, PropensityScoreDistance,
    PropensityScoreEstimate, PropensityScoreOutputScale, RandomSelection, ReconstructedEpisode,
    ReconstructedMatchedPair, ReconstructedMatchingOptions, ReconstructedMatchingOutput,
    ReconstructedMatchingTier, ResidentAtIndexRecord, RoleIndexedRecord, Rounds, ScoreTable,
    SelectionStrategy, SubclassReferenceGroup, SubclassSummary, SubclassedUnits,
    SubclassificationConfig, SubclassificationOutcome, UnitRole, UnitWeightRow, UnitWeightSet,
    UnitWeightTable, VariableRatio, VectorTable, cross_validation_folds,
    estimate_propensity_and_match, match_weights_from_pairs, pair_weights_from_pairs,
    prepare_euclidean_distance_config, prepare_mahalanobis_distance_config,
    prepare_propensity_distance_config, reconstruct_case_control_pairs,
    subclassify_by_propensity_score_map, subclassify_with_set_aside_scores,
};
pub use matching::{
    BalanceCategory, BalanceLevel, BalanceMode, FineBalance, FineBalanceOutcome, match_fine_balance,
};
pub use matching::{
    BalanceMeasure, CardinalityLimits, CardinalityOutcome, CardinalityRequest, GeneticFit,
    GeneticLoss, GeneticMatchOptions, GeneticMatches, GeneticOutcome, GeneticProblem,
    GeneticSearch, LossRule, StratumReport, match_cardinality, pair_units,
};
pub use matching::{
    CemBinning, CemKind, CemRule, CemStrata, CemVariable, cem_k2k_pairs, cem_strata, cem_subclasses,
};
pub use matching::{
    FullMatchConfig, OptimalRequest, QuickMatchOptions, Retain, SeedMethod, match_full,
    match_optimal, quick_clusters, quick_match_subclasses,
};
pub use role_transition::{AtRiskSpan, DefaultAtRiskRule, DefaultRiskSetPolicy, RiskSetPolicy};
pub use role_transition::{
    ExposedPerson, ExposureDensityControl, ExposureDensityOutcome, exposure_density_sets,
};
pub use trials::{
    ControlReuse, PeriodDates, PeriodStrata, TrialColumns, TrialMatching, TrialMember,
    TrialOptions, TrialOutcome, TrialPanel, TrialPanelBuilder, TrialPool, TrialRecord, TrialResult,
    TrialView, expand_trial_columns, match_trials, matched_trial_columns, trial_seed,
};
pub use types::{
    AgeLimitYears, BalanceDiagnostics, BalanceRecord, BalanceRecordBuilder, BalanceReport,
    BalanceReportOptions, BalanceThresholdSummary, BalanceThresholds, BaseRecord,
    BirthDateWindowDays, CategoricalBalance, CategoricalLevelBalance, CommonSupport,
    CommonSupportFailureReason, ConstraintReason, ControlIdx, CovariateRef, CovariateValue,
    DistanceCaliper, DistanceCaliperReason, Estimand, EstimandDriftReason, ExclusionReason,
    InvalidCriteriaReason, MatchDiagnostics, MatchOutcome, MatchRatio, MatchedPair,
    MatchingCriteria, NumericBalance, NumericBalanceThresholdCheck, NumericBalanceTransform,
    RoleTransitionOptions, RoleTransitionRecord, SupplementalBalanceCovariates, TrimmedUnits,
    UniqueValueId, ValidatedMatchingCriteria,
};
pub use types::{
    BalanceImprovement, BalanceSummary, GroupSampleSizes, LevelImprovement, NumericImprovement,
    PairDistance, SampleSizeTable,
};

pub use spec::{MatchSpec, SpecInput, SpecOutput, run_spec};

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

/// The names a first matching job needs: the builder, the record types, the
/// typed options of a risk-set job, the outcome and the error, and the spec
/// with its runner for options read at run time.
///
/// Everything else is imported from the crate root or from its modules
/// (`constraints`, `output`, `spec`, `effects`, `sensitivity`).
pub mod prelude {
    pub use crate::{
        AgeLimitYears, BaseRecord, BirthDateWindowDays, Error, MatchJob, MatchOutcome, MatchRatio,
        MatchSpec, MatchedPair, MatchingRecord, RoleIndexedRecord, RoleTransitionRecord, SpecInput,
        run_spec,
    };
}
