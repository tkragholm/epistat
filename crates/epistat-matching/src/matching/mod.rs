mod cardinality;
mod cem;
mod codes;
mod constraints;
mod distance;
mod engine;
pub(crate) use engine::{StandardCommonSupport, standard_common_support, trimmed_ids};
mod estimation;
mod exact;
mod fine_balance;
mod genetic;
mod index;
pub mod job;
mod nearest;
mod optimal;
mod quick;
pub mod ratio;
mod reconstructed;
mod records;
mod selection;
mod subclassification;
mod weights;

pub use cardinality::{
    CardinalityError, CardinalityLimits, CardinalityOutcome, CardinalityRequest, StratumReport,
    match_cardinality, pair_units,
};
pub use cem::{
    CemBinning, CemError, CemKind, CemRule, CemStrata, CemVariable, cem_k2k_pairs, cem_strata,
    cem_subclasses,
};
pub use constraints::{
    CaliperConstraint, Chained, Constraint, ConstraintContext, ConstraintGroup,
    ExactMatchConstraint, StrataExactConstraint, UsedControlsVec,
};
pub use distance::{
    DateDistance, DistanceChannel, DistanceConfig, DistanceMetric, IdMapEuclideanDistance,
    IdMapMahalanobisDistance, IdMapPropensityScoreDistance, MahalanobisDistance, MahalanobisError,
    MatrixDistance, PairDistanceMatrix, PairDistanceMatrixError, PropensityScoreDistance,
    ScoreTable,
};
pub use engine::{
    DiscardedUnits, StandardMatchRequest, match_standard, match_standard_with_discard,
};
pub(crate) use estimation::encode_covariates_pooled;
pub use estimation::{
    CbpsConfig, CbpsEstimand, CbpsFit, CbpsMethod, CovariateEncodingConfig, CovariateMatrix,
    CrossValidation, ElasticNetCvConfig, ElasticNetLogisticConfig, EstimationError,
    EuclideanDistancePreparation, EuclideanPreparationConfig, EuclideanScale, LambdaRule,
    LogisticRegressionConfig, MahalanobisCovarianceStrategy, MahalanobisDistancePreparation,
    MahalanobisPreparationConfig, MahalanobisTransform, MissingValuePolicy,
    PropensityDistancePreparation, PropensityEstimator, PropensityLink, PropensityMatchedOutcome,
    PropensityScoreConfig, PropensityScoreEstimate, PropensityScoreOutputScale,
    cross_validation_folds, encode_covariates, estimate_propensity_and_match,
    prepare_euclidean_distance_config, prepare_mahalanobis_distance_config,
    prepare_propensity_distance_config, prepare_propensity_distance_config_with_discard,
};
pub use exact::{ExactMatchError, ExactMatchOutcome, SubclassedUnits, exact_match_subclasses};
pub use fine_balance::{
    BalanceCategory, BalanceLevel, BalanceMode, FineBalance, FineBalanceOutcome, match_fine_balance,
};
pub use genetic::{
    BalanceMeasure, GeneticError, GeneticFit, GeneticLoss, GeneticMatchOptions, GeneticMatches,
    GeneticOutcome, GeneticProblem, GeneticSearch, KsTest, LossRule, ks_limit_upper, ks_two_sample,
};
pub use nearest::{MatchOrder, NearestSelection, VariableRatio, VariableRatioError};
pub use optimal::{
    FullMatchConfig, FullMatchOutcome, OptimalError, OptimalRequest, Retain, match_full,
    match_optimal,
};
pub use quick::{
    QuickMatchError, QuickMatchOptions, QuickMatchOutcome, SeedMethod, apply_transposed,
    cholesky_lower, quick_clusters, quick_match_subclasses,
};
pub use reconstructed::{
    ReconstructedEpisode, ReconstructedMatchedPair, ReconstructedMatchingError,
    ReconstructedMatchingOptions, ReconstructedMatchingOutput, ReconstructedMatchingTier,
    reconstruct_case_control_pairs,
};
pub use records::{CovariateRecord, MatchingRecord, ResidentAtIndexRecord, RoleIndexedRecord};
pub use selection::{
    Assignment, DeterministicSelection, NearestBirthDateSelection, RandomSelection, Rounds,
    SelectionStrategy,
};
pub use subclassification::{
    SubclassReferenceGroup, SubclassSummary, SubclassificationConfig, SubclassificationError,
    SubclassificationOutcome, subclassify_by_propensity_score_map,
};
pub use weights::{
    InvalidWeightError, MatchWeightMethod, PairWeight, PairWeightRow, PairWeightSet,
    PairWeightTable, UnitRole, UnitWeightRow, UnitWeightSet, UnitWeightTable,
    effective_sample_size, match_weights_from_pairs, pair_weights_from_pairs,
};

pub use constraints::unique_value;
pub(crate) use constraints::{
    FieldValueSource, RefusesFromDate, ReplacementConstraint, ResidenceSpellSource, StatusRule,
    StatusRuleKind, build_strata_values, caliper_allows,
};
pub use engine::{build_outcome, invalid_cohort_outcome, invalid_criteria_outcome, to_f64};
pub(crate) use index::CandidateIndex;

pub use engine::CandidatePoolRequest;
pub use engine::{MatchEngine, finalize_estimand_diagnostics, invalid_common_support_outcome};
pub(crate) use engine::{id_prefix, intern_unique};
pub use job::MatchJob;
