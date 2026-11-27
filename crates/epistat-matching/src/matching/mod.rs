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
#[cfg(test)]
mod invariant_tests;
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
    CardinalityLimits, CardinalityOutcome, CardinalityRequest, StratumReport, match_cardinality,
    pair_units,
};
pub use cem::{
    CemBinning, CemKind, CemRule, CemStrata, CemVariable, cem_k2k_pairs, cem_strata, cem_subclasses,
};
pub(crate) use constraints::UsedControlsVec;
pub use constraints::{
    CaliperConstraint, Chained, Constraint, ConstraintContext, ConstraintGroup,
    ExactMatchConstraint,
};
pub use distance::{
    DateDistance, DistanceChannel, DistanceConfig, DistanceMetric, IdMapEuclideanDistance,
    IdMapMahalanobisDistance, IdMapPropensityScoreDistance, MahalanobisDistance, MatrixDistance,
    PairDistanceMatrix, PropensityScoreDistance, ScoreTable, VectorTable,
};
pub use engine::DiscardedUnits;
pub(crate) use engine::{StandardMatchRequest, match_standard};
pub(crate) use estimation::encode_covariates_pooled;
pub use estimation::{
    CbpsConfig, CbpsFit, CbpsMethod, CovariateEncodingConfig, CovariateMatrix, CrossValidation,
    ElasticNetCvConfig, ElasticNetLogisticConfig, EuclideanDistancePreparation,
    EuclideanPreparationConfig, EuclideanScale, LambdaRule, LogisticRegressionConfig,
    MahalanobisCovarianceStrategy, MahalanobisDistancePreparation, MahalanobisPreparationConfig,
    MahalanobisTransform, MissingValuePolicy, PropensityDistancePreparation, PropensityEstimator,
    PropensityLink, PropensityMatchedOutcome, PropensityScoreConfig, PropensityScoreEstimate,
    PropensityScoreOutputScale, cross_validation_folds, encode_covariates,
    estimate_propensity_and_match, prepare_euclidean_distance_config,
    prepare_mahalanobis_distance_config, prepare_propensity_distance_config,
};
pub use exact::{SubclassedUnits, exact_match_subclasses};
pub use fine_balance::{
    BalanceCategory, BalanceLevel, BalanceMode, FineBalance, FineBalanceOutcome, match_fine_balance,
};
pub use genetic::{
    BalanceMeasure, GeneticFit, GeneticLoss, GeneticMatchOptions, GeneticMatches, GeneticOutcome,
    GeneticProblem, GeneticSearch, LossRule,
};
pub use nearest::{MatchOrder, NearestSelection, VariableRatio};
pub use optimal::{FullMatchConfig, OptimalRequest, Retain, match_full, match_optimal};
pub use quick::{
    QuickMatchOptions, SeedMethod, apply_transposed, quick_clusters, quick_match_subclasses,
};
pub use reconstructed::{
    ReconstructedEpisode, ReconstructedMatchedPair, ReconstructedMatchingOptions,
    ReconstructedMatchingOutput, ReconstructedMatchingTier, reconstruct_case_control_pairs,
};
pub(crate) use records::{CachedSlot, ColumnReader, NameScan, SlotCache};
pub use records::{CovariateRecord, MatchingRecord, ResidentAtIndexRecord, RoleIndexedRecord};
pub use selection::{
    Assignment, DeterministicSelection, NearestBirthDateSelection, RandomSelection, Rounds,
    SelectionStrategy,
};
pub use subclassification::{
    SubclassReferenceGroup, SubclassSummary, SubclassificationConfig, SubclassificationOutcome,
    subclassify_by_propensity_score_map, subclassify_with_set_aside_scores,
};
#[expect(deprecated, reason = "the released path, kept for one release")]
pub use weights::effective_sample_size;
pub use weights::{
    MatchWeightMethod, PairWeight, PairWeightRow, PairWeightSet, PairWeightTable, UnitRole,
    UnitWeightRow, UnitWeightSet, UnitWeightTable, match_weights_from_pairs,
    pair_weights_from_pairs,
};

pub use constraints::unique_value;
pub(crate) use constraints::{
    FieldValueSource, RefusesFromDate, ReplacementConstraint, ResidenceSpellSource, StatusRule,
    StatusRuleKind, build_strata_values, caliper_allows,
};
pub use engine::{build_outcome, refused_outcome, to_f64};
pub(crate) use index::CandidateIndex;

pub use engine::CandidatePoolRequest;
pub use engine::{MatchEngine, finalize_estimand_diagnostics};
pub(crate) use engine::{id_keys, intern_unique};
pub use job::MatchJob;
