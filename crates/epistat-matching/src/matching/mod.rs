mod cem;
mod constraints;
mod distance;
mod engine;
mod estimation;
mod exact;
mod index;
pub mod job;
mod nearest;
mod quick;
pub mod ratio;
mod reconstructed;
mod records;
mod selection;
mod subclassification;
mod weights;

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
};
pub use engine::{
    DiscardedUnits, StandardMatchRequest, match_standard, match_standard_with_discard,
};
pub use estimation::{
    CovariateEncodingConfig, CovariateMatrix, CrossValidation, ElasticNetCvConfig,
    ElasticNetLogisticConfig, EstimationError, EuclideanDistancePreparation,
    EuclideanPreparationConfig, EuclideanScale, LambdaRule, LogisticRegressionConfig,
    MahalanobisCovarianceStrategy, MahalanobisDistancePreparation, MahalanobisPreparationConfig,
    MahalanobisTransform, MissingValuePolicy, PropensityDistancePreparation, PropensityEstimator,
    PropensityLink, PropensityMatchedOutcome, PropensityScoreConfig, PropensityScoreEstimate,
    PropensityScoreOutputScale, RidgeReport, cross_validation_folds, estimate_propensity_and_match,
    prepare_euclidean_distance_config, prepare_mahalanobis_distance_config,
    prepare_propensity_distance_config, prepare_propensity_distance_config_with_discard,
};
pub use exact::{ExactMatchError, ExactMatchOutcome, exact_match_subclasses};
pub use nearest::{MatchOrder, NearestSelection, VariableRatio, VariableRatioError};
pub use quick::{
    QuickMatchError, QuickMatchOptions, QuickMatchOutcome, SeedMethod, apply_transposed,
    cholesky_lower, quick_clusters, quick_match_subclasses,
};
pub use reconstructed::{
    ReconstructedEpisode, ReconstructedMatchedPair, ReconstructedMatchingOptions,
    ReconstructedMatchingOutput, ReconstructedMatchingTier, reconstruct_case_control_pairs,
};
pub use records::{CovariateRecord, MatchingRecord, ResidentAtIndexRecord, RoleIndexedRecord};
pub use selection::{
    Assignment, DeterministicSelection, NearestBirthDateSelection, RandomSelection, Rounds,
    SelectionStrategy,
};
pub use subclassification::{
    SubclassReferenceGroup, SubclassSummary, SubclassificationConfig, SubclassificationOutcome,
    subclassify_by_propensity_score_map,
};
pub use weights::{
    MatchWeightMethod, PairWeight, PairWeightRow, PairWeightSet, PairWeightTable, UnitRole,
    UnitWeightRow, UnitWeightSet, UnitWeightTable, effective_sample_size, match_weights_from_pairs,
    pair_weights_from_pairs,
};

pub use constraints::unique_value;
pub use engine::{build_outcome, invalid_criteria_outcome, to_f64};

pub use engine::CandidatePoolRequest;
pub(crate) use engine::intern_unique;
pub use engine::{MatchEngine, finalize_estimand_diagnostics, invalid_common_support_outcome};
pub use job::MatchJob;
