#![warn(missing_docs)]

//! Models for semicontinuous outcomes, non-negative with a mass at zero, such as
//! costs: the two-part model, the Tweedie GLM, log-normal regression with smearing,
//! and the marginalised two-part model for panels ([`models::mtp`]), with model
//! comparison and selection. The [`models`] modules explain each method.

/// Model inputs and their validation.
pub mod input;
/// The models, their fits, comparison and selection.
pub mod models;
/// Outcome counts and the removal of constant design columns.
pub mod preprocess;
pub(crate) mod utils;

pub use input::{
    CounterfactualInputBuildError, InputError, LongitudinalInputError, LongitudinalModelInput,
    ModelInput, ModelInputRef, PreparedCounterfactualInput, TWO_PART_NOT_APPLICABLE_POSITIVE_SHARE,
    build_counterfactual_input, sanitize_model_outcome, two_part_applicability,
};
pub use preprocess::{
    OutcomeDiagnostics, column_has_variation, drop_constant_columns, outcome_diagnostics,
};

pub use crate::models::{FitMetadata, Model, SolverKind};

pub use models::two_part::{
    BootstrapOptions, BootstrapResult, BootstrapSummary, ConfidenceInterval, FitOptions,
    Regularization, TwoPartError, TwoPartModel, TwoPartPrediction, TwoPartReport, TwoPartTrainer,
    TwoPartWarmStart, bootstrap as bootstrap_two_part, bootstrap_percentile_ci, bootstrap_summary,
    coefficient_confidence_intervals, fit_two_part_clustered_input, fit_two_part_input,
    fit_two_part_input_warm, fit_two_part_ref_warm, fit_two_part_ref_warm_prevalidated,
    fit_two_part_weighted_input, log_likelihood as two_part_log_likelihood,
};

pub use models::tweedie::{
    TweedieError, TweedieModel, TweedieOptions, TweediePrediction, TweedieReport, TweedieScratch,
    TweedieTrainer, deviance as tweedie_deviance, fit_tweedie_input, fit_tweedie_input_warm,
    fit_tweedie_ref_warm, fit_tweedie_ref_warm_prevalidated,
    fit_tweedie_ref_warm_prevalidated_with_scratch,
    quasi_log_likelihood as tweedie_quasi_log_likelihood,
};

pub use models::prepared::{
    PreparedFitError, PreparedGammaDesign, PreparedGammaFit, PreparedPredictionDesign,
    PreparedScratch, PreparedTwoPartDesign, PreparedTwoPartFit,
};

pub use models::lognormal::{
    LogNormalError, LogNormalModel, LogNormalOptions, LogNormalPrediction, LogNormalReport,
    LogNormalTrainer, fit_lognormal_smearing_input, log_likelihood as lognormal_log_likelihood,
};

pub use models::selection::{
    CrossValidationError, CrossValidationOptions, CrossValidationResult, InformationCriteria,
    LogNormalCandidate, ModelFitMetrics, ParkTestResult, SelectionResult, TweedieCandidate,
    TweedieCvCandidate, classic_park_test, compute_information_criteria, compute_model_fit_metrics,
    cross_validate_models_input, fit_tweedie_candidates_input, recommend_from_cv,
    select_models_input,
};

pub use models::comparison::{
    FittedComparisonModels, ModelComparison, ModelComparisonError, ModelComparisonOptions,
    ModelInformationCriteria, ModelScore, TweedieRankingRow, compare_models_input,
    compare_models_input_with_fits,
};

pub use models::mtp::{
    CalibrationBinSummary, CounterfactualEffects, CounterfactualEffectsSummary,
    CounterfactualScenario, EffectIntervalSummary, FamilyRandomEffects, MtpAcceptanceRates,
    MtpConvergenceSummary, MtpError, MtpFitOptions, MtpModel, MtpMultiChainOptions,
    MtpMultiChainReport, MtpPosteriorDraw, MtpPosteriorSamples, MtpPosteriorSummary,
    MtpPriorConfig, MtpProposalTuning, MtpReport, MtpSamplerConfig, MtpSamplerDiagnostics,
    ParameterSummary, PeriodEffect, PeriodEffectSummary, PositivePartDistribution,
    PosteriorPredictiveSummary, RandomEffectsStructure, autocorrelation,
    compute_counterfactual_effects, compute_counterfactual_effects_summary, effective_sample_size,
    fit_mtp_input, fit_mtp_input_multi_chain, fit_mtp_input_multi_chain_with_config,
    fit_mtp_input_multi_chain_with_posterior, fit_mtp_input_multi_chain_with_posterior_config,
    fit_mtp_input_with_config, fit_mtp_input_with_posterior, fit_mtp_input_with_posterior_config,
    posterior_predictive_summary, summarize_multi_chain_convergence, summarize_posterior,
};
