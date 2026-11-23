//! The marginalised two-part model for person-period panels, fitted by MCMC.
//!
//! A logit gives the probability of a positive outcome and a log link the marginal
//! mean over zeros and positives, each with subject random intercepts and, optionally,
//! time slopes, correlated across the two parts, and optional family intercepts. The
//! positive part is log-skew-normal or log-normal. The posterior draws give period
//! effects of a counterfactual exposure, posterior predictive checks and split R-hat
//! over several chains \[1, 2\].
//!
//! # Algorithm
//!
//! Metropolis-Hastings in blocks: the fixed effects and the scale of the positive part
//! together, the skewness, the subject effects, the family effects and their
//! covariance. Proposal scales adapt during burn-in only, by a Robbins-Monro rule on
//! the log step size towards an acceptance band, so the retained draws come from a
//! fixed kernel. Chain `i` draws from `epistat_rng::replicate_seed(seed, i)`.
//!
//! # References
//!
//! 1. Smith VA, Preisser JS, Neelon B, Maciejewski ML. A marginalized two-part model
//!    for semicontinuous data. Stat Med. 2014;33:4891-903.
//! 2. Smith VA, Neelon B, Preisser JS, Maciejewski ML. A marginalized two-part model
//!    for longitudinal semicontinuous data. Stat Methods Med Res. 2017;26:1949-68.

pub mod diagnostics;
pub mod effects;
pub mod input;
pub mod likelihood;
pub mod posterior;
pub mod priors;
pub mod sampler;
pub mod types;

pub use crate::utils::{CalibrationBinSummary, EffectIntervalSummary};

pub use diagnostics::{
    PosteriorPredictiveSummary, autocorrelation, effective_sample_size,
    posterior_predictive_summary, summarize_multi_chain_convergence,
};
pub use effects::{
    CounterfactualEffects, CounterfactualEffectsSummary, CounterfactualScenario, PeriodEffect,
    PeriodEffectSummary, compute_counterfactual_effects, compute_counterfactual_effects_summary,
};
pub use posterior::{
    MtpPosteriorDraw, MtpPosteriorSamples, MtpPosteriorSummary, ParameterSummary,
    summarize_posterior,
};
pub use priors::MtpPriorConfig;
pub use sampler::{
    fit_mtp_input, fit_mtp_input_multi_chain, fit_mtp_input_multi_chain_with_config,
    fit_mtp_input_multi_chain_with_posterior, fit_mtp_input_multi_chain_with_posterior_config,
    fit_mtp_input_with_config, fit_mtp_input_with_posterior, fit_mtp_input_with_posterior_config,
};
pub use types::{
    FamilyRandomEffects, MtpAcceptanceRates, MtpConvergenceSummary, MtpError, MtpFitOptions,
    MtpModel, MtpMultiChainOptions, MtpMultiChainReport, MtpProposalTuning, MtpReport,
    MtpSamplerConfig, MtpSamplerDiagnostics, PositivePartDistribution, RandomEffectsStructure,
};
