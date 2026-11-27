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

mod diagnostics;
mod effects;
pub(crate) mod input;
mod likelihood;
mod posterior;
mod priors;
mod sampler;
mod types;

pub use crate::utils::{CalibrationBinSummary, EffectIntervalSummary};

pub use diagnostics::{
    PosteriorPredictiveSummary, autocorrelation, convergence, effective_sample_size,
    posterior_predictive_summary,
};
pub use effects::{
    CounterfactualEffects, CounterfactualEffectsSummary, CounterfactualScenario, PeriodEffect,
    PeriodEffectSummary, counterfactual_effects, counterfactual_effects_summary,
};
pub use posterior::{
    MtpPosteriorDraw, MtpPosteriorSamples, MtpPosteriorSummary, ParameterSummary,
    summarize_posterior,
};
pub use priors::MtpPriors;
pub use types::{
    FamilyRandomEffects, MtpAcceptanceRates, MtpConvergenceSummary, MtpModel, MtpOptions,
    MtpProposalTuning, MtpReport, MtpSamplerDiagnostics, PositivePartDistribution,
    RandomEffectsStructure,
};

#[cfg(feature = "bench-internals")]
pub use sampler::{benchmark_weighted_xtx, benchmark_weighted_xtz};

use crate::{Error, LongitudinalData};

/// Fit the marginalised two-part model by MCMC.
///
/// One chain draws from `options.seed`; several draw from
/// `epistat_rng::replicate_seed(options.seed, i)`, run on scoped threads, and
/// add the split-R-hat summary. The posterior summary is that of the draws of
/// every chain together.
///
/// # Errors
///
/// [`Error::Input`] for no chains or an iteration count, burn-in or thinning
/// that keeps no draw; [`Error::Invalid`] for priors or proposal tuning that
/// are not valid; [`Error::MissingFamilies`] for family effects without
/// family codes; [`Error::NoPositiveOutcome`]; the numerical errors of the
/// starting values.
pub fn fit(data: &LongitudinalData<'_>, options: &MtpOptions) -> Result<MtpModel, Error> {
    if options.chains == 0 {
        return Err(crate::error::domain("chains", 0.0, "at least 1"));
    }
    let config = options.sampler_config();
    if config.family_random_effects != FamilyRandomEffects::Disabled && data.family().is_none() {
        return Err(Error::MissingFamilies);
    }
    let input = data.to_owned_input();
    let (chains, posterior_summary, convergence, draws) = if options.chains == 1 {
        let (report, samples) = sampler::fit_single_chain(&input, config)?;
        let summary = report.posterior_summary.clone();
        (vec![report], summary, None, vec![samples])
    } else {
        let fit = sampler::fit_chains(&input, config, options.chains)?;
        (
            fit.chain_reports,
            fit.pooled_posterior_summary,
            Some(fit.convergence),
            fit.chain_posteriors,
        )
    };
    Ok(MtpModel {
        n_binary_covariates: data.x_binary().ncols(),
        n_mean_covariates: data.x_mean().ncols(),
        random_effect_dimension: options.random_effects.random_effect_dimension(),
        chains,
        posterior_summary,
        convergence,
        draws: options.keep_draws.then_some(draws),
    })
}
