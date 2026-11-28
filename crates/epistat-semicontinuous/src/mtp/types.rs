//! Core public types for the MTP module.

use super::posterior::MtpPosteriorSummary;
use super::priors::MtpPriors;
use crate::Error;
use crate::fit::FitMetadata;
use epistat_core::Seed;

/// At least `min` chains.
pub(crate) fn validate_chain_count(found: usize, min: usize) -> Result<(), Error> {
    if found < min {
        return Err(crate::error::domain("chains", found as f64, "at least 2"));
    }
    Ok(())
}

/// Random-effects structure used by the longitudinal MTP model.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RandomEffectsStructure {
    /// Subject-level random intercepts for binary and mean parts.
    InterceptsOnly,
    /// Subject-level random intercepts and linear time slopes for both parts.
    InterceptsAndTimeSlopes,
}

impl RandomEffectsStructure {
    /// Number of random-effect dimensions per subject implied by the structure.
    #[must_use]
    pub const fn random_effect_dimension(self) -> usize {
        match self {
            Self::InterceptsOnly => 2,
            Self::InterceptsAndTimeSlopes => 4,
        }
    }
}

/// Positive-part distribution for outcomes where `y > 0`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PositivePartDistribution {
    /// Log-skew-normal positive part with skewness parameter `kappa`.
    #[default]
    LogSkewNormal,
    /// Log-normal positive part (equivalent to skew-normal with fixed `kappa = 0`).
    LogNormal,
}

/// Optional family-level random-effects layer.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FamilyRandomEffects {
    /// No family-level random effects.
    #[default]
    Disabled,
    /// Family-level random intercepts for binary and mean components.
    InterceptsOnly,
}

/// The iteration controls of one chain.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MtpFitOptions {
    pub iterations: usize,
    pub burn_in: usize,
    pub thin: usize,
    pub seed: u64,
    pub random_effects: RandomEffectsStructure,
    pub adapt_during_burn_in: bool,
}

impl MtpFitOptions {
    pub(crate) fn validate(self) -> Result<(), Error> {
        if self.iterations == 0 {
            return Err(crate::error::domain("iterations", 0.0, "at least 1"));
        }
        if self.burn_in >= self.iterations {
            return Err(crate::error::domain(
                "burn_in",
                self.burn_in as f64,
                "below the iterations",
            ));
        }
        if self.thin == 0 {
            return Err(crate::error::domain("thin", 0.0, "at least 1"));
        }
        Ok(())
    }

    /// Number of retained draws implied by `(iterations, burn_in, thin)`.
    pub(crate) const fn retained_draws(self) -> usize {
        crate::utils::retained_draws(self.iterations, self.burn_in, self.thin)
    }
}

/// Validate the posterior draw dimensions used by MTP summary calculations.
pub(super) fn validate_draw_dimensions(
    alpha_len: usize,
    beta_len: usize,
    draw_alpha_len: usize,
    draw_beta_len: usize,
) -> Result<(), Error> {
    if draw_alpha_len != alpha_len {
        return Err(Error::Input(epistat_core::InputError::Length {
            what: "coefficients",
            expected: alpha_len,
            found: draw_alpha_len,
        }));
    }
    if draw_beta_len != beta_len {
        return Err(Error::Input(epistat_core::InputError::Length {
            what: "coefficients",
            expected: beta_len,
            found: draw_beta_len,
        }));
    }
    Ok(())
}

/// Proposal-scale and adaptation controls for MTP MCMC.
///
/// `alpha`, `beta` and `log(omega_sq)` are updated as one block whose proposal
/// covariance is estimated from the chain during burn-in. `kappa` has its own random
/// walk, which holds the variance of the log positive part fixed. A subject's
/// random-effect step is a multiple of the Cholesky factor of the current random-effects
/// covariance.
///
/// Every `adaptation_interval` burn-in iterations each step size `s` takes a
/// Robbins-Monro step on the log scale towards the middle of the acceptance band,
/// `log(s) += adaptation_gain / sqrt(k) * (rate - target)`, where `rate` is its
/// acceptance rate over the interval and `k` counts its adaptations (Andrieu and Thoms
/// 2008). The step is not capped, so the gains sum without bound and any starting scale
/// can be corrected given enough intervals. All adaptation stops at the end of burn-in.
///
/// Andrieu, C. and Thoms, J. (2008). A tutorial on adaptive MCMC. Statistics and
/// Computing 18(4), 343-373.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MtpProposalTuning {
    /// Minimum allowed proposal scale.
    pub min_draw_scale: f64,
    /// Initial random-walk scale for `kappa`.
    pub kappa_draw_scale: f64,
    /// Initial proposal standard deviation of `log(omega_sq)` in the fixed-effect block,
    /// used until the block covariance is estimated from the chain.
    pub log_omega_draw_scale: f64,
    /// Adapt every `adaptation_interval` iterations during burn-in.
    pub adaptation_interval: usize,
    /// Lower end of the acceptance band; adaptation targets the band's midpoint.
    pub acceptance_target_low: f64,
    /// Upper end of the acceptance band; adaptation targets the band's midpoint.
    pub acceptance_target_high: f64,
    /// Gain `c` of the Robbins-Monro step `c / sqrt(k)` on the log step size.
    pub adaptation_gain: f64,
    /// Initial random-walk scale for family-level random effects.
    pub family_effect_draw_scale: f64,
}

impl Default for MtpProposalTuning {
    fn default() -> Self {
        Self {
            min_draw_scale: 1.0e-3,
            kappa_draw_scale: 0.05,
            log_omega_draw_scale: 0.03,
            adaptation_interval: 50,
            acceptance_target_low: 0.2,
            acceptance_target_high: 0.35,
            adaptation_gain: 3.0,
            family_effect_draw_scale: 0.05,
        }
    }
}

impl MtpProposalTuning {
    /// Whether proposal tuning settings are numerically valid.
    #[must_use]
    pub(crate) fn is_valid(self) -> bool {
        self.min_draw_scale > 0.0
            && self.kappa_draw_scale > 0.0
            && self.log_omega_draw_scale > 0.0
            && self.adaptation_interval > 0
            && self.acceptance_target_low >= 0.0
            && self.acceptance_target_high <= 1.0
            && self.acceptance_target_low < self.acceptance_target_high
            && self.adaptation_gain > 0.0
            && self.family_effect_draw_scale > 0.0
    }
}

/// The options of [`fit`](super::fit).
///
/// No default: the seed of the chains is the caller's. [`MtpOptions::new`]
/// gives 4,000 iterations with a burn-in of 1,000 and every fourth draw kept,
/// random intercepts only, a log-skew-normal positive part, no family effects
/// and four chains.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MtpOptions {
    /// MCMC iterations of each chain, burn-in included.
    pub iterations: usize,
    /// Burn-in iterations discarded before draws are kept.
    pub burn_in: usize,
    /// Keep every `thin`-th draw after burn-in.
    pub thin: usize,
    /// The root seed. A single chain draws from it; chain `i` of several
    /// draws from `epistat_rng::replicate_seed(seed, i)`, so a chain does
    /// not depend on the other chains or on the thread that runs it.
    pub seed: Seed,
    /// The subject random effects.
    pub random_effects: RandomEffectsStructure,
    /// Whether the proposal scales adapt during burn-in.
    pub adapt_during_burn_in: bool,
    /// The priors.
    pub priors: MtpPriors,
    /// The proposal scales and their adaptation.
    pub proposal: MtpProposalTuning,
    /// The distribution of the positive part.
    pub positive_part: PositivePartDistribution,
    /// Whether to add family-level random intercepts.
    pub family_random_effects: FamilyRandomEffects,
    /// The number of chains. With one chain there is no convergence summary.
    pub chains: usize,
    /// Whether the fit keeps every chain's draws.
    pub keep_draws: bool,
}

impl MtpOptions {
    /// The options described on [`MtpOptions`], with root seed `seed`.
    #[must_use]
    pub fn new(seed: Seed) -> Self {
        Self {
            iterations: 4_000,
            burn_in: 1_000,
            thin: 4,
            seed,
            random_effects: RandomEffectsStructure::InterceptsOnly,
            adapt_during_burn_in: true,
            priors: MtpPriors::default(),
            proposal: MtpProposalTuning::default(),
            positive_part: PositivePartDistribution::LogSkewNormal,
            family_random_effects: FamilyRandomEffects::Disabled,
            chains: 4,
            keep_draws: false,
        }
    }

    /// The number of draws each chain keeps.
    #[must_use]
    pub const fn retained_draws(&self) -> usize {
        crate::utils::retained_draws(self.iterations, self.burn_in, self.thin)
    }

    pub(crate) const fn sampler_config(&self) -> MtpSamplerConfig {
        MtpSamplerConfig {
            fit_options: MtpFitOptions {
                iterations: self.iterations,
                burn_in: self.burn_in,
                thin: self.thin,
                seed: self.seed.value(),
                random_effects: self.random_effects,
                adapt_during_burn_in: self.adapt_during_burn_in,
            },
            prior_config: self.priors,
            proposal_tuning: self.proposal,
            positive_part_distribution: self.positive_part,
            family_random_effects: self.family_random_effects,
        }
    }
}

/// The default iteration controls, for the tests of the sampler.
#[cfg(test)]
pub(crate) fn test_fit_options() -> MtpFitOptions {
    test_config().fit_options
}

/// The sampler's options of [`MtpOptions::new`] with seed 42, for the tests
/// of the sampler.
#[cfg(test)]
pub(crate) fn test_config() -> MtpSamplerConfig {
    MtpOptions::new(Seed(42)).sampler_config()
}

/// The sampler's options.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MtpSamplerConfig {
    pub fit_options: MtpFitOptions,
    pub prior_config: MtpPriors,
    pub proposal_tuning: MtpProposalTuning,
    pub positive_part_distribution: PositivePartDistribution,
    pub family_random_effects: FamilyRandomEffects,
}

impl MtpSamplerConfig {
    pub(crate) fn validate(self) -> Result<(), Error> {
        self.fit_options.validate()?;
        if !self.prior_config.is_valid() {
            return Err(Error::Invalid {
                what: "the priors: a variance, shape, scale or degrees of freedom is not positive, or the kappa bounds are not increasing",
            });
        }
        if !self.proposal_tuning.is_valid() {
            return Err(Error::Invalid {
                what: "the proposal tuning: a scale, interval or gain is not positive, or the acceptance band is not inside [0, 1]",
            });
        }
        Ok(())
    }
}

/// The fitted MTP model: its dimensions, the report of each chain, and the
/// posterior summary over all chains.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct MtpModel {
    /// Number of covariates in the binary design matrix.
    pub n_binary_covariates: usize,
    /// Number of covariates in the marginal-mean design matrix.
    pub n_mean_covariates: usize,
    /// Random-effects dimensionality per subject.
    pub random_effect_dimension: usize,
    /// The report of each chain, in chain order.
    pub chains: Vec<MtpReport>,
    /// The posterior summary of the draws of all chains, `None` when no draw
    /// was kept.
    pub posterior_summary: Option<MtpPosteriorSummary>,
    /// Split-R-hat over the chains, `None` for one chain.
    pub convergence: Option<MtpConvergenceSummary>,
    /// Every chain's draws, when the options keep them.
    pub draws: Option<Vec<super::posterior::MtpPosteriorSamples>>,
}

/// Component-wise acceptance rates from the MCMC sampler, over all iterations.
///
/// `alpha`, `beta` and `omega_sq` are proposed together, so the three rates are equal.
/// A rate is `None` when its component had no proposals.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, Default)]
pub struct MtpAcceptanceRates {
    /// Acceptance rate of the binary-part coefficients.
    pub alpha: Option<f64>,
    /// Acceptance rate of the mean coefficients.
    pub beta: Option<f64>,
    /// Acceptance rate of the subject effects.
    pub random_effects: Option<f64>,
    /// Acceptance rate of the family effects.
    pub family_effects: Option<f64>,
    /// Acceptance rate of kappa.
    pub kappa: Option<f64>,
    /// Acceptance rate of omega squared.
    pub omega_sq: Option<f64>,
}

/// Sampler diagnostics summary.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct MtpSamplerDiagnostics {
    /// Iterations run.
    pub iterations_completed: usize,
    /// Draws kept after burn-in and thinning.
    pub retained_draws: usize,
    /// Acceptance rates over all iterations.
    pub acceptance_rates: Option<MtpAcceptanceRates>,
}

/// The report of one chain.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct MtpReport {
    /// Standardized fit metadata.
    pub meta: FitMetadata,
    /// Sampler diagnostics.
    pub diagnostics: MtpSamplerDiagnostics,
    /// Summaries of the retained draws.
    pub posterior_summary: Option<MtpPosteriorSummary>,
}

/// Multi-chain split-R-hat diagnostics summary.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct MtpConvergenceSummary {
    /// Number of chains included.
    pub chain_count: usize,
    /// Draws per chain used after truncation to equal even length.
    pub draws_per_chain_used: usize,
    /// Split-R-hat for each alpha coefficient, `None` where a half-chain is constant.
    pub alpha_split_rhat: Vec<Option<f64>>,
    /// Split-R-hat for each beta coefficient, `None` where a half-chain is constant.
    pub beta_split_rhat: Vec<Option<f64>>,
    /// Split-R-hat for kappa.
    pub kappa_split_rhat: Option<f64>,
    /// Split-R-hat for omega squared.
    pub omega_sq_split_rhat: Option<f64>,
    /// Maximum split-R-hat across all tracked scalar parameters, `None` when any of
    /// them is undefined.
    pub max_split_rhat: Option<f64>,
}

impl MtpConvergenceSummary {
    /// Whether every tracked parameter has a defined split-R-hat below `threshold`.
    /// A parameter whose chains are constant is not converged.
    #[must_use]
    pub fn is_converged(&self, threshold: f64) -> bool {
        self.max_split_rhat.is_some_and(|max| max < threshold)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn options_validate_and_retained_draws() {
        let options = MtpOptions {
            iterations: 100,
            burn_in: 20,
            ..MtpOptions::new(Seed(1))
        };
        assert!(options.sampler_config().validate().is_ok());
        assert_eq!(options.retained_draws(), 20);
        let bad = MtpOptions {
            burn_in: 100,
            ..options
        };
        assert!(matches!(
            bad.sampler_config().validate(),
            Err(Error::Input(_))
        ));
    }

    #[test]
    fn random_effect_dimension_matches_structure() {
        assert_eq!(
            RandomEffectsStructure::InterceptsOnly.random_effect_dimension(),
            2
        );
        assert_eq!(
            RandomEffectsStructure::InterceptsAndTimeSlopes.random_effect_dimension(),
            4
        );
    }

    #[test]
    fn proposal_tuning_defaults_are_valid() {
        assert!(MtpProposalTuning::default().is_valid());
    }

    #[test]
    fn sampler_config_rejects_invalid_priors_and_tuning() {
        let priors = MtpOptions {
            priors: MtpPriors {
                alpha_variance: 0.0,
                ..MtpPriors::default()
            },
            ..MtpOptions::new(Seed(1))
        };
        assert!(matches!(
            priors.sampler_config().validate(),
            Err(Error::Invalid { .. })
        ));
        let tuning = MtpOptions {
            proposal: MtpProposalTuning {
                adaptation_interval: 0,
                ..MtpProposalTuning::default()
            },
            ..MtpOptions::new(Seed(1))
        };
        assert!(matches!(
            tuning.sampler_config().validate(),
            Err(Error::Invalid { .. })
        ));
    }
}
