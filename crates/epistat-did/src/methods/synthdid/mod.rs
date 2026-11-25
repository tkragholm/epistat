//! Synthetic difference-in-differences, synthetic control and the plain
//! difference-in-differences estimator beside them, as R's `synthdid` 0.0.9
//! computes them, with its placebo, bootstrap and jackknife standard errors,
//! and a cohort-by-cohort extension to staggered adoption.
//!
//! # Setting
//!
//! A balanced panel of `N` units over `T` periods. The first `N0` units are
//! controls, never treated. The other `N1 = N - N0` units are treated from
//! period `T0 + 1` on, so `T1 = T - T0` periods are post-treatment. `synthdid`
//! handles this single adoption date only; [`estimate_synthdid_staggered`]
//! handles several, one cohort at a time (below).
//!
//! # Algorithm
//!
//! The collapsed panel `Yc` is `(N0 + 1) × (T0 + 1)`: the controls'
//! pre-period outcomes with the mean of their post periods as a last column,
//! and a last row of the treated units' means (`collapsed.form`).
//!
//! The noise level is the standard deviation (divisor `n - 1`) of the
//! `N0 (T0 - 1)` first differences `Y_it - Y_i,t-1` of the controls over the
//! pre-periods, and the defaults are
//!
//! ```text
//! zeta_omega  = (N1 T1)^(1/4) sigma,   zeta_lambda = 1e-6 sigma,
//! min.decrease = 1e-5 sigma.
//! ```
//!
//! The unit weights `omega` solve, over the simplex,
//!
//! ```text
//! min  zeta_omega^2 |omega|^2 + (1/T0) sum_{t <= T0} (omega_0 + sum_j omega_j Y_jt - mean_tr Y_t)^2
//! ```
//!
//! and the time weights `lambda`, over the simplex,
//!
//! ```text
//! min  zeta_lambda^2 |lambda|^2 + (1/N0) sum_{j <= N0} (lambda_0 + sum_t lambda_t Y_jt - mean_post Y_j)^2,
//! ```
//!
//! with the intercepts `omega_0` and `lambda_0` removed by centring the
//! columns of the problem before solving (the `intercept` arguments). Each is
//! solved by [`epistat_optim::frank_wolfe`] with penalty `eta = m zeta^2`,
//! `m` the number of rows, as `sc.weight.fw` does: at most
//! `max.iter.pre.sparsify = 100` steps from uniform weights, then weights no
//! larger than a quarter of the largest are set to zero and the rest
//! rescaled (`sparsify_function`), then at most `max.iter = 10000` more
//! steps from there. A run stops after step `t >= 2` when the objective fell
//! by no more than `min.decrease^2`. The estimate is
//!
//! ```text
//! tau_t = mean_tr Y_t - sum_j omega_j Y_jt                 (t = 1..T)
//! tau   = sum_{t > T0} tau_t / T1 - sum_{t <= T0} lambda_t tau_t,
//! ```
//!
//! and the effect curve is `tau_{T0+s} - sum_{t <= T0} lambda_t tau_t` for
//! `s = 1..T1` (`synthdid_effect_curve`); `tau` is its mean.
//!
//! Synthetic control (`sc_estimate`) is the same with `lambda = 0` held
//! fixed, no intercept in the unit-weight problem and `eta_omega = 1e-6`, so
//! that `zeta_omega = 1e-6 sigma`. Difference-in-differences (`did_estimate`)
//! holds `omega = 1/N0` and `lambda = 1/T0`.
//!
//! # Covariates
//!
//! With `K` time-varying covariates `X_k`, the outcome is replaced by
//! `Y - sum_k beta_k X_k` (`sc.weight.fw.covariates`). From `beta = 0`, each
//! iteration takes one Frank-Wolfe step for `lambda` and one for `omega` on
//! the adjusted collapsed panel, then a gradient step of size `1/t` on `beta`
//! of the sum of the two objectives, until the sum changes by no more than
//! `min.decrease^2` either way or `max.iter` iterations. There is no
//! sparsification in this case. The noise level is that of the unadjusted
//! outcome.
//!
//! # Standard errors
//!
//! Each refit of the placebo and bootstrap methods uses the options of the
//! estimate (the penalties, `min.decrease`, `max.iter`, the intercepts and
//! which weights are estimated) and starts its weights from the estimate's,
//! the unit weights restricted to the units kept and rescaled to sum to one
//! (`sum_normalize`), as `synthdid`'s `vcov` does \[1\].
//!
//! - Placebo (Algorithm 4 of \[1\]): permute the controls, call the last `N1`
//!   of them treated and refit on the controls alone. Needs `N0 > N1`.
//! - Bootstrap (Algorithm 2): draw `N` units with replacement, sorted, and
//!   refit; a draw with no control or no treated unit is drawn again.
//!   Undefined with one treated unit.
//! - Jackknife (Algorithm 3): leave out each unit in turn with the weights
//!   held fixed, the unit weights rescaled. Undefined with one treated unit
//!   or a single control with positive weight.
//!
//! With replicate estimates `u_1..u_B`, the placebo and bootstrap variance is
//! `((B - 1)/B) var(u)` and the jackknife variance `((n - 1)/n) sum (u_i -
//! mean u)^2`. The same formulas over the effect curves give a standard error
//! for each post period. Without covariates the jackknife is computed in
//! closed form: with `h_j = sum_t c_t Y_jt` for the contrast `c = (-lambda,
//! 1/T1)`, `tau = mean_tr h - sum_j omega_j h_j`, so leaving out a control
//! `i` gives `mean_tr h - (sum_j omega_j h_j - omega_i h_i) / (1 - omega_i)`
//! and leaving out a treated unit drops it from the mean, `O(N T)` in all.
//! With covariates each leave-one-out refits `beta` with the weights fixed,
//! as `synthdid` does.
//!
//! The jackknife also gives an influence value for each unit,
//! `psi_i = sqrt(n (n - 1)) (mean u - u_i)`, scaled so that
//! [`did_standard_error`](crate::did_standard_error) with one cluster per
//! unit is the jackknife standard error. With other clusters it is the
//! linearised cluster jackknife; [`SynthVariance::Jackknife`] with clusters
//! leaves out whole clusters instead, with `n` the number of clusters.
//!
//! Random draws come from [`epistat_rng::Stream`], replicate `r` seeded with
//! [`epistat_rng::replicate_seed`]`(seed, r)`: a placebo draw is a
//! Fisher-Yates shuffle of the controls and a bootstrap draw `N` calls of
//! `below(N)`, repeated within the same stream until it holds a control and a
//! treated unit. Replicates run in parallel and their results are combined in
//! replicate order, so they do not depend on the number of threads. The draws
//! are not R's; [`SynthDraws::Given`] takes the draws from outside, which is
//! how the tests feed R's draws through and compare replicate by replicate.
//!
//! # Staggered adoption
//!
//! `synthdid` takes one adoption date. For several, the estimate is formed
//! cohort by cohort, as in the appendix of \[1\] and in Stata's `sdid` \[4\]:
//! for each adoption period `a`, the units first treated at `a` against the
//! never-treated units over all periods, with `T0 = a` pre-periods, each with
//! its own noise level and penalties. The aggregate weights cohort `a` by its
//! number of treated cells, `N_a (T - a)`:
//!
//! ```text
//! tau = sum_a N_a (T - a) tau_a / sum_a N_a (T - a),
//! ```
//!
//! and the effect at event time `e` (periods since adoption) is the mean of
//! the cohorts' effect curves at `e`, weighted by `N_a`, over the cohorts
//! observed `e` periods after adoption. The standard errors replay the whole
//! procedure: the jackknife leaves a unit out of every cohort it belongs to
//! (undefined when a cohort has a single treated unit), the bootstrap draws
//! units from the whole panel and drops cohorts that the draw leaves without
//! treated units, and the placebo permutes the never-treated units and gives
//! the last of them the adoption periods of the treated units, in panel
//! order. With one cohort each reduces to the single-date method above.
//!
//! # Differences from `synthdid`
//!
//! - [`SynthPanel::from_observations`] orders the units by treatment, then by
//!   their integer identifier; `panel.matrices` sorts the names as text.
//! - The refits of the variance methods use the estimate's sparsification
//!   setting and first-run limit. `synthdid` passes neither to its refits,
//!   which then use the defaults whatever the estimate used.
//! - A Frank-Wolfe step whose line search has a zero denominator (a flat
//!   direction, possible only without a penalty) takes the full step when the
//!   slope is negative and none otherwise, where R divides by zero.
//! - The sums run over eight fixed lanes, not through BLAS, so the weights
//!   agree with R's to rounding; in the tests the step counts and the zero
//!   weights agree exactly.
//!
//! # References
//!
//! 1. Arkhangelsky D, Athey S, Hirshberg DA, Imbens GW, Wager S. Synthetic
//!    difference-in-differences. Am Econ Rev 2021;111(12):4088-118.
//!    doi:10.1257/aer.20190159. R package `synthdid` 0.0.9.
//! 2. Abadie A, Diamond A, Hainmueller J. Synthetic control methods for
//!    comparative case studies: estimating the effect of California's tobacco
//!    control program. J Am Stat Assoc 2010;105(490):493-505.
//!    doi:10.1198/jasa.2009.ap08746
//! 3. Frank M, Wolfe P. An algorithm for quadratic programming. Nav Res Logist
//!    Q 1956;3(1-2):95-110. doi:10.1002/nav.3800030109
//! 4. Clarke D, Pailañir D, Athey S, Imbens G. On synthetic
//!    difference-in-differences and related estimation methods in Stata.
//!    Stata J 2024;24(4):557-98.

mod fit;
mod panel;
mod staggered;
mod variance;

use epistat_inference::Clusters;
use epistat_optim::frank_wolfe::FrankWolfeError;

use fit::{Data, Opts, Start};
pub use panel::SynthObservation;
pub use staggered::{
    StaggeredSynthEstimate, StaggeredSynthPanel, SynthCohort, estimate_synthdid_staggered,
    synthdid_staggered_variance,
};

/// Why an estimate or a standard error was refused.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SynthDidError {
    /// The values do not fill `units × periods` (or `count` such blocks of
    /// covariates), or a weight vector or draw has the wrong length.
    #[error("the sizes do not match: {0}")]
    Dimension(&'static str),
    /// An outcome or covariate is not finite.
    #[error("outcomes and covariates must be finite")]
    NotFinite,
    /// No control unit, no treated unit, no pre-period or no post-period.
    #[error("the panel needs controls, treated units, and periods before and after adoption")]
    Design,
    /// The default noise level needs at least two first differences of the
    /// controls before adoption; give the noise level instead.
    #[error("the noise level needs two first differences before adoption; give it explicitly")]
    NoiseLevel,
    /// A configured number is not finite or out of range.
    #[error("invalid setting: {0}")]
    Setting(&'static str),
    /// Weights that are not finite, starting weights off the simplex, or
    /// weights that a refit needs and that are missing.
    #[error("weights must be finite, and starting weights non-negative and summing to one")]
    Weights,
    /// The placebo method needs more controls than treated units.
    #[error("the placebo method needs more controls than treated units")]
    PlaceboControls,
    /// A given draw is not a permutation of the controls (placebo) or holds
    /// an index outside the panel or no control or no treated unit
    /// (bootstrap).
    #[error("draw {0} is not a valid draw for this method")]
    Draw(usize),
    /// Fewer than two replicates.
    #[error("at least two replicates are needed")]
    Replications,
    /// The estimate was not computed on this panel.
    #[error("the estimate does not belong to this panel")]
    Mismatch,
    /// The observations do not form a balanced panel with the adoption
    /// pattern the method needs.
    #[error("invalid panel: {0}")]
    Panel(String),
    /// The Frank-Wolfe solver refused its problem.
    #[error("Frank-Wolfe: {0}")]
    Solver(#[from] FrankWolfeError),
}

/// A balanced panel with a single adoption date: `units × periods` outcomes
/// by unit, the first `controls` units never treated and the rest treated
/// after `pre_periods` periods, as `synthdid` takes `Y`, `N0` and `T0`.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthPanel {
    outcome: Vec<f64>,
    covariates: Vec<f64>,
    covariate_count: usize,
    units: usize,
    periods: usize,
    controls: usize,
    pre_periods: usize,
    unit_ids: Vec<i64>,
    times: Vec<i32>,
}

impl SynthPanel {
    /// The panel from its outcomes, unit by unit (`outcome[i * periods + t]`).
    ///
    /// # Errors
    ///
    /// [`SynthDidError::Dimension`], [`SynthDidError::NotFinite`] or
    /// [`SynthDidError::Design`] when there are no controls, no treated
    /// units, no pre-period or no post-period.
    pub fn new(
        outcome: Vec<f64>,
        units: usize,
        periods: usize,
        controls: usize,
        pre_periods: usize,
    ) -> Result<Self, SynthDidError> {
        if units.checked_mul(periods) != Some(outcome.len()) {
            return Err(SynthDidError::Dimension("outcome is not units × periods"));
        }
        if outcome.iter().any(|v| !v.is_finite()) {
            return Err(SynthDidError::NotFinite);
        }
        if controls == 0 || controls >= units || pre_periods == 0 || pre_periods >= periods {
            return Err(SynthDidError::Design);
        }
        Ok(Self {
            outcome,
            covariates: Vec::new(),
            covariate_count: 0,
            units,
            periods,
            controls,
            pre_periods,
            unit_ids: (0..units).map(crate::util::usize_to_i64).collect(),
            times: (0..periods).map(staggered::period_label).collect(),
        })
    }

    /// Adds `count` time-varying covariates, each a `units × periods` block
    /// laid out as the outcome (`covariates[(k * units + i) * periods + t]`),
    /// the array `X` of `synthdid` with its third index first.
    ///
    /// # Errors
    ///
    /// [`SynthDidError::Dimension`] or [`SynthDidError::NotFinite`].
    pub fn with_covariates(
        mut self,
        covariates: Vec<f64>,
        count: usize,
    ) -> Result<Self, SynthDidError> {
        if self.outcome.len().checked_mul(count) != Some(covariates.len()) {
            return Err(SynthDidError::Dimension(
                "covariates are not count × units × periods",
            ));
        }
        if covariates.iter().any(|v| !v.is_finite()) {
            return Err(SynthDidError::NotFinite);
        }
        self.covariates = covariates;
        self.covariate_count = count;
        Ok(self)
    }

    /// Number of units.
    #[must_use]
    pub fn units(&self) -> usize {
        self.units
    }

    /// Number of periods.
    #[must_use]
    pub fn periods(&self) -> usize {
        self.periods
    }

    /// Number of controls, `N0`.
    #[must_use]
    pub fn controls(&self) -> usize {
        self.controls
    }

    /// Number of pre-treatment periods, `T0`.
    #[must_use]
    pub fn pre_periods(&self) -> usize {
        self.pre_periods
    }

    /// Number of covariates.
    #[must_use]
    pub fn covariate_count(&self) -> usize {
        self.covariate_count
    }

    /// The outcomes, unit by unit.
    #[must_use]
    pub fn outcome(&self) -> &[f64] {
        &self.outcome
    }

    /// The unit of each row: the identifiers of
    /// [`SynthPanel::from_observations`], or `0..units`.
    #[must_use]
    pub fn unit_ids(&self) -> &[i64] {
        &self.unit_ids
    }

    /// The period of each column: the times of
    /// [`SynthPanel::from_observations`], or `0..periods`.
    #[must_use]
    pub fn times(&self) -> &[i32] {
        &self.times
    }

    fn data(&self) -> Data<'_> {
        Data {
            outcome: &self.outcome,
            covariates: &self.covariates,
            count: self.covariate_count,
            units: self.units,
            periods: self.periods,
        }
    }
}

/// How a set of weights is found.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SynthWeights {
    /// Estimated by Frank-Wolfe from uniform weights.
    #[default]
    Estimate,
    /// Estimated by Frank-Wolfe from these weights, which must lie on the
    /// simplex.
    Start(Vec<f64>),
    /// Held at these weights.
    Fixed(Vec<f64>),
    /// Held at uniform weights, `1/N0` or `1/T0`.
    FixedUniform,
    /// Held at zero, as the time weights of synthetic control.
    FixedZero,
}

/// The arguments of `synthdid_estimate`, with `None` for the defaults that
/// depend on the data. [`SynthDidConfig::synthdid`],
/// [`SynthDidConfig::synthetic_control`] and [`SynthDidConfig::did`] are the
/// three estimators of the package.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthDidConfig {
    /// `noise.level`; `None` for the standard deviation of the controls'
    /// first differences before adoption.
    pub noise_level: Option<f64>,
    /// `eta.omega`; `None` for `(N1 T1)^(1/4)`.
    pub eta_omega: Option<f64>,
    /// `eta.lambda`, default `1e-6`.
    pub eta_lambda: f64,
    /// `zeta.omega`; `None` for `eta_omega * noise_level`.
    pub zeta_omega: Option<f64>,
    /// `zeta.lambda`; `None` for `eta_lambda * noise_level`.
    pub zeta_lambda: Option<f64>,
    /// `omega.intercept`, default `true`.
    pub omega_intercept: bool,
    /// `lambda.intercept`, default `true`.
    pub lambda_intercept: bool,
    /// The unit weights.
    pub omega: SynthWeights,
    /// The time weights.
    pub lambda: SynthWeights,
    /// `min.decrease`; `None` for `1e-5 * noise_level`.
    pub min_decrease: Option<f64>,
    /// `max.iter`, default 10000.
    pub max_iter: usize,
    /// `max.iter.pre.sparsify`, default 100.
    pub max_iter_pre_sparsify: usize,
    /// Whether to sparsify between the two runs (`sparsify =
    /// sparsify_function`, the default) or run once (`sparsify = NULL`).
    pub sparsify: bool,
}

impl SynthDidConfig {
    /// `synthdid_estimate` with its defaults.
    #[must_use]
    pub fn synthdid() -> Self {
        Self {
            noise_level: None,
            eta_omega: None,
            eta_lambda: 1e-6,
            zeta_omega: None,
            zeta_lambda: None,
            omega_intercept: true,
            lambda_intercept: true,
            omega: SynthWeights::Estimate,
            lambda: SynthWeights::Estimate,
            min_decrease: None,
            max_iter: 10_000,
            max_iter_pre_sparsify: 100,
            sparsify: true,
        }
    }

    /// `sc_estimate`: time weights fixed at zero, no intercept in the unit
    /// weights, `eta_omega = 1e-6`.
    #[must_use]
    pub fn synthetic_control() -> Self {
        Self {
            eta_omega: Some(1e-6),
            omega_intercept: false,
            lambda: SynthWeights::FixedZero,
            ..Self::synthdid()
        }
    }

    /// `did_estimate`: uniform unit and time weights.
    #[must_use]
    pub fn did() -> Self {
        Self {
            omega: SynthWeights::FixedUniform,
            lambda: SynthWeights::FixedUniform,
            ..Self::synthdid()
        }
    }
}

impl Default for SynthDidConfig {
    fn default() -> Self {
        Self::synthdid()
    }
}

/// The objective after each Frank-Wolfe step of the last run, as
/// `synthdid` keeps them in `lambda.vals`, `omega.vals` and, with
/// covariates, `vals`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SynthTrace {
    /// The time-weight problem, empty when the time weights are held fixed
    /// or there are covariates.
    pub lambda: Vec<f64>,
    /// The unit-weight problem, likewise.
    pub omega: Vec<f64>,
    /// The joint objective with covariates, empty without.
    pub joint: Vec<f64>,
}

/// An estimate with its weights.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthDidEstimate {
    /// The average effect on the treated over the post periods.
    pub estimate: f64,
    /// The effect in each post period, whose mean is `estimate`.
    pub effect_curve: Vec<f64>,
    /// Unit weights, one per control.
    pub omega: Vec<f64>,
    /// Time weights, one per pre-period.
    pub lambda: Vec<f64>,
    /// Covariate coefficients, empty without covariates.
    pub beta: Vec<f64>,
    /// The noise level used.
    pub noise_level: f64,
    /// The penalty of the unit weights.
    pub zeta_omega: f64,
    /// The penalty of the time weights.
    pub zeta_lambda: f64,
    /// The stopping tolerance's `min.decrease`.
    pub min_decrease: f64,
    /// The objective after each step of the last runs.
    pub trace: SynthTrace,
    /// `false` when a last run stopped at `max_iter` with the objective still
    /// falling by more than `min_decrease^2`. `synthdid` reports nothing in
    /// that case.
    pub converged: bool,
    pub(crate) opts: Opts,
    pub(crate) shape: (usize, usize, usize, usize, usize),
}

/// A standard error.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthDidVariance {
    /// The standard error of the estimate, `None` where `synthdid` returns NA
    /// (see [`SynthUndefined`]).
    pub se: Option<f64>,
    /// The standard error of the effect in each post period (or event time,
    /// for staggered adoption), from the same replicates.
    pub effect_curve_se: Vec<Option<f64>>,
    /// The replicate estimates: leave-one-out estimates for the jackknife, in
    /// the order of the units (or clusters), refits for the placebo and the
    /// bootstrap, in replicate order.
    pub replicates: Vec<f64>,
    /// The jackknife influence values of the units, scaled as described in
    /// the module documentation; `None` for the other methods and the
    /// cluster jackknife.
    pub influence: Option<Vec<f64>>,
    /// Why the standard error is undefined, when it is.
    pub undefined: Option<SynthUndefined>,
}

/// Why a standard error is undefined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthUndefined {
    /// A cohort has a single treated unit (jackknife and bootstrap), or a
    /// cluster holds all treated units of a cohort.
    OneTreatedUnit,
    /// The unit weights of a cohort have a single positive entry, or a
    /// cluster holds all controls with positive weight (jackknife).
    OneWeightedControl,
}

/// The method of a standard error.
#[derive(Debug, Clone, Copy)]
pub enum SynthVariance<'a> {
    /// Leave one unit, or with `clusters` one cluster, out at a time with
    /// the weights fixed.
    Jackknife {
        /// One label per unit, in panel order; `None` for one per unit.
        clusters: Option<&'a Clusters>,
    },
    /// Refits on the controls with some of them called treated.
    Placebo(SynthDraws<'a>),
    /// Refits on units drawn with replacement.
    Bootstrap(SynthDraws<'a>),
}

/// The random draws of the placebo and the bootstrap.
#[derive(Debug, Clone, Copy)]
pub enum SynthDraws<'a> {
    /// `replications` draws from `seed`, as described in the module
    /// documentation; `synthdid` uses 200.
    Seeded {
        /// Number of replicates, at least 2.
        replications: usize,
        /// Base seed.
        seed: u64,
    },
    /// The draws themselves, 0-based: for the placebo a permutation of the
    /// controls' positions `0..N0`, for the bootstrap `N` unit indices with
    /// repeats, in the order drawn.
    Given(&'a [Vec<usize>]),
}

/// Resolves the configuration for one cohort: the noise level, the
/// penalties and the weight rules.
pub(crate) fn resolve(
    config: &SynthDidConfig,
    data: &Data<'_>,
    controls: &[usize],
    treated: usize,
    pre: usize,
) -> Result<(Opts, Start, f64), SynthDidError> {
    let n0 = controls.len();
    let post = data.periods - pre;
    let noise = match config.noise_level {
        Some(v) if v.is_finite() && v >= 0.0 => v,
        Some(_) => return Err(SynthDidError::Setting("noise_level")),
        None => fit::noise_level(data, controls, pre).ok_or(SynthDidError::NoiseLevel)?,
    };
    let positive = |v: f64, name| {
        if v.is_finite() && v >= 0.0 {
            Ok(v)
        } else {
            Err(SynthDidError::Setting(name))
        }
    };
    let eta_omega = match config.eta_omega {
        Some(v) => positive(v, "eta_omega")?,
        None => ((treated * post) as f64).powf(0.25),
    };
    let eta_lambda = positive(config.eta_lambda, "eta_lambda")?;
    let zeta_omega = match config.zeta_omega {
        Some(v) => positive(v, "zeta_omega")?,
        None => eta_omega * noise,
    };
    let zeta_lambda = match config.zeta_lambda {
        Some(v) => positive(v, "zeta_lambda")?,
        None => eta_lambda * noise,
    };
    let min_decrease = match config.min_decrease {
        Some(v) => positive(v, "min_decrease")?,
        None => 1e-5 * noise,
    };
    if config.max_iter == 0 || config.max_iter_pre_sparsify == 0 {
        return Err(SynthDidError::Setting("max_iter"));
    }
    let rule =
        |weights: &SynthWeights, len: usize| -> Result<(bool, Option<Vec<f64>>), SynthDidError> {
            let check = |v: &Vec<f64>, simplex: bool| -> Result<(), SynthDidError> {
                if v.len() != len {
                    return Err(SynthDidError::Dimension("weights"));
                }
                if v.iter().any(|x| !x.is_finite()) {
                    return Err(SynthDidError::Weights);
                }
                let total: f64 = v.iter().sum();
                if simplex && (v.iter().any(|&x| x < 0.0) || (total - 1.0).abs() > 1e-8) {
                    return Err(SynthDidError::Weights);
                }
                Ok(())
            };
            Ok(match weights {
                SynthWeights::Estimate => (true, None),
                SynthWeights::Start(v) => {
                    check(v, true)?;
                    (true, Some(v.clone()))
                }
                SynthWeights::Fixed(v) => {
                    check(v, false)?;
                    (false, Some(v.clone()))
                }
                SynthWeights::FixedUniform => (false, Some(vec![1.0 / len as f64; len])),
                SynthWeights::FixedZero => (false, Some(vec![0.0; len])),
            })
        };
    let (update_omega, omega) = rule(&config.omega, n0)?;
    let (update_lambda, lambda) = rule(&config.lambda, pre)?;
    Ok((
        Opts {
            zeta_omega,
            zeta_lambda,
            omega_intercept: config.omega_intercept,
            lambda_intercept: config.lambda_intercept,
            update_omega,
            update_lambda,
            min_decrease,
            max_iter: config.max_iter,
            max_iter_pre_sparsify: config.max_iter_pre_sparsify,
            sparsify: config.sparsify,
        },
        Start { lambda, omega },
        noise,
    ))
}

/// The estimate of `synthdid_estimate`, `sc_estimate` or `did_estimate`,
/// according to `config`.
///
/// # Errors
///
/// [`SynthDidError::NoiseLevel`] when the default noise level is undefined,
/// [`SynthDidError::Setting`], [`SynthDidError::Dimension`] or
/// [`SynthDidError::Weights`] for a configuration that does not fit the
/// panel.
pub fn estimate_synthdid(
    panel: &SynthPanel,
    config: &SynthDidConfig,
) -> Result<SynthDidEstimate, SynthDidError> {
    let data = panel.data();
    let controls: Vec<usize> = (0..panel.controls).collect();
    let treated: Vec<usize> = (panel.controls..panel.units).collect();
    let pre = panel.pre_periods;
    let (opts, start, noise) = resolve(config, &data, &controls, treated.len(), pre)?;
    let fitted = fit::fit(&data, &controls, &treated, pre, &opts, start)?;
    Ok(SynthDidEstimate {
        estimate: fitted.estimate,
        effect_curve: fitted.curve,
        omega: fitted.omega,
        lambda: fitted.lambda,
        beta: fitted.beta,
        noise_level: noise,
        zeta_omega: opts.zeta_omega,
        zeta_lambda: opts.zeta_lambda,
        min_decrease: opts.min_decrease,
        trace: fitted.trace,
        converged: fitted.converged,
        opts,
        shape: shape(panel),
    })
}

fn shape(panel: &SynthPanel) -> (usize, usize, usize, usize, usize) {
    (
        panel.units,
        panel.periods,
        panel.controls,
        panel.pre_periods,
        panel.covariate_count,
    )
}

/// The standard error of `synthdid`'s `vcov(estimate, method)`, with a
/// standard error for each post period beside it.
///
/// # Errors
///
/// [`SynthDidError::Mismatch`] when `estimate` was not computed on `panel`,
/// [`SynthDidError::PlaceboControls`], [`SynthDidError::Replications`],
/// [`SynthDidError::Draw`] or [`SynthDidError::Dimension`] for clusters or
/// draws that do not fit, and the errors of a refit.
pub fn synthdid_variance(
    panel: &SynthPanel,
    estimate: &SynthDidEstimate,
    method: SynthVariance<'_>,
) -> Result<SynthDidVariance, SynthDidError> {
    if estimate.shape != shape(panel) {
        return Err(SynthDidError::Mismatch);
    }
    let plan = variance::Plan {
        data: panel.data(),
        controls: (0..panel.controls).collect(),
        cohorts: vec![variance::PlanCohort {
            pre: panel.pre_periods,
            treated: (panel.controls..panel.units).collect(),
            opts: estimate.opts,
            omega: estimate.omega.clone(),
            lambda: estimate.lambda.clone(),
        }],
    };
    variance::variance(&plan, method)
}
