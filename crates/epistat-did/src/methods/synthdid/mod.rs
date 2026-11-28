//! Synthetic difference-in-differences, synthetic control and the plain
//! difference-in-differences estimator beside them, as R's `synthdid` 0.0.9
//! computes them, with its placebo, bootstrap and jackknife standard errors,
//! and a cohort-by-cohort extension to staggered adoption.
//!
//! # Setting
//!
//! A balanced panel of `N` units over `T` periods, a [`StaggeredPanel`] in
//! which every unit is observed in every period. `N0` units are controls,
//! never treated. The other `N1 = N - N0` units are treated from period
//! `T0 + 1` on, so `T1 = T - T0` periods are post-treatment. `synthdid`
//! handles this single adoption date only, which is [`estimate`];
//! [`estimate_staggered`] handles several, one cohort at a time (below). The
//! units are taken in increasing code and the periods in increasing order.
//! The standard error is chosen by [`SynthDidOptions::variance`].
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
//! `psi_i = sqrt(n (n - 1)) (mean u - u_i)`, under
//! [`ClusterRule::NoSmallSample`](epistat_inference::ClusterRule), so that
//! [`Influence::variance`] without clusters is the jackknife variance. With
//! clusters it is the linearised cluster jackknife; on a panel with
//! clusters, [`SynthVariance::Jackknife`] leaves out whole clusters instead,
//! with `n` the number of clusters, and gives no influence values.
//!
//! Random draws come from [`epistat_rng::Stream`], replicate `r` seeded with
//! [`epistat_rng::replicate_seed`]`(seed, r)`: a placebo draw is a
//! Fisher-Yates shuffle of the controls and a bootstrap draw `N` calls of
//! `below(N)`, repeated within the same stream until it holds a control and a
//! treated unit. Replicates run in parallel and their results are combined in
//! replicate order, so they do not depend on the number of threads. The draws
//! are not R's; [`SynthDraws::Given`] takes the draws from outside, which is
//! how the tests feed R's draws through and compare replicate by replicate.
//! A bootstrap draw names units by their position in increasing code.
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
//! - The units are in increasing code; `panel.matrices` puts the controls
//!   first and sorts the names as text. The order decides which unit a
//!   bootstrap draw names and the order of the jackknife's replicates, not
//!   the estimate.
//! - The estimate's interval is the normal one at
//!   [`SynthDidOptions::level`]; `synthdid` reports the standard error
//!   alone.
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

use epistat_core::{Bootstrap, Estimate, InputError, Level, Scale};
use epistat_inference::{ClusterRule, Influence};

use crate::error::Error;
use crate::methods::staggered::StaggeredPanel;

use fit::{Data, Opts, Start};
use panel::Grid;
pub use staggered::{StaggeredSynthDid, SynthCohort, estimate_staggered};

/// How a set of weights is found.
#[non_exhaustive]
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

/// The standard error, as `vcov(estimate, method)` of `synthdid`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SynthVariance {
    /// No standard error, as `synthdid_estimate` alone.
    #[default]
    None,
    /// Leave one unit out at a time with the weights fixed, or one cluster
    /// when the panel has clusters (`method = "jackknife"`).
    Jackknife,
    /// Refits on the controls with some of them called treated (`method =
    /// "placebo"`).
    Placebo(SynthDraws),
    /// Refits on units drawn with replacement (`method = "bootstrap"`).
    Bootstrap(SynthDraws),
}

/// The random draws of the placebo and the bootstrap.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum SynthDraws {
    /// Draws from the root seed, as described in the module documentation;
    /// `synthdid` uses 200 replicates.
    Seeded(Bootstrap),
    /// The draws themselves, 0-based: for the placebo a permutation of the
    /// controls' positions `0..N0` among the never-treated units, for the
    /// bootstrap `N` positions among all units, with repeats, in the order
    /// drawn. Positions count units in increasing code.
    Given(Vec<Vec<usize>>),
}

/// The options of [`estimate`] and [`estimate_staggered`]: the arguments of
/// `synthdid_estimate`, with `None` for the defaults that depend on the
/// data, the standard error of `vcov` and the level of the intervals. The
/// default is `synthdid_estimate`; [`SynthDidOptions::synthetic_control`]
/// and [`SynthDidOptions::did`] are the package's `sc_estimate` and
/// `did_estimate`, the same function with other defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthDidOptions {
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
    pub max_iterations: usize,
    /// `max.iter.pre.sparsify`, default 100.
    pub max_iterations_pre_sparsify: usize,
    /// Whether to sparsify between the two runs (`sparsify =
    /// sparsify_function`, the default) or run once (`sparsify = NULL`).
    pub sparsify: bool,
    /// The standard error; by default none.
    pub variance: SynthVariance,
    /// The level of the normal intervals; by default 95%.
    pub level: Level,
}

impl Default for SynthDidOptions {
    /// `synthdid_estimate` with its defaults.
    fn default() -> Self {
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
            max_iterations: 10_000,
            max_iterations_pre_sparsify: 100,
            sparsify: true,
            variance: SynthVariance::None,
            level: Level::default(),
        }
    }
}

impl SynthDidOptions {
    /// `sc_estimate`: time weights fixed at zero, no intercept in the unit
    /// weights, `eta_omega = 1e-6`.
    #[must_use]
    pub fn synthetic_control() -> Self {
        Self {
            eta_omega: Some(1e-6),
            omega_intercept: false,
            lambda: SynthWeights::FixedZero,
            ..Self::default()
        }
    }

    /// `did_estimate`: uniform unit and time weights.
    #[must_use]
    pub fn did() -> Self {
        Self {
            omega: SynthWeights::FixedUniform,
            lambda: SynthWeights::FixedUniform,
            ..Self::default()
        }
    }
}

/// The objective after each Frank-Wolfe step of the last run, as
/// `synthdid` keeps them in `lambda.vals`, `omega.vals` and, with
/// covariates, `vals`.
#[non_exhaustive]
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

/// The weights of one adoption cohort and the settings they were found
/// with.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SynthFit {
    /// Unit weights, one per control, in increasing unit code.
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
    /// `false` when a last run stopped at `max_iterations` with the
    /// objective still falling by more than `min_decrease^2`. `synthdid`
    /// reports nothing in that case.
    pub converged: bool,
}

/// The replicates of a standard error.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SynthReplicates {
    /// The replicate estimates: leave-one-out estimates for the jackknife, in
    /// increasing code of the units (or clusters) left out, refits for the
    /// placebo and the bootstrap, in replicate order.
    pub replicates: Vec<f64>,
    /// The jackknife influence values of the units, as described in the
    /// module documentation; `None` for the other methods and the cluster
    /// jackknife.
    pub influence: Option<Influence>,
    /// Why the standard error is undefined, when it is.
    pub undefined: Option<SynthUndefined>,
}

/// Why a standard error is undefined.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynthUndefined {
    /// A cohort has a single treated unit (jackknife and bootstrap), or a
    /// cluster holds all treated units of a cohort.
    OneTreatedUnit,
    /// The unit weights of a cohort have a single positive entry, or a
    /// cluster holds all controls with positive weight (jackknife).
    OneWeightedControl,
}

/// A single-adoption estimate: `synthdid_estimate`, `sc_estimate` or
/// `did_estimate`, with the standard error of `vcov` when asked for.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct SynthDid {
    /// The average effect on the treated over the post periods, with the
    /// standard error of [`SynthDidOptions::variance`] and its normal
    /// interval and p-value; the standard error `None` without a variance or
    /// where `synthdid` returns NA (see [`SynthReplicates::undefined`]).
    pub estimate: Estimate,
    /// The effect in each post period, whose mean is the estimate, with a
    /// standard error from the same replicates.
    pub effect_curve: Vec<Estimate>,
    /// The period the treated units are first treated in.
    pub adoption: i64,
    /// The codes of the never-treated units, the order of
    /// [`SynthFit::omega`].
    pub controls: Vec<u32>,
    /// The codes of the treated units.
    pub treated: Vec<u32>,
    /// The weights.
    pub fit: SynthFit,
    /// The replicates of the standard error, `None` without a variance.
    pub variance: Option<SynthReplicates>,
}

/// The method of a standard error as the variance routines take it.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Method<'a> {
    Jackknife(Option<&'a epistat_inference::Clusters>),
    Placebo(Draws<'a>),
    Bootstrap(Draws<'a>),
}

/// The draws of a placebo or a bootstrap as the variance routines take them.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Draws<'a> {
    Seeded { replications: usize, seed: u64 },
    Given(&'a [Vec<usize>]),
}

impl SynthVariance {
    /// The method of the options on the panel `grid`, `None` for no
    /// variance.
    fn method<'a>(&'a self, grid: &'a Grid) -> Option<Method<'a>> {
        let draws = |d: &'a SynthDraws| match d {
            SynthDraws::Seeded(b) => Draws::Seeded {
                replications: b.replicates,
                seed: b.seed.value(),
            },
            SynthDraws::Given(given) => Draws::Given(given),
        };
        match self {
            Self::None => None,
            Self::Jackknife => Some(Method::Jackknife(grid.clusters.as_ref())),
            Self::Placebo(d) => Some(Method::Placebo(draws(d))),
            Self::Bootstrap(d) => Some(Method::Bootstrap(draws(d))),
        }
    }
}

/// A standard error with its replicates, as the variance routines return it.
pub(crate) struct Variance {
    pub se: Option<f64>,
    pub curve_se: Vec<Option<f64>>,
    pub replicates: Vec<f64>,
    pub influence: Option<Vec<f64>>,
    pub undefined: Option<SynthUndefined>,
}

impl Variance {
    /// The replicates of the result, with the influence values of the units
    /// `units`.
    fn replicates(self, units: &[u32]) -> Result<SynthReplicates, Error> {
        let influence = self
            .influence
            .map(|values| Influence::new(values, units.to_vec(), 1, ClusterRule::NoSmallSample))
            .transpose()?;
        Ok(SynthReplicates {
            replicates: self.replicates,
            influence,
            undefined: self.undefined,
        })
    }
}

/// `value` with the standard error `se` and its normal interval at `level`.
fn with_se(value: f64, se: Option<f64>, level: Level) -> Estimate {
    epistat_stats::wald(value, se, level, Scale::Identity)
}

fn domain(what: &'static str, value: f64, domain: &'static str) -> Error {
    InputError::Domain {
        what,
        value,
        domain,
        row: None,
    }
    .into()
}

/// Resolves the options for one cohort: the noise level, the penalties and
/// the weight rules.
pub(crate) fn resolve(
    options: &SynthDidOptions,
    data: &Data<'_>,
    controls: &[usize],
    treated: usize,
    pre: usize,
) -> Result<(Opts, Start, f64), Error> {
    let n0 = controls.len();
    let post = data.periods - pre;
    let positive = |v: f64, name| {
        if v.is_finite() && v >= 0.0 {
            Ok(v)
        } else {
            Err(domain(name, v, "[0, inf)"))
        }
    };
    let noise = match options.noise_level {
        Some(v) => positive(v, "noise_level")?,
        None => fit::noise_level(data, controls, pre).ok_or(Error::NoiseLevel)?,
    };
    let eta_omega = match options.eta_omega {
        Some(v) => positive(v, "eta_omega")?,
        None => ((treated * post) as f64).powf(0.25),
    };
    let eta_lambda = positive(options.eta_lambda, "eta_lambda")?;
    let zeta_omega = match options.zeta_omega {
        Some(v) => positive(v, "zeta_omega")?,
        None => eta_omega * noise,
    };
    let zeta_lambda = match options.zeta_lambda {
        Some(v) => positive(v, "zeta_lambda")?,
        None => eta_lambda * noise,
    };
    let min_decrease = match options.min_decrease {
        Some(v) => positive(v, "min_decrease")?,
        None => 1e-5 * noise,
    };
    if options.max_iterations == 0 {
        return Err(domain("max_iterations", 0.0, "1 or more"));
    }
    if options.max_iterations_pre_sparsify == 0 {
        return Err(domain("max_iterations_pre_sparsify", 0.0, "1 or more"));
    }
    let rule = |what: &'static str,
                weights: &SynthWeights,
                len: usize|
     -> Result<(bool, Option<Vec<f64>>), Error> {
        let check = |v: &Vec<f64>, simplex: bool| -> Result<(), Error> {
            epistat_core::check::same_length(what, len, v.len())?;
            epistat_core::check::finite(what, v)?;
            let total: f64 = v.iter().sum();
            if let Some(&w) = v.iter().find(|&&x| x < 0.0).filter(|_| simplex) {
                return Err(domain(what, w, "the simplex"));
            }
            if simplex && (total - 1.0).abs() > 1e-8 {
                return Err(domain(what, total, "the simplex"));
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
    let (update_omega, omega) = rule("omega", &options.omega, n0)?;
    let (update_lambda, lambda) = rule("lambda", &options.lambda, pre)?;
    Ok((
        Opts {
            zeta_omega,
            zeta_lambda,
            omega_intercept: options.omega_intercept,
            lambda_intercept: options.lambda_intercept,
            update_omega,
            update_lambda,
            min_decrease,
            max_iter: options.max_iterations,
            max_iter_pre_sparsify: options.max_iterations_pre_sparsify,
            sparsify: options.sparsify,
        },
        Start { lambda, omega },
        noise,
    ))
}

/// The estimate of `synthdid_estimate`, `sc_estimate` or `did_estimate` on a
/// panel with a single adoption period, according to `options`, with the
/// standard error of [`SynthDidOptions::variance`].
///
/// # Errors
///
/// The errors of the panel's matrices: [`Error::Unsupported`] for prior
/// weights or comparison cohorts, [`Error::Unbalanced`],
/// [`Error::Adoption`], [`Error::NoNeverTreated`] and
/// [`Error::NoTreatment`]; [`Error::Unsupported`] for more than one
/// adoption period, which [`estimate_staggered`] takes;
/// [`Error::NoiseLevel`] when the default noise level is undefined;
/// [`Error::Input`] for options that do not fit the panel; and for the
/// variance, [`Error::PlaceboControls`], [`Error::Draw`] and
/// [`Error::Input`] for fewer than two replicates.
pub fn estimate(panel: &StaggeredPanel<'_>, options: &SynthDidOptions) -> Result<SynthDid, Error> {
    let grid = Grid::new(panel)?;
    let cohorts = grid.cohorts();
    let [(pre, treated)] = cohorts.as_slice() else {
        return Err(Error::Unsupported {
            what: "synthdid::estimate with several adoption periods; estimate_staggered takes them",
        });
    };
    let (pre, treated) = (*pre, treated.clone());
    let data = grid.data();
    let controls = grid.controls();
    let (opts, start, noise) = resolve(options, &data, &controls, treated.len(), pre)?;
    let fitted = fit::fit(&data, &controls, &treated, pre, &opts, start)?;
    let variance = match options.variance.method(&grid) {
        None => None,
        Some(method) => {
            let plan = variance::Plan {
                data,
                controls: controls.clone(),
                cohorts: vec![variance::PlanCohort {
                    pre,
                    treated: treated.clone(),
                    opts,
                    omega: fitted.omega.clone(),
                    lambda: fitted.lambda.clone(),
                }],
            };
            Some(variance::variance(&plan, method)?)
        }
    };
    let level = options.level;
    let se = variance.as_ref().and_then(|v| v.se);
    let curve_se = |s: usize| variance.as_ref().and_then(|v| v.curve_se[s]);
    Ok(SynthDid {
        estimate: with_se(fitted.estimate, se, level),
        effect_curve: fitted
            .curve
            .iter()
            .enumerate()
            .map(|(s, &v)| with_se(v, curve_se(s), level))
            .collect(),
        adoption: grid.periods[pre],
        controls: grid.codes(&controls),
        treated: grid.codes(&treated),
        fit: SynthFit {
            omega: fitted.omega,
            lambda: fitted.lambda,
            beta: fitted.beta,
            noise_level: noise,
            zeta_omega: opts.zeta_omega,
            zeta_lambda: opts.zeta_lambda,
            min_decrease: opts.min_decrease,
            trace: fitted.trace,
            converged: fitted.converged,
        },
        variance: variance.map(|v| v.replicates(&grid.units)).transpose()?,
    })
}
