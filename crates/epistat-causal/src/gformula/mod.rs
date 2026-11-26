//! The parametric g-formula for time-varying treatments and confounders on
//! person-period data.
//!
//! # Algorithm
//!
//! The data are one row per person and period `k = 0, 1, ..., K - 1`, with
//! baseline variables `V`, time-varying variables `L_k = (L_k1, ..., L_kp)` in a
//! fixed order (covariates and treatments alike), and for a survival outcome an
//! event indicator `Y_k` and optionally a competing event `D_k`. A person's rows
//! end at the first event or at censoring. Write `H_kj` for the history that
//! variable `j` of period `k` may depend on: `V`, every variable of earlier
//! periods and variables `1..j-1` of period `k`, through the terms of
//! [`Term`] (lags, cumulative averages, lagged cumulative averages, products).
//!
//! Models, fitted by [`epistat_glm`] as gfoRmula fits them with `glm`:
//!
//! ```text
//! binary L_kj:   logit P(L_kj = 1 | H_kj) = x(H_kj)' beta_j         rows with k >= 1
//! normal L_kj:   L_kj = x(H_kj)' beta_j + e,  e ~ N(0, s_j^2)        rows with k >= 1
//! categorical:   log P(L_kj = c) / P(L_kj = 0) = x(H_kj)' beta_jc    rows with k >= 1
//! competing D:   logit P(D_k = 1 | H_k)   = x(H_k)' gamma            all rows
//! outcome Y:     logit P(Y_k = 1 | H_k, D_k = 0) = x(H_k)' alpha     rows with D_k = 0
//! ```
//!
//! with `s_j^2 = sum (L - fitted)^2 / n` (the residual mean square over the
//! number of rows, gfoRmula's `rmse`) and `H_k` the whole history of period
//! `k`. A [`Restriction`] fits a covariate's model only to the rows where it
//! applies. For an outcome at the end of follow-up the outcome model is fitted
//! to the rows of the last period, logistic for [`Outcome::BinaryEnd`] and
//! linear for [`Outcome::ContinuousEnd`].
//!
//! Simulation. Each simulated person starts from the period-0 row of an
//! observed person: every person once by default, or `m` persons drawn with
//! replacement. For each strategy, and `k = 0..K-1`, the variables of period
//! `k` are taken in order. At `k = 0` a variable keeps its observed value; at
//! `k >= 1` it is drawn from its model given the simulated history (a binary
//! variable is 1 when a uniform falls below its probability, a categorical one
//! is the first class whose cumulative probability exceeds a uniform, a normal
//! one is its
//! mean plus `s_j` times a standard normal, cut to the range observed in the
//! data unless [`GFormulaOptions::truncate`] is off), or set as its
//! restriction says where the restriction does not apply. The intervention's
//! rule for the variable, if any at period `k`, then replaces the value: a set
//! value, a schedule, a threshold `min(max(x, lower), upper)` or a function of
//! the history and the natural value. For a survival outcome, with `p_d` and
//! `p_y` the probabilities of the two models on the history of period `k`,
//!
//! ```text
//! risk_k = risk_{k-1} + p_y,k (1 - p_d,k) S_{k-1},      S_k = S_{k-1} (1 - p_y,k)(1 - p_d,k)
//! ```
//!
//! with `risk_{-1} = 0` and `S_{-1} = 1`, and every simulated person is followed
//! through all `K` periods; this is the risk of the event in the presence of the
//! competing event (the total effect). For an end-of-follow-up outcome the
//! person's value is the outcome model's mean at period `K - 1`. The estimate of
//! a strategy is the mean of its persons' values, and ratios and differences are
//! taken against a reference strategy, the natural course (no intervention) by
//! default.
//!
//! Every strategy is simulated from the same draws: simulated person `j` of a
//! run with seed `s` draws from the stream seeded with `replicate_seed(s, j + 2)`
//! of `epistat_rng`, which every strategy restarts, and every variable takes one
//! draw at every period `k >= 1` whether or not it is restricted or intervened
//! on (a uniform for a binary or categorical variable, a Box-Muller normal for
//! a normal one).
//! The draws of a person therefore do not depend on the order in which persons
//! are simulated, and the results do not depend on the number of threads.
//!
//! # Monte Carlo error and the bootstrap
//!
//! The Monte Carlo standard error of an estimate is the standard deviation of
//! the persons' values over the square root of the number simulated, of the
//! persons' differences for a difference (the strategies share their draws), and
//! for a ratio `R = a / b` that of `a_i - R b_i` over `b`. When `m` persons are
//! drawn with replacement this is the standard error of the simulation given the
//! fitted models; when every person is simulated once it is an upper bound, since
//! the persons' baseline differences are part of the estimand and not of the
//! simulation error. Means and variances are accumulated per block of 512
//! persons by Welford's update and merged in block order by the formula of Chan,
//! Golub and LeVeque.
//!
//! The nonparametric bootstrap resamples persons with replacement
//! (`epistat_rng::resample_counts` with the seed `replicate_seed(r_b, 0)`, where
//! `r_b = replicate_seed(seed, b + 1)`), refits every model with the counts as
//! prior weights, which equals fitting to the resampled rows, and simulates
//! again from the resampled persons with the seed `r_b` (the point estimate uses
//! `replicate_seed(seed, 0)`). The cut to the observed range keeps the range of
//! the original data, as gfoRmula does. The standard error is the standard
//! deviation of the replicates, and the interval is the percentile interval
//! (type 7 quantiles, gfoRmula's default) or `estimate -/+ z se`. A replicate in
//! which a model cannot be fitted is left out and counted in
//! [`GFormula::failed_replicates`].
//!
//! The nonparametric observed risk, against which the natural course is
//! checked, is `risk_k = risk_{k-1} + h_k (1 - d_k) S_{k-1}` with `h_k` the
//! proportion of events among the rows of period `k` without a competing event
//! and `d_k` the proportion of competing events among all rows of period `k`;
//! for an end-of-follow-up outcome it is the mean of the observed outcomes at
//! the last period.
//!
//! With a [`Censoring`] model the observed risk is weighted by the inverse
//! probability of remaining uncensored, as gfoRmula's "IP weighted risk": a
//! logistic model of censoring `p_c` is fitted on every row, a row that is not
//! censored has the weight `w_r = prod over the person's rows s <= r of 1 / (1
//! - p_c(s))`, a censored row the weight 0, and `h_k` and `d_k` are the
//! weighted proportions among the same rows. The censored rows leave the
//! outcome and competing-event models, as rows without an observed outcome.
//!
//! # Departures from gfoRmula
//!
//! gfoRmula applies the interventions of a period after all its covariates are
//! drawn, so a covariate after an intervened variable in the order is drawn from
//! the variable's natural value. Here a rule is applied as soon as its variable is
//! drawn, so later variables see the intervened value. The two agree when the
//! intervened variables are the last of each period, the usual layout. A person
//! counts as intervened on in a period when a rule changed a value at all, where
//! gfoRmula needs a change above `1e-6`. A model may read the current period only
//! of variables before it in the order (gfoRmula would read the previous period's
//! value of a later one); anything else is an error. A lag before period 0 is 0
//! (without `baseline_lags`), which for a categorical variable is its reference
//! class, as gfoRmula sets it.
//!
//! Not built: zero-inflated, bounded and truncated normal covariates,
//! categorical covariates with time as their only term (gfoRmula's
//! `categorical time`); restrictions on the outcome and competing-event models;
//! truncation of the censoring weights (`ipw_cutoff_quantile`) and censoring
//! weights for an outcome at the end of follow-up; the hazard ratio of a pair
//! of strategies; continuous outcomes over time.
//!
//! # Edge cases
//!
//! Every person's rows must be contiguous, at periods `0, 1, 2, ...` in order,
//! with baseline values that do not change, finite time-varying values, binary
//! variables of 0 or 1, and no rows after an event. A model that does not
//! converge or has an aliased column is an error (in a bootstrap replicate, a
//! failed replicate), as is a function of a restriction or an intervention that
//! gives a value that is not finite. The ratio is `None` where the reference is
//! 0, and its bootstrap limits are `None` when a replicate has a reference of 0.
//! The Monte Carlo standard errors are `None` for one simulated person, and the
//! bootstrap's for fewer than two successful replicates. The observed risk is
//! `None` from the first period without rows.
//!
//! # References
//!
//! 1. Robins J. A new approach to causal inference in mortality studies with a
//!    sustained exposure period: application to control of the healthy worker
//!    survivor effect. Math Model. 1986;7(9-12):1393-512.
//! 2. Taubman SL, Robins JM, Mittleman MA, Hernán MA. Intervening on risk
//!    factors for coronary heart disease: an application of the parametric
//!    g-formula. Int J Epidemiol. 2009;38(6):1599-611.
//! 3. Young JG, Cain LE, Robins JM, O'Reilly EJ, Hernán MA. Comparative
//!    effectiveness of dynamic treatment regimes: an application of the
//!    parametric g-formula. Stat Biosci. 2011;3(1):119-43.
//! 4. Daniel RM, De Stavola BL, Cousens SN. gformula: estimating causal effects
//!    in the presence of time-varying confounding or mediation using the
//!    g-computation formula. Stata J. 2011;11(4):479-517.
//! 5. Keil AP, Edwards JK, Richardson DB, Naimi AI, Cole SR. The parametric
//!    g-formula for time-to-event data: intuition and a worked example.
//!    Epidemiology. 2014;25(6):889-97.
//! 6. McGrath S, Lin V, Zhang Z, Petito LC, Logan RW, Hernán MA, Young JG.
//!    gfoRmula: an R package for estimating the effects of sustained treatment
//!    strategies via the parametric g-formula. Patterns. 2020;1(3):100008.
//! 7. Young JG, Stensrud MJ, Tchetgen Tchetgen EJ, Hernán MA. A causal framework
//!    for classical statistical estimands in failure-time settings with
//!    competing events. Stat Med. 2020;39(8):1199-236.
//! 8. Chan TF, Golub GH, LeVeque RJ. Algorithms for computing the sample
//!    variance: analysis and recommendations. Am Stat. 1983;37(3):242-7.

mod fit;
mod history;
mod prepare;
mod simulate;

use epistat_inference::interval::{self, Percentile};
use epistat_rng::replicate_seed;
use rayon::prelude::*;

use crate::error::{Error, Result};
pub use history::{History, Term};

/// The person-period table, one row per person and period.
#[derive(Debug, Clone, Copy)]
pub struct PersonPeriods<'a> {
    /// The person of each row. A person's rows are contiguous.
    pub id: &'a [u64],
    /// The period of each row, `0, 1, 2, ...` within a person.
    pub period: &'a [usize],
    /// Variables that do not change within a person, one slice of rows each.
    pub baseline: &'a [&'a [f64]],
    /// The time-varying variables, one slice of rows each, in the order in
    /// which they are simulated within a period: confounders before the
    /// treatments they affect.
    pub time_varying: &'a [&'a [f64]],
}

/// The distribution of a time-varying covariate given its history.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CovariateKind {
    /// 0 or 1, by logistic regression.
    Binary,
    /// Normal around a linear regression, with the residual root mean square
    /// as its standard deviation.
    Normal,
    /// One of `levels` classes coded `0, 1, ..., levels - 1`, by multinomial
    /// logistic regression with class 0 the reference (gfoRmula's
    /// `categorical`, fitted by `nnet::multinom`). Terms read its classes
    /// through [`Term::Is`].
    Categorical(usize),
}

/// What a restricted covariate takes where its model does not apply.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Otherwise {
    /// A fixed value.
    Value(f64),
    /// Its value at the previous period.
    CarryForward,
}

/// A condition on the history under which a covariate is drawn from its
/// model, as gfoRmula's `restrictions`.
pub type Condition = Box<dyn Fn(&History<'_>) -> bool + Send + Sync>;

/// Where a covariate's model applies; elsewhere it is set without a draw from
/// the model, and the model is fitted only to the rows where it applies.
pub enum Restriction {
    /// A binary covariate that stays 1 once it is 1: modelled while its value
    /// at the previous period is 0, carried forward otherwise (gfoRmula's
    /// `absorbing`).
    Absorbing,
    /// Modelled where `applies` is true; `otherwise` gives the value elsewhere.
    When {
        /// Reads the history of the person at the period, with the current
        /// values of the variables before this one visible.
        applies: Condition,
        /// The value where the model does not apply.
        otherwise: Otherwise,
    },
}

/// A time-varying covariate's model.
pub struct Covariate {
    /// Binary or normal.
    pub kind: CovariateKind,
    /// The terms of the linear predictor. They may read the current period
    /// only of variables before this one.
    pub terms: Vec<Term>,
    /// Where the model applies; everywhere when `None`.
    pub restriction: Option<Restriction>,
}

/// The competing event of a survival outcome.
#[derive(Debug, Clone)]
pub struct Competing<'a> {
    /// Whether the competing event happened in the row's period.
    pub event: &'a [bool],
    /// The terms of its logistic model.
    pub terms: Vec<Term>,
}

/// Censoring of a survival outcome, observed as a row of its own (gfoRmula's
/// `censor_name` and `censor_model`).
#[derive(Debug, Clone)]
pub struct Censoring<'a> {
    /// Whether the person is censored in the row's period, before the outcome
    /// of the period. A censored row is the person's last, has neither the
    /// event nor the competing event, and does not enter the outcome and
    /// competing-event models; its covariates do enter their models.
    pub censored: &'a [bool],
    /// The terms of the logistic model of censoring, fitted on every row with
    /// the whole history of the period visible.
    pub terms: Vec<Term>,
}

/// The outcome and its model.
#[derive(Debug, Clone)]
pub enum Outcome<'a> {
    /// An event in discrete time: the risk by each period.
    Survival {
        /// Whether the event happened in the row's period. A row with the
        /// competing event does not enter the outcome model.
        event: &'a [bool],
        /// The terms of the logistic model of the event.
        terms: Vec<Term>,
        /// A competing event, which ends follow-up without the event.
        competing: Option<Competing<'a>>,
        /// Censoring with its model, for the observed risk weighted by the
        /// inverse probability of remaining uncensored. `None` treats a
        /// person's rows as ending at censoring after the last of them.
        censoring: Option<Censoring<'a>>,
    },
    /// A binary outcome measured at the last period: its probability. Values at
    /// other periods are not read; NaN marks a missing outcome.
    BinaryEnd {
        /// The outcome of each row.
        value: &'a [f64],
        /// The terms of its logistic model.
        terms: Vec<Term>,
    },
    /// A continuous outcome measured at the last period: its mean. Values at
    /// other periods are not read; NaN marks a missing outcome.
    ContinuousEnd {
        /// The outcome of each row.
        value: &'a [f64],
        /// The terms of its linear model.
        terms: Vec<Term>,
    },
}

/// A function of the history and the natural value of the variable, giving
/// its value under a dynamic strategy.
pub type DynamicRule = Box<dyn Fn(&History<'_>, f64) -> f64 + Send + Sync>;

/// How a rule sets its variable.
pub enum Action {
    /// The same value at every period (gfoRmula's `static` with one value).
    Set(f64),
    /// The value at each period, at least as many as there are periods.
    Schedule(Vec<f64>),
    /// The natural value cut to `[lower, upper]` (gfoRmula's `threshold`).
    Threshold {
        /// The smallest value allowed.
        lower: f64,
        /// The largest value allowed.
        upper: f64,
    },
    /// A function of the history, with the current values of the variables
    /// before this one visible, and of the natural value.
    Dynamic(DynamicRule),
}

/// How one variable is set under a strategy.
pub struct Rule {
    /// The time-varying variable.
    pub variable: usize,
    /// How it is set.
    pub action: Action,
    /// The periods at which the rule applies, every period when `None`; at the
    /// others the variable keeps its natural value. Periods after the last one
    /// simulated have no effect.
    pub periods: Option<Vec<usize>>,
}

/// A strategy: rules for one or more variables.
pub struct Intervention {
    /// At most one rule per variable.
    pub rules: Vec<Rule>,
}

impl Intervention {
    /// The strategy that sets `variable` to `value` at every period.
    #[must_use]
    pub fn set(variable: usize, value: f64) -> Self {
        Self {
            rules: vec![Rule {
                variable,
                action: Action::Set(value),
                periods: None,
            }],
        }
    }
}

/// How the bootstrap interval is formed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntervalMethod {
    /// The percentile interval of the replicates, with type 7 quantiles.
    #[default]
    Percentile,
    /// `estimate -/+ z se` with the bootstrap standard error.
    Normal,
}

/// What the g-formula does besides the data and the models.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GFormulaOptions {
    /// The number of persons simulated, drawn with replacement from the
    /// observed (or resampled) persons; each person once when `None`, the
    /// default, as gfoRmula's default `nsimul`.
    pub simulated: Option<usize>,
    /// The number of periods simulated; the number observed when `None`.
    pub periods: Option<usize>,
    /// The number of bootstrap replicates, 0 (the default) for none.
    pub bootstrap: usize,
    /// The seed of the simulation and the bootstrap.
    pub seed: u64,
    /// One minus the level of the bootstrap limits. The default is 0.05.
    pub alpha: f64,
    /// How the bootstrap limits are formed.
    pub interval: IntervalMethod,
    /// The strategy that ratios and differences are taken against: 0 for the
    /// natural course (the default), `i` for intervention `i` (from 1).
    pub reference: usize,
    /// Whether lags before period 0 take the value at period 0 (gfoRmula's
    /// `baselags`); 0 when false, the default.
    pub baseline_lags: bool,
    /// Whether simulated normal covariates are cut to the range observed in
    /// the data (gfoRmula's `sim_trunc`). The default is true.
    pub truncate: bool,
    /// Whether the simulated courses of the point estimate are returned in
    /// [`GFormula::courses`] (gfoRmula's `sim_data`), one row per simulated
    /// person and period for each strategy. The default is false: the memory
    /// is `strategies * simulated * periods * (variables + 6)` values.
    pub keep_simulated: bool,
}

impl Default for GFormulaOptions {
    fn default() -> Self {
        Self {
            simulated: None,
            periods: None,
            bootstrap: 0,
            seed: 20_261_124,
            alpha: 0.05,
            interval: IntervalMethod::Percentile,
            reference: 0,
            baseline_lags: false,
            truncate: true,
            keep_simulated: false,
        }
    }
}

/// An estimate with its Monte Carlo error and its bootstrap limits.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Estimate {
    /// The estimate, `None` where it is undefined.
    pub estimate: Option<f64>,
    /// The standard error of the simulation (see the module documentation).
    pub monte_carlo_se: Option<f64>,
    /// The bootstrap standard error.
    pub se: Option<f64>,
    /// The lower bootstrap limit.
    pub lower: Option<f64>,
    /// The upper bootstrap limit.
    pub upper: Option<f64>,
}

/// The results of one strategy.
#[derive(Debug, Clone, PartialEq)]
pub struct Strategy {
    /// The risk by each period, or the mean at the end of follow-up.
    pub value: Vec<Estimate>,
    /// The ratio to the reference strategy.
    pub ratio: Vec<Estimate>,
    /// The difference from the reference strategy.
    pub difference: Vec<Estimate>,
    /// The proportion of simulated persons whose value of some variable a rule
    /// changed at some period.
    pub intervened_persons: f64,
    /// The proportion of simulated person-periods in which a rule changed a
    /// value.
    pub intervened_person_periods: f64,
}

/// A fitted model.
#[derive(Debug, Clone, PartialEq)]
pub struct FittedModel {
    /// The coefficients, in the order of the terms.
    pub coefficients: Vec<f64>,
    /// The residual root mean square `sqrt(sum (y - fitted)^2 / n)`, for normal
    /// covariates and a continuous outcome.
    pub rmse: Option<f64>,
    /// The number of rows the model was fitted to.
    pub rows: usize,
}

/// The fitted models.
#[derive(Debug, Clone, PartialEq)]
pub struct Models {
    /// One per time-varying covariate, `None` when only one period is
    /// simulated and no covariate is drawn.
    pub covariates: Vec<Option<FittedModel>>,
    /// The outcome model.
    pub outcome: FittedModel,
    /// The competing-event model.
    pub competing: Option<FittedModel>,
    /// The censoring model, fitted once to the data for the observed risk.
    pub censoring: Option<FittedModel>,
}

/// The simulated course of one strategy: one row per simulated person and
/// period, in the order of the persons and then of the periods.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SimulatedCourse {
    /// The simulated person, from 0.
    pub person: Vec<usize>,
    /// The observed person (its position in the order of the data) whose
    /// period-0 row the simulated person starts from.
    pub source: Vec<usize>,
    /// The period.
    pub period: Vec<usize>,
    /// The value of each time-varying variable, one column per variable in
    /// the order of [`PersonPeriods::time_varying`], after the strategy's rules.
    pub values: Vec<Vec<f64>>,
    /// The probability of the event in the period given the history and no
    /// competing event, for a survival outcome; the outcome model's mean at the
    /// last period for an outcome at the end of follow-up, NaN at the others.
    pub outcome_probability: Vec<f64>,
    /// The probability of the competing event in the period, 0 without one and
    /// NaN for an outcome at the end of follow-up.
    pub competing_probability: Vec<f64>,
    /// The person's risk by the end of the period (gfoRmula's `poprisk`), or
    /// the mean at the last period for an outcome at the end of follow-up, NaN
    /// at the others. The estimate of the strategy is its mean over the
    /// persons.
    pub value: Vec<f64>,
}

impl SimulatedCourse {
    fn new(variables: usize) -> Self {
        Self {
            values: vec![Vec::new(); variables],
            ..Self::default()
        }
    }

    fn extend(&mut self, other: &Self) {
        self.person.extend_from_slice(&other.person);
        self.source.extend_from_slice(&other.source);
        self.period.extend_from_slice(&other.period);
        for (a, b) in self.values.iter_mut().zip(&other.values) {
            a.extend_from_slice(b);
        }
        self.outcome_probability
            .extend_from_slice(&other.outcome_probability);
        self.competing_probability
            .extend_from_slice(&other.competing_probability);
        self.value.extend_from_slice(&other.value);
    }
}

/// The result of [`gformula`].
#[derive(Debug, Clone, PartialEq)]
pub struct GFormula {
    /// The periods the values refer to: `0..K` for a survival outcome, `K - 1`
    /// for an outcome at the end of follow-up.
    pub periods: Vec<usize>,
    /// The nonparametric observed risk (or mean) at each of those periods.
    pub observed: Vec<Option<f64>>,
    /// The natural course first, then the interventions in the order given.
    pub strategies: Vec<Strategy>,
    /// The fitted models.
    pub models: Models,
    /// The number of persons simulated for each strategy.
    pub simulated: usize,
    /// The bootstrap replicates asked for.
    pub replicates: usize,
    /// The replicates left out because a model could not be fitted.
    pub failed_replicates: usize,
    /// The simulated courses of the point estimate, the natural course first,
    /// when [`GFormulaOptions::keep_simulated`] asks for them.
    pub courses: Option<Vec<SimulatedCourse>>,
}

/// The parametric g-formula: fits the models of `covariates` (one per
/// time-varying variable of `data`) and of `outcome`, simulates the natural
/// course and each of `interventions`, and gives the risks or means with their
/// ratios and differences, Monte Carlo errors and, with
/// [`GFormulaOptions::bootstrap`], bootstrap limits.
///
/// # Errors
///
/// [`Error::Input`] for data or models that cannot be used (see the module
/// documentation), [`Error::NoConvergence`] for a model that does not
/// converge on the data.
pub fn gformula(
    data: &PersonPeriods<'_>,
    covariates: &[Covariate],
    outcome: &Outcome<'_>,
    interventions: &[Intervention],
    options: &GFormulaOptions,
) -> Result<GFormula> {
    let prepared = prepare::prepare(data, covariates, outcome, interventions, options)?;
    let n = prepared.persons();
    let ones = vec![1.0; n];
    let fitted = fit::fit_all(&prepared, covariates, outcome, &ones)?;
    let point_seed = replicate_seed(options.seed, 0);
    let sources = simulate::sources(&ones, options.simulated, point_seed);
    let mut point = simulate::run(
        &prepared,
        covariates,
        outcome,
        &fitted,
        &sources,
        point_seed,
        options.keep_simulated,
    )?;

    let replicates: Vec<Option<Vec<Vec<f64>>>> = (0..options.bootstrap)
        .into_par_iter()
        .map(|b| -> Result<Option<Vec<Vec<f64>>>> {
            let seed = replicate_seed(options.seed, b as u64 + 1);
            let counts: Vec<f64> = epistat_rng::resample_counts(n, replicate_seed(seed, 0))
                .into_iter()
                .map(f64::from)
                .collect();
            let Ok(fitted) = fit::fit_all(&prepared, covariates, outcome, &counts) else {
                return Ok(None);
            };
            let sources = simulate::sources(&counts, options.simulated, seed);
            let run = simulate::run(
                &prepared, covariates, outcome, &fitted, &sources, seed, false,
            )?;
            Ok(Some(run.means()))
        })
        .collect::<Result<_>>()?;
    let failed = replicates.iter().filter(|r| r.is_none()).count();
    let replicates: Vec<Vec<Vec<f64>>> = replicates.into_iter().flatten().collect();

    let strategies = summarise(&point, &replicates, prepared.reference, options)?;
    let mut models = fitted.report();
    let mut observed = prepared.observed.clone();
    if let Outcome::Survival {
        event,
        competing,
        censoring: Some(censoring),
        ..
    } = outcome
    {
        let model = fit::fit_censoring(&prepared, censoring, &ones)?;
        observed = prepare::weighted_observed_risk(
            &prepared,
            event,
            competing.as_ref().map(|c| c.event),
            censoring,
            &model,
        );
        models.censoring = Some(model.model);
    }
    Ok(GFormula {
        periods: prepared.reported_periods(),
        observed,
        strategies,
        models,
        simulated: sources.len(),
        replicates: options.bootstrap,
        failed_replicates: failed,
        courses: point.courses.take(),
    })
}

/// The estimates of each strategy with their Monte Carlo errors and bootstrap
/// limits. `replicates[b][s][t]` is the mean of strategy `s` at reported
/// period `t` in replicate `b`.
fn summarise(
    point: &simulate::Run,
    replicates: &[Vec<Vec<f64>>],
    reference: usize,
    options: &GFormulaOptions,
) -> Result<Vec<Strategy>> {
    let strategies = point.value.len();
    let periods = point.value[0].len();
    let limits = |estimate: f64, values: Option<Vec<f64>>| -> Result<Estimate> {
        let mut e = Estimate {
            estimate: Some(estimate),
            ..Estimate::default()
        };
        let Some(mut values) = values else {
            return Ok(e);
        };
        if values.len() < 2 {
            return Ok(e);
        }
        let b = values.len() as f64;
        let mean = values.iter().sum::<f64>() / b;
        let se = (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (b - 1.0)).sqrt();
        e.se = Some(se);
        let (lower, upper) = match options.interval {
            IntervalMethod::Percentile => {
                interval::percentile(&mut values, options.alpha, Percentile::Type7)
            }
            IntervalMethod::Normal => interval::normal(estimate, se, options.alpha),
        }
        .map_err(|e| Error::input(e.to_string()))?;
        e.lower = Some(lower);
        e.upper = Some(upper);
        Ok(e)
    };
    let mut out = Vec::with_capacity(strategies);
    for s in 0..strategies {
        let mut value = Vec::with_capacity(periods);
        let mut ratio = Vec::with_capacity(periods);
        let mut difference = Vec::with_capacity(periods);
        for t in 0..periods {
            let m = &point.value[s][t];
            let r = &point.value[reference][t];
            let d = &point.difference[s][t];
            let reps = |f: &dyn Fn(&[Vec<f64>]) -> Option<f64>| -> Option<Vec<f64>> {
                if replicates.is_empty() {
                    return None;
                }
                replicates.iter().map(|rep| f(rep)).collect()
            };
            let mut v = limits(m.mean, reps(&|rep| Some(rep[s][t])))?;
            v.monte_carlo_se = m.standard_error();
            value.push(v);
            let mut dd = limits(
                m.mean - r.mean,
                reps(&|rep| Some(rep[s][t] - rep[reference][t])),
            )?;
            dd.monte_carlo_se = d.standard_error();
            difference.push(dd);
            if r.mean == 0.0 {
                ratio.push(Estimate::default());
            } else {
                let q = m.mean / r.mean;
                let mut rr = limits(
                    q,
                    reps(&|rep| (rep[reference][t] != 0.0).then(|| rep[s][t] / rep[reference][t])),
                )?;
                rr.monte_carlo_se = simulate::ratio_standard_error(m, r, d);
                ratio.push(rr);
            }
        }
        out.push(Strategy {
            value,
            ratio,
            difference,
            intervened_persons: point.intervened_persons[s],
            intervened_person_periods: point.intervened_person_periods[s],
        });
    }
    Ok(out)
}
