//! Cox regression for a multi-state process: one hazard per transition, with
//! coefficients that are specific to a transition or shared between some, the
//! cumulative hazards of a covariate profile, and the transition probabilities
//! they give.
//!
//! [`fit`] is `coxph` with a multi-state outcome, `Surv(start, stop, state)`
//! with `id` and `istate`, and the stacked `coxph` of `mstate` that
//! `msprep` prepares. [`MsModel::msfit`] is `mstate::msfit` and [`probtrans`]
//! is `mstate::probtrans`.
//!
//! # The model
//!
//! The process has states `1..=n_codes`. Each row of a subject is an interval
//! `(start, stop]` spent in one state, the state the row starts in, which
//! ends with status 0 (censored) or the state entered. The hazard of the
//! transition `k` from state `a` to state `b` is
//!
//! ```text
//! h_k(t | x) = h_0,g(k)(t) exp(x_k' beta),
//! ```
//!
//! for a subject in state `a` at `t`, where `g(k)` is the baseline hazard
//! the transition has and `x_k` is the covariate vector of the transition.
//! Each coefficient of `beta` belongs to a term. A term is one column of the
//! covariates, or the constant 1, and the set of transitions it applies to,
//! so a coefficient is specific to one transition or shared by several. The
//! stacked design has one column per term, which holds the covariate in the
//! rows of the transitions of the term and 0 in the others.
//!
//! The partial likelihood is the sum of the partial likelihoods of the
//! transitions, with the transitions that share a baseline hazard in one
//! stratum. It is the likelihood of the stacked data, fitted by [`crate::cox::fit`]:
//! every row of a subject is repeated once for each transition that leaves the
//! state the row starts in, with the event indicator 1 when the row ends in the
//! state that transition enters, and 0 otherwise, so a move to another state is
//! a censoring for it. This is the cause-specific hazard model for every state
//! with competing exits, which is what `coxph` does with a multi-state
//! outcome and what `msprep` followed by `coxph(... + strata(trans))` does.
//!
//! # Terms and baselines
//!
//! [`MsTerms::Separate`] gives every covariate a coefficient in every transition,
//! as `coxph(Surv(...) ~ x)` does for a multi-state outcome. [`MsTerms::Common`]
//! gives it one coefficient for all, the `common` of `coxph`'s formula list.
//! [`MsTerms::Custom`] takes the terms, which is the formula list with one
//! formula for each set of transitions. [`MsBaseline::Separate`] gives each
//! transition its own baseline hazard. [`MsBaseline::Shared`] gives all
//! transitions one, and [`MsBaseline::Groups`] groups them. `coxph`'s `shared`
//! is a shared baseline with a term that is the constant 1 for each transition
//! but the first, the proportional baselines `ph(a:b/c:d)`, and it is built here
//! with [`MsTerm::indicator`].
//!
//! The transitions are numbered from 0 in the order of the state they leave and
//! then the state they enter, which is the numbering of `mstate`'s transition
//! matrix. [`observed_transitions`] lists those the data has, and [`MsOptions`]
//! takes a list, in which case an event that is not on it is a censoring for the
//! transitions out of its state.
//!
//! # Rows
//!
//! The data must have an id column. The times go through the tie rule of
//! [`TimeFix`], `aeqSurv` by default, then
//! [`survcheck`] checks the follow-up of each subject under
//! [`Policy::Strict`], so overlapping rows, jumps and teleports are errors
//! and gaps are allowed, as in `coxph`. The state a row starts in is the one
//! `survcheck` resolves: the given `istate`, or the state the earlier rows of the
//! subject led to. A row whose state is unknown is an error, and so is a row
//! that ends in the state it starts in. Rows of weight 0 are not used. A
//! subject's rows are one cluster, and the covariance has a robust version
//! with them, the way `coxph` reports a robust variance when it is given an
//! `id`. [`MsOptions::robust`] is true by default as in `coxph`. A cluster
//! column in the data is used in place of the id.
//!
//! # Predictions
//!
//! For a covariate profile `z`, [`MsModel::msfit`] gives the cumulative
//! hazard `H_k(t) = theta_k H_0,g(k)(t)` of every transition at every event
//! time of the model and at the last follow-up time, where `H_0,g` is the
//! baseline of [`basehaz`](crate::cox::predict::basehaz) and `theta_k = exp((x_k - c)' beta)`
//! with `c` the reference of the baseline. This is `msfit` of `mstate`,
//! whose model has one stratum for each transition. The variance is the delta
//! method of Tsiatis \[4\] as in [`cox::predict`](mod@crate::cox::predict),
//!
//! ```text
//! cov(H_a, H_b) = theta_a theta_b ( g_a' V g_b + [g(a) = g(b)] sum of v(s) ),
//! g_k(t) = H_0(t) (x_k - c) - sum over s <= t of xbar_k(s),
//! ```
//!
//! with `V` the covariance of the coefficients, model-based as `mstate` uses it
//! and by default here, and `sum of v(s)` the variance of the baseline hazard
//! with `beta` known. The covariances between transitions come from the
//! coefficients they share, and from the baseline they share, which `mstate`
//! cannot have because each of its transitions has its own stratum.
//!
//! [`probtrans`] takes those cumulative hazards and forms the transition
//! matrix by the product integral of Aalen and Johansen \[3\]
//!
//! ```text
//! P(s, t) = prod over event times u in (s, t] of (I + dA(u)),
//! ```
//!
//! where `dA(u)` has the increment of the cumulative hazard of the
//! transition `a -> b` in its entry `(a, b)` and minus the row sum on the
//! diagonal. The variance of `P` is the recursion of Andersen et al. \[2\] in
//! the form `mstate` uses by default (`method = "aalen"`):
//!
//! ```text
//! P_j      = P_{j-1} (I + dA_j),
//! var P_j  = (B' x I) var P_{j-1} (B x I)  +  (I x P_j) var dA_j (I x P_j'),
//! ```
//!
//! with `B = I + dA_j`, `x` the Kronecker product, `var` the covariance matrix
//! of the entries of the matrix in column order, and `var dA_j` the increment of
//! the covariance matrix of the cumulative hazards between the event times,
//! completed on the diagonal because the rows of `dA` sum to 0.
//!
//! [`ProbTransMethod::Greenwood`] is `method = "greenwood"` of `mstate`. It takes
//! the derivative of the product at the matrix before the step instead of after
//! it, `(I x P_{j-1}) var dA_j (I x P_{j-1}')` in the second term.
//!
//! # Probabilities at a fixed horizon
//!
//! With [`ProbTransDirection::FixedHorizon`] the product runs from `predt`
//! backwards over the event times up to `predt`, `P(t, predt) = prod (I + dA(u))`
//! over `u` in `(t, predt]` in the order of the times, so that row `k` of the
//! result is the probability of being in each state at `predt` for a subject
//! in each state at the time `t_k` of the row. The recursion for the variance
//! is the one of `mstate`,
//!
//! ```text
//! P_j     = (I + dA_j) P_{j+1},
//! var P_j = (I x B) var P_{j+1} (I x B')  +  (P_{j+1}' x B) var dA_j (P_{j+1} x B'),
//! ```
//!
//! with `B = I + dA_j`, and `P_{j+1}' x I` in place of `P_{j+1}' x B` for the
//! Greenwood form. [`ProbTransOptions::covariance`] gives the whole covariance
//! matrix of the probabilities at each time. In the fixed horizon `mstate`
//! returns that array one slice off the rows of its result, the slice `k + 1`
//! holding the covariance of the matrix of row `k` (its standard errors in row
//! `k` are the square roots of that slice's diagonal, which the tests assert).
//! The covariances here are aligned to the rows.
//!
//! # Several profiles, and profiles that differ between transitions
//!
//! [`MsModel::msfit_many`] makes the cumulative hazards of several profiles with
//! one computation of the baseline hazards, and [`probtrans_many`] the
//! probabilities of several. A profile is [`MsProfile::Same`], the values of
//! the covariate columns for every transition, or [`MsProfile::ByTransition`],
//! a vector for each transition, the `newdata` of `msfit` with one row for each
//! transition. A coefficient of a term takes the value of its column in the
//! vector of each transition it applies to.
//!
//! # Time-varying covariates
//!
//! A covariate that changes during follow-up is given as several rows for a
//! subject, each with the values it has in its interval. Nothing else is
//! needed: the covariates are per row, the stacked data repeat each row for the
//! transitions that leave its state, the risk sets of the partial likelihood
//! are formed from the intervals, and a prediction is for a covariate profile
//! held fixed, as in `mstate`. A covariate that is a function of time alone is
//! not identified (every row at risk at a time has the same value), and one
//! that does not vary among those at risk in the events of a transition is
//! aliased with the baseline hazard, in any Cox model.
//!
//! # The Greenwood variance of the hazards
//!
//! [`MsVarType::Greenwood`] is `vartype = "greenwood"` of `mstate`'s `msfit`,
//! which `mstate` allows only for a model with the transitions as strata and no
//! coefficients. It is an error here for any other model, and without variances
//! it asks for nothing.
//!
//! # Weights
//!
//! `mstate`'s `msfit` stops with "cannot (yet) compute the result for a
//! weighted model". The model here can be weighted, the weights being those of
//! the data, and [`MsModel::msfit`] is then the cumulative hazard and the
//! variance of [`basehaz`](crate::cox::predict::basehaz) with the delta method
//! above. Each transition's hazard and variance equals `survfit` of `coxph` on
//! the stacked weighted data, with the model-based covariance of the
//! coefficients, which is what the tests compare. No R function gives the
//! covariances between transitions of a weighted model, and they are not
//! checked against R: the expression is the one of the unweighted model with
//! weighted sums, as in the cumulative hazards themselves.
//!
//! # Agreement with R
//!
//! The coefficients and covariances are those of `coxph` on the same rows, and
//! the cumulative hazards, their variances, the probabilities and their
//! standard errors those of `mstate` 0.3.3 on `ebmt3`, for both forms of the
//! variance and both directions, with time-varying covariates, and the hazards
//! of weighted models those of `survival` 3.8-12. The tests give the
//! tolerances.
//!
//! # References
//!
//! 1. Putter H, Fiocco M, Geskus RB. Tutorial in biostatistics: competing risks
//!    and multi-state models. Stat Med 2007;26(11):2389-2430. doi:10.1002/sim.2712
//! 2. Andersen PK, Borgan O, Gill RD, Keiding N. Statistical models based on
//!    counting processes. New York: Springer; 1993. Section IV.4.
//! 3. Aalen OO, Johansen S. An empirical transition matrix for non-homogeneous
//!    Markov chains based on censored observations. Scand J Stat 1978;5(3):141-150.
//! 4. Tsiatis AA. A large sample study of Cox's regression model. Ann Stat
//!    1981;9(1):93-108. doi:10.1214/aos/1176345335
//! 5. de Wreede LC, Fiocco M, Putter H. The mstate package for estimation and
//!    prediction in non- and semi-parametric multi-state and competing risks
//!    models. Comput Methods Programs Biomed 2010;99(3):261-274.
//!    doi:10.1016/j.cmpb.2010.01.001
//! 6. Therneau TM, Crowson C, Atkinson E. Multi-state models and competing
//!    risks. Vignette of the survival package, version 3.8-12.

mod hazard;
mod probtrans;
mod stack;

use epistat_linalg::{Mat, MatRef};

use crate::cox::{CoxModelFit, CoxOptions, Ties};
use crate::error::{Error, Result};
use crate::surv::{SurvBuf, SurvData};
use crate::survcheck::{Policy, survcheck};
use crate::timefix::TimeFix;

pub use hazard::{CumulativeHazards, MsFitOptions, MsProfile, MsVarType};
pub use probtrans::{
    ProbTrans, ProbTransDirection, ProbTransMethod, ProbTransOptions, probtrans, probtrans_many,
};

/// One transition of the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MsTransition {
    /// The state it leaves.
    pub from: u32,
    /// The state it enters.
    pub to: u32,
    /// The index of its baseline hazard among the baselines of the model, in
    /// the order of the group codes.
    pub baseline: usize,
    /// The number of rows of positive weight in which the transition happens.
    pub n_events: usize,
}

/// A coefficient: a column of the covariates, or the constant 1, in a set of
/// transitions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MsTerm {
    /// The column of the covariate matrix, or `None` for the constant 1.
    pub column: Option<usize>,
    /// The transitions the coefficient applies to, as indices into
    /// [`MsModel::transitions`].
    pub transitions: Vec<usize>,
}

impl MsTerm {
    /// A covariate column with one coefficient in the given transitions.
    #[must_use]
    pub fn covariate(column: usize, transitions: &[usize]) -> Self {
        Self {
            column: Some(column),
            transitions: transitions.to_vec(),
        }
    }

    /// The constant 1 in the given transitions, an indicator of them, as the
    /// proportional baselines of `coxph`'s `shared`.
    #[must_use]
    pub fn indicator(transitions: &[usize]) -> Self {
        Self {
            column: None,
            transitions: transitions.to_vec(),
        }
    }
}

/// The coefficients of the model.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum MsTerms {
    /// A coefficient for each covariate in each transition, covariate by
    /// covariate and within a covariate in the order of the transitions.
    #[default]
    Separate,
    /// One coefficient for each covariate, shared by all transitions.
    Common,
    /// These terms, in this order.
    Custom(Vec<MsTerm>),
}

/// The baseline hazards of the transitions.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum MsBaseline {
    /// A baseline hazard of its own for each transition.
    #[default]
    Separate,
    /// One baseline hazard for all transitions.
    Shared,
    /// One group code for each transition, in the order of the transitions.
    /// Transitions with the same code share a baseline hazard.
    Groups(Vec<u32>),
}

/// What [`fit`] takes besides the data and the covariates.
#[derive(Debug, Clone, PartialEq)]
pub struct MsOptions {
    /// How tied events enter the partial likelihood, as in [`CoxOptions`].
    /// `Efron`, as in `coxph`.
    pub ties: Ties,
    /// The rule that makes times equal when they differ by rounding error,
    /// applied before the follow-up is checked. The default is the rule of
    /// `survival::aeqSurv`, which `coxph` applies.
    pub timefix: TimeFix,
    /// The transitions of the model, from state and to state, in any order. The
    /// default is those the data has, [`observed_transitions`].
    pub transitions: Option<Vec<(u32, u32)>>,
    /// The coefficients.
    pub terms: MsTerms,
    /// The baseline hazards.
    pub baseline: MsBaseline,
    /// Whether to give the cluster-robust covariance, with the subject (or the
    /// cluster column of the data) as the cluster. True, as `coxph` does with
    /// an `id`.
    pub robust: bool,
    /// The starting values of the coefficients, one per term, as in
    /// [`CoxOptions`].
    pub start: Option<Vec<f64>>,
    /// The largest number of trial points, as in [`CoxOptions`].
    pub max_iterations: usize,
    /// The relative change of the log partial likelihood at which the iteration
    /// stops, as in [`CoxOptions`].
    pub tolerance: f64,
}

impl Default for MsOptions {
    fn default() -> Self {
        let cox = CoxOptions::default();
        Self {
            ties: cox.ties,
            timefix: cox.timefix,
            transitions: None,
            terms: MsTerms::Separate,
            baseline: MsBaseline::Separate,
            robust: true,
            start: None,
            max_iterations: cox.max_iterations,
            tolerance: cox.tolerance,
        }
    }
}

/// The result of [`fit`]: the coefficients of every term, the covariance, and
/// what the predictions need.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MsModel {
    /// The transitions, in the order of the state they leave and the state they
    /// enter.
    pub transitions: Vec<MsTransition>,
    /// The terms, one coefficient each, in the order of the columns of the
    /// fit.
    pub terms: Vec<MsTerm>,
    /// The number of states, `n_codes` of the data.
    pub n_states: u32,
    /// The Cox fit of the stacked rows. Its coefficients and covariance are by
    /// term. `n_rows` and `n_used` are those of the data, and the rows of the
    /// stacked data are counted in [`n_stacked`](Self::n_stacked).
    pub fit: CoxModelFit,
    /// The number of rows of the stacked data, a row for each transition that
    /// leaves the state of a row of positive weight.
    pub n_stacked: usize,
    /// The number of subjects.
    pub n_ids: usize,
    stacked: SurvBuf,
    design: Mat<f64>,
    n_covariates: usize,
    ties: Ties,
    last_time: f64,
}

impl MsModel {
    /// The coefficients of the terms, `None` for an aliased term, one per term.
    #[must_use]
    pub fn coefficients(&self) -> &[Option<f64>] {
        &self.fit.coefficients
    }

    /// The number of covariate columns the profile of a prediction has.
    #[must_use]
    pub fn n_covariates(&self) -> usize {
        self.n_covariates
    }
}

/// The transitions the data has among the rows of positive weight: the pairs of
/// the state a row starts in and the state it ends in, in the order of the
/// state they leave and the state they enter.
///
/// The times go through `timefix` and the follow-up is checked as in [`fit`].
///
/// # Errors
///
/// An input error as for [`fit`], for the data and the follow-up.
pub fn observed_transitions(data: &SurvData<'_>, timefix: TimeFix) -> Result<Vec<(u32, u32)>> {
    let (data, istate) = resolve(data, timefix)?;
    stack::observed(&data, &istate)
}

/// The data with the tie rule applied and the state each row starts in.
fn resolve<'a>(data: &SurvData<'a>, timefix: TimeFix) -> Result<(SurvData<'a>, Vec<u32>)> {
    let data = data.clone().with_timefix(timefix)?;
    let surv = data.followup()?;
    if surv.n_codes() < 2 {
        return Err(Error::not_applicable(
            "the status has one code, so there is one transition and no multi-state model; \
             use `cox::fit` for it",
        ));
    }
    let check = survcheck(&data)?;
    check.enforce(Policy::Strict)?;
    Ok((data, check.istate))
}

/// Fit the transition-specific Cox models of a multi-state process.
///
/// `data` has an id, `n_codes` states and, usually, a start column and an
/// `istate`. `x` has one row for each row of `data` and the covariates in its
/// columns. See the [module documentation](self) for the model, the rows and
/// the terms.
///
/// Without covariates and without terms, which is `x` with no column and
/// [`MsTerms::Separate`] or [`MsTerms::Common`], the model is the baseline
/// hazards only, and the predictions are the Nelson-Aalen type estimates of
/// each transition. [`MsModel::coefficients`] is then empty, and `fit` is the
/// [`crate::cox::fit`] of a design without columns.
///
/// # Errors
///
/// An input error when `data` has no id column, one status code, or
/// interval-censored times, when the follow-up of a subject has an overlap, a
/// jump or a teleport, when a row of positive weight starts in an unknown state
/// or ends in the state it starts in, when `x` has another number of rows than
/// `data` or a value that is not finite, when a transition is out of range, a
/// duplicate or leaves and enters one state, when a term names a column or a
/// transition that does not exist or two terms give one covariate in one
/// transition two coefficients, when `baseline` or `start` have the
/// wrong length, and as [`crate::cox::fit`] for its options.
pub fn fit(data: &SurvData<'_>, x: MatRef<'_, f64>, opts: &MsOptions) -> Result<MsModel> {
    stack::fit(data, x, opts)
}
