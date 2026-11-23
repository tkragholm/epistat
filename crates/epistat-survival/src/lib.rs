//! Survival and multi-state methods: Cox regression, the Fine-Gray model for
//! competing risks, the Aalen-Johansen estimator, state occupation in a
//! multi-state process and the Titman-Putter test of the Markov property.
//!
//! Most estimators read a [`SurvData`], the checked columns of a follow-up
//! table (exit and entry time, status, weight, strata, clusters, subject and
//! starting state), and a design matrix where they have covariates. They return
//! plain structs and give the same bits on the same input, whatever the number
//! of threads. Where an estimator bootstraps, the caller supplies the seeds,
//! and each replicate depends on its seed only, through [`epistat_rng`].
//!
//! # Contents
//!
//! - [`absrisk`]: the absolute risk of a cause from cause-specific Cox models, with
//!   an influence-function standard error and confidence limits, as riskRegression's
//!   `CSC` and `predict.CauseSpecificCox`.
//! - [`concordance`]: Harrell's C and its relatives for a numeric score, with weights,
//!   strata, delayed entry, time weights and an infinitesimal jackknife variance.
//! - [`cox`]: Cox regression on a design matrix, with Breslow or
//!   Efron or exact ties, delayed entry, strata, case weights, offsets, starting values
//!   and iteration limits, a cluster-robust variance, and the Wald, score and
//!   likelihood-ratio tests of the coefficients, its residuals as `residuals.coxph`,
//!   the test of proportional hazards of `cox.zph`, and time-transform
//!   covariates as `coxph(tt = )` ([`cox::fit_tt`]).
//! - [`clogit`]: conditional logistic regression for matched sets, as `clogit`,
//!   with the exact partial likelihood of [`cox`] by default.
//! - [`fine_gray`]: the Fine-Gray model on a design matrix, fitted by a sweep
//!   over the event times with censoring strata, and its weighted rows as
//!   `survival::finegray` writes them.
//! - [`aalen_johansen`]: cumulative incidence under competing risks with
//!   groups, weights and delayed entry, its infinitesimal jackknife standard
//!   errors, and a clustered bootstrap of the incidence in two arms over many
//!   cells.
//! - [`multistate`]: occupation probabilities of a multi-state process by the
//!   product integral, their infinitesimal jackknife standard errors, and their
//!   clustered bootstrap.
//! - [`msreg`]: Cox regression for a multi-state process, a hazard for each transition with
//!   coefficients specific to a transition or shared, as `coxph` with a multi-state outcome,
//!   and the cumulative hazards and transition probabilities of a covariate profile, as
//!   `msfit` and `probtrans` of `mstate`.
//! - [`markov`]: the landmark test of the Markov property.
//! - [`metrics`]: redistribute-to-the-right weights, the Brier score, the IPCW
//!   Brier score and index of prediction accuracy, the time-dependent AUC and
//!   calibration by risk group, as `rttright`, `brier`, `riskRegression::Score`
//!   and `timeROC`, with a Cox model for the censoring times, contrasts between
//!   models, and calibration by pseudo-values.
//!   Royston and Sauerbrei's D with the measures of Kent and O'Quigley and of
//!   Gonen and Heller, as `royston`.
//! - [`survfit`]: Kaplan-Meier and Nelson-Aalen curves for single-event data with
//!   standard errors, confidence limits, quantiles and restricted means, read at chosen
//!   times as `summary.survfit(times = )` with the starting point of `survfit0`.
//! - [`pseudo`]: pseudo-observations of survival, cumulative incidence, restricted
//!   means and cumulative hazards from the infinitesimal or the exact jackknife, as
//!   `survival::pseudo` and the `pseudo` package, of a multi-state process (state
//!   occupation, time in state, transition hazards), by inverse probability of
//!   censoring weighting with a Cox model of the censoring, as `eventglm`, and
//!   generalised estimating equations on them.
//! - [`rmst`]: restricted mean survival time and time lost in two groups with their
//!   difference and ratios, and their regression on covariates, as `survRM2::rmst2`.
//! - [`pyears`]: person-time and event counts in the cells of factors and
//!   time-dependent cut points such as age and calendar period.
//! - [`ratetable`]: a table of reference hazards by age, sex and calendar year, and the
//!   events expected in a cohort that had them, as `survival::ratetable` and
//!   `pyears(expect = "event")`.
//! - [`cox::predict`](mod@cox::predict): the baseline hazard, linear predictor, risk, expected
//!   number of events, terms and predicted survival curves of a Cox fit, as
//!   `basehaz`, `predict.coxph` and `survfit.coxph`, including the
//!   Kalbfleisch-Prentice curve, `start.time` and curves for covariates that
//!   change over time.
//! - [`cox::anova`](mod@cox::anova): likelihood-ratio tests between nested Cox models, as
//!   `anova.coxph`.
//! - [`SurvData`]: the checked input type that the estimators share, for
//!   follow-up data and for interval-censored times.
//! - [`survreg`]: accelerated failure time models for right, left and interval
//!   censored times, with offsets, the t distribution, robust variance, their
//!   predictions and residuals, as R's `survreg`.
//! - [`survcheck`]: structural checks of follow-up with several rows per subject, as
//!   R's `survcheck`.
//! - [`survdiff`]: the log-rank and G-rho tests of equal hazards, with strata, and
//!   the one-sample test, as R's `survdiff`.
//! - [`split`], [`tmerge`], [`condense`]: the data tools `survSplit`, `tmerge` and
//!   `survcondense`, which return rows and the row of the input each came from.
//! - [`lexis`]: follow-up split on several time scales at once, as `Epi::splitLexis` and
//!   `popEpi::splitMulti`, summing pieces into cells, and Poisson models for rates on them with
//!   `ci.lin` and `ci.exp` limits.
//! - [`surv2`]: timeline data (one row per observation) turned into counting-process
//!   intervals, as `Surv2` and `fromtimeline`.
//! - [`weighting`]: inverse probability of treatment and censoring weights, with
//!   the pooled logistic censoring model, and Cox and Kaplan-Meier estimates with a
//!   variance that accounts for the estimated weights of treatment, of censoring
//!   or of both.
//! - `arrow` (feature `arrow`): [`SurvData`] built from Apache Arrow arrays and record batches
//!   without copying the columns.
//! - [`standardise`]: standardised risks and average treatment effects of Cox and
//!   cause-specific Cox models by the g-formula, with influence-function standard
//!   errors, as `riskRegression::ate`, and population-averaged predictions and
//!   contrasts of a fitted model, as `survival::yates` ([`standardise::yates`]).
//! - [`timefix`]: the tie policy of `survival::aeqSurv`, which makes times equal
//!   up to rounding error equal.
//!
//! # Arithmetic
//!
//! The operations and their order are fixed, with no fused multiply-add and no
//! reassociation by thread count. Sums that must agree with a compensated
//! reference use [`epistat_stats::sum::compensated`], and the rest are plain
//! loops in input order. `ln` and `cos`, used for the cluster weights and the
//! normal multipliers, come from the platform's maths library.
#![warn(missing_docs)]

pub mod aalen_johansen;
pub mod absrisk;
#[cfg(feature = "arrow")]
pub mod arrow;
mod check;
pub mod clogit;
pub mod concordance;
pub mod condense;
pub mod cox;
mod error;
pub mod fine_gray;
mod ldl;
pub mod lexis;
pub mod markov;
pub mod metrics;
pub mod msreg;
pub mod multistate;
mod order;
mod par;
pub mod pseudo;
pub mod pyears;
pub mod ratetable;
pub mod rmst;
pub mod split;
pub mod standardise;
mod step;
pub mod surv;
pub mod surv2;
pub mod survcheck;
pub mod survdiff;
pub mod survfit;
pub mod survreg;
pub mod timefix;
pub mod tmerge;
pub mod weighting;

pub use error::{Error, Result};
pub use surv::{
    Cause, IntervalDataBuilder, IntervalKind, IntervalSurv, Levels, Outcome, Surv, SurvBuf,
    SurvBufBuilder, SurvData, SurvDataBuilder, UsedRows,
};
