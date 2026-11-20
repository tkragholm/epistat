//! Survival and multi-state methods: Cox regression, the Fine-Gray model for
//! competing risks, the Aalen-Johansen estimator, state occupation in a
//! multi-state process and the Titman-Putter test of the Markov property.
//!
//! The estimators take slices, return plain structs and give the same bits on
//! the same input, whatever the number of threads. Where an estimator
//! bootstraps, the caller supplies the seeds, and each replicate depends on its
//! seed only, through [`epistat_rng`].
//!
//! # Contents
//!
//! - [`concordance`]: Harrell's C and its relatives for a numeric score, with weights,
//!   strata, delayed entry, time weights and an infinitesimal jackknife variance.
//! - [`cox`]: Cox regression on one covariate or a design matrix, with Breslow or
//!   Efron ties, delayed entry, strata, case weights and a cluster-robust
//!   variance.
//! - [`fine_gray`]: the rows of the Fine-Gray model, the Kaplan-Meier estimate of
//!   the censoring distribution they need, and the fit on one covariate.
//! - [`aalen_johansen`]: cumulative incidence under competing risks with
//!   groups, weights and delayed entry, its infinitesimal jackknife standard
//!   errors, and a clustered bootstrap of the incidence in two arms over many
//!   cells.
//! - [`multistate`]: occupation probabilities of a multi-state process by the
//!   product integral, their infinitesimal jackknife standard errors, and their
//!   clustered bootstrap.
//! - [`markov`]: the landmark test of the Markov property.
//! - [`pyears`]: person-time and event counts in the cells of factors and
//!   time-dependent cut points such as age and calendar period.
//! - [`SurvData`]: the checked input type that the estimators will share.
//! - [`survcheck`]: structural checks of follow-up with several rows per subject, as
//!   R's `survcheck`.
//! - [`survdiff`]: the log-rank and G-rho tests of equal hazards, with strata, and
//!   the one-sample test, as R's `survdiff`.
//! - [`split`], [`tmerge`], [`condense`]: the data tools `survSplit`, `tmerge` and
//!   `survcondense`, which return rows and the row of the input each came from.
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

pub mod aalen_johansen;
mod check;
pub mod concordance;
pub mod condense;
pub mod cox;
mod error;
pub mod fine_gray;
pub mod markov;
pub mod multistate;
pub mod pyears;
pub mod split;
pub mod surv;
pub mod survcheck;
pub mod survdiff;
pub mod timefix;
pub mod tmerge;

pub use error::{Error, Result};
pub use surv::{
    Cause, Levels, Outcome, Surv, SurvBuf, SurvBufBuilder, SurvData, SurvDataBuilder, UsedRows,
};
