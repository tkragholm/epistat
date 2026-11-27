//! Causal effect estimators for cohort data.
//!
//! - [`aipw`]: augmented inverse probability weighting for a binary or
//!   continuous outcome. [`aipw::aipw`] takes the predictions of the outcome and
//!   propensity models and gives the treatment-specific means, their difference
//!   and ratio, with the variance of the efficient influence function.
//!   [`aipw::aipw_glm`] fits generalised linear models for both and gives the
//!   variance that accounts for their estimation.
//! - [`balance`]: entropy balancing and the covariate balancing propensity score,
//!   weights that balance the covariates directly, with outcome estimates whose
//!   variance accounts for the estimation of the weights.
//! - [`gformula`]: the parametric g-formula for time-varying treatments and
//!   confounders on person-period data. It fits a model for each time-varying
//!   covariate and for the outcome, simulates the natural course and static,
//!   threshold and dynamic strategies, and gives the risk by each period (with
//!   a competing event) or the mean at the end of follow-up, with Monte Carlo
//!   errors and a bootstrap over persons.
//! - [`tmle`]: targeted maximum likelihood estimation for a binary or bounded
//!   continuous outcome, as the R package `tmle` does it, on given predictions
//!   ([`tmle::tmle`]) or with fitted generalised linear models
//!   ([`tmle::tmle_glm`]), with the variance of the efficient influence function;
//!   and [`tmle::survival`], the risk by a time with right censoring in discrete
//!   time.
//!
//! The risk by a time with right censoring is in `epistat_survival::aipw`, which
//! has the Cox and censoring models it needs.
//!
//! # Results
//!
//! The estimators of a contrast report it as a [`Contrast`] of two
//! [`Estimate`]s, the mean (or risk) in arm 1 and in arm 0, with their
//! difference and ratio, at the confidence [`Level`] of their options. The
//! population is an [`Estimand`]: the ATE, ATT or ATC. They also return the
//! [`Influence`] function of the two means on the per-unit scale, `theta_hat -
//! theta ~ sum(psi_i) / n`, each person a unit coded by its row number, with
//! which the standard errors were computed, so that a caller can form the
//! variance of any other combination of the two means. The [`ClusterRule`]
//! of each estimator is its R reference's and is stored on the influence
//! function: [`aipw`] (`riskRegression::ate`), [`balance`] (`WeightIt`) and
//! [`tmle::survival`] (`survtmle`) have the `n` divisor without clusters,
//! `NoSmallSample`, and the factor `G / (G - 1)` with them, `SmallSample`;
//! [`tmle`] has `tmle`'s `n - 1` divisor, `SmallSample`, with and without
//! clusters.
//!
//! Results do not depend on the number of threads.
#![warn(missing_docs)]

pub mod aipw;
pub mod balance;
mod error;
pub mod gformula;
mod shared;
pub mod tmle;

pub use epistat_core::{Contrast, Estimand, Estimate, Level, Panel};
pub use epistat_inference::{ClusterRule, Clusters, Influence, KeyedBy};
pub use error::{Error, Result};
