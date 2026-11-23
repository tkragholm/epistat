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
//! - [`tmle`]: targeted maximum likelihood estimation for a binary or bounded
//!   continuous outcome, as the R package `tmle` does it, on given predictions
//!   ([`tmle::tmle`]) or with fitted generalised linear models
//!   ([`tmle::tmle_glm`]), with the variance of the efficient influence function.
//!
//! The risk by a time with right censoring is in `epistat_survival::aipw`, which
//! has the Cox and censoring models it needs.
//!
//! Results do not depend on the number of threads.
#![warn(missing_docs)]

pub mod aipw;
pub mod balance;
mod error;
pub mod tmle;

pub use error::{Error, Result};
