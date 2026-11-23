//! Causal effect estimators for cohort data.
//!
//! - [`aipw`]: augmented inverse probability weighting for a binary or
//!   continuous outcome. [`aipw::aipw`] takes the predictions of the outcome and
//!   propensity models and gives the treatment-specific means, their difference
//!   and ratio, with the variance of the efficient influence function.
//!   [`aipw::aipw_glm`] fits generalised linear models for both and gives the
//!   variance that accounts for their estimation.
//!
//! Results do not depend on the number of threads.
#![warn(missing_docs)]

pub mod aipw;
mod error;

pub use error::{Error, Result};
