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
//! [`aipw::Aipw`], [`tmle::Tmle`], [`balance::MeanDifference`] and
//! [`balance::WeightedRegression`] implement [`Tidy`](epistat_core::Tidy): a
//! table of the two means (`mean1`, `mean0`) or of the regression's
//! coefficients with the influence-function covariance, which is a sandwich
//! and is reported as [`VarianceKind::Robust`](epistat_core::VarianceKind).
//!
//! # Inputs
//!
//! Case weights are [`CaseWeights`]; a person of weight 0 leaves the
//! estimate. Predictions given for a binary outcome, every outcome 0 or 1,
//! are probabilities, and [`aipw::aipw`] and [`tmle::tmle`] refuse one
//! outside `[0, 1]`. The designs of the fitted nuisance models need a
//! constant column, the intercept.
//!
//! | field | meaning |
//! |---|---|
//! | `contrast` | `mean1`, `mean0`, their difference and ratio, as [`Estimate`]s |
//! | `influence` | the influence functions of the two means, per unit |
//! | `coefficients`, `covariance` | a fitted model's estimates and their covariance |
//! | `se`, `interval` | a standard error and an interval, `None` where undefined |
//! | `n_used`, `n_trimmed` | the persons used, and those whose propensity was bounded |
//! | `weight`, `cluster` | the case weights and the cluster codes of the input |
//!
//! Results do not depend on the number of threads.
#![warn(missing_docs)]

pub mod aipw;
pub mod balance;
mod error;
pub mod gformula;
mod shared;
pub mod tmle;

pub use epistat_core::{Bootstrap, CaseWeights, Contrast, Estimand, Estimate, Level, Panel, Seed};
pub use epistat_inference::{ClusterRule, Clusters, Influence, KeyedBy};
pub use epistat_stats::weighted::ProbabilityBounds;
pub use error::{Error, Model, Result};
