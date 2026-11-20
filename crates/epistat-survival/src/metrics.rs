//! Metrics of predictions for censored outcomes: redistribute-to-the-right
//! weights, the Brier score, the index of prediction accuracy, the
//! time-dependent area under the ROC curve and the calibration of predicted
//! risks.
//!
//! - [`rttright`]: the weights of `survival::rttright`, which turn a censored
//!   sample into a complete one.
//! - [`brier`]: the Brier score and R-squared of `survival::brier`, for
//!   predicted probabilities supplied by the caller.
//! - [`score`]: the inverse-probability-of-censoring weighted Brier score with
//!   its standard error, the null model and the index of prediction accuracy,
//!   and the time-dependent area under the curve, for survival and competing
//!   risks, as `riskRegression::Score` and `timeROC::timeROC`.
//! - [`calibration`]: observed against predicted risk in groups of the
//!   predictions, as the quantile groups of `riskRegression::plotCalibration`.
//!
//! The predictions are the caller's. For a Cox model they come from
//! [`cox::predict::survfit`](crate::cox::predict::survfit) and for a
//! competing-risks model from the caller's own absolute risks.

mod censoring;
mod marginal;
mod prepare;

pub mod brier;
pub mod calibration;
pub mod royston;
pub mod rttright;
pub mod score;
