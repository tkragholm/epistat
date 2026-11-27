//! Truncation of the weights of the outcome model, as `TrialEmulation`'s
//! `analysis_weights = "p99"` and `"weight_limits"`.
//!
//! The limits are [`WeightLimits`] of `epistat-stats`, whose
//! [`apply`](WeightLimits::apply) cuts the weights at R's type 1 quantiles
//! or at fixed values, as `TrialEmulation`'s `limit_weight` does.

pub use epistat_stats::weighted::WeightLimits;
