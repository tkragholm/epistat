//! Doubly robust difference-in-differences estimators.
//!
//! - [`improved`] holds the improved, locally efficient entry points.
//! - [`panel`] holds the panel estimator. `estimate_drdid_panel` is the
//!   locally efficient panel estimator of `DRDID::drdid_panel`.
//! - [`repeated`] holds the traditional repeated cross-section estimator.

mod design;
pub mod improved;
pub mod moments;
pub mod panel;
pub mod repeated;
pub mod repeated_efficient;

pub use design::{DesignPruning, independent_design_columns};

use crate::inference::{multiplier_bootstrap_ci, normal_interval};
use crate::types::DrDidConfig;

/// The confidence interval of a two-period estimate.
///
/// With `config.bootstrap_reps` 0 it is the normal interval `att +- z se` at
/// `config.confidence_level`, which is what `DRDID` reports with its default
/// `boot = FALSE`. A positive count makes it the percentile interval of that
/// many Rademacher multiplier replicates of `influence`, seeded from
/// `config.bootstrap_seed`.
pub(crate) fn confidence_interval(
    att: f64,
    se: f64,
    influence: &[f64],
    config: DrDidConfig,
) -> (f64, f64) {
    if config.bootstrap_reps == 0 {
        normal_interval(att, se, config.confidence_level)
    } else {
        multiplier_bootstrap_ci(att, influence, config.inference(), config.bootstrap())
    }
}
