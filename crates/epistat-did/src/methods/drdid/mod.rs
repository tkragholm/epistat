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
