#![warn(missing_docs)]

//! Models for semicontinuous outcomes, non-negative with a mass at zero, such as
//! costs: the two-part model, the Tweedie GLM, log-normal regression with smearing,
//! and the marginalised two-part model for panels, with model comparison and
//! selection.
//!
//! | method | entry point | data | result |
//! |---|---|---|---|
//! | two-part model | [`two_part::fit`] | [`CostData`] | [`two_part::TwoPartModel`] |
//! | Tweedie GLM | [`tweedie::fit`] | [`CostData`] | [`tweedie::TweedieModel`] |
//! | log-normal with smearing | [`lognormal::fit`] | [`CostData`] | [`lognormal::LogNormalModel`] |
//! | marginalised two-part model | [`mtp::fit`] | [`LongitudinalData`] | [`mtp::MtpModel`] |
//! | comparison, cross-validation, selection | [`compare::compare`], [`compare::cross_validate`], [`compare::select`] | [`CostData`] | [`compare::ModelComparison`], [`compare::CrossValidation`], [`compare::Selection`] |
//!
//! Each module explains its method. The data views check their columns once
//! ([`data`]); a bootstrap builds the view of the full sample once and refits it
//! under many weight columns with [`CostData::with_weights`], or, for designs of
//! indicator columns, with the compact designs of [`prepared`], a supported API
//! for bootstraps that refit one design many times. Every function returns
//! the one [`Error`].
//!
//! The fits use their own IRLS with fused multiply-add, so results are the same
//! on any number of threads but can differ between platforms in the last bits.

pub mod compare;
pub mod counterfactual;
pub mod data;
mod error;
mod fit;
pub mod lognormal;
pub mod mtp;
pub mod prepared;
pub mod preprocess;
pub mod tweedie;
pub mod two_part;

mod covariance;
mod matrix_ops;
pub(crate) mod utils;

pub(crate) use data::check_columns;
pub use data::{CostData, CostDataBuilder, LongitudinalData, LongitudinalDataBuilder};
pub use error::Error;
pub use fit::{AttemptDiagnostics, AttemptOutcome, FitMetadata, FitStrategy, Model, SolverKind};
