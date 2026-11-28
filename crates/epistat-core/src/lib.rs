//! The types every epistat crate shares.
//!
//! This crate sits at the bottom of the workspace and depends on nothing
//! (serde is optional). It holds plain data types and integer arithmetic,
//! with no floating-point algorithm beyond fixed-order loops, so that the
//! crates above it can exchange results without conversion and a binding can
//! convert one set of types.
//!
//! | item | what it is |
//! |---|---|
//! | [`Level`] | a confidence level, kept as the confidence or alpha the caller gave |
//! | [`Interval`], [`Estimate`], [`Scale`], [`Contrast`] | how an estimator reports a number |
//! | [`Covariance`] | a covariance matrix that knows its dimension |
//! | [`EventStudy`] | event-study coefficients, their covariance and the reference period |
//! | [`Bootstrap`] | the replicates and root seed of a randomised call |
//! | [`Estimand`] | the population an average treatment effect is averaged over |
//! | [`Codes`] | labels as dense `u32` codes in sorted order, and back |
//! | [`Panel`] | the validated unit-and-period key of person-period data |
//! | [`day`], [`Day`] | civil dates as `i64` days since 1970-01-01, and one such date |
//! | [`check`], [`InputError`] | the input checks every estimator shares |
//! | [`ErrorKind`], [`Classify`] | what kind of failure an error is, and its stable code |
//!
//! # Conventions
//!
//! Undefined numbers are `None`, never NaN. Units, clusters and strata are
//! `u32` codes, periods and days `i64`. Results are `#[non_exhaustive]` and
//! built through constructors, so that adding a field breaks no caller.
//! Types whose fields must stay consistent with each other ([`Covariance`],
//! [`EventStudy`], [`Codes`], [`Panel`]) keep them private and are checked
//! once, when built.

pub mod bootstrap;
pub mod check;
pub mod codes;
pub mod covariance;
pub mod day;
pub mod error;
pub mod estimand;
pub mod estimate;
pub mod event_study;
pub mod level;
pub mod panel;

pub use bootstrap::Bootstrap;
pub use check::InputError;
pub use codes::Codes;
pub use covariance::Covariance;
pub use day::Day;
pub use error::{Classify, ErrorKind};
pub use estimand::Estimand;
pub use estimate::{Contrast, Estimate, Interval, Scale};
pub use event_study::EventStudy;
pub use level::Level;
pub use panel::Panel;
