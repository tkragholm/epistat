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
//! | [`Interval`], [`Coverage`], [`Estimate`], [`Scale`], [`Contrast`] | how an estimator reports a number |
//! | [`Coefficients`], [`Term`], [`Glance`], [`Tidy`] | a fit's coefficient table and summary, and the trait fits implement |
//! | [`Col`] | a column with null rows, as Arrow holds it: values and a validity bitmap |
//! | [`Covariance`] | a covariance matrix that knows its dimension |
//! | [`Matrix`] | a rectangular matrix of results, row-major |
//! | [`Test`], [`Reference`] | a test statistic, its degrees of freedom and p-value |
//! | [`EventStudy`] | event-study coefficients, their covariance and the reference period |
//! | [`Bootstrap`], [`Seed`] | the replicates and root seed of a randomised call |
//! | [`Estimand`] | the population an average treatment effect is averaged over |
//! | [`Codes`] | labels as dense `u32` codes in sorted order, and back |
//! | [`Panel`] | the validated unit-and-period key of person-period data |
//! | [`Permutation`] | a reordering of rows, applied to each column alike |
//! | [`day`], [`Day`] | civil dates as `i64` days since 1970-01-01, and one such date |
//! | [`CaseWeights`], [`WeightKind`] | a checked column of case weights and what they mean |
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
//!
//! # Options
//!
//! Options are plain structs with public fields and a `Default`, written as
//! a literal that ends in `..Default::default()`. Each options struct in
//! epistat also has a hidden field of the hidden type `Extensible`:
//!
//! ```
//! use epistat_core::Extensible;
//!
//! #[derive(Clone, Debug, PartialEq)]
//! pub struct FitOptions {
//!     /// The most iterations before the fit gives up.
//!     pub max_iterations: usize,
//!     /// The change in deviance taken as convergence.
//!     pub tolerance: f64,
//!     #[doc(hidden)]
//!     pub _extensible: Extensible,
//! }
//!
//! impl Default for FitOptions {
//!     fn default() -> Self {
//!         Self {
//!             max_iterations: 25,
//!             tolerance: 1e-8,
//!             _extensible: Extensible::NEW,
//!         }
//!     }
//! }
//!
//! let options = FitOptions { max_iterations: 50, ..Default::default() };
//! assert_eq!(options.tolerance, 1e-8);
//! ```
//!
//! A literal that names only the documented fields does not compile, so no
//! caller depends on the struct having exactly those fields. A new option is
//! then a minor release, and cargo-semver-checks, which ignores hidden
//! fields, agrees:
//!
//! ```compile_fail,E0063
//! # use epistat_core::Extensible;
//! # #[derive(Default)]
//! # pub struct FitOptions {
//! #     pub max_iterations: usize,
//! #     pub tolerance: f64,
//! #     #[doc(hidden)]
//! #     pub _extensible: Extensible,
//! # }
//! let options = FitOptions { max_iterations: 50, tolerance: 1e-8 };
//! ```
//!
//! A crate adopting it adds the field to every options struct, the structs
//! whose names end in `Options`, `Spec`, `Settings`, `Config` or `Control`.
//! A derived `Default` fills it. A `Default` or a `const` written out by hand
//! sets it to `Extensible::NEW`, which only code inside epistat names. Where
//! the struct is serialised, the field is marked `#[cfg_attr(feature =
//! "serde", serde(skip))]`.

pub mod bootstrap;
pub mod check;
pub mod codes;
pub mod coefficients;
pub mod col;
pub mod covariance;
pub mod day;
pub mod error;
pub mod estimand;
pub mod estimate;
pub mod event_study;
mod extensible;
mod format;
pub mod hypothesis;
pub mod level;
pub mod matrix;
pub mod panel;
pub mod permutation;
pub mod weights;

pub use bootstrap::{Bootstrap, Seed};
pub use check::InputError;
pub use codes::Codes;
pub use coefficients::{Coefficients, EffectScale, Glance, Term, Tidy, TidyRow, VarianceKind};
pub use col::Col;
pub use covariance::Covariance;
pub use day::Day;
pub use error::{Classify, ErrorKind};
pub use estimand::Estimand;
pub use estimate::{Contrast, Coverage, Estimate, Interval, Scale};
pub use event_study::EventStudy;
#[doc(hidden)]
pub use extensible::Extensible;
pub use hypothesis::{Reference, Test};
pub use level::Level;
pub use matrix::Matrix;
pub use panel::Panel;
pub use permutation::Permutation;
pub use weights::{CaseWeights, WeightKind};
