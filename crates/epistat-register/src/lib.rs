//! Tools for register data: person-time by calendar year, merging of
//! overlapping intervals, connected components of rows that share a key, and
//! constant-price deflation.
//!
//! Each module is one small, self-contained method on plain slices and integers,
//! with no dependence on a data-frame library.
//!
//! - [`person_time`] converts between day counts and civil dates, and splits an
//!   interval of days across the calendar years it spans, with the length of
//!   each year for prorating an annual value.
//! - [`spans`] merges intervals that overlap, or lie within a grace of each
//!   other, in one sweep over rows sorted by key and start.
//! - [`relations`] finds the connected components of rows that share a code in
//!   any of several key columns, for example siblings through either parent.
//! - [`cpi`] deflates nominal values to the prices of a reference year with an
//!   annual price index the caller supplies. The Statistics Denmark index is
//!   available behind the cargo feature `denmark`.
//!
//! # Conventions
//!
//! Dates are counts of days since 1970-01-01 as `i64`. Row numbers and
//! component numbers are `u32`, and a function that takes more rows than that
//! refuses them. Every function returns results that depend only on its
//! inputs, never on a thread count or hash order.

pub mod cpi;
mod error;
pub mod person_time;
pub mod relations;
pub mod spans;

pub use cpi::CpiTable;
#[cfg(feature = "denmark")]
pub use cpi::DENMARK_ANNUAL_CPI;
pub use error::RegisterError;
pub use person_time::{
    YearOverlap, calendar_year_overlaps, civil_from_days, days_from_civil, year_from_days,
};
pub use relations::connected_components;
pub use spans::{MergedSpans, merge_sorted_spans};
