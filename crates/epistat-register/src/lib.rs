//! Tools for register data: person-time by calendar year, merging of
//! overlapping intervals, connected components of rows that share a key,
//! constant-price deflation, normalisation of ICD-10 codes and age in whole
//! years.
//!
//! Each module is one small, self-contained method on plain slices and integers,
//! with no dependence on a data-frame library.
//!
//! - [`person_time`] converts between day counts and civil dates, with the
//!   SAS epoch of 1960 beside the Unix epoch, and splits an
//!   interval of days across the calendar years it spans, with the length of
//!   each year for prorating an annual value.
//! - [`spans`] merges intervals that overlap, or lie within a grace of each
//!   other, in one sweep over rows sorted by key and start.
//! - [`relations`] finds the connected components of rows that share a code in
//!   any of several key columns, for example siblings through either parent.
//! - [`icd10`] normalizes ICD-10 diagnosis codes, including the Danish register
//!   prefix, and classifies them into the 22 chapters.
//! - [`age`] computes age in whole years from a birth date under a named rule.
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
#![warn(missing_docs)]

pub mod age;
pub mod cpi;
mod error;
pub mod icd10;
pub mod person_time;
pub mod relations;
pub mod spans;

pub use age::{AgeRule, LeapBirthday, age_in_years};
pub use cpi::CpiTable;
#[cfg(feature = "denmark")]
pub use cpi::DENMARK_ANNUAL_CPI;
pub use error::RegisterError;
pub use icd10::{
    Icd10Chapter, NormalizedDiagnosis, UnknownChapter, icd10_chapter_label,
    normalize_diagnosis_code,
};
pub use person_time::{
    SAS_EPOCH_OFFSET_DAYS, SAS_EPOCH_OFFSET_SECONDS, YearOverlap, calendar_year_overlaps,
    civil_from_days, civil_from_days_sas, civil_from_sas_seconds, days_from_civil,
    days_from_civil_sas, sas_days_to_unix, sas_seconds_to_unix, unix_days_to_sas,
    unix_seconds_to_sas, year_from_days,
};
pub use relations::connected_components;
pub use spans::{MergedSpans, merge_sorted_spans};
