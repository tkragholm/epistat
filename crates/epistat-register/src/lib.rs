//! Tools for register data: person-time by calendar year, merging of
//! overlapping intervals, connected components of rows that share a key,
//! constant-price deflation, normalisation of ICD-10 codes, age in whole
//! years, code sets for ICD-10, ICD-8 and ATC, event definitions, comorbidity
//! indices, drug exposure episodes, interval sets, Lexis splitting, pedigrees and
//! residence spells.
//!
//! Each module is one small, self-contained method on plain slices and integers,
//! with no dependence on a data-frame library.
//!
//! - [`person_time`] converts between day counts and civil dates, with the
//!   SAS epoch of 1960 beside the Unix epoch, and splits an
//!   interval of days across the calendar years it spans, with the length of
//!   each year for prorating an annual value. The civil-date arithmetic is
//!   that of [`epistat_core::day`].
//! - [`spans`] merges intervals that overlap, or lie within a grace of each
//!   other, in one sweep over rows sorted by key and start.
//! - [`relations`] finds the connected components of rows that share a code in
//!   any of several key columns, for example siblings through either parent.
//! - [`icd10`] normalizes ICD-10 diagnosis codes, including the Danish register
//!   prefix, and classifies them into the 22 chapters.
//! - [`age`] computes age in whole years from a birth date under a named rule.
//! - [`codes`] holds sets of codes written as prefixes and ranges, with
//!   classification of the distinct codes of a column, the levels of ATC and
//!   ICD-10, and ICD-8 beside ICD-10.
//! - [`events`] finds the first event of a definition in a window, whether a
//!   person was already a case before it, and whether a second record confirms
//!   it.
//! - [`comorbidity`] scores the Charlson index and the Elixhauser measure from
//!   diagnosis codes, with look-back windows and hierarchy rules.
//! - [`exposure`] turns dispensings into periods of drug exposure and
//!   cumulative doses, under assumptions the caller declares.
//! - [`intervals`] intersects, subtracts, complements and clips per-key sets of
//!   half-open intervals.
//! - [`lexis`] cuts follow-up on integer days at calendar years, civil
//!   birthdays, periods and times since entry.
//! - [`pedigree`] finds siblings, cousins, multiple births, family clusters and
//!   relatedness from the mother and father of each person.
//! - [`residence`] turns dated emigrations and immigrations into residence
//!   spells and gives the end of observation.
//! - [`wtd`] fits the parametric waiting time distribution and gives the
//!   duration of a prescription as a percentile of the time between
//!   redemptions.
//! - [`pre2dup`] estimates drug use periods from each person's own pattern of
//!   purchases, by the PRE2DUP method.
//! - [`cpi`] deflates nominal values to the prices of a reference year with an
//!   annual price index the caller supplies, checked by
//!   [`CpiTable::try_new`]. The Statistics Denmark index is available behind
//!   the cargo feature `denmark` as `CpiTable::denmark`.
//!
//! # Conventions
//!
//! Dates are counts of days since 1970-01-01 as `i64`. Row numbers and
//! component numbers are `u32`, and a function that takes more rows than that
//! refuses them. Every function returns results that depend only on its
//! inputs, never on a thread count or hash order.
//!
//! # Paths
//!
//! Thirty-three names are reachable at the root and in the module that defines
//! them. The module path is the canonical one, the path the documentation of
//! the item and the other modules use. The root path is a shorthand for the
//! names most callers need and is kept.
//!
//! | at the root | canonical path |
//! |---|---|
//! | [`Error`], [`RegisterError`], [`CodeItemProblem`], [`CrosswalkProblem`], [`PedigreeProblem`] | `error` (private module, named at the root only) |
//! | [`CpiTable`], `DENMARK_ANNUAL_CPI` | [`cpi`] |
//! | [`AgeRule`], [`LeapBirthday`], [`age_in_years`], [`birthday_on_age`] | [`age`] |
//! | [`Icd10Chapter`], [`NormalizedDiagnosis`], [`UnknownChapter`], [`icd10_chapter_label`], [`normalize_diagnosis_code`] | [`icd10`] |
//! | [`SAS_EPOCH_OFFSET_DAYS`], [`SAS_EPOCH_OFFSET_SECONDS`], [`YearOverlap`], [`add_months`], [`add_years`], [`calendar_year_overlaps`], [`civil_from_days`], [`civil_from_days_sas`], [`civil_from_sas_seconds`], [`days_from_civil`], [`days_from_civil_sas`], [`sas_days_to_unix`], [`sas_seconds_to_unix`], [`unix_days_to_sas`], [`unix_seconds_to_sas`], [`year_from_days`] | [`person_time`] |
//! | [`connected_components`] | [`relations`] |
//! | [`MergedSpans`], [`merge_sorted_spans`] | [`spans`] |
//!
//! The date arithmetic in [`person_time`] is defined in [`epistat_core::day`],
//! which names the conversions `from_civil`, `to_civil` and `year_of`.
#![warn(missing_docs)]

pub mod age;
pub mod codes;
pub mod comorbidity;
pub mod cpi;
mod error;
pub mod events;
pub mod exposure;
pub mod icd10;
pub mod intervals;
pub mod lexis;
pub mod pedigree;
pub mod person_time;
pub mod pre2dup;
pub mod relations;
pub mod residence;
pub mod spans;
pub mod wtd;

pub use age::{AgeRule, LeapBirthday, age_in_years, birthday_on_age};
pub use cpi::CpiTable;
#[cfg(feature = "denmark")]
pub use cpi::DENMARK_ANNUAL_CPI;
pub use error::{CodeItemProblem, CrosswalkProblem, Error, PedigreeProblem};

/// The name of the error before it was shared with the other crates' naming.
#[deprecated(since = "0.2.0", note = "use `epistat_register::Error`")]
pub type RegisterError = Error;

pub use icd10::{
    Icd10Chapter, NormalizedDiagnosis, UnknownChapter, icd10_chapter_label,
    normalize_diagnosis_code,
};
pub use person_time::{
    SAS_EPOCH_OFFSET_DAYS, SAS_EPOCH_OFFSET_SECONDS, YearOverlap, add_months, add_years,
    calendar_year_overlaps, civil_from_days, civil_from_days_sas, civil_from_sas_seconds,
    days_from_civil, days_from_civil_sas, sas_days_to_unix, sas_seconds_to_unix, unix_days_to_sas,
    unix_seconds_to_sas, year_from_days,
};
pub use relations::connected_components;
pub use spans::{MergedSpans, merge_sorted_spans};
