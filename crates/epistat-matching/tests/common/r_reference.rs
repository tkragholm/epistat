//! What the tests against an R reference share: loading the reference, checking
//! the version of R that made it, and the unit ids and groups they build.
//!
//! Included by `#[path = "common/r_reference.rs"] mod r_reference;`.

#![allow(dead_code)]

use chrono::NaiveDate;
use epistat_matching::{BalanceRecord, CovariateValue};
use epistat_testkit::load_json;
use serde::de::DeserializeOwned;
use std::collections::HashMap;

/// The reference in `tests/data/{file}`.
pub fn reference<T: DeserializeOwned>(file: &str) -> T {
    load_json(format!("{}/tests/data/{file}", env!("CARGO_MANIFEST_DIR")))
}

/// The versions a reference script recorded, by the name of R or of the
/// package as the script spelled it. Indexing a name the script did not
/// record panics.
pub type Versions = HashMap<String, String>;

/// Fails unless `version`, the `R.version.string` a reference script
/// recorded, is a release of R 4.6.
pub fn assert_r_version(version: &str) {
    assert!(version.starts_with("R version 4.6"), "{version}");
}

/// The id `u{idx}` of the unit at row `idx`.
pub fn unit_id(idx: usize) -> String {
    format!("u{idx}")
}

/// The units flagged first and the others, each in the order given.
pub fn split_by_group<T>(units: impl IntoIterator<Item = (bool, T)>) -> (Vec<T>, Vec<T>) {
    let mut first = Vec::new();
    let mut rest = Vec::new();
    for (in_first, unit) in units {
        if in_first {
            first.push(unit);
        } else {
            rest.push(unit);
        }
    }
    (first, rest)
}

/// A unit with the given covariates, for a test whose distance, propensity
/// score or effect estimate does not read the birth date.
pub fn covariate_unit(
    id: String,
    covariates: impl IntoIterator<Item = (String, CovariateValue)>,
) -> BalanceRecord {
    let mut record =
        BalanceRecord::new(id, NaiveDate::from_ymd_opt(2010, 1, 1).expect("valid date"));
    record.covariates.extend(covariates);
    record
}
