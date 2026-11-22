//! What the tests against an R reference share: loading the reference, checking
//! the version of R that made it, and the unit ids and groups they build.
//!
//! Included by `#[path = "common/r_reference.rs"] mod r_reference;`.

#![allow(dead_code)]

use epistat_testkit::load_json;
use serde::de::DeserializeOwned;

/// The reference in `tests/data/{file}`.
pub fn reference<T: DeserializeOwned>(file: &str) -> T {
    load_json(format!("{}/tests/data/{file}", env!("CARGO_MANIFEST_DIR")))
}

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
