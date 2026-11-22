//! Test helpers for numerical code: tolerances, bit fingerprints, checks that
//! results do not depend on the number of threads, ratios of deviations to
//! tolerances, a `SplitMix64` generator for test problems, a marker for undefined values, and loading of reference values
//! written by R.

mod absent;
mod deviation;
mod fingerprint;
mod generator;
mod reference;
mod threads;
mod tolerance;

pub use absent::{absent, absent_all, bits, none_as_nan};
pub use deviation::{
    NanRule, worst_deviation, worst_deviation_defined, worst_deviation_defined_each,
};
pub use fingerprint::Fingerprint;
pub use generator::SplitMix64;
pub use reference::{load_json, na_f64, na_f64_vec};
pub use threads::{POOLS, assert_same_on_pools, assert_same_on_threads, on_threads};
pub use tolerance::{Tolerance, assert_close, assert_within_standard_errors, ulps_apart};
