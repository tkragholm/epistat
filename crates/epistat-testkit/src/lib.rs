//! Test helpers for numerical code: tolerances, bit fingerprints, checks that
//! results do not depend on the number of threads, and loading of reference
//! values written by R.

mod fingerprint;
mod reference;
mod threads;
mod tolerance;

pub use fingerprint::Fingerprint;
pub use reference::{load_json, na_f64, na_f64_vec};
pub use threads::{assert_same_on_threads, on_threads};
pub use tolerance::{Tolerance, assert_close, ulps_apart};
