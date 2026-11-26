//! Building the core result types from the limits a method computes.

use epistat_core::{Interval, Level};

/// The interval `[lower, upper]` at `level`, or `None` when neither limit is
/// defined.
pub(crate) fn interval(lower: Option<f64>, upper: Option<f64>, level: Level) -> Option<Interval> {
    (lower.is_some() || upper.is_some()).then(|| Interval::new(lower, upper, level))
}
