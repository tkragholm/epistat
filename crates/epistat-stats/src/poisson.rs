//! Poisson confidence intervals and the gamma quantile at their paths before
//! 0.3.0.
//!
//! The functions are in [`crate::dist::poisson`] and [`crate::dist::gamma`].
//! These forwarders give the same bits and the signatures of 0.2; they are
//! removed in 0.4.0.
#![expect(
    deprecated,
    reason = "the forwarders name the types they keep for one release"
)]

use std::fmt;

use epistat_core::Level;

use crate::dist;

/// How the limits of a Poisson interval are found.
#[deprecated(since = "0.3.0", note = "use `epistat_stats::dist::poisson::Method`")]
pub type Method = dist::poisson::Method;

/// The limits of an interval. A limit that is undefined is `None`.
#[deprecated(
    since = "0.3.0",
    note = "`epistat_stats::dist::poisson::cipoisson` returns an `epistat_core::Interval`"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Interval {
    /// The lower limit.
    pub lower: Option<f64>,
    /// The upper limit.
    pub upper: Option<f64>,
}

/// Why an interval was refused.
#[deprecated(since = "0.3.0", note = "use `epistat_stats::Error`")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoissonError {
    /// The count is negative, NaN or infinite.
    Count,
    /// The person-time is NaN or infinite.
    Time,
    /// The level is not strictly between 0 and 1.
    Level,
}

impl fmt::Display for PoissonError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Count => "the count must be finite and not negative",
            Self::Time => "the person-time must be finite",
            Self::Level => "the level must be strictly between 0 and 1",
        })
    }
}

impl std::error::Error for PoissonError {}

/// The confidence interval of a Poisson count `count` divided by `time`, at
/// the confidence `level`.
///
/// # Errors
///
/// [`PoissonError`] when the count is negative or not finite, the time is not
/// finite, or the level is not in `(0, 1)`.
#[deprecated(
    since = "0.3.0",
    note = "use `epistat_stats::dist::poisson::cipoisson` with a `Level`"
)]
pub fn cipoisson(
    count: f64,
    time: f64,
    level: f64,
    method: Method,
) -> Result<Interval, PoissonError> {
    if !(count.is_finite() && count >= 0.0) {
        return Err(PoissonError::Count);
    }
    if !time.is_finite() {
        return Err(PoissonError::Time);
    }
    let level = Level::confidence(level).map_err(|_| PoissonError::Level)?;
    let interval =
        dist::poisson::cipoisson(count, time, level, method).map_err(|_| PoissonError::Count)?;
    Ok(Interval {
        lower: interval.lower,
        upper: interval.upper,
    })
}

/// The `p` quantile of the gamma distribution with the given shape and scale
/// 1, or `None` when `shape` is not positive and finite or `p` is outside
/// `[0, 1]`.
#[deprecated(
    since = "0.3.0",
    note = "use `epistat_stats::dist::gamma::quantile`, NaN where this is None"
)]
#[must_use]
pub fn gamma_quantile(p: f64, shape: f64) -> Option<f64> {
    dist::gamma::quantile_or_none(p, shape)
}
