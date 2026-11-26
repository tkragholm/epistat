//! The restricted mean survival time of a curve.

use super::GroupCurve;
use crate::error::{Error, Result};

/// A restricted mean survival time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RestrictedMean {
    /// The horizon the curve was integrated to.
    pub tau: f64,
    /// The area under the curve from its origin to `tau`.
    pub mean: f64,
    /// Its standard error.
    pub se: f64,
}

pub(super) fn restricted_mean(curve: &GroupCurve, tau: f64) -> Result<RestrictedMean> {
    if !tau.is_finite() {
        return Err(Error::invalid(
            "the horizon of a restricted mean must be finite",
        ));
    }
    if tau < curve.origin {
        return Err(Error::invalid(format!(
            "the horizon {tau} of a restricted mean is before the start of the curve, {}",
            curve.origin
        )));
    }
    let kept = curve.time.partition_point(|&t| t <= tau);
    // The rectangles: the curve is 1 from the origin to the first time, and
    // its value at each time to the next, the last to tau.
    let mut rectangles = Vec::with_capacity(kept + 1);
    let mut previous_time = curve.origin;
    let mut previous_surv = 1.0;
    for i in 0..kept {
        rectangles.push((curve.time[i] - previous_time) * previous_surv);
        previous_time = curve.time[i];
        previous_surv = curve.surv[i];
    }
    rectangles.push((tau - previous_time) * previous_surv);
    let mean: f64 = rectangles.iter().sum();
    // The area after each time, accumulated from the end, times Greenwood's
    // increment at that time.
    let mut variance = 0.0;
    let mut after = 0.0;
    for i in (0..kept).rev() {
        after += rectangles[i + 1];
        let risk = curve.n_risk[i];
        let events = curve.n_event[i];
        let increment = if risk - events == 0.0 {
            0.0
        } else {
            events / (risk * (risk - events))
        };
        variance += after * after * increment;
    }
    Ok(RestrictedMean {
        tau,
        mean,
        se: variance.sqrt(),
    })
}
