//! The range of point estimates and the confidence interval of an additive
//! effect under hidden bias, from an m-statistic over matched sets.
//!
//! For an effect `tau` added to every treated response, the m-statistic of
//! the sets with `tau` taken from each treated response has, under `gamma`,
//! the bound of [`super::set_bounds`] on its deviate. The confidence interval
//! at `gamma` holds the `tau` that the bound does not reject, and the range
//! of point estimates holds the `tau` at which the deviate can be zero
//! \[1\].
//!
//! # Algorithm
//!
//! As `sensitivitymw::senmwCI`: the lower end of the point estimates is the
//! `tau` where the deviate bound of the shifted sets crosses 0, and the lower
//! limit of the interval the `tau` where it crosses `z`, the normal quantile
//! at the level's [`Level::confidence_value`], `1 - alpha`, for a one-sided
//! interval and at its [`Level::upper_probability`], `1 - alpha / 2`, for a
//! two-sided one. The upper ends
//! come from the same search on the negated responses. The scores, scale
//! included, are recomputed at each `tau`, as `senmwCI` does. Each crossing
//! is bracketed by `min(treated) - max(controls)`, where every treated unit
//! is the largest of its set and the deviate is at its largest, and
//! `max(treated) - min(controls)`, where it is at its smallest, and bisected
//! to adjacent doubles. `senmwCI` finds the same crossings with `uniroot` to
//! a tolerance of the bracket's width over `5e5`.
//!
//! # References
//!
//! 1. Rosenbaum PR. Sensitivity analysis for m-estimates, tests, and
//!    confidence intervals in matched observational studies. Biometrics.
//!    2007;63(2):456-64.

use super::{MScore, SensitivityError, check_gamma, m_scores, set_bounds};
use crate::Error;
use epistat_core::{Interval, Level};
use epistat_stats::dist::normal;

/// Whether an interval has one limit or two.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sides {
    /// A lower limit only, with the level's alpha in its one tail, and no
    /// upper limit.
    One,
    /// Both limits, with half the level's alpha in each tail.
    Two,
}

/// The range of point estimates and the confidence interval of an additive
/// effect at one `gamma`.
///
/// The point estimates are a range, not an interval at a level: every
/// effect in it is a point estimate under some assignment probabilities
/// that `gamma` allows.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EffectInterval {
    /// The sensitivity parameter.
    pub gamma: f64,
    /// The smallest point estimate.
    pub estimate_lower: f64,
    /// The largest point estimate.
    pub estimate_upper: f64,
    /// The confidence interval: the smallest lower limit and the largest
    /// upper limit, `None` for the upper limit of a one-sided interval.
    pub interval: Interval,
}

/// The deviate bound of the sets with `tau` taken from each treated response.
fn deviate(sets: &[Vec<f64>], tau: f64, gamma: f64, spec: &MScore) -> Result<f64, Error> {
    let shifted: Vec<Vec<f64>> = sets
        .iter()
        .map(|s| {
            let mut v = s.clone();
            v[0] -= tau;
            v
        })
        .collect();
    Ok(set_bounds(&m_scores(&shifted, spec)?, gamma)?.deviate)
}

/// The `tau` in `[low, high]` where the deviate falls to `target`.
fn crossing(
    sets: &[Vec<f64>],
    gamma: f64,
    spec: &MScore,
    target: f64,
    mut low: f64,
    mut high: f64,
) -> Result<f64, Error> {
    let above = |tau: f64| deviate(sets, tau, gamma, spec).map(|d| d > target);
    if !above(low)? || above(high)? {
        return Err(SensitivityError::Parameter(
            "the deviate does not cross its target between the ends of the search",
        )
        .into());
    }
    loop {
        let mid = low + (high - low) / 2.0;
        if mid <= low || mid >= high {
            return Ok(high);
        }
        if above(mid)? {
            low = mid;
        } else {
            high = mid;
        }
    }
}

/// The range of point estimates and the confidence interval of an additive
/// effect under `gamma`, from the m-statistic of `spec` over matched sets of
/// raw responses, the treated unit first in each, as
/// `sensitivitymw::senmwCI` for sets of any size.
///
/// # Errors
///
/// [`crate::Error`] for a `gamma` below 1 or a deviate that does not cross
/// its target in the bracket, and those of [`m_scores`] and [`set_bounds`] at any `tau`
/// tried.
pub fn m_statistic_interval(
    sets: &[Vec<f64>],
    gamma: f64,
    spec: &MScore,
    level: Level,
    sides: Sides,
) -> Result<EffectInterval, Error> {
    check_gamma(gamma)?;
    // Validates the sets and the score.
    m_scores(sets, spec)?;
    let treated = sets.iter().map(|s| s[0]);
    let controls = sets.iter().flat_map(|s| s[1..].iter().copied());
    let (t_min, t_max) = treated.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    });
    let (c_min, c_max) = controls.fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
        (lo.min(v), hi.max(v))
    });
    let (low, high) = (t_min - c_max, t_max - c_min);
    if !(low < high && (high - low).is_finite()) {
        return Err(SensitivityError::Parameter(
            "the responses give no range of effects to search",
        )
        .into());
    }
    let z = normal::quantile(match sides {
        Sides::One => level.confidence_value(),
        Sides::Two => level.upper_probability(),
    });
    let negated: Vec<Vec<f64>> = sets
        .iter()
        .map(|s| s.iter().map(|v| -v).collect())
        .collect();
    let estimate_lower = crossing(sets, gamma, spec, 0.0, low, high)?;
    let estimate_upper = -crossing(&negated, gamma, spec, 0.0, -high, -low)?;
    let lower = crossing(sets, gamma, spec, z, low, high)?;
    let upper = match sides {
        Sides::One => None,
        Sides::Two => Some(-crossing(&negated, gamma, spec, z, -high, -low)?),
    };
    Ok(EffectInterval {
        gamma,
        estimate_lower,
        estimate_upper,
        interval: Interval::new(Some(lower), upper, level),
    })
}
