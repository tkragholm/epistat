//! Steps of the estimators that a caller can run on sums it forms itself.
//!
//! These are not estimators: they take raw slices and assume what an
//! estimator checks before it calls them, apart from the lengths, which they
//! check. They are public for callers that accumulate the same sums as
//! [`aalen_johansen::estimate`](crate::aalen_johansen::estimate) or
//! [`multistate::bootstrap`](crate::multistate::bootstrap) on data of their
//! own, such as reweighted or resampled data, and want the same bits. Prefer
//! the estimators. This module is public through 0.4 and may be narrowed in
//! a later release.

use epistat_core::InputError;

use crate::error::{Error, Result};

/// Advances the Aalen-Johansen estimator over one distinct time.
///
/// `survival` is `S(t-)` on entry and `S(t)` on return. With `Y` the weight
/// at risk just before the time, `d_any` the weighted events of every cause
/// and `d_causes[k]` those of cause `k`,
///
/// ```text
/// F_k  <- F_k + S(t-) d_k / Y,        S(t) = S(t-) (1 - d_any / Y).
/// ```
///
/// `incidence[k]` is `F_k` and gains the increment of cause `k`. The
/// increment is `S(t-) * d / Y`, formed in that order, and the survival is
/// `S(t-) * (1 - d_any / Y)`. A time with nobody at risk, `at_risk <= 0` or
/// NaN, moves nothing. This is the step of
/// [`aalen_johansen::estimate`](crate::aalen_johansen::estimate) and of its
/// bootstrap, so a caller that accumulates the same sums gets the same bits.
///
/// # Errors
///
/// [`InputError::Length`] when `d_causes` and `incidence` differ in length.
pub fn advance(
    survival: &mut f64,
    at_risk: f64,
    d_any: f64,
    d_causes: &[f64],
    incidence: &mut [f64],
) -> Result<()> {
    if d_causes.len() != incidence.len() {
        return Err(InputError::Length {
            what: "d_causes",
            expected: incidence.len(),
            found: d_causes.len(),
        }
        .into());
    }
    crate::aalen_johansen::advance_step(survival, at_risk, d_any, d_causes, incidence);
    Ok(())
}

/// The restricted mean time in each state from `origin` up to `horizon`,
/// from one curve of occupation probabilities.
///
/// The integral of the occupation probability, a sum of rectangles because
/// the curve is a step function. The segment from the origin to the first
/// event time uses `initial`, the starting distribution. `curve` holds one
/// row of `states` probabilities for each of `times`, row major, the row at a
/// time being the distribution from that time on. Rows at or before `origin`
/// only set the current distribution. The totals are 0 when `horizon` equals
/// `origin`. This is the integral of
/// [`multistate::bootstrap`](crate::multistate::bootstrap) with horizons.
///
/// # Errors
///
/// [`InputError::Length`] when `initial` does not have `states` entries or
/// `curve` does not have `states` for each of `times`, and an input error
/// when `horizon` or `origin` is not finite or `horizon` is before
/// `origin`.
pub fn integrate(
    curve: &[f64],
    times: &[f64],
    initial: &[f64],
    states: usize,
    horizon: f64,
    origin: f64,
) -> Result<Vec<f64>> {
    epistat_core::check::same_length("initial", states, initial.len())?;
    let expected = times
        .len()
        .checked_mul(states)
        .ok_or(InputError::TooLarge {
            what: "curve",
            limit: usize::MAX as u64,
            found: u64::MAX,
        })?;
    epistat_core::check::same_length("curve", expected, curve.len())?;
    if !(horizon.is_finite() && origin.is_finite()) {
        return Err(Error::invalid("horizon and origin must be finite"));
    }
    if horizon < origin {
        return Err(Error::invalid(format!(
            "horizon: {horizon} is before the origin {origin}"
        )));
    }
    Ok(crate::multistate::integrate_unchecked(
        curve, times, initial, states, horizon, origin,
    ))
}

/// Codes for the distinct values of a float column, in order of first
/// appearance, by bit pattern, so that `0.0` and `-0.0` are two values and
/// every NaN of one bit pattern is one.
#[must_use]
pub fn float_codes(values: &[f64]) -> Vec<u32> {
    crate::fine_gray::codes_of(values)
}
