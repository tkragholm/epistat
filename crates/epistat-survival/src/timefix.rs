//! Times that are equal up to rounding error, made equal: the tie policy of
//! `survival::aeqSurv`.
//!
//! Two times that a data pipeline meant to be the same can differ in the last
//! bits, for example a date computed as `a + b - b`. The estimators treat
//! times as tied only when they are exactly equal, so such a pair would split
//! a tied event or start a risk set one step late. [`fix_times`] finds these
//! near-ties and replaces each by one common value before estimation.
//!
//! # Algorithm
//!
//! Let `y` be the finite start and stop values pooled, sorted and
//! de-duplicated, with `n` values and mean absolute value `m`. With the
//! tolerance `tol`, consecutive values `y[i]` and `y[i+1]` are tied when
//!
//! ```text
//! dy = y[i+1] - y[i]
//! dy <= tol   or   dy / m <= tol
//! ```
//!
//! which is the rule of `all.equal`: an absolute comparison for values near
//! zero and a relative one elsewhere. The default tolerance is
//! `sqrt(f64::EPSILON)`, about 1.49e-8. Ties chain. A run of consecutive tied
//! values is one chain, and every value in it becomes the smallest of the
//! chain. A chain can therefore span more than `tol` from end to end, because
//! each link is only compared with its neighbour. Every start and stop is then
//! replaced by the smallest value of its chain, and a value in no chain is
//! unchanged. When no consecutive pair is tied, nothing is returned and the
//! input stands.
//!
//! A row whose start and stop differed but became equal has lost its length,
//! and the call fails. A row that was already of length zero is left alone.
//! The status column is not touched, so it is not an argument.
//!
//! Zero has two representations, and a chain containing both becomes -0.0. R
//! keeps whichever came first.
//!
//! Only values of the input are returned, so the output does not depend on
//! how the replacement is computed. The replacement is a map of each value
//! through the sorted chain minima, and the sort is of values that compare
//! equal only when they are the same number, so the result is the same on any
//! number of threads.
//!
//! # The mean and where it can differ from R
//!
//! `m` is the sum of the absolute values divided by `n`. The sum is the
//! correctly rounded sum ([`epistat_stats::sum::exact`]), then one division.
//! R's `mean` sums in `long double` and adds a second-pass correction. On
//! hardware where `long double` is the 80-bit format the two means can differ
//! in the last bit, and where it is `f64` (Apple silicon, the platform of the
//! reference tests) the correction is also in `f64`. A different `m` changes
//! the outcome only for a pair whose `dy / m` lies exactly on `tol` to within
//! one unit in the last place, which an input of day numbers or of times
//! measured to a fixed precision does not produce. A sum that overflows is an
//! error here, where R divides by infinity and ties everything.
//!
//! # Fast path
//!
//! If every finite value is an integer and `tol < 1` and `max|y| * tol <= 1/2`,
//! no two values can be tied and the sort is skipped. Proof: distinct integers
//! differ by at least one, and the subtraction of two integers in `f64` is
//! monotone, so `dy >= 1`. Then `dy <= tol` is false since `tol < 1`. The mean
//! is at most `max|y|` up to the relative error of one sum, one division and
//! the rounding of `dy / m`, at most a few units of `2^-53`, so
//! `dy / m >= (1 - 4 * 2^-53) / max|y| >= 2 * (1 - 4 * 2^-53) * tol > tol`
//! using `max|y| * tol <= 1/2`. Neither branch of the rule holds, so there is
//! no chain and the result is the same as the full computation. Day numbers
//! (about 2e4) are far inside the bound of about 3.3e7 at the default
//! tolerance.
//!
//! # Differences from `aeqSurv`
//!
//! - R ignores infinite start and stop values when it looks for ties, but then
//!   maps an infinite stop to the largest finite value and breaks on an
//!   infinite start. Here any value that is not finite is an error, as in the
//!   other estimators' checks.
//! - R accepts a negative tolerance and treats it as zero. Here a negative or
//!   non-finite tolerance is an error. `tolerance: 0.0` and [`TimeFix::Exact`]
//!   are the two ways to ask for no change.
//! - A start and stop of different lengths are an error.
//!
//! # References
//!
//! 1. Therneau TM. A Package for Survival Analysis in R. R package `survival`
//!    3.8: `aeqSurv`, and its help page, which describes the rule as that of
//!    `all.equal` and the replacement by the smaller value. The `timefix`
//!    argument of `coxph.control` and `survfit` switches it off.

use rayon::prelude::*;

use crate::check::{check_lengths, refuse_non_finite};
use crate::error::{Error, Result};

/// How times that differ by rounding error are treated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TimeFix {
    /// Times are compared exactly and nothing changes.
    Exact,
    /// Times closer than the tolerance, absolutely or relative to the mean
    /// absolute time, are made equal. A tolerance of zero changes nothing.
    Aeq {
        /// The tolerance, finite and not negative.
        tolerance: f64,
    },
}

impl Default for TimeFix {
    /// `survival`'s default: `Aeq` with `sqrt(f64::EPSILON)`.
    fn default() -> Self {
        Self::Aeq {
            tolerance: f64::EPSILON.sqrt(),
        }
    }
}

/// Times after the tie policy, where it found a tie.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedTimes {
    /// The start times, or `None` when the input had none.
    pub start: Option<Vec<f64>>,
    /// The stop times.
    pub stop: Vec<f64>,
}

/// Apply the tie policy to the times of one data set.
///
/// `start` is `None` for right-censored data, where only `stop` is compared.
/// The result is `None` when nothing is tied, or when the policy changes
/// nothing, and the input stands. Otherwise it holds the full columns.
///
/// # Errors
///
/// [`Error::Input`] when the columns differ in length, a value is not finite,
/// the tolerance is negative or not finite, the sum of the absolute values
/// overflows, or a row whose start and stop differed ends with equal values.
#[expect(
    clippy::float_cmp,
    reason = "a row collapses when its two times become the same number"
)]
pub fn fix_times(
    start: Option<&[f64]>,
    stop: &[f64],
    policy: TimeFix,
) -> Result<Option<FixedTimes>> {
    let tolerance = match policy {
        TimeFix::Exact => 0.0,
        TimeFix::Aeq { tolerance } => {
            if !tolerance.is_finite() || tolerance < 0.0 {
                return Err(Error::input(format!(
                    "time tolerance must be finite and not negative, got {tolerance}"
                )));
            }
            tolerance
        }
    };
    if let Some(start) = start {
        check_lengths(&[("stop", stop.len()), ("start", start.len())])?;
        refuse_non_finite(start, "start")?;
    }
    refuse_non_finite(stop, "stop")?;
    if tolerance == 0.0 || cannot_tie(start, stop, tolerance) {
        return Ok(None);
    }

    let mut pooled: Vec<f64> = Vec::with_capacity(stop.len() * (1 + usize::from(start.is_some())));
    pooled.extend_from_slice(stop);
    if let Some(start) = start {
        pooled.extend_from_slice(start);
    }
    // `total_cmp` orders -0.0 before 0.0, and `dedup` then keeps -0.0, so the
    // sign of a zero in the output does not depend on the input order.
    pooled.sort_by(f64::total_cmp);
    pooled.dedup();
    let unique = pooled;
    if unique.len() < 2 {
        return Ok(None);
    }

    let total = epistat_stats::sum::exact(unique.iter().map(|v| v.abs()));
    if !total.is_finite() {
        return Err(Error::input(
            "times are too large to average; rescale them before estimating",
        ));
    }
    #[expect(
        clippy::cast_precision_loss,
        reason = "the count of unique times is far below 2^53"
    )]
    let mean = total / unique.len() as f64;

    let mut cuts: Vec<f64> = vec![unique[0]];
    let mut any_tied = false;
    for pair in unique.windows(2) {
        let dy = pair[1] - pair[0];
        if dy <= tolerance || dy / mean <= tolerance {
            any_tied = true;
        } else {
            cuts.push(pair[1]);
        }
    }
    if !any_tied {
        return Ok(None);
    }

    let map = |values: &[f64]| -> Vec<f64> {
        values
            .par_iter()
            .map(|&v| cuts[cuts.partition_point(|&c| c <= v) - 1])
            .collect()
    };
    let new_stop = map(stop);
    let new_start = start.map(map);
    if let (Some(old), Some(new)) = (start, new_start.as_deref()) {
        let collapsed = old
            .iter()
            .zip(stop)
            .zip(new.iter().zip(&new_stop))
            .any(|((&a, &b), (&na, &nb))| na == nb && a != b);
        if collapsed {
            return Err(Error::input(
                "an interval has effective length 0 after times equal up to rounding \
                 error are made equal",
            ));
        }
    }
    Ok(Some(FixedTimes {
        start: new_start,
        stop: new_stop,
    }))
}

/// Whether the values are integers small enough that no two can be tied.
///
/// This is the fast path proved in the module documentation.
fn cannot_tie(start: Option<&[f64]>, stop: &[f64], tolerance: f64) -> bool {
    if tolerance >= 1.0 {
        return false;
    }
    let bound = 0.5 / tolerance;
    start
        .into_iter()
        .flatten()
        .chain(stop)
        .all(|&v| v.fract() == 0.0 && v.abs() <= bound)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn a_tied_pair_takes_the_smaller_value() {
        let stop = [1.0, 1.0 + EPS, 5.0];
        let fixed = fix_times(None, &stop, TimeFix::default()).unwrap().unwrap();
        assert_eq!(fixed.stop, [1.0, 1.0, 5.0]);
        assert!(fixed.start.is_none());
    }

    #[test]
    fn exact_zero_and_integers_change_nothing() {
        let stop = [1.0, 1.0 + EPS, 5.0];
        assert!(fix_times(None, &stop, TimeFix::Exact).unwrap().is_none());
        let zero = TimeFix::Aeq { tolerance: 0.0 };
        assert!(fix_times(None, &stop, zero).unwrap().is_none());
        let days = [1.0, 2.0, 3.0];
        assert!(
            fix_times(None, &days, TimeFix::default())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn a_chain_spans_more_than_the_tolerance() {
        let link = 0.9 * f64::EPSILON.sqrt();
        let stop = [0.0, link, 2.0 * link, 3.0 * link, 1.0];
        let fixed = fix_times(None, &stop, TimeFix::default()).unwrap().unwrap();
        assert_eq!(fixed.stop, [0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn a_row_that_loses_its_length_is_refused() {
        let start = [1.0];
        let stop = [1.0 + EPS];
        let error = fix_times(Some(&start), &stop, TimeFix::default()).unwrap_err();
        assert!(error.to_string().contains("effective length 0"));
        // A row of length zero from the start is left alone.
        let start = [1.0, 2.0];
        let stop = [1.0, 3.0];
        let fixed = fix_times(Some(&start), &stop, TimeFix::default()).unwrap();
        assert!(fixed.is_none());
        let start = [1.0, 1.0 + EPS, 4.0];
        let stop = [1.0, 2.0, 5.0];
        let fixed = fix_times(Some(&start), &stop, TimeFix::default())
            .unwrap()
            .unwrap();
        assert_eq!(fixed.start.unwrap(), [1.0, 1.0, 4.0]);
    }

    #[test]
    fn invalid_input_is_refused() {
        let bad = |tolerance| fix_times(None, &[1.0], TimeFix::Aeq { tolerance });
        assert!(bad(-1e-8).is_err());
        assert!(bad(f64::NAN).is_err());
        assert!(bad(f64::INFINITY).is_err());
        assert!(fix_times(None, &[1.0, f64::INFINITY], TimeFix::default()).is_err());
        assert!(fix_times(Some(&[f64::NAN]), &[1.0], TimeFix::default()).is_err());
        assert!(fix_times(Some(&[0.0]), &[1.0, 2.0], TimeFix::default()).is_err());
        assert!(fix_times(None, &[1e308, 1e308 + 1e292], TimeFix::default()).is_err());
    }

    #[test]
    fn the_fast_path_agrees_with_the_full_computation() {
        // Integers up to the bound: the full computation finds no tie either.
        let bound = (0.5 / f64::EPSILON.sqrt()).floor();
        let stop: Vec<f64> = (0..1000).map(|i| bound - f64::from(i)).collect();
        assert!(cannot_tie(None, &stop, f64::EPSILON.sqrt()));
        assert!(!cannot_tie(None, &[bound + 2.0], f64::EPSILON.sqrt()));
        assert!(
            fix_times(None, &stop, TimeFix::default())
                .unwrap()
                .is_none()
        );
    }
}
