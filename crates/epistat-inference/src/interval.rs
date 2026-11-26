//! Intervals from an estimate and its standard error, or from bootstrap
//! replicates.
//!
//! Every function takes the confidence level as a [`Level`] and returns an
//! [`Interval`] that carries it. A level given as an alpha keeps the
//! arithmetic of `1 - alpha / 2`, and one given as a confidence that of
//! `1 - (1 - c) / 2`, so an alpha of `1.0 - 0.95` and a confidence of `0.95`
//! can differ in the last bits; the caller's number is used as given. The
//! replicate functions sort their input in place, except
//! [`percentile_sorted`], which reads replicates already sorted. A NaN
//! replicate is an input error, since a percentile of a sample with a missing
//! value does not exist.
//!
//! # Normal interval
//!
//! [`normal()`] returns `estimate -/+ z * se`, where `z` is the standard normal
//! quantile at the upper probability of the level, `1 - alpha / 2`. A NaN
//! limit, as from a NaN standard error, is `None`; an infinite standard error
//! gives infinite limits.
//!
//! # Percentile intervals
//!
//! [`percentile`] reads the bounds of the percentile interval \[1\]\[2\] from
//! the `n` sorted replicates `x(0) <= ... <= x(n-1)`. [`Percentile`] chooses
//! the rule. Type 7 is the rule for new work: it is R's default, it is the
//! quantile of \[3\] that most software computes, and it does not depend on a
//! convention about which side to round to. The other two reproduce existing
//! software and are kept so that its intervals can be reproduced exactly:
//! `OutwardFromLast` the percentile interval of the PHD kernels, and
//! `OutwardFromCount` that of the semicontinuous cost models. With
//! `last = n - 1`:
//!
//! ```text
//! Type7:            quantiles at alpha/2 and 1 - alpha/2, type 7 of [3]
//! OutwardFromLast:  x(floor(alpha/2 * last)),  x(min(ceil((1 - alpha/2) * last), last))
//! OutwardFromCount: x(floor(alpha/2 * n)),     x(ceil((1 - alpha/2) * n) - 1)
//! ```
//!
//! Type 7 interpolates linearly between order statistics, so it can return a
//! value that is not a replicate. The two outward rules always return
//! replicates and round away from the centre. For `OutwardFromCount` the lower
//! position is capped at `last` and the upper position is raised to at least
//! the lower, so the bounds are in order.
//!
//! [`trimmed_percentile`] first drops `floor(trim * n)` replicates from each
//! end of the sorted sample, with `trim` capped to `[0, 0.49]`, and then
//! applies [`percentile`] to what is left. If nothing would be dropped, or the
//! trim would leave no replicates, it returns the untrimmed interval.
//!
//! # Bias-corrected interval
//!
//! [`bias_corrected`] is the bias-corrected percentile interval of Efron
//! \[4\] without the acceleration constant, so `a = 0`. The bias constant is
//!
//! ```text
//! z0 = Phi^-1( clamp(#{x < estimate} / n, 0.5/n, 1 - 0.5/n) )
//! ```
//!
//! The clamp keeps `z0` finite when the estimate lies outside the replicates.
//! With `z = Phi^-1(alpha/2)` and `Phi^-1(1 - alpha/2)` the lower and upper
//! probabilities are
//!
//! ```text
//! p_lower = Phi(2 z0 + z_lower),    p_upper = Phi(2 z0 + z_upper)
//! ```
//!
//! and each bound is the replicate at the nearest rank of `p * (n - 1)`,
//! rounding halves away from zero. There is no interpolation. The function
//! uses the Abramowitz-Stegun 7.1.26 approximation of `Phi` \[5\], with
//! absolute error below `1.5e-7`, and Acklam's rational approximation of
//! `Phi^-1`, with relative error below `1.2e-9`. Both are fixed
//! approximations rather than the library's full-precision functions, so the
//! interval matches implementations that use them.
//!
//! # Edge cases
//!
//! - No replicates is [`InputError::Empty`], a NaN among them is
//!   [`InputError::NotFinite`] at its position, and replicates that
//!   [`percentile_sorted`] finds out of order are [`InputError::Unsorted`].
//! - One replicate gives that replicate as both bounds.
//!
//! # References
//!
//! 1. Efron B. Bootstrap methods: another look at the jackknife. Ann Stat.
//!    1979;7(1):1-26.
//! 2. Efron B, Tibshirani RJ. An introduction to the bootstrap. New York:
//!    Chapman & Hall; 1993.
//! 3. Hyndman RJ, Fan Y. Sample quantiles in statistical packages. Am Stat.
//!    1996;50(4):361-5.
//! 4. Efron B. Better bootstrap confidence intervals. J Am Stat Assoc.
//!    1987;82(397):171-85.
//! 5. Abramowitz M, Stegun IA, editors. Handbook of mathematical functions.
//!    Washington, DC: National Bureau of Standards; 1964.

use epistat_core::{InputError, Interval, Level, check};
use epistat_stats::{normal, quantile};

use crate::{Error, empty};

/// How a percentile interval reads its bounds from the sorted replicates.
///
/// Use [`Percentile::Type7`] for new work. The other two rules reproduce the
/// intervals of existing software.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Percentile {
    /// R's type 7 quantiles at `alpha / 2` and `1 - alpha / 2`.
    Type7,
    /// For reproducing the PHD kernels' `percentile_ci`. The replicates at
    /// positions `floor(alpha / 2 * (n - 1))` and
    /// `ceil((1 - alpha / 2) * (n - 1))`, counting from 0.
    OutwardFromLast,
    /// For reproducing semicontinuous_models' `bootstrap_percentile_ci`. The
    /// replicates at positions `floor(alpha / 2 * n)` and
    /// `ceil((1 - alpha / 2) * n) - 1`, counting from 0.
    OutwardFromCount,
}

/// `estimate -/+ z se` with `z` the normal quantile at the level's upper
/// probability, `1 - alpha / 2`. A NaN limit is `None`.
#[must_use]
pub fn normal(estimate: f64, se: f64, level: Level) -> Interval {
    let margin = normal::quantile(level.upper_probability()) * se;
    let defined = |v: f64| (!v.is_nan()).then_some(v);
    Interval::new(
        defined(estimate - margin),
        defined(estimate + margin),
        level,
    )
}

/// The percentile interval of the replicates, which it sorts.
///
/// # Errors
///
/// [`InputError::Empty`] for no replicates and [`InputError::NotFinite`] for a
/// NaN among them.
pub fn percentile(
    replicates: &mut [f64],
    level: Level,
    rule: Percentile,
) -> Result<Interval, Error> {
    check_replicates(replicates)?;
    replicates.sort_by(f64::total_cmp);
    sorted_bounds(replicates, level, rule)
}

/// The percentile interval of replicates sorted in increasing order, which it
/// leaves as they are. [`percentile`] sorts a copy the caller can lend.
///
/// # Errors
///
/// As [`percentile`], and [`InputError::Unsorted`] for replicates out of
/// order.
pub fn percentile_sorted(
    sorted: &[f64],
    level: Level,
    rule: Percentile,
) -> Result<Interval, Error> {
    check_replicates(sorted)?;
    check::sorted_by("replicates", sorted, f64::total_cmp)?;
    sorted_bounds(sorted, level, rule)
}

#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is a non-negative position below the length"
)]
fn sorted_bounds(sorted: &[f64], level: Level, rule: Percentile) -> Result<Interval, Error> {
    let last = sorted.len() - 1;
    let n = sorted.len() as f64;
    let (lower_p, upper_p) = (level.lower_probability(), level.upper_probability());
    let (lower, upper) = match rule {
        Percentile::Type7 => (
            quantile::type7(sorted, lower_p).ok_or_else(|| empty("replicates"))?,
            quantile::type7(sorted, upper_p).ok_or_else(|| empty("replicates"))?,
        ),
        Percentile::OutwardFromLast => {
            let lower = (lower_p * last as f64).floor() as usize;
            let upper = (upper_p * last as f64).ceil() as usize;
            (sorted[lower], sorted[upper.min(last)])
        }
        Percentile::OutwardFromCount => {
            let lower = ((lower_p * n).floor() as usize).min(last);
            let upper = ((upper_p * n).ceil() as usize)
                .saturating_sub(1)
                .clamp(lower, last);
            (sorted[lower], sorted[upper])
        }
    };
    Ok(Interval::new(Some(lower), Some(upper), level))
}

/// [`percentile`] after dropping `floor(trim * n)` replicates from each tail,
/// with `trim` capped at 0.49. Without trimming, or when the trim would leave
/// nothing, it is the untrimmed interval.
///
/// # Errors
///
/// As [`percentile`].
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is a non-negative position below the length"
)]
pub fn trimmed_percentile(
    replicates: &mut [f64],
    level: Level,
    trim: f64,
    rule: Percentile,
) -> Result<Interval, Error> {
    check_replicates(replicates)?;
    let n = replicates.len();
    let cut = (trim.clamp(0.0, 0.49) * n as f64).floor() as usize;
    if cut == 0 || 2 * cut >= n {
        return percentile(replicates, level, rule);
    }
    replicates.sort_by(f64::total_cmp);
    percentile(&mut replicates[cut..n - cut], level, rule)
}

/// Efron's bias-corrected percentile interval, without acceleration.
///
/// `z0` is the normal quantile of the share of replicates below `estimate`,
/// kept within `[0.5 / n, 1 - 0.5 / n]`, and the bounds are the replicates at
/// probabilities `Phi(2 z0 -/+ z)`, read at the nearest rank of `p (n - 1)`.
/// It uses the Abramowitz-Stegun normal CDF and Acklam's quantile, as the
/// cost models whose intervals it reproduces did.
///
/// # Errors
///
/// As [`percentile`].
pub fn bias_corrected(
    replicates: &mut [f64],
    estimate: f64,
    level: Level,
) -> Result<Interval, Error> {
    check_replicates(replicates)?;
    replicates.sort_by(f64::total_cmp);
    let sorted = &*replicates;
    let n = sorted.len() as f64;
    let below = sorted.iter().filter(|&&v| v < estimate).count() as f64;
    let z0 = normal::quantile_acklam((below / n).clamp(0.5 / n, 1.0 - 0.5 / n));
    let z_lower = normal::quantile_acklam(level.lower_probability());
    let z_upper = normal::quantile_acklam(level.upper_probability());
    let at = |z: f64| {
        quantile::rounded_rank(sorted, normal::cdf_as26(2.0 * z0 + z))
            .ok_or_else(|| empty("replicates"))
    };
    Ok(Interval::new(Some(at(z_lower)?), Some(at(z_upper)?), level))
}

fn check_replicates(replicates: &[f64]) -> Result<(), Error> {
    check::non_empty("replicates", replicates.len())?;
    match replicates.iter().position(|v| v.is_nan()) {
        Some(row) => Err(InputError::NotFinite {
            what: "replicates",
            row,
        }
        .into()),
        None => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replicates() -> Vec<f64> {
        (0..100).rev().map(f64::from).collect()
    }

    fn alpha(a: f64) -> Level {
        Level::alpha(a).unwrap()
    }

    fn bounds(interval: Interval) -> (f64, f64) {
        (interval.lower.unwrap(), interval.upper.unwrap())
    }

    #[test]
    fn the_three_percentile_rules() {
        let at = |rule| bounds(percentile(&mut replicates(), alpha(0.1), rule).unwrap());
        assert_eq!(at(Percentile::Type7), (4.95, 94.05));
        assert_eq!(at(Percentile::OutwardFromLast), (4.0, 95.0));
        assert_eq!(at(Percentile::OutwardFromCount), (5.0, 94.0));
        let interval = percentile(&mut replicates(), alpha(0.1), Percentile::Type7).unwrap();
        assert_eq!(interval.level, alpha(0.1));
    }

    #[test]
    fn sorted_replicates_give_the_same_interval_without_a_copy() {
        let mut sorted = replicates();
        sorted.sort_by(f64::total_cmp);
        for rule in [
            Percentile::Type7,
            Percentile::OutwardFromLast,
            Percentile::OutwardFromCount,
        ] {
            assert_eq!(
                percentile_sorted(&sorted, alpha(0.1), rule).unwrap(),
                percentile(&mut replicates(), alpha(0.1), rule).unwrap()
            );
        }
        assert!(matches!(
            percentile_sorted(&replicates(), alpha(0.1), Percentile::Type7),
            Err(Error::Input(InputError::Unsorted { row: 1, .. }))
        ));
    }

    #[test]
    fn trimming_and_bias_correction() {
        let trimmed = trimmed_percentile(
            &mut replicates(),
            alpha(0.1),
            0.1,
            Percentile::OutwardFromLast,
        )
        .unwrap();
        assert_eq!(bounds(trimmed), (13.0, 86.0));
        // A point estimate at the median leaves the interval centred.
        let (lower, upper) = bounds(bias_corrected(&mut replicates(), 49.5, alpha(0.1)).unwrap());
        assert!((lower + upper - 99.0).abs() <= 1.0);
    }

    #[test]
    fn a_nan_replicate_is_an_error() {
        let nan_at = |row| {
            Err(Error::Input(InputError::NotFinite {
                what: "replicates",
                row,
            }))
        };
        for rule in [
            Percentile::Type7,
            Percentile::OutwardFromLast,
            Percentile::OutwardFromCount,
        ] {
            let mut sample = vec![1.0, f64::NAN, 3.0];
            assert_eq!(percentile(&mut sample, alpha(0.1), rule), nan_at(1));
        }
        let mut sample: Vec<f64> = (0..20).map(f64::from).collect();
        sample[0] = f64::NAN;
        assert_eq!(
            trimmed_percentile(&mut sample, alpha(0.1), 0.1, Percentile::Type7),
            nan_at(0)
        );
        assert_eq!(bias_corrected(&mut sample, 5.0, alpha(0.1)), nan_at(0));
        assert!(matches!(
            percentile(&mut [], alpha(0.1), Percentile::Type7),
            Err(Error::Input(InputError::Empty { .. }))
        ));
    }

    #[test]
    fn the_normal_interval() {
        let (lower, upper) = bounds(normal(1.0, 0.5, alpha(0.05)));
        assert!((upper - 1.0 - 1.959_963_984_540_054 * 0.5).abs() < 1e-15);
        assert_eq!((lower + upper).to_bits(), 2.0_f64.to_bits());
        let confidence = normal(1.0, 0.5, Level::default());
        assert!((confidence.upper.unwrap() - upper).abs() < 1e-15);
        assert_eq!(normal(0.0, f64::NAN, Level::default()).lower, None);
    }
}
