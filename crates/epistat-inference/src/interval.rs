//! Intervals from an estimate and its standard error, or from bootstrap
//! replicates.
//!
//! Levels are given as `alpha`, the probability outside the interval, which
//! must lie in `(0, 1)`. The replicate functions sort their input in place.
//! A NaN replicate is [`InferenceError::NanReplicate`], since a percentile of a
//! sample with a missing value does not exist.
//!
//! # Normal interval
//!
//! [`normal()`] returns `estimate -/+ z * se`, where `z` is the standard normal
//! quantile at `1 - alpha / 2`.
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
//! - `alpha` outside `(0, 1)` is [`InferenceError::Alpha`].
//! - No replicates is [`InferenceError::Empty`], and a NaN among them is
//!   [`InferenceError::NanReplicate`].
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

use epistat_stats::dist::normal;
use epistat_stats::{compat, quantile};

use crate::InferenceError;

/// How a percentile interval reads its bounds from the sorted replicates.
///
/// Use [`Percentile::Type7`] for new work. The other two rules reproduce the
/// intervals of existing software.
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

/// `estimate -/+ z se` with `z` the normal quantile at `1 - alpha / 2`.
///
/// # Errors
///
/// [`InferenceError::Alpha`] unless `0 < alpha < 1`.
pub fn normal(estimate: f64, se: f64, alpha: f64) -> Result<(f64, f64), InferenceError> {
    check_alpha(alpha)?;
    let margin = normal::quantile(1.0 - alpha / 2.0) * se;
    Ok((estimate - margin, estimate + margin))
}

/// The percentile interval of the replicates, which it sorts.
///
/// # Errors
///
/// [`InferenceError::Empty`] for no replicates, [`InferenceError::NanReplicate`]
/// for a NaN among them and [`InferenceError::Alpha`] unless `0 < alpha < 1`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is a non-negative position below the length"
)]
pub fn percentile(
    replicates: &mut [f64],
    alpha: f64,
    rule: Percentile,
) -> Result<(f64, f64), InferenceError> {
    check_alpha(alpha)?;
    check_replicates(replicates)?;
    replicates.sort_by(f64::total_cmp);
    let sorted = &*replicates;
    let last = sorted.len() - 1;
    let n = sorted.len() as f64;
    Ok(match rule {
        Percentile::Type7 => (
            quantile::type7(sorted, alpha / 2.0).ok_or(InferenceError::Empty)?,
            quantile::type7(sorted, 1.0 - alpha / 2.0).ok_or(InferenceError::Empty)?,
        ),
        Percentile::OutwardFromLast => {
            let lower = (alpha / 2.0 * last as f64).floor() as usize;
            let upper = ((1.0 - alpha / 2.0) * last as f64).ceil() as usize;
            (sorted[lower], sorted[upper.min(last)])
        }
        Percentile::OutwardFromCount => {
            let lower = ((alpha / 2.0 * n).floor() as usize).min(last);
            let upper = (((1.0 - alpha / 2.0) * n).ceil() as usize)
                .saturating_sub(1)
                .clamp(lower, last);
            (sorted[lower], sorted[upper])
        }
    })
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
    alpha: f64,
    trim: f64,
    rule: Percentile,
) -> Result<(f64, f64), InferenceError> {
    check_alpha(alpha)?;
    check_replicates(replicates)?;
    let n = replicates.len();
    let cut = (trim.clamp(0.0, 0.49) * n as f64).floor() as usize;
    if cut == 0 || 2 * cut >= n {
        return percentile(replicates, alpha, rule);
    }
    replicates.sort_by(f64::total_cmp);
    percentile(&mut replicates[cut..n - cut], alpha, rule)
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
    alpha: f64,
) -> Result<(f64, f64), InferenceError> {
    check_alpha(alpha)?;
    check_replicates(replicates)?;
    replicates.sort_by(f64::total_cmp);
    let sorted = &*replicates;
    let n = sorted.len() as f64;
    let below = sorted.iter().filter(|&&v| v < estimate).count() as f64;
    let z0 = compat::quantile_acklam((below / n).clamp(0.5 / n, 1.0 - 0.5 / n));
    let z_lower = compat::quantile_acklam(alpha / 2.0);
    let z_upper = compat::quantile_acklam(1.0 - alpha / 2.0);
    let at = |z: f64| {
        compat::rounded_rank(sorted, compat::cdf_as26(2.0 * z0 + z)).ok_or(InferenceError::Empty)
    };
    Ok((at(z_lower)?, at(z_upper)?))
}

fn check_replicates(replicates: &[f64]) -> Result<(), InferenceError> {
    if replicates.is_empty() {
        Err(InferenceError::Empty)
    } else if replicates.iter().any(|v| v.is_nan()) {
        Err(InferenceError::NanReplicate)
    } else {
        Ok(())
    }
}

fn check_alpha(alpha: f64) -> Result<(), InferenceError> {
    if alpha > 0.0 && alpha < 1.0 {
        Ok(())
    } else {
        Err(InferenceError::Alpha(alpha))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replicates() -> Vec<f64> {
        (0..100).rev().map(f64::from).collect()
    }

    #[test]
    fn the_three_percentile_rules() {
        assert_eq!(
            percentile(&mut replicates(), 0.1, Percentile::Type7).unwrap(),
            (4.95, 94.05)
        );
        assert_eq!(
            percentile(&mut replicates(), 0.1, Percentile::OutwardFromLast).unwrap(),
            (4.0, 95.0)
        );
        assert_eq!(
            percentile(&mut replicates(), 0.1, Percentile::OutwardFromCount).unwrap(),
            (5.0, 94.0)
        );
    }

    #[test]
    fn trimming_and_bias_correction() {
        let trimmed =
            trimmed_percentile(&mut replicates(), 0.1, 0.1, Percentile::OutwardFromLast).unwrap();
        assert_eq!(trimmed, (13.0, 86.0));
        // A point estimate at the median leaves the interval centred.
        let (lower, upper) = bias_corrected(&mut replicates(), 49.5, 0.1).unwrap();
        assert!((lower + upper - 99.0).abs() <= 1.0);
    }

    #[test]
    fn a_nan_replicate_is_an_error() {
        for rule in [
            Percentile::Type7,
            Percentile::OutwardFromLast,
            Percentile::OutwardFromCount,
        ] {
            let mut sample = vec![1.0, f64::NAN, 3.0];
            assert_eq!(
                percentile(&mut sample, 0.1, rule),
                Err(InferenceError::NanReplicate)
            );
        }
        let mut sample: Vec<f64> = (0..20).map(f64::from).collect();
        sample[0] = f64::NAN;
        assert_eq!(
            trimmed_percentile(&mut sample, 0.1, 0.1, Percentile::Type7),
            Err(InferenceError::NanReplicate)
        );
        assert_eq!(
            bias_corrected(&mut sample, 5.0, 0.1),
            Err(InferenceError::NanReplicate)
        );
    }

    #[test]
    fn the_normal_interval() {
        let (lower, upper) = normal(1.0, 0.5, 0.05).unwrap();
        assert!((upper - 1.0 - 1.959_963_984_540_054 * 0.5).abs() < 1e-15);
        assert_eq!((lower + upper).to_bits(), 2.0_f64.to_bits());
        assert_eq!(normal(0.0, 1.0, 0.0), Err(InferenceError::Alpha(0.0)));
    }
}
