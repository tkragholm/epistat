//! Intervals from an estimate and its standard error, or from bootstrap
//! replicates.
//!
//! Levels are given as `alpha`, the probability outside the interval, which
//! must lie in `(0, 1)`. The replicate functions sort their input in place.
//! Replicates are ordered with `f64::total_cmp`, so a NaN sorts to an end
//! rather than failing.
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
//! the rule. With `last = n - 1`:
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
//! - No replicates is [`InferenceError::Empty`].
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

use epistat_stats::{normal, quantile};

use crate::InferenceError;

/// How a percentile interval reads its bounds from the sorted replicates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Percentile {
    /// R's type 7 quantiles at `alpha / 2` and `1 - alpha / 2`.
    Type7,
    /// The replicates at positions `floor(alpha / 2 * (n - 1))` and
    /// `ceil((1 - alpha / 2) * (n - 1))`, counting from 0.
    OutwardFromLast,
    /// The replicates at positions `floor(alpha / 2 * n)` and
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
/// [`InferenceError::Empty`] for no replicates and [`InferenceError::Alpha`]
/// unless `0 < alpha < 1`.
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
    if replicates.is_empty() {
        return Err(InferenceError::Empty);
    }
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
    if replicates.is_empty() {
        return Err(InferenceError::Empty);
    }
    replicates.sort_by(f64::total_cmp);
    let sorted = &*replicates;
    let n = sorted.len() as f64;
    let below = sorted.iter().filter(|&&v| v < estimate).count() as f64;
    let z0 = normal::quantile_acklam((below / n).clamp(0.5 / n, 1.0 - 0.5 / n));
    let z_lower = normal::quantile_acklam(alpha / 2.0);
    let z_upper = normal::quantile_acklam(1.0 - alpha / 2.0);
    let at = |z: f64| {
        quantile::rounded_rank(sorted, normal::cdf_as26(2.0 * z0 + z)).ok_or(InferenceError::Empty)
    };
    Ok((at(z_lower)?, at(z_upper)?))
}

fn check_alpha(alpha: f64) -> Result<(), InferenceError> {
    if alpha > 0.0 && alpha < 1.0 {
        Ok(())
    } else {
        Err(InferenceError::Alpha(alpha))
    }
}

#[cfg(test)]
#[expect(clippy::float_cmp, reason = "the tests compare exact values")]
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
    fn the_normal_interval() {
        let (lower, upper) = normal(1.0, 0.5, 0.05).unwrap();
        assert!((upper - 1.0 - 1.959_963_984_540_054 * 0.5).abs() < 1e-15);
        assert_eq!(lower + upper, 2.0);
        assert_eq!(normal(0.0, 1.0, 0.0), Err(InferenceError::Alpha(0.0)));
    }
}
