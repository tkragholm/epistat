//! Intervals from an estimate and its standard error, or from bootstrap
//! replicates. Levels are given as `alpha`, the probability outside the
//! interval.

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
