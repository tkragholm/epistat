//! Ranks with ties and the signed rank statistic under biased assignment.
//!
//! The Wilcoxon signed rank statistic of a sample of paired differences is
//! the sum of the ranks of the absolute differences over the pairs whose
//! difference is positive. Without hidden bias each pair is positive with
//! probability one half, and the distribution is the usual one. When the
//! odds of treatment within a pair may differ by up to a factor `gamma`
//! between the two units, each pair is positive with a probability between
//! `1 / (1 + gamma)` and `gamma / (1 + gamma)` \[1\].
//!
//! # Algorithm
//!
//! [`average_ranks`] ranks a sample, giving tied values the mean of their
//! ranks. Every average rank is a multiple of one half, so twice a rank is an
//! integer. [`exact_upper_tail`] computes `P(sum r_i B_i >= t)` for
//! independent Bernoulli(`p`) variables `B_i` and ranks `r_i`, by the
//! recursion over the pairs
//!
//! ```text
//! f_k(s) = (1 - p) f_{k-1}(s) + p f_{k-1}(s - 2 r_k)
//! ```
//!
//! on the integer grid of twice the sum, which is exact up to rounding of the
//! probabilities and handles ties among the ranks. The table has
//! `2 sum r_i + 1` cells and the recursion `n` passes over it, so the
//! computation stops at [`MAX_TABLE_WORK`] cells in all. [`normal_upper_tail`]
//! is the large-sample approximation, with mean `p sum r_i` and variance
//! `p (1 - p) sum r_i^2`.
//!
//! # Validation
//!
//! The exact tail at `p = 1/2` agrees with R's `psignrank` for samples
//! without ties, and at other `p` with a brute-force enumeration over all
//! `2^n` sign patterns; both are in the tests of `epistat-matching`
//! (`r_sensitivity`) and of this module.
//!
//! # References
//!
//! 1. Rosenbaum PR. Observational Studies, 2nd ed. Springer; 2002. Chapter 4.

use crate::normal;

/// The largest `n * (2 sum r_i + 1)` [`exact_upper_tail`] works through.
pub const MAX_TABLE_WORK: f64 = 2.0e9;

/// What goes wrong in the signed rank computations.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SignedRankError {
    /// A value that is not finite.
    NotFinite,
    /// A probability outside `[0, 1]`.
    Probability(f64),
    /// The exact distribution would need more than [`MAX_TABLE_WORK`] cells.
    TooLarge,
}

impl std::fmt::Display for SignedRankError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotFinite => write!(f, "a value is not finite"),
            Self::Probability(p) => write!(f, "{p} is not a probability"),
            Self::TooLarge => write!(
                f,
                "the exact distribution is too large; use the normal approximation"
            ),
        }
    }
}

impl std::error::Error for SignedRankError {}

/// The ranks of `values` from 1 upwards, tied values sharing the mean of
/// their ranks. The result is in the order of `values`.
///
/// # Errors
///
/// [`SignedRankError::NotFinite`] when a value is NaN or infinite.
#[expect(clippy::float_cmp, reason = "tied values are equal")]
pub fn average_ranks(values: &[f64]) -> Result<Vec<f64>, SignedRankError> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(SignedRankError::NotFinite);
    }
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && values[order[end]] == values[order[start]] {
            end += 1;
        }
        // Ranks start + 1 ..= end share their mean.
        let mean = (start + 1 + end) as f64 / 2.0;
        for &i in &order[start..end] {
            ranks[i] = mean;
        }
        start = end;
    }
    Ok(ranks)
}

/// `P(sum r_i B_i >= observed)` for independent Bernoulli(`p`) variables
/// `B_i`, with the ranks `r_i` multiples of one half.
///
/// A rank that is not a multiple of one half is rounded to the nearest
/// one, so the ranks of [`average_ranks`] are exact.
///
/// # Errors
///
/// [`SignedRankError::Probability`] for `p` outside `[0, 1]`,
/// [`SignedRankError::NotFinite`] and [`SignedRankError::TooLarge`].
pub fn exact_upper_tail(ranks: &[f64], observed: f64, p: f64) -> Result<f64, SignedRankError> {
    if !(0.0..=1.0).contains(&p) {
        return Err(SignedRankError::Probability(p));
    }
    if !observed.is_finite() || ranks.iter().any(|r| !r.is_finite() || *r < 0.0) {
        return Err(SignedRankError::NotFinite);
    }
    // The table size is checked in floating point before any conversion, so
    // ranks too large for the table are an error and never an overflow.
    let total_f: f64 = ranks.iter().map(|r| (2.0 * r).round()).sum();
    if !total_f.is_finite() || (ranks.len() as f64) * (total_f + 1.0) > MAX_TABLE_WORK {
        return Err(SignedRankError::TooLarge);
    }
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "ranks are checked non-negative and the table size is bounded"
    )]
    let doubled: Vec<usize> = ranks.iter().map(|r| (2.0 * r).round() as usize).collect();
    let total: usize = doubled.iter().sum();
    let mut f = vec![0.0_f64; total + 1];
    f[0] = 1.0;
    let mut reach = 0;
    for &r in &doubled {
        reach += r;
        for s in (0..=reach).rev() {
            let shifted = if s >= r { f[s - r] } else { 0.0 };
            f[s] = (1.0 - p) * f[s] + p * shifted;
        }
    }
    let threshold = (2.0 * observed).ceil();
    if threshold <= 0.0 {
        return Ok(1.0);
    }
    if threshold > total as f64 {
        return Ok(0.0);
    }
    // Smallest tail terms first.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the threshold is between 0 and the table size"
    )]
    let from = threshold as usize;
    let mut sum = 0.0;
    for &v in f[from..].iter().rev() {
        sum += v;
    }
    Ok(sum.min(1.0))
}

/// The large-sample upper tail of `sum r_i B_i`, `1 - Phi(z)` with the mean
/// `p sum r_i` and the variance `p (1 - p) sum r_i^2`. `None` when the
/// variance is zero.
#[must_use]
pub fn normal_upper_tail(ranks: &[f64], observed: f64, p: f64) -> Option<f64> {
    let mean = p * ranks.iter().sum::<f64>();
    let variance = p * (1.0 - p) * ranks.iter().map(|r| r * r).sum::<f64>();
    if variance > 0.0 && variance.is_finite() {
        Some(normal::cdf(-(observed - mean) / variance.sqrt()))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_share_the_mean_rank() {
        let r = average_ranks(&[3.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(r, vec![3.5, 1.0, 3.5, 2.0]);
    }

    fn brute(ranks: &[f64], observed: f64, p: f64) -> f64 {
        let n = ranks.len();
        let mut total = 0.0;
        for mask in 0..(1_u32 << n) {
            let mut stat = 0.0;
            let mut prob = 1.0;
            for (i, r) in ranks.iter().enumerate() {
                if mask >> i & 1 == 1 {
                    stat += r;
                    prob *= p;
                } else {
                    prob *= 1.0 - p;
                }
            }
            if stat >= observed - 1e-12 {
                total += prob;
            }
        }
        total
    }

    #[test]
    fn exact_tail_matches_enumeration_with_ties() {
        let ranks = [1.0, 2.5, 2.5, 4.0, 5.0, 6.5, 6.5, 8.0];
        for p in [0.5, 0.6, 0.75, 0.9] {
            for observed in [0.0, 3.5, 10.0, 17.5, 25.0, 36.0, 40.0] {
                let a = exact_upper_tail(&ranks, observed, p).unwrap();
                let b = brute(&ranks, observed, p);
                assert!((a - b).abs() < 1e-14, "{p} {observed} {a} {b}");
            }
        }
    }

    #[test]
    fn exact_tail_at_half_for_three_pairs() {
        // Sums of {1,2,3}: 0,1,2,3,3,4,5,6, each with probability 1/8.
        let t = exact_upper_tail(&[1.0, 2.0, 3.0], 5.0, 0.5).unwrap();
        assert!((t - 2.0 / 8.0).abs() < 1e-15);
    }

    #[test]
    fn exact_upper_tail_with_huge_ranks_is_too_large() {
        assert_eq!(
            exact_upper_tail(&[1e19, 1e19], 1.0, 0.5),
            Err(SignedRankError::TooLarge)
        );
    }
}
