//! Deprecated: this module moved to `epistat_epi::signed_rank` in 0.3.0, where its
//! documentation, explainer and tests are. The copy here keeps the items and
//! the results of 0.2 for one release and is removed in 0.4.0.

#![expect(deprecated, reason = "the copy of 0.2 calls its own deprecated items")]

use crate::dist::normal;
use crate::sum::lanes;

/// The largest `n * (2 sum r_i + 1)` [`exact_upper_tail`] works through.
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::signed_rank::MAX_TABLE_WORK`"
)]
pub const MAX_TABLE_WORK: f64 = 2.0e9;

/// What goes wrong in the signed rank computations.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::signed_rank::SignedRankError`"
)]
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
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::signed_rank::average_ranks`"
)]
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
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::signed_rank::exact_upper_tail`"
)]
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
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::signed_rank::normal_upper_tail`"
)]
pub fn normal_upper_tail(ranks: &[f64], observed: f64, p: f64) -> Option<f64> {
    let mean = p * lanes::sum(ranks);
    let variance = p * (1.0 - p) * lanes::sum_squares(ranks);
    if variance > 0.0 && variance.is_finite() {
        Some(normal::cdf(-(observed - mean) / variance.sqrt()))
    } else {
        None
    }
}
