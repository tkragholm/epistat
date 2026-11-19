//! Sample quantiles.
//!
//! The functions take a sample sorted ascending and a probability in `[0, 1]`.
//! Types 1 and 7 are those of Hyndman and Fan \[1\] and give the same doubles as
//! R's `quantile()`.
//!
//! # Algorithm
//!
//! Let `x[1] <= ... <= x[n]` be the order statistics, counted from one as R
//! defines them, and `p` the probability. [`type7`], the default of R, uses
//!
//! ```text
//! index = 1 + (n - 1) p
//! lo    = floor(index),    hi = ceil(index),    h = index - lo
//! Q(p)  = (1 - h) x[lo] + h x[hi]
//! ```
//!
//! When `h = 0` or `x[lo] = x[hi]` the result is `x[lo]` unchanged, which is
//! what R does and avoids the rounding of `(1 - h) a + h b` when both terms are
//! equal. The code subtracts one from `lo` and `hi` to index the slice.
//!
//! [`type1`] is the inverse of the empirical distribution function. With
//! `np = n p`, `eps` the machine epsilon and `j = floor(np + 4 eps)`,
//!
//! ```text
//! rank = j + 1   if np > j,    rank = j   otherwise,
//! Q(p) = x[clamp(rank, 1, n)]
//! ```
//!
//! R adds the four epsilons, so that a product `n p` which rounded to just below an
//! integer still counts as that integer.
//!
//! # Other forms
//!
//! - [`type7_ratio`] takes `p = numerator / denominator` as integers. The position
//!   `(n - 1) numerator` is split by integer division into `lo` and a remainder,
//!   and `h = remainder / denominator`. The probability is never rounded. The
//!   result can differ from [`type7`] in the last bits, because `h` is formed
//!   differently.
//! - [`type7_unsorted`] needs no sorted input. It selects the element at `lo` and
//!   then the element at `hi` among the larger ones, using `f64::total_cmp`, and
//!   reorders the slice. It returns the same doubles as [`type7`] on the sorted
//!   sample.
//! - [`rounded_rank`] returns `x` at zero-based position `round(p (n - 1))`, with
//!   halves rounded away from zero. It is not a Hyndman and Fan type.
//!
//! # Edge cases
//!
//! All functions return `None` for an empty sample and for `p` outside `[0, 1]`,
//! including NaN. [`type7_ratio`] also returns `None` for a zero denominator, a
//! numerator above the denominator, or an overflow in `numerator * (n - 1)`. NaN
//! values in a sample are not checked. The sorted forms assume the sample is
//! sorted, and the unsorted form orders NaN by `total_cmp`.
//!
//! # References
//!
//! 1. Hyndman RJ, Fan Y. Sample quantiles in statistical packages. The American
//!    Statistician 1996;50:361-365.

/// R's type 7, the default of R and of most software: linear interpolation
/// between the order statistics at `1 + (n - 1) p`.
#[must_use]
pub fn type7(sorted: &[f64], p: f64) -> Option<f64> {
    let (lo, hi, h) = type7_position(sorted.len(), p)?;
    Some(interpolate(sorted[lo], sorted[hi], h))
}

/// [`type7`] for an unsorted sample, which it partially reorders.
#[must_use]
pub fn type7_unsorted(values: &mut [f64], p: f64) -> Option<f64> {
    let (lo, hi, h) = type7_position(values.len(), p)?;
    values.select_nth_unstable_by(lo, f64::total_cmp);
    let low = values[lo];
    if hi == lo {
        return Some(low);
    }
    let (_, high, _) = values[lo + 1..].select_nth_unstable_by(hi - lo - 1, f64::total_cmp);
    Some(interpolate(low, *high, h))
}

/// [`type7`] at the probability `numerator / denominator`, which avoids
/// rounding the probability.
#[must_use]
pub fn type7_ratio(sorted: &[f64], numerator: usize, denominator: usize) -> Option<f64> {
    let n = sorted.len();
    if n == 0 || denominator == 0 || numerator > denominator {
        return None;
    }
    let position = numerator.checked_mul(n - 1)?;
    let lo = position / denominator;
    let remainder = position % denominator;
    if remainder == 0 {
        return Some(sorted[lo]);
    }
    let h = remainder as f64 / denominator as f64;
    Some(interpolate(sorted[lo], sorted[lo + 1], h))
}

/// R's type 1, the inverse of the empirical distribution function: the
/// smallest order statistic whose rank is at least `n p`.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is a non-negative position below the length"
)]
pub fn type1(sorted: &[f64], p: f64) -> Option<f64> {
    let n = sorted.len();
    if n == 0 || !(0.0..=1.0).contains(&p) {
        return None;
    }
    // R adds four machine epsilons before taking the floor, so that n p
    // computed a rounding below an integer still counts as that integer.
    let np = n as f64 * p;
    let j = (np + 4.0 * f64::EPSILON).floor();
    let rank = if np > j { j as usize + 1 } else { j as usize };
    Some(sorted[rank.clamp(1, n) - 1])
}

/// The order statistic nearest to position `p (n - 1)`, rounding halves away
/// from zero. Not one of the Hyndman and Fan types.
#[must_use]
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is a non-negative position below the length"
)]
pub fn rounded_rank(sorted: &[f64], p: f64) -> Option<f64> {
    let n = sorted.len();
    if n == 0 || !(0.0..=1.0).contains(&p) {
        return None;
    }
    let index = (p * (n - 1) as f64).round() as usize;
    Some(sorted[index.min(n - 1)])
}

/// Zero-based indices of the two order statistics and the weight of the
/// upper one, computed as R does from the one-based index `1 + (n - 1) p`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the value is a non-negative position below the length"
)]
fn type7_position(n: usize, p: f64) -> Option<(usize, usize, f64)> {
    if n == 0 || !(0.0..=1.0).contains(&p) {
        return None;
    }
    let index = 1.0 + (n - 1) as f64 * p;
    let lo = index.floor();
    let hi = index.ceil();
    Some((lo as usize - 1, hi as usize - 1, index - lo))
}

/// `(1 - h) a + h b`, or `a` when the two are equal, as R computes it.
#[expect(
    clippy::float_cmp,
    reason = "equal order statistics are returned unchanged, as R does"
)]
fn interpolate(a: f64, b: f64, h: f64) -> f64 {
    if h == 0.0 || a == b {
        a
    } else {
        (1.0 - h) * a + h * b
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_samples_and_bad_probabilities_give_none() {
        assert_eq!(type7(&[], 0.5), None);
        assert_eq!(type1(&[], 0.5), None);
        assert_eq!(rounded_rank(&[], 0.5), None);
        assert_eq!(type7(&[1.0], 1.5), None);
        assert_eq!(type7(&[1.0], f64::NAN), None);
        assert_eq!(type7_ratio(&[1.0], 1, 0), None);
        assert_eq!(type7_ratio(&[1.0], 3, 2), None);
    }

    #[test]
    fn the_unsorted_and_ratio_forms_agree_with_the_sorted_one() {
        let sorted = [-3.0, 0.25, 1.0, 1.0, 2.5, 8.0, 13.0];
        for (numerator, denominator) in [(0, 4), (1, 4), (1, 3), (1, 2), (5, 7), (4, 4)] {
            let p = numerator as f64 / denominator as f64;
            let mut shuffled = [8.0, 1.0, -3.0, 13.0, 2.5, 1.0, 0.25];
            let expected = type7(&sorted, p).unwrap();
            assert_eq!(
                type7_unsorted(&mut shuffled, p).unwrap().to_bits(),
                expected.to_bits()
            );
            let ratio = type7_ratio(&sorted, numerator, denominator).unwrap();
            assert!((ratio - expected).abs() <= 1e-15 * expected.abs().max(1.0));
        }
    }

    #[test]
    fn rounded_rank_rounds_half_away_from_zero() {
        let sorted = [1.0, 2.0, 3.0];
        assert_eq!(rounded_rank(&sorted, 0.25), Some(2.0));
        assert_eq!(rounded_rank(&sorted, 0.24), Some(1.0));
        assert_eq!(rounded_rank(&sorted, 1.0), Some(3.0));
    }
}
