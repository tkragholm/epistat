//! Sample quantiles, weighted and unweighted.
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
//! [`weighted_type1`] is the same rule on a weighted sample. With cumulative
//! weights `W[i] = w[1] + ... + w[i]` and total `W[n]`,
//!
//! ```text
//! Q(p) = x[i]   for the smallest i with w[i] > 0 and W[i] >= p W[n]
//! ```
//!
//! This is the inverse of the weighted empirical distribution function, the
//! rule numpy applies for `method="inverted_cdf"` with `weights`. With equal
//! weights `W[i] = i w` and the condition reads `i >= n p`, which is [`type1`]
//! (`ceil(n p)` clamped to `1..=n`). A value with weight zero is never
//! returned, so `p = 0` gives the smallest value that has weight. A weight
//! that is a positive integer `k` acts as `k` copies of the value. Multiplying
//! all weights by a constant changes no result, apart from the rounding of the
//! sums when the weights are not exactly summable. The cumulative weights are
//! plain left-to-right sums, so a rank can differ by one from [`type1`] at a
//! boundary when equal weights such as 0.1 do not add up exactly.
//!
//! There is no weighted [`type7`]. Type 7 interpolates between order
//! statistics at `1 + (n - 1) p`, a position that counts observations, and
//! weights do not say how many observations there are. The frequency-weight
//! form in Hmisc's `wtd.quantile` depends on the scale of the weights, and no
//! standard definition keeps both equal-weight type 7 and scale invariance, so
//! the crate offers none.
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
//! All functions return `None` for an empty sample, for a sample that contains
//! NaN, and for `p` outside `[0, 1]`, including NaN. A quantile of a sample with
//! a missing value does not exist, and a sort would place the NaN at one end and
//! return a number. [`weighted_type1`] also returns `None` for a negative or
//! non-finite weight, a total weight of zero or one that overflows, and for
//! values and weights of different lengths. Weights of zero are allowed.
//! [`type7_ratio`] also returns `None` for a zero denominator, a
//! numerator above the denominator, or an overflow in `numerator * (n - 1)`. The
//! sorted forms assume the sample is sorted. The check for NaN reads the whole
//! sample, which is linear in its length.
//!
//! # References
//!
//! 1. Hyndman RJ, Fan Y. Sample quantiles in statistical packages. The American
//!    Statistician 1996;50:361-365.

/// R's type 7, the default of R and of most software: linear interpolation
/// between the order statistics at `1 + (n - 1) p`.
#[must_use]
pub fn type7(sorted: &[f64], p: f64) -> Option<f64> {
    if has_nan(sorted) {
        return None;
    }
    let (lo, hi, h) = type7_position(sorted.len(), p)?;
    Some(interpolate(sorted[lo], sorted[hi], h))
}

/// [`type7`] for an unsorted sample, which it partially reorders.
#[must_use]
pub fn type7_unsorted(values: &mut [f64], p: f64) -> Option<f64> {
    if has_nan(values) {
        return None;
    }
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
    if n == 0 || denominator == 0 || numerator > denominator || has_nan(sorted) {
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
    if n == 0 || !(0.0..=1.0).contains(&p) || has_nan(sorted) {
        return None;
    }
    // R adds four machine epsilons before taking the floor, so that n p
    // computed a rounding below an integer still counts as that integer.
    let np = n as f64 * p;
    let j = (np + 4.0 * f64::EPSILON).floor();
    let rank = if np > j { j as usize + 1 } else { j as usize };
    Some(sorted[rank.clamp(1, n) - 1])
}

/// The inverse of the weighted empirical distribution function: the smallest
/// value, among those with positive weight, whose cumulative weight reaches
/// `p` times the total. `sorted` is ascending and `weights[i]` belongs to
/// `sorted[i]`. With equal weights the result is [`type1`].
#[must_use]
pub fn weighted_type1(sorted: &[f64], weights: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty()
        || sorted.len() != weights.len()
        || !(0.0..=1.0).contains(&p)
        || has_nan(sorted)
        || weights.iter().any(|w| !w.is_finite() || *w < 0.0)
    {
        return None;
    }
    let total: f64 = weights.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        return None;
    }
    let target = p * total;
    let mut cumulative = 0.0;
    let mut last = None;
    for (x, w) in sorted.iter().zip(weights) {
        if *w > 0.0 {
            cumulative += w;
            last = Some(*x);
            if cumulative >= target {
                return last;
            }
        }
    }
    last
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
    if n == 0 || !(0.0..=1.0).contains(&p) || has_nan(sorted) {
        return None;
    }
    let index = (p * (n - 1) as f64).round() as usize;
    Some(sorted[index.min(n - 1)])
}

fn has_nan(sample: &[f64]) -> bool {
    sample.iter().any(|v| v.is_nan())
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
    fn a_nan_in_the_sample_gives_none() {
        let sample = [1.0, 2.0, f64::NAN];
        assert_eq!(type7(&sample, 0.5), None);
        assert_eq!(type1(&sample, 0.5), None);
        assert_eq!(rounded_rank(&sample, 0.5), None);
        assert_eq!(type7_ratio(&sample, 1, 2), None);
        assert_eq!(type7_unsorted(&mut [f64::NAN, 1.0, 2.0], 0.5), None);
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

    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "the weights are small non-negative integers"
    )]
    fn repeated(values: &[f64], weights: &[f64]) -> Vec<f64> {
        values
            .iter()
            .zip(weights)
            .flat_map(|(x, w)| std::iter::repeat_n(*x, *w as usize))
            .collect()
    }

    const PROBABILITIES: [f64; 13] = [
        0.0, 0.01, 0.1, 0.25, 0.3, 0.5, 0.6, 0.7, 0.75, 0.9, 0.99, 0.999, 1.0,
    ];

    #[test]
    fn equal_weights_give_type1_to_the_bit() {
        let sorted = [-3.0, 0.25, 1.0, 1.0, 2.5, 8.0, 13.0, 21.0, 34.0, 55.0];
        for w in [1.0, 2.0, 0.5, 4.0] {
            let weights = vec![w; sorted.len()];
            for n in 1..=sorted.len() {
                for p in PROBABILITIES {
                    assert_eq!(
                        weighted_type1(&sorted[..n], &weights[..n], p).map(f64::to_bits),
                        type1(&sorted[..n], p).map(f64::to_bits),
                        "n {n} p {p} w {w}"
                    );
                }
            }
        }
    }

    #[test]
    fn integer_weights_equal_the_repeated_sample_to_the_bit() {
        let values = [-2.0, 0.5, 1.0, 3.0, 3.5, 9.0];
        let weights = [3.0, 1.0, 0.0, 5.0, 2.0, 1.0];
        let expanded = repeated(&values, &weights);
        for p in PROBABILITIES
            .into_iter()
            .chain([0.05, 0.2, 0.45, 0.55, 0.8])
        {
            assert_eq!(
                weighted_type1(&values, &weights, p).map(f64::to_bits),
                type1(&expanded, p).map(f64::to_bits),
                "p {p}"
            );
        }
    }

    #[test]
    fn scaling_the_weights_changes_nothing() {
        let values = [1.0, 2.0, 4.0, 8.0, 16.0];
        let weights = [0.5, 1.5, 1.0, 2.0, 0.25];
        for scale in [2.0, 8.0, 0.125, 1024.0] {
            let scaled: Vec<f64> = weights.iter().map(|w| w * scale).collect();
            for p in PROBABILITIES {
                assert_eq!(
                    weighted_type1(&values, &scaled, p).map(f64::to_bits),
                    weighted_type1(&values, &weights, p).map(f64::to_bits),
                    "p {p} scale {scale}"
                );
            }
        }
    }

    #[test]
    fn zero_weights_are_never_returned() {
        let values = [1.0, 2.0, 3.0, 4.0];
        let weights = [0.0, 1.0, 1.0, 0.0];
        assert_eq!(weighted_type1(&values, &weights, 0.0), Some(2.0));
        assert_eq!(weighted_type1(&values, &weights, 1.0), Some(3.0));
        assert_eq!(weighted_type1(&values, &weights, 0.5), Some(2.0));
    }

    #[test]
    fn weighted_type1_is_none_where_it_is_undefined() {
        let values = [1.0, 2.0];
        assert_eq!(weighted_type1(&[], &[], 0.5), None);
        assert_eq!(weighted_type1(&values, &[1.0], 0.5), None);
        assert_eq!(weighted_type1(&values, &[1.0, 1.0], 1.5), None);
        assert_eq!(weighted_type1(&values, &[1.0, 1.0], -0.1), None);
        assert_eq!(weighted_type1(&values, &[1.0, 1.0], f64::NAN), None);
        assert_eq!(weighted_type1(&[1.0, f64::NAN], &[1.0, 1.0], 0.5), None);
        assert_eq!(weighted_type1(&values, &[1.0, f64::NAN], 0.5), None);
        assert_eq!(weighted_type1(&values, &[1.0, -1.0], 0.5), None);
        assert_eq!(weighted_type1(&values, &[1.0, f64::INFINITY], 0.5), None);
        assert_eq!(weighted_type1(&values, &[0.0, 0.0], 0.5), None);
        assert_eq!(weighted_type1(&values, &[f64::MAX, f64::MAX], 0.5), None);
    }
}
