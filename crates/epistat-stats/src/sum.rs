//! Floating-point sums that do not depend on how the terms happen to round.
//!
//! [`compensated`] is a compensated sum that matches Python's `sum()` over floats,
//! [`exact`] is the correctly rounded sum that matches Python's `math.fsum()`,
//! and [`blocked`] is the pairwise sum that polars uses for a float column.
//!
//! # Compensated sum
//!
//! [`compensated`] is Neumaier's improvement of Kahan's summation \[1\], in the
//! order of operations of CPython 3.12 and later. With running total `s` and
//! compensation `c`, for each term `v`:
//!
//! ```text
//! t = s + v
//! c = c + ((s - t) + v)    if |s| >= |v|
//! c = c + ((v - t) + s)    otherwise
//! s = t
//! ```
//!
//! The expression in `c` is the rounding error of `s + v`, recovered exactly
//! when the larger operand is subtracted first. The result is `s + c`. The
//! compensation is added only when it is finite and non-zero, so a sum with an
//! infinite term stays infinite instead of becoming NaN, as in CPython. The
//! error is not exact in general. It is about `2^-53` times the sum of the
//! absolute values, plus terms of second order in the rounding error.
//!
//! # Exact sum
//!
//! [`exact`] is Shewchuk's method of exact partials \[2\], the algorithm of
//! `math.fsum()`. It keeps a list of non-overlapping partial sums whose exact
//! total equals the exact sum of the terms so far. Adding a term `x` runs through
//! the partials `y`. For each, with `big` the larger and `small` the smaller in
//! absolute value:
//!
//! ```text
//! hi = big + small,    lo = small - (hi - big)
//! ```
//!
//! Here `hi + lo = big + small` exactly. A non-zero `lo` is kept as a partial, and
//! `hi` carries on as `x`. A non-zero last `x` becomes a new partial. The partials
//! are then summed from the largest down until a remainder `lo` is non-zero.
//!
//! The result is rounded half to even. If the remainder `lo` and the next
//! partial have the same sign, the exact sum lies on the far side of a tie or
//! beyond it, so the code doubles `lo`, adds it to `hi`, and uses that sum if the
//! addition was exact. This is the correction that makes the sum correctly
//! rounded in the tie case, where the partial sum alone would round the wrong way.
//! The result does not depend on the order of the terms. Terms must be finite. A
//! sum beyond the range of `f64` is infinite, and an empty sum is 0.
//!
//! # Blocked sum
//!
//! [`blocked`] adds in the order of the float sum of polars 1.41, a
//! pairwise sum over blocks. With `r = len mod 128`, the first `r` terms are
//! added left to right into one total, and the rest, a multiple of 128 terms,
//! is summed recursively. A run of at least two blocks is split after
//! `floor(blocks / 2)` blocks and the two halves are added. A single block of 128
//! terms keeps 16 running sums, one per position modulo 16, each adding
//! its terms in order. The 16 sums are folded onto each other: sum `j` gains sum
//! `j + 8`, then `j + 4`, so four remain, and the block's sum is `(s0 + s2) +
//! (s1 + s3)`. The result is main sum plus remainder sum. It is a fixed
//! sequence of roundings, so it does not depend on a thread count, but it is not
//! the correctly rounded sum.
//!
//! # References
//!
//! 1. Neumaier A. Rundungsfehleranalyse einiger Verfahren zur Summation endlicher
//!    Summen. Zeitschrift für Angewandte Mathematik und Mechanik 1974;54:39-51.
//! 2. Shewchuk JR. Adaptive precision floating-point arithmetic and fast robust
//!    geometric predicates. Discrete and Computational Geometry 1997;18:305-363.

/// A running Neumaier compensated sum, for terms that arrive one at a time.
///
/// [`compensated`] is the sum of an iterator by this accumulator, so a sum
/// built with [`add`](Self::add) and read with [`value`](Self::value) has the
/// bits of [`compensated`] over the same terms.
#[derive(Clone, Copy, Debug, Default)]
pub struct Compensated {
    total: f64,
    compensation: f64,
}

impl Compensated {
    /// An empty sum, 0.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            total: 0.0,
            compensation: 0.0,
        }
    }

    /// Adds `value`.
    pub fn add(&mut self, value: f64) {
        let candidate = self.total + value;
        if self.total.abs() >= value.abs() {
            self.compensation += (self.total - candidate) + value;
        } else {
            self.compensation += (value - candidate) + self.total;
        }
        self.total = candidate;
    }

    /// The sum so far, with the compensation added only when it is finite and
    /// non-zero.
    #[must_use]
    pub fn value(&self) -> f64 {
        if self.compensation != 0.0 && self.compensation.is_finite() {
            self.total + self.compensation
        } else {
            self.total
        }
    }
}

/// Neumaier's compensated sum, in the order of operations of Python's
/// built-in `sum()` over floats (Python 3.12 and later).
///
/// The compensation is added only when it is finite and non-zero, so a sum
/// with an infinite term stays infinite rather than becoming NaN.
pub fn compensated(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = Compensated::new();
    for value in values {
        sum.add(value);
    }
    sum.value()
}

/// The correctly rounded sum of finite `values`, whatever their order, by
/// Shewchuk's exact partials, as Python's `math.fsum()` computes it.
///
/// A sum beyond the range of `f64` is infinite.
#[expect(
    clippy::float_cmp,
    reason = "the algorithm tests whether a rounding was exact"
)]
pub fn exact(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    for value in values {
        let mut x = value;
        let mut kept = 0;
        for j in 0..partials.len() {
            let y = partials[j];
            let (big, small) = if x.abs() < y.abs() { (y, x) } else { (x, y) };
            let hi = big + small;
            let lo = small - (hi - big);
            if lo != 0.0 {
                partials[kept] = lo;
                kept += 1;
            }
            x = hi;
        }
        partials.truncate(kept);
        if x != 0.0 {
            partials.push(x);
        }
    }
    let Some(mut n) = partials.len().checked_sub(1) else {
        return 0.0;
    };
    let mut hi = partials[n];
    let mut lo = 0.0;
    while n > 0 {
        let x = hi;
        n -= 1;
        let y = partials[n];
        hi = x + y;
        lo = y - (hi - x);
        if lo != 0.0 {
            break;
        }
    }
    // Round half to even when the remaining partials decide a tie.
    if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
        let y = lo * 2.0;
        let x = hi + y;
        if y == x - hi {
            hi = x;
        }
    }
    hi
}

const BLOCK: usize = 128;
const LANES: usize = 16;

/// The sum of `values` in the order of the float sum of polars 1.41: the first
/// `len mod 128` terms in order, the remainder as a pairwise sum of blocks of
/// 128 terms with 16 lanes. An empty sum is 0.
#[must_use]
pub fn blocked(values: &[f64]) -> f64 {
    let remainder = values.len() % BLOCK;
    let (rest, main) = values.split_at(remainder);
    let main_sum = if main.is_empty() {
        0.0
    } else {
        blocked_pairwise(main)
    };
    let mut rest_sum = 0.0;
    for &value in rest {
        rest_sum += value;
    }
    main_sum + rest_sum
}

/// `values.len()` is a positive multiple of [`BLOCK`].
fn blocked_pairwise(values: &[f64]) -> f64 {
    if values.len() == BLOCK {
        let mut lanes = [0.0_f64; LANES];
        for chunk in values.as_chunks::<LANES>().0 {
            for (lane, &value) in lanes.iter_mut().zip(chunk) {
                *lane += value;
            }
        }
        let mut width = LANES;
        while width > 4 {
            for j in 0..width / 2 {
                lanes[j] += lanes[width / 2 + j];
            }
            width /= 2;
        }
        return (lanes[0] + lanes[2]) + (lanes[1] + lanes[3]);
    }
    let left = (values.len() / BLOCK / 2) * BLOCK;
    blocked_pairwise(&values[..left]) + blocked_pairwise(&values[left..])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_running_sum_has_the_bits_of_the_iterator_sum() {
        let terms = [1e16, 1.0, -1e16, 0.1, 0.2, 0.3, f64::INFINITY, 1.0];
        for n in 0..=terms.len() {
            let mut sum = Compensated::new();
            for &t in &terms[..n] {
                sum.add(t);
            }
            assert_eq!(
                sum.value().to_bits(),
                compensated(terms[..n].iter().copied()).to_bits()
            );
        }
        assert_eq!(Compensated::new().value().to_bits(), 0.0_f64.to_bits());
    }

    // Each expected value is what Python 3.12 returns.
    #[test]
    fn compensated_matches_python_sum() {
        assert_eq!(f64::to_bits(compensated([])), f64::to_bits(0.0));
        assert_eq!(f64::to_bits(compensated([0.1; 10])), f64::to_bits(1.0));
        assert_eq!(
            f64::to_bits(compensated([0.1, 0.2, 0.3])),
            f64::to_bits(0.6)
        );
        assert_eq!(
            f64::to_bits(compensated([1e100, 1.0, -1e100])),
            f64::to_bits(1.0)
        );
        assert_eq!(
            f64::to_bits(compensated([1.0, f64::INFINITY, 3.0])),
            f64::to_bits(f64::INFINITY)
        );
        assert_eq!(
            f64::to_bits(compensated([1e308, 1e308, -1e308])),
            f64::to_bits(f64::INFINITY)
        );
        assert!(compensated([f64::INFINITY, f64::NEG_INFINITY]).is_nan());
    }

    #[test]
    fn exact_matches_python_fsum() {
        assert_eq!(f64::to_bits(exact([])), f64::to_bits(0.0));
        assert_eq!(exact([-0.0, -0.0]).to_bits(), 0.0_f64.to_bits());
        assert_eq!(f64::to_bits(exact([0.1; 10])), f64::to_bits(1.0));
        assert_eq!(
            f64::to_bits(exact([1.0, 1e-16, 1e-16, -1.0])),
            f64::to_bits(2e-16)
        );
        assert_eq!(
            f64::to_bits(exact([9_007_199_254_740_992.0, 1.0, 1.0, 1.0])),
            f64::to_bits(9_007_199_254_740_996.0)
        );
        assert_eq!(
            f64::to_bits(exact([1e-17, 1.0, -1.0, 3.5e-16])),
            f64::to_bits(3.600_000_000_000_000_3e-16)
        );
        assert_eq!(
            f64::to_bits(exact([4_503_599_627_370_496.5, 1e-300])),
            f64::to_bits(4_503_599_627_370_496.0)
        );
    }
}
