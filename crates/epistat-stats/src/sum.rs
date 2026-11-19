//! Floating-point sums that do not depend on how the terms happen to round.

/// Neumaier's compensated sum, in the order of operations of Python's
/// built-in `sum()` over floats (Python 3.12 and later).
///
/// The compensation is added only when it is finite and non-zero, so a sum
/// with an infinite term stays infinite rather than becoming NaN.
pub fn compensated(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut total = 0.0_f64;
    let mut compensation = 0.0_f64;
    for value in values {
        let candidate = total + value;
        if total.abs() >= value.abs() {
            compensation += (total - candidate) + value;
        } else {
            compensation += (value - candidate) + total;
        }
        total = candidate;
    }
    if compensation != 0.0 && compensation.is_finite() {
        total += compensation;
    }
    total
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

#[cfg(test)]
#[expect(clippy::float_cmp, reason = "the tests compare exact values")]
mod tests {
    use super::*;

    // Each expected value is what Python 3.12 returns.
    #[test]
    fn compensated_matches_python_sum() {
        assert_eq!(compensated([]), 0.0);
        assert_eq!(compensated([0.1; 10]), 1.0);
        assert_eq!(compensated([0.1, 0.2, 0.3]), 0.6);
        assert_eq!(compensated([1e100, 1.0, -1e100]), 1.0);
        assert_eq!(compensated([1.0, f64::INFINITY, 3.0]), f64::INFINITY);
        assert_eq!(compensated([1e308, 1e308, -1e308]), f64::INFINITY);
        assert!(compensated([f64::INFINITY, f64::NEG_INFINITY]).is_nan());
    }

    #[test]
    fn exact_matches_python_fsum() {
        assert_eq!(exact([]), 0.0);
        assert_eq!(exact([-0.0, -0.0]).to_bits(), 0.0_f64.to_bits());
        assert_eq!(exact([0.1; 10]), 1.0);
        assert_eq!(exact([1.0, 1e-16, 1e-16, -1.0]), 2e-16);
        assert_eq!(
            exact([9_007_199_254_740_992.0, 1.0, 1.0, 1.0]),
            9_007_199_254_740_996.0
        );
        assert_eq!(
            exact([1e-17, 1.0, -1.0, 3.5e-16]),
            3.600_000_000_000_000_3e-16
        );
        assert_eq!(
            exact([4_503_599_627_370_496.5, 1e-300]),
            4_503_599_627_370_496.0
        );
    }
}
