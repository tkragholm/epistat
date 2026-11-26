//! The mean, variance and standard deviation of a sample.
//!
//! # Formulas
//!
//! With `n` values `x_i`,
//!
//! ```text
//! mean     = sum(x_i) / n,
//! variance = sum((x_i - mean)^2) / (n - d),    d = 1 or 0,
//! sd       = sqrt(variance),
//! ```
//!
//! where the divisor is `n - 1`, R's `var` and `sd` and the unbiased
//! estimator, or `n`, the variance of the sample as a population. The
//! deviations are taken from the mean in a second pass rather than through
//! `E[x^2] - mean^2`, which loses precision for a large mean and a small
//! spread.
//!
//! # Summation order
//!
//! Both sums run over the eight fixed lanes of [`crate::sum::lanes`], so the
//! result has the same bits on every target and thread count. It is not the
//! correctly rounded sum; the error bound is that of [`crate::sum::lanes`].
//!
//! # Edge cases
//!
//! `None` for an empty sample, a value that is NaN or infinite, and a
//! variance whose divisor is not positive (one value with `n - 1`).

use crate::sum::lanes;

/// The divisor of a variance.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Divisor {
    /// `n - 1`, R's `var` and `sd`.
    #[default]
    NMinusOne,
    /// `n`.
    N,
}

/// The mean of `values`; `None` for an empty sample or a value that is not
/// finite.
#[must_use]
pub fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(lanes::sum(values) / values.len() as f64)
}

/// The variance of `values` with `divisor`; `None` as for [`mean`] and when
/// the divisor is not positive.
#[must_use]
pub fn variance(values: &[f64], divisor: Divisor) -> Option<f64> {
    mean_variance(values, divisor).map(|(_, v)| v)
}

/// The mean and the standard deviation of `values` with `divisor`; `None` as
/// for [`variance`].
#[must_use]
pub fn mean_sd(values: &[f64], divisor: Divisor) -> Option<(f64, f64)> {
    mean_variance(values, divisor).map(|(m, v)| (m, v.sqrt()))
}

fn mean_variance(values: &[f64], divisor: Divisor) -> Option<(f64, f64)> {
    let mean = mean(values)?;
    let n = values.len();
    let denominator = match divisor {
        Divisor::NMinusOne => n - 1,
        Divisor::N => n,
    };
    if denominator == 0 {
        return None;
    }
    Some((
        mean,
        lanes::centred_squares(values, mean) / denominator as f64,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_values() {
        let x = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert_eq!(mean(&x), Some(5.0));
        // var(x) in R is 32 / 7; the population variance is 4.
        assert!((variance(&x, Divisor::NMinusOne).unwrap() - 32.0 / 7.0).abs() < 1e-15);
        assert_eq!(variance(&x, Divisor::N), Some(4.0));
        assert_eq!(mean_sd(&x, Divisor::N), Some((5.0, 2.0)));
    }

    #[test]
    fn a_large_mean_does_not_cancel_the_spread() {
        let x: Vec<f64> = (0..100).map(|i| 1e9 + f64::from(i % 2)).collect();
        // Half the values are 1e9 and half 1e9 + 1: the population variance
        // is exactly 1/4.
        assert_eq!(variance(&x, Divisor::N), Some(0.25));
    }

    #[test]
    fn edges() {
        assert_eq!(mean(&[]), None);
        assert_eq!(mean(&[1.0, f64::NAN]), None);
        assert_eq!(variance(&[1.0], Divisor::NMinusOne), None);
        assert_eq!(variance(&[1.0], Divisor::N), Some(0.0));
        assert_eq!(variance(&[1.0, f64::INFINITY], Divisor::N), None);
    }
}
