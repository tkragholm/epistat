//! Truncation of the weights of the outcome model, as `TrialEmulation`'s
//! `analysis_weights = "p99"` and `"weight_limits"`.

use crate::error::{Error, Result};
use epistat_core::InputError;

/// Where the weights of the rows of the trials are cut.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightLimits {
    /// At the `lower` and `upper` quantiles of the weights, R's type 1 (the
    /// inverse of the empirical distribution function), as `TrialEmulation`
    /// takes them. [`WeightLimits::P99`] is its `"p99"`.
    Quantiles {
        /// The probability of the lower quantile.
        lower: f64,
        /// The probability of the upper quantile.
        upper: f64,
    },
    /// At fixed values, as its `"weight_limits"`.
    Values {
        /// The smallest weight kept.
        lower: f64,
        /// The largest weight kept.
        upper: f64,
    },
}

impl WeightLimits {
    /// The 1st and 99th percentiles.
    pub const P99: Self = Self::Quantiles {
        lower: 0.01,
        upper: 0.99,
    };
}

/// R's type 1 quantile at `p` of the sorted values `x`: with `j = floor(n p +
/// 4 eps)`, `x_(j + 1)` when `n p > j` and `x_(j)` otherwise, counting from 1
/// and with `x_(0) = x_(1)`.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "n p lies in [0, n] and is floored to an index"
)]
fn quantile_type1(x: &[f64], p: f64) -> f64 {
    let n = x.len();
    let np = n as f64 * p;
    let j = (np + 4.0 * f64::EPSILON).floor();
    let at = if np > j { j as usize + 1 } else { j as usize };
    x[at.clamp(1, n) - 1]
}

/// The weights cut to the limits: each weight below the lower limit set to it
/// and each above the upper set to it, as `TrialEmulation`'s `limit_weight`.
///
/// # Errors
///
/// An input error for no weights, a weight that is negative or not finite,
/// quantile probabilities outside `[0, 1]` or limits with the lower above the
/// upper.
pub fn limit_weights(weight: &[f64], limits: WeightLimits) -> Result<Vec<f64>> {
    if weight.is_empty() {
        return Err(Error::from(InputError::Empty { what: "weight" }));
    }
    if let Some(i) = weight.iter().position(|w| !(w.is_finite() && *w >= 0.0)) {
        return Err(Error::invalid(format!(
            "weight: value {i} is negative or not finite"
        )));
    }
    let (lower, upper) = match limits {
        WeightLimits::Quantiles { lower, upper } => {
            if !((0.0..=1.0).contains(&lower) && (0.0..=1.0).contains(&upper) && lower <= upper) {
                return Err(Error::invalid(format!(
                    "limits: the quantiles {lower} and {upper} are not probabilities in order"
                )));
            }
            let mut sorted = weight.to_vec();
            sorted.sort_by(f64::total_cmp);
            (
                quantile_type1(&sorted, lower),
                quantile_type1(&sorted, upper),
            )
        }
        WeightLimits::Values { lower, upper } => {
            if lower.is_nan() || upper.is_nan() || lower > upper {
                return Err(Error::invalid(format!(
                    "limits: {lower} and {upper} are not in order"
                )));
            }
            (lower, upper)
        }
    };
    Ok(weight.iter().map(|&w| w.max(lower).min(upper)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn type_1_quantiles_are_order_statistics() {
        // R: quantile(1:10, c(0, 0.01, 0.1, 0.15, 0.99, 1), type = 1) is
        // 1 1 1 2 10 10.
        let x: Vec<f64> = (1..=10).map(f64::from).collect();
        let q: Vec<f64> = [0.0, 0.01, 0.1, 0.15, 0.99, 1.0]
            .iter()
            .map(|&p| quantile_type1(&x, p))
            .collect();
        assert_eq!(q, vec![1.0, 1.0, 1.0, 2.0, 10.0, 10.0]);
        let w = limit_weights(
            &[5.0, 0.5, 2.0],
            WeightLimits::Values {
                lower: 1.0,
                upper: 3.0,
            },
        )
        .expect("valid");
        assert_eq!(w, vec![3.0, 1.0, 2.0]);
        assert!(
            limit_weights(
                &[1.0],
                WeightLimits::Values {
                    lower: 2.0,
                    upper: 1.0
                }
            )
            .is_err()
        );
    }
}
