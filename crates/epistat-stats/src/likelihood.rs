//! Log-likelihood helpers: log-sum-exp, softmax and information criteria.

/// `ln(sum(exp(values)))`, shifted by the maximum so that no term overflows.
///
/// Returns `-inf` for no values or when every value is `-inf`, and `inf` when
/// any value is `inf`.
#[must_use]
pub fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    if max.is_infinite() {
        return max;
    }
    let sum: f64 = values.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

/// Replaces log-weights by the probabilities they define,
/// `exp(v - log_sum_exp(values))`, and returns the log-sum.
///
/// Values that are all `-inf` are left as they are, since they define no
/// distribution.
pub fn softmax(values: &mut [f64]) -> f64 {
    let log_sum = log_sum_exp(values);
    if log_sum.is_finite() {
        for value in values.iter_mut() {
            *value = (*value - log_sum).exp();
        }
    }
    log_sum
}

/// Akaike's information criterion, `-2 ll + 2 k`, for `k` parameters.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn aic(log_likelihood: f64, parameters: usize) -> f64 {
    -2.0 * log_likelihood + 2.0 * parameters as f64
}

/// Schwarz's Bayesian information criterion, `-2 ll + k ln(n)`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn bic(log_likelihood: f64, parameters: usize, observations: usize) -> f64 {
    -2.0 * log_likelihood + parameters as f64 * (observations as f64).ln()
}

/// Sclove's sample-size adjusted BIC, `-2 ll + k ln((n + 2) / 24)`.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn sabic(log_likelihood: f64, parameters: usize, observations: usize) -> f64 {
    -2.0 * log_likelihood + parameters as f64 * ((observations as f64 + 2.0) / 24.0).ln()
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn log_sum_exp_does_not_overflow() {
        assert!((log_sum_exp(&[1000.0, 1000.0]) - (1000.0 + 2f64.ln())).abs() < 1e-12);
        assert!((log_sum_exp(&[0.0, 1.0f64.ln()]) - 2f64.ln()).abs() < 1e-15);
        assert_eq!(log_sum_exp(&[]), f64::NEG_INFINITY);
        assert_eq!(log_sum_exp(&[f64::NEG_INFINITY; 2]), f64::NEG_INFINITY);
        assert_eq!(log_sum_exp(&[1.0, f64::INFINITY]), f64::INFINITY);
    }

    #[test]
    fn softmax_gives_probabilities() {
        let mut values = [1.0, 2.0, 3.0];
        let log_sum = softmax(&mut values);
        assert!((values.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        assert!((values[2] / values[1] - std::f64::consts::E).abs() < 1e-12);
        assert!((log_sum - log_sum_exp(&[1.0, 2.0, 3.0])).abs() < f64::EPSILON);
    }

    #[test]
    fn criteria() {
        assert!((aic(-100.0, 3) - 206.0).abs() < 1e-12);
        assert!((bic(-100.0, 3, 50) - (200.0 + 3.0 * 50f64.ln())).abs() < 1e-12);
        assert!((sabic(-100.0, 3, 46) - (200.0 + 3.0 * 2f64.ln())).abs() < 1e-12);
    }
}
