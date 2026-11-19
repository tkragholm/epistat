//! Log-likelihood helpers: log-sum-exp, softmax and information criteria.
//!
//! # Formulas
//!
//! [`log_sum_exp`] subtracts the maximum `M` so that no term overflows:
//!
//! ```text
//! ln(sum_i exp(v_i)) = M + ln(sum_i exp(v_i - M))
//! ```
//!
//! [`softmax`] replaces `v_i` by `exp(v_i - log_sum_exp(v))`, so that the values
//! sum to 1, and returns the log-sum.
//!
//! With log-likelihood `ll`, `k` parameters and `n` observations, the criteria are
//!
//! ```text
//! AIC   = -2 ll + 2 k                   (Akaike \[1\])
//! BIC   = -2 ll + k ln(n)               (Schwarz \[2\])
//! SABIC = -2 ll + k ln((n + 2) / 24)    (Sclove \[3\])
//! ```
//!
//! [`sabic`] replaces `n` in the BIC by `(n + 2) / 24`, so its penalty is negative
//! for fewer than 22 observations.
//!
//! [`information_criteria`] returns the three together and gives `None` where a
//! criterion is undefined. The BIC needs at least one observation, since
//! `ln(0)` is `-inf`. The SABIC is defined for every `n >= 0`. A NaN
//! log-likelihood has no criteria. The terms are added without a fused
//! multiply-add, so a result can differ in the last bit from one computed with
//! `mul_add`, as the latent class and semicontinuous model packages do.
//!
//! # References
//!
//! 1. Akaike H. A new look at the statistical model identification. IEEE
//!    Transactions on Automatic Control 1974;19:716-723.
//! 2. Schwarz G. Estimating the dimension of a model. The Annals of Statistics
//!    1978;6:461-464.
//! 3. Sclove SL. Application of model-selection criteria to some problems in
//!    multivariate analysis. Psychometrika 1987;52:333-343.

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
pub fn aic(log_likelihood: f64, parameters: usize) -> f64 {
    -2.0 * log_likelihood + 2.0 * parameters as f64
}

/// Schwarz's Bayesian information criterion, `-2 ll + k ln(n)`.
#[must_use]
pub fn bic(log_likelihood: f64, parameters: usize, observations: usize) -> f64 {
    -2.0 * log_likelihood + parameters as f64 * (observations as f64).ln()
}

/// Sclove's sample-size adjusted BIC, `-2 ll + k ln((n + 2) / 24)`.
#[must_use]
pub fn sabic(log_likelihood: f64, parameters: usize, observations: usize) -> f64 {
    -2.0 * log_likelihood + parameters as f64 * ((observations as f64 + 2.0) / 24.0).ln()
}

/// The three information criteria of one fit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InformationCriteria {
    /// Akaike's criterion.
    pub aic: f64,
    /// Schwarz's criterion, `None` for fewer than one observation.
    pub bic: Option<f64>,
    /// Sclove's adjusted criterion.
    pub sabic: f64,
}

/// [`aic`], [`bic`] and [`sabic`] for a fit, `None` when the log-likelihood is
/// NaN.
#[must_use]
pub fn information_criteria(
    log_likelihood: f64,
    parameters: usize,
    observations: usize,
) -> Option<InformationCriteria> {
    if log_likelihood.is_nan() {
        return None;
    }
    Some(InformationCriteria {
        aic: aic(log_likelihood, parameters),
        bic: (observations >= 1).then(|| bic(log_likelihood, parameters, observations)),
        sabic: sabic(log_likelihood, parameters, observations),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_sum_exp_does_not_overflow() {
        assert!((log_sum_exp(&[1000.0, 1000.0]) - (1000.0 + 2f64.ln())).abs() < 1e-12);
        assert!((log_sum_exp(&[0.0, 1.0f64.ln()]) - 2f64.ln()).abs() < 1e-15);
        assert_eq!(
            f64::to_bits(log_sum_exp(&[])),
            f64::to_bits(f64::NEG_INFINITY)
        );
        assert_eq!(
            f64::to_bits(log_sum_exp(&[f64::NEG_INFINITY; 2])),
            f64::to_bits(f64::NEG_INFINITY)
        );
        assert_eq!(
            f64::to_bits(log_sum_exp(&[1.0, f64::INFINITY])),
            f64::to_bits(f64::INFINITY)
        );
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

    #[test]
    fn criteria_are_none_where_undefined() {
        let c = information_criteria(-100.0, 3, 50).unwrap();
        assert_eq!(c.aic.to_bits(), aic(-100.0, 3).to_bits());
        assert_eq!(c.bic.unwrap().to_bits(), bic(-100.0, 3, 50).to_bits());
        assert_eq!(c.sabic.to_bits(), sabic(-100.0, 3, 50).to_bits());
        let one = information_criteria(-1.0, 2, 1).unwrap();
        assert_eq!(one.bic.unwrap().to_bits(), 2.0_f64.to_bits());
        let none = information_criteria(-1.0, 2, 0).unwrap();
        assert_eq!(none.bic, None);
        assert!(none.sabic.is_finite());
        assert_eq!(information_criteria(f64::NAN, 2, 10), None);
    }

    /// `estep.rs` of latent_class: the maximum by `fold`, the sum of the
    /// shifted exponentials and the posterior as `exp(v - log_sum)`.
    #[test]
    fn latent_class_copy_gives_the_same_bits() {
        for values in [
            vec![-3.1, -0.2, -8.75],
            vec![-1000.0, -1001.5, -999.25, -1e3],
            vec![-0.5],
            vec![-2.0; 7],
        ] {
            let max_log = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let sum_exp: f64 = values.iter().map(|&lp| (lp - max_log).exp()).sum();
            let log_sum = max_log + sum_exp.ln();
            let posterior: Vec<f64> = values.iter().map(|&lp| (lp - log_sum).exp()).collect();
            let mut ours = values.clone();
            let ours_log_sum = softmax(&mut ours);
            assert_eq!(ours_log_sum.to_bits(), log_sum.to_bits());
            assert_eq!(log_sum_exp(&values).to_bits(), log_sum.to_bits());
            for (a, b) in ours.iter().zip(&posterior) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }
}
