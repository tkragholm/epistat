//! The logistic function and its relatives.
//!
//! # Formulas
//!
//! ```text
//! expit(eta)     = 1 / (1 + exp(-eta))           = exp(eta) / (1 + exp(eta)),
//! logit(p)       = ln(p / (1 - p)),
//! log1p_exp(x)   = ln(1 + exp(x))                = x + ln(1 + exp(-x)),
//! log_expit(eta) = ln(expit(eta))                = -log1p_exp(-eta).
//! ```
//!
//! [`expit`] takes the first form for `eta >= 0` and the second below, so the
//! exponential always has a non-positive argument: it neither overflows nor
//! rounds a small tail probability to 0 before it is formed. It does not keep
//! the result off 0 and 1; a caller that needs that clamps, as R's
//! `binomial()$linkinv` does. [`log1p_exp`] likewise takes the form whose
//! exponential has a non-positive argument, so it is accurate for both large
//! positive arguments, where it is `x` plus a small correction, and large
//! negative ones, where it is `exp(x)`. These are the two branches of R's
//! `plogis` and `log1pexp`.
//!
//! The functions use plain operations and no fused multiply-add, so their
//! results are the same on every platform.

/// The logistic function `1 / (1 + exp(-eta))`, R's `plogis(eta)`, evaluated
/// with the exponential of a non-positive argument so that it neither
/// overflows nor loses the small tail.
#[must_use]
pub fn expit(eta: f64) -> f64 {
    if eta >= 0.0 {
        1.0 / (1.0 + (-eta).exp())
    } else {
        let e = eta.exp();
        e / (1.0 + e)
    }
}

/// The log odds `ln(p / (1 - p))`, R's `qlogis(p)`: `-inf` at 0, `inf` at 1
/// and NaN outside `[0, 1]`.
#[must_use]
pub fn logit(p: f64) -> f64 {
    if !(0.0..=1.0).contains(&p) {
        return f64::NAN;
    }
    (p / (1.0 - p)).ln()
}

/// `ln(1 + exp(x))`, without overflow for large `x` and without losing
/// `exp(x)` for very negative `x`.
#[must_use]
pub fn log1p_exp(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// `ln(expit(eta))`, the log-likelihood contribution of an event, accurate
/// where `expit` rounds to 0 or 1.
#[must_use]
pub fn log_expit(eta: f64) -> f64 {
    -log1p_exp(-eta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_branches_meet_and_the_tails_are_kept() {
        assert_eq!(expit(0.0), 0.5);
        // plogis(-800) in R is about 3.7e-348, which underflows; plogis(-700)
        // is exp(-700) to relative 1e-304.
        assert!((expit(-700.0) / (-700.0_f64).exp() - 1.0).abs() < 1e-15);
        assert_eq!(expit(800.0), 1.0);
        for eta in [-30.0, -2.5, -0.1, 0.1, 2.5, 30.0] {
            assert!((expit(eta) + expit(-eta) - 1.0).abs() <= f64::EPSILON);
            assert!((logit(expit(eta)) - eta).abs() <= 1e-15 * eta.abs().max(1.0) * 1e3);
        }
    }

    #[test]
    fn logs_are_accurate_in_both_tails() {
        // ln(1 + e^x) = x + ln(1 + e^-x); at x = 50 the correction is e^-50.
        assert_eq!(log1p_exp(1000.0), 1000.0);
        assert!((log1p_exp(-50.0) / (-50.0_f64).exp() - 1.0).abs() < 1e-15);
        assert!((log1p_exp(0.0) - std::f64::consts::LN_2).abs() <= f64::EPSILON);
        assert!((log_expit(-1000.0) + 1000.0).abs() <= f64::EPSILON * 1000.0);
        for eta in [-5.0, -0.3, 0.0, 0.7, 4.0] {
            assert!((log_expit(eta) - expit(eta).ln()).abs() < 1e-15);
        }
    }

    #[test]
    fn logit_edges() {
        assert_eq!(logit(0.0), f64::NEG_INFINITY);
        assert_eq!(logit(1.0), f64::INFINITY);
        assert_eq!(logit(0.5), 0.0);
        assert!(logit(1.5).is_nan());
        assert!(logit(f64::NAN).is_nan());
    }
}
