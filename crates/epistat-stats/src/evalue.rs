//! E-values for unmeasured confounding.
//!
//! The E-value of VanderWeele and Ding \[1\] is the smallest strength of
//! association, on the risk ratio scale, that an unmeasured confounder would
//! need to have with both the exposure and the outcome, given the measured
//! covariates, to explain away an observed association. It depends only on the
//! estimate and its confidence limit, so it applies to any design, including
//! a matched one.
//!
//! # Algorithm
//!
//! For an estimated risk ratio `x` the E-value is
//!
//! ```text
//! E(x) = x + sqrt(x (x - 1))          x >= 1
//! E(x) = E(1 / x)                      x < 1
//! ```
//!
//! To move the estimate to a risk ratio `t` other than the null the same
//! quantity is taken for the ratio of the estimate to `t`. With `x` and `t`
//! both on the side above 1 (a value below 1 is replaced by its reciprocal,
//! for both), `E(x, t) = (x + sqrt(x (x - t))) / t` when `t <= x`. When `t > x`
//! the true value lies beyond the estimate, and the E-value is that of the
//! ratio `t / x`, `r + sqrt(r (r - 1))` with `r = t / x`.
//!
//! The confidence limit that matters is the one nearer the true value. If
//! the interval contains the true value, its E-value is 1. The limit on the
//! far side has no E-value.
//!
//! Other effect measures are converted to an approximate risk ratio first:
//!
//! * An odds ratio is the risk ratio when the outcome is rare, and its
//!   square root when it is common \[1\].
//! * A hazard ratio is the risk ratio when the outcome is rare. Otherwise
//!   `(1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1 / HR))` \[1\].
//! * A standardised mean difference `d` is `exp(0.91 d)`. The limits of its
//!   interval are `exp(0.91 d -+ 1.78 se)`, where `1.78` is `1.96 * 0.91`
//!   \[1\].
//!
//! # Validation
//!
//! Checked against `EValue::evalues.RR`, `evalues.OR`, `evalues.HR` and
//! `evalues.MD` 4.1.4 in `epistat-matching`'s `r_sensitivity` test. An
//! estimate that is zero is an error here, where R returns an infinite
//! E-value.
//!
//! # References
//!
//! 1. VanderWeele TJ, Ding P. Sensitivity analysis in observational research:
//!    introducing the E-value. Ann Intern Med. 2017;167(4):268-74.

use std::error::Error;
use std::fmt::{Display, Formatter};

/// What is wrong with an E-value input.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum EValueError {
    /// A ratio that is not finite and positive.
    NotPositive {
        /// The argument.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A value that is not finite where a number is needed.
    NotFinite {
        /// The argument.
        what: &'static str,
    },
    /// A standard error that is negative.
    NegativeStandardError,
    /// A lower limit above the upper limit.
    LimitsReversed,
    /// An estimate outside its own confidence interval.
    EstimateOutsideInterval,
}

impl Display for EValueError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotPositive { what, value } => {
                write!(f, "{what} must be finite and positive, got {value}")
            }
            Self::NotFinite { what } => write!(f, "{what} must be finite"),
            Self::NegativeStandardError => write!(f, "the standard error cannot be negative"),
            Self::LimitsReversed => {
                write!(f, "the lower limit is above the upper limit")
            }
            Self::EstimateOutsideInterval => {
                write!(f, "the estimate is outside its confidence interval")
            }
        }
    }
}

impl Error for EValueError {}

/// The E-values of an estimate and of its confidence limits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EValues {
    /// The risk ratio the E-values are for: the estimate, on the risk ratio
    /// scale.
    pub rr: f64,
    /// The risk ratio of the lower limit, when one was given.
    pub rr_lower: Option<f64>,
    /// The risk ratio of the upper limit, when one was given.
    pub rr_upper: Option<f64>,
    /// The E-value of the estimate.
    pub estimate: f64,
    /// The E-value of the lower limit. It is given when the estimate is above
    /// the true value, and 1 when the interval contains it.
    pub lower: Option<f64>,
    /// The E-value of the upper limit. It is given when the estimate is below
    /// the true value, and 1 when the interval contains it.
    pub upper: Option<f64>,
}

/// Whether the outcome is rare enough to treat an odds or hazard ratio as a
/// risk ratio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// Rare: the ratio is taken as the risk ratio.
    Rare,
    /// Common: the ratio is converted.
    Common,
}

fn positive(what: &'static str, value: f64) -> Result<f64, EValueError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(EValueError::NotPositive { what, value })
    }
}

/// The E-value of a risk ratio `x` for moving it to `true_rr`, which is 1 for
/// the null.
///
/// # Errors
///
/// [`EValueError::NotPositive`] when `x` or `true_rr` is not finite and
/// positive.
pub fn evalue_threshold(x: f64, true_rr: f64) -> Result<f64, EValueError> {
    let mut x = positive("risk ratio", x)?;
    let mut t = positive("true value", true_rr)?;
    if x <= 1.0 {
        x = 1.0 / x;
        t = 1.0 / t;
    }
    if t <= x {
        Ok((x + (x * (x - t)).sqrt()) / t)
    } else {
        let r = t / x;
        Ok(r + (r * (r - 1.0)).sqrt())
    }
}

/// The E-values of a risk ratio and its confidence limits, for moving the
/// estimate and the interval to `true_rr`.
///
/// # Errors
///
/// [`EValueError::NotPositive`] for a ratio that is not positive,
/// [`EValueError::LimitsReversed`] when `lower > upper` and
/// [`EValueError::EstimateOutsideInterval`] when the estimate is not between
/// the limits given.
pub fn evalues_rr(
    estimate: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    true_rr: f64,
) -> Result<EValues, EValueError> {
    positive("risk ratio", estimate)?;
    positive("true value", true_rr)?;
    if let Some(lo) = lower {
        positive("lower limit", lo)?;
    }
    if let Some(hi) = upper {
        positive("upper limit", hi)?;
    }
    if let (Some(lo), Some(hi)) = (lower, upper) {
        if lo > hi {
            return Err(EValueError::LimitsReversed);
        }
    }
    if lower.is_some_and(|lo| estimate < lo) || upper.is_some_and(|hi| estimate > hi) {
        return Err(EValueError::EstimateOutsideInterval);
    }
    let point = evalue_threshold(estimate, true_rr)?;
    let mut lower_e = lower.map(|v| evalue_threshold(v, true_rr)).transpose()?;
    let mut upper_e = upper.map(|v| evalue_threshold(v, true_rr)).transpose()?;
    let crosses = if estimate > true_rr {
        lower.map(|lo| lo < true_rr)
    } else if estimate < true_rr {
        upper.map(|hi| hi > true_rr)
    } else {
        None
    };
    if crosses == Some(true) {
        lower_e = lower_e.map(|_| 1.0);
        upper_e = upper_e.map(|_| 1.0);
    }
    if lower.is_some() || upper.is_some() {
        if estimate > true_rr {
            upper_e = None;
        } else if estimate < true_rr {
            lower_e = None;
        } else {
            lower_e = Some(1.0);
            upper_e = None;
        }
    }
    Ok(EValues {
        rr: estimate,
        rr_lower: lower,
        rr_upper: upper,
        estimate: point,
        lower: lower_e,
        upper: upper_e,
    })
}

/// The risk ratio an odds ratio stands for.
///
/// # Errors
///
/// [`EValueError::NotPositive`] for an odds ratio that is not positive.
pub fn odds_ratio_to_rr(odds_ratio: f64, outcome: Outcome) -> Result<f64, EValueError> {
    let or = positive("odds ratio", odds_ratio)?;
    Ok(match outcome {
        Outcome::Rare => or,
        Outcome::Common => or.sqrt(),
    })
}

/// The risk ratio a hazard ratio stands for.
///
/// # Errors
///
/// [`EValueError::NotPositive`] for a hazard ratio that is not positive.
pub fn hazard_ratio_to_rr(hazard_ratio: f64, outcome: Outcome) -> Result<f64, EValueError> {
    let hr = positive("hazard ratio", hazard_ratio)?;
    Ok(match outcome {
        Outcome::Rare => hr,
        Outcome::Common => {
            (1.0 - 0.5_f64.powf(hr.sqrt())) / (1.0 - 0.5_f64.powf((1.0 / hr).sqrt()))
        }
    })
}

/// The risk ratio a standardised mean difference stands for, `exp(0.91 d)`.
///
/// # Errors
///
/// [`EValueError::NotFinite`] for a difference that is not finite.
pub fn mean_difference_to_rr(difference: f64) -> Result<f64, EValueError> {
    if difference.is_finite() {
        Ok((0.91 * difference).exp())
    } else {
        Err(EValueError::NotFinite { what: "difference" })
    }
}

/// E-values for an odds ratio. The limits and the true value are converted
/// the same way as the estimate.
///
/// # Errors
///
/// As [`evalues_rr`] and [`odds_ratio_to_rr`].
pub fn evalues_or(
    estimate: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    outcome: Outcome,
    true_or: f64,
) -> Result<EValues, EValueError> {
    let convert = |v: f64| odds_ratio_to_rr(v, outcome);
    evalues_rr(
        convert(estimate)?,
        lower.map(convert).transpose()?,
        upper.map(convert).transpose()?,
        convert(true_or)?,
    )
}

/// E-values for a hazard ratio. The limits and the true value are converted
/// the same way as the estimate.
///
/// # Errors
///
/// As [`evalues_rr`] and [`hazard_ratio_to_rr`].
pub fn evalues_hr(
    estimate: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    outcome: Outcome,
    true_hr: f64,
) -> Result<EValues, EValueError> {
    let convert = |v: f64| hazard_ratio_to_rr(v, outcome);
    evalues_rr(
        convert(estimate)?,
        lower.map(convert).transpose()?,
        upper.map(convert).transpose()?,
        convert(true_hr)?,
    )
}

/// E-values for a standardised mean difference with its standard error. The
/// interval is `exp(0.91 d -+ 1.78 se)` on the risk ratio scale. The true
/// value is a difference, 0 for the null.
///
/// # Errors
///
/// [`EValueError::NegativeStandardError`], [`EValueError::NotFinite`], and
/// those of [`evalues_rr`].
pub fn evalues_md(
    estimate: f64,
    standard_error: Option<f64>,
    true_difference: f64,
) -> Result<EValues, EValueError> {
    let rr = mean_difference_to_rr(estimate)?;
    let true_rr = mean_difference_to_rr(true_difference)?;
    let (lower, upper) = match standard_error {
        None => (None, None),
        Some(se) if !se.is_finite() => {
            return Err(EValueError::NotFinite {
                what: "standard error",
            });
        }
        Some(se) if se < 0.0 => return Err(EValueError::NegativeStandardError),
        Some(se) => (
            Some((0.91 * estimate - 1.78 * se).exp()),
            Some((0.91 * estimate + 1.78 * se).exp()),
        ),
    };
    evalues_rr(rr, lower, upper, true_rr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn null_evalue_of_two_is_the_published_value() {
        // VanderWeele and Ding: RR 2 has E-value 2 + sqrt(2) = 3.41.
        let e = evalue_threshold(2.0, 1.0).unwrap();
        assert!((e - (2.0 + 2.0_f64.sqrt())).abs() < 1e-15);
        assert!((evalue_threshold(0.5, 1.0).unwrap() - e).abs() < 1e-15);
    }

    #[test]
    fn the_null_has_evalue_one() {
        assert_eq!(
            evalue_threshold(1.0, 1.0).unwrap().to_bits(),
            1.0_f64.to_bits()
        );
    }

    #[test]
    fn interval_crossing_the_null_gives_one() {
        let e = evalues_rr(1.5, Some(0.9), Some(2.5), 1.0).unwrap();
        assert_eq!(e.lower, Some(1.0));
        assert_eq!(e.upper, None);
    }

    #[test]
    fn protective_estimate_uses_the_upper_limit() {
        let e = evalues_rr(0.5, Some(0.3), Some(0.8), 1.0).unwrap();
        assert_eq!(e.lower, None);
        let expected = evalue_threshold(0.8, 1.0).unwrap();
        assert_eq!(e.upper, Some(expected));
    }

    #[test]
    fn invalid_input_is_an_error() {
        assert!(evalue_threshold(0.0, 1.0).is_err());
        assert!(evalue_threshold(f64::NAN, 1.0).is_err());
        assert!(evalues_rr(2.0, Some(3.0), None, 1.0).is_err());
        assert!(evalues_rr(2.0, Some(2.5), Some(2.0), 1.0).is_err());
        assert!(evalues_md(0.3, Some(-1.0), 0.0).is_err());
    }
}
