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
//! The bound behind the E-value is the bounding factor of Ding and VanderWeele
//! \[2\]: a confounder with risk ratio `RR_EU` with the exposure and `RR_UD`
//! with the outcome, given the measured covariates, can move a risk ratio by
//! at most `RR_EU RR_UD / (RR_EU + RR_UD - 1)`, whatever its prevalence
//! ([`bounding_factor`]). The E-value is the common value of the two at which
//! the bound reaches the observed ratio. A measured covariate's two
//! associations, put into the bound, give a benchmark.
//!
//! A risk difference is moved by a bias factor through both risks and the
//! exposed share, and its E-values are in [`evalues_rd`].
//!
//! Selection that depends on a factor `U` which also affects the outcome has
//! an E-value of its own \[3\]: the common value of the risk ratios of `U`
//! with selection and with the outcome, within exposure groups, that would
//! explain the estimate. It depends on what is assumed about the selection,
//! [`SelectionBias`], and is computed by [`selection_evalues_rr`].
//!
//! # Validation
//!
//! Checked against `EValue::evalues.RR`, `evalues.OR`, `evalues.HR` and
//! `evalues.MD` 4.1.4 in `epistat-matching`'s `r_sensitivity` test, and
//! against `evalues.RD` and `svalues.RR` 4.1.4 in this crate's `r_evalue`
//! test. An estimate that is zero is an error here, where R returns an
//! infinite E-value.
//!
//! # References
//!
//! 1. VanderWeele TJ, Ding P. Sensitivity analysis in observational research:
//!    introducing the E-value. Ann Intern Med. 2017;167(4):268-74.
//! 2. Ding P, VanderWeele TJ. Sensitivity analysis without assumptions.
//!    Epidemiology. 2016;27(3):368-77.
//! 3. Smith LH, VanderWeele TJ. Bounding bias due to selection. Epidemiology.
//!    2019;30(4):509-16.

use crate::Error;
use epistat_core::{Estimate, Interval, Level};
use std::fmt::{Display, Formatter};

/// What is wrong with an E-value input.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub(crate) enum EValueError {
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
    /// A risk ratio of a confounder below 1, where at least 1 is needed.
    BelowOne {
        /// The argument.
        what: &'static str,
        /// Its value.
        value: f64,
    },
    /// A table that cannot be used.
    Table(&'static str),
    /// A risk difference below zero, where the exposure groups should be
    /// exchanged.
    NegativeDifference,
    /// A true value at or above the estimate.
    TrueNotBelowEstimate,
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
            Self::BelowOne { what, value } => {
                write!(f, "{what} must be at least 1, got {value}")
            }
            Self::Table(why) => write!(f, "invalid table: {why}"),
            Self::NegativeDifference => write!(
                f,
                "the risk difference is negative; exchange the exposure groups"
            ),
            Self::TrueNotBelowEstimate => {
                write!(f, "the true value must be below the estimate")
            }
        }
    }
}

impl std::error::Error for EValueError {}

/// The E-values of an estimate and of its confidence limits.
///
/// The fields are E-values and the risk ratios they are for, not an estimate
/// with an interval: the limits come in as numbers without a level, and the
/// E-value of a limit is not a limit of the E-value.
#[non_exhaustive]
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

fn positive(what: &'static str, value: f64) -> Result<f64, Error> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(EValueError::NotPositive { what, value }.into())
    }
}

/// The E-value of a risk ratio `x` for moving it to `true_rr`, which is 1 for
/// the null.
///
/// # Errors
///
/// [`crate::Error`] when `x` or `true_rr` is not finite and
/// positive.
pub fn evalue_threshold(x: f64, true_rr: f64) -> Result<f64, Error> {
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
/// [`crate::Error`] for a ratio that is not positive,
/// [`crate::Error`] when `lower > upper` and
/// [`crate::Error`] when the estimate is not between
/// the limits given.
pub fn evalues_rr(
    estimate: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    true_rr: f64,
) -> Result<EValues, Error> {
    interval_evalues(estimate, lower, upper, true_rr, evalue_threshold)
}

/// The E-values of an estimate and its limits by `threshold`, with the rules
/// for the limit nearer the true value: 1 when the interval contains it, and
/// none for the limit on the far side.
fn interval_evalues(
    estimate: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    true_rr: f64,
    threshold: impl Fn(f64, f64) -> Result<f64, Error>,
) -> Result<EValues, Error> {
    positive("risk ratio", estimate)?;
    positive("true value", true_rr)?;
    if let Some(lo) = lower {
        positive("lower limit", lo)?;
    }
    if let Some(hi) = upper {
        positive("upper limit", hi)?;
    }
    if let (Some(lo), Some(hi)) = (lower, upper)
        && lo > hi
    {
        return Err(EValueError::LimitsReversed.into());
    }
    if lower.is_some_and(|lo| estimate < lo) || upper.is_some_and(|hi| estimate > hi) {
        return Err(EValueError::EstimateOutsideInterval.into());
    }
    let point = threshold(estimate, true_rr)?;
    let mut lower_e = lower.map(|v| threshold(v, true_rr)).transpose()?;
    let mut upper_e = upper.map(|v| threshold(v, true_rr)).transpose()?;
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
/// [`crate::Error`] for an odds ratio that is not positive.
pub fn odds_ratio_to_rr(odds_ratio: f64, outcome: Outcome) -> Result<f64, Error> {
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
/// [`crate::Error`] for a hazard ratio that is not positive.
pub fn hazard_ratio_to_rr(hazard_ratio: f64, outcome: Outcome) -> Result<f64, Error> {
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
/// [`crate::Error`] for a difference that is not finite.
pub fn mean_difference_to_rr(difference: f64) -> Result<f64, Error> {
    if difference.is_finite() {
        Ok((0.91 * difference).exp())
    } else {
        Err(EValueError::NotFinite { what: "difference" }.into())
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
) -> Result<EValues, Error> {
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
) -> Result<EValues, Error> {
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
/// [`crate::Error`], [`crate::Error`], and
/// those of [`evalues_rr`].
pub fn evalues_md(
    estimate: f64,
    standard_error: Option<f64>,
    true_difference: f64,
) -> Result<EValues, Error> {
    let rr = mean_difference_to_rr(estimate)?;
    let true_rr = mean_difference_to_rr(true_difference)?;
    let (lower, upper) = match standard_error {
        None => (None, None),
        Some(se) if !se.is_finite() => {
            return Err(EValueError::NotFinite {
                what: "standard error",
            }
            .into());
        }
        Some(se) if se < 0.0 => return Err(EValueError::NegativeStandardError.into()),
        Some(se) => (
            Some((0.91 * estimate - 1.78 * se).exp()),
            Some((0.91 * estimate + 1.78 * se).exp()),
        ),
    };
    evalues_rr(rr, lower, upper, true_rr)
}

/// The largest factor by which a confounder can move a risk ratio, given its
/// risk ratio `rr_eu` with the exposure and `rr_ud` with the outcome, both at
/// least 1: `rr_eu rr_ud / (rr_eu + rr_ud - 1)` \[2\]. The observed risk ratio
/// divided by it is the most the confounder could take it towards the null.
/// An infinite argument gives the other argument.
///
/// # Errors
///
/// [`crate::Error`] for an argument below 1 or not a number.
pub fn bounding_factor(rr_eu: f64, rr_ud: f64) -> Result<f64, Error> {
    for (what, value) in [
        ("the risk ratio with the exposure", rr_eu),
        ("the risk ratio with the outcome", rr_ud),
    ] {
        if value.is_nan() || value < 1.0 {
            return Err(EValueError::BelowOne { what, value }.into());
        }
    }
    if rr_eu.is_infinite() {
        return Ok(rr_ud);
    }
    if rr_ud.is_infinite() {
        return Ok(rr_eu);
    }
    Ok(rr_eu * rr_ud / (rr_eu + rr_ud - 1.0))
}

/// What is assumed about selection when computing a selection-bias E-value
/// \[3\]. The names follow `EValue::svalues.RR`'s arguments. `U` is a factor
/// that affects both selection `S` and the outcome `Y`; `A` is the exposure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectionBias {
    /// No assumption, the default of `svalues.RR`. The E-value is the common
    /// value of `RR_UY|A=0`, `RR_UY|A=1`, `RR_SU|A=0` and `RR_SU|A=1`.
    General,
    /// The inference is about the selected population only (`sel_pop`). The
    /// E-value is the common value of `RR_UY|S=1` and `RR_AU|S=1`.
    SelectedPopulation,
    /// The risk of the outcome is higher in the selected exposed than in the
    /// unselected exposed (`risk_inc`). The E-value is the common value of
    /// `RR_UY|A=1` and `RR_SU|A=1`.
    IncreasedRisk,
    /// The risk of the outcome is lower in the selected unexposed than in the
    /// unselected unexposed (`risk_dec`). The E-value is the common value of
    /// `RR_UY|A=0` and `RR_SU|A=0`.
    DecreasedRisk,
    /// The factor is selection itself, `S = U` (`S_eq_U`). The E-value is the
    /// common value of `RR_UY|A=0` and `RR_UY|A=1`.
    SelectionIsTheFactor,
    /// `S = U` and increased risk. The E-value is `RR_UY|A=1`.
    SelectionIsTheFactorIncreasedRisk,
    /// `S = U` and decreased risk. The E-value is `RR_UY|A=0`.
    SelectionIsTheFactorDecreasedRisk,
}

/// The selection-bias E-value of a risk ratio `x` for moving it to `true_rr`,
/// as `EValue::threshold_selection`. With `x` and `true_rr` on the side above
/// 1 (a value below 1 is replaced by its reciprocal, for both) and `r` the
/// ratio of the larger to the smaller, it is `E(sqrt(r))` for
/// [`SelectionBias::General`], `sqrt(r)` when selection is the factor, `r`
/// when selection is the factor and the risk is increased or decreased, and
/// `E(r)` otherwise, with `E(x) = x + sqrt(x (x - 1))`.
///
/// # Errors
///
/// [`crate::Error`] when `x` or `true_rr` is not finite and
/// positive.
pub fn selection_evalue_threshold(
    x: f64,
    true_rr: f64,
    assumption: SelectionBias,
) -> Result<f64, Error> {
    let mut x = positive("risk ratio", x)?;
    let mut t = positive("true value", true_rr)?;
    if x <= 1.0 {
        x = 1.0 / x;
        t = 1.0 / t;
    }
    let r = if t <= x { x / t } else { t / x };
    let e = |v: f64| v + (v * (v - 1.0)).sqrt();
    Ok(match assumption {
        SelectionBias::General => e(r.sqrt()),
        SelectionBias::SelectedPopulation
        | SelectionBias::IncreasedRisk
        | SelectionBias::DecreasedRisk => e(r),
        SelectionBias::SelectionIsTheFactor => r.sqrt(),
        SelectionBias::SelectionIsTheFactorIncreasedRisk
        | SelectionBias::SelectionIsTheFactorDecreasedRisk => r,
    })
}

/// The selection-bias E-values of a risk ratio and its confidence limits
/// \[3\], as `EValue::svalues.RR`: how strong the selection would have to be,
/// under `assumption`, to move the estimate and the interval to `true_rr`.
/// The rules for the limits are those of [`evalues_rr`]. An odds or hazard
/// ratio is converted first with [`odds_ratio_to_rr`] or
/// [`hazard_ratio_to_rr`], as `svalues.OR` and `svalues.HR` do.
///
/// # Errors
///
/// As [`evalues_rr`].
pub fn selection_evalues_rr(
    estimate: f64,
    lower: Option<f64>,
    upper: Option<f64>,
    true_rr: f64,
    assumption: SelectionBias,
) -> Result<EValues, Error> {
    interval_evalues(estimate, lower, upper, true_rr, |x, t| {
        selection_evalue_threshold(x, t, assumption)
    })
}

/// The E-values of a risk difference and of its lower confidence limit.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RdEValues {
    /// The risk difference of the table with its standard error and its Wald
    /// interval at the level given. The E-value of the limit is for the
    /// lower limit; there is no p-value.
    pub risk_difference: Estimate,
    /// The bias factor that moves the risk difference to the true value.
    pub bias_factor: f64,
    /// The bias factor that moves the lower limit to the true value, 1 when
    /// the interval contains it.
    pub bias_factor_lower: f64,
    /// The E-value of the estimate.
    pub estimate: f64,
    /// The E-value of the lower limit, 1 when the interval contains the true
    /// value.
    pub lower: f64,
}

/// The E-values of the risk difference of a table and of its lower limit, as
/// `EValue::evalues.RD` \[2\]. The table is that of [`crate::bias`]: `a`
/// exposed cases, `b` unexposed cases, `c` exposed non-cases and `d`
/// unexposed non-cases. `true_rd` is the value to move to, 0 for the null,
/// and the lower limit is the Wald limit at `level`.
///
/// # Algorithm
///
/// With `f` the exposed share of the table, `p1` and `p0` the risks of the
/// exposed and unexposed, and `t = true_rd`, the bias factor `B` that moves
/// the difference to `t` solves `(p1 - p0 B) (f + (1 - f) / B) = t`, the
/// positive root of a quadratic, and the E-value is `B + sqrt(B (B - 1))`.
/// For the lower limit, the corrected limit under a bias factor `B` is
///
/// ```text
/// L(B) = (p1 - p0 B) f_B - z sqrt( (v1 + v0 B^2) f_B^2 + (p1 - p0 B)^2 (1 - 1/B)^2 vf )
/// ```
///
/// with `f_B = f + (1 - f) / B`, `v1` and `v0` the binomial variances of the
/// two risks, `vf` that of `f` and `z` the normal quantile at the level's
/// [`Level::upper_probability`], `1 - alpha / 2` for a level given as alpha.
/// The bias factor of the limit is the smallest `B` with `L(B) <= t`.
/// `evalues.RD` finds it on a grid of step `1e-4` in `B`. Here `B` is scanned
/// on a grid of relative step `1e-4` and the first crossing is then bisected
/// to adjacent doubles, so the value is the crossing itself and lies at most
/// one grid step below that of `evalues.RD`.
///
/// # Errors
///
/// [`crate::Error`] for a table with a negative or non-finite cell, an
/// empty exposure group or no unexposed case, [`crate::Error`] for a
/// `true_rd` that is not finite,
/// [`crate::Error`] when the risk difference is negative
/// (exchange the exposure groups), and [`crate::Error`]
/// when the difference is not above `true_rd`.
pub fn evalues_rd(
    t: &crate::tables::Counts2x2,
    true_rd: f64,
    level: Level,
) -> Result<RdEValues, Error> {
    if [
        t.exposed_cases,
        t.unexposed_cases,
        t.exposed_noncases,
        t.unexposed_noncases,
    ]
    .iter()
    .any(|x| !x.is_finite() || *x < 0.0)
    {
        return Err(EValueError::Table("cells must be finite and not negative").into());
    }
    let (n1, n0) = (
        t.exposed_cases + t.exposed_noncases,
        t.unexposed_cases + t.unexposed_noncases,
    );
    if !(n1 > 0.0 && n0 > 0.0) {
        return Err(EValueError::Table("both exposure groups need people").into());
    }
    if t.unexposed_cases <= 0.0 {
        return Err(
            EValueError::Table("the unexposed need a case, or no bias factor is finite").into(),
        );
    }
    if !true_rd.is_finite() {
        return Err(EValueError::NotFinite { what: "true value" }.into());
    }
    let n = n1 + n0;
    let f = n1 / n;
    let p1 = t.exposed_cases / n1;
    let p0 = t.unexposed_cases / n0;
    if p1 < p0 {
        return Err(EValueError::NegativeDifference.into());
    }
    if p1 - p0 <= true_rd {
        return Err(EValueError::TrueNotBelowEstimate.into());
    }
    let v1 = p1 * (1.0 - p1) / n1;
    let v0 = p0 * (1.0 - p0) / n0;
    let vf = f * (1.0 - f) / n;
    let diff = p0 * (1.0 - f) - p1 * f;
    let shifted = true_rd + diff;
    let bias_factor =
        ((shifted * shifted + 4.0 * p1 * p0 * f * (1.0 - f)).sqrt() - shifted) / (2.0 * p0 * f);
    let estimate = evalue_threshold(bias_factor, 1.0)?;
    let z = epistat_stats::dist::normal::two_sided_z(level);
    let se = (v1 + v0).sqrt();
    let lower_limit = p1 - p0 - z * se;
    let risk_difference = Estimate::point(p1 - p0)
        .with_se(Some(se))
        .with_interval(Some(Interval::new(
            Some(lower_limit),
            Some(p1 - p0 + z * se),
            level,
        )));
    if lower_limit <= true_rd {
        return Ok(RdEValues {
            risk_difference,
            bias_factor,
            bias_factor_lower: 1.0,
            estimate,
            lower: 1.0,
        });
    }
    let corrected_limit = |b: f64| {
        let rd = p1 - p0 * b;
        let fb = f + (1.0 - f) / b;
        let shrink = 1.0 - 1.0 / b;
        rd * fb - z * ((v1 + v0 * b * b) * fb * fb + rd * rd * shrink * shrink * vf).sqrt()
    };
    let bias_factor_lower = first_crossing(1.0, bias_factor, |b| corrected_limit(b) <= true_rd);
    Ok(RdEValues {
        risk_difference,
        bias_factor,
        bias_factor_lower,
        estimate,
        lower: evalue_threshold(bias_factor_lower, 1.0)?,
    })
}

/// The first `x` in `(from, to]` at which `crossed` holds, scanning on a grid
/// of relative step `1e-4` and bisecting the first step that crosses to
/// adjacent doubles; `to` when no grid point before it crosses.
fn first_crossing(from: f64, to: f64, crossed: impl Fn(f64) -> bool) -> f64 {
    const STEP: f64 = 1e-4;
    let mut previous = from;
    let mut k = 1.0_f64;
    let mut high = to;
    loop {
        let x = from * (k * STEP).exp();
        if x >= to {
            break;
        }
        if crossed(x) {
            high = x;
            break;
        }
        previous = x;
        k += 1.0;
    }
    let mut low = previous;
    loop {
        let mid = low + (high - low) / 2.0;
        if mid <= low || mid >= high {
            return high;
        }
        if crossed(mid) {
            high = mid;
        } else {
            low = mid;
        }
    }
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
    fn the_bounding_factor_of_the_evalue_is_the_estimate() {
        // B(E, E) = E^2 / (2 E - 1) = RR when E = RR + sqrt(RR (RR - 1)).
        // E has a few roundings, and B doubles their relative error at most:
        // 1e-14 relative.
        for rr in [1.01, 1.5, 2.0, 3.7, 25.0] {
            let e = evalue_threshold(rr, 1.0).unwrap();
            let b = bounding_factor(e, e).unwrap();
            assert!((b - rr).abs() <= 1e-14 * rr, "{rr} {b}");
        }
        assert_eq!(bounding_factor(1.0, 7.0).unwrap(), 1.0);
        assert_eq!(bounding_factor(f64::INFINITY, 2.5).unwrap(), 2.5);
        assert!(bounding_factor(0.9, 2.0).is_err());
        assert!(bounding_factor(2.0, f64::NAN).is_err());
    }

    #[test]
    fn selection_evalues_follow_the_assumptions() {
        let rr: f64 = 2.25;
        let e = |x: f64| x + (x * (x - 1.0)).sqrt();
        let get = |a| selection_evalue_threshold(rr, 1.0, a).unwrap();
        assert_eq!(get(SelectionBias::General), e(1.5));
        assert_eq!(get(SelectionBias::IncreasedRisk), e(rr));
        assert_eq!(get(SelectionBias::SelectionIsTheFactor), 1.5);
        assert_eq!(get(SelectionBias::SelectionIsTheFactorDecreasedRisk), rr);
        // A protective estimate is taken by its reciprocal.
        let p = selection_evalue_threshold(1.0 / rr, 1.0, SelectionBias::General).unwrap();
        assert!((p - e(1.5)).abs() < 1e-15);
    }

    #[test]
    fn the_risk_difference_evalue_of_the_estimate_is_that_of_the_risk_ratio() {
        // At the null, the bias factor that removes the difference is the
        // ratio of the risks, here 1.5, to rounding.
        let t = crate::tables::Counts2x2 {
            exposed_cases: 150.0,
            unexposed_cases: 900.0,
            exposed_noncases: 9_850.0,
            unexposed_noncases: 89_100.0,
        };
        let level = Level::alpha(0.05).unwrap();
        let e = evalues_rd(&t, 0.0, level).unwrap();
        assert!((e.bias_factor - 1.5).abs() < 1e-12);
        assert!(e.bias_factor_lower > 1.0 && e.bias_factor_lower < e.bias_factor);
        assert!(e.lower < e.estimate);
        let swapped = crate::tables::Counts2x2 {
            exposed_cases: 900.0,
            unexposed_cases: 150.0,
            exposed_noncases: 89_100.0,
            unexposed_noncases: 9_850.0,
        };
        assert_eq!(
            evalues_rd(&swapped, 0.0, level),
            Err(EValueError::NegativeDifference.into())
        );
        assert_eq!(
            evalues_rd(&t, 0.01, level),
            Err(EValueError::TrueNotBelowEstimate.into())
        );
        assert!(evalues_rd(&t, f64::NAN, level).is_err());
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
