//! Point estimates with their standard error, interval and p-value.
//!
//! Every estimator reports a number the same way: an [`Estimate`] holds the
//! estimate, its standard error, a confidence [`Interval`] at a [`Level`] and
//! a p-value, each `None` where the method does not define it. A [`Scale`]
//! says whether the standard error and the interval were formed on the log
//! scale, as for a ratio whose interval is `exp(log r +- z se)`. A
//! [`Contrast`] holds two means and their difference and ratio, the shape of
//! a causal contrast or a matched comparison.
//!
//! The types hold numbers; they compute nothing beyond reading them.
//! `epistat_stats::wald` forms the Wald estimate `estimate +- z se` with the
//! p-value `2 Phi(-|estimate / se|)` from a standard error and a level.
//!
//! An interval says whether its level holds for it alone or for a whole
//! family of intervals at once: [`Coverage::Pointwise`] or
//! [`Coverage::Simultaneous`], as for a uniform band over event times. A
//! p-value is a pointwise statement, so an estimate whose interval is
//! simultaneous carries none.
//!
//! The structs are `#[non_exhaustive]`: a crate builds them through the
//! constructors here, and a field added later breaks no caller.
//!
//! ```
//! use epistat_core::{Estimate, Interval, Level, Scale};
//!
//! let hazard_ratio = Estimate::point(0.510)
//!     .with_se(Some(0.0893))
//!     .with_interval(Some(Interval::new(Some(0.428), Some(0.608), Level::P95)))
//!     .with_p_value(Some(4.9e-14))
//!     .with_scale(Scale::Log);
//! assert_eq!(hazard_ratio.to_string(), "0.510 (95% CI 0.428 to 0.608), p = 4.9e-14");
//! assert_eq!(hazard_ratio.null(), 1.0);
//! assert_eq!(hazard_ratio.se_log(), Some(0.0893));
//! ```

use core::fmt;

use crate::check::{self, InputError};
use crate::format;
use crate::level::Level;

/// Whether an interval's level holds for the interval alone or jointly for a
/// family of intervals.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(try_from = "CoverageForm", into = "CoverageForm")
)]
pub enum Coverage {
    /// The level holds for this interval alone, as for a Wald interval.
    #[default]
    Pointwise,
    /// The level holds for all intervals of a family at once, as for a
    /// uniform band, whose limits are `estimate +- critical se`. Built with
    /// [`Coverage::simultaneous`], which checks the critical value.
    #[non_exhaustive]
    Simultaneous {
        /// The critical value of the family, finite and above 0, such as a
        /// sup-t quantile.
        critical: f64,
    },
}

/// The serialised form: `"pointwise"` or `{"simultaneous": {"critical": c}}`.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
enum CoverageForm {
    Pointwise,
    Simultaneous { critical: f64 },
}

impl TryFrom<CoverageForm> for Coverage {
    type Error = InputError;

    fn try_from(form: CoverageForm) -> Result<Self, InputError> {
        match form {
            CoverageForm::Pointwise => Ok(Self::Pointwise),
            CoverageForm::Simultaneous { critical } => Self::simultaneous(critical),
        }
    }
}

impl From<Coverage> for CoverageForm {
    fn from(coverage: Coverage) -> Self {
        match coverage {
            Coverage::Pointwise => Self::Pointwise,
            Coverage::Simultaneous { critical } => Self::Simultaneous { critical },
        }
    }
}

impl Coverage {
    /// Simultaneous coverage with the family's critical value `critical`.
    ///
    /// # Errors
    ///
    /// [`InputError::Domain`] unless `critical` is finite and above 0.
    pub fn simultaneous(critical: f64) -> Result<Self, InputError> {
        check::positive("critical value", critical)?;
        Ok(Self::Simultaneous { critical })
    }

    /// Whether the level holds jointly for a family of intervals.
    #[must_use]
    pub const fn is_simultaneous(self) -> bool {
        matches!(self, Self::Simultaneous { .. })
    }

    /// The critical value of a simultaneous family, `None` for pointwise
    /// coverage.
    #[must_use]
    pub const fn critical(self) -> Option<f64> {
        match self {
            Self::Pointwise => None,
            Self::Simultaneous { critical } => Some(critical),
        }
    }
}

impl fmt::Display for Coverage {
    /// `pointwise`, or `simultaneous` with the critical value.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pointwise => f.write_str("pointwise"),
            Self::Simultaneous { critical } => write!(
                f,
                "simultaneous, critical value {}",
                format::Significant(*critical, 4)
            ),
        }
    }
}

/// A confidence interval at a level, pointwise or simultaneous.
///
/// A bound is `None` where the method does not define it, as for an interval
/// whose standard error is undefined or a one-sided bound that does not
/// exist.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Interval {
    /// The lower limit.
    pub lower: Option<f64>,
    /// The upper limit.
    pub upper: Option<f64>,
    /// The confidence level.
    pub level: Level,
    /// Whether the level holds for this interval alone or for its family;
    /// pointwise when a serialised interval does not say.
    #[cfg_attr(feature = "serde", serde(default))]
    pub coverage: Coverage,
}

impl Interval {
    /// The pointwise interval `[lower, upper]` at `level`.
    #[must_use]
    pub const fn new(lower: Option<f64>, upper: Option<f64>, level: Level) -> Self {
        Self {
            lower,
            upper,
            level,
            coverage: Coverage::Pointwise,
        }
    }

    /// The interval `[lower, upper]` of a family whose intervals cover
    /// jointly at `level`, with the family's critical value `critical`.
    ///
    /// ```
    /// use epistat_core::{Interval, Level};
    ///
    /// let band = Interval::simultaneous(Some(0.1), Some(0.9), Level::P95, 2.73)?;
    /// assert!(band.coverage.is_simultaneous());
    /// assert_eq!(band.to_string(), "95% simultaneous CI 0.100 to 0.900");
    /// # Ok::<(), epistat_core::InputError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`InputError::Domain`] unless `critical` is finite and above 0.
    pub fn simultaneous(
        lower: Option<f64>,
        upper: Option<f64>,
        level: Level,
        critical: f64,
    ) -> Result<Self, InputError> {
        Ok(Self::new(lower, upper, level).with_coverage(Coverage::simultaneous(critical)?))
    }

    /// The same interval with the coverage `coverage`.
    #[must_use]
    pub const fn with_coverage(mut self, coverage: Coverage) -> Self {
        self.coverage = coverage;
        self
    }

    /// `upper - lower`, when both are defined.
    #[must_use]
    pub fn width(&self) -> Option<f64> {
        Some(self.upper? - self.lower?)
    }

    /// Whether `value` lies in the closed interval, when both limits are
    /// defined.
    #[must_use]
    pub fn contains(&self, value: f64) -> Option<bool> {
        Some(self.lower? <= value && value <= self.upper?)
    }
}

impl fmt::Display for Interval {
    /// `95% CI 0.428 to 0.608`, or `95% simultaneous CI ...` for a band; an
    /// undefined limit prints as `NA`. The limits have three significant
    /// digits, or the decimals of a precision such as `{:.2}`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let decimals = f.precision();
        let kind = if self.coverage.is_simultaneous() {
            " simultaneous"
        } else {
            ""
        };
        write!(
            f,
            "{}{kind} CI {} to {}",
            self.level,
            format::optional(self.lower, 3, decimals),
            format::optional(self.upper, 3, decimals)
        )
    }
}

/// The scale on which an estimate's standard error and interval were formed.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum Scale {
    /// The scale of the estimate itself.
    #[default]
    Identity,
    /// The log of the estimate: the standard error is that of `log estimate`
    /// and the interval is `exp` of an interval on that scale, as for ratios.
    Log,
}

/// A point estimate with what the method defines about its uncertainty.
///
/// The p-value is pointwise. An estimate whose interval is
/// [simultaneous](Coverage::Simultaneous) has none, and deserialising one
/// that has both is refused.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "EstimateParts"))]
pub struct Estimate {
    /// The point estimate.
    pub estimate: f64,
    /// The standard error, on [`scale`](Self::scale).
    pub se: Option<f64>,
    /// The confidence interval, on the scale of the estimate.
    pub interval: Option<Interval>,
    /// The two-sided p-value of the null hypothesis the method tests, `None`
    /// when the interval is simultaneous.
    pub p_value: Option<f64>,
    /// The scale of the standard error and of the interval's construction.
    pub scale: Scale,
}

#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize))]
struct EstimateParts {
    estimate: f64,
    se: Option<f64>,
    interval: Option<Interval>,
    p_value: Option<f64>,
    scale: Scale,
}

impl TryFrom<EstimateParts> for Estimate {
    type Error = InputError;

    fn try_from(parts: EstimateParts) -> Result<Self, InputError> {
        if let (Some(p), Some(interval)) = (parts.p_value, parts.interval)
            && interval.coverage.is_simultaneous()
        {
            return Err(InputError::Domain {
                what: "p_value",
                value: p,
                domain: "no value beside a simultaneous interval",
                row: None,
            });
        }
        Ok(Self::new(
            parts.estimate,
            parts.se,
            parts.interval,
            parts.p_value,
            parts.scale,
        ))
    }
}

impl Estimate {
    /// A point estimate alone: no standard error, interval or p-value, on the
    /// identity scale.
    #[must_use]
    pub const fn point(estimate: f64) -> Self {
        Self {
            estimate,
            se: None,
            interval: None,
            p_value: None,
            scale: Scale::Identity,
        }
    }

    /// An estimate with every part given.
    #[must_use]
    pub const fn new(
        estimate: f64,
        se: Option<f64>,
        interval: Option<Interval>,
        p_value: Option<f64>,
        scale: Scale,
    ) -> Self {
        Self {
            estimate,
            se,
            interval,
            p_value,
            scale,
        }
    }

    /// The same estimate with the standard error `se`.
    #[must_use]
    pub const fn with_se(mut self, se: Option<f64>) -> Self {
        self.se = se;
        self
    }

    /// The same estimate with the interval `interval`.
    #[must_use]
    pub const fn with_interval(mut self, interval: Option<Interval>) -> Self {
        self.interval = interval;
        self
    }

    /// The same estimate with the p-value `p_value`.
    #[must_use]
    pub const fn with_p_value(mut self, p_value: Option<f64>) -> Self {
        self.p_value = p_value;
        self
    }

    /// The same estimate on the scale `scale`.
    #[must_use]
    pub const fn with_scale(mut self, scale: Scale) -> Self {
        self.scale = scale;
        self
    }

    /// The interval's lower limit, when there is one.
    #[must_use]
    pub fn lower(&self) -> Option<f64> {
        self.interval.and_then(|interval| interval.lower)
    }

    /// The interval's upper limit, when there is one.
    #[must_use]
    pub fn upper(&self) -> Option<f64> {
        self.interval.and_then(|interval| interval.upper)
    }

    /// The value of the null hypothesis the p-value tests: 0 on
    /// [`Scale::Identity`], 1 on [`Scale::Log`], where the estimate is a
    /// ratio.
    #[must_use]
    pub const fn null(&self) -> f64 {
        match self.scale {
            Scale::Identity => 0.0,
            Scale::Log => 1.0,
        }
    }

    /// The standard error of `log estimate`, `Some` only on [`Scale::Log`],
    /// where [`se`](Self::se) is that standard error. On the identity scale
    /// it is `None`, so the log-scale reading of an identity-scale error
    /// cannot happen by accident.
    #[must_use]
    pub const fn se_log(&self) -> Option<f64> {
        match self.scale {
            Scale::Identity => None,
            Scale::Log => self.se,
        }
    }
}

impl fmt::Display for Estimate {
    /// `0.510 (95% CI 0.428 to 0.608), p = 4.9e-14`. Without an interval the
    /// standard error is shown, `(se 0.12)`, or `(se of log 0.12)` on the log
    /// scale. Numbers have three significant digits, or the decimals of a
    /// precision such as `{:.2}`; the p-value has two significant digits.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let decimals = f.precision();
        write!(f, "{}", format::number(self.estimate, 3, decimals))?;
        match (self.interval, self.se) {
            (Some(interval), _) => match decimals {
                Some(d) => write!(f, " ({interval:.d$})")?,
                None => write!(f, " ({interval})")?,
            },
            (None, Some(se)) => {
                let of = match self.scale {
                    Scale::Identity => "se",
                    Scale::Log => "se of log",
                };
                write!(f, " ({of} {})", format::number(se, 3, decimals))?;
            }
            (None, None) => {}
        }
        if let Some(p) = self.p_value {
            write!(f, ", p {}", format::p_clause(p))?;
        }
        Ok(())
    }
}

/// Two means and their contrast: `mean1` in the exposed or treated arm,
/// `mean0` in the other, the difference `mean1 - mean0` and, where defined,
/// the ratio `mean1 / mean0`.
///
/// The ratio is usually reported with [`Scale::Log`]. It is `None` where the
/// method does not form it, as for a difference of means that can be
/// negative.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Contrast {
    /// The mean in arm 1.
    pub mean1: Estimate,
    /// The mean in arm 0.
    pub mean0: Estimate,
    /// `mean1 - mean0`.
    pub difference: Estimate,
    /// `mean1 / mean0`.
    pub ratio: Option<Estimate>,
}

impl Contrast {
    /// The contrast of `mean1` and `mean0` with its difference and ratio.
    #[must_use]
    pub const fn new(
        mean1: Estimate,
        mean0: Estimate,
        difference: Estimate,
        ratio: Option<Estimate>,
    ) -> Self {
        Self {
            mean1,
            mean0,
            difference,
            ratio,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_interval_reads_its_limits() {
        let level = Level::default();
        let interval = Interval::new(Some(1.0), Some(3.0), level);
        assert_eq!(interval.width(), Some(2.0));
        assert_eq!(interval.contains(2.0), Some(true));
        assert_eq!(interval.contains(3.5), Some(false));
        let open = Interval::new(None, Some(3.0), level);
        assert_eq!(open.width(), None);
        assert_eq!(open.contains(2.0), None);
    }

    #[test]
    fn an_estimate_is_built_part_by_part() {
        let level = Level::alpha(0.05).unwrap();
        let estimate = Estimate::point(2.0)
            .with_se(Some(0.5))
            .with_interval(Some(Interval::new(Some(1.0), Some(3.0), level)))
            .with_p_value(Some(0.01))
            .with_scale(Scale::Log);
        assert_eq!(
            estimate,
            Estimate::new(
                2.0,
                Some(0.5),
                Some(Interval::new(Some(1.0), Some(3.0), level)),
                Some(0.01),
                Scale::Log
            )
        );
        assert_eq!(estimate.lower(), Some(1.0));
        assert_eq!(estimate.upper(), Some(3.0));
        let bare = Estimate::point(1.0);
        assert_eq!((bare.se, bare.lower(), bare.p_value), (None, None, None));
        assert_eq!(bare.scale, Scale::Identity);
    }

    #[test]
    fn a_contrast_holds_both_arms() {
        let contrast = Contrast::new(
            Estimate::point(3.0),
            Estimate::point(2.0),
            Estimate::point(1.0),
            Some(Estimate::point(1.5).with_scale(Scale::Log)),
        );
        assert_eq!(contrast.difference.estimate, 1.0);
        assert_eq!(contrast.ratio.map(|r| r.scale), Some(Scale::Log));
    }

    #[test]
    fn an_interval_says_its_coverage() {
        let pointwise = Interval::new(Some(1.0), Some(2.0), Level::P95);
        assert_eq!(pointwise.coverage, Coverage::Pointwise);
        assert_eq!(pointwise.coverage.critical(), None);
        let band = Interval::simultaneous(Some(1.0), Some(2.0), Level::P95, 2.5).unwrap();
        assert!(band.coverage.is_simultaneous());
        assert_eq!(band.coverage.critical(), Some(2.5));
        assert_ne!(band, pointwise);
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(matches!(
                Interval::simultaneous(Some(1.0), Some(2.0), Level::P95, bad),
                Err(InputError::Domain {
                    what: "critical value",
                    ..
                })
            ));
        }
        assert_eq!(
            band.coverage.to_string(),
            "simultaneous, critical value 2.500"
        );
    }

    #[test]
    fn the_null_and_the_log_scale_error_follow_the_scale() {
        let difference = Estimate::point(0.3).with_se(Some(0.1));
        assert_eq!(difference.null(), 0.0);
        assert_eq!(difference.se_log(), None);
        let ratio = difference.with_scale(Scale::Log);
        assert_eq!(ratio.null(), 1.0);
        assert_eq!(ratio.se_log(), Some(0.1));
    }

    #[test]
    fn estimates_and_intervals_print_as_a_sentence() {
        let level = Level::P95;
        let interval = Interval::new(Some(0.428_31), Some(0.608), level);
        assert_eq!(interval.to_string(), "95% CI 0.428 to 0.608");
        assert_eq!(format!("{interval:.1}"), "95% CI 0.4 to 0.6");
        let open = Interval::new(None, Some(3.0), Level::P90);
        assert_eq!(open.to_string(), "90% CI NA to 3.00");
        let estimate = Estimate::point(0.510)
            .with_interval(Some(interval))
            .with_p_value(Some(0.2));
        assert_eq!(estimate.to_string(), "0.510 (95% CI 0.428 to 0.608), p = 0.20");
        assert_eq!(
            format!("{estimate:.2}"),
            "0.51 (95% CI 0.43 to 0.61), p = 0.20"
        );
        let tiny = Estimate::point(2.0).with_se(Some(0.25)).with_p_value(Some(0.0));
        assert_eq!(tiny.to_string(), "2.00 (se 0.250), p < 2e-16");
        let ratio = Estimate::point(1.5).with_se(Some(0.2)).with_scale(Scale::Log);
        assert_eq!(ratio.to_string(), "1.50 (se of log 0.200)");
        assert_eq!(Estimate::point(-3.0).to_string(), "-3.00");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn coverage_round_trips_and_defaults_to_pointwise() {
        let band = Interval::simultaneous(Some(1.0), Some(2.0), Level::P95, 2.5).unwrap();
        let text = serde_json::to_string(&band).unwrap();
        assert!(text.contains(r#""coverage":{"simultaneous":{"critical":2.5}}"#), "{text}");
        assert_eq!(serde_json::from_str::<Interval>(&text).unwrap(), band);
        let old = r#"{"lower":1.0,"upper":2.0,"level":{"confidence":0.95}}"#;
        assert_eq!(
            serde_json::from_str::<Interval>(old).unwrap().coverage,
            Coverage::Pointwise
        );
        let bad = r#"{"lower":1.0,"upper":2.0,"level":{"confidence":0.95},"coverage":{"simultaneous":{"critical":-1.0}}}"#;
        assert!(serde_json::from_str::<Interval>(bad).is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn a_p_value_beside_a_simultaneous_interval_is_refused() {
        let band = Interval::simultaneous(Some(1.0), Some(2.0), Level::P95, 2.5).unwrap();
        let estimate = Estimate::point(1.5).with_interval(Some(band));
        let text = serde_json::to_string(&estimate).unwrap();
        assert_eq!(serde_json::from_str::<Estimate>(&text).unwrap(), estimate);
        let with_p = serde_json::to_string(&estimate.with_p_value(Some(0.01))).unwrap();
        assert!(serde_json::from_str::<Estimate>(&with_p).is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn estimates_round_trip_through_serde() {
        let estimate = Estimate::point(0.1 + 0.2)
            .with_se(Some(1.0 / 3.0))
            .with_interval(Some(Interval::new(None, Some(2.0), Level::default())));
        let text = serde_json::to_string(&estimate).unwrap();
        let back: Estimate = serde_json::from_str(&text).unwrap();
        assert_eq!(back, estimate);
        assert_eq!(back.estimate.to_bits(), (0.1_f64 + 0.2).to_bits());
    }
}
