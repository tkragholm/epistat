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
//! The structs are `#[non_exhaustive]`: a crate builds them through the
//! constructors here, and a field added later breaks no caller.

use crate::level::Level;

/// A confidence interval at a level.
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
}

impl Interval {
    /// The interval `[lower, upper]` at `level`.
    #[must_use]
    pub const fn new(lower: Option<f64>, upper: Option<f64>, level: Level) -> Self {
        Self {
            lower,
            upper,
            level,
        }
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
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Estimate {
    /// The point estimate.
    pub estimate: f64,
    /// The standard error, on [`scale`](Self::scale).
    pub se: Option<f64>,
    /// The confidence interval, on the scale of the estimate.
    pub interval: Option<Interval>,
    /// The two-sided p-value of the null hypothesis the method tests.
    pub p_value: Option<f64>,
    /// The scale of the standard error and of the interval's construction.
    pub scale: Scale,
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
