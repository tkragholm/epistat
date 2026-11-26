//! The Wald estimate: an estimate with its normal interval and p-value.
//!
//! # Formulas
//!
//! With the estimate `b`, its standard error `s` and `z` the
//! [`Level::upper_probability`] quantile of the standard normal,
//!
//! ```text
//! Identity:  interval = b - z s, b + z s,            p = 2 Phi(-|b / s|),
//! Log:       interval = exp(ln b - z s), exp(ln b + z s),   p = 2 Phi(-|ln b / s|),
//! ```
//!
//! where on [`Scale::Log`] `s` is the standard error of `ln b`, as for a
//! ratio. The p-value tests `b = 0` on the identity scale and `b = 1` on the
//! log scale. `Phi(-|t|)` is the lower tail, computed directly, so a small
//! p-value keeps its relative precision.
//!
//! The interval is `None` without a finite, non-negative standard error, and
//! the p-value is `None` unless the standard error is positive. On the log
//! scale both are `None` unless `b` is positive.

use epistat_core::{Estimate, Interval, Level, Scale};

use crate::dist::normal;

/// The Wald estimate of `estimate` with the standard error `se` at `level`,
/// on `scale`.
#[must_use]
pub fn wald(estimate: f64, se: Option<f64>, level: Level, scale: Scale) -> Estimate {
    let point = Estimate::point(estimate).with_se(se).with_scale(scale);
    let Some(se) = se.filter(|s| s.is_finite() && *s >= 0.0) else {
        return point;
    };
    let centre = match scale {
        Scale::Log if estimate > 0.0 && estimate.is_finite() => estimate.ln(),
        Scale::Log => return point,
        _ => estimate,
    };
    if !centre.is_finite() {
        return point;
    }
    let z = normal::two_sided_z(level);
    let (lower, upper) = (centre - z * se, centre + z * se);
    let (lower, upper) = match scale {
        Scale::Log => (lower.exp(), upper.exp()),
        _ => (lower, upper),
    };
    let p_value = (se > 0.0).then(|| 2.0 * normal::cdf(-(centre / se).abs()));
    point
        .with_interval(Some(Interval::new(Some(lower), Some(upper), level)))
        .with_p_value(p_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_and_log_scales() {
        let level = Level::default();
        let z = normal::quantile(0.975);
        let e = wald(2.0, Some(0.5), level, Scale::Identity);
        assert_eq!(e.lower(), Some(2.0 - z * 0.5));
        assert_eq!(e.upper(), Some(2.0 + z * 0.5));
        assert_eq!(e.p_value, Some(2.0 * normal::cdf(-4.0)));
        let r = wald(2.0, Some(0.5), level, Scale::Log);
        assert_eq!(r.lower(), Some((2.0_f64.ln() - z * 0.5).exp()));
        assert_eq!(r.p_value, Some(2.0 * normal::cdf(-(2.0_f64.ln() / 0.5))));
        assert_eq!(r.scale, Scale::Log);
        assert_eq!(r.interval.map(|i| i.level), Some(level));
    }

    #[test]
    fn undefined_parts_are_none() {
        let level = Level::default();
        let none = wald(1.0, None, level, Scale::Identity);
        assert_eq!((none.interval, none.p_value), (None, None));
        let zero = wald(1.0, Some(0.0), level, Scale::Identity);
        assert_eq!(zero.lower(), Some(1.0));
        assert_eq!(zero.p_value, None);
        assert_eq!(wald(-1.0, Some(0.2), level, Scale::Log).interval, None);
        assert_eq!(
            wald(1.0, Some(f64::NAN), level, Scale::Identity).interval,
            None
        );
        assert_eq!(
            wald(f64::NAN, Some(1.0), level, Scale::Identity).interval,
            None
        );
    }
}
