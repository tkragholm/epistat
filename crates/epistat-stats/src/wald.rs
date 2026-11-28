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

use epistat_core::{
    Coefficients, Covariance, EffectScale, Estimate, Interval, Level, Reference, Scale, Term,
    VarianceKind, check,
};

use crate::Error;
use crate::dist::{normal, t};

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

/// The coefficient table of a fit from its coefficients and their
/// covariance: each term's Wald interval at `level` and its p-value, the
/// statistic `b / se`, referred to the standard normal or, with `df`, to
/// Student's t on `df` degrees of freedom.
///
/// The standard errors are the square roots of the covariance's diagonal,
/// `None` where it is undefined or negative. A coefficient that is `None`,
/// such as an aliased column, gives a term without an estimate. With the
/// normal reference each term is [`wald`]`(b, se, level, Scale::Identity)`;
/// with a t reference the quantile and p-value are those of `t`:
///
/// ```text
/// interval = b -+ q se,   q = t quantile at the level's upper probability on df,
/// p        = 2 P(T_df > |b / se|).
/// ```
///
/// This is the table a fit's `Tidy::coefficients` returns, built in one
/// place so that every fit forms it the same way.
///
/// ```
/// use epistat_core::{Covariance, EffectScale, Level, VarianceKind};
/// use epistat_stats::coefficients;
///
/// let v = Covariance::from_values(2, &[0.04, 0.0, 0.0, 0.01])?;
/// let table = coefficients(
///     vec!["(Intercept)".into(), "treated".into()],
///     &[Some(-1.2), Some(0.4)],
///     v,
///     Level::P95,
///     VarianceKind::ModelBased,
///     EffectScale::LogOddsRatio,
///     None,
/// )?;
/// let or = table.exp().expect("log odds ratios");
/// assert_eq!(or.get("treated").and_then(|t| t.estimate).map(|e| e.se), Some(Some(0.1)));
/// # Ok::<(), epistat_stats::Error>(())
/// ```
///
/// # Errors
///
/// [`Error::Input`] with [`InputError::Length`](epistat_core::InputError::Length)
/// when `names` or the covariance does not have one entry per coefficient,
/// with [`InputError::Domain`](epistat_core::InputError::Domain) for `df`
/// that is not finite and above 0, and the refusals of
/// [`Coefficients::new`].
pub fn coefficients(
    names: Vec<String>,
    estimates: &[Option<f64>],
    covariance: Covariance,
    level: Level,
    variance: VarianceKind,
    effect: EffectScale,
    df: Option<f64>,
) -> Result<Coefficients, Error> {
    check::same_length("names", estimates.len(), names.len())?;
    check::same_length("covariance", estimates.len(), covariance.dim())?;
    if let Some(df) = df {
        check::positive("df", df)?;
    }
    let se = covariance.standard_errors();
    let terms = names
        .into_iter()
        .zip(estimates)
        .zip(se)
        .map(|((name, &b), se)| {
            let Some(b) = b else {
                return Term::new(name, None, None);
            };
            let statistic = se.filter(|&s| s > 0.0).map(|s| b / s);
            let estimate = match df {
                None => wald(b, se, level, Scale::Identity),
                Some(df) => t_wald(b, se, level, df),
            };
            Term::new(name, Some(estimate), statistic)
        })
        .collect();
    let table = Coefficients::new(terms, covariance, level, variance, effect)?;
    Ok(match df {
        None => table,
        Some(df) => table.with_reference(Reference::StudentT, Some(df))?,
    })
}

/// The Wald estimate on the identity scale with a t quantile and p-value.
fn t_wald(b: f64, se: Option<f64>, level: Level, df: f64) -> Estimate {
    let point = Estimate::point(b).with_se(se);
    let Some(se) = se.filter(|s| s.is_finite() && *s >= 0.0) else {
        return point;
    };
    let q = t::quantile(level.upper_probability(), df);
    let p_value = (se > 0.0).then(|| t::two_sided_p(b / se, df));
    point
        .with_interval(Some(Interval::new(
            Some(b - q * se),
            Some(b + q * se),
            level,
        )))
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

    #[test]
    fn a_table_is_the_wald_estimate_of_each_term() {
        let v = Covariance::new(
            3,
            vec![
                Some(0.25),
                None,
                None,
                None,
                Some(0.04),
                None,
                None,
                None,
                Some(-1.0),
            ],
        )
        .unwrap();
        let names = vec!["a".to_owned(), "b".to_owned(), "c".to_owned()];
        let estimates = [Some(1.0), None, Some(2.0)];
        let level = Level::P95;
        let table = coefficients(
            names.clone(),
            &estimates,
            v.clone(),
            level,
            VarianceKind::Robust,
            EffectScale::Identity,
            None,
        )
        .unwrap();
        let a = table.get("a").unwrap();
        assert_eq!(
            a.estimate,
            Some(wald(1.0, Some(0.5), level, Scale::Identity))
        );
        assert_eq!(a.statistic, Some(2.0));
        assert_eq!(table.get("b").unwrap().estimate, None);
        let c = table.get("c").unwrap();
        assert_eq!((c.estimate.unwrap().se, c.statistic), (None, None));
        assert_eq!(table.variance(), VarianceKind::Robust);
        let with_t = coefficients(
            names,
            &estimates,
            v,
            level,
            VarianceKind::ModelBased,
            EffectScale::Identity,
            Some(10.0),
        )
        .unwrap();
        let a = with_t.get("a").unwrap().estimate.unwrap();
        let q = t::quantile(0.975, 10.0);
        assert_eq!(a.lower(), Some(1.0 - q * 0.5));
        assert_eq!(a.p_value, Some(t::two_sided_p(2.0, 10.0)));
        assert_eq!(with_t.reference(), Reference::StudentT);
    }

    #[test]
    fn a_table_of_the_wrong_shape_is_refused() {
        let v = Covariance::from_values(1, &[1.0]).unwrap();
        let build = |names: Vec<String>, df| {
            coefficients(
                names,
                &[Some(1.0)],
                v.clone(),
                Level::P95,
                VarianceKind::ModelBased,
                EffectScale::Identity,
                df,
            )
        };
        assert!(build(Vec::new(), None).is_err());
        assert!(build(vec!["a".into()], Some(0.0)).is_err());
        assert!(build(vec!["a".into()], Some(f64::NAN)).is_err());
    }
}
