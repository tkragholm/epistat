//! The outputs the fixed-effects family shares: the interval and p-value of
//! a coefficient under the reference distribution its R package uses, and
//! the influence functions of the coefficients that were estimated.
//!
//! # Intervals
//!
//! With the estimate `b`, its standard error `s` and `q` the
//! [`Level::upper_probability`] quantile of the reference distribution,
//!
//! ```text
//! interval = b - q s, b + q s,       p = 2 P(T > |b / s|),
//! ```
//!
//! where `T` is standard normal ([`Reference::Normal`]) or Student t with
//! `nu` degrees of freedom ([`Reference::StudentT`]). `fixest` uses the t
//! distribution for linear models, with `nu = G - 1` for the smallest number
//! of clusters `G` of a clustered covariance and `nu = n - K` otherwise
//! (`ssc(t.df = "min")`, its default), and the normal for Poisson and logit
//! models; `marginaleffects` and `didimputation` use the normal.
//!
//! The interval is `None` without a finite, non-negative standard error, and
//! the p-value is `None` unless the standard error is positive, as in
//! [`epistat_stats::wald`], which forms the normal case.

use epistat_core::{Estimate, Interval, Level, Scale};
use epistat_glm::FeolsCovariance;
use epistat_inference::{ClusterRule, Influence};
use epistat_stats::dist::t;

use crate::error::Error;

/// The distribution of `b / s` under the null hypothesis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Reference {
    /// The standard normal.
    Normal,
    /// Student t with these degrees of freedom.
    StudentT(f64),
}

impl Reference {
    /// The reference of `fixest` for a linear model with the covariance
    /// `cov` of `n` rows: t with `G - 1` degrees of freedom when the
    /// covariance is clustered, `n - K` otherwise, and at least one.
    pub(crate) fn fixest_linear(cov: &FeolsCovariance, n: usize) -> Self {
        let nu = cov
            .min_clusters
            .map_or_else(|| n.saturating_sub(cov.df_k), |g| g.saturating_sub(1));
        Self::StudentT(nu.max(1) as f64)
    }
}

/// The estimate `value` with the standard error `se`, its interval at
/// `level` and its p-value under `reference`.
pub(crate) fn estimate(
    value: f64,
    se: Option<f64>,
    level: Level,
    reference: Reference,
) -> Estimate {
    match reference {
        Reference::Normal => epistat_stats::wald(value, se, level, Scale::Identity),
        Reference::StudentT(nu) => {
            let point = Estimate::point(value).with_se(se);
            let Some(se) = se.filter(|s| s.is_finite() && *s >= 0.0) else {
                return point;
            };
            if !value.is_finite() {
                return point;
            }
            let q = t::quantile(level.upper_probability(), nu);
            let interval = Interval::new(Some(value - q * se), Some(value + q * se), level);
            let p_value = (se > 0.0).then(|| 2.0 * t::sf((value / se).abs(), nu));
            point.with_interval(Some(interval)).with_p_value(p_value)
        }
    }
}

/// The influence function of the defined columns of `columns`, each one
/// value per unit of `units`, under `rule`; `None` when no column is defined.
/// The values are kept column after column in the order given.
pub(crate) fn influence(
    columns: Vec<Option<Vec<f64>>>,
    units: Vec<u32>,
    rule: ClusterRule,
) -> Result<Option<Influence>, Error> {
    let defined: Vec<Vec<f64>> = columns.into_iter().flatten().collect();
    if defined.is_empty() {
        return Ok(None);
    }
    let k = defined.len();
    let values: Vec<f64> = defined.into_iter().flatten().collect();
    Ok(Some(Influence::new(values, units, k, rule)?))
}

/// The column of `influence` that belongs to entry `j` of a list whose
/// entries `defined` marks: the entries with a column come in order, and an
/// entry without one has none.
pub(crate) fn column_of(
    influence: Option<&Influence>,
    mut defined: impl Iterator<Item = bool>,
    j: usize,
) -> Option<&[f64]> {
    let influence = influence?;
    let mut position = 0;
    for _ in 0..j {
        position += usize::from(defined.next()?);
    }
    defined.next()?.then(|| influence.column(position))
}

/// The cluster rule of a covariance with `fixest`'s factor `G / (G - 1)` when
/// `g_adj` is set.
pub(crate) const fn fixest_rule(g_adj: bool) -> ClusterRule {
    if g_adj {
        ClusterRule::SmallSample
    } else {
        ClusterRule::NoSmallSample
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn t_intervals_widen_the_normal_ones() {
        let level = Level::default();
        let normal = estimate(1.0, Some(0.5), level, Reference::Normal);
        let t5 = estimate(1.0, Some(0.5), level, Reference::StudentT(5.0));
        let q = t::quantile(0.975, 5.0);
        assert_eq!(t5.lower(), Some(1.0 - q * 0.5));
        assert_eq!(t5.upper(), Some(1.0 + q * 0.5));
        assert_eq!(t5.p_value, Some(2.0 * t::sf(2.0, 5.0)));
        assert!(t5.lower() < normal.lower());
        assert!(t5.p_value > normal.p_value);
        assert_eq!(t5.interval.map(|i| i.level), Some(level));
    }

    #[test]
    fn undefined_parts_are_none() {
        let level = Level::default();
        let none = estimate(1.0, None, level, Reference::StudentT(3.0));
        assert_eq!((none.se, none.interval, none.p_value), (None, None, None));
        let zero = estimate(1.0, Some(0.0), level, Reference::StudentT(3.0));
        assert_eq!(zero.lower(), Some(1.0));
        assert_eq!(zero.p_value, None);
    }
}
