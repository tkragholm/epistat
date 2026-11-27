//! Net survival standardised by age: a weighted mean of the curves of age
//! groups.

use super::{NetSurvival, Transform};
use crate::error::{Error, Result};
use epistat_core::Level;
use epistat_stats::dist::normal;

/// The weights of the International Cancer Survival Standards 1, 2 and 3 for
/// the age groups 15 to 44, 45 to 54, 55 to 64, 65 to 74 and 75 and over at
/// diagnosis (Corazziari et al. 2004), in that order. Standard 1 is for
/// cancers whose incidence rises with age, 2 for those with little change by
/// age, and 3 for those of young adults. They are the shares of the five
/// groups in `popEpi::ICSS`.
pub const ICSS: [[f64; 5]; 3] = [
    [0.07, 0.12, 0.23, 0.29, 0.29],
    [0.28, 0.17, 0.21, 0.20, 0.14],
    [0.60, 0.10, 0.10, 0.10, 0.10],
];

/// A net survival curve standardised by age.
#[derive(Clone, Debug, PartialEq)]
pub struct StandardisedNet {
    /// The times asked for.
    pub times: Vec<f64>,
    /// The standardised net survival, `None` beyond the last reported time of
    /// a group with a positive weight.
    pub survival: Vec<Option<f64>>,
    /// Its standard error.
    pub se: Vec<Option<f64>>,
    /// The lower limit, `None` where the transform is undefined.
    pub lower: Vec<Option<f64>>,
    /// The upper limit.
    pub upper: Vec<Option<f64>>,
}

/// The net survival of age groups standardised to `weights`, at `times`.
///
/// `curves` holds one curve per age group, such as the strata of
/// [`super::net_survival`] by age group, and `weights` one weight per curve,
/// such as a row of [`ICSS`], scaled to sum to 1. The curve of group `k` at
/// `t` is its estimate at its last reported time at or before `t`, 1 with no
/// variance before its first, and undefined after its last, since the group
/// has no follow-up there. With `R_k` and `V_k` the net survival and the
/// variance of the cumulative excess hazard of group `k`,
///
/// ```text
/// R(t)      = sum_k w_k R_k(t)
/// Var R(t)  = sum_k w_k^2 R_k(t)^2 V_k(t)
/// ```
///
/// by the delta method for `R_k = exp(-Lambda_k)` and independent groups, as
/// in the traditional approach of Corazziari et al. and Lambert et al. The
/// limits are normal on the scale of `transform`: `R -/+ z se` for
/// [`Transform::Plain`], `R exp(-/+ z se / R)` for [`Transform::Log`], and for
/// [`Transform::LogLog`] normal limits of `ln Lambda` with `Lambda = -ln R`
/// and the standard error `se / (R Lambda)`, undefined when `Lambda` is not
/// positive, as for one curve. A group of weight 0 is left out.
///
/// # Errors
///
/// An input error for no curve, a number of weights other than the number of
/// curves, a weight that is negative or not finite, weights that sum to 0,
/// times that are not finite.
pub fn standardise_net(
    curves: &[NetSurvival],
    weights: &[f64],
    times: &[f64],
    level: Level,
    transform: Transform,
) -> Result<StandardisedNet> {
    if curves.is_empty() {
        return Err(Error::invalid("no curve to standardise"));
    }
    if weights.len() != curves.len() {
        return Err(Error::invalid(format!(
            "{} weights for {} curves",
            weights.len(),
            curves.len()
        )));
    }
    if weights.iter().any(|w| !(w.is_finite() && *w >= 0.0)) {
        return Err(Error::invalid(
            "the weights must be finite and not negative",
        ));
    }
    let total = epistat_stats::sum::exact(weights.iter().copied());
    if total <= 0.0 {
        return Err(Error::invalid("the weights sum to 0"));
    }
    if times.iter().any(|t| !t.is_finite()) {
        return Err(Error::invalid("the times must be finite"));
    }
    let z = normal::quantile(level.upper_probability());
    let mut out = StandardisedNet {
        times: times.to_vec(),
        survival: Vec::with_capacity(times.len()),
        se: Vec::with_capacity(times.len()),
        lower: Vec::with_capacity(times.len()),
        upper: Vec::with_capacity(times.len()),
    };
    for &t in times {
        let mut estimate = Some((0.0_f64, 0.0_f64));
        for (curve, &w) in curves.iter().zip(weights) {
            if w == 0.0 {
                continue;
            }
            let share = w / total;
            let at = match curve.times.iter().rposition(|&x| x <= t) {
                None => Some((1.0, 0.0)),
                Some(k) if k + 1 == curve.times.len() && t > curve.times[k] => None,
                Some(k) => Some((curve.survival[k], curve.variance[k])),
            };
            estimate = estimate
                .zip(at)
                .map(|((r, v), (rk, vk))| (r + share * rk, v + share * share * rk * rk * vk));
        }
        let Some((r, v)) = estimate else {
            out.survival.push(None);
            out.se.push(None);
            out.lower.push(None);
            out.upper.push(None);
            continue;
        };
        let se = v.sqrt();
        let limits = match transform {
            Transform::Plain => Some((r - z * se, r + z * se)),
            Transform::Log => Some((r * (-z * se / r).exp(), r * (z * se / r).exp())),
            Transform::LogLog => {
                let lambda = -r.ln();
                (lambda > 0.0).then(|| {
                    let shift = z * se / (r * lambda);
                    (
                        (-(lambda.ln() + shift).exp()).exp(),
                        (-(lambda.ln() - shift).exp()).exp(),
                    )
                })
            }
        };
        out.survival.push(Some(r));
        out.se.push(Some(se));
        out.lower.push(limits.map(|l| l.0));
        out.upper.push(limits.map(|l| l.1));
    }
    Ok(out)
}
