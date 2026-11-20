//! Confidence limits of a survival probability from its standard error.

use super::{ConfLower, ConfType};

/// Which scale a standard error is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Scale {
    /// The standard error of `log S`, which is what Greenwood's variance and
    /// the variance of the cumulative hazard give.
    Log,
    /// The standard error of `S`, which the infinitesimal jackknife gives for
    /// the product-limit estimate.
    Surv,
}

/// The maximum, undefined when either side is, so that an undefined bound
/// stays undefined.
fn max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// The minimum, undefined when either side is.
fn min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// The two limits for one time. NaN stands for an undefined limit and becomes
/// `None` at the edge of the module.
///
/// `p` is the survival probability, `se` its standard error on `scale` and
/// `z` the normal quantile of the level. `lower_se`, when given, replaces the
/// standard error of the lower limit, on the same scale as `se`, as the Peto
/// and modified lower limits do.
#[expect(clippy::float_cmp, reason = "R's rules at exact zero and one")]
pub(super) fn limits(
    kind: ConfType,
    z: f64,
    p: f64,
    se: f64,
    scale: Scale,
    lower_se: Option<f64>,
) -> (f64, f64) {
    // The factor that takes the standard error to that of the lower limit.
    let stretch = match lower_se {
        None => 1.0,
        Some(low) => {
            if low == 0.0 {
                1.0
            } else {
                low / se
            }
        }
    };
    let se = match scale {
        Scale::Log => se,
        Scale::Surv if se == 0.0 => 0.0,
        Scale::Surv => se / p,
    };
    // Zero probability has no logarithm, and one has no log-log.
    let positive = if p == 0.0 { f64::NAN } else { p };
    let interior = if p == 0.0 || p == 1.0 { f64::NAN } else { p };
    match kind {
        ConfType::Plain => {
            let half = se * p * z;
            (max(p - half * stretch, 0.0), min(p + half, 1.0))
        }
        ConfType::Log => {
            let half = z * se;
            let lower = if se == 0.0 {
                p
            } else {
                (positive.ln() - half * stretch).exp()
            };
            let upper = if se == 0.0 {
                p
            } else {
                (positive.ln() + half).exp()
            };
            (lower, min(upper, 1.0))
        }
        ConfType::LogLog => {
            let half = z * se / interior.ln();
            let base = (-interior.ln()).ln();
            let lower = if se == 0.0 {
                p
            } else {
                (-(base - half * stretch).exp()).exp()
            };
            let upper = if se == 0.0 {
                p
            } else {
                (-(base + half).exp()).exp()
            };
            (lower, upper)
        }
        ConfType::Logit => {
            let half = z * se * (1.0 + positive / (1.0 - positive));
            let centre = (p / (1.0 - p)).ln();
            let lower = if se == 0.0 {
                p
            } else {
                1.0 - 1.0 / (1.0 + (centre - half * stretch).exp())
            };
            let upper = if se == 0.0 {
                p
            } else {
                1.0 - 1.0 / (1.0 + (centre + half).exp())
            };
            (lower, upper)
        }
        ConfType::Arcsin => {
            let half = 0.5 * z * se * (positive / (1.0 - positive)).sqrt();
            let centre = positive.sqrt().asin();
            let lower = max(0.0, centre - half * stretch).sin();
            let upper = min(std::f64::consts::FRAC_PI_2, centre + half).sin();
            (lower * lower, upper * upper)
        }
        ConfType::None => (f64::NAN, f64::NAN),
    }
}

/// The standard error of the lower limit for the Peto and modified variants,
/// on the scale of the standard error, or `None` for the usual limit.
///
/// `n_risk` is the weighted number at risk, `lag` the weighted number at risk
/// at the last time with an event.
pub(super) fn lower_se(variant: ConfLower, p: f64, se: f64, n_risk: f64, lag: f64) -> Option<f64> {
    match variant {
        ConfLower::Usual => None,
        ConfLower::Peto => Some(((1.0 - p) / n_risk).sqrt()),
        ConfLower::Modified => Some(se * (lag / n_risk).sqrt()),
    }
}
