//! Confidence limits of an absolute risk on a transformed scale, as
//! `confint.predictCSC`.

/// The scale on which the confidence limits of an absolute risk are formed,
/// `absRisk.transform` of `confint.predictCSC`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RiskLimits {
    /// No limits.
    None,
    /// `F -+ z se`, restricted to `[0, 1]`.
    Plain,
    /// `F exp(+- z se / F)`, the upper limit at most 1.
    Log,
    /// `F^exp(-+ z se / (F |log F|))`, the log-log scale.
    #[default]
    LogLog,
    /// `1 - (1 - F)^exp(+- z se / ((1 - F) |log(1 - F)|))`.
    CLogLog,
}

type Limits = (Vec<Option<f64>>, Vec<Option<f64>>);

/// The limits for every risk with a standard error, `z` the upper normal
/// quantile of the level. A limit that is not finite is `None`.
pub(super) fn limits(
    risk: &[Option<f64>],
    std_err: &[Option<f64>],
    kind: RiskLimits,
    z: f64,
) -> Limits {
    let mut lower = Vec::with_capacity(risk.len());
    let mut upper = Vec::with_capacity(risk.len());
    for (f, s) in risk.iter().zip(std_err) {
        let pair = match (f, s) {
            (Some(f), Some(s)) => one(*f, *s, kind, z),
            _ => None,
        };
        lower.push(pair.map(|p| p.0));
        upper.push(pair.map(|p| p.1));
    }
    (lower, upper)
}

fn one(f: f64, se: f64, kind: RiskLimits, z: f64) -> Option<(f64, f64)> {
    // A standard error of zero stays zero on every scale.
    let scaled = |divisor: f64| if se == 0.0 { 0.0 } else { se / divisor };
    let (lo, up) = match kind {
        RiskLimits::None => return None,
        RiskLimits::Plain => {
            let q = scaled(1.0);
            ((f - z * q).max(0.0), (f + z * q).min(1.0))
        }
        RiskLimits::Log => {
            let q = scaled(f);
            (f * (-z * q).exp(), (f * (z * q).exp()).min(1.0))
        }
        RiskLimits::LogLog => {
            let q = scaled(-f * f.ln());
            (f.powf((z * q).exp()), f.powf((-z * q).exp()))
        }
        RiskLimits::CLogLog => {
            let q = scaled(-(1.0 - f) * (1.0 - f).ln());
            (
                1.0 - (1.0 - f).powf((-z * q).exp()),
                1.0 - (1.0 - f).powf((z * q).exp()),
            )
        }
    };
    (lo.is_finite() && up.is_finite()).then_some((lo, up))
}
