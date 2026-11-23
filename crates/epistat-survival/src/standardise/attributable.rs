//! The attributable fraction function of a time-to-event outcome from the
//! standardised risks of a model: `AF(t) = 1 - F_0(t) / F(t)`.
//!
//! # The estimate
//!
//! `F(t)` is the risk of the event by `t` in the population as it is, and
//! `F_0(t)` the risk had the exposure been removed, so `AF(t)` is the
//! proportion of the events by `t` that the exposure causes under the model
//! \[1, 2\]. Both are standardised risks of
//! [`standardise_cox`](super::standardise_cox) or
//! [`standardise_cause_specific`](super::standardise_cause_specific): the level
//! `observed` has the covariates as they are and the level `unexposed` has the
//! exposure column, and every column that depends on it, set to the unexposed
//! value in every row. The caller makes that level, by
//! [`Counterfactual::Columns`](super::Counterfactual::Columns) or a full design
//! matrix when there is an interaction. Nothing here can check it: a level that
//! does not remove the exposure gives a fraction of 0, and one that is built
//! from the wrong column gives a number with no meaning. The model must
//! also be adjusted for what confounds the exposure and the outcome, or the
//! fraction is not the effect of the exposure.
//!
//! With competing risks the standardised risk of a cause by
//! [`standardise_cause_specific`](super::standardise_cause_specific) gives the
//! fraction of the events of that cause, and the risk of the Cox model for the
//! cause as if the other events did not occur gives the fraction in the world
//! without them; the two answer different questions.
//!
//! # The variance
//!
//! `AF = 1 - r` with `r = F_0 / F`, and [`Standardised::contrast`] gives the
//! standard error of `r` from the influence functions of the two standardised
//! risks, including the covariance between them, which is large because the
//! same persons and the same model give both. The standard error of `AF` is that
//! of `r`. The influence function is that of the standardised risk, so it
//! accounts for the estimated coefficients and baseline hazard and for the
//! sample over which the risks are averaged, as the sandwich of `AFcoxph` of the
//! R package AF does \[3\].
//!
//! The limits are `AF +- z se`, or on the scale of `log(1 - AF) = log r`, which
//! keeps them below 1: `1 - r exp(+- z se / r)`.
//!
//! # Results that are not numbers
//!
//! `AF(t)` is `None` where either risk is `None` or the risk as observed is 0.
//! The standard error is `None` without standard errors in the standardised
//! risks, and the limits on the log scale are also `None` where `r` is 0, that
//! is where the risk without the exposure is 0 and `AF` is 1.
//!
//! # Agreement with R
//!
//! The tests compare `AF(t)` and the two risks with `AFcoxph` of AF 0.1.5 at
//! relative `1e-9`, with continuous and with tied times. The variance is
//! compared with the delete-one jackknife of the risks and of `AF(t)` refitted
//! in R, to 3% relative, because the variance of `AFcoxph` is too small: its
//! score for the Breslow baseline leaves out the compensator term of the
//! influence function of the cumulative hazard. Its variances of the risks are
//! 9% to 40% below this code's and the jackknife's, and its variance of `AF(t)`
//! within 7%.
//!
//! # References
//!
//! 1. Chen L, Lin DY, Zeng D. Attributable fraction functions for censored event
//!    times. Biometrika 2010;97(3):713-26.
//! 2. Sjolander A, Vansteelandt S. Doubly robust estimation of attributable
//!    fractions in survival analysis. Stat Methods Med Res 2017;26(2):948-69.
//! 3. Dahlqwist E, Zetterqvist J, Pawitan Y, Sjolander A. Model-based
//!    estimation of the attributable fraction for cross-sectional, case-control
//!    and cohort studies using the R package AF. Eur J Epidemiol
//!    2016;31(6):575-82.

use epistat_glm::attributable::IntervalScale;
use epistat_stats::normal;

use super::Standardised;
use crate::error::{Error, Result};

/// The attributable fraction at each time of a [`Standardised`] result.
#[derive(Debug, Clone, PartialEq)]
pub struct AfCurve {
    /// The times, as given.
    pub times: Vec<f64>,
    /// `AF(t)`.
    pub af: Vec<Option<f64>>,
    /// The standard error of `AF(t)`.
    pub se: Vec<Option<f64>>,
    /// The lower confidence limit.
    pub lower: Vec<Option<f64>>,
    /// The upper confidence limit.
    pub upper: Vec<Option<f64>>,
    /// The ratio `F_0(t) / F(t)` of the risks, the complement of `AF(t)`.
    pub ratio: Vec<Option<f64>>,
}

/// `AF(t) = 1 - F_0(t) / F(t)` from the standardised risks of the level
/// `unexposed` and the level `observed` at each of `times`, the times of the
/// call that made `s`.
///
/// # Errors
///
/// [`Error::Input`] for a level outside the levels of `s`, the same level
/// twice, `times` of another length than the times of `s`, and a `level` outside
/// `(0, 1)`.
pub fn af_from_standardised(
    s: &Standardised,
    times: &[f64],
    unexposed: usize,
    observed: usize,
    level: f64,
    scale: IntervalScale,
) -> Result<AfCurve> {
    if times.len() != s.n_times {
        return Err(Error::input(format!(
            "times has {} entries and the standardised risks {}",
            times.len(),
            s.n_times
        )));
    }
    if unexposed == observed {
        return Err(Error::input(
            "the unexposed and the observed level must be different levels",
        ));
    }
    if !(level > 0.0 && level < 1.0) {
        return Err(Error::input("level must be in (0, 1)"));
    }
    let contrast = s.contrast(unexposed, observed)?;
    let z = normal::two_sided_z(level);
    let t = s.n_times;
    let mut curve = AfCurve {
        times: times.to_vec(),
        af: Vec::with_capacity(t),
        se: Vec::with_capacity(t),
        lower: Vec::with_capacity(t),
        upper: Vec::with_capacity(t),
        ratio: contrast.ratio.clone(),
    };
    for k in 0..t {
        let r = contrast.ratio[k];
        let af = r.map(|r| 1.0 - r);
        let se = contrast
            .ratio_inference
            .as_ref()
            .and_then(|inf| inf.std_err[k]);
        curve.af.push(af);
        curve.se.push(se);
        let limits = match (af, r, se) {
            (Some(af), Some(r), Some(se)) => match scale {
                IntervalScale::Identity => Some((af - z * se, af + z * se)),
                IntervalScale::LogComplement if r > 0.0 => {
                    let half = z * se / r;
                    Some((1.0 - r * half.exp(), 1.0 - r * (-half).exp()))
                }
                IntervalScale::LogComplement => None,
            },
            _ => None,
        };
        curve.lower.push(limits.map(|l| l.0));
        curve.upper.push(limits.map(|l| l.1));
    }
    Ok(curve)
}
