//! The number of cases a self-controlled case series needs, by the formula of
//! Musonda, Farrington and Whitaker \[1\].
//!
//! # Algorithm
//!
//! All cases share one observation period, cut into age bands `j` of length
//! `e_j` with relative incidence `exp(alpha_j)` (1 in the first band). Each
//! exposed case has one risk window of length `r`, which falls in age band `j`
//! with probability `p_j`; `1 - sum_j p_j` of the cases are unexposed. With
//! the design relative incidence `rho`,
//!
//! ```text
//! r_j  = r exp(alpha_j) / sum_k exp(alpha_k) e_k
//! pi_j = r_j rho / (r_j rho + 1 - r_j)
//! v_j  = p_j (r_j rho + 1 - r_j) / (1 - sum_k p_k + sum_k p_k (r_k rho + 1 - r_k))
//! A    = 2 sum_j v_j (pi_j ln rho - ln(r_j rho + 1 - r_j))
//! B    = (ln rho)^2 / A  sum_j v_j pi_j (1 - pi_j)
//! n    = (z_(1 - alpha/2) + z_power sqrt(B))^2 / A
//! ```
//!
//! `r_j` is the share of the case's weighted time that the window takes when
//! it lies in band `j`, `pi_j` the probability that the case's event falls in
//! it, and `v_j` the probability that a case is exposed in band `j`. `A` is
//! twice the expected log likelihood ratio per case, and `n` is the number of
//! cases, exposed or not, for a two-sided Wald test of level `alpha` to reject
//! `rho = 1` with the stated power. The study needs `ceil(n)` cases.
//!
//! # Agreement with R
//!
//! The formula is that of `SCCS::samplesize` (SCCS 1.8), which rounds the two
//! normal quantiles to four decimals before it applies them; this function
//! does not, so `n` differs from the unrounded R value in the last bits and
//! its ceiling from `samplesize` only when `n` lies within about `1e-4 n` of an
//! integer. Times are half open, so `end` is the last day of observation plus
//! one, which gives the last age band the extra day `samplesize` adds.
//!
//! # References
//!
//! 1. Musonda P, Farrington CP, Whitaker HJ. Sample sizes for self-controlled
//!    case series studies. Stat Med 2006;25(15):2618-31.

use crate::error::{Error, Result};

/// The design of a self-controlled case series whose size [`sample_size`]
/// computes.
#[derive(Debug, Clone, Copy)]
pub struct SampleSizeSpec<'a> {
    /// The relative incidence to detect, positive and not 1.
    pub relative_incidence: f64,
    /// The length of the risk window, shorter than every age band.
    pub risk_length: f64,
    /// The start of observation, the same for every case.
    pub start: f64,
    /// The end of observation, exclusive.
    pub end: f64,
    /// The strictly increasing age cut points inside `(start, end)`.
    pub age_cuts: &'a [f64],
    /// The relative incidence of each age band after the first, positive.
    pub age_relative_incidence: &'a [f64],
    /// The probability that a case is exposed in each age band, one per band,
    /// not negative and with a sum in `(0, 1]`.
    pub exposed: &'a [f64],
    /// The two-sided level of the test, in `(0, 1)`.
    pub alpha: f64,
    /// The power, in `(0, 1)`.
    pub power: f64,
}

fn invalid(message: impl Into<String>) -> Error {
    Error::input(message)
}

/// The number of cases a self-controlled case series needs to detect
/// [`SampleSizeSpec::relative_incidence`], before rounding up. See the module
/// documentation for the formula.
///
/// # Errors
///
/// [`Error::Input`] when the relative incidence is not positive and finite or
/// is 1, the risk window is not positive or not shorter than every age band,
/// the observation period is empty, the age cuts are not strictly increasing
/// inside it, the age relative incidences or the exposure probabilities do not
/// have one entry per band (after the first, for the former), a probability is
/// negative or their sum is not in `(0, 1]`, or `alpha` or `power` is not in
/// `(0, 1)`.
#[expect(
    clippy::float_cmp,
    reason = "a relative incidence of exactly 1 has nothing to detect"
)]
pub fn sample_size(spec: &SampleSizeSpec<'_>) -> Result<f64> {
    let rho = spec.relative_incidence;
    if !(rho.is_finite() && rho > 0.0) || rho == 1.0 {
        return Err(invalid(format!(
            "the relative incidence is {rho}, which is not positive, finite and other than 1"
        )));
    }
    if !(spec.start.is_finite() && spec.end.is_finite() && spec.start < spec.end) {
        return Err(invalid(
            "the observation period is not a finite, non-empty interval",
        ));
    }
    let mut bounds = vec![spec.start];
    bounds.extend_from_slice(spec.age_cuts);
    bounds.push(spec.end);
    if bounds
        .windows(2)
        .any(|w| w[0].partial_cmp(&w[1]) != Some(std::cmp::Ordering::Less))
    {
        return Err(invalid(
            "the age cuts must be strictly increasing and inside the observation period",
        ));
    }
    let bands = bounds.len() - 1;
    if spec.age_relative_incidence.len() + 1 != bands || spec.exposed.len() != bands {
        return Err(invalid(format!(
            "{bands} age bands need {} age relative incidences and {bands} exposure probabilities, found {} and {}",
            bands - 1,
            spec.age_relative_incidence.len(),
            spec.exposed.len()
        )));
    }
    if spec
        .age_relative_incidence
        .iter()
        .any(|&a| !(a.is_finite() && a > 0.0))
    {
        return Err(invalid(
            "the age relative incidences must be positive and finite",
        ));
    }
    let total: f64 = spec.exposed.iter().sum();
    if spec.exposed.iter().any(|&p| !(p.is_finite() && p >= 0.0)) || !(total > 0.0 && total <= 1.0)
    {
        return Err(invalid(
            "the exposure probabilities must not be negative and must sum to more than 0 and at most 1",
        ));
    }
    let length: Vec<f64> = bounds.windows(2).map(|w| w[1] - w[0]).collect();
    let r = spec.risk_length;
    if !(r.is_finite() && r > 0.0) || length.iter().any(|&e| r >= e) {
        return Err(invalid(
            "the risk window must be positive and shorter than every age band",
        ));
    }
    for (name, value) in [("alpha", spec.alpha), ("power", spec.power)] {
        if !(value > 0.0 && value < 1.0) {
            return Err(invalid(format!(
                "{name} is {value}, which is not in (0, 1)"
            )));
        }
    }
    let age: Vec<f64> = std::iter::once(1.0)
        .chain(spec.age_relative_incidence.iter().copied())
        .collect();
    let weighted: f64 = age.iter().zip(&length).map(|(a, e)| a * e).sum();
    let share: Vec<f64> = age.iter().map(|a| a * r / weighted).collect();
    let odds: Vec<f64> = share.iter().map(|&s| s * rho + 1.0 - s).collect();
    let pi: Vec<f64> = share.iter().zip(&odds).map(|(s, o)| s * rho / o).collect();
    let denominator = 1.0 - total
        + spec
            .exposed
            .iter()
            .zip(&odds)
            .map(|(p, o)| p * o)
            .sum::<f64>();
    let v: Vec<f64> = spec
        .exposed
        .iter()
        .zip(&odds)
        .map(|(p, o)| p * o / denominator)
        .collect();
    let log_rho = rho.ln();
    let a: f64 = 2.0
        * (0..bands)
            .map(|j| v[j] * (pi[j] * log_rho - odds[j].ln()))
            .sum::<f64>();
    let b = log_rho * log_rho / a
        * (0..bands)
            .map(|j| v[j] * pi[j] * (1.0 - pi[j]))
            .sum::<f64>();
    let za = epistat_stats::dist::normal::quantile(1.0 - spec.alpha / 2.0);
    let zb = epistat_stats::dist::normal::quantile(spec.power);
    let root = za + zb * b.sqrt();
    Ok(root * root / a)
}
