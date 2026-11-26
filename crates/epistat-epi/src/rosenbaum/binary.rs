//! Bounds for a binary outcome in matched sets or strata: the
//! Mantel-Haenszel test under hidden bias.
//!
//! A stratum is a 2 x 2 table of treated and control units by whether they
//! had the outcome. Given its margins, the number of treated cases `X` has,
//! without hidden bias, the hypergeometric distribution. When the odds of
//! treatment of two units in a stratum may differ by up to `gamma`, the
//! distribution of `X` that gives the largest upper tail is Fisher's
//! noncentral hypergeometric distribution with odds ratio `gamma`, and the
//! one that gives the smallest has odds ratio `1 / gamma` \[1, chapter 4\].
//! The bounds on the one-sided p-value of the Mantel-Haenszel
//! statistic, the total number of treated cases, are the upper tails of the
//! sum of independent such variables over the strata.
//!
//! # Algorithm
//!
//! With `m1` cases, `m2` non-cases and `n` treated units in a stratum, `X`
//! takes the values `max(0, n - m2) ..= min(n, m1)` with probabilities
//! proportional to `C(m1, x) C(m2, n - x) w^x`. The ratio of successive
//! weights is `(m1 - x) (n - x) w / ((x + 1) (m2 - n + x + 1))`, and the
//! weights are formed on the log scale from these ratios and normalised. A
//! stratum with one treated unit, or one case, is a Bernoulli variable; the
//! strata that share their margins are added up as one binomial, whose
//! probabilities come from the ratio `(N - k) p / ((k + 1) (1 - p))` outwards
//! from the mode. The distributions are then convolved, and the tail is summed
//! from its smallest terms. [`BinaryDistribution::Normal`] uses the sum of the
//! means and variances of the strata instead, without a continuity
//! correction, as `sensitivity2x2xk::mhLS`.
//!
//! In matched pairs only the discordant pairs carry information, and the
//! bound is the binomial tail of the pairs in which the treated unit is the
//! case, with probability `gamma / (1 + gamma)`.
//!
//! # Validation
//!
//! Checked against `sensitivity2x2xk::mh` and `mhLS` 1.1 \[2\] and the
//! binomial tail of `pbinom` in `epistat-matching`'s `r_rosenbaum_methods`
//! test.
//!
//! # References
//!
//! 1. Rosenbaum PR. Observational Studies, 2nd ed. Springer; 2002.
//! 2. Rosenbaum PR, Small DS. An adaptive Mantel-Haenszel test for
//!    sensitivity analysis in observational studies. Biometrics.
//!    2017;73(2):422-30.

use crate::Error;
use std::collections::BTreeMap;

use super::{SensitivityError, check_gamma, sensitivity_value};
use epistat_stats::dist::normal;

/// The largest number of multiplications the exact convolution may take.
pub const MAX_CONVOLUTION_WORK: f64 = 2.0e9;

/// One matched set or stratum with a binary outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BinaryStratum {
    /// Treated units with the outcome.
    pub treated_cases: u64,
    /// Treated units without it.
    pub treated_noncases: u64,
    /// Control units with the outcome.
    pub control_cases: u64,
    /// Control units without it.
    pub control_noncases: u64,
}

/// How the bounds of the Mantel-Haenszel test are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryDistribution {
    /// The exact distribution, as `sensitivity2x2xk::mh`.
    Exact,
    /// The normal approximation, as `sensitivity2x2xk::mhLS`.
    Normal,
}

/// The bounds of the one-sided Mantel-Haenszel test at one `gamma`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinaryBounds {
    /// The sensitivity parameter.
    pub gamma: f64,
    /// The strata whose margins leave the number of treated cases free.
    pub informative: usize,
    /// The number of treated cases, over all strata.
    pub statistic: u64,
    /// The largest expectation of the statistic under `gamma`.
    pub expectation: f64,
    /// The variance at that expectation.
    pub variance: f64,
    /// The smaller bound of the one-sided p-value.
    pub p_lower: f64,
    /// The larger bound of the one-sided p-value.
    pub p_upper: f64,
}

/// The margins of a stratum: cases, non-cases, treated, and the range of
/// the number of treated cases.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct Margins {
    cases: u64,
    noncases: u64,
    treated: u64,
}

impl Margins {
    fn of(s: &BinaryStratum) -> Option<Self> {
        Some(Self {
            cases: s.treated_cases.checked_add(s.control_cases)?,
            noncases: s.treated_noncases.checked_add(s.control_noncases)?,
            treated: s.treated_cases.checked_add(s.treated_noncases)?,
        })
    }

    fn low(&self) -> u64 {
        self.treated.saturating_sub(self.noncases)
    }

    fn high(&self) -> u64 {
        self.treated.min(self.cases)
    }

    /// The distribution of the number of treated cases minus `low`, with odds
    /// ratio `w`.
    fn distribution(&self, w: f64) -> Vec<f64> {
        let (low, high) = (self.low(), self.high());
        let (m1, m2, n) = (self.cases as f64, self.noncases as f64, self.treated as f64);
        let ln_w = w.ln();
        let mut log_weights = Vec::with_capacity(usize::try_from(high - low + 1).unwrap_or(0));
        let mut current = 0.0;
        log_weights.push(current);
        for x in low..high {
            let x = x as f64;
            current += ((m1 - x) * (n - x) / ((x + 1.0) * (m2 - n + x + 1.0))).ln() + ln_w;
            log_weights.push(current);
        }
        normalise_logs(&log_weights)
    }
}

fn normalise_logs(log_weights: &[f64]) -> Vec<f64> {
    let top = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let mut out: Vec<f64> = log_weights.iter().map(|l| (l - top).exp()).collect();
    let total: f64 = out.iter().sum();
    for v in &mut out {
        *v /= total;
    }
    out
}

/// The binomial distribution of `trials` with probability `p`, from the
/// ratio of successive probabilities outwards from the mode.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the mode lies between 0 and the number of trials"
)]
fn binomial(trials: u64, p: f64) -> Vec<f64> {
    let len = trials as usize + 1;
    if p <= 0.0 {
        let mut out = vec![0.0; len];
        out[0] = 1.0;
        return out;
    }
    if p >= 1.0 {
        let mut out = vec![0.0; len];
        out[len - 1] = 1.0;
        return out;
    }
    let n = trials as f64;
    let odds = p / (1.0 - p);
    let mode = (((n + 1.0) * p).floor() as usize).min(len - 1);
    let mut out = vec![0.0; len];
    out[mode] = 1.0;
    for k in mode..len - 1 {
        let kf = k as f64;
        out[k + 1] = out[k] * (n - kf) / (kf + 1.0) * odds;
    }
    for k in (0..mode).rev() {
        let kf = k as f64;
        out[k] = out[k + 1] * (kf + 1.0) / ((n - kf) * odds);
    }
    let total: f64 = out.iter().sum();
    for v in &mut out {
        *v /= total;
    }
    out
}

fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; a.len() + b.len() - 1];
    for (i, x) in a.iter().enumerate() {
        if *x == 0.0 {
            continue;
        }
        for (j, y) in b.iter().enumerate() {
            out[i + j] += x * y;
        }
    }
    out
}

/// The strata grouped by their margins, the uninformative ones left out,
/// and the statistic less the smallest value it can take.
struct Prepared {
    groups: Vec<(Margins, u64)>,
    statistic: u64,
    shifted_statistic: u64,
    informative: usize,
}

fn prepare(strata: &[BinaryStratum]) -> Result<Prepared, Error> {
    if strata.is_empty() {
        return Err(SensitivityError::Empty("strata").into());
    }
    let mut groups: BTreeMap<Margins, u64> = BTreeMap::new();
    let mut statistic = 0_u64;
    let mut floor = 0_u64;
    let mut informative = 0;
    for s in strata {
        let m = Margins::of(s).ok_or(SensitivityError::Parameter("a count overflows"))?;
        statistic = statistic
            .checked_add(s.treated_cases)
            .ok_or(SensitivityError::Parameter("a count overflows"))?;
        floor += m.low();
        if m.low() < m.high() {
            informative += 1;
            *groups.entry(m).or_insert(0) += 1;
        }
    }
    if informative == 0 {
        return Err(SensitivityError::NoInformation.into());
    }
    Ok(Prepared {
        groups: groups.into_iter().collect(),
        statistic,
        shifted_statistic: statistic - floor,
        informative,
    })
}

impl Prepared {
    /// The mean and variance of the shifted statistic with odds ratio `w`.
    fn moments(&self, w: f64) -> (f64, f64) {
        let mut mean = 0.0;
        let mut variance = 0.0;
        for (m, count) in &self.groups {
            let d = m.distribution(w);
            let mu: f64 = d.iter().enumerate().map(|(k, p)| k as f64 * p).sum();
            let var: f64 = d
                .iter()
                .enumerate()
                .map(|(k, p)| (k as f64 - mu).powi(2) * p)
                .sum();
            let c = *count as f64;
            mean += c * mu;
            variance += c * var;
        }
        (mean, variance)
    }

    fn work(&self) -> f64 {
        let mut length = 1.0_f64;
        let mut work = 0.0_f64;
        for (m, count) in &self.groups {
            let width = (m.high() - m.low()) as f64;
            let c = *count as f64;
            if m.high() - m.low() == 1 {
                work += length * (c + 1.0) + c;
                length += c;
            } else {
                // `c` convolutions with a distribution of `width + 1` values.
                work += c * (length + c * width / 2.0) * (width + 1.0);
                length += c * width;
            }
        }
        work
    }

    fn exact_upper_tail(&self, w: f64) -> f64 {
        let mut total = vec![1.0];
        for (m, count) in &self.groups {
            if m.high() - m.low() == 1 {
                let d = m.distribution(w);
                total = convolve(&total, &binomial(*count, d[1]));
            } else {
                let d = m.distribution(w);
                for _ in 0..*count {
                    total = convolve(&total, &d);
                }
            }
        }
        let from = usize::try_from(self.shifted_statistic).unwrap_or(usize::MAX);
        if from >= total.len() {
            return 0.0;
        }
        let mut sum = 0.0;
        for v in total[from..].iter().rev() {
            sum += v;
        }
        sum.min(1.0)
    }

    fn normal_upper_tail(&self, w: f64) -> Result<f64, Error> {
        let (mean, variance) = self.moments(w);
        if !(variance > 0.0 && variance.is_finite()) {
            return Err(SensitivityError::ZeroVariance.into());
        }
        Ok(normal::cdf(
            -(self.shifted_statistic as f64 - mean) / variance.sqrt(),
        ))
    }

    fn bounds(&self, gamma: f64, distribution: BinaryDistribution) -> Result<BinaryBounds, Error> {
        check_gamma(gamma)?;
        let (mean, variance) = self.moments(gamma);
        let floor = (self.statistic - self.shifted_statistic) as f64;
        let (p_lower, p_upper) = match distribution {
            BinaryDistribution::Exact => {
                if self.work() > MAX_CONVOLUTION_WORK {
                    return Err(SensitivityError::Parameter(
                        "the exact distribution is too large; use the normal approximation",
                    )
                    .into());
                }
                (
                    self.exact_upper_tail(1.0 / gamma),
                    self.exact_upper_tail(gamma),
                )
            }
            BinaryDistribution::Normal => (
                self.normal_upper_tail(1.0 / gamma)?,
                self.normal_upper_tail(gamma)?,
            ),
        };
        Ok(BinaryBounds {
            gamma,
            informative: self.informative,
            statistic: self.statistic,
            expectation: floor + mean,
            variance,
            p_lower,
            p_upper,
        })
    }
}

/// The bounds of the one-sided Mantel-Haenszel test that the treated have
/// the outcome more often, over strata with a binary outcome.
///
/// Strata whose margins fix the number of treated cases, such as a set where
/// everyone or no one had the outcome, add a constant and are counted out of
/// `informative`.
///
/// # Errors
///
/// [`crate::Error`], [`crate::Error`] when no
/// stratum is informative, [`crate::Error`] and
/// [`crate::Error`] when a count overflows or the exact
/// distribution would take more than [`MAX_CONVOLUTION_WORK`]
/// multiplications.
pub fn binary_bounds(
    strata: &[BinaryStratum],
    gamma: f64,
    distribution: BinaryDistribution,
) -> Result<BinaryBounds, Error> {
    check_gamma(gamma)?;
    prepare(strata)?.bounds(gamma, distribution)
}

/// The sensitivity value of the Mantel-Haenszel test: the smallest `gamma`
/// at which the upper bound of [`binary_bounds`] reaches `alpha`.
///
/// # Errors
///
/// As [`binary_bounds`] and [`sensitivity_value`].
pub fn binary_sensitivity_value(
    strata: &[BinaryStratum],
    alpha: f64,
    distribution: BinaryDistribution,
) -> Result<f64, Error> {
    let prepared = prepare(strata)?;
    sensitivity_value(alpha, |gamma| {
        prepared.bounds(gamma, distribution).map(|b| b.p_upper)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(treated: u64, control: u64) -> BinaryStratum {
        BinaryStratum {
            treated_cases: treated,
            treated_noncases: 1 - treated,
            control_cases: control,
            control_noncases: 1 - control,
        }
    }

    /// The binomial tail by the sum of `C(n, k) p^k (1 - p)^(n - k)` with the
    /// coefficients built up exactly in integers.
    fn binomial_tail(n: u64, from: u64, p: f64) -> f64 {
        let mut c = 1.0_f64;
        let mut total = 0.0;
        for k in 0..=n {
            if k > 0 {
                c = c * (n - k + 1) as f64 / k as f64;
            }
            if k >= from {
                let (up, down) = (i32::try_from(k).unwrap(), i32::try_from(n - k).unwrap());
                total += c * p.powi(up) * (1.0 - p).powi(down);
            }
        }
        total
    }

    #[test]
    fn pairs_give_the_binomial_tail_of_the_discordant_pairs() {
        let mut strata = Vec::new();
        strata.extend(std::iter::repeat_n(pair(1, 0), 30));
        strata.extend(std::iter::repeat_n(pair(0, 1), 15));
        strata.extend(std::iter::repeat_n(pair(1, 1), 10));
        strata.extend(std::iter::repeat_n(pair(0, 0), 345));
        for gamma in [1.0, 1.25, 1.5, 3.0] {
            let b = binary_bounds(&strata, gamma, BinaryDistribution::Exact).unwrap();
            assert_eq!(b.informative, 45);
            assert_eq!(b.statistic, 40);
            let want = binomial_tail(45, 30, gamma / (1.0 + gamma));
            // A sum of 46 terms, each with a few roundings: 1e-13 relative.
            assert!((b.p_upper - want).abs() <= 1e-13 * want, "{gamma}");
            let low = binomial_tail(45, 30, 1.0 / (1.0 + gamma));
            assert!((b.p_lower - low).abs() <= 1e-13 * low, "{gamma}");
        }
    }

    #[test]
    fn one_stratum_is_the_noncentral_hypergeometric_tail() {
        // Three cases among two treated and four controls: x in 0..=2, with
        // weights C(3, x) C(3, 2 - x) w^x = 3, 9 w, 3 w^2.
        let s = BinaryStratum {
            treated_cases: 2,
            treated_noncases: 0,
            control_cases: 1,
            control_noncases: 3,
        };
        let w = 2.0;
        let b = binary_bounds(&[s], w, BinaryDistribution::Exact).unwrap();
        let total = 3.0 + 9.0 * w + 3.0 * w * w;
        assert!((b.p_upper - 3.0 * w * w / total).abs() < 1e-15);
        let mean = (9.0 * w + 6.0 * w * w) / total;
        assert!((b.expectation - mean).abs() < 1e-15);
    }

    #[test]
    fn uninformative_strata_only_is_an_error() {
        let s = [pair(1, 1), pair(0, 0)];
        assert_eq!(
            binary_bounds(&s, 1.0, BinaryDistribution::Exact),
            Err(SensitivityError::NoInformation.into())
        );
        assert!(binary_bounds(&[pair(1, 0)], 0.5, BinaryDistribution::Exact).is_err());
    }

    #[test]
    fn the_normal_bound_has_the_exact_moments() {
        let mut strata = Vec::new();
        strata.extend(std::iter::repeat_n(pair(1, 0), 30));
        strata.extend(std::iter::repeat_n(pair(0, 1), 15));
        let b = binary_bounds(&strata, 2.0, BinaryDistribution::Normal).unwrap();
        let p = 2.0 / 3.0;
        assert!((b.expectation - 45.0 * p).abs() < 1e-12);
        assert!((b.variance - 45.0 * p * (1.0 - p)).abs() < 1e-12);
    }
}
