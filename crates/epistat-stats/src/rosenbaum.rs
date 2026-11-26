//! Deprecated: this module moved to `epistat_epi::rosenbaum` in 0.3.0, where its
//! documentation, explainer and tests are. The copy here keeps the items and
//! the results of 0.2 for one release and is removed in 0.4.0.

#![expect(deprecated, reason = "the copy of 0.2 calls its own deprecated items")]

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::dist::normal;
use crate::quantile::type7_unsorted;
use crate::signed_rank::{self, SignedRankError, average_ranks};
use crate::sum::lanes;

/// What goes wrong in a sensitivity analysis.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::rosenbaum::SensitivityError`"
)]
pub enum SensitivityError {
    /// `gamma` below 1 or not finite.
    Gamma(f64),
    /// A response that is not finite.
    NotFinite,
    /// No pair with a non-zero difference, or no pairs.
    NoInformation,
    /// A matched set with fewer than two units.
    SetTooSmall,
    /// The scale of the scores, a quantile of the differences within sets,
    /// is zero.
    ZeroScale,
    /// A tuning value outside its range.
    Parameter(&'static str),
    /// The variance of the statistic is zero.
    ZeroVariance,
    /// From the exact signed rank distribution.
    SignedRank(SignedRankError),
}

impl Display for SensitivityError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gamma(g) => write!(f, "gamma must be at least 1, got {g}"),
            Self::NotFinite => write!(f, "a response is not finite"),
            Self::NoInformation => write!(f, "no pair has a non-zero difference"),
            Self::SetTooSmall => write!(f, "a matched set has fewer than two units"),
            Self::ZeroScale => write!(
                f,
                "the scale of the differences within sets is zero; increase lambda, or for a binary outcome use binary_bounds"
            ),
            Self::Parameter(what) => write!(f, "{what} is out of range"),
            Self::ZeroVariance => write!(f, "the variance of the statistic is zero"),
            Self::SignedRank(e) => write!(f, "{e}"),
        }
    }
}

impl Error for SensitivityError {}

impl From<SignedRankError> for SensitivityError {
    fn from(e: SignedRankError) -> Self {
        Self::SignedRank(e)
    }
}

fn check_gamma(gamma: f64) -> Result<(), SensitivityError> {
    if gamma.is_finite() && gamma >= 1.0 {
        Ok(())
    } else {
        Err(SensitivityError::Gamma(gamma))
    }
}

/// How the p-value bounds of the signed rank test are computed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::rosenbaum::WilcoxonDistribution`"
)]
pub enum WilcoxonDistribution {
    /// The normal approximation.
    Normal,
    /// The exact distribution, for up to a few hundred pairs.
    Exact,
}

/// The bounds of the signed rank test at one `gamma`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::rosenbaum::WilcoxonBounds`"
)]
pub struct WilcoxonBounds {
    /// The sensitivity parameter.
    pub gamma: f64,
    /// The number of pairs with a non-zero difference.
    pub pairs: usize,
    /// The signed rank statistic, the rank sum of the positive differences.
    pub statistic: f64,
    /// The smaller bound of the one-sided p-value.
    pub p_lower: f64,
    /// The larger bound of the one-sided p-value.
    pub p_upper: f64,
}

/// The bounds of the one-sided signed rank test of positive differences.
///
/// # Errors
///
/// [`SensitivityError::Gamma`], [`SensitivityError::NotFinite`],
/// [`SensitivityError::NoInformation`] when every difference is zero, and
/// from the exact distribution.
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::rosenbaum::wilcoxon_bounds`"
)]
pub fn wilcoxon_bounds(
    differences: &[f64],
    gamma: f64,
    distribution: WilcoxonDistribution,
) -> Result<WilcoxonBounds, SensitivityError> {
    check_gamma(gamma)?;
    SignedRanks::new(differences)?.bounds(gamma, distribution)
}

/// The ranks of the non-zero differences and the signed rank statistic.
struct SignedRanks {
    ranks: Vec<f64>,
    statistic: f64,
}

impl SignedRanks {
    fn new(differences: &[f64]) -> Result<Self, SensitivityError> {
        if differences.iter().any(|d| !d.is_finite()) {
            return Err(SensitivityError::NotFinite);
        }
        let kept: Vec<f64> = differences.iter().copied().filter(|d| *d != 0.0).collect();
        if kept.is_empty() {
            return Err(SensitivityError::NoInformation);
        }
        let absolute: Vec<f64> = kept.iter().map(|d| d.abs()).collect();
        let ranks = average_ranks(&absolute)?;
        let statistic = lanes::sum_map2(&ranks, &kept, |r, d| if d > 0.0 { r } else { 0.0 });
        Ok(Self { ranks, statistic })
    }

    fn bounds(
        &self,
        gamma: f64,
        distribution: WilcoxonDistribution,
    ) -> Result<WilcoxonBounds, SensitivityError> {
        check_gamma(gamma)?;
        let p_plus = gamma / (1.0 + gamma);
        let p_minus = 1.0 / (1.0 + gamma);
        let tail = |p: f64| -> Result<f64, SensitivityError> {
            match distribution {
                WilcoxonDistribution::Normal => {
                    signed_rank::normal_upper_tail(&self.ranks, self.statistic, p)
                        .ok_or(SensitivityError::ZeroVariance)
                }
                WilcoxonDistribution::Exact => Ok(signed_rank::exact_upper_tail(
                    &self.ranks,
                    self.statistic,
                    p,
                )?),
            }
        };
        Ok(WilcoxonBounds {
            gamma,
            pairs: self.ranks.len(),
            statistic: self.statistic,
            p_lower: tail(p_minus)?,
            p_upper: tail(p_plus)?,
        })
    }
}

/// The signed rank statistic of the sorted differences shifted by `tau`, with
/// the zeros at the lowest ranks and tied absolute values at their average
/// rank. The shifted differences below `tau` and above it are each in order of
/// their absolute values, so the ranks come from one merge.
#[expect(clippy::float_cmp, reason = "tied values are equal")]
fn shifted_statistic(sorted: &[f64], tau: f64) -> f64 {
    let n = sorted.len();
    let below = sorted.partition_point(|d| d - tau < 0.0);
    let zeros = sorted[below..].partition_point(|d| d - tau == 0.0);
    let mut negative = below;
    let mut positive = below + zeros;
    let mut used = zeros;
    let mut statistic = 0.0;
    while negative > 0 || positive < n {
        let low = if negative > 0 {
            (sorted[negative - 1] - tau).abs()
        } else {
            f64::INFINITY
        };
        let high = if positive < n {
            sorted[positive] - tau
        } else {
            f64::INFINITY
        };
        let value = low.min(high);
        let mut from_below = 0;
        while negative > 0 && (sorted[negative - 1] - tau).abs() == value {
            negative -= 1;
            from_below += 1;
        }
        let mut from_above = 0;
        while positive < n && sorted[positive] - tau == value {
            positive += 1;
            from_above += 1;
        }
        let group = from_below + from_above;
        // The group takes ranks used + 1 ..= used + group.
        let mean = (2 * used + 1 + group) as f64 / 2.0;
        statistic += from_above as f64 * mean;
        used += group;
    }
    statistic
}

/// The bounds of the Hodges-Lehmann estimate of the shift at one `gamma`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::rosenbaum::HodgesLehmannBounds`"
)]
pub struct HodgesLehmannBounds {
    /// The sensitivity parameter.
    pub gamma: f64,
    /// The lower bound of the estimate.
    pub lower: f64,
    /// The upper bound of the estimate.
    pub upper: f64,
}

/// The bounds of the Hodges-Lehmann estimate of the shift of paired
/// differences, for each `gamma`.
///
/// # Errors
///
/// [`SensitivityError::Gamma`], [`SensitivityError::NotFinite`],
/// [`SensitivityError::NoInformation`] for no pairs, and
/// [`SensitivityError::SignedRank`] when the range of the differences is not
/// finite, since a shifted difference could then overflow.
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::rosenbaum::hodges_lehmann_bounds`"
)]
pub fn hodges_lehmann_bounds(
    differences: &[f64],
    gammas: &[f64],
) -> Result<Vec<HodgesLehmannBounds>, SensitivityError> {
    for g in gammas {
        check_gamma(*g)?;
    }
    if differences.iter().any(|d| !d.is_finite()) {
        return Err(SensitivityError::NotFinite);
    }
    let n = differences.len();
    if n == 0 {
        return Err(SensitivityError::NoInformation);
    }
    let mut sorted = differences.to_vec();
    sorted.sort_by(f64::total_cmp);
    // Every shift tried lies between the smallest and the largest difference.
    if !(sorted[n - 1] - sorted[0]).is_finite() {
        return Err(SensitivityError::SignedRank(SignedRankError::NotFinite));
    }
    let negative_zero = sorted.iter().any(|d| *d == 0.0 && d.is_sign_negative());
    let total = (n * (n + 1)) as f64 / 2.0;
    let mut walsh = WalshSearch::new(&sorted);
    let mut out = Vec::with_capacity(gammas.len());
    for &gamma in gammas {
        let t_low = total / (1.0 + gamma);
        let t_high = total * gamma / (1.0 + gamma);
        // Upper bound: the first average above which T is at most t_low.
        let upper = walsh.first_crossing(|tau| shifted_statistic(&sorted, tau) <= t_low);
        // Lower bound: the first average above which T is below t_high.
        let lower = walsh.first_crossing(|tau| shifted_statistic(&sorted, tau) < t_high);
        // A zero average is -0.0 when a difference is.
        let sign = |v: f64| if v == 0.0 && negative_zero { -0.0 } else { v };
        out.push(HodgesLehmannBounds {
            gamma,
            lower: sign(lower),
            upper: sign(upper),
        });
    }
    Ok(out)
}

/// Search among the Walsh averages `(d_i + d_j) / 2`, `i <= j`, of sorted
/// differences, without listing them.
struct WalshSearch<'a> {
    sorted: &'a [f64],
    /// Per row, the first column of the averages still possible.
    first: Vec<usize>,
    /// Per row, one past the last column of those.
    end: Vec<usize>,
    /// Per row, the first column whose average is above the value last tried.
    cut: Vec<usize>,
    state: u64,
}

impl<'a> WalshSearch<'a> {
    fn new(sorted: &'a [f64]) -> Self {
        let n = sorted.len();
        Self {
            sorted,
            first: Vec::with_capacity(n),
            end: Vec::with_capacity(n),
            cut: vec![0; n],
            state: 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn average(&self, row: usize, column: usize) -> f64 {
        self.sorted[row].midpoint(self.sorted[column])
    }

    fn next_random(&mut self) -> u64 {
        self.state = self.state.wrapping_add(epistat_rng::GAMMA);
        epistat_rng::mix64(self.state)
    }

    /// Fill `cut` with, per row, the first column from the diagonal on whose
    /// average is above `value`, and return the smallest average above `value`
    /// anywhere, or `None` when there is none.
    fn cut_above(&mut self, value: f64) -> Option<f64> {
        let n = self.sorted.len();
        let mut columns = n;
        let mut next: Option<f64> = None;
        for row in 0..n {
            while columns > 0 && self.average(row, columns - 1) > value {
                columns -= 1;
            }
            let cut = columns.max(row);
            self.cut[row] = cut;
            if cut < n {
                let candidate = self.average(row, cut);
                if next.is_none_or(|best| candidate < best) {
                    next = Some(candidate);
                }
            }
        }
        next
    }

    /// The smallest average `v` such that `above(tau)` holds for `tau` halfway
    /// between `v` and the next larger average, or `v + 1` when there is none.
    /// `above` must go from false to true as `tau` increases, and be true above
    /// every average.
    fn first_crossing(&mut self, above: impl Fn(f64) -> bool) -> f64 {
        let n = self.sorted.len();
        self.first.clear();
        self.first.extend(0..n);
        self.end.clear();
        self.end.resize(n, n);
        // The answer is in (low, high]: `high` is known to cross, and every
        // average up to `low` is known not to.
        let mut high = self.sorted[n - 1];
        loop {
            let mut active = 0_usize;
            let mut least: Option<f64> = None;
            for row in 0..n {
                if self.first[row] < self.end[row] {
                    active += self.end[row] - self.first[row];
                    let candidate = self.average(row, self.first[row]);
                    if least.is_none_or(|best| candidate < best) {
                        least = Some(candidate);
                    }
                }
            }
            let Some(least) = least else { return high };
            if least >= high {
                return high;
            }
            // A random average of the active ones below `high`; the least
            // one when the draws keep landing on `high`.
            let mut pivot = least;
            for _ in 0..4 {
                let bound = u64::try_from(active).unwrap_or(u64::MAX);
                let mut draw = usize::try_from(self.next_random() % bound).unwrap_or(0);
                let mut row = 0;
                while draw >= self.end[row].saturating_sub(self.first[row]) {
                    draw -= self.end[row].saturating_sub(self.first[row]);
                    row += 1;
                }
                let candidate = self.average(row, self.first[row] + draw);
                if candidate < high {
                    pivot = candidate;
                    break;
                }
            }
            let next = self.cut_above(pivot);
            let tau = match next {
                Some(next) => pivot + (next - pivot) / 2.0,
                None => pivot + 1.0,
            };
            if above(tau) {
                high = pivot;
                for row in 0..n {
                    self.end[row] = self.end[row].min(self.cut[row]);
                }
            } else {
                for row in 0..n {
                    self.first[row] = self.first[row].max(self.cut[row]);
                }
            }
        }
    }
}

/// The scores of an m-statistic over matched sets.
#[derive(Debug, Clone, Copy, PartialEq)]
#[deprecated(since = "0.3.0", note = "moved to `epistat_epi::rosenbaum::MScore`")]
pub struct MScore {
    /// The inner trim `a` in units of the scale.
    pub inner: f64,
    /// The outer trim `b`; infinite for no trimming.
    pub trim: f64,
    /// The quantile of the differences within sets that sets the scale, in
    /// the open interval (0, 1).
    pub lambda: f64,
    /// Scores for the mean difference between the treated unit and its
    /// controls instead.
    pub treated_on_treated: bool,
}

impl Default for MScore {
    /// Huber's m-statistic as `sensitivitymv::senmv` has it: no inner trim,
    /// outer trim 2.5, the median scale.
    fn default() -> Self {
        Self {
            inner: 0.0,
            trim: 2.5,
            lambda: 0.5,
            treated_on_treated: false,
        }
    }
}

/// The scores of every unit of every set, the treated unit first in each.
///
/// # Errors
///
/// [`SensitivityError::SetTooSmall`], [`SensitivityError::NotFinite`],
/// [`SensitivityError::Parameter`] and [`SensitivityError::ZeroScale`].
#[deprecated(since = "0.3.0", note = "moved to `epistat_epi::rosenbaum::m_scores`")]
pub fn m_scores(sets: &[Vec<f64>], spec: &MScore) -> Result<Vec<Vec<f64>>, SensitivityError> {
    if !(spec.lambda > 0.0 && spec.lambda < 1.0) {
        return Err(SensitivityError::Parameter("lambda"));
    }
    if !(spec.inner >= 0.0 && spec.inner <= spec.trim) {
        return Err(SensitivityError::Parameter("inner trim"));
    }
    if sets.is_empty() {
        return Err(SensitivityError::NoInformation);
    }
    for set in sets {
        if set.len() < 2 {
            return Err(SensitivityError::SetTooSmall);
        }
        if set.iter().any(|v| !v.is_finite()) {
            return Err(SensitivityError::NotFinite);
        }
    }
    let rescale = spec.trim < f64::INFINITY || spec.inner > 0.0;
    let scale = if rescale {
        let mut all = Vec::new();
        for set in sets {
            for (j, a) in set.iter().enumerate() {
                for (k, b) in set.iter().enumerate() {
                    if j != k {
                        all.push((a - b).abs());
                    }
                }
            }
        }
        let h = type7_unsorted(&mut all, spec.lambda).ok_or(SensitivityError::NotFinite)?;
        if h > 0.0 {
            h
        } else {
            return Err(SensitivityError::ZeroScale);
        }
    } else {
        1.0
    };
    let psi = |x: f64| -> f64 {
        if !rescale {
            return x;
        }
        let z = x / scale;
        let magnitude = if spec.trim < f64::INFINITY && spec.inner < spec.trim {
            ((z.abs() - spec.inner).max(0.0) / (spec.trim - spec.inner)).min(1.0)
        } else if spec.trim < f64::INFINITY {
            f64::from(z.abs() > spec.inner)
        } else {
            (z.abs() - spec.inner).max(0.0)
        };
        if z > 0.0 {
            magnitude
        } else if z < 0.0 {
            -magnitude
        } else {
            0.0
        }
    };
    let n_sets = sets.len() as f64;
    Ok(sets
        .iter()
        .map(|set| {
            let m = set.len() as f64;
            set.iter()
                .enumerate()
                .map(|(j, a)| {
                    let mut sum = 0.0;
                    for (k, b) in set.iter().enumerate() {
                        if k != j {
                            sum += psi(a - b);
                        }
                    }
                    if spec.treated_on_treated {
                        sum / (m - 1.0) / n_sets
                    } else {
                        sum / m
                    }
                })
                .collect()
        })
        .collect())
}

/// The bound of a test on the treated unit's score summed over sets.
#[derive(Debug, Clone, Copy, PartialEq)]
#[deprecated(since = "0.3.0", note = "moved to `epistat_epi::rosenbaum::SetBounds`")]
pub struct SetBounds {
    /// The sensitivity parameter.
    pub gamma: f64,
    /// The sum over sets of the treated unit's score.
    pub statistic: f64,
    /// The largest expectation of the statistic under `gamma`.
    pub expectation: f64,
    /// The variance at that expectation.
    pub variance: f64,
    /// The standardised statistic, `(statistic - expectation) / sqrt(variance)`.
    pub deviate: f64,
    /// The upper bound of the one-sided p-value, the normal tail of the
    /// deviate.
    pub p_upper: f64,
}

/// The bound of the one-sided test from the scores of matched sets, the
/// treated unit's score first in each.
///
/// # Errors
///
/// [`SensitivityError::Gamma`], [`SensitivityError::SetTooSmall`],
/// [`SensitivityError::NotFinite`] and [`SensitivityError::ZeroVariance`].
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::rosenbaum::set_bounds`"
)]
pub fn set_bounds(scores: &[Vec<f64>], gamma: f64) -> Result<SetBounds, SensitivityError> {
    check_gamma(gamma)?;
    if scores.is_empty() {
        return Err(SensitivityError::NoInformation);
    }
    let mut statistic = 0.0;
    let mut expectation = 0.0;
    let mut variance = 0.0;
    for set in scores {
        if set.len() < 2 {
            return Err(SensitivityError::SetTooSmall);
        }
        if set.iter().any(|v| !v.is_finite()) {
            return Err(SensitivityError::NotFinite);
        }
        statistic += set[0];
        let mut sorted = set.clone();
        sorted.sort_by(f64::total_cmp);
        let m = sorted.len();
        let mut best_mean = f64::NEG_INFINITY;
        let mut best_var = f64::NEG_INFINITY;
        for a in 1..m {
            let (low, high) = sorted.split_at(a);
            let denom = a as f64 + gamma * (m - a) as f64;
            let low_sum: f64 = low.iter().sum();
            let high_sum: f64 = high.iter().sum();
            let low_sq: f64 = low.iter().map(|v| v * v).sum();
            let high_sq: f64 = high.iter().map(|v| v * v).sum();
            let mean = (low_sum + gamma * high_sum) / denom;
            let var = (low_sq + gamma * high_sq) / denom - mean * mean;
            if mean > best_mean {
                best_mean = mean;
                best_var = var;
            } else if mean.total_cmp(&best_mean).is_eq() {
                best_var = best_var.max(var);
            }
        }
        expectation += best_mean;
        variance += best_var;
    }
    if !(variance > 0.0 && variance.is_finite()) {
        return Err(SensitivityError::ZeroVariance);
    }
    let deviate = (statistic - expectation) / variance.sqrt();
    Ok(SetBounds {
        gamma,
        statistic,
        expectation,
        variance,
        deviate,
        p_upper: normal::cdf(-deviate),
    })
}

/// The m-statistic bound for matched sets of raw responses, the treated unit
/// first in each: [`m_scores`] followed by [`set_bounds`].
///
/// # Errors
///
/// Those of the two.
#[deprecated(
    since = "0.3.0",
    note = "moved to `epistat_epi::rosenbaum::matched_sets_bounds`"
)]
pub fn matched_sets_bounds(
    sets: &[Vec<f64>],
    gamma: f64,
    spec: &MScore,
) -> Result<SetBounds, SensitivityError> {
    set_bounds(&m_scores(sets, spec)?, gamma)
}
