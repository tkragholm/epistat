//! Rosenbaum bounds for sensitivity to hidden bias in matched studies.
//!
//! In a matched study without hidden bias, the units in a matched set have the
//! same chance of being the treated one. If an unmeasured covariate changes
//! the odds of treatment of two units in a set by at most a factor `gamma >= 1`,
//! each unit's chance of being the treated one lies in bounds that depend
//! on `gamma` \[1\]. A test statistic then has a range of null distributions,
//! and the largest p-value over the range is the p-value bound. `gamma = 1`
//! is no hidden bias. The functions here report the bounds for a given
//! `gamma`; the study's sensitivity is the `gamma` at which the upper bound
//! crosses the significance level.
//!
//! # Matched pairs: the signed rank test
//!
//! For pair differences `d_i` (treated minus control), with zero differences
//! left out and `r_i` the average ranks of `|d_i|`, the statistic is
//! `T = sum r_i 1[d_i > 0]`. Each pair is positive with probability at most
//! `p+ = gamma / (1 + gamma)` and at least `p- = 1 / (1 + gamma)`. The bounds
//! on the one-sided p-value of `T` are the upper tails of `sum r_i B_i` with
//! `B_i` Bernoulli(`p-`), the smaller, and Bernoulli(`p+`), the larger, by the
//! normal approximation with mean `p sum r_i` and variance
//! `p (1 - p) sum r_i^2` ([`WilcoxonDistribution::Normal`]) or exactly
//! ([`WilcoxonDistribution::Exact`], by [`crate::signed_rank`]).
//!
//! # Matched pairs: the Hodges-Lehmann estimate
//!
//! For a shift `tau` the signed rank statistic of `d_i - tau` is
//! `T(tau)`, which does not increase in `tau`. With `n` pairs, the point
//! estimate of the shift is where `T` crosses `n (n + 1) / 4`, the median of
//! the Walsh averages `(d_i + d_j) / 2`. Under `gamma` the expectation of `T`
//! lies between `n (n + 1) / (2 (1 + gamma))` and `gamma n (n + 1) / (2
//! (1 + gamma))`. The bounds of the estimate are the shifts where `T` crosses
//! these. The upper bound is the smallest `tau` with `T(tau)` at most the
//! lower expectation, and the lower bound the largest `tau` with `T(tau)` at
//! least the upper one. `T` is a step function that changes only at Walsh
//! averages, so the bounds are Walsh averages. Zero differences stay in, at the
//! lowest ranks.
//!
//! ## Algorithm
//!
//! The Walsh averages are never listed. With the differences sorted, the
//! averages form a matrix `w[i][j] = (d_i + d_j) / 2`, `i <= j`, whose rows and
//! columns do not decrease. A bound is the first Walsh average `v` for which
//! `T` evaluated just above `v` (halfway to the next larger average) has
//! crossed its threshold. `T` does not increase in `tau`, so the crossing is
//! found as a quickselect does it, in the manner of Johnson and Mizoguchi
//! \[3\] and Monahan \[4\]. Two values bracket the answer, and per row the
//! columns whose averages lie between them. A random average between them is
//! drawn, the number of averages at most it in each row is found by a pointer
//! that only moves one way along the sorted differences, `T` is evaluated just
//! above it, and the bracket narrows to one side. One step costs `O(n)`: the
//! row pointers, the next larger average (the least entry just past the pointer
//! of each row), and `T` itself, which is a merge of the shifted differences
//! below and above `tau`, already in order of their absolute values. The
//! expected number of steps is `O(log n)` because the number of averages in
//! the bracket falls by a constant factor in expectation, so a bound at one
//! `gamma` costs expected `O(n log n)` time, after one `O(n log n)` sort, and
//! `O(n)` memory. The number of pairs is not limited. The random draws come
//! from a fixed generator, so a call gives the same result every time.
//!
//! `T` is a sum of ranks that are multiples of one half, which is exact in
//! floating point while `n (n + 1) / 2` is below `2^52`, so the order in which
//! ranks are added does not matter.
//!
//! # Matched sets: m-statistics
//!
//! For matched sets of one treated unit and one or more controls, the
//! statistic is the sum over sets of a score of the treated unit \[2\].
//! In a set with responses `y_1 .. y_m` (the treated unit first) the score of
//! unit `j` is
//!
//! ```text
//! q_j = (1 / m) sum_{k != j} psi((y_j - y_k) / h)
//! ```
//!
//! with `psi` the Huber-type function with inner trim `a` and outer trim `b`,
//! `psi(x) = sign(x) min(1, max(0, |x| - a) / (b - a))`, and `h` the
//! `lambda` quantile of all `|y_j - y_k|`, `lambda = 1/2` by default (the
//! scores are the raw differences when there is no trim at all, `b` infinite
//! and `a = 0`; with an infinite `b` and `a > 0`, `psi` is soft thresholding,
//! and with `a = b` it is the sign of the differences larger than `a`).
//! Under `gamma`, the unit in a set that is treated has
//! probability proportional to `gamma` for the units with the larger scores,
//! so with the scores of a set sorted, the expectation of the treated unit's
//! score is largest when the `a` lowest scores have weight 1 and the rest
//! `gamma`, for the `a` that maximises it. The upper bounds on the expectation
//! and the variance, summed over sets, standardise `T`, and the p-value
//! bound is the upper normal tail of the deviate.
//!
//! The scores can instead be divided by `m - 1` and by the number of sets
//! (`treated_on_treated`), the form of the "treatment-on-the-treated"
//! statistic with `b` infinite, which is the mean difference between the
//! treated unit and its controls.
//!
//! # Validation
//!
//! Checked against `rbounds::psens` and `hlsens` 2.2, `sensitivitymv::senmv`
//! 1.4.4 and `sensitivitymw::senmw` 2.1 in `epistat-matching`'s
//! `r_sensitivity` test. `psens` rounds its p-values to four digits and
//! `hlsens` finds its bounds by steps of its argument `pr`, which the
//! comparisons allow for. Weighting of sets by the range of their scores
//! (`method` `"w"`, `"q"`, `"s"`, `"l"`) is not implemented.
//!
//! # References
//!
//! 1. Rosenbaum PR. Observational Studies, 2nd ed. Springer; 2002. Chapter 4.
//! 2. Rosenbaum PR. Sensitivity analysis for m-estimates, tests, and
//!    confidence intervals in matched observational studies. Biometrics.
//!    2007;63(2):456-64.
//! 3. Johnson DB, Mizoguchi T. Selecting the K-th element in X + Y and
//!    X1 + X2 + ... + Xm. SIAM Journal on Computing. 1978;7(2):147-53.
//! 4. Monahan JF. Algorithm 616: fast computation of the Hodges-Lehmann
//!    location estimator. ACM Transactions on Mathematical Software.
//!    1984;10(3):265-70.

use std::error::Error;
use std::fmt::{Display, Formatter};

use crate::normal;
use crate::quantile::type7_unsorted;
use crate::signed_rank::{self, SignedRankError, average_ranks};

/// What goes wrong in a sensitivity analysis.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
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
                "the scale of the differences within sets is zero; increase lambda"
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
pub enum WilcoxonDistribution {
    /// The normal approximation.
    Normal,
    /// The exact distribution, for up to a few hundred pairs.
    Exact,
}

/// The bounds of the signed rank test at one `gamma`.
#[derive(Debug, Clone, Copy, PartialEq)]
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
pub fn wilcoxon_bounds(
    differences: &[f64],
    gamma: f64,
    distribution: WilcoxonDistribution,
) -> Result<WilcoxonBounds, SensitivityError> {
    check_gamma(gamma)?;
    if differences.iter().any(|d| !d.is_finite()) {
        return Err(SensitivityError::NotFinite);
    }
    let kept: Vec<f64> = differences.iter().copied().filter(|d| *d != 0.0).collect();
    if kept.is_empty() {
        return Err(SensitivityError::NoInformation);
    }
    let absolute: Vec<f64> = kept.iter().map(|d| d.abs()).collect();
    let ranks = average_ranks(&absolute)?;
    let statistic: f64 = ranks
        .iter()
        .zip(&kept)
        .filter(|(_, d)| **d > 0.0)
        .map(|(r, _)| *r)
        .sum();
    let p_plus = gamma / (1.0 + gamma);
    let p_minus = 1.0 / (1.0 + gamma);
    let tail = |p: f64| -> Result<f64, SensitivityError> {
        match distribution {
            WilcoxonDistribution::Normal => signed_rank::normal_upper_tail(&ranks, statistic, p)
                .ok_or(SensitivityError::ZeroVariance),
            WilcoxonDistribution::Exact => Ok(signed_rank::exact_upper_tail(&ranks, statistic, p)?),
        }
    };
    Ok(WilcoxonBounds {
        gamma,
        pairs: kept.len(),
        statistic,
        p_lower: tail(p_minus)?,
        p_upper: tail(p_plus)?,
    })
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
        // SplitMix64.
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
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
                let mut draw = (self.next_random() % active as u64) as usize;
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
pub fn matched_sets_bounds(
    sets: &[Vec<f64>],
    gamma: f64,
    spec: &MScore,
) -> Result<SetBounds, SensitivityError> {
    set_bounds(&m_scores(sets, spec)?, gamma)
}

#[cfg(test)]
mod tests {
    use super::*;


    /// The bounds from all the Walsh averages, sorted, bisected one by one.
    fn reference(differences: &[f64], gammas: &[f64]) -> Vec<HodgesLehmannBounds> {
        let shifted = |tau: f64| -> f64 {
            let s: Vec<f64> = differences.iter().map(|d| d - tau).collect();
            let a: Vec<f64> = s.iter().map(|d| d.abs()).collect();
            let ranks = average_ranks(&a).unwrap();
            ranks
                .iter()
                .zip(&s)
                .filter(|(_, d)| **d > 0.0)
                .map(|(r, _)| *r)
                .sum()
        };
        let n = differences.len();
        let mut walsh = Vec::new();
        for i in 0..n {
            for j in i..n {
                walsh.push(differences[i].midpoint(differences[j]));
            }
        }
        walsh.sort_by(f64::total_cmp);
        walsh.dedup();
        let k = walsh.len();
        let representative = |m: usize| -> f64 {
            if m == 0 {
                walsh[0] - 1.0
            } else if m == k {
                walsh[k - 1] + 1.0
            } else {
                walsh[m - 1] + (walsh[m] - walsh[m - 1]) / 2.0
            }
        };
        let total = (n * (n + 1)) as f64 / 2.0;
        let mut out = Vec::new();
        for &gamma in gammas {
            let t_low = total / (1.0 + gamma);
            let t_high = total * gamma / (1.0 + gamma);
            let (mut lo, mut hi) = (0, k);
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if shifted(representative(mid)) <= t_low {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            let upper = if lo == 0 { walsh[0] } else { walsh[lo - 1] };
            let (mut lo, mut hi) = (0, k);
            while lo < hi {
                let mid = lo + (hi - lo).div_ceil(2);
                if shifted(representative(mid)) >= t_high {
                    lo = mid;
                } else {
                    hi = mid - 1;
                }
            }
            let lower = if lo == k { walsh[k - 1] } else { walsh[lo] };
            out.push(HodgesLehmannBounds {
                gamma,
                lower,
                upper,
            });
        }
        out
    }

    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0 >> 33
        }
    }

    #[test]
    fn hodges_lehmann_equals_the_listing_of_all_averages_to_the_bit() {
        let gammas = [1.0, 1.1, 1.5, 2.0, 3.7, 9.0];
        let mut rng = Lcg(12345);
        for case in 0..400 {
            let n = 1 + (rng.next() % 120) as usize;
            let kind = case % 4;
            let d: Vec<f64> = (0..n)
                .map(|_| match kind {
                    // Continuous.
                    0 => (rng.next() % 1_000_000) as f64 / 1000.0 - 400.0,
                    // Small integers, many ties and zeros.
                    1 => (rng.next() % 9) as f64 - 3.0,
                    // Halves and negative zeros.
                    2 => {
                        let v = (rng.next() % 7) as f64 / 2.0 - 1.5;
                        if v == 0.0 && rng.next() % 2 == 0 { -0.0 } else { v }
                    }
                    // Mostly positive.
                    _ => (rng.next() % 100) as f64 / 10.0 - 1.0,
                })
                .collect();
            let want = reference(&d, &gammas);
            let got = hodges_lehmann_bounds(&d, &gammas).unwrap();
            for (w, g) in want.iter().zip(&got) {
                assert_eq!(w.lower.to_bits(), g.lower.to_bits(), "case {case} lower");
                assert_eq!(w.upper.to_bits(), g.upper.to_bits(), "case {case} upper");
            }
        }
    }

    #[test]
    fn hodges_lehmann_takes_more_than_five_thousand_pairs() {
        let mut rng = Lcg(7);
        let d: Vec<f64> = (0..40_000)
            .map(|_| (rng.next() % 2_000_001) as f64 / 1000.0 - 900.0)
            .collect();
        let b = hodges_lehmann_bounds(&d, &[1.0, 2.0]).unwrap();
        assert!(b[0].lower <= b[0].upper);
        assert!(b[1].lower <= b[0].lower && b[0].upper <= b[1].upper);
    }

    #[test]
    fn gamma_one_signed_rank_bounds_coincide() {
        let d = [1.2, -0.4, 2.2, 0.7, 3.1, -1.5, 0.9];
        let b = wilcoxon_bounds(&d, 1.0, WilcoxonDistribution::Normal).unwrap();
        assert!((b.p_lower - b.p_upper).abs() < 1e-15);
        let e = wilcoxon_bounds(&d, 1.0, WilcoxonDistribution::Exact).unwrap();
        assert!((e.p_lower - e.p_upper).abs() < 1e-15);
    }

    #[test]
    fn bounds_widen_with_gamma() {
        let d = [1.2, -0.4, 2.2, 0.7, 3.1, -1.5, 0.9, 2.4, 1.1];
        let a = wilcoxon_bounds(&d, 1.5, WilcoxonDistribution::Exact).unwrap();
        let b = wilcoxon_bounds(&d, 3.0, WilcoxonDistribution::Exact).unwrap();
        assert!(b.p_upper > a.p_upper);
        assert!(b.p_lower < a.p_lower);
    }

    #[test]
    fn hodges_lehmann_at_gamma_one_is_the_median_walsh_average() {
        let d = [1.0, 2.0, 4.0, 7.0, 8.0];
        let b = hodges_lehmann_bounds(&d, &[1.0]).unwrap();
        // The 15 Walsh averages have median 4.5.
        assert!(b[0].lower <= 4.5 && 4.5 <= b[0].upper);
    }

    #[test]
    fn hodges_lehmann_bounds_are_ordered_and_widen() {
        let d = [0.3, 1.2, -0.5, 2.0, 1.7, 0.9, 2.6, 0.1];
        let b = hodges_lehmann_bounds(&d, &[1.0, 1.5, 2.0]).unwrap();
        assert!(b[1].lower <= b[0].lower && b[0].upper <= b[1].upper);
        assert!(b[2].lower <= b[1].lower && b[1].upper <= b[2].upper);
    }

    #[test]
    fn set_bound_at_gamma_one_has_the_plain_expectation() {
        let scores = vec![vec![0.5, -0.1, -0.4], vec![0.2, 0.3, -0.5]];
        let b = set_bounds(&scores, 1.0).unwrap();
        assert!(b.expectation.abs() < 1e-12);
    }

    #[test]
    fn invalid_input_is_an_error() {
        assert!(wilcoxon_bounds(&[1.0], 0.5, WilcoxonDistribution::Normal).is_err());
        assert!(wilcoxon_bounds(&[0.0, 0.0], 1.0, WilcoxonDistribution::Normal).is_err());
        assert!(m_scores(&[vec![1.0]], &MScore::default()).is_err());
        assert!(m_scores(&[vec![1.0, 1.0], vec![2.0, 2.0]], &MScore::default()).is_err());
    }

    #[test]
    fn hodges_lehmann_with_overflowing_shift_is_an_error() {
        let d = [1.7e308, -1.7e308, 1.0e308, -1.0e308, 5.0e307];
        assert!(hodges_lehmann_bounds(&d, &[1.0, 2.0]).is_err());
    }
}
