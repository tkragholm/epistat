//! The balance measures that rank a set of weights, GenMatch's fitness
//! functions.
//!
//! After a match for given weights, each balance covariate is compared between
//! the focal units and their matched controls, one row per matched pair:
//! the rows of a focal unit matched to several controls repeat it, and a
//! control used twice appears twice. The default measure is the set of p-values
//! of a paired t-test and a Kolmogorov-Smirnov test on every covariate, ranked
//! by the smallest first. The alternatives are summaries of the differences
//! between the empirical quantile functions of the two groups.

use std::cmp::Ordering;

use epistat_rng::Stream;
use epistat_stats::balance::ecdf_gaps_sorted;
use epistat_stats::student::t_cdf_sf;

use super::ks;

/// How the balance after a match is measured. The names follow `fit.func` of
/// `GenMatch`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BalanceMeasure {
    /// The p-values of tests on every covariate, to be made as large as
    /// possible, the smallest first. This is `fit.func = "pvals"`.
    #[default]
    PValues,
    /// The mean over covariates of the mean absolute difference between the
    /// empirical distribution functions, `"qqmean.mean"`.
    QqMeanMean,
    /// The mean differences of every covariate, largest first, `"qqmean.max"`.
    QqMeanMax,
    /// The mean over covariates of the largest difference, `"qqmax.mean"`.
    QqMaxMean,
    /// The largest differences of every covariate, largest first,
    /// `"qqmax.max"`.
    QqMaxMax,
    /// The median over covariates of the median difference,
    /// `"qqmedian.median"`.
    QqMedianMedian,
    /// The median differences of every covariate, largest first,
    /// `"qqmedian.max"`.
    QqMedianMax,
}

/// What the p-values are reduced to, GenMatch's `loss`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LossRule {
    /// All p-values, sorted from smallest to largest and compared from the
    /// first. `loss = 1`.
    #[default]
    Sorted,
    /// Only the smallest p-value. `loss = 2`.
    Minimum,
}

/// The balance measure, and the choices that go with it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GeneticFit {
    /// The measure.
    pub measure: BalanceMeasure,
    /// The reduction of the p-values. Read for [`BalanceMeasure::PValues`]
    /// only.
    pub loss: LossRule,
    /// Whether the p-values include a Kolmogorov-Smirnov test per covariate.
    pub ks: bool,
    /// A paired t-test when true, as `paired = TRUE` of `GenMatch`, and
    /// Welch's unpaired test otherwise.
    pub paired: bool,
    /// Bootstrap replicates of the Kolmogorov-Smirnov p-value. With fewer than
    /// 10 the p-value of the test itself is used, which is exact for small
    /// samples and asymptotic for large ones, as `ks.test` has it.
    pub bootstrap: usize,
}

impl Default for GeneticFit {
    fn default() -> Self {
        Self {
            measure: BalanceMeasure::PValues,
            loss: LossRule::Sorted,
            ks: true,
            paired: true,
            bootstrap: 0,
        }
    }
}

/// The value of a fitness function. It is a list of numbers compared from the
/// first, and either as large as possible or as small as possible.
#[derive(Debug, Clone, PartialEq)]
pub struct GeneticLoss {
    values: Vec<f64>,
    maximise: bool,
}

impl GeneticLoss {
    pub(super) fn new(values: Vec<f64>, maximise: bool) -> Self {
        Self { values, maximise }
    }

    /// The numbers, most important first.
    #[must_use]
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Whether a larger list is better.
    #[must_use]
    pub const fn maximise(&self) -> bool {
        self.maximise
    }

    /// `Greater` when `self` is better than `other`. Two values are equal
    /// when they differ by at most `tolerance`, so a list is better at its
    /// first entry that differs by more.
    #[must_use]
    pub fn compare(&self, other: &Self, tolerance: f64) -> Ordering {
        for (a, b) in self.values.iter().zip(&other.values) {
            if (a - b).abs() > tolerance {
                let larger = a.total_cmp(b);
                return if self.maximise {
                    larger
                } else {
                    larger.reverse()
                };
            }
        }
        Ordering::Equal
    }
}

/// The rows of a match, as the focal and control values of one covariate.
pub(super) struct MatchedRows<'a> {
    pub(super) treated: &'a [f64],
    pub(super) control: &'a [f64],
    pub(super) weight: &'a [f64],
}

/// The p-value of a paired t-test of the weighted differences. `None` when
/// the degrees of freedom `sum(weight) - 1` are not positive.
fn paired_t_pvalue(rows: &MatchedRows<'_>) -> Option<f64> {
    let total: f64 = rows.weight.iter().sum();
    let df = total - 1.0;
    if df.partial_cmp(&0.0) != Some(Ordering::Greater) {
        return None;
    }
    let mut weighted = 0.0;
    for ((t, c), w) in rows.treated.iter().zip(rows.control).zip(rows.weight) {
        weighted += (t - c) * w;
    }
    let estimate = weighted / total;
    let mut spread = 0.0;
    for ((t, c), w) in rows.treated.iter().zip(rows.control).zip(rows.weight) {
        let centred = (t - c) - estimate;
        spread += centred * centred * w;
    }
    let variance = spread / (total * total);
    if estimate == 0.0 && variance == 0.0 {
        return Some(1.0);
    }
    if variance == 0.0 {
        return Some(0.0);
    }
    let statistic = (estimate / variance.sqrt()).abs();
    Some(2.0 * t_cdf_sf(statistic, df).1)
}

/// The p-value of Welch's test on the weighted groups, as `GenMatch` computes
/// it when the rows are not paired.
fn unpaired_t_pvalue(rows: &MatchedRows<'_>) -> Option<f64> {
    let obs: f64 = rows.weight.iter().sum();
    if (obs - 1.0).partial_cmp(&0.0) != Some(Ordering::Greater) {
        return None;
    }
    let mut mean_t = 0.0;
    let mut mean_c = 0.0;
    for ((t, c), w) in rows.treated.iter().zip(rows.control).zip(rows.weight) {
        mean_t += t * w;
        mean_c += c * w;
    }
    mean_t /= obs;
    mean_c /= obs;
    let mut var_t = 0.0;
    let mut var_c = 0.0;
    for ((t, c), w) in rows.treated.iter().zip(rows.control).zip(rows.weight) {
        var_t += (t - mean_t) * (t - mean_t) * w;
        var_c += (c - mean_c) * (c - mean_c) * w;
    }
    var_t /= obs - 1.0;
    var_c /= obs - 1.0;
    let estimate = mean_t - mean_c;
    let spread = (var_t / obs + var_c / obs).sqrt();
    if estimate == 0.0 && spread == 0.0 {
        return Some(1.0);
    }
    if spread == 0.0 {
        return Some(0.0);
    }
    let a1 = var_t / obs;
    let a2 = var_c / obs;
    let dof = (a1 + a2) * (a1 + a2) / (a1 * a1 / (obs - 1.0) + a2 * a2 / (obs - 1.0));
    Some(2.0 * t_cdf_sf((estimate / spread).abs(), dof).1)
}

/// The bootstrap p-value of the Kolmogorov-Smirnov statistic: the share of
/// resamples of the pooled values, split in two halves, whose statistic is at
/// least the observed one less a rounding margin.
fn ks_bootstrap(treated: &[f64], control: &[f64], replicates: usize, stream: &mut Stream) -> f64 {
    let observed = ks::ks_statistic(treated, control).unwrap_or(0.0);
    let pooled: Vec<f64> = treated.iter().chain(control).copied().collect();
    let obs = pooled.len();
    let cut = obs.div_ceil(2);
    let margin = f64::EPSILON * 100.0;
    let mut first = vec![0.0; cut];
    let mut second = vec![0.0; obs - cut];
    let mut count = 0_usize;
    for _ in 0..replicates {
        for slot in &mut first {
            *slot = pooled[stream.below(obs)];
        }
        for slot in &mut second {
            *slot = pooled[stream.below(obs)];
        }
        first.sort_by(f64::total_cmp);
        second.sort_by(f64::total_cmp);
        if ks::ks_d_sorted(&first, &second) >= observed - margin {
            count += 1;
        }
    }
    count as f64 / replicates as f64
}

/// The summaries of the absolute differences between the two empirical
/// distribution functions on the pooled distinct values: mean, median and
/// maximum.
fn qq_summaries(treated: &[f64], control: &[f64]) -> (f64, f64, f64) {
    let mut xs = treated.to_vec();
    let mut ys = control.to_vec();
    xs.sort_by(f64::total_cmp);
    ys.sort_by(f64::total_cmp);
    let mut diffs: Vec<f64> = ecdf_gaps_sorted(&xs, &ys).collect();
    let mut sum = 0.0;
    for d in &diffs {
        sum += d;
    }
    let mean = sum / diffs.len() as f64;
    let max = diffs.iter().copied().fold(0.0, f64::max);
    let median = median_of(&mut diffs);
    (mean, median, max)
}

/// The median, the mean of the two middle values for an even count. Sorts
/// `values`.
fn median_of(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let half = values.len() / 2;
    if values.len() % 2 == 1 {
        values[half]
    } else {
        f64::midpoint(values[half - 1], values[half])
    }
}

/// The loss of a match. `columns[k]` holds the treated and control values of
/// balance covariate `k` on the matched rows, `few_values[k]` whether the
/// covariate takes fewer than three distinct values in the data, and
/// `stream` supplies the bootstrap draws. `None` when the rows cannot give a
/// t-test.
pub(super) fn loss_of_rows(
    fit: &GeneticFit,
    columns: &[(Vec<f64>, Vec<f64>)],
    weight: &[f64],
    few_values: &[bool],
    stream: &mut Stream,
) -> Option<GeneticLoss> {
    match fit.measure {
        BalanceMeasure::PValues => {
            let mut t_values = Vec::with_capacity(columns.len());
            let mut k_values = Vec::with_capacity(columns.len());
            for (k, (treated, control)) in columns.iter().enumerate() {
                let rows = MatchedRows {
                    treated,
                    control,
                    weight,
                };
                let t = if fit.paired {
                    paired_t_pvalue(&rows)?
                } else {
                    unpaired_t_pvalue(&rows)?
                };
                t_values.push(t);
                if fit.ks {
                    let value = if few_values[k] {
                        t
                    } else if fit.bootstrap > 9 {
                        ks_bootstrap(treated, control, fit.bootstrap, stream)
                    } else {
                        ks::ks_test(treated, control)?.1
                    };
                    k_values.push(value);
                }
            }
            let mut all = t_values;
            all.extend(k_values);
            match fit.loss {
                LossRule::Sorted => all.sort_by(f64::total_cmp),
                LossRule::Minimum => {
                    let smallest = all.iter().copied().fold(f64::INFINITY, f64::min);
                    all = vec![smallest];
                }
            }
            Some(GeneticLoss::new(all, true))
        }
        measure => {
            let mut means = Vec::with_capacity(columns.len());
            let mut medians = Vec::with_capacity(columns.len());
            let mut maxes = Vec::with_capacity(columns.len());
            for (treated, control) in columns {
                let (mean, median, max) = qq_summaries(treated, control);
                means.push(mean);
                medians.push(median);
                maxes.push(max);
            }
            let descending = |mut v: Vec<f64>| {
                v.sort_by(|a, b| b.total_cmp(a));
                v
            };
            let mean_of = |v: &[f64]| {
                let mut sum = 0.0;
                for x in v {
                    sum += x;
                }
                sum / v.len() as f64
            };
            let values = match measure {
                BalanceMeasure::QqMeanMean => vec![mean_of(&means)],
                BalanceMeasure::QqMeanMax => descending(means),
                BalanceMeasure::QqMaxMean => vec![mean_of(&maxes)],
                BalanceMeasure::QqMaxMax => descending(maxes),
                BalanceMeasure::QqMedianMedian => vec![median_of(&mut medians)],
                BalanceMeasure::QqMedianMax => descending(medians),
                BalanceMeasure::PValues => unreachable!("handled above"),
            };
            Some(GeneticLoss::new(values, false))
        }
    }
}

/// The loss when no unit could be matched, or too few for a t-test: worse
/// than any loss a match can have.
pub(super) fn worst_loss(fit: &GeneticFit, columns: usize) -> GeneticLoss {
    match fit.measure {
        BalanceMeasure::PValues => {
            let length = match fit.loss {
                LossRule::Minimum => 1,
                LossRule::Sorted => columns * (1 + usize::from(fit.ks)),
            };
            GeneticLoss::new(vec![f64::NEG_INFINITY; length], true)
        }
        BalanceMeasure::QqMeanMean | BalanceMeasure::QqMaxMean | BalanceMeasure::QqMedianMedian => {
            GeneticLoss::new(vec![f64::INFINITY], false)
        }
        BalanceMeasure::QqMeanMax | BalanceMeasure::QqMaxMax | BalanceMeasure::QqMedianMax => {
            GeneticLoss::new(vec![f64::INFINITY; columns], false)
        }
    }
}
