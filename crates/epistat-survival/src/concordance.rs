//! Concordance of a numeric score with a survival outcome, with case weights,
//! strata, delayed entry, time weights and an infinitesimal jackknife
//! variance, in `O(n log n)`.
//!
//! [`estimate`] takes a [`SurvData`] and one score per row and returns the
//! weighted counts of concordant, discordant and tied pairs, the concordance,
//! and two variances. It reports what `survival::concordance` reports for a
//! single numeric predictor.
//!
//! # Orientation
//!
//! With [`ConcordanceOptions::reverse`] false, a larger score predicts a
//! longer survival, as for a linear model or a parametric survival model. A
//! Cox linear predictor is the other way round, so pass `reverse: true` for
//! it. Reversing swaps the concordant and discordant counts, turns the
//! concordance `c` into `1 - c` and leaves both variances unchanged.
//!
//! # Pairs
//!
//! Row `i` is a death at time `t_i` when its status is 1. A pair is
//! comparable when one row is a death at `t` and the other is at risk at `t`:
//! `start < t <= stop`, with `start` absent meaning at risk from the
//! beginning. A censoring at `t` is at risk at `t`, so a death comes before a
//! censoring at the same time. Two deaths at the same time are not
//! comparable. Pairs are formed within strata only. For a comparable pair with
//! scores `x_i` of the death and `x_j` of the other row, and `v(t)` the time
//! weight below,
//!
//! ```text
//! concordant   x_j > x_i      the death had the smaller score
//! discordant   x_j < x_i
//! tied on x    x_j = x_i
//! ```
//!
//! and each pair counts `w_i w_j v(t)`. Two deaths at one time with different
//! scores count `w_i w_j v(t)` as tied on y and with equal scores as tied on
//! both. The concordance is
//!
//! ```text
//! c = (1 + (C - D) / (C + D + T)) / 2
//! ```
//!
//! for the concordant, discordant and tied-on-x totals `C`, `D` and `T`, so a
//! pair tied on x counts one half. It is `None` when no pair is comparable.
//!
//! # Time weights
//!
//! [`TimeWeight`] sets `v(t)` for the event times of a stratum. `n(t)` is the
//! weighted number at risk, `S(t-)` the weighted Kaplan-Meier estimate of
//! survival and `G(t-)` the one of censoring, each just before `t`, and `W`
//! the weight of the stratum. Censorings follow deaths at a tied time, so
//! `G(t-)` counts only censorings strictly before `t`, and the risk set of
//! a censoring at `t` is the rows with a later time and those censored at `t`.
//!
//! ```text
//! N        1                          Harrell
//! S        W S(t-) / n(t)             Peto-Wilcoxon
//! S/G      W S(t-) / (G(t-) n(t))     Schemper
//! N/G2     1 / G(t-)^2                Uno
//! I        1 / n(t)                   log-rank
//! ```
//!
//! With `n(t) = W S(t-) G(t-)`, which holds exactly for right-censored data,
//! `S/G` and `N/G2` are the same weight. They need a censoring distribution,
//! which is not defined with delayed entry, so they are refused for data with
//! a start column. A stratum with fewer than two deaths uses `N`, since a time
//! weight cannot change a stratum with one event time and R makes the same
//! choice. [`ConcordanceOptions::ymax`] gives weight 0 to the event times
//! above it, and [`ConcordanceOptions::ymin`] raises every stop time below it
//! to it before anything else, as `survival` does.
//!
//! # Algorithm
//!
//! The scores of a stratum are ranked, and two Fenwick trees over the ranks
//! hold the weights of the rows at risk and the time-weighted weights of the
//! deaths seen so far. The rows are swept from the latest stop time to the
//! earliest. A death at `t` asks the first tree for the weight at risk above,
//! below and equal to its rank, which are its concordant, discordant and tied
//! partners, and then enters the tree. A row with a start time leaves the tree
//! when the sweep passes its start. Every step costs `O(log n)`, so a stratum
//! of `n` rows costs `O(n log n)` time and `O(n)` memory, and no pair is
//! enumerated.
//!
//! # Variance
//!
//! The infinitesimal jackknife variance treats the time weights as fixed. Let
//! `a_k`, `b_k` and `e_k` be the sums, over the comparable pairs of row `k`
//! that are concordant, discordant and tied on x, of `w v` of the other row,
//! which are the derivatives of `C`, `D` and `T` with respect to `w_k`. With
//! `s = (C - D) / (C + D + T)` and `P = C + D + T`, the influence of row `k`
//! on `c` is
//!
//! ```text
//! dfbeta_k = w_k ( (1 - s) a_k - (1 + s) b_k - s e_k ) / (2 P)
//! ```
//!
//! and the variance is the sum of `dfbeta_k^2`, over clusters first when
//! the data has them. The two sums per row of the second tree give `a_k`,
//! `b_k` and `e_k` in the same sweep: row `k` as the death takes them from the
//! first tree, and as the other row of a pair it takes them from the second
//! tree, over the deaths between its start and its stop. They cost 24 bytes per
//! used row. The other variance, [`Concordance::cox_variance`], is the
//! variance of the score statistic of a Cox model with the rank as a
//! time-dependent covariate: the sum over event times of the time-weighted
//! variance of the mid-ranks in the risk set, divided by `4 P^2`. It
//! ignores that the weights of rows are estimated and is the older of the two.
//! The risk-set variance is kept as a running sum that is updated when a row
//! enters or leaves the tree: a row of weight `w` entering a set of weight `T`
//! of which `e` has its score adds `w (T - e) (T + e + w)`.
//!
//! # Arithmetic and tolerance
//!
//! Sums run in a fixed order: the sweep is sequential within a stratum, the
//! strata run in parallel and their results are added in stratum order, and
//! rows that tie on every sort key are ordered by row number. The result is
//! the same on any number of threads. A weight sum held in a tree is a sum of
//! `n` entering and leaving terms, so a count differs from the exact value by a
//! few `n` machine epsilons of the largest pair weight sum `W^2` at most. With
//! integer weights every sum is exact below `2^53`, and so are the counts when
//! `v = 1`. The risk-set variance subtracts when a row leaves, so with delayed
//! entry its error is relative to the largest value the running sum took.
//!
//! # Differences from `survival`
//!
//! - Rows of weight 0 are dropped before anything else. `survival` keeps
//!   them, which changes nothing but the count of deaths that decides whether a
//!   stratum falls back to `N`.
//! - A non-finite score is an error, and an undefined concordance or variance
//!   is `None`, where `survival` returns `NaN`.
//! - Only one numeric score is taken, per call. Several scores are several
//!   calls.
//! - The result carries per-stratum counts always, where `survival` keeps them
//!   up to a limit.
//! - The `ranks` residuals are not computed.
//!
//! # References
//!
//! 1. Harrell FE, Califf RM, Pryor DB, Lee KL, Rosati RA. Evaluating the yield
//!    of medical tests. JAMA 1982;247(18):2543-6.
//!    doi:10.1001/jama.1982.03320430047030
//! 2. Therneau TM, Watson DD. The concordance statistic. Vignette of the R
//!    package survival, 2024.
//! 3. Uno H, Cai T, Pencina MJ, D'Agostino RB, Wei LJ. On the C-statistics for
//!    evaluating overall adequacy of risk prediction procedures with censored
//!    survival data. Stat Med 2011;30(10):1105-17. doi:10.1002/sim.4154
//! 4. Schemper M, Wakounig S, Heinze G. The estimation of average hazard
//!    ratios by weighted Cox regression. Stat Med 2009;28(19):2473-89.
//!    doi:10.1002/sim.3623
//! 5. Fenwick PM. A new data structure for cumulative frequency tables.
//!    Software: Practice and Experience 1994;24(3):327-36.
//!    doi:10.1002/spe.4380240306

#![expect(
    clippy::cast_possible_truncation,
    reason = "the data was checked to have no more rows than fit in u32"
)]
#![expect(
    clippy::float_cmp,
    reason = "times and scores tie when they are equal, which is the definition of a tie"
)]

use std::cmp::Ordering;

use epistat_stats::sum::compensated;
use rayon::prelude::*;

use crate::check::{check_lengths, refuse_non_finite};
use crate::error::{Error, Result};
use crate::surv::SurvData;
use crate::timefix::TimeFix;

/// How the event times are weighted. See the module documentation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeWeight {
    /// Every event time counts the same: Harrell's C.
    N,
    /// `W S(t-) / n(t)`: the Peto-Wilcoxon weight.
    S,
    /// `W S(t-) / (G(t-) n(t))`: Schemper's weight. Right-censored data only.
    SOverG,
    /// `1 / G(t-)^2`: Uno's weight. Right-censored data only.
    NOverG2,
    /// `1 / n(t)`: the log-rank weight.
    I,
}

/// What [`estimate`] takes besides the data and the score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConcordanceOptions {
    /// The rule that makes times equal when they differ by rounding error,
    /// applied to the data first. The default is that of `survival::aeqSurv`,
    /// which `concordance` applies.
    pub timefix: TimeFix,
    /// The weight of each event time. `N`, as in `concordance`.
    pub time_weight: TimeWeight,
    /// Whether a larger score predicts a shorter survival, as a Cox linear
    /// predictor does. False, as in `concordance`, takes a larger score to
    /// predict a longer survival.
    pub reverse: bool,
    /// Raise every stop time below this to it, as `ymin` does.
    pub ymin: Option<f64>,
    /// Give weight 0 to the event times above this, as `ymax` does.
    pub ymax: Option<f64>,
    /// Compute the two variances. Costs 24 bytes per used row.
    pub variance: bool,
    /// Return the influence of each row on the concordance, `dfbeta` of
    /// `concordance`. Needs `variance`.
    pub influence: bool,
}

impl Default for ConcordanceOptions {
    fn default() -> Self {
        Self {
            timefix: TimeFix::default(),
            time_weight: TimeWeight::N,
            reverse: false,
            ymin: None,
            ymax: None,
            variance: true,
            influence: false,
        }
    }
}

/// Weighted counts of pairs, as `concordance` reports them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PairCounts {
    /// Pairs where the death had the smaller score, or the larger when the
    /// orientation is reversed.
    pub concordant: f64,
    /// Pairs where the death had the larger score, or the smaller when the
    /// orientation is reversed.
    pub discordant: f64,
    /// Comparable pairs with equal scores.
    pub tied_x: f64,
    /// Pairs of deaths at one time with different scores.
    pub tied_y: f64,
    /// Pairs of deaths at one time with equal scores.
    pub tied_xy: f64,
}

/// The counts of one stratum, labelled by the caller's stratum code.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StratumCounts {
    /// The caller's code of the stratum.
    pub code: u32,
    /// Its counts.
    pub counts: PairCounts,
}

/// The result of [`estimate`].
#[derive(Debug, Clone, PartialEq)]
pub struct Concordance {
    /// The counts over all strata.
    pub counts: PairCounts,
    /// The concordance, `None` when no pair is comparable.
    pub concordance: Option<f64>,
    /// The infinitesimal jackknife variance of the concordance, `None` when
    /// it was not asked for or no pair is comparable.
    pub variance: Option<f64>,
    /// The variance from the Cox score statistic, `None` as `variance`.
    pub cox_variance: Option<f64>,
    /// The counts of each stratum with a used row, in the order of the
    /// caller's codes. Empty without a stratum column.
    pub strata: Vec<StratumCounts>,
    /// The influence of each row, zero for rows not used. `None` unless asked
    /// for.
    pub influence: Option<Vec<f64>>,
    /// The number of rows given.
    pub n_rows: usize,
    /// The number of rows of positive weight.
    pub n_used: usize,
}

/// The ranks, times and weights of one stratum, as contiguous columns.
struct Local {
    stop: Vec<f64>,
    start: Option<Vec<f64>>,
    death: Vec<bool>,
    weight: Vec<f64>,
    /// 1-based dense rank of the score within the stratum.
    rank: Vec<u32>,
    n_ranks: usize,
}

/// A Fenwick tree of `f64` over ranks `1..=n`.
struct Fenwick {
    tree: Vec<f64>,
}

impl Fenwick {
    fn new(n: usize) -> Self {
        Self {
            tree: vec![0.0; n + 1],
        }
    }

    fn add(&mut self, rank: usize, delta: f64) {
        let mut i = rank;
        while i < self.tree.len() {
            self.tree[i] += delta;
            i += i & i.wrapping_neg();
        }
    }

    /// The sum over ranks `1..=rank`.
    fn prefix(&self, rank: usize) -> f64 {
        let mut i = rank;
        let mut sum = 0.0;
        while i > 0 {
            sum += self.tree[i];
            i &= i - 1;
        }
        sum
    }
}

/// Weights by rank with a running total: the tree, the weight at each rank and
/// the sum.
struct Bag {
    tree: Fenwick,
    at: Vec<f64>,
    total: f64,
}

impl Bag {
    fn new(n: usize) -> Self {
        Self {
            tree: Fenwick::new(n),
            at: vec![0.0; n + 1],
            total: 0.0,
        }
    }

    fn add(&mut self, rank: usize, weight: f64) {
        self.tree.add(rank, weight);
        self.at[rank] += weight;
        self.total += weight;
    }

    /// The weight above, below and at `rank`.
    fn around(&self, rank: usize) -> (f64, f64, f64) {
        let below = self.tree.prefix(rank - 1);
        let equal = self.at[rank];
        (self.total - below - equal, below, equal)
    }
}

/// What one stratum gives: the six sums and the three partner sums per row.
struct Swept {
    counts: [f64; 6],
    partners: Option<Vec<[f64; 3]>>,
}

/// Concordance of `score` with the outcome of `data`.
///
/// `score` has one value per row of `data`. Rows of weight 0 are left out.
/// See the module documentation for the pairs, the weights and the variance.
///
/// # Errors
///
/// An input error when `score` has another length than the data or holds a
/// NaN or an infinity, when the status has more than one event code, when
/// `ymin` or `ymax` is not finite, when `influence` is asked for without
/// `variance`, when `S/G` or `N/G2` is asked for with a start column, when a
/// time weight is not finite, and as [`SurvData::with_timefix`] for the tie
/// rule.
pub fn estimate(
    data: &SurvData<'_>,
    score: &[f64],
    opts: &ConcordanceOptions,
) -> Result<Concordance> {
    check_lengths(&[("data", data.n_rows()), ("score", score.len())])?;
    refuse_non_finite(score, "score")?;
    for (name, value) in [("ymin", opts.ymin), ("ymax", opts.ymax)] {
        if value.is_some_and(|v| !v.is_finite()) {
            return Err(Error::invalid(format!("{name} must be a finite number")));
        }
    }
    if opts.influence && !opts.variance {
        return Err(Error::invalid(
            "influence: the influence of the rows is part of the variance, which is off",
        ));
    }
    let data = data.clone().with_timefix(opts.timefix)?;
    let surv = data.followup()?;
    if surv.n_codes() != 1 {
        return Err(Error::invalid(format!(
            "status has {} event codes, and concordance takes one; pass one cause as 0 and 1",
            surv.n_codes()
        )));
    }
    let counting = surv.start().is_some();
    if counting && matches!(opts.time_weight, TimeWeight::SOverG | TimeWeight::NOverG2) {
        return Err(Error::invalid(
            "time weight S/G and N/G2 need a censoring distribution, which data with delayed entry does not have",
        ));
    }

    let stop_cow;
    let stop: &[f64] = match opts.ymin {
        Some(ymin) => {
            stop_cow = surv.stop().iter().map(|&t| t.max(ymin)).collect::<Vec<_>>();
            &stop_cow
        }
        None => surv.stop(),
    };

    let tasks = stratum_rows(&data);
    let swept: Vec<Result<Swept>> = tasks
        .par_iter()
        .map(|(_, rows)| {
            let local = gather(&data, stop, score, rows);
            sweep(&local, opts)
        })
        .collect();
    let mut sweeps = Vec::with_capacity(swept.len());
    for result in swept {
        sweeps.push(result?);
    }

    let mut total = [0.0_f64; 6];
    for s in &sweeps {
        for (t, c) in total.iter_mut().zip(&s.counts) {
            *t += c;
        }
    }
    let strata = data.stratum_levels().map_or_else(Vec::new, |levels| {
        tasks
            .iter()
            .zip(&sweeps)
            .filter_map(|((level, _), s)| {
                Some(StratumCounts {
                    code: levels.code(*level)?,
                    counts: counts_of(&s.counts, opts.reverse),
                })
            })
            .collect()
    });

    let pairs = total[0] + total[1] + total[2];
    let defined = pairs > 0.0;
    let somer = (total[0] - total[1]) / pairs;
    let mut concordance = defined.then(|| f64::midpoint(somer, 1.0));
    if opts.reverse {
        concordance = concordance.map(|c| 1.0 - c);
    }
    let cox_variance = (defined && opts.variance).then(|| total[5] / (4.0 * pairs * pairs));

    let mut variance = None;
    let mut influence = None;
    if defined && opts.variance {
        let mut dfbeta = vec![0.0_f64; data.n_rows()];
        let c = [
            (1.0 - somer) / (2.0 * pairs),
            -(1.0 + somer) / (2.0 * pairs),
            -somer / (2.0 * pairs),
        ];
        for ((_, rows), s) in tasks.iter().zip(&mut sweeps) {
            let Some(partners) = s.partners.take() else {
                continue;
            };
            for (&row, p) in rows.iter().zip(&partners) {
                let w = data.weight().map_or(1.0, |w| w[row as usize]);
                dfbeta[row as usize] = w * (c[0] * p[0] + c[1] * p[1] + c[2] * p[2]);
            }
        }
        variance = Some(sum_of_squares(&data, &dfbeta));
        if opts.influence {
            if opts.reverse {
                for d in &mut dfbeta {
                    *d = -*d;
                }
            }
            influence = Some(dfbeta);
        }
    }

    Ok(Concordance {
        counts: counts_of(&total, opts.reverse),
        concordance,
        variance,
        cox_variance,
        strata,
        influence,
        n_rows: data.n_rows(),
        n_used: data.n_used(),
    })
}

fn counts_of(sums: &[f64; 6], reverse: bool) -> PairCounts {
    let (concordant, discordant) = if reverse {
        (sums[1], sums[0])
    } else {
        (sums[0], sums[1])
    };
    PairCounts {
        concordant,
        discordant,
        tied_x: sums[2],
        tied_y: sums[3],
        tied_xy: sums[4],
    }
}

/// The sum of squares of the influences, over clusters when there are any.
fn sum_of_squares(data: &SurvData<'_>, dfbeta: &[f64]) -> f64 {
    match data.cluster() {
        Some(cluster) => {
            let mut sums = vec![0.0_f64; data.n_clusters()];
            for (&c, &d) in cluster.iter().zip(dfbeta) {
                sums[c as usize] += d;
            }
            compensated(sums.iter().map(|s| s * s))
        }
        None => compensated(dfbeta.iter().map(|d| d * d)),
    }
}

/// The used rows of each stratum with a used row, by stratum level, with each
/// stratum's rows in ascending order.
fn stratum_rows(data: &SurvData<'_>) -> Vec<(usize, Vec<u32>)> {
    match data.stratum() {
        None => vec![(0, data.used_rows().collect())],
        Some(stratum) => {
            let mut counts = vec![0_usize; data.n_strata()];
            for row in data.used_rows() {
                counts[stratum[row as usize] as usize] += 1;
            }
            let mut tasks: Vec<(usize, Vec<u32>)> = counts
                .iter()
                .enumerate()
                .map(|(level, &n)| (level, Vec::with_capacity(n)))
                .collect();
            for row in data.used_rows() {
                tasks[stratum[row as usize] as usize].1.push(row);
            }
            tasks.retain(|(_, rows)| !rows.is_empty());
            tasks
        }
    }
}

/// The columns of the rows of one stratum, with the scores ranked.
fn gather(data: &SurvData<'_>, stop: &[f64], score: &[f64], rows: &[u32]) -> Local {
    let surv = data
        .followup()
        .expect("the data was checked as follow-up data");
    let pick = |column: &[f64]| rows.iter().map(|&r| column[r as usize]).collect::<Vec<_>>();
    let status = surv.status();
    let x = pick(score);
    let mut by_score: Vec<u32> = (0..rows.len() as u32).collect();
    by_score.sort_unstable_by(|&a, &b| {
        x[a as usize]
            .partial_cmp(&x[b as usize])
            .expect("scores were checked finite")
            .then(a.cmp(&b))
    });
    let mut rank = vec![0_u32; rows.len()];
    let mut current = 0_u32;
    let mut previous = None;
    for &i in &by_score {
        let value = x[i as usize];
        if previous != Some(value) {
            current += 1;
            previous = Some(value);
        }
        rank[i as usize] = current;
    }
    Local {
        stop: pick(stop),
        start: surv.start().map(pick),
        death: rows.iter().map(|&r| status[r as usize] == 1).collect(),
        weight: data.weight().map_or_else(
            || vec![1.0; rows.len()],
            |w| rows.iter().map(|&r| w[r as usize]).collect(),
        ),
        rank,
        n_ranks: current as usize,
    }
}

/// Rows from the latest stop to the earliest, censorings before deaths at a
/// tied time, then by rank and row.
fn sweep_order(local: &Local) -> Vec<u32> {
    let mut order: Vec<u32> = (0..local.stop.len() as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        let (a, b) = (a as usize, b as usize);
        local.stop[b]
            .partial_cmp(&local.stop[a])
            .unwrap_or(Ordering::Equal)
            .then(local.death[a].cmp(&local.death[b]))
            .then(local.rank[a].cmp(&local.rank[b]))
            .then(a.cmp(&b))
    });
    order
}

/// Rows from the latest start to the earliest, or `None` without start times.
fn start_order(local: &Local) -> Option<Vec<u32>> {
    local.start.as_ref().map(|start| {
        let mut by_start: Vec<u32> = (0..start.len() as u32).collect();
        by_start.sort_unstable_by(|&a, &b| {
            start[b as usize]
                .partial_cmp(&start[a as usize])
                .unwrap_or(Ordering::Equal)
                .then(a.cmp(&b))
        });
        by_start
    })
}

/// One step of the risk-set summary: a distinct stop time.
struct Step {
    time: f64,
    censored: f64,
    deaths: f64,
    /// Weight at risk before the deaths of the step enter, with the censorings.
    at_risk_censor: f64,
    /// Weight at risk at the time, deaths included.
    at_risk: f64,
}

/// The weight `v(t)` of each event time, latest first.
fn time_weights(local: &Local, order: &[u32], opts: &ConcordanceOptions) -> Result<Vec<f64>> {
    let n_deaths = local.death.iter().filter(|&&d| d).count();
    let mut weights: Vec<f64> = Vec::new();
    let mut times: Vec<f64> = Vec::new();
    let kind = if n_deaths < 2 {
        TimeWeight::N
    } else {
        opts.time_weight
    };
    if kind == TimeWeight::N {
        let mut previous: Option<f64> = None;
        for &i in order {
            let i = i as usize;
            if local.death[i] && previous != Some(local.stop[i]) {
                previous = Some(local.stop[i]);
                times.push(local.stop[i]);
                weights.push(1.0);
            }
        }
    } else {
        let steps = risk_steps(local, order);
        // Fold from the earliest time: S(t-) and G(t-) before the step.
        let total: f64 = local.weight.iter().sum();
        let mut surv = 1.0_f64;
        let mut cens = 1.0_f64;
        for step in steps.iter().rev() {
            if step.deaths > 0.0 {
                let v = match kind {
                    TimeWeight::S => total * surv / step.at_risk,
                    TimeWeight::SOverG => total * surv / (cens * step.at_risk),
                    TimeWeight::NOverG2 => 1.0 / (cens * cens),
                    TimeWeight::I => 1.0 / step.at_risk,
                    TimeWeight::N => 1.0,
                };
                times.push(step.time);
                weights.push(v);
                surv *= (step.at_risk - step.deaths) / step.at_risk;
            }
            if step.censored > 0.0 {
                cens *= (step.at_risk_censor - step.censored) / step.at_risk_censor;
            }
        }
        // Collected earliest first; the sweep wants the latest first.
        weights.reverse();
        times.reverse();
    }
    if weights.iter().any(|v| !v.is_finite()) {
        return Err(Error::invalid(
            "a time weight is not finite: the censoring distribution or the risk set is empty at an event time",
        ));
    }
    if let Some(ymax) = opts.ymax {
        for (v, &t) in weights.iter_mut().zip(&times) {
            if t > ymax {
                *v = 0.0;
            }
        }
    }
    Ok(weights)
}

/// The risk-set summary at each distinct stop time, latest first, for the rows
/// of a stratum with at least one death. Steps without a death or a censoring
/// cannot occur, since every row sits at its stop time.
fn risk_steps(local: &Local, order: &[u32]) -> Vec<Step> {
    let m = order.len();
    let by_start = start_order(local);
    let mut steps = Vec::new();
    let mut at_risk = 0.0_f64;
    let mut left = 0_usize;
    let mut i = 0;
    while i < m {
        let t = local.stop[order[i] as usize];
        let mut censored = 0.0;
        let mut deaths = 0.0;
        let mut before_deaths = at_risk;
        while i < m && local.stop[order[i] as usize] == t {
            let row = order[i] as usize;
            at_risk += local.weight[row];
            if local.death[row] {
                deaths += local.weight[row];
            } else {
                censored += local.weight[row];
                before_deaths = at_risk;
            }
            i += 1;
        }
        if let (Some(by_start), Some(start)) = (&by_start, &local.start) {
            while left < m && start[by_start[left] as usize] >= t {
                at_risk -= local.weight[by_start[left] as usize];
                before_deaths -= local.weight[by_start[left] as usize];
                left += 1;
            }
        }
        steps.push(Step {
            time: t,
            censored,
            deaths,
            at_risk_censor: before_deaths,
            at_risk,
        });
    }
    steps
}

/// Enter row `rank` of weight `w` in the risk set and the running variance.
fn enter(risk: &mut Bag, z2: &mut f64, rank: usize, w: f64) {
    *z2 += w * (risk.total - risk.at[rank]) * (risk.total + risk.at[rank] + w);
    risk.add(rank, w);
}

/// Take row `rank` of weight `w` out of the risk set and the running variance.
fn leave(risk: &mut Bag, z2: &mut f64, rank: usize, w: f64) {
    risk.add(rank, -w);
    *z2 -= w * (risk.total - risk.at[rank]) * (risk.total + risk.at[rank] + w);
}

/// Sweep one stratum.
fn sweep(local: &Local, opts: &ConcordanceOptions) -> Result<Swept> {
    let m = local.stop.len();
    let mut counts = [0.0_f64; 6];
    if !local.death.iter().any(|&d| d) {
        return Ok(Swept {
            counts,
            partners: opts.variance.then(|| vec![[0.0; 3]; m]),
        });
    }
    let order = sweep_order(local);
    let v = time_weights(local, &order, opts)?;
    let by_start = start_order(local);
    let mut partners = opts.variance.then(|| vec![[0.0_f64; 3]; m]);
    let mut risk = Bag::new(local.n_ranks);
    let mut deaths = Bag::new(local.n_ranks);
    let mut z2 = 0.0_f64;
    let mut left = 0_usize;
    let mut group = 0_usize;
    let mut i = 0;
    while i < m {
        let row = order[i] as usize;
        let rank = local.rank[row] as usize;
        let w = local.weight[row];
        if !local.death[row] {
            if let Some(p) = partners.as_mut() {
                let (above, below, equal) = deaths.around(rank);
                p[row][0] -= below;
                p[row][1] -= above;
                p[row][2] -= equal;
            }
            enter(&mut risk, &mut z2, rank, w);
            i += 1;
            continue;
        }
        let t = local.stop[row];
        if let (Some(by_start), Some(start)) = (&by_start, &local.start) {
            while left < m && start[by_start[left] as usize] >= t {
                let out = by_start[left] as usize;
                let out_rank = local.rank[out] as usize;
                if let Some(p) = partners.as_mut() {
                    let (above, below, equal) = deaths.around(out_rank);
                    p[out][0] += below;
                    p[out][1] += above;
                    p[out][2] += equal;
                }
                leave(&mut risk, &mut z2, out_rank, local.weight[out]);
                left += 1;
            }
        }
        let mut j = i;
        while j < m && local.death[order[j] as usize] && local.stop[order[j] as usize] == t {
            j += 1;
        }
        let vt = v[group];
        group += 1;
        let mut dwt = 0.0_f64;
        let mut dwt_same = 0.0_f64;
        let mut rank_now = rank;
        for &p in &order[i..j] {
            let p = p as usize;
            let r = local.rank[p] as usize;
            let wp = local.weight[p];
            if r != rank_now {
                rank_now = r;
                dwt_same = 0.0;
            }
            counts[3] += wp * (dwt - dwt_same) * vt;
            counts[4] += wp * dwt_same * vt;
            dwt += wp;
            dwt_same += wp;
            let (above, below, equal) = risk.around(r);
            counts[0] += wp * above * vt;
            counts[1] += wp * below * vt;
            counts[2] += wp * equal * vt;
            if let Some(part) = partners.as_mut() {
                part[p][0] += above * vt;
                part[p][1] += below * vt;
                part[p][2] += equal * vt;
            }
            deaths.add(r, vt * wp);
        }
        for &p in &order[i..j] {
            let p = p as usize;
            let r = local.rank[p] as usize;
            if let Some(part) = partners.as_mut() {
                let (above, below, equal) = deaths.around(r);
                part[p][0] -= below;
                part[p][1] -= above;
                part[p][2] -= equal;
            }
            enter(&mut risk, &mut z2, r, local.weight[p]);
        }
        counts[5] += dwt * vt * z2 / risk.total;
        i = j;
    }
    if let Some(p) = partners.as_mut() {
        let remaining: &[u32] = match &by_start {
            Some(by_start) => &by_start[left..],
            None => &order,
        };
        for &out in remaining {
            let out = out as usize;
            let (above, below, equal) = deaths.around(local.rank[out] as usize);
            p[out][0] += below;
            p[out][1] += above;
            p[out][2] += equal;
        }
    }
    Ok(Swept { counts, partners })
}

#[cfg(test)]
mod tests;
