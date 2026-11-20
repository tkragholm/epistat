//! The Aalen-Johansen estimator of cumulative incidence under competing risks,
//! with weights, groups and delayed entry, its infinitesimal jackknife standard
//! errors, and a clustered bootstrap of the incidence in two arms over many
//! cells.
//!
//! # Algorithm
//!
//! Within a group, let `t_1 < t_2 < ...` be the distinct exit times. At `t` let
//! `Y(t)` be the weight at risk just before `t`, `d_k(t)` the weight of events
//! of cause `k` at `t` and `d(t) = sum_k d_k(t)`. The event-free probability
//! `S` and the cumulative incidence `F_k` of cause `k` advance by
//!
//! ```text
//! F_k(t) = F_k(t-) + S(t-) d_k(t) / Y(t)
//! S(t)   = S(t-) (1 - d(t) / Y(t))
//! ```
//!
//! starting from `S = 1` and `F_k = 0` \[1\]\[2\]. Each cause increment is weighted by
//! the event-free probability just before the time, not at it, which is what
//! separates the estimator from one minus the Kaplan-Meier estimate when
//! competing causes are present. A time with nobody at risk moves nothing.
//! Times that carry only censoring appear in the output and leave the
//! probabilities unchanged.
//!
//! Without delayed entry the weight at risk is the weight of everyone who has
//! not yet left, a reverse cumulative sum over the distinct times. With entry
//! times a row is at risk on `(entry, time]`, so at `t` the risk set is
//! everyone who entered strictly before `t` less everyone who left strictly
//! before it. A running balance over the merged entry and exit times gives both
//! terms: the changes at each distinct time are summed, and the balance is read
//! before they are added, which makes the entry inequality strict and the exit
//! inequality inclusive.
//!
//! Rows are ordered by group and time with a stable sort, so rows tied on both
//! are summed in input order and the result does not depend on the number of
//! threads. Times are compared for exact equality once the tie rule below has
//! been applied.
//!
//! # Input
//!
//! [`estimate`] and [`bootstrap`] take a [`SurvData`]. It supplies the exit
//! time (`stop`), the status, the optional entry time (`start`), the weight and
//! the groups, which are its strata, with the caller's stratum codes carried
//! into the output. Status 0 is censoring and a code from 1 to `n_codes` is a
//! cause. The estimate has a column for every code, observed or not, as
//! `survfit` has a level for every level of a factor status. The tie rule of
//! `survival::aeqSurv` is applied to the times first and is the default, as in
//! `survfit`. [`AalenJohansenOptions::timefix`] and [`BootstrapOptions::timefix`]
//! change it, and `TimeFix::Exact` compares times exactly.
//!
//! Rows of weight 0 are not used. [`Curves::n_rows`] counts every row,
//! as `n` of `survfit` does, and [`Curves::n_used`] those of positive weight. A
//! time carried only by rows of weight 0 does not appear in the output.
//!
//! The deprecated [`curves`], [`bootstrap_arm_incidence`] and [`ArmColumns`]
//! behave as before. Times are compared exactly, every row is used and weights
//! of 0 count as rows at risk, the causes listed are those observed, and the
//! family codes are taken as given. They run the same code on their slices
//! rather than on a [`SurvData`], which would refuse some input they accept,
//! such as a weight of NaN or an entry time not before the exit time.
//!
//! # Agreement with R
//!
//! The estimate, the weight at risk, the number at risk and the number of events
//! at each distinct exit time are those of `survival::survfit` on a factor
//! status (`pstate`, `n.risk`, `n.event`), with case weights, groups and
//! `Surv(entry, time, status)` where a row is at risk on `(entry, time]`, so an
//! entry that equals another row's exit time does not put it at risk at that
//! time. `survfit` lists the same times, censoring-only times included. The
//! unweighted estimate also agrees with `etm::etm`, `cmprsk::cuminc` and
//! `prodlim::prodlim`. `prodlim` reads a case weight as a count, so it is
//! compared only without weights.
//!
//! # The variance
//!
//! [`variance`] gives the standard errors that `survfit` reports for a fit with
//! a factor status: of the event-free probability and the incidence of each
//! cause (`std.err`), of the cumulative cause-specific hazards (`std.chaz`) and of
//! the restricted time in each state (`std.auc`), by the infinitesimal jackknife
//! \[4\]\[5\]. Each row has an influence on each of them, the derivative of the
//! estimate with respect to its case weight times the weight. The influences are
//! summed within a cluster, the cluster column, else the id column, else the row
//! itself as `survfit` does without an id, and the variance is the sum of squares
//! over the clusters. The recursion is that of `multistate::influence`, which
//! the multi-state estimate shares, with the event-free state and one state per
//! cause. The result lines up with the rows of [`estimate`], and can carry the
//! influence of every cluster at chosen times, as pseudo-values need. It does
//! not keep the influence at all times, which would be clusters by times by
//! states values: each block of 256 clusters runs the whole recursion alone.
//!
//! # The bootstrap
//!
//! [`bootstrap`] repeats the estimate for the exposed and the
//! comparator arm of many cells, each a subset of the rows, under random
//! cluster weights. A replicate draws one `Exp(1)` variate per cluster, scales
//! the variates to sum to the number of clusters, which is the Bayesian
//! bootstrap with a Dirichlet(1, ..., 1) weight vector scaled to mean 1 \[3\],
//! and multiplies each row's weight by the weight of its cluster. The incidence
//! of one cause is read at the requested ages. Two properties make this cheaper
//! than repeating the point estimate:
//!
//! * A replicate changes only the weights, never which rows exist or when their
//!   events happen, so the sort, the distinct times and the block boundaries of
//!   a cell are computed once and reused by every replicate.
//! * Cells and replicates are independent and run on all cores. The weights of
//!   replicate `r` of a cell come from the seed `replicate_seed(cell_seed, r)`,
//!   so the output does not depend on the number of threads.
//!
//! The clusters are the cluster column of the [`SurvData`], and the arm of each
//! row is a separate argument, since it is not part of the survival data. The
//! cells stay a [`RowWindows`] over the rows of the data.
//!
//! Seeds are supplied per cell, so the caller decides where a seed comes from.
//! The output holds the incidence of each arm, not their difference, so a
//! caller can form a difference, a ratio or per-arm bands from the same draws.
//!
//! # Edge cases
//!
//! A cell with an empty arm has no estimate, is flagged not estimable, and its
//! block of the output is zero. An age before the first event time takes the
//! incidence so far, which is zero. An age after the last takes the final
//! incidence. Everyone is at risk from a common origin in the bootstrap, which
//! takes no entry times, so data with a start column is refused.
//!
//! # References
//!
//! 1. Aalen OO, Johansen S. An empirical transition matrix for non-homogeneous
//!    Markov chains based on censored observations. Scand J Stat
//!    1978;5(3):141-50.
//! 2. Andersen PK, Borgan O, Gill RD, Keiding N. Statistical models based on
//!    counting processes. New York: Springer; 1993.
//! 3. Rubin DB. The Bayesian bootstrap. Ann Stat 1981;9(1):130-4.
//!    doi:10.1214/aos/1176345338
//! 4. Therneau TM, Grambsch PM. Modeling survival data: extending the Cox
//!    model. New York: Springer; 2000.
//! 5. Therneau TM. Methods of the survfit function, vignette `methods` of the
//!    survival package, on the infinitesimal jackknife for the Aalen-Johansen
//!    estimate.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the entry points check that the row count fits in u32"
)]

use std::borrow::Cow;
use std::collections::HashMap;

use epistat_rng::{dirichlet_weights, replicate_seed};
use rayon::prelude::*;

use crate::check::{check_lengths, check_row_count};
use crate::error::{Error, Result};
use crate::multistate::{check_ascending, influence, snapshot_positions};
use crate::surv::SurvData;
use crate::timefix::TimeFix;

const CENSORED: i32 = 0;

/// Advances the estimator over one distinct time.
///
/// `survival` is `S(t-)` on entry and `S(t)` on return. With `Y` the weight at
/// risk just before the time, `d_any` the weighted events of every cause and
/// `d_causes[k]` those of cause `k`,
///
/// ```text
/// F_k  <- F_k + S(t-) d_k / Y,        S(t) = S(t-) (1 - d_any / Y).
/// ```
///
/// `incidence[k]` is `F_k` and gains the increment of cause `k`. The
/// increment is `S(t-) * d / Y`, formed in that order, and the survival is
/// `S(t-) * (1 - d_any / Y)`. A time with nobody at risk, `at_risk <= 0` or NaN,
/// moves nothing. That can arise only with zero weights, where every `d` is zero
/// too. The shorter of `incidence` and `d_causes` decides how many causes
/// are advanced.
///
/// This is the step [`estimate`] and the bootstrap use, so a caller that
/// accumulates the same sums itself, for example for reweighted data, gets the
/// same bits by calling it.
#[inline]
pub fn advance(
    survival: &mut f64,
    at_risk: f64,
    d_any: f64,
    d_causes: &[f64],
    incidence: &mut [f64],
) {
    if at_risk > 0.0 {
        let before = *survival;
        for (cif, &d) in incidence.iter_mut().zip(d_causes) {
            *cif += before * d / at_risk;
        }
        *survival = before * (1.0 - d_any / at_risk);
    }
}

/// The estimate for every group, one row per group and distinct exit time.
///
/// `cif` holds one column per cause, in the order of `causes`.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Curves {
    /// The causes, ascending. [`estimate`] lists every code from 1 to
    /// `n_codes`, observed or not. The deprecated [`curves`] lists the causes
    /// observed.
    pub causes: Vec<i32>,
    /// The group of each row. [`estimate`] gives the caller's stratum codes.
    pub group: Vec<u32>,
    /// The distinct exit time of each row.
    pub time: Vec<f64>,
    /// Weighted at risk just before `time`.
    pub n_risk: Vec<f64>,
    /// The same in individuals.
    pub n_at_risk: Vec<f64>,
    /// Weighted events of every cause.
    pub n_event: Vec<f64>,
    /// Weighted events of each cause, one column per cause in the order of
    /// `causes`. The columns sum to `n_event` at each time, added in cause
    /// order.
    pub n_event_cause: Vec<Vec<f64>>,
    /// `S(t)`.
    pub event_free: Vec<f64>,
    /// `F_k(t)`, one column per cause in the order of `causes`.
    pub cif: Vec<Vec<f64>>,
    /// The number of rows given.
    pub n_rows: usize,
    /// The number of rows the estimate is built from. [`estimate`] leaves out
    /// the rows of weight 0, so this is [`SurvData::n_used`]. The deprecated
    /// [`curves`] uses every row, so it equals `n_rows`.
    pub n_used: usize,
}

/// One group's rows at one distinct exit time.
struct Block {
    time: f64,
    leaving: f64,
    people: f64,
    d: Vec<f64>,
}

/// The weighted Aalen-Johansen estimate per group, on checked input.
///
/// Rows are ordered by group and time with a stable sort. `causes` is the
/// ascending list of codes that get a column, and every nonzero event is in it.
/// `entry`, when given, is the delayed-entry time: a row is at risk on `(entry, time]`. Every exit
/// time appears, including times with censoring only, which leave the
/// probabilities unchanged.
// `==` on times is exact on purpose, as in `Arm::build`: rows share a time or
// they do not.
#[expect(clippy::float_cmp, reason = "rows share a time or they do not")]
fn curves_checked(
    time: &[f64],
    event: &[i32],
    weight: &[f64],
    group: &[u32],
    entry: Option<&[f64]>,
    causes: Vec<i32>,
) -> Curves {
    let k = causes.len();

    let mut order: Vec<u32> = (0..time.len() as u32).collect();
    order.par_sort_by(|&a, &b| {
        let (a, b) = (a as usize, b as usize);
        group[a].cmp(&group[b]).then(time[a].total_cmp(&time[b]))
    });
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < order.len() {
        let g = group[order[start] as usize];
        let mut end = start + 1;
        while end < order.len() && group[order[end] as usize] == g {
            end += 1;
        }
        ranges.push((start, end));
        start = end;
    }

    let per_group: Vec<Vec<(Block, f64, f64)>> = ranges
        .par_iter()
        .map(|&(lo, hi)| {
            let rows = &order[lo..hi];
            let mut blocks: Vec<Block> = Vec::new();
            for &row in rows {
                let i = row as usize;
                if blocks.last().is_none_or(|b| b.time != time[i]) {
                    blocks.push(Block {
                        time: time[i],
                        leaving: 0.0,
                        people: 0.0,
                        d: vec![0.0; k],
                    });
                }
                let block = blocks.last_mut().expect("pushed above");
                block.leaving += weight[i];
                block.people += 1.0;
                if event[i] != CENSORED {
                    let c = causes.binary_search(&event[i]).expect("collected above");
                    block.d[c] += weight[i];
                }
            }
            let (risk, people) = match entry {
                None => risk_without_entry(&blocks),
                Some(entry) => risk_with_entry(&blocks, rows, entry, time, weight),
            };
            blocks
                .into_iter()
                .zip(risk.into_iter().zip(people))
                .map(|(block, (risk, people))| (block, risk, people))
                .collect()
        })
        .collect();

    let mut out = Curves {
        cif: vec![Vec::new(); k],
        n_event_cause: vec![Vec::new(); k],
        causes,
        ..Curves::default()
    };
    let mut incidence = vec![0.0f64; k];
    for (&(lo, _), blocks) in ranges.iter().zip(per_group) {
        let g = group[order[lo] as usize];
        let mut survival = 1.0f64;
        incidence.fill(0.0);
        for (block, at_risk, people) in blocks {
            // Summed over causes in cause order, which is `n_event`.
            let d_any = block.d.iter().fold(0.0, |total, &d| total + d);
            advance(&mut survival, at_risk, d_any, &block.d, &mut incidence);
            out.group.push(g);
            out.time.push(block.time);
            out.n_risk.push(at_risk);
            out.n_at_risk.push(people);
            out.n_event.push(d_any);
            for (column, &d) in out.n_event_cause.iter_mut().zip(&block.d) {
                column.push(d);
            }
            out.event_free.push(survival);
            for (column, &value) in out.cif.iter_mut().zip(&incidence) {
                column.push(value);
            }
        }
    }
    out
}

/// At risk just before each time without delayed entry: everyone who has not
/// yet left, a reverse cumulative sum over the blocks in time order.
fn risk_without_entry(blocks: &[Block]) -> (Vec<f64>, Vec<f64>) {
    let mut risk = vec![0.0; blocks.len()];
    let mut people = vec![0.0; blocks.len()];
    let (mut w, mut p) = (0.0f64, 0.0f64);
    for (b, block) in blocks.iter().enumerate().rev() {
        w += block.leaving;
        p += block.people;
        risk[b] = w;
        people[b] = p;
    }
    (risk, people)
}

/// At risk just before each exit time under delayed entry.
///
/// A row is at risk on `(entry, time]`, so at `t` the risk set is everyone who
/// entered strictly before `t` less everyone who left strictly before it. One
/// running balance over the merged entry and exit times gives both terms: the
/// changes at each distinct time are summed, and the balance is read before
/// they are added, which makes both inequalities strict.
#[expect(clippy::float_cmp, reason = "changes at one time are summed together")]
fn risk_with_entry(
    blocks: &[Block],
    rows: &[u32],
    entry: &[f64],
    time: &[f64],
    weight: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    // (time, weight change, people change), entries before exits at a tie so the
    // stable sort below keeps the order fixed.
    let mut changes: Vec<(f64, f64, f64)> = Vec::with_capacity(2 * rows.len());
    for &row in rows {
        let i = row as usize;
        changes.push((entry[i], weight[i], 1.0));
    }
    for &row in rows {
        let i = row as usize;
        changes.push((time[i], -weight[i], -1.0));
    }
    changes.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut risk = Vec::with_capacity(blocks.len());
    let mut people = Vec::with_capacity(blocks.len());
    let (mut w, mut p) = (0.0f64, 0.0f64);
    let mut next = 0usize;
    let mut start = 0usize;
    while start < changes.len() {
        let t = changes[start].0;
        let mut end = start;
        let (mut dw, mut dp) = (0.0f64, 0.0f64);
        while end < changes.len() && changes[end].0 == t {
            dw += changes[end].1;
            dp += changes[end].2;
            end += 1;
        }
        if next < blocks.len() && blocks[next].time == t {
            risk.push(w);
            people.push(p);
            next += 1;
        }
        w += dw;
        p += dp;
        start = end;
    }
    (risk, people)
}

/// The causes observed in `event`, ascending.
fn observed_causes(event: &[i32]) -> Vec<i32> {
    let mut causes: Vec<i32> = event.iter().copied().filter(|&e| e != CENSORED).collect();
    causes.sort_unstable();
    causes.dedup();
    causes
}

/// What [`estimate`] and [`bootstrap`] do besides the data.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AalenJohansenOptions {
    /// How near-equal times are treated. The default is `survival`'s rule,
    /// see [`TimeFix`].
    pub timefix: TimeFix,
}

/// The weighted Aalen-Johansen estimate for every stratum of `data`.
///
/// Status 0 is censoring and a code `k` in `1..=n_codes` is an event of cause
/// `k`. The strata are the groups, and the result carries the caller's stratum
/// codes. Without a stratum column there is one group, coded 0. A start time
/// is the delayed-entry time: a row is at risk on `(start, stop]`. The tie rule
/// of `opts` is applied to the times first, `survival`'s by default. Rows of
/// weight 0 are not used. They are counted in `n_rows` and not in `n_used`, and
/// a time that only they carry does not appear.
///
/// Every cause from 1 to `n_codes` has a column in [`Curves::cif`], whether it
/// is observed or not, as `survfit` reports every level of a factor status. A
/// cause that is never observed has an incidence of 0.
///
/// # Errors
///
/// [`Error::Input`] when the tie rule fails, see
/// [`SurvData::with_timefix`], or `n_codes` does not fit an `i32`.
pub fn estimate(data: &SurvData<'_>, opts: &AalenJohansenOptions) -> Result<Curves> {
    let data = data.clone().with_timefix(opts.timefix)?;
    let surv = data.followup()?;
    let n_codes = i32::try_from(surv.n_codes())
        .map_err(|_| Error::input("n_codes must fit an i32 status"))?;
    let used = data.used_index();
    let index = used.as_deref();
    let time = pick(surv.stop(), index);
    let event = pick(surv.status(), index);
    let entry = surv.start().map(|start| pick(start, index));
    let weight = data.weight().map_or_else(
        || Cow::Owned(vec![1.0; data.n_used()]),
        |weight| pick(weight, index),
    );
    let group = data.stratum().map_or_else(
        || Cow::Owned(vec![0u32; data.n_used()]),
        |stratum| pick(stratum, index),
    );
    let mut out = curves_checked(
        &time,
        &event,
        &weight,
        &group,
        entry.as_deref(),
        (1..=n_codes).collect(),
    );
    if let Some(levels) = data.stratum_levels() {
        let codes = levels.to_vec();
        for g in &mut out.group {
            *g = codes[*g as usize];
        }
    }
    out.n_rows = data.n_rows();
    out.n_used = data.n_used();
    Ok(out)
}

/// The values of a column at `index`, or the column itself when `index` is
/// `None` because every row is used.
fn pick<'a, T: Copy>(column: &'a [T], index: Option<&[u32]>) -> Cow<'a, [T]> {
    match index {
        None => Cow::Borrowed(column),
        Some(index) => Cow::Owned(index.iter().map(|&i| column[i as usize]).collect()),
    }
}

/// What [`variance`] reports besides the standard errors at each time.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VarianceOptions {
    /// Horizons at which the standard error of the restricted time in each
    /// state, from the start of the curves, is wanted. Ascending. Empty, the
    /// default, asks for none.
    pub horizons: Vec<f64>,
    /// Times at which the influence of every cluster is returned. Ascending.
    /// The influence at a time is that after the last exit time not after it,
    /// and zero before the first. Empty, the default, asks for none: the
    /// influence of all clusters at all times is not kept.
    pub influence_times: Vec<f64>,
}

/// The influence part of [`Variance`] that is not one row per time: one entry
/// for each group.
#[derive(Debug, Clone, PartialEq)]
pub struct GroupInfluence {
    /// The group, as in [`Curves::group`].
    pub group: u32,
    /// The caller's code of each cluster of the group, ascending: the cluster
    /// column, else the id column, else the row number of the data.
    pub clusters: Vec<u32>,
    /// `horizon * state`, row-major, the states being the event-free state and
    /// then each cause: the standard error of the restricted time in the state
    /// from the origin to each horizon.
    pub std_rmean: Vec<f64>,
    /// `cluster * influence time * state`, row-major: the influence of each
    /// cluster on the event-free probability and the incidence of each cause.
    /// The squares of a column sum to the variance.
    pub influence: Vec<f64>,
}

/// The standard errors of the Aalen-Johansen estimate by the infinitesimal
/// jackknife, one row per group and exit time as in [`Curves`].
///
/// A state is the event-free state, `0` in the vectors that say `state`, and then
/// each cause in the order of `causes`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Variance {
    /// The causes, as in [`Curves::causes`].
    pub causes: Vec<i32>,
    /// The group of each row, as in [`Curves::group`].
    pub group: Vec<u32>,
    /// The distinct exit time of each row, as in [`Curves::time`].
    pub time: Vec<f64>,
    /// `S(t)` from the recursion the errors come from, equal to
    /// [`Curves::event_free`] up to rounding.
    pub event_free: Vec<f64>,
    /// `F_k(t)`, one column per cause, equal to [`Curves::cif`] up to rounding.
    pub cif: Vec<Vec<f64>>,
    /// The standard error of `S(t)`, `fit$std.err` of `survfit` in the column of
    /// the event-free state.
    pub std_event_free: Vec<f64>,
    /// The standard error of `F_k(t)`, one column per cause.
    pub std_cif: Vec<Vec<f64>>,
    /// The cumulative hazard of each cause from the event-free state, one column
    /// per cause, 0 for a cause that is not observed.
    pub cumhaz: Vec<Vec<f64>>,
    /// The standard error of each cumulative hazard, `fit$std.chaz`.
    pub std_cumhaz: Vec<Vec<f64>>,
    /// The standard error of the restricted time event-free from the origin to
    /// each time, `fit$std.auc` of the event-free state.
    pub std_time_event_free: Vec<f64>,
    /// The same for the restricted time lost to each cause, one column per
    /// cause.
    pub std_time: Vec<Vec<f64>>,
    /// The origin of the restricted times: the earliest start time, or without
    /// start times the smaller of 0 and the earliest exit time, over every row
    /// used, as `survfit` takes its `t0`.
    pub origin: f64,
    /// The horizons of [`VarianceOptions::horizons`].
    pub horizons: Vec<f64>,
    /// The times of [`VarianceOptions::influence_times`].
    pub influence_times: Vec<f64>,
    /// What is not one value per row, for each group in ascending order.
    pub groups: Vec<GroupInfluence>,
    /// The number of rows given.
    pub n_rows: usize,
    /// The number of rows of positive weight, which are the rows used.
    pub n_used: usize,
}

/// The standard errors of the incidence, the event-free probability, the
/// cumulative hazards and the restricted times of the Aalen-Johansen
/// estimate, by the infinitesimal jackknife that `survfit` reports.
///
/// The influence of each row on the estimate is summed within its cluster, and
/// the standard error at a time is the square root of the sum of squares over
/// the clusters. A cluster is a value of the cluster column, else of the id
/// column, and without either every row is its own. A weight enters as a case
/// weight, and the influence of a row is its weight times the derivative of the
/// estimate with respect to it. A start time is the delayed-entry time, as in
/// [`estimate`]. The rows and times are those of [`estimate`], so the result
/// lines up with [`Curves`] row by row. See the module `multistate::influence`
/// for the recursion, and for how memory is bounded: the influence of every
/// cluster at every time is never held, only that at
/// [`VarianceOptions::influence_times`], and the time is
/// `O(clusters * times * causes)`.
///
/// `survfit` starts the restricted times at the earliest start time over
/// every stratum together, and so does this. Its `start.time` argument, which
/// drops data before a time, has no counterpart.
///
/// # Errors
///
/// [`Error::Input`] when the tie rule fails, see [`SurvData::with_timefix`],
/// `n_codes` does not fit an `i32`, or `horizons` or `influence_times` is not
/// ascending or holds a value that is not finite.
pub fn variance(
    data: &SurvData<'_>,
    opts: &AalenJohansenOptions,
    v: &VarianceOptions,
) -> Result<Variance> {
    check_ascending(&v.horizons, "horizons", false)?;
    check_ascending(&v.influence_times, "influence_times", false)?;
    let data = data.clone().with_timefix(opts.timefix)?;
    let surv = data.followup()?;
    let n_codes = i32::try_from(surv.n_codes())
        .map_err(|_| Error::input("n_codes must fit an i32 status"))?;
    let causes: Vec<i32> = (1..=n_codes).collect();
    let k = causes.len();
    let states = k + 1;
    let used = data.used_index();
    let index = used.as_deref();
    let time = pick(surv.stop(), index);
    let event = pick(surv.status(), index);
    let entry = surv.start().map(|start| pick(start, index));
    let weight = data.weight().map_or_else(
        || Cow::Owned(vec![1.0; data.n_used()]),
        |weight| pick(weight, index),
    );
    let group = data.stratum().map_or_else(
        || Cow::Owned(vec![0u32; data.n_used()]),
        |stratum| pick(stratum, index),
    );
    let cluster = data.cluster().or_else(|| data.id()).map(|c| pick(c, index));
    let cluster_levels = if data.cluster().is_some() {
        data.cluster_levels()
    } else {
        data.id_levels()
    };
    let origin = entry.as_deref().map_or_else(
        || time.iter().fold(0.0f64, |m, &t| m.min(t)),
        |entry| entry.iter().fold(f64::INFINITY, |m, &t| m.min(t)),
    );
    let n_used = data.n_used();
    let row_number = |position: usize| index.map_or(position as u32, |rows| rows[position]);

    let mut members: Vec<Vec<u32>> = vec![Vec::new(); data.n_strata().max(1)];
    for (position, &g) in group.iter().enumerate() {
        members[g as usize].push(position as u32);
    }

    let mut out = Variance {
        cif: vec![Vec::new(); k],
        std_cif: vec![Vec::new(); k],
        cumhaz: vec![Vec::new(); k],
        std_cumhaz: vec![Vec::new(); k],
        std_time: vec![Vec::new(); k],
        causes,
        origin,
        horizons: v.horizons.clone(),
        influence_times: v.influence_times.clone(),
        n_rows: data.n_rows(),
        n_used,
        ..Variance::default()
    };
    let mut p0 = vec![0.0f64; states];
    p0[0] = 1.0;
    for (dense, rows) in members.iter().enumerate() {
        if rows.is_empty() {
            continue;
        }
        let g_code = data
            .stratum_levels()
            .map_or(dense as u32, |l| l.code(dense).unwrap_or(dense as u32));
        let mut times: Vec<f64> = rows.iter().map(|&r| time[r as usize]).collect();
        times.sort_by(f64::total_cmp);
        #[expect(clippy::float_cmp, reason = "rows share an exit time or they do not")]
        times.dedup_by(|a, b| *a == *b);

        let (labels, local): (Vec<u32>, Vec<u32>) = match &cluster {
            Some(codes) => {
                let mut present: Vec<u32> = rows.iter().map(|&r| codes[r as usize]).collect();
                present.sort_unstable();
                present.dedup();
                let local = rows
                    .iter()
                    .map(|&r| present.partition_point(|&c| c < codes[r as usize]) as u32)
                    .collect();
                let labels = present
                    .iter()
                    .map(|&c| cluster_levels.map_or(c, |l| l.code(c as usize).unwrap_or(c)))
                    .collect();
                (labels, local)
            }
            None => (
                rows.iter().map(|&r| row_number(r as usize)).collect(),
                (0..rows.len() as u32).collect(),
            ),
        };
        let pick_rows =
            |values: &[f64]| -> Vec<f64> { rows.iter().map(|&r| values[r as usize]).collect() };
        let start = entry
            .as_deref()
            .map_or_else(|| vec![f64::NEG_INFINITY; rows.len()], &pick_rows);
        let stop = pick_rows(&time);
        let weights = pick_rows(&weight);
        let from = vec![0u32; rows.len()];
        let to: Vec<u32> = rows
            .iter()
            .map(|&r| match event[r as usize] {
                CENSORED => u32::MAX,
                code => u32::try_from(code).unwrap_or(u32::MAX),
            })
            .collect();
        let snapshots = snapshot_positions(&times, &v.influence_times);
        let result = influence::run(
            &influence::Process {
                start: &start,
                stop: &stop,
                from: &from,
                to: &to,
                weight: &weights,
                cluster: &local,
            },
            &influence::Request {
                states,
                n_clusters: labels.len(),
                times: &times,
                p0: &p0,
                u0: None,
                origin,
                horizons: &v.horizons,
                snapshots: &snapshots,
            },
        );
        let h = result.transitions.len();
        for (t, &time) in times.iter().enumerate() {
            out.group.push(g_code);
            out.time.push(time);
            out.event_free.push(result.p[t * states]);
            out.std_event_free.push(result.std_p[t * states]);
            out.std_time_event_free.push(result.std_a[t * states]);
            for c in 0..k {
                out.cif[c].push(result.p[t * states + 1 + c]);
                out.std_cif[c].push(result.std_p[t * states + 1 + c]);
                out.std_time[c].push(result.std_a[t * states + 1 + c]);
                out.cumhaz[c].push(0.0);
                out.std_cumhaz[c].push(0.0);
            }
            for (jk, &(_, to)) in result.transitions.iter().enumerate() {
                let c = to as usize - 1;
                let last = out.cumhaz[c].len() - 1;
                out.cumhaz[c][last] = result.cumhaz[t * h + jk];
                out.std_cumhaz[c][last] = result.std_c[t * h + jk];
            }
        }
        out.groups.push(GroupInfluence {
            group: g_code,
            clusters: labels,
            std_rmean: result.std_h,
            influence: result.influence,
        });
    }
    Ok(out)
}

/// The weighted Aalen-Johansen estimate for every group.
///
/// `event` is the cause code of each row, with 0 for censoring. `group` is a
/// dense code per row, built by the caller from its grouping. `entry`, when
/// given, is the delayed-entry time: a row is at risk on `(entry, time]`.
/// Returns one row per group and distinct exit time, ordered by both. See
/// [`Curves`] for the columns.
///
/// Times are compared exactly, and every row is used, rows of weight 0
/// included. The causes listed are those observed. It accepts weights and times
/// that [`SurvData`] refuses, so its numbers stay what they were.
///
/// # Errors
///
/// [`Error::Input`] when the slices differ in length or a time or entry is NaN.
#[deprecated(since = "0.3.0", note = "use `estimate` with a `SurvData`")]
pub fn curves(
    time: &[f64],
    event: &[i32],
    weight: &[f64],
    group: &[u32],
    entry: Option<&[f64]>,
) -> Result<Curves> {
    let n = time.len();
    check_lengths(&[
        ("time", n),
        ("event", event.len()),
        ("weight", weight.len()),
        ("group", group.len()),
        ("entry", entry.map_or(n, <[f64]>::len)),
    ])?;
    check_row_count(n)?;
    if time
        .iter()
        .chain(entry.into_iter().flatten())
        .any(|t| t.is_nan())
    {
        return Err(Error::input("time and entry must not be NaN"));
    }
    let mut out = curves_checked(time, event, weight, group, entry, observed_causes(event));
    out.n_rows = n;
    out.n_used = n;
    Ok(out)
}

/// How the cluster weights of a replicate are drawn for the cells of a
/// bootstrap.
///
/// `PerCell` treats each cell as its own analysis. The weights are drawn over
/// the clusters present in the cell, numbered from zero within it, and
/// normalised to sum to the number of those clusters. The seed of the cell
/// drives the draw, so the same seed in two cells gives one weight to two
/// different clusters, a coincidence rather than a shared draw.
///
/// `Shared` draws one weight vector over every cluster of the frame, numbered
/// by their codes, normalises it over the whole frame, with the first seed of
/// `cell_seeds`, and then reads it in each cell. Cells then share the weight of
/// a cluster they have in common, which is what an interaction between two
/// cells needs. The two modes give different draws, not the same draws at a
/// different cost.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Draw {
    /// Weights over the clusters of each cell, drawn from that cell's seed.
    PerCell,
    /// One weight vector over the clusters of the whole frame.
    Shared,
}

/// One cell's rows, gathered compactly.
///
/// Clusters are re-indexed to the cell under [`Draw::PerCell`] and keep their
/// codes under [`Draw::Shared`]. The difference is part of the estimator.
struct Cell {
    n_families: usize,
    exposed: Arm,
    comparator: Arm,
}

impl Cell {
    fn gather(
        rows: &[u32],
        time: &[f64],
        event: &[i32],
        weight: &[f64],
        family: &[u32],
        exposed: &[bool],
        draw: Draw,
        skip_unused: bool,
    ) -> Option<Self> {
        let mut cell_time = Vec::with_capacity(rows.len());
        let mut cell_event = Vec::with_capacity(rows.len());
        let mut cell_weight = Vec::with_capacity(rows.len());
        let mut cell_family = Vec::with_capacity(rows.len());
        let mut local: HashMap<u32, u32> = HashMap::new();
        let mut yes: Vec<u32> = Vec::new();
        let mut no: Vec<u32> = Vec::new();

        for &row in rows {
            let i = row as usize;
            // A row of weight 0 is not used, and a cluster it alone belongs to
            // is not a cluster of the cell.
            if skip_unused && weight[i] <= 0.0 {
                continue;
            }
            let position = cell_time.len() as u32;
            cell_time.push(time[i]);
            cell_event.push(event[i]);
            cell_weight.push(weight[i]);
            match draw {
                Draw::PerCell => {
                    let next = local.len() as u32;
                    cell_family.push(*local.entry(family[i]).or_insert(next));
                }
                Draw::Shared => cell_family.push(family[i]),
            }
            if exposed[i] { &mut yes } else { &mut no }.push(position);
        }
        if yes.is_empty() || no.is_empty() {
            // One arm empty is not an estimate.
            return None;
        }
        Some(Self {
            exposed: Arm::build(&yes, &cell_time, &cell_event, &cell_weight, &cell_family),
            comparator: Arm::build(&no, &cell_time, &cell_event, &cell_weight, &cell_family),
            n_families: local.len(), // meaningful only under `Draw::PerCell`
        })
    }
}

/// One arm of one cell: positions ordered by time, grouped into distinct times.
struct Arm {
    /// The arm's rows in time order, laid out contiguously so a replicate reads
    /// them in sequence.
    rows: Vec<ArmRow>,
    blocks: Vec<(u32, u32)>,
    times: Vec<f64>,
}

#[derive(Clone, Copy)]
struct ArmRow {
    weight: f64,
    family: u32,
    event: i32,
}

impl Arm {
    // `time[..] == t` is an exact comparison on purpose: this groups rows that
    // share an event time into blocks, and two rows are at the same time or they
    // are not. A tolerance would merge distinct event times into one block and
    // the risk sets would then be wrong at both of them.
    #[expect(clippy::float_cmp, reason = "rows share an event time or they do not")]
    fn build(
        positions: &[u32],
        time: &[f64],
        event: &[i32],
        weight: &[f64],
        family: &[u32],
    ) -> Self {
        let mut order: Vec<u32> = positions.to_vec();
        order.sort_unstable_by(|&a, &b| time[a as usize].total_cmp(&time[b as usize]));
        let mut blocks = Vec::new();
        let mut times = Vec::new();
        let mut start = 0usize;
        while start < order.len() {
            let t = time[order[start] as usize];
            let mut end = start + 1;
            while end < order.len() && time[order[end] as usize] == t {
                end += 1;
            }
            blocks.push((start as u32, end as u32));
            times.push(t);
            start = end;
        }
        let rows = order
            .iter()
            .map(|&position| ArmRow {
                weight: weight[position as usize],
                family: family[position as usize],
                event: event[position as usize],
            })
            .collect();
        Self {
            rows,
            blocks,
            times,
        }
    }

    /// Cumulative incidence of `cause` at each age, under one set of weights.
    fn cif_at_ages(
        &self,
        weights: &[f64],
        ages: &[f64],
        cause: i32,
        scratch: &mut Scratch,
        out: &mut [f64],
    ) {
        let n = self.blocks.len();
        scratch.reset(n);

        for (b, &(lo, hi)) in self.blocks.iter().enumerate() {
            let (mut leaving, mut d_any, mut d_cause) = (0.0, 0.0, 0.0);
            for row in &self.rows[lo as usize..hi as usize] {
                let w = row.weight * weights[row.family as usize];
                leaving += w;
                if row.event != CENSORED {
                    d_any += w;
                    if row.event == cause {
                        d_cause += w;
                    }
                }
            }
            scratch.leaving[b] = leaving;
            scratch.d_any[b] = d_any;
            scratch.d_cause[b] = d_cause;
        }

        // At risk just before t: everyone who has not yet left, a reverse
        // cumulative sum, which is why the blocks are ordered by time.
        let mut running = 0.0;
        for b in (0..n).rev() {
            running += scratch.leaving[b];
            scratch.at_risk[b] = running;
        }

        // Forward pass. Survival just BEFORE the time is the factor on each
        // increment, and is 1 before the first event time by definition.
        let mut survival = 1.0f64;
        let mut cif = 0.0f64;
        let mut age_at = 0usize;
        for b in 0..n {
            // Ages falling before this block take the incidence so far, which
            // for ages before the first event is zero. The estimator documents
            // that as an estimate rather than an absence of one: everyone is at
            // risk from a common origin.
            while age_at < ages.len() && ages[age_at] < self.times[b] {
                out[age_at] = cif;
                age_at += 1;
            }
            advance(
                &mut survival,
                scratch.at_risk[b],
                scratch.d_any[b],
                &[scratch.d_cause[b]],
                std::slice::from_mut(&mut cif),
            );
        }
        for slot in out.iter_mut().skip(age_at) {
            *slot = cif;
        }
    }
}

/// Per-replicate working vectors, reused across the replicates of a cell.
#[derive(Default)]
struct Scratch {
    leaving: Vec<f64>,
    d_any: Vec<f64>,
    d_cause: Vec<f64>,
    at_risk: Vec<f64>,
    weights: Vec<f64>,
}

impl Scratch {
    fn reset(&mut self, n: usize) {
        for buffer in [
            &mut self.leaving,
            &mut self.d_any,
            &mut self.d_cause,
            &mut self.at_risk,
        ] {
            buffer.clear();
            buffer.resize(n, 0.0);
        }
    }
}

/// Every row of a bootstrapped frame: the columns both arms are read from.
///
/// Superseded by a [`SurvData`] with a cluster column and the arm indicator
/// passed to [`bootstrap`].
#[deprecated(since = "0.3.0", note = "use a `SurvData` with `bootstrap`")]
#[derive(Debug, Clone, PartialEq)]
pub struct ArmColumns {
    time: Vec<f64>,
    event: Vec<i32>,
    weight: Vec<f64>,
    family: Vec<u32>,
    exposed: Vec<bool>,
}

#[expect(deprecated, reason = "the methods of the deprecated type")]
impl ArmColumns {
    /// The rows: exit `time`, cause code `event` (0 for censoring), `weight`,
    /// dense cluster code `family` and whether the row is in the exposed arm.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the columns differ in length.
    pub fn new(
        time: Vec<f64>,
        event: Vec<i32>,
        weight: Vec<f64>,
        family: Vec<u32>,
        exposed: Vec<bool>,
    ) -> Result<Self> {
        check_lengths(&[
            ("time", time.len()),
            ("weight", weight.len()),
            ("event", event.len()),
            ("family", family.len()),
            ("exposed", exposed.len()),
        ])?;
        check_row_count(time.len())?;
        Ok(Self {
            time,
            event,
            weight,
            family,
            exposed,
        })
    }

    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.time.len()
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }
}

/// Which rows each of several windows reads, as a CSR list.
///
/// Window `w` reads `rows[offsets[w]..offsets[w + 1]]`, and a row may belong
/// to several windows. Built only from offsets that are non-decreasing and
/// within `rows`, and from rows within the frame, so reading a window cannot
/// go out of bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowWindows {
    rows: Vec<u32>,
    offsets: Vec<u32>,
    n_rows: usize,
}

impl RowWindows {
    /// `label` names the arguments in an error, as `{label}_rows` and
    /// `{label}_offsets`. `n_rows` is the number of rows of the frame the
    /// windows index.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the offsets are empty, decrease or run past
    /// `rows`, or a row indexes past `n_rows`.
    pub fn new(rows: Vec<u32>, offsets: Vec<u32>, n_rows: usize, label: &str) -> Result<Self> {
        let Some(&last) = offsets.last() else {
            return Err(Error::input(format!("{label}_offsets must not be empty")));
        };
        if offsets.windows(2).any(|pair| pair[0] > pair[1]) || last as usize > rows.len() {
            return Err(Error::input(format!(
                "{label}_offsets must be non-decreasing and within {label}_rows"
            )));
        }
        if rows.iter().any(|&row| row as usize >= n_rows) {
            return Err(Error::input(format!("{label}_rows indexes past the frame")));
        }
        Ok(Self {
            rows,
            offsets,
            n_rows,
        })
    }

    /// The number of windows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Whether there are no windows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The rows of window `window`.
    ///
    /// # Panics
    ///
    /// When `window` is not below [`len`](Self::len).
    #[must_use]
    pub fn get(&self, window: usize) -> &[u32] {
        &self.rows[self.offsets[window] as usize..self.offsets[window + 1] as usize]
    }

    /// The number of rows of the frame the windows index.
    #[must_use]
    pub const fn frame_rows(&self) -> usize {
        self.n_rows
    }
}

/// `(exposed, comparator, estimable)`, as [`bootstrap`] returns them.
pub type ArmDraws = (Vec<f64>, Vec<f64>, Vec<bool>);

/// What [`bootstrap`] does besides the data.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BootstrapOptions {
    /// The number of replicates.
    pub n_replicates: usize,
    /// How the cluster weights are drawn for the cells.
    pub draw: Draw,
    /// How near-equal times are treated. The default is `survival`'s rule,
    /// see [`TimeFix`].
    pub timefix: TimeFix,
}

impl BootstrapOptions {
    /// Options with the given number of replicates and draw, and `survival`'s
    /// tie rule. Neither the number of replicates nor the draw has a default:
    /// the first sets the Monte Carlo error and the second what is resampled,
    /// so the caller states both.
    #[must_use]
    pub fn new(n_replicates: usize, draw: Draw) -> Self {
        Self {
            n_replicates,
            draw,
            timefix: TimeFix::default(),
        }
    }
}

/// Bootstrap each arm's cumulative incidence at each age, for every cell.
///
/// The clusters are the cluster column of `data`, which is required, and the
/// arm of each row is `exposed`. Cell `c` owns the rows `cells.get(c)`, indexed
/// in `data`. A row may appear in several cells, for example when the cells
/// are several partitions of the same cohort. The weights of replicate `r` of a
/// cell are drawn from `replicate_seed(cell_seeds[c], r)`. The event is status
/// `cause`, and every other nonzero status is a competing event.
///
/// Returns `(exposed, comparator, estimable)`. The two arms are of shape
/// `(cells, ages, n_replicates)`, row-major, with the ages sorted.
/// `estimable[c]` is false for a cell with an empty arm, which has no draws:
/// its block is zero and is not an estimate. `opts.draw` selects how the
/// cluster weights are drawn; see [`Draw`].
///
/// The tie rule of `opts` is applied to the times first. Rows of weight 0 are
/// not used, and a cluster that only they belong to is not a cluster of a
/// cell. Under [`Draw::Shared`] the weight vector still covers every cluster of
/// `data`. The bootstrap takes no delayed entry, since everyone is at risk from
/// a common origin, so `data` must have no start column.
///
/// Given the same rows, clusters, seeds and cluster codes `0..k`, the draws
/// are those of the deprecated [`bootstrap_arm_incidence`], to the bit.
///
/// # Errors
///
/// [`Error::Input`] when the tie rule fails, `data` has no cluster column or
/// has a start column, `cause` is outside `1..=n_codes`, `exposed` does not
/// have one value per row, `cell_seeds` does not hold one seed per cell, or
/// `cells` was built for a frame of another size than `data`.
pub fn bootstrap(
    data: &SurvData<'_>,
    exposed: &[bool],
    cells: &RowWindows,
    cell_seeds: &[u64],
    ages: &[f64],
    cause: u32,
    opts: &BootstrapOptions,
) -> Result<ArmDraws> {
    let data = data.clone().with_timefix(opts.timefix)?;
    let code = i32::try_from(data.cause(cause)?.code())
        .map_err(|_| Error::input("the cause must fit an i32 status"))?;
    let surv = data.followup()?;
    if surv.start().is_some() {
        return Err(Error::input(
            "the bootstrap takes no start times, everyone is at risk from a common origin",
        ));
    }
    let Some(cluster) = data.cluster() else {
        return Err(Error::input("the bootstrap needs a cluster column"));
    };
    check_lengths(&[("stop", data.n_rows()), ("exposed", exposed.len())])?;
    let weight = data
        .weight()
        .map_or_else(|| Cow::Owned(vec![1.0; data.n_rows()]), Cow::Borrowed);
    bootstrap_core(
        &Frame {
            time: surv.stop(),
            event: surv.status(),
            weight: &weight,
            family: cluster,
            exposed,
        },
        data.n_clusters(),
        true,
        cells,
        cell_seeds,
        ages.to_vec(),
        code,
        opts.n_replicates,
        opts.draw,
    )
}

/// The columns of a frame, one value per row.
struct Frame<'a> {
    time: &'a [f64],
    event: &'a [i32],
    weight: &'a [f64],
    family: &'a [u32],
    exposed: &'a [bool],
}

/// Bootstrap each arm's cumulative incidence at each age, for every cell.
///
/// Cell `c` owns the rows `cells.get(c)`. A row may appear in several cells,
/// for example when the cells are several partitions of the same cohort. The
/// weights of replicate `r` of a cell are drawn from
/// `replicate_seed(cell_seeds[c], r)`.
///
/// Returns `(exposed, comparator, estimable)`. The two arms are of shape
/// `(cells, ages, n_replicates)`, row-major, with the ages sorted.
/// `estimable[c]` is false for a cell with an empty arm, which has no draws:
/// its block is zero and is not an estimate. `draw` selects how the cluster
/// weights are drawn; see [`Draw`].
///
/// Times are compared exactly and every row is used, rows of weight 0
/// included. The family codes are taken as the caller numbered them.
///
/// # Errors
///
/// [`Error::Input`] when `cell_seeds` does not hold one seed per cell, or
/// `cells` was built for a frame of another size than `columns`.
#[deprecated(since = "0.3.0", note = "use `bootstrap` with a `SurvData`")]
#[expect(deprecated, reason = "the argument is the deprecated type")]
pub fn bootstrap_arm_incidence(
    columns: &ArmColumns,
    cells: &RowWindows,
    cell_seeds: &[u64],
    ages: Vec<f64>,
    cause: i32,
    n_replicates: usize,
    draw: Draw,
) -> Result<ArmDraws> {
    let ArmColumns {
        time,
        event,
        weight,
        family,
        exposed,
    } = columns;
    let n_families = family.iter().copied().max().map_or(0, |m| m as usize + 1);
    bootstrap_core(
        &Frame {
            time,
            event,
            weight,
            family,
            exposed,
        },
        n_families,
        false,
        cells,
        cell_seeds,
        ages,
        cause,
        n_replicates,
        draw,
    )
}

/// The bootstrap of both entry points. `n_families` is the size of the weight
/// vector under [`Draw::Shared`], and `skip_unused` leaves out rows of weight 0.
fn bootstrap_core(
    frame: &Frame<'_>,
    n_families: usize,
    skip_unused: bool,
    cells: &RowWindows,
    cell_seeds: &[u64],
    ages: Vec<f64>,
    cause: i32,
    n_replicates: usize,
    draw: Draw,
) -> Result<ArmDraws> {
    let Frame {
        time,
        event,
        weight,
        family,
        exposed,
    } = *frame;
    let n_cells = cells.len();
    if cells.frame_rows() != time.len() {
        return Err(Error::input(
            "the cell windows were built for a frame of another size than the columns",
        ));
    }
    if cell_seeds.len() != n_cells {
        return Err(Error::input("cell_seeds must have one seed per cell"));
    }
    if n_cells == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let mut ages_sorted = ages;
    ages_sorted.sort_by(f64::total_cmp);

    let n_ages = ages_sorted.len();
    // One cell's draws: age-major, `n_replicates` values per age.
    let block = n_ages * n_replicates;

    let gathered: Vec<Option<Cell>> = (0..n_cells)
        .into_par_iter()
        .map(|c| {
            Cell::gather(
                cells.get(c),
                time,
                event,
                weight,
                family,
                exposed,
                draw,
                skip_unused,
            )
        })
        .collect();
    let estimable: Vec<bool> = gathered.iter().map(Option::is_some).collect();
    let mut exposed_out = vec![0.0f64; n_cells * block];
    let mut comparator_out = vec![0.0f64; n_cells * block];
    if block == 0 {
        return Ok((exposed_out, comparator_out, estimable));
    }

    match draw {
        // Parallel over cells and then replicates, which lets rayon balance a
        // handful of huge cells (`sex`) against a long tail of small ones
        // (`diagnosis_group`).
        Draw::PerCell => exposed_out
            .par_chunks_mut(block)
            .zip(comparator_out.par_chunks_mut(block))
            .zip(gathered.par_iter())
            .enumerate()
            .for_each(|(c, ((exposed_block, comparator_block), cell))| {
                let Some(cell) = cell else {
                    return;
                };
                let seed = cell_seeds[c];
                let per_replicate: Vec<ArmLevels> = (0..n_replicates)
                    .into_par_iter()
                    .map_init(Scratch::default, |scratch, replicate| {
                        let mut weights = std::mem::take(&mut scratch.weights);
                        dirichlet_weights(
                            cell.n_families,
                            replicate_seed(seed, replicate as u64),
                            &mut weights,
                        );
                        let arms = both_arms(cell, &weights, &ages_sorted, cause, scratch);
                        scratch.weights = weights;
                        arms
                    })
                    .collect();
                for (replicate, (a, b)) in per_replicate.iter().enumerate() {
                    scatter(a, replicate, n_replicates, exposed_block);
                    scatter(b, replicate, n_replicates, comparator_block);
                }
            }),

        // Parallel over replicates, because the weights are what is shared:
        // one draw over the frame's families, then every cell measured
        // against it. Cells cannot be the outer loop here without either
        // redrawing per cell, which is the bug this mode exists to avoid, or
        // holding one weight vector per replicate, which is `G` by
        // replicates and gigabytes at cohort scale.
        Draw::Shared => {
            let seed = cell_seeds[0];
            let per_replicate: Vec<Vec<Option<ArmLevels>>> = (0..n_replicates)
                .into_par_iter()
                .map_init(Scratch::default, |scratch, replicate| {
                    let mut weights = std::mem::take(&mut scratch.weights);
                    dirichlet_weights(
                        n_families,
                        replicate_seed(seed, replicate as u64),
                        &mut weights,
                    );
                    let row = gathered
                        .iter()
                        .map(|cell| {
                            cell.as_ref()
                                .map(|cell| both_arms(cell, &weights, &ages_sorted, cause, scratch))
                        })
                        .collect();
                    scratch.weights = weights;
                    row
                })
                .collect();
            for (replicate, row) in per_replicate.iter().enumerate() {
                for (c, arms) in row.iter().enumerate() {
                    let Some((a, b)) = arms else {
                        continue;
                    };
                    let range = c * block..(c + 1) * block;
                    scatter(a, replicate, n_replicates, &mut exposed_out[range.clone()]);
                    scatter(b, replicate, n_replicates, &mut comparator_out[range]);
                }
            }
        }
    }
    Ok((exposed_out, comparator_out, estimable))
}

/// One cell's exposed and comparator incidence at each age, in one replicate.
type ArmLevels = (Vec<f64>, Vec<f64>);

/// Both arms of one cell under one set of cluster weights.
fn both_arms(
    cell: &Cell,
    weights: &[f64],
    ages: &[f64],
    cause: i32,
    scratch: &mut Scratch,
) -> ArmLevels {
    let mut a = vec![0.0f64; ages.len()];
    let mut b = vec![0.0f64; ages.len()];
    cell.exposed
        .cif_at_ages(weights, ages, cause, scratch, &mut a);
    cell.comparator
        .cif_at_ages(weights, ages, cause, scratch, &mut b);
    (a, b)
}

/// Writes one replicate's value at each age into an age-major block.
///
/// `block` holds `n_replicates` values per age, so the value at age `a` of
/// replicate `r` goes to `block[a * n_replicates + r]`.
fn scatter(by_age: &[f64], replicate: usize, n_replicates: usize, block: &mut [f64]) {
    for (age, &value) in by_age.iter().enumerate() {
        block[age * n_replicates + replicate] = value;
    }
}

#[cfg(test)]
#[expect(deprecated, reason = "these tests are of the deprecated function")]
mod curve_tests {
    use super::curves;

    fn close(got: f64, expected: f64) -> bool {
        (got - expected).abs() < 1e-12
    }

    /// `survival::survfit(Surv(time, state) ~ 1, weights = wt)`, R 4.6.1,
    /// survival 3.8.6. Rows are
    /// `(time, n_risk, event_free, cause 1, cause 2)`.
    #[test]
    fn matches_r_survfit_with_weights() {
        let time = [1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 6.0, 8.0, 9.0, 10.0];
        let event = [1, 2, 1, 0, 1, 2, 1, 0, 2, 1];
        let weight = [1.0, 1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25];
        let reference = [
            (1.0, 5.5, 0.818_181_818_181_818, 0.181_818_181_818_182, 0.0),
            (
                2.0,
                4.5,
                0.454_545_454_545_455,
                0.363_636_363_636_364,
                0.181_818_181_818_182,
            ),
            (
                4.0,
                2.5,
                0.454_545_454_545_455,
                0.363_636_363_636_364,
                0.181_818_181_818_182,
            ),
            (
                5.0,
                1.5,
                0.378_787_878_787_879,
                0.439_393_939_393_939,
                0.181_818_181_818_182,
            ),
            (
                6.0,
                1.25,
                0.227_272_727_272_727,
                0.515_151_515_151_515,
                0.257_575_757_575_758,
            ),
            (
                8.0,
                0.75,
                0.227_272_727_272_727,
                0.515_151_515_151_515,
                0.257_575_757_575_758,
            ),
            (
                9.0,
                0.5,
                0.113_636_363_636_364,
                0.515_151_515_151_515,
                0.371_212_121_212_121,
            ),
            (
                10.0,
                0.25,
                0.0,
                0.628_787_878_787_879,
                0.371_212_121_212_121,
            ),
        ];
        let out = curves(&time, &event, &weight, &[0; 10], None).unwrap();
        assert_eq!(out.causes, vec![1, 2]);
        assert_eq!(out.time.len(), reference.len());
        for (row, &(t, risk, free, c1, c2)) in reference.iter().enumerate() {
            assert!(close(out.time[row], t));
            assert!(close(out.n_risk[row], risk), "n_risk at {t}");
            assert!(close(out.event_free[row], free), "S at {t}");
            assert!(close(out.cif[0][row], c1), "F_1 at {t}");
            assert!(close(out.cif[1][row], c2), "F_2 at {t}");
        }
    }

    /// Groups are estimated separately and come back in code order, whatever
    /// the row order.
    #[test]
    fn groups_are_estimated_separately() {
        let time = [2.0, 1.0, 1.0, 3.0];
        let event = [1, 1, 0, 0];
        let out = curves(&time, &event, &[1.0; 4], &[1, 1, 0, 0], None).unwrap();
        assert_eq!(out.group, vec![0, 0, 1, 1]);
        assert_eq!(out.time, vec![1.0, 3.0, 1.0, 2.0]);
        assert_eq!(out.cif[0], vec![0.0, 0.0, 0.5, 1.0]);
        assert_eq!(out.n_at_risk, vec![2.0, 1.0, 2.0, 1.0]);
    }

    /// Hand-computed: two people from 0, a third only from 2.5, so at t = 1 the
    /// risk set is 2 and the hazard 1/2. At t = 3 the late entrant and the
    /// person leaving at 4 are at risk.
    #[test]
    fn a_late_entrant_is_not_at_risk_before_they_enter() {
        let entry = [0.0, 0.0, 2.5];
        let time = [1.0, 4.0, 3.0];
        let event = [1, 0, 1];
        let out = curves(&time, &event, &[1.0; 3], &[0; 3], Some(&entry)).unwrap();
        assert_eq!(out.time, vec![1.0, 3.0, 4.0]);
        assert_eq!(out.n_risk, vec![2.0, 2.0, 1.0]);
        assert!(close(out.cif[0][0], 0.5));
        assert!(close(out.cif[0][1], 0.75));
        assert!(close(out.event_free[1], 0.25));
    }

    /// Someone entering exactly at an exit time is not at risk at it.
    #[test]
    fn the_risk_set_is_strict_on_entry_and_inclusive_on_exit() {
        let entry = [0.0, 0.0, 2.0];
        let time = [2.0, 5.0, 6.0];
        let out = curves(&time, &[1, 0, 0], &[1.0; 3], &[0; 3], Some(&entry)).unwrap();
        assert_eq!(out.n_risk, vec![2.0, 2.0, 1.0]);
    }

    #[test]
    fn everyone_censored_has_no_cause_and_stays_event_free() {
        let out = curves(&[1.0, 2.0], &[0, 0], &[1.0; 2], &[0; 2], None).unwrap();
        assert!(out.causes.is_empty() && out.cif.is_empty());
        assert_eq!(out.event_free, vec![1.0, 1.0]);
        assert_eq!(out.n_event, vec![0.0, 0.0]);
    }
}

#[cfg(test)]
#[expect(deprecated, reason = "these tests are of the deprecated functions")]
mod tests {
    use super::{
        ArmColumns, Cell, Draw, Error, RowWindows, Scratch, advance, bootstrap_arm_incidence,
        scatter,
    };

    #[test]
    fn advance_steps_the_survival_and_each_incidence() {
        // Four at risk, one event of cause 0: S = 3/4, F = (1/4, 0).
        let mut survival = 1.0;
        let mut incidence = [0.0, 0.0];
        advance(&mut survival, 4.0, 1.0, &[1.0, 0.0], &mut incidence);
        assert_eq!(survival.to_bits(), 0.75_f64.to_bits());
        assert_eq!(
            incidence.map(f64::to_bits),
            [0.25_f64, 0.0].map(f64::to_bits)
        );
        // Three at risk, one event of each cause: S = 3/4 * 1/3 = 1/4, and each
        // incidence gains (3/4) (1/3) = 1/4.
        advance(&mut survival, 3.0, 2.0, &[1.0, 1.0], &mut incidence);
        assert_eq!(survival.to_bits(), (0.75_f64 * (1.0 - 2.0 / 3.0)).to_bits());
        assert_eq!(
            incidence[0].to_bits(),
            (0.25_f64 + 0.75 * 1.0 / 3.0).to_bits()
        );
        assert_eq!(incidence[1].to_bits(), (0.75_f64 * 1.0 / 3.0).to_bits());
        // Nobody at risk moves nothing.
        let before = (survival, incidence);
        advance(&mut survival, 0.0, 0.0, &[0.0, 0.0], &mut incidence);
        assert_eq!((survival, incidence), before);
    }

    fn cell(time: &[f64], event: &[i32], exposed: &[bool], family: &[u32]) -> Cell {
        let weight = vec![1.0; time.len()];
        let rows: Vec<u32> = (0..time.len() as u32).collect();
        Cell::gather(
            &rows,
            time,
            event,
            &weight,
            family,
            exposed,
            Draw::PerCell,
            false,
        )
        .expect("both arms present")
    }

    /// Worked by hand.
    ///
    /// The exposed arm is two people. At time 1 one has the outcome: 1 of 2 at
    /// risk, so the increment is `S(1-) * 1/2 = 1 * 0.5`. At time 2 the other is
    /// censored, which moves the number at risk and not the incidence.
    #[test]
    fn cumulative_incidence_matches_a_hand_computed_curve() {
        let c = cell(
            &[1.0, 2.0, 5.0],
            &[1, 0, 0],
            &[true, true, false],
            &[0, 1, 2],
        );
        let ages = [0.5, 1.0, 2.0, 3.0];
        let mut out = [0.0; 4];
        let mut scratch = Scratch::default();
        c.exposed
            .cif_at_ages(&[1.0, 1.0, 1.0], &ages, 1, &mut scratch, &mut out);

        assert!((out[0] - 0.0).abs() < 1e-12, "before any event: {out:?}");
        assert!((out[1] - 0.5).abs() < 1e-12, "at the event: {out:?}");
        assert!(
            (out[2] - 0.5).abs() < 1e-12,
            "censoring moves nothing: {out:?}"
        );
        assert!(
            (out[3] - 0.5).abs() < 1e-12,
            "flat past the last time: {out:?}"
        );
    }

    /// A competing event must hold the other cause's incidence below what
    /// one-minus-Kaplan-Meier would give.
    #[test]
    fn a_competing_event_holds_the_incidence_below_naive_survival() {
        let c = cell(
            &[1.0, 2.0, 3.0, 9.0],
            &[2, 1, 0, 0],
            &[true, true, true, false],
            &[0, 1, 2, 3],
        );
        let mut out = [0.0; 1];
        let mut scratch = Scratch::default();
        c.exposed
            .cif_at_ages(&[1.0; 4], &[3.0], 1, &mut scratch, &mut out);

        // S(2-) = 1 - 1/3 = 2/3, and the cause-1 event has 2 at risk:
        // F_1(3) = (2/3)(1/2) = 1/3, below the 1/2 that ignoring the competing
        // cause would give.
        assert!((out[0] - 1.0 / 3.0).abs() < 1e-12, "{out:?}");
        assert!(out[0] < 0.5);
    }

    /// The re-indexing a code shared across cells would get wrong.
    #[test]
    fn families_are_numbered_within_the_cell_not_the_cohort() {
        let time = [1.0, 2.0, 3.0, 4.0];
        let event = [1, 0, 1, 0];
        let weight = [1.0; 4];
        let family = [10, 20, 30, 40];
        let exposed = [true, false, true, false];
        let c = Cell::gather(
            &[0, 3],
            &time,
            &event,
            &weight,
            &family,
            &exposed,
            Draw::PerCell,
            false,
        )
        .expect("both arms present");

        assert_eq!(c.n_families, 2, "two families in the cell, not forty-one");
        let families = |arm: &super::Arm| arm.rows.iter().map(|row| row.family).collect::<Vec<_>>();
        assert_eq!(families(&c.exposed), vec![0], "renumbered from zero");
        assert_eq!(families(&c.comparator), vec![1], "renumbered from zero");
    }

    #[test]
    fn a_cell_with_one_arm_empty_is_not_an_estimate() {
        let time = [1.0, 2.0];
        let event = [1, 0];
        let weight = [1.0; 2];
        let family = [0, 1];
        let exposed = [true, true];
        assert!(
            Cell::gather(
                &[0, 1],
                &time,
                &event,
                &weight,
                &family,
                &exposed,
                Draw::PerCell,
                false
            )
            .is_none()
        );
    }

    /// The block the caller reads as `[age][replicate]`.
    #[test]
    fn a_replicate_lands_in_its_column_of_the_age_major_block() {
        let mut block = vec![0.0; 6];
        scatter(&[1.0, 2.0], 0, 3, &mut block);
        scatter(&[10.0, 20.0], 2, 3, &mut block);
        assert_eq!(block, vec![1.0, 0.0, 10.0, 2.0, 0.0, 20.0]);
    }

    #[test]
    fn a_malformed_row_list_is_refused_rather_than_read() {
        let refused = |rows: Vec<u32>, offsets: Vec<u32>, n_rows: usize| {
            RowWindows::new(rows, offsets, n_rows, "cell").err()
        };
        assert_eq!(
            refused(vec![0], vec![], 1),
            Some(Error::input("cell_offsets must not be empty"))
        );
        assert!(
            refused(vec![0, 1], vec![0, 2, 1], 2).is_some(),
            "decreasing"
        );
        assert!(refused(vec![0], vec![0, 2], 1).is_some(), "past the rows");
        assert_eq!(
            refused(vec![0, 5], vec![0, 2], 3),
            Some(Error::input("cell_rows indexes past the frame"))
        );
    }

    #[test]
    fn a_window_reads_its_own_slice_of_the_row_list() {
        let windows = RowWindows::new(vec![0, 2, 1, 2], vec![0, 2, 2, 4], 3, "cell").unwrap();
        assert_eq!(windows.len(), 3);
        assert_eq!(windows.get(0), &[0, 2]);
        assert_eq!(windows.get(1), &[] as &[u32]);
        assert_eq!(windows.get(2), &[1, 2]);
    }

    #[test]
    fn the_bootstrap_is_reproducible_and_flags_a_cell_with_one_arm() {
        let columns = ArmColumns::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1, 0, 2, 1, 0, 1],
            vec![1.0; 6],
            vec![0, 1, 2, 3, 4, 5],
            vec![true, true, true, false, false, false],
        )
        .unwrap();
        let cells =
            RowWindows::new(vec![0, 1, 2, 3, 4, 5, 0, 1], vec![0, 6, 8], 6, "cell").unwrap();
        let run = || {
            bootstrap_arm_incidence(
                &columns,
                &cells,
                &[7, 8],
                vec![4.0, 2.0],
                1,
                5,
                Draw::PerCell,
            )
            .unwrap()
        };
        let (exposed, comparator, estimable) = run();
        assert_eq!(estimable, vec![true, false]);
        assert_eq!(exposed.len(), 2 * 2 * 5);
        assert!(exposed[..10].iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(exposed[10..].iter().all(|&v| v.to_bits() == 0));
        assert!(comparator[10..].iter().all(|&v| v.to_bits() == 0));
        let again = run();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&exposed), bits(&again.0));
        assert_eq!(bits(&comparator), bits(&again.1));
    }

    #[test]
    fn seeds_and_windows_that_do_not_fit_are_refused() {
        let columns = ArmColumns::new(vec![1.0], vec![1], vec![1.0], vec![0], vec![true]).unwrap();
        let cells = RowWindows::new(vec![0], vec![0, 1], 1, "cell").unwrap();
        assert!(
            bootstrap_arm_incidence(&columns, &cells, &[], vec![1.0], 1, 2, Draw::PerCell).is_err()
        );
        let wide = RowWindows::new(vec![0], vec![0, 1], 2, "cell").unwrap();
        assert!(
            bootstrap_arm_incidence(&columns, &wide, &[1], vec![1.0], 1, 2, Draw::PerCell).is_err()
        );
        let none = RowWindows::new(vec![], vec![0], 1, "cell").unwrap();
        let (exposed, _, estimable) =
            bootstrap_arm_incidence(&columns, &none, &[], vec![1.0], 1, 2, Draw::Shared).unwrap();
        assert!(exposed.is_empty() && estimable.is_empty());
    }
}

#[cfg(test)]
#[expect(
    deprecated,
    reason = "the new entry points are compared with the deprecated ones"
)]
mod estimate_tests {
    use super::{
        AalenJohansenOptions, ArmColumns, BootstrapOptions, Curves, Draw, RowWindows, bootstrap,
        bootstrap_arm_incidence, curves, estimate,
    };
    use crate::surv::SurvData;
    use crate::timefix::TimeFix;

    const EXACT: AalenJohansenOptions = AalenJohansenOptions {
        timefix: TimeFix::Exact,
    };

    fn bits(values: &[f64]) -> Vec<u64> {
        values.iter().map(|v| v.to_bits()).collect()
    }

    fn same(a: &Curves, b: &Curves) {
        assert_eq!(a.causes, b.causes);
        assert_eq!(a.group, b.group);
        for (x, y) in [
            (&a.time, &b.time),
            (&a.n_risk, &b.n_risk),
            (&a.n_at_risk, &b.n_at_risk),
            (&a.n_event, &b.n_event),
            (&a.event_free, &b.event_free),
        ] {
            assert_eq!(bits(x), bits(y));
        }
        for (x, y) in a.cif.iter().zip(&b.cif) {
            assert_eq!(bits(x), bits(y));
        }
    }

    const TIME: [f64; 8] = [1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0];
    const EVENT: [i32; 8] = [1, 2, 0, 1, 1, 2, 0, 1];
    const WEIGHT: [f64; 8] = [1.0, 0.5, 2.0, 1.5, 0.25, 1.0, 1.0, 3.0];
    const START: [f64; 8] = [0.0, 0.5, 0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
    const GROUP: [u32; 8] = [0, 1, 0, 1, 0, 1, 0, 1];

    #[test]
    fn the_estimate_is_that_of_the_old_function_on_the_same_rows() {
        let data = SurvData::builder(&TIME, &EVENT)
            .start(&START)
            .n_codes(2)
            .weight(&WEIGHT)
            .stratum(&GROUP)
            .build()
            .unwrap();
        let new = estimate(&data, &EXACT).unwrap();
        let old = curves(&TIME, &EVENT, &WEIGHT, &GROUP, Some(&START)).unwrap();
        same(&new, &old);
        assert_eq!((new.n_rows, new.n_used), (8, 8));
    }

    #[test]
    fn strata_come_back_as_the_callers_codes() {
        let strata: Vec<u32> = GROUP.iter().map(|&g| 10 + 20 * g).collect();
        let data = SurvData::builder(&TIME, &EVENT)
            .n_codes(2)
            .stratum(&strata)
            .build()
            .unwrap();
        let new = estimate(&data, &EXACT).unwrap();
        let old = curves(&TIME, &EVENT, &[1.0; 8], &GROUP, None).unwrap();
        assert!(new.group.iter().all(|g| [10, 30].contains(g)));
        let relabelled: Vec<u32> = old.group.iter().map(|&g| 10 + 20 * g).collect();
        assert_eq!(new.group, relabelled);
        assert_eq!(bits(&new.event_free), bits(&old.event_free));
    }

    #[test]
    fn rows_of_weight_zero_are_not_used_and_are_counted() {
        let weight = [1.0, 0.0, 2.0, 1.5, 0.0, 1.0, 1.0, 3.0];
        let data = SurvData::builder(&TIME, &EVENT)
            .n_codes(2)
            .weight(&weight)
            .stratum(&GROUP)
            .build()
            .unwrap();
        let new = estimate(&data, &EXACT).unwrap();
        let keep = [0usize, 2, 3, 5, 6, 7];
        let pick = |c: &[f64]| keep.iter().map(|&i| c[i]).collect::<Vec<_>>();
        let old = curves(
            &pick(&TIME),
            &keep.iter().map(|&i| EVENT[i]).collect::<Vec<_>>(),
            &pick(&weight),
            &keep.iter().map(|&i| GROUP[i]).collect::<Vec<_>>(),
            None,
        )
        .unwrap();
        same(&new, &old);
        assert_eq!((new.n_rows, new.n_used), (8, 6));
    }

    #[test]
    fn every_code_is_a_cause_whether_observed_or_not() {
        let data = SurvData::builder(&TIME, &EVENT).n_codes(4).build().unwrap();
        let new = estimate(&data, &EXACT).unwrap();
        assert_eq!(new.causes, vec![1, 2, 3, 4]);
        assert_eq!(new.cif.len(), 4);
        assert!(new.cif[2].iter().chain(&new.cif[3]).all(|&v| v == 0.0));
        let old = curves(&TIME, &EVENT, &[1.0; 8], &[0; 8], None).unwrap();
        assert_eq!(old.causes, vec![1, 2]);
    }

    #[test]
    fn the_default_tie_rule_merges_times_that_differ_by_rounding() {
        let time = [1.0, 1.0 + 1e-12, 2.0];
        let status = [1, 0, 1];
        let data = SurvData::builder(&time, &status).build().unwrap();
        let fixed = estimate(&data, &AalenJohansenOptions::default()).unwrap();
        assert_eq!(fixed.time, vec![1.0, 2.0]);
        assert_eq!(fixed.n_at_risk, vec![3.0, 1.0]);
        let exact = estimate(&data, &EXACT).unwrap();
        assert_eq!(exact.time.len(), 3);
    }

    #[test]
    fn nothing_used_gives_no_rows() {
        let data = SurvData::builder(&TIME, &EVENT)
            .weight(&[0.0; 8])
            .n_codes(2)
            .build()
            .unwrap();
        let new = estimate(&data, &EXACT).unwrap();
        assert!(new.time.is_empty() && new.group.is_empty());
        assert_eq!((new.n_rows, new.n_used), (8, 0));
        assert_eq!(new.cif.len(), 2);
    }

    struct Fixture {
        time: Vec<f64>,
        event: Vec<i32>,
        weight: Vec<f64>,
        family: Vec<u32>,
        exposed: Vec<bool>,
        cells: RowWindows,
    }

    fn frame() -> Fixture {
        let n = 60;
        let mut rows = Vec::new();
        let mut offsets = vec![0u32];
        for (start, len) in [(0usize, 40usize), (10, 30), (20, 40)] {
            rows.extend((start..start + len).map(|i| i as u32));
            offsets.push(rows.len() as u32);
        }
        Fixture {
            time: (0..n).map(|i| 1.0 + ((i * 7) % 11) as f64).collect(),
            event: (0..n).map(|i| [0, 1, 2, 1][i % 4]).collect(),
            weight: (0..n).map(|i| 0.5 + (i % 3) as f64).collect(),
            family: (0..n).map(|i| ((i * 5) % 9) as u32).collect(),
            exposed: (0..n).map(|i| i % 3 != 0).collect(),
            cells: RowWindows::new(rows, offsets, n, "cell").unwrap(),
        }
    }

    #[test]
    fn the_bootstrap_draws_are_those_of_the_old_function() {
        let Fixture {
            time,
            event,
            weight,
            family,
            exposed,
            cells,
        } = frame();
        let seeds = [3, 5, 8];
        let ages = [0.5, 6.0, 3.0, 30.0];
        let data = SurvData::builder(&time, &event)
            .n_codes(2)
            .weight(&weight)
            .cluster(&family)
            .build()
            .unwrap();
        let columns = ArmColumns::new(
            time.clone(),
            event.clone(),
            weight.clone(),
            family.clone(),
            exposed.clone(),
        )
        .unwrap();
        for draw in [Draw::PerCell, Draw::Shared] {
            let opts = BootstrapOptions {
                n_replicates: 7,
                draw,
                timefix: TimeFix::Exact,
            };
            let new = bootstrap(&data, &exposed, &cells, &seeds, &ages, 1, &opts).unwrap();
            let old = bootstrap_arm_incidence(&columns, &cells, &seeds, ages.to_vec(), 1, 7, draw)
                .unwrap();
            assert_eq!(bits(&new.0), bits(&old.0));
            assert_eq!(bits(&new.1), bits(&old.1));
            assert_eq!(new.2, old.2);
        }
    }

    /// A row of weight 0 and the cluster it alone belongs to are not in the cell.
    #[test]
    fn the_bootstrap_leaves_out_rows_of_weight_zero() {
        let time = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = [1, 0, 2, 1, 0, 1];
        let weight = [1.0, 1.0, 2.0, 1.0, 0.5, 0.0];
        let family = [0, 1, 2, 0, 1, 3];
        let exposed = [true, true, true, false, false, false];
        let data = SurvData::builder(&time, &event)
            .n_codes(2)
            .weight(&weight)
            .cluster(&family)
            .build()
            .unwrap();
        let all = RowWindows::new((0..6).collect(), vec![0, 6], 6, "cell").unwrap();
        let opts = BootstrapOptions {
            n_replicates: 5,
            draw: Draw::PerCell,
            timefix: TimeFix::Exact,
        };
        let new = bootstrap(&data, &exposed, &all, &[9], &[2.0, 6.0], 1, &opts).unwrap();
        let columns = ArmColumns::new(
            time[..5].to_vec(),
            event[..5].to_vec(),
            weight[..5].to_vec(),
            family[..5].to_vec(),
            exposed[..5].to_vec(),
        )
        .unwrap();
        let first = RowWindows::new((0..5).collect(), vec![0, 5], 5, "cell").unwrap();
        let old =
            bootstrap_arm_incidence(&columns, &first, &[9], vec![2.0, 6.0], 1, 5, Draw::PerCell)
                .unwrap();
        assert_eq!(bits(&new.0), bits(&old.0));
        assert_eq!(bits(&new.1), bits(&old.1));
    }

    #[test]
    fn the_bootstrap_refuses_what_it_cannot_read() {
        let Fixture {
            time,
            event,
            family,
            exposed,
            cells,
            ..
        } = frame();
        let run = |data: &SurvData<'_>, exposed: &[bool], cause: u32| {
            bootstrap(
                data,
                exposed,
                &cells,
                &[1, 2, 3],
                &[1.0],
                cause,
                &BootstrapOptions::new(10, Draw::PerCell),
            )
        };
        let build = || SurvData::builder(&time, &event).n_codes(2);
        let with = build().cluster(&family).build().unwrap();
        assert!(run(&with, &exposed, 1).is_ok());
        assert!(run(&with, &exposed[1..], 1).is_err(), "exposed length");
        assert!(run(&with, &exposed, 0).is_err(), "censoring is no cause");
        assert!(run(&with, &exposed, 3).is_err(), "cause above n_codes");
        let without = build().build().unwrap();
        assert!(run(&without, &exposed, 1).is_err(), "no cluster column");
        let start = vec![0.0; time.len()];
        let entry = build().cluster(&family).start(&start).build().unwrap();
        assert!(run(&entry, &exposed, 1).is_err(), "delayed entry");
    }
}

#[cfg(test)]
#[expect(
    clippy::cast_possible_wrap,
    reason = "the sizes in these tests are small"
)]
mod variance_tests {
    use super::{AalenJohansenOptions, Error, VarianceOptions, variance};
    use crate::surv::SurvData;

    /// With no censoring and one cause the estimate is the empirical
    /// distribution function, and the influence of a subject is
    /// `(1[T <= t] - F(t)) / n`, so the variance is `F (1 - F) / n`. Worked by
    /// hand at four exit times for `n = 8`, with ties.
    #[test]
    fn without_censoring_the_variance_is_binomial() {
        let stop = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0];
        let status = [1; 8];
        let data = SurvData::builder(&stop, &status)
            .n_codes(1)
            .build()
            .unwrap();
        let got = variance(
            &data,
            &AalenJohansenOptions::default(),
            &VarianceOptions::default(),
        )
        .unwrap();
        let cumulative = [1.0, 3.0, 4.0, 5.0, 7.0, 8.0];
        for (t, &count) in cumulative.iter().enumerate() {
            let f = count / 8.0;
            let expected = (f * (1.0 - f) / 8.0_f64).sqrt();
            assert!((got.std_cif[0][t] - expected).abs() < 1e-15, "cif at {t}");
            assert!(
                (got.std_event_free[t] - expected).abs() < 1e-15,
                "event free at {t}"
            );
            assert!((got.cif[0][t] - f).abs() < 1e-15);
        }
    }

    /// Two rows of one id are one cluster: their influences are added before
    /// they are squared, so a subject followed in two pieces has the variance of
    /// the subject, which is not that of two independent rows.
    #[test]
    fn rows_of_one_id_are_one_cluster() {
        // Subject 0 is at risk on (0, 2] and has an event of cause 1 on (2, 3].
        let start = [0.0, 2.0, 0.0, 0.0];
        let stop = [2.0, 3.0, 2.0, 3.0];
        let status = [0, 1, 1, 0];
        let id = [0, 0, 1, 2];
        let whole = SurvData::builder(&stop, &status)
            .start(&start)
            .id(&id)
            .n_codes(1)
            .build()
            .unwrap();
        let rows = SurvData::builder(&stop, &status)
            .start(&start)
            .n_codes(1)
            .build()
            .unwrap();
        let opts = AalenJohansenOptions::default();
        let by_id = variance(&whole, &opts, &VarianceOptions::default()).unwrap();
        let by_row = variance(&rows, &opts, &VarianceOptions::default()).unwrap();
        assert_eq!(by_id.groups[0].clusters, vec![0, 1, 2]);
        assert_eq!(by_row.groups[0].clusters, vec![0, 1, 2, 3]);
        // the estimates are the same, the errors are not
        assert_eq!(by_id.cif, by_row.cif);
        assert!(
            by_id.std_cif[0]
                .iter()
                .zip(&by_row.std_cif[0])
                .any(|(a, b)| (a - b).abs() > 1e-6)
        );
    }

    /// The influence at the times asked for squares and sums to the variance,
    /// for more clusters than one block holds and a last block that is not full.
    #[test]
    fn the_influence_values_sum_to_the_variance_across_blocks() {
        let n = 3 * super::influence::BLOCK + 17;
        let stop: Vec<f64> = (0..n).map(|i| 1.0 + ((i * 7919) % 23) as f64).collect();
        let outcome: Vec<i32> = (0..n).map(|i| ((i * 31) % 4) as i32).collect();
        let weight: Vec<f64> = (0..n).map(|i| 0.5 + ((i * 13) % 5) as f64 * 0.25).collect();
        let data = SurvData::builder(&stop, &outcome)
            .n_codes(3)
            .weight(&weight)
            .build()
            .unwrap();
        let times: Vec<f64> = (1..=24).map(f64::from).collect();
        let v = VarianceOptions {
            horizons: vec![],
            influence_times: times.clone(),
        };
        let got = variance(&data, &AalenJohansenOptions::default(), &v).unwrap();
        let g = &got.groups[0];
        assert_eq!(g.clusters.len(), n);
        let width = 4;
        for (t, &time) in got.time.iter().enumerate() {
            let k = times
                .iter()
                .position(|&x| (x - time).abs() < 1e-12)
                .unwrap();
            for s in 0..width {
                let sum: f64 = (0..n)
                    .map(|c| g.influence[(c * times.len() + k) * width + s].powi(2))
                    .sum();
                let reported = if s == 0 {
                    got.std_event_free[t]
                } else {
                    got.std_cif[s - 1][t]
                };
                assert!(
                    (sum.sqrt() - reported).abs() <= 1e-13 * reported.max(1e-3),
                    "t={time} s={s}: {} against {reported}",
                    sum.sqrt()
                );
            }
        }
    }

    #[test]
    fn unsorted_requests_are_refused() {
        let stop = [1.0, 2.0];
        let status = [1, 0];
        let data = SurvData::builder(&stop, &status)
            .n_codes(1)
            .build()
            .unwrap();
        let opts = AalenJohansenOptions::default();
        for v in [
            VarianceOptions {
                horizons: vec![2.0, 1.0],
                influence_times: vec![],
            },
            VarianceOptions {
                horizons: vec![],
                influence_times: vec![f64::NAN],
            },
        ] {
            assert!(matches!(variance(&data, &opts, &v), Err(Error::Input(_))));
        }
    }
}
