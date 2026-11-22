//! The Fine-Gray subdistribution hazard model for one cause, fitted on a design
//! matrix, with the Kaplan-Meier estimate of the censoring distribution it
//! needs.
//!
//! The model is fitted as a weighted Cox model on rows built from the data \[1\],
//! so the fit itself is [`crate::cox::fit`] and the result is a
//! [`CoxModelFit`]: coefficients, model-based and robust covariance, and the
//! Wald, score and likelihood ratio tests. This module builds the rows.
//!
//! # Input
//!
//! [`rows`], [`risk_rows`] and [`fit`] take a [`SurvData`]. Its status is the
//! cause code, with 0 for censoring, the cause of interest is a code in
//! `1..=n_codes`, and every other nonzero code is a competing event. Its
//! stop is the time, its weight the case weight and, for [`fit`], its cluster
//! the cluster of the robust variance, as the dense codes the type holds. The
//! covariates of [`fit`] are the columns of a matrix with one row per row of
//! the data. The model follows every individual from time 0, so data with a
//! start column is an error.
//!
//! The tie rule of [`FineGrayOptions::timefix`] and [`RowsOptions::timefix`]
//! is applied to the times first. It defaults to the rule of
//! `survival::aeqSurv`, which `finegray` and `coxph` apply, so times that
//! differ by rounding error are one time. Integer times are unaffected.
//!
//! Rows of weight 0 are individuals who do not take part. They are left out
//! before the censoring distribution is estimated, so they have no rows and no
//! say in `G` or in the last time the rows run to. The rows keep the row
//! numbers of the data. The fit reports the rows of the form it used in
//! `n_rows`, and `n_used` is the same, since the rows of the form fitted have
//! positive weight.
//!
//! # Algorithm
//!
//! Individual `i` has a time `x_i`, a cause code `c_i` with 0 for censoring, a
//! weight `w_i` and a covariate. The cause of interest is `k`. The subdistribution
//! hazard of `k` keeps an individual with a competing event, a cause other than
//! `k` and 0, in the risk set after the event, because that individual could
//! no longer have had the event `k` but would have been observed for it if the
//! competing event had not happened \[1\]. The individual's weight after the
//! event decreases with the chance that a censoring would have removed them:
//!
//! ```text
//! weight at time s >= x_i:   w_i G(s) / G(x_i-)
//! ```
//!
//! where `G` is the Kaplan-Meier estimate of the censoring distribution \[2\],
//! censoring being the event and every failure of any cause being censored for
//! it. The risk set at a censoring time `t` is the weight with a time strictly
//! after `t` plus the weight censored at `t`, so a failure and a censoring at
//! the same time are ordered failure first. Using `time >= t` would add every
//! tied failure to the risk set and change every weight derived from `G`.
//! `G(t-)` is `G` just before `t`, which is 1 before the first censoring.
//!
//! The rows are counting-process intervals. [`rows`] writes one
//! row per individual and interval, which is the form `survival::finegray` in R
//! produces. Individuals with the cause of interest or censored have the single
//! row `(0, x_i]`. An individual with a competing event has rows at full weight
//! to the first censoring time at or after the event, then at
//! `w_i G(max(s, x_i)) / G(x_i-)` for each interval `(s, s']` between later
//! censoring times, ending at the largest observed time. The denominator is `G`
//! before the event, so an event that ties with a censoring time loses the
//! weight of that censoring. `G` is read at `max(s, x_i)` because `G(s)` alone
//! gives weights above 1 when the event falls after censoring has begun. Rows of
//! weight zero are left out.
//!
//! [`risk_rows`] sums the competing-event rows that share an
//! interval and a covariate value. These rows all end without an event, so
//! they enter the partial likelihood only through weighted sums, and adding
//! them is exact. The weight on the interval opening at censoring time `c_j`
//! factorises as `G(c_j) w_i / G(x_i-)`, so one prefix sum of `w_i / G(x_i-)`
//! over the event times, per covariate level, gives every interval.
//! [`fit`] fits on these collapsed rows, with the rows of the design matrix
//! as levels, or on the expanded rows when the data has clusters, because the
//! robust variance needs each individual's own rows.
//!
//! # Agreement with R
//!
//! `tests/r_fine_gray.rs` compares with `survival::finegray` followed by
//! `coxph` on its rows, with weights `fgwt` and, for the robust variance, the
//! individual as cluster, and with `cmprsk::crr`. A factor is written as
//! indicator columns, as `model.matrix` does.
//!
//! - `survival::finegray` writes no row whose interval holds no event time of
//!   the cause, because such a row enters no risk set. [`rows`] writes every
//!   interval between censoring times. Every row of R is a row of [`rows`]
//!   with the same start, stop, status and weight, and every row of [`rows`]
//!   that R leaves out holds no event time of the cause.
//! - Case weights enter the censoring distribution here, which is then the
//!   weighted Kaplan-Meier estimate, as if each individual were repeated
//!   `weight` times. `finegray(weights =)` estimates it without the weights
//!   and multiplies the row weights by them afterwards, so its rows are
//!   those of unit weights times the case weight. The weighted estimate is the
//!   one the test compares with, through `finegray` on repeated individuals.
//! - `crr` estimates the censoring distribution with `survfit` on the
//!   censoring indicator, which keeps a failure at a censoring time in the risk
//!   set of that censoring. Here, as in `finegray`, the failure is taken out
//!   first. The two coefficients agree, to the tolerance `crr` stops at, when
//!   no failure ties with a censoring, with Breslow's approximation for
//!   ties among failures in both, and differ otherwise. `crr` has no Efron
//!   option.
//! - `crr` fits several covariates as well, and the comparison on a matrix
//!   has the same conditions.
//! - The standard errors of `crr` include the variance from estimating the
//!   censoring distribution. The robust standard error here is the sandwich
//!   of `coxph` on the rows, which treats the weights as known, and does not.
//!
//! # Edge cases
//!
//! Cause 0 is censoring and is refused as a cause of interest. An individual
//! whose `G(x_i-)` is zero contributes no weight after the event. The estimates
//! of [`rows`] and [`risk_rows`] are the same, and `n_rows` of the fit counts the
//! rows of the form used.
//!
//! # References
//!
//! 1. Fine JP, Gray RJ. A proportional hazards model for the subdistribution
//!    of a competing risk. J Am Stat Assoc 1999;94(446):496-509.
//!    doi:10.1080/01621459.1999.10474144
//! 2. Kaplan EL, Meier P. Nonparametric estimation from incomplete
//!    observations. J Am Stat Assoc 1958;53(282):457-81.
//!    doi:10.1080/01621459.1958.10501452
//! 3. Therneau TM, Crowson C, Atkinson E. Multi-state models and competing
//!    risks. Vignette of the R package survival.
//! 4. Gray RJ. cmprsk: subdistribution analysis of competing risks. R package.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the entry points check that the row count fits in u32"
)]

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;

use epistat_linalg::{Mat, MatRef};

use crate::check::{check_lengths, check_row_count};
use crate::cox::{CoxModelFit, CoxOptions, Ties};
use crate::error::{Error, Result};
use crate::step::Steps;
use crate::surv::SurvData;
use crate::timefix::TimeFix;

/// The Kaplan-Meier estimate of the censoring distribution `G`, as a step
/// function.
///
/// Censoring is the event and every failure, of any cause, is censored for it.
/// The risk set at a censoring time `t` is the weight with a time strictly
/// after `t` plus the weight censored at `t`, so a failure and a censoring at
/// the same time are ordered failure first. Using `time >= t` would add every
/// tied failure to the risk set and change every weight derived from `G`.
///
/// The curve is 1 before the first censoring. Its times are also the times the
/// Fine-Gray weights change, so the rows are split there.
// Exact `==` groups rows that share a time.
#[expect(clippy::float_cmp, reason = "rows that share a time are grouped")]
fn censoring_curve(time: &[f64], event: &[i32], weight: &[f64]) -> Steps {
    let mut order: Vec<usize> = (0..time.len()).collect();
    order.sort_by(|&a, &b| time[a].total_cmp(&time[b]));

    // Per distinct time: its value, total weight, censored weight, and
    // whether anyone was censored there.
    let mut blocks: Vec<(f64, f64, f64, bool)> = Vec::new();
    for &i in &order {
        let censored = event[i] == 0;
        match blocks.last_mut() {
            Some(block) if block.0 == time[i] => {
                block.1 += weight[i];
                if censored {
                    block.2 += weight[i];
                    block.3 = true;
                }
            }
            _ => blocks.push((
                time[i],
                weight[i],
                if censored { weight[i] } else { 0.0 },
                censored,
            )),
        }
    }

    // Weight strictly after each block, as a suffix total.
    let mut after = vec![0.0f64; blocks.len()];
    let mut running = 0.0f64;
    for (slot, block) in after.iter_mut().zip(&blocks).rev() {
        *slot = running;
        running += block.1;
    }

    let mut survival = 1.0f64;
    let mut times = Vec::new();
    let mut steps = Vec::new();
    for (block, &later) in blocks.iter().zip(&after) {
        let (t, _, censored_here, any_censored) = *block;
        if !any_censored {
            continue;
        }
        let at_risk = later + censored_here;
        if at_risk > 0.0 && censored_here > 0.0 {
            survival *= 1.0 - censored_here / at_risk;
        }
        times.push(t);
        steps.push(survival);
    }
    Steps {
        time: times,
        first: 1.0,
        value: steps,
    }
}

/// Counting-process rows for a Cox fit. `row` is an input row: the individual
/// for [`rows`], and for [`risk_rows`] the individual or any individual with
/// the covariate value the collapsed row carries.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Rows {
    /// The input row each row comes from.
    pub row: Vec<u32>,
    /// Interval start.
    pub start: Vec<f64>,
    /// Interval stop.
    pub stop: Vec<f64>,
    /// 1 when the interval ends in the cause of interest, 0 otherwise.
    pub event: Vec<i32>,
    /// Weight of the row.
    pub weight: Vec<f64>,
}

impl Rows {
    fn push(&mut self, row: usize, start: f64, stop: f64, event: bool, weight: f64) {
        self.row.push(row as u32);
        self.start.push(start);
        self.stop.push(stop);
        self.event.push(i32::from(event));
        self.weight.push(weight);
    }
}

/// A competing event: its time, its weight and the weight over `G` just
/// before the time.
type TiedEvent = (f64, f64, f64);

/// What every Fine-Gray construction starts from: the data, the cause of
/// interest and the Kaplan-Meier estimate of the censoring distribution.
///
/// The columns are slices of the individuals that take part, the rows of
/// positive weight of the data.
struct Expansion<'a> {
    time: &'a [f64],
    event: &'a [i32],
    weight: &'a [f64],
    cause: i32,
    curve: Steps,
    horizon: f64,
}

impl<'a> Expansion<'a> {
    /// The data of one analysis: one `time`, cause code `event` (0 for
    /// censoring) and `weight` per individual, and the `cause` of interest.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the slices differ in length or `cause` is 0.
    fn new(time: &'a [f64], event: &'a [i32], weight: &'a [f64], cause: i32) -> Result<Self> {
        check_lengths(&[
            ("time", time.len()),
            ("event", event.len()),
            ("weight", weight.len()),
        ])?;
        check_row_count(time.len())?;
        if cause == 0 {
            return Err(Error::input("cause 0 is censoring, not a cause"));
        }
        Ok(Self {
            curve: censoring_curve(time, event, weight),
            horizon: time.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            time,
            event,
            weight,
            cause,
        })
    }

    fn competing(&self, i: usize) -> bool {
        self.event[i] != self.cause && self.event[i] != 0
    }

    /// One row `(0, x_i]` for each individual with the cause of interest or
    /// censored. There is nothing to expand for them.
    fn direct_rows(&self, rows: &mut Rows) {
        for i in 0..self.time.len() {
            if !self.competing(i) {
                rows.push(
                    i,
                    0.0,
                    self.time[i],
                    self.event[i] == self.cause,
                    self.weight[i],
                );
            }
        }
    }

    /// One row per individual and interval, as `survival::finegray` writes it.
    ///
    /// A competing-event individual runs at full weight to the first
    /// censoring time at or after the event and at `w * G(max(start, x)) /
    /// G(x-)` after it, split at every later censoring time and ending at the
    /// largest observed time. The denominator is `G` before the event, so an
    /// event that ties with a censoring time loses the weight of that
    /// censoring, as in `survival::finegray`. `G` is read at `max(start, x)`
    /// because `G(start)` alone gives weights above 1 when the event falls
    /// after censoring has begun. Rows of weight zero are left out.
    fn expand(&self) -> Rows {
        let mut rows = Rows::default();
        for i in 0..self.time.len() {
            let (t, w) = (self.time[i], self.weight[i]);
            if !self.competing(i) {
                rows.push(i, 0.0, t, self.event[i] == self.cause, w);
                continue;
            }
            let g_before_event = self.curve.before(t);
            let weight_from = |previous: f64| {
                if previous < t {
                    w
                } else if g_before_event > 0.0 {
                    w * (self.curve.at(previous) / g_before_event)
                } else {
                    0.0
                }
            };
            let first = self.curve.time.partition_point(|&c| c < t);
            let mut previous = 0.0f64;
            for &boundary in &self.curve.time[first..] {
                let weight_here = weight_from(previous);
                if weight_here > 0.0 {
                    rows.push(i, previous, boundary, false, weight_here);
                }
                previous = boundary;
            }
            if previous < self.horizon {
                let weight_here = weight_from(previous);
                if weight_here > 0.0 {
                    rows.push(i, previous, self.horizon, false, weight_here);
                }
            }
        }
        rows
    }

    /// The expansion with competing-event rows summed per interval and level.
    ///
    /// The weight on the interval opening at censoring time `c_j` factorises as
    /// `G(c_j) * w_i / G(x_i-)`, so one prefix sum of `w_i / G(x_i-)` over event
    /// times, per level, gives every interval. `level` codes the covariate;
    /// levels are emitted in order of their first competing-event individual.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when `level` does not hold one code per individual.
    fn risk_rows(&self, level: &[u32]) -> Result<Rows> {
        check_lengths(&[("time", self.time.len()), ("level", level.len())])?;
        let mut rows = Rows::default();
        self.direct_rows(&mut rows);

        // Per level: (event time, w, w / G(event time-)) and a representative
        // row.
        let mut slot_of: HashMap<u32, usize> = HashMap::new();
        let mut by_level: Vec<(usize, Vec<TiedEvent>)> = Vec::new();
        for (i, &code) in level.iter().enumerate() {
            if !self.competing(i) {
                continue;
            }
            let g_before_event = self.curve.before(self.time[i]);
            if g_before_event <= 0.0 {
                // Every weight after the event is zero.
                continue;
            }
            let slot = *slot_of.entry(code).or_insert_with(|| {
                by_level.push((i, Vec::new()));
                by_level.len() - 1
            });
            by_level[slot].1.push((
                self.time[i],
                self.weight[i],
                self.weight[i] / g_before_event,
            ));
        }

        let censoring = &self.curve.time;
        for (representative, mut entries) in by_level {
            entries.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
            let event_times: Vec<f64> = entries.iter().map(|entry| entry.0).collect();
            let mut prefix = Vec::with_capacity(entries.len() + 1);
            prefix.push(0.0f64);
            for &(_, _, ratio) in &entries {
                prefix.push(prefix[prefix.len() - 1] + ratio);
            }

            // The leading interval, from 0 to the first censoring time at or
            // after the event, at the individual's own weight. Entries are in
            // time order, so a shared stop is always the last.
            let mut leading: Vec<(f64, f64)> = Vec::new();
            for &(t, weight, _) in &entries {
                if weight <= 0.0 {
                    continue;
                }
                let index = censoring.partition_point(|&c| c < t);
                let stop = censoring.get(index).copied().unwrap_or(self.horizon);
                match leading.last_mut() {
                    Some(last) if last.0.total_cmp(&stop) == Ordering::Equal => last.1 += weight,
                    _ => leading.push((stop, weight)),
                }
            }
            for (stop, weight) in leading {
                rows.push(representative, 0.0, stop, false, weight);
            }

            // One row per interval between censoring times, carrying everyone
            // whose event is at or before the time the interval opens. An
            // event on a censoring time has its leading interval end there,
            // so the interval that opens there is its first reduced one.
            for (position, &boundary) in censoring.iter().enumerate() {
                let stop = censoring.get(position + 1).copied().unwrap_or(self.horizon);
                if stop <= boundary {
                    continue;
                }
                let covered = prefix[event_times.partition_point(|&v| v <= boundary)];
                let weight = self.curve.at(boundary) * covered;
                if weight > 0.0 {
                    rows.push(representative, boundary, stop, false, weight);
                }
            }
        }
        Ok(rows)
    }
}

/// Codes for the distinct values of a float column, in order of first
/// appearance, by bit pattern.
#[must_use]
pub fn float_codes(values: &[f64]) -> Vec<u32> {
    let mut codes: HashMap<u64, u32> = HashMap::new();
    values
        .iter()
        .map(|value| {
            let next = codes.len() as u32;
            *codes.entry(value.to_bits()).or_insert(next)
        })
        .collect()
}

/// Codes for the distinct rows of a matrix, in order of first appearance, by
/// bit pattern.
fn row_codes(x: MatRef<'_, f64>) -> Vec<u32> {
    let mut codes: HashMap<Vec<u64>, u32> = HashMap::new();
    (0..x.nrows())
        .map(|i| {
            let next = codes.len() as u32;
            let key: Vec<u64> = (0..x.ncols()).map(|j| x[(i, j)].to_bits()).collect();
            *codes.entry(key).or_insert(next)
        })
        .collect()
}

/// The options of [`rows`] and [`risk_rows`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RowsOptions {
    /// How times that differ by rounding error are treated. The default is the
    /// rule of `survival::aeqSurv`, which `finegray` applies.
    pub timefix: TimeFix,
}

/// The options of [`fit`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FineGrayOptions {
    /// The cause of interest, a status code in `1..=n_codes` of the data.
    pub cause: u32,
    /// How tied events enter the partial likelihood.
    pub ties: Ties,
    /// How times that differ by rounding error are treated. The default is the
    /// rule of `survival::aeqSurv`, which `coxph` applies.
    pub timefix: TimeFix,
    /// The relative change of the log partial likelihood at which the
    /// iteration stops, `eps` of `coxph.control`. It must be positive.
    pub eps: f64,
    /// The largest number of trial points, `iter.max` of `coxph.control`.
    pub iter_max: u32,
}

impl FineGrayOptions {
    /// The options for `cause` with the defaults of `coxph`: Efron's
    /// approximation for ties, the `aeqSurv` rule for times and the
    /// convergence limits of [`CoxOptions::default`].
    #[must_use]
    pub fn new(cause: u32) -> Self {
        let cox = CoxOptions::default();
        Self {
            cause,
            ties: cox.ties,
            timefix: cox.timefix,
            eps: cox.eps,
            iter_max: cox.iter_max,
        }
    }
}

/// `values` of the used individuals, or all of them when `index` is `None`.
fn gather<'a, T: Copy>(values: &'a [T], index: Option<&[u32]>) -> Cow<'a, [T]> {
    match index {
        None => Cow::Borrowed(values),
        Some(index) => Cow::Owned(index.iter().map(|&i| values[i as usize]).collect()),
    }
}

/// Run `body` on the Fine-Gray construction for `data` and `cause`, with the
/// tie rule applied and the rows of weight zero left out.
///
/// `body` also gets the positions of the used rows in `data`, `None` when all
/// are used, to gather its own columns by.
fn with_expansion<T>(
    data: &SurvData<'_>,
    cause: u32,
    timefix: TimeFix,
    body: impl FnOnce(&Expansion<'_>, Option<&[u32]>) -> Result<T>,
) -> Result<T> {
    if data.followup()?.start().is_some() {
        return Err(Error::input(
            "Fine-Gray follows every individual from time 0, so data with a start column is not supported",
        ));
    }
    let code = i32::try_from(data.cause(cause)?.code())
        .expect("SurvData::cause checked that the code is a status");
    let fixed = data.clone().with_timefix(timefix)?;
    let surv = fixed.followup()?;
    let index = fixed.used_index();
    let time = gather(surv.stop(), index.as_deref());
    let event = gather(surv.status(), index.as_deref());
    let weight: Cow<'_, [f64]> = match fixed.weight() {
        None => Cow::Owned(vec![1.0; fixed.n_rows()]),
        Some(weight) => gather(weight, index.as_deref()),
    };
    let expansion = Expansion::new(&time, &event, &weight, code)?;
    body(&expansion, index.as_deref())
}

/// Point the rows at the rows of the data they came from.
fn original_rows(mut rows: Rows, index: Option<&[u32]>) -> Rows {
    if let Some(index) = index {
        for row in &mut rows.row {
            *row = index[*row as usize];
        }
    }
    rows
}

/// The counting-process rows of the Fine-Gray model for `cause`, one row per
/// individual and interval, as `survival::finegray` writes them.
///
/// The cause is a status code of `data`, a code other than 0 and another
/// cause is a competing event. The tie rule of `opts` is applied to the times
/// first, and rows of weight 0 are left out of the data, of the censoring
/// distribution and of the rows. [`Rows::row`] is the row of `data` each row
/// comes from.
///
/// # Errors
///
/// [`Error::Input`] when `data` has a start column, when `cause` is not a
/// status code of `data`, or as for [`SurvData::with_timefix`].
pub fn rows(data: &SurvData<'_>, cause: u32, opts: &RowsOptions) -> Result<Rows> {
    with_expansion(data, cause, opts.timefix, |expansion, index| {
        Ok(original_rows(expansion.expand(), index))
    })
}

/// The expansion of [`rows`] with the competing-event rows summed per
/// interval and covariate level.
///
/// `level` holds one code per row of `data`, the covariate level of the row.
/// The rows are the ones a fit on a covariate with those levels uses.
/// [`Rows::row`] is the first row of `data` of the level that has a
/// competing event, and levels are emitted in order of that first row.
///
/// # Errors
///
/// [`Error::Input`] as for [`rows`], and when `level` does not hold one code
/// per row of `data`.
pub fn risk_rows(
    data: &SurvData<'_>,
    cause: u32,
    level: &[u32],
    opts: &RowsOptions,
) -> Result<Rows> {
    check_lengths(&[("stop", data.n_rows()), ("level", level.len())])?;
    with_expansion(data, cause, opts.timefix, |expansion, index| {
        let level = gather(level, index);
        Ok(original_rows(expansion.risk_rows(&level)?, index))
    })
}

/// Fit the subdistribution hazard model of a cause on a design matrix.
///
/// `x` holds the covariates, one row per row of `data` and one column per
/// covariate, with no intercept column. A factor is its indicator columns. The
/// cause is [`FineGrayOptions::cause`], a status code of `data`. Rows of
/// weight 0 are dropped, the tie rule of `opts` is applied to the times, and
/// every individual is at risk from time 0. The weighted Cox model with delayed
/// entry is fitted to the rows of the model by [`crate::cox::fit`], so the
/// result is that of the Cox fit: coefficients, covariances, tests, aliased
/// columns and the convergence report.
///
/// The fit is on the collapsed rows ([`risk_rows`], with the distinct rows of
/// `x` as levels). When `data` has clusters it is on the uncollapsed rows
/// ([`rows`]), because the robust variance needs each individual's rows, and
/// clusters are the dense codes of `data`. The robust covariance is then the
/// sandwich of the Cox fit on those rows, which treats the weights as known.
/// The estimate is the same in both forms. `n_rows` and `n_used` of the result
/// count the rows of the form used.
///
/// # Errors
///
/// [`Error::Input`] when `x` has no column, does not hold one row per row of
/// `data` or has a value that is not finite, when `opts.eps` is not positive
/// and finite, and as for [`rows`].
pub fn fit(data: &SurvData<'_>, x: MatRef<'_, f64>, opts: &FineGrayOptions) -> Result<CoxModelFit> {
    check_lengths(&[("stop", data.n_rows()), ("x", x.nrows())])?;
    if x.ncols() == 0 {
        return Err(Error::input("x has no columns"));
    }
    if (0..x.ncols()).any(|j| (0..x.nrows()).any(|i| !x[(i, j)].is_finite())) {
        return Err(Error::input("x: a value is not finite"));
    }
    with_expansion(data, opts.cause, opts.timefix, |expansion, index| {
        let source = |row: u32| index.map_or(row as usize, |index| index[row as usize] as usize);
        let rows = if data.cluster().is_some() {
            expansion.expand()
        } else {
            let used = Mat::from_fn(expansion.time.len(), x.ncols(), |i, j| {
                x[(source(i as u32), j)]
            });
            expansion.risk_rows(&row_codes(used.as_ref()))?
        };
        let design = Mat::from_fn(rows.row.len(), x.ncols(), |k, j| {
            x[(source(rows.row[k]), j)]
        });
        let cluster: Option<Vec<u32>> = data
            .cluster()
            .map(|cluster| rows.row.iter().map(|&row| cluster[source(row)]).collect());
        let mut builder = SurvData::builder(&rows.stop, &rows.event)
            .n_codes(1)
            .start(&rows.start)
            .weight(&rows.weight);
        if let Some(cluster) = &cluster {
            builder = builder.cluster(cluster);
        }
        // The tie rule is applied to the individuals' times above; the rows
        // carry those times and the censoring times, so there is nothing to
        // merge.
        crate::cox::fit(
            &builder.build()?,
            design.as_ref(),
            &CoxOptions {
                ties: opts.ties,
                timefix: TimeFix::Exact,
                eps: opts.eps,
                iter_max: opts.iter_max,
                ..CoxOptions::default()
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use epistat_linalg::Mat;

    use super::{
        Expansion, FineGrayOptions, Rows, RowsOptions, censoring_curve, fit, risk_rows, rows,
    };
    use crate::cox::{CoxModelFit, CoxOptions, Ties};
    use crate::surv::SurvData;
    use crate::timefix::TimeFix;

    // Two groups, two causes, with ties within a cause, across causes, and
    // between a censoring time and an event time.
    const TIME: [f64; 20] = [
        2., 3., 3., 5., 6., 8., 9., 9., 11., 12., 1., 4., 4., 5., 7., 7., 10., 10., 13., 14.,
    ];
    const EVENT: [i32; 20] = [1, 2, 1, 0, 1, 2, 1, 0, 2, 1, 2, 1, 1, 0, 2, 1, 1, 0, 2, 1];

    fn group() -> Vec<f64> {
        (0..20).map(|i| if i < 10 { 1.0 } else { 0.0 }).collect()
    }

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() < tolerance
    }

    /// Subject 1 of the R example (time 3, competing event) is split at the
    /// censoring times after it, 5, 9 and 10, and runs to 14. Intervals and
    /// weights from `survival::finegray(..., etype = "c1")`, R 4.6.1,
    /// survival 3.8.6, which has 30 rows in all.
    #[test]
    fn expansion_matches_r() {
        let weight = [1.0; 20];
        let data = Expansion::new(&TIME, &EVENT, &weight, 1).unwrap();
        let rows = data.expand();
        assert_eq!(rows.row.len(), 30);
        let subject: Vec<usize> = (0..rows.row.len()).filter(|&k| rows.row[k] == 1).collect();
        let stops: Vec<f64> = subject.iter().map(|&k| rows.stop[k]).collect();
        assert_eq!(stops, [5.0, 9.0, 10.0, 14.0]);
        let expected = [
            1.0,
            0.857_142_857_142_857_1,
            0.734_693_877_551_020_3,
            0.587_755_102_040_816_2,
        ];
        for (&k, want) in subject.iter().zip(expected) {
            assert!(close(rows.weight[k], want, 1e-12));
        }
        assert!(rows.weight.iter().all(|&w| w > 0.0 && w <= 1.0 + 1e-12));
    }

    /// A competing event that ties with a censoring time keeps the weight
    /// `G(c) / G(x-)` after it. `survival::finegray(Surv(time, cf) ~ 1,
    /// etype = "a")` on times 1 to 6 with causes 0, 2, 0, 1, 0, 1, 1 gives
    /// the second individual the rows `(0, 2]` at weight 1, `(2, 4]` at 0.8
    /// and `(4, 6]` at 0.5333333333, R 4.6.1, survival 3.8.6.
    #[test]
    fn a_competing_event_tied_with_censoring_matches_r() {
        let time = [1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let event = [0, 2, 0, 1, 0, 1, 1];
        let weight = [1.0; 7];
        let data = Expansion::new(&time, &event, &weight, 1).unwrap();
        let rows = data.expand();
        let subject: Vec<usize> = (0..rows.row.len()).filter(|&k| rows.row[k] == 1).collect();
        let stops: Vec<f64> = subject.iter().map(|&k| rows.stop[k]).collect();
        assert_eq!(stops, [2.0, 4.0, 6.0]);
        for (&k, want) in subject.iter().zip([1.0, 0.8, 0.533_333_333_3]) {
            assert!(close(rows.weight[k], want, 1e-9), "{:?}", rows.weight);
        }
    }

    fn fit_rows(rows: &Rows, x: &[f64]) -> CoxModelFit {
        let design = Mat::from_fn(rows.row.len(), 1, |k, _| x[rows.row[k] as usize]);
        let data = SurvData::builder(&rows.stop, &rows.event)
            .n_codes(1)
            .start(&rows.start)
            .weight(&rows.weight)
            .build()
            .unwrap();
        crate::cox::fit(
            &data,
            design.as_ref(),
            &CoxOptions {
                ties: Ties::Breslow,
                timefix: TimeFix::Exact,
                ..CoxOptions::default()
            },
        )
        .expect("consistent lengths")
    }

    /// `survival::finegray(..., etype = "c1")`, then
    /// `coxph(..., weights = fgwt, ties = "breslow")`; the model-based
    /// (`naive.var`) standard error. R 4.6.1, survival 3.8.6.
    #[test]
    fn fine_gray_fit_matches_r_on_both_forms() {
        let weight = [1.0; 20];
        let x = group();
        let data = Expansion::new(&TIME, &EVENT, &weight, 1).unwrap();
        let levels: Vec<u32> = x.iter().map(|&v| u32::from(v > 0.5)).collect();
        for rows in [data.expand(), data.risk_rows(&levels).unwrap()] {
            let fit = fit_rows(&rows, &x);
            assert!(fit.converged);
            assert!(
                close(fit.coefficients[0].unwrap(), 0.151_472_434_751_664, 1e-10),
                "{fit:?}"
            );
            assert!(
                close(
                    fit.standard_errors[0].unwrap(),
                    0.634_301_730_060_493,
                    1e-10
                ),
                "{fit:?}"
            );
        }
    }

    /// Worked by hand: five people, censored at 1, 2 and 3. At 1 all five are
    /// at risk, so G is 4/5. At 2 a failure ties with the censoring and is
    /// ordered first, so the risk set is the censored one plus the two later
    /// times, and G falls by a factor 2/3. At 3 two are at risk and G halves.
    #[test]
    fn censoring_curve_orders_failures_before_tied_censoring() {
        let curve = censoring_curve(&[1.0, 2.0, 2.0, 3.0, 4.0], &[0, 0, 1, 0, 1], &[1.0; 5]);
        assert_eq!(curve.time, [1.0, 2.0, 3.0]);
        assert!(close(curve.at(0.5), 1.0, 1e-15));
        assert!(close(curve.at(1.0), 0.8, 1e-15));
        assert!(close(curve.at(2.5), 0.8 * (1.0 - 1.0 / 3.0), 1e-15));
        assert!(close(
            curve.at(3.0),
            0.8 * (1.0 - 1.0 / 3.0) * (1.0 - 1.0 / 2.0),
            1e-15
        ));
    }

    fn data<'a>(
        time: &'a [f64],
        event: &'a [i32],
        weight: &'a [f64],
        cluster: Option<&'a [u32]>,
    ) -> SurvData<'a> {
        let mut builder = SurvData::builder(time, event).n_codes(2).weight(weight);
        if let Some(cluster) = cluster {
            builder = builder.cluster(cluster);
        }
        builder.build().unwrap()
    }

    fn options(cause: u32, ties: Ties) -> FineGrayOptions {
        FineGrayOptions {
            ties,
            ..FineGrayOptions::new(cause)
        }
    }

    fn column(x: &[f64]) -> Mat<f64> {
        Mat::from_fn(x.len(), 1, |i, _| x[i])
    }

    #[test]
    fn the_entry_point_fits_both_forms_to_the_same_estimate() {
        let weight = [1.0; 20];
        let x = group();
        let plain = fit(
            &data(&TIME, &EVENT, &weight, None),
            column(&x).as_ref(),
            &options(1, Ties::Breslow),
        )
        .unwrap();
        let family: Vec<u32> = (0..20).map(|i| i % 10).collect();
        let clustered = fit(
            &data(&TIME, &EVENT, &weight, Some(&family)),
            column(&x).as_ref(),
            &options(1, Ties::Breslow),
        )
        .unwrap();
        assert!(close(
            plain.coefficients[0].unwrap(),
            clustered.coefficients[0].unwrap(),
            1e-10
        ));
        assert!(plain.n_rows < clustered.n_rows);
        assert_eq!(plain.n_rows, plain.n_used);
        assert!(clustered.robust_standard_errors.is_some());
        assert!(plain.robust_standard_errors.is_none());
    }

    /// Three columns, one of them a factor's indicator pair: the collapsed
    /// and the expanded form give the same model, and the model is the Cox
    /// fit of the rows.
    #[test]
    fn a_design_matrix_gives_the_same_model_in_both_forms() {
        let weight: Vec<f64> = (0..20).map(|i| 1.0 + f64::from(i % 2)).collect();
        let g = group();
        let level: Vec<usize> = (0..20)
            .map(|i| (i * 7 + usize::try_from(EVENT[i]).unwrap()) % 3)
            .collect();
        let x = Mat::from_fn(20, 3, |i, j| match j {
            0 => g[i],
            1 => f64::from(u8::from(level[i] == 1)),
            _ => f64::from(u8::from(level[i] == 2)),
        });
        let each: Vec<u32> = (0..20).collect();
        let plain = fit(
            &data(&TIME, &EVENT, &weight, None),
            x.as_ref(),
            &options(1, Ties::Efron),
        )
        .unwrap();
        let clustered = fit(
            &data(&TIME, &EVENT, &weight, Some(&each)),
            x.as_ref(),
            &options(1, Ties::Efron),
        )
        .unwrap();
        assert_eq!(plain.rank, 3);
        for (a, b) in plain.coefficients.iter().zip(&clustered.coefficients) {
            assert!(close(a.unwrap(), b.unwrap(), 1e-9), "{plain:?}");
        }
        for (a, b) in plain.covariance.iter().zip(&clustered.covariance) {
            assert!(close(a.unwrap(), b.unwrap(), 1e-9));
        }
        assert!(close(
            plain.tests.score.unwrap().statistic,
            clustered.tests.score.unwrap().statistic,
            1e-9
        ));
        let robust = clustered.robust_covariance.as_ref().unwrap();
        assert_eq!(robust.len(), 9);
        assert!(clustered.tests.robust_wald.is_some());
        assert!(plain.tests.wald.is_some() && plain.tests.robust_wald.is_none());
        assert!(plain.n_events > 0.0);
    }

    #[test]
    fn invalid_causes_columns_and_start_are_refused() {
        let weight = [1.0; 20];
        let x = group();
        let d = data(&TIME, &EVENT, &weight, None);
        let zero = fit(&d, column(&x).as_ref(), &options(0, Ties::Breslow));
        assert_eq!(
            zero.unwrap_err().to_string(),
            "cause 0 is outside 1..=2; 0 is censoring"
        );
        let above = rows(&d, 3, &RowsOptions::default());
        assert!(above.is_err());
        let short = fit(&d, column(&x[..19]).as_ref(), &options(1, Ties::Breslow));
        assert_eq!(
            short.unwrap_err().to_string(),
            "columns differ in length: x has 19 values against 20 in stop"
        );
        let none = Mat::<f64>::zeros(20, 0);
        let error = fit(&d, none.as_ref(), &options(1, Ties::Breslow)).unwrap_err();
        assert_eq!(error.to_string(), "x has no columns");
        let levels = vec![0u32; 19];
        assert!(risk_rows(&d, 1, &levels, &RowsOptions::default()).is_err());
        let mut nan = x.clone();
        nan[3] = f64::NAN;
        assert!(fit(&d, column(&nan).as_ref(), &options(1, Ties::Breslow)).is_err());
        let loose = FineGrayOptions {
            eps: 0.0,
            ..options(1, Ties::Breslow)
        };
        assert!(fit(&d, column(&x).as_ref(), &loose).is_err());

        let start = [0.0; 20];
        let delayed = SurvData::builder(&TIME, &EVENT)
            .n_codes(2)
            .start(&start)
            .build()
            .unwrap();
        let error = fit(&delayed, column(&x).as_ref(), &options(1, Ties::Breslow)).unwrap_err();
        assert!(error.to_string().contains("start column"), "{error}");
        assert!(rows(&delayed, 1, &RowsOptions::default()).is_err());
    }

    /// With no weights or all weights positive, the entry points use the same
    /// code as the slice functions, so the rows are the same.
    #[test]
    fn the_rows_are_those_of_the_core_when_every_row_is_used() {
        let weight: Vec<f64> = (0..20).map(|i| 0.5 + f64::from(i % 3)).collect();
        let d = data(&TIME, &EVENT, &weight, None);
        let core = Expansion::new(&TIME, &EVENT, &weight, 1).unwrap();
        let ours = rows(&d, 1, &RowsOptions::default()).unwrap();
        assert_eq!(ours, core.expand());
        let levels: Vec<u32> = (0..20).map(|i| i % 4).collect();
        let ours = risk_rows(&d, 1, &levels, &RowsOptions::default()).unwrap();
        assert_eq!(ours, core.risk_rows(&levels).unwrap());
    }

    /// A row of weight 0 is no individual: it does not enter the censoring
    /// distribution and has no rows, and the others keep their row numbers.
    #[test]
    fn rows_of_weight_zero_are_dropped_and_the_others_keep_their_numbers() {
        let mut weight = vec![1.0; 20];
        for dropped in [0, 5, 13, 19] {
            weight[dropped] = 0.0;
        }
        let kept: Vec<usize> = (0..20).filter(|&i| weight[i] > 0.0).collect();
        let time: Vec<f64> = kept.iter().map(|&i| TIME[i]).collect();
        let event: Vec<i32> = kept.iter().map(|&i| EVENT[i]).collect();
        let x: Vec<f64> = group();
        let kept_x: Vec<f64> = kept.iter().map(|&i| x[i]).collect();
        let clusters: Vec<u32> = (0..20).map(|i| i % 7 + 10).collect();
        let kept_clusters: Vec<u32> = kept.iter().map(|&i| clusters[i]).collect();

        let all = data(&TIME, &EVENT, &weight, Some(&clusters));
        let unit = vec![1.0; kept.len()];
        let only = data(&time, &event, &unit, Some(&kept_clusters));
        let opts = RowsOptions::default();
        let full = rows(&all, 1, &opts).unwrap();
        let small = rows(&only, 1, &opts).unwrap();
        let renumbered: Vec<u32> = small.row.iter().map(|&r| kept[r as usize] as u32).collect();
        assert_eq!(full.row, renumbered);
        assert_eq!(full.start, small.start);
        assert_eq!(full.stop, small.stop);
        assert_eq!(full.event, small.event);
        assert_eq!(full.weight, small.weight);
        for ties in [Ties::Breslow, Ties::Efron] {
            let a = fit(&all, column(&x).as_ref(), &options(1, ties)).unwrap();
            let b = fit(&only, column(&kept_x).as_ref(), &options(1, ties)).unwrap();
            assert_eq!(a.coefficients, b.coefficients);
            assert_eq!(a.robust_covariance, b.robust_covariance);
            assert_eq!(a.n_rows, b.n_rows);
        }
    }

    /// Times that differ by rounding error are one time under the default
    /// rule and two under `Exact`.
    #[test]
    fn the_default_rule_merges_near_ties_and_exact_does_not() {
        // Individual 5 has a competing event just before the censoring of
        // individual 7. The rule sets both to the smaller time.
        let mut time = TIME.map(|t| t * 0.1);
        time[5] = time[7] * (1.0 - 1e-12);
        let mut merged = time;
        merged[7] = time[5];
        let weight = [1.0; 20];
        let d = data(&time, &EVENT, &weight, None);
        let by_default = rows(&d, 1, &RowsOptions::default()).unwrap();
        let exact = rows(
            &d,
            1,
            &RowsOptions {
                timefix: TimeFix::Exact,
            },
        )
        .unwrap();
        let reference = rows(
            &data(&merged, &EVENT, &weight, None),
            1,
            &RowsOptions::default(),
        )
        .unwrap();
        assert_eq!(by_default, reference);
        assert_ne!(exact, reference);
    }
}
