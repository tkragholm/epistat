//! The Fine-Gray subdistribution hazard model for one cause, fitted on a design
//! matrix, with the Kaplan-Meier estimate of the censoring distribution it
//! needs.
//!
//! The model is the weighted Cox model on rows built from the data \[1\]:
//! `coxph` on the rows of `survival::finegray` in R. [`rows`] and
//! [`risk_rows`] build those rows. [`fit`] fits the model without building
//! them, by a sweep over the event times that forms the sums of the rows'
//! risk sets from running sums over the individuals, and gives the result of
//! the Cox fit, a [`CoxModelFit`]: coefficients, model-based and robust
//! covariance, and the Wald, score and likelihood ratio tests, by the same
//! Newton iteration as [`crate::cox::fit`].
//!
//! # Input
//!
//! [`rows`], [`risk_rows`] and [`fit`] take a [`SurvData`]. Its status is the
//! cause code, with 0 for censoring, the cause of interest is a code in
//! `1..=n_codes`, and every other nonzero code is a competing event. Its
//! stop is the time, its weight the case weight, its stratum the censoring
//! stratum and, for [`fit`], its cluster the cluster of the robust variance,
//! as the dense codes the type holds. The covariates of [`fit`] are the
//! columns of a matrix with one row per row of the data. The model follows
//! every individual from time 0, so data with a start column is an error.
//!
//! A stratum is a censoring stratum, `cengroup` of `cmprsk::crr`: the
//! censoring distribution is estimated within each stratum and each competing
//! event is reweighted by that of its own stratum. The model has one baseline
//! subdistribution hazard. Without a stratum the censoring distribution is
//! estimated from everyone.
//!
//! The tie rule of [`FineGrayOptions::timefix`] and [`RowsOptions::timefix`]
//! is applied to the times first. It defaults to the rule of
//! `survival::aeqSurv`, which `finegray` and `coxph` apply, so times that
//! differ by rounding error are one time. Integer times are unaffected.
//!
//! Rows of weight 0 are individuals who do not take part. They are left out
//! before the censoring distribution is estimated, so they have no rows and no
//! say in `G` or in the last time the rows run to. The rows keep the row
//! numbers of the data. The fit reports the rows of the data in `n_rows` and
//! those of positive weight in `n_used`, as [`crate::cox::fit`] does.
//!
//! # The model
//!
//! Individual `i` has a time `x_i`, a cause code `c_i` with 0 for censoring, a
//! weight `w_i` and covariates `z_i`. The cause of interest is `k`. The
//! subdistribution hazard of `k` keeps an individual with a competing event, a
//! cause other than `k` and 0, in the risk set after the event, because that
//! individual could no longer have had the event `k` but would have been
//! observed for it if the competing event had not happened \[1\]. The
//! individual's weight after the event decreases with the chance that a
//! censoring would have removed them:
//!
//! ```text
//! weight at time s >= x_i:   w_i G(s) / G(x_i-)
//! ```
//!
//! where `G` is the Kaplan-Meier estimate of the censoring distribution \[2\]
//! of the individual's censoring stratum, censoring being the event and every
//! failure of any cause being censored for it. The risk set at a censoring
//! time `t` is the weight with a time strictly after `t` plus the weight
//! censored at `t`, so a failure and a censoring at the same time are ordered
//! failure first. Using `time >= t` would add every tied failure to the risk
//! set and change every weight derived from `G`. `G(t-)` is `G` just before
//! `t`, which is 1 before the first censoring. Past the largest time of a
//! stratum, when that is below the largest time of all, its `G` is not
//! estimated and is 0, as in `crr`, so a competing event of the stratum
//! leaves the risk set there.
//!
//! # The rows
//!
//! The rows are counting-process intervals. [`rows`] writes one
//! row per individual and interval, which is the form `survival::finegray` in R
//! produces. Individuals with the cause of interest or censored have the single
//! row `(0, x_i]`. An individual with a competing event has rows at full weight
//! to the first censoring time of its stratum at or after the event, then at
//! `w_i G(max(s, x_i)) / G(x_i-)` for each interval `(s, s']` between later
//! censoring times of its stratum, ending at the largest observed time. The
//! denominator is `G` before the event, so an event that ties with a censoring
//! time loses the weight of that censoring. `G` is read at `max(s, x_i)`
//! because `G(s)` alone gives weights above 1 when the event falls after
//! censoring has begun. Rows of weight zero are left out.
//!
//! [`risk_rows`] sums the competing-event rows that share an interval, a
//! covariate value and a stratum. These rows all end without an event, so
//! they enter the partial likelihood only through weighted sums, and adding
//! them is exact. The weight on the interval opening at censoring time `c_j`
//! factorises as `G(c_j) w_i / G(x_i-)`, so one prefix sum of `w_i / G(x_i-)`
//! over the event times, per covariate level and stratum, gives every
//! interval.
//!
//! # Algorithm of the fit
//!
//! A row of [`rows`] is in the risk set at an event time `t` of the cause
//! when its interval holds `t`. For an individual with the cause or censored
//! that is `t <= x_i`, at weight `w_i`. For a competing event it is every `t`
//! up to the largest time, at weight `w_i` for `t <= x_i` and
//! `w_i G_g(t-) / G_g(x_i-)` for `t > x_i`, with `g` the stratum: the rows
//! between censoring times carry `G` of the censoring time that opens them,
//! which is `G(t-)` for every `t` inside. `G(t-)` is common to everyone of the
//! stratum at `t`, and `1 / G(x_i-)` belongs to the individual \[1\]. With
//! `theta_i = exp(beta' z_i)` the sums over the risk set at `t` are therefore
//!
//! ```text
//! S_r(t) = sum over x_i >= t of w_i theta_i z_i^r
//!        + sum over strata g of G_g(t-) A_r,g(t),
//! A_r,g(t) = sum over competing events of g with x_i < t of
//!            (w_i / G_g(x_i-)) theta_i z_i^r,
//! ```
//!
//! for `r = 0, 1, 2`, with `z^0 = 1`, `z^1 = z` and `z^2 = z z'`. The first sum
//! is the risk set of an ordinary Cox model, accumulated as the time descends,
//! and `A` a running sum accumulated as the time ascends. Given `S_r` the log
//! partial likelihood, score and information are those of
//! [`crate::cox::design`] with Breslow's or Efron's approximation, from the
//! same code: with `D(t)` the events at `t`, `E_r` the sums over them, and the
//! terms `(u, f)` of the approximation (one term of the summed event weight
//! and `f = 0` for Breslow, `d` terms of the mean weight and `f = j / d` for
//! Efron),
//!
//! ```text
//! l(beta) = sum over t of [ sum over D(t) of w_i beta' z_i - sum over terms of u log D_0 ]
//! U(beta) = sum over t of [ sum over D(t) of w_i z_i - sum over terms of u D_1 / D_0 ]
//! I(beta) = sum over t, terms of u [ D_2 / D_0 - (D_1 / D_0)(D_1 / D_0)' ],
//! D_r = S_r(t) - f E_r(t).
//! ```
//!
//! An individual with a competing event and `G(x_i-) = 0` has no weight after
//! the first censoring time at or after the event, so it counts at `w_i` up to
//! that time and has no `A` term. With positive weights `G(x_i-)` is positive
//! for every individual, since the individual is in the risk set of every
//! censoring before `x_i`, so this arises only from rounding.
//!
//! The descending sums are one pass over the individuals. The ascending sums
//! are kept at checkpoints, one per block of about `sqrt(m s)` event times
//! with `m` event times and `s` strata, and each block is summed again from its
//! checkpoint when the descending pass reaches it, so each sum is that of one
//! ascending pass and the memory is that of the checkpoints and one block,
//! `O(sqrt(m s) s p^2)`, besides the data. Each evaluation of the likelihood
//! costs three passes over the individuals of `p^2` each and `m s p^2` for the
//! strata. Nothing of the size of the rows is held: the rows of an individual
//! with a competing event are as many as the censoring times after the event,
//! which came to about 10 KB per individual on the cohort of
//! `benches/survival_compare.rs`.
//!
//! The covariates are centred and scaled as `coxph` does it on the rows, by the
//! weighted mean and mean absolute deviation with each individual's total row
//! weight `w_i (1 + sum_j G(c_j) / G(x_i-))` over the censoring times `c_j` that
//! open an interval of the individual, so the iteration, the aliasing rule and
//! the test for a diverging coefficient are those of `coxph` on the rows. The
//! iteration is that of `coxph` on counting-process data (`agreg.fit`).
//!
//! # Robust variance
//!
//! With clusters the robust covariance is the sandwich `V (sum_g U_g U_g') V`
//! of `coxph` on the rows with the clusters of the individuals \[3\], where
//! `U_g` sums the score residuals of the rows in cluster `g`. The residuals of
//! an individual's rows add up to
//!
//! ```text
//! w_i (z_i - m(x_i)) [the event, with the correction of Efron's approximation]
//! - w_i theta_i sum over t <= end_i of (z_i h_t - c_t)
//! - (w_i / G_g(x_i-)) theta_i sum over t > x_i of G_g(t-) (z_i h_t - c_t)
//! ```
//!
//! with `h_t` the sum over the terms at `t` of `u / D_0`, `c_t` that of
//! `u D_1 / D_0^2`, `m(t)` the mean of `D_1 / D_0` over the terms, and `end_i`
//! the last time at the individual's own weight. The first sum is a prefix sum
//! over the event times; the second is summed per stratum over the event times
//! in descending order. The sandwich treats `G` as known, as `coxph` on the
//! rows does. The variance of `crr` adds a term for the estimation of `G`
//! \[1\], which this one does not have.
//!
//! # Agreement with R and with the rows
//!
//! `tests/r_fine_gray.rs` compares with `survival::finegray` followed by
//! `coxph` on its rows, with weights `fgwt` and, for the robust variance, the
//! individual or a group of four as cluster, and with `cmprsk::crr`. A factor
//! is written as indicator columns, as `model.matrix` does. With censoring
//! strata it compares with `coxph` on rows built in R with the censoring
//! distribution of each stratum, and with `crr(cengroup =)`.
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
//! - `finegray` with a `strata()` term estimates the censoring distribution per
//!   stratum for a model stratified by the same variable, and writes each
//!   stratum's rows only up to its own last time and event, so it is not the
//!   reference for censoring strata with one baseline.
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
//! The unit tests compare [`fit`] with [`crate::cox::fit`] on the rows of
//! [`rows`], on random data with ties, fractional weights with zeros,
//! censoring strata and clusters, for both approximations: coefficients,
//! standard errors, covariance, robust covariance and log partial likelihood
//! within tolerances derived from the order of the sums, and the iteration
//! count exactly.
//!
//! # Edge cases
//!
//! Cause 0 is censoring and is refused as a cause of interest. Data without
//! an event of the cause has no estimate. `Ties::Exact` is the Cox fit with
//! exact ties of the rows, which `coxph` has for rows of weight 1 only: case
//! weights of 1 and no censoring in a competing event's stratum after the
//! event, with a later time after that. Each individual is then one row, and
//! other data is refused, as are clusters.
//!
//! # References
//!
//! 1. Fine JP, Gray RJ. A proportional hazards model for the subdistribution
//!    of a competing risk. J Am Stat Assoc 1999;94(446):496-509.
//!    doi:10.1080/01621459.1999.10474144
//! 2. Kaplan EL, Meier P. Nonparametric estimation from incomplete
//!    observations. J Am Stat Assoc 1958;53(282):457-81.
//!    doi:10.1080/01621459.1958.10501452
//! 3. Lin DY, Wei LJ. The robust inference for the Cox proportional hazards
//!    model. J Am Stat Assoc 1989;84(408):1074-8.
//!    doi:10.1080/01621459.1989.10478874
//! 4. Therneau TM, Crowson C, Atkinson E. Multi-state models and competing
//!    risks. Vignette of the R package survival.
//! 5. Gray RJ. cmprsk: subdistribution analysis of competing risks. R package.

#![expect(
    clippy::cast_possible_truncation,
    reason = "the entry points check that the row count fits in u32"
)]

use std::borrow::Cow;
use std::cmp::Ordering;
use std::collections::HashMap;

use epistat_core::InputError;
use epistat_linalg::{Mat, MatRef};

use crate::check::{check_lengths, check_row_count, refuse_non_finite_matrix};
use crate::cox::design::{DenseClusters, Newton, Stepping, newton};
use crate::cox::{CoxModelFit, CoxOptions, Ties};
use crate::error::{Error, Result};
use crate::metrics::censoring::CensoringKm;
use crate::order::pick_used;
use crate::step::Steps;
use crate::surv::SurvData;
use crate::timefix::TimeFix;

mod sweep;

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
fn censoring_curve(time: &[f64], event: &[i32], weight: &[f64]) -> Steps {
    let mut order: Vec<usize> = (0..time.len()).collect();
    order.sort_by(|&a, &b| time[a].total_cmp(&time[b]));
    let sorted_time: Vec<f64> = order.iter().map(|&i| time[i]).collect();
    let censored: Vec<bool> = order.iter().map(|&i| event[i] == 0).collect();
    let sorted_weight: Vec<f64> = order.iter().map(|&i| weight[i]).collect();
    CensoringKm::new(&sorted_time, &censored, &sorted_weight, true).curve
}

/// Counting-process rows for a Cox fit. `row` is an input row: the individual
/// for [`rows`], and for [`risk_rows`] the individual or any individual with
/// the covariate value the collapsed row carries.
#[derive(Debug, Clone, Default, PartialEq)]
#[non_exhaustive]
pub struct FineGrayRows {
    /// The input row each row comes from.
    pub row: Vec<u32>,
    /// Interval start.
    pub start: Vec<f64>,
    /// Interval stop.
    pub stop: Vec<f64>,
    /// 1 when the interval ends in the cause of interest, 0 otherwise, the
    /// status of the row as `finegray`'s `fgstatus`.
    pub status: Vec<i32>,
    /// Weight of the row.
    pub weight: Vec<f64>,
}

impl FineGrayRows {
    /// The value of `column`, one value per row of the input, at the input row
    /// of each row here.
    ///
    /// # Errors
    ///
    /// An input error when `column` has no value at some input row.
    pub fn gather<T: Copy>(&self, column: &[T]) -> Result<Vec<T>> {
        crate::expanded::gather(&self.row, column, "column")
    }

    /// The rows as follow-up data, with their weights and the largest status as the number of
    /// codes, at least 1, for a fit on them.
    ///
    /// # Errors
    ///
    /// The errors of building a [`SurvBuf`](crate::SurvBuf) from the columns.
    pub fn to_surv_buf(&self) -> Result<crate::SurvBuf> {
        crate::expanded::surv_buf(
            Some(&self.start),
            &self.stop,
            &self.status,
            Some(&self.weight),
        )
    }
}

impl FineGrayRows {
    fn push(&mut self, row: usize, start: f64, stop: f64, event: bool, weight: f64) {
        self.row.push(row as u32);
        self.start.push(start);
        self.stop.push(stop);
        self.status.push(i32::from(event));
        self.weight.push(weight);
    }
}

/// `FineGrayRows`, under its name in 0.3.
#[deprecated(since = "0.4.0", note = "renamed `FineGrayRows`")]
pub type Rows = FineGrayRows;

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
    /// The censoring stratum of each individual, empty without strata.
    group: &'a [u32],
    /// The censoring distribution of each stratum, one without strata.
    curves: Vec<Steps>,
    horizon: f64,
}

impl<'a> Expansion<'a> {
    /// The data of one analysis: one `time`, cause code `event` (0 for
    /// censoring) and `weight` per individual, and the `cause` of interest.
    ///
    /// # Errors
    ///
    /// An input error when the slices differ in length or `cause` is 0.
    #[cfg(test)]
    fn new(time: &'a [f64], event: &'a [i32], weight: &'a [f64], cause: i32) -> Result<Self> {
        Self::with_strata(time, event, weight, cause, &[], 0)
    }

    /// [`Expansion::new`] with a censoring stratum `group` per individual, a
    /// code below `n_groups`, or `group` empty for one stratum.
    fn with_strata(
        time: &'a [f64],
        event: &'a [i32],
        weight: &'a [f64],
        cause: i32,
        group: &'a [u32],
        n_groups: usize,
    ) -> Result<Self> {
        check_lengths(&[
            ("time", time.len()),
            ("event", event.len()),
            ("weight", weight.len()),
        ])?;
        if !group.is_empty() {
            check_lengths(&[("time", time.len()), ("stratum", group.len())])?;
        }
        check_row_count(time.len())?;
        if cause == 0 {
            return Err(Error::out_of_range("cause 0 is censoring, not a cause"));
        }
        let horizon = time.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let curves = if group.is_empty() {
            vec![censoring_curve(time, event, weight)]
        } else {
            (0..n_groups)
                .map(|g| {
                    let members: Vec<usize> = (0..time.len())
                        .filter(|&i| group[i] as usize == g)
                        .collect();
                    let pick = |column: &[f64]| -> Vec<f64> {
                        members.iter().map(|&i| column[i]).collect()
                    };
                    let own_time = pick(time);
                    let own_event: Vec<i32> = members.iter().map(|&i| event[i]).collect();
                    let mut curve = censoring_curve(&own_time, &own_event, &pick(weight));
                    // Past the stratum's largest time its censoring
                    // distribution is not estimated, and is 0 as in
                    // `cmprsk::crr`. A censoring at that time has made it 0
                    // already.
                    let last = own_time.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    if last < horizon && curve.time.last().is_none_or(|&c| c < last) {
                        curve.time.push(last);
                        curve.value.push(0.0);
                    }
                    curve
                })
                .collect()
        };
        Ok(Self {
            time,
            event,
            weight,
            cause,
            group,
            curves,
            horizon,
        })
    }

    fn competing(&self, i: usize) -> bool {
        self.event[i] != self.cause && self.event[i] != 0
    }

    /// The censoring stratum of individual `i`.
    fn group_of(&self, i: usize) -> usize {
        self.group.get(i).map_or(0, |&g| g as usize)
    }

    /// The censoring distribution of individual `i`.
    fn curve(&self, i: usize) -> &Steps {
        &self.curves[self.group_of(i)]
    }

    /// One row `(0, x_i]` for each individual with the cause of interest or
    /// censored. There is nothing to expand for them.
    fn direct_rows(&self, rows: &mut FineGrayRows) {
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
    fn expand(&self) -> FineGrayRows {
        let mut rows = FineGrayRows::default();
        for i in 0..self.time.len() {
            let (t, w) = (self.time[i], self.weight[i]);
            if !self.competing(i) {
                rows.push(i, 0.0, t, self.event[i] == self.cause, w);
                continue;
            }
            let curve = self.curve(i);
            let g_before_event = curve.before(t);
            let weight_from = |previous: f64| {
                if previous < t {
                    w
                } else if g_before_event > 0.0 {
                    w * (curve.at(previous) / g_before_event)
                } else {
                    0.0
                }
            };
            let first = curve.time.partition_point(|&c| c < t);
            let mut previous = 0.0f64;
            for &boundary in &curve.time[first..] {
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

    /// The expansion with competing-event rows summed per interval, level and
    /// censoring stratum.
    ///
    /// The weight on the interval opening at censoring time `c_j` factorises as
    /// `G(c_j) * w_i / G(x_i-)`, so one prefix sum of `w_i / G(x_i-)` over event
    /// times, per level and stratum, gives every interval. `level` codes the
    /// covariate; levels are emitted in order of their first competing-event
    /// individual.
    ///
    /// # Errors
    ///
    /// An input error when `level` does not hold one code per individual.
    fn risk_rows(&self, level: &[u32]) -> Result<FineGrayRows> {
        check_lengths(&[("time", self.time.len()), ("level", level.len())])?;
        let mut rows = FineGrayRows::default();
        self.direct_rows(&mut rows);

        // Per level and stratum: (event time, w, w / G(event time-)) and a
        // representative row.
        let mut slot_of: HashMap<(usize, u32), usize> = HashMap::new();
        let mut by_level: Vec<(usize, Vec<TiedEvent>)> = Vec::new();
        for (i, &code) in level.iter().enumerate() {
            if !self.competing(i) {
                continue;
            }
            let g_before_event = self.curve(i).before(self.time[i]);
            if g_before_event <= 0.0 {
                // Every weight after the event is zero.
                continue;
            }
            let slot = *slot_of.entry((self.group_of(i), code)).or_insert_with(|| {
                by_level.push((i, Vec::new()));
                by_level.len() - 1
            });
            by_level[slot].1.push((
                self.time[i],
                self.weight[i],
                self.weight[i] / g_before_event,
            ));
        }

        for (representative, mut entries) in by_level {
            let curve = self.curve(representative);
            let censoring = &curve.time;
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
                let weight = curve.at(boundary) * covered;
                if weight > 0.0 {
                    rows.push(representative, boundary, stop, false, weight);
                }
            }
        }
        Ok(rows)
    }
}

/// Codes for the distinct values of a float column, in order of first
/// appearance, by bit pattern, under its name in 0.3.
#[deprecated(since = "0.4.0", note = "use `lowlevel::float_codes`")]
#[must_use]
pub fn float_codes(values: &[f64]) -> Vec<u32> {
    codes_of(values)
}

/// Codes for the distinct values of a float column, in order of first
/// appearance, by bit pattern.
#[must_use]
pub(crate) fn codes_of(values: &[f64]) -> Vec<u32> {
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
#[cfg(test)]
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
    /// How tied events enter the partial likelihood. `Ties::Exact` needs every
    /// row weight to be 1; see [`fit`].
    pub ties: Ties,
    /// How times that differ by rounding error are treated. The default is the
    /// rule of `survival::aeqSurv`, which `coxph` applies.
    pub timefix: TimeFix,
    /// The relative change of the log partial likelihood at which the
    /// iteration stops, `eps` of `coxph.control`. It must be positive.
    pub tolerance: f64,
    /// The largest number of trial points, `iter.max` of `coxph.control`.
    pub max_iterations: usize,
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
            tolerance: cox.tolerance,
            max_iterations: cox.max_iterations,
        }
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
        return Err(Error::unsupported(
            "start",
            "Fine-Gray follows every individual from time 0, so data with a start column is not supported",
        ));
    }
    let code = i32::try_from(data.cause(cause)?.code())
        .expect("SurvData::cause checked that the code is a status");
    let fixed = data.clone().with_timefix(timefix)?;
    let surv = fixed.followup()?;
    let index = fixed.used_index();
    let time = pick_used(surv.stop(), index.as_deref());
    let event = pick_used(surv.status(), index.as_deref());
    let weight: Cow<'_, [f64]> = match fixed.weight() {
        None => Cow::Owned(vec![1.0; fixed.n_rows()]),
        Some(weight) => pick_used(weight, index.as_deref()),
    };
    let group: Cow<'_, [u32]> = match fixed.stratum() {
        None => Cow::Borrowed(&[]),
        Some(stratum) => pick_used(stratum, index.as_deref()),
    };
    let expansion = Expansion::with_strata(&time, &event, &weight, code, &group, fixed.n_strata())?;
    body(&expansion, index.as_deref())
}

/// Point the rows at the rows of the data they came from.
fn original_rows(mut rows: FineGrayRows, index: Option<&[u32]>) -> FineGrayRows {
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
/// cause is a competing event. The stratum of `data`, if any, is the
/// censoring stratum: a competing event's rows are split at the censoring
/// times of its stratum and weighted by its censoring distribution. The tie
/// rule of `opts` is applied to the times first, and rows of weight 0 are left
/// out of the data, of the censoring distribution and of the rows.
/// [`FineGrayRows::row`] is the row of `data` each row comes from.
///
/// # Errors
///
/// An input error when `data` has a start column, when `cause` is not a
/// status code of `data`, or as for [`SurvData::with_timefix`].
pub fn rows(data: &SurvData<'_>, cause: u32, opts: &RowsOptions) -> Result<FineGrayRows> {
    with_expansion(data, cause, opts.timefix, |expansion, index| {
        Ok(original_rows(expansion.expand(), index))
    })
}

/// The expansion of [`rows`] with the competing-event rows summed per
/// interval, covariate level and censoring stratum.
///
/// `level` holds one code per row of `data`, the covariate level of the row.
/// The rows have the risk sets of [`rows`] for a covariate with those levels.
/// [`FineGrayRows::row`] is the first row of `data` of the level and stratum that has
/// a competing event, and levels are emitted in order of that first row.
///
/// # Errors
///
/// An input error as for [`rows`], and when `level` does not hold one code
/// per row of `data`.
pub fn risk_rows(
    data: &SurvData<'_>,
    cause: u32,
    level: &[u32],
    opts: &RowsOptions,
) -> Result<FineGrayRows> {
    check_lengths(&[("stop", data.n_rows()), ("level", level.len())])?;
    with_expansion(data, cause, opts.timefix, |expansion, index| {
        let level = pick_used(level, index);
        Ok(original_rows(expansion.risk_rows(&level)?, index))
    })
}

/// Fit the subdistribution hazard model of a cause on a design matrix.
///
/// `x` holds the covariates, one row per row of `data` and one column per
/// covariate, with no intercept column. A factor is its indicator columns. The
/// cause is [`FineGrayOptions::cause`], a status code of `data`, and the
/// stratum of `data`, if any, is the censoring stratum. Rows of weight 0 are
/// dropped, the tie rule of `opts` is applied to the times, and every
/// individual is at risk from time 0. The result is that of
/// [`crate::cox::fit`] on the rows of [`rows`] with their weights:
/// coefficients, covariances, tests, aliased columns and the convergence
/// report. It is computed by the sweep of the module documentation, without
/// the rows.
///
/// With clusters in `data`, dense codes, the robust covariance is the sandwich
/// of the Cox fit on the rows with each row in its individual's cluster, which
/// treats the weights as known. `n_rows` counts the rows of `data` and
/// `n_used` those of positive weight.
///
/// # Errors
///
/// An input error when `x` has no column, does not hold one row per row of
/// `data` or has a value that is not finite, when `opts.tolerance` is not positive
/// and finite, for `Ties::Exact` with clusters, with case weights other than 1
/// or with a censoring after a competing event in its stratum, and as for
/// [`rows`].
pub fn fit(data: &SurvData<'_>, x: MatRef<'_, f64>, opts: &FineGrayOptions) -> Result<CoxModelFit> {
    check_lengths(&[("stop", data.n_rows()), ("x", x.nrows())])?;
    if x.ncols() == 0 {
        return Err(InputError::Empty { what: "x" }.into());
    }
    refuse_non_finite_matrix(x, "x")?;
    if !(opts.tolerance.is_finite() && opts.tolerance > 0.0) {
        return Err(Error::options("tolerance: must be finite and positive"));
    }
    if opts.ties == Ties::Exact && data.cluster().is_some() {
        return Err(Error::unsupported(
            "ties",
            "ties: the exact method has no cluster-robust variance",
        ));
    }
    let mut fit = with_expansion(data, opts.cause, opts.timefix, |expansion, index| {
        let source = |i: usize| index.map_or(i, |index| index[i] as usize);
        if opts.ties == Ties::Exact {
            return exact(expansion, x, &source, opts);
        }
        let p = x.ncols();
        let n = expansion.time.len();
        let n_events: f64 = (0..n)
            .filter(|&i| expansion.event[i] == expansion.cause)
            .map(|i| expansion.weight[i])
            .sum();
        let clusters = data.cluster().map(|cluster| {
            let used: Vec<u32> = (0..n).map(|i| cluster[source(i)]).collect();
            let count = used.iter().map(|&c| c as usize + 1).max().unwrap_or(0);
            (used, count)
        });
        let with_clusters = clusters.is_some();
        if !expansion.event.contains(&expansion.cause) {
            return CoxModelFit::degenerate(p, n_events, n, with_clusters);
        }
        let (model, factor) = prepare(expansion, x, &source, opts.ties, clusters);
        let summary = with_clusters
            .then(|| DenseClusters::of(&model.cluster, 0..model.cluster.len(), model.n_clusters));
        newton(
            &model,
            &Newton {
                p,
                factor: &factor,
                init: None,
                max_iterations: opts.max_iterations,
                tolerance: opts.tolerance,
                stepping: Stepping::Counting,
                ties: opts.ties,
                n_events,
                n_rows: n,
                clusters: summary,
            },
        )
    })?;
    fit.n_rows = data.n_rows();
    fit.n_used = data.n_used();
    Ok(fit)
}

/// The fit with exact ties, which the Cox fit has for rows of weight 1 only.
///
/// Every row of [`rows`] has weight 1 when the case weights are 1 and no
/// competing event has a later interval of positive weight, that is no
/// censoring time at or after it in its stratum with a later time after
/// that. Each individual is then the one row `(0, x_i]`, or for a competing
/// event `(0, c]` with `c` the first censoring time at or after it or the
/// largest time, and the fit is the Cox fit of those rows.
fn exact(
    expansion: &Expansion<'_>,
    x: MatRef<'_, f64>,
    source: &impl Fn(usize) -> usize,
    opts: &FineGrayOptions,
) -> Result<CoxModelFit> {
    if expansion.weight.iter().any(|&w| w - 1.0 != 0.0) {
        return Err(Error::unsupported(
            "weight",
            "weight: the exact method supports case weights of 1 only",
        ));
    }
    let n = expansion.time.len();
    let mut stop = Vec::with_capacity(n);
    for i in 0..n {
        let t = expansion.time[i];
        if !expansion.competing(i) {
            stop.push(t);
            continue;
        }
        let curve = expansion.curve(i);
        let first = curve.time.partition_point(|&c| c < t);
        let later = (first..curve.time.len()).any(|j| {
            let next = curve.time.get(j + 1).copied().unwrap_or(expansion.horizon);
            next > curve.time[j] && curve.value[j] > 0.0
        });
        if later && curve.before(t) > 0.0 {
            return Err(Error::not_applicable(
                "ties: the exact method needs every Fine-Gray weight to be 1, so no \
                 censoring after a competing event in its stratum",
            ));
        }
        stop.push(curve.time.get(first).copied().unwrap_or(expansion.horizon));
    }
    let start = vec![0.0f64; n];
    let status: Vec<i32> = (0..n)
        .map(|i| i32::from(expansion.event[i] == expansion.cause))
        .collect();
    let design = Mat::from_fn(n, x.ncols(), |i, j| x[(source(i), j)]);
    let rows = SurvData::builder(&stop, &status)
        .n_codes(1)
        .start(&start)
        .weight(expansion.weight)
        .build()?;
    crate::cox::fit(
        &rows,
        design.as_ref(),
        &CoxOptions {
            ties: Ties::Exact,
            timefix: TimeFix::Exact,
            tolerance: opts.tolerance,
            max_iterations: opts.max_iterations,
            ..CoxOptions::default()
        },
    )
}

/// The sweep of [`sweep::Sweep`] for `expansion` and the covariates `x` of
/// the data, with `source` the row of `x` of each individual, and the factor
/// of each column from the scaled to the given scale.
///
/// The covariates are centred and scaled as `coxph` does it on the rows of
/// [`rows`]: by the mean and the mean absolute deviation weighted by the total
/// weight of each individual's rows. A competing event at `x_i` has the row
/// weight `w_i` up to the first censoring time at or after it and
/// `w_i G(c_j) / G(x_i-)` on the interval opening at each later censoring time
/// `c_j`, so its total is `w_i (1 + sum_j G(c_j) / G(x_i-))` over the
/// intervals that have a length.
fn prepare(
    expansion: &Expansion<'_>,
    x: MatRef<'_, f64>,
    source: &impl Fn(usize) -> usize,
    ties: Ties,
    clusters: Option<(Vec<u32>, usize)>,
) -> (sweep::Sweep, Vec<f64>) {
    let n = expansion.time.len();
    let p = x.ncols();
    let groups = expansion.curves.len();

    // Per stratum, the sum over the censoring times from each on of `G(c_j)`
    // for the intervals that open there and have a length.
    let tails: Vec<Vec<f64>> = expansion
        .curves
        .iter()
        .map(|curve| {
            let m = curve.time.len();
            let mut tail = vec![0.0f64; m + 1];
            for j in (0..m).rev() {
                let stop = curve.time.get(j + 1).copied().unwrap_or(expansion.horizon);
                let here = if stop > curve.time[j] {
                    curve.value[j]
                } else {
                    0.0
                };
                tail[j] = tail[j + 1] + here;
            }
            tail
        })
        .collect();

    let mut end = Vec::with_capacity(n);
    let mut total = Vec::with_capacity(n);
    let mut late: Vec<(usize, u32, f64, f64)> = Vec::new();
    for i in 0..n {
        let (t, w) = (expansion.time[i], expansion.weight[i]);
        if !expansion.competing(i) {
            end.push(t);
            total.push(w);
            continue;
        }
        let g = expansion.group_of(i);
        let curve = &expansion.curves[g];
        let first = curve.time.partition_point(|&c| c < t);
        let g_before_event = curve.before(t);
        if g_before_event > 0.0 {
            let ratio = w / g_before_event;
            end.push(t);
            total.push(w + ratio * tails[g][first]);
            late.push((g, i as u32, t, ratio));
        } else {
            // Every weight after the first censoring time at or after the
            // event is zero.
            end.push(curve.time.get(first).copied().unwrap_or(expansion.horizon));
            total.push(w);
        }
    }

    // Centre and scale as `coxph` does.
    let total_weight: f64 = total.iter().sum();
    let mut centre = vec![0.0f64; p];
    let mut factor = vec![0.0f64; p];
    for a in 0..p {
        let first = x[(source(0), a)];
        if (0..n).all(|i| x[(source(i), a)] - first == 0.0) {
            continue;
        }
        let mean: f64 = (0..n).map(|i| total[i] * x[(source(i), a)]).sum::<f64>() / total_weight;
        let deviation: f64 = (0..n)
            .map(|i| total[i] * (x[(source(i), a)] - mean).abs())
            .sum::<f64>()
            / total_weight;
        centre[a] = mean;
        factor[a] = if deviation > 0.0 {
            1.0 / deviation
        } else {
            1.0
        };
    }
    let mut scaled = Vec::with_capacity(n * p);
    for i in 0..n {
        for a in 0..p {
            scaled.push(if factor[a] == 0.0 {
                0.0
            } else {
                (x[(source(i), a)] - centre[a]) * factor[a]
            });
        }
    }

    let mut by_end: Vec<u32> = (0..n as u32).collect();
    by_end.sort_by(|&a, &b| end[b as usize].total_cmp(&end[a as usize]));
    late.sort_by(|a, b| a.0.cmp(&b.0).then(a.2.total_cmp(&b.2)));
    let mut late_offset = vec![0usize; groups + 1];
    for &(g, ..) in &late {
        late_offset[g + 1] += 1;
    }
    for g in 0..groups {
        late_offset[g + 1] += late_offset[g];
    }

    let is_event: Vec<bool> = (0..n)
        .map(|i| expansion.event[i] == expansion.cause)
        .collect();
    let mut by_time: Vec<u32> = (0..n as u32).filter(|&i| is_event[i as usize]).collect();
    by_time.sort_by(|&a, &b| expansion.time[a as usize].total_cmp(&expansion.time[b as usize]));
    let mut event_times: Vec<f64> = Vec::new();
    let mut dying_offset = vec![0usize];
    for (position, &i) in by_time.iter().enumerate() {
        let t = expansion.time[i as usize];
        if event_times
            .last()
            .is_some_and(|&last| last.total_cmp(&t) == Ordering::Equal)
        {
            continue;
        }
        if !event_times.is_empty() {
            dying_offset.push(position);
        }
        event_times.push(t);
    }
    dying_offset.push(by_time.len());
    let (cluster, n_clusters) = clusters.unwrap_or_default();
    (
        sweep::Sweep {
            p,
            ties,
            x: scaled,
            weight: expansion.weight.to_vec(),
            end,
            is_event,
            by_end,
            late: late.iter().map(|entry| entry.1).collect(),
            late_time: late.iter().map(|entry| entry.2).collect(),
            late_ratio: late.iter().map(|entry| entry.3).collect(),
            late_offset,
            curves: expansion.curves.clone(),
            times: event_times,
            dying: by_time,
            dying_offset,
            cluster,
            n_clusters,
        },
        factor,
    )
}

#[cfg(test)]
mod tests {
    use epistat_core::Covariance;
    use epistat_linalg::Mat;

    use epistat_linalg::MatRef;
    use epistat_rng::Stream;

    use super::{
        Expansion, FineGrayOptions, FineGrayRows, RowsOptions, censoring_curve, fit, risk_rows,
        row_codes, rows, with_expansion,
    };
    use crate::check::check_lengths;
    use crate::cox::{CoxModelFit, CoxOptions, Ties};
    use crate::error::Result;
    use crate::step::Steps;
    use crate::surv::SurvData;
    use crate::timefix::TimeFix;

    /// The fit as a weighted Cox fit on the rows of [`rows`], or with
    /// `collapsed` on those of [`risk_rows`] with the distinct rows of `x` as
    /// levels, by [`crate::cox::fit`]. The rows of a competing event are
    /// clustered by individual through the cluster of the data. This is the
    /// definition the sweep of [`fit`] is checked against.
    fn reference_fit(
        data: &SurvData<'_>,
        x: MatRef<'_, f64>,
        opts: &FineGrayOptions,
        collapsed: bool,
    ) -> Result<CoxModelFit> {
        check_lengths(&[("stop", data.n_rows()), ("x", x.nrows())])?;
        if x.ncols() == 0 {
            return Err(epistat_core::InputError::Empty { what: "x" }.into());
        }
        with_expansion(data, opts.cause, opts.timefix, |expansion, index| {
            let source =
                |row: u32| index.map_or(row as usize, |index| index[row as usize] as usize);
            let rows = if collapsed {
                let used = Mat::from_fn(expansion.time.len(), x.ncols(), |i, j| {
                    x[(source(i as u32), j)]
                });
                expansion.risk_rows(&row_codes(used.as_ref()))?
            } else {
                expansion.expand()
            };
            let design = Mat::from_fn(rows.row.len(), x.ncols(), |k, j| {
                x[(source(rows.row[k]), j)]
            });
            let cluster: Option<Vec<u32>> = data
                .cluster()
                .map(|cluster| rows.row.iter().map(|&row| cluster[source(row)]).collect());
            let mut builder = SurvData::builder(&rows.stop, &rows.status)
                .n_codes(1)
                .start(&rows.start)
                .weight(&rows.weight);
            if let Some(cluster) = &cluster {
                builder = builder.cluster(cluster);
            }
            crate::cox::fit(
                &builder.build()?,
                design.as_ref(),
                &CoxOptions {
                    ties: opts.ties,
                    timefix: TimeFix::Exact,
                    tolerance: opts.tolerance,
                    max_iterations: opts.max_iterations,
                    ..CoxOptions::default()
                },
            )
        })
    }

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

    fn fit_rows(rows: &FineGrayRows, x: &[f64]) -> CoxModelFit {
        let design = Mat::from_fn(rows.row.len(), 1, |k, _| x[rows.row[k] as usize]);
        let data = SurvData::builder(&rows.stop, &rows.status)
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
                close(fit.se[0].unwrap(), 0.634_301_730_060_493, 1e-10),
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

    /// The censoring curve as the blocks of tied times, which sums the weight
    /// after each time from block totals.
    #[expect(clippy::float_cmp, reason = "rows that share a time are grouped")]
    fn censoring_curve_by_blocks(time: &[f64], event: &[i32], weight: &[f64]) -> Steps {
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

    /// The shared Kaplan-Meier estimate and the one built from block totals
    /// sum the same weights in another order, so they differ by rounding only.
    /// The bound is `n` units in the last place of each value, the error of a
    /// sum of at most `n` weights, applied at each of at most `n` steps.
    #[test]
    fn the_shared_censoring_curve_equals_the_block_form_up_to_rounding() {
        let time: Vec<f64> = (0..300).map(|i| 1.0 + f64::from((i * 7) % 40)).collect();
        let event: Vec<i32> = (0..300).map(|i| [0, 1, 2, 0, 1][i % 5]).collect();
        let weight: Vec<f64> = (0..300)
            .map(|i| 0.25 + 0.1 * f64::from((i * 13) % 17))
            .collect();
        for weights in [vec![1.0; 300], weight] {
            let shared = censoring_curve(&time, &event, &weights);
            let blocks = censoring_curve_by_blocks(&time, &event, &weights);
            assert_eq!(shared.time, blocks.time);
            let bound = 300.0 * 300.0 * f64::EPSILON;
            for (a, b) in shared.value.iter().zip(&blocks.value) {
                assert!((a - b).abs() <= bound * a.abs().max(*b), "{a} {b}");
            }
        }
        // With unit weights every sum is exact, so the curves are the same.
        let ones = vec![1.0; 300];
        assert_eq!(
            censoring_curve(&time, &event, &ones).value,
            censoring_curve_by_blocks(&time, &event, &ones).value
        );
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
        assert_eq!((plain.n_rows, plain.n_used), (20, 20));
        assert_eq!((clustered.n_rows, clustered.n_used), (20, 20));
        assert!(clustered.robust_se.is_some());
        assert!(plain.robust_se.is_none());
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
        for (a, b) in plain
            .covariance
            .as_slice()
            .iter()
            .zip(clustered.covariance.as_slice())
        {
            assert!(close(a.unwrap(), b.unwrap(), 1e-9));
        }
        assert!(close(
            plain.tests.score.unwrap().statistic,
            clustered.tests.score.unwrap().statistic,
            1e-9
        ));
        let robust = clustered.robust_covariance.as_ref().unwrap();
        assert_eq!(robust.dim(), 3);
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
            "x has length 19, expected 20"
        );
        let none = Mat::<f64>::zeros(20, 0);
        let error = fit(&d, none.as_ref(), &options(1, Ties::Breslow)).unwrap_err();
        assert_eq!(error.to_string(), "x is empty");
        let levels = vec![0u32; 19];
        assert!(risk_rows(&d, 1, &levels, &RowsOptions::default()).is_err());
        let mut nan = x.clone();
        nan[3] = f64::NAN;
        assert!(fit(&d, column(&nan).as_ref(), &options(1, Ties::Breslow)).is_err());
        let loose = FineGrayOptions {
            tolerance: 0.0,
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

    /// With no weights or all weights positive, every row is used, so the rows
    /// are those of the core on the columns as they are.
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
        assert_eq!(full.status, small.status);
        assert_eq!(full.weight, small.weight);
        for ties in [Ties::Breslow, Ties::Efron] {
            let a = fit(&all, column(&x).as_ref(), &options(1, ties)).unwrap();
            let b = fit(&only, column(&kept_x).as_ref(), &options(1, ties)).unwrap();
            assert_eq!(a.coefficients, b.coefficients);
            assert_eq!(a.robust_covariance, b.robust_covariance);
            assert_eq!((a.n_rows, a.n_used), (20, 16));
            assert_eq!((b.n_rows, b.n_used), (16, 16));
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

    /// A random cohort with ties on a time grid, three causes, fractional
    /// weights with zeros, censoring strata and clusters.
    struct Cohort {
        time: Vec<f64>,
        event: Vec<i32>,
        weight: Vec<f64>,
        stratum: Vec<u32>,
        cluster: Vec<u32>,
        x: Mat<f64>,
    }

    fn cohort(n: usize, seed: u64, grid: usize, censored: usize) -> Cohort {
        let mut rng = Stream::seeded(seed);
        let mut c = Cohort {
            time: Vec::new(),
            event: Vec::new(),
            weight: Vec::new(),
            stratum: Vec::new(),
            cluster: Vec::new(),
            x: Mat::zeros(n, 3),
        };
        for i in 0..n {
            c.time.push((1 + rng.below(grid)) as f64 * 0.5);
            c.event.push(if rng.below(100) < censored {
                0
            } else {
                1 + i32::try_from(rng.below(3)).unwrap()
            });
            c.weight.push(if rng.below(20) == 0 {
                0.0
            } else {
                0.25 + rng.uniform() * 2.0
            });
            c.stratum.push(rng.below(3) as u32);
            c.cluster.push(rng.below(n / 3) as u32);
            c.x[(i, 0)] = rng.normal();
            c.x[(i, 1)] = f64::from(u8::from(rng.below(2) == 0));
            c.x[(i, 2)] = rng.below(4) as f64 - 1.5;
        }
        c
    }

    impl Cohort {
        fn data(&self, strata: bool, clusters: bool) -> SurvData<'_> {
            let mut builder = SurvData::builder(&self.time, &self.event)
                .n_codes(3)
                .weight(&self.weight);
            if strata {
                builder = builder.stratum(&self.stratum);
            }
            if clusters {
                builder = builder.cluster(&self.cluster);
            }
            builder.build().unwrap()
        }
    }

    /// The tolerances of the sweep against the Cox fit of the rows.
    ///
    /// Both compute the same weights and risk-set sums in another order. A
    /// weight after a competing event is the product of the case weight,
    /// `G(t-)` and `1 / G(x-)`, rounded in another order, so the two differ by
    /// `2 epsilon` relative. A risk-set sum of the Cox fit adds and removes up
    /// to `k` row terms, and the sweep adds up to `2 n` terms, so every sum
    /// differs by at most `f = (k + 2 n + 4) epsilon` relative. The rest is the
    /// derivation of `tests/r_fine_gray.rs`: a relative change `f` of the
    /// sums moves the score in column `k` by at most `2 f W r_k`, with `W` the
    /// weight of the events and `r_k` the range of the column, the root by
    /// `db_j = sum_k |V_jk| 2 f W r_k`, the information by
    /// `dI_kl = r_k r_l (2 f W + W sum_m r_m db_m)` and the covariance by
    /// `dV = |V| dI |V|`. The solves add `8 p kappa epsilon` relative, with
    /// `kappa` the 1-norm condition number of the information scaled to a
    /// unit diagonal, which bounds the 2-norm condition number of the scaled
    /// covariates the fit solves on up to the factor `p`.
    struct Bounds {
        f: f64,
        solve: f64,
        kappa: f64,
        db: Vec<f64>,
        dv: Vec<f64>,
    }

    fn bounds(reference: &CoxModelFit, x: MatRef<'_, f64>, rows: usize, n: usize) -> Bounds {
        let p = x.ncols();
        let v: Vec<f64> = reference
            .covariance
            .as_slice()
            .iter()
            .map(|c| c.unwrap())
            .collect();
        let mut factor = v.clone();
        let pivots = crate::ldl::factorise(&mut factor, p, 0.0);
        assert_eq!(pivots.rank, p);
        let information = crate::ldl::inverse(&factor, p);
        let norm = |m: &dyn Fn(usize, usize) -> f64| {
            (0..p)
                .map(|b| (0..p).map(|a| m(a, b).abs()).sum::<f64>())
                .fold(0.0, f64::max)
        };
        let d = |a: usize| information[a * p + a].sqrt();
        let kappa = norm(&|a, b| information[a * p + b] / (d(a) * d(b)))
            * norm(&|a, b| v[a * p + b] * d(a) * d(b));
        let range: Vec<f64> = (0..p)
            .map(|j| {
                let column = (0..x.nrows()).map(|i| x[(i, j)]);
                column.clone().fold(f64::NEG_INFINITY, f64::max)
                    - column.fold(f64::INFINITY, f64::min)
            })
            .collect();
        let w = reference.n_events;
        let f = (rows + 2 * n + 4) as f64 * f64::EPSILON;
        let db: Vec<f64> = (0..p)
            .map(|j| {
                (0..p)
                    .map(|k| v[j * p + k].abs() * 2.0 * f * w * range[k])
                    .sum()
            })
            .collect();
        let beta_term: f64 = (0..p).map(|m| range[m] * db[m]).sum();
        let di = |k: usize, l: usize| range[k] * range[l] * (2.0 * f * w + w * beta_term);
        let dv = (0..p * p)
            .map(|ab| {
                let (a, b) = (ab / p, ab % p);
                (0..p * p)
                    .map(|kl| {
                        let (k, l) = (kl / p, kl % p);
                        v[a * p + k].abs() * di(k, l) * v[l * p + b].abs()
                    })
                    .sum()
            })
            .collect();
        Bounds {
            f,
            solve: 8.0 * p as f64 * kappa * f64::EPSILON,
            kappa,
            db,
            dv,
        }
    }

    /// The largest deviations of `got` from `want` in units of the bounds:
    /// coefficients, standard errors, covariance, robust covariance and log
    /// partial likelihood.
    fn deviations(
        got: &CoxModelFit,
        want: &CoxModelFit,
        x: MatRef<'_, f64>,
        rows: usize,
        n: usize,
    ) -> [f64; 5] {
        let p = x.ncols();
        let b = bounds(want, x, rows, n);
        let units = |g: f64, w: f64, tolerance: f64| (g - w).abs() / tolerance;
        let v = |a: usize, c: usize| want.covariance.get(a, c).unwrap();
        let mut worst = [0.0f64; 5];
        for a in 0..p {
            let beta = want.coefficients[a].unwrap();
            worst[0] = worst[0].max(units(
                got.coefficients[a].unwrap(),
                beta,
                b.db[a] + b.solve * beta.abs().max(1.0),
            ));
            let sd = v(a, a).sqrt();
            worst[1] = worst[1].max(units(
                got.se[a].unwrap(),
                sd,
                f64::midpoint(b.dv[a * p + a], p as f64 * b.solve * v(a, a)) / sd,
            ));
            for c in 0..p {
                worst[2] = worst[2].max(units(
                    got.covariance.get(a, c).unwrap(),
                    v(a, c),
                    b.dv[a * p + c] + p as f64 * b.solve * (v(a, a) * v(c, c)).sqrt(),
                ));
            }
        }
        if let (Some(robust), Some(reference)) = (&got.robust_covariance, &want.robust_covariance) {
            // The robust covariance `V M V`: the relative changes of `V` enter
            // twice, amplified by at most kappa, the meat has the weights
            // squared, and the sums of the residuals add `2 k epsilon`.
            let relative_v = (0..p * p)
                .map(|ac| b.dv[ac] / (v(ac / p, ac / p) * v(ac % p, ac % p)).sqrt())
                .fold(0.0, f64::max);
            let relative = 4.0 * p as f64 * b.solve
                + 2.0 * rows as f64 * f64::EPSILON
                + 2.0 * b.kappa * relative_v
                + 4.0 * b.f;
            let r = |a: usize, c: usize| reference.get(a, c).unwrap();
            for (ac, value) in robust.as_slice().iter().enumerate() {
                let (a, c) = (ac / p, ac % p);
                worst[3] = worst[3].max(units(
                    value.unwrap(),
                    r(a, c),
                    relative * (r(a, a) * r(c, c)).sqrt(),
                ));
            }
        }
        // `W f` from the weights in the log of each risk-set sum, `W f
        // max|x| sum |beta|` in the linear predictor, and the rounding of the
        // sums.
        let eta: f64 = (0..p)
            .map(|j| {
                let largest = (0..x.nrows()).map(|i| x[(i, j)].abs()).fold(0.0, f64::max);
                largest * want.coefficients[j].unwrap().abs()
            })
            .sum();
        let loglik = want.log_partial_likelihood.unwrap();
        worst[4] = units(
            got.log_partial_likelihood.unwrap(),
            loglik,
            2.0 * b.f * want.n_events * (1.0 + eta)
                + (b.solve + rows as f64 * f64::EPSILON) * loglik.abs().max(1.0),
        );
        worst
    }

    /// The sweep against the Cox fit of the rows of [`rows`], on random data
    /// with ties within and across causes and with censoring, fractional
    /// weights with zeros, censoring strata, clusters and three covariates,
    /// for Breslow and Efron ties and every cause.
    #[test]
    fn the_sweep_is_the_cox_fit_of_the_rows() {
        let mut worst = [0.0f64; 5];
        let mut largest = [0.0f64; 5];
        for (n, seed, grid, censored) in [(300, 7, 30, 35), (500, 8, 12, 50), (200, 9, 200, 20)] {
            let c = cohort(n, seed, grid, censored);
            for strata in [false, true] {
                for clusters in [false, true] {
                    for ties in [Ties::Breslow, Ties::Efron] {
                        for cause in 1..=3 {
                            let d = c.data(strata, clusters);
                            let o = options(cause, ties);
                            let got = fit(&d, c.x.as_ref(), &o).unwrap();
                            let want = reference_fit(&d, c.x.as_ref(), &o, false).unwrap();
                            let label = format!(
                                "n {n} strata {strata} clusters {clusters} {ties:?} cause {cause}"
                            );
                            assert!(got.converged && want.converged, "{label}");
                            assert_eq!(got.iterations, want.iterations, "{label}");
                            assert_eq!(got.rank, want.rank, "{label}");
                            assert_eq!(got.n_events.to_bits(), want.n_events.to_bits(), "{label}");
                            assert_eq!(
                                got.robust_covariance.is_some(),
                                want.robust_covariance.is_some(),
                                "{label}"
                            );
                            let rows = rows(&d, cause, &RowsOptions::default()).unwrap().row.len();
                            let units = deviations(&got, &want, c.x.as_ref(), rows, n);
                            for (k, &value) in units.iter().enumerate() {
                                assert!(value <= 1.0, "{label}: {value} of the bound in {k}");
                                worst[k] = worst[k].max(value);
                            }
                            let absolute = [
                                got.coefficients
                                    .iter()
                                    .zip(&want.coefficients)
                                    .map(|(a, b)| (a.unwrap() - b.unwrap()).abs())
                                    .fold(0.0, f64::max),
                                got.se
                                    .iter()
                                    .zip(&want.se)
                                    .map(|(a, b)| (a.unwrap() / b.unwrap() - 1.0).abs())
                                    .fold(0.0, f64::max),
                                got.covariance
                                    .as_slice()
                                    .iter()
                                    .zip(want.covariance.as_slice())
                                    .map(|(a, b)| (a.unwrap() - b.unwrap()).abs())
                                    .fold(0.0, f64::max),
                                got.robust_covariance
                                    .iter()
                                    .flat_map(Covariance::as_slice)
                                    .zip(
                                        want.robust_covariance
                                            .iter()
                                            .flat_map(Covariance::as_slice),
                                    )
                                    .map(|(a, b)| (a.unwrap() - b.unwrap()).abs())
                                    .fold(0.0, f64::max),
                                (got.log_partial_likelihood.unwrap()
                                    - want.log_partial_likelihood.unwrap())
                                .abs(),
                            ];
                            for (slot, value) in largest.iter_mut().zip(absolute) {
                                *slot = slot.max(value);
                            }
                            let score = |fit: &CoxModelFit| fit.tests.score.unwrap().statistic;
                            assert!(
                                (score(&got) - score(&want)).abs()
                                    <= 1e3 * rows as f64 * f64::EPSILON * score(&want).max(1.0),
                                "{label}: score test"
                            );
                        }
                    }
                }
            }
        }
        eprintln!(
            "sweep against rows, worst in units of the bound: coefficient {:.2e}, se {:.2e}, \
             covariance {:.2e}, robust covariance {:.2e}, loglik {:.2e}",
            worst[0], worst[1], worst[2], worst[3], worst[4]
        );
        eprintln!(
            "largest deviations: coefficient {:.2e}, relative se {:.2e}, covariance {:.2e}, \
             robust covariance {:.2e}, loglik {:.2e}",
            largest[0], largest[1], largest[2], largest[3], largest[4]
        );
    }

    /// With one censoring stratum for everyone the fit is the fit without
    /// strata, bit for bit.
    #[test]
    fn one_censoring_stratum_is_no_strata() {
        let c = cohort(300, 11, 25, 35);
        let single = vec![4u32; 300];
        let one = SurvData::builder(&c.time, &c.event)
            .n_codes(3)
            .weight(&c.weight)
            .stratum(&single)
            .cluster(&c.cluster)
            .build()
            .unwrap();
        for ties in [Ties::Breslow, Ties::Efron] {
            let o = options(1, ties);
            assert_eq!(
                fit(&one, c.x.as_ref(), &o).unwrap(),
                fit(&c.data(false, true), c.x.as_ref(), &o).unwrap()
            );
            assert_eq!(
                rows(&one, 1, &RowsOptions::default()).unwrap(),
                rows(&c.data(false, false), 1, &RowsOptions::default()).unwrap()
            );
        }
    }

    /// Censoring strata change the weights: each competing event is reweighted
    /// by the censoring distribution of its own stratum, and its rows are split
    /// at the censoring times of that stratum.
    #[test]
    fn censoring_strata_reweight_by_their_own_curve() {
        let c = cohort(300, 12, 25, 35);
        let o = options(1, Ties::Breslow);
        let pooled = fit(&c.data(false, false), c.x.as_ref(), &o).unwrap();
        let stratified = fit(&c.data(true, false), c.x.as_ref(), &o).unwrap();
        assert_ne!(pooled.coefficients, stratified.coefficients);
        let expanded = rows(&c.data(true, false), 1, &RowsOptions::default()).unwrap();
        for k in 0..expanded.row.len() {
            let i = expanded.row[k] as usize;
            if expanded.start[k] > 0.0 {
                // A later row starts at a censoring time of the individual's
                // own stratum.
                assert!((0..300).any(|j| {
                    c.event[j] == 0
                        && c.weight[j] > 0.0
                        && c.stratum[j] == c.stratum[i]
                        && c.time[j].to_bits() == expanded.start[k].to_bits()
                }));
            }
        }
    }

    /// Exact ties are the Cox fit of one row per individual, which needs every
    /// row weight to be 1: case weights of 1 and no censoring after a competing
    /// event. The fit is then that of the rows, bit for bit, and otherwise it is
    /// refused.
    #[test]
    fn exact_ties_fit_the_rows_when_every_weight_is_one() {
        let time = [1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 8.0, 8.0, 9.0];
        let event = [0, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1];
        let x = column(&[
            0.5, 1.0, -1.0, 0.0, 2.0, 1.0, -0.5, 0.3, 1.5, -1.0, 0.2, 0.7,
        ]);
        let unit = [1.0; 12];
        let d = data(&time, &event, &unit, None);
        let o = options(1, Ties::Exact);
        let got = fit(&d, x.as_ref(), &o).unwrap();
        let want = reference_fit(&d, x.as_ref(), &o, false).unwrap();
        assert!(got.converged);
        assert_eq!(got.coefficients, want.coefficients);
        assert_eq!(got.covariance, want.covariance);
        assert_eq!(got.log_partial_likelihood, want.log_partial_likelihood);
        assert_eq!(got.tests, want.tests);

        let mut weight = unit;
        weight[3] = 2.0;
        let error = fit(&data(&time, &event, &weight, None), x.as_ref(), &o).unwrap_err();
        assert!(error.to_string().contains("case weights of 1"), "{error}");
        let mut late = event;
        late[10] = 0;
        let error = fit(&data(&time, &late, &unit, None), x.as_ref(), &o).unwrap_err();
        assert!(error.to_string().contains("censoring after"), "{error}");
        let cluster: Vec<u32> = (0..12).collect();
        let error = fit(&data(&time, &event, &unit, Some(&cluster)), x.as_ref(), &o).unwrap_err();
        assert!(error.to_string().contains("cluster"), "{error}");
    }
}
