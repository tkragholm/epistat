//! The Fine-Gray subdistribution hazard model for one cause, fitted on one
//! covariate, with the Kaplan-Meier estimate of the censoring distribution it
//! needs.
//!
//! The model is fitted as a weighted Cox model on rows built from the data \[1\],
//! so the fit itself is [`crate::cox`]. This module builds the rows. The fit is
//! on one covariate, like the Cox fit, and a general design matrix is later
//! work.
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
//! The rows are counting-process intervals. [`FineGrayData::expand`] writes one
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
//! [`FineGrayData::risk_rows`] sums the competing-event rows that share an
//! interval and a covariate value. These rows all end without an event, so
//! they enter the partial likelihood only through weighted sums, and adding
//! them is exact. The weight on the interval opening at censoring time `c_j`
//! factorises as `G(c_j) w_i / G(x_i-)`, so one prefix sum of `w_i / G(x_i-)`
//! over the event times, per covariate level, gives every interval.
//! [`fine_gray_fit`] fits on these collapsed rows, or on the expanded rows when
//! cluster codes are given, because the robust variance needs each individual's
//! own rows.
//!
//! # Edge cases
//!
//! Cause 0 is censoring and is refused as a cause of interest. An individual
//! whose `G(x_i-)` is zero contributes no weight after the event. The estimates
//! of `expand` and `risk_rows` are the same, and `n_rows` of the fit counts the
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

#![expect(
    clippy::cast_possible_truncation,
    reason = "the entry points check that the row count fits in u32"
)]

use std::cmp::Ordering;
use std::collections::HashMap;

use crate::check::{check_lengths, check_row_count};
use crate::cox::{CoxData, CoxFit, Ties, fit_cox_single_covariate};
use crate::error::{Error, Result};

/// The Kaplan-Meier estimate of the censoring distribution `G`, as a step
/// function.
///
/// Censoring is the event and every failure, of any cause, is censored for it.
/// The risk set at a censoring time `t` is the weight with a time strictly
/// after `t` plus the weight censored at `t`, so a failure and a censoring at
/// the same time are ordered failure first. Using `time >= t` would add every
/// tied failure to the risk set and change every weight derived from `G`.
struct CensoringCurve {
    /// Censoring times, ascending. They are also the times the Fine-Gray
    /// weights change, so the rows are split here.
    times: Vec<f64>,
    /// `G` from each of `times` onward.
    survival: Vec<f64>,
}

impl CensoringCurve {
    // Exact `==` groups rows that share a time.
    #[expect(clippy::float_cmp, reason = "rows that share a time are grouped")]
    fn estimate(time: &[f64], event: &[i32], weight: &[f64]) -> Self {
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
        Self {
            times,
            survival: steps,
        }
    }

    /// `G` at `t`, after any censoring at `t`. 1 before the first censoring.
    fn at(&self, t: f64) -> f64 {
        match self.times.partition_point(|&c| c <= t) {
            0 => 1.0,
            index => self.survival[index - 1],
        }
    }

    /// `G` just before `t`, so without any censoring at `t`.
    fn before(&self, t: f64) -> f64 {
        match self.times.partition_point(|&c| c < t) {
            0 => 1.0,
            index => self.survival[index - 1],
        }
    }
}

/// Counting-process rows for a Cox fit. `row` is an input row: the individual
/// for [`FineGrayData::expand`], and for [`FineGrayData::risk_rows`] the
/// individual or any individual with the covariate value the collapsed row
/// carries.
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
pub struct FineGrayData<'a> {
    time: &'a [f64],
    event: &'a [i32],
    weight: &'a [f64],
    cause: i32,
    curve: CensoringCurve,
    horizon: f64,
}

impl<'a> FineGrayData<'a> {
    /// The data of one analysis: one `time`, cause code `event` (0 for
    /// censoring) and `weight` per individual, and the `cause` of interest.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the slices differ in length or `cause` is 0.
    pub fn new(time: &'a [f64], event: &'a [i32], weight: &'a [f64], cause: i32) -> Result<Self> {
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
            curve: CensoringCurve::estimate(time, event, weight),
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
    #[must_use]
    pub fn expand(&self) -> Rows {
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
            let first = self.curve.times.partition_point(|&c| c < t);
            let mut previous = 0.0f64;
            for &boundary in &self.curve.times[first..] {
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
    pub fn risk_rows(&self, level: &[u32]) -> Result<Rows> {
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

        let censoring = &self.curve.times;
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

/// Fit the subdistribution log hazard ratio of `cause` for one covariate.
///
/// Builds the collapsed rows ([`FineGrayData::risk_rows`]) and fits the weighted
/// Cox model with delayed entry on them. `n_rows` in the result counts the
/// collapsed rows. Every individual is at risk from time 0.
///
/// With `cluster`, one code per individual, the fit is on the uncollapsed rows
/// ([`FineGrayData::expand`]), because the robust variance needs each
/// individual's rows. The estimate is the same and `n_rows` counts the expanded
/// rows.
///
/// # Errors
///
/// [`Error::Input`] when the slices differ in length or `cause` is 0.
pub fn fine_gray_fit(
    time: &[f64],
    event: &[i32],
    weight: &[f64],
    x: &[f64],
    cause: i32,
    ties: Ties,
    cluster: Option<&[u32]>,
) -> Result<CoxFit> {
    check_lengths(&[
        ("time", time.len()),
        ("x", x.len()),
        ("cluster", cluster.map_or(time.len(), <[u32]>::len)),
    ])?;
    let data = FineGrayData::new(time, event, weight, cause)?;
    let rows = if cluster.is_some() {
        data.expand()
    } else {
        data.risk_rows(&float_codes(x))?
    };
    let row_x: Vec<f64> = rows.row.iter().map(|&i| x[i as usize]).collect();
    let row_cluster: Option<Vec<u32>> =
        cluster.map(|cluster| rows.row.iter().map(|&i| cluster[i as usize]).collect());
    let is_event: Vec<bool> = rows.event.iter().map(|&code| code == 1).collect();
    let stratum = vec![0u32; rows.row.len()];
    fit_cox_single_covariate(&CoxData {
        x: &row_x,
        is_event: &is_event,
        start: &rows.start,
        stop: &rows.stop,
        weight: &rows.weight,
        stratum: &stratum,
        ties,
        cluster: row_cluster.as_deref(),
    })
}

#[cfg(test)]
mod tests {
    use super::{CensoringCurve, FineGrayData, Rows, fine_gray_fit};
    use crate::cox::{CoxData, CoxFit, Ties, fit_cox_single_covariate};

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
        let data = FineGrayData::new(&TIME, &EVENT, &weight, 1).unwrap();
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
        let data = FineGrayData::new(&time, &event, &weight, 1).unwrap();
        let rows = data.expand();
        let subject: Vec<usize> = (0..rows.row.len()).filter(|&k| rows.row[k] == 1).collect();
        let stops: Vec<f64> = subject.iter().map(|&k| rows.stop[k]).collect();
        assert_eq!(stops, [2.0, 4.0, 6.0]);
        for (&k, want) in subject.iter().zip([1.0, 0.8, 0.533_333_333_3]) {
            assert!(close(rows.weight[k], want, 1e-9), "{:?}", rows.weight);
        }
    }

    fn fit_rows(rows: &Rows, x: &[f64]) -> CoxFit {
        let row_x: Vec<f64> = rows.row.iter().map(|&i| x[i as usize]).collect();
        let is_event: Vec<bool> = rows.event.iter().map(|&e| e == 1).collect();
        fit_cox_single_covariate(&CoxData {
            x: &row_x,
            is_event: &is_event,
            start: &rows.start,
            stop: &rows.stop,
            weight: &rows.weight,
            stratum: &vec![0; rows.row.len()],
            ties: Ties::Breslow,
            cluster: None,
        })
        .expect("consistent lengths")
    }

    /// `survival::finegray(..., etype = "c1")`, then
    /// `coxph(..., weights = fgwt, ties = "breslow")`; the model-based
    /// (`naive.var`) standard error. R 4.6.1, survival 3.8.6.
    #[test]
    fn fine_gray_fit_matches_r_on_both_forms() {
        let weight = [1.0; 20];
        let x = group();
        let data = FineGrayData::new(&TIME, &EVENT, &weight, 1).unwrap();
        let levels: Vec<u32> = x.iter().map(|&v| u32::from(v > 0.5)).collect();
        for rows in [data.expand(), data.risk_rows(&levels).unwrap()] {
            let fit = fit_rows(&rows, &x);
            assert!(fit.converged);
            assert!(
                close(fit.log_hazard_ratio, 0.151_472_434_751_664, 1e-10),
                "{fit:?}"
            );
            assert!(
                close(fit.standard_error, 0.634_301_730_060_493, 1e-10),
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
        let curve =
            CensoringCurve::estimate(&[1.0, 2.0, 2.0, 3.0, 4.0], &[0, 0, 1, 0, 1], &[1.0; 5]);
        assert_eq!(curve.times, [1.0, 2.0, 3.0]);
        assert!(close(curve.at(0.5), 1.0, 1e-15));
        assert!(close(curve.at(1.0), 0.8, 1e-15));
        assert!(close(curve.at(2.5), 0.8 * (1.0 - 1.0 / 3.0), 1e-15));
        assert!(close(
            curve.at(3.0),
            0.8 * (1.0 - 1.0 / 3.0) * (1.0 - 1.0 / 2.0),
            1e-15
        ));
    }

    #[test]
    fn the_entry_point_fits_both_forms_to_the_same_estimate() {
        let weight = [1.0; 20];
        let x = group();
        let plain = fine_gray_fit(&TIME, &EVENT, &weight, &x, 1, Ties::Breslow, None).unwrap();
        let family: Vec<u32> = (0..20).map(|i| i % 10).collect();
        let clustered =
            fine_gray_fit(&TIME, &EVENT, &weight, &x, 1, Ties::Breslow, Some(&family)).unwrap();
        assert!(close(
            plain.log_hazard_ratio,
            clustered.log_hazard_ratio,
            1e-10
        ));
        assert!(plain.n_rows < clustered.n_rows);
        assert!(clustered.robust_standard_error.is_some());
    }

    #[test]
    fn cause_zero_and_mismatched_slices_are_refused() {
        let weight = [1.0; 20];
        let x = group();
        let zero = fine_gray_fit(&TIME, &EVENT, &weight, &x, 0, Ties::Breslow, None);
        assert_eq!(
            zero.unwrap_err().to_string(),
            "cause 0 is censoring, not a cause"
        );
        let short = fine_gray_fit(&TIME, &EVENT, &weight, &x[..19], 1, Ties::Breslow, None);
        assert!(short.is_err());
    }
}
