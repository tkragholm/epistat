//! Windows for the case-crossover and case-time-control designs: the hazard
//! window before each event, its referent windows, and the exposure in each.
//! The analysis of the windows is [`clogit`](crate::clogit::clogit).
//!
//! # Design
//!
//! Each case is compared with itself. For an event on day `t`, the hazard
//! window is the `h` days `[t - l - h, t - l - 1]` before it, with `l` the lag,
//! and each referent window is the hazard window moved by a shift `s`,
//! `[t - l - h + s, t - l - 1 + s]` \[1\]. A set holds the hazard window, the
//! case, and its referent windows, the controls, and the analysis is the
//! conditional logistic regression of the exposure in the window on the case
//! indicator within sets. Days are integers, as days since 1970-01-01, and both
//! ends of a window are included. The shifts are those of the [`Referent`]
//! chosen:
//!
//! - [`Referent::Unidirectional`]: `-step, -2 step, .., -n step`, windows
//!   before the hazard window only.
//! - [`Referent::Bidirectional`]: the same on both sides, `+-step, ..,
//!   +-n step` \[2\].
//! - [`Referent::TimeStratified`]: every shift that is a multiple of 7 days
//!   and puts the day `t + s` in the same calendar month, or year, as `t`. The
//!   referents are the days with the same weekday as the event in the same
//!   stratum, which keeps the referent days balanced around the event and
//!   avoids the bias of unidirectional selection when the exposure trends
//!   \[3, 4\].
//!
//! A referent window is dropped when it contains the event day or does not lie
//! within the observation period of the person, and a set that has no referent
//! left is dropped with its case. A set whose hazard window is not within
//! observation is dropped too. Each of these is counted or listed in
//! [`Windows`]. The shifts must be at least `h` days apart from zero (a step,
//! and for time stratification 7 days, of at least `h`) so that referent windows
//! do not overlap the hazard window.
//!
//! # Exposure
//!
//! [`exposure_in_windows`] measures each window against the intervals of an
//! exposure series, such as the days covered by a dispensing: the number of days
//! of the window covered by any interval of the person, which counts a day
//! covered by two intervals once, or whether that number is above zero. The
//! intervals are sorted and merged by person, and each window is found by binary
//! search, so the cost is `O((I + W) log I)` for `I` intervals and `W` windows.
//!
//! # Case-time-control
//!
//! The case-time-control design \[5\] adds sets of control persons, built the
//! same way at a date matched to each case, and fits the conditional logistic
//! model with the exposure and the exposure times the group, 1 for the sets of
//! the cases and 0 for the sets of the controls. [`case_time_control_design`]
//! forms those two columns. `exp` of the exposure coefficient is the odds ratio
//! of the control persons, the time trend of the exposure, and `exp` of the
//! interaction is the ratio of the case odds ratio to it, the case-time-control
//! estimate. It removes the bias of a trend in the exposure only if the trend is
//! the same in cases and controls, which fails when it differs for reasons
//! related to the event \[6\]. The matching of the control persons is the
//! caller's, for example by `epistat-matching`.
//!
//! # Agreement with R
//!
//! The tests compare every window and every count of the construction with an
//! explicit R loop over calendar days (R's `Date`, for weekday and month
//! length), the exposure with the union of days in R, and the analysis with
//! `survival::clogit`, with the exact likelihood and with a cluster-robust
//! variance by the approximate method and `cluster(person)`.
//!
//! # References
//!
//! 1. Maclure M. The case-crossover design: a method for studying transient
//!    effects on the risk of acute events. Am J Epidemiol 1991;133(2):144-53.
//! 2. Navidi W. Bidirectional case-crossover designs for exposures with time
//!    trends. Biometrics 1998;54(2):596-605.
//! 3. Lumley T, Levy D. Bias in the case-crossover design: implications for
//!    studies of air pollution. Environmetrics 2000;11(6):689-704.
//! 4. Janes H, Sheppard L, Lumley T. Case-crossover analyses of air pollution
//!    exposure data: referent selection strategies and their implications for
//!    bias. Epidemiology 2005;16(6):717-26.
//! 5. Suissa S. The case-time-control design. Epidemiology 1995;6(3):248-53.
//! 6. Greenland S. Confounding and exposure trends in case-crossover and
//!    case-time-control designs. Epidemiology 1996;7(3):231-9.

use epistat_linalg::Mat;
use rayon::prelude::*;

use crate::error::{Error, Result};

/// The calendar stratum within which a time-stratified referent shares the
/// weekday of the event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StratumUnit {
    /// The same calendar month and year.
    Month,
    /// The same calendar year.
    Year,
}

/// How the referent windows of an event are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Referent {
    /// `n` windows before the hazard window, `step` days apart.
    Unidirectional {
        /// The number of referent windows.
        n: usize,
        /// The days between windows, at least the hazard window length.
        step: i64,
    },
    /// `n` windows on each side of the hazard window, `step` days apart.
    Bidirectional {
        /// The number of referent windows on each side.
        n: usize,
        /// The days between windows, at least the hazard window length.
        step: i64,
    },
    /// The same weekday within the calendar stratum of the event.
    TimeStratified {
        /// The stratum.
        unit: StratumUnit,
    },
}

/// The days of observation of each event's person, both ends included.
#[derive(Debug, Clone, Copy)]
pub struct Observation<'a> {
    /// The first observed day of each event's person.
    pub start: &'a [i64],
    /// The last observed day of each event's person.
    pub end: &'a [i64],
}

/// The windows of a set of events, grouped by set, the hazard window first.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Windows {
    /// The set of each window: the position of its event in the input.
    pub set: Vec<u32>,
    /// The person of each window.
    pub person: Vec<u32>,
    /// The first day of each window.
    pub start: Vec<i64>,
    /// The last day of each window.
    pub end: Vec<i64>,
    /// Whether the window is the hazard window of its set.
    pub is_case: Vec<bool>,
    /// The sets dropped because no referent window was left.
    pub sets_without_referent: Vec<u32>,
    /// The sets dropped because the hazard window is not within observation.
    pub sets_case_outside_observation: Vec<u32>,
    /// The referent windows dropped because they are not within observation.
    pub n_referents_outside_observation: usize,
    /// The referent windows dropped because they contain the event day.
    pub n_referents_on_event_day: usize,
}

impl Windows {
    /// The number of windows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.set.len()
    }

    /// Whether there is no window.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.set.is_empty()
    }

    /// A value of each event repeated on every window of its set, for a
    /// covariate that belongs to the event, such as the group.
    ///
    /// # Panics
    ///
    /// When `per_event` is shorter than the largest set number.
    #[must_use]
    pub fn expand<T: Copy>(&self, per_event: &[T]) -> Vec<T> {
        self.set.iter().map(|&s| per_event[s as usize]).collect()
    }
}

/// How an exposure interval series is turned into a number per window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Measure {
    /// The days of the window covered by the series.
    Days,
    /// 1 when any day of the window is covered, 0 otherwise.
    Indicator,
}

/// Intervals of exposure, both ends included, one per element.
#[derive(Debug, Clone, Copy)]
pub struct ExposureIntervals<'a> {
    /// The person of each interval.
    pub person: &'a [u32],
    /// The first day of each interval.
    pub start: &'a [i64],
    /// The last day of each interval, not before the first.
    pub end: &'a [i64],
}

/// Events per task of the construction.
const BLOCK: usize = 1024;

fn invalid(message: impl Into<String>) -> Error {
    Error::input(message)
}

/// Days since 1970-01-01 of a civil date.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The civil date `(year, month, day)` of a day count.
fn civil_from_days(days: i64) -> (i64, i64, i64) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe + era * 400 + i64::from(m <= 2), m, d)
}

/// The first and last day of the stratum that holds `t`.
fn stratum_of(t: i64, unit: StratumUnit) -> (i64, i64) {
    let (y, m, _) = civil_from_days(t);
    match unit {
        StratumUnit::Month => {
            let first = days_from_civil(y, m, 1);
            let next = if m == 12 {
                days_from_civil(y + 1, 1, 1)
            } else {
                days_from_civil(y, m + 1, 1)
            };
            (first, next - 1)
        }
        StratumUnit::Year => (days_from_civil(y, 1, 1), days_from_civil(y + 1, 1, 1) - 1),
    }
}

/// The shifts of the referent windows of an event on day `t`, in increasing
/// order.
#[expect(
    clippy::cast_possible_wrap,
    reason = "the number of referent windows is a small count"
)]
fn shifts(t: i64, referent: Referent) -> Vec<i64> {
    match referent {
        Referent::Unidirectional { n, step } => (1..=n as i64).rev().map(|k| -k * step).collect(),
        Referent::Bidirectional { n, step } => {
            let mut s: Vec<i64> = (1..=n as i64).rev().map(|k| -k * step).collect();
            s.extend((1..=n as i64).map(|k| k * step));
            s
        }
        Referent::TimeStratified { unit } => {
            let (first, last) = stratum_of(t, unit);
            let below = (t - first) / 7;
            let above = (last - t) / 7;
            (-below..=above)
                .filter(|&k| k != 0)
                .map(|k| 7 * k)
                .collect()
        }
    }
}

/// Builds the hazard and referent windows of each event.
///
/// `event_day` and `person` have one entry per event, and so does each column
/// of `observation` when given. `hazard_days` is the length `h` of the hazard
/// window, at least 1, and `lag` the days between its end and the event, at
/// least 0. See the module documentation for what is dropped.
///
/// # Errors
///
/// [`Error::Input`] when the columns differ in length, `hazard_days` is below
/// 1, `lag` is negative, a referent has no window or a step below
/// `hazard_days` (below 7 days for a time-stratified referent, whose smallest
/// shift is 7), an observation period ends before it starts, or there are more
/// events than a `u32` addresses.
#[expect(
    clippy::cast_possible_truncation,
    reason = "the number of events is checked against u32 first"
)]
pub fn build_windows(
    event_day: &[i64],
    person: &[u32],
    hazard_days: i64,
    lag: i64,
    referent: Referent,
    observation: Option<Observation<'_>>,
) -> Result<Windows> {
    let n = event_day.len();
    if person.len() != n {
        return Err(invalid(
            "event_day and person must have one entry per event",
        ));
    }
    if let Some(o) = observation {
        if o.start.len() != n || o.end.len() != n {
            return Err(invalid(
                "the observation columns must have one entry per event",
            ));
        }
        if let Some(i) = (0..n).find(|&i| o.end[i] < o.start[i]) {
            return Err(invalid(format!(
                "event {i} has an observation period that ends before it starts"
            )));
        }
    }
    if u32::try_from(n).is_err() {
        return Err(invalid("more events than a u32 addresses"));
    }
    if hazard_days < 1 || lag < 0 {
        return Err(invalid("hazard_days must be at least 1 and lag at least 0"));
    }
    let smallest = match referent {
        Referent::Unidirectional { n, step } | Referent::Bidirectional { n, step } => {
            if n == 0 {
                return Err(invalid("a referent needs at least one window"));
            }
            step
        }
        Referent::TimeStratified { .. } => 7,
    };
    if smallest < hazard_days {
        return Err(invalid(format!(
            "referent windows are {smallest} days apart, fewer than the {hazard_days} days of the hazard window, so they would overlap it"
        )));
    }
    let blocks: Vec<Windows> = (0..n.div_ceil(BLOCK))
        .into_par_iter()
        .map(|block| {
            let mut w = Windows::default();
            for i in block * BLOCK..((block + 1) * BLOCK).min(n) {
                let t = event_day[i];
                let (hs, he) = (t - lag - hazard_days, t - lag - 1);
                let within =
                    |a: i64, b: i64| observation.is_none_or(|o| a >= o.start[i] && b <= o.end[i]);
                if !within(hs, he) {
                    w.sets_case_outside_observation.push(i as u32);
                    continue;
                }
                let mut referents = Vec::new();
                for s in shifts(t, referent) {
                    let (a, b) = (hs + s, he + s);
                    if a <= t && t <= b {
                        w.n_referents_on_event_day += 1;
                    } else if !within(a, b) {
                        w.n_referents_outside_observation += 1;
                    } else {
                        referents.push((a, b));
                    }
                }
                if referents.is_empty() {
                    w.sets_without_referent.push(i as u32);
                    continue;
                }
                for (k, (a, b)) in std::iter::once((hs, he)).chain(referents).enumerate() {
                    w.set.push(i as u32);
                    w.person.push(person[i]);
                    w.start.push(a);
                    w.end.push(b);
                    w.is_case.push(k == 0);
                }
            }
            w
        })
        .collect();
    let mut out = Windows::default();
    for w in blocks {
        out.set.extend(w.set);
        out.person.extend(w.person);
        out.start.extend(w.start);
        out.end.extend(w.end);
        out.is_case.extend(w.is_case);
        out.sets_without_referent.extend(w.sets_without_referent);
        out.sets_case_outside_observation
            .extend(w.sets_case_outside_observation);
        out.n_referents_outside_observation += w.n_referents_outside_observation;
        out.n_referents_on_event_day += w.n_referents_on_event_day;
    }
    Ok(out)
}

/// The exposure in each window: the days of the window covered by the
/// intervals of the window's person, or whether there are any, as
/// [`Measure`] says.
///
/// Intervals that overlap or touch are merged first, so a day is counted once.
///
/// # Errors
///
/// [`Error::Input`] when the columns of `exposure` differ in length or an
/// interval ends before it starts.
pub fn exposure_in_windows(
    windows: &Windows,
    exposure: &ExposureIntervals<'_>,
    measure: Measure,
) -> Result<Vec<f64>> {
    let m = exposure.person.len();
    if exposure.start.len() != m || exposure.end.len() != m {
        return Err(invalid(
            "person, start and end of the exposure must have one entry per interval",
        ));
    }
    if let Some(i) = (0..m).find(|&i| exposure.end[i] < exposure.start[i]) {
        return Err(invalid(format!(
            "exposure interval {i} ends before it starts"
        )));
    }
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by_key(|&i| (exposure.person[i], exposure.start[i], exposure.end[i]));
    // Merged intervals as (person, start, end), sorted.
    let mut merged: Vec<(u32, i64, i64)> = Vec::with_capacity(m);
    for &i in &order {
        let (p, a, b) = (exposure.person[i], exposure.start[i], exposure.end[i]);
        match merged.last_mut() {
            Some(last) if last.0 == p && a <= last.2.saturating_add(1) => last.2 = last.2.max(b),
            _ => merged.push((p, a, b)),
        }
    }
    let out = (0..windows.len())
        .into_par_iter()
        .map(|r| {
            let (p, a, b) = (windows.person[r], windows.start[r], windows.end[r]);
            let mut k = merged.partition_point(|&(q, _, e)| (q, e) < (p, a));
            let mut days = 0_i64;
            while k < merged.len() && merged[k].0 == p && merged[k].1 <= b {
                days += merged[k].2.min(b) - merged[k].1.max(a) + 1;
                k += 1;
            }
            match measure {
                Measure::Days => days as f64,
                Measure::Indicator => f64::from(u8::from(days > 0)),
            }
        })
        .collect();
    Ok(out)
}

/// The two columns of the case-time-control model: the exposure, and the
/// exposure times the group, 1 for the sets of the cases and 0 for the sets of
/// the control persons.
///
/// # Errors
///
/// [`Error::Input`] when the two have different lengths.
pub fn case_time_control_design(exposure: &[f64], case_group: &[bool]) -> Result<Mat<f64>> {
    if exposure.len() != case_group.len() {
        return Err(invalid(
            "exposure and case_group must have one entry per window",
        ));
    }
    Ok(Mat::from_fn(exposure.len(), 2, |i, j| {
        if j == 0 || case_group[i] {
            exposure[i]
        } else {
            0.0
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn civil_dates_round_trip() {
        for days in [
            -800_000_i64,
            -1,
            0,
            59,
            60,
            11_016,
            18_321,
            18_322,
            2_932_896,
        ] {
            let (y, m, d) = civil_from_days(days);
            assert_eq!(days_from_civil(y, m, d), days);
        }
        assert_eq!(civil_from_days(18_321), (2020, 2, 29));
        assert_eq!(days_from_civil(1970, 1, 1), 0);
    }

    #[test]
    fn time_stratified_shifts_stay_in_the_month_and_keep_the_weekday() {
        // 2020-02-29 was a Saturday: the other Saturdays of February are the 1st, 8th, 15th, 22nd.
        let t = days_from_civil(2020, 2, 29);
        let s = shifts(
            t,
            Referent::TimeStratified {
                unit: StratumUnit::Month,
            },
        );
        assert_eq!(s, vec![-28, -21, -14, -7]);
        // 2019-01-31: Thursdays of January 2019 are the 3rd, 10th, 17th, 24th.
        let t = days_from_civil(2019, 1, 31);
        let s = shifts(
            t,
            Referent::TimeStratified {
                unit: StratumUnit::Month,
            },
        );
        assert_eq!(s, vec![-28, -21, -14, -7]);
    }
}
