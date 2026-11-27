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
//!   and puts the day `t + s` in the same stratum as `t`: the calendar month,
//!   quarter or year, or a block of fixed length ([`StratumUnit`]). The
//!   referents are the days with the same weekday as the event in the same
//!   stratum, which keeps the referent days balanced around the event and
//!   avoids the bias of unidirectional selection when the exposure trends
//!   \[3, 4\].
//! - [`Referent::Stratified`]: the same with a spacing other than 7 days, 14
//!   or 28 to keep the weekday with fewer referents, or at least the hazard
//!   window for a hazard window longer than a week, such as 35 days in the
//!   calendar year for a 30-day window.
//! - [`build_windows_in_strata`]: the same with a stratum the caller gives for
//!   each event, such as one that follows a holiday calendar.
//!
//! A referent window is dropped when it contains the event day or does not lie
//! within the observation period of the person, and a set that has no referent
//! left is dropped with its case. A set whose hazard window is not within
//! observation is dropped too. Each of these is counted or listed in
//! [`Windows`]. The shifts must be at least `h` days apart from zero (a step,
//! and for time stratification 7 days or the spacing, of at least `h`) so that
//! referent windows do not overlap the hazard window.
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
//! caller's, by `epistat-matching` or by [`sample_controls`], which draws
//! control persons at the date of each case: people without the event by
//! then for the case-time-control design, or people who have it later for the
//! case-case-time-control design \[7\], whose controls are closer to the cases
//! in whatever leads to the event. [`case_time_control_windows`] builds the
//! windows of both and the group of each window.
//!
//! # Agreement with R
//!
//! The tests compare every window and every count of the construction with an
//! explicit R loop over calendar days (R's `Date`, for weekday and month
//! length), the exposure with the union of days in R, and the analysis with
//! `survival::clogit`, with the exact likelihood and with a cluster-robust
//! variance by the approximate method and `cluster(person)`. The
//! time-stratified referents by month, 7 and 14 days apart, are compared with
//! the referent days of `rSPARCS::CXover.data` (rSPARCS 0.1.1), and the
//! analysis in fixed 28- and 35-day strata and by month with
//! `season::casecross` (season 0.3.16), which puts the hazard on the event day
//! itself; a hazard window here ends the day before, so its one-day windows
//! stand for the day after them. No R package samples control persons at the
//! case's date by these rules, so [`sample_controls`] is checked for
//! eligibility, uniformity and thread independence on its own.
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
//! 7. Wang S, Linkletter C, Maclure M, Dore D, Mor V, Buka S, et al. Future
//!    cases as present controls to adjust for exposure trend bias in
//!    case-only studies. Epidemiology 2011;22(4):568-74.

mod controls;

pub use controls::{
    Candidates, CaseEvents, CaseTimeControlWindows, ControlRule, Controls,
    case_time_control_windows, sample_controls,
};

use epistat_core::day;
use epistat_linalg::Mat;
use rayon::prelude::*;

use crate::error::{Error, Result};

/// The calendar stratum within which time-stratified referents are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StratumUnit {
    /// The same calendar month and year.
    Month,
    /// The same calendar quarter and year: January to March, April to June,
    /// July to September, October to December.
    Quarter,
    /// The same calendar year.
    Year,
    /// Consecutive blocks of `length` days, the first starting on day
    /// `origin`, as the strata of `season::casecross` with `stratalength`.
    Fixed {
        /// The first day of a stratum.
        origin: i64,
        /// The days of each stratum, at least 1.
        length: i64,
    },
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
    /// The same weekday within the calendar stratum of the event: every
    /// shift that is a multiple of 7 days and stays in the stratum.
    TimeStratified {
        /// The stratum.
        unit: StratumUnit,
    },
    /// Every shift that is a multiple of `spacing` days and stays in the
    /// calendar stratum of the event. A spacing of 7 is
    /// [`Referent::TimeStratified`]; 14 or 28 keep the weekday with fewer
    /// referents, and a spacing of at least the hazard window allows a hazard
    /// window longer than a week.
    Stratified {
        /// The stratum.
        unit: StratumUnit,
        /// The days between referents, at least the hazard window length.
        spacing: i64,
    },
}

/// The first and last day of the stratum of each event, both included, for
/// strata the caller defines.
#[derive(Debug, Clone, Copy)]
pub struct EventStrata<'a> {
    /// The first day of the stratum of each event.
    pub first: &'a [i64],
    /// The last day of the stratum of each event.
    pub last: &'a [i64],
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
    Error::invalid(message)
}

/// The first day of the month `months` months after January of year `y`.
fn first_of_month(y: i64, months: i64) -> i64 {
    day::from_civil(y + months.div_euclid(12), months.rem_euclid(12) + 1, 1)
}

/// The first and last day of the stratum that holds `t`.
fn stratum_of(t: i64, unit: StratumUnit) -> (i64, i64) {
    let (y, m, _) = day::to_civil(t);
    match unit {
        StratumUnit::Month => (first_of_month(y, m - 1), first_of_month(y, m) - 1),
        StratumUnit::Quarter => {
            let q = (m - 1) / 3;
            (first_of_month(y, 3 * q), first_of_month(y, 3 * q + 3) - 1)
        }
        StratumUnit::Year => (day::from_civil(y, 1, 1), day::from_civil(y + 1, 1, 1) - 1),
        StratumUnit::Fixed { origin, length } => {
            let first = origin + (t - origin).div_euclid(length) * length;
            (first, first + length - 1)
        }
    }
}

/// The shifts that are nonzero multiples of `spacing` and keep `t` within
/// `[first, last]`, in increasing order.
fn stratum_shifts(t: i64, first: i64, last: i64, spacing: i64) -> Vec<i64> {
    let below = (t - first) / spacing;
    let above = (last - t) / spacing;
    (-below..=above)
        .filter(|&k| k != 0)
        .map(|k| spacing * k)
        .collect()
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
            stratum_shifts(t, first, last, 7)
        }
        Referent::Stratified { unit, spacing } => {
            let (first, last) = stratum_of(t, unit);
            stratum_shifts(t, first, last, spacing)
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
/// An input error when the columns differ in length, `hazard_days` is below
/// 1, `lag` is negative, a referent has no window or a step below
/// `hazard_days` (below 7 days for a time-stratified referent, whose smallest
/// shift is 7, and below the spacing of a stratified one), a fixed stratum or
/// a spacing is shorter than a day, an observation period ends before it
/// starts, or there are more events than a `u32` addresses.
pub fn build_windows(
    event_day: &[i64],
    person: &[u32],
    hazard_days: i64,
    lag: i64,
    referent: Referent,
    observation: Option<Observation<'_>>,
) -> Result<Windows> {
    let smallest = match referent {
        Referent::Unidirectional { n, step } | Referent::Bidirectional { n, step } => {
            if n == 0 {
                return Err(invalid("a referent needs at least one window"));
            }
            step
        }
        Referent::TimeStratified { unit } => {
            check_unit(unit)?;
            7
        }
        Referent::Stratified { unit, spacing } => {
            check_unit(unit)?;
            if spacing < 1 {
                return Err(invalid(format!(
                    "the spacing of the referents is {spacing}, which is less than a day"
                )));
            }
            spacing
        }
    };
    check_events(event_day, person, hazard_days, lag, smallest, observation)?;
    Ok(build_with(
        event_day,
        person,
        hazard_days,
        lag,
        observation,
        |_, t| shifts(t, referent),
    ))
}

/// Builds the hazard and referent windows of each event with strata the
/// caller defines: the referents of event `i` are the shifts that are
/// nonzero multiples of `spacing` days and keep the event day within
/// `[strata.first[i], strata.last[i]]`. Otherwise as [`build_windows`].
///
/// # Errors
///
/// As [`build_windows`], and when the strata do not have one entry per event
/// or an event day is not within its stratum, or `spacing` is below
/// `hazard_days`.
pub fn build_windows_in_strata(
    event_day: &[i64],
    person: &[u32],
    hazard_days: i64,
    lag: i64,
    strata: EventStrata<'_>,
    spacing: i64,
    observation: Option<Observation<'_>>,
) -> Result<Windows> {
    let n = event_day.len();
    if strata.first.len() != n || strata.last.len() != n {
        return Err(invalid("the strata columns must have one entry per event"));
    }
    if let Some(i) =
        (0..n).find(|&i| !(strata.first[i] <= event_day[i] && event_day[i] <= strata.last[i]))
    {
        return Err(invalid(format!(
            "event {i} on day {} is not within its stratum [{}, {}]",
            event_day[i], strata.first[i], strata.last[i]
        )));
    }
    if spacing < 1 {
        return Err(invalid(format!(
            "the spacing of the referents is {spacing}, which is less than a day"
        )));
    }
    check_events(event_day, person, hazard_days, lag, spacing, observation)?;
    Ok(build_with(
        event_day,
        person,
        hazard_days,
        lag,
        observation,
        |i, t| stratum_shifts(t, strata.first[i], strata.last[i], spacing),
    ))
}

fn check_unit(unit: StratumUnit) -> Result<()> {
    match unit {
        StratumUnit::Fixed { length, .. } if length < 1 => Err(invalid(format!(
            "a fixed stratum of {length} days is shorter than a day"
        ))),
        _ => Ok(()),
    }
}

fn check_events(
    event_day: &[i64],
    person: &[u32],
    hazard_days: i64,
    lag: i64,
    smallest: i64,
    observation: Option<Observation<'_>>,
) -> Result<()> {
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
    if smallest < hazard_days {
        return Err(invalid(format!(
            "referent windows are {smallest} days apart, fewer than the {hazard_days} days of the hazard window, so they would overlap it"
        )));
    }
    Ok(())
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the number of events is checked against u32 first"
)]
fn build_with(
    event_day: &[i64],
    person: &[u32],
    hazard_days: i64,
    lag: i64,
    observation: Option<Observation<'_>>,
    shifts_of: impl Fn(usize, i64) -> Vec<i64> + Sync,
) -> Windows {
    let n = event_day.len();
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
                for s in shifts_of(i, t) {
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
    out
}

/// The exposure in each window: the days of the window covered by the
/// intervals of the window's person, or whether there are any, as
/// [`Measure`] says.
///
/// Intervals that overlap or touch are merged first, so a day is counted once.
///
/// # Errors
///
/// An input error when the columns of `exposure` differ in length or an
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
/// An input error when the two have different lengths.
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
            let (y, m, d) = day::to_civil(days);
            assert_eq!(day::from_civil(y, m, d), days);
        }
        assert_eq!(day::to_civil(18_321), (2020, 2, 29));
        assert_eq!(day::from_civil(1970, 1, 1), 0);
    }

    #[test]
    fn time_stratified_shifts_stay_in_the_month_and_keep_the_weekday() {
        // 2020-02-29 was a Saturday: the other Saturdays of February are the 1st, 8th, 15th, 22nd.
        let t = day::from_civil(2020, 2, 29);
        let s = shifts(
            t,
            Referent::TimeStratified {
                unit: StratumUnit::Month,
            },
        );
        assert_eq!(s, vec![-28, -21, -14, -7]);
        // 2019-01-31: Thursdays of January 2019 are the 3rd, 10th, 17th, 24th.
        let t = day::from_civil(2019, 1, 31);
        let s = shifts(
            t,
            Referent::TimeStratified {
                unit: StratumUnit::Month,
            },
        );
        assert_eq!(s, vec![-28, -21, -14, -7]);
    }
}
