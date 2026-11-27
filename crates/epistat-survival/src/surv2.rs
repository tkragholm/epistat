//! Timeline data turned into counting-process intervals: R's `Surv2` and the
//! conversion that `coxph`, `survfit` and `survcheck` run on it
//! (`survival:::surv2counting`, exported as `fromtimeline`).
//!
//! # Timeline data
//!
//! A timeline data set has one row for each observation of a subject: the
//! subject, the time, what was observed at that time (the status) and any
//! covariates. A subject seen `k` times has `k` rows. [`counting`] turns the
//! `k` observations of a subject into `k - 1` intervals, each from one
//! observation to the next, with the status found at the end of the interval
//! and the covariates and state of its beginning.
//!
//! # The rules
//!
//! These are `surv2counting`'s, in the order they are applied. Each is
//! checked against R, and the reasons for the ones that go beyond the help page
//! are in the code of `survival` and its `timeline` vignette.
//!
//! 1. The subject and time are never missing, and a subject has no two
//!    observations at the same time. Both are errors. The rows need not be
//!    sorted. They are put in order of subject and time, and the intervals
//!    come out in the order of the observation each begins at.
//! 2. An interval begins at observation `j` and ends at `j + 1`, so a subject
//!    with one observation gives no interval, and the last observation of a
//!    subject gives only the end of an interval. Its covariates are never used.
//! 3. A missing covariate value in observation `j` is replaced by the most
//!    recent non-missing value of an earlier observation of the same subject,
//!    when `lvcf` is set (last value carried forward). A missing value of the
//!    first observation stays missing, and the last observation does not
//!    supply or receive a value.
//! 4. When the status is a set of states (`R`: a factor, whose first level
//!    means "no event at this time"), a missing status is taken to be that
//!    first level, 0. A subject whose data come from merging two tables often
//!    has missing status. With a binary status a missing value is kept, and
//!    the interval that ends at it is not complete: `survfit` and `coxph`
//!    remove it with their `na.action`. [`Counting::complete_rows`] lists
//!    the rows that remain.
//! 5. With [`Repeated::First`], only the first appearance in a subject of each
//!    state is kept, and every later one, adjacent or not, is turned into 0.
//!    This includes the first observation. With `Events` and `Collapse`
//!    nothing is changed here.
//! 6. The current state of a subject is the last non-zero status up to and
//!    including the observation an interval begins at. If every subject has
//!    a non-zero status at its first observation, the intervals carry that
//!    current state as [`Counting::istate`], the initial state of the
//!    interval, as `istate` of a multi-state fit. If every subject has 0 or a
//!    missing value there, there is no initial state, as in survival with
//!    competing risks. If some have one and some do not it is an error. A
//!    binary status cannot have an initial state, so a non-zero status
//!    at every first observation is an error.
//! 7. The status of an interval is the status of the observation it ends at.
//!    With [`Repeated::Collapse`] and an initial state, an interval whose
//!    status equals its own initial state is given status 0, since the
//!    subject stayed where it was. Without an initial state, `Collapse`
//!    changes nothing, although the help page of `Surv2` says that a repeated
//!    code is not a new event. That is what `surv2counting` does, and what is
//!    done here. [`Repeated::Events`] counts every repeat as an event.
//! 8. If no subject has more than one interval, the response has no start
//!    time, as `Surv(stop, status)`, and [`Counting::start`] is `None`. Every
//!    interval then starts at time 0 in the fit that follows. Otherwise it is
//!    `Surv(start, stop, status)`.
//!
//! The values of `Surv2` that are not accepted: `Surv2` reads a numeric status
//! of 1 and 2 as 0 and 1 (an old convention), and turns any other value into
//! missing with a warning. Both are errors here, so a status is 0 or 1 for
//! [`StatusKind::Binary`] and 0 to the number of states for
//! [`StatusKind::States`].
//!
//! # Missing values
//!
//! A missing status is `None`. A missing covariate is NaN on input, since
//! that is how a column arrives from R or a data frame, and `None` on output.
//! Times and subjects are never missing, and an infinite time is refused.
//!
//! # Algorithm
//!
//! A sort of the row numbers by (subject, time, row), a check of adjacent
//! pairs for repeated times, then one pass over the subjects for the status
//! rules and one pass per covariate. The intervals are put back in the order
//! of the rows they begin at by a second sort. Both sorts have a total order,
//! so the result does not depend on the number of threads.
//!
//! # References
//!
//! The `Surv2` and `fromtimeline` help pages and the `timeline` vignette of
//! the `survival` package, Therneau T. M., Grambsch P. M. (2000) Modeling
//! Survival Data, Springer.

use epistat_core::InputError;
use rayon::slice::ParallelSliceMut;

use crate::check::{check_lengths, check_row_count};
use crate::error::{Error, Result};

/// What the status column of a timeline holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum StatusKind {
    /// 0 for no event at that time and 1 for an event, `Surv2(time, 0/1)`.
    Binary,
    /// 0 for no event at that time and `1..=n` for the `n` states, a factor
    /// whose first level means no event. Missing values are taken as 0.
    States(u32),
}

/// What a repeated status means, `repeated` of `Surv2`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Repeated {
    /// Every repeat of a status is a new event, `repeated = TRUE`.
    Events,
    /// A status that equals the current state is not a new event,
    /// `repeated = FALSE`. It has an effect only with an initial state.
    Collapse,
    /// Only the first appearance of each state of a subject counts,
    /// `repeated = "first"`.
    First,
}

/// Settings of [`counting`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TimelineOptions {
    /// The meaning of the status column.
    pub status: StatusKind,
    /// How a repeated status is treated. `Surv2`'s default is
    /// [`Repeated::Collapse`].
    pub repeated: Repeated,
    /// Carry the last value of a covariate forward over missing values.
    /// `fromtimeline`'s default is `true`.
    pub lvcf: bool,
}

/// The observations of a timeline data set, one entry per row in each column.
#[derive(Debug, Clone, Copy)]
pub struct Timeline<'a> {
    /// The subject of each row.
    pub id: &'a [u32],
    /// The time of the observation, finite.
    pub time: &'a [f64],
    /// What was observed, `None` when missing.
    pub status: &'a [Option<u32>],
    /// Covariate columns, with NaN for a missing value.
    pub covariates: &'a [&'a [f64]],
}

/// The intervals of a timeline, as columns of equal length.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Counting {
    /// The subject of each interval.
    pub id: Vec<u32>,
    /// The start of each interval, `None` when no subject has more than one
    /// interval (rule 8).
    pub start: Option<Vec<f64>>,
    /// The end of each interval.
    pub stop: Vec<f64>,
    /// The status at the end of each interval, 0 for none, and `None` where it
    /// is missing.
    pub status: Vec<Option<u32>>,
    /// The state at the start of each interval, when the subjects have an
    /// initial state (rule 6).
    pub istate: Option<Vec<u32>>,
    /// The covariates of the observation each interval begins at, after the
    /// last value is carried forward if asked. One column per input column.
    pub covariates: Vec<Vec<Option<f64>>>,
    /// The input row each interval begins at. Ascending.
    pub row: Vec<u32>,
    /// The input row each interval ends at.
    pub stop_row: Vec<u32>,
}

impl Counting {
    /// The value of `column`, one value per row of the input, at the input row
    /// of each row here.
    ///
    /// # Errors
    ///
    /// An input error when `column` has no value at some input row.
    pub fn gather<T: Copy>(&self, column: &[T]) -> Result<Vec<T>> {
        crate::expanded::gather(&self.row, column, "column")
    }
}

impl Counting {
    /// The number of intervals.
    #[must_use]
    pub fn len(&self) -> usize {
        self.stop.len()
    }

    /// Whether there are no intervals.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.stop.is_empty()
    }

    /// The intervals with a status and every covariate present, which are the
    /// ones `coxph` and `survfit` keep under their default `na.action`, as
    /// indices into the columns, ascending.
    ///
    /// # Errors
    ///
    /// An input error when `status` or a column of `covariates` has another
    /// length than `stop`, or there are more intervals than `u32::MAX`.
    pub fn complete_rows(&self) -> Result<Vec<u32>> {
        let n = self.len();
        check_lengths(&[("stop", n), ("status", self.status.len())])?;
        for (k, c) in self.covariates.iter().enumerate() {
            if c.len() != n {
                return Err(Error::mismatch(format!(
                    "columns differ in length: covariate {k} has {} values against {n} in stop",
                    c.len()
                )));
            }
        }
        check_row_count(n)?;
        Ok((0..n)
            .filter(|&i| self.status[i].is_some() && self.covariates.iter().all(|c| c[i].is_some()))
            .map(index_u32)
            .collect())
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the row count was checked to fit u32"
)]
fn index_u32(i: usize) -> u32 {
    i as u32
}

/// Turn timeline data into counting-process intervals.
///
/// See the [module documentation](self) for the rules.
///
/// # Errors
///
/// An input error when the columns differ in length, there are more rows
/// than `u32::MAX`, a time is not finite, a status is outside its range, a
/// subject has two observations at one time, the subjects disagree on having
/// an initial state, or a binary status has one.
///
/// # Panics
///
/// Never on checked input. The `expect`s mark invariants.
#[expect(
    clippy::float_cmp,
    reason = "two observations are at the same time only when the times are equal"
)]
pub fn counting(timeline: &Timeline<'_>, opts: &TimelineOptions) -> Result<Counting> {
    let n = timeline.id.len();
    check_lengths(&[
        ("id", n),
        ("time", timeline.time.len()),
        ("status", timeline.status.len()),
    ])?;
    for (k, c) in timeline.covariates.iter().enumerate() {
        if c.len() != n {
            return Err(Error::mismatch(format!(
                "columns differ in length: covariate {k} has {} values against {n} in id",
                c.len()
            )));
        }
    }
    check_row_count(n)?;
    let n_states = match opts.status {
        StatusKind::Binary => 1,
        StatusKind::States(0) => {
            return Err(Error::options(
                "a status of states needs at least one state",
            ));
        }
        StatusKind::States(m) => m,
    };
    if let Some(i) = timeline.time.iter().position(|t| !t.is_finite()) {
        return Err(InputError::NotFinite {
            what: "time",
            row: i,
            column: None,
        }
        .into());
    }
    if let Some(i) = timeline
        .status
        .iter()
        .position(|s| s.is_some_and(|v| v > n_states))
    {
        return Err(Error::out_of_range(format!(
            "the status of row {i} is above {n_states}"
        )));
    }

    let (id, time) = (timeline.id, timeline.time);
    let mut order: Vec<u32> = (0..n).map(index_u32).collect();
    order.par_sort_unstable_by(|&a, &b| {
        let (a, b) = (a as usize, b as usize);
        id[a]
            .cmp(&id[b])
            .then(time[a].total_cmp(&time[b]))
            .then(a.cmp(&b))
    });
    for w in order.windows(2) {
        let (a, b) = (w[0] as usize, w[1] as usize);
        if id[a] == id[b] && time[a] == time[b] {
            return Err(Error::inconsistent(format!(
                "subject {} has two observations at time {}",
                id[a], time[a]
            )));
        }
    }

    // Status in sorted order: missing is 0 for states (rule 4), and only the
    // first appearance of each state stays with `First` (rule 5).
    let mut status: Vec<Option<u32>> = order
        .iter()
        .map(|&r| match (timeline.status[r as usize], opts.status) {
            (None, StatusKind::States(_)) => Some(0),
            (s, _) => s,
        })
        .collect();
    let bounds = subject_bounds(&order, id);
    if opts.repeated == Repeated::First {
        let mut seen = vec![usize::MAX; n_states as usize + 1];
        for (g, &(a, b)) in bounds.iter().enumerate() {
            for s in &mut status[a..b] {
                if let Some(v) = *s
                    && v != 0
                {
                    if seen[v as usize] == g {
                        *s = Some(0);
                    } else {
                        seen[v as usize] = g;
                    }
                }
            }
        }
    }

    // The initial state (rule 6).
    let censored_first = bounds
        .iter()
        .filter(|&&(a, _)| status[a].is_none_or(|v| v == 0))
        .count();
    let has_istate = if censored_first == bounds.len() {
        false
    } else if censored_first > 0 {
        return Err(Error::inconsistent(
            "everyone or no one should have an initial state: \
             some subjects have a status at their first observation and some do not",
        ));
    } else if opts.status == StatusKind::Binary {
        return Err(Error::not_applicable(
            "a binary status cannot have an initial state, \
             but every subject has an event at its first observation",
        ));
    } else {
        true
    };

    // The intervals, in the order of the sorted observations.
    let m = n - bounds.len();
    let mut out = Counting {
        id: Vec::with_capacity(m),
        start: None,
        stop: Vec::with_capacity(m),
        status: Vec::with_capacity(m),
        istate: has_istate.then(|| Vec::with_capacity(m)),
        covariates: Vec::new(),
        row: Vec::with_capacity(m),
        stop_row: Vec::with_capacity(m),
    };
    let mut start = Vec::with_capacity(m);
    for &(a, b) in &bounds {
        let mut current = 0;
        for p in a..b - 1 {
            if let Some(v) = status[p]
                && v != 0
            {
                current = v;
            }
            let mut s = status[p + 1];
            if opts.repeated == Repeated::Collapse && has_istate && s == Some(current) {
                s = Some(0);
            }
            out.id.push(id[order[p] as usize]);
            start.push(time[order[p] as usize]);
            out.stop.push(time[order[p + 1] as usize]);
            out.status.push(s);
            if let Some(istate) = &mut out.istate {
                istate.push(current);
            }
            out.row.push(order[p]);
            out.stop_row.push(order[p + 1]);
        }
    }
    let multiple = bounds.iter().any(|&(a, b)| b - a > 2);

    // Covariates, with the last value carried forward (rule 3).
    for column in timeline.covariates {
        let mut values = Vec::with_capacity(m);
        for &(a, b) in &bounds {
            let mut carry = f64::NAN;
            for p in a..b - 1 {
                let v = column[order[p] as usize];
                if v.is_nan() {
                    values.push((opts.lvcf && !carry.is_nan()).then_some(carry));
                } else {
                    carry = v;
                    values.push(Some(v));
                }
            }
        }
        out.covariates.push(values);
    }

    // Back to the order of the input rows.
    let mut perm: Vec<u32> = (0..m).map(index_u32).collect();
    perm.par_sort_unstable_by_key(|&i| out.row[i as usize]);
    out.id = take(&out.id, &perm);
    out.stop = take(&out.stop, &perm);
    out.status = take(&out.status, &perm);
    out.row = take(&out.row, &perm);
    out.stop_row = take(&out.stop_row, &perm);
    if let Some(istate) = &out.istate {
        out.istate = Some(take(istate, &perm));
    }
    for c in &mut out.covariates {
        *c = take(c, &perm);
    }
    if multiple {
        out.start = Some(take(&start, &perm));
    }
    Ok(out)
}

fn take<T: Copy>(v: &[T], perm: &[u32]) -> Vec<T> {
    perm.iter().map(|&i| v[i as usize]).collect()
}

/// The half-open ranges of the sorted observations that belong to one subject.
fn subject_bounds(order: &[u32], id: &[u32]) -> Vec<(usize, usize)> {
    let mut bounds = Vec::new();
    let mut a = 0;
    for p in 1..=order.len() {
        if p == order.len() || id[order[p] as usize] != id[order[a] as usize] {
            bounds.push((a, p));
            a = p;
        }
    }
    bounds
}
