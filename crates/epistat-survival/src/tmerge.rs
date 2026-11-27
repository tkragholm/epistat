//! Building start-stop data from a base table and time-dependent covariates
//! and events: the row logic of R's `survival::tmerge`.
//!
//! A [`Tmerge`] holds the intervals `(tstart, tstop]` of each subject and the
//! variables added to them. It starts from the base table's ids and the
//! follow-up spans ([`Tmerge::new`]), and each call of [`Tmerge::tdc`],
//! [`Tmerge::cumtdc`], [`Tmerge::event`] or [`Tmerge::cumevent`] adds or
//! updates one variable, splitting intervals where it must. The base table
//! itself is not touched. [`Tmerge::row`] gives for each output row the row of
//! the base table it came from, so the caller carries every other column of
//! the base table along with one indexed copy.
//!
//! The values are numbers. A caller with a factor, a string or a logical
//! codes it as a number, and an event status with several causes is the code
//! of the cause, 0 being no event. A missing value is `None`.
//!
//! # The rules
//!
//! Each of these is tested against R.
//!
//! **Spans.** The first call gives each subject one or more intervals, rows of
//! the table `spans`, with an id, a start and a stop. The ids of the base
//! table must be unique, every id of `spans` must be in the base table, and
//! every id of the base table must be in `spans`. Each interval needs
//! `tstart < tstop`. Without a start, every interval starts at 0, and every
//! stop must then be above 0. Several intervals of one subject must not
//! overlap. When a subject has several rows in `spans`, the output is in order
//! of first appearance of the id in `spans`, and within a subject by stop;
//! otherwise the output is in the order of `spans`. With
//! [`Tmerge::from_first_events`] the span of a subject is `(0, t]`, where `t`
//! is the time in its row of the event table, in which each subject appears
//! once.
//!
//! **Where an observation falls.** An observation is an id, a time and
//! optionally a value, a row of the table given to an operation. Against the
//! intervals of its subject, its time is one of
//!
//! - early, before the start of the first interval,
//! - late, after the stop of the last interval,
//! - in a gap, between two intervals that do not touch,
//! - within, strictly inside an interval,
//! - on a boundary, equal to the stop of an interval and the start of the next,
//! - leading, equal to the start of an interval that does not touch the one
//!   before it, including the first,
//! - trailing, equal to the stop of an interval that the next does not touch,
//!   including the last.
//!
//! The counts of each are in [`Tmerge::counts`].
//!
//! **Observations that are dropped.** An observation with no time, and one
//! whose id is not in the base table (counted as `missid`), is dropped. With
//! `na_rm`, which is the default, so is an observation with a value and no
//! value. The time of an observation must be finite.
//!
//! **Splitting.** Whichever operation, an observation within an interval cuts
//! that interval at its time, once for each distinct time. Earlier events
//! recorded in the interval are on the last piece only and the earlier pieces
//! have the censoring value 0 for them. Other variables are copied to every
//! piece. Observations that are early, late, in a gap or on a boundary or edge
//! cut nothing.
//!
//! **`tdc`.** A step function. In the interval that starts at time `s` the
//! variable has the value of the last observation of the subject with time at
//! or before `s`, so a change takes effect at the start of an interval and an
//! observation in a gap or before the first interval is carried into the next
//! interval. Intervals before the first observation have the value `init`, or
//! the option `tdcstart` when there is no `init`, or missing when neither is
//! given. Without a value column, the variable is 0 before the first
//! observation and 1 from then on. When several observations of a subject have
//! the same time, the last one in the table wins. An observation after the
//! last interval is never used. Calling `tdc` again with the name of a `tdc` or
//! `cumtdc` variable updates it: an interval with a observation at or before
//! its start gets the new value, and the others keep theirs. Without a value
//! column the update is allowed only on a variable that is missing, 0 or 1.
//!
//! **`cumtdc`.** As `tdc`, but the observations add up. The variable in an
//! interval is the sum of the values of all observations of the subject with
//! time at or before its start, added to the initial value, which is 0 without
//! a value column and `init` or `tdcstart` otherwise. An interval with no
//! observation at or before its start keeps the initial value, which is
//! missing when none is given. Without a value column, each observation is
//! counted as 1. A sum with a missing value is missing, which can happen only
//! without `na_rm`.
//!
//! **`event`.** An event at time `t` is recorded on the interval that stops at
//! `t`, if there is one, and the interval stops there on a boundary or as the
//! trailing edge. An event on a leading edge, early, late or in a gap is
//! discarded. An event within an interval cuts it first. The variable is 0 where
//! there is no event. Without a value column, the value is 1. Several events
//! at one time on one subject: the last one in the table wins.
//!
//! **`cumevent`.** As `event`, but the value recorded is the running total of
//! the values of the subject's observations, in order of time and counted over
//! every observation of the subject that was not dropped, including those
//! that fall outside the follow-up. An observation with value 0 adds nothing
//! and records nothing. Without a value column, each counts 1. A missing
//! value is refused.
//!
//! **`delay`.** For `tdc` and `cumtdc`, the time of an observation later than
//! the first start of its subject is moved later by `delay`, so that a
//! covariate known on day `t` is used from `t + delay`. An observation at or
//! before the first start is not moved.
//!
//! **Names.** Each variable has a name. An operation with the name of an
//! existing variable of the same kind updates it as above, and with the name
//! of a variable of the other kind (an event or a covariate) it is refused. R
//! silently replaces a covariate by an event.
//!
//! # Differences from R
//!
//! - The base table's own columns are not copied, see [`Tmerge::row`].
//! - A variable has numbers only.
//! - `init` cannot be given as a missing value when `tdcstart` is set.
//! - R replaces a covariate by an event of the same name, and this refuses it.
//! - `from_first_events` takes a start of 0 only.
//!
//! # Algorithm
//!
//! Each operation sorts its observations by subject and time (ties in table
//! order), classifies each by two binary searches in the subject's intervals,
//! cuts intervals once for all the observations that fall within, and writes
//! the variable in one pass over the intervals. The cost is
//! `O((n + m) log m)` for `n` intervals and `m` observations, sorting
//! included. All arithmetic is sums in table order, so the result is
//! deterministic and independent of the number of threads, and nothing runs in
//! parallel.
//!
//! # References
//!
//! Therneau TM, Crowson C, Atkinson E. Using time dependent covariates and
//! time dependent coefficients in the Cox model (the `timedep` vignette of the
//! `survival` package). The `tmerge` help page of the `survival` package.

#![expect(
    clippy::float_cmp,
    reason = "times are equal or they are not: a change takes effect at an exact time"
)]

use crate::error::{Error, Result};

/// Settings of a [`Tmerge`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TmergeOptions {
    /// Move the time of a `tdc` or `cumtdc` observation later than the first
    /// start of its subject by this much. Finite, not negative.
    pub delay: f64,
    /// Drop observations with a missing value, so that a covariate carries the
    /// last value that is not missing forward.
    pub na_rm: bool,
    /// The value of a `tdc` or `cumtdc` before its first observation when the
    /// operation has no `init`. `None` is missing.
    pub tdcstart: Option<f64>,
}

impl Default for TmergeOptions {
    /// `tmerge`'s defaults: no delay, `na_rm` and a missing `tdcstart`.
    fn default() -> Self {
        Self {
            delay: 0.0,
            na_rm: true,
            tdcstart: None,
        }
    }
}

/// The follow-up spans of the first call: one row per interval.
#[derive(Debug, Clone, Copy)]
pub struct Spans<'a> {
    /// The subject of each interval.
    pub id: &'a [u32],
    /// The start of each interval, or `None` for 0.
    pub tstart: Option<&'a [f64]>,
    /// The stop of each interval.
    pub tstop: &'a [f64],
}

/// The observations of one operation: rows with an id, a time and optionally
/// a value.
#[derive(Debug, Clone, Copy)]
pub struct Observations<'a> {
    /// The subject of each observation.
    pub id: &'a [u32],
    /// The time of each observation, `None` when missing.
    pub time: &'a [Option<f64>],
    /// The value of each observation, `None` for an operation without
    /// values.
    pub value: Option<&'a [Option<f64>]>,
}

/// What kind of variable an operation made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// From `tdc` or `cumtdc`.
    Covariate,
    /// From `event` or `cumevent`.
    Event,
}

/// A variable of the output.
#[derive(Debug, Clone, PartialEq)]
pub struct Variable {
    /// The name given at the first operation.
    pub name: String,
    /// Covariate or event.
    pub kind: Kind,
    /// One value per output row.
    pub values: Vec<Option<f64>>,
}

/// How the observations of one operation fell against the intervals.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Counts {
    /// Before the first interval of the subject.
    pub early: usize,
    /// After the last interval.
    pub late: usize,
    /// In a gap between intervals.
    pub gap: usize,
    /// Strictly inside an interval.
    pub within: usize,
    /// At the join of two touching intervals.
    pub boundary: usize,
    /// At the start of an interval that does not touch the one before.
    pub leading: usize,
    /// At the stop of an interval that the next does not touch.
    pub trailing: usize,
    /// Observations whose subject and time repeat those of the one before.
    pub tied: usize,
    /// Observations whose id is not in the base table.
    pub missid: usize,
}

/// Start-stop data being built.
#[derive(Debug, Clone, PartialEq)]
pub struct Tmerge {
    id: Vec<u32>,
    tstart: Vec<f64>,
    tstop: Vec<f64>,
    row: Vec<u32>,
    variables: Vec<Variable>,
    counts: Vec<(String, Counts)>,
    options: TmergeOptions,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Op {
    Tdc,
    Cumtdc,
    Event,
    Cumevent,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Place {
    Early,
    Late,
    Gap,
    Within,
    Boundary,
    Leading,
    Trailing,
}

/// An observation that passed the filters.
#[derive(Clone, Copy)]
struct Ob {
    group: usize,
    time: f64,
    value: Option<f64>,
    seq: usize,
}

impl Tmerge {
    /// The first call: set the intervals of each subject.
    ///
    /// `base_id` is the id of each row of the base table and must be unique.
    ///
    /// # Errors
    ///
    /// An input error for each failure of the rules under **Spans** in the
    /// module documentation, for columns of different lengths, and for a time
    /// that is not finite.
    pub fn new(base_id: &[u32], spans: &Spans<'_>, options: TmergeOptions) -> Result<Self> {
        check_options(&options)?;
        let m = spans.id.len();
        if spans.tstop.len() != m || spans.tstart.is_some_and(|s| s.len() != m) {
            return Err(Error::invalid("spans columns differ in length"));
        }
        let base_rows = sorted_unique(base_id).map_err(|id| {
            Error::invalid(format!("the base table has the id {id} more than once"))
        })?;
        let mut row = Vec::with_capacity(m);
        for &id in spans.id {
            let at = base_rows
                .binary_search_by_key(&id, |&(k, _)| k)
                .map_err(|_| {
                    Error::invalid(format!("spans has the id {id}, not in the base table"))
                })?;
            row.push(base_rows[at].1);
        }
        let mut seen = vec![false; base_id.len()];
        for &r in &row {
            seen[r as usize] = true;
        }
        if let Some(r) = seen.iter().position(|&s| !s) {
            return Err(Error::invalid(format!(
                "the base table has the id {}, not in spans",
                base_id[r]
            )));
        }
        let mut tstart = Vec::with_capacity(m);
        for i in 0..m {
            let stop = spans.tstop[i];
            if stop.is_nan() {
                return Err(Error::invalid(format!("missing stop in spans row {i}")));
            }
            let start = if let Some(s) = spans.tstart {
                s[i]
            } else {
                if stop <= 0.0 {
                    return Err(Error::invalid(format!(
                        "found an ending time of {stop}, the default starting time of 0 is invalid"
                    )));
                }
                0.0
            };
            if !start.is_finite() || !stop.is_finite() {
                return Err(Error::invalid(format!(
                    "times in spans must be finite, row {i}"
                )));
            }
            if start >= stop {
                return Err(Error::invalid(format!(
                    "tstart must be below tstop, row {i} has {start} and {stop}"
                )));
            }
            tstart.push(start);
        }
        // First appearance of each id in spans, for the order of the output.
        let first = first_appearance(spans.id);
        let duplicated = {
            let mut ids = spans.id.to_vec();
            ids.sort_unstable();
            ids.windows(2).any(|w| w[0] == w[1])
        };
        let mut order: Vec<usize> = (0..m).collect();
        if duplicated {
            order.sort_by(|&a, &b| {
                first[a]
                    .cmp(&first[b])
                    .then(spans.tstop[a].total_cmp(&spans.tstop[b]))
            });
        }
        let id: Vec<u32> = order.iter().map(|&i| spans.id[i]).collect();
        let t0: Vec<f64> = order.iter().map(|&i| tstart[i]).collect();
        let t1: Vec<f64> = order.iter().map(|&i| spans.tstop[i]).collect();
        let row: Vec<u32> = order.iter().map(|&i| row[i]).collect();
        if duplicated {
            for i in 1..id.len() {
                if id[i] == id[i - 1] && t1[i - 1] > t0[i] {
                    return Err(Error::invalid(
                        "the first call has created overlapping or duplicated time intervals",
                    ));
                }
            }
        }
        Ok(Self {
            id,
            tstart: t0,
            tstop: t1,
            row,
            variables: Vec::new(),
            counts: Vec::new(),
            options,
        })
    }

    /// The first call when an event table sets the span: each subject's
    /// interval is `(0, t]`, with `t` the time in its row of the table.
    ///
    /// # Errors
    ///
    /// As [`Tmerge::new`], when an id appears twice or a time is missing.
    pub fn from_first_events(
        base_id: &[u32],
        id: &[u32],
        time: &[Option<f64>],
        options: TmergeOptions,
    ) -> Result<Self> {
        if id.len() != time.len() {
            return Err(Error::invalid("id and time differ in length"));
        }
        let first = first_appearance(id);
        if let Some(i) = (0..id.len()).find(|&i| first[i] != i) {
            return Err(Error::invalid(format!(
                "the id {} is repeated in the table that sets the span, row {i}",
                id[i]
            )));
        }
        let ids = id.to_vec();
        let mut stops = Vec::with_capacity(id.len());
        for t in time {
            stops.push(t.ok_or_else(|| {
                Error::invalid("missing time value, when that variable defines the span")
            })?);
        }
        Self::new(
            base_id,
            &Spans {
                id: &ids,
                tstart: None,
                tstop: &stops,
            },
            options,
        )
    }

    /// The number of output rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.id.len()
    }

    /// Whether there are no output rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }

    /// The subject of each output row.
    #[must_use]
    pub fn id(&self) -> &[u32] {
        &self.id
    }

    /// The start of each output row, `tstart` of `tmerge`.
    #[must_use]
    pub fn start(&self) -> &[f64] {
        &self.tstart
    }

    /// The stop of each output row, `tstop` of `tmerge`.
    #[must_use]
    pub fn stop(&self) -> &[f64] {
        &self.tstop
    }

    /// The start of each output row, under its name in 0.3.
    #[deprecated(since = "0.4.0", note = "use `start`")]
    #[must_use]
    pub fn tstart(&self) -> &[f64] {
        &self.tstart
    }

    /// The stop of each output row, under its name in 0.3.
    #[deprecated(since = "0.4.0", note = "use `stop`")]
    #[must_use]
    pub fn tstop(&self) -> &[f64] {
        &self.tstop
    }

    /// The row of the base table that each output row comes from.
    #[must_use]
    pub fn row(&self) -> &[u32] {
        &self.row
    }

    /// The value of `column`, one value per row of the base table, at the
    /// base row of each output row.
    ///
    /// # Errors
    ///
    /// An input error when `column` has no value at some base row.
    pub fn gather<T: Copy>(&self, column: &[T]) -> Result<Vec<T>> {
        crate::expanded::gather(&self.row, column, "column")
    }

    /// The variables, in the order they were first made.
    #[must_use]
    pub fn variables(&self) -> &[Variable] {
        &self.variables
    }

    /// The variable of this name.
    #[must_use]
    pub fn variable(&self, name: &str) -> Option<&Variable> {
        self.variables.iter().find(|v| v.name == name)
    }

    /// How the observations of each operation fell, one entry per call, with
    /// the name of the variable.
    #[must_use]
    pub fn counts(&self) -> &[(String, Counts)] {
        &self.counts
    }

    /// A covariate that changes value at the observation times.
    ///
    /// `init` is the value before the first observation, and `None` takes the
    /// option `tdcstart`.
    ///
    /// # Errors
    ///
    /// An input error when the observations are malformed, the name is
    /// empty or belongs to an event, or an update without values meets a
    /// variable that is not missing, 0 or 1.
    pub fn tdc(&mut self, name: &str, obs: &Observations<'_>, init: Option<f64>) -> Result<()> {
        self.apply(Op::Tdc, name, obs, init)
    }

    /// A covariate that is the running total of the observations.
    ///
    /// # Errors
    ///
    /// As [`Tmerge::tdc`].
    pub fn cumtdc(&mut self, name: &str, obs: &Observations<'_>, init: Option<f64>) -> Result<()> {
        self.apply(Op::Cumtdc, name, obs, init)
    }

    /// An event at the observation times.
    ///
    /// # Errors
    ///
    /// An input error when the observations are malformed, or the name
    /// belongs to a covariate.
    pub fn event(&mut self, name: &str, obs: &Observations<'_>) -> Result<()> {
        self.apply(Op::Event, name, obs, None)
    }

    /// An event variable that is the running total of the observations.
    ///
    /// # Errors
    ///
    /// As [`Tmerge::event`], and when a value is missing.
    pub fn cumevent(&mut self, name: &str, obs: &Observations<'_>) -> Result<()> {
        self.apply(Op::Cumevent, name, obs, None)
    }

    fn apply(
        &mut self,
        op: Op,
        name: &str,
        obs: &Observations<'_>,
        init: Option<f64>,
    ) -> Result<()> {
        if name.is_empty() {
            return Err(Error::invalid("a variable needs a name"));
        }
        let is_cov = matches!(op, Op::Tdc | Op::Cumtdc);
        let existing = self.variables.iter().position(|v| v.name == name);
        if let Some(v) = existing {
            let kind = self.variables[v].kind;
            if (kind == Kind::Event) == is_cov {
                return Err(Error::invalid(format!(
                    "{name} is already {} and cannot become {}",
                    if kind == Kind::Event {
                        "an event"
                    } else {
                        "a covariate"
                    },
                    if is_cov { "a covariate" } else { "an event" }
                )));
            }
        }
        let n = obs.id.len();
        if obs.time.len() != n || obs.value.is_some_and(|v| v.len() != n) {
            return Err(Error::invalid(format!(
                "the observations of {name} differ in length"
            )));
        }
        if let Some(i) = obs
            .time
            .iter()
            .position(|t| t.is_some_and(|t| !t.is_finite()))
        {
            return Err(Error::invalid(format!(
                "time of {name} must be finite, row {i}"
            )));
        }
        if let Some(v) = obs.value
            && let Some(i) = v.iter().position(|x| x.is_some_and(|x| !x.is_finite()))
        {
            return Err(Error::invalid(format!(
                "value of {name} must be finite, row {i}"
            )));
        }
        if let Some(x) = init.filter(|x| !x.is_finite()) {
            return Err(Error::invalid(format!(
                "init of {name} must be finite, not {x}"
            )));
        }
        let default = init.or(self.options.tdcstart);

        // Subjects: runs of equal id in the output, sorted by id for lookup.
        let groups = self.groups();
        let mut by_id: Vec<(u32, usize)> = groups
            .iter()
            .enumerate()
            .map(|(g, &(lo, _))| (self.id[lo], g))
            .collect();
        by_id.sort_unstable();

        // Filter, shift and sort the observations.
        let mut missid = 0;
        let mut kept: Vec<Ob> = Vec::new();
        for i in 0..n {
            let Ok(at) = by_id.binary_search_by_key(&obs.id[i], |&(k, _)| k) else {
                missid += 1;
                continue;
            };
            let Some(mut time) = obs.time[i] else {
                continue;
            };
            let value = obs.value.map(|v| v[i]);
            if let Some(v) = value
                && self.options.na_rm
                && v.is_none()
            {
                continue;
            }
            let group = by_id[at].1;
            if is_cov && self.options.delay > 0.0 {
                let first_start = self.tstart[groups[group].0];
                if time > first_start {
                    time += self.options.delay;
                }
            }
            kept.push(Ob {
                group,
                time,
                value: value.flatten(),
                seq: i,
            });
        }
        kept.sort_unstable_by(|a, b| {
            a.group
                .cmp(&b.group)
                .then(a.time.total_cmp(&b.time))
                .then(a.seq.cmp(&b.seq))
        });
        let has_value = obs.value.is_some();
        if op == Op::Cumevent && has_value && kept.iter().any(|o| o.value.is_none()) {
            return Err(Error::invalid(format!(
                "cumevent {name} has a missing value, which has no running total"
            )));
        }

        // Classify, count, and cut.
        let mut counts = Counts {
            missid,
            ..Counts::default()
        };
        for w in kept.windows(2) {
            if w[0].group == w[1].group && w[0].time == w[1].time {
                counts.tied += 1;
            }
        }
        let mut within: Vec<(usize, f64)> = Vec::new();
        for o in &kept {
            let (place, i1, _) = self.place(&groups, o);
            match place {
                Place::Early => counts.early += 1,
                Place::Late => counts.late += 1,
                Place::Gap => counts.gap += 1,
                Place::Within => {
                    counts.within += 1;
                    let row = i1.unwrap_or(0);
                    if within.last() != Some(&(row, o.time)) {
                        within.push((row, o.time));
                    }
                }
                Place::Boundary => counts.boundary += 1,
                Place::Leading => counts.leading += 1,
                Place::Trailing => counts.trailing += 1,
            }
        }
        if !within.is_empty() {
            self.cut(&within);
        }
        let groups = self.groups();

        // Write the variable.
        let rows = self.len();
        let mut values = match existing {
            Some(v) => std::mem::take(&mut self.variables[v].values),
            None => match op {
                Op::Tdc | Op::Cumtdc => vec![if has_value { default } else { Some(0.0) }; rows],
                Op::Event | Op::Cumevent => vec![Some(0.0); rows],
            },
        };
        match op {
            Op::Tdc | Op::Cumtdc => {
                let mut o = 0;
                for (g, &(lo, hi)) in groups.iter().enumerate() {
                    while o < kept.len() && kept[o].group < g {
                        o += 1;
                    }
                    let mut end = o;
                    while end < kept.len() && kept[end].group == g {
                        end += 1;
                    }
                    let mine = &kept[o..end];
                    if op == Op::Tdc {
                        self.write_tdc(
                            &mut values,
                            lo,
                            hi,
                            mine,
                            has_value,
                            name,
                            existing.is_some(),
                        )?;
                    } else {
                        let mut k = 0;
                        let mut csum = Some(0.0_f64);
                        let mut hasone = false;
                        for (r, slot) in values.iter_mut().enumerate().take(hi).skip(lo) {
                            while k < mine.len() && mine[k].time <= self.tstart[r] {
                                let y = if has_value { mine[k].value } else { Some(1.0) };
                                csum = match (csum, y) {
                                    (Some(a), Some(b)) => Some(a + b),
                                    _ => None,
                                };
                                hasone = true;
                                k += 1;
                            }
                            if hasone {
                                *slot = match (*slot, csum) {
                                    (Some(v), Some(c)) => Some(v + c),
                                    _ => csum,
                                };
                            }
                        }
                    }
                    o = end;
                }
            }
            Op::Event | Op::Cumevent => {
                let mut running = 0.0_f64;
                let mut current = usize::MAX;
                for o in &kept {
                    let (place, _, i2) = self.place(&groups, o);
                    let inc = if has_value { o.value } else { Some(1.0) };
                    let recorded = if op == Op::Cumevent {
                        if o.group != current {
                            current = o.group;
                            running = 0.0;
                        }
                        let y = inc.unwrap_or(0.0);
                        running += y;
                        if y == 0.0 { None } else { Some(Some(running)) }
                    } else {
                        Some(inc)
                    };
                    if matches!(place, Place::Boundary | Place::Trailing)
                        && let (Some(v), Some(r)) = (recorded, i2)
                    {
                        values[r] = v;
                    }
                }
            }
        }
        let kind = if is_cov { Kind::Covariate } else { Kind::Event };
        match existing {
            Some(v) => self.variables[v].values = values,
            None => self.variables.push(Variable {
                name: name.to_owned(),
                kind,
                values,
            }),
        }
        self.counts.push((name.to_owned(), counts));
        Ok(())
    }

    fn write_tdc(
        &self,
        values: &mut [Option<f64>],
        lo: usize,
        hi: usize,
        mine: &[Ob],
        has_value: bool,
        name: &str,
        update: bool,
    ) -> Result<()> {
        if update && !has_value {
            let bad = values[lo..hi]
                .iter()
                .any(|v| v.is_some_and(|v| v != 0.0 && v != 1.0));
            if bad {
                return Err(Error::invalid(format!(
                    "tdc update does not match prior variable: {name} is not 0 or 1"
                )));
            }
        }
        let mut k = 0;
        for (r, slot) in values.iter_mut().enumerate().take(hi).skip(lo) {
            while k < mine.len() && mine[k].time <= self.tstart[r] {
                k += 1;
            }
            if k == 0 {
                continue;
            }
            *slot = if has_value {
                mine[k - 1].value
            } else {
                Some(1.0)
            };
        }
        Ok(())
    }

    /// The runs of equal id: `(first row, one past the last)`.
    fn groups(&self) -> Vec<(usize, usize)> {
        let mut groups = Vec::new();
        let mut lo = 0;
        for i in 1..=self.id.len() {
            if i == self.id.len() || self.id[i] != self.id[lo] {
                groups.push((lo, i));
                lo = i;
            }
        }
        groups
    }

    /// Where an observation falls, the row of the last interval that starts
    /// at or before it, and the row of the first interval that stops at or
    /// after it.
    fn place(&self, groups: &[(usize, usize)], o: &Ob) -> (Place, Option<usize>, Option<usize>) {
        let (lo, hi) = groups[o.group];
        let t = o.time;
        let c = self.tstart[lo..hi].partition_point(|&s| s <= t);
        let d = self.tstop[lo..hi].partition_point(|&s| s < t);
        let i1 = (c > 0).then(|| lo + c - 1);
        let i2 = (d < hi - lo).then(|| lo + d);
        let place = match (i1, i2) {
            (None, _) => Place::Early,
            (_, None) => Place::Late,
            (Some(a), Some(b)) => {
                if b > a {
                    Place::Gap
                } else if t == self.tstart[a] || t == self.tstop[b] {
                    if a == b + 1 {
                        Place::Boundary
                    } else if t == self.tstart[a] {
                        Place::Leading
                    } else {
                        Place::Trailing
                    }
                } else {
                    Place::Within
                }
            }
        };
        (place, i1, i2)
    }

    /// Cut rows at the given (row, time) pairs, sorted, without repeats.
    fn cut(&mut self, within: &[(usize, f64)]) {
        let old = self.len();
        let mut source: Vec<usize> = Vec::with_capacity(old + within.len());
        let mut start = Vec::with_capacity(old + within.len());
        let mut stop = Vec::with_capacity(old + within.len());
        let mut earlier = Vec::with_capacity(old + within.len());
        let mut w = 0;
        for r in 0..old {
            let begin = w;
            while w < within.len() && within[w].0 == r {
                w += 1;
            }
            let mut s = self.tstart[r];
            for &(_, t) in &within[begin..w] {
                source.push(r);
                start.push(s);
                stop.push(t);
                earlier.push(true);
                s = t;
            }
            source.push(r);
            start.push(s);
            stop.push(self.tstop[r]);
            earlier.push(false);
        }
        self.id = source.iter().map(|&r| self.id[r]).collect();
        self.row = source.iter().map(|&r| self.row[r]).collect();
        for v in &mut self.variables {
            let old_values = std::mem::take(&mut v.values);
            v.values = source
                .iter()
                .zip(&earlier)
                .map(|(&r, &e)| {
                    if e && v.kind == Kind::Event {
                        Some(0.0)
                    } else {
                        old_values[r]
                    }
                })
                .collect();
        }
        self.tstart = start;
        self.tstop = stop;
    }
}

fn check_options(options: &TmergeOptions) -> Result<()> {
    if !options.delay.is_finite() || options.delay < 0.0 {
        return Err(Error::invalid(
            "delay must be a number that is not negative",
        ));
    }
    if options.tdcstart.is_some_and(|x| !x.is_finite()) {
        return Err(Error::invalid("tdcstart must be finite or missing"));
    }
    Ok(())
}

/// Ids with the row of each, sorted by id; the offending id if one repeats.
fn sorted_unique(ids: &[u32]) -> std::result::Result<Vec<(u32, u32)>, u32> {
    let mut v: Vec<(u32, u32)> = ids
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, u32::try_from(i).unwrap_or(u32::MAX)))
        .collect();
    v.sort_unstable();
    match v.windows(2).find(|w| w[0].0 == w[1].0) {
        Some(w) => Err(w[0].0),
        None => Ok(v),
    }
}

/// For each element, the index of the first element with the same value.
fn first_appearance(ids: &[u32]) -> Vec<usize> {
    let mut v: Vec<(u32, usize)> = ids.iter().copied().zip(0..).collect();
    v.sort_unstable();
    let mut first = vec![0; ids.len()];
    let mut head = 0;
    for k in 0..v.len() {
        if k == 0 || v[k].0 != v[k - 1].0 {
            head = v[k].1;
        }
        first[v[k].1] = head;
    }
    first
}
