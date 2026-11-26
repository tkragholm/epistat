//! Intersection, difference, complement and clipping of interval sets per key.
//!
//! A [`SpanSet`] holds, for each key (a person), a sorted list of disjoint
//! half-open intervals `[start, end)` of integer days, in CSR layout: the
//! intervals of key `k` are `key_offsets[k]..key_offsets[k + 1]`. Follow-up,
//! residence, exposure and enrolment are such sets, and the operations here
//! answer "follow-up within residence" ([`intersect`]), "exposure outside
//! hospital stays" ([`subtract`]), "the part of a window not covered"
//! ([`complement_within`]) and "follow-up within a study period" ([`clip`]).
//!
//! # Convention
//!
//! Every function here takes half-open intervals, so a day belongs to
//! `[start, end)` when `start <= day < end`, the length is `end - start`, and
//! two intervals that share only an endpoint do not overlap. The closed
//! intervals of [`crate::spans::merge_sorted_spans`] and
//! [`crate::person_time::calendar_year_overlaps`] convert with
//! [`closed_to_half_open`] (`end + 1`) and [`half_open_to_closed`], and a
//! [`SpanSet`] is built from closed ends with [`SpanSet::from_closed`].
//!
//! # Conversions
//!
//! | Form | Day `t` is inside when | Into a [`SpanSet`] |
//! | --- | --- | --- |
//! | Residence spell | `from <= t <= to`, open ends `None` | [`crate::residence::Spells::span_set`] |
//! | Closed interval, open ends | `start <= t <= last` | [`SpanSet::from_closed_within`] |
//! | Merged span | `start <= t <= last` | [`SpanSet::from_merged_spans`] |
//! | Exposure period | `start <= t < end`, `end` real | [`crate::exposure::Periods::span_set`] |
//! | Follow-up row | `tstart < time <= tstop`, real | [`SpanSet::switches`], [`SpanSet::follow_up_rows`] |
//!
//! An open end becomes the edge of the key's window, a closed last day `d`
//! the half-open end `d + 1`, and a real end of supply `e` the first whole day
//! not covered, `ceil(e)`. On the real time scale of a follow-up row, day `t`
//! counted from an origin `o` is the interval `(t - o, t - o + 1]`, so the
//! days `[start, end)` are `(start - o, end - o]` and an event on day `t` has
//! the time `t - o + 1`.
//!
//! # Algorithm
//!
//! For one key, `a` and `b` are sorted and disjoint, so both are walked once
//! with a pointer in each.
//!
//! - [`intersect`] emits `[max(a.start, b.start), min(a.end, b.end))` when it
//!   is not empty, then advances the pointer whose interval ends first (both
//!   when they end together). Each emitted piece lies in one `a` interval and
//!   one `b` interval, and every common day is in exactly one piece.
//! - [`subtract`] keeps a cursor `c`, the first day of the current `a` interval
//!   not yet decided. It skips the `b` intervals that end at or before `c`,
//!   emits `[c, b.start)` when `b.start > c`, moves `c` to `b.end`, and emits
//!   what remains of the `a` interval when no `b` interval reaches it.
//! - [`complement_within`] is the difference of the window, one interval per
//!   key, and the set.
//! - [`clip`] intersects each interval with the key's `[lo, hi)`.
//!
//! The work is `O(|a| + |b|)` per key and the output is in key and time order.
//! Pieces that touch, such as `[1, 5)` and `[5, 8)` produced from two input
//! intervals that touched, are not joined: the input structure is kept.
//!
//! # Edge cases
//!
//! - An input set must have every interval non-empty (`start < end`), sorted,
//!   and not overlapping its predecessor (`start >= previous end`). Anything
//!   else is refused with the key, because a result computed on it would not
//!   mean what the caller thinks. [`validate_intervals`] counts missing,
//!   inverted and empty rows in raw columns so a caller can raise, drop or
//!   warn before building a set.
//! - Two sets must have the same number of keys. A key with no interval is
//!   an empty list, not an error.
//! - A window or clip bound with `hi < lo` is refused. `hi == lo` is an empty
//!   window.
//! - Days are `i64`, and `end + 1` in a conversion from closed ends that would
//!   overflow is refused.

use crate::RegisterError;
use crate::spans::MergedSpans;

/// Per-key lists of disjoint half-open intervals `[start, end)` of days.
///
/// `key_offsets` has one more entry than there are keys, starts at 0 and ends
/// at `start.len()`. Within a key the intervals are sorted, non-empty and do
/// not overlap. [`SpanSet::new`] checks this, and every function that takes a
/// set checks it again, so a set built by filling the fields directly is
/// still refused when it is malformed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpanSet {
    /// First day of each interval.
    pub start: Vec<i64>,
    /// One past the last day of each interval.
    pub end: Vec<i64>,
    /// Row range of each key, of length keys plus one.
    pub key_offsets: Vec<usize>,
}

impl SpanSet {
    /// A set from its parts, after checking that it is well formed.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] when the columns differ in length, the offsets
    /// are not a non-decreasing sequence from 0 to the number of intervals, or
    /// an interval is empty, inverted or overlaps the one before it.
    pub fn new(
        start: Vec<i64>,
        end: Vec<i64>,
        key_offsets: Vec<usize>,
    ) -> Result<Self, RegisterError> {
        let set = Self {
            start,
            end,
            key_offsets,
        };
        set.check("span set")?;
        Ok(set)
    }

    /// A set whose intervals are given with inclusive last days, as
    /// [`crate::spans::merge_sorted_spans`] and
    /// [`crate::person_time::calendar_year_overlaps`] use them.
    ///
    /// # Errors
    ///
    /// As [`SpanSet::new`], and when a last day is `i64::MAX`.
    pub fn from_closed(
        start: Vec<i64>,
        last_day: &[i64],
        key_offsets: Vec<usize>,
    ) -> Result<Self, RegisterError> {
        let end = last_day
            .iter()
            .map(|&day| closed_to_half_open(day))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| RegisterError::input("a closed end of i64::MAX has no half-open end"))?;
        Self::new(start, end, key_offsets)
    }

    /// A set from closed intervals whose ends may be open, inside a window
    /// per key.
    ///
    /// Key `k` has the intervals `key_offsets[k]..key_offsets[k + 1]` and the
    /// window `[window_start[k], window_end[k])`. A missing start is the start
    /// of the window, a missing last day the end of the window, a last day `d`
    /// becomes the half-open end `d + 1`, and each interval is clipped to the
    /// window. An interval that lies outside the window is dropped, and a key
    /// with no interval inside it is empty.
    ///
    /// Residence spells with an unknown arrival or no departure are such
    /// intervals. A person with no spell at all is empty here; the residence
    /// reading of such a person, resident on every day, is
    /// [`crate::residence::Spells::span_set`].
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] when the columns differ in length, the offsets
    /// are not a non-decreasing sequence from 0 to the number of intervals, the
    /// window columns do not have one entry per key, a window ends before it
    /// starts, an interval has a last day before its start, or the intervals of
    /// a key overlap or are out of order.
    pub fn from_closed_within(
        start: &[Option<i64>],
        last_day: &[Option<i64>],
        key_offsets: &[usize],
        window_start: &[i64],
        window_end: &[i64],
    ) -> Result<Self, RegisterError> {
        if start.len() != last_day.len() {
            return Err(RegisterError::input(
                "start and last_day must have one entry per interval",
            ));
        }
        check_offsets(key_offsets, start.len(), "closed intervals")?;
        let keys = key_offsets.len() - 1;
        check_window(window_start, window_end, keys)?;
        let mut out = Builder::new(keys);
        for key in 0..keys {
            let (lo, hi) = (window_start[key], window_end[key]);
            for i in key_offsets[key]..key_offsets[key + 1] {
                if let (Some(s), Some(d)) = (start[i], last_day[i])
                    && d < s
                {
                    return Err(RegisterError::input(format!(
                        "key {key} has a closed interval [{s}, {d}] that ends before it starts"
                    )));
                }
                let s = start[i].map_or(lo, |s| s.max(lo));
                let e = last_day[i].map_or(hi, |d| d.saturating_add(1).min(hi));
                out.push(s, e);
            }
            out.close_key();
        }
        out.out.check("closed intervals")?;
        Ok(out.out)
    }

    /// The set of the spans of [`crate::spans::merge_sorted_spans`], whose
    /// columns hold inclusive last days.
    ///
    /// The keys are the runs of `new_key`: key 0 begins at row 0 and key `k`
    /// at the `k`-th row with `new_key` set after it. `start` and `last_day`
    /// are the columns given to the merge, so span `j` is
    /// `[start[first_row[j]], last_day[last_row[j]] + 1)`. Merged spans of one
    /// key never share a day, so the set is well formed.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] when the columns differ in length, the first
    /// row does not begin a key, a span names a row out of range, or a last
    /// day is `i64::MAX`.
    pub fn from_merged_spans(
        merged: &MergedSpans,
        new_key: &[bool],
        start: &[i64],
        last_day: &[i64],
    ) -> Result<Self, RegisterError> {
        let n = new_key.len();
        if start.len() != n || last_day.len() != n {
            return Err(RegisterError::input(
                "new_key, start and last_day must have one entry per row",
            ));
        }
        if merged.first_row.len() != merged.last_row.len() {
            return Err(RegisterError::input(
                "the merged spans have first and last rows of different lengths",
            ));
        }
        if n > 0 && !new_key[0] {
            return Err(RegisterError::input("the first row must begin a key"));
        }
        let mut key_of_row = Vec::with_capacity(n);
        let mut keys = 0_usize;
        for &begins in new_key {
            if begins {
                keys += 1;
            }
            key_of_row.push(keys - 1);
        }
        let mut out = Builder::new(keys);
        let mut key = 0;
        for (&first, &last) in merged.first_row.iter().zip(&merged.last_row) {
            let (first, last) = (first as usize, last as usize);
            if first >= n || last >= n {
                return Err(RegisterError::input(format!(
                    "a merged span names row {} of {n}",
                    first.max(last)
                )));
            }
            if key_of_row[first] < key || key_of_row[last] != key_of_row[first] {
                return Err(RegisterError::input(
                    "the merged spans are not in key order, or a span crosses keys",
                ));
            }
            while key < key_of_row[first] {
                out.close_key();
                key += 1;
            }
            let end = closed_to_half_open(last_day[last]).ok_or_else(|| {
                RegisterError::input("a closed end of i64::MAX has no half-open end")
            })?;
            out.out.start.push(start[first]);
            out.out.end.push(end);
        }
        while key < keys {
            out.close_key();
            key += 1;
        }
        out.out.check("merged spans")?;
        Ok(out.out)
    }

    /// The inclusive last day of each interval.
    #[must_use]
    pub fn last_days(&self) -> Vec<i64> {
        self.end
            .iter()
            .map(|&end| half_open_to_closed(end))
            .collect()
    }

    /// The number of keys.
    #[must_use]
    pub fn n_keys(&self) -> usize {
        self.key_offsets.len().saturating_sub(1)
    }

    /// The number of intervals.
    #[must_use]
    pub fn len(&self) -> usize {
        self.start.len()
    }

    /// Whether there are no intervals.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start.is_empty()
    }

    /// The total number of days in the set.
    #[must_use]
    pub fn total_days(&self) -> i128 {
        self.start
            .iter()
            .zip(&self.end)
            .map(|(&s, &e)| i128::from(e) - i128::from(s))
            .sum()
    }

    fn range(&self, key: usize) -> std::ops::Range<usize> {
        self.key_offsets[key]..self.key_offsets[key + 1]
    }

    fn check(&self, what: &str) -> Result<(), RegisterError> {
        if self.start.len() != self.end.len() {
            return Err(RegisterError::input(format!(
                "{what}: start and end differ in length"
            )));
        }
        if self.key_offsets.first() != Some(&0)
            || self.key_offsets.last() != Some(&self.start.len())
        {
            return Err(RegisterError::input(format!(
                "{what}: key offsets must run from 0 to the number of intervals"
            )));
        }
        if self.key_offsets.windows(2).any(|w| w[0] > w[1]) {
            return Err(RegisterError::input(format!(
                "{what}: key offsets must not decrease"
            )));
        }
        if u32::try_from(self.n_keys()).is_err() {
            return Err(RegisterError::input(format!(
                "{what}: more keys than a u32 holds"
            )));
        }
        for key in 0..self.n_keys() {
            let mut previous_end = i64::MIN;
            for row in self.range(key) {
                let (s, e) = (self.start[row], self.end[row]);
                if s >= e {
                    return Err(RegisterError::input(format!(
                        "{what}: key {key} has an empty or inverted interval [{s}, {e})"
                    )));
                }
                if s < previous_end {
                    return Err(RegisterError::input(format!(
                        "{what}: key {key} has intervals that are unsorted or overlap, at day {s}"
                    )));
                }
                previous_end = e;
            }
        }
        Ok(())
    }
}

/// The half-open end for an inclusive last day: `last_day + 1`, or [`None`]
/// for `i64::MAX`.
#[must_use]
pub fn closed_to_half_open(last_day: i64) -> Option<i64> {
    last_day.checked_add(1)
}

/// The inclusive last day for a half-open end: `end - 1`, saturating at
/// `i64::MIN`.
#[must_use]
pub fn half_open_to_closed(end: i64) -> i64 {
    end.saturating_sub(1)
}

/// Counts of the rows of raw start and end columns by what is wrong with them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct IntervalCounts {
    /// Rows with both ends present and `start < end`.
    pub valid: usize,
    /// Rows with a missing start or end.
    pub missing: usize,
    /// Rows with `end < start`.
    pub inverted: usize,
    /// Rows with `end == start`: nothing in a half-open interval.
    pub empty: usize,
}

/// Counts the rows of raw columns that are valid half-open intervals and the
/// rows that are missing, inverted or empty, so that the caller decides
/// whether to raise, drop or warn.
///
/// # Errors
///
/// [`RegisterError::Input`] when the columns differ in length.
pub fn validate_intervals(
    start: &[Option<i64>],
    end: &[Option<i64>],
) -> Result<IntervalCounts, RegisterError> {
    if start.len() != end.len() {
        return Err(RegisterError::input(
            "start and end must have one entry per row",
        ));
    }
    let mut counts = IntervalCounts::default();
    for (s, e) in start.iter().zip(end) {
        match (s, e) {
            (Some(s), Some(e)) if s < e => counts.valid += 1,
            (Some(s), Some(e)) if s == e => counts.empty += 1,
            (Some(_), Some(_)) => counts.inverted += 1,
            _ => counts.missing += 1,
        }
    }
    Ok(counts)
}

struct Builder {
    out: SpanSet,
}

impl Builder {
    fn new(keys: usize) -> Self {
        let mut key_offsets = Vec::with_capacity(keys + 1);
        key_offsets.push(0);
        Self {
            out: SpanSet {
                start: Vec::new(),
                end: Vec::new(),
                key_offsets,
            },
        }
    }

    fn push(&mut self, start: i64, end: i64) {
        if start < end {
            self.out.start.push(start);
            self.out.end.push(end);
        }
    }

    fn close_key(&mut self) {
        self.out.key_offsets.push(self.out.start.len());
    }
}

fn same_keys(a: &SpanSet, b: &SpanSet) -> Result<(), RegisterError> {
    a.check("first span set")?;
    b.check("second span set")?;
    if a.n_keys() != b.n_keys() {
        return Err(RegisterError::input(format!(
            "the span sets have {} and {} keys",
            a.n_keys(),
            b.n_keys()
        )));
    }
    Ok(())
}

/// The days that are in both sets, key by key.
///
/// # Errors
///
/// [`RegisterError::Input`] when either set is malformed or the key counts
/// differ.
pub fn intersect(a: &SpanSet, b: &SpanSet) -> Result<SpanSet, RegisterError> {
    same_keys(a, b)?;
    let mut out = Builder::new(a.n_keys());
    for key in 0..a.n_keys() {
        let (ra, rb) = (a.range(key), b.range(key));
        let (mut ia, mut ib) = (ra.start, rb.start);
        while ia < ra.end && ib < rb.end {
            out.push(a.start[ia].max(b.start[ib]), a.end[ia].min(b.end[ib]));
            match a.end[ia].cmp(&b.end[ib]) {
                std::cmp::Ordering::Less => ia += 1,
                std::cmp::Ordering::Greater => ib += 1,
                std::cmp::Ordering::Equal => {
                    ia += 1;
                    ib += 1;
                }
            }
        }
        out.close_key();
    }
    Ok(out.out)
}

/// The days that are in `a` and not in `b`, key by key.
///
/// # Errors
///
/// [`RegisterError::Input`] when either set is malformed or the key counts
/// differ.
pub fn subtract(a: &SpanSet, b: &SpanSet) -> Result<SpanSet, RegisterError> {
    same_keys(a, b)?;
    let mut out = Builder::new(a.n_keys());
    for key in 0..a.n_keys() {
        let sub = b.range(key);
        let mut first_b = sub.start;
        for ia in a.range(key) {
            let mut cursor = a.start[ia];
            let end = a.end[ia];
            // `b` intervals that end before this `a` interval starts cannot
            // matter to it or to any later one.
            while first_b < sub.end && b.end[first_b] <= cursor {
                first_b += 1;
            }
            let mut jb = first_b;
            while jb < sub.end && b.start[jb] < end {
                if b.start[jb] > cursor {
                    out.push(cursor, b.start[jb]);
                }
                cursor = cursor.max(b.end[jb]);
                jb += 1;
            }
            out.push(cursor, end);
        }
        out.close_key();
    }
    Ok(out.out)
}

/// The days of each key's window `[window_start, window_end)` that are not in
/// `a`. The set comes first and the window after it, as in [`clip`].
///
/// # Errors
///
/// [`RegisterError::Input`] when `a` is malformed, the window columns do not
/// have one entry per key of `a`, or a window ends before it starts.
pub fn complement_within(
    a: &SpanSet,
    window_start: &[i64],
    window_end: &[i64],
) -> Result<SpanSet, RegisterError> {
    let window = window_set(window_start, window_end, a.n_keys())?;
    subtract(&window, a)
}

/// The part of each interval of `a` inside the key's `[lo, hi)`.
///
/// # Errors
///
/// [`RegisterError::Input`] when `a` is malformed, `lo` and `hi` do not have
/// one entry per key of `a`, or `hi < lo`.
pub fn clip(a: &SpanSet, lo: &[i64], hi: &[i64]) -> Result<SpanSet, RegisterError> {
    let window = window_set(lo, hi, a.n_keys())?;
    intersect(a, &window)
}

fn check_window(lo: &[i64], hi: &[i64], keys: usize) -> Result<(), RegisterError> {
    if lo.len() != keys || hi.len() != keys {
        return Err(RegisterError::input(format!(
            "the window needs one entry per key, {keys}; got {} and {}",
            lo.len(),
            hi.len()
        )));
    }
    if let Some(key) = (0..keys).find(|&k| hi[k] < lo[k]) {
        return Err(RegisterError::input(format!(
            "key {key} has a window that ends before it starts"
        )));
    }
    Ok(())
}

fn check_offsets(offsets: &[usize], rows: usize, what: &str) -> Result<(), RegisterError> {
    if offsets.first() != Some(&0)
        || offsets.last() != Some(&rows)
        || offsets.windows(2).any(|w| w[0] > w[1])
    {
        return Err(RegisterError::input(format!(
            "{what}: key offsets must run from 0 to the number of intervals and not decrease"
        )));
    }
    if u32::try_from(offsets.len() - 1).is_err() {
        return Err(RegisterError::input(format!(
            "{what}: more keys than a u32 holds"
        )));
    }
    Ok(())
}

fn window_set(lo: &[i64], hi: &[i64], keys: usize) -> Result<SpanSet, RegisterError> {
    check_window(lo, hi, keys)?;
    let mut out = Builder::new(keys);
    for (&s, &e) in lo.iter().zip(hi) {
        out.push(s, e);
        out.close_key();
    }
    Ok(out.out)
}

/// Switches of a covariate that is 1 inside a set and 0 outside it, in the
/// form of `tmerge`'s time-dependent covariate: `(key, time, value)` rows in
/// key and time order.
///
/// The columns are those of `Observations` in `epistat_survival::tmerge`, so
/// `Observations { id: &s.key, time: &s.time, value: Some(&s.value) }` passes
/// them to `Tmerge::tdc`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Switches {
    /// The key of each switch.
    pub key: Vec<u32>,
    /// The time of each switch on the real scale, days since the key's origin.
    pub time: Vec<Option<f64>>,
    /// 1 where the covariate switches on and 0 where it switches off.
    pub value: Vec<Option<f64>>,
}

/// Follow-up rows that tile a window, each inside or outside a set, in the
/// counting-process form `(tstart, tstop]` of `tmerge` and `SurvData`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FollowUpRows {
    /// The key of each row.
    pub key: Vec<u32>,
    /// The start of each row, days since the key's origin, excluded.
    pub tstart: Vec<f64>,
    /// The stop of each row, days since the key's origin, included.
    pub tstop: Vec<f64>,
    /// Whether the days of the row are in the set.
    pub inside: Vec<bool>,
}

impl SpanSet {
    /// The switches on and off of the indicator of the set, with times in days
    /// since `origin[k]` for key `k`.
    ///
    /// Day `t` is the interval `(t - origin, t - origin + 1]` of the real time
    /// scale, so the days `[start, end)` of an interval are
    /// `(start - origin, end - origin]`: the covariate switches on at
    /// `start - origin` and off at `end - origin`, and an event on day `t` has
    /// the time `t - origin + 1`. Intervals of a key that touch, `end` of one
    /// equal to `start` of the next, give no switch between them.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] when the set is malformed or `origin` does not
    /// have one entry per key.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "day differences of register data are far below 2^53, and keys are checked against u32"
    )]
    pub fn switches(&self, origin: &[i64]) -> Result<Switches, RegisterError> {
        self.check("span set")?;
        if origin.len() != self.n_keys() {
            return Err(RegisterError::input(format!(
                "origin needs one entry per key, {}; got {}",
                self.n_keys(),
                origin.len()
            )));
        }
        let mut out = Switches::default();
        for key in 0..self.n_keys() {
            let range = self.range(key);
            let o = origin[key];
            let mut row = range.start;
            while row < range.end {
                let s = self.start[row];
                let mut e = self.end[row];
                row += 1;
                while row < range.end && self.start[row] == e {
                    e = self.end[row];
                    row += 1;
                }
                for (day, value) in [(s, 1.0), (e, 0.0)] {
                    out.key.push(key as u32);
                    out.time.push(Some((i128::from(day) - i128::from(o)) as f64));
                    out.value.push(Some(value));
                }
            }
        }
        Ok(out)
    }

    /// Rows that tile each key's window `[window_start, window_end)` of days,
    /// alternately inside and outside the set, as `(tstart, tstop]` in days
    /// since `origin`.
    ///
    /// The days `[a, b)` of a row are `(a - origin, b - origin]` on the real
    /// scale, as in [`SpanSet::switches`]. Parts of the set outside the window
    /// are not used, touching intervals give one row, and an empty window
    /// gives no row. A key whose set is empty in its window has one row
    /// outside.
    ///
    /// # Errors
    ///
    /// [`RegisterError::Input`] when the set is malformed, a window or origin
    /// column does not have one entry per key, or a window ends before it
    /// starts.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        reason = "day differences of register data are far below 2^53, and keys are checked against u32"
    )]
    pub fn follow_up_rows(
        &self,
        window_start: &[i64],
        window_end: &[i64],
        origin: &[i64],
    ) -> Result<FollowUpRows, RegisterError> {
        let inside = clip(self, window_start, window_end)?;
        if origin.len() != self.n_keys() {
            return Err(RegisterError::input(format!(
                "origin needs one entry per key, {}; got {}",
                self.n_keys(),
                origin.len()
            )));
        }
        let mut out = FollowUpRows::default();
        let time = |day: i64, o: i64| (i128::from(day) - i128::from(o)) as f64;
        for key in 0..self.n_keys() {
            let (lo, hi, o) = (window_start[key], window_end[key], origin[key]);
            let mut push = |a: i64, b: i64, flag: bool| {
                if a < b {
                    out.key.push(key as u32);
                    out.tstart.push(time(a, o));
                    out.tstop.push(time(b, o));
                    out.inside.push(flag);
                }
            };
            let mut cursor = lo;
            let range = inside.range(key);
            let mut row = range.start;
            while row < range.end {
                let s = inside.start[row];
                let mut e = inside.end[row];
                row += 1;
                while row < range.end && inside.start[row] == e {
                    e = inside.end[row];
                    row += 1;
                }
                push(cursor, s, false);
                push(s, e, true);
                cursor = e;
            }
            push(cursor, hi, false);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    #![expect(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        reason = "small whole numbers in test data"
    )]

    use super::*;

    fn set(keys: &[&[(i64, i64)]]) -> SpanSet {
        let mut s = SpanSet {
            key_offsets: vec![0],
            ..SpanSet::default()
        };
        for key in keys {
            for &(a, b) in *key {
                s.start.push(a);
                s.end.push(b);
            }
            s.key_offsets.push(s.start.len());
        }
        s
    }

    fn pieces(s: &SpanSet, key: usize) -> Vec<(i64, i64)> {
        s.range(key).map(|r| (s.start[r], s.end[r])).collect()
    }

    #[test]
    fn hand_cases() {
        let a = set(&[&[(0, 10), (20, 30)]]);
        let b = set(&[&[(5, 8), (9, 22), (25, 26)]]);
        assert_eq!(
            pieces(&intersect(&a, &b).unwrap(), 0),
            [(5, 8), (9, 10), (20, 22), (25, 26)]
        );
        assert_eq!(
            pieces(&subtract(&a, &b).unwrap(), 0),
            [(0, 5), (8, 9), (22, 25), (26, 30)]
        );
    }

    #[test]
    fn touching_intervals_do_not_intersect() {
        let a = set(&[&[(0, 5)]]);
        let b = set(&[&[(5, 9)]]);
        assert!(intersect(&a, &b).unwrap().is_empty());
        assert_eq!(pieces(&subtract(&a, &b).unwrap(), 0), [(0, 5)]);
    }

    #[test]
    fn empty_keys_and_windows() {
        let a = set(&[&[], &[(1, 4)]]);
        let b = set(&[&[(0, 9)], &[]]);
        let i = intersect(&a, &b).unwrap();
        assert!(i.is_empty());
        assert_eq!(i.n_keys(), 2);
        assert_eq!(pieces(&subtract(&a, &b).unwrap(), 1), [(1, 4)]);
        let c = complement_within(&a, &[0, 0], &[0, 6]).unwrap();
        assert_eq!(pieces(&c, 0), []);
        assert_eq!(pieces(&c, 1), [(0, 1), (4, 6)]);
    }

    #[test]
    fn clip_cuts_both_sides() {
        let a = set(&[&[(0, 10), (12, 20)]]);
        assert_eq!(
            pieces(&clip(&a, &[5], &[15]).unwrap(), 0),
            [(5, 10), (12, 15)]
        );
    }

    #[test]
    fn refusals() {
        let ok = set(&[&[(0, 5)]]);
        assert!(SpanSet::new(vec![5], vec![5], vec![0, 1]).is_err());
        assert!(SpanSet::new(vec![0, 4], vec![5, 9], vec![0, 2]).is_err());
        assert!(SpanSet::new(vec![4, 0], vec![5, 3], vec![0, 2]).is_err());
        assert!(SpanSet::new(vec![0], vec![5], vec![0, 2]).is_err());
        assert!(intersect(&ok, &set(&[&[], &[]])).is_err());
        assert!(clip(&ok, &[5], &[4]).is_err());
        assert!(clip(&ok, &[], &[]).is_err());
        assert!(SpanSet::from_closed(vec![0], &[i64::MAX], vec![0, 1]).is_err());
    }

    #[test]
    fn closed_ends_convert() {
        let s = SpanSet::from_closed(vec![1, 6], &[5, 7], vec![0, 2]).unwrap();
        assert_eq!(s.end, [6, 8]);
        assert_eq!(s.last_days(), [5, 7]);
        assert_eq!(s.total_days(), 7);
    }

    #[test]
    fn closed_intervals_with_open_ends_fill_the_window() {
        let s = SpanSet::from_closed_within(
            &[None, Some(20), Some(5), Some(40)],
            &[Some(9), None, Some(5), Some(60)],
            &[0, 2, 3, 4],
            &[0, 0, 0],
            &[30, 30, 30],
        )
        .unwrap();
        assert_eq!(pieces(&s, 0), [(0, 10), (20, 30)]);
        // A one-day spell is one day long.
        assert_eq!(pieces(&s, 1), [(5, 6)]);
        // A spell after the window is dropped, and the key is empty.
        assert_eq!(pieces(&s, 2), []);
        let inverted = SpanSet::from_closed_within(&[Some(5)], &[Some(4)], &[0, 1], &[0], &[9]);
        assert!(inverted.is_err());
        let overlap = SpanSet::from_closed_within(
            &[Some(0), Some(5)],
            &[Some(5), Some(9)],
            &[0, 2],
            &[0],
            &[20],
        );
        assert!(overlap.is_err());
        assert!(SpanSet::from_closed_within(&[None], &[None], &[0, 1], &[3], &[2]).is_err());
    }

    #[test]
    fn merged_spans_convert_key_by_key() {
        use crate::spans::merge_sorted_spans;
        let new_key = [true, false, false, true, true, false];
        let start = [1, 4, 9, 0, 3, 7];
        let last = [5, 6, 12, 2, 4, 8];
        let merged = merge_sorted_spans(&new_key, &start, &last, 0).unwrap();
        let s = SpanSet::from_merged_spans(&merged, &new_key, &start, &last).unwrap();
        assert_eq!(s.n_keys(), 3);
        assert_eq!(pieces(&s, 0), [(1, 7), (9, 13)]);
        assert_eq!(pieces(&s, 1), [(0, 3)]);
        assert_eq!(pieces(&s, 2), [(3, 5), (7, 9)]);
        // With a grace the spans of a key still never share a day.
        let graced = merge_sorted_spans(&new_key, &start, &last, 3).unwrap();
        let s = SpanSet::from_merged_spans(&graced, &new_key, &start, &last).unwrap();
        assert_eq!(pieces(&s, 0), [(1, 13)]);
        assert_eq!(pieces(&s, 2), [(3, 9)]);
        let empty = merge_sorted_spans(&[], &[], &[], 0).unwrap();
        assert!(SpanSet::from_merged_spans(&empty, &[], &[], &[]).unwrap().is_empty());
        assert!(SpanSet::from_merged_spans(&merged, &new_key[..3], &start, &last).is_err());
    }

    #[test]
    fn switches_join_touching_intervals_and_count_from_the_origin() {
        let a = set(&[&[(10, 15), (15, 20), (30, 31)], &[]]);
        let sw = a.switches(&[10, 0]).unwrap();
        assert_eq!(sw.key, [0, 0, 0, 0]);
        assert_eq!(
            sw.time,
            [Some(0.0), Some(10.0), Some(20.0), Some(21.0)]
        );
        assert_eq!(sw.value, [Some(1.0), Some(0.0), Some(1.0), Some(0.0)]);
        assert!(a.switches(&[0]).is_err());
    }

    #[test]
    fn follow_up_rows_tile_the_window() {
        let a = set(&[&[(5, 10), (10, 12), (20, 40)], &[], &[(0, 3)]]);
        let rows = a
            .follow_up_rows(&[0, 0, 5], &[30, 7, 5], &[0, 0, 5])
            .unwrap();
        let got: Vec<(u32, f64, f64, bool)> = (0..rows.key.len())
            .map(|i| (rows.key[i], rows.tstart[i], rows.tstop[i], rows.inside[i]))
            .collect();
        assert_eq!(
            got,
            [
                (0, 0.0, 5.0, false),
                (0, 5.0, 12.0, true),
                (0, 12.0, 20.0, false),
                (0, 20.0, 30.0, true),
                (1, 0.0, 7.0, false),
            ]
        );
        assert!(a.follow_up_rows(&[0, 0], &[1, 1], &[0, 0]).is_err());
    }

    #[test]
    fn counts_of_raw_rows() {
        let c = validate_intervals(
            &[Some(0), Some(5), None, Some(3), Some(1)],
            &[Some(4), Some(5), Some(2), Some(1), None],
        )
        .unwrap();
        assert_eq!(
            c,
            IntervalCounts {
                valid: 1,
                empty: 1,
                missing: 2,
                inverted: 1
            }
        );
    }

    /// Day-by-day arrays decide every operation on random small sets.
    #[test]
    fn agrees_with_boolean_arrays_on_random_sets() {
        const DAYS: usize = 31;
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let random_set = |next: &mut dyn FnMut() -> u64| {
            let mut days = [false; DAYS];
            let mut pos = 0;
            let mut list = Vec::new();
            while pos < DAYS {
                pos += (next() % 4) as usize;
                let len = 1 + (next() % 5) as usize;
                if pos >= DAYS {
                    break;
                }
                let end = (pos + len).min(DAYS);
                list.push((pos as i64, end as i64));
                for d in &mut days[pos..end] {
                    *d = true;
                }
                pos = end + (next() % 2) as usize;
            }
            (list, days)
        };
        let to_days = |s: &SpanSet, key: usize| {
            let mut d = [false; DAYS];
            for (a, b) in pieces(s, key) {
                for x in a..b {
                    assert!(!d[x as usize], "overlap in output");
                    d[x as usize] = true;
                }
            }
            d
        };
        for _ in 0..4000 {
            let (la, da) = random_set(&mut next);
            let (lb, db) = random_set(&mut next);
            let (a, b) = (set(&[&la]), set(&[&lb]));
            let i = to_days(&intersect(&a, &b).unwrap(), 0);
            let s = to_days(&subtract(&a, &b).unwrap(), 0);
            let c = to_days(&complement_within(&a, &[0], &[DAYS as i64]).unwrap(), 0);
            let lo = (next() % 20) as i64;
            let hi = lo + (next() % 12) as i64;
            let k = to_days(&clip(&a, &[lo], &[hi]).unwrap(), 0);
            for x in 0..DAYS {
                assert_eq!(i[x], da[x] && db[x]);
                assert_eq!(s[x], da[x] && !db[x]);
                assert_eq!(c[x], !da[x]);
                assert_eq!(k[x], da[x] && (lo..hi).contains(&(x as i64)));
            }
        }
    }
}
