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
/// `a`.
///
/// # Errors
///
/// [`RegisterError::Input`] when `a` is malformed, the window columns do not
/// have one entry per key of `a`, or a window ends before it starts.
pub fn complement_within(
    window_start: &[i64],
    window_end: &[i64],
    a: &SpanSet,
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

fn window_set(lo: &[i64], hi: &[i64], keys: usize) -> Result<SpanSet, RegisterError> {
    if lo.len() != keys || hi.len() != keys {
        return Err(RegisterError::input(format!(
            "the window needs one entry per key, {keys}; got {} and {}",
            lo.len(),
            hi.len()
        )));
    }
    let mut out = Builder::new(keys);
    for (key, (&s, &e)) in lo.iter().zip(hi).enumerate() {
        if e < s {
            return Err(RegisterError::input(format!(
                "key {key} has a window that ends before it starts"
            )));
        }
        out.push(s, e);
        out.close_key();
    }
    Ok(out.out)
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
        let c = complement_within(&[0, 0], &[0, 6], &a).unwrap();
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
            let c = to_days(&complement_within(&[0], &[DAYS as i64], &a).unwrap(), 0);
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
