//! Merging overlapping intervals in one sweep.
//!
//! The intervals arrive sorted by key, then start, then end. Within a key an
//! interval begins a new span when its start is later than the latest end of
//! every earlier interval of the key plus the grace, and extends the current
//! span otherwise. [`merge_sorted_spans`] returns, for each span, the row that
//! holds its start and the row that holds its latest end, so the caller reads
//! both bounds, with their own type, from its columns.
//!
//! # Algorithm
//!
//! Let `reach` be the latest end seen in the current span. For row `i`,
//!
//! ```text
//! begins(i) = new_key[i]  or  start[i] > reach + grace
//! ```
//!
//! A row that begins a span closes the previous span, sets `reach = end[i]` and
//! records `i` as the row of the latest end. Otherwise `reach` and the row of
//! the latest end move to `i` only when `end[i] > reach`, so the row of the
//! latest end is the first row that holds that end. With `grace = 0`, intervals
//! that share a day merge, and intervals that touch with no day between
//! them, such as days 1 to 5 and 6 to 7, do not. A grace of `g` allows `g`
//! days between them.
//!
//! The sweep is one pass and takes no sorting. It relies on the sort order,
//! which is not checked beyond the first row beginning a key.
//!
//! # Edge cases
//!
//! - Times are integers in one unit. The sum of `reach` and `grace` saturates at
//!   the `i64` limits, so a very late end plus a grace cannot wrap round and
//!   merge unrelated intervals.
//! - An interval inside an earlier one does not shorten the reach.
//! - Empty input gives no spans.
//! - Columns of different lengths, a first row that does not begin a key, a row
//!   that ends before it starts, and more rows than a `u32` holds are refused.

#![expect(
    clippy::cast_possible_truncation,
    reason = "row numbers are checked against the u32 limit before they are narrowed"
)]

use crate::RegisterError;

/// Refuse columns of different lengths, naming the first that differs from the
/// first column.
fn check_lengths(columns: &[(&str, usize)]) -> Result<(), RegisterError> {
    let Some(&(reference, expected)) = columns.first() else {
        return Ok(());
    };
    match columns.iter().find(|&&(_, len)| len != expected) {
        Some(&(name, len)) => Err(RegisterError::input(format!(
            "columns differ in length: {name} has {len} values against {expected} in {reference}"
        ))),
        None => Ok(()),
    }
}

/// The merged spans of a sorted set of intervals.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MergedSpans {
    /// For each span, its first row, which holds its start.
    pub first_row: Vec<u32>,
    /// For each span, the first row whose end is the latest end of the span.
    pub last_row: Vec<u32>,
}

/// Merge intervals sorted by key, start and end.
///
/// `new_key[i]` says row `i` is the first row of a key, and must be true for
/// row 0. `start` and `end` are in one unit and `end[i] >= start[i]`. A row
/// extends the current span when its start is at most the latest end of the
/// earlier rows of the key plus `grace`.
///
/// # Errors
///
/// [`RegisterError::Input`] when the columns differ in length, the first row
/// does not begin a key, a row has an end before its start, or there are more
/// rows than a `u32` row number holds.
pub fn merge_sorted_spans(
    new_key: &[bool],
    start: &[i64],
    end: &[i64],
    grace: i64,
) -> Result<MergedSpans, RegisterError> {
    let n = start.len();
    check_lengths(&[("start", n), ("end", end.len()), ("new_key", new_key.len())])?;
    if u32::try_from(n).is_err() {
        return Err(RegisterError::input(
            "more rows than a UInt32 row number holds",
        ));
    }
    if n == 0 {
        return Ok(MergedSpans::default());
    }
    if !new_key[0] {
        return Err(RegisterError::input("the first row must begin a key"));
    }
    if let Some(row) = (0..n).find(|&row| end[row] < start[row]) {
        return Err(RegisterError::input(format!(
            "row {row} has an end before its start"
        )));
    }

    let mut merged = MergedSpans::default();
    let mut reach = i64::MIN;
    let mut latest = 0_u32;
    for row in 0..n {
        let i = row as u32;
        let begins = new_key[row] || start[row] > reach.saturating_add(grace);
        if begins {
            if row > 0 {
                merged.last_row.push(latest);
            }
            merged.first_row.push(i);
            reach = end[row];
            latest = i;
        } else if end[row] > reach {
            reach = end[row];
            latest = i;
        }
    }
    merged.last_row.push(latest);
    Ok(merged)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn merge(keys: &[bool], start: &[i64], end: &[i64], grace: i64) -> Vec<(u32, u32)> {
        let m = merge_sorted_spans(keys, start, end, grace).unwrap();
        m.first_row.into_iter().zip(m.last_row).collect()
    }

    #[test]
    fn overlapping_and_touching_intervals_merge() {
        let keys = [true, false, false];
        assert_eq!(merge(&keys, &[1, 4, 9], &[5, 9, 12], 0), [(0, 2)]);
        assert_eq!(
            merge(&keys, &[1, 6, 9], &[5, 7, 12], 0),
            [(0, 0), (1, 1), (2, 2)]
        );
    }

    #[test]
    fn a_gap_within_the_grace_merges_and_one_beyond_it_does_not() {
        let keys = [true, false];
        assert_eq!(merge(&keys, &[0, 7], &[5, 9], 2), [(0, 1)]);
        assert_eq!(merge(&keys, &[0, 8], &[5, 9], 2), [(0, 0), (1, 1)]);
    }

    #[test]
    fn the_latest_end_is_the_first_row_that_holds_it() {
        let keys = [true, false, false];
        assert_eq!(merge(&keys, &[0, 1, 2], &[10, 3, 10], 0), [(0, 0)]);
    }

    #[test]
    fn an_interval_inside_an_earlier_one_does_not_shorten_the_reach() {
        let keys = [true, false, false];
        assert_eq!(merge(&keys, &[0, 1, 9], &[10, 2, 11], 0), [(0, 2)]);
    }

    #[test]
    fn a_new_key_always_begins_a_span() {
        let keys = [true, false, true, false];
        assert_eq!(
            merge(&keys, &[0, 1, 0, 1], &[5, 6, 5, 6], 100),
            [(0, 1), (2, 3)]
        );
    }

    #[test]
    fn a_grace_near_the_range_limit_saturates() {
        let keys = [true, false];
        assert_eq!(
            merge(&keys, &[0, 1], &[i64::MAX - 1, i64::MAX], 10),
            [(0, 1)]
        );
    }

    #[test]
    fn empty_input_gives_no_spans() {
        assert_eq!(
            merge_sorted_spans(&[], &[], &[], 0).unwrap(),
            MergedSpans::default()
        );
    }

    #[test]
    fn refusals() {
        assert!(merge_sorted_spans(&[false], &[0], &[1], 0).is_err());
        assert!(merge_sorted_spans(&[true], &[2], &[1], 0).is_err());
        assert!(merge_sorted_spans(&[true], &[0, 1], &[1], 0).is_err());
    }
}
