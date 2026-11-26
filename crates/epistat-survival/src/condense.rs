//! Merging adjacent intervals of one subject: R's `survival::survcondense`.
//!
//! [`condense`] undoes the rows that [`crate::split::split`] and
//! [`crate::tmerge`] add. Two rows of one subject are merged when the second
//! starts exactly where the first stops and the caller's covariate key is the
//! same on both. The merged row runs from the start of the first to the stop
//! of the second and has the status of the second. A run of rows that each
//! meet the next is merged into one.
//!
//! # The covariate key
//!
//! The caller gives one `u64` per row. Rows are merged only if their keys are
//! equal, so the key is whatever the caller wants to stay constant: a code of
//! the covariate values, a hash, or the index of the row in a table of
//! distinct covariate rows. Values that `survcondense` takes from the formula
//! and from `weights` belong in the key.
//!
//! # Events
//!
//! `survcondense` compares covariates and times only. It does not look at the
//! status of the first row of a pair, so a row that ended in an event is
//! merged into the next one when the covariates match and its event is lost.
//! [`CondenseOptions::merge_over_events`] chooses. With `true` the rule is
//! R's. With `false` a row that ended in an event is never merged into the next
//! row, so every event survives, and a row without an event is merged into a
//! next row that has one. There is no default, since both are defensible and
//! one of them silently drops events.
//!
//! # Order and ties
//!
//! The rows of one subject are put in order of stop, and rows with the same
//! stop in the order of the input, as R's `order(id, stop)`. The pair
//! compared are neighbours in that order. The rows kept come out in the order
//! of the input, each with its row number, as R keeps the rows of the data
//! frame that survive. Times are compared exactly.
//!
//! # Algorithm
//!
//! A sort of the row numbers by (id, stop, row), then one pass over the sorted
//! rows that marks each row merged into its successor and carries the start of
//! a run forward to its last row.
//!
//! # References
//!
//! The `survcondense` help page of the `survival` package.

use rayon::slice::ParallelSliceMut;

use crate::check::{check_lengths, check_row_count};
use crate::error::{Error, Result};
use crate::surv::SurvData;

/// Settings of [`condense`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CondenseOptions {
    /// Merge a row that ended in an event into the next row, as R does. When
    /// `false`, only rows without an event are merged into their successor.
    pub merge_over_events: bool,
}

/// The rows left after merging, as columns of equal length.
#[derive(Debug, Clone, PartialEq)]
pub struct Condensed {
    /// The start of each row, the earliest start of the rows merged into it.
    pub start: Vec<f64>,
    /// The stop of each row.
    pub stop: Vec<f64>,
    /// The status of each row.
    pub status: Vec<i32>,
    /// The row of the input that each row stands for, the last of the rows
    /// merged. The other columns of that row, such as its covariates, are the
    /// covariates of all the rows merged into it. Ascending.
    pub row: Vec<u32>,
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "the row count was checked to fit u32"
)]
fn order_index(i: usize) -> u32 {
    i as u32
}

/// Merge the adjacent rows of one subject that have the same key.
///
/// `data` needs a start column and an id column. `key` has one value per row.
///
/// # Errors
///
/// An input error when `data` has no start column or no id, or `key` has
/// the wrong length.
#[expect(
    clippy::float_cmp,
    reason = "a row continues the one before only at exactly its stop"
)]
pub fn condense(data: &SurvData<'_>, key: &[u64], options: &CondenseOptions) -> Result<Condensed> {
    let id = data
        .id()
        .ok_or_else(|| Error::invalid("condense needs an id column"))?;
    let data = data.followup()?;
    let n = data.len();
    check_lengths(&[("status", n), ("key", key.len())])?;
    let start = data
        .start()
        .ok_or_else(|| Error::invalid("condense needs a start column"))?;
    let stop = data.stop();
    let status = data.status();

    check_row_count(n)?;
    let mut order: Vec<u32> = (0..n).map(order_index).collect();
    order.par_sort_unstable_by(|&a, &b| {
        let (a, b) = (a as usize, b as usize);
        id[a]
            .cmp(&id[b])
            .then(stop[a].total_cmp(&stop[b]))
            .then(a.cmp(&b))
    });

    let mut new_start = start.to_vec();
    let mut dropped = vec![false; n];
    for w in order.windows(2) {
        let (i, j) = (w[0] as usize, w[1] as usize);
        if id[i] == id[j]
            && key[i] == key[j]
            && start[j] == stop[i]
            && (options.merge_over_events || status[i] == 0)
        {
            dropped[i] = true;
            new_start[j] = new_start[i];
        }
    }
    let kept = n - dropped.iter().filter(|&&d| d).count();
    let mut out = Condensed {
        start: Vec::with_capacity(kept),
        stop: Vec::with_capacity(kept),
        status: Vec::with_capacity(kept),
        row: Vec::with_capacity(kept),
    };
    for i in (0..n).filter(|&i| !dropped[i]) {
        out.start.push(new_start[i]);
        out.stop.push(stop[i]);
        out.status.push(status[i]);
        out.row.push(order_index(i));
    }
    Ok(out)
}
