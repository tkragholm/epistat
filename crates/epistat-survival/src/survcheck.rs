//! Structural checks of follow-up data that has several rows per subject:
//! overlaps, gaps, jumps and teleports, the transitions seen, and the state
//! each row starts in. The checks of R's `survival::survcheck`.
//!
//! [`survcheck`] reads a [`SurvData`] with an id column and reports. It does
//! not refuse anything, and [`SurvCheck::enforce`] turns a report into an
//! error under a [`Policy`], so each estimator can decide what it tolerates.
//!
//! # What the flags mean
//!
//! The rows of one subject are put in time order, by stop, then start. Each
//! row after the first is compared with the row before it. The state a row is
//! in is the status of the nearest earlier row of that subject that ended in
//! an event, so a censored row does not change the state, and the first row of
//! a subject is in its `istate` (0 when there is none). A row is flagged by
//! at most one of four labels.
//!
//! - **Overlap.** The row starts before the previous row stopped, so the
//!   subject is at risk twice over a stretch of time. Counted whatever the
//!   states.
//! - **Gap.** The row starts after the previous row stopped, and its `istate`
//!   agrees with the state the previous rows led to. The subject was not
//!   observed in between, which is legitimate for a subject who is out of the
//!   data for a while.
//! - **Jump.** As a gap, but the `istate` disagrees. The subject changed
//!   state while unobserved.
//! - **Teleport.** The row starts exactly when the previous one stopped, but
//!   its `istate` disagrees with the state reached. The subject changed
//!   state without any event being recorded.
//!
//! Without an `istate` column and with a start column, each row is taken to
//! be in the state the earlier rows led to, so there are no jumps and no
//! teleports. Without a start column the rows are right-censored follow-up and
//! every row starts at time 0, so a second row of a subject is an overlap
//! whenever the row before it stopped after time 0.
//!
//! Times are compared exactly. Times that differ by rounding error are made
//! equal beforehand by the tie rule of [`crate::timefix`], as `survcheck` in
//! R applies `aeqSurv` before it checks. A caller that skips the rule asks for
//! exact comparisons.
//!
//! # The transitions
//!
//! [`SurvCheck::transitions`] counts, for each pair of the state a row starts
//! in and the status it ends with, the rows with an event and the last row of
//! each subject. A subject's last row counts as a transition to 0 when it
//! ends censored, so the table includes censoring, as in R. The state a row
//! starts in is its given `istate` where there is one. With a start column and
//! no `istate` it is the state reached by the earlier rows, and with neither
//! it is 0. Pairs with a count of zero are absent. In `from`, 0 is "not
//! given", which R calls `(s0)`. In `to`, 0 is censoring.
//!
//! # Algorithm
//!
//! A permutation of the row numbers is sorted by the key (id, stop, start,
//! row), with start taken as 0 when there is no start column. The key is
//! unique, so the order does not depend on the sort algorithm or on the number
//! of threads, and the sort runs in parallel. One pass over the sorted rows
//! then carries the previous stop and the current state along each subject's
//! rows, labels the row, resolves its state and counts the transition. The
//! labels and states are written by row, so the lists come out in ascending
//! row order without another sort. The pass is sequential, which costs about
//! as much as reading the columns.
//!
//! R orders the rows of one subject by stop and then start, so a row that
//! lies inside a longer one is compared with the row before it in that order.
//! The same order is used here, since it decides which rows are flagged.
//!
//! # Memory
//!
//! The permutation, the state of each row and the label of each row take 9
//! bytes per row, 90 MB at 10^7 rows. The row lists of the four flags are not
//! capped. Each row has at most one label, so together they are at most 4
//! bytes per row, the size of the data's own id column, and a cap would make
//! the counts and the report disagree.
//!
//! # Ids
//!
//! The ids in a report are the dense codes of [`SurvData::id`], which are
//! the position of each original id in sorted order. The original codes are
//! not kept by [`SurvData`].
//!
//! # Rows of weight 0
//!
//! The check reads all rows, including rows of weight 0, since the structure
//! of a subject's follow-up does not depend on case weights.
//!
//! # References
//!
//! Therneau TM. survival: Survival Analysis. R package, `survcheck()`, the
//! definitions of overlap, gap, jump and teleport and of the transitions
//! table (version 3.8-12). The code here is written from those definitions
//! and checked against the package on the cases in the test suite.

use std::cmp::Ordering;
use std::collections::HashMap;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::surv::SurvData;

/// The rows and subjects that one flag marks.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Flagged {
    /// Row numbers from 0 in the order of the input, ascending. The row is
    /// the one that starts the interval compared with the one before it.
    pub rows: Vec<u32>,
    /// The dense id codes of the subjects with a flagged row, ascending.
    pub ids: Vec<u32>,
    /// The dense id code of the subject of the first row.
    first_id: u32,
}

impl Flagged {
    /// The number of flagged rows.
    #[must_use]
    pub fn count(&self) -> usize {
        self.rows.len()
    }
}

/// The number of rows that start in one state and end in one status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Transition {
    /// The state the row starts in, 0 when it is not given.
    pub from: u32,
    /// The status the row ends with, 0 for censored.
    pub to: u32,
    /// The number of rows.
    pub count: usize,
}

/// What [`survcheck`] found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SurvCheck {
    /// Rows that start before the row before them stopped.
    pub overlap: Flagged,
    /// Rows that start after the row before them stopped, in the expected
    /// state.
    pub gap: Flagged,
    /// Rows that start after the row before them stopped, in another state.
    pub jump: Flagged,
    /// Rows that start when the row before them stopped, in another state.
    pub teleport: Flagged,
    /// The transitions, by `from` and then `to`, with a count above zero.
    pub transitions: Vec<Transition>,
    /// The state each row is in at its start, by row: the state the earlier
    /// rows of the subject led to, or the given `istate` for a subject's
    /// first row. 0 for a first row without one.
    pub istate: Vec<u32>,
    /// The number of distinct ids.
    pub n_ids: usize,
}

impl SurvCheck {
    /// The number of transitions to an event, which leaves out censoring.
    #[must_use]
    pub fn n_transitions(&self) -> usize {
        self.transitions
            .iter()
            .filter(|t| t.to != 0)
            .map(|t| t.count)
            .sum()
    }

    /// Whether any flag marks a row.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        [&self.overlap, &self.gap, &self.jump, &self.teleport]
            .iter()
            .all(|flag| flag.rows.is_empty())
    }

    /// Apply a policy to the report.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] under [`Policy::Strict`] when there is an overlap, a
    /// jump or a teleport. The message names the flag of the lowest row among
    /// them, that row, its subject's dense id code and the number of rows with
    /// that flag.
    pub fn enforce(&self, policy: Policy) -> Result<()> {
        if policy == Policy::Report {
            return Ok(());
        }
        let first = [
            ("overlap", &self.overlap),
            ("jump", &self.jump),
            ("teleport", &self.teleport),
        ]
        .into_iter()
        .filter_map(|(name, flag)| Some((flag.rows.first().copied()?, name, flag)))
        .min_by_key(|&(row, ..)| row);
        let Some((row, name, flag)) = first else {
            return Ok(());
        };
        let id = flag.first_id;
        Err(Error::input(format!(
            "follow-up of one subject is inconsistent: {count} rows with {name}, first at row {row} \
             (id code {id}); multi-state data allows gaps only",
            count = flag.count(),
        )))
    }
}

/// What a caller tolerates in a report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Everything is reported and nothing is an error. For binary status,
    /// which `survival` accepts as it is.
    Report,
    /// Overlaps, jumps and teleports are errors and gaps are allowed. For
    /// multi-state data.
    Strict,
}

impl Policy {
    /// The policy for these data: [`Policy::Strict`] when the status has more
    /// than one event code, and [`Policy::Report`] for a binary status.
    #[must_use]
    pub fn for_data(data: &SurvData<'_>) -> Self {
        match data.followup() {
            Ok(surv) if surv.n_codes() > 1 => Self::Strict,
            _ => Self::Report,
        }
    }
}

/// Check the follow-up structure of each subject.
///
/// See the [module documentation](self) for the flags, the transitions and
/// the algorithm.
///
/// # Errors
///
/// [`Error::Input`] when the data have no id column.
///
/// # Panics
///
/// Only when a [`SurvData`] invariant is broken: a row or id count above
/// `u32::MAX`, or a time that is not finite.
pub fn survcheck(data: &SurvData<'_>) -> Result<SurvCheck> {
    let ids = data.id().ok_or_else(|| {
        Error::input("survcheck needs an id column to know which rows belong together".to_owned())
    })?;
    let surv = data.followup()?;
    let (stop, start, status) = (surv.stop(), surv.start(), surv.status());
    let given = data.istate();
    let n = ids.len();

    let mut order: Vec<u32> = (0..n)
        .map(|row| u32::try_from(row).expect("the builder limits rows to u32::MAX"))
        .collect();
    order.par_sort_unstable_by(|&a, &b| {
        let (a, b) = (a as usize, b as usize);
        ids[a]
            .cmp(&ids[b])
            .then_with(|| compare_times(stop[a], stop[b]))
            .then_with(|| start.map_or(Ordering::Equal, |s| compare_times(s[a], s[b])))
            .then(a.cmp(&b))
    });

    let mut label = vec![Label::None; n];
    let mut resolved = vec![0_u32; n];
    let mut table = Table::new(surv.n_codes());
    let mut previous: Option<usize> = None;
    for &position in &order {
        let row = position as usize;
        let (reached, comparison) = match previous {
            Some(p) if ids[p] == ids[row] => {
                let reached = match status[p] {
                    0 => resolved[p],
                    code => u32::try_from(code).expect("status is at least 0"),
                };
                let begin = start.map_or(0.0, |s| s[row]);
                (reached, compare_times(begin, stop[p]))
            }
            _ => {
                if let Some(p) = previous {
                    table.end_of_subject(status[p], from_state(given, start, &resolved, p));
                }
                (given.map_or(0, |g| g[row]), Ordering::Equal)
            }
        };
        resolved[row] = reached;
        let from = from_state(given, start, &resolved, row);
        let mismatch = from != reached;
        label[row] = match comparison {
            Ordering::Less => Label::Overlap,
            Ordering::Greater if mismatch => Label::Jump,
            Ordering::Greater => Label::Gap,
            Ordering::Equal if mismatch => Label::Teleport,
            Ordering::Equal => Label::None,
        };
        if status[row] > 0 {
            table.add(from, status[row]);
        }
        previous = Some(row);
    }
    if let Some(p) = previous {
        table.end_of_subject(status[p], from_state(given, start, &resolved, p));
    }

    let n_ids = data.n_ids();
    let mut seen = vec![0_u8; n_ids];
    let mut flags: [Flagged; 4] = Default::default();
    for (row, &mark) in label.iter().enumerate() {
        let Some(slot) = mark.slot() else { continue };
        let id = ids[row];
        let flag = &mut flags[slot];
        if flag.rows.is_empty() {
            flag.first_id = id;
        }
        flag.rows
            .push(u32::try_from(row).expect("the builder limits rows to u32::MAX"));
        seen[id as usize] |= 1 << slot;
    }
    for (id, &bits) in seen.iter().enumerate() {
        for (slot, flag) in flags.iter_mut().enumerate() {
            if bits >> slot & 1 == 1 {
                flag.ids
                    .push(u32::try_from(id).expect("the builder limits ids to u32::MAX"));
            }
        }
    }
    let [overlap, gap, jump, teleport] = flags;
    Ok(SurvCheck {
        overlap,
        gap,
        jump,
        teleport,
        transitions: table.into_sorted(),
        istate: resolved,
        n_ids,
    })
}

/// How a row relates to the row before it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Label {
    None,
    Overlap,
    Gap,
    Jump,
    Teleport,
}

impl Label {
    fn slot(self) -> Option<usize> {
        match self {
            Self::None => None,
            Self::Overlap => Some(0),
            Self::Gap => Some(1),
            Self::Jump => Some(2),
            Self::Teleport => Some(3),
        }
    }
}

/// Times are finite when the data was built, so they are totally ordered.
/// Zero and negative zero compare equal, as they do in R.
fn compare_times(a: f64, b: f64) -> Ordering {
    a.partial_cmp(&b)
        .expect("times are finite when a SurvData is built")
}

/// The state a row is taken to start in: the given one, else the state
/// reached when there is a start column, else 0.
fn from_state(given: Option<&[u32]>, start: Option<&[f64]>, resolved: &[u32], row: usize) -> u32 {
    match (given, start) {
        (Some(g), _) => g[row],
        (None, Some(_)) => resolved[row],
        (None, None) => 0,
    }
}

/// Counts of (from, to) pairs. A dense table while it stays small, a map when
/// the number of codes is large.
enum Table {
    Dense { width: usize, counts: Vec<usize> },
    Sparse(HashMap<(u32, u32), usize>),
}

impl Table {
    const DENSE_LIMIT: usize = 1 << 16;

    fn new(n_codes: u32) -> Self {
        let width = n_codes as usize + 1;
        match width.checked_mul(width) {
            Some(cells) if cells <= Self::DENSE_LIMIT => Self::Dense {
                width,
                counts: vec![0; cells],
            },
            _ => Self::Sparse(HashMap::new()),
        }
    }

    fn add(&mut self, from: u32, to: i32) {
        let to = u32::try_from(to).expect("status is at least 0");
        match self {
            Self::Dense { width, counts } => counts[from as usize * *width + to as usize] += 1,
            Self::Sparse(map) => *map.entry((from, to)).or_insert(0) += 1,
        }
    }

    /// The last row of a subject counts as a transition when it is censored.
    /// An event row was counted when it was visited.
    fn end_of_subject(&mut self, status: i32, from: u32) {
        if status == 0 {
            self.add(from, 0);
        }
    }

    fn into_sorted(self) -> Vec<Transition> {
        match self {
            Self::Dense { width, counts } => counts
                .iter()
                .enumerate()
                .filter(|&(_, &count)| count > 0)
                .map(|(cell, &count)| Transition {
                    from: u32::try_from(cell / width).expect("below n_codes + 1"),
                    to: u32::try_from(cell % width).expect("below n_codes + 1"),
                    count,
                })
                .collect(),
            Self::Sparse(map) => {
                let mut all: Vec<Transition> = map
                    .into_iter()
                    .map(|((from, to), count)| Transition { from, to, count })
                    .collect();
                all.sort_unstable_by_key(|t| (t.from, t.to));
                all
            }
        }
    }
}

#[cfg(test)]
mod tests;
