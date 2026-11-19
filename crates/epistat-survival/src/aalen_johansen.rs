//! The Aalen-Johansen estimator of cumulative incidence under competing risks,
//! with weights, groups and delayed entry, and a clustered bootstrap of the
//! incidence in two arms over many cells.
//!
//! # Algorithm
//!
//! Within a group, let `t_1 < t_2 < ...` be the distinct exit times. At `t` let
//! `Y(t)` be the weight at risk just before `t`, `d_k(t)` the weight of events
//! of cause `k` at `t` and `d(t) = sum_k d_k(t)`. The event-free probability
//! `S` and the cumulative incidence `F_k` of cause `k` advance by
//!
//! ```text
//! F_k(t) = F_k(t-) + S(t-) d_k(t) / Y(t)
//! S(t)   = S(t-) (1 - d(t) / Y(t))
//! ```
//!
//! starting from `S = 1` and `F_k = 0` \[1\]\[2\]. Each cause increment is weighted by
//! the event-free probability just before the time, not at it, which is what
//! separates the estimator from one minus the Kaplan-Meier estimate when
//! competing causes are present. A time with nobody at risk moves nothing.
//! Times that carry only censoring appear in the output and leave the
//! probabilities unchanged.
//!
//! Without delayed entry the weight at risk is the weight of everyone who has
//! not yet left, a reverse cumulative sum over the distinct times. With entry
//! times a row is at risk on `(entry, time]`, so at `t` the risk set is
//! everyone who entered strictly before `t` less everyone who left strictly
//! before it. A running balance over the merged entry and exit times gives both
//! terms: the changes at each distinct time are summed, and the balance is read
//! before they are added, which makes the entry inequality strict and the exit
//! inequality inclusive.
//!
//! Rows are ordered by group and time with a stable sort, so rows tied on both
//! are summed in input order and the result does not depend on the number of
//! threads. Times are compared for exact equality.
//!
//! # The bootstrap
//!
//! [`bootstrap_arm_incidence`] repeats the estimate for the exposed and the
//! comparator arm of many cells, each a subset of the rows, under random
//! cluster weights. A replicate draws one `Exp(1)` variate per cluster, scales
//! the variates to sum to the number of clusters, which is the Bayesian
//! bootstrap with a Dirichlet(1, ..., 1) weight vector scaled to mean 1 \[3\],
//! and multiplies each row's weight by the weight of its cluster. The incidence
//! of one cause is read at the requested ages. Two properties make this cheaper
//! than repeating the point estimate:
//!
//! * A replicate changes only the weights, never which rows exist or when their
//!   events happen, so the sort, the distinct times and the block boundaries of
//!   a cell are computed once and reused by every replicate.
//! * Cells and replicates are independent and run on all cores. The weights of
//!   replicate `r` of a cell come from the seed `replicate_seed(cell_seed, r)`,
//!   so the output does not depend on the number of threads.
//!
//! Seeds are supplied per cell, so the caller decides where a seed comes from.
//! The output holds the incidence of each arm, not their difference, so a
//! caller can form a difference, a ratio or per-arm bands from the same draws.
//!
//! # Edge cases
//!
//! A cell with an empty arm has no estimate, is flagged not estimable, and its
//! block of the output is zero. An age before the first event time takes the
//! incidence so far, which is zero. An age after the last takes the final
//! incidence. Everyone is at risk from a common origin in the bootstrap, which
//! takes no entry times.
//!
//! # References
//!
//! 1. Aalen OO, Johansen S. An empirical transition matrix for non-homogeneous
//!    Markov chains based on censored observations. Scand J Stat
//!    1978;5(3):141-50.
//! 2. Andersen PK, Borgan O, Gill RD, Keiding N. Statistical models based on
//!    counting processes. New York: Springer; 1993.
//! 3. Rubin DB. The Bayesian bootstrap. Ann Stat 1981;9(1):130-4.
//!    doi:10.1214/aos/1176345338

#![expect(
    clippy::cast_possible_truncation,
    reason = "the entry points check that the row count fits in u32"
)]

use std::collections::HashMap;

use epistat_rng::{dirichlet_weights, replicate_seed};
use rayon::prelude::*;

use crate::check::{check_lengths, check_row_count};
use crate::error::{Error, Result};

const CENSORED: i32 = 0;

/// One distinct time of the estimator, shared by the point estimate and the
/// bootstrap, so a curve and the replicates around it are one computation.
///
/// `survival` enters as `S(t-)` and leaves as `S(t)`, and each cause's incidence
/// gains `S(t-) d_k / Y`. A time with nobody at risk moves nothing; it can only
/// arise with zero weights, where every `d_k` is zero too.
#[inline]
pub(crate) fn advance(
    survival: &mut f64,
    at_risk: f64,
    d_any: f64,
    d_causes: &[f64],
    incidence: &mut [f64],
) {
    if at_risk > 0.0 {
        let before = *survival;
        for (cif, &d) in incidence.iter_mut().zip(d_causes) {
            *cif += before * d / at_risk;
        }
        *survival = before * (1.0 - d_any / at_risk);
    }
}

/// The estimate for every group, one row per group and distinct exit time.
///
/// `cif` holds one column per cause, in the order of `causes`.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Curves {
    /// The causes observed, ascending.
    pub causes: Vec<i32>,
    /// The group of each row.
    pub group: Vec<u32>,
    /// The distinct exit time of each row.
    pub time: Vec<f64>,
    /// Weighted at risk just before `time`.
    pub n_risk: Vec<f64>,
    /// The same in individuals.
    pub n_at_risk: Vec<f64>,
    /// Weighted events of every cause.
    pub n_event: Vec<f64>,
    /// `S(t)`.
    pub event_free: Vec<f64>,
    /// `F_k(t)`, one column per cause in the order of `causes`.
    pub cif: Vec<Vec<f64>>,
}

/// One group's rows at one distinct exit time.
struct Block {
    time: f64,
    leaving: f64,
    people: f64,
    d: Vec<f64>,
}

/// The weighted Aalen-Johansen estimate per group, on checked input.
///
/// Rows are ordered by group and time with a stable sort. `entry`, when given,
/// is the delayed-entry time: a row is at risk on `(entry, time]`. Every exit
/// time appears, including times with censoring only, which leave the
/// probabilities unchanged.
// `==` on times is exact on purpose, as in `Arm::build`: rows share a time or
// they do not.
#[expect(clippy::float_cmp, reason = "rows share a time or they do not")]
fn curves_checked(
    time: &[f64],
    event: &[i32],
    weight: &[f64],
    group: &[u32],
    entry: Option<&[f64]>,
) -> Curves {
    let mut causes: Vec<i32> = event.iter().copied().filter(|&e| e != CENSORED).collect();
    causes.sort_unstable();
    causes.dedup();
    let k = causes.len();

    let mut order: Vec<u32> = (0..time.len() as u32).collect();
    order.par_sort_by(|&a, &b| {
        let (a, b) = (a as usize, b as usize);
        group[a].cmp(&group[b]).then(time[a].total_cmp(&time[b]))
    });
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < order.len() {
        let g = group[order[start] as usize];
        let mut end = start + 1;
        while end < order.len() && group[order[end] as usize] == g {
            end += 1;
        }
        ranges.push((start, end));
        start = end;
    }

    let per_group: Vec<Vec<(Block, f64, f64)>> = ranges
        .par_iter()
        .map(|&(lo, hi)| {
            let rows = &order[lo..hi];
            let mut blocks: Vec<Block> = Vec::new();
            for &row in rows {
                let i = row as usize;
                if blocks.last().is_none_or(|b| b.time != time[i]) {
                    blocks.push(Block {
                        time: time[i],
                        leaving: 0.0,
                        people: 0.0,
                        d: vec![0.0; k],
                    });
                }
                let block = blocks.last_mut().expect("pushed above");
                block.leaving += weight[i];
                block.people += 1.0;
                if event[i] != CENSORED {
                    let c = causes.binary_search(&event[i]).expect("collected above");
                    block.d[c] += weight[i];
                }
            }
            let (risk, people) = match entry {
                None => risk_without_entry(&blocks),
                Some(entry) => risk_with_entry(&blocks, rows, entry, time, weight),
            };
            blocks
                .into_iter()
                .zip(risk.into_iter().zip(people))
                .map(|(block, (risk, people))| (block, risk, people))
                .collect()
        })
        .collect();

    let mut out = Curves {
        cif: vec![Vec::new(); k],
        causes,
        ..Curves::default()
    };
    let mut incidence = vec![0.0f64; k];
    for (&(lo, _), blocks) in ranges.iter().zip(per_group) {
        let g = group[order[lo] as usize];
        let mut survival = 1.0f64;
        incidence.fill(0.0);
        for (block, at_risk, people) in blocks {
            // Summed over causes in cause order, which is `n_event`.
            let d_any = block.d.iter().fold(0.0, |total, &d| total + d);
            advance(&mut survival, at_risk, d_any, &block.d, &mut incidence);
            out.group.push(g);
            out.time.push(block.time);
            out.n_risk.push(at_risk);
            out.n_at_risk.push(people);
            out.n_event.push(d_any);
            out.event_free.push(survival);
            for (column, &value) in out.cif.iter_mut().zip(&incidence) {
                column.push(value);
            }
        }
    }
    out
}

/// At risk just before each time without delayed entry: everyone who has not
/// yet left, a reverse cumulative sum over the blocks in time order.
fn risk_without_entry(blocks: &[Block]) -> (Vec<f64>, Vec<f64>) {
    let mut risk = vec![0.0; blocks.len()];
    let mut people = vec![0.0; blocks.len()];
    let (mut w, mut p) = (0.0f64, 0.0f64);
    for (b, block) in blocks.iter().enumerate().rev() {
        w += block.leaving;
        p += block.people;
        risk[b] = w;
        people[b] = p;
    }
    (risk, people)
}

/// At risk just before each exit time under delayed entry.
///
/// A row is at risk on `(entry, time]`, so at `t` the risk set is everyone who
/// entered strictly before `t` less everyone who left strictly before it. One
/// running balance over the merged entry and exit times gives both terms: the
/// changes at each distinct time are summed, and the balance is read before
/// they are added, which makes both inequalities strict.
#[expect(clippy::float_cmp, reason = "changes at one time are summed together")]
fn risk_with_entry(
    blocks: &[Block],
    rows: &[u32],
    entry: &[f64],
    time: &[f64],
    weight: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    // (time, weight change, people change), entries before exits at a tie so the
    // stable sort below keeps the order fixed.
    let mut changes: Vec<(f64, f64, f64)> = Vec::with_capacity(2 * rows.len());
    for &row in rows {
        let i = row as usize;
        changes.push((entry[i], weight[i], 1.0));
    }
    for &row in rows {
        let i = row as usize;
        changes.push((time[i], -weight[i], -1.0));
    }
    changes.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut risk = Vec::with_capacity(blocks.len());
    let mut people = Vec::with_capacity(blocks.len());
    let (mut w, mut p) = (0.0f64, 0.0f64);
    let mut next = 0usize;
    let mut start = 0usize;
    while start < changes.len() {
        let t = changes[start].0;
        let mut end = start;
        let (mut dw, mut dp) = (0.0f64, 0.0f64);
        while end < changes.len() && changes[end].0 == t {
            dw += changes[end].1;
            dp += changes[end].2;
            end += 1;
        }
        if next < blocks.len() && blocks[next].time == t {
            risk.push(w);
            people.push(p);
            next += 1;
        }
        w += dw;
        p += dp;
        start = end;
    }
    (risk, people)
}

/// The weighted Aalen-Johansen estimate for every group.
///
/// `event` is the cause code of each row, with 0 for censoring. `group` is a
/// dense code per row, built by the caller from its grouping. `entry`, when
/// given, is the delayed-entry time: a row is at risk on `(entry, time]`.
/// Returns one row per group and distinct exit time, ordered by both. See
/// [`Curves`] for the columns.
///
/// # Errors
///
/// [`Error::Input`] when the slices differ in length or a time or entry is NaN.
pub fn curves(
    time: &[f64],
    event: &[i32],
    weight: &[f64],
    group: &[u32],
    entry: Option<&[f64]>,
) -> Result<Curves> {
    let n = time.len();
    check_lengths(&[
        ("time", n),
        ("event", event.len()),
        ("weight", weight.len()),
        ("group", group.len()),
        ("entry", entry.map_or(n, <[f64]>::len)),
    ])?;
    check_row_count(n)?;
    if time
        .iter()
        .chain(entry.into_iter().flatten())
        .any(|t| t.is_nan())
    {
        return Err(Error::input("time and entry must not be NaN"));
    }
    Ok(curves_checked(time, event, weight, group, entry))
}

/// How the cluster weights of a replicate are drawn for the cells of a
/// bootstrap.
///
/// `PerCell` treats each cell as its own analysis. The weights are drawn over
/// the clusters present in the cell, numbered from zero within it, and
/// normalised to sum to the number of those clusters. The seed of the cell
/// drives the draw, so the same seed in two cells gives one weight to two
/// different clusters, a coincidence rather than a shared draw.
///
/// `Shared` draws one weight vector over every cluster of the frame, numbered
/// by their codes, normalises it over the whole frame, with the first seed of
/// `cell_seeds`, and then reads it in each cell. Cells then share the weight of
/// a cluster they have in common, which is what an interaction between two
/// cells needs. The two modes give different draws, not the same draws at a
/// different cost.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Draw {
    /// Weights over the clusters of each cell, drawn from that cell's seed.
    PerCell,
    /// One weight vector over the clusters of the whole frame.
    Shared,
}

/// One cell's rows, gathered compactly.
///
/// Clusters are re-indexed to the cell under [`Draw::PerCell`] and keep their
/// codes under [`Draw::Shared`]. The difference is part of the estimator.
struct Cell {
    n_families: usize,
    exposed: Arm,
    comparator: Arm,
}

impl Cell {
    fn gather(
        rows: &[u32],
        time: &[f64],
        event: &[i32],
        weight: &[f64],
        family: &[u32],
        exposed: &[bool],
        draw: Draw,
    ) -> Option<Self> {
        let mut cell_time = Vec::with_capacity(rows.len());
        let mut cell_event = Vec::with_capacity(rows.len());
        let mut cell_weight = Vec::with_capacity(rows.len());
        let mut cell_family = Vec::with_capacity(rows.len());
        let mut local: HashMap<u32, u32> = HashMap::new();
        let mut yes: Vec<u32> = Vec::new();
        let mut no: Vec<u32> = Vec::new();

        for &row in rows {
            let i = row as usize;
            let position = cell_time.len() as u32;
            cell_time.push(time[i]);
            cell_event.push(event[i]);
            cell_weight.push(weight[i]);
            match draw {
                Draw::PerCell => {
                    let next = local.len() as u32;
                    cell_family.push(*local.entry(family[i]).or_insert(next));
                }
                Draw::Shared => cell_family.push(family[i]),
            }
            if exposed[i] { &mut yes } else { &mut no }.push(position);
        }
        if yes.is_empty() || no.is_empty() {
            // One arm empty is not an estimate.
            return None;
        }
        Some(Self {
            exposed: Arm::build(&yes, &cell_time, &cell_event, &cell_weight, &cell_family),
            comparator: Arm::build(&no, &cell_time, &cell_event, &cell_weight, &cell_family),
            n_families: local.len(), // meaningful only under `Draw::PerCell`
        })
    }
}

/// One arm of one cell: positions ordered by time, grouped into distinct times.
struct Arm {
    /// The arm's rows in time order, laid out contiguously so a replicate reads
    /// them in sequence.
    rows: Vec<ArmRow>,
    blocks: Vec<(u32, u32)>,
    times: Vec<f64>,
}

#[derive(Clone, Copy)]
struct ArmRow {
    weight: f64,
    family: u32,
    event: i32,
}

impl Arm {
    // `time[..] == t` is an exact comparison on purpose: this groups rows that
    // share an event time into blocks, and two rows are at the same time or they
    // are not. A tolerance would merge distinct event times into one block and
    // the risk sets would then be wrong at both of them.
    #[expect(clippy::float_cmp, reason = "rows share an event time or they do not")]
    fn build(
        positions: &[u32],
        time: &[f64],
        event: &[i32],
        weight: &[f64],
        family: &[u32],
    ) -> Self {
        let mut order: Vec<u32> = positions.to_vec();
        order.sort_unstable_by(|&a, &b| time[a as usize].total_cmp(&time[b as usize]));
        let mut blocks = Vec::new();
        let mut times = Vec::new();
        let mut start = 0usize;
        while start < order.len() {
            let t = time[order[start] as usize];
            let mut end = start + 1;
            while end < order.len() && time[order[end] as usize] == t {
                end += 1;
            }
            blocks.push((start as u32, end as u32));
            times.push(t);
            start = end;
        }
        let rows = order
            .iter()
            .map(|&position| ArmRow {
                weight: weight[position as usize],
                family: family[position as usize],
                event: event[position as usize],
            })
            .collect();
        Self {
            rows,
            blocks,
            times,
        }
    }

    /// Cumulative incidence of `cause` at each age, under one set of weights.
    fn cif_at_ages(
        &self,
        weights: &[f64],
        ages: &[f64],
        cause: i32,
        scratch: &mut Scratch,
        out: &mut [f64],
    ) {
        let n = self.blocks.len();
        scratch.reset(n);

        for (b, &(lo, hi)) in self.blocks.iter().enumerate() {
            let (mut leaving, mut d_any, mut d_cause) = (0.0, 0.0, 0.0);
            for row in &self.rows[lo as usize..hi as usize] {
                let w = row.weight * weights[row.family as usize];
                leaving += w;
                if row.event != CENSORED {
                    d_any += w;
                    if row.event == cause {
                        d_cause += w;
                    }
                }
            }
            scratch.leaving[b] = leaving;
            scratch.d_any[b] = d_any;
            scratch.d_cause[b] = d_cause;
        }

        // At risk just before t: everyone who has not yet left, a reverse
        // cumulative sum, which is why the blocks are ordered by time.
        let mut running = 0.0;
        for b in (0..n).rev() {
            running += scratch.leaving[b];
            scratch.at_risk[b] = running;
        }

        // Forward pass. Survival just BEFORE the time is the factor on each
        // increment, and is 1 before the first event time by definition.
        let mut survival = 1.0f64;
        let mut cif = 0.0f64;
        let mut age_at = 0usize;
        for b in 0..n {
            // Ages falling before this block take the incidence so far, which
            // for ages before the first event is zero. The estimator documents
            // that as an estimate rather than an absence of one: everyone is at
            // risk from a common origin.
            while age_at < ages.len() && ages[age_at] < self.times[b] {
                out[age_at] = cif;
                age_at += 1;
            }
            advance(
                &mut survival,
                scratch.at_risk[b],
                scratch.d_any[b],
                &[scratch.d_cause[b]],
                std::slice::from_mut(&mut cif),
            );
        }
        for slot in out.iter_mut().skip(age_at) {
            *slot = cif;
        }
    }
}

/// Per-replicate working vectors, reused across the replicates of a cell.
#[derive(Default)]
struct Scratch {
    leaving: Vec<f64>,
    d_any: Vec<f64>,
    d_cause: Vec<f64>,
    at_risk: Vec<f64>,
    weights: Vec<f64>,
}

impl Scratch {
    fn reset(&mut self, n: usize) {
        for buffer in [
            &mut self.leaving,
            &mut self.d_any,
            &mut self.d_cause,
            &mut self.at_risk,
        ] {
            buffer.clear();
            buffer.resize(n, 0.0);
        }
    }
}

/// Every row of a bootstrapped frame: the columns both arms are read from.
#[derive(Debug, Clone, PartialEq)]
pub struct ArmColumns {
    time: Vec<f64>,
    event: Vec<i32>,
    weight: Vec<f64>,
    family: Vec<u32>,
    exposed: Vec<bool>,
}

impl ArmColumns {
    /// The rows: exit `time`, cause code `event` (0 for censoring), `weight`,
    /// dense cluster code `family` and whether the row is in the exposed arm.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the columns differ in length.
    pub fn new(
        time: Vec<f64>,
        event: Vec<i32>,
        weight: Vec<f64>,
        family: Vec<u32>,
        exposed: Vec<bool>,
    ) -> Result<Self> {
        check_lengths(&[
            ("time", time.len()),
            ("weight", weight.len()),
            ("event", event.len()),
            ("family", family.len()),
            ("exposed", exposed.len()),
        ])?;
        check_row_count(time.len())?;
        Ok(Self {
            time,
            event,
            weight,
            family,
            exposed,
        })
    }

    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.time.len()
    }

    /// Whether there are no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.time.is_empty()
    }
}

/// Which rows each of several windows reads, as a CSR list.
///
/// Window `w` reads `rows[offsets[w]..offsets[w + 1]]`, and a row may belong
/// to several windows. Built only from offsets that are non-decreasing and
/// within `rows`, and from rows within the frame, so reading a window cannot
/// go out of bounds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RowWindows {
    rows: Vec<u32>,
    offsets: Vec<u32>,
    n_rows: usize,
}

impl RowWindows {
    /// `label` names the arguments in an error, as `{label}_rows` and
    /// `{label}_offsets`. `n_rows` is the number of rows of the frame the
    /// windows index.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the offsets are empty, decrease or run past
    /// `rows`, or a row indexes past `n_rows`.
    pub fn new(rows: Vec<u32>, offsets: Vec<u32>, n_rows: usize, label: &str) -> Result<Self> {
        let Some(&last) = offsets.last() else {
            return Err(Error::input(format!("{label}_offsets must not be empty")));
        };
        if offsets.windows(2).any(|pair| pair[0] > pair[1]) || last as usize > rows.len() {
            return Err(Error::input(format!(
                "{label}_offsets must be non-decreasing and within {label}_rows"
            )));
        }
        if rows.iter().any(|&row| row as usize >= n_rows) {
            return Err(Error::input(format!("{label}_rows indexes past the frame")));
        }
        Ok(Self {
            rows,
            offsets,
            n_rows,
        })
    }

    /// The number of windows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Whether there are no windows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The rows of window `window`.
    ///
    /// # Panics
    ///
    /// When `window` is not below [`len`](Self::len).
    #[must_use]
    pub fn get(&self, window: usize) -> &[u32] {
        &self.rows[self.offsets[window] as usize..self.offsets[window + 1] as usize]
    }

    /// The number of rows of the frame the windows index.
    #[must_use]
    pub const fn frame_rows(&self) -> usize {
        self.n_rows
    }
}

/// `(exposed, comparator, estimable)`, as [`bootstrap_arm_incidence`]
/// returns them.
pub type ArmDraws = (Vec<f64>, Vec<f64>, Vec<bool>);

/// Bootstrap each arm's cumulative incidence at each age, for every cell.
///
/// Cell `c` owns the rows `cells.get(c)`. A row may appear in several cells,
/// for example when the cells are several partitions of the same cohort. The
/// weights of replicate `r` of a cell are drawn from
/// `replicate_seed(cell_seeds[c], r)`.
///
/// Returns `(exposed, comparator, estimable)`. The two arms are of shape
/// `(cells, ages, n_replicates)`, row-major, with the ages sorted.
/// `estimable[c]` is false for a cell with an empty arm, which has no draws:
/// its block is zero and is not an estimate. `draw` selects how the cluster
/// weights are drawn; see [`Draw`].
///
/// # Errors
///
/// [`Error::Input`] when `cell_seeds` does not hold one seed per cell, or
/// `cells` was built for a frame of another size than `columns`.
pub fn bootstrap_arm_incidence(
    columns: &ArmColumns,
    cells: &RowWindows,
    cell_seeds: &[u64],
    ages: Vec<f64>,
    cause: i32,
    n_replicates: usize,
    draw: Draw,
) -> Result<ArmDraws> {
    let ArmColumns {
        time,
        event,
        weight,
        family,
        exposed,
    } = columns;
    let n_cells = cells.len();
    if cells.frame_rows() != time.len() {
        return Err(Error::input(
            "the cell windows were built for a frame of another size than the columns",
        ));
    }
    if cell_seeds.len() != n_cells {
        return Err(Error::input("cell_seeds must have one seed per cell"));
    }
    if n_cells == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let mut ages_sorted = ages;
    ages_sorted.sort_by(f64::total_cmp);

    let n_ages = ages_sorted.len();
    // One cell's draws: age-major, `n_replicates` values per age.
    let block = n_ages * n_replicates;

    let gathered: Vec<Option<Cell>> = (0..n_cells)
        .into_par_iter()
        .map(|c| Cell::gather(cells.get(c), time, event, weight, family, exposed, draw))
        .collect();
    let estimable: Vec<bool> = gathered.iter().map(Option::is_some).collect();
    let mut exposed_out = vec![0.0f64; n_cells * block];
    let mut comparator_out = vec![0.0f64; n_cells * block];
    if block == 0 {
        return Ok((exposed_out, comparator_out, estimable));
    }

    match draw {
        // Parallel over cells and then replicates, which lets rayon balance a
        // handful of huge cells (`sex`) against a long tail of small ones
        // (`diagnosis_group`).
        Draw::PerCell => exposed_out
            .par_chunks_mut(block)
            .zip(comparator_out.par_chunks_mut(block))
            .zip(gathered.par_iter())
            .enumerate()
            .for_each(|(c, ((exposed_block, comparator_block), cell))| {
                let Some(cell) = cell else {
                    return;
                };
                let seed = cell_seeds[c];
                let per_replicate: Vec<ArmLevels> = (0..n_replicates)
                    .into_par_iter()
                    .map_init(Scratch::default, |scratch, replicate| {
                        let mut weights = std::mem::take(&mut scratch.weights);
                        dirichlet_weights(
                            cell.n_families,
                            replicate_seed(seed, replicate as u64),
                            &mut weights,
                        );
                        let arms = both_arms(cell, &weights, &ages_sorted, cause, scratch);
                        scratch.weights = weights;
                        arms
                    })
                    .collect();
                for (replicate, (a, b)) in per_replicate.iter().enumerate() {
                    scatter(a, replicate, n_replicates, exposed_block);
                    scatter(b, replicate, n_replicates, comparator_block);
                }
            }),

        // Parallel over replicates, because the weights are what is shared:
        // one draw over the frame's families, then every cell measured
        // against it. Cells cannot be the outer loop here without either
        // redrawing per cell, which is the bug this mode exists to avoid, or
        // holding one weight vector per replicate, which is `G` by
        // replicates and gigabytes at cohort scale.
        Draw::Shared => {
            let n_families = family.iter().copied().max().map_or(0, |m| m as usize + 1);
            let seed = cell_seeds[0];
            let per_replicate: Vec<Vec<Option<ArmLevels>>> = (0..n_replicates)
                .into_par_iter()
                .map_init(Scratch::default, |scratch, replicate| {
                    let mut weights = std::mem::take(&mut scratch.weights);
                    dirichlet_weights(
                        n_families,
                        replicate_seed(seed, replicate as u64),
                        &mut weights,
                    );
                    let row = gathered
                        .iter()
                        .map(|cell| {
                            cell.as_ref()
                                .map(|cell| both_arms(cell, &weights, &ages_sorted, cause, scratch))
                        })
                        .collect();
                    scratch.weights = weights;
                    row
                })
                .collect();
            for (replicate, row) in per_replicate.iter().enumerate() {
                for (c, arms) in row.iter().enumerate() {
                    let Some((a, b)) = arms else {
                        continue;
                    };
                    let range = c * block..(c + 1) * block;
                    scatter(a, replicate, n_replicates, &mut exposed_out[range.clone()]);
                    scatter(b, replicate, n_replicates, &mut comparator_out[range]);
                }
            }
        }
    }
    Ok((exposed_out, comparator_out, estimable))
}

/// One cell's exposed and comparator incidence at each age, in one replicate.
type ArmLevels = (Vec<f64>, Vec<f64>);

/// Both arms of one cell under one set of cluster weights.
fn both_arms(
    cell: &Cell,
    weights: &[f64],
    ages: &[f64],
    cause: i32,
    scratch: &mut Scratch,
) -> ArmLevels {
    let mut a = vec![0.0f64; ages.len()];
    let mut b = vec![0.0f64; ages.len()];
    cell.exposed
        .cif_at_ages(weights, ages, cause, scratch, &mut a);
    cell.comparator
        .cif_at_ages(weights, ages, cause, scratch, &mut b);
    (a, b)
}

/// Writes one replicate's value at each age into an age-major block.
///
/// `block` holds `n_replicates` values per age, so the value at age `a` of
/// replicate `r` goes to `block[a * n_replicates + r]`.
fn scatter(by_age: &[f64], replicate: usize, n_replicates: usize, block: &mut [f64]) {
    for (age, &value) in by_age.iter().enumerate() {
        block[age * n_replicates + replicate] = value;
    }
}

#[cfg(test)]
mod curve_tests {
    use super::curves;

    fn close(got: f64, expected: f64) -> bool {
        (got - expected).abs() < 1e-12
    }

    /// `survival::survfit(Surv(time, state) ~ 1, weights = wt)`, R 4.6.1,
    /// survival 3.8.6. Rows are
    /// `(time, n_risk, event_free, cause 1, cause 2)`.
    #[test]
    fn matches_r_survfit_with_weights() {
        let time = [1.0, 2.0, 2.0, 4.0, 5.0, 6.0, 6.0, 8.0, 9.0, 10.0];
        let event = [1, 2, 1, 0, 1, 2, 1, 0, 2, 1];
        let weight = [1.0, 1.0, 1.0, 1.0, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25];
        let reference = [
            (1.0, 5.5, 0.818_181_818_181_818, 0.181_818_181_818_182, 0.0),
            (
                2.0,
                4.5,
                0.454_545_454_545_455,
                0.363_636_363_636_364,
                0.181_818_181_818_182,
            ),
            (
                4.0,
                2.5,
                0.454_545_454_545_455,
                0.363_636_363_636_364,
                0.181_818_181_818_182,
            ),
            (
                5.0,
                1.5,
                0.378_787_878_787_879,
                0.439_393_939_393_939,
                0.181_818_181_818_182,
            ),
            (
                6.0,
                1.25,
                0.227_272_727_272_727,
                0.515_151_515_151_515,
                0.257_575_757_575_758,
            ),
            (
                8.0,
                0.75,
                0.227_272_727_272_727,
                0.515_151_515_151_515,
                0.257_575_757_575_758,
            ),
            (
                9.0,
                0.5,
                0.113_636_363_636_364,
                0.515_151_515_151_515,
                0.371_212_121_212_121,
            ),
            (
                10.0,
                0.25,
                0.0,
                0.628_787_878_787_879,
                0.371_212_121_212_121,
            ),
        ];
        let out = curves(&time, &event, &weight, &[0; 10], None).unwrap();
        assert_eq!(out.causes, vec![1, 2]);
        assert_eq!(out.time.len(), reference.len());
        for (row, &(t, risk, free, c1, c2)) in reference.iter().enumerate() {
            assert!(close(out.time[row], t));
            assert!(close(out.n_risk[row], risk), "n_risk at {t}");
            assert!(close(out.event_free[row], free), "S at {t}");
            assert!(close(out.cif[0][row], c1), "F_1 at {t}");
            assert!(close(out.cif[1][row], c2), "F_2 at {t}");
        }
    }

    /// Groups are estimated separately and come back in code order, whatever
    /// the row order.
    #[test]
    fn groups_are_estimated_separately() {
        let time = [2.0, 1.0, 1.0, 3.0];
        let event = [1, 1, 0, 0];
        let out = curves(&time, &event, &[1.0; 4], &[1, 1, 0, 0], None).unwrap();
        assert_eq!(out.group, vec![0, 0, 1, 1]);
        assert_eq!(out.time, vec![1.0, 3.0, 1.0, 2.0]);
        assert_eq!(out.cif[0], vec![0.0, 0.0, 0.5, 1.0]);
        assert_eq!(out.n_at_risk, vec![2.0, 1.0, 2.0, 1.0]);
    }

    /// Hand-computed: two people from 0, a third only from 2.5, so at t = 1 the
    /// risk set is 2 and the hazard 1/2. At t = 3 the late entrant and the
    /// person leaving at 4 are at risk.
    #[test]
    fn a_late_entrant_is_not_at_risk_before_they_enter() {
        let entry = [0.0, 0.0, 2.5];
        let time = [1.0, 4.0, 3.0];
        let event = [1, 0, 1];
        let out = curves(&time, &event, &[1.0; 3], &[0; 3], Some(&entry)).unwrap();
        assert_eq!(out.time, vec![1.0, 3.0, 4.0]);
        assert_eq!(out.n_risk, vec![2.0, 2.0, 1.0]);
        assert!(close(out.cif[0][0], 0.5));
        assert!(close(out.cif[0][1], 0.75));
        assert!(close(out.event_free[1], 0.25));
    }

    /// Someone entering exactly at an exit time is not at risk at it.
    #[test]
    fn the_risk_set_is_strict_on_entry_and_inclusive_on_exit() {
        let entry = [0.0, 0.0, 2.0];
        let time = [2.0, 5.0, 6.0];
        let out = curves(&time, &[1, 0, 0], &[1.0; 3], &[0; 3], Some(&entry)).unwrap();
        assert_eq!(out.n_risk, vec![2.0, 2.0, 1.0]);
    }

    #[test]
    fn everyone_censored_has_no_cause_and_stays_event_free() {
        let out = curves(&[1.0, 2.0], &[0, 0], &[1.0; 2], &[0; 2], None).unwrap();
        assert!(out.causes.is_empty() && out.cif.is_empty());
        assert_eq!(out.event_free, vec![1.0, 1.0]);
        assert_eq!(out.n_event, vec![0.0, 0.0]);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArmColumns, Cell, Draw, Error, RowWindows, Scratch, bootstrap_arm_incidence, scatter,
    };

    fn cell(time: &[f64], event: &[i32], exposed: &[bool], family: &[u32]) -> Cell {
        let weight = vec![1.0; time.len()];
        let rows: Vec<u32> = (0..time.len() as u32).collect();
        Cell::gather(&rows, time, event, &weight, family, exposed, Draw::PerCell)
            .expect("both arms present")
    }

    /// Worked by hand.
    ///
    /// The exposed arm is two people. At time 1 one has the outcome: 1 of 2 at
    /// risk, so the increment is `S(1-) * 1/2 = 1 * 0.5`. At time 2 the other is
    /// censored, which moves the number at risk and not the incidence.
    #[test]
    fn cumulative_incidence_matches_a_hand_computed_curve() {
        let c = cell(
            &[1.0, 2.0, 5.0],
            &[1, 0, 0],
            &[true, true, false],
            &[0, 1, 2],
        );
        let ages = [0.5, 1.0, 2.0, 3.0];
        let mut out = [0.0; 4];
        let mut scratch = Scratch::default();
        c.exposed
            .cif_at_ages(&[1.0, 1.0, 1.0], &ages, 1, &mut scratch, &mut out);

        assert!((out[0] - 0.0).abs() < 1e-12, "before any event: {out:?}");
        assert!((out[1] - 0.5).abs() < 1e-12, "at the event: {out:?}");
        assert!(
            (out[2] - 0.5).abs() < 1e-12,
            "censoring moves nothing: {out:?}"
        );
        assert!(
            (out[3] - 0.5).abs() < 1e-12,
            "flat past the last time: {out:?}"
        );
    }

    /// A competing event must hold the other cause's incidence below what
    /// one-minus-Kaplan-Meier would give.
    #[test]
    fn a_competing_event_holds_the_incidence_below_naive_survival() {
        let c = cell(
            &[1.0, 2.0, 3.0, 9.0],
            &[2, 1, 0, 0],
            &[true, true, true, false],
            &[0, 1, 2, 3],
        );
        let mut out = [0.0; 1];
        let mut scratch = Scratch::default();
        c.exposed
            .cif_at_ages(&[1.0; 4], &[3.0], 1, &mut scratch, &mut out);

        // S(2-) = 1 - 1/3 = 2/3, and the cause-1 event has 2 at risk:
        // F_1(3) = (2/3)(1/2) = 1/3, below the 1/2 that ignoring the competing
        // cause would give.
        assert!((out[0] - 1.0 / 3.0).abs() < 1e-12, "{out:?}");
        assert!(out[0] < 0.5);
    }

    /// The re-indexing a code shared across cells would get wrong.
    #[test]
    fn families_are_numbered_within_the_cell_not_the_cohort() {
        let time = [1.0, 2.0, 3.0, 4.0];
        let event = [1, 0, 1, 0];
        let weight = [1.0; 4];
        let family = [10, 20, 30, 40];
        let exposed = [true, false, true, false];
        let c = Cell::gather(
            &[0, 3],
            &time,
            &event,
            &weight,
            &family,
            &exposed,
            Draw::PerCell,
        )
        .expect("both arms present");

        assert_eq!(c.n_families, 2, "two families in the cell, not forty-one");
        let families = |arm: &super::Arm| arm.rows.iter().map(|row| row.family).collect::<Vec<_>>();
        assert_eq!(families(&c.exposed), vec![0], "renumbered from zero");
        assert_eq!(families(&c.comparator), vec![1], "renumbered from zero");
    }

    #[test]
    fn a_cell_with_one_arm_empty_is_not_an_estimate() {
        let time = [1.0, 2.0];
        let event = [1, 0];
        let weight = [1.0; 2];
        let family = [0, 1];
        let exposed = [true, true];
        assert!(
            Cell::gather(
                &[0, 1],
                &time,
                &event,
                &weight,
                &family,
                &exposed,
                Draw::PerCell
            )
            .is_none()
        );
    }

    /// The block the caller reads as `[age][replicate]`.
    #[test]
    fn a_replicate_lands_in_its_column_of_the_age_major_block() {
        let mut block = vec![0.0; 6];
        scatter(&[1.0, 2.0], 0, 3, &mut block);
        scatter(&[10.0, 20.0], 2, 3, &mut block);
        assert_eq!(block, vec![1.0, 0.0, 10.0, 2.0, 0.0, 20.0]);
    }

    #[test]
    fn a_malformed_row_list_is_refused_rather_than_read() {
        let refused = |rows: Vec<u32>, offsets: Vec<u32>, n_rows: usize| {
            RowWindows::new(rows, offsets, n_rows, "cell").err()
        };
        assert_eq!(
            refused(vec![0], vec![], 1),
            Some(Error::input("cell_offsets must not be empty"))
        );
        assert!(
            refused(vec![0, 1], vec![0, 2, 1], 2).is_some(),
            "decreasing"
        );
        assert!(refused(vec![0], vec![0, 2], 1).is_some(), "past the rows");
        assert_eq!(
            refused(vec![0, 5], vec![0, 2], 3),
            Some(Error::input("cell_rows indexes past the frame"))
        );
    }

    #[test]
    fn a_window_reads_its_own_slice_of_the_row_list() {
        let windows = RowWindows::new(vec![0, 2, 1, 2], vec![0, 2, 2, 4], 3, "cell").unwrap();
        assert_eq!(windows.len(), 3);
        assert_eq!(windows.get(0), &[0, 2]);
        assert_eq!(windows.get(1), &[] as &[u32]);
        assert_eq!(windows.get(2), &[1, 2]);
    }

    #[test]
    fn the_bootstrap_is_reproducible_and_flags_a_cell_with_one_arm() {
        let columns = ArmColumns::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1, 0, 2, 1, 0, 1],
            vec![1.0; 6],
            vec![0, 1, 2, 3, 4, 5],
            vec![true, true, true, false, false, false],
        )
        .unwrap();
        let cells =
            RowWindows::new(vec![0, 1, 2, 3, 4, 5, 0, 1], vec![0, 6, 8], 6, "cell").unwrap();
        let run = || {
            bootstrap_arm_incidence(
                &columns,
                &cells,
                &[7, 8],
                vec![4.0, 2.0],
                1,
                5,
                Draw::PerCell,
            )
            .unwrap()
        };
        let (exposed, comparator, estimable) = run();
        assert_eq!(estimable, vec![true, false]);
        assert_eq!(exposed.len(), 2 * 2 * 5);
        assert!(exposed[..10].iter().all(|v| (0.0..=1.0).contains(v)));
        assert!(exposed[10..].iter().all(|&v| v.to_bits() == 0));
        assert!(comparator[10..].iter().all(|&v| v.to_bits() == 0));
        let again = run();
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&exposed), bits(&again.0));
        assert_eq!(bits(&comparator), bits(&again.1));
    }

    #[test]
    fn seeds_and_windows_that_do_not_fit_are_refused() {
        let columns = ArmColumns::new(vec![1.0], vec![1], vec![1.0], vec![0], vec![true]).unwrap();
        let cells = RowWindows::new(vec![0], vec![0, 1], 1, "cell").unwrap();
        assert!(
            bootstrap_arm_incidence(&columns, &cells, &[], vec![1.0], 1, 2, Draw::PerCell).is_err()
        );
        let wide = RowWindows::new(vec![0], vec![0, 1], 2, "cell").unwrap();
        assert!(
            bootstrap_arm_incidence(&columns, &wide, &[1], vec![1.0], 1, 2, Draw::PerCell).is_err()
        );
        let none = RowWindows::new(vec![], vec![0], 1, "cell").unwrap();
        let (exposed, _, estimable) =
            bootstrap_arm_incidence(&columns, &none, &[], vec![1.0], 1, 2, Draw::Shared).unwrap();
        assert!(exposed.is_empty() && estimable.is_empty());
    }
}
