//! The sweep over the event times of a stratum with the risk-set sums, shared
//! by the fit and by the residuals.
//!
//! A stratum's rows are in descending exit order. At each event time, in
//! descending order, rows enter the risk set when their exit is at or after the
//! time and leave it when their entry is at or after the time, and the sums of
//! the weights `c`, of `c x` and of `c x x'` over the risk set and over the
//! events at the time are handed to a visitor.
//!
//! # Blocks
//!
//! A large stratum is swept in blocks of consecutive event times that run in
//! parallel. A block begins at the first event time, or at the first event
//! time at which at least [`BLOCK_ROWS`] rows have entered the risk set since
//! the block before it began, so the blocks depend on the times alone and not
//! on the number of threads. The risk-set sums at the start of block `b` are
//!
//! ```text
//! S(b) = S(b - 1) + E(b - 1) - L(b - 1),        S(0) = 0,
//! ```
//!
//! where `E(b)` and `L(b)` are the sums over the rows that enter and leave
//! during block `b`, each formed in row order by one task and added in block
//! order on one thread. Within a block the sweep is the sequential one,
//! started from `S(b)`. The sums differ from those of one running sum over
//! the whole stratum only by the grouping of the additions, so a sum carries
//! a rounding error of the order of the machine epsilon times the sum of the
//! absolute values of its terms in either case, and every result is the same
//! on every number of threads.

use std::ops::Range;

use rayon::prelude::*;

use super::moments;

/// The rows that enter the risk set during one block of event times, at
/// least, before the next block begins. A stratum with fewer rows is one
/// block and is swept as one running sum.
pub(crate) const BLOCK_ROWS: usize = 1 << 14;

/// The sums `sum c x^r` for `r = 0, 1, 2`, with the lower triangle of `r = 2`
/// when the second moment is wanted.
#[derive(Clone)]
pub(crate) struct Moments {
    pub(crate) m0: f64,
    pub(crate) m1: Vec<f64>,
    pub(crate) m2: Vec<f64>,
}

impl Moments {
    fn new(p: usize, second: bool) -> Self {
        Self {
            m0: 0.0,
            m1: vec![0.0; p],
            m2: vec![0.0; if second { p * p } else { 0 }],
        }
    }

    fn clear(&mut self) {
        self.m0 = 0.0;
        self.m1.fill(0.0);
        self.m2.fill(0.0);
    }

    fn add(&mut self, c: f64, x: &[f64]) {
        self.m0 += c;
        moments::add(c, x, &mut self.m1, &mut self.m2);
    }

    /// Adds `other`, cell by cell.
    fn add_moments(&mut self, other: &Self) {
        self.m0 += other.m0;
        for (a, b) in self.m1.iter_mut().zip(&other.m1) {
            *a += b;
        }
        for (a, b) in self.m2.iter_mut().zip(&other.m2) {
            *a += b;
        }
    }

    fn copy_from(&mut self, other: &Self) {
        self.m0 = other.m0;
        self.m1.copy_from_slice(&other.m1);
        self.m2.copy_from_slice(&other.m2);
    }
}

/// A block of consecutive event times of a stratum, with the number of rows
/// that have entered and that have left the risk set before its first event
/// time.
#[derive(Clone, Debug)]
pub(crate) struct Block {
    pub(crate) events: Range<usize>,
    pub(crate) entered: usize,
    left: usize,
}

/// The sums over the risk set and over the events at one time.
pub(crate) struct Sums {
    pub(crate) risk: Moments,
    pub(crate) dying: Moments,
}

impl Sums {
    pub(crate) fn new(p: usize, second: bool) -> Self {
        Self {
            risk: Moments::new(p, second),
            dying: Moments::new(p, second),
        }
    }
}

/// What the sweep reads of one stratum.
pub(crate) struct RiskSets<'a> {
    /// Covariates, `p` per row, in descending exit order.
    pub(crate) x: &'a [f64],
    /// Exit times in descending order.
    pub(crate) stop: &'a [f64],
    /// Positions in descending entry order, and the entry times in that order.
    /// Both are empty when no row has a delayed entry.
    pub(crate) by_start: &'a [u32],
    pub(crate) start_sorted: &'a [f64],
    /// Distinct event times, descending.
    pub(crate) event_times: &'a [f64],
    /// The positions of the events at each time:
    /// `dying[dying_offset[k]..dying_offset[k + 1]]`.
    pub(crate) dying: &'a [u32],
    pub(crate) dying_offset: &'a [usize],
}

impl RiskSets<'_> {
    /// Visit every event time in descending order with the risk-set sums and
    /// the sums over the events at that time, at the per-row weights `risk`,
    /// which are `w exp(beta' x)`.
    pub(crate) fn sweep(
        &self,
        p: usize,
        risk: &[f64],
        active: &mut Vec<bool>,
        sums: &mut Sums,
        mut visit: impl FnMut(usize, &Sums),
    ) {
        let size = self.stop.len();
        active.clear();
        active.resize(size, false);
        sums.risk.clear();
        let (mut entering, mut leaving) = (0usize, 0usize);
        for (position, &time) in self.event_times.iter().enumerate() {
            while entering < size && self.stop[entering] >= time {
                sums.risk
                    .add(risk[entering], &self.x[entering * p..(entering + 1) * p]);
                active[entering] = true;
                entering += 1;
            }
            while leaving < self.by_start.len() && self.start_sorted[leaving] >= time {
                let row = self.by_start[leaving] as usize;
                if active[row] {
                    sums.risk.add(-risk[row], &self.x[row * p..(row + 1) * p]);
                    active[row] = false;
                }
                leaving += 1;
            }
            sums.dying.clear();
            for &row in &self.dying[self.dying_offset[position]..self.dying_offset[position + 1]] {
                let row = row as usize;
                sums.dying.add(risk[row], &self.x[row * p..(row + 1) * p]);
            }
            visit(position, sums);
        }
    }
}

impl RiskSets<'_> {
    /// The blocks of the event times, as the module documentation describes,
    /// with `rows` for [`BLOCK_ROWS`]. One block when there are fewer rows.
    pub(crate) fn blocks(&self, rows: usize) -> Vec<Block> {
        let size = self.stop.len();
        let mut blocks = Vec::new();
        let mut current = Block {
            events: 0..0,
            entered: 0,
            left: 0,
        };
        let (mut entering, mut leaving) = (0usize, 0usize);
        for (position, &time) in self.event_times.iter().enumerate() {
            if position > current.events.start && entering - current.entered >= rows {
                current.events.end = position;
                blocks.push(current);
                current = Block {
                    events: position..position,
                    entered: entering,
                    left: leaving,
                };
            }
            while entering < size && self.stop[entering] >= time {
                entering += 1;
            }
            while leaving < self.by_start.len() && self.start_sorted[leaving] >= time {
                leaving += 1;
            }
        }
        current.events.end = self.event_times.len();
        blocks.push(current);
        blocks
    }

    /// The risk-set sums at the start of each block, at the per-row weights
    /// `risk`: the sums of the rows entering and leaving in each block are
    /// formed in parallel and added in block order.
    pub(crate) fn block_starts(&self, p: usize, risk: &[f64], blocks: &[Block]) -> Vec<Moments> {
        let deltas: Vec<Moments> = blocks
            .par_windows(2)
            .map(|pair| {
                let mut delta = Moments::new(p, true);
                let entering = pair[0].entered..pair[1].entered;
                for (row, &weight) in entering.clone().zip(&risk[entering]) {
                    delta.add(weight, &self.x[row * p..(row + 1) * p]);
                }
                for &row in &self.by_start[pair[0].left..pair[1].left] {
                    let row = row as usize;
                    delta.add(-risk[row], &self.x[row * p..(row + 1) * p]);
                }
                delta
            })
            .collect();
        let mut starts = Vec::with_capacity(blocks.len());
        let mut current = Moments::new(p, true);
        starts.push(current.clone());
        for delta in &deltas {
            current.add_moments(delta);
            starts.push(current.clone());
        }
        starts
    }

    /// The sweep of [`sweep`](Self::sweep) over the event times of `block`,
    /// from the risk-set sums `start` just before its first event time.
    ///
    /// A row leaves the risk set when its entry is at or after the time, and
    /// its exit is after its entry, so it has entered by then; no record of
    /// the rows in the risk set is needed.
    pub(crate) fn sweep_block(
        &self,
        p: usize,
        risk: &[f64],
        block: &Block,
        start: &Moments,
        sums: &mut Sums,
        mut visit: impl FnMut(usize, &Sums),
    ) {
        let size = self.stop.len();
        sums.risk.copy_from(start);
        let (mut entering, mut leaving) = (block.entered, block.left);
        for position in block.events.clone() {
            let time = self.event_times[position];
            while entering < size && self.stop[entering] >= time {
                sums.risk
                    .add(risk[entering], &self.x[entering * p..(entering + 1) * p]);
                entering += 1;
            }
            while leaving < self.by_start.len() && self.start_sorted[leaving] >= time {
                let row = self.by_start[leaving] as usize;
                sums.risk.add(-risk[row], &self.x[row * p..(row + 1) * p]);
                leaving += 1;
            }
            sums.dying.clear();
            for &row in &self.dying[self.dying_offset[position]..self.dying_offset[position + 1]] {
                let row = row as usize;
                sums.dying.add(risk[row], &self.x[row * p..(row + 1) * p]);
            }
            visit(position, sums);
        }
    }
}

/// The terms of the partial likelihood at one event time as `(weight,
/// fraction)`: the denominator is the risk-set sum less `fraction` of the sum
/// over the events, and the term is multiplied by `weight`. `dying` holds the
/// positions of the events at the time and `weight` the case weight of each
/// position. Breslow's approximation has one term, Efron's one per event.
pub(crate) fn terms(
    weight: &[f64],
    dying: &[u32],
    efron: bool,
) -> impl Iterator<Item = (f64, f64)> + use<> {
    let count = dying.len();
    let total: f64 = dying.iter().map(|&row| weight[row as usize]).sum();
    let (terms, weight, spread) = if efron {
        (count, total / count as f64, 1.0 / count as f64)
    } else {
        (1, total, 0.0)
    };
    (0..terms).map(move |k| (weight, k as f64 * spread))
}
