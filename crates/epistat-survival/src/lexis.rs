//! Lexis splitting: follow-up that advances on several time scales at once,
//! cut at breaks on each scale, as `Epi::splitLexis` applied scale by scale
//! and `popEpi::splitMulti` do.
//!
//! A person's follow-up in a register study moves along several time scales
//! together: time since entry, age and calendar time. Each row has its own
//! value on every scale at the start of the row, and a scale has its own break
//! points. [`split_lexis`] cuts every row wherever any scale meets one of its
//! breaks, and returns the pieces as columns: the value of every scale at the
//! start of the piece, the piece's length (person-time), the status, and the
//! row of the input each piece came from. [`rates`] fits Poisson models for
//! rates on the pieces, and [`cells`] sums pieces into cells first.
//!
//! # What a split does
//!
//! Row `i` is a stretch of follow-up of duration `d = stop - start`, with
//! `start` taken as 0 when the data has no start column. Scale `j` has the
//! value `x_j` at the start of the row and advances at unit speed, so it has
//! the value `x_j + t` after elapsed time `t`. A break `b` of scale `j` is met
//! at elapsed time `b - x_j`. The row is cut at every elapsed time in
//! `(0, d)` at which some scale meets a break, so a row cut at `m` distinct
//! instants gives `m + 1` pieces, in time order.
//!
//! A piece carries, for every scale, the value at its start. When the piece
//! starts at a break of the scale this value is the break itself, exactly.
//! Otherwise it is `x_j + t`. The length of a piece is the difference of the
//! elapsed times at its two ends, and every elapsed time is computed from the
//! start of the row, so rounding error does not build up over the pieces. The
//! status of the row is on its last piece and every earlier piece has status
//! 0, as a piece that is censored there.
//!
//! [`Lexis::interval`] gives, for every piece and scale, the number of breaks
//! at or before the start of the piece. It is the cell of the piece on that
//! scale, `0` below the first break and `breaks.len()` at or above the last.
//! It follows the instants, not the stored value, so a piece that starts at
//! the break `5` of age has interval `k` even when its stored age is `5` less
//! a rounding error. Use it, not the stored values, to put pieces in cells.
//!
//! # Breaks met at the same instant
//!
//! Two scales often meet their breaks together: a person born on a birthday
//! reaches age 40 on the same day as a calendar year ends, and a scale with
//! every row at its break at entry. Computed with floating-point numbers the
//! two elapsed times can differ by rounding error, and cutting at both would
//! leave a piece of length about `1e-16` in a cell the person never entered.
//! As [`pyears`](crate::pyears) does, two breaks whose elapsed
//! times differ by no more than rounding error are the same instant, and no
//! piece is made between them. The bound is four times the machine epsilon of
//! the sum of the magnitudes of the numbers that formed the elapsed time: the
//! break, the value of the scale at the start, and the start and stop of the
//! row. A break met at the very end of the row, up to that bound, belongs to
//! the end and starts no piece, so an event on a break stays below it, and a
//! break met at the start of the row, up to the bound, is already passed. The
//! value stored for a scale at a break that was met within the bound is the
//! break itself.
//!
//! `Epi::splitLexis` drops any piece shorter than its `tol`, `1.5e-8` of a
//! time unit, and so loses real person-time, and the event of a row if the
//! dropped piece is its last. `popEpi::splitMulti` drops its pieces of length
//! below the same bound. Here only pieces from coincident breaks are avoided,
//! so a stretch of `1e-9` years is a piece. The two agree whenever no real
//! piece is shorter than `1.5e-8` and no instants coincide up to rounding. In
//! the tests this difference is shown in R and asserted.
//!
//! A row of duration 0 is one piece of length 0, with its status, as
//! [`crate::split::split`] returns it. The R functions drop it.
//!
//! # Cropping
//!
//! With [`LexisOptions::crop`] a scale with at least two breaks keeps only the
//! follow-up between its first and last break, as `splitMulti`'s default
//! `drop = TRUE` does: the row is cut to the elapsed times at which every such
//! scale is inside its range, the status is cleared if the end is cut off, and
//! a row with nothing left gives no piece. Without it every row is split along
//! the whole of every scale and the pieces below the first break and above the
//! last are kept, as `splitLexis` does.
//!
//! # Rows and weights
//!
//! Every row is split, whatever its weight, as in [`crate::split::split`]. The
//! timefix rule is not applied: neither R function has one. The weights,
//! strata and clusters of the data are not read. A caller that carries them
//! uses [`Lexis::row`].
//!
//! # Algorithm
//!
//! A row is walked from its start. At each step the next elapsed time is the
//! earliest of the end and the elapsed times at which the scales meet their
//! next break, a piece is emitted, and every scale whose next break is met at
//! that time, up to rounding, moves past it, and past several breaks if it
//! has met several. The work for a row is proportional to the number of
//! pieces times the number of scales. Rows are handled in chunks of a fixed
//! size, in parallel: one pass counts the pieces of each chunk, the counts
//! give every chunk its place in the output, and a second pass writes the
//! pieces there. Every piece depends on its row alone, so the output is the
//! same whatever the number of threads.
//!
//! # Memory and time
//!
//! The pieces take `12 S + 16` bytes each for `S` scales: `S` values and `S`
//! intervals, the length, the status and the row. Ten million rows of
//! follow-up of up to 25 years from ages up to 80, with breaks every 5 years of
//! age, every calendar year over 30 years and at 0, 1, 5, 10 and 20 years since
//! entry, give 152.8 million pieces, 15 per row, and so 7.9 GB, with 1.2 GB
//! more for the events as a column. On a 12-core Apple machine with 18 GB the
//! split took 2.4 s and summing the pieces into the 2028 cells took 2.4 s more,
//! with a peak resident set of 5.4 GB (the operating system compresses idle
//! pages, so plan for the arithmetic).
//!
//! [`split_cells`] does both in one pass and keeps no pieces: 0.6 s for the
//! same data, with a peak resident set of 0.37 GB, which is the input of five
//! columns of 80 MB. Its memory is `24 * 33 * K` bytes for `K` possible cells,
//! whatever the number of rows. The fit on the cells is a fraction of a
//! millisecond. This is the route at register scale when the design depends on
//! cells only. The measurement is the test `scale_of_ten_million_rows` in
//! `tests/lexis_edges.rs`, run with `--ignored`.
//!
//! # References
//!
//! 1. Carstensen B, Plummer M, Laara E, Hills M. Epi: A Package for
//!    Statistical Analysis in Epidemiology. R package, help pages of `Lexis`
//!    and `splitLexis`.
//! 2. Seppa K, Rantanen M, Miettinen J, Hakulinen T. popEpi: Functions for
//!    Epidemiological Analysis using Population Data. R package, help pages
//!    of `splitMulti` and `lexpand`.
//! 3. Carstensen B. Age-period-cohort models for the Lexis diagram. Stat Med.
//!    2007;26(15):3018-45. doi:10.1002/sim.2764
//! 4. Breslow NE, Day NE. Statistical Methods in Cancer Research, volume II.
//!    IARC 1987, chapter 2.

pub mod cells;
pub mod rates;

use std::ops::Range;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::lexis::cells::{CellKey, Cells};
use crate::par;
use crate::surv::SurvData;

/// Rows per parallel chunk. A constant, so the work is the same on any pool.
const CHUNK: usize = 8192;

/// One time scale: the value at the start of every row, and its breaks.
#[derive(Debug, Clone, Copy)]
pub struct TimeScale<'a> {
    /// The value of the scale at the start of each row, finite, as many as
    /// the data has rows. Time since entry is a scale whose entry is the
    /// start of the row, or 0.
    pub entry: &'a [f64],
    /// The breaks, finite, in any order. They are sorted and made unique.
    /// Empty is allowed: the scale then only advances.
    pub breaks: &'a [f64],
}

/// What [`split_lexis`] does besides the data and the scales.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct LexisOptions {
    /// Keep only the follow-up inside the first and last break of every scale
    /// with at least two breaks, see the module documentation. `splitMulti`'s
    /// `drop = TRUE`. The default is no cropping, as `splitLexis`.
    pub crop: bool,
}

/// The pieces of a split, as columns of equal length.
#[derive(Debug, Clone, PartialEq)]
pub struct Lexis {
    /// For each scale, the value at the start of each piece.
    pub entry: Vec<Vec<f64>>,
    /// For each scale, the number of its breaks at or before the start of
    /// each piece, up to rounding, see the module documentation.
    pub interval: Vec<Vec<u32>>,
    /// The length of each piece: its person-time, in the unit of the data.
    pub length: Vec<f64>,
    /// The status of each piece: the status of the row on its last piece and
    /// 0 on the others.
    pub status: Vec<i32>,
    /// The row of the input each piece came from, counted from 0.
    pub row: Vec<u32>,
    /// For each scale, its breaks sorted and unique.
    pub breaks: Vec<Vec<f64>>,
}

impl Lexis {
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

impl Lexis {
    /// The number of pieces.
    #[must_use]
    pub fn len(&self) -> usize {
        self.length.len()
    }

    /// Whether there are no pieces.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.length.is_empty()
    }

    /// The number of time scales.
    #[must_use]
    pub fn n_scales(&self) -> usize {
        self.entry.len()
    }

    /// The number of intervals of a scale: one more than its breaks.
    #[must_use]
    pub fn n_intervals(&self, scale: usize) -> usize {
        self.breaks[scale].len() + 1
    }

    /// The events of each piece: 1 where the status is `code`, else 0.
    #[must_use]
    pub fn events(&self, code: i32) -> Vec<f64> {
        self.status.iter().map(|&s| f64::from(s == code)).collect()
    }
}

/// Everything a walk of one row reads.
struct Walk<'a> {
    entry: Vec<&'a [f64]>,
    breaks: Vec<Vec<f64>>,
    start: Option<&'a [f64]>,
    stop: &'a [f64],
    status: &'a [i32],
    crop: bool,
}

/// What a walk changes: the state of the scales, reused between rows.
struct State {
    x0: Vec<f64>,
    reach: Vec<f64>,
    above: Vec<usize>,
    /// The value to store for a scale at the start of the current piece when
    /// it is a break, else NaN.
    at_break: Vec<f64>,
}

impl State {
    fn new(scales: usize) -> Self {
        Self {
            x0: vec![0.0; scales],
            reach: vec![0.0; scales],
            above: vec![0; scales],
            at_break: vec![f64::NAN; scales],
        }
    }
}

/// One piece, as the walk hands it to its sink.
struct Piece {
    /// Elapsed time at the start of the piece.
    from: f64,
    length: f64,
    status: i32,
}

/// Split the follow-up of `data` at the breaks of `scales`.
///
/// The duration of a row is its stop minus its start, with the start taken as
/// 0 without a start column. The output is in input order, by row and then in
/// time.
///
/// # Errors
///
/// An input error when a scale's entry has not as many values as the data
/// has rows, an entry or a break is not finite, a stop is below its start (or
/// below 0 without a start column), there are more rows than `u32::MAX`,
/// the pieces cannot be indexed, or [`LexisOptions::crop`] is set and a scale
/// has exactly one break.
pub fn split_lexis(
    data: &SurvData<'_>,
    scales: &[TimeScale<'_>],
    options: &LexisOptions,
) -> Result<Lexis> {
    let walk = Walk::new(data, scales, *options)?;
    let n = data.n_rows();
    let s = scales.len();

    // Pass one: the pieces of each chunk.
    let chunks = n.div_ceil(CHUNK);
    let counts: Vec<usize> = (0..chunks)
        .into_par_iter()
        .map(|c| {
            let mut state = State::new(s);
            let mut count = 0_usize;
            for i in c * CHUNK..((c + 1) * CHUNK).min(n) {
                walk.row(i, &mut state, &mut |_, _| count += 1);
            }
            count
        })
        .collect();
    let mut total = 0_usize;
    for &c in &counts {
        total = total
            .checked_add(c)
            .ok_or_else(|| Error::invalid("too many pieces to index"))?;
    }

    let mut out = Lexis {
        entry: vec![vec![0.0; total]; s],
        interval: vec![vec![0; total]; s],
        length: vec![0.0; total],
        status: vec![0; total],
        row: vec![0; total],
        breaks: walk.breaks.clone(),
    };

    // Pass two: every chunk writes its own stretch of every column.
    {
        let mut rest_entry: Vec<&mut [f64]> = out.entry.iter_mut().map(Vec::as_mut_slice).collect();
        let mut rest_interval: Vec<&mut [u32]> =
            out.interval.iter_mut().map(Vec::as_mut_slice).collect();
        let mut rest_length = out.length.as_mut_slice();
        let mut rest_status = out.status.as_mut_slice();
        let mut rest_row = out.row.as_mut_slice();
        let mut jobs = Vec::with_capacity(chunks);
        for (c, &count) in counts.iter().enumerate() {
            let entry: Vec<&mut [f64]> = rest_entry
                .iter_mut()
                .map(|r| {
                    let (head, tail) = std::mem::take(r).split_at_mut(count);
                    *r = tail;
                    head
                })
                .collect();
            let interval: Vec<&mut [u32]> = rest_interval
                .iter_mut()
                .map(|r| {
                    let (head, tail) = std::mem::take(r).split_at_mut(count);
                    *r = tail;
                    head
                })
                .collect();
            let (length, tail) = std::mem::take(&mut rest_length).split_at_mut(count);
            rest_length = tail;
            let (status, tail) = std::mem::take(&mut rest_status).split_at_mut(count);
            rest_status = tail;
            let (row, tail) = std::mem::take(&mut rest_row).split_at_mut(count);
            rest_row = tail;
            jobs.push(ChunkOut {
                first: c * CHUNK,
                entry,
                interval,
                length,
                status,
                row,
            });
        }
        jobs.par_iter_mut().for_each(|job| {
            let mut state = State::new(s);
            let mut at = 0;
            let end = (job.first + CHUNK).min(n);
            for i in job.first..end {
                walk.row(i, &mut state, &mut |state, piece| {
                    for j in 0..s {
                        job.entry[j][at] = if state.at_break[j].is_nan() {
                            state.x0[j] + piece.from
                        } else {
                            state.at_break[j]
                        };
                        job.interval[j][at] = u32::try_from(state.above[j]).unwrap_or(u32::MAX);
                    }
                    job.length[at] = piece.length;
                    job.status[at] = piece.status;
                    // The row count was checked against u32 above.
                    job.row[at] = u32::try_from(i).unwrap_or(u32::MAX);
                    at += 1;
                });
            }
            debug_assert_eq!(at, job.length.len());
        });
    }
    Ok(out)
}

/// Blocks of rows whose tables are held at once in [`split_cells`].
const WAVE: usize = 32;

/// The sums of a block of rows in [`split_cells`], per possible cell.
struct Table {
    length: Vec<f64>,
    events: Vec<f64>,
    count: Vec<u64>,
}

/// Split the follow-up of `data` and sum the pieces into cells in one pass,
/// without keeping the pieces.
///
/// The cells are the combinations of the interval on every scale of `scales`
/// (the first keys, in order, with `breaks.len() + 1` levels each) and the
/// categorical `keys` of the rows (as many codes as the data has rows). A
/// cell has the person-time of the pieces in it, the number of events, which
/// is the number of rows whose status is `cause`, counted on their last piece,
/// and the number of pieces. Only cells with a piece are returned, in the order of
/// their combined key, the first key varying fastest.
///
/// `cause` is as in [`crate::pyears::PyearsOptions::cause`]: `None` is allowed
/// only when the data has one event code.
///
/// This is what [`split_lexis`] followed by [`cells::aggregate`] gives, without
/// the memory of the pieces, which is 52 bytes per piece for three scales. The
/// sums are not the same bits as theirs: a block of 8192 rows is summed in row
/// order into its own table, and the tables are added in block order, so the
/// association of the additions differs, up to rounding of the sums. The
/// result does not depend on the number of threads.
///
/// # Memory
///
/// The table of a block takes 24 bytes per possible cell, `K`, the product of
/// the numbers of levels. Thirty-two tables are held at once, so the memory is
/// about `24 * 33 * K` bytes, whatever the number of rows or pieces. For
/// `K` above a few hundred thousand, sum the pieces of [`split_lexis`]
/// instead.
///
/// # Errors
///
/// As [`split_lexis`], and an input error when a key has the wrong length,
/// no levels or a code outside its levels, the cause is missing with several
/// event codes or is not one of them, or the number of possible cells is too
/// large to index.
pub fn split_cells(
    data: &SurvData<'_>,
    scales: &[TimeScale<'_>],
    keys: &[CellKey<'_>],
    cause: Option<u32>,
    options: &LexisOptions,
) -> Result<Cells> {
    let surv = data.followup()?;
    let code = match cause {
        Some(code) => data.cause(code)?.code(),
        None if surv.n_codes() == 1 => 1,
        None => {
            return Err(Error::invalid(format!(
                "the status has {} event codes, so the cause to count must be given",
                surv.n_codes()
            )));
        }
    };
    let code = i32::try_from(code).map_err(|_| Error::invalid("cause does not fit an i32"))?;
    let walk = Walk::new(data, scales, *options)?;
    let n = data.n_rows();
    let mut levels: Vec<usize> = walk.breaks.iter().map(|b| b.len() + 1).collect();
    for (k, key) in keys.iter().enumerate() {
        if key.codes.len() != n {
            return Err(Error::invalid(format!(
                "key {k} has {} codes against {n} rows",
                key.codes.len()
            )));
        }
        if key.levels == 0 {
            return Err(Error::invalid(format!("key {k} has no levels")));
        }
        if let Some(i) = key
            .codes
            .iter()
            .position(|&c| usize::try_from(c).map_or(true, |c| c >= key.levels))
        {
            return Err(Error::invalid(format!(
                "key {k}: the code {} of row {i} is not below its {} levels",
                key.codes[i], key.levels
            )));
        }
        levels.push(key.levels);
    }
    let s = scales.len();
    let mut strides = Vec::with_capacity(levels.len());
    let mut possible = 1_usize;
    for &l in &levels {
        strides.push(possible);
        possible = possible
            .checked_mul(l)
            .filter(|&p| p <= (isize::MAX as usize) / 24 / (WAVE + 1))
            .ok_or_else(|| Error::invalid("the number of possible cells is too large to index"))?;
    }
    let size = possible;

    let make = |rows: Range<usize>| -> Table {
        let mut t = Table {
            length: vec![0.0; size],
            events: vec![0.0; size],
            count: vec![0; size],
        };
        let mut state = State::new(s);
        for i in rows {
            let base: usize = keys
                .iter()
                .enumerate()
                .map(|(k, key)| key.codes[i] as usize * strides[s + k])
                .sum();
            walk.row(i, &mut state, &mut |state, piece| {
                let cell = base + (0..s).map(|j| state.above[j] * strides[j]).sum::<usize>();
                t.length[cell] += piece.length;
                t.events[cell] += f64::from(piece.status == code);
                t.count[cell] += 1;
            });
        }
        t
    };
    let Table {
        length,
        events,
        count,
    } = par::ordered_reduce(n, CHUNK, WAVE, make, |total, part| {
        for cell in 0..size {
            total.length[cell] += part.length[cell];
            total.events[cell] += part.events[cell];
            total.count[cell] += part.count[cell];
        }
    });
    let mut ids = Vec::new();
    let (mut l, mut e, mut c) = (Vec::new(), Vec::new(), Vec::new());
    let mut pieces = 0_u64;
    for cell in 0..size {
        pieces += count[cell];
        if count[cell] > 0 {
            ids.push(cell);
            l.push(length[cell]);
            e.push(events[cell]);
            c.push(count[cell]);
        }
    }
    let pieces = usize::try_from(pieces).map_err(|_| Error::invalid("too many pieces to count"))?;
    Ok(cells::finish(&levels, &strides, &ids, l, e, c, pieces))
}

struct ChunkOut<'a> {
    first: usize,
    entry: Vec<&'a mut [f64]>,
    interval: Vec<&'a mut [u32]>,
    length: &'a mut [f64],
    status: &'a mut [i32],
    row: &'a mut [u32],
}

/// The rounding error to allow in an elapsed time `cut - x0`, whose terms
/// come from the value at the start and the start and stop of the row.
fn reach(cut: f64, x0: f64, begin: f64, end: f64) -> f64 {
    4.0 * f64::EPSILON * (cut.abs() + x0.abs() + begin.abs() + end.abs())
}

impl<'a> Walk<'a> {
    /// Check the input and sort the breaks.
    fn new(
        data: &'a SurvData<'a>,
        scales: &[TimeScale<'a>],
        options: LexisOptions,
    ) -> Result<Self> {
        let surv = data.followup()?;
        let n = data.n_rows();
        u32::try_from(n).map_err(|_| Error::invalid("more rows than u32::MAX"))?;
        let stop = surv.stop();
        let start = surv.start();
        for i in 0..n {
            let begin = start.map_or(0.0, |s| s[i]);
            if stop[i] < begin || stop[i].is_nan() {
                return Err(Error::invalid(format!(
                    "row {i} stops at {}, before its start {begin}",
                    stop[i]
                )));
            }
        }
        let mut breaks = Vec::with_capacity(scales.len());
        for (j, scale) in scales.iter().enumerate() {
            if scale.entry.len() != n {
                return Err(Error::invalid(format!(
                    "scale {j} has {} entry values against {n} rows",
                    scale.entry.len()
                )));
            }
            if let Some(i) = scale.entry.iter().position(|v| !v.is_finite()) {
                return Err(Error::invalid(format!(
                    "scale {j}: the entry value of row {i} is not finite"
                )));
            }
            if let Some(b) = scale.breaks.iter().find(|b| !b.is_finite()) {
                return Err(Error::invalid(format!(
                    "scale {j}: the breaks must be finite, found {b}"
                )));
            }
            let mut b = scale.breaks.to_vec();
            b.sort_unstable_by(f64::total_cmp);
            b.dedup();
            if options.crop && b.len() == 1 {
                return Err(Error::invalid(format!(
                    "scale {j} has one break, and cropping needs none or at least two"
                )));
            }
            breaks.push(b);
        }
        Ok(Walk {
            entry: scales.iter().map(|s| s.entry).collect(),
            breaks,
            start,
            stop,
            status: surv.status(),
            crop: options.crop,
        })
    }
}

impl Walk<'_> {
    /// Walk row `i`, handing every piece to `sink` with the state of the
    /// scales at its start.
    fn row(&self, i: usize, state: &mut State, sink: &mut impl FnMut(&State, Piece)) {
        let scales = self.entry.len();
        let begin = self.start.map_or(0.0, |s| s[i]);
        let end = self.stop[i];
        let duration = end - begin;
        let mut from = 0.0_f64;
        let mut to = duration;
        let mut cropped_end = false;
        for j in 0..scales {
            state.x0[j] = self.entry[j][i];
            state.above[j] = 0;
            state.at_break[j] = f64::NAN;
        }
        if self.crop {
            for j in 0..scales {
                let b = &self.breaks[j];
                if b.len() < 2 {
                    continue;
                }
                let (low, high) = (b[0], b[b.len() - 1]);
                let x0 = state.x0[j];
                let r_low = reach(low, x0, begin, end);
                let r_high = reach(high, x0, begin, end);
                if (low - x0) > r_low {
                    from = from.max(low - x0);
                }
                if high - x0 < duration - r_high {
                    to = to.min(high - x0);
                    cropped_end = true;
                }
            }
            if to < from || (to <= from && duration > 0.0) {
                return;
            }
            if to >= duration {
                to = duration;
                cropped_end = false;
            }
        }
        // The scales that put the start of the piece at a break.
        for j in 0..scales {
            let x0 = state.x0[j];
            state.reach[j] = 0.0;
            let b = &self.breaks[j];
            let mut above = b.partition_point(|&c| c <= x0);
            while above < b.len() {
                let r = reach(b[above], x0, begin, end);
                if b[above] - x0 <= from + r {
                    above += 1;
                } else {
                    break;
                }
            }
            state.above[j] = above;
            if from > 0.0
                && above > 0
                && (b[above - 1] - x0 - from).abs() <= reach(b[above - 1], x0, begin, end)
            {
                state.at_break[j] = b[above - 1];
            }
        }
        let final_status = if cropped_end { 0 } else { self.status[i] };
        let mut elapsed = from;
        loop {
            let mut next = to;
            for j in 0..scales {
                let b = &self.breaks[j];
                if state.above[j] == b.len() {
                    continue;
                }
                let cut = b[state.above[j]];
                let offset = cut - state.x0[j];
                if to - offset <= reach(cut, state.x0[j], begin, end) {
                    continue;
                }
                if offset < next {
                    next = offset;
                }
            }
            let last = next >= to;
            sink(
                state,
                Piece {
                    from: elapsed,
                    length: next - elapsed,
                    status: if last { final_status } else { 0 },
                },
            );
            if last {
                return;
            }
            elapsed = next;
            for j in 0..scales {
                let b = &self.breaks[j];
                state.at_break[j] = f64::NAN;
                while state.above[j] < b.len() {
                    let cut = b[state.above[j]];
                    if cut - state.x0[j] > elapsed + reach(cut, state.x0[j], begin, end) {
                        break;
                    }
                    state.above[j] += 1;
                    state.at_break[j] = cut;
                }
            }
        }
    }
}
