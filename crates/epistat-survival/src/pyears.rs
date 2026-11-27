//! Person-time and event counts in the cells of a table, as `survival::pyears`
//! without a rate table.
//!
//! The table has one dimension per grouping variable. A dimension is either a
//! factor, whose level is fixed for each row, or a time-dependent cut, as
//! `tcut` in R, whose value advances with follow-up, so that a row contributes
//! time to every cell of an age or calendar-period dimension it passes through.
//! The expected counts that `pyears` computes from a rate table are in
//! [`crate::ratetable`], which shares this module's walk over the cells.
//!
//! # What is counted
//!
//! A row at risk during `(start, stop]` with the weight `w` adds, to each cell
//! it spends time in, the time spent there times `w`. Without a start column
//! the row is at risk from 0. For a time-dependent dimension the caller gives
//! each row the value the variable has at time 0 of the follow-up scale, such
//! as age at the origin of the time axis, and the value at time `s` is that
//! value plus `s`, as `tcut` does in R. A cell of a time dimension with cut
//! points `c_0 <= c_1 <= ... <= c_m` is `[c_j, c_{j+1})`, so a value on a cut
//! belongs to the cell above it. Time spent with any dimension below `c_0` or
//! at or above `c_m` is off the table and is added to
//! [`Pyears::offtable`]. A cell of zero width is never entered.
//!
//! For each cell the result holds
//!
//! - the person-time, divided by the scale;
//! - `n`, the number of rows that contributed a positive amount of time to
//!   the cell, whatever their weight. R calls this the number of subjects, and
//!   a subject with several rows in the data counts once for each row. Rows
//!   of weight 0 are not used and are not counted;
//! - the weighted number of events: a row with the event status counts its
//!   weight in the cell where its follow-up ends, which is the last cell it
//!   spent time in. An event that ends off the table is not counted anywhere.
//!   A row that ends before it has spent any time, which only happens with no
//!   start column and a stop of 0, is placed by its values at time 0.
//!
//! Cells are in the order of R's arrays, the first dimension varying fastest.
//!
//! # Algorithm
//!
//! One pass over the rows of positive weight, in row order, with the sums
//! taken in that order, so the result is the same on any machine and does not
//! depend on threads. A row is cut into pieces of follow-up that stay in one
//! cell. At elapsed time `t` each time dimension `j` is in the cell above the
//! last cut not exceeding `x_j + t`, where `x_j` is its value at the start of
//! the row. The next piece ends at the earliest of the end of follow-up and the
//! time `c - x_j` at which some dimension reaches its next cut `c`, except
//! that while a dimension is below its first cut the row is off the table until
//! every such dimension has entered, and once a dimension is at or beyond its
//! last cut the rest of the row is off the table. The offsets `c - x_j` are
//! computed from the start of the row, so rounding error does not build up
//! over the pieces.
//!
//! A cut that the follow-up reaches at its very end, up to rounding in
//! `x_j + (stop - start)`, belongs to the end and does not start a piece in
//! the next cell, so an event on a boundary stays in the cell below it, as in
//! R. Two dimensions that reach their cuts at the same time up to rounding
//! change cell together, and no piece of rounding size is made between them.
//! Two offsets are the same, and a cut is reached at the end, when they are
//! within four times the machine epsilon of the sum of the magnitudes of the
//! numbers that formed the offset, which bounds their rounding error. R instead drops any final
//! stretch shorter than 1e-8 times the shortest follow-up in the data. The two
//! agree unless a real follow-up ends within that distance of a cut without
//! reaching it.
//!
//! The work for a row is proportional to the number of cells it crosses.
//!
//! # Memory
//!
//! The table takes 24 bytes per cell: person-time, count and events.
//!
//! # References
//!
//! 1. Therneau TM, Grambsch PM. Modeling Survival Data: Extending the Cox
//!    Model. Springer 2000, section on person-years.
//! 2. Breslow NE, Day NE. Statistical Methods in Cancer Research, volume II.
//!    IARC 1987, chapter 3.

use epistat_core::InputError;

use crate::check::check_ascending;
use crate::error::{Error, Result};
use crate::surv::SurvData;
use crate::timefix::TimeFix;

/// One dimension of the table.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum Dimension<'a> {
    /// A factor with the level of every row as a code in `0..levels`. Every
    /// level is a cell, whether any row has it or not.
    Factor {
        /// The level of each row.
        codes: &'a [u32],
        /// The number of levels, at least 1.
        levels: usize,
    },
    /// A time-dependent variable cut into intervals.
    Time {
        /// The value of the variable for each row at time 0 of the follow-up
        /// scale, finite for every row used.
        origin: &'a [f64],
        /// The cut points, finite, not decreasing and at least two. There are
        /// `cuts.len() - 1` cells, the cell `j` being `[cuts[j], cuts[j + 1])`.
        cuts: &'a [f64],
    },
}

/// What [`pyears`] does besides the data and the table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PyearsOptions {
    /// Person-time and the time off the table are divided by this, for
    /// instance 365.25 for follow-up in days and a table in years. Finite and
    /// positive. The default is 1, no rescaling. R's default is 365.25.
    pub scale: f64,
    /// The status code that counts as an event. `None` is allowed only when
    /// `n_codes` is 1, which makes status 1 the event.
    pub cause: Option<u32>,
    /// How near-equal times are treated. The default is exact times, as R's
    /// `pyears` has no tie rule; [`TimeFix::default`] gives `survival`'s
    /// `aeqSurv` rule for callers who want near-ties merged.
    pub timefix: TimeFix,
}

impl Default for PyearsOptions {
    fn default() -> Self {
        Self {
            scale: 1.0,
            cause: None,
            timefix: TimeFix::Exact,
        }
    }
}

/// The cells of a person-time table.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Pyears {
    /// The number of cells in each dimension, in the order given. Empty when
    /// no dimension was given, and then there is one cell.
    pub dims: Vec<usize>,
    /// Person-time in each cell, divided by the scale.
    pub person_time: Vec<f64>,
    /// The number of rows that contributed time to each cell.
    pub n: Vec<u64>,
    /// The weighted number of events in each cell.
    pub events: Vec<f64>,
    /// Person-time that fell outside the table, divided by the scale.
    pub offtable: f64,
    /// The number of rows of the data, including rows of weight 0.
    pub n_rows: usize,
    /// The number of rows used, those with positive weight.
    pub n_used: usize,
}

impl Pyears {
    /// The position of a cell in the vectors, from its coordinate in each
    /// dimension, or `None` when the coordinates do not fit [`dims`](Self::dims).
    #[must_use]
    pub fn index(&self, coordinates: &[usize]) -> Option<usize> {
        if coordinates.len() != self.dims.len() {
            return None;
        }
        let mut stride = 1;
        let mut position = 0;
        for (&c, &size) in coordinates.iter().zip(&self.dims) {
            if c >= size {
                return None;
            }
            position += c * stride;
            stride *= size;
        }
        Some(position)
    }
}

/// A time dimension while a row is walked.
#[derive(Clone, Copy)]
pub(crate) struct Walker<'a> {
    cuts: &'a [f64],
    origin: &'a [f64],
    stride: usize,
    /// The value at the start of the row.
    x0: f64,
    /// The number of cuts at or below the current value. 0 is below the first
    /// cut, `cuts.len()` is at or beyond the last.
    above: usize,
    /// Whether the first cell reaches without limit below the first cut and
    /// the last cell without limit above the last, so that no value is off
    /// the table and only the inner cuts are boundaries.
    open: bool,
}

impl<'a> Walker<'a> {
    /// The time dimension with these cuts, the value of each row at time 0
    /// and the stride of its cells.
    pub(crate) const fn new(cuts: &'a [f64], origin: &'a [f64], stride: usize) -> Self {
        Self {
            cuts,
            origin,
            stride,
            x0: 0.0,
            above: 0,
            open: false,
        }
    }

    /// The same dimension with open ends when `open` is set.
    pub(crate) const fn with_open(mut self, open: bool) -> Self {
        self.open = open;
        self
    }

    /// The offset of the current cell in the cells of the table, `None` off
    /// the table.
    fn cell(&self) -> Option<usize> {
        let n = self.cuts.len();
        let above = if self.open {
            self.above.clamp(1, n - 1)
        } else if self.above == 0 || self.above == n {
            return None;
        } else {
            self.above
        };
        Some((above - 1) * self.stride)
    }
}

/// Person-time, counts and events for the cells of `dims`.
///
/// Every dimension is as long as `data` has rows. Rows of weight 0 are not
/// used. The tie rule of `opts` is applied to the times first.
///
/// # Errors
///
/// An input error when a dimension has the wrong length, a factor code is
/// not below its number of levels, the cuts are fewer than two, not finite or
/// decreasing, a value used is not finite, the table has too many cells to
/// index, the scale is not finite and positive, the cause is missing with
/// several event codes or outside them, a row has a negative stop with no
/// start column, or the tie rule fails, see [`SurvData::with_timefix`].
pub fn pyears(data: &SurvData<'_>, dims: &[Dimension<'_>], opts: &PyearsOptions) -> Result<Pyears> {
    let Layout {
        data,
        code,
        sizes,
        cells,
        mut cursor,
    } = Layout::new(data, dims, opts, true)?;
    let mut person_time = vec![0.0; cells];
    let mut count = vec![0_u64; cells];
    let mut events = vec![0.0; cells];
    let mut offtable = 0.0;
    let n_rows = data.n_rows();
    {
        let surv = data.followup()?;
        let status = surv.status();
        let weight = data.weight();
        let weight_of = |i: usize| weight.map_or(1.0, |w| w[i]);
        cursor.walk(
            &data,
            &mut |i, cell, length| {
                let w = weight_of(i);
                match cell {
                    Some(cell) => {
                        person_time[cell] += length * w;
                        count[cell] += 1;
                    }
                    None => offtable += length * w,
                }
            },
            &mut |i, last| {
                if status[i] == code
                    && let Some(cell) = last
                {
                    events[cell] += weight_of(i);
                }
            },
        )?;
    }
    for value in &mut person_time {
        *value /= opts.scale;
    }
    Ok(Pyears {
        dims: sizes,
        person_time,
        n: count,
        events,
        offtable: offtable / opts.scale,
        n_rows,
        n_used: data.n_used(),
    })
}

/// The checked data and dimensions of a table.
pub(crate) struct Layout<'d, 'x> {
    pub(crate) data: SurvData<'d>,
    pub(crate) code: i32,
    pub(crate) sizes: Vec<usize>,
    pub(crate) cells: usize,
    pub(crate) cursor: Cursor<'x>,
}

/// The dimensions of a table and the state of the walk over a row.
pub(crate) struct Cursor<'x> {
    factors: Vec<(&'x [u32], usize)>,
    walkers: Vec<Walker<'x>>,
}

impl<'d, 'x> Layout<'d, 'x> {
    /// Checks the data and the dimensions as [`pyears`] documents.
    pub(crate) fn new(
        data: &SurvData<'d>,
        dims: &[Dimension<'x>],
        opts: &PyearsOptions,
        needs_cause: bool,
    ) -> Result<Self> {
        if !(opts.scale.is_finite() && opts.scale > 0.0) {
            return Err(Error::options("scale must be finite and positive"));
        }
        let data = data.clone().with_timefix(opts.timefix)?;
        let surv = data.followup()?;
        let code = match opts.cause {
            _ if !needs_cause => 0,
            Some(code) => data.cause(code)?.code(),
            None if surv.n_codes() == 1 => 1,
            None => {
                return Err(Error::missing(format!(
                    "the status has {} event codes, so the cause to count must be given",
                    surv.n_codes()
                )));
            }
        };
        let code =
            i32::try_from(code).map_err(|_| Error::out_of_range("cause does not fit an i32"))?;
        let n_rows = data.n_rows();

        let mut sizes = Vec::with_capacity(dims.len());
        let mut cells = 1_usize;
        for (d, dim) in dims.iter().enumerate() {
            let size = match *dim {
                Dimension::Factor { codes, levels } => {
                    if codes.len() != n_rows {
                        return Err(length_error("codes", codes.len(), n_rows));
                    }
                    if levels == 0 {
                        return Err(Error::value(format!("dimension {d} has no levels")));
                    }
                    if let Some(row) = codes
                        .iter()
                        .position(|&c| usize::try_from(c).map_or(true, |c| c >= levels))
                    {
                        return Err(Error::out_of_range(format!(
                            "dimension {d}: the code {} in row {row} is not below the {levels} levels",
                            codes[row]
                        )));
                    }
                    levels
                }
                Dimension::Time { origin, cuts } => {
                    if origin.len() != n_rows {
                        return Err(length_error("origin", origin.len(), n_rows));
                    }
                    if cuts.is_empty() {
                        return Err(Error::from(InputError::Empty { what: "cuts" }));
                    }
                    if cuts.len() < 2 {
                        return Err(Error::no_information(format!(
                            "dimension {d} needs at least two cut points"
                        )));
                    }
                    check_ascending(cuts, "cuts", false)?;
                    let bad = data.used_rows().find(|&r| !origin[r as usize].is_finite());
                    if let Some(row) = bad {
                        return Err(Error::from(InputError::NotFinite {
                            what: "origin",
                            row: row as usize,
                            column: None,
                        }));
                    }
                    cuts.len() - 1
                }
            };
            cells = cells
                .checked_mul(size)
                .filter(|&c| c <= isize::MAX as usize / 24)
                .ok_or_else(|| Error::value("the table has too many cells"))?;
            sizes.push(size);
        }

        let stop = surv.stop();
        if surv.start().is_none()
            && let Some(row) = data.used_rows().find(|&r| stop[r as usize] < 0.0)
        {
            return Err(Error::value(format!(
                "row {row} has a negative follow-up time and no start time"
            )));
        }

        // Strides, first dimension fastest.
        let mut strides = Vec::with_capacity(dims.len());
        let mut stride = 1_usize;
        for &size in &sizes {
            strides.push(stride);
            stride *= size;
        }
        let mut factors: Vec<(&'x [u32], usize)> = Vec::new();
        let mut walkers: Vec<Walker<'x>> = Vec::new();
        for (dim, &stride) in dims.iter().zip(&strides) {
            match *dim {
                Dimension::Factor { codes, .. } => factors.push((codes, stride)),
                Dimension::Time { origin, cuts } => walkers.push(Walker::new(cuts, origin, stride)),
            }
        }
        Ok(Self {
            data,
            code,
            sizes,
            cells,
            cursor: Cursor { factors, walkers },
        })
    }
}

impl<'x> Cursor<'x> {
    /// The part of the cell index of row `i` that comes from the factors.
    pub(crate) fn base(&self, i: usize) -> usize {
        self.factors
            .iter()
            .map(|&(codes, stride)| codes[i] as usize * stride)
            .sum()
    }

    /// The cuts and the stride of each time dimension, in the order of the
    /// dimensions.
    pub(crate) fn time_dimensions(&self) -> impl Iterator<Item = (&'x [f64], usize)> + '_ {
        self.walkers.iter().map(|w| (w.cuts, w.stride))
    }

    /// Gives every time dimension open ends: its first and last cells reach
    /// without limit, as the cells of R's `ratetable` do.
    pub(crate) fn open_ends(&mut self) {
        for walker in &mut self.walkers {
            walker.open = true;
        }
    }
}

impl Cursor<'_> {
    /// Walks the rows of positive weight in row order. `on_piece(row, cell,
    /// length)` is called for each stretch of follow-up that stays in one
    /// cell, with `cell` `None` for time off the table and `length` not
    /// weighted. `on_end(row, last)` is called after each row with the last
    /// cell it spent time in, or the cell its values at time 0 fall in when it
    /// spent none.
    pub(crate) fn walk(
        &mut self,
        data: &SurvData<'_>,
        on_piece: &mut impl FnMut(usize, Option<usize>, f64),
        on_end: &mut impl FnMut(usize, Option<usize>),
    ) -> Result<()> {
        let surv = data.followup()?;
        let stop = surv.stop();
        let start = surv.start();
        let factors = &self.factors;
        let walkers = &mut self.walkers;

        for row in data.used_rows() {
            let i = row as usize;
            let begin = start.map_or(0.0, |s| s[i]);
            let base: usize = factors
                .iter()
                .map(|&(codes, stride)| codes[i] as usize * stride)
                .sum();
            let last = walk_row(walkers, i, base, begin, stop[i], &mut |cell, length| {
                on_piece(i, cell, length);
            });
            on_end(i, last);
        }
        Ok(())
    }
}

/// Walks row `i` from `begin` to `end` through the time dimensions of
/// `walkers`, with `base` the part of its cell index from the factors.
/// `on_piece(cell, length)` is called for each stretch of follow-up that stays
/// in one cell, with `cell` `None` for time off the table. Returns the last
/// cell the row spent time in, or the cell its values at time 0 fall in when
/// it spent none. The walk depends only on these arguments and the origins of
/// row `i`, so a row walked again gives the same pieces.
pub(crate) fn walk_row(
    walkers: &mut [Walker<'_>],
    i: usize,
    base: usize,
    begin: f64,
    end: f64,
    on_piece: &mut impl FnMut(Option<usize>, f64),
) -> Option<usize> {
    let duration = end - begin;
    for walker in walkers.iter_mut() {
        walker.x0 = walker.origin[i] + begin;
        walker.above = walker.cuts.partition_point(|&c| c <= walker.x0);
    }

    let mut elapsed = 0.0_f64;
    let mut last: Option<usize> = None;
    let mut any_piece = false;
    while elapsed < duration {
        let mut next = duration;
        let mut entry = 0.0_f64;
        let mut below = false;
        let mut beyond = false;
        for walker in walkers.iter() {
            if walker.open {
                // Only the inner cuts are boundaries: the next one above the
                // value, if it is not the last cut.
                let k = walker.above.max(1);
                if k + 1 < walker.cuts.len() {
                    let cut = walker.cuts[k];
                    let mut offset = cut - walker.x0;
                    if duration - offset <= reach(cut, walker.x0, begin, end) {
                        offset = duration;
                    }
                    next = next.min(offset);
                }
                continue;
            }
            if walker.above == walker.cuts.len() {
                beyond = true;
                continue;
            }
            let cut = walker.cuts[walker.above];
            let mut offset = cut - walker.x0;
            let reach = reach(cut, walker.x0, begin, end);
            if duration - offset <= reach {
                offset = duration;
            }
            if walker.above == 0 {
                below = true;
                entry = entry.max(offset);
            } else if offset < next {
                next = offset;
            }
        }
        let to = if beyond {
            duration
        } else if below {
            entry.min(duration)
        } else {
            next
        };
        let length = to - elapsed;
        if length > 0.0 {
            let cell = if beyond || below {
                None
            } else {
                walkers
                    .iter()
                    .map(Walker::cell)
                    .sum::<Option<usize>>()
                    .map(|offset| base + offset)
            };
            on_piece(cell, length);
            last = cell;
            any_piece = true;
        }
        elapsed = to;
        for walker in walkers.iter_mut() {
            while walker.above < walker.cuts.len() {
                let cut = walker.cuts[walker.above];
                if cut - walker.x0 > elapsed + reach(cut, walker.x0, begin, end) {
                    break;
                }
                walker.above += 1;
            }
        }
    }
    if !any_piece {
        last = walkers
            .iter()
            .map(Walker::cell)
            .sum::<Option<usize>>()
            .map(|offset| base + offset);
    }
    last
}

/// The rounding error to allow in the offset `cut - x0` of a boundary, whose
/// terms come from `x0 = origin + begin` and a duration `end - begin`. Four
/// units in the last place of the sum of the magnitudes bound the error of the
/// additions and subtractions that form it.
fn reach(cut: f64, x0: f64, begin: f64, end: f64) -> f64 {
    4.0 * f64::EPSILON * (cut.abs() + x0.abs() + begin.abs() + end.abs())
}

/// The error for the column `what` of a dimension with `got` values against
/// the `rows` rows of the data.
fn length_error(what: &'static str, got: usize, rows: usize) -> Error {
    if got == 0 {
        return Error::from(InputError::Empty { what });
    }
    Error::from(InputError::Length {
        what,
        expected: rows,
        found: got,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> PyearsOptions {
        PyearsOptions {
            timefix: TimeFix::Exact,
            ..PyearsOptions::default()
        }
    }

    #[test]
    fn cuts_met_together_up_to_rounding_make_no_piece_between() {
        // Age 68.1 and year 1998.1 at the start reach 70 and 2000 at the same
        // time, 1.9 years on, up to rounding. Time between would add a visit
        // to the cell where age is above 70 and year is below 2000.
        let stop = [3.0];
        let status = [0];
        let data = SurvData::builder(&stop, &status).build().unwrap();
        let age = [68.1];
        let year = [1998.1];
        let dims = [
            Dimension::Time {
                origin: &age,
                cuts: &[60.0, 70.0, 80.0],
            },
            Dimension::Time {
                origin: &year,
                cuts: &[1990.0, 2000.0, 2010.0],
            },
        ];
        let table = pyears(&data, &dims, &opts()).unwrap();
        assert_eq!(table.n, [1, 0, 0, 1]);
        assert!((table.person_time[0] - 1.9).abs() < 1e-12);
        assert!((table.person_time[3] - 1.1).abs() < 1e-12);
    }

    #[test]
    fn event_on_a_cut_stays_in_the_cell_below() {
        let stop = [0.2];
        let status = [1];
        let data = SurvData::builder(&stop, &status).build().unwrap();
        let origin = [0.1];
        let dims = [Dimension::Time {
            origin: &origin,
            cuts: &[0.0, 0.3, 1.0],
        }];
        // 0.3 - 0.1 is below 0.2 by rounding, though the follow-up only
        // reaches the cut at its end.
        let table = pyears(&data, &dims, &opts()).unwrap();
        assert_eq!(table.events, [1.0, 0.0]);
        assert_eq!(table.n, [1, 0]);
    }

    #[test]
    fn off_table_time_and_events_are_not_cells() {
        let stop = [10.0, 4.0];
        let status = [1, 1];
        let data = SurvData::builder(&stop, &status).build().unwrap();
        let origin = [-2.0, 0.0];
        let dims = [Dimension::Time {
            origin: &origin,
            cuts: &[0.0, 5.0],
        }];
        let table = pyears(&data, &dims, &opts()).unwrap();
        // Row 0 is off the table for 2, in for 5 and off again for 3, and its
        // event is off the table. Row 1 is in for 4 and has its event there.
        assert_eq!(table.person_time, [9.0]);
        assert_eq!(table.offtable.to_bits(), 5.0_f64.to_bits());
        assert_eq!(table.events, [1.0]);
        assert_eq!(table.n, [2]);
    }

    #[test]
    fn rows_of_weight_zero_are_not_used() {
        let stop = [2.0, 3.0];
        let status = [1, 1];
        let weight = [1.0, 0.0];
        let data = SurvData::builder(&stop, &status)
            .weight(&weight)
            .build()
            .unwrap();
        let table = pyears(&data, &[], &opts()).unwrap();
        assert_eq!(
            (table.person_time[0], table.n[0], table.events[0]),
            (2.0, 1, 1.0)
        );
        assert_eq!((table.n_rows, table.n_used), (2, 1));
    }

    #[test]
    fn a_cause_is_needed_with_several_codes() {
        let stop = [2.0, 3.0, 4.0];
        let status = [1, 2, 0];
        let data = SurvData::builder(&stop, &status)
            .n_codes(2)
            .build()
            .unwrap();
        assert!(pyears(&data, &[], &opts()).is_err());
        let two = PyearsOptions {
            cause: Some(2),
            ..opts()
        };
        assert_eq!(pyears(&data, &[], &two).unwrap().events, [1.0]);
    }

    #[test]
    fn refuses_bad_dimensions() {
        let stop = [2.0, 3.0];
        let status = [1, 0];
        let data = SurvData::builder(&stop, &status).build().unwrap();
        let short = [1_u32];
        let codes = [0_u32, 2];
        let origin = [1.0, f64::NAN];
        for dim in [
            Dimension::Factor {
                codes: &short,
                levels: 2,
            },
            Dimension::Factor {
                codes: &codes,
                levels: 2,
            },
            Dimension::Factor {
                codes: &codes,
                levels: 0,
            },
            Dimension::Time {
                origin: &origin,
                cuts: &[0.0, 1.0],
            },
            Dimension::Time {
                origin: &[1.0, 2.0],
                cuts: &[1.0],
            },
            Dimension::Time {
                origin: &[1.0, 2.0],
                cuts: &[2.0, 1.0],
            },
        ] {
            assert!(pyears(&data, &[dim], &opts()).is_err(), "{dim:?}");
        }
        let bad_scale = PyearsOptions {
            scale: 0.0,
            ..opts()
        };
        assert!(pyears(&data, &[], &bad_scale).is_err());
    }

    #[test]
    fn index_follows_the_first_dimension_fastest() {
        let table = Pyears {
            dims: vec![2, 3],
            person_time: vec![],
            n: vec![],
            events: vec![],
            offtable: 0.0,
            n_rows: 0,
            n_used: 0,
        };
        assert_eq!(table.index(&[1, 2]), Some(5));
        assert_eq!(table.index(&[2, 0]), None);
        assert_eq!(table.index(&[0]), None);
    }
}
