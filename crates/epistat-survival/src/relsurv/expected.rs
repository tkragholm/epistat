//! The reference hazard of each row of a cohort over its follow-up.

use std::ops::Range;

use crate::error::{Error, Result};
use crate::pyears::{Layout, PyearsOptions, Walker, walk_row};
use crate::ratetable::RateTable;
use crate::surv::SurvData;

/// The reference hazard of every row of a cohort as a function of the time
/// since the origin of the follow-up scale.
///
/// The hazard of a row is piecewise constant between the cuts of the rate
/// table, so its cumulative hazard `H(t)` is piecewise linear and its expected
/// survival is `exp(-H(t))`. The pieces run from time 0, not from the entry
/// time of a row with delayed entry, because the weights of the Pohar Perme
/// estimator are the inverse of the expected survival since the origin.
///
/// The pieces are not stored. A row keeps its value of each time dimension at
/// time 0, the part of its cell index that comes from the factors and the end
/// of its follow-up, 8 bytes each, and whether it is used, and its pieces are formed by walking it
/// through the table when they are read. The walk of a row depends on these
/// values alone, so it gives the same pieces, and the same cumulative hazards,
/// every time.
#[derive(Debug, Clone)]
pub struct ExpectedHazard {
    /// The cuts of each time dimension of the table.
    cuts: Vec<Vec<f64>>,
    /// The stride in the cells of each time dimension.
    strides: Vec<usize>,
    /// The value of each time dimension at time 0, by dimension and row.
    origins: Vec<Vec<f64>>,
    /// The part of the cell index of each row that comes from the factors.
    base: Vec<usize>,
    /// The end of the follow-up of each row, 0 for a row without pieces.
    horizon: Vec<f64>,
    /// Whether the row was used, that is has a positive weight.
    used: Vec<bool>,
    /// The hazard of each cell, its rate divided by the scale of the table.
    cell_hazard: Vec<f64>,
    /// Whether the time dimensions have extended ends.
    open: bool,
    /// The potential follow-up of every row, for Hakulinen's estimator.
    potential: Option<Vec<f64>>,
}

impl ExpectedHazard {
    /// Builds the hazards of the rows of `data` from the table, the values of
    /// each time dimension at time 0 of the follow-up scale (`origins`, one
    /// slice per time dimension, such as age in days at the origin and the
    /// date of the origin) and the levels of each factor (`factors`), as
    /// [`RateTable::expected`] takes them. Rows of weight 0 get no pieces.
    ///
    /// The hazards are per unit of the follow-up time: the rate of the table
    /// divided by its scale. The cumulative hazard of a row is summed over its
    /// pieces in order, as [`RateTable::expected`] sums the count of a row.
    ///
    /// # Errors
    ///
    /// An input error as for [`RateTable::expected`], and when a used row
    /// has follow-up off the table, which has no expected hazard and is not
    /// assumed to be zero.
    pub fn from_table(
        table: &RateTable,
        data: &SurvData<'_>,
        origins: &[&[f64]],
        factors: &[&[u32]],
    ) -> Result<Self> {
        let surv = data.followup();
        let n = data.n_rows();
        let from_origin = {
            let mut b = SurvData::builder(surv.stop(), surv.status());
            if let Some(w) = data.weight() {
                b = b.weight(w);
            }
            b.build()?
        };
        let dims = table.dimensions(origins, factors)?;
        let Layout {
            data: walked,
            mut cursor,
            ..
        } = Layout::new(&from_origin, &dims, &PyearsOptions::default(), false)?;
        let open = table.ends_extended();
        if open {
            cursor.open_ends();
        }
        let mut used = vec![false; n];
        for row in walked.used_rows() {
            used[row as usize] = true;
        }
        let mut any = vec![false; n];
        let mut off: Option<usize> = None;
        cursor.walk(
            &walked,
            &mut |i, cell, _| {
                if cell.is_some() {
                    any[i] = true;
                } else {
                    off.get_or_insert(i);
                }
            },
            &mut |_, _| {},
        )?;
        if let Some(row) = off {
            return Err(Error::out_of_range(format!(
                "row {row} has follow-up off the rate table, which has no expected hazard"
            )));
        }
        let stop = walked.followup().stop();
        let horizon = any
            .iter()
            .zip(stop)
            .map(|(&any, &end)| if any { end } else { 0.0 })
            .collect();
        let base = (0..n)
            .map(|i| if used[i] { cursor.base(i) } else { 0 })
            .collect();
        let (cuts, strides) = cursor
            .time_dimensions()
            .map(|(cuts, stride)| (cuts.to_vec(), stride))
            .unzip();
        let scale = table.scale();
        Ok(Self {
            cuts,
            strides,
            origins: origins.iter().map(|o| o.to_vec()).collect(),
            base,
            horizon,
            used,
            cell_hazard: table.rates().iter().map(|r| r / scale).collect(),
            open,
            potential: None,
        })
    }

    /// These hazards with the potential follow-up of every row of the data,
    /// which [`NetMethod::Hakulinen`](super::NetMethod::Hakulinen) needs and
    /// the other methods of [`net_survival`](super::net_survival) do not
    /// take: the time a row would have been followed to had it not died, its
    /// exit time if it was censored. `relsurv::rs.surv` sets it to `fin.date`
    /// less the date of diagnosis for those who died. [`net_survival`](super::net_survival) checks
    /// it against the data.
    #[must_use]
    pub fn with_potential(mut self, potential: &[f64]) -> Self {
        self.potential = Some(potential.to_vec());
        self
    }

    /// The potential follow-up of every row, `None` unless it was given with
    /// [`Self::with_potential`].
    #[must_use]
    pub fn potential(&self) -> Option<&[f64]> {
        self.potential.as_deref()
    }

    /// The number of rows of the data it was built from.
    #[must_use]
    pub fn n_rows(&self) -> usize {
        self.horizon.len()
    }

    /// The end of the follow-up of row `i`, 0 for a row without pieces.
    #[must_use]
    pub fn horizon(&self, i: usize) -> f64 {
        self.horizon[i]
    }

    /// The cumulative reference hazard of row `i` from time 0 to `t`.
    ///
    /// Beyond the follow-up of the row it is the cumulative hazard at the end
    /// of the follow-up plus that of the row's walk through the table from
    /// there to `t`, as if the row were followed on, which the expected
    /// survival of Ederer I and Hakulinen needs. `None` for `t` below 0 or not
    /// a number, for a row of weight 0, and for a `t` beyond the follow-up
    /// whose walk leaves the table, which has no rate there (see the rule in
    /// [`crate::ratetable`]).
    #[must_use]
    pub fn cumulative(&self, i: usize, t: f64) -> Option<f64> {
        if !(t >= 0.0 && self.used[i]) {
            return None;
        }
        let end = self.horizon(i);
        let within = |t: f64| {
            self.locate(i, t)
                .map_or(0.0, |at| at.before + (t - at.begin) * at.hazard)
        };
        if t <= end {
            return Some(within(t));
        }
        let mut beyond = 0.0_f64;
        let mut off = false;
        self.walk_beyond(i, t, &mut |cell, length| match cell {
            Some(cell) => beyond += length * self.cell_hazard[cell],
            None => off = true,
        });
        (!off).then(|| within(end) + beyond)
    }

    /// The reference hazard of row `i` at time `t`, that of the piece that
    /// ends at or after `t`, so that at a cut of the table it is the hazard
    /// before the cut. Beyond the follow-up of the row it is that of the row's
    /// walk through the table from there, as in [`ExpectedHazard::cumulative`].
    /// `None` for `t` not above 0 or not a number, for a row of weight 0, and
    /// for a `t` beyond the follow-up whose walk leaves the table.
    #[must_use]
    pub fn hazard(&self, i: usize, t: f64) -> Option<f64> {
        if t.is_nan() || t <= 0.0 || !self.used[i] {
            return None;
        }
        if t <= self.horizon[i] {
            return self.locate(i, t).filter(|at| at.within).map(|at| at.hazard);
        }
        let mut last: Option<Option<usize>> = None;
        self.walk_beyond(i, t, &mut |cell, _| {
            if last != Some(None) {
                last = Some(cell);
            }
        });
        last.flatten().map(|cell| self.cell_hazard[cell])
    }

    /// Walks row `i` through the table from the end of its follow-up to `t`.
    fn walk_beyond(&self, i: usize, t: f64, on_piece: &mut impl FnMut(Option<usize>, f64)) {
        let mut walkers: Vec<Walker<'_>> = self.walkers().collect();
        walk_row(&mut walkers, i, self.base[i], self.horizon[i], t, on_piece);
    }

    /// The piece of row `i` that ends at or after `t`, or the last piece when
    /// `t` is beyond them all, as `RowPieces` finds it, from one walk of the
    /// row without storing its pieces. `None` for a row without pieces.
    fn locate(&self, i: usize, t: f64) -> Option<Located> {
        let end = self.horizon[i];
        if end == 0.0 {
            return None;
        }
        let mut stack = [Walker::new(&[], &[], 0); 4];
        let mut heap = Vec::new();
        let walkers: &mut [Walker<'_>] = if self.cuts.len() <= stack.len() {
            for (k, w) in self.walkers().enumerate() {
                stack[k] = w;
            }
            &mut stack[..self.cuts.len()]
        } else {
            heap.extend(self.walkers());
            &mut heap
        };
        // The latest piece, whose end is not known to be the last, and the
        // cumulative hazard and end of the piece before it.
        let mut latest: Option<(f64, f64, f64)> = None;
        let (mut before, mut begin) = (0.0_f64, 0.0_f64);
        let mut found: Option<Located> = None;
        let (mut elapsed, mut total) = (0.0_f64, 0.0_f64);
        let mut penultimate = f64::NEG_INFINITY;
        walk_row(walkers, i, self.base[i], 0.0, end, &mut |cell, length| {
            let Some(cell) = cell else { return };
            if let Some((e, _, _)) = latest {
                penultimate = e;
            }
            if found.is_none()
                && let Some((e, h, c)) = latest
            {
                if e >= t {
                    found = Some(Located {
                        before,
                        begin,
                        hazard: h,
                        within: true,
                    });
                } else {
                    (before, begin) = (c, e);
                }
            }
            let hazard = self.cell_hazard[cell];
            elapsed += length;
            total += length * hazard;
            latest = Some((elapsed, hazard, total));
        });
        if end < penultimate {
            // The rounding of the sum of the lengths put the end of the
            // follow-up before the end of the piece before the last, so the
            // ends are not in order and the search over the stored ends is
            // the one to follow.
            let pieces = self.reader().row(i).located(t);
            return Some(pieces);
        }
        // The last piece ends at the end of the follow-up.
        found.or_else(|| {
            latest.map(|(_, h, _)| Located {
                before,
                begin,
                hazard: h,
                within: end >= t,
            })
        })
    }

    fn walkers(&self) -> impl Iterator<Item = Walker<'_>> {
        self.cuts
            .iter()
            .zip(&self.strides)
            .zip(&self.origins)
            .map(|((cuts, &stride), origin)| Walker::new(cuts, origin, stride).with_open(self.open))
    }

    /// A reader of the pieces of rows, which keeps its buffers from one row
    /// to the next.
    pub(crate) fn reader(&self) -> Reader<'_> {
        Reader {
            walkers: self.walkers().collect(),
            expected: self,
            pieces: Pieces::default(),
        }
    }
}

/// A piece found by [`ExpectedHazard::locate`]: the cumulative hazard and
/// the time at the end of the piece before it, its hazard, and whether it
/// ends at or after the time looked for.
#[derive(Clone, Copy)]
struct Located {
    before: f64,
    begin: f64,
    hazard: f64,
    within: bool,
}

/// The pieces of rows laid end to end: the end of each piece in the time
/// since the origin, its hazard, and the cumulative hazard at its end.
#[derive(Debug, Default)]
pub(crate) struct Pieces {
    ends: Vec<f64>,
    hazards: Vec<f64>,
    cumulative: Vec<f64>,
}

impl Pieces {
    /// The number of pieces held.
    pub(crate) fn len(&self) -> usize {
        self.ends.len()
    }

    /// The pieces in `range`, the range [`Reader::append`] returned for a row.
    pub(crate) fn slice(&self, range: Range<usize>) -> RowPieces<'_> {
        RowPieces {
            ends: &self.ends[range.clone()],
            hazards: &self.hazards[range.clone()],
            cumulative: &self.cumulative[range],
        }
    }

    fn clear(&mut self) {
        self.ends.clear();
        self.hazards.clear();
        self.cumulative.clear();
    }
}

/// Forms the pieces of rows by walking them through the table.
pub(crate) struct Reader<'e> {
    expected: &'e ExpectedHazard,
    walkers: Vec<Walker<'e>>,
    pieces: Pieces,
}

impl Reader<'_> {
    /// Adds the pieces of row `i` to the end of `into` and returns where they
    /// are. The end of a piece is the sum of the lengths up to it and its
    /// cumulative hazard the sum of the lengths times the hazards, both in
    /// order, and the last piece ends where the follow-up ends, whatever the
    /// rounding of the sum. A row without pieces adds none.
    pub(crate) fn append(&mut self, i: usize, into: &mut Pieces) -> Range<usize> {
        let end = self.expected.horizon[i];
        // `from_table` refused a row with time off the table, found by the
        // same walk, so every piece has a cell.
        self.append_until(i, end, into)
            .unwrap_or(into.len()..into.len())
    }

    /// Adds the pieces of row `i` from time 0 to `end` to `into`, as
    /// [`Reader::append`] does for the follow-up, with `end` at or beyond the
    /// follow-up for the expected survival of a row followed on. `None`, and
    /// nothing added, when the walk leaves the table or the row has weight 0.
    pub(crate) fn append_until(
        &mut self,
        i: usize,
        end: f64,
        into: &mut Pieces,
    ) -> Option<Range<usize>> {
        let e = self.expected;
        let first = into.len();
        if !e.used[i] {
            return None;
        }
        if end == 0.0 {
            return Some(first..first);
        }
        let mut elapsed = 0.0_f64;
        let mut total = 0.0_f64;
        let mut off = false;
        walk_row(
            &mut self.walkers,
            i,
            e.base[i],
            0.0,
            end,
            &mut |cell, length| {
                let Some(cell) = cell else {
                    off = true;
                    return;
                };
                let hazard = e.cell_hazard[cell];
                elapsed += length;
                total += length * hazard;
                into.ends.push(elapsed);
                into.hazards.push(hazard);
                into.cumulative.push(total);
            },
        );
        if off {
            into.ends.truncate(first);
            into.hazards.truncate(first);
            into.cumulative.truncate(first);
            return None;
        }
        let last = into.len();
        if last > first {
            into.ends[last - 1] = end;
        }
        Some(first..last)
    }

    /// The pieces of row `i` from time 0 to `end`, held until the next call,
    /// or `None` as for [`Reader::append_until`].
    pub(crate) fn row_until(&mut self, i: usize, end: f64) -> Option<RowPieces<'_>> {
        let mut pieces = std::mem::take(&mut self.pieces);
        pieces.clear();
        let range = self.append_until(i, end, &mut pieces);
        self.pieces = pieces;
        range.map(|r| self.pieces.slice(r))
    }

    /// The pieces of row `i`, held until the next call.
    pub(crate) fn row(&mut self, i: usize) -> RowPieces<'_> {
        let mut pieces = std::mem::take(&mut self.pieces);
        pieces.clear();
        let range = self.append(i, &mut pieces);
        self.pieces = pieces;
        self.pieces.slice(range)
    }
}

/// The pieces of one row.
#[derive(Clone, Copy)]
pub(crate) struct RowPieces<'a> {
    ends: &'a [f64],
    hazards: &'a [f64],
    cumulative: &'a [f64],
}

impl RowPieces<'_> {
    /// The cumulative hazard at a `t` known to be in `[0, horizon]`, 0 for a
    /// row without pieces.
    pub(crate) fn cumulative_within(&self, t: f64) -> f64 {
        if self.ends.is_empty() {
            return 0.0;
        }
        let at = self.located(t);
        at.before + (t - at.begin) * at.hazard
    }

    /// The piece that ends at or after `t`, or the last piece, by a search
    /// over the ends. The row has pieces.
    fn located(&self, t: f64) -> Located {
        let ends = self.ends;
        let found = ends.partition_point(|&e| e < t);
        let k = found.min(ends.len() - 1);
        let (before, begin) = if k == 0 {
            (0.0, 0.0)
        } else {
            (self.cumulative[k - 1], ends[k - 1])
        };
        Located {
            before,
            begin,
            hazard: self.hazards[k],
            within: found < ends.len(),
        }
    }

    /// Calls `f(a, b, hazard)` for each stretch `(a, b]` of `(lo, hi]` over
    /// which the hazard is constant, in order, for `0 <= lo < hi <= horizon`.
    /// The stretches end exactly at `hi`.
    pub(crate) fn pieces_within(&self, lo: f64, hi: f64, mut f: impl FnMut(f64, f64, f64)) {
        let ends = self.ends;
        if ends.is_empty() {
            return;
        }
        let last = ends.len() - 1;
        let mut k = ends.partition_point(|&e| e <= lo).min(last);
        let mut from = lo;
        loop {
            let to = if k == last { hi } else { ends[k].min(hi) };
            if to > from {
                f(from, to, self.hazards[k]);
            }
            if to >= hi {
                break;
            }
            from = to;
            k += 1;
        }
    }
}
