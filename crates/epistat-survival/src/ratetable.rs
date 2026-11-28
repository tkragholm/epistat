//! A table of reference rates by age, sex and calendar year, and the number
//! of events a cohort would have if it had those rates.
//!
//! A [`RateTable`] is the `ratetable` of `survival`: a dense array of
//! hazards with one dimension per classification, each either a time
//! dimension, whose cell is found from a value that advances with follow-up
//! (age, calendar date), or a factor (sex). [`RateTable::expected`] walks the
//! follow-up of each row through the cells of the table, as [`crate::pyears`]
//! does, and adds `length * rate / scale` for every stretch of follow-up that
//! stays in one cell. The total over rows is the expected number of events
//! `E = sum_s y_s lambda_s` of a standardised incidence or mortality ratio,
//! and the expected count of one row is its cumulative reference hazard, from
//! which its expected survival `exp(-H)` follows, as in relative survival.
//!
//! # Cells
//!
//! A time dimension is given by its cut points, strictly increasing, with
//! cell `j` the interval `[c_j, c_{j+1})`, as the cut points of
//! [`crate::pyears::Dimension::Time`]. A factor has `levels` cells, coded
//! `0..levels`. [`RateTable::new`] takes the rates in the order of the cells,
//! and [`RateTable::from_long`] takes one row per cell, with the value of
//! each dimension in the row, and puts them in that order.
//!
//! # Time off the table
//!
//! One rule decides where the table has a rate, for every function that reads
//! it. A value of a time dimension below its first cut, or at or above its
//! last, has no rate, unless the table has been given extended ends with
//! [`RateTable::extend_ends`]: then the first cell reaches without limit
//! below and the last without limit above, as the cells of R's `ratetable`
//! do, and every value has a rate. Follow-up without a rate is never given
//! one: [`RateTable::rate`] returns `None` for it, [`RateTable::expected`] and
//! [`RateTable::expected_keyed`] leave it out of the expected count and add it
//! to `offtable`, as [`crate::pyears::Pyears::offtable`] does, so the caller
//! sees how much is missing, and
//! [`ExpectedHazard`](crate::relsurv::ExpectedHazard) refuses a row with such
//! follow-up and returns `None` for a time beyond its follow-up that has no
//! rate, since an expected survival over time without a rate would be that
//! of a population without deaths there. The choice between a table that
//! ends and one with extended ends is the caller's, and belongs in the
//! methods section: rates usually fall over calendar time, so the last year's
//! rates applied to later years slightly overstate them.
//!
//! The rates are hazards per `scale` units of the follow-up time. With
//! follow-up in days and daily hazards, as `survexp.us` has, the scale is 1.
//! With follow-up in days and hazards per year, it is 365.25. The cut points
//! of a time dimension are in the units of the follow-up time, since the value
//! of the dimension advances one for one with the follow-up, as in
//! [`crate::pyears`]. Hazards are not negative and finite, and a missing rate is
//! an error at construction.
//!
//! # Algorithm
//!
//! [`RateTable::expected`] shares its walk with [`crate::pyears`], so a row is
//! cut at the same places and a cut reached at the end of follow-up belongs
//! to the end. The person-time of a cell is summed over the pieces in row
//! order, weighted, and the expected count of the cell is that person-time
//! times its rate divided by the scale, one product per cell, so that
//! [`RateTable::expected_cells`] of a [`crate::pyears::pyears`] table on the
//! table's own dimensions has the same bits. The total is the compensated sum
//! over the cells in cell order. The expected count of a row is summed over its
//! pieces in order and is not weighted. The walk is a single serial pass in
//! row order, so the result does not depend on threads.
//!
//! For a cohort of many rows, [`crate::lexis::split_cells`] sums the pieces
//! into cells in parallel, with the table's cut points as the breaks of its
//! time scales, and [`RateTable::expected_keyed`] turns those cells into
//! expected counts. The work is then the split, and the lookup costs one
//! product per cell.
//!
//! # Lookup
//!
//! [`RateTable::rate`] finds the cell of each time dimension by a binary
//! search of its cuts, `O(log m)`, and the position by the strides of the
//! dimensions, first dimension fastest as in R.
//!
//! # Accuracy
//!
//! The expected count of a cell is a product of a sum of at most `p` weighted
//! lengths, so its relative error is about `p` units of machine epsilon, where
//! `p` is the number of pieces in the cell. The total adds the compensated sum
//! of the cells. The tests compare with `survival::pyears(expect = "event")`
//! on `survexp.us` and state the deviation.
//!
//! # References
//!
//! 1. Therneau TM, Grambsch PM. Modeling Survival Data: Extending the Cox
//!    Model. Springer 2000, chapter on expected survival.
//! 2. Breslow NE, Day NE. Statistical Methods in Cancer Research, volume II.
//!    IARC 1987, chapter 3.

use epistat_core::check;
use epistat_stats::sum::compensated;

use crate::check::{check_ascending, refuse_non_finite};
use crate::error::{Error, Result};
use crate::pyears::{Dimension, Layout, PyearsOptions};
use crate::surv::SurvData;

/// One dimension of a rate table.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum RateDim {
    /// A classification that advances with follow-up, given by the cut points
    /// of its cells.
    Time {
        /// The cut points, finite, strictly increasing and at least two. There
        /// are `cuts.len() - 1` cells.
        cuts: Vec<f64>,
    },
    /// A classification fixed for a row.
    Factor {
        /// The number of levels, at least 1.
        levels: usize,
    },
}

impl RateDim {
    fn len(&self) -> usize {
        match self {
            Self::Time { cuts } => cuts.len() - 1,
            Self::Factor { levels } => *levels,
        }
    }
}

/// One dimension of a rate table given as one value per row of a long table,
/// for [`RateTable::from_long`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum LongDim<'a> {
    /// A time dimension: the start of the cell of each row, such as the age
    /// or the calendar year it holds, and the end of the last cell. The cuts
    /// are the distinct starts and the end.
    Time {
        /// The start of the cell of each row, finite.
        start: &'a [f64],
        /// The end of the last cell, finite and above every start.
        end: f64,
    },
    /// A factor: the level of each row, coded from 0. The levels are `0` to
    /// the largest code.
    Factor {
        /// The level of each row.
        level: &'a [u32],
    },
}

/// A table of reference hazards.
#[derive(Debug, Clone, PartialEq)]
pub struct RateTable {
    dims: Vec<RateDim>,
    sizes: Vec<usize>,
    rates: Vec<f64>,
    scale: f64,
    extended: bool,
}

/// Expected counts of a cohort.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Expected {
    /// The expected count of each row of the data, its cumulative reference
    /// hazard over the follow-up on the table, not weighted. 0 for a row that
    /// was not used.
    pub per_row: Vec<f64>,
    /// The expected count of each cell of the table, weighted, in the order of
    /// the table: the first dimension varies fastest.
    pub cells: Vec<f64>,
    /// The person-time in each cell, weighted, in the units of the follow-up.
    pub person_time: Vec<f64>,
    /// The total expected count, the compensated sum of the cells.
    pub total: f64,
    /// Weighted person-time off the table, in the units of the follow-up.
    pub offtable: f64,
}

/// Expected counts of cells that were summed elsewhere.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct KeyedExpected {
    /// The expected count of each cell given, 0 for a cell off the table.
    pub per_cell: Vec<f64>,
    /// The total expected count, the compensated sum of the cells.
    pub total: f64,
    /// The person-time of the cells off the table.
    pub offtable: f64,
}

impl RateTable {
    /// Builds a table from its dimensions, its hazards and the number of
    /// follow-up time units each hazard refers to. The hazards are in the order
    /// of the cells, the first dimension varying fastest.
    ///
    /// # Errors
    ///
    /// An input error for no dimension, a time dimension with fewer than two cuts or with
    /// cuts that are not finite and strictly increasing, a factor with no
    /// level, a number of hazards other than the number of cells, a hazard
    /// that is negative, NaN or infinite, or a scale that is not finite and
    /// positive.
    pub fn new(dims: Vec<RateDim>, rates: Vec<f64>, scale: f64) -> Result<Self> {
        check::non_empty("dims", dims.len())?;
        check::positive("scale", scale)?;
        let mut sizes = Vec::with_capacity(dims.len());
        let mut cells = 1_usize;
        for (d, dim) in dims.iter().enumerate() {
            match dim {
                RateDim::Time { cuts } => {
                    check::non_empty("cuts", cuts.len())?;
                    if cuts.len() < 2 {
                        return Err(Error::order(format!(
                            "dimension {d} needs at least two cut points"
                        )));
                    }
                    check_ascending(cuts, "cuts", true)?;
                }
                RateDim::Factor { levels } => {
                    if *levels == 0 {
                        return Err(Error::value(format!("dimension {d} has no levels")));
                    }
                }
            }
            cells = cells
                .checked_mul(dim.len())
                .ok_or_else(|| Error::value("the table has too many cells"))?;
            sizes.push(dim.len());
        }
        check::same_length("rates", cells, rates.len())?;
        check::weights("rates", &rates)?;
        Ok(Self {
            dims,
            sizes,
            rates,
            scale,
            extended: false,
        })
    }

    /// The same table with the first and last cells of every time dimension
    /// reaching without limit below the first cut and above the last, as the
    /// cells of R's `ratetable` do. No time is then off the table: follow-up
    /// beyond the last calendar year has the rates of that year, and follow-up
    /// beyond the oldest age those of that age.
    #[must_use]
    pub fn extend_ends(mut self) -> Self {
        self.extended = true;
        self
    }

    /// Whether the first and last cells of the time dimensions reach without
    /// limit, see [`RateTable::extend_ends`].
    #[must_use]
    pub fn ends_extended(&self) -> bool {
        self.extended
    }

    /// Builds a table from a long table with one row per cell: the value of
    /// each dimension in the row and its hazard, such as rows of age, sex,
    /// calendar year and death rate from a statistics office. Every
    /// combination of the cuts of the time dimensions and the levels of the
    /// factors must appear exactly once.
    ///
    /// # Errors
    ///
    /// An input error for no dimension or no row, a dimension of another
    /// length than the rates, a start that is not finite, an end that is not
    /// finite or not above every start, a combination that appears twice or
    /// not at all (the error names it), and the cases of [`RateTable::new`].
    pub fn from_long(dims: &[LongDim<'_>], rates: &[f64], scale: f64) -> Result<Self> {
        let n = rates.len();
        check::non_empty("dims", dims.len())?;
        check::non_empty("rates", n)?;
        let mut table_dims = Vec::with_capacity(dims.len());
        for (d, dim) in dims.iter().enumerate() {
            let (name, length) = match dim {
                LongDim::Time { start, .. } => ("start", start.len()),
                LongDim::Factor { level } => ("level", level.len()),
            };
            check::same_length(name, n, length)?;
            table_dims.push(match *dim {
                LongDim::Time { start, end } => {
                    refuse_non_finite(start, "start")?;
                    let mut cuts = start.to_vec();
                    cuts.sort_by(f64::total_cmp);
                    cuts.dedup();
                    let last = cuts[cuts.len() - 1];
                    if !(end.is_finite() && end > last) {
                        return Err(Error::value(format!(
                            "dimension {d}: the end {end} must be finite and above the last start {last}"
                        )));
                    }
                    cuts.push(end);
                    RateDim::Time { cuts }
                }
                LongDim::Factor { level } => RateDim::Factor {
                    levels: level.iter().max().map_or(0, |&m| m as usize + 1),
                },
            });
        }
        let sizes: Vec<usize> = table_dims.iter().map(RateDim::len).collect();
        let cells = sizes
            .iter()
            .try_fold(1_usize, |c, &s| c.checked_mul(s))
            .ok_or_else(|| Error::value("the table has too many cells"))?;
        let mut dense: Vec<Option<usize>> = vec![None; cells];
        let describe = |position: usize| -> String {
            let mut rest = position;
            let parts: Vec<String> = table_dims
                .iter()
                .zip(&sizes)
                .enumerate()
                .map(|(d, (dim, &size))| {
                    let k = rest % size;
                    rest /= size;
                    match dim {
                        RateDim::Time { cuts } => format!("dimension {d} at {}", cuts[k]),
                        RateDim::Factor { .. } => format!("dimension {d} level {k}"),
                    }
                })
                .collect();
            parts.join(", ")
        };
        for row in 0..n {
            let mut position = 0;
            let mut stride = 1;
            for ((dim, table_dim), &size) in dims.iter().zip(&table_dims).zip(&sizes) {
                let k = match (dim, table_dim) {
                    (LongDim::Time { start, .. }, RateDim::Time { cuts }) => {
                        cuts.partition_point(|&c| c < start[row])
                    }
                    (LongDim::Factor { level }, _) => level[row] as usize,
                    _ => unreachable!("the dimensions are built in the same order"),
                };
                position += k * stride;
                stride *= size;
            }
            if let Some(first) = dense[position] {
                return Err(Error::inconsistent(format!(
                    "rows {first} and {row} are both the cell with {}",
                    describe(position)
                )));
            }
            dense[position] = Some(row);
        }
        if let Some(position) = dense.iter().position(Option::is_none) {
            return Err(Error::missing(format!(
                "no row gives the cell with {}",
                describe(position)
            )));
        }
        let ordered = dense.iter().flatten().map(|&row| rates[row]).collect();
        Self::new(table_dims, ordered, scale)
    }

    /// The dimensions.
    #[must_use]
    pub fn dims(&self) -> &[RateDim] {
        &self.dims
    }

    /// The hazards, in the order of the cells.
    #[must_use]
    pub fn rates(&self) -> &[f64] {
        &self.rates
    }

    /// The number of follow-up time units each hazard refers to.
    #[must_use]
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// The number of cells in each dimension.
    #[must_use]
    pub fn sizes(&self) -> &[usize] {
        &self.sizes
    }

    /// The hazard at `coords`, one value per dimension: the value of a time
    /// dimension, or the level of a factor as an integer value. `None` when
    /// the length is wrong, a time value is not finite or is off the table, or
    /// a factor value is not a level.
    #[must_use]
    pub fn rate(&self, coords: &[f64]) -> Option<f64> {
        if coords.len() != self.dims.len() {
            return None;
        }
        let mut position = 0;
        let mut stride = 1;
        for ((dim, &x), &size) in self.dims.iter().zip(coords).zip(&self.sizes) {
            let cell = match dim {
                RateDim::Time { cuts } => {
                    if !x.is_finite() {
                        return None;
                    }
                    let above = cuts.partition_point(|&c| c <= x);
                    if self.extended {
                        above.clamp(1, cuts.len() - 1) - 1
                    } else if above == 0 || above == cuts.len() {
                        return None;
                    } else {
                        above - 1
                    }
                }
                RateDim::Factor { levels } => {
                    if !(x >= 0.0 && x.fract() == 0.0 && x < *levels as f64) {
                        return None;
                    }
                    #[expect(
                        clippy::cast_possible_truncation,
                        clippy::cast_sign_loss,
                        reason = "x is a non-negative integer value below the number of levels"
                    )]
                    let level = x as usize;
                    level
                }
            };
            position += cell * stride;
            stride *= size;
        }
        Some(self.rates[position])
    }

    /// The expected count of each cell given its person-time, which is
    /// `person_time * rate / scale`. The person-time is in the order of the
    /// cells and in the units of the follow-up, as a
    /// [`crate::pyears::pyears`] table on this table's dimensions gives it with
    /// a scale of 1.
    ///
    /// # Errors
    ///
    /// An input error when the length is not the number of cells or a
    /// person-time is negative or not finite.
    pub fn expected_cells(&self, person_time: &[f64]) -> Result<Vec<f64>> {
        check::same_length("person_time", self.rates.len(), person_time.len())?;
        check::weights("person_time", person_time)?;
        Ok(person_time
            .iter()
            .zip(&self.rates)
            .map(|(&y, &rate)| y * rate / self.scale)
            .collect())
    }

    /// The expected counts of cells that were summed with
    /// [`crate::lexis::split_cells`] or [`crate::lexis::cells::aggregate`],
    /// where the time scales have this table's cuts as breaks and the factors
    /// are keys, in the order of the table's dimensions. `codes[d][k]` is the
    /// level of cell `k` in dimension `d`: for a time dimension 0 is below the
    /// first cut, which is off the table, and `j + 1` is the cell `j`, and for a
    /// factor it is the level. `length` is the person-time of each cell.
    ///
    /// # Errors
    ///
    /// An input error for a number of dimensions other than the table's, slices of
    /// different length, a code outside the levels of its dimension (a time
    /// dimension has one more level than it has cuts), or a
    /// length that is negative or not finite.
    pub fn expected_keyed(&self, codes: &[&[u32]], length: &[f64]) -> Result<KeyedExpected> {
        if codes.len() != self.dims.len() {
            return Err(Error::mismatch(format!(
                "{} code columns for a table of {} dimensions",
                codes.len(),
                self.dims.len()
            )));
        }
        let n = length.len();
        check::weights("length", length)?;
        for (d, column) in codes.iter().enumerate() {
            check::same_length("codes", n, column.len())?;
            let top = match &self.dims[d] {
                RateDim::Time { cuts } => cuts.len() + 1,
                RateDim::Factor { levels } => *levels,
            };
            if let Some(k) = column.iter().position(|&c| c as usize >= top) {
                return Err(Error::out_of_range(format!(
                    "dimension {d}: the code {} of cell {k} is not below {top}",
                    column[k]
                )));
            }
        }
        let mut per_cell = Vec::with_capacity(n);
        let mut offtable = 0.0;
        for k in 0..n {
            let mut position = 0;
            let mut stride = 1;
            let mut off = false;
            for (d, dim) in self.dims.iter().enumerate() {
                let code = codes[d][k] as usize;
                let cell = match dim {
                    RateDim::Time { cuts } => {
                        if self.extended {
                            code.clamp(1, cuts.len() - 1) - 1
                        } else if code == 0 || code == cuts.len() {
                            off = true;
                            0
                        } else {
                            code - 1
                        }
                    }
                    RateDim::Factor { .. } => code,
                };
                position += cell * stride;
                stride *= self.sizes[d];
            }
            if off {
                per_cell.push(0.0);
                offtable += length[k];
            } else {
                per_cell.push(length[k] * self.rates[position] / self.scale);
            }
        }
        let total = compensated(per_cell.iter().copied());
        Ok(KeyedExpected {
            per_cell,
            total,
            offtable,
        })
    }

    /// The dimensions of the table with the values of the rows attached.
    pub(crate) fn dimensions<'a>(
        &'a self,
        origins: &'a [&'a [f64]],
        factors: &'a [&'a [u32]],
    ) -> Result<Vec<Dimension<'a>>> {
        let n_time = self
            .dims
            .iter()
            .filter(|d| matches!(d, RateDim::Time { .. }))
            .count();
        if origins.len() != n_time || factors.len() != self.dims.len() - n_time {
            return Err(Error::mismatch(format!(
                "the table has {n_time} time dimensions and {} factors, and {} origins and {} factors were given",
                self.dims.len() - n_time,
                origins.len(),
                factors.len()
            )));
        }
        let (mut next_time, mut next_factor) = (0, 0);
        Ok(self
            .dims
            .iter()
            .map(|d| match d {
                RateDim::Time { cuts } => {
                    let dim = Dimension::Time {
                        origin: origins[next_time],
                        cuts,
                    };
                    next_time += 1;
                    dim
                }
                RateDim::Factor { levels } => {
                    let dim = Dimension::Factor {
                        codes: factors[next_factor],
                        levels: *levels,
                    };
                    next_factor += 1;
                    dim
                }
            })
            .collect())
    }

    /// The expected count of the follow-up in `data`, by the walk of
    /// [`crate::pyears::pyears`]. `origins` has one slice for each time
    /// dimension of the table, in order, with the value of the dimension at
    /// time 0 of the follow-up scale for every row, such as age at the origin,
    /// and `factors` one slice of levels for each factor, in order. Rows of
    /// weight 0 are not used.
    ///
    /// # Errors
    ///
    /// An input error when the number of slices does not match the table's
    /// dimensions, a slice has the wrong length, a level is outside its
    /// factor, a value used is not finite, or the follow-up is invalid as in
    /// [`crate::pyears::pyears`].
    pub fn expected(
        &self,
        data: &SurvData<'_>,
        origins: &[&[f64]],
        factors: &[&[u32]],
    ) -> Result<Expected> {
        let dims = self.dimensions(origins, factors)?;
        let Layout {
            data,
            cells,
            mut cursor,
            ..
        } = Layout::new(data, &dims, &PyearsOptions::default(), false)?;
        if self.extended {
            cursor.open_ends();
        }
        let mut person_time = vec![0.0; cells];
        let mut per_row = vec![0.0; data.n_rows()];
        let mut offtable = 0.0;
        let weight = data.weight();
        let weight_of = |i: usize| weight.map_or(1.0, |w| w[i]);
        cursor.walk(
            &data,
            &mut |i, cell, length| match cell {
                Some(cell) => {
                    person_time[cell] += length * weight_of(i);
                    per_row[i] += length * self.rates[cell] / self.scale;
                }
                None => offtable += length * weight_of(i),
            },
            &mut |_, _| {},
        );
        let expected_cells = self.expected_cells(&person_time)?;
        let total = compensated(expected_cells.iter().copied());
        Ok(Expected {
            per_row,
            cells: expected_cells,
            person_time,
            total,
            offtable,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table() -> RateTable {
        // Age bands 0 to 10 and 10 to 100, sex of two levels, hazards per year.
        RateTable::new(
            vec![
                RateDim::Time {
                    cuts: vec![0.0, 10.0, 100.0],
                },
                RateDim::Factor { levels: 2 },
            ],
            vec![0.01, 0.02, 0.1, 0.2],
            1.0,
        )
        .unwrap()
    }

    #[test]
    fn lookup_follows_the_first_dimension_fastest() {
        let t = table();
        assert_eq!(t.rate(&[5.0, 0.0]), Some(0.01));
        assert_eq!(t.rate(&[5.0, 1.0]), Some(0.1));
        assert_eq!(t.rate(&[10.0, 0.0]), Some(0.02));
        assert_eq!(t.rate(&[99.0, 1.0]), Some(0.2));
        assert_eq!(t.rate(&[-1.0, 0.0]), None);
        assert_eq!(t.rate(&[100.0, 0.0]), None);
        assert_eq!(t.rate(&[5.0, 2.0]), None);
        assert_eq!(t.rate(&[5.0, 0.5]), None);
        assert_eq!(t.rate(&[5.0]), None);
    }

    #[test]
    fn a_row_crossing_a_cut_adds_both_pieces() {
        let t = table();
        // Entering at age 8 for 4 years: 2 years at 0.01, 2 years at 0.02.
        let stop = [4.0];
        let status = [0];
        let data = SurvData::builder(&stop, &status).build().unwrap();
        let e = t.expected(&data, &[&[8.0]], &[&[0]]).unwrap();
        assert!((e.total - (2.0 * 0.01 + 2.0 * 0.02)).abs() < 1e-15);
        assert_eq!(
            e.per_row[0].to_bits(),
            (2.0_f64 * 0.01 + 2.0 * 0.02).to_bits()
        );
        assert_eq!(e.offtable.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn time_below_the_table_is_off_it() {
        let t = table();
        let stop = [5.0];
        let status = [0];
        let data = SurvData::builder(&stop, &status).build().unwrap();
        let e = t.expected(&data, &[&[-2.0]], &[&[1]]).unwrap();
        assert!((e.offtable - 2.0).abs() < 1e-15);
        assert!((e.total - 3.0 * 0.1).abs() < 1e-15);
    }

    #[test]
    fn refuses_bad_tables() {
        let dim = || vec![RateDim::Factor { levels: 2 }];
        assert!(RateTable::new(vec![], vec![], 1.0).is_err());
        assert!(RateTable::new(dim(), vec![0.1], 1.0).is_err());
        assert!(RateTable::new(dim(), vec![0.1, f64::NAN], 1.0).is_err());
        assert!(RateTable::new(dim(), vec![0.1, -0.2], 1.0).is_err());
        assert!(RateTable::new(dim(), vec![0.1, 0.2], 0.0).is_err());
        let bad = vec![RateDim::Time {
            cuts: vec![1.0, 1.0, 2.0],
        }];
        assert!(RateTable::new(bad, vec![0.1, 0.2], 1.0).is_err());
        let one = vec![RateDim::Time { cuts: vec![1.0] }];
        assert!(RateTable::new(one, vec![], 1.0).is_err());
    }

    #[test]
    fn extended_ends_reach_without_limit() {
        let t = table().extend_ends();
        assert!(t.ends_extended());
        assert_eq!(t.rate(&[-1.0, 0.0]), Some(0.01));
        assert_eq!(t.rate(&[100.0, 1.0]), Some(0.2));
        assert_eq!(t.rate(&[1e9, 1.0]), Some(0.2));
        assert_eq!(t.rate(&[5.0, 2.0]), None);
        // From age -2 for 5 years: 2 years below the first cut and 3 inside,
        // all at the rate of the first cell.
        let stop = [5.0];
        let status = [0];
        let data = SurvData::builder(&stop, &status).build().unwrap();
        let e = t.expected(&data, &[&[-2.0]], &[&[1]]).unwrap();
        assert_eq!(e.offtable.to_bits(), 0.0_f64.to_bits());
        assert!((e.total - 5.0 * 0.1).abs() < 1e-15);
        let k = t.expected_keyed(&[&[0, 3], &[0, 1]], &[1.0, 2.0]).unwrap();
        assert_eq!(k.per_cell, vec![0.01, 2.0 * 0.2]);
        assert_eq!(k.offtable.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn extended_ends_give_the_bits_of_far_cuts() {
        // R's reach by moving the first and last cut far out, as the rates
        // guide did by hand, and the extended table walk the same pieces.
        let far = RateTable::new(
            vec![
                RateDim::Time {
                    cuts: vec![-1e6, 10.0, 1e6],
                },
                RateDim::Factor { levels: 2 },
            ],
            vec![0.01, 0.02, 0.1, 0.2],
            1.0,
        )
        .unwrap();
        let t = table().extend_ends();
        let stop = [4.0, 30.0, 200.0, 0.5];
        let start = [0.0, 2.0, 0.0, 0.0];
        let status = [0, 1, 0, 1];
        let data = SurvData::builder(&stop, &status)
            .start(&start)
            .build()
            .unwrap();
        let age = [8.0, -5.0, 60.0, 99.9];
        let sex = [0, 1, 1, 0];
        let a = t.expected(&data, &[&age], &[&sex]).unwrap();
        let b = far.expected(&data, &[&age], &[&sex]).unwrap();
        assert_eq!(a.per_row, b.per_row);
        assert_eq!(a.total.to_bits(), b.total.to_bits());
    }

    #[test]
    fn a_long_table_gives_the_dense_one() {
        // The rows of the table above in another order: age start, sex, rate.
        let age = [10.0, 0.0, 0.0, 10.0];
        let sex = [1, 0, 1, 0];
        let rates = [0.2, 0.01, 0.1, 0.02];
        let long = RateTable::from_long(
            &[
                LongDim::Time {
                    start: &age,
                    end: 100.0,
                },
                LongDim::Factor { level: &sex },
            ],
            &rates,
            1.0,
        )
        .unwrap();
        assert_eq!(long, table());
        let twice = RateTable::from_long(
            &[
                LongDim::Time {
                    start: &[0.0, 0.0],
                    end: 10.0,
                },
                LongDim::Factor { level: &[1, 1] },
            ],
            &[0.1, 0.2],
            1.0,
        )
        .unwrap_err();
        assert!(twice.to_string().contains("rows 0 and 1"), "{twice}");
        let missing = RateTable::from_long(
            &[
                LongDim::Time {
                    start: &age[..3],
                    end: 100.0,
                },
                LongDim::Factor { level: &sex[..3] },
            ],
            &rates[..3],
            1.0,
        )
        .unwrap_err();
        assert!(
            missing
                .to_string()
                .contains("dimension 0 at 10, dimension 1 level 0"),
            "{missing}"
        );
        let short_end = RateTable::from_long(
            &[LongDim::Time {
                start: &[0.0, 10.0],
                end: 10.0,
            }],
            &[0.1, 0.2],
            1.0,
        );
        assert!(short_end.is_err());
        assert!(
            RateTable::from_long(&[LongDim::Factor { level: &[0] }], &[0.1, 0.2], 1.0).is_err()
        );
    }

    #[test]
    fn keyed_cells_match_the_walk() {
        let t = table();
        let k = t
            .expected_keyed(
                &[&[0, 1, 2, 2, 3], &[0, 0, 0, 1, 0]],
                &[1.0, 2.0, 3.0, 4.0, 5.0],
            )
            .unwrap();
        assert_eq!(k.offtable.to_bits(), 6.0_f64.to_bits());
        assert_eq!(
            k.per_cell,
            vec![0.0, 2.0 * 0.01, 3.0 * 0.02, 4.0 * 0.2, 0.0]
        );
    }
}
