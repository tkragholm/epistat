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
//! [`crate::pyears::Dimension::Time`]. A value below the first cut or at or
//! above the last is off the table. Follow-up off the table has no expected
//! count and is added to [`Expected::offtable`], as
//! [`crate::pyears::Pyears::offtable`] does, so the caller sees it. A
//! `ratetable` of R instead uses the first or last cell for a value outside the
//! cut points, and a table read from R can be given the same reach by moving
//! its first and last cut far out. A factor has `levels` cells, coded
//! `0..levels`.
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

use epistat_stats::sum::compensated;

use crate::error::{Error, Result};
use crate::pyears::{Dimension, Layout, PyearsOptions};
use crate::surv::SurvData;

/// One dimension of a rate table.
#[derive(Debug, Clone, PartialEq)]
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

/// A table of reference hazards.
#[derive(Debug, Clone, PartialEq)]
pub struct RateTable {
    dims: Vec<RateDim>,
    sizes: Vec<usize>,
    rates: Vec<f64>,
    scale: f64,
}

/// Expected counts of a cohort.
#[derive(Debug, Clone, PartialEq)]
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
    /// [`Error::Input`] for no dimension, a time dimension with fewer than two cuts or with
    /// cuts that are not finite and strictly increasing, a factor with no
    /// level, a number of hazards other than the number of cells, a hazard
    /// that is negative, NaN or infinite, or a scale that is not finite and
    /// positive.
    pub fn new(dims: Vec<RateDim>, rates: Vec<f64>, scale: f64) -> Result<Self> {
        if dims.is_empty() {
            return Err(Error::input("a rate table needs at least one dimension"));
        }
        if !(scale.is_finite() && scale > 0.0) {
            return Err(Error::input("scale must be finite and positive"));
        }
        let mut sizes = Vec::with_capacity(dims.len());
        let mut cells = 1_usize;
        for (d, dim) in dims.iter().enumerate() {
            match dim {
                RateDim::Time { cuts } => {
                    if cuts.len() < 2 {
                        return Err(Error::input(format!(
                            "dimension {d} needs at least two cut points"
                        )));
                    }
                    if cuts.iter().any(|c| !c.is_finite())
                        || cuts
                            .windows(2)
                            .any(|w| w[0].partial_cmp(&w[1]) != Some(std::cmp::Ordering::Less))
                    {
                        return Err(Error::input(format!(
                            "dimension {d}: the cut points must be finite and strictly increasing"
                        )));
                    }
                }
                RateDim::Factor { levels } => {
                    if *levels == 0 {
                        return Err(Error::input(format!("dimension {d} has no levels")));
                    }
                }
            }
            cells = cells
                .checked_mul(dim.len())
                .ok_or_else(|| Error::input("the table has too many cells"))?;
            sizes.push(dim.len());
        }
        if rates.len() != cells {
            return Err(Error::input(format!(
                "{} rates for a table of {cells} cells",
                rates.len()
            )));
        }
        if let Some(i) = rates.iter().position(|r| !(r.is_finite() && *r >= 0.0)) {
            return Err(Error::input(format!(
                "the rate of cell {i} is not finite and non-negative"
            )));
        }
        Ok(Self {
            dims,
            sizes,
            rates,
            scale,
        })
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
                    if above == 0 || above == cuts.len() {
                        return None;
                    }
                    above - 1
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
    /// [`Error::Input`] when the length is not the number of cells or a
    /// person-time is negative or not finite.
    pub fn expected_cells(&self, person_time: &[f64]) -> Result<Vec<f64>> {
        if person_time.len() != self.rates.len() {
            return Err(Error::input(format!(
                "{} person-times for a table of {} cells",
                person_time.len(),
                self.rates.len()
            )));
        }
        if let Some(i) = person_time
            .iter()
            .position(|v| !(v.is_finite() && *v >= 0.0))
        {
            return Err(Error::input(format!(
                "the person-time of cell {i} is not finite and non-negative"
            )));
        }
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
    /// [`Error::Input`] for a number of dimensions other than the table's, slices of
    /// different length, a code outside the levels of its dimension (a time
    /// dimension has one more level than it has cuts), or a
    /// length that is negative or not finite.
    pub fn expected_keyed(&self, codes: &[&[u32]], length: &[f64]) -> Result<KeyedExpected> {
        if codes.len() != self.dims.len() {
            return Err(Error::input(format!(
                "{} code columns for a table of {} dimensions",
                codes.len(),
                self.dims.len()
            )));
        }
        let n = length.len();
        if let Some(i) = length.iter().position(|v| !(v.is_finite() && *v >= 0.0)) {
            return Err(Error::input(format!(
                "the length of cell {i} is not finite and non-negative"
            )));
        }
        for (d, column) in codes.iter().enumerate() {
            if column.len() != n {
                return Err(Error::input(format!(
                    "dimension {d} has {} codes against {n} cells",
                    column.len()
                )));
            }
            let top = match &self.dims[d] {
                RateDim::Time { cuts } => cuts.len() + 1,
                RateDim::Factor { levels } => *levels,
            };
            if let Some(k) = column.iter().position(|&c| c as usize >= top) {
                return Err(Error::input(format!(
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
                        if code == 0 || code == cuts.len() {
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

    /// The expected count of the follow-up in `data`, by the walk of
    /// [`crate::pyears::pyears`]. `origins` has one slice for each time
    /// dimension of the table, in order, with the value of the dimension at
    /// time 0 of the follow-up scale for every row, such as age at the origin,
    /// and `factors` one slice of levels for each factor, in order. Rows of
    /// weight 0 are not used.
    ///
    /// # Errors
    ///
    /// [`Error::Input`] when the number of slices does not match the table's
    /// dimensions, a slice has the wrong length, a level is outside its
    /// factor, a value used is not finite, or the follow-up is invalid as in
    /// [`crate::pyears::pyears`].
    pub fn expected(
        &self,
        data: &SurvData<'_>,
        origins: &[&[f64]],
        factors: &[&[u32]],
    ) -> Result<Expected> {
        let n_time = self
            .dims
            .iter()
            .filter(|d| matches!(d, RateDim::Time { .. }))
            .count();
        if origins.len() != n_time || factors.len() != self.dims.len() - n_time {
            return Err(Error::input(format!(
                "the table has {n_time} time dimensions and {} factors, and {} origins and {} factors were given",
                self.dims.len() - n_time,
                origins.len(),
                factors.len()
            )));
        }
        let (mut next_time, mut next_factor) = (0, 0);
        let dims: Vec<Dimension<'_>> = self
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
            .collect();
        let Layout {
            data,
            cells,
            mut cursor,
            ..
        } = Layout::new(data, &dims, &PyearsOptions::default(), false)?;
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
        )?;
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
        assert_eq!(e.offtable, 0.0);
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
    fn keyed_cells_match_the_walk() {
        let t = table();
        let k = t
            .expected_keyed(
                &[&[0, 1, 2, 2, 3], &[0, 0, 0, 1, 0]],
                &[1.0, 2.0, 3.0, 4.0, 5.0],
            )
            .unwrap();
        assert_eq!(k.offtable, 6.0);
        assert_eq!(
            k.per_cell,
            vec![0.0, 2.0 * 0.01, 3.0 * 0.02, 4.0 * 0.2, 0.0]
        );
    }
}
