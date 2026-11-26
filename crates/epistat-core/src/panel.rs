//! The validated key of person-period data: a unit code and a period per
//! row, sorted, each pair once.
//!
//! Panel data, target-trial expansions, clone-censor-weight designs and
//! sequential trials all hold one row per unit and period. [`Panel`] is the
//! key they share, so the same two columns pass from crate to crate and are
//! checked once. Each crate's own data view adds its columns to the key:
//! outcomes, treatment, weights, clusters.
//!
//! The rows are sorted by unit and then by period, with no repeated pair:
//!
//! ```text
//! (unit[i], period[i]) < (unit[i + 1], period[i + 1])   for every i,
//! ```
//!
//! in the lexicographic order. Units are `u32` codes, as
//! [`Codes`](crate::Codes) gives them, and periods are `i64` (years, months
//! or days). Periods need not be consecutive. [`Panel::sort_order`] gives the
//! permutation that sorts unsorted input.

use core::ops::Range;

use crate::check::{self, InputError};

/// Unit codes and periods, sorted by unit then period, no pair repeated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Panel<'a> {
    unit: &'a [u32],
    period: &'a [i64],
}

impl<'a> Panel<'a> {
    /// The key with the columns `unit` and `period`.
    ///
    /// # Errors
    ///
    /// [`InputError::Length`] when the columns differ in length,
    /// [`InputError::Empty`] for no rows, [`InputError::Unsorted`] at the
    /// first row out of order and [`InputError::Duplicate`] at the first row
    /// that repeats the unit and period of the row above it.
    pub fn new(unit: &'a [u32], period: &'a [i64]) -> Result<Self, InputError> {
        check::same_length("period", unit.len(), period.len())?;
        check::non_empty("unit", unit.len())?;
        for row in 1..unit.len() {
            match (unit[row - 1], period[row - 1]).cmp(&(unit[row], period[row])) {
                core::cmp::Ordering::Less => {}
                core::cmp::Ordering::Equal => {
                    return Err(InputError::Duplicate { what: "panel", row });
                }
                core::cmp::Ordering::Greater => {
                    return Err(InputError::Unsorted { what: "panel", row });
                }
            }
        }
        Ok(Self { unit, period })
    }

    /// The rows of `unit` and `period` in sorted order: position `k` of the
    /// result is the row that comes `k`-th. Rows with equal unit and period
    /// keep their input order, which [`Panel::new`] then refuses.
    ///
    /// # Errors
    ///
    /// [`InputError::Length`] when the columns differ in length and
    /// [`InputError::TooLarge`] for more rows than a `u32` row number holds.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the number of rows was checked to fit a u32"
    )]
    pub fn sort_order(unit: &[u32], period: &[i64]) -> Result<Vec<u32>, InputError> {
        check::same_length("period", unit.len(), period.len())?;
        let limit = u64::from(u32::MAX) + 1;
        if unit.len() as u64 > limit {
            return Err(InputError::TooLarge {
                what: "unit",
                limit,
                found: unit.len() as u64,
            });
        }
        let mut order: Vec<u32> = (0..unit.len()).map(|row| row as u32).collect();
        order.sort_by_key(|&row| (unit[row as usize], period[row as usize]));
        Ok(order)
    }

    /// The unit code of each row.
    #[must_use]
    pub const fn unit(&self) -> &'a [u32] {
        self.unit
    }

    /// The period of each row.
    #[must_use]
    pub const fn period(&self) -> &'a [i64] {
        self.period
    }

    /// The number of rows.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.unit.len()
    }

    /// Whether the key has no rows, which [`Panel::new`] refuses, so always
    /// false.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.unit.is_empty()
    }

    /// Each unit's code and the range of its rows, in unit order.
    pub fn units(&self) -> impl Iterator<Item = (u32, Range<usize>)> + 'a {
        let unit = self.unit;
        let mut start = 0;
        core::iter::from_fn(move || {
            let &code = unit.get(start)?;
            let end = start + unit[start..].partition_point(|&u| u == code);
            let range = start..end;
            start = end;
            Some((code, range))
        })
    }

    /// The number of distinct units.
    #[must_use]
    pub fn n_units(&self) -> usize {
        1 + self
            .unit
            .windows(2)
            .filter(|pair| pair[0] != pair[1])
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Classify;

    #[test]
    fn a_sorted_key_is_accepted_and_read_by_unit() {
        let unit = [0, 0, 0, 3, 3, 7];
        let period = [2000, 2001, 2004, 1999, 2000, 2010];
        let panel = Panel::new(&unit, &period).unwrap();
        assert_eq!(panel.len(), 6);
        assert!(!panel.is_empty());
        assert_eq!(panel.n_units(), 3);
        let runs: Vec<(u32, Range<usize>)> = panel.units().collect();
        assert_eq!(runs, vec![(0, 0..3), (3, 3..5), (7, 5..6)]);
        assert_eq!(panel.unit(), &unit);
        assert_eq!(panel.period(), &period);
    }

    #[test]
    fn unsorted_repeated_or_mismatched_keys_are_refused() {
        assert_eq!(
            Panel::new(&[0, 0], &[2001, 2000]),
            Err(InputError::Unsorted {
                what: "panel",
                row: 1
            })
        );
        assert_eq!(
            Panel::new(&[1, 0], &[2000, 2001]),
            Err(InputError::Unsorted {
                what: "panel",
                row: 1
            })
        );
        assert_eq!(
            Panel::new(&[0, 1, 1], &[5, 5, 5]),
            Err(InputError::Duplicate {
                what: "panel",
                row: 2
            })
        );
        assert_eq!(
            Panel::new(&[], &[]),
            Err(InputError::Empty { what: "unit" })
        );
        assert_eq!(
            Panel::new(&[0], &[1, 2]).unwrap_err().code(),
            "input.length"
        );
    }

    #[test]
    fn the_sort_order_sorts_and_keeps_ties_in_input_order() {
        let unit = [3, 0, 3, 0, 0];
        let period = [2000, 2001, 1999, 2001, 1990];
        let order = Panel::sort_order(&unit, &period).unwrap();
        assert_eq!(order, vec![4, 1, 3, 2, 0]);
        let sorted_unit: Vec<u32> = order.iter().map(|&r| unit[r as usize]).collect();
        let sorted_period: Vec<i64> = order.iter().map(|&r| period[r as usize]).collect();
        // Rows 1 and 3 repeat a pair, so the sorted key is still refused.
        assert_eq!(
            Panel::new(&sorted_unit, &sorted_period),
            Err(InputError::Duplicate {
                what: "panel",
                row: 2
            })
        );
        assert!(Panel::sort_order(&[0], &[]).is_err());
    }
}
