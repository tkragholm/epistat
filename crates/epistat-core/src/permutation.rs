//! A reordering of rows, applied to every column the same way.
//!
//! [`Panel::sort_order`](crate::Panel::sort_order) returns the order that
//! sorts person-period rows by unit and period. Each column of the table must
//! then be put in that order. [`Permutation::apply`] does it for one column
//! and refuses a column of another length, so a column taken from the wrong
//! table is an error rather than a misaligned input; [`Permutation::inverse`]
//! puts per-row results back in the caller's order.
//!
//! Position `k` of a permutation is the input row that comes `k`-th:
//!
//! ```text
//! apply(column)[k]   = column[order[k]]
//! inverse[order[k]]  = k
//! ```
//!
//! ```
//! use epistat_core::Panel;
//!
//! let unit = [1_u32, 0, 1, 0];
//! let year = [2001_i64, 2001, 2000, 2000];
//! let y = [4.0, 2.0, 3.0, 1.0];
//! let order = Panel::sort_order(&unit, &year)?;
//! let (unit, year, y) = (order.apply(&unit)?, order.apply(&year)?, order.apply(&y)?);
//! assert_eq!(y, [1.0, 2.0, 3.0, 4.0]);
//! let panel = Panel::new(&unit, &year)?;
//! assert_eq!(panel.n_units(), 2);
//! // Results per sorted row go back to the input rows.
//! assert_eq!(order.inverse().apply(&y)?, [4.0, 2.0, 3.0, 1.0]);
//! # Ok::<(), epistat_core::InputError>(())
//! ```

use core::ops::Deref;

use crate::check::{self, InputError};

/// A permutation of the rows `0..n`: position `k` holds the row that comes
/// `k`-th.
///
/// It derefs to `[u32]`, so it reads as the slice of row numbers it holds.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "Vec<u32>", into = "Vec<u32>"))]
pub struct Permutation(Vec<u32>);

impl Permutation {
    /// The permutation that takes row `order[k]` to position `k`.
    ///
    /// # Errors
    ///
    /// [`InputError::TooLarge`] for more rows than a `u32` row number holds,
    /// [`InputError::Domain`] for a row number of `order.len()` or more, and
    /// [`InputError::Duplicate`] at the first position that repeats a row.
    pub fn new(order: Vec<u32>) -> Result<Self, InputError> {
        fits_rows(order.len())?;
        let mut seen = vec![false; order.len()];
        for (position, &row) in order.iter().enumerate() {
            let Some(slot) = seen.get_mut(row as usize) else {
                return Err(InputError::Domain {
                    what: "permutation",
                    value: f64::from(row),
                    domain: "the row numbers 0 to n - 1",
                    row: Some(position),
                });
            };
            if *slot {
                return Err(InputError::Duplicate {
                    what: "permutation",
                    row: position,
                });
            }
            *slot = true;
        }
        Ok(Self(order))
    }

    /// The order of the rows of `unit` and `period` sorted by unit, then
    /// period, ties in input order. [`Panel::sort_order`](crate::Panel::sort_order)
    /// is this function.
    pub(crate) fn sorting(unit: &[u32], period: &[i64]) -> Result<Self, InputError> {
        check::same_length("period", unit.len(), period.len())?;
        fits_rows(unit.len())?;
        let mut order: Vec<u32> = (0..unit.len()).map(row_number).collect();
        order.sort_by_key(|&row| (unit[row as usize], period[row as usize]));
        Ok(Self(order))
    }

    /// The identity on `n` rows, which leaves every column as it is.
    ///
    /// # Errors
    ///
    /// [`InputError::TooLarge`] for more rows than a `u32` row number holds.
    pub fn identity(n: usize) -> Result<Self, InputError> {
        fits_rows(n)?;
        Ok(Self((0..n).map(row_number).collect()))
    }

    /// Whether every row stays where it is, so the columns need no copy.
    #[must_use]
    pub fn is_identity(&self) -> bool {
        self.0.iter().enumerate().all(|(k, &row)| row as usize == k)
    }

    /// The permutation that undoes this one: applied to columns in sorted
    /// order, it gives them back in the input order.
    #[must_use]
    pub fn inverse(&self) -> Self {
        let mut inverse = vec![0; self.0.len()];
        for (k, &row) in self.0.iter().enumerate() {
            inverse[row as usize] = row_number(k);
        }
        Self(inverse)
    }

    /// `column` in this order: entry `k` of the result is
    /// `column[self[k]]`.
    ///
    /// # Errors
    ///
    /// [`InputError::Length`] when `column` does not have one entry per row.
    pub fn apply<T: Clone>(&self, column: &[T]) -> Result<Vec<T>, InputError> {
        let mut out = Vec::new();
        self.apply_into(column, &mut out)?;
        Ok(out)
    }

    /// As [`apply`](Self::apply), into `out`, which is cleared first, so a
    /// buffer can serve several columns.
    ///
    /// # Errors
    ///
    /// [`InputError::Length`] when `column` does not have one entry per row.
    pub fn apply_into<T: Clone>(&self, column: &[T], out: &mut Vec<T>) -> Result<(), InputError> {
        check::same_length("column", self.0.len(), column.len())?;
        out.clear();
        out.extend(self.0.iter().map(|&row| column[row as usize].clone()));
        Ok(())
    }

    /// The row numbers, position `k` holding the row that comes `k`-th.
    #[must_use]
    pub fn as_slice(&self) -> &[u32] {
        &self.0
    }

    /// The row numbers, consuming the permutation.
    #[must_use]
    pub fn into_vec(self) -> Vec<u32> {
        self.0
    }
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "callers check that the number of rows fits a u32 row number"
)]
const fn row_number(row: usize) -> u32 {
    row as u32
}

fn fits_rows(n: usize) -> Result<(), InputError> {
    let limit = u64::from(u32::MAX) + 1;
    let found = u64::try_from(n).unwrap_or(u64::MAX);
    if found > limit {
        return Err(InputError::TooLarge {
            what: "rows",
            limit,
            found,
        });
    }
    Ok(())
}

impl Deref for Permutation {
    type Target = [u32];

    fn deref(&self) -> &[u32] {
        &self.0
    }
}

impl AsRef<[u32]> for Permutation {
    fn as_ref(&self) -> &[u32] {
        &self.0
    }
}

impl TryFrom<Vec<u32>> for Permutation {
    type Error = InputError;

    fn try_from(order: Vec<u32>) -> Result<Self, InputError> {
        Self::new(order)
    }
}

impl From<Permutation> for Vec<u32> {
    fn from(permutation: Permutation) -> Self {
        permutation.0
    }
}

impl IntoIterator for Permutation {
    type Item = u32;
    type IntoIter = std::vec::IntoIter<u32>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a Permutation {
    type Item = &'a u32;
    type IntoIter = core::slice::Iter<'a, u32>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl PartialEq<[u32]> for Permutation {
    fn eq(&self, other: &[u32]) -> bool {
        self.0 == other
    }
}

impl<const N: usize> PartialEq<[u32; N]> for Permutation {
    fn eq(&self, other: &[u32; N]) -> bool {
        self.0 == other
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Classify;

    #[test]
    fn a_permutation_applies_inverts_and_knows_the_identity() {
        let order = Permutation::new(vec![2, 0, 1]).unwrap();
        assert_eq!(order.apply(&['a', 'b', 'c']).unwrap(), ['c', 'a', 'b']);
        let inverse = order.inverse();
        assert_eq!(inverse, [1, 2, 0]);
        let back = inverse.apply(&order.apply(&[10, 20, 30]).unwrap()).unwrap();
        assert_eq!(back, [10, 20, 30]);
        assert!(!order.is_identity());
        assert!(Permutation::identity(3).unwrap().is_identity());
        assert_eq!(order.inverse().inverse(), order);
        let mut out = vec![99.0; 7];
        order.apply_into(&[1.0, 2.0, 3.0], &mut out).unwrap();
        assert_eq!(out, [3.0, 1.0, 2.0]);
        assert_eq!(order.len(), 3);
        assert_eq!(order.iter().copied().max(), Some(2));
        assert_eq!(order.clone().into_vec(), vec![2, 0, 1]);
    }

    #[test]
    fn a_column_of_another_length_is_refused() {
        let order = Permutation::identity(3).unwrap();
        assert_eq!(
            order.apply(&[1, 2]),
            Err(InputError::Length {
                what: "column",
                expected: 3,
                found: 2
            })
        );
    }

    #[test]
    fn a_list_that_is_not_a_permutation_is_refused() {
        assert_eq!(
            Permutation::new(vec![0, 2, 0]),
            Err(InputError::Duplicate {
                what: "permutation",
                row: 2
            })
        );
        assert_eq!(
            Permutation::new(vec![0, 3, 1]).unwrap_err().code(),
            "input.domain"
        );
        assert!(Permutation::new(Vec::new()).unwrap().is_identity());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_writes_the_rows_and_validates() {
        let order = Permutation::new(vec![1, 0]).unwrap();
        assert_eq!(serde_json::to_string(&order).unwrap(), "[1,0]");
        assert_eq!(serde_json::from_str::<Permutation>("[1,0]").unwrap(), order);
        assert!(serde_json::from_str::<Permutation>("[1,1]").is_err());
    }
}
