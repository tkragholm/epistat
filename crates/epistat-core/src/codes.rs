//! Dense `u32` codes for a column of labels, and the labels back.
//!
//! Estimators take units, clusters and strata as `u32` codes. A caller holds
//! them as labels: personal ids, county numbers, strings. [`Codes`] turns a
//! label column into codes once and keeps the distinct labels, so a result
//! indexed by code can be reported by label.
//!
//! # Algorithm
//!
//! The distinct labels are sorted, and a row's code is the position of its
//! label in that sorted list:
//!
//! ```text
//! labels   = sort(distinct(column))
//! code(i)  = position of column(i) in labels
//! ```
//!
//! found by binary search, `O(n log k)` for `n` rows and `k` distinct labels.
//! Sorted order makes the codes depend on the set of labels alone, not on the
//! row order, so a sort by code is a sort by label: for string ids the order
//! of `str::cmp`, for integer ids the numeric order. An estimator that orders
//! its units by code therefore orders them as a sort of the caller's ids
//! would.

use crate::check::InputError;

/// A label column as dense codes `0..k`, with the `k` distinct labels in
/// sorted order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Codes<L> {
    codes: Vec<u32>,
    labels: Vec<L>,
}

impl<L: Ord + Clone> Codes<L> {
    /// The codes of `column`, in the sorted order of its distinct labels.
    ///
    /// # Errors
    ///
    /// [`InputError::TooLarge`] for more distinct labels than `u32` codes can
    /// number, `2^32`.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "positions are below the count of labels, checked to fit a u32 code"
    )]
    pub fn from_labels(column: &[L]) -> Result<Self, InputError> {
        let mut labels = column.to_vec();
        labels.sort_unstable();
        labels.dedup();
        let limit = u64::from(u32::MAX) + 1;
        if labels.len() as u64 > limit {
            return Err(InputError::TooLarge {
                what: "labels",
                limit,
                found: labels.len() as u64,
            });
        }
        let codes = column
            .iter()
            .map(|label| {
                // Every label of the column is among the distinct labels, so
                // the search finds it.
                labels
                    .binary_search(label)
                    .unwrap_or_else(|position| position) as u32
            })
            .collect();
        Ok(Self { codes, labels })
    }
}

impl<L: Ord> Codes<L> {
    /// The code of `label`, `None` when the column did not contain it.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "positions are below the count of labels, checked to fit a u32 code"
    )]
    pub fn code_of(&self, label: &L) -> Option<u32> {
        self.labels
            .binary_search(label)
            .ok()
            .map(|position| position as u32)
    }
}

impl<L> Codes<L> {
    /// The code of each row, in row order.
    #[must_use]
    pub fn codes(&self) -> &[u32] {
        &self.codes
    }

    /// The distinct labels, sorted; the label of code `c` is `labels()[c]`.
    #[must_use]
    pub fn labels(&self) -> &[L] {
        &self.labels
    }

    /// The label of code `code`, `None` for a code that does not occur.
    #[must_use]
    pub fn label(&self, code: u32) -> Option<&L> {
        self.labels.get(usize::try_from(code).ok()?)
    }

    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Whether the column had no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// The number of distinct labels, one more than the largest code.
    #[must_use]
    pub fn n_codes(&self) -> usize {
        self.labels.len()
    }

    /// The codes and the sorted labels, consuming the coding.
    #[must_use]
    pub fn into_parts(self) -> (Vec<u32>, Vec<L>) {
        (self.codes, self.labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_follow_the_sorted_order_of_the_labels() {
        let column = ["dk-17", "dk-03", "dk-17", "dk-10"];
        let codes = Codes::from_labels(&column).unwrap();
        assert_eq!(codes.labels(), &["dk-03", "dk-10", "dk-17"]);
        assert_eq!(codes.codes(), &[2, 0, 2, 1]);
        assert_eq!(codes.label(1), Some(&"dk-10"));
        assert_eq!(codes.label(3), None);
        assert_eq!(codes.code_of(&"dk-17"), Some(2));
        assert_eq!(codes.code_of(&"dk-99"), None);
        assert_eq!((codes.len(), codes.n_codes()), (4, 3));
        assert!(!codes.is_empty());
    }

    #[test]
    fn integer_labels_code_in_numeric_order_and_row_order_does_not_matter() {
        let a = Codes::from_labels(&[905_i64, -3, 17, 905]).unwrap();
        let b = Codes::from_labels(&[17_i64, 905, -3]).unwrap();
        assert_eq!(a.labels(), b.labels());
        assert_eq!(a.codes(), &[2, 0, 1, 2]);
        assert_eq!(b.codes(), &[1, 2, 0]);
        let (codes, labels) = a.into_parts();
        let back: Vec<i64> = codes.iter().map(|&c| labels[c as usize]).collect();
        assert_eq!(back, [905, -3, 17, 905]);
    }

    #[test]
    fn owned_strings_and_an_empty_column() {
        let column: Vec<String> = ["b", "a"].iter().map(|s| (*s).to_owned()).collect();
        let codes = Codes::from_labels(&column).unwrap();
        assert_eq!(codes.codes(), &[1, 0]);
        let empty = Codes::<u64>::from_labels(&[]).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.n_codes(), 0);
    }
}
