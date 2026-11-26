//! The input checks every estimator shares, and the error they return.
//!
//! An argument of one kind gets one check and one message in every crate: a
//! column of the wrong length is [`InputError::Length`] whether it reaches a
//! Cox model or a difference-in-differences estimator. Each crate's error enum
//! wraps [`InputError`] in an `Input` variant, so a caller matches on the same
//! variants everywhere and a binding reads one set of codes.
//!
//! `what` names the argument as the function's documentation does, such as
//! `"weights"` or `"period"`. Row numbers count from 0 in the caller's input.
//!
//! | check | refuses | error |
//! |---|---|---|
//! | [`same_length`] | a length other than the expected one | `Length` |
//! | [`non_empty`] | no rows | `Empty` |
//! | [`finite`] | NaN or an infinity | `NotFinite` |
//! | [`probability`] | a value outside `[0, 1]` | `Domain` |
//! | [`sorted_by`] | a row before its predecessor | `Unsorted` |
//! | [`strictly_sorted_by`] | that, or a row equal to its predecessor | `Unsorted`, `Duplicate` |
//! | [`csr_offsets`] | offsets that are not a layout of the rows | `Empty`, `Domain`, `Unsorted`, `Length` |
//! | [`constant_within`] | a value that changes within a unit | `NotConstant` |
//!
//! The checks are written for each argument once, where it enters, and an
//! estimator does not repeat them on input a validated view already holds.

use core::cmp::Ordering;
use core::fmt;
use std::collections::HashMap;
use std::collections::hash_map::Entry;

use crate::error::{Classify, ErrorKind};

/// Input an estimator cannot use.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq)]
pub enum InputError {
    /// An argument has `found` rows or entries where `expected` were needed.
    Length {
        /// The argument.
        what: &'static str,
        /// The length it must have.
        expected: usize,
        /// The length it has.
        found: usize,
    },
    /// An argument has no rows where at least one is needed.
    Empty {
        /// The argument.
        what: &'static str,
    },
    /// An argument holds NaN or an infinity at `row`.
    NotFinite {
        /// The argument.
        what: &'static str,
        /// The first such row.
        row: usize,
    },
    /// An argument's value lies outside the set it must lie in.
    Domain {
        /// The argument.
        what: &'static str,
        /// The value given.
        value: f64,
        /// The set the value must lie in, as text, such as `"(0, 1)"`.
        domain: &'static str,
    },
    /// Row `row` of an argument sorts before row `row - 1`.
    Unsorted {
        /// The argument.
        what: &'static str,
        /// The first such row.
        row: usize,
    },
    /// Row `row` of an argument repeats row `row - 1` where the rows must be
    /// distinct.
    Duplicate {
        /// The argument.
        what: &'static str,
        /// The first such row.
        row: usize,
    },
    /// An argument that must be constant within a unit takes two values in
    /// unit `unit`.
    NotConstant {
        /// The argument.
        what: &'static str,
        /// The unit, as its code.
        unit: u32,
    },
    /// An argument has more entries than its index type can number.
    TooLarge {
        /// The argument.
        what: &'static str,
        /// The largest count allowed.
        limit: u64,
        /// The count given.
        found: u64,
    },
}

impl fmt::Display for InputError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Length {
                what,
                expected,
                found,
            } => write!(f, "{what} has length {found}, expected {expected}"),
            Self::Empty { what } => write!(f, "{what} is empty"),
            Self::NotFinite { what, row } => {
                write!(f, "{what} is not finite at row {row}")
            }
            Self::Domain {
                what,
                value,
                domain,
            } => write!(f, "{what} is {value}, outside {domain}"),
            Self::Unsorted { what, row } => {
                write!(
                    f,
                    "{what} is not sorted: row {row} sorts before the row above it"
                )
            }
            Self::Duplicate { what, row } => {
                write!(f, "{what} repeats at row {row} the row above it")
            }
            Self::NotConstant { what, unit } => {
                write!(f, "{what} is not constant within unit {unit}")
            }
            Self::TooLarge { what, limit, found } => {
                write!(
                    f,
                    "{what} has {found} entries, more than the {limit} allowed"
                )
            }
        }
    }
}

impl std::error::Error for InputError {}

impl Classify for InputError {
    fn kind(&self) -> ErrorKind {
        ErrorKind::Input
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Length { .. } => "input.length",
            Self::Empty { .. } => "input.empty",
            Self::NotFinite { .. } => "input.not_finite",
            Self::Domain { .. } => "input.domain",
            Self::Unsorted { .. } => "input.unsorted",
            Self::Duplicate { .. } => "input.duplicate",
            Self::NotConstant { .. } => "input.not_constant",
            Self::TooLarge { .. } => "input.too_large",
        }
    }
}

/// `found` equals `expected`.
///
/// # Errors
///
/// [`InputError::Length`] otherwise.
pub const fn same_length(
    what: &'static str,
    expected: usize,
    found: usize,
) -> Result<(), InputError> {
    if expected == found {
        Ok(())
    } else {
        Err(InputError::Length {
            what,
            expected,
            found,
        })
    }
}

/// `len` is at least 1.
///
/// # Errors
///
/// [`InputError::Empty`] for 0.
pub const fn non_empty(what: &'static str, len: usize) -> Result<(), InputError> {
    if len == 0 {
        Err(InputError::Empty { what })
    } else {
        Ok(())
    }
}

/// Every value is finite.
///
/// # Errors
///
/// [`InputError::NotFinite`] at the first NaN or infinity.
pub fn finite(what: &'static str, values: &[f64]) -> Result<(), InputError> {
    match values.iter().position(|value| !value.is_finite()) {
        Some(row) => Err(InputError::NotFinite { what, row }),
        None => Ok(()),
    }
}

/// `value` is a probability, in `[0, 1]`.
///
/// # Errors
///
/// [`InputError::Domain`] for a value outside `[0, 1]` or NaN.
pub fn probability(what: &'static str, value: f64) -> Result<(), InputError> {
    if (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(InputError::Domain {
            what,
            value,
            domain: "[0, 1]",
        })
    }
}

/// The values are in the order `compare` defines, equal neighbours allowed.
///
/// # Errors
///
/// [`InputError::Unsorted`] at the first row that compares less than the row
/// before it.
pub fn sorted_by<T>(
    what: &'static str,
    values: &[T],
    mut compare: impl FnMut(&T, &T) -> Ordering,
) -> Result<(), InputError> {
    for (row, pair) in values.windows(2).enumerate() {
        if compare(&pair[0], &pair[1]) == Ordering::Greater {
            return Err(InputError::Unsorted { what, row: row + 1 });
        }
    }
    Ok(())
}

/// The values are in the order `compare` defines, with no two neighbours
/// equal.
///
/// # Errors
///
/// [`InputError::Unsorted`] at the first row that compares less than the row
/// before it, or [`InputError::Duplicate`] at the first that compares equal,
/// whichever comes first.
pub fn strictly_sorted_by<T>(
    what: &'static str,
    values: &[T],
    mut compare: impl FnMut(&T, &T) -> Ordering,
) -> Result<(), InputError> {
    for (row, pair) in values.windows(2).enumerate() {
        match compare(&pair[0], &pair[1]) {
            Ordering::Less => {}
            Ordering::Equal => return Err(InputError::Duplicate { what, row: row + 1 }),
            Ordering::Greater => return Err(InputError::Unsorted { what, row: row + 1 }),
        }
    }
    Ok(())
}

/// `offsets` lays out `rows` rows in groups, compressed-sparse-row style:
/// group `k` holds rows `offsets[k]..offsets[k + 1]`.
///
/// The offsets start at 0, end at `rows` and do not decrease, so there are
/// `offsets.len() - 1` groups, possibly empty ones.
///
/// # Errors
///
/// [`InputError::Empty`] for no offsets, [`InputError::Domain`] for a first
/// offset other than 0, [`InputError::Unsorted`] at the first offset below
/// its predecessor and [`InputError::Length`] for a last offset other than
/// `rows`.
#[expect(
    clippy::cast_precision_loss,
    reason = "the offset is reported, not computed with"
)]
pub fn csr_offsets(what: &'static str, offsets: &[usize], rows: usize) -> Result<(), InputError> {
    let (Some(&first), Some(&last)) = (offsets.first(), offsets.last()) else {
        return Err(InputError::Empty { what });
    };
    if first != 0 {
        return Err(InputError::Domain {
            what,
            value: first as f64,
            domain: "a first offset of 0",
        });
    }
    sorted_by(what, offsets, usize::cmp)?;
    same_length(what, rows, last)
}

/// `value` takes one value within each unit: rows with the same `unit` code
/// have equal values.
///
/// The rows need not be sorted. The check visits them in order and reports
/// the unit of the first row that disagrees with an earlier row of its unit.
///
/// # Errors
///
/// [`InputError::Length`] when `value` and `unit` differ in length, and
/// [`InputError::NotConstant`] for a unit with two values.
pub fn constant_within<T: PartialEq + Copy>(
    what: &'static str,
    unit: &[u32],
    value: &[T],
) -> Result<(), InputError> {
    same_length(what, unit.len(), value.len())?;
    let mut seen: HashMap<u32, T> = HashMap::new();
    for (&code, &v) in unit.iter().zip(value) {
        match seen.entry(code) {
            Entry::Vacant(slot) => {
                slot.insert(v);
            }
            Entry::Occupied(slot) => {
                if *slot.get() != v {
                    return Err(InputError::NotConstant { what, unit: code });
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lengths_and_emptiness() {
        assert_eq!(same_length("weights", 3, 3), Ok(()));
        let error = same_length("weights", 3, 2).unwrap_err();
        assert_eq!(
            error,
            InputError::Length {
                what: "weights",
                expected: 3,
                found: 2
            }
        );
        assert_eq!(error.to_string(), "weights has length 2, expected 3");
        assert_eq!(error.code(), "input.length");
        assert_eq!(error.kind(), ErrorKind::Input);
        assert_eq!(
            non_empty("time", 0),
            Err(InputError::Empty { what: "time" })
        );
        assert_eq!(non_empty("time", 1), Ok(()));
    }

    #[test]
    fn the_first_non_finite_row_is_reported() {
        assert_eq!(finite("y", &[1.0, 2.0]), Ok(()));
        assert_eq!(
            finite("y", &[1.0, f64::INFINITY, f64::NAN]),
            Err(InputError::NotFinite { what: "y", row: 1 })
        );
    }

    #[test]
    fn probabilities_are_closed_unit_interval() {
        assert_eq!(probability("p", 0.0), Ok(()));
        assert_eq!(probability("p", 1.0), Ok(()));
        assert!(probability("p", f64::NAN).is_err());
        assert!(probability("p", -1e-300).is_err());
        assert_eq!(probability("p", 1.5).unwrap_err().code(), "input.domain");
    }

    #[test]
    fn sortedness_reports_the_offending_row() {
        assert_eq!(sorted_by("t", &[1, 1, 2], i32::cmp), Ok(()));
        assert_eq!(
            sorted_by("t", &[1, 3, 2], i32::cmp),
            Err(InputError::Unsorted { what: "t", row: 2 })
        );
        assert_eq!(
            strictly_sorted_by("t", &[1, 1, 2], i32::cmp),
            Err(InputError::Duplicate { what: "t", row: 1 })
        );
        assert_eq!(
            strictly_sorted_by("t", &[2, 1, 1], i32::cmp),
            Err(InputError::Unsorted { what: "t", row: 1 })
        );
        assert_eq!(strictly_sorted_by::<i32>("t", &[], i32::cmp), Ok(()));
    }

    #[test]
    fn csr_offsets_must_lay_out_every_row() {
        assert_eq!(csr_offsets("o", &[0, 2, 2, 5], 5), Ok(()));
        assert_eq!(csr_offsets("o", &[0], 0), Ok(()));
        assert_eq!(
            csr_offsets("o", &[], 0),
            Err(InputError::Empty { what: "o" })
        );
        assert_eq!(
            csr_offsets("o", &[1, 5], 5).unwrap_err().code(),
            "input.domain"
        );
        assert_eq!(
            csr_offsets("o", &[0, 3, 2, 5], 5),
            Err(InputError::Unsorted { what: "o", row: 2 })
        );
        assert_eq!(
            csr_offsets("o", &[0, 4], 5),
            Err(InputError::Length {
                what: "o",
                expected: 5,
                found: 4
            })
        );
    }

    #[test]
    fn constancy_within_units_in_any_row_order() {
        assert_eq!(
            constant_within("cluster", &[2, 1, 2, 1], &[7, 3, 7, 3]),
            Ok(())
        );
        assert_eq!(
            constant_within("cluster", &[2, 1, 2, 1], &[7, 3, 7, 4]),
            Err(InputError::NotConstant {
                what: "cluster",
                unit: 1
            })
        );
        assert_eq!(
            constant_within("cluster", &[1, 2], &[1])
                .unwrap_err()
                .code(),
            "input.length"
        );
    }

    #[test]
    fn codes_are_unique() {
        let errors = [
            InputError::Length {
                what: "a",
                expected: 1,
                found: 0,
            },
            InputError::Empty { what: "a" },
            InputError::NotFinite { what: "a", row: 0 },
            InputError::Domain {
                what: "a",
                value: 0.0,
                domain: "x",
            },
            InputError::Unsorted { what: "a", row: 1 },
            InputError::Duplicate { what: "a", row: 1 },
            InputError::NotConstant { what: "a", unit: 0 },
            InputError::TooLarge {
                what: "a",
                limit: 1,
                found: 2,
            },
        ];
        let mut codes: Vec<&str> = errors.iter().map(Classify::code).collect();
        codes.sort_unstable();
        codes.dedup();
        assert_eq!(codes.len(), errors.len());
        for error in &errors {
            assert!(error.code().starts_with("input."));
            assert_ne!(error.to_string(), "");
        }
    }
}
