//! The input checks every estimator shares, and the error they return.
//!
//! An argument of one kind gets one check and one message in every crate: a
//! column of the wrong length is [`InputError::Length`] whether it reaches a
//! Cox model or a difference-in-differences estimator. Each crate's error enum
//! wraps [`InputError`] in an `Input` variant, so a caller matches on the same
//! variants everywhere and a binding reads one set of codes.
//!
//! `what` names the argument as the function's documentation does, such as
//! `"weights"` or `"period"`. Row and column numbers count from 0 in the
//! caller's input, and an error names the position of the first value it
//! refuses wherever the check sees one.
//!
//! | check | refuses | error |
//! |---|---|---|
//! | [`same_length`] | a length other than the expected one | `Length` |
//! | [`non_empty`] | no rows | `Empty` |
//! | [`finite`] | NaN or an infinity | `NotFinite` |
//! | [`finite_column`] | NaN or an infinity in one column of several | `NotFinite` |
//! | [`weights`] | NaN, an infinity or a negative weight | `NotFinite`, `Domain` |
//! | [`positive_weights`] | that, or a weight of 0 | `NotFinite`, `Domain` |
//! | [`probability`] | a value outside `[0, 1]` | `Domain` |
//! | [`positive`] | a value that is not finite and above 0 | `Domain` |
//! | [`below`] | a value not below a limit the call sets | `NotBelow` |
//! | [`sorted_by`] | a row before its predecessor | `Unsorted` |
//! | [`strictly_sorted_by`] | that, or a row equal to its predecessor | `Unsorted`, `Duplicate` |
//! | [`csr_offsets`] | offsets that are not a layout of the rows | `Empty`, `Domain`, `Unsorted`, `Length` |
//! | [`constant_within`] | a value that changes within a unit | `NotConstant` |
//! | [`fits_u32`] | a count above `u32::MAX` | `TooLarge` |
//! | [`CaseWeights::new`](crate::CaseWeights::new) | a weight [`weights`] refuses, or weights all zero | `NotFinite`, `Domain`, `Empty`, `AllZero` |
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
    /// `input.length`: an argument has `found` rows or entries where
    /// `expected` were needed.
    Length {
        /// The argument.
        what: &'static str,
        /// The length it must have.
        expected: usize,
        /// The length it has.
        found: usize,
    },
    /// `input.empty`: an argument has no rows where at least one is needed.
    Empty {
        /// The argument.
        what: &'static str,
    },
    /// `input.not_finite`: an argument holds NaN or an infinity at `row`, and
    /// in `column` when the argument has several.
    NotFinite {
        /// The argument.
        what: &'static str,
        /// The first such row.
        row: usize,
        /// The column of that row, for an argument of several columns.
        column: Option<usize>,
    },
    /// `input.domain`: an argument's value lies outside the set it must lie
    /// in.
    Domain {
        /// The argument.
        what: &'static str,
        /// The value given.
        value: f64,
        /// The set the value must lie in, as text, such as `"(0, 1)"`.
        domain: &'static str,
        /// The row of the value, for an argument with one value per row.
        row: Option<usize>,
    },
    /// `input.not_below`: an argument's value is not below the limit the call
    /// sets for it, such as a lag that must be below the number of periods.
    NotBelow {
        /// The argument.
        what: &'static str,
        /// The value given.
        value: f64,
        /// The limit the value must lie below.
        limit: f64,
    },
    /// `input.unsorted`: row `row` of an argument sorts before row `row - 1`.
    Unsorted {
        /// The argument.
        what: &'static str,
        /// The first such row.
        row: usize,
    },
    /// `input.duplicate`: row `row` of an argument repeats row `row - 1`
    /// where the rows must be distinct.
    Duplicate {
        /// The argument.
        what: &'static str,
        /// The first such row.
        row: usize,
    },
    /// `input.not_constant`: an argument that must be constant within a unit
    /// takes two values in unit `unit`.
    NotConstant {
        /// The argument.
        what: &'static str,
        /// The unit, as its code.
        unit: u32,
    },
    /// `input.too_large`: an argument has more entries than its index type
    /// can number.
    TooLarge {
        /// The argument.
        what: &'static str,
        /// The largest count allowed.
        limit: u64,
        /// The count given.
        found: u64,
    },
    /// `input.all_zero`: an argument whose values may be zero is zero in
    /// every row, as case weights that leave no row in the fit.
    AllZero {
        /// The argument.
        what: &'static str,
    },
    /// `input.percentage`: a proportion in `(0, 1)`, such as a confidence
    /// level, was given as a percentage between 1 and 100, as `95` for
    /// `0.95`.
    Percentage {
        /// The argument.
        what: &'static str,
        /// The value given.
        value: f64,
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
            Self::NotFinite {
                what,
                row,
                column: None,
            } => write!(f, "{what} is not finite at row {row}"),
            Self::NotFinite {
                what,
                row,
                column: Some(column),
            } => write!(f, "{what} is not finite at row {row}, column {column}"),
            Self::Domain {
                what,
                value,
                domain,
                row: None,
            } => write!(f, "{what} is {value}, outside {domain}"),
            Self::Domain {
                what,
                value,
                domain,
                row: Some(row),
            } => write!(f, "{what} is {value} at row {row}, outside {domain}"),
            Self::NotBelow { what, value, limit } => {
                write!(f, "{what} is {value}, not below {limit}")
            }
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
            Self::AllZero { what } => write!(f, "{what} are zero in every row"),
            Self::Percentage { what, value } => write!(
                f,
                "{what} is {value}, outside (0, 1); as a proportion, {value}% is {}",
                value / 100.0
            ),
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
            Self::NotBelow { .. } => "input.not_below",
            Self::Unsorted { .. } => "input.unsorted",
            Self::Duplicate { .. } => "input.duplicate",
            Self::NotConstant { .. } => "input.not_constant",
            Self::TooLarge { .. } => "input.too_large",
            Self::AllZero { .. } => "input.all_zero",
            Self::Percentage { .. } => "input.percentage",
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
        Some(row) => Err(InputError::NotFinite {
            what,
            row,
            column: None,
        }),
        None => Ok(()),
    }
}

/// Every value of column `column` of a multi-column argument is finite.
///
/// # Errors
///
/// [`InputError::NotFinite`] at the first NaN or infinity, naming the column.
pub fn finite_column(what: &'static str, column: usize, values: &[f64]) -> Result<(), InputError> {
    match values.iter().position(|value| !value.is_finite()) {
        Some(row) => Err(InputError::NotFinite {
            what,
            row,
            column: Some(column),
        }),
        None => Ok(()),
    }
}

/// Case weights: every weight finite and not negative. A weight of 0 is
/// allowed. [`CaseWeights`](crate::CaseWeights) runs this check and also
/// refuses weights that are all zero.
///
/// # Errors
///
/// At the first weight refused: [`InputError::NotFinite`] for NaN or an
/// infinity and [`InputError::Domain`], with its row, for a negative weight.
pub fn weights(what: &'static str, values: &[f64]) -> Result<(), InputError> {
    weights_in(what, values, |w| w >= 0.0, "[0, inf)")
}

/// Weights that must be positive: every weight finite and above 0.
///
/// # Errors
///
/// At the first weight refused: [`InputError::NotFinite`] for NaN or an
/// infinity and [`InputError::Domain`], with its row, for a weight of 0 or
/// below.
pub fn positive_weights(what: &'static str, values: &[f64]) -> Result<(), InputError> {
    weights_in(what, values, |w| w > 0.0, "(0, inf)")
}

fn weights_in(
    what: &'static str,
    values: &[f64],
    allowed: impl Fn(f64) -> bool,
    domain: &'static str,
) -> Result<(), InputError> {
    let Some(row) = values.iter().position(|&w| !(w.is_finite() && allowed(w))) else {
        return Ok(());
    };
    let value = values[row];
    Err(if value.is_finite() {
        InputError::Domain {
            what,
            value,
            domain,
            row: Some(row),
        }
    } else {
        InputError::NotFinite {
            what,
            row,
            column: None,
        }
    })
}

/// `value` is finite and above 0, as a tolerance or a scale must be.
///
/// # Errors
///
/// [`InputError::Domain`] for 0, a negative value, NaN or an infinity.
pub fn positive(what: &'static str, value: f64) -> Result<(), InputError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(InputError::Domain {
            what,
            value,
            domain: "(0, inf)",
            row: None,
        })
    }
}

/// `value` lies below `limit`, a bound that depends on the call, such as the
/// number of periods for a lag.
///
/// # Errors
///
/// [`InputError::NotBelow`] for a value at or above the limit, or NaN.
pub fn below(what: &'static str, value: f64, limit: f64) -> Result<(), InputError> {
    if value < limit {
        Ok(())
    } else {
        Err(InputError::NotBelow { what, value, limit })
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
            row: None,
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
            row: None,
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

/// `count` as a `u32`, for codes numbered `0..count` or a count stored as
/// one.
///
/// # Errors
///
/// [`InputError::TooLarge`] for a count above `u32::MAX`.
pub fn fits_u32(what: &'static str, count: usize) -> Result<u32, InputError> {
    u32::try_from(count).map_err(|_| InputError::TooLarge {
        what,
        limit: u64::from(u32::MAX),
        found: u64::try_from(count).unwrap_or(u64::MAX),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_fit_u32_up_to_its_maximum() {
        assert_eq!(fits_u32("clusters", 0), Ok(0));
        assert_eq!(fits_u32("clusters", u32::MAX as usize), Ok(u32::MAX));
        #[cfg(target_pointer_width = "64")]
        assert_eq!(
            fits_u32("clusters", u32::MAX as usize + 1),
            Err(InputError::TooLarge {
                what: "clusters",
                limit: u64::from(u32::MAX),
                found: u64::from(u32::MAX) + 1,
            })
        );
    }

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
            Err(InputError::NotFinite {
                what: "y",
                row: 1,
                column: None
            })
        );
        assert_eq!(
            finite_column("x", 2, &[0.0, f64::NAN])
                .unwrap_err()
                .to_string(),
            "x is not finite at row 1, column 2"
        );
    }

    #[test]
    fn weights_report_the_row_of_the_first_refused() {
        assert_eq!(weights("w", &[0.0, 1.5]), Ok(()));
        assert_eq!(
            weights("w", &[1.0, -0.5, f64::NAN]),
            Err(InputError::Domain {
                what: "w",
                value: -0.5,
                domain: "[0, inf)",
                row: Some(1)
            })
        );
        assert_eq!(
            weights("w", &[1.0, f64::INFINITY, -1.0]),
            Err(InputError::NotFinite {
                what: "w",
                row: 1,
                column: None
            })
        );
        let zero = positive_weights("w", &[1.0, 0.0]).unwrap_err();
        assert_eq!(zero.code(), "input.domain");
        assert_eq!(zero.to_string(), "w is 0 at row 1, outside (0, inf)");
        assert_eq!(positive_weights("w", &[1e-300]), Ok(()));
    }

    #[test]
    fn positive_and_below_refuse_the_boundary_and_nan() {
        assert_eq!(positive("tol", 1e-9), Ok(()));
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert_eq!(positive("tol", bad).unwrap_err().code(), "input.domain");
        }
        assert_eq!(below("lag", 2.0, 3.0), Ok(()));
        let error = below("lag", 3.0, 3.0).unwrap_err();
        assert_eq!(
            error,
            InputError::NotBelow {
                what: "lag",
                value: 3.0,
                limit: 3.0
            }
        );
        assert_eq!(error.code(), "input.not_below");
        assert_eq!(error.to_string(), "lag is 3, not below 3");
        assert!(below("lag", f64::NAN, 3.0).is_err());
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
            InputError::NotFinite {
                what: "a",
                row: 0,
                column: None,
            },
            InputError::Domain {
                what: "a",
                value: 0.0,
                domain: "x",
                row: None,
            },
            InputError::NotBelow {
                what: "a",
                value: 1.0,
                limit: 1.0,
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
