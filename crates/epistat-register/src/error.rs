//! The one error of the crate.
//!
//! Every function that can refuse its input returns [`Error`]. A column of the
//! wrong length, offsets that are not a layout over the rows, rows out of
//! order, a value outside its domain and more rows than a `u32` holds are
//! [`Error::Input`] with the shared [`InputError`], so they read as in every
//! other epistat crate. Input that is well formed but breaks a contract of
//! this crate has a variant of its own, with the row or the name it concerns.
//! [`Classify`] gives a binding the kind and a stable code.

use std::fmt;

use epistat_core::{Classify, ErrorKind, InputError, check};

use crate::codes::MapKind;

/// Why a function refused its input or a price index lacked a year.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Input the function cannot take: columns of different lengths, offsets
    /// that are not a layout over the rows, rows out of order, a value
    /// outside its domain or more rows than a `u32` row number holds.
    #[error(transparent)]
    Input(#[from] InputError),
    /// Row `row` of `what`, an interval, window or span, ends before it
    /// starts (for a half-open interval, ends where it starts).
    #[error("{what} ends before it starts at row {row}")]
    EndBeforeStart {
        /// The argument.
        what: &'static str,
        /// The row, key, person or window, as the function documents.
        row: usize,
    },
    /// A value that names a row, person, group, key or condition is not
    /// below the number there are.
    #[error("{what}{} is {value}, not below {limit}", at_row(*.row))]
    OutOfRange {
        /// The argument.
        what: &'static str,
        /// The row of the value, for an argument with one value per row.
        row: Option<usize>,
        /// The value given.
        value: u64,
        /// The number of rows, persons, groups or keys there are.
        limit: u64,
    },
    /// The first row of `what` does not begin a key.
    #[error("the first row of {what} does not begin a key")]
    FirstRowBeginsNoKey {
        /// The argument that marks the first row of each key.
        what: &'static str,
    },
    /// A day computed for row `row` does not fit in an `i64`.
    #[error("a day of {what} at row {row} does not fit in an i64")]
    DayOverflow {
        /// The day: a birthday, an end or a half-open end.
        what: &'static str,
        /// The row it was computed for.
        row: usize,
    },
    /// An item of a code specification cannot be read.
    #[error("code item {item:?} {problem}")]
    CodeItem {
        /// The item as written.
        item: String,
        /// What is wrong with it.
        problem: CodeItemProblem,
    },
    /// A name that must be unique is given twice.
    #[error("{what} {name:?} is given twice")]
    DuplicateName {
        /// What the name names: a condition, a set, a label, a person.
        what: &'static str,
        /// The name.
        name: String,
    },
    /// A name that the map does not hold.
    #[error("there is no {what} {name:?}")]
    UnknownName {
        /// What the name should name.
        what: &'static str,
        /// The name.
        name: String,
    },
    /// A condition supersedes itself or a key that names no condition of the
    /// map.
    #[error(
        "condition {key:?} supersedes {target:?}, which is itself or not a condition of the map"
    )]
    Supersedes {
        /// The condition.
        key: String,
        /// The key it supersedes.
        target: String,
    },
    /// Two entries of `what` claim the same code, where each code belongs to
    /// one entry.
    #[error("{what} {first:?} and {second:?} share a code")]
    SharedCode {
        /// The map or hierarchy.
        what: &'static str,
        /// The first entry.
        first: String,
        /// The second entry.
        second: String,
    },
    /// A set built by intersection or difference has no original items, so a
    /// longest prefix cannot be read from it.
    #[error("the set {set:?} has no original items, so a longest prefix cannot be read from it")]
    NoItems {
        /// The name of the set.
        set: String,
    },
    /// A pair of a crosswalk cannot be used.
    #[error("crosswalk pair {from} to {to}: {problem}")]
    Crosswalk {
        /// The source code as given.
        from: String,
        /// The target code as given.
        to: String,
        /// What is wrong with the pair.
        problem: CrosswalkProblem,
    },
    /// The parents of a person are impossible.
    #[error("person {person} {problem}")]
    Pedigree {
        /// The person's identifier.
        person: u64,
        /// What is wrong.
        problem: PedigreeProblem,
    },
    /// A group of relatives is larger than the limit the caller set.
    #[error("{what} {id} has {size} members among the rows, more than the limit {limit}")]
    GroupTooLarge {
        /// The group: the children of a parent, the descendants of an
        /// ancestor.
        what: &'static str,
        /// The identifier of the parent or ancestor.
        id: u64,
        /// The size of the group.
        size: usize,
        /// The limit.
        limit: usize,
    },
    /// An option or a column that `by` needs is not given.
    #[error("{by} needs {what}")]
    Required {
        /// What is missing.
        what: &'static str,
        /// The option or function that needs it.
        by: &'static str,
    },
    /// An argument has fewer values than the method needs.
    #[error("{what} has {found} values, fewer than the {needed} needed")]
    TooFew {
        /// The argument.
        what: &'static str,
        /// The number needed.
        needed: usize,
        /// The number given.
        found: usize,
    },
    /// The optimiser of a fit failed.
    #[error(transparent)]
    Optim(#[from] epistat_optim::Error),
    /// The reference year has no price index.
    #[error("Reference price year {0} is missing from the CPI table.")]
    MissingReferenceYear(i32),
    /// One or more years have no price index, sorted and without duplicates.
    #[error("Missing CPI adjustment factors for years: {}", join_years(.0))]
    MissingYears(Vec<i32>),
}

/// What is wrong with an item of a code specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum CodeItemProblem {
    /// It holds a character other than a letter, a digit, a dot, a hyphen or
    /// an en dash.
    Character(char),
    /// An end is longer than eight characters.
    TooLong,
    /// An end is empty.
    Empty,
    /// It has more than one hyphen or en dash.
    RangeMarks,
    /// The ends of the range differ in length.
    EndLengths,
    /// The range ends before it starts.
    Reversed,
}

impl fmt::Display for CodeItemProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Character(ch) => write!(
                f,
                "holds the character {ch:?}, which is neither a letter, a digit nor a dot"
            ),
            Self::TooLong => f.write_str("is longer than 8 characters"),
            Self::Empty => f.write_str("has an empty code"),
            Self::RangeMarks => f.write_str("has more than one hyphen or en dash"),
            Self::EndLengths => f.write_str("has ends of different lengths"),
            Self::Reversed => f.write_str("ends before it starts"),
        }
    }
}

/// What is wrong with a pair of a crosswalk.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum CrosswalkProblem {
    /// The source is not an ICD-8 code.
    SourceNotIcd8,
    /// The target is not an ICD-10 code.
    TargetNotIcd10,
    /// The pair is given twice.
    Repeated,
    /// The declared kind does not fit the number of targets of the source and
    /// sources of the target.
    Kind {
        /// The kind declared.
        declared: MapKind,
        /// The number of targets of the source.
        targets: usize,
        /// The number of sources of the target.
        sources: usize,
    },
}

impl fmt::Display for CrosswalkProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceNotIcd8 => f.write_str("the source is not an ICD-8 code"),
            Self::TargetNotIcd10 => f.write_str("the target is not an ICD-10 code"),
            Self::Repeated => f.write_str("the pair is given twice"),
            Self::Kind {
                declared,
                targets,
                sources,
            } => write!(
                f,
                "declared {declared:?}, but its source has {targets} targets and its target {sources} sources"
            ),
        }
    }
}

/// What is wrong with the parents of a person.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PedigreeProblem {
    /// The person is their own ancestor.
    OwnAncestor,
    /// The mother and the father are the same person.
    SameParents,
}

impl fmt::Display for PedigreeProblem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OwnAncestor => f.write_str("is their own ancestor"),
            Self::SameParents => f.write_str("has the same mother and father"),
        }
    }
}

fn at_row(row: Option<usize>) -> String {
    row.map_or_else(String::new, |row| format!(" at row {row}"))
}

impl Classify for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Self::Input(error) => error.kind(),
            Self::Optim(error) => error.kind(),
            Self::EndBeforeStart { .. }
            | Self::OutOfRange { .. }
            | Self::FirstRowBeginsNoKey { .. }
            | Self::DayOverflow { .. }
            | Self::CodeItem { .. }
            | Self::DuplicateName { .. }
            | Self::UnknownName { .. }
            | Self::Supersedes { .. }
            | Self::SharedCode { .. }
            | Self::NoItems { .. }
            | Self::Crosswalk { .. }
            | Self::Pedigree { .. }
            | Self::GroupTooLarge { .. }
            | Self::Required { .. }
            | Self::TooFew { .. }
            | Self::MissingReferenceYear(_)
            | Self::MissingYears(_) => ErrorKind::Input,
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input(error) => error.code(),
            Self::Optim(error) => error.code(),
            Self::EndBeforeStart { .. } => "register.end_before_start",
            Self::OutOfRange { .. } => "register.out_of_range",
            Self::FirstRowBeginsNoKey { .. } => "register.first_row_begins_no_key",
            Self::DayOverflow { .. } => "register.day_overflow",
            Self::CodeItem { .. } => "register.code_item",
            Self::DuplicateName { .. } => "register.duplicate_name",
            Self::UnknownName { .. } => "register.unknown_name",
            Self::Supersedes { .. } => "register.supersedes",
            Self::SharedCode { .. } => "register.shared_code",
            Self::NoItems { .. } => "register.no_items",
            Self::Crosswalk { .. } => "register.crosswalk",
            Self::Pedigree { .. } => "register.pedigree",
            Self::GroupTooLarge { .. } => "register.group_too_large",
            Self::Required { .. } => "register.required",
            Self::TooFew { .. } => "register.too_few",
            Self::MissingReferenceYear(_) => "register.missing_reference_year",
            Self::MissingYears(_) => "register.missing_years",
        }
    }
}

/// `found` equals `expected`, or [`InputError::Length`].
pub(crate) fn length(what: &'static str, expected: usize, found: usize) -> Result<(), Error> {
    Ok(check::same_length(what, expected, found)?)
}

/// `offsets` lay out `rows` rows, or the error of the first fault.
pub(crate) fn csr(what: &'static str, offsets: &[usize], rows: usize) -> Result<(), Error> {
    Ok(check::csr_offsets(what, offsets, rows)?)
}

/// `n` fits a `u32` row number, or [`InputError::TooLarge`].
pub(crate) fn row_count(what: &'static str, n: usize) -> Result<u32, Error> {
    u32::try_from(n).map_err(|_| {
        InputError::TooLarge {
            what,
            limit: u64::from(u32::MAX),
            found: n as u64,
        }
        .into()
    })
}

fn join_years(years: &[i32]) -> String {
    years
        .iter()
        .map(i32::to_string)
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// One error of every variant of the crate's own.
    pub(crate) fn one_of_each() -> Vec<Error> {
        vec![
            Error::EndBeforeStart { what: "x", row: 0 },
            Error::OutOfRange {
                what: "x",
                row: None,
                value: 1,
                limit: 1,
            },
            Error::FirstRowBeginsNoKey { what: "x" },
            Error::DayOverflow { what: "x", row: 0 },
            Error::CodeItem {
                item: String::new(),
                problem: CodeItemProblem::Empty,
            },
            Error::DuplicateName {
                what: "x",
                name: String::new(),
            },
            Error::UnknownName {
                what: "x",
                name: String::new(),
            },
            Error::Supersedes {
                key: String::new(),
                target: String::new(),
            },
            Error::SharedCode {
                what: "x",
                first: String::new(),
                second: String::new(),
            },
            Error::NoItems { set: String::new() },
            Error::Crosswalk {
                from: String::new(),
                to: String::new(),
                problem: CrosswalkProblem::Repeated,
            },
            Error::Pedigree {
                person: 0,
                problem: PedigreeProblem::OwnAncestor,
            },
            Error::GroupTooLarge {
                what: "x",
                id: 0,
                size: 2,
                limit: 1,
            },
            Error::Required { what: "x", by: "y" },
            Error::TooFew {
                what: "x",
                needed: 3,
                found: 2,
            },
            Error::MissingReferenceYear(1900),
            Error::MissingYears(vec![1900]),
        ]
    }

    #[test]
    fn codes_are_unique_and_prefixed() {
        let errors = one_of_each();
        epistat_testkit::assert_codes_unique("register", &errors);
        assert!(errors.iter().all(|e| e.kind() == ErrorKind::Input));
        assert_eq!(
            Error::from(InputError::Empty { what: "x" }).code(),
            "input.empty"
        );
        assert_eq!(
            Error::from(epistat_optim::Error::NotDescent).code(),
            epistat_optim::Error::NotDescent.code()
        );
    }

    #[test]
    fn messages_name_the_row_and_the_value() {
        let error = Error::OutOfRange {
            what: "person",
            row: Some(3),
            value: 9,
            limit: 4,
        };
        assert_eq!(error.to_string(), "person at row 3 is 9, not below 4");
        let item = Error::CodeItem {
            item: "C0$".into(),
            problem: CodeItemProblem::Character('$'),
        };
        assert!(item.to_string().contains("'$'"));
    }
}
