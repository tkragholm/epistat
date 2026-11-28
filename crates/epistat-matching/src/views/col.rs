//! Columns as the caller holds them, and the labels of ids and categories.

use std::fmt;
use std::sync::Arc;

use chrono::{Datelike, NaiveDate};
use epistat_core::Day;

/// A column of `n` values, borrowed or owned, with or without nulls.
///
/// A column is made from a slice or a vector of values, or of `Option`s
/// where some rows have none, or from values with a validity mask, as Arrow
/// and polars hold them ([`Col::with_validity`]). None of these copies the
/// values.
///
/// A date column is a `Col<'_, Day>`, and the only ways to make one name the
/// epoch of the numbers: [`Col::unix_days`] for days since 1970-01-01, as
/// registers, Arrow's `Date32` and polars hold them, [`Col::sas_days`] for
/// days since 1960-01-01, as SAS holds them, or [`Col::dates`] for chrono
/// dates. A bare `&[i64]` is never read as dates.
///
/// ```
/// use epistat_matching::{Col, Day};
///
/// let age = [34.0, 51.0, 29.0];
/// let ages: Col<'_, f64> = Col::from(&age[..]);
/// assert_eq!(ages.len(), 3);
///
/// let income = [Some(310.0), None, Some(275.5)];
/// let incomes: Col<'_, f64> = Col::from(&income[..]);
/// assert_eq!(incomes.null_count(), 1);
///
/// // Days since 1970-01-01, as a register or Arrow's Date32 gives them.
/// let diagnosis = [Some(19_782_i32), None, Some(20_000)];
/// let days: Col<'_, Day> = Col::unix_days_nullable(&diagnosis[..]);
/// assert_eq!(days.len(), 3);
/// ```
pub struct Col<'a, T> {
    repr: Repr<'a, T>,
}

enum Repr<'a, T> {
    Slice(&'a [T]),
    Owned(Vec<T>),
    Masked {
        values: &'a [T],
        valid: &'a [bool],
    },
    Options(&'a [Option<T>]),
    OwnedOptions(Vec<Option<T>>),
    /// Day counts or chrono dates; built only by the constructors of
    /// `Col<'_, Day>`, which is the only column that reads them.
    Days(Days<'a>),
}

/// The day columns a `Col<'_, Day>` can read, and their epoch.
#[derive(Clone, Copy)]
enum Days<'a> {
    Unix(&'a [i64]),
    Unix32(&'a [i32]),
    UnixOptions(&'a [Option<i64>]),
    Unix32Options(&'a [Option<i32>]),
    Sas(&'a [i64]),
    SasOptions(&'a [Option<i64>]),
    Dates(&'a [NaiveDate]),
    DateOptions(&'a [Option<NaiveDate>]),
}

impl Days<'_> {
    const fn len(&self) -> usize {
        match self {
            Self::Unix(days) | Self::Sas(days) => days.len(),
            Self::Unix32(days) => days.len(),
            Self::UnixOptions(days) | Self::SasOptions(days) => days.len(),
            Self::Unix32Options(days) => days.len(),
            Self::Dates(dates) => dates.len(),
            Self::DateOptions(dates) => dates.len(),
        }
    }

    /// The day count since 1970-01-01 of a row, `None` for a null.
    fn unix(&self, row: usize) -> Option<i64> {
        let sas = |days: i64| days.checked_sub(epistat_core::day::SAS_OFFSET_DAYS);
        match *self {
            Self::Unix(days) => Some(days[row]),
            Self::Unix32(days) => Some(i64::from(days[row])),
            Self::UnixOptions(days) => days[row],
            Self::Unix32Options(days) => days[row].map(i64::from),
            // A SAS day too small for the shift is out of range: i64::MIN
            // is refused by `Day::from_unix_days` like any other.
            Self::Sas(days) => Some(sas(days[row]).unwrap_or(i64::MIN)),
            Self::SasOptions(days) => days[row].map(|day| sas(day).unwrap_or(i64::MIN)),
            Self::Dates(dates) => Some(unix_day_of(dates[row])),
            Self::DateOptions(dates) => dates[row].map(unix_day_of),
        }
    }
}

/// The Unix day count of a chrono date.
fn unix_day_of(date: NaiveDate) -> i64 {
    i64::from(date.num_days_from_ce() - crate::UNIX_EPOCH_DAYS_FROM_CE)
}

impl<T> fmt::Debug for Col<'_, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Col")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}

impl<'a, T> From<&'a [T]> for Col<'a, T> {
    fn from(values: &'a [T]) -> Self {
        Self {
            repr: Repr::Slice(values),
        }
    }
}

impl<'a, T> From<&'a Vec<T>> for Col<'a, T> {
    fn from(values: &'a Vec<T>) -> Self {
        Self::from(values.as_slice())
    }
}

impl<T> From<Vec<T>> for Col<'_, T> {
    fn from(values: Vec<T>) -> Self {
        Self {
            repr: Repr::Owned(values),
        }
    }
}

impl<'a, T> From<&'a [Option<T>]> for Col<'a, T> {
    fn from(values: &'a [Option<T>]) -> Self {
        Self {
            repr: Repr::Options(values),
        }
    }
}

impl<'a, T> From<&'a Vec<Option<T>>> for Col<'a, T> {
    fn from(values: &'a Vec<Option<T>>) -> Self {
        Self::from(values.as_slice())
    }
}

impl<T> From<Vec<Option<T>>> for Col<'_, T> {
    fn from(values: Vec<Option<T>>) -> Self {
        Self {
            repr: Repr::OwnedOptions(values),
        }
    }
}

impl<'a, T> Col<'a, T> {
    /// Values with a validity mask, as Arrow and polars hold them: row `i`
    /// is null where `valid[i]` is false. A mask of another length than the
    /// values is refused when the column is added to a view.
    #[must_use]
    pub const fn with_validity(values: &'a [T], valid: &'a [bool]) -> Self {
        Self {
            repr: Repr::Masked { values, valid },
        }
    }

    /// The number of rows.
    #[must_use]
    pub fn len(&self) -> usize {
        match &self.repr {
            Repr::Slice(values) | Repr::Masked { values, .. } => values.len(),
            Repr::Owned(values) => values.len(),
            Repr::Options(values) => values.len(),
            Repr::OwnedOptions(values) => values.len(),
            Repr::Days(days) => days.len(),
        }
    }

    /// Whether the column has no rows.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The number of rows without a value.
    #[must_use]
    pub fn null_count(&self) -> usize {
        match &self.repr {
            Repr::Slice(_) | Repr::Owned(_) => 0,
            Repr::Masked { values, valid } => valid
                .iter()
                .take(values.len())
                .filter(|valid| !**valid)
                .count(),
            Repr::Options(values) => values.iter().filter(|value| value.is_none()).count(),
            Repr::OwnedOptions(values) => values.iter().filter(|value| value.is_none()).count(),
            Repr::Days(days) => (0..days.len())
                .filter(|&row| days.unix(row).is_none())
                .count(),
        }
    }

    /// The length of the validity mask, where it differs from the values'.
    pub(crate) const fn mask_mismatch(&self) -> Option<usize> {
        match &self.repr {
            Repr::Masked { values, valid } if valid.len() != values.len() => Some(valid.len()),
            _ => None,
        }
    }
}

impl<T: Clone> Col<'_, T> {
    /// The value of `row`, `None` for a null.
    ///
    /// Values and labels are read here; dates through [`Col::day`].
    pub(crate) fn get(&self, row: usize) -> Option<T> {
        match &self.repr {
            Repr::Slice(values) => Some(values[row].clone()),
            Repr::Owned(values) => Some(values[row].clone()),
            Repr::Masked { values, valid } => valid[row].then(|| values[row].clone()),
            Repr::Options(values) => values[row].clone(),
            Repr::OwnedOptions(values) => values[row].clone(),
            Repr::Days(_) => {
                unreachable!("day counts are built only as Col<Day>, read by Col::day")
            }
        }
    }
}

/// A row of a date column whose day count is outside the range of [`Day`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OutOfRange;

impl<'a> Col<'a, Day> {
    /// Days since 1970-01-01, as registers, Arrow's `Date32` and polars hold
    /// them: a slice or vector of `i64` or `i32`. No copy.
    #[must_use]
    pub fn unix_days(days: impl UnixDays<'a>) -> Self {
        days.into_unix_col()
    }

    /// Days since 1970-01-01 with nulls: a slice or vector of `Option<i64>`
    /// or `Option<i32>`. No copy.
    #[must_use]
    pub fn unix_days_nullable(days: impl UnixDaysNullable<'a>) -> Self {
        days.into_unix_col()
    }

    /// Days since 1960-01-01, as SAS holds them. Each is read as the Unix day
    /// 3653 days earlier.
    #[must_use]
    pub const fn sas_days(days: &'a [i64]) -> Self {
        Self::days(Days::Sas(days))
    }

    /// Days since 1960-01-01 with nulls.
    #[must_use]
    pub const fn sas_days_nullable(days: &'a [Option<i64>]) -> Self {
        Self::days(Days::SasOptions(days))
    }

    /// chrono dates.
    #[must_use]
    pub const fn dates(dates: &'a [NaiveDate]) -> Self {
        Self::days(Days::Dates(dates))
    }

    /// chrono dates with nulls.
    #[must_use]
    pub const fn dates_nullable(dates: &'a [Option<NaiveDate>]) -> Self {
        Self::days(Days::DateOptions(dates))
    }

    const fn days(days: Days<'a>) -> Self {
        Self {
            repr: Repr::Days(days),
        }
    }

    /// The day of `row`, `None` for a null.
    pub(crate) fn day(&self, row: usize) -> Result<Option<Day>, OutOfRange> {
        let unix = match &self.repr {
            Repr::Days(days) => days.unix(row),
            Repr::Slice(_)
            | Repr::Owned(_)
            | Repr::Masked { .. }
            | Repr::Options(_)
            | Repr::OwnedOptions(_) => {
                return Ok(self.get(row));
            }
        };
        unix.map(|days| Day::from_unix_days(days).ok_or(OutOfRange))
            .transpose()
    }
}

/// Columns of day counts since 1970-01-01 without nulls, for
/// [`Col::unix_days`]: slices and vectors of `i64` and `i32`.
pub trait UnixDays<'a> {
    /// The date column of these days.
    fn into_unix_col(self) -> Col<'a, Day>;
}

impl<'a> UnixDays<'a> for &'a [i64] {
    fn into_unix_col(self) -> Col<'a, Day> {
        Col::days(Days::Unix(self))
    }
}

impl<'a> UnixDays<'a> for &'a Vec<i64> {
    fn into_unix_col(self) -> Col<'a, Day> {
        Col::days(Days::Unix(self))
    }
}

impl<'a> UnixDays<'a> for &'a [i32] {
    fn into_unix_col(self) -> Col<'a, Day> {
        Col::days(Days::Unix32(self))
    }
}

impl<'a> UnixDays<'a> for &'a Vec<i32> {
    fn into_unix_col(self) -> Col<'a, Day> {
        Col::days(Days::Unix32(self))
    }
}

/// Columns of day counts since 1970-01-01 with nulls, for
/// [`Col::unix_days_nullable`]: slices and vectors of `Option<i64>` and
/// `Option<i32>`.
pub trait UnixDaysNullable<'a> {
    /// The date column of these days.
    fn into_unix_col(self) -> Col<'a, Day>;
}

impl<'a> UnixDaysNullable<'a> for &'a [Option<i64>] {
    fn into_unix_col(self) -> Col<'a, Day> {
        Col::days(Days::UnixOptions(self))
    }
}

impl<'a> UnixDaysNullable<'a> for &'a Vec<Option<i64>> {
    fn into_unix_col(self) -> Col<'a, Day> {
        Col::days(Days::UnixOptions(self))
    }
}

impl<'a> UnixDaysNullable<'a> for &'a [Option<i32>] {
    fn into_unix_col(self) -> Col<'a, Day> {
        Col::days(Days::Unix32Options(self))
    }
}

impl<'a> UnixDaysNullable<'a> for &'a Vec<Option<i32>> {
    fn into_unix_col(self) -> Col<'a, Day> {
        Col::days(Days::Unix32Options(self))
    }
}

/// A column of labels as the caller holds it, for the categorical columns
/// of a view: a slice or a vector of labels, or of `Option`s of labels
/// where some rows have none, or a [`Col`] of labels.
pub trait LabelColumn<'a> {
    /// The type of the labels.
    type Label: Label + 'a;
    /// The column.
    fn into_col(self) -> Col<'a, Self::Label>;
}

impl<'a, L: Label + 'a> LabelColumn<'a> for Col<'a, L> {
    type Label = L;
    fn into_col(self) -> Col<'a, L> {
        self
    }
}

impl<'a, L: Label + 'a> LabelColumn<'a> for &'a [L] {
    type Label = L;
    fn into_col(self) -> Col<'a, L> {
        Col::from(self)
    }
}

impl<'a, L: Label + 'a> LabelColumn<'a> for &'a Vec<L> {
    type Label = L;
    fn into_col(self) -> Col<'a, L> {
        Col::from(self)
    }
}

impl<'a, L: Label + 'a> LabelColumn<'a> for Vec<L> {
    type Label = L;
    fn into_col(self) -> Col<'a, L> {
        Col::from(self)
    }
}

impl<'a, L: Label + 'a> LabelColumn<'a> for &'a [Option<L>] {
    type Label = L;
    fn into_col(self) -> Col<'a, L> {
        Col::from(self)
    }
}

impl<'a, L: Label + 'a> LabelColumn<'a> for &'a Vec<Option<L>> {
    type Label = L;
    fn into_col(self) -> Col<'a, L> {
        Col::from(self)
    }
}

impl<'a, L: Label + 'a> LabelColumn<'a> for Vec<Option<L>> {
    type Label = L;
    fn into_col(self) -> Col<'a, L> {
        Col::from(self)
    }
}

/// A type of ids and category values: strings and integers.
///
/// A view codes a label column once, in the sorted order of its distinct
/// labels (`epistat_core::Codes`): `str::cmp` for strings, numeric for
/// integers, so the integer ids 9 and 10 are in that order and not in the
/// order of their text. A label's `Display` is its text in messages and in
/// the engine's records, so distinct labels must display differently; a view
/// refuses a column where two do not.
pub trait Label: Ord + Clone + fmt::Debug + fmt::Display + Send + Sync {}

impl Label for String {}
impl Label for &str {}
impl Label for Box<str> {}
impl Label for Arc<str> {}
impl Label for char {}
impl Label for u8 {}
impl Label for u16 {}
impl Label for u32 {}
impl Label for u64 {}
impl Label for u128 {}
impl Label for usize {}
impl Label for i8 {}
impl Label for i16 {}
impl Label for i32 {}
impl Label for i64 {}
impl Label for i128 {}
impl Label for isize {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_form_of_a_column_reads_its_values_and_nulls() {
        let values = [1.0, 2.0, 3.0];
        let valid = [true, false, true];
        let options = [Some(1.0), None, Some(3.0)];
        let expected = [Some(1.0), None, Some(3.0)];
        for col in [
            Col::with_validity(&values, &valid),
            Col::from(&options[..]),
            Col::from(options.to_vec()),
            Col::from(&options.to_vec()),
        ] {
            assert_eq!(col.len(), 3);
            assert_eq!(col.null_count(), 1);
            assert_eq!((0..3).map(|row| col.get(row)).collect::<Vec<_>>(), expected);
        }
        for col in [
            Col::from(&values[..]),
            Col::from(values.to_vec()),
            Col::from(&values.to_vec()),
        ] {
            assert_eq!(col.null_count(), 0);
            assert_eq!(col.get(1), Some(2.0));
        }
        assert_eq!(
            Col::with_validity(&values, &valid[..2]).mask_mismatch(),
            Some(2)
        );
        assert!(Col::from(Vec::<f64>::new()).is_empty());
    }

    #[test]
    fn day_columns_name_their_epoch() {
        let leap = Day::from_civil(2024, 2, 29).expect("a date");
        let unix = [leap.unix_days()];
        let unix32 = [i32::try_from(leap.unix_days()).expect("fits")];
        let sas = [leap.sas_days()];
        let date = [NaiveDate::from_ymd_opt(2024, 2, 29).expect("a date")];
        for col in [
            Col::unix_days(&unix[..]),
            Col::unix_days(&unix32[..]),
            Col::unix_days(&unix.to_vec()),
            Col::sas_days(&sas),
            Col::dates(&date),
        ] {
            assert_eq!(col.day(0), Ok(Some(leap)));
        }
        let nulls = [None, Some(leap.unix_days())];
        let nulls32 = [None, Some(unix32[0])];
        let sas_nulls = [None, Some(leap.sas_days())];
        let date_nulls = [None, Some(date[0])];
        for col in [
            Col::unix_days_nullable(&nulls[..]),
            Col::unix_days_nullable(&nulls32[..]),
            Col::sas_days_nullable(&sas_nulls),
            Col::dates_nullable(&date_nulls),
        ] {
            assert_eq!((col.day(0), col.day(1)), (Ok(None), Ok(Some(leap))));
            assert_eq!(col.null_count(), 1);
        }
        let days = [leap];
        assert_eq!(Col::from(&days[..]).day(0), Ok(Some(leap)));
        assert_eq!(Col::unix_days(&[i64::MAX][..]).day(0), Err(OutOfRange));
        assert_eq!(Col::sas_days(&[i64::MIN]).day(0), Err(OutOfRange));
    }
}
