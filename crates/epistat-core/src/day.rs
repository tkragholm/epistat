//! Civil dates as day counts: days since 1970-01-01 in the proleptic
//! Gregorian calendar, negative before it.
//!
//! A date column is `i64` days throughout the workspace, the type registers
//! deliver and Arrow's `Date32` widens to. This module converts day counts to
//! and from year, month and day, and adds whole years and months. [`Day`] is
//! one such date as a value of its own: it is built only from a valid
//! calendar date or a day count with its epoch named, and prints as an ISO
//! date.
//!
//! SAS counts days from 1960-01-01, [`SAS_OFFSET_DAYS`] days before the
//! Unix epoch of 1970-01-01. A SAS day passed where a Unix day is read is ten
//! years off and nothing in the number shows it, so [`Day::from_sas_days`]
//! and [`Day::from_unix_days`] name the epoch at the call.
//!
//! # Algorithm
//!
//! The conversions are the algorithms of Hinnant \[1\], exact over every day
//! count whose intermediate products fit an `i64`. A Gregorian era is 400
//! years, or 146 097 days. Dates are counted from 1 March, so that the leap
//! day is the last day of the year. With `y` the year, shifted back by one for
//! January and February, the era `e = floor(y / 400)`, the year of the era
//! `yoe = y - 400 e`, and the day of the shifted year
//! `doy = (153 m' + 2) / 5 + d - 1`, where `m' = m - 3` for `m > 2` and
//! `m + 9` otherwise,
//!
//! ```text
//! doe  = 365 yoe + yoe/4 - yoe/100 + doy        day of the era, 0 to 146 096
//! days = 146 097 e + doe - 719 468              719 468 days from 0000-03-01 to 1970-01-01
//! ```
//!
//! with integer division. [`to_civil`] inverts it.
//!
//! A year is a leap year when it is divisible by 4 and not by 100, or by
//! 400. [`add_years`] keeps the month and day; a 29 February with no
//! counterpart in the target year goes to 1 March or to 28 February, as the
//! caller's [`LeapBirthday`] says. [`add_months`] keeps the day of the month
//! or, when the target month is shorter, takes its last day.
//!
//! # References
//!
//! 1. Hinnant H. chrono-Compatible Low-Level Date Algorithms.
//!    howardhinnant.github.io/date_algorithms.html

use crate::check::InputError;

/// The days from 1960-01-01, the day 0 of SAS dates, to 1970-01-01, the day 0
/// of Unix days: a SAS day `s` is the Unix day `s - SAS_OFFSET_DAYS`.
pub const SAS_OFFSET_DAYS: i64 = 3_653;

/// Days from 1970-01-01 to the proleptic Gregorian date `(y, m, d)`, with the
/// month from 1 to 12 and the day from 1.
///
/// The date is not checked. A month or a day outside its range gives the day
/// count the formula gives, not an error: `from_civil(2020, 13, 45)` is a
/// day in 2021. [`checked_from_civil`] and [`Day::from_civil`] refuse such a
/// date.
#[must_use]
pub const fn from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = (if y >= 0 { y } else { y - 399 }) / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// Days from 1970-01-01 to the proleptic Gregorian date `(y, m, d)`, or
/// `None` when it is not a date: a month outside 1 to 12, a day outside the
/// days of its month, or a year outside the range of [`Day`].
#[must_use]
pub const fn checked_from_civil(y: i64, m: i64, d: i64) -> Option<i64> {
    if y < Day::MIN_YEAR || y > Day::MAX_YEAR || m < 1 || m > 12 {
        return None;
    }
    if d < 1 || d > days_in_month(y, m) {
        return None;
    }
    Some(from_civil(y, m, d))
}

/// The proleptic Gregorian date `(year, month, day)` of a day count from
/// 1970-01-01.
#[must_use]
pub const fn to_civil(day: i64) -> (i64, i64, i64) {
    let z = day + 719_468;
    let era = (if z >= 0 { z } else { z - 146_096 }) / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// The calendar year of a day count.
#[must_use]
pub const fn year_of(day: i64) -> i64 {
    to_civil(day).0
}

/// Whether `year` has a 29 February.
#[must_use]
pub const fn is_leap_year(year: i64) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

/// The number of days in month `month` (1 to 12) of `year`.
#[must_use]
pub const fn days_in_month(year: i64, month: i64) -> i64 {
    match month {
        2 => 28 + is_leap_year(year) as i64,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Where a 29 February falls in a year that has none.
///
/// The rule is always named, because two conventions are in use and a
/// cohort's ages depend on which one it uses.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum LeapBirthday {
    /// On 1 March: a year is completed when the month and day compare as not
    /// earlier than the birth month and day.
    March1,
    /// On 28 February, the date that adding whole years to 29 February gives
    /// in most date libraries.
    February28,
}

/// The day `years` calendar years after `day`, on the same month and day; a
/// negative `years` goes back.
///
/// A 29 February that has no counterpart in the target year goes to 1 March
/// or to 28 February, as `leap` says. The result is the day on which a person
/// born on `day` completes `years` years under the same convention.
#[must_use]
pub const fn add_years(day: i64, years: i64, leap: LeapBirthday) -> i64 {
    let (y, m, d) = to_civil(day);
    let target = y + years;
    if m == 2 && d == 29 && !is_leap_year(target) {
        return match leap {
            LeapBirthday::March1 => from_civil(target, 3, 1),
            LeapBirthday::February28 => from_civil(target, 2, 28),
        };
    }
    from_civil(target, m, d)
}

/// The day `months` calendar months after `day`, on the same day of the month
/// or on the last day of the target month when it is shorter; a negative
/// `months` goes back.
#[must_use]
pub const fn add_months(day: i64, months: i64) -> i64 {
    let (y, m, d) = to_civil(day);
    let index = y * 12 + (m - 1) + months;
    let (ty, tm) = (index.div_euclid(12), index.rem_euclid(12) + 1);
    let last = days_in_month(ty, tm);
    from_civil(ty, tm, if d < last { d } else { last })
}

/// A civil date: one day of the proleptic Gregorian calendar, held as the
/// days since 1970-01-01.
///
/// A `Day` is made from a date that exists ([`Day::from_civil`]) or from a day
/// count whose epoch the call names ([`Day::from_unix_days`],
/// [`Day::from_sas_days`]), and reads back in either. Its years run from
/// [`Day::MIN_YEAR`] to [`Day::MAX_YEAR`], those of an `i32`, so every
/// conversion is exact. It orders as the dates do and prints as an ISO 8601
/// date, `yyyy-mm-dd`, with a sign and at least four digits for a year
/// outside 0 to 9999.
///
/// ```
/// use epistat_core::day::Day;
///
/// let day = Day::from_civil(2024, 2, 29).expect("a leap day");
/// assert_eq!(day.unix_days(), 19_782);
/// assert_eq!(day.sas_days(), 23_435);
/// assert_eq!(Day::from_sas_days(23_435), Some(day));
/// assert_eq!(day.to_string(), "2024-02-29");
/// assert_eq!(Day::from_civil(2023, 2, 29), None);
/// assert_eq!(Day::from_civil(2020, 13, 45), None);
/// ```
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "i64", into = "i64"))]
pub struct Day(i64);

impl Day {
    /// The first year a `Day` can hold.
    pub const MIN_YEAR: i64 = i32::MIN as i64;
    /// The last year a `Day` can hold.
    pub const MAX_YEAR: i64 = i32::MAX as i64;
    /// The first day, 1 January of [`Day::MIN_YEAR`].
    pub const MIN: Self = Self(from_civil(Self::MIN_YEAR, 1, 1));
    /// The last day, 31 December of [`Day::MAX_YEAR`].
    pub const MAX: Self = Self(from_civil(Self::MAX_YEAR, 12, 31));

    /// The date `(year, month, day)`, or `None` when it does not exist: a
    /// month outside 1 to 12, a day outside its month, or a year outside
    /// [`Day::MIN_YEAR`] to [`Day::MAX_YEAR`].
    #[must_use]
    pub const fn from_civil(year: i64, month: i64, day: i64) -> Option<Self> {
        match checked_from_civil(year, month, day) {
            Some(days) => Some(Self(days)),
            None => None,
        }
    }

    /// The day `days` after 1970-01-01, or `None` outside [`Day::MIN`] to
    /// [`Day::MAX`].
    #[must_use]
    pub const fn from_unix_days(days: i64) -> Option<Self> {
        if days < Self::MIN.0 || days > Self::MAX.0 {
            None
        } else {
            Some(Self(days))
        }
    }

    /// The day `days` after 1960-01-01, the epoch of SAS dates, or `None`
    /// outside [`Day::MIN`] to [`Day::MAX`].
    #[must_use]
    pub const fn from_sas_days(days: i64) -> Option<Self> {
        match days.checked_sub(SAS_OFFSET_DAYS) {
            Some(unix) => Self::from_unix_days(unix),
            None => None,
        }
    }

    /// The days since 1970-01-01, negative before it.
    #[must_use]
    pub const fn unix_days(self) -> i64 {
        self.0
    }

    /// The days since 1960-01-01, as SAS counts them.
    #[must_use]
    pub const fn sas_days(self) -> i64 {
        self.0 + SAS_OFFSET_DAYS
    }

    /// The date as `(year, month, day)`.
    #[must_use]
    pub const fn to_civil(self) -> (i64, i64, i64) {
        to_civil(self.0)
    }

    /// The calendar year.
    #[must_use]
    pub const fn year(self) -> i64 {
        year_of(self.0)
    }

    /// The day `days` later, or earlier for a negative `days`, or `None`
    /// outside [`Day::MIN`] to [`Day::MAX`].
    #[must_use]
    pub const fn checked_add_days(self, days: i64) -> Option<Self> {
        match self.0.checked_add(days) {
            Some(sum) => Self::from_unix_days(sum),
            None => None,
        }
    }

    /// The days from `earlier` to this day, negative when `earlier` is
    /// later; `day - earlier` is the same.
    #[must_use]
    pub const fn days_since(self, earlier: Self) -> i64 {
        // Both counts lie within the years of an i32, about 7.8e11 days from
        // the epoch, so the difference cannot overflow.
        self.0 - earlier.0
    }

    /// The day `years` calendar years later, as [`add_years`] gives it under
    /// the rule `leap` for 29 February, or `None` outside [`Day::MIN`] to
    /// [`Day::MAX`].
    #[must_use]
    pub const fn checked_add_years(self, years: i64, leap: LeapBirthday) -> Option<Self> {
        let (y, _, _) = self.to_civil();
        match y.checked_add(years) {
            Some(target) if target >= Self::MIN_YEAR && target <= Self::MAX_YEAR => {
                Some(Self(add_years(self.0, years, leap)))
            }
            _ => None,
        }
    }

    /// The day `months` calendar months later, as [`add_months`] gives it, or
    /// `None` outside [`Day::MIN`] to [`Day::MAX`].
    #[must_use]
    pub const fn checked_add_months(self, months: i64) -> Option<Self> {
        let (y, m, _) = self.to_civil();
        let Some(index) = (y * 12 + (m - 1)).checked_add(months) else {
            return None;
        };
        let target = index.div_euclid(12);
        if target < Self::MIN_YEAR || target > Self::MAX_YEAR {
            return None;
        }
        Some(Self(add_months(self.0, months)))
    }

    /// The day of an Arrow `Date32` value, days since 1970-01-01 as an
    /// `i32`; every such value is a `Day`.
    #[must_use]
    pub const fn from_unix_days_i32(days: i32) -> Self {
        Self(days as i64)
    }

    /// The days since 1970-01-01 as an `i32`, Arrow's `Date32`, or `None` for
    /// a day more than about 5.8 million years from 1970.
    #[must_use]
    pub const fn unix_days_i32(self) -> Option<i32> {
        if self.0 < i32::MIN as i64 || self.0 > i32::MAX as i64 {
            None
        } else {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "the count was checked to fit an i32"
            )]
            let days = self.0 as i32;
            Some(days)
        }
    }

    /// A column of days since 1970-01-01 as days.
    ///
    /// ```
    /// use epistat_core::Day;
    ///
    /// let days = Day::from_unix_column(&[0, 19_782])?;
    /// assert_eq!(days[1].to_string(), "2024-02-29");
    /// assert_eq!(Day::to_unix_column(&days), [0, 19_782]);
    /// assert!(Day::from_unix_column(&[0, i64::MAX]).is_err());
    /// # Ok::<(), epistat_core::InputError>(())
    /// ```
    ///
    /// # Errors
    ///
    /// [`InputError::Domain`] at the first count outside [`Day::MIN`] to
    /// [`Day::MAX`], with its row.
    pub fn from_unix_column(days: &[i64]) -> Result<Vec<Self>, InputError> {
        column(days, Self::from_unix_days, UNIX_DOMAIN)
    }

    /// A column of SAS dates, days since 1960-01-01, as days.
    ///
    /// # Errors
    ///
    /// [`InputError::Domain`] at the first count outside [`Day::MIN`] to
    /// [`Day::MAX`], with its row.
    pub fn from_sas_column(days: &[i64]) -> Result<Vec<Self>, InputError> {
        column(
            days,
            Self::from_sas_days,
            "the SAS days of the years of an i32",
        )
    }

    /// A column of Arrow `Date32` values as days.
    #[must_use]
    pub fn from_unix_column_i32(days: &[i32]) -> Vec<Self> {
        days.iter().map(|&d| Self::from_unix_days_i32(d)).collect()
    }

    /// The days since 1970-01-01 of each day, the column the estimators take.
    #[must_use]
    pub fn to_unix_column(days: &[Self]) -> Vec<i64> {
        days.iter().map(|day| day.0).collect()
    }
}

const UNIX_DOMAIN: &str = "the Unix days of the years of an i32";

fn column(
    days: &[i64],
    convert: impl Fn(i64) -> Option<Day>,
    domain: &'static str,
) -> Result<Vec<Day>, InputError> {
    days.iter()
        .enumerate()
        .map(|(row, &d)| {
            convert(d).ok_or(InputError::Domain {
                what: "day",
                value: d as f64,
                domain,
                row: Some(row),
            })
        })
        .collect()
}

impl core::ops::Sub for Day {
    type Output = i64;

    /// The days from `earlier` to `self`, [`Day::days_since`].
    fn sub(self, earlier: Self) -> i64 {
        self.days_since(earlier)
    }
}

impl core::fmt::Display for Day {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let (y, m, d) = self.to_civil();
        if (0..=9999).contains(&y) {
            write!(f, "{y:04}-{m:02}-{d:02}")
        } else {
            write!(f, "{y:+05}-{m:02}-{d:02}")
        }
    }
}

impl core::fmt::Debug for Day {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Day({self})")
    }
}

impl TryFrom<i64> for Day {
    type Error = InputError;

    /// The day of a Unix day count.
    fn try_from(days: i64) -> Result<Self, Self::Error> {
        Self::from_unix_days(days).ok_or(InputError::Domain {
            what: "day",
            value: days as f64,
            domain: UNIX_DOMAIN,
            row: None,
        })
    }
}

impl From<Day> for i64 {
    /// The Unix day count.
    fn from(day: Day) -> Self {
        day.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_day_is_made_only_from_a_date_that_exists() {
        assert_eq!(checked_from_civil(2020, 13, 45), None);
        assert_eq!(from_civil(2020, 13, 45), from_civil(2021, 2, 14));
        assert_eq!(Day::from_civil(2023, 2, 29), None);
        assert_eq!(Day::from_civil(2023, 4, 31), None);
        assert_eq!(Day::from_civil(2023, 0, 1), None);
        assert_eq!(Day::from_civil(2023, 1, 0), None);
        assert_eq!(Day::from_civil(Day::MAX_YEAR + 1, 1, 1), None);
        assert_eq!(
            Day::from_civil(2024, 2, 29).map(Day::unix_days),
            Some(19_782)
        );
        assert_eq!(Day::from_civil(1960, 1, 1).map(Day::sas_days), Some(0));
        assert_eq!(Day::from_civil(1970, 1, 1).map(Day::unix_days), Some(0));
    }

    #[test]
    fn the_epochs_differ_by_ten_years() {
        let day = Day::from_civil(1970, 1, 1).expect("a date");
        assert_eq!(day.sas_days(), SAS_OFFSET_DAYS);
        assert_eq!(Day::from_sas_days(SAS_OFFSET_DAYS), Some(day));
        assert_eq!(Day::from_sas_days(0), Day::from_civil(1960, 1, 1));
        assert_eq!(Day::from_sas_days(i64::MIN), None);
        assert_eq!(Day::from_unix_days(i64::MAX), None);
        assert_eq!(Day::from_unix_days(Day::MAX.unix_days()), Some(Day::MAX));
        assert_eq!(Day::MAX.checked_add_days(1), None);
        assert_eq!(Day::MIN.checked_add_days(-1), None);
        assert_eq!(
            day.checked_add_days(-1).map(Day::to_civil),
            Some((1969, 12, 31))
        );
        assert_eq!(Day::MIN.to_civil(), (Day::MIN_YEAR, 1, 1));
        assert_eq!(Day::MAX.to_civil(), (Day::MAX_YEAR, 12, 31));
    }

    #[test]
    fn a_day_prints_as_an_iso_date() {
        let print = |y, m, d| Day::from_civil(y, m, d).expect("a date").to_string();
        assert_eq!(print(2024, 2, 29), "2024-02-29");
        assert_eq!(print(5, 3, 1), "0005-03-01");
        assert_eq!(print(-1, 12, 31), "-0001-12-31");
        assert_eq!(print(12_345, 1, 1), "+12345-01-01");
        assert_eq!(
            format!("{:?}", Day::from_civil(2000, 1, 1).expect("a date")),
            "Day(2000-01-01)"
        );
        assert!(Day::from_civil(1999, 12, 31) < Day::from_civil(2000, 1, 1));
    }

    #[test]
    fn days_subtract_and_add_calendar_units() {
        let day = Day::from_civil(2000, 2, 29).expect("a date");
        let later = Day::from_civil(2019, 3, 1).expect("a date");
        assert_eq!(later - day, 6_940);
        assert_eq!(day.days_since(later), -6_940);
        assert_eq!(day.checked_add_years(19, LeapBirthday::March1), Some(later));
        assert_eq!(
            day.checked_add_months(12).map(Day::to_civil),
            Some((2001, 2, 28))
        );
        assert_eq!(Day::MAX.checked_add_years(1, LeapBirthday::March1), None);
        assert_eq!(Day::MIN.checked_add_months(-1), None);
        assert_eq!(Day::MAX.checked_add_months(i64::MAX), None);
        assert_eq!(
            Day::MAX - Day::MIN,
            Day::MAX.unix_days() - Day::MIN.unix_days()
        );
    }

    #[test]
    fn columns_convert_with_the_epoch_named() {
        let days = Day::from_sas_column(&[0, SAS_OFFSET_DAYS]).unwrap();
        assert_eq!(days[0].to_string(), "1960-01-01");
        assert_eq!(Day::to_unix_column(&days), [-SAS_OFFSET_DAYS, 0]);
        assert_eq!(
            Day::from_unix_column(&[1, i64::MIN]),
            Err(InputError::Domain {
                what: "day",
                value: i64::MIN as f64,
                domain: UNIX_DOMAIN,
                row: Some(1)
            })
        );
        let date32 = Day::from_unix_column_i32(&[i32::MIN, 0, i32::MAX]);
        assert_eq!(date32[1].unix_days(), 0);
        assert_eq!(date32[2].unix_days_i32(), Some(i32::MAX));
        assert_eq!(Day::MAX.unix_days_i32(), None);
    }

    #[test]
    fn a_day_converts_to_and_from_its_unix_count() {
        let day = Day::try_from(19_782_i64).expect("in range");
        assert_eq!(i64::from(day), 19_782);
        assert_eq!(day.year(), 2024);
        assert!(Day::try_from(i64::MIN).is_err());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serde_writes_the_unix_count_and_refuses_one_out_of_range() {
        let day = Day::from_civil(2024, 2, 29).expect("a date");
        assert_eq!(serde_json::to_string(&day).unwrap(), "19782");
        assert_eq!(serde_json::from_str::<Day>("19782").unwrap(), day);
        assert!(serde_json::from_str::<Day>("9223372036854775807").is_err());
    }

    #[test]
    fn known_dates() {
        assert_eq!(from_civil(1970, 1, 1), 0);
        assert_eq!(from_civil(1969, 12, 31), -1);
        assert_eq!(from_civil(2000, 3, 1), 11_017);
        assert_eq!(from_civil(1960, 1, 1), -3_653);
        assert_eq!(to_civil(19_782), (2024, 2, 29));
        assert_eq!(year_of(-1), 1969);
    }

    /// A day-by-day walk of the calendar from 1599-01-01 to 2401-12-31 agrees
    /// with both conversions on every day.
    #[test]
    fn the_conversions_agree_with_a_walk_of_the_calendar() {
        let mut day = from_civil(1599, 1, 1);
        for year in 1599..=2401 {
            for month in 1..=12 {
                for d in 1..=days_in_month(year, month) {
                    assert_eq!(from_civil(year, month, d), day);
                    assert_eq!(to_civil(day), (year, month, d));
                    day += 1;
                }
            }
        }
    }

    #[test]
    fn conversions_round_trip_far_from_the_epoch() {
        for day in (-200_000_000_i64..200_000_000).step_by(9_973) {
            let (y, m, d) = to_civil(day);
            assert!((1..=12).contains(&m) && (1..=days_in_month(y, m)).contains(&d));
            assert_eq!(from_civil(y, m, d), day);
        }
    }

    #[test]
    fn leap_years() {
        assert!(is_leap_year(2000) && is_leap_year(2024) && is_leap_year(-4));
        assert!(!is_leap_year(1900) && !is_leap_year(2023) && !is_leap_year(2100));
        assert_eq!(days_in_month(2024, 2), 29);
        assert_eq!(days_in_month(2023, 2), 28);
        assert_eq!(days_in_month(2023, 4), 30);
    }

    #[test]
    fn whole_years_and_months_keep_the_calendar_date() {
        let leap_day = from_civil(2000, 2, 29);
        assert_eq!(
            add_years(leap_day, 19, LeapBirthday::March1),
            from_civil(2019, 3, 1)
        );
        assert_eq!(
            add_years(leap_day, 19, LeapBirthday::February28),
            from_civil(2019, 2, 28)
        );
        assert_eq!(
            add_years(leap_day, 4, LeapBirthday::March1),
            from_civil(2004, 2, 29)
        );
        assert_eq!(
            add_years(from_civil(2010, 6, 15), -10, LeapBirthday::March1),
            from_civil(2000, 6, 15)
        );
        assert_eq!(
            add_months(from_civil(2023, 1, 31), 1),
            from_civil(2023, 2, 28)
        );
        assert_eq!(
            add_months(from_civil(2024, 1, 31), 1),
            from_civil(2024, 2, 29)
        );
        assert_eq!(
            add_months(from_civil(2024, 3, 15), -15),
            from_civil(2022, 12, 15)
        );
    }
}
